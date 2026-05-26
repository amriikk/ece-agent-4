"""
memory.py — Single source of truth for all conversation and execution state.

Holds:
  - chat_history      : conversation turns (role + content)
  - active_datasets   : {year: DataFrame} dict of loaded financial CSVs
  - namespace         : persistent Python exec() namespace across iterations
  - last_result       : last DataFrame / scalar from analytics agent
  - iteration_traces  : per-iteration logs [{n, reasoning, code, observation}]
  - last_verdict      : last validator verdict string
"""

from __future__ import annotations
import json
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Helpers ───────────────────────────────────────────────────────────────────

def _json_safe(obj: Any) -> Any:
    """Recursively strip non-JSON-serialisable types for SSE payloads."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(i) for i in obj]
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.head(5).to_json(orient="records"))
    if isinstance(obj, pd.Series):
        return json.loads(obj.to_json())
    if hasattr(obj, "item"):          # numpy scalar
        return obj.item()
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


# ── ConversationMemory ────────────────────────────────────────────────────────

@dataclass
class ConversationMemory:
    """All mutable state for a single user session."""

    # Conversation history — list of {"role": "user"|"assistant", "content": str}
    chat_history: List[Dict[str, str]] = field(default_factory=list)

    # Loaded datasets — {year_str: DataFrame}, e.g. {"2014": df2014, "2018": df2018}
    active_datasets: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # Persistent Python namespace for the analytics agent's exec() calls.
    # Variables created in iteration N survive into iteration N+1.
    namespace: Dict[str, Any] = field(default_factory=dict)

    # Last DataFrame result returned by the analytics agent (for follow-ups)
    last_result: Optional[pd.DataFrame] = None

    # Last validator verdict: "APPROVED" | "RETRY: <reason>" | "SUSPICIOUS: <reason>"
    last_verdict: str = "NONE"

    # Per-query iteration log — reset at the start of each new query
    iteration_traces: List[Dict[str, Any]] = field(default_factory=list)

    # Dataset file paths currently active
    dataset_paths: List[str] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self.chat_history.append({"role": "user", "content": content})
        self.iteration_traces = []   # fresh trace per query

    def add_assistant_message(self, content: str) -> None:
        self.chat_history.append({"role": "assistant", "content": content})

    def add_iteration_trace(
        self,
        n: int,
        reasoning: str,
        code: str,
        observation: str,
    ) -> None:
        self.iteration_traces.append({
            "n": n,
            "reasoning": reasoning,
            "code": code,
            "observation": observation,
        })

    def reset_namespace(self) -> None:
        """
        Clear the exec namespace. Called when a new dataset is loaded so
        stale variables from a previous session don't bleed in.
        """
        self.namespace.clear()
        self.last_result = None
        self.last_verdict = "NONE"
        self.iteration_traces = []

    def load_dataset(self, path: str, year: str) -> pd.DataFrame:
        """
        Load a financial CSV, rename the ticker column, and store in
        active_datasets[year] AND inject into the exec namespace as df_{year}.
        """
        df = pd.read_csv(path, encoding="latin-1")
        df = df.rename(columns={"Unnamed: 0": "Ticker"})

        # Numeric cleaning: strip commas, coerce to float
        for col in df.columns:
            if col in ("Ticker", "Sector", "Class"):
                continue
            try:
                cleaned = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
                converted = pd.to_numeric(cleaned, errors="coerce")
                if converted.notna().sum() > 10:
                    df[col] = converted
            except Exception:
                pass

        # Tag with year for multi-table unions
        df["Year"] = int(year)

        self.active_datasets[year] = df
        self.namespace[f"df_{year}"] = df

        if path not in self.dataset_paths:
            self.dataset_paths.append(path)

        return df

    def get_schema_context(self) -> str:
        """
        Returns a compact schema description for all loaded datasets —
        injected into LLM prompts so the model knows exactly what's available.
        """
        if not self.active_datasets:
            return "No datasets loaded."

        lines = []
        for year, df in sorted(self.active_datasets.items()):
            price_col = f"{int(year)+1} PRICE VAR [%]"
            lines.append(
                f"df_{year}  ({len(df):,} rows × {len(df.columns)} cols) — "
                f"year {year}, ticker col='Ticker', "
                f"target col='{price_col}' (if present), "
                f"Class col=0/1 buy signal"
            )

        # Show shared column groups once (same across all years)
        if self.active_datasets:
            sample = next(iter(self.active_datasets.values()))
            cols = list(sample.columns)
            groups = {
                "Identity": [c for c in cols if c in ("Ticker","Sector","Class","Year")],
                "Income": [c for c in cols if c in (
                    "Revenue","Revenue Growth","Gross Profit","Gross Margin",
                    "Net Income","EPS","EBITDA","Operating Income")],
                "Profitability": [c for c in cols if c in (
                    "returnOnEquity","returnOnAssets","Profit Margin",
                    "Net Profit Margin","EBITDA Margin","ROIC")],
                "Valuation": [c for c in cols if c in (
                    "PE ratio","PB ratio","Market Cap","Enterprise Value",
                    "priceToSalesRatio","Enterprise Value over EBITDA")],
                "Growth": [c for c in cols if c in (
                    "Revenue Growth","Net Income Growth","EPS Growth",
                    "5Y Revenue Growth (per Share)","3Y Revenue Growth (per Share)")],
                "Balance Sheet": [c for c in cols if c in (
                    "Total assets","Total debt","Net Debt","currentRatio",
                    "debtEquityRatio","Total shareholders equity")],
                "Cash Flow": [c for c in cols if c in (
                    "Free Cash Flow","Operating Cash Flow",
                    "Free Cash Flow Yield","Free Cash Flow per Share")],
            }
            lines.append("\nShared column groups (same in all years):")
            for group, gcols in groups.items():
                if gcols:
                    lines.append(f"  {group}: {gcols}")

        return "\n".join(lines)

    def recent_history_text(self, n: int = 6) -> str:
        """Last n turns formatted as a string for LLM context."""
        recent = self.chat_history[-n:]
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)

    def snapshot_namespace_keys(self) -> List[str]:
        """Return non-private variable names in the exec namespace."""
        return [k for k in self.namespace if not k.startswith("_")]
