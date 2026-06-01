"""
analytics_agent.py — Iterative analytics sandbox with persistent exec namespace.

Unlike ECE-A2's one-shot code generation, this agent:
  1. Auto-inspects datasets on the first iteration
  2. Reasons about what to do next after each observation
  3. Executes code in a persistent namespace (variables survive across iterations)
  4. Extracts Plotly figures automatically
  5. Decides when to stop (DONE) based on whether the question is answered

MAX_ITERATIONS = 4 per query (safety cap)
"""

from __future__ import annotations
import io
import json
import re
import sys
import traceback
import anthropic
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Any, Dict, List, Optional, Tuple

from memory import ConversationMemory


# ── Config ────────────────────────────────────────────────────────────────────

MAX_ITERATIONS = 4

# Libraries pre-injected into every exec() call so generated code never imports them
EXEC_GLOBALS = {
    "pd":  pd,
    "np":  np,
    "px":  px,
    "go":  go,
    "json": json,
}

# ── Auto-inspection code run on iteration 0 ───────────────────────────────────

AUTO_INSPECT_CODE = """
# Auto-inspection — runs before the first user-driven iteration
_inspection_lines = []
for _varname, _df in [(k,v) for k,v in vars().items() if isinstance(v, __import__('pandas').DataFrame) and k.startswith('df_')]:
    _inspection_lines.append(f"=== {_varname} ===")
    _inspection_lines.append(f"Shape: {_df.shape}")
    _inspection_lines.append(f"Columns (first 20): {list(_df.columns[:20])}")
    if 'Sector' in _df.columns:
        _inspection_lines.append(f"Sectors: {_df['Sector'].value_counts().to_dict()}")
    if 'Class' in _df.columns:
        _inspection_lines.append(f"Class dist: {_df['Class'].value_counts().to_dict()}")
    num_cols = _df.select_dtypes('number')
    _inspection_lines.append(f"Numeric cols: {len(num_cols.columns)}, null%: {(num_cols.isna().mean()*100).round(1).to_dict()}")
_auto_inspection = '\\n'.join(_inspection_lines)
print(_auto_inspection)
"""

# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert financial data analyst working inside an interactive Python sandbox.

You have access to a persistent Python namespace. Variables you create in one iteration are available in the next.

Pre-loaded in scope:
  pd, np, px (plotly.express), go (plotly.graph_objects), json
  df_YEAR  — one DataFrame per loaded year (e.g. df_2014, df_2017)

Rules:
1. Output ONLY valid JSON in this format:
   {"action": "CODE", "reasoning": "...", "code": "..."}
   OR
   {"action": "DONE", "reasoning": "...", "final_answer": "..."}

2. action=CODE means you want to run more Python. Put code in "code".
3. action=DONE means you have enough information to give a complete final answer.
4. reasoning: explain in 1-2 sentences what you plan to do or what you concluded.
5. Code must be valid Python. No markdown fences inside the JSON.
6. For visualisations: use px or go, assign figure to a variable ending in _fig
   (e.g. trend_fig = px.line(...)). The system will auto-extract all _fig variables.
   NEVER call fig.show(), fig.write_html(), or fig.write_image() — these open browser tabs.
   Just assign: my_fig = px.bar(...) and stop. The framework handles display.
7. Always assign your final analysis result to a variable named `result` (DataFrame or scalar).
8. Do NOT try to import libraries — pd, np, px, go are already available.
9. Be specific with column names — they are case-sensitive.
10. For multi-year analysis, use variables like df_2014, df_2015 etc. and pd.concat().
11. IMPORTANT: Once you have computed the result AND created a visualisation, output action=DONE immediately.
    Do NOT create multiple charts in separate iterations. One chart per answer is enough.
    Do NOT keep running CODE after a chart has been created.
"""

ITERATION_PROMPT_TEMPLATE = """
## Loaded Datasets
{schema_context}

## Recent Conversation
{recent_history}

## Question to Answer
{question}

## Namespace (variables available right now)
{namespace_keys}

## Iteration History
{iteration_history}

## Current Observation (from last code run)
{observation}

What do you want to do next? Reply in JSON only.
"""


# ── Main public function ──────────────────────────────────────────────────────

def run(question: str, memory: ConversationMemory) -> Dict[str, Any]:
    """
    Iterative analytics loop.

    Returns:
        {
          "answer":  str,
          "plots":   List[{"title": str, "plotly_json": dict}],
          "trace":   List[{"n": int, "reasoning": str, "code": str, "observation": str}],
          "result":  pd.DataFrame | None,
        }
    """
    namespace = memory.namespace   # shared, persistent across calls
    plots: List[Dict] = []
    traces: List[Dict] = []
    final_answer = ""

    # Clear any _fig variables left over from previous queries so stale
    # charts don't get re-extracted and show up as duplicates
    stale_figs = [k for k in list(namespace.keys()) if k.endswith("_fig")]
    for k in stale_figs:
        del namespace[k]

    # ── Iteration 0: auto-inspect ─────────────────────────────────────────────
    observation = _exec_code(AUTO_INSPECT_CODE, namespace)
    traces.append({
        "n": 0,
        "reasoning": "Auto-inspecting loaded datasets before analysis.",
        "code": AUTO_INSPECT_CODE.strip(),
        "observation": observation,
    })
    memory.add_iteration_trace(0, "Auto-inspection", AUTO_INSPECT_CODE.strip(), observation)

    # ── Iterations 1 → MAX_ITERATIONS ─────────────────────────────────────────
    client = anthropic.Anthropic()

    for iteration in range(1, MAX_ITERATIONS + 1):
        # Build prompt
        iteration_history_text = _format_iteration_history(traces)
        prompt = ITERATION_PROMPT_TEMPLATE.format(
            schema_context     = memory.get_schema_context(),
            recent_history     = memory.recent_history_text(n=4),
            question           = question,
            namespace_keys     = memory.snapshot_namespace_keys(),
            iteration_history  = iteration_history_text,
            observation        = observation,
        )

        # Call LLM
        try:
            response = client.messages.create(
                model      = "claude-sonnet-4-6",
                max_tokens = 1500,
                system     = SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
        except Exception as e:
            observation = f"LLM call failed: {e}"
            traces.append({"n": iteration, "reasoning": "LLM error", "code": "", "observation": observation})
            break

        # Parse LLM response
        parsed, parse_error = _parse_response(raw)
        if parse_error:
            observation = f"Parse error: {parse_error}\nRaw: {raw[:300]}"
            traces.append({"n": iteration, "reasoning": "Parse error", "code": raw, "observation": observation})
            continue

        reasoning    = parsed.get("reasoning", "")
        action       = parsed.get("action", "CODE")
        code         = parsed.get("code", "")
        final_answer = parsed.get("final_answer", "")

        # ── DONE ──────────────────────────────────────────────────────────────
        if action == "DONE":
            traces.append({
                "n": iteration,
                "reasoning": reasoning,
                "code": "",
                "observation": "Agent decided analysis is complete.",
            })
            memory.add_iteration_trace(iteration, reasoning, "", "DONE")
            break

        # ── CODE ──────────────────────────────────────────────────────────────
        observation = _exec_code(code, namespace)

        # Extract any Plotly figures created (variables ending in _fig)
        new_plots = _extract_plots(namespace)
        plots.extend(new_plots)

        traces.append({
            "n": iteration,
            "reasoning": reasoning,
            "code": code,
            "observation": observation,
        })
        memory.add_iteration_trace(iteration, reasoning, code, observation)

    # ── If agent never said DONE, generate answer from last observation ────────
    if not final_answer:
        final_answer = _generate_fallback_answer(question, traces, client)

    # ── Extract final result DataFrame ────────────────────────────────────────
    result_df = namespace.get("result")
    if isinstance(result_df, pd.DataFrame):
        memory.last_result = result_df
    else:
        memory.last_result = None

    return {
        "answer": final_answer,
        "plots":  plots,
        "trace":  traces,
        "result": memory.last_result,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _exec_code(code: str, namespace: dict) -> str:
    """
    Execute code in the persistent namespace.
    Captures stdout + any exception. Returns observation string.
    """
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf

    exec_ns = {**EXEC_GLOBALS, **namespace}

    # Strip any fig.show() / fig.write_html() calls that would open browser tabs
    code = re.sub(r'\bfig\.show\s*\([^)]*\)', '# fig.show() blocked', code)
    code = re.sub(r'\w+_fig\.show\s*\([^)]*\)', '# fig.show() blocked', code)
    code = re.sub(r'\.write_html\s*\([^)]*\)', '# write_html blocked', code)
    code = re.sub(r'\.write_image\s*\([^)]*\)', '# write_image blocked', code)

    try:
        exec(code, exec_ns)  # noqa: S102
        # Sync any new variables back into the shared namespace
        for k, v in exec_ns.items():
            if not k.startswith("_") and k not in EXEC_GLOBALS:
                namespace[k] = v
        output = buf.getvalue()
        return output.strip() if output.strip() else "Code executed successfully (no output)."
    except Exception:
        err = traceback.format_exc()
        return f"EXECUTION ERROR:\n{err}"
    finally:
        sys.stdout = old_stdout


def _parse_response(raw: str) -> Tuple[dict, Optional[str]]:
    """
    Strip markdown fences and parse JSON from LLM output.
    Also handles LLMs that embed literal newlines inside JSON string values,
    which causes standard json.loads to fail with 'Extra data' or 'Invalid control'.
    """
    clean = re.sub(r"^```json\s*", "", raw.strip())
    clean = re.sub(r"^```\s*",     "", clean)
    clean = re.sub(r"\s*```$",     "", clean).strip()

    # Try direct parse first
    try:
        return json.loads(clean), None
    except json.JSONDecodeError:
        pass

    # Fallback: find the outermost {...} block and parse that
    try:
        start = clean.index("{")
        end   = clean.rindex("}") + 1
        return json.loads(clean[start:end]), None
    except (ValueError, json.JSONDecodeError) as e:
        return {}, str(e)


def _extract_plots(namespace: dict) -> List[Dict]:
    """
    Extract Plotly figures from namespace (variables ending in _fig).
    Returns at most ONE plot — the most recently created one.
    Multiple charts per answer clutters the UI; the agent should pick one chart type.
    Iterates over list() copy to avoid "dictionary changed size during iteration".
    """
    candidates = []
    to_delete  = []

    for k, v in list(namespace.items()):
        if k.endswith("_fig") and hasattr(v, "to_json"):
            try:
                fig_json = json.loads(v.to_json())
                title = (
                    v.layout.title.text
                    if hasattr(v, "layout") and v.layout.title and v.layout.title.text
                    else k.replace("_fig", "").replace("_", " ").title()
                )
                candidates.append({"title": title, "plotly_json": fig_json})
                to_delete.append(k)
            except Exception:
                pass

    for k in to_delete:
        del namespace[k]

    # Return only the last figure — it's the most refined version
    return candidates[-1:] if candidates else []


def _format_iteration_history(traces: List[Dict]) -> str:
    if not traces:
        return "No iterations yet."
    lines = []
    for t in traces:
        lines.append(f"--- Iteration {t['n']} ---")
        lines.append(f"Reasoning: {t['reasoning']}")
        if t["code"]:
            code_preview = t["code"][:400] + ("..." if len(t["code"]) > 400 else "")
            lines.append(f"Code:\n{code_preview}")
        obs_preview = t["observation"][:500] + ("..." if len(t["observation"]) > 500 else "")
        lines.append(f"Observation:\n{obs_preview}")
    return "\n".join(lines)


def _generate_fallback_answer(
    question: str,
    traces: List[Dict],
    client: anthropic.Anthropic,
) -> str:
    """If the agent hit MAX_ITERATIONS without saying DONE, synthesise an answer."""
    last_obs = traces[-1]["observation"] if traces else "No observations."
    prompt = (
        f"You were analysing financial data to answer: '{question}'\n\n"
        f"After {len(traces)} iterations, the final observation was:\n{last_obs}\n\n"
        f"Based on all observations so far, write a concise, specific final answer. "
        f"Include actual numbers where available."
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()