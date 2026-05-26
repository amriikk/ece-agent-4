"""
executor.py — Deterministic plan executor used by the validator agent.

Takes a plan dict {steps: [...]} and a namespace dict {var_name: DataFrame}
and executes each step against the evolving current DataFrame.

Multi-table steps (join_tables, union_tables) reference named DataFrames
from the namespace rather than just the running current table.
"""

import pandas as pd
from operators import OPERATORS


def execute(df_or_namespace, plan: dict):
    """
    Execute a plan.

    Args:
        df_or_namespace:
            - If a DataFrame: single-table mode (HW3 compatible)
            - If a dict: multi-table mode; keys are variable names (e.g. "df_2014")
              and the plan's first step specifies which one to start from via
              step["source"] (defaults to the first value if omitted).
        plan: {"steps": [...]}

    Returns:
        result_df: final DataFrame
        trace:     list of step dicts with row counts
    """
    trace = []

    # ── Resolve starting DataFrame ────────────────────────────────────────────
    if isinstance(df_or_namespace, dict):
        namespace = df_or_namespace
        # Use "source" key from first step, or fall back to first df in namespace
        first_step = plan["steps"][0] if plan["steps"] else {}
        source_key = first_step.get("source")
        if source_key and source_key in namespace:
            current = namespace[source_key].copy()
        else:
            # Default: first DataFrame value in the namespace
            dfs = [v for v in namespace.values() if isinstance(v, pd.DataFrame)]
            if not dfs:
                raise ValueError("No DataFrames found in namespace")
            current = dfs[0].copy()
    else:
        namespace = {}
        current = df_or_namespace.copy()

    # ── Execute each step ─────────────────────────────────────────────────────
    for i, step in enumerate(plan["steps"]):
        op_name = step["op"]

        if op_name not in OPERATORS:
            raise ValueError(f"Step {i+1}: Unknown operator '{op_name}'")

        params = {k: v for k, v in step.items() if k not in ("op", "source")}
        input_rows = len(current)

        # ── Multi-table operators need extra DataFrames from namespace ─────────
        if op_name == "join_tables":
            right_key = params.pop("right_df")
            if right_key not in namespace:
                raise ValueError(
                    f"Step {i+1} join_tables: '{right_key}' not in namespace. "
                    f"Available: {list(namespace.keys())}"
                )
            current = OPERATORS[op_name](current, namespace[right_key], **params)

        elif op_name == "union_tables":
            # params["sources"] = list of var names to union
            sources = params.pop("sources", [])
            dfs_to_union = []
            for src in sources:
                if src == "__current__":
                    dfs_to_union.append(current)
                elif src in namespace:
                    dfs_to_union.append(namespace[src])
                else:
                    raise ValueError(f"Step {i+1} union_tables: '{src}' not in namespace")
            current = OPERATORS[op_name](dfs_to_union, **params)

        elif op_name == "add_year_column":
            current = OPERATORS[op_name](current, **params)

        else:
            # Standard single-table operator
            current = OPERATORS[op_name](current, **params)

        trace.append({
            "step":        i + 1,
            "op":          op_name,
            "input_rows":  input_rows,
            "output_rows": len(current),
            "output_cols": list(current.columns),
        })

        if len(current) == 0:
            print(f"  ⚠️  Step {i+1} ({op_name}) produced an empty table")

    return current, trace


def format_trace(trace: list) -> str:
    """Pretty-print execution trace."""
    header  = f"{'Step':<6} {'Operation':<25} {'Input Rows':<14} {'Output Rows':<12}"
    divider = "-" * len(header)
    lines   = [header, divider]
    for t in trace:
        lines.append(
            f"{t['step']:<6} {t['op']:<25} {t['input_rows']:<14} {t['output_rows']:<12}"
        )
    return "\n".join(lines)
