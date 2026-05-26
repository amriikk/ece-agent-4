"""
operators.py — Deterministic DataFrame operators used by the validator agent.

HW3 operators kept as-is:
  filter_rows, group_and_aggregate, sort_rows, limit_rows,
  select_columns, derive_columns, distinct_rows

New operators for HW4 multi-table support:
  add_year_column, join_tables, union_tables
"""

import pandas as pd
import numpy as np
from typing import List, Optional


# ── HW3 Operators (unchanged) ─────────────────────────────────────────────────

def derive_columns(df: pd.DataFrame, derive: list) -> pd.DataFrame:
    df = df.copy()
    for spec in derive:
        new_col = spec["new_column"]
        dtype   = spec["type"]

        if dtype == "arithmetic":
            left  = _resolve_operand(df, spec["left"])
            right = _resolve_operand(df, spec["right"])
            op    = spec["operation"]
            if op == "add":
                df[new_col] = left + right
            elif op == "subtract":
                df[new_col] = left - right
            elif op == "multiply":
                df[new_col] = left * right
            elif op == "divide":
                if isinstance(right, pd.Series):
                    df[new_col] = left / right.replace(0, np.nan)
                else:
                    df[new_col] = left / right if right != 0 else np.nan

        elif dtype == "date_diff":
            col_a = pd.to_datetime(df[spec["start"]], errors="coerce")
            col_b = pd.to_datetime(df[spec["end"]],   errors="coerce")
            df[new_col] = (col_b - col_a).dt.days

        elif dtype == "extract_date_part":
            col  = pd.to_datetime(df[spec["column"]], errors="coerce")
            part = spec["part"]
            df[new_col] = getattr(col.dt, part)

    return df


def _resolve_operand(df: pd.DataFrame, operand: dict):
    if operand["type"] == "column":
        return pd.to_numeric(df[operand["value"]], errors="coerce")
    elif operand["type"] == "literal":
        return operand["value"]


def filter_rows(df: pd.DataFrame, conditions: list) -> pd.DataFrame:
    df   = df.copy()
    mask = pd.Series([True] * len(df), index=df.index)

    for cond in conditions:
        col = df[cond["column"]]
        val = cond["value"]
        op  = cond["operator"]

        if op in [">", "<", ">=", "<="]:
            col = pd.to_numeric(col, errors="coerce")
            val = float(val)

        if   op == ">":        mask &= col > val
        elif op == "<":        mask &= col < val
        elif op == ">=":       mask &= col >= val
        elif op == "<=":       mask &= col <= val
        elif op == "==":       mask &= col == val
        elif op == "!=":       mask &= col != val
        elif op == "contains": mask &= col.astype(str).str.contains(val, case=False, na=False)

    return df[mask]


def group_and_aggregate(df: pd.DataFrame, group_by: list, metrics: list) -> pd.DataFrame:
    df = df.copy()
    agg_dict    = {}
    rename_dict = {}

    for m in metrics:
        col   = m["column"]
        func  = m["function"]
        alias = m["as"]

        if func != "count":
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if col not in agg_dict:
            agg_dict[col] = []
        agg_dict[col].append(func)
        rename_dict[(col, func)] = alias

    result = df.groupby(group_by).agg(agg_dict).reset_index()
    result.columns = [
        rename_dict.get(col, col[0] if isinstance(col, tuple) else col)
        for col in result.columns
    ]
    return result


def sort_rows(df: pd.DataFrame, sort_by: list) -> pd.DataFrame:
    columns   = [s["column"] for s in sort_by]
    ascending = [s["direction"] == "asc" for s in sort_by]
    df = df.copy()
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(
        by=columns, ascending=ascending, na_position="last"
    ).reset_index(drop=True)


def limit_rows(df: pd.DataFrame, k: int) -> pd.DataFrame:
    return df.head(k).reset_index(drop=True)


def select_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return df[columns]


def distinct_rows(df: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
    return df.drop_duplicates(subset=columns).reset_index(drop=True)


# ── HW4 Multi-Table Operators (new) ──────────────────────────────────────────

def add_year_column(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Tags a DataFrame with df['Year'] = year.
    Called before union_tables so temporal context is preserved.
    """
    df = df.copy()
    df["Year"] = int(year)
    return df


def join_tables(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    on: list,
    how: str = "inner",
    suffixes: Optional[list] = None,
) -> pd.DataFrame:
    """
    Join two DataFrames on shared key columns.

    Args:
        left_df:  left DataFrame
        right_df: right DataFrame
        on:       list of column names to join on (e.g. ["Ticker"])
        how:      "inner" | "left" | "right" | "outer"
        suffixes: column-name suffixes for overlapping columns
                  (default: ["_left", "_right"])
    """
    sfx = suffixes if suffixes else ["_left", "_right"]
    return pd.merge(left_df, right_df, on=on, how=how, suffixes=sfx)


def union_tables(
    dfs: List[pd.DataFrame],
    add_year_labels: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Vertically concatenate a list of DataFrames.

    Args:
        dfs:             list of DataFrames (must share key columns)
        add_year_labels: if provided, tags each df with df['Year'] = label
                         before stacking (e.g. [2014, 2015, 2016])
    """
    if add_year_labels is not None:
        if len(add_year_labels) != len(dfs):
            raise ValueError("add_year_labels must have same length as dfs")
        tagged = []
        for df, yr in zip(dfs, add_year_labels):
            d = df.copy()
            d["Year"] = int(yr)
            tagged.append(d)
        dfs = tagged

    return pd.concat(dfs, ignore_index=True)


# ── Registry ──────────────────────────────────────────────────────────────────

OPERATORS = {
    # HW3
    "derive_columns":      derive_columns,
    "filter_rows":         filter_rows,
    "group_and_aggregate": group_and_aggregate,
    "sort_rows":           sort_rows,
    "limit_rows":          limit_rows,
    "select_columns":      select_columns,
    "distinct_rows":       distinct_rows,
    # HW4
    "add_year_column":     add_year_column,
    "join_tables":         join_tables,
    "union_tables":        union_tables,
}


def get_operator(name: str):
    if name not in OPERATORS:
        raise ValueError(f"Unknown operator: '{name}'. Available: {list(OPERATORS)}")
    return OPERATORS[name]
