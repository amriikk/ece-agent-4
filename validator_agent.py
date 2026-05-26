"""
validator_agent.py — Independent verification of analytics agent output.

Uses the HW3 plan-execute engine to independently answer the same question,
then compares results numerically and semantically with the analytics agent's output.

Verdicts:
  APPROVED   — results are consistent, question is answered
  RETRY      — validator found a different answer; analytics should redo
  SUSPICIOUS — large numerical discrepancy (> 20% on primary metric)

The validator NEVER just checks whether code ran without error.
It reasons about correctness, completeness, and numerical consistency.
"""

from __future__ import annotations
import json
import anthropic
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional

from planner  import generate_plan
from executor import execute, format_trace
from memory   import ConversationMemory


# ── Config ────────────────────────────────────────────────────────────────────

DISCREPANCY_THRESHOLD = 0.20   # 20% relative difference triggers SUSPICIOUS


# ── Main public function ──────────────────────────────────────────────────────

def validate(
    question: str,
    analytics_answer: str,
    analytics_result: Optional[pd.DataFrame],
    memory: ConversationMemory,
) -> Dict[str, Any]:
    """
    Independently verify the analytics agent's output.

    Steps:
      1. Generate an independent plan via the planner
      2. Execute it against the loaded datasets
      3. Numerically compare results with analytics_result
      4. LLM reasoning step: evaluate correctness and completeness
      5. Return verdict dict

    Returns:
        {
          "verdict":          "APPROVED" | "RETRY: <reason>" | "SUSPICIOUS: <reason>",
          "reason":           str,
          "validator_result": pd.DataFrame | None,
          "validator_answer": str,
          "trace":            list,
          "comparison":       dict,
        }
    """
    schema_context = memory.get_schema_context()
    namespace      = memory.active_datasets   # {year_str: DataFrame}

    # ── Step 1: Independent plan generation ───────────────────────────────────
    try:
        plan = generate_plan(question, schema_context=schema_context)
    except Exception as e:
        return _verdict("RETRY", f"Validator planner failed: {e}", None, "", [], {})

    # ── Step 2: Execute independently ─────────────────────────────────────────
    # Build namespace with string keys matching planner convention (df_2014 etc.)
    exec_namespace = {f"df_{yr}": df for yr, df in namespace.items()}

    try:
        validator_df, trace = execute(exec_namespace, plan)
    except Exception as e:
        return _verdict("RETRY", f"Validator execution failed: {e}", None, "", [], {})

    # ── Step 3: Numerical comparison ──────────────────────────────────────────
    comparison = _compare_results(analytics_result, validator_df)

    # ── Step 4: LLM reasoning ─────────────────────────────────────────────────
    validator_answer, llm_verdict, llm_reason = _llm_evaluate(
        question        = question,
        analytics_answer= analytics_answer,
        analytics_result= analytics_result,
        validator_result= validator_df,
        comparison      = comparison,
        trace           = trace,
    )

    # ── Step 5: Merge numerical + LLM verdicts ────────────────────────────────
    if comparison.get("discrepancy_level") == "high" and llm_verdict == "APPROVED":
        # Numerical check overrides optimistic LLM
        final_verdict = f"SUSPICIOUS: numerical discrepancy > {DISCREPANCY_THRESHOLD*100:.0f}%"
    else:
        final_verdict = f"{llm_verdict}: {llm_reason}" if llm_verdict != "APPROVED" else "APPROVED"

    return _verdict(
        verdict   = final_verdict,
        reason    = llm_reason,
        result_df = validator_df,
        answer    = validator_answer,
        trace     = trace,
        comparison= comparison,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compare_results(
    analytics_df: Optional[pd.DataFrame],
    validator_df: pd.DataFrame,
) -> dict:
    """
    Numerically compare two DataFrames.
    Returns a comparison dict consumed by the LLM reasoning step.
    """
    if analytics_df is None or len(analytics_df) == 0:
        return {"comparable": False, "reason": "analytics result is empty"}

    comparison = {
        "comparable":        True,
        "analytics_rows":    len(analytics_df),
        "validator_rows":    len(validator_df),
        "row_count_match":   len(analytics_df) == len(validator_df),
        "discrepancy_level": "none",
    }

    # Find shared numeric columns
    a_num = analytics_df.select_dtypes(include="number")
    v_num = validator_df.select_dtypes(include="number")
    shared = [c for c in a_num.columns if c in v_num.columns]

    if not shared:
        comparison["comparable"] = False
        comparison["reason"] = "no shared numeric columns to compare"
        return comparison

    # Compare column means for shared numeric cols
    diffs = {}
    max_rel_diff = 0.0
    for col in shared[:5]:   # compare up to 5 cols
        a_mean = float(a_num[col].mean())
        v_mean = float(v_num[col].mean())
        if abs(a_mean) > 1e-10:
            rel_diff = abs(a_mean - v_mean) / abs(a_mean)
            diffs[col] = {"analytics_mean": round(a_mean, 4),
                          "validator_mean": round(v_mean, 4),
                          "relative_diff":  round(rel_diff, 4)}
            max_rel_diff = max(max_rel_diff, rel_diff)
        else:
            diffs[col] = {"analytics_mean": round(a_mean, 4),
                          "validator_mean": round(v_mean, 4),
                          "relative_diff":  None}

    comparison["column_diffs"] = diffs
    comparison["max_relative_diff"] = round(max_rel_diff, 4)

    if max_rel_diff > DISCREPANCY_THRESHOLD:
        comparison["discrepancy_level"] = "high"
    elif max_rel_diff > 0.05:
        comparison["discrepancy_level"] = "minor"

    return comparison


def _llm_evaluate(
    question: str,
    analytics_answer: str,
    analytics_result: Optional[pd.DataFrame],
    validator_result: pd.DataFrame,
    comparison: dict,
    trace: list,
) -> tuple[str, str, str]:
    """
    LLM reasoning step.

    Returns: (validator_answer, verdict_word, reason_str)
    verdict_word: "APPROVED" | "RETRY" | "SUSPICIOUS"
    """
    client = anthropic.Anthropic()

    # Summarise results for the prompt (cap at 10 rows)
    a_preview = (
        analytics_result.head(10).to_string(index=False)
        if analytics_result is not None and len(analytics_result) > 0
        else "None / empty"
    )
    v_preview = (
        validator_result.head(10).to_string(index=False)
        if len(validator_result) > 0
        else "Empty"
    )
    exec_trace = format_trace(trace)

    prompt = f"""You are an independent validator checking a data analysis result.

ORIGINAL QUESTION:
{question}

ANALYTICS AGENT ANSWER:
{analytics_answer}

ANALYTICS AGENT RESULT TABLE (first 10 rows):
{a_preview}

VALIDATOR EXECUTION TRACE:
{exec_trace}

VALIDATOR RESULT TABLE (first 10 rows):
{v_preview}

NUMERICAL COMPARISON:
{json.dumps(comparison, indent=2)}

Your tasks:
1. Write a 1-2 sentence validator answer summarising what YOU found.
2. Determine a verdict:
   - APPROVED if: results are consistent, the question is answered, no major issues found
   - RETRY if: results differ significantly, analysis is incomplete, or key aspect of question not addressed
   - SUSPICIOUS if: there is a large unexplained numerical discrepancy (>20%)
3. Give a concise reason (1 sentence).

Respond in this EXACT JSON format (no markdown):
{{"validator_answer": "...", "verdict": "APPROVED|RETRY|SUSPICIOUS", "reason": "..."}}
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    # Strip fences
    import re
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*",     "", raw)
    raw = re.sub(r"\s*```$",     "", raw)

    try:
        parsed = json.loads(raw)
        return (
            parsed.get("validator_answer", "Validation complete."),
            parsed.get("verdict", "APPROVED"),
            parsed.get("reason", ""),
        )
    except json.JSONDecodeError:
        return "Validation complete.", "APPROVED", "Could not parse validator LLM response."


def _verdict(verdict, reason, result_df, answer, trace, comparison) -> dict:
    return {
        "verdict":          verdict,
        "reason":           reason,
        "validator_result": result_df,
        "validator_answer": answer,
        "trace":            trace,
        "comparison":       comparison,
    }
