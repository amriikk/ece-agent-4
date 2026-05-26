"""
orchestrator.py — LangGraph supervisor that routes and coordinates all agents.

Flow:
  classify → [analytics | generic]
  analytics → validate → [retry | respond]
  generic   → web_search → respond

Emits SSE-compatible events into state["sse_events"] which main.py streams.
"""

from __future__ import annotations
import json
import anthropic
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from memory          import ConversationMemory
from analytics_agent import run as run_analytics
from validator_agent import validate
from web_search      import run as run_web_search


# ── Config ────────────────────────────────────────────────────────────────────

MAX_RETRIES = 2   # max validator-triggered retries per query


# ── State ─────────────────────────────────────────────────────────────────────

class OrchestratorState(TypedDict):
    query:            str
    query_type:       Optional[str]     # "analytics" | "generic"
    memory:           ConversationMemory

    # Analytics path
    analytics_output: Optional[Dict[str, Any]]
    validator_output: Optional[Dict[str, Any]]
    retry_count:      int

    # Generic path
    web_output:       Optional[Dict[str, Any]]

    # Final
    final_response:   Optional[Dict[str, Any]]

    # SSE event queue — list of JSON-serialisable dicts
    sse_events:       List[Dict[str, Any]]


# ── SSE event helpers ─────────────────────────────────────────────────────────

def _emit(state: OrchestratorState, event: dict) -> None:
    """Append an SSE event to the queue."""
    state["sse_events"].append(event)


def _progress(state, message: str) -> None:
    _emit(state, {"type": "progress", "message": message})


def _iteration_event(state, n: int, reasoning: str, code: str, observation: str) -> None:
    _emit(state, {
        "type":        "iteration",
        "n":           n,
        "reasoning":   reasoning,
        "code":        code[:600],    # cap for payload size
        "observation": observation[:600],
    })


def _validator_event(state, verdict: str, reason: str) -> None:
    _emit(state, {"type": "validator", "verdict": verdict, "reason": reason})


def _web_event(state, title: str, url: str, snippet: str) -> None:
    _emit(state, {"type": "web_result", "title": title, "url": url, "snippet": snippet})


def _done_event(state, answer: str, plots: list, citations: list) -> None:
    _emit(state, {
        "type":      "done",
        "answer":    answer,
        "plots":     plots,
        "citations": citations,
    })


def _error_event(state, message: str) -> None:
    _emit(state, {"type": "error", "message": message})


# ── Node: classify ─────────────────────────────────────────────────────────────

def classify_node(state: OrchestratorState) -> OrchestratorState:
    """
    Determine whether the query is analytics-related or generic-domain.
    Analytics: requires dataset analysis, financial metrics, stock comparisons.
    Generic:   factual/conceptual questions answered by web search.
    """
    _progress(state, "⚙ Classifying intent...")

    query   = state["query"]
    memory  = state["memory"]
    client  = anthropic.Anthropic()

    has_datasets = bool(memory.active_datasets)

    prompt = (
        f'User query: "{query}"\n\n'
        f"Loaded datasets: {list(memory.active_datasets.keys()) or 'none'}\n\n"
        "Classify this query as ANALYTICS or GENERIC.\n\n"
        "ANALYTICS: requires analysis of the loaded financial dataset "
        "(stock metrics, sector comparisons, temporal trends, portfolio screening, "
        "any question referencing stocks, companies, returns, PE ratio, ROE, revenue, etc.)\n\n"
        "GENERIC: factual/conceptual question that can be answered by web search "
        "(what is X, why did Y happen, explain Z, history of W)\n\n"
        "Output ONLY one word: ANALYTICS or GENERIC"
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    decision = response.content[0].text.strip().upper()

    # If no datasets loaded, force generic regardless
    if not has_datasets and "ANALYTICS" in decision:
        decision = "GENERIC"
        _progress(state, "⚠ No datasets loaded — routing to web search.")

    state["query_type"] = "analytics" if "ANALYTICS" in decision else "generic"
    return state


# ── Node: analytics ────────────────────────────────────────────────────────────

def analytics_node(state: OrchestratorState) -> OrchestratorState:
    """
    Run the iterative analytics agent and stream per-iteration events.
    """
    query  = state["query"]
    memory = state["memory"]

    retry = state["retry_count"]
    if retry == 0:
        _progress(state, "🔬 Auto-inspecting datasets...")
    else:
        _progress(state, f"🔄 Re-running analysis (retry {retry}/{MAX_RETRIES})...")

    # Run analytics agent
    try:
        output = run_analytics(query, memory)
    except Exception as e:
        _error_event(state, f"Analytics agent failed: {e}")
        state["analytics_output"] = {
            "answer": f"Analysis failed: {e}",
            "plots":  [],
            "trace":  [],
            "result": None,
        }
        return state

    # Stream per-iteration events to the frontend
    for t in output["trace"]:
        if t["n"] == 0:
            continue   # auto-inspect already signalled above
        _progress(state, f"✍ Reasoning + coding (iteration {t['n']}/{6})...")
        _iteration_event(
            state,
            n           = t["n"],
            reasoning   = t["reasoning"],
            code        = t["code"],
            observation = t["observation"],
        )

    state["analytics_output"] = output
    return state


# ── Node: validate ─────────────────────────────────────────────────────────────

def validate_node(state: OrchestratorState) -> OrchestratorState:
    """
    Run the validator agent and emit its verdict.
    """
    _progress(state, "🧪 Validator independently checking results...")

    output  = state["analytics_output"]
    memory  = state["memory"]
    query   = state["query"]

    import pandas as pd
    result_df = output.get("result")
    if not isinstance(result_df, pd.DataFrame):
        result_df = None

    try:
        v_output = validate(
            question         = query,
            analytics_answer = output.get("answer", ""),
            analytics_result = result_df,
            memory           = memory,
        )
    except Exception as e:
        # Validator failure shouldn't crash the system
        v_output = {
            "verdict":  "APPROVED",
            "reason":   f"Validator errored ({e}); defaulting to APPROVED.",
            "validator_result": None,
            "validator_answer": "",
            "trace":    [],
            "comparison": {},
        }

    verdict = v_output["verdict"]
    reason  = v_output.get("reason", "")
    _validator_event(state, verdict, reason)
    memory.last_verdict = verdict

    state["validator_output"] = v_output
    return state


# ── Node: web_search ───────────────────────────────────────────────────────────

def web_search_node(state: OrchestratorState) -> OrchestratorState:
    """
    Run DuckDuckGo search and emit per-result events.
    """
    query = state["query"]
    _progress(state, f"🔍 Searching DuckDuckGo: {query[:60]}...")

    try:
        output = run_web_search(query)
    except Exception as e:
        _error_event(state, f"Web search failed: {e}")
        state["web_output"] = {
            "answer":    f"Web search failed: {e}",
            "citations": [],
        }
        return state

    for r in output.get("raw_results", []):
        _web_event(state, r.title, r.url, r.snippet)

    state["web_output"] = output
    return state


# ── Node: respond ──────────────────────────────────────────────────────────────

def respond_node(state: OrchestratorState) -> OrchestratorState:
    """
    Assemble the final response and emit the done event.
    """
    memory = state["memory"]

    if state["query_type"] == "generic":
        web = state.get("web_output", {})
        answer    = web.get("answer", "No answer found.")
        citations = web.get("citations", [])
        plots     = []
    else:
        analytics = state.get("analytics_output", {})
        answer    = analytics.get("answer", "Analysis complete.")
        plots     = analytics.get("plots", [])
        citations = []

    # Serialise plots for SSE (Plotly JSON is already a dict)
    safe_plots = []
    for p in plots:
        safe_plots.append({
            "title":       p.get("title", "Chart"),
            "plotly_json": p.get("plotly_json", {}),
        })

    _done_event(state, answer, safe_plots, citations)
    memory.add_assistant_message(answer)

    state["final_response"] = {
        "answer":    answer,
        "plots":     safe_plots,
        "citations": citations,
    }
    return state


# ── Routing functions ─────────────────────────────────────────────────────────

def _route_after_classify(state: OrchestratorState) -> str:
    return state["query_type"]   # "analytics" | "generic"


def _route_after_validate(state: OrchestratorState) -> str:
    verdict     = state["validator_output"]["verdict"]
    retry_count = state["retry_count"]

    if verdict.startswith("RETRY") and retry_count < MAX_RETRIES:
        state["retry_count"] += 1
        return "retry"
    return "respond"


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_graph() -> Any:
    g = StateGraph(OrchestratorState)

    g.add_node("classify",   classify_node)
    g.add_node("analytics",  analytics_node)
    g.add_node("validate",   validate_node)
    g.add_node("web_search", web_search_node)
    g.add_node("respond",    respond_node)

    g.set_entry_point("classify")

    g.add_conditional_edges(
        "classify",
        _route_after_classify,
        {"analytics": "analytics", "generic": "web_search"},
    )

    g.add_edge("analytics", "validate")

    g.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"retry": "analytics", "respond": "respond"},
    )

    g.add_edge("web_search", "respond")
    g.add_edge("respond",    END)

    return g.compile()


# Compile once at import time
orchestrator = build_graph()


# ── Public API ────────────────────────────────────────────────────────────────

def run(query: str, memory: ConversationMemory) -> OrchestratorState:
    """
    Entry point called by the FastAPI backend.
    Returns the final OrchestratorState including sse_events.
    """
    memory.add_user_message(query)

    initial_state: OrchestratorState = {
        "query":            query,
        "query_type":       None,
        "memory":           memory,
        "analytics_output": None,
        "validator_output": None,
        "retry_count":      0,
        "web_output":       None,
        "final_response":   None,
        "sse_events":       [],
    }

    return orchestrator.invoke(initial_state)
