# HW4 Architecture — Orchestrated AI Analytics System

## Dataset Facts (inform every design decision)
- 5 CSV files: 2014–2018, each 225 columns, 3,808–4,960 rows
- Column 0 is ticker symbol (`Unnamed: 0` → renamed to `Ticker` on load)
- Schema is identical across years; only the price-var target column differs
  (`2015 PRICE VAR [%]`, `2016 PRICE VAR [%]`, …)
- 3,726 tickers appear in all 5 years → clean temporal join on `Ticker`
- `Sector` (10 values), `Class` (0/1 buy signal), ~1–8% null rate on numeric cols
- Key column groups: identity, income-statement, profitability-ratios,
  valuation, growth, balance-sheet, cash-flow, target

---

## Component Map

```
User Query (React frontend)
        │  SSE stream
        ▼
 backend/main.py   (FastAPI — SSE gateway)
        │
        ▼
 orchestrator.py   (LangGraph supervisor)
        │
   ┌────┴────────────────────┐
   │                         │
   ▼                         ▼
web_search.py        analytics_agent.py
(DuckDuckGo)         (iterative sandbox)
   │                         │
   │                    validator_agent.py
   │                    (HW3 plan-execute)
   │                         │
   └──────────┬──────────────┘
              ▼
        final output
   {answer, plots, trace, citations}
```

---

## File Responsibilities

### `memory.py`
Single source of truth for conversation state.
- `ConversationMemory` dataclass:
  - `chat_history`: List[dict] — `{role, content}` pairs
  - `active_datasets`: List[str] — paths currently loaded
  - `namespace`: dict — persistent Python exec namespace (analytics agent reuses this)
  - `last_result`: Any — last DataFrame/scalar from analytics agent
  - `last_validator_verdict`: str — `APPROVED | RETRY | SUSPICIOUS`
  - `iteration_traces`: List[dict] — per-iteration reasoning logs
- `reset_namespace()` — clears exec namespace on new dataset load
- `snapshot_namespace()` — returns JSON-safe summary for SSE stream

### `web_search.py`
Thin wrapper around LangChain's DuckDuckGoSearchRun.
- `search(query: str) -> List[SearchResult]`
  - `SearchResult`: `{title, url, snippet}`
- `synthesize_answer(question, results, llm) -> str`
  - calls Claude to write a grounded answer with inline citations
- Returns: `{answer: str, citations: List[{title, url}]}`

### `operators.py` (extended from HW3)
Deterministic DataFrame operators used by the validator.
New additions for multi-table support:
- `join_tables(left_df, right_df, on, how)` — inner/left/outer join
- `union_tables(dfs: List[DataFrame])` — vertical stack with `Year` column added
- `add_year_column(df, year: int)` — tags a loaded DataFrame with its year
Existing operators kept as-is: `filter_rows`, `group_and_aggregate`, `sort_rows`,
`limit_rows`, `select_columns`, `derive_columns`, `distinct_rows`

### `planner.py` (extended from HW3)
LLM planner — now financial-schema-aware.
- Updated `SYSTEM_PROMPT` with financial column groups and multi-table examples
- New operator docs for `join_tables`, `union_tables`, `add_year_column`
- `generate_plan(question, schema_context) -> dict`
  - `schema_context`: injected dynamically from loaded datasets

### `validator_agent.py`
Independent verification of analytics agent output.
Inputs: `question`, `analytics_answer`, `result_df`, `loaded_datasets`
Process:
  1. Independently generate a plan via `planner.py`
  2. Execute it via `executor.py`
  3. Compare its result with analytics_agent's result numerically
  4. LLM reasoning step: evaluate correctness + completeness
Outputs: `{verdict: APPROVED|RETRY|SUSPICIOUS, reason: str, validator_result_df}`
- `APPROVED`: results consistent, question answered
- `RETRY`: missing analysis, validator found different answer
- `SUSPICIOUS`: large numerical discrepancy (> 20% on key metrics)

### `analytics_agent.py`
Iterative code-generation + execution loop with persistent namespace.
```
for iteration in range(MAX_ITERATIONS=6):
    1. LLM reasons: what do I know so far? what should I do next?
    2. LLM outputs: {action: CODE|DONE, code: str, reasoning: str}
    3. if DONE: break
    4. exec(code, namespace)  ← persistent — variables survive across iterations
    5. observe: capture stdout, last assigned variable, any DataFrames
    6. feed observation back into next LLM context
    7. if plots detected: extract Plotly JSON
```
Auto-inspection on iteration 0:
  - Runs `df.shape`, `df.dtypes`, `df.describe()`, `df['Sector'].value_counts()`
  - Injects result into LLM context before first user-driven iteration
Stopping condition: LLM outputs `action: DONE` or MAX_ITERATIONS reached
Output: `{answer: str, plots: List[{title, plotly_json}], trace: List[dict]}`

### `orchestrator.py`
LangGraph `StateGraph` with these nodes:

```
classify → [route] → web_search_node → synthesize → respond
                   → analytics_node  → validate   → [retry_router] → respond
                                                   ↑______________|
```

State: `OrchestratorState` (TypedDict):
- `query`, `query_type` (analytics|generic), `memory`
- `analytics_output`, `validator_verdict`, `retry_count`
- `web_search_results`, `final_response`
- `sse_events`: List[str] — consumed by FastAPI SSE stream

Retry logic: if `RETRY` and `retry_count < 2`: loop back to analytics_node
Max retries: 2 (so max 3 total analytics runs per question)

### `backend/main.py`
FastAPI with SSE streaming. Endpoints:
- `POST /api/chat/stream` — primary, streams `OrchestratorState.sse_events`
- `POST /api/upload` — saves CSV, calls `memory.reset_namespace()`
- `POST /api/reset` — clears memory, resets namespace
- `GET /api/datasets` — returns list of loaded dataset filenames

SSE event types:
```json
{"type": "progress",   "message": "🔬 Inspecting datasets..."}
{"type": "iteration",  "n": 2, "reasoning": "...", "code": "..."}
{"type": "validator",  "verdict": "RETRY", "reason": "..."}
{"type": "web_result", "title": "...", "url": "...", "snippet": "..."}
{"type": "done",       "answer": "...", "plots": [...], "citations": [...]}
{"type": "error",      "message": "..."}
```

### `frontend/src/App.jsx`
Extended DATUM UI with two new panels:
1. **Agent Trace Panel** (collapsible, right side): shows per-iteration reasoning,
   code executed, observation, validator verdict
2. **Citation strip** (below web-search answers): linked source cards

---

## Data Flow for an Analytics Query

```
User: "Compare ROE of Technology vs Healthcare across 2015-2018"

orchestrator: classify → analytics
analytics_agent iteration 0 (auto-inspect):
  → runs df.dtypes, df['Sector'].value_counts() on each year
  → observes: 4 DataFrames loaded, Sector col present, ROE col present
analytics_agent iteration 1:
  → code: merge all years, filter Tech+Healthcare, groupby Sector+Year, mean ROE
  → observes: result DataFrame with 8 rows (2 sectors × 4 years)
analytics_agent iteration 2:
  → code: px.line(result, x='Year', y='ROE', color='Sector')
  → observes: fig created, extracts plotly_json
analytics_agent: action=DONE
validator_agent:
  → independently plans: union 4 years, filter, groupby, mean
  → executes, compares numbers with analytics result
  → verdict: APPROVED (values within 5%)
final output → SSE done event → React renders answer + line chart
```

---

## Key Design Decisions

### Why persistent namespace over fresh exec each time (HW2 approach)
HW2 ran fresh `exec()` per query. HW4 queries require multi-step analysis
where intermediate DataFrames (e.g. a merged 5-year table) are reused.
Persistent namespace means iteration 2 can reference `merged_df` created in
iteration 1 without re-reading CSV. This matches a real code-interpreter environment.

### Why validator uses the HW3 plan-execute engine instead of LLM-only checking
LLM-only validators hallucinate numeric results. By independently running
deterministic operators on the same data and comparing numbers, the validator
can detect real discrepancies rather than just reasoning about whether the
answer "sounds right."

### Why `union_tables` adds a `Year` column automatically
Without year tagging, a union of 5 DataFrames loses temporal context.
All validator and analytics code can then do `groupby(['Ticker','Year'])` cleanly.

### Why SSE iteration events (not just final answer)
The assignment requires showing "agent execution in real time." Each iteration
event carries `{n, reasoning, code, observation}` so the trace panel renders
the full decision chain as it happens — not reconstructed after the fact.

---

## Operator Extension for Multi-Table (HW3 → HW4)

```python
# New operators added to operators.py

def join_tables(left_df, right_df, on, how="inner"):
    """Join two DataFrames. 'on' is a list of column names."""

def union_tables(dfs, add_year_labels=None):
    """Vertical concat. add_year_labels=[2014,2015,...] tags each df."""

def add_year_column(df, year):
    """Tags df with df['Year'] = year. Used before union."""
```

Validator planner system prompt extended with:
- Multi-table schema context (both DataFrames' columns injected)
- Examples of join_tables and union_tables plans
- Cross-year temporal query examples

---

## SSE Progress Message Taxonomy

| Stage | Message |
|---|---|
| Classifying | `⚙ Classifying intent...` |
| Web search | `🔍 Searching DuckDuckGo for: {query}` |
| Auto-inspect | `🔬 Inspecting datasets (auto-explore)...` |
| Iteration N | `✍ Reasoning + coding (iteration {n}/{max})...` |
| Executing | `⚡ Executing code (iteration {n})...` |
| Observing | `👁 Observing results...` |
| Validator | `🧪 Validator checking results...` |
| Retry | `🔄 Validator requested retry ({n}/2) — {reason}` |
| Done | `✅ Analysis complete` |
| Error | `✕ {error message}` |
