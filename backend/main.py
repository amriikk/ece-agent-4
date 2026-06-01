"""
backend/main.py — FastAPI with direct SSE streaming.

Instead of running the full LangGraph orchestrator in a thread and
flushing all events at the end, this version calls each agent step
directly inside an async generator and yields SSE events in real-time.

"""

import asyncio
import json
import os
import re
import shutil
import sys
import traceback

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory import ConversationMemory

app = FastAPI(title="Datum HW4")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session memory ────────────────────────────────────────────────────────────
_memory = ConversationMemory()

DATASETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"
)
os.makedirs(DATASETS_DIR, exist_ok=True)


def _auto_load_datasets():
    for fname in sorted(os.listdir(DATASETS_DIR)):
        if not fname.endswith(".csv"):
            continue
        match = re.search(r"(\d{4})", fname)
        if not match:
            continue
        year = match.group(1)
        path = os.path.join(DATASETS_DIR, fname)
        try:
            _memory.load_dataset(path, year)
            print(f"  Auto-loaded: {fname} as df_{year}")
        except Exception as e:
            print(f"  Failed to load {fname}: {e}")


_auto_load_datasets()


# ── Request model ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[dict]] = []


# ── SSE helpers ───────────────────────────────────────────────────────────────
def _sse(event: dict) -> str:
    return f"data: {json.dumps(_safe(event))}\n\n"


def _safe(obj: Any) -> Any:
    import pandas as pd, numpy as np
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(i) for i in obj]
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.head(10).to_json(orient="records"))
    if isinstance(obj, pd.Series):
        return json.loads(obj.to_json())
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if "Dtype" in type(obj).__name__: return str(obj)
    if isinstance(obj, (int, float, str, bool, type(None))): return obj
    return str(obj)


# ── Classify query ────────────────────────────────────────────────────────────
def _classify(query: str, has_datasets: bool) -> str:
    """Returns 'analytics' or 'generic'."""
    if not has_datasets:
        return "generic"
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": (
            f'Query: "{query}"\n\n'
            "Is this about financial data analysis (stocks, sectors, ROE, revenue, "
            "companies, metrics, returns) or is it a general knowledge question?\n"
            "Output ONLY: ANALYTICS or GENERIC"
        )}],
    )
    return "analytics" if "ANALYTICS" in resp.content[0].text.upper() else "generic"


# ── Main streaming endpoint ───────────────────────────────────────────────────
@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):

    async def generate():
        query = request.query
        _memory.add_user_message(query)

        # ── Step 1: Classify ───────────────────────────────────────────────
        yield _sse({"type": "progress", "message": "⚙ Classifying intent..."})
        await asyncio.sleep(0)

        try:
            loop = asyncio.get_event_loop()
            query_type = await loop.run_in_executor(
                None, _classify, query, bool(_memory.active_datasets)
            )
        except Exception as e:
            yield _sse({"type": "error", "message": f"Classification failed: {e}"})
            return

        print(f"\n--- QUERY TYPE: {query_type} | '{query}' ---\n")

        # ── Generic path: web search ───────────────────────────────────────
        if query_type == "generic":
            yield _sse({"type": "progress", "message": f"🔍 Searching the web..."})
            await asyncio.sleep(0)
            try:
                from web_search import run as web_run
                result = await loop.run_in_executor(None, web_run, query)
                answer    = result.get("answer", "No results found.")
                citations = result.get("citations", [])
                for r in result.get("raw_results", []):
                    yield _sse({"type": "web_result",
                                "title": r.title, "url": r.url, "snippet": r.snippet})
                    await asyncio.sleep(0)
                yield _sse({"type": "progress", "message": "💬 Composing answer..."})
                yield _sse({"type": "done", "answer": answer,
                             "plots": [], "citations": citations})
                _memory.add_assistant_message(answer)
            except Exception as e:
                traceback.print_exc()
                yield _sse({"type": "error", "message": f"Web search failed: {e}"})
            return

        # ── Analytics path ─────────────────────────────────────────────────
        yield _sse({"type": "progress", "message": "🔬 Auto-inspecting datasets..."})
        await asyncio.sleep(0)

        analytics_output = None
        try:
            from analytics_agent import run as analytics_run

            # Run analytics agent in thread (it's synchronous + slow)
            def _run_analytics():
                return analytics_run(query, _memory)

            yield _sse({"type": "progress",
                        "message": "✍ Reasoning + writing analysis code..."})
            await asyncio.sleep(0)

            analytics_output = await loop.run_in_executor(None, _run_analytics)

            # Stream per-iteration trace events
            for t in analytics_output.get("trace", []):
                if t["n"] == 0:
                    continue
                yield _sse({"type": "iteration",
                             "n": t["n"],
                             "reasoning": t.get("reasoning", ""),
                             "code": t.get("code", "")[:600],
                             "observation": t.get("observation", "")[:600]})
                await asyncio.sleep(0)

        except Exception as e:
            traceback.print_exc()
            yield _sse({"type": "error",
                        "message": f"Analytics agent failed: {e}"})
            return

        # ── Validator ──────────────────────────────────────────────────────
        yield _sse({"type": "progress",
                    "message": "🧪 Validator independently checking results..."})
        await asyncio.sleep(0)

        validator_verdict = "APPROVED"
        retry_count       = 0
        MAX_RETRIES       = 1

        while True:
            try:
                import pandas as pd
                from validator_agent import validate as validator_run

                result_df = analytics_output.get("result")
                if not isinstance(result_df, pd.DataFrame):
                    result_df = None

                v_out = await loop.run_in_executor(
                    None, validator_run,
                    query,
                    analytics_output.get("answer", ""),
                    result_df,
                    _memory,
                )

                validator_verdict = v_out.get("verdict", "APPROVED")
                reason            = v_out.get("reason", "")

                yield _sse({"type": "validator",
                             "verdict": validator_verdict, "reason": reason})
                await asyncio.sleep(0)

                # Retry if validator says RETRY and we haven't exceeded limit
                if validator_verdict.startswith("RETRY") and retry_count < MAX_RETRIES:
                    retry_count += 1
                    yield _sse({"type": "progress",
                                "message": f"🔄 Re-running analysis (retry {retry_count})..."})
                    await asyncio.sleep(0)
                    analytics_output = await loop.run_in_executor(None, _run_analytics)
                else:
                    break

            except Exception as e:
                traceback.print_exc()
                # Validator failure is non-fatal — just log and continue
                print(f"Validator error: {e}")
                yield _sse({"type": "validator",
                             "verdict": "APPROVED",
                             "reason": f"Validator skipped: {e}"})
                break

        # ── Final response ─────────────────────────────────────────────────
        answer = analytics_output.get("answer", "Analysis complete.")
        plots  = [_safe(p) for p in analytics_output.get("plots", [])]

        yield _sse({"type": "progress", "message": "✅ Analysis complete"})
        yield _sse({"type": "done", "answer": answer,
                     "plots": plots, "citations": []})
        _memory.add_assistant_message(answer)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Upload ────────────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    fname = file.filename or "upload.csv"
    match = re.search(r"(\d{4})", fname)
    if not match:
        raise HTTPException(400, "Year not found in filename")
    year = match.group(1)
    dest = os.path.join(DATASETS_DIR, fname)
    with open(dest, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    try:
        df = _memory.load_dataset(dest, year)
    except Exception as e:
        raise HTTPException(500, f"Failed to load: {e}")
    return {"filename": fname, "year": year, "rows": len(df),
            "cols": len(df.columns), "loaded_years": list(_memory.active_datasets.keys())}


# ── Reset ─────────────────────────────────────────────────────────────────────
@app.post("/api/reset")
async def reset():
    _memory.chat_history.clear()
    _memory.reset_namespace()
    for year, df in _memory.active_datasets.items():
        _memory.namespace[f"df_{year}"] = df
    return {"status": "reset", "datasets": list(_memory.active_datasets.keys())}


# ── Datasets list ─────────────────────────────────────────────────────────────
@app.get("/api/datasets")
async def list_datasets():
    result = []
    for year, df in sorted(_memory.active_datasets.items()):
        result.append({"year": year, "rows": len(df), "cols": len(df.columns)})
    return {"datasets": result}


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok",
            "loaded_years": list(_memory.active_datasets.keys()),
            "history_turns": len(_memory.chat_history)}