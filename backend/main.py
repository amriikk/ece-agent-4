"""
backend/main.py — FastAPI gateway with SSE streaming for the HW4 orchestrated system.

Endpoints:
  POST /api/chat/stream  — primary SSE endpoint; streams per-event updates
  POST /api/upload       — upload a financial CSV (year detected from filename)
  POST /api/reset        — clear conversation memory and exec namespace
  GET  /api/datasets     — list currently loaded datasets
  GET  /api/health       — healthcheck
"""

import asyncio
import json
import os
import re
import shutil
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, List, Optional

# Project root on path so imports resolve regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory      import ConversationMemory
from orchestrator import run as orchestrator_run

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="DATUM HW4 — Orchestrated Analytics System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session memory (single-user; extend to per-session dict for multi-user) ───

_memory = ConversationMemory()

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

# Auto-load any CSVs already present in datasets/ at startup
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

# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    # chat_history sent from frontend for display only; memory object is authoritative
    chat_history: Optional[List[dict]] = []


# ── SSE streaming endpoint ────────────────────────────────────────────────────

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Run the full orchestrator pipeline and stream SSE events as they're generated.

    Event types (see ARCHITECTURE.md):
      progress, iteration, validator, web_result, done, error
    """
    async def event_generator():
        try:
            # Run orchestrator synchronously in a thread so it doesn't block event loop
            import concurrent.futures
            loop = asyncio.get_event_loop()

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = loop.run_in_executor(
                    pool,
                    orchestrator_run,
                    request.query,
                    _memory,
                )

                # Poll for completion while yielding a heartbeat every 0.5s
                # so the SSE connection stays alive
                while not future.done():
                    await asyncio.sleep(0.5)
                    yield ": heartbeat\n\n"

                final_state = await future

            # Stream all queued SSE events
            for event in final_state.get("sse_events", []):
                payload = json.dumps(_make_serialisable(event))
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0)

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_payload = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering":"no",
        },
    )


# ── Upload endpoint ───────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a financial CSV. Year is detected from filename (e.g. 2014_Financial_Data.csv).
    Saves to datasets/, resets exec namespace, loads into memory.
    """
    fname = file.filename or "upload.csv"
    match = re.search(r"(\d{4})", fname)
    if not match:
        raise HTTPException(
            status_code=400,
            detail="Cannot detect year from filename. Include year (e.g. 2014_Financial_Data.csv)."
        )
    year = match.group(1)
    dest = os.path.join(DATASETS_DIR, fname)

    with open(dest, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        df = _memory.load_dataset(dest, year)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")

    return {
        "filename":     fname,
        "year":         year,
        "rows":         len(df),
        "cols":         len(df.columns),
        "dataset_path": dest,
        "loaded_years": list(_memory.active_datasets.keys()),
    }


# ── Reset endpoint ────────────────────────────────────────────────────────────

@app.post("/api/reset")
async def reset_memory():
    """Clear conversation history and exec namespace. Datasets remain loaded."""
    _memory.chat_history.clear()
    _memory.reset_namespace()
    # Re-inject datasets into namespace
    for year, df in _memory.active_datasets.items():
        _memory.namespace[f"df_{year}"] = df
    return {"status": "reset", "datasets_preserved": list(_memory.active_datasets.keys())}


# ── Dataset list endpoint ─────────────────────────────────────────────────────

@app.get("/api/datasets")
async def list_datasets():
    """Return currently loaded datasets with basic stats."""
    result = []
    for year, df in sorted(_memory.active_datasets.items()):
        result.append({
            "year":    year,
            "rows":    len(df),
            "cols":    len(df.columns),
            "sectors": df["Sector"].nunique() if "Sector" in df.columns else 0,
        })
    return {"datasets": result}


# ── Healthcheck ───────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status":          "ok",
        "loaded_years":    list(_memory.active_datasets.keys()),
        "history_turns":   len(_memory.chat_history),
        "namespace_vars":  _memory.snapshot_namespace_keys(),
    }


# ── Serialisation helper ──────────────────────────────────────────────────────

def _make_serialisable(obj: Any) -> Any:
    """Recursively ensure an object is JSON-serialisable for SSE payloads."""
    import pandas as pd
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(i) for i in obj]
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.head(10).to_json(orient="records"))
    if isinstance(obj, pd.Series):
        return json.loads(obj.to_json())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "name") and "Dtype" in type(obj).__name__:
        return str(obj)
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)
