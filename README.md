# DATUM HW4 — Orchestrated AI Analytics System

**ECE 272C · University of California, Santa Barbara**

---

## Setup

```bash
# 1. Install Python dependencies
pip install fastapi uvicorn langgraph langchain-anthropic anthropic \
    pandas numpy plotly duckduckgo-search langchain-community python-multipart

# 2. Set API key
export ANTHROPIC_API_KEY="your-key-here"

# 3. Place datasets in datasets/ folder
#    (2014_Financial_Data.csv … 2018_Financial_Data.csv)
#    They are auto-loaded on startup.

# 4. Start backend (from project root)
uvicorn backend.main:app --reload

# 5. Start frontend (separate terminal)
cd frontend && npm install && npm run dev
```

The backend auto-loads any CSVs in `datasets/` matching `YYYY_*.csv`.

---

## Architecture

See `ARCHITECTURE.md` for the full design document.

```
User → backend/main.py (FastAPI SSE)
            → orchestrator.py (LangGraph supervisor)
                 ├── analytics_agent.py  (iterative sandbox)
                 │        └── validator_agent.py (HW3 plan-execute)
                 └── web_search.py (DuckDuckGo)
```

## File Map

| File | Role |
|---|---|
| `memory.py` | Session state — chat history, exec namespace, loaded datasets |
| `operators.py` | Deterministic DataFrame operators (HW3 + join/union for HW4) |
| `planner.py` | LLM plan generator for validator (financial schema-aware) |
| `executor.py` | Runs operator plans against DataFrames |
| `validator_agent.py` | Independent verification via HW3 plan-execute engine |
| `analytics_agent.py` | Iterative reasoning + code sandbox (persistent namespace) |
| `web_search.py` | DuckDuckGo search + grounded answer synthesis |
| `orchestrator.py` | LangGraph graph: classify → route → validate → respond |
| `backend/main.py` | FastAPI SSE gateway, upload, reset, dataset list endpoints |
| `frontend/src/App.jsx` | React UI: chat + trace panel + citation strip + charts |

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/chat/stream` | POST | SSE stream — primary chat endpoint |
| `/api/upload` | POST | Upload a financial CSV (year detected from filename) |
| `/api/reset` | POST | Clear conversation + namespace (datasets preserved) |
| `/api/datasets` | GET | List loaded datasets with row/col counts |
| `/api/health` | GET | Health check |
