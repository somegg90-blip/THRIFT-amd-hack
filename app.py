"""
THRIFT demo server.

Serves the dashboard at GET / and exposes the agent over HTTP.
All real logic lives in thrift.py -- this file is just the HTTP boundary.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from thrift import Thrift

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("thrift.app")

# Single shared agent instance
agent = Thrift()

# Lifespan event to preload the model into memory (if Tier 1 is enabled)
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing THRIFT agent...")
    try:
        # Check if Tier 1 is actually enabled before trying to load it
        if agent.cascade.tier1 is not None:
            logger.info("Pre-loading local model into memory...")
            agent.cascade.tier1._ensure_loaded()
            logger.info("Local model loaded successfully and ready for inference.")
        else:
            logger.info("Tier 1 (local model) is skipped. Using Fireworks API only.")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
    yield

# Pass the lifespan to FastAPI
app = FastAPI(title="THRIFT", description="Token-efficient routing agent", lifespan=lifespan)

# Serve static files (dashboard)
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class QueryRequest(BaseModel):
    query: str
    context: Optional[str] = ""


class QueryResponse(BaseModel):
    answer: str
    tokens_used: int
    latency_sec: float
    subtask_count: int
    tiers_used: list
    any_degraded: bool
    subtasks: list
    cascade_results: list   # full per-subtask detail for the dashboard


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the demo dashboard."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>THRIFT running</h1><p>Dashboard not found — place index.html in static/</p>")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query field cannot be empty")

    try:
        run = agent.answer(request.query, context=request.context or "")
    except Exception as e:
        logger.error(f"[app] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    return QueryResponse(
        answer=run.final_answer.text,
        tokens_used=run.tokens_used,
        latency_sec=run.latency_sec,
        subtask_count=len(run.subtasks),
        tiers_used=run.final_answer.tiers_used,
        any_degraded=run.any_degraded,
        subtasks=[s.to_dict() for s in run.subtasks],
        cascade_results=[r.to_dict() for r in run.cascade_results],
    )


@app.get("/stats")
def stats():
    try:
        return agent.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute stats: {e}")


@app.post("/reset")
def reset():
    agent.reset_stats()
    return {"status": "reset"}