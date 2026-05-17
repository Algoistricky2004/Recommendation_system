"""
main.py — FastAPI application for the SHL Assessment Recommender.

Endpoints:
  GET  /health  → {"status": "ok"}
  POST /chat    → ChatResponse (reply, recommendations, end_of_conversation)
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent import process_chat
from app.catalog import get_catalog
from app.models import ChatRequest, ChatResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: pre-load catalog at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading SHL catalog…")
    t0 = time.perf_counter()
    try:
        catalog = get_catalog()
        logger.info(
            "Catalog loaded: %d products indexed in %.2fs",
            len(catalog.items),
            time.perf_counter() - t0,
        )
    except Exception as e:
        logger.error("Catalog load failed: %s", e)
        raise
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent that recommends SHL assessments from the product catalog.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow cross-origin requests (useful for front-end or evaluator)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: request timing
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=dict, tags=["Health"])
def health_check():
    """Readiness probe. Returns HTTP 200 with {'status': 'ok'}."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest):
    """
    Stateless conversation endpoint.

    The full conversation history is sent on every call.
    Returns the next agent reply plus an optional structured shortlist.
    """
    if not request.messages:
        raise HTTPException(status_code=422, detail="messages list must not be empty")

    # Enforce turn cap at API level
    total_turns = len(request.messages)
    if total_turns > 8:
        logger.warning("Request exceeds turn cap (%d turns). Clamping.", total_turns)

    try:
        catalog = get_catalog()
        response = process_chat(request, catalog=catalog)
        return response

    except RuntimeError as e:
        # Config errors (missing API key, etc.)
        logger.error("Configuration error: %s", e)
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        logger.exception("Unexpected error in /chat: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error — please retry.")


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )