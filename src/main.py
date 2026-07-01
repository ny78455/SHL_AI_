"""
FastAPI application entry point.

Responsibilities:
- Create the FastAPI app instance
- Register the lifespan context (startup/shutdown)
- Mount routers
- Configure structured logging

The lifespan hook calls initialise_services() which builds the catalog and
retrieval index ONCE. After startup, all requests are served from in-memory
state with no further cold-start latency.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import initialise_services
from src.api.routes import chat, health

# ── Structured logging setup ──────────────────────────────────────────────────

def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer()
            if sys.stderr.isatty()
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )


_configure_logging()
logger = structlog.get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Build all singletons at startup so /health and /chat are immediately ready.
    Nothing is lazy-loaded per-request.
    """
    logger.info("startup_begin")
    initialise_services()
    logger.info("startup_complete — service ready")
    yield
    logger.info("shutdown_complete")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SHL Conversational Assessment Recommender",
    description=(
        "Stateless conversational agent that turns a vague hiring need into a "
        "grounded shortlist of SHL Individual Test Solutions through multi-turn dialogue."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)
