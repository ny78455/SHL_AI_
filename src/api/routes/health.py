"""GET /health — liveness probe."""

from __future__ import annotations

from fastapi import APIRouter
from src.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """
    Liveness probe. Returns 200 {status: ok} once the process is running.
    The retrieval index and catalog are loaded at startup; once /health passes,
    /chat is ready with no further cold-start latency.
    """
    return HealthResponse(status="ok")
