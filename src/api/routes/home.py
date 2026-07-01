"""GET /health — liveness probe and default route info."""

from __future__ import annotations

from typing import Any, Dict
from fastapi import APIRouter

router = APIRouter()


@router.get("/", tags=["info"])
async def root() -> Dict[str, Any]:
    """
    Default route. Returns information about available API endpoints 
    and their expected schemas.
    """
    return {
        "service": "SHL Conversational Assessment Recommender API",
        "endpoints": {
            "/health": {
                "method": "GET",
                "description": "Liveness probe. Returns 200 {status: ok} when ready."
            },
            "/chat": {
                "method": "POST",
                "description": "Main conversational endpoint for assessment recommendations.",
                "schema": {
                    "request": {
                        "messages": [
                            {
                                "role": "string (user or assistant)",
                                "content": "string"
                            }
                        ]
                    },
                    "response": {
                        "reply": "string",
                        "recommendations": [
                            {
                                "name": "string",
                                "url": "string",
                                "test_type": "string"
                            }
                        ],
                        "end_of_conversation": "boolean"
                    },
                    "notes": "recommendations can be null when clarifying, comparing, or refusing."
                }
            }
        }
    }