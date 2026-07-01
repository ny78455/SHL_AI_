"""
API request/response schemas.

These are the ONLY Pydantic models the HTTP layer uses.
They are deliberately kept separate from domain models so the API contract
can evolve independently of internal business objects.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class MessageSchema(BaseModel):
    """A single message in the conversation history."""

    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("user", "assistant"):
            # Accept but normalise unknown roles rather than reject
            # (harness may send 'system' or other roles)
            return "user"
        return v

    @field_validator("content", mode="before")
    @classmethod
    def ensure_string(cls, v: object) -> str:
        return str(v) if v is not None else ""


class ChatRequest(BaseModel):
    """POST /chat request body."""

    messages: list[MessageSchema] = Field(
        default_factory=list,
        description="Full conversation history. Client resends all turns each call.",
    )

    @field_validator("messages", mode="before")
    @classmethod
    def coerce_none(cls, v: object) -> list:
        return v if isinstance(v, list) else []


class RecommendationItem(BaseModel):
    """A single recommended SHL assessment."""

    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    """POST /chat response body — schema is fixed per spec (hard-eval gate)."""

    reply: str = Field(..., description="Conversational response text")
    recommendations: list[RecommendationItem] = Field(
        default_factory=list,
        description="Recommended assessments (empty on clarify/compare/refuse turns)",
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True when the agent considers the task complete",
    )


class HealthResponse(BaseModel):
    """GET /health response body."""

    status: str = "ok"
