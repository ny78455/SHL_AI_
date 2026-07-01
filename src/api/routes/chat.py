"""POST /chat — conversational assessment recommender endpoint."""

from __future__ import annotations

import logging
import structlog

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from src.api.dependencies import get_orchestrator
from src.api.schemas import ChatRequest, ChatResponse, RecommendationItem
from src.domain import models as domain
from src.orchestration.orchestrator import ConversationOrchestrator

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(
    request: ChatRequest,
    orchestrator: ConversationOrchestrator = Depends(get_orchestrator),
) -> ChatResponse:
    """
    Stateless conversational endpoint.

    The client sends the FULL conversation history on every call.
    No session state is stored server-side.

    Returns a grounded shortlist of SHL Individual Test Solutions,
    a clarifying question, a comparison, or a scope-refusal — depending
    on the accumulated context and detected intent.
    """
    # Convert API schema → domain models
    domain_messages = [
        domain.Message(role=msg.role, content=msg.content)
        for msg in request.messages
    ]

    # Delegate entirely to the orchestrator — route stays thin
    domain_response: domain.ChatResponse = await orchestrator.process(domain_messages)

    # Convert domain response → API schema
    return ChatResponse(
        reply=domain_response.reply,
        recommendations=[
            RecommendationItem(
                name=rec.name,
                url=rec.url,
                test_type=rec.test_type,
            )
            for rec in domain_response.recommendations
        ],
        end_of_conversation=domain_response.end_of_conversation,
    )
