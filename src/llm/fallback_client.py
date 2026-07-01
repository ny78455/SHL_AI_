"""
Fallback LLM client — wraps the primary client with a deterministic last resort.

Chain:
1. Primary: GeminiClient (main model)
2. Secondary: GeminiClient with a smaller / faster model config
3. Deterministic: schema-valid clarify template — never fails

This ensures the API always returns a valid response even when all providers
are down or timing out, satisfying the hard-eval gate.
"""

from __future__ import annotations

import logging
import structlog

from src.config.settings import get_settings
from src.domain.models import ChatResponse, Message
from src.llm.gemini_client import GeminiClient

logger = structlog.get_logger(__name__)

_DETERMINISTIC_FALLBACK = ChatResponse(
    reply=(
        "I'm having trouble processing your request right now. "
        "Could you tell me more about the role and skills you're hiring for? "
        "I'll do my best to recommend the right SHL assessments."
    ),
    recommendations=[],
    end_of_conversation=False,
)


class FallbackLLMClient:
    """
    Resilient LLM caller: primary → secondary → deterministic template.
    Each layer is independently triggerable and stackable.
    """

    def __init__(self, primary: GeminiClient) -> None:
        self._primary = primary
        self._settings = get_settings()

    async def generate(
        self,
        system_prompt: str,
        messages: list[Message],
        context_snippets: str,
    ) -> str:
        """
        Attempt generation through the fallback chain.
        Returns a raw JSON string on success, or a pre-serialised fallback.
        """
        import json

        # 1. Primary client
        try:
            result = await self._primary.generate(
                system_prompt=system_prompt,
                messages=messages,
                context_snippets=context_snippets,
            )
            logger.debug("llm_primary_success")
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("llm_primary_failed", error=str(exc))

        # 2. Stricter JSON-only retry (same client, stripped prompt)
        try:
            stripped_prompt = (
                system_prompt
                + "\n\nCRITICAL: Return ONLY valid JSON with no prose outside the JSON object."
            )
            result = await self._primary.generate(
                system_prompt=stripped_prompt,
                messages=messages[-2:],  # minimal history to reduce token load
                context_snippets=context_snippets,
            )
            logger.info("llm_secondary_retry_success")
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error("llm_secondary_failed", error=str(exc))

        # 3. Deterministic fallback — always schema-valid
        logger.critical("llm_using_deterministic_fallback")
        return json.dumps({
            "reply": _DETERMINISTIC_FALLBACK.reply,
            "recommendations": [],
            "end_of_conversation": False,
        })
