"""
Gemini LLM client — implements LLMPort.

Responsibilities:
- Enforce JSON-mode output via response_mime_type
- Retry transient errors (rate-limit, 5xx) with tenacity
- Hard timeout on each call (well under the 30s budget)
- Return raw LLM string; schema validation is the caller's job
"""

from __future__ import annotations

import json
import logging
import structlog
from typing import Any

import google.generativeai as genai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config.settings import get_settings
from src.domain.models import Message
from src.domain.ports import LLMPort

logger = structlog.get_logger(__name__)


class GeminiClient(LLMPort):
    """
    Thin, focused LLM adapter for Gemini.

    Uses generate_content with JSON response_mime_type so the model is
    constrained to emit valid JSON, reducing post-hoc parsing failures.
    """

    def __init__(self) -> None:
        settings = get_settings()
        genai.configure(api_key=settings.gemini_api_key)
        self._model_name = settings.llm_model
        self._timeout = settings.llm_timeout_seconds
        self._max_retries = settings.llm_max_retries

    async def generate(
        self,
        system_prompt: str,
        messages: list[Message],
        context_snippets: str,
    ) -> str:
        """
        Generate a response.  Returns the raw text from the model.
        Retries on transient failures; raises on permanent failure so
        the fallback client can take over.
        """
        model = genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction=system_prompt + "\n\n" + context_snippets,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.2,
                max_output_tokens=2048,
            ),
        )

        gemini_history = self._to_gemini_history(messages)
        return await self._call_with_retry(model, gemini_history)

    @staticmethod
    def _to_gemini_history(
        messages: list[Message],
    ) -> list[dict[str, Any]]:
        """Convert domain Message list to Gemini's content format."""
        history = []
        for msg in messages:
            role = "user" if msg.role == "user" else "model"
            history.append({"role": role, "parts": [{"text": msg.content}]})
        return history

    async def _call_with_retry(
        self,
        model: genai.GenerativeModel,
        history: list[dict[str, Any]],
    ) -> str:
        """Attempt the LLM call with exponential back-off retries."""

        @retry(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            reraise=True,
        )
        def _call() -> str:
            # Build chat session
            chat = model.start_chat(history=history[:-1] if history else [])
            last_message = history[-1]["parts"][0]["text"] if history else ""
            response = chat.send_message(
                last_message,
                request_options={"timeout": self._timeout},
            )
            return response.text

        return _call()
