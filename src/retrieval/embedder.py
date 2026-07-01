"""
Gemini embedding adapter.

Wraps google-generativeai to produce dense embeddings for the Qdrant index.
Loaded once at startup; embeddings are batched to stay within API rate limits.
"""

from __future__ import annotations

import logging
import structlog
import time
from typing import Sequence

import google.generativeai as genai

from src.config.settings import get_settings

logger = structlog.get_logger(__name__)

_TASK_TYPE_DOC = "RETRIEVAL_DOCUMENT"
_TASK_TYPE_QUERY = "RETRIEVAL_QUERY"
_BATCH_SIZE = 50          # Gemini embedding API batch limit
_RETRY_DELAY_SECONDS = 2


class GeminiEmbedder:
    """
    Produces dense embeddings using Gemini's embedding-001 model.

    Two task types:
    - RETRIEVAL_DOCUMENT — used when indexing catalog records
    - RETRIEVAL_QUERY    — used for each user retrieval query
    """

    def __init__(self) -> None:
        settings = get_settings()
        genai.configure(api_key=settings.gemini_api_key)
        self._model = settings.embedding_model
        self._dim = settings.embedding_dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of document strings (catalog records)."""
        return self._embed_batched(list(texts), task_type=_TASK_TYPE_DOC)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single retrieval query string."""
        result = genai.embed_content(
            model=self._model,
            content=text,
            task_type=_TASK_TYPE_QUERY,
        )
        return result["embedding"]

    def _embed_batched(
        self, texts: list[str], task_type: str
    ) -> list[list[float]]:
        """Embed texts in batches, respecting API limits."""
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            try:
                result = genai.embed_content(
                    model=self._model,
                    content=batch,
                    task_type=task_type,
                )
                vectors = result["embedding"]
                # Gemini returns a list of lists for batch input
                if isinstance(vectors[0], float):
                    # Single item batch returned flat — wrap it
                    all_vectors.append(vectors)  # type: ignore[arg-type]
                else:
                    all_vectors.extend(vectors)
            except Exception as exc:  # noqa: BLE001
                logger.error("embedding_batch_failed", batch_index=i, error=str(exc))
                # Fill with zero vectors so index positions stay aligned
                for _ in batch:
                    all_vectors.append([0.0] * self._dim)
                time.sleep(_RETRY_DELAY_SECONDS)
        return all_vectors
