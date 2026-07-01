"""
Gemini embedding adapter.
Wraps google-generativeai to produce dense embeddings for the Qdrant index.
Loaded once at startup; embeddings are batched to stay within API rate limits.

Recall@k fix applied: the previous implementation zeroed out an ENTIRE batch
(up to 50 catalog items) on the first exception — a single transient
rate-limit or network blip permanently removed those items from dense
retrieval for the life of the index, since a zero vector never scores highly
against anything. This version retries with exponential backoff, and if a
batch still fails after retries, falls back to embedding items one at a time
so a single bad/oversized item doesn't take the rest of the batch down with
it. Zero vectors are now a last resort per-item, not a default per-batch
outcome, and every fallback is logged with the specific item so it can be
re-embedded later rather than silently degrading recall.
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
_MAX_RETRIES = 3
_BASE_RETRY_DELAY_SECONDS = 2


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
        """Embed a single retrieval query string, with retries."""
        for attempt in range(_MAX_RETRIES):
            try:
                result = genai.embed_content(
                    model=self._model,
                    content=text,
                    task_type=_TASK_TYPE_QUERY,
                )
                return result["embedding"]
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "embed_query_attempt_failed",
                    attempt=attempt + 1,
                    error=str(exc),
                )
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_BASE_RETRY_DELAY_SECONDS * (2 ** attempt))

        logger.error("embed_query_failed_all_retries", text=text[:100])
        # Caller (HybridRetriever) already treats an empty/zero dense vector
        # as "dense unavailable" and lets QdrantIndex.search() fall back to
        # sparse-only — raise so that path is exercised explicitly instead
        # of silently returning a zero vector that looks like a real query.
        raise RuntimeError("embed_query failed after all retries")

    def _embed_batched(
        self, texts: list[str], task_type: str
    ) -> list[list[float]]:
        """Embed texts in batches, respecting API limits, with retries and
        per-item fallback so a single failure doesn't zero out a whole batch.
        """
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            vectors = self._embed_batch_with_retries(batch, task_type, batch_index=i)
            all_vectors.extend(vectors)
        return all_vectors

    def _embed_batch_with_retries(
        self, batch: list[str], task_type: str, batch_index: int
    ) -> list[list[float]]:
        for attempt in range(_MAX_RETRIES):
            try:
                result = genai.embed_content(
                    model=self._model,
                    content=batch,
                    task_type=task_type,
                )
                vectors = result["embedding"]
                # Gemini returns a flat list for single-item input — wrap it.
                if isinstance(vectors[0], float):
                    return [vectors]  # type: ignore[list-item]
                return vectors
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "embedding_batch_attempt_failed",
                    batch_index=batch_index,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_BASE_RETRY_DELAY_SECONDS * (2 ** attempt))

        # Whole-batch retries exhausted — fall back to embedding items
        # individually rather than zeroing the entire batch. This isolates
        # a single problematic item (e.g. unusually long description) from
        # the rest of the batch, which previously lost all ~50 items.
        logger.error(
            "embedding_batch_failed_falling_back_to_individual",
            batch_index=batch_index,
            batch_size=len(batch),
        )
        return [
            self._embed_single_with_retries(text, task_type, batch_index + j)
            for j, text in enumerate(batch)
        ]

    def _embed_single_with_retries(
        self, text: str, task_type: str, item_index: int
    ) -> list[float]:
        for attempt in range(_MAX_RETRIES):
            try:
                result = genai.embed_content(
                    model=self._model,
                    content=text,
                    task_type=task_type,
                )
                return result["embedding"]
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "embedding_single_attempt_failed",
                    item_index=item_index,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_BASE_RETRY_DELAY_SECONDS * (2 ** attempt))

        logger.error(
            "embedding_single_failed_all_retries_using_zero_vector",
            item_index=item_index,
            text_preview=text[:100],
        )
        # True last resort: this item is genuinely unembeddable right now.
        # It will still be retrievable via the BM25 sparse leg, so it isn't
        # fully lost from recall — only its dense signal is missing.
        return [0.0] * self._dim