"""
Hybrid retriever — the primary retrieval port implementation.

Pipeline per query:
1. Build retrieval query string (query_builder.py)
2. Embed with Gemini (dense)
3. Encode sparse (BM25-style, client-side)
4. Qdrant prefetch + RRF fusion → entity_ids
5. Resolve entity_ids → Assessment domain objects via CatalogRepository
6. Dedup/diversity re-rank (removes near-duplicate name variants)
7. Return top-K assessments

Degrades gracefully:
- Qdrant unavailable → sparse-only fallback inside QdrantIndex.search()
- Zero results → returns empty list (orchestrator handles this case)
- Retrieved duplicates → diversity filter applied
"""

from __future__ import annotations

import logging
import structlog
import re

from src.catalog.repository import InMemoryCatalogRepository
from src.domain.models import Assessment
from src.domain.ports import CatalogRepository, RetrievalPort
from src.retrieval.embedder import GeminiEmbedder
from src.retrieval.qdrant_index import QdrantIndex

logger = structlog.get_logger(__name__)


class HybridRetriever(RetrievalPort):
    """
    Implements RetrievalPort using Qdrant hybrid search (dense + sparse, RRF).
    All dependencies injected; never constructs its own.
    """

    def __init__(
        self,
        catalog: CatalogRepository,
        embedder: GeminiEmbedder,
        qdrant_index: QdrantIndex,
    ) -> None:
        self._catalog = catalog
        self._embedder = embedder
        self._qdrant = qdrant_index

    async def search(self, query: str, top_k: int) -> list[Assessment]:
        """
        Run hybrid retrieval for the query and return up to top_k diverse results.
        """
        logger.debug("retrieval_start", query=query[:100], top_k=top_k)

        # 1. Embed query
        try:
            dense_vec = self._embedder.embed_query(query)
        except Exception as exc:  # noqa: BLE001
            logger.error("retrieval_embed_failed", error=str(exc))
            dense_vec = []

        # 2. Search Qdrant (handles fallback internally)
        entity_ids = self._qdrant.search(
            dense_query=dense_vec,
            text_query=query,
            top_k=top_k * 2,  # over-fetch before diversity filter
        )

        # 3. Resolve to domain objects
        assessments: list[Assessment] = []
        for eid in entity_ids:
            a = self._catalog.get_by_entity_id(eid)
            if a is not None:
                assessments.append(a)

        # 4. Dedup / diversity re-rank
        assessments = self._deduplicate(assessments)

        result = assessments[:top_k]
        logger.debug("retrieval_done", returned=len(result))
        return result

    # ── Diversity re-rank ──────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(assessments: list[Assessment]) -> list[Assessment]:
        """
        Remove near-duplicate entries (same base name, different '(New)' / version suffix).
        Keeps the first (highest-ranked) occurrence of each normalised name.

        Exception: if both old and new variants are present and the query
        explicitly asks to compare them, both should survive — the orchestrator
        handles that case by passing a sufficiently large top_k and letting the
        LLM select from the full set.
        """
        seen_normalised: set[str] = set()
        unique: list[Assessment] = []
        for a in assessments:
            normalised = _normalise_name(a.name)
            if normalised not in seen_normalised:
                seen_normalised.add(normalised)
                unique.append(a)
        return unique


def _normalise_name(name: str) -> str:
    """
    Strip version markers for dedup purposes.
    'Core Java (Advanced Level) (New)' → 'core java (advanced level)'
    'MS Excel (New)' → 'ms excel'
    """
    name = name.lower()
    name = re.sub(r"\(new\)", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name
