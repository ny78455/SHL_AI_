"""
Qdrant index manager — builds and queries the hybrid search collection.

Collection design:
- Dense vectors  : Gemini embedding-001 (768-dim, cosine)
- Sparse vectors : BM25-style term weights via qdrant_client SparseVector
- Hybrid scoring : Reciprocal Rank Fusion (RRF) via Qdrant's built-in prefetch

The collection is created / rebuilt by the ingestion script (scripts/ingest_catalog.py)
and only queried at inference time.  Inference code never writes to Qdrant.
"""

from __future__ import annotations

import logging
import structlog
import math
import re
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.config.settings import get_settings
from src.domain.models import Assessment

logger = structlog.get_logger(__name__)

_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "sparse"


# ── BM25-style sparse encoder (client-side, no extra service needed) ────────────

class _SparseEncoder:
    """
    Lightweight TF-IDF-like sparse encoder.
    Produces {token_index: weight} sparse vectors compatible with Qdrant's
    SparseVector format.  Used for the lexical retrieval leg of hybrid search.
    """

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._df: dict[int, int] = {}   # document frequency per token index
        self._n_docs: int = 0

    def fit(self, corpus: list[str]) -> None:
        """Build vocabulary and document-frequency table from corpus."""
        self._n_docs = len(corpus)
        for doc in corpus:
            tokens = set(self._tokenize(doc))
            for tok in tokens:
                idx = self._vocab.setdefault(tok, len(self._vocab))
                self._df[idx] = self._df.get(idx, 0) + 1

    def encode(self, text: str) -> SparseVector:
        """Encode a single text into a Qdrant SparseVector."""
        tokens = self._tokenize(text)
        tf: dict[int, int] = {}
        for tok in tokens:
            if tok in self._vocab:
                idx = self._vocab[tok]
                tf[idx] = tf.get(idx, 0) + 1
        indices: list[int] = []
        values: list[float] = []
        n = max(self._n_docs, 1)
        for idx, count in tf.items():
            df = max(self._df.get(idx, 1), 1)
            idf = math.log(1 + n / df)
            indices.append(idx)
            values.append(float(count) * idf)
        return SparseVector(indices=indices, values=values)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())


# ── Qdrant index ────────────────────────────────────────────────────────────────

class QdrantIndex:
    """
    Manages the Qdrant collection for hybrid retrieval.

    At startup (via build_from_assessments):
      - Creates / recreates the collection
      - Upserts all assessment records as points with dense + sparse vectors

    At inference (via search):
      - Queries with dense + sparse prefetches + RRF fusion
      - Returns ordered list of matching entity_ids
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._collection = settings.qdrant_collection_name
        self._dim = settings.embedding_dimension
        api_key = settings.qdrant_api_key or None
        self._client = QdrantClient(url=settings.qdrant_url, api_key=api_key, timeout=60.0)
        self._sparse_encoder = _SparseEncoder()

    def setup_sparse_encoder(self, assessments: list[Assessment]) -> None:
        """Called at API startup to build the in-memory term frequency vocab."""
        corpus = [a.searchable_text() for a in assessments]
        self._sparse_encoder.fit(corpus)
        logger.info("qdrant_sparse_encoder_ready", vocab_size=len(self._sparse_encoder._vocab))

    # ── Build (called by ingestion script, not at inference time) ───────────

    def build_from_assessments(
        self,
        assessments: list[Assessment],
        dense_vectors: list[list[float]],
    ) -> None:
        """
        (Re)build the Qdrant collection from scratch.
        dense_vectors[i] corresponds to assessments[i].
        """
        corpus = [a.searchable_text() for a in assessments]
        self._sparse_encoder.fit(corpus)

        # Recreate collection
        self._client.recreate_collection(
            collection_name=self._collection,
            vectors_config={
                _DENSE_VECTOR_NAME: VectorParams(
                    size=self._dim, distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                _SPARSE_VECTOR_NAME: SparseVectorParams()
            },
        )

        # Build and upsert points in batches
        points: list[PointStruct] = []
        for i, (assessment, dense_vec) in enumerate(
            zip(assessments, dense_vectors)
        ):
            sparse_vec = self._sparse_encoder.encode(corpus[i])
            points.append(
                PointStruct(
                    id=int(assessment.entity_id)
                    if assessment.entity_id.isdigit()
                    else abs(hash(assessment.entity_id)) % (2**31),
                    vector={
                        _DENSE_VECTOR_NAME: dense_vec,
                        _SPARSE_VECTOR_NAME: sparse_vec,
                    },
                    payload={"entity_id": assessment.entity_id},
                )
            )

        batch_size = 100
        for i in range(0, len(points), batch_size):
            self._client.upsert(
                collection_name=self._collection,
                points=points[i : i + batch_size],
            )

        logger.info("qdrant_index_built", count=len(points))

    # ── Query (called at inference time) ────────────────────────────────────

    def search(
        self,
        dense_query: list[float],
        text_query: str,
        top_k: int,
    ) -> list[str]:
        """
        Hybrid search: dense + sparse prefetches fused with RRF.
        Returns ordered list of entity_ids (most relevant first).
        """
        sparse_query = self._sparse_encoder.encode(text_query)

        try:
            results = self._client.query_points(
                collection_name=self._collection,
                prefetch=[
                    qmodels.Prefetch(
                        query=dense_query,
                        using=_DENSE_VECTOR_NAME,
                        limit=top_k * 2,
                    ),
                    qmodels.Prefetch(
                        query=qmodels.SparseVector(
                            indices=sparse_query.indices,
                            values=sparse_query.values,
                        ),
                        using=_SPARSE_VECTOR_NAME,
                        limit=top_k * 2,
                    ),
                ],
                query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )
            return [
                str(point.payload.get("entity_id", point.id))
                for point in results.points
            ]
        except Exception as exc:  # noqa: BLE001
            logger.error("qdrant_search_failed", error=str(exc))
            # Fall back to sparse-only
            return self._sparse_fallback(text_query, top_k)

    def _sparse_fallback(self, text_query: str, top_k: int) -> list[str]:
        """Sparse-only fallback when Qdrant dense path fails."""
        try:
            sparse_query = self._sparse_encoder.encode(text_query)
            results = self._client.search(
                collection_name=self._collection,
                query_vector=(
                    _SPARSE_VECTOR_NAME,
                    qmodels.SparseVector(
                        indices=sparse_query.indices,
                        values=sparse_query.values,
                    ),
                ),
                limit=top_k,
                with_payload=True,
            )
            return [
                str(r.payload.get("entity_id", r.id)) for r in results
            ]
        except Exception as exc:  # noqa: BLE001
            logger.error("qdrant_sparse_fallback_failed", error=str(exc))
            return []

    def load_sparse_encoder(self, encoder: _SparseEncoder) -> None:
        """Restore sparse encoder state (called after loading from pickle)."""
        self._sparse_encoder = encoder
