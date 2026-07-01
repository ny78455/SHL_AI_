"""
Qdrant index manager — builds and queries the hybrid search collection.

Collection design:
- Dense vectors  : Gemini embedding-001 (768-dim, cosine)
- Sparse vectors : true Okapi BM25 term weights via qdrant_client SparseVector
- Hybrid scoring : Reciprocal Rank Fusion (RRF) via Qdrant's built-in prefetch

The collection is created / rebuilt by the ingestion script (scripts/ingest_catalog.py)
and only queried at inference time.  Inference code never writes to Qdrant.

Recall@k fixes applied (see approach doc for before/after numbers):
1. Sparse vocabulary is now built with a DETERMINISTIC token->index assignment
   (sorted tokens, not `set()` iteration order) and PERSISTED to disk at
   ingestion time. Inference now LOADS that exact same vocab instead of
   re-fitting from scratch at startup — previously the startup refit could
   assign different indices to the same tokens than the ones baked into the
   vectors already stored in Qdrant, silently corrupting every lexical match.
2. The sparse encoder now implements real Okapi BM25 (tf saturation via k1,
   document-length normalization via b) instead of unsaturated TF * log-IDF.
   Documents also get a light bigram signal so multi-word product names
   ("core java", "ms excel") aren't reduced to easily-confused unigrams.
3. Point IDs for non-numeric entity_ids use a stable hash (md5-derived) instead
   of Python's per-process-salted `hash()`, so re-ingestion never reassigns a
   different Qdrant point ID to the same catalog entity.
4. The sparse-only fallback path now uses the current qdrant-client
   query_points API (the old `.search(query_vector=(name, SparseVector))`
   shape is deprecated/fragile) so a dense-path outage degrades gracefully
   instead of silently returning zero results.
5. Prefetch over-fetch margin increased (top_k * 4, floor of 40) to give the
   downstream diversity/dedup pass in hybrid_retriever.py enough headroom to
   still land top_k *distinct* items instead of being starved by near-dupes.
"""

from __future__ import annotations

import hashlib
import math
import pickle
import re
import time
from pathlib import Path
from typing import Any

import structlog
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

# Okapi BM25 hyperparameters (standard defaults; tuned on catalog description
# lengths which range ~15-200 tokens — b=0.75 is the right regime for that
# spread, k1=1.5 is the conventional default and worked well in local eval).
_BM25_K1 = 1.5
_BM25_B = 0.75

_SPARSE_ENCODER_FILENAME = "sparse_encoder.pkl"


# ── BM25 sparse encoder (client-side, no extra service needed) ──────────────

class BM25SparseEncoder:
    """
    True Okapi BM25 sparse encoder, deterministic across processes.

    Produces {token_index: weight} sparse vectors compatible with Qdrant's
    SparseVector format. Used for the lexical retrieval leg of hybrid search.

    Key property: the SAME encoder instance (same vocab, same doc stats) must
    be used both when the collection is built and when queries are encoded at
    inference time. Callers must persist/load this object rather than
    re-fitting independently at each process start — see save()/load().
    """

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._df: dict[int, int] = {}      # document frequency per token index
        self._n_docs: int = 0
        self._avg_doc_len: float = 0.0

    # ── fit ──────────────────────────────────────────────────────────────

    def fit(self, corpus: list[str]) -> None:
        """
        Build vocabulary and document-frequency table from corpus.

        Vocab assignment is made deterministic by sorting the token set for
        each document before assigning new indices, and by processing
        documents in the order given (i.e. caller must pass a stable corpus
        order — this is what makes the vocab reproducible run-to-run, which
        is required since the vocab must match between ingestion and
        inference).
        """
        self._n_docs = len(corpus)
        total_len = 0
        for doc in corpus:
            tokens = self._tokenize_with_bigrams(doc)
            total_len += len(tokens)
            for tok in sorted(set(tokens)):
                idx = self._vocab.setdefault(tok, len(self._vocab))
                self._df[idx] = self._df.get(idx, 0) + 1
        self._avg_doc_len = (total_len / self._n_docs) if self._n_docs else 1.0

    # ── encode ───────────────────────────────────────────────────────────

    def encode(self, text: str, is_query: bool = False) -> SparseVector:
        """
        Encode text into a Qdrant SparseVector using Okapi BM25 weighting.

        Document-side weights apply full BM25 saturation + length
        normalization. Query-side weights skip length normalization (b term
        dropped) since queries are short and length-normalizing them against
        the corpus's average *document* length would under-weight legitimate
        multi-concept queries (e.g. "senior java engineer sql aws docker").
        This asymmetric encoding is the standard approach for BM25-as-sparse-
        vector-dot-product retrieval.
        """
        tokens = self._tokenize_with_bigrams(text)
        tf: dict[int, int] = {}
        for tok in tokens:
            idx = self._vocab.get(tok)
            if idx is not None:
                tf[idx] = tf.get(idx, 0) + 1

        doc_len = len(tokens)
        n = max(self._n_docs, 1)
        avgdl = max(self._avg_doc_len, 1e-6)

        indices: list[int] = []
        values: list[float] = []
        for idx, count in tf.items():
            df = max(self._df.get(idx, 1), 1)
            # BM25 idf with +1 smoothing to keep weights non-negative even
            # when a term appears in every document.
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5)) if n > df else math.log(1 + n / df)
            idf = max(idf, 1e-4)

            if is_query:
                # Simple saturated term frequency, no length normalization.
                tf_component = (count * (_BM25_K1 + 1)) / (count + _BM25_K1)
            else:
                norm = 1 - _BM25_B + _BM25_B * (doc_len / avgdl)
                tf_component = (count * (_BM25_K1 + 1)) / (count + _BM25_K1 * norm)

            weight = idf * tf_component
            if weight > 0:
                indices.append(idx)
                values.append(float(weight))

        return SparseVector(indices=indices, values=values)

    # ── persistence ──────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vocab": self._vocab,
                    "df": self._df,
                    "n_docs": self._n_docs,
                    "avg_doc_len": self._avg_doc_len,
                },
                f,
            )
        logger.info("bm25_encoder_saved", path=str(path), vocab_size=len(self._vocab))

    @classmethod
    def load(cls, path: Path) -> "BM25SparseEncoder":
        with open(path, "rb") as f:
            state = pickle.load(f)
        enc = cls()
        enc._vocab = state["vocab"]
        enc._df = state["df"]
        enc._n_docs = state["n_docs"]
        enc._avg_doc_len = state["avg_doc_len"]
        return enc

    # ── tokenization ─────────────────────────────────────────────────────

    _STOPWORDS = frozenset(
        {
            "a", "an", "the", "and", "or", "of", "to", "in", "for", "is",
            "on", "with", "this", "that", "are", "at", "by", "be", "as",
        }
    )

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        raw = re.findall(r"[a-z0-9]+", text.lower())
        return [t for t in raw if t not in cls._STOPWORDS or len(raw) <= 3]

    @classmethod
    def _tokenize_with_bigrams(cls, text: str) -> list[str]:
        """
        Unigrams plus adjacent-pair bigrams, so multi-word product names
        ("core java", "ms excel", "customer service") get a distinct token
        from their generic constituent unigrams. This directly targets the
        catalog's many same-family / near-duplicate product names (§7.4 of
        the spec) that pure-unigram BM25 tends to conflate.
        """
        unigrams = cls._tokenize(text)
        bigrams = [f"{a}_{b}" for a, b in zip(unigrams, unigrams[1:])]
        return unigrams + bigrams


# ── Qdrant index ──────────────────────────────────────────────────────────

class QdrantIndex:
    """
    Manages the Qdrant collection for hybrid retrieval.

    At startup (via build_from_assessments):
      - Creates / recreates the collection
      - Upserts all assessment records as points with dense + sparse vectors
      - Persists the fitted BM25 encoder to disk

    At inference (via search):
      - Loads the persisted BM25 encoder (never re-fits independently)
      - Queries with dense + sparse prefetches + RRF fusion
      - Returns ordered list of matching entity_ids
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._collection = settings.qdrant_collection_name
        self._dim = settings.embedding_dimension
        api_key = settings.qdrant_api_key or None
        self._client = QdrantClient(url=settings.qdrant_url, api_key=api_key, timeout=60.0)
        self._sparse_encoder = BM25SparseEncoder()
        # `index_cache_dir` may not exist yet on older settings objects —
        # fall back to a sane default rather than crashing at startup.
        cache_dir = getattr(settings, "index_cache_dir", None) or "./data/index_cache"
        self._encoder_path = Path(cache_dir) / _SPARSE_ENCODER_FILENAME

    def setup_sparse_encoder(self, assessments: list[Assessment]) -> None:
        """
        Called at API startup. Loads the BM25 encoder PERSISTED at ingestion
        time rather than re-fitting from the in-memory assessment list — the
        vocab must be byte-identical to the one used to encode the vectors
        already stored in Qdrant, or every sparse-vector token index will
        silently point at the wrong term (see module docstring, fix #1).

        Falls back to a fresh fit only if no persisted encoder is found
        (e.g. first-ever run before any ingestion), and logs a loud warning
        in that case since it means query-time and index-time vocabs may
        drift on the very next ingestion run.
        """
        if self._encoder_path.exists():
            try:
                self._sparse_encoder = BM25SparseEncoder.load(self._encoder_path)
                logger.info(
                    "qdrant_sparse_encoder_loaded",
                    path=str(self._encoder_path),
                    vocab_size=len(self._sparse_encoder._vocab),
                )
                return
            except Exception as exc:  # noqa: BLE001
                logger.error("qdrant_sparse_encoder_load_failed", error=str(exc))

        logger.warning(
            "qdrant_sparse_encoder_refitting_from_scratch",
            reason="no_persisted_encoder_found",
        )
        corpus = [a.searchable_text() for a in assessments]
        self._sparse_encoder.fit(corpus)

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
        self._sparse_encoder = BM25SparseEncoder()
        self._sparse_encoder.fit(corpus)
        # Persist immediately so inference-time startup loads this exact
        # vocab instead of refitting (fix #1).
        self._sparse_encoder.save(self._encoder_path)

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
            sparse_vec = self._sparse_encoder.encode(corpus[i], is_query=False)
            points.append(
                PointStruct(
                    id=self._stable_point_id(assessment.entity_id),
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

    @staticmethod
    def _stable_point_id(entity_id: str) -> int:
        """
        Deterministic point ID that is stable across processes and re-runs.

        The previous implementation used Python's builtin `hash()` for
        non-numeric entity_ids, which is salted per-process (PYTHONHASHSEED)
        by default — the same entity_id could map to a different Qdrant
        point ID on every ingestion run, risking silent point collisions /
        overwrites between unrelated catalog entries (fix #3).
        """
        if entity_id.isdigit():
            return int(entity_id)
        digest = hashlib.md5(entity_id.encode("utf-8")).hexdigest()
        return int(digest[:15], 16)  # fits comfortably under 2**63

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
        sparse_query = self._sparse_encoder.encode(text_query, is_query=True)

        # Wider over-fetch margin than before (was top_k*2) so that the
        # diversity/dedup pass downstream in hybrid_retriever.py has enough
        # candidates to still surface top_k *distinct* items rather than
        # being starved by near-duplicate name variants (fix #5).
        prefetch_limit = max(top_k * 4, 40)

        try:
            results = self._client.query_points(
                collection_name=self._collection,
                prefetch=[
                    qmodels.Prefetch(
                        query=dense_query,
                        using=_DENSE_VECTOR_NAME,
                        limit=prefetch_limit,
                    ),
                    qmodels.Prefetch(
                        query=qmodels.SparseVector(
                            indices=sparse_query.indices,
                            values=sparse_query.values,
                        ),
                        using=_SPARSE_VECTOR_NAME,
                        limit=prefetch_limit,
                    ),
                ],
                query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
                limit=max(top_k, prefetch_limit // 2),
                with_payload=True,
            )
            return [
                str(point.payload.get("entity_id", point.id))
                for point in results.points
            ]
        except Exception as exc:  # noqa: BLE001
            logger.error("qdrant_search_failed", error=str(exc))
            # Fall back to sparse-only
            return self._sparse_fallback(sparse_query, top_k)

    def search_by_text(self, text_query: str, top_k: int = 1) -> list[str]:
        """
        Sparse-only (BM25) lookup, public entry point.

        Used by HybridRetriever to resolve "pinned default" items — e.g. the
        spec's called-out default OPQ32r inclusion (§11 known risk #2) — by
        their exact catalog name, without the caller needing to know a
        stable entity_id up front. Because it's an exact/near-exact name
        string being encoded, BM25 puts the matching record at rank 1 with
        very high confidence, so top_k=1 is normally sufficient.
        """
        sparse_query = self._sparse_encoder.encode(text_query, is_query=True)
        return self._sparse_fallback(sparse_query, top_k)

    def _sparse_fallback(self, sparse_query: SparseVector, top_k: int) -> list[str]:
        """
        Sparse-only fallback when the dense path fails (e.g. embedding
        provider outage). Uses the current `query_points` API — the previous
        implementation called the deprecated `.search(query_vector=(name,
        SparseVector))` tuple shape, which is fragile across qdrant-client
        versions and, if it raised, was silently swallowed into an empty
        result list, meaning a dense outage could zero out recall for the
        whole turn instead of degrading to lexical-only as the spec's
        §13.2 fallback table requires (fix #4).
        """
        try:
            results = self._client.query_points(
                collection_name=self._collection,
                query=qmodels.SparseVector(
                    indices=sparse_query.indices,
                    values=sparse_query.values,
                ),
                using=_SPARSE_VECTOR_NAME,
                limit=top_k,
                with_payload=True,
            )
            return [
                str(point.payload.get("entity_id", point.id))
                for point in results.points
            ]
        except Exception as exc:  # noqa: BLE001
            logger.error("qdrant_sparse_fallback_failed", error=str(exc))
            return []

    def load_sparse_encoder(self, encoder: BM25SparseEncoder) -> None:
        """Restore sparse encoder state (e.g. injected directly in tests)."""
        self._sparse_encoder = encoder