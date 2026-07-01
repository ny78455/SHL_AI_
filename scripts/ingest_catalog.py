"""
Catalog ingestion script.

Run this ONCE before deployment (or after catalog updates) to:
1. Fetch + validate + classify the SHL catalog
2. Build Gemini embeddings for all Individual Test Solutions
3. Upsert dense + sparse vectors into Qdrant
4. Write catalog_cache.json to disk

Usage:
    python -m scripts.ingest_catalog

Environment:
    Requires GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY in .env

This script is deliberately NOT called at inference time — the index is
pre-built so /chat can serve requests without any startup rebuild cost.
"""

from __future__ import annotations

import json
import logging
import structlog
import pickle
import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.catalog.repository import InMemoryCatalogRepository
from src.config.settings import get_settings
from src.retrieval.embedder import GeminiEmbedder
from src.retrieval.qdrant_index import QdrantIndex

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = structlog.get_logger("ingest_catalog")


def main() -> None:
    settings = get_settings()
    logger.info("=== SHL Catalog Ingestion ===")

    # ── 1. Fetch & classify catalog ────────────────────────────────────────
    logger.info("Step 1: Building catalog repository...")
    catalog = InMemoryCatalogRepository.build()

    if catalog.size == 0:
        logger.critical("Catalog is empty — check CATALOG_JSON_URL and cache. Aborting.")
        sys.exit(1)

    logger.info(f"  → {catalog.size} Individual Test Solutions loaded.")

    # ── 2. Write/refresh cache ─────────────────────────────────────────────
    logger.info("Step 2: Catalog cache already written by fetcher.")

    # ── 3. Build embeddings ────────────────────────────────────────────────
    assessments = catalog.get_all()
    texts = [a.searchable_text() for a in assessments]

    logger.info(f"Step 3: Embedding {len(texts)} assessments with {settings.embedding_model}...")
    embedder = GeminiEmbedder()

    start = time.time()
    dense_vectors = embedder.embed_documents(texts)
    elapsed = time.time() - start
    logger.info(f"  → Embeddings done in {elapsed:.1f}s. Vectors: {len(dense_vectors)}")

    if len(dense_vectors) != len(assessments):
        logger.error("Embedding count mismatch. Aborting.")
        sys.exit(1)

    # ── 4. Upsert into Qdrant ──────────────────────────────────────────────
    logger.info(f"Step 4: Building Qdrant collection '{settings.qdrant_collection_name}'...")
    qdrant_index = QdrantIndex()

    start = time.time()
    qdrant_index.build_from_assessments(
        assessments=assessments,
        dense_vectors=dense_vectors,
    )
    elapsed = time.time() - start
    logger.info(f"  → Qdrant index built in {elapsed:.1f}s.")

    logger.info("=== Ingestion complete. Service is ready to start. ===")


if __name__ == "__main__":
    main()
