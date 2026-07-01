"""
FastAPI dependency providers.

Single source of truth for all singleton services.
All heavy objects (catalog, index, embedder, orchestrator) are built ONCE
at application startup via the lifespan hook and stored here.
Per-request routes call get_orchestrator() — they never construct anything.
"""

from __future__ import annotations

import logging
import structlog
from typing import Optional

from src.catalog.repository import InMemoryCatalogRepository
from src.llm.fallback_client import FallbackLLMClient
from src.llm.gemini_client import GeminiClient
from src.llm.response_validator import ResponseValidator
from src.orchestration.constraint_extractor import ConstraintExtractor
from src.orchestration.intent_classifier import IntentClassifier
from src.orchestration.orchestrator import ConversationOrchestrator
from src.orchestration.shortlist_parser import ShortlistParser
from src.orchestration.turn_counter import TurnCounter
from src.retrieval.embedder import GeminiEmbedder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.qdrant_index import QdrantIndex
from src.retrieval.query_builder import QueryBuilder

logger = structlog.get_logger(__name__)

# ── Module-level singletons (populated by startup lifespan) ───────────────────
_catalog: Optional[InMemoryCatalogRepository] = None
_orchestrator: Optional[ConversationOrchestrator] = None


def initialise_services() -> None:
    """
    Build and wire all singleton services.
    Called once from the FastAPI lifespan hook — never called per-request.
    """
    global _catalog, _orchestrator

    logger.info("startup_building_catalog")
    catalog = InMemoryCatalogRepository.build()
    _catalog = catalog
    logger.info("startup_catalog_ready", size=catalog.size)

    logger.info("startup_building_retrieval_index")
    embedder = GeminiEmbedder()
    qdrant_index = QdrantIndex()
    qdrant_index.setup_sparse_encoder(catalog.get_all())
    retriever = HybridRetriever(
        catalog=catalog,
        embedder=embedder,
        qdrant_index=qdrant_index,
    )

    logger.info("startup_building_llm_client")
    primary_llm = GeminiClient()
    fallback_llm = FallbackLLMClient(primary=primary_llm)
    validator = ResponseValidator(catalog=catalog)

    _orchestrator = ConversationOrchestrator(
        catalog=catalog,
        retriever=retriever,
        llm=fallback_llm,
        validator=validator,
        intent_classifier=IntentClassifier(),
        shortlist_parser=ShortlistParser(),
        turn_counter=TurnCounter(),
        constraint_extractor=ConstraintExtractor(),
        query_builder=QueryBuilder(),
    )
    logger.info("startup_all_services_ready")


def get_orchestrator() -> ConversationOrchestrator:
    """FastAPI dependency — returns the singleton orchestrator."""
    if _orchestrator is None:
        raise RuntimeError("Services not initialised. Call initialise_services() at startup.")
    return _orchestrator


def get_catalog() -> InMemoryCatalogRepository:
    """FastAPI dependency — returns the singleton catalog repository."""
    if _catalog is None:
        raise RuntimeError("Services not initialised. Call initialise_services() at startup.")
    return _catalog
