"""
In-memory catalog repository.

Implements CatalogRepository (§domain/ports.py).
Converts validated RawCatalogRecord → Assessment domain objects and exposes
fast O(1) lookups by entity_id and URL via pre-built dicts.

Loaded once at application startup via the FastAPI lifespan hook; never
rebuilt per-request.
"""

from __future__ import annotations

import logging
import structlog
from typing import Optional

from src.catalog.classifier import IndividualTestClassifier
from src.catalog.fetcher import CatalogFetcher
from src.catalog.schema_validator import RawCatalogRecord
from src.domain.models import Assessment
from src.domain.ports import CatalogRepository

logger = structlog.get_logger(__name__)

# ── test_type derivation map (§3.2 of spec) ───────────────────────────────────
_KEYS_TO_TEST_TYPE: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
    "Assessment Exercises": "E",
}


def _derive_test_type(record: RawCatalogRecord) -> str:
    """
    Derive test_type codes from `keys` when not explicitly present in JSON.
    Preserves explicit test_type if available; falls back to key-map derivation.
    Multiple codes joined by comma, e.g. "K,S".
    """
    if record.test_type:
        return record.test_type.strip()
    codes = []
    seen: set[str] = set()
    for key in record.keys:
        code = _KEYS_TO_TEST_TYPE.get(key)
        if code and code not in seen:
            codes.append(code)
            seen.add(code)
    return ",".join(codes) if codes else "K"  # default to K if unknown


def _raw_to_domain(record: RawCatalogRecord) -> Assessment:
    return Assessment(
        entity_id=record.entity_id,
        name=record.name,
        url=record.link,
        description=record.description or "Not specified",
        keys=record.keys or [],
        test_type=_derive_test_type(record),
        job_levels=record.job_levels or [],
        languages=record.languages or [],
        duration=record.duration or "Not specified",
        remote=record.remote == "yes",
        adaptive=record.adaptive == "yes",
    )


class InMemoryCatalogRepository(CatalogRepository):
    """
    Thread-safe (reads only after startup) in-memory catalog store.

    Index strategy:
    - `_by_id`  : dict[entity_id → Assessment]  O(1) lookup
    - `_by_url` : dict[url → Assessment]         O(1) URL validation
    - `_all`    : list[Assessment]               iteration for retrieval indexing
    """

    def __init__(self) -> None:
        self._all: list[Assessment] = []
        self._by_id: dict[str, Assessment] = {}
        self._by_url: dict[str, Assessment] = {}

    # ── Factory / builder ──────────────────────────────────────────────────

    @classmethod
    def build(cls) -> "InMemoryCatalogRepository":
        """
        Fetch, classify, convert, and index the full catalog.
        Called once at startup by the FastAPI lifespan hook.
        """
        fetcher = CatalogFetcher()
        classifier = IndividualTestClassifier()

        raw_records = fetcher.fetch()
        repo = cls()

        included = 0
        excluded = 0
        for raw in raw_records:
            if not classifier.is_individual_test(raw):
                excluded += 1
                continue
            assessment = _raw_to_domain(raw)
            repo._all.append(assessment)
            repo._by_id[assessment.entity_id] = assessment
            repo._by_url[assessment.url] = assessment
            included += 1

        logger.info(
            "catalog_repository_built",
            included=included,
            excluded=excluded,
            total=len(raw_records),
        )
        return repo

    # ── CatalogRepository interface ────────────────────────────────────────

    def get_all(self) -> list[Assessment]:
        return list(self._all)

    def get_by_entity_id(self, entity_id: str) -> Optional[Assessment]:
        return self._by_id.get(entity_id)

    def get_by_url(self, url: str) -> Optional[Assessment]:
        return self._by_url.get(url)

    def url_exists(self, url: str) -> bool:
        return url in self._by_url

    @property
    def size(self) -> int:
        return len(self._all)
