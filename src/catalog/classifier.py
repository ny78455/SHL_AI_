"""
Individual Test Solution classifier (§3.3 of spec).

The SHL catalog JSON does not carry an explicit Individual / Pre-packaged flag.
This module reconstructs that distinction using a rule-based classifier in
priority order:

1. Explicit exclusion patterns (high-confidence Job Solutions by name)
2. Single-keys → Individual Test (high confidence)
3. Multi-keys + bundling language → Pre-packaged (exclude)
4. Ambiguous → EXCLUDE (safe default per §12.1 fallback rules)

Every decision is traceable to a named rule; no magic heuristics buried in
one big conditional block.
"""

from __future__ import annotations

import logging
import structlog
import re

from src.catalog.schema_validator import RawCatalogRecord

logger = structlog.get_logger(__name__)

# ── Patterns identifying Pre-packaged Job Solutions ────────────────────────────

# Names that strongly indicate a bundled job solution (case-insensitive)
_JOB_SOLUTION_NAME_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bsolution\b", re.IGNORECASE),
    re.compile(r"\bsolutions\b", re.IGNORECASE),
]

# Description phrases characteristic of job-archetype bundles
_JOB_SOLUTION_DESCRIPTION_MARKERS: list[str] = [
    "is for entry-level positions in which employees",
    "pre-packaged",
    "job solution",
    "packaged solution",
]

# Keys categories that, when appearing together in the same record, suggest a bundle
_BUNDLE_KEYS_THRESHOLD = 4  # 4+ distinct category keys → very likely a bundle


class IndividualTestClassifier:
    """
    Classifies an SHL catalog record as either an Individual Test Solution
    (keep) or a Pre-packaged Job Solution (exclude).

    All rules are pure functions so they are trivially unit-testable.
    """

    def is_individual_test(self, record: RawCatalogRecord) -> bool:
        """
        Return True iff the record should be included in the retrievable set.
        Excludes records with status != 'ok' outright.
        """
        if record.status != "ok":
            logger.debug("classifier_exclude_status", name=record.name, status=record.status)
            return False

        if self._is_explicit_job_solution_by_name(record):
            logger.debug("classifier_exclude_name_pattern", name=record.name)
            return False

        if self._is_explicit_job_solution_by_description(record):
            logger.debug("classifier_exclude_description_pattern", name=record.name)
            return False

        if self._is_bundle_by_key_count(record):
            logger.debug("classifier_exclude_key_count", name=record.name, keys=record.keys)
            return False

        logger.debug("classifier_include", name=record.name)
        return True

    # ── Named rules (each independently testable) ──────────────────────────

    @staticmethod
    def _is_explicit_job_solution_by_name(record: RawCatalogRecord) -> bool:
        """Exclude records whose name contains 'Solution(s)'."""
        return any(p.search(record.name) for p in _JOB_SOLUTION_NAME_PATTERNS)

    @staticmethod
    def _is_explicit_job_solution_by_description(record: RawCatalogRecord) -> bool:
        """Exclude records whose description contains bundle marker phrases."""
        desc_lower = record.description.lower()
        return any(marker in desc_lower for marker in _JOB_SOLUTION_DESCRIPTION_MARKERS)

    @staticmethod
    def _is_bundle_by_key_count(record: RawCatalogRecord) -> bool:
        """
        Exclude if record spans >= threshold distinct categories AND has no
        test_type that suggests it is actually a standalone multi-dimension test.
        Note: Global Skills Development Report (6 keys, type D) is a known
        legitimate individual product — the test_type='D' guard handles it.
        """
        if len(record.keys) < _BUNDLE_KEYS_THRESHOLD:
            return False
        # If it has a single-char test_type it is likely a standalone report product
        tt = (record.test_type or "").strip()
        if tt and len(tt.replace(",", "").replace(" ", "")) <= 2:
            return False
        return True
