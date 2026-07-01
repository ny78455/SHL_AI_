"""
Catalog fetcher — retrieves the SHL product catalog JSON and persists a
local snapshot for offline / cold-start fallback.

Design decisions:
- Never blocks startup: if the live endpoint is unavailable, load from cache.
- Sanity-checks record count before swapping in a new fetch (prevents a
  partial/empty payload from silently shrinking the retrievable catalog).
- Validates payload shape against RawCatalogRecord before accepting it.
- Cache is always written on a successful fetch so the next cold-start is fast.
"""

from __future__ import annotations

import json
import logging
import structlog
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from src.catalog.schema_validator import RawCatalogRecord
from src.config.settings import get_settings

logger = structlog.get_logger(__name__)


class CatalogFetcher:
    """
    Fetches and validates the SHL catalog JSON.

    Responsibilities:
    1. HTTP GET → raw JSON list
    2. Validate each record (RawCatalogRecord)
    3. Sanity-check count vs. historical baseline / env minimum
    4. Persist valid payload to cache
    5. Fall back to cached snapshot on any failure
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    # ── Public ──────────────────────────────────────────────────────────────

    def fetch(self) -> list[RawCatalogRecord]:
        """
        Return validated raw records.  Never raises; always returns a list
        (possibly from cache).
        """
        cached = self._load_cache()

        try:
            raw = self._http_get()
            records = self._parse_and_validate(raw)
            if not self._count_sane(records, cached):
                logger.warning(
                    "catalog_fetch_rejected",
                    fetched=len(records),
                    cached=len(cached) if cached else 0,
                )
                return cached or []
            self._write_cache(records)
            logger.info("catalog_fetched_ok", count=len(records))
            return records

        except Exception as exc:  # noqa: BLE001
            logger.error("catalog_fetch_failed", error=str(exc))
            if cached:
                logger.warning("catalog_using_cache", count=len(cached))
                return cached
            logger.critical("catalog_no_cache_available")
            return []

    # ── Private ─────────────────────────────────────────────────────────────

    def _http_get(self) -> list[dict[str, Any]]:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(self._settings.catalog_json_url)
            resp.raise_for_status()
            # The upstream source occasionally includes invalid control chars
            data = json.loads(resp.text, strict=False)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")
        return data  # type: ignore[return-value]

    def _parse_and_validate(
        self, raw: list[dict[str, Any]]
    ) -> list[RawCatalogRecord]:
        records: list[RawCatalogRecord] = []
        errors = 0
        for item in raw:
            try:
                records.append(RawCatalogRecord.model_validate(item))
            except ValidationError as exc:
                errors += 1
                logger.debug("catalog_record_invalid", error=str(exc))
        if errors:
            logger.warning("catalog_records_skipped", count=errors)
        return records

    def _count_sane(
        self,
        fetched: list[RawCatalogRecord],
        cached: list[RawCatalogRecord] | None,
    ) -> bool:
        """Reject if fetched count is implausibly small."""
        min_absolute = self._settings.catalog_min_record_count
        if len(fetched) < min_absolute:
            return False
        if cached and len(fetched) < len(cached) * 0.5:
            # More than 50% shrinkage vs. last known good → suspicious
            return False
        return True

    def _load_cache(self) -> list[RawCatalogRecord] | None:
        path: Path = self._settings.catalog_cache_path
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                return None
            return [RawCatalogRecord.model_validate(r) for r in raw]
        except Exception as exc:  # noqa: BLE001
            logger.warning("catalog_cache_load_failed", error=str(exc))
            return None

    def _write_cache(self, records: list[RawCatalogRecord]) -> None:
        path: Path = self._settings.catalog_cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.model_dump() for r in records]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("catalog_cache_written", path=str(path))
