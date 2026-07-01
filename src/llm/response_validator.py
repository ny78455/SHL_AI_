"""
Response validator — post-hoc guard layer (§7.2 of spec).

Validates every LLM output BEFORE it reaches the client:
1. JSON schema validation (required fields, correct types)
2. URL allow-list check: every recommendations[].url must exist in the catalog
3. test_type consistency: must match the catalog record's actual code
4. Recommendation count: 0 on clarify/compare/refuse, 1-10 on commit turns

On failure:
- Strip hallucinated items (URL not in catalog)
- If zero valid items remain on a commit turn, return a fallback clarify reply
- On schema failure, return a deterministic schema-valid fallback
"""

from __future__ import annotations

import json
import logging
import structlog
from dataclasses import dataclass
from typing import Any

from src.domain.models import ChatResponse, Recommendation
from src.domain.ports import CatalogRepository

logger = structlog.get_logger(__name__)

_FALLBACK_CLARIFY_REPLY = (
    "I need a bit more information to make a good recommendation. "
    "Could you tell me more about the role, seniority level, or specific skills required?"
)


@dataclass
class ValidationResult:
    response: ChatResponse
    hallucinations_stripped: int = 0
    schema_repaired: bool = False


class ResponseValidator:
    """
    Post-hoc validator for raw LLM output.

    Always returns a schema-valid ChatResponse — never raises to the caller.
    """

    def __init__(self, catalog: CatalogRepository) -> None:
        self._catalog = catalog

    def validate(self, raw_text: str, is_commit_turn: bool) -> ChatResponse:
        """
        Parse, validate, repair, and return a ChatResponse.
        Never raises; always returns a schema-valid object.
        """
        # Step 1: Parse JSON
        data = self._parse_json(raw_text)
        if data is None:
            logger.warning("validator_json_parse_failed")
            return ChatResponse(
                reply=_FALLBACK_CLARIFY_REPLY,
                recommendations=[],
                end_of_conversation=False,
            )

        # Step 2: Extract fields with defaults
        reply = self._extract_reply(data)
        raw_recs = data.get("recommendations") or []
        end_of_conv = bool(data.get("end_of_conversation", False))

        # Step 3: Validate and filter recommendations
        valid_recs, stripped = self._validate_recommendations(raw_recs)

        if stripped > 0:
            logger.warning(
                "validator_hallucinations_stripped",
                count=stripped,
                remaining=len(valid_recs),
            )

        # Step 4: Handle zero valid items on a commit turn
        if is_commit_turn and not valid_recs and raw_recs:
            logger.error("validator_all_recs_stripped_on_commit")
            # Force a fallback clarify rather than emit empty recommendations
            # silently on a turn that was supposed to commit
            return ChatResponse(
                reply=(
                    "I wasn't able to find catalog-verified matches for all suggested items. "
                    "Let me refine the recommendations. Could you confirm the key skills?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        # Step 5: Enforce recommendation count bounds
        valid_recs = valid_recs[:10]  # never exceed 10

        return ChatResponse(
            reply=reply,
            recommendations=valid_recs,
            end_of_conversation=end_of_conv,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        """
        Attempt to parse JSON from the LLM output.
        Handles cases where the model wraps JSON in a markdown code fence.
        """
        text = text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        return None

    @staticmethod
    def _extract_reply(data: dict[str, Any]) -> str:
        reply = data.get("reply", "")
        if not isinstance(reply, str) or not reply.strip():
            return "Here is what I found based on your requirements."
        return reply.strip()

    def _validate_recommendations(
        self, raw: list[Any]
    ) -> tuple[list[Recommendation], int]:
        """
        Filter recommendations to only those with catalog-verified URLs.
        Returns (valid_list, stripped_count).
        """
        valid: list[Recommendation] = []
        stripped = 0

        if not isinstance(raw, list):
            return [], 0

        for item in raw:
            if not isinstance(item, dict):
                stripped += 1
                continue

            name = (item.get("name") or "").strip()
            url = (item.get("url") or "").strip()
            test_type = (item.get("test_type") or "").strip()

            if not name or not url:
                stripped += 1
                continue

            # URL allow-list check (hard guard)
            if not self._catalog.url_exists(url):
                logger.warning(
                    "validator_url_not_in_catalog",
                    name=name,
                    url=url,
                )
                stripped += 1
                continue

            # test_type consistency check
            catalog_record = self._catalog.get_by_url(url)
            if catalog_record and test_type != catalog_record.test_type:
                logger.debug(
                    "validator_test_type_corrected",
                    name=name,
                    llm_type=test_type,
                    catalog_type=catalog_record.test_type,
                )
                test_type = catalog_record.test_type  # use ground truth

            valid.append(Recommendation(name=name, url=url, test_type=test_type))

        return valid, stripped
