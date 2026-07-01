"""
Shortlist parser — re-derives the prior shortlist from conversation history.

Since the API is stateless, the server never stores the current shortlist.
Instead, the orchestrator re-parses its own previous structured output from
the message history to determine what to diff on a Refine turn.

Strategy (in descending reliability order):
1. Parse recommendations JSON from the most recent assistant message that
   contains a valid "recommendations" field with 1+ items.
2. Fall back to parsing a Markdown table from the assistant reply text.
3. Return empty list if nothing can be confidently parsed.
"""

from __future__ import annotations

import json
import logging
import structlog
import re
from typing import Any

from src.domain.models import Message, Recommendation

logger = structlog.get_logger(__name__)

# Matches a markdown table row:  | 1 | Name | K | ... | url |
_TABLE_ROW_RE = re.compile(
    r"\|\s*\d+\s*\|"           # row number
    r"\s*(?P<name>[^|]+?)\s*\|"
    r"\s*(?P<type>[^|]+?)\s*\|"
    r".*?"                      # keys / duration / languages (ignored)
    r"\s*(?P<url>https?://[^\s|]+)\s*\|",
    re.IGNORECASE,
)


class ShortlistParser:
    """
    Scans the message history backwards to reconstruct the most recent
    committed shortlist the agent emitted.
    """

    def parse(self, messages: list[Message]) -> list[Recommendation]:
        """
        Return the most recently committed shortlist, or [] if none found.
        Scans from the tail of messages to find the last assistant turn that
        carried a non-empty recommendations block.
        """
        for message in reversed(messages):
            if message.role != "assistant":
                continue

            recs = self._try_parse_json(message.content)
            if recs:
                logger.debug("shortlist_parsed_from_json", count=len(recs))
                return recs

            recs = self._try_parse_markdown_table(message.content)
            if recs:
                logger.debug("shortlist_parsed_from_markdown", count=len(recs))
                return recs

        logger.debug("shortlist_no_prior_found")
        return []

    # ── Parsing strategies ─────────────────────────────────────────────────

    @staticmethod
    def _try_parse_json(content: str) -> list[Recommendation]:
        """Try to parse recommendations from a JSON block in the message."""
        try:
            # The message might be raw JSON or JSON embedded in prose
            # Try direct parse first
            data: Any = json.loads(content)
            if isinstance(data, dict):
                recs = data.get("recommendations") or []
                if isinstance(recs, list) and recs:
                    results = []
                    for item in recs:
                        if isinstance(item, dict) and item.get("url") and item.get("name"):
                            results.append(Recommendation(
                                name=item["name"],
                                url=item["url"],
                                test_type=item.get("test_type", ""),
                            ))
                    return results
        except (json.JSONDecodeError, TypeError):
            pass
        return []

    @staticmethod
    def _try_parse_markdown_table(content: str) -> list[Recommendation]:
        """Try to extract recommendations from a Markdown table in the reply text."""
        results: list[Recommendation] = []
        for match in _TABLE_ROW_RE.finditer(content):
            name = match.group("name").strip()
            test_type = match.group("type").strip()
            url = match.group("url").strip()
            if name and url:
                results.append(Recommendation(
                    name=name,
                    url=url,
                    test_type=test_type,
                ))
        return results
