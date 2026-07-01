"""
Constraint extractor — parses accumulated hiring constraints from the full
conversation history to enrich the retrieval query.

Extracted constraints:
- role / job family
- seniority / level
- skills (comma-separated list)
- language (delivery language)
- purpose (selection / development / audit)

These are used by QueryBuilder to synthesise a richer retrieval query that
includes context from earlier turns, not just the latest message.
"""

from __future__ import annotations

import re

from src.domain.models import Message


# ── Signal patterns ────────────────────────────────────────────────────────────

_SENIORITY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(cxo|c-?suite|director|vp|executive|head\s+of)\b", re.I), "executive"),
    (re.compile(r"\b(senior|sr\.?|lead|principal|staff)\b", re.I), "senior"),
    (re.compile(r"\b(mid[\s-]?level?|mid[\s-]?professional|3[\s-]?to[\s-]?7|4[\s-]?years?)\b", re.I), "mid"),
    (re.compile(r"\b(junior|jr\.?|entry[\s-]?level|graduate|fresh(er)?)\b", re.I), "entry"),
]

_PURPOSE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(select|hiring|recruit|screen)\b", re.I), "selection"),
    (re.compile(r"\b(develop|grow|coaching|re[\s-]?skill|upskill|audit)\b", re.I), "development"),
]

_LANGUAGE_PATTERN = re.compile(
    r"\b(english|spanish|french|german|portuguese|chinese|arabic|dutch|italian|"
    r"hindi|japanese|korean|russian|swedish|danish|norwegian|finnish|polish)\b",
    re.I,
)


class ConstraintExtractor:
    """
    Extracts key hiring constraints from the full conversation history.
    Returns a dict suitable for QueryBuilder and prompt context.
    """

    def extract(self, messages: list[Message]) -> dict[str, str]:
        """
        Scan all user messages (in order) and return the most recently
        mentioned value for each constraint dimension.
        """
        constraints: dict[str, str] = {}
        all_user_text = " ".join(
            m.content for m in messages if m.role == "user"
        )

        # Seniority
        for pattern, label in _SENIORITY_PATTERNS:
            if pattern.search(all_user_text):
                constraints["seniority"] = label
                break

        # Purpose
        for pattern, label in _PURPOSE_PATTERNS:
            if pattern.search(all_user_text):
                constraints["purpose"] = label
                break

        # Language
        lang_match = _LANGUAGE_PATTERN.search(all_user_text)
        if lang_match:
            constraints["language"] = lang_match.group(0).lower()

        # Role / skills — use the first user message as role signal
        # (subsequent messages typically refine rather than reset the role)
        first_user = next((m for m in messages if m.role == "user"), None)
        if first_user:
            constraints["role"] = first_user.content[:200]

        # Skills — collect technology/domain keywords across all user messages
        skills = self._extract_skills(all_user_text)
        if skills:
            constraints["skills"] = ", ".join(skills)

        return constraints

    @staticmethod
    def _extract_skills(text: str) -> list[str]:
        """
        Extract technology and domain keywords from user messages.
        Not exhaustive — used for retrieval query enrichment only.
        """
        skill_pattern = re.compile(
            r"\b("
            r"java|python|javascript|typescript|rust|go|golang|c\+\+|c#|\.net|php|ruby|swift|kotlin|scala|"
            r"spring|angular|react|vue|node\.?js|django|flask|fastapi|"
            r"sql|postgresql|mysql|oracle|mongodb|redis|"
            r"aws|azure|gcp|docker|kubernetes|k8s|terraform|ci/?cd|devops|"
            r"machine\s*learning|data\s*science|analytics|finance|accounting|"
            r"hipaa|medical|healthcare|nursing|"
            r"sales|customer\s*service|contact\s*cent(?:er|re)|retail|"
            r"manufacturing|industrial|safety|plant|chemical|"
            r"networking|linux|windows|excel|word|powerpoint|office\s*365"
            r")\b",
            re.I,
        )
        matches = skill_pattern.findall(text)
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for m in matches:
            lower = m.lower()
            if lower not in seen:
                seen.add(lower)
                unique.append(lower)
        return unique
