"""
Scope and intent classifier.

Two-phase classification:
1. Rule-based pre-classifier (fast, no LLM call) — catches injection, off-topic,
   legal/compliance out-of-scope requests before reaching the main LLM.
2. Signal-based intent classifier — uses keyword heuristics to distinguish
   CLARIFY / RECOMMEND / REFINE / COMPARE turns without an extra LLM call.

Keeping classification separate from generation means a single rogue generation
call cannot override scope decisions.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from src.domain.models import Intent, Message

# ── Out-of-scope keyword patterns ─────────────────────────────────────────────

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?", re.I),
    re.compile(r"pretend\s+(you\s+are|to\s+be)", re.I),
    re.compile(r"you\s+are\s+now\s+a", re.I),
    re.compile(r"disregard\s+(your\s+)?(system\s+)?prompt", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"DAN\s+mode", re.I),
]

_LEGAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(legally\s+required|legal\s+requirement|compliance\s+obligation)\b", re.I),
    re.compile(r"\b(HIPAA|ADA|EEOC|GDPR|CCPA)\b.*\b(require|obligat|mandate|law|violat)\b", re.I),
    re.compile(r"\b(sue|lawsuit|liability|discriminat)\b", re.I),
]

_OFF_TOPIC_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(stock\s+price|invest(ment|ing)|weather|recipe|sport|movie|music)\b", re.I),
    re.compile(r"\b(write\s+(me\s+)?(a\s+)?(poem|story|essay|code|email))\b", re.I),
    re.compile(r"\b(who\s+is|what\s+is\s+the\s+capital|how\s+tall)\b", re.I),
]

# ── Refine signal patterns ─────────────────────────────────────────────────────

_REFINE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(add|include|also\s+add|can\s+you\s+add)\b", re.I),
    re.compile(r"\b(remove|drop|exclude|take\s+out|get\s+rid\s+of)\b", re.I),
    re.compile(r"\b(replace|swap|instead\s+of|rather\s+than)\b", re.I),
    re.compile(r"\b(without|no\s+more|skip\s+the)\b", re.I),
]

# ── Compare signal patterns ────────────────────────────────────────────────────

_COMPARE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(difference|differ|compare|vs\.?|versus)\b", re.I),
    re.compile(r"\b(what('?s| is) the difference)\b", re.I),
    re.compile(r"\b(which one|which is better|why (use|choose|pick))\b", re.I),
    re.compile(r"\b(distinguish|how does .+ differ)\b", re.I),
]

# ── Confirm / end patterns ─────────────────────────────────────────────────────

_CONFIRM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(perfect|confirmed|that('?s| is) (good|great|fine|what we need)|looks? good|go ahead|yes)[\.\!]*$", re.I),
    re.compile(r"\b(keep (the )?(shortlist|list)|locking it in|final list)\b", re.I),
]


class ScopeDecision(NamedTuple):
    is_in_scope: bool
    refuse_reason: str  # empty string if in scope


class IntentClassifier:
    """
    Stateless intent classifier.  Pure functions — no I/O, no LLM calls.
    All classify_* methods accept the latest user message and history.
    """

    # ── Scope guard ────────────────────────────────────────────────────────

    def check_scope(self, user_message: str) -> ScopeDecision:
        """
        Run the scope guard.  Returns a ScopeDecision indicating whether
        the message is in scope and the reason if not.
        """
        if self._is_injection(user_message):
            return ScopeDecision(False, "prompt_injection")
        if self._is_legal_query(user_message):
            return ScopeDecision(False, "legal_compliance")
        if self._is_off_topic(user_message):
            return ScopeDecision(False, "off_topic")
        return ScopeDecision(True, "")

    # ── Intent classification ──────────────────────────────────────────────

    def classify_intent(
        self,
        user_message: str,
        history: list[Message],
        has_prior_shortlist: bool,
    ) -> Intent:
        """
        Classify the intent of the latest user turn.
        Checks for REFUSE first, then COMPARE, then REFINE, then
        RECOMMEND/CLARIFY based on context.
        """
        scope = self.check_scope(user_message)
        if not scope.is_in_scope:
            return Intent.REFUSE

        if self._is_compare(user_message):
            return Intent.COMPARE

        if has_prior_shortlist and self._is_refine(user_message):
            return Intent.REFINE

        if self._is_confirm(user_message) and has_prior_shortlist:
            # Confirm of existing shortlist → Refine (re-emit unchanged)
            return Intent.REFINE

        # Decide CLARIFY vs RECOMMEND based on context richness
        if self._has_enough_context(user_message, history):
            return Intent.RECOMMEND

        return Intent.CLARIFY

    # ── Named pattern checks ───────────────────────────────────────────────

    @staticmethod
    def _is_injection(text: str) -> bool:
        return any(p.search(text) for p in _INJECTION_PATTERNS)

    @staticmethod
    def _is_legal_query(text: str) -> bool:
        return any(p.search(text) for p in _LEGAL_PATTERNS)

    @staticmethod
    def _is_off_topic(text: str) -> bool:
        return any(p.search(text) for p in _OFF_TOPIC_PATTERNS)

    @staticmethod
    def _is_refine(text: str) -> bool:
        return any(p.search(text) for p in _REFINE_PATTERNS)

    @staticmethod
    def _is_compare(text: str) -> bool:
        return any(p.search(text) for p in _COMPARE_PATTERNS)

    @staticmethod
    def _is_confirm(text: str) -> bool:
        return any(p.search(text) for p in _CONFIRM_PATTERNS)

    @staticmethod
    def _has_enough_context(user_message: str, history: list[Message]) -> bool:
        """
        Heuristic: we have enough context to recommend if:
        - The conversation has had at least 2 user turns, OR
        - The user message contains a specific role + one other signal
        """
        user_turns = sum(1 for m in history if m.role == "user")
        if user_turns >= 2:
            return True

        # Single-turn richness check: mention of a job title AND some skill/level
        has_role = bool(re.search(
            r"\b(engineer|developer|analyst|manager|sales|admin|nurse|operator|agent|trainee)\b",
            user_message, re.I
        ))
        has_signal = bool(re.search(
            r"\b(senior|junior|mid|entry|graduate|level|years?|cognitive|personality|skill)\b",
            user_message, re.I
        ))
        return has_role and has_signal
