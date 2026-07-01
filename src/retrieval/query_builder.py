"""
Query builder — synthesises a retrieval query from conversation context.

Design: The retrieval query is NOT just the latest user message. It is a
combination of:
1. Latest user message (most recent signal, repeated for extra weight)
2. The last few user turns, to preserve context across a Refine ("add AWS")
   without over-diluting the current-turn signal
3. Accumulated constraints from the whole conversation (role, seniority,
   skills, language, purpose), phrased as natural language
4. Prior shortlist item names (on REFINE / Confirm-of-shortlist turns only) —
   injected as a compact text block so BM25 registers the product tokens and
   Qdrant can surface those exact items again even when the user's message is
   as thin as "That's good." or "Locking it in."  Without this, a
   confirmation turn with a clean message retrieves almost none of the
   shortlist items, causing the LLM to hallucinate replacements or produce an
   incomplete re-emission — directly responsible for C8 recall collapse
   (four Microsoft Office items dropped on the final confirmation turn).

Recall@k fixes applied:
- Constraints were previously emitted as literal "role: X | seniority: Y"
  key-value pairs. Dense embedding models are trained on natural language,
  not schema-like tag lists, and score noticeably worse on unnatural phrasing
  for paraphrased/vague queries — the exact case (§3.4 of the spec) hybrid
  retrieval is meant to win on. This version renders constraints as a plain
  sentence instead.
- Only the single latest user message was ever included. On a Refine turn
  ("add AWS") this meant the retrieval query for that turn's sparse leg
  could be as thin as "add aws" if accumulated_constraints hadn't yet
  absorbed the new term from a prior orchestrator pass. This version also
  folds in the last 2 user turns (deduplicated) as light additional context,
  so a bare follow-up instruction still carries the surrounding conversation
  even if constraint extraction lags a turn behind.
- The latest user message is repeated once. This is a standard, cheap way to
  bias both the BM25 leg (higher term frequency for genuinely current-turn
  terms) and, marginally, the dense leg (many embedding models are
  order/frequency sensitive) toward what the user is asking about *right
  now*, without discarding the accumulated context entirely.
- Prior shortlist names are now injected on Refine / Confirm turns (fix #4
  above). The names are appended ONCE (not repeated) so they provide a recall
  floor without drowning out the actual intent signal of the user's message.
"""
from __future__ import annotations

from src.domain.models import ConversationContext, Intent

_MAX_PRIOR_USER_TURNS = 2


class QueryBuilder:
    """
    Builds a dense+sparse retrieval query from a ConversationContext.
    The output is a plain string that the embedder and sparse encoder both
    receive as their input.
    """

    def build(self, context: ConversationContext) -> str:
        """
        Synthesise a retrieval query from the conversation context.
        Returns a human-readable sentence-like string optimised for embedding.
        """
        parts: list[str] = []

        user_messages = [
            msg.content.strip()
            for msg in context.messages
            if msg.role == "user" and msg.content and msg.content.strip()
        ]

        if user_messages:
            latest = user_messages[-1]
            # Repeat the latest message for recency weighting (see docstring).
            parts.append(latest)
            parts.append(latest)

            # Light additional context from the turns just before it, so a
            # short Refine instruction ("add AWS") doesn't retrieve in a
            # vacuum. Deduplicated against the latest message.
            prior = [m for m in user_messages[-(1 + _MAX_PRIOR_USER_TURNS):-1] if m != latest]
            if prior:
                parts.append(" ".join(prior))

        # Accumulated constraints, rendered as natural language rather than
        # key:value tags — embedding models handle prose far better.
        constraints = context.accumulated_constraints
        if constraints:
            sentence = self._constraints_to_sentence(constraints)
            if sentence:
                parts.append(sentence)

        # On Refine / Confirm-of-shortlist turns, inject prior shortlist
        # product names so BM25 can recall those exact items even when the
        # user's message is a bare acknowledgement ("That's good.",
        # "Locking it in."). Names are appended once — enough to register as
        # vocabulary hits without swamping the intent signal (fix #4).
        if context.intent in (Intent.REFINE,) and context.prior_shortlist:
            shortlist_names = " ".join(
                rec.name for rec in context.prior_shortlist
            )
            if shortlist_names:
                parts.append(shortlist_names)

        return " ".join(parts) if parts else "SHL assessment"

    @staticmethod
    def _constraints_to_sentence(constraints: dict) -> str:
        role = constraints.get("role")
        seniority = constraints.get("seniority")
        skills = constraints.get("skills")
        language = constraints.get("language")
        purpose = constraints.get("purpose")

        fragments: list[str] = []

        subject = " ".join(p for p in [seniority, role] if p)
        if subject:
            fragments.append(f"Hiring for a {subject} position.")
        elif role:
            fragments.append(f"Hiring for a {role} position.")

        if skills:
            skills_text = skills if isinstance(skills, str) else ", ".join(skills)
            fragments.append(f"Required skills: {skills_text}.")

        if language:
            fragments.append(f"Assessment should be available in {language}.")

        if purpose:
            fragments.append(f"Purpose: {purpose}.")

        return " ".join(fragments)