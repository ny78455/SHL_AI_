"""
Query builder — synthesises a retrieval query from conversation context.

Design: The retrieval query is NOT just the latest user message.  It is a
combination of:
1. Latest user message (most recent signal)
2. Accumulated constraints from prior turns (role, seniority, skills, language, purpose)

This ensures that on a Refine turn ("add AWS") the retrieval still includes
the full context ("senior full-stack Java engineer, SQL, AWS, Docker") rather
than only searching for "AWS".
"""

from __future__ import annotations

from src.domain.models import ConversationContext


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

        # Latest user message (highest weight — put first)
        if context.messages:
            for msg in reversed(context.messages):
                if msg.role == "user":
                    parts.append(msg.content.strip())
                    break

        # Accumulated constraints
        constraints = context.accumulated_constraints
        if constraints:
            constraint_parts = []
            if role := constraints.get("role"):
                constraint_parts.append(f"role: {role}")
            if seniority := constraints.get("seniority"):
                constraint_parts.append(f"seniority: {seniority}")
            if skills := constraints.get("skills"):
                constraint_parts.append(f"skills: {skills}")
            if language := constraints.get("language"):
                constraint_parts.append(f"language: {language}")
            if purpose := constraints.get("purpose"):
                constraint_parts.append(f"purpose: {purpose}")
            if constraint_parts:
                parts.append(" | ".join(constraint_parts))

        return " ".join(parts) if parts else "SHL assessment"
