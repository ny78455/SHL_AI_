"""
Domain models — pure dataclasses with no external dependencies.

These are the core business objects that every layer speaks in.
Infrastructure adapters (Qdrant rows, Gemini JSON, HTTP schemas) are
converted TO these types at their layer boundary, never passed through raw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ── Catalog entity ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Assessment:
    """A single SHL Individual Test Solution from the catalog."""

    entity_id: str
    name: str
    url: str
    description: str
    keys: list[str]          # category taxonomy, e.g. ["Knowledge & Skills"]
    test_type: str           # SHL code(s), e.g. "K" or "K,S"
    job_levels: list[str]
    languages: list[str]
    duration: str            # free-form, e.g. "30 minutes", "Untimed", ""
    remote: bool
    adaptive: bool

    def searchable_text(self) -> str:
        """Concatenated text used for embedding and BM25 indexing."""
        parts = [
            self.name,
            self.description,
            " ".join(self.keys),
            " ".join(self.job_levels),
            self.test_type,
        ]
        return " ".join(p for p in parts if p)


# ── Conversation primitives ────────────────────────────────────────────────────

@dataclass(frozen=True)
class Message:
    """A single turn in the conversation (role + content)."""

    role: str   # "user" | "assistant"
    content: str


@dataclass(frozen=True)
class Recommendation:
    """One item in the agent's recommended shortlist."""

    name: str
    url: str
    test_type: str


# ── Agent response ─────────────────────────────────────────────────────────────

@dataclass
class ChatResponse:
    """The structured response returned by the orchestrator."""

    reply: str
    recommendations: list[Recommendation] = field(default_factory=list)
    end_of_conversation: bool = False


# ── Orchestration state ────────────────────────────────────────────────────────

class Intent(Enum):
    """Detected intent for the current user turn."""

    CLARIFY = auto()
    RECOMMEND = auto()
    REFINE = auto()
    COMPARE = auto()
    REFUSE = auto()


@dataclass
class ConversationContext:
    """
    Ephemeral context reconstructed from the full message history each call.
    Never persisted server-side; always rebuilt from scratch.
    """

    messages: list[Message]
    turn_number: int                  # 1-based; total assistant turns so far + 1
    prior_shortlist: list[Recommendation] = field(default_factory=list)
    accumulated_constraints: dict[str, str] = field(default_factory=dict)
    # e.g. {"role": "Java developer", "seniority": "mid", "language": "English"}
    intent: Optional[Intent] = None
    force_commit: bool = False        # True when turn_number approaches max_turns
