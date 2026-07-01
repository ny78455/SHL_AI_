"""
Turn counter — tracks budget relative to the 8-turn conversation cap.
"""

from __future__ import annotations

from src.domain.models import Message
from src.config.settings import get_settings


class TurnCounter:
    """
    Analyses the message history to determine the current turn number
    and whether a forced-commit is required.
    """

    def __init__(self) -> None:
        self._max_turns = get_settings().max_turns

    def count_assistant_turns(self, messages: list[Message]) -> int:
        """Number of completed assistant turns in the history."""
        return sum(1 for m in messages if m.role == "assistant")

    def current_turn_number(self, messages: list[Message]) -> int:
        """
        The turn number of the NEXT assistant response (1-based).
        = number of completed assistant turns + 1
        """
        return self.count_assistant_turns(messages) + 1

    def should_force_commit(self, messages: list[Message]) -> bool:
        """
        True when the next assistant turn is turn 7 or 8 (last two turns).
        The agent must commit to a shortlist rather than keep clarifying.
        """
        next_turn = self.current_turn_number(messages)
        return next_turn >= self._max_turns - 1

    def is_at_cap(self, messages: list[Message]) -> bool:
        """True if we have already hit the max turn limit."""
        return self.count_assistant_turns(messages) >= self._max_turns

    def has_repeated_clarification(self, messages: list[Message]) -> bool:
        """
        Detect looping: returns True if the last 2 assistant messages both
        ended without recommendations (clarify loop detected).
        """
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        if len(assistant_msgs) < 2:
            return False
        # Heuristic: if neither of the last 2 assistant replies contains "http"
        # (no recommendation URLs), we are likely looping
        last_two = assistant_msgs[-2:]
        return all("http" not in m.content for m in last_two)
