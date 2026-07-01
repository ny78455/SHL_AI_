"""
Conversation Orchestrator — the central pipeline controller.

Implements the full per-turn processing loop:
1. Validate request structure
2. Count turns / forced-commit check
3. Extract accumulated constraints from history
4. Reconstruct prior shortlist (for Refine turns)
5. Classify intent (scope guard → rule-based intent)
6. Build retrieval query
7. Run hybrid retrieval
8. Build LLM context (system prompt + catalog snippets + behavior block)
9. Generate with fallback LLM chain
10. Post-hoc validate (URL allow-list, test_type, schema)
11. Return ChatResponse

This class contains ZERO business rules itself — it only orchestrates the
injected collaborators. Every decision is delegated to a named component.
"""

from __future__ import annotations

import json
import logging
import structlog

from src.catalog.repository import InMemoryCatalogRepository
from src.config.settings import get_settings
from src.domain.models import (
    Assessment,
    ChatResponse,
    ConversationContext,
    Intent,
    Message,
    Recommendation,
)
from src.domain.ports import CatalogRepository, RetrievalPort
from src.llm.fallback_client import FallbackLLMClient
from src.llm.response_validator import ResponseValidator
from src.orchestration.constraint_extractor import ConstraintExtractor
from src.orchestration.intent_classifier import IntentClassifier
from src.orchestration.shortlist_parser import ShortlistParser
from src.orchestration.turn_counter import TurnCounter
from src.prompts.system_prompt import (
    CLARIFY_INSTRUCTIONS,
    COMPARE_INSTRUCTIONS,
    FORCE_COMMIT_INSTRUCTIONS,
    RECOMMEND_INSTRUCTIONS,
    REFINE_INSTRUCTIONS,
    REFUSE_INSTRUCTIONS,
    SYSTEM_PROMPT,
    build_context_block,
)
from src.retrieval.query_builder import QueryBuilder

logger = structlog.get_logger(__name__)


class ConversationOrchestrator:
    """
    The main per-turn pipeline.  All collaborators are injected via constructor
    so the class is fully testable without any real I/O.
    """

    def __init__(
        self,
        catalog: CatalogRepository,
        retriever: RetrievalPort,
        llm: FallbackLLMClient,
        validator: ResponseValidator,
        intent_classifier: IntentClassifier,
        shortlist_parser: ShortlistParser,
        turn_counter: TurnCounter,
        constraint_extractor: ConstraintExtractor,
        query_builder: QueryBuilder,
    ) -> None:
        self._catalog = catalog
        self._retriever = retriever
        self._llm = llm
        self._validator = validator
        self._intent_classifier = intent_classifier
        self._shortlist_parser = shortlist_parser
        self._turn_counter = turn_counter
        self._constraint_extractor = constraint_extractor
        self._query_builder = query_builder
        self._settings = get_settings()

    async def process(self, messages: list[Message]) -> ChatResponse:
        """
        Process one turn of the conversation and return a ChatResponse.
        Never raises — all exceptions are caught and a safe fallback returned.
        """
        try:
            return await self._process_internal(messages)
        except Exception as exc:  # noqa: BLE001
            logger.exception("orchestrator_unhandled_error", error=str(exc))
            return ChatResponse(
                reply=(
                    "I encountered an unexpected issue. "
                    "Could you rephrase or tell me more about the role you're hiring for?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

    async def _process_internal(self, messages: list[Message]) -> ChatResponse:
        # ── 1. Sanitise input ──────────────────────────────────────────────
        messages = self._sanitise_messages(messages)
        if not messages:
            return ChatResponse(
                reply="Hello! I'm an SHL Assessment Recommender. Tell me about the role you're hiring for and I'll suggest the right assessments.",
                recommendations=[],
                end_of_conversation=False,
            )

        last_user_msg = self._last_user_message(messages)

        # ── 2. Turn budget ─────────────────────────────────────────────────
        force_commit = self._turn_counter.should_force_commit(messages)
        is_looping = self._turn_counter.has_repeated_clarification(messages)

        # ── 3. Extract constraints & reconstruct shortlist ─────────────────
        constraints = self._constraint_extractor.extract(messages)
        prior_shortlist = self._shortlist_parser.parse(messages)

        # ── 4. Classify intent ─────────────────────────────────────────────
        intent = self._intent_classifier.classify_intent(
            user_message=last_user_msg,
            history=messages,
            has_prior_shortlist=bool(prior_shortlist),
        )

        # Force commit overrides CLARIFY when budget is exhausted
        if force_commit or is_looping:
            if intent == Intent.CLARIFY:
                intent = Intent.RECOMMEND
                logger.info("orchestrator_forced_commit", reason="turn_budget")

        logger.info(
            "orchestrator_turn",
            intent=intent.name,
            turn=self._turn_counter.current_turn_number(messages),
            force_commit=force_commit,
            prior_shortlist_size=len(prior_shortlist),
        )

        # ── 5. Build context ───────────────────────────────────────────────
        context = ConversationContext(
            messages=messages,
            turn_number=self._turn_counter.current_turn_number(messages),
            prior_shortlist=prior_shortlist,
            accumulated_constraints=constraints,
            intent=intent,
            force_commit=force_commit,
        )

        # ── 6. Retrieve catalog snippets ───────────────────────────────────
        retrieval_query = self._query_builder.build(context)
        top_k = self._settings.retrieval_top_k
        retrieved: list[Assessment] = []

        # On REFUSE turns, skip retrieval (not needed)
        if intent != Intent.REFUSE:
            retrieved = await self._retriever.search(retrieval_query, top_k)

        catalog_snippets = self._format_snippets(retrieved)

        # ── 7. Select behavior instruction ────────────────────────────────
        behavior_instruction = self._select_behavior_instruction(intent)

        # ── 8. Format prior shortlist for prompt ──────────────────────────
        prior_sl_text = (
            self._format_shortlist_for_prompt(prior_shortlist)
            if prior_shortlist and intent in (Intent.REFINE, Intent.COMPARE)
            else None
        )

        # ── 9. Build full context block ────────────────────────────────────
        context_block = build_context_block(
            catalog_snippets=catalog_snippets,
            current_shortlist=prior_sl_text,
            behavior_instruction=behavior_instruction,
            force_commit=force_commit,
        )

        # ── 10. LLM generation ─────────────────────────────────────────────
        raw_output = await self._llm.generate(
            system_prompt=SYSTEM_PROMPT,
            messages=messages,
            context_snippets=context_block,
        )

        # ── 11. Post-hoc validation ────────────────────────────────────────
        is_commit_turn = intent in (Intent.RECOMMEND, Intent.REFINE)
        response = self._validator.validate(raw_output, is_commit_turn=is_commit_turn)

        return response

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _sanitise_messages(messages: list[Message]) -> list[Message]:
        """
        Remove messages with empty content or unknown roles.
        An empty history is valid (cold start).
        """
        valid_roles = {"user", "assistant"}
        return [
            m for m in messages
            if m.role in valid_roles and m.content and m.content.strip()
        ]

    @staticmethod
    def _last_user_message(messages: list[Message]) -> str:
        for m in reversed(messages):
            if m.role == "user":
                return m.content
        return ""

    @staticmethod
    def _select_behavior_instruction(intent: Intent) -> str:
        return {
            Intent.CLARIFY: CLARIFY_INSTRUCTIONS,
            Intent.RECOMMEND: RECOMMEND_INSTRUCTIONS,
            Intent.REFINE: REFINE_INSTRUCTIONS,
            Intent.COMPARE: COMPARE_INSTRUCTIONS,
            Intent.REFUSE: REFUSE_INSTRUCTIONS,
        }[intent]

    @staticmethod
    def _format_snippets(assessments: list[Assessment]) -> str:
        """
        Format retrieved catalog records into a compact text block for the prompt.
        Each record is clearly delimited so the LLM can easily extract URLs verbatim.
        """
        if not assessments:
            return "(No matching assessments found in the catalog for this query.)"

        lines: list[str] = []
        for i, a in enumerate(assessments, 1):
            lines.append(f"--- Assessment {i} ---")
            lines.append(f"Name      : {a.name}")
            lines.append(f"URL       : {a.url}")
            lines.append(f"TestType  : {a.test_type}")
            lines.append(f"Keys      : {', '.join(a.keys)}")
            lines.append(f"Duration  : {a.duration}")
            lines.append(f"JobLevels : {', '.join(a.job_levels) or 'Not specified'}")
            lines.append(f"Languages : {', '.join(a.languages[:5]) if a.languages else 'Not specified'}")
            lines.append(f"Remote    : {'Yes' if a.remote else 'No'}")
            lines.append(f"Adaptive  : {'Yes' if a.adaptive else 'No'}")
            lines.append(f"Desc      : {a.description[:300]}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _format_shortlist_for_prompt(recs: list[Recommendation]) -> str:
        """Format the prior shortlist as a readable block for Refine/Compare prompts."""
        lines = ["Current shortlist:"]
        for i, r in enumerate(recs, 1):
            lines.append(f"  {i}. {r.name} | {r.test_type} | {r.url}")
        return "\n".join(lines)
