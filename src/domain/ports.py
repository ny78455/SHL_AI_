"""
Abstract port interfaces (dependency inversion boundaries).

Business logic depends only on these abstract types, never on concrete
implementations (Qdrant, Gemini, httpx).  Concrete adapters implement
these protocols and are injected via FastAPI's dependency system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.models import Assessment, ChatResponse, Message, Recommendation


class CatalogRepository(ABC):
    """Read-only access to the indexed catalog of SHL Individual Test Solutions."""

    @abstractmethod
    def get_all(self) -> list[Assessment]:
        """Return every assessment in the catalog."""

    @abstractmethod
    def get_by_entity_id(self, entity_id: str) -> Assessment | None:
        """Return a single assessment by its stable catalog ID, or None."""

    @abstractmethod
    def get_by_url(self, url: str) -> Assessment | None:
        """Return a single assessment by its catalog URL, or None."""

    @abstractmethod
    def url_exists(self, url: str) -> bool:
        """True iff the URL belongs to a verified catalog record."""


class RetrievalPort(ABC):
    """Semantic + lexical hybrid retrieval over the catalog index."""

    @abstractmethod
    async def search(self, query: str, top_k: int) -> list[Assessment]:
        """
        Return the top-K most relevant assessments for the given query.
        Implementations must handle partial failures (e.g., vector store
        unavailable) by degrading gracefully rather than raising.
        """


class LLMPort(ABC):
    """Text generation via a language model."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        messages: list[Message],
        context_snippets: str,
    ) -> str:
        """
        Generate a response string.  The implementation is responsible for
        JSON-mode enforcement, retries, and timeouts.
        Returns the raw LLM output string; schema validation is done upstream.
        """


class IntentClassifierPort(ABC):
    """Classify the intent of the latest user turn."""

    @abstractmethod
    async def classify(
        self,
        last_user_message: str,
        history: list[Message],
    ) -> str:
        """
        Return one of: "CLARIFY", "RECOMMEND", "REFINE", "COMPARE", "REFUSE".
        Rule-based implementation should run synchronously but the interface
        is async to allow an LLM-backed fallback without changing callers.
        """
