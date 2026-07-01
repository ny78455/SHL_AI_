"""
Centralised application configuration.

All tuneable parameters live here.  Business logic imports `get_settings()`
and never reads `os.environ` directly, keeping concerns isolated and the
settings trivially mockable in tests.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings, sourced from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Gemini ────────────────────────────────────────────────────────────────
    gemini_api_key: str = Field(..., description="Google Gemini API key")
    llm_model: str = Field(
        default="gemini-2.5-flash-lite-preview-06-17",
        description="Gemini model identifier",
    )
    llm_timeout_seconds: int = Field(default=20, ge=5, le=30)
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="models/gemini-embedding-001",
        description="Gemini embedding model name",
    )
    embedding_dimension: int = Field(default=768, ge=1)

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str = Field(default="")
    qdrant_collection_name: str = Field(default="shl_assessments")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=15, ge=1, le=50)

    # ── Catalog ───────────────────────────────────────────────────────────────
    catalog_json_url: str = Field(
        default="https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
    )
    catalog_cache_path: Path = Field(default=Path("data/catalog_cache.json"))
    catalog_min_record_count: int = Field(
        default=100,
        description="Minimum accepted record count; reject fetches below this.",
    )

    # ── Conversation ──────────────────────────────────────────────────────────
    max_turns: int = Field(default=8, ge=2, le=20)

    @field_validator("catalog_cache_path", mode="before")
    @classmethod
    def _coerce_path(cls, v: object) -> Path:
        return Path(str(v))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()  # type: ignore[call-arg]
