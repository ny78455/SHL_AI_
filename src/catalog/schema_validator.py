"""
Raw catalog record validator.

Validates the shape of each JSON record fetched from the SHL catalog endpoint
before any business logic touches it.  Unknown fields are silently ignored;
missing required fields raise a clear ValidationError.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class RawCatalogRecord(BaseModel):
    """
    Pydantic model mirroring the SHL catalog JSON shape (§3.1 of spec).
    Only `entity_id`, `name`, `link`, and `status` are strictly required;
    all other fields default gracefully so that records with sparse data
    are still usable rather than silently dropped.
    """

    entity_id: str
    name: str
    link: str
    status: str = "ok"

    # Optional fields — default to safe empty values
    description: str = ""
    job_levels: list[str] = Field(default_factory=list)
    job_levels_raw: str = ""
    languages: list[str] = Field(default_factory=list)
    languages_raw: str = ""
    duration: str = ""
    duration_raw: str = ""
    remote: str = "no"
    adaptive: str = "no"
    keys: list[str] = Field(default_factory=list)

    # test_type may be absent from JSON; classifier will derive it from keys
    test_type: Optional[str] = None

    scraped_at: Optional[str] = None

    @field_validator("entity_id", "name", "link", mode="before")
    @classmethod
    def _strip_whitespace(cls, v: object) -> str:
        if isinstance(v, str):
            return v.strip()
        return str(v)

    @field_validator("keys", "job_levels", "languages", mode="before")
    @classmethod
    def _coerce_list(cls, v: object) -> list[str]:
        """Accept either a pre-parsed list or a comma-separated raw string."""
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str) and v.strip():
            return [x.strip() for x in v.split(",") if x.strip()]
        return []

    @field_validator("remote", "adaptive", mode="before")
    @classmethod
    def _coerce_bool_string(cls, v: object) -> str:
        """Normalise to lowercase 'yes'/'no'."""
        return str(v).strip().lower() if v is not None else "no"
