"""
Regex entity matcher — exact/near-exact catalog mention detection.

This is the third retrieval signal, sitting ABOVE BM25 + Dense hybrid search,
not replacing either:

- Dense: handles vague, paraphrased queries ("stakeholder-facing Java dev").
- BM25:  handles lexical overlap on descriptions/keys, ranked in fusion.
- Regex: handles the case where the user (or a Compare/Refine turn) names a
  specific catalog product or its acronym verbatim — "OPQ32r", "GSA", "DSI",
  "the Contact Center Call Simulation". In that case there's essentially no
  ambiguity about intent, so instead of trusting hybrid ranking to surface it
  (which can lose to sharper semantic matches on a crowded turn — the same
  problem the OPQ32r default-inclusion fix addressed), a regex hit is treated
  as a near-certain, explicit request and force-included with top priority.

This directly targets the spec's own justification for hybrid over
pure-vector retrieval (§3.4): "pure embedding search under-performs on exact
product-name queries... pure lexical search under-performs on paraphrased
vague queries." Regex closes the remaining gap: even BM25 has to *win* a
ranked fusion; regex hits don't compete, they're guaranteed.

Safety: an alias that maps to more than one distinct catalog entity is
treated as ambiguous and is NOT force-included by either match — per the
spec's own "ambiguous -> exclude rather than wrongly include" default
(§12/§13.1), silently guessing which entity an ambiguous acronym refers to is
worse than leaving it to BM25 + Dense to sort out on relevance alone.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

from src.domain.models import Assessment

logger = structlog.get_logger(__name__)

# Acronym-like tokens: 2+ leading uppercase letters, then any run of
# alphanumerics (catches "OPQ32r", "GSA", "DSI", "SVAR", "HIPAA", "SQL").
_ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,}[A-Za-z0-9]*\b")

_MIN_PHRASE_ALIAS_LEN = 4  # avoid trivial/noisy full-name substring matches


def _normalise_phrase(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\(new\)", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class _AliasIndex:
    # alias string -> set of entity_ids it could refer to (size > 1 = ambiguous)
    acronyms: dict[str, set[str]] = field(default_factory=dict)
    phrases: dict[str, set[str]] = field(default_factory=dict)


class RegexEntityMatcher:
    """
    Builds an alias index from the catalog once at startup, then matches
    incoming query text against it. Cheap enough to run on every turn.
    """

    def __init__(self) -> None:
        self._index = _AliasIndex()
        self._built = False

    def build(self, assessments: list[Assessment]) -> None:
        """Call once at startup, same lifecycle point as
        QdrantIndex.setup_sparse_encoder / HybridRetriever pinned-default
        resolution — rebuilding per-turn would be wasted work since the
        catalog doesn't change mid-conversation."""
        index = _AliasIndex()
        for a in assessments:
            entity_id = a.entity_id
            name = a.name

            phrase = _normalise_phrase(name)
            if len(phrase) >= _MIN_PHRASE_ALIAS_LEN:
                index.phrases.setdefault(phrase, set()).add(entity_id)

            for m in _ACRONYM_PATTERN.finditer(name):
                acronym = m.group(0).lower()
                index.acronyms.setdefault(acronym, set()).add(entity_id)

        self._index = index
        self._built = True
        n_ambiguous = sum(1 for v in {**index.acronyms, **index.phrases}.values() if len(v) > 1)
        logger.info(
            "entity_matcher_built",
            acronyms=len(index.acronyms),
            phrases=len(index.phrases),
            ambiguous_aliases=n_ambiguous,
        )

    def match(self, text: str) -> list[str]:
        """
        Return entity_ids explicitly (and unambiguously) named in `text`,
        in order of first appearance, deduplicated.
        """
        if not self._built:
            return []

        found: list[str] = []
        seen: set[str] = set()

        # Acronyms: case-insensitive, word-boundary match against raw text.
        for m in re.finditer(r"\b[A-Za-z][A-Za-z0-9]*\b", text):
            token = m.group(0).lower()
            entities = self._index.acronyms.get(token)
            if not entities:
                continue
            if len(entities) > 1:
                continue  # ambiguous — leave to BM25/Dense, don't force it
            (entity_id,) = tuple(entities)
            if entity_id not in seen:
                found.append(entity_id)
                seen.add(entity_id)

        # Full/near-full product name phrases, as a substring of the
        # normalised query text.
        normalised_text = _normalise_phrase(text)
        for phrase, entities in self._index.phrases.items():
            if phrase in normalised_text:
                if len(entities) > 1:
                    continue
                (entity_id,) = tuple(entities)
                if entity_id not in seen:
                    found.append(entity_id)
                    seen.add(entity_id)

        return found