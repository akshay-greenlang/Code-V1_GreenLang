# -*- coding: utf-8 -*-
"""Mapping base types."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MappingConfidence(str, Enum):
    """Banded confidence score for a mapping.

    The numeric ``confidence`` field is still the source of truth; the
    banded enum is for operators who think in "high / medium / low".
    """

    EXACT = "exact"           # 0.95–1.00
    HIGH = "high"             # 0.80–0.95
    MEDIUM = "medium"         # 0.60–0.80
    LOW = "low"               # 0.30–0.60
    UNKNOWN = "unknown"       # < 0.30

    @classmethod
    def from_score(cls, score: float) -> "MappingConfidence":
        if score >= 0.95:
            return cls.EXACT
        if score >= 0.80:
            return cls.HIGH
        if score >= 0.60:
            return cls.MEDIUM
        if score >= 0.30:
            return cls.LOW
        return cls.UNKNOWN


class MappingError(ValueError):
    """Raised when a mapping cannot be produced with any confidence."""


@dataclass
class MappingResult:
    """Return value of every mapping function.

    Attributes:
        canonical: The canonical key (string) or structured key (dict).
        confidence: Float 0.0–1.0.
        band: Banded confidence (EXACT / HIGH / MEDIUM / LOW / UNKNOWN).
        rationale: Human-readable explanation of why this match was picked.
        matched_pattern: The synonym / regex / code that triggered the match.
        alternates: Other candidate keys considered (top 5).
        raw_input: The original user input (for audit trails).
    """

    canonical: Any
    confidence: float
    band: MappingConfidence
    rationale: str
    matched_pattern: Optional[str] = None
    alternates: List[Any] = field(default_factory=list)
    raw_input: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical": self.canonical,
            "confidence": round(self.confidence, 3),
            "band": self.band.value,
            "rationale": self.rationale,
            "matched_pattern": self.matched_pattern,
            "alternates": self.alternates,
            "raw_input": self.raw_input,
        }


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------


_PUNCT = re.compile(r"[^\w\s-]")
_MULTI_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation (except `-`), collapse whitespace."""
    if s is None:
        return ""
    out = _PUNCT.sub(" ", str(s).lower())
    out = _MULTI_WS.sub(" ", out).strip()
    return out


# ---------------------------------------------------------------------------
# Shared mixin for table-driven mappings
# ---------------------------------------------------------------------------


class BaseMapping:
    """Shared helpers for synonym-table-driven mappings.

    Subclasses define::

        TAXONOMY = {
            "diesel": {
                "synonyms": ["diesel fuel", "distillate", "no. 2 diesel", ...],
                "meta": {"fuel_family": "liquid_fossil", ...},
            },
            ...
        }

    and call :meth:`_lookup` from their top-level ``map_*`` function.
    """

    TAXONOMY: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def _build_reverse_index(cls) -> Dict[str, str]:
        """Synonym → canonical, one-shot per process."""
        cache_key = f"_reverse_index_cache_{cls.__name__}"
        cache = getattr(cls, cache_key, None)
        if cache is not None:
            return cache
        index: Dict[str, str] = {}
        for canonical, payload in cls.TAXONOMY.items():
            index[normalize_text(canonical)] = canonical
            for syn in payload.get("synonyms", []):
                index[normalize_text(syn)] = canonical
        setattr(cls, cache_key, index)
        return index

    @classmethod
    def _lookup(cls, text: str) -> Optional[MappingResult]:
        """Exact-match lookup against the synonym table."""
        if not text:
            return None
        needle = normalize_text(text)
        index = cls._build_reverse_index()
        if needle in index:
            canonical = index[needle]
            return MappingResult(
                canonical=canonical,
                confidence=1.0,
                band=MappingConfidence.EXACT,
                rationale=f"Exact match on synonym '{needle}'",
                matched_pattern=needle,
                raw_input=text,
            )
        # Token-overlap fallback: return best partial match.
        tokens = set(needle.split())
        best: Optional[str] = None
        best_score = 0.0
        for syn_norm, canonical in index.items():
            syn_tokens = set(syn_norm.split())
            if not syn_tokens:
                continue
            overlap = tokens & syn_tokens
            if not overlap:
                continue
            score = len(overlap) / max(len(tokens), len(syn_tokens))
            if score > best_score:
                best_score = score
                best = canonical
        if best is not None and best_score >= 0.3:
            return MappingResult(
                canonical=best,
                confidence=best_score,
                band=MappingConfidence.from_score(best_score),
                rationale=f"Token-overlap match ({best_score:.2f})",
                matched_pattern=needle,
                raw_input=text,
            )
        return None


__all__ = [
    "BaseMapping",
    "MappingConfidence",
    "MappingError",
    "MappingResult",
    "normalize_text",
]
