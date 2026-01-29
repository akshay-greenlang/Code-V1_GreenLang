"""
Data models for Entity Resolution Pipeline (GL-FOUND-X-003).

This module defines the core data models used throughout the entity resolution
pipeline, including resolution results, candidates, and match type enumeration.

All models use Pydantic for validation and serialization, ensuring type safety
and clear documentation of all fields.

Example:
    >>> from gl_normalizer_core.resolution.models import (
    ...     ResolutionResult, Candidate, MatchType
    ... )
    >>> candidate = Candidate(
    ...     id="FUEL-001",
    ...     name="Diesel Fuel",
    ...     score=0.95,
    ...     source="ecoinvent"
    ... )
    >>> result = ResolutionResult(
    ...     canonical_id="FUEL-001",
    ...     confidence=0.95,
    ...     needs_review=False,
    ...     candidates=[candidate],
    ...     match_type=MatchType.EXACT
    ... )
"""

from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, field_validator


class MatchType(str, Enum):
    """
    Enumeration of match types in the entity resolution pipeline.

    Match types indicate how a candidate was matched, which affects
    the confidence level and review requirements.

    Attributes:
        EXACT: Case-insensitive exact match (highest confidence)
        FUZZY: Levenshtein distance-based fuzzy match (deterministic)
        SEMANTIC: Embedding-based semantic similarity match
        LLM_SUGGESTED: LLM-suggested candidate (always requires review)

    Example:
        >>> match_type = MatchType.EXACT
        >>> assert match_type.value == "exact"
        >>> assert match_type.is_deterministic == True
    """

    EXACT = "exact"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"
    LLM_SUGGESTED = "llm_suggested"

    @property
    def is_deterministic(self) -> bool:
        """
        Check if this match type produces deterministic results.

        Returns:
            bool: True if the matching algorithm is deterministic

        Example:
            >>> MatchType.EXACT.is_deterministic
            True
            >>> MatchType.LLM_SUGGESTED.is_deterministic
            False
        """
        return self in (MatchType.EXACT, MatchType.FUZZY)

    @property
    def requires_review_by_default(self) -> bool:
        """
        Check if this match type requires human review by default.

        Returns:
            bool: True if human review is required by default

        Example:
            >>> MatchType.LLM_SUGGESTED.requires_review_by_default
            True
            >>> MatchType.EXACT.requires_review_by_default
            False
        """
        return self in (MatchType.SEMANTIC, MatchType.LLM_SUGGESTED)


class Candidate(BaseModel):
    """
    A candidate entity match from the resolution pipeline.

    Represents a potential match found during entity resolution,
    including its identifier, name, confidence score, and source.

    Attributes:
        id: Canonical identifier of the candidate entity
        name: Human-readable name of the candidate
        score: Confidence score for this candidate (0.0 to 1.0)
        source: Source vocabulary or database where candidate was found
        metadata: Additional metadata about the candidate (optional)

    Example:
        >>> candidate = Candidate(
        ...     id="MAT-STEEL-001",
        ...     name="Carbon Steel",
        ...     score=0.92,
        ...     source="materials_vocab"
        ... )
        >>> assert candidate.score >= 0.0 and candidate.score <= 1.0
    """

    id: str = Field(
        ...,
        description="Canonical identifier of the candidate entity",
        min_length=1,
        max_length=256,
    )
    name: str = Field(
        ...,
        description="Human-readable name of the candidate",
        min_length=1,
        max_length=512,
    )
    score: float = Field(
        ...,
        description="Confidence score for this candidate (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    source: str = Field(
        ...,
        description="Source vocabulary or database where candidate was found",
        min_length=1,
        max_length=128,
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the candidate",
    )

    @field_validator("score")
    @classmethod
    def round_score(cls, v: float) -> float:
        """Round score to 6 decimal places for consistency."""
        return round(v, 6)

    def __lt__(self, other: "Candidate") -> bool:
        """Enable sorting candidates by score (descending)."""
        return self.score > other.score  # Higher score = higher priority


class ResolutionResult(BaseModel):
    """
    Result of an entity resolution operation.

    Contains the resolved canonical entity (if found), confidence score,
    review status, list of candidates considered, and the match type used.

    Attributes:
        canonical_id: Resolved canonical entity ID (None if no match)
        confidence: Overall confidence score (0.0 to 1.0)
        needs_review: Whether human review is required
        candidates: List of candidate matches considered
        match_type: Type of match that produced the result
        error_code: Error code if resolution failed (optional)
        error_message: Human-readable error message (optional)
        provenance_hash: SHA-256 hash for audit trail (optional)

    Example:
        >>> result = ResolutionResult(
        ...     canonical_id="PROC-WELD-001",
        ...     confidence=0.88,
        ...     needs_review=False,
        ...     candidates=[...],
        ...     match_type=MatchType.FUZZY
        ... )
        >>> if result.needs_review:
        ...     queue_for_review(result)
    """

    canonical_id: Optional[str] = Field(
        None,
        description="Resolved canonical entity ID (None if no match)",
        max_length=256,
    )
    confidence: float = Field(
        ...,
        description="Overall confidence score (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    needs_review: bool = Field(
        ...,
        description="Whether human review is required",
    )
    candidates: List[Candidate] = Field(
        default_factory=list,
        description="List of candidate matches considered",
    )
    match_type: Optional[MatchType] = Field(
        None,
        description="Type of match that produced the result",
    )
    error_code: Optional[str] = Field(
        None,
        description="Error code if resolution failed (GLNORM-E4xx)",
        pattern=r"^GLNORM-E4\d{2}$",
    )
    error_message: Optional[str] = Field(
        None,
        description="Human-readable error message",
        max_length=1024,
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
        pattern=r"^[a-f0-9]{64}$",
    )

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        """Round confidence to 6 decimal places for consistency."""
        return round(v, 6)

    @property
    def is_resolved(self) -> bool:
        """
        Check if resolution was successful.

        Returns:
            bool: True if a canonical ID was found with sufficient confidence

        Example:
            >>> result = ResolutionResult(canonical_id="X", confidence=0.9, ...)
            >>> assert result.is_resolved == True
        """
        return self.canonical_id is not None and self.error_code is None

    @property
    def top_candidate(self) -> Optional[Candidate]:
        """
        Get the highest-scoring candidate.

        Returns:
            Candidate or None: The candidate with the highest score

        Example:
            >>> result.top_candidate.score
            0.95
        """
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda c: c.score)

    @property
    def runner_up_candidate(self) -> Optional[Candidate]:
        """
        Get the second-highest scoring candidate.

        Returns:
            Candidate or None: The candidate with the second highest score

        Example:
            >>> result.runner_up_candidate.score
            0.89
        """
        if len(self.candidates) < 2:
            return None
        sorted_candidates = sorted(self.candidates, key=lambda c: c.score, reverse=True)
        return sorted_candidates[1]

    @property
    def margin(self) -> Optional[float]:
        """
        Calculate the margin between top two candidates.

        Returns:
            float or None: Score difference between top two candidates

        Example:
            >>> result.margin
            0.06
        """
        top = self.top_candidate
        runner_up = self.runner_up_candidate
        if top is None or runner_up is None:
            return None
        return round(top.score - runner_up.score, 6)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "canonical_id": "FUEL-DIESEL-001",
                    "confidence": 0.96,
                    "needs_review": False,
                    "candidates": [
                        {
                            "id": "FUEL-DIESEL-001",
                            "name": "Diesel Fuel",
                            "score": 0.96,
                            "source": "ecoinvent",
                        }
                    ],
                    "match_type": "exact",
                }
            ]
        }
    }


class EntityType(str, Enum):
    """
    Enumeration of entity types supported by the resolution pipeline.

    Each entity type has its own confidence threshold requirement.

    Attributes:
        FUEL: Fuel entities (threshold: 0.95)
        MATERIAL: Material entities (threshold: 0.90)
        PROCESS: Process entities (threshold: 0.85)
        ACTIVITY: Activity entities (threshold: 0.85)
        EMISSION_FACTOR: Emission factor entities (threshold: 0.95)
        LOCATION: Location/geographic entities (threshold: 0.90)

    Example:
        >>> entity_type = EntityType.FUEL
        >>> assert entity_type.value == "fuel"
    """

    FUEL = "fuel"
    MATERIAL = "material"
    PROCESS = "process"
    ACTIVITY = "activity"
    EMISSION_FACTOR = "emission_factor"
    LOCATION = "location"


class ResolutionContext(BaseModel):
    """
    Context information for entity resolution.

    Provides additional context that can improve resolution accuracy,
    such as industry sector, geographic region, or temporal scope.

    Attributes:
        industry_sector: Industry sector code (e.g., "energy", "manufacturing")
        region: Geographic region code (e.g., "US", "EU", "GLOBAL")
        temporal_scope: Time period (e.g., "2024", "2020-2025")
        vocabulary_ids: Specific vocabularies to search (optional)
        additional_context: Free-form additional context (optional)

    Example:
        >>> context = ResolutionContext(
        ...     industry_sector="manufacturing",
        ...     region="EU",
        ...     temporal_scope="2024"
        ... )
    """

    industry_sector: Optional[str] = Field(
        None,
        description="Industry sector code",
        max_length=64,
    )
    region: Optional[str] = Field(
        None,
        description="Geographic region code",
        max_length=64,
    )
    temporal_scope: Optional[str] = Field(
        None,
        description="Time period for resolution",
        max_length=32,
    )
    vocabulary_ids: List[str] = Field(
        default_factory=list,
        description="Specific vocabularies to search",
    )
    additional_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form additional context",
    )


__all__ = [
    "MatchType",
    "Candidate",
    "ResolutionResult",
    "EntityType",
    "ResolutionContext",
]
