"""
Reference Resolver module for the GreenLang Normalizer.

This module provides reference data resolution capabilities,
matching input strings to canonical vocabulary entries using
fuzzy matching and confidence scoring.

Example:
    >>> from gl_normalizer_core.resolver import ReferenceResolver
    >>> resolver = ReferenceResolver()
    >>> result = resolver.resolve("natural gas", vocabulary="fuels")
    >>> print(result.resolved_id)
    FUEL_NAT_GAS_001
"""

from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import hashlib
from datetime import datetime

from pydantic import BaseModel, Field
import structlog

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

from gl_normalizer_core.errors import ResolutionError

logger = structlog.get_logger(__name__)


class MatchConfidence(str, Enum):
    """Confidence levels for reference matches."""

    EXACT = "exact"  # 100% match
    HIGH = "high"  # >= 90% match
    MEDIUM = "medium"  # >= 70% match
    LOW = "low"  # >= 50% match
    NONE = "none"  # < 50% or no match


class MatchMethod(str, Enum):
    """Methods used for matching."""

    EXACT = "exact"
    FUZZY = "fuzzy"
    ALIAS = "alias"
    EMBEDDING = "embedding"


class ResolvedReference(BaseModel):
    """
    A resolved reference with confidence scoring.

    Attributes:
        resolved_id: Canonical ID of the resolved reference
        resolved_name: Canonical name of the resolved reference
        original_query: Original query string
        vocabulary: Vocabulary used for resolution
        confidence: Confidence level of the match
        confidence_score: Numeric confidence score (0-100)
        match_method: Method used to find the match
        alternatives: Alternative matches if any
        metadata: Additional metadata from the vocabulary
        provenance_hash: SHA-256 hash for audit trail
    """

    resolved_id: str = Field(..., description="Canonical reference ID")
    resolved_name: str = Field(..., description="Canonical reference name")
    original_query: str = Field(..., description="Original query string")
    vocabulary: str = Field(..., description="Vocabulary used")
    confidence: MatchConfidence = Field(..., description="Confidence level")
    confidence_score: float = Field(..., ge=0, le=100, description="Confidence score 0-100")
    match_method: MatchMethod = Field(..., description="Method used for matching")
    alternatives: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Alternative matches"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")
    resolution_time_ms: float = Field(..., description="Resolution time in milliseconds")

    @classmethod
    def create(
        cls,
        resolved_id: str,
        resolved_name: str,
        original_query: str,
        vocabulary: str,
        confidence_score: float,
        match_method: MatchMethod,
        resolution_time_ms: float,
        alternatives: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ResolvedReference":
        """Create a ResolvedReference with calculated confidence level."""
        # Determine confidence level
        if confidence_score >= 100:
            confidence = MatchConfidence.EXACT
        elif confidence_score >= 90:
            confidence = MatchConfidence.HIGH
        elif confidence_score >= 70:
            confidence = MatchConfidence.MEDIUM
        elif confidence_score >= 50:
            confidence = MatchConfidence.LOW
        else:
            confidence = MatchConfidence.NONE

        # Calculate provenance hash
        provenance_str = f"{original_query}|{resolved_id}|{vocabulary}|{confidence_score}"
        provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

        return cls(
            resolved_id=resolved_id,
            resolved_name=resolved_name,
            original_query=original_query,
            vocabulary=vocabulary,
            confidence=confidence,
            confidence_score=confidence_score,
            match_method=match_method,
            alternatives=alternatives or [],
            metadata=metadata or {},
            provenance_hash=provenance_hash,
            resolution_time_ms=resolution_time_ms,
        )


class ResolutionResult(BaseModel):
    """
    Result of a resolution operation.

    Attributes:
        success: Whether resolution succeeded
        resolved: The resolved reference (if successful)
        candidates: Candidate matches found
        warnings: Any warnings generated
    """

    success: bool = Field(..., description="Whether resolution succeeded")
    resolved: Optional[ResolvedReference] = Field(None, description="Resolved reference")
    candidates: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Candidate matches"
    )
    warnings: List[str] = Field(default_factory=list, description="Resolution warnings")


class VocabEntry(BaseModel):
    """
    Entry in a vocabulary.

    Attributes:
        id: Unique identifier
        name: Canonical name
        aliases: Alternative names/spellings
        metadata: Additional metadata
    """

    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Canonical name")
    aliases: List[str] = Field(default_factory=list, description="Alternative names")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ReferenceResolver:
    """
    Resolver for reference data.

    This class matches input strings to canonical vocabulary entries
    using exact matching, alias lookup, and fuzzy matching.

    Attributes:
        vocabularies: Loaded vocabularies
        min_confidence: Minimum confidence threshold for matches
        fuzzy_enabled: Whether fuzzy matching is enabled

    Example:
        >>> resolver = ReferenceResolver()
        >>> resolver.load_vocabulary("fuels", entries)
        >>> result = resolver.resolve("natural gas", "fuels")
        >>> print(result.resolved.resolved_name)
        Natural Gas
    """

    # Default vocabularies with common entries
    DEFAULT_FUELS: List[VocabEntry] = [
        VocabEntry(
            id="FUEL_NAT_GAS_001",
            name="Natural Gas",
            aliases=["nat gas", "natural gas", "methane", "ng"],
            metadata={"category": "fossil", "ghg_factor": 2.02}
        ),
        VocabEntry(
            id="FUEL_DIESEL_001",
            name="Diesel",
            aliases=["diesel fuel", "diesel oil", "derv", "agr diesel"],
            metadata={"category": "fossil", "ghg_factor": 2.68}
        ),
        VocabEntry(
            id="FUEL_PETROL_001",
            name="Petrol",
            aliases=["gasoline", "gas", "motor spirit", "premium unleaded"],
            metadata={"category": "fossil", "ghg_factor": 2.31}
        ),
        VocabEntry(
            id="FUEL_ELEC_001",
            name="Electricity",
            aliases=["electric", "power", "grid electricity"],
            metadata={"category": "secondary", "ghg_factor": 0.233}
        ),
    ]

    DEFAULT_MATERIALS: List[VocabEntry] = [
        VocabEntry(
            id="MAT_STEEL_001",
            name="Steel",
            aliases=["carbon steel", "mild steel", "structural steel"],
            metadata={"category": "metal", "density": 7850}
        ),
        VocabEntry(
            id="MAT_ALUM_001",
            name="Aluminum",
            aliases=["aluminium", "alu", "al"],
            metadata={"category": "metal", "density": 2700}
        ),
        VocabEntry(
            id="MAT_CONC_001",
            name="Concrete",
            aliases=["cement", "portland cement", "ready-mix"],
            metadata={"category": "construction", "density": 2400}
        ),
    ]

    def __init__(
        self,
        min_confidence: float = 70.0,
        fuzzy_enabled: bool = True,
        load_defaults: bool = True,
    ) -> None:
        """
        Initialize ReferenceResolver.

        Args:
            min_confidence: Minimum confidence threshold (0-100)
            fuzzy_enabled: Enable fuzzy matching
            load_defaults: Load default vocabularies
        """
        self.min_confidence = min_confidence
        self.fuzzy_enabled = fuzzy_enabled and RAPIDFUZZ_AVAILABLE
        self.vocabularies: Dict[str, List[VocabEntry]] = {}

        if load_defaults:
            self.vocabularies["fuels"] = self.DEFAULT_FUELS.copy()
            self.vocabularies["materials"] = self.DEFAULT_MATERIALS.copy()

        logger.info(
            "ReferenceResolver initialized",
            min_confidence=min_confidence,
            fuzzy_enabled=self.fuzzy_enabled,
            vocabularies=list(self.vocabularies.keys()),
        )

    def load_vocabulary(
        self,
        vocabulary_name: str,
        entries: List[VocabEntry],
    ) -> None:
        """
        Load a vocabulary.

        Args:
            vocabulary_name: Name of the vocabulary
            entries: List of vocabulary entries
        """
        self.vocabularies[vocabulary_name] = entries
        logger.info(
            "Loaded vocabulary",
            name=vocabulary_name,
            entry_count=len(entries),
        )

    def resolve(
        self,
        query: str,
        vocabulary: str,
        confidence_threshold: Optional[float] = None,
    ) -> ResolutionResult:
        """
        Resolve a query string to a vocabulary entry.

        Args:
            query: Query string to resolve
            vocabulary: Vocabulary to search
            confidence_threshold: Override minimum confidence threshold

        Returns:
            ResolutionResult with resolved reference or candidates

        Raises:
            ResolutionError: If vocabulary not found or resolution fails
        """
        start_time = datetime.now()
        threshold = confidence_threshold or self.min_confidence
        warnings: List[str] = []

        try:
            # Validate vocabulary exists
            if vocabulary not in self.vocabularies:
                raise ResolutionError(
                    f"Vocabulary '{vocabulary}' not found",
                    query=query,
                    vocabulary=vocabulary,
                    hint=f"Available vocabularies: {list(self.vocabularies.keys())}",
                )

            entries = self.vocabularies[vocabulary]
            query_normalized = query.strip().lower()

            # Try exact match first
            for entry in entries:
                if entry.name.lower() == query_normalized:
                    resolution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                    resolved = ResolvedReference.create(
                        resolved_id=entry.id,
                        resolved_name=entry.name,
                        original_query=query,
                        vocabulary=vocabulary,
                        confidence_score=100.0,
                        match_method=MatchMethod.EXACT,
                        resolution_time_ms=resolution_time_ms,
                        metadata=entry.metadata,
                    )
                    return ResolutionResult(success=True, resolved=resolved)

            # Try alias match
            for entry in entries:
                for alias in entry.aliases:
                    if alias.lower() == query_normalized:
                        resolution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                        resolved = ResolvedReference.create(
                            resolved_id=entry.id,
                            resolved_name=entry.name,
                            original_query=query,
                            vocabulary=vocabulary,
                            confidence_score=100.0,
                            match_method=MatchMethod.ALIAS,
                            resolution_time_ms=resolution_time_ms,
                            metadata=entry.metadata,
                        )
                        return ResolutionResult(success=True, resolved=resolved)

            # Try fuzzy matching
            if self.fuzzy_enabled:
                candidates = self._fuzzy_match(query, entries, vocabulary)

                if candidates:
                    # Return best match if above threshold
                    best = candidates[0]
                    if best["score"] >= threshold:
                        resolution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                        resolved = ResolvedReference.create(
                            resolved_id=best["id"],
                            resolved_name=best["name"],
                            original_query=query,
                            vocabulary=vocabulary,
                            confidence_score=best["score"],
                            match_method=MatchMethod.FUZZY,
                            resolution_time_ms=resolution_time_ms,
                            alternatives=candidates[1:5],  # Top 4 alternatives
                            metadata=best.get("metadata", {}),
                        )
                        return ResolutionResult(
                            success=True,
                            resolved=resolved,
                            candidates=candidates,
                        )
                    else:
                        # Return candidates for review
                        return ResolutionResult(
                            success=False,
                            candidates=candidates,
                            warnings=[
                                f"Best match '{best['name']}' has confidence "
                                f"{best['score']:.1f}% below threshold {threshold}%"
                            ],
                        )

            # No match found
            resolution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            return ResolutionResult(
                success=False,
                warnings=[f"No match found for '{query}' in vocabulary '{vocabulary}'"],
            )

        except ResolutionError:
            raise
        except Exception as e:
            logger.error("Resolution failed", error=str(e), exc_info=True)
            raise ResolutionError(
                f"Resolution failed: {str(e)}",
                query=query,
                vocabulary=vocabulary,
            ) from e

    def _fuzzy_match(
        self,
        query: str,
        entries: List[VocabEntry],
        vocabulary: str,
    ) -> List[Dict[str, Any]]:
        """
        Perform fuzzy matching against vocabulary entries.

        Args:
            query: Query string
            entries: Vocabulary entries to match against
            vocabulary: Vocabulary name

        Returns:
            List of candidate matches with scores
        """
        if not RAPIDFUZZ_AVAILABLE:
            return []

        candidates = []
        query_lower = query.lower()

        for entry in entries:
            # Match against name
            name_score = fuzz.ratio(query_lower, entry.name.lower())

            # Match against aliases
            alias_scores = [fuzz.ratio(query_lower, alias.lower()) for alias in entry.aliases]
            best_alias_score = max(alias_scores) if alias_scores else 0

            # Use best score
            best_score = max(name_score, best_alias_score)

            if best_score > 0:
                candidates.append({
                    "id": entry.id,
                    "name": entry.name,
                    "score": best_score,
                    "vocabulary": vocabulary,
                    "metadata": entry.metadata,
                })

        # Sort by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates

    def get_vocabulary_entries(self, vocabulary: str) -> List[VocabEntry]:
        """
        Get all entries in a vocabulary.

        Args:
            vocabulary: Vocabulary name

        Returns:
            List of vocabulary entries

        Raises:
            ResolutionError: If vocabulary not found
        """
        if vocabulary not in self.vocabularies:
            raise ResolutionError(
                f"Vocabulary '{vocabulary}' not found",
                vocabulary=vocabulary,
            )
        return self.vocabularies[vocabulary]

    def list_vocabularies(self) -> List[str]:
        """List all available vocabularies."""
        return list(self.vocabularies.keys())


__all__ = [
    "ReferenceResolver",
    "ResolvedReference",
    "ResolutionResult",
    "MatchConfidence",
    "MatchMethod",
    "VocabEntry",
]
