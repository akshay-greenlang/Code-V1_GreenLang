"""
Matching strategies for Entity Resolution Pipeline (GL-FOUND-X-003).

This module implements the core matching algorithms used in the entity
resolution pipeline. Matchers are organized by determinism:

- ExactMatcher: Deterministic case-insensitive exact matching
- FuzzyMatcher: Deterministic Levenshtein distance-based matching
- SemanticMatcher: Non-deterministic embedding-based matching (stubbed)

All deterministic matchers produce consistent, reproducible results.
The semantic matcher is optional and always flags results for review.

Example:
    >>> from gl_normalizer_core.resolution.matchers import (
    ...     ExactMatcher, FuzzyMatcher, SemanticMatcher
    ... )
    >>> exact = ExactMatcher(reference_data)
    >>> result = exact.match("Diesel Fuel")
    >>> if not result:
    ...     fuzzy = FuzzyMatcher(reference_data)
    ...     result = fuzzy.match("Deisel Fuel")  # typo
"""

import logging
import re
import unicodedata
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any, Tuple

from gl_normalizer_core.resolution.models import (
    Candidate,
    MatchType,
    ResolutionContext,
)
from gl_normalizer_core.resolution.thresholds import get_config

logger = logging.getLogger(__name__)


# =============================================================================
# Reference Data Types
# =============================================================================

class ReferenceEntity:
    """
    A reference entity from the vocabulary.

    Attributes:
        id: Canonical identifier
        name: Primary name
        aliases: Alternative names
        source: Source vocabulary
        metadata: Additional metadata
    """

    def __init__(
        self,
        id: str,
        name: str,
        aliases: Optional[List[str]] = None,
        source: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize ReferenceEntity."""
        self.id = id
        self.name = name
        self.aliases = aliases or []
        self.source = source
        self.metadata = metadata or {}

    @property
    def all_names(self) -> List[str]:
        """Get all names including primary and aliases."""
        return [self.name] + self.aliases


# =============================================================================
# Base Matcher Interface
# =============================================================================

class BaseMatcher(ABC):
    """
    Abstract base class for all matchers.

    Defines the interface that all matchers must implement.
    Subclasses provide specific matching algorithms.

    Attributes:
        reference_data: List of reference entities to match against
        match_type: The type of match this matcher produces
    """

    match_type: MatchType = MatchType.FUZZY

    def __init__(
        self,
        reference_data: List[ReferenceEntity],
        **kwargs: Any,
    ) -> None:
        """
        Initialize BaseMatcher.

        Args:
            reference_data: List of reference entities to match against
            **kwargs: Additional configuration options
        """
        self.reference_data = reference_data
        self._config = kwargs

    @abstractmethod
    def match(
        self,
        input_text: str,
        context: Optional[ResolutionContext] = None,
        max_candidates: int = 5,
    ) -> List[Candidate]:
        """
        Match input text against reference data.

        Args:
            input_text: Text to match
            context: Optional resolution context
            max_candidates: Maximum number of candidates to return

        Returns:
            List of Candidate objects sorted by score (descending)
        """
        pass

    @property
    def is_deterministic(self) -> bool:
        """
        Check if this matcher produces deterministic results.

        Returns:
            bool: True if results are deterministic
        """
        return self.match_type.is_deterministic

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for matching.

        Performs case normalization, Unicode normalization, and
        whitespace standardization.

        Args:
            text: Text to normalize

        Returns:
            str: Normalized text
        """
        # Unicode normalization (NFKC for compatibility)
        text = unicodedata.normalize("NFKC", text)

        # Case normalization
        text = text.lower()

        # Whitespace normalization
        text = re.sub(r"\s+", " ", text).strip()

        return text


# =============================================================================
# Exact Matcher
# =============================================================================

class ExactMatcher(BaseMatcher):
    """
    Case-insensitive exact matcher.

    This matcher performs exact matching after normalizing both input
    and reference text. It is fully deterministic and produces the
    highest confidence results.

    Matching process:
    1. Normalize input text (lowercase, Unicode normalization, whitespace)
    2. Compare against all names and aliases in reference data
    3. Return exact matches with score 1.0

    Attributes:
        reference_data: List of reference entities
        _name_index: Pre-built index mapping normalized names to entities

    Example:
        >>> matcher = ExactMatcher(reference_data)
        >>> candidates = matcher.match("Diesel Fuel")
        >>> if candidates and candidates[0].score == 1.0:
        ...     print("Exact match found!")
    """

    match_type = MatchType.EXACT

    def __init__(
        self,
        reference_data: List[ReferenceEntity],
        **kwargs: Any,
    ) -> None:
        """
        Initialize ExactMatcher.

        Builds an index for efficient exact matching.

        Args:
            reference_data: List of reference entities
            **kwargs: Additional configuration
        """
        super().__init__(reference_data, **kwargs)
        self._name_index = self._build_index()

    def _build_index(self) -> Dict[str, List[ReferenceEntity]]:
        """
        Build a lookup index mapping normalized names to entities.

        Returns:
            Dict mapping normalized names to list of matching entities
        """
        index: Dict[str, List[ReferenceEntity]] = {}

        for entity in self.reference_data:
            for name in entity.all_names:
                normalized = self._normalize_text(name)
                if normalized not in index:
                    index[normalized] = []
                index[normalized].append(entity)

        logger.debug(f"ExactMatcher index built with {len(index)} entries")
        return index

    def match(
        self,
        input_text: str,
        context: Optional[ResolutionContext] = None,
        max_candidates: int = 5,
    ) -> List[Candidate]:
        """
        Perform exact matching.

        Args:
            input_text: Text to match
            context: Optional resolution context (used for filtering)
            max_candidates: Maximum number of candidates to return

        Returns:
            List of Candidate objects (all with score 1.0)

        Example:
            >>> candidates = matcher.match("diesel fuel")
            >>> assert all(c.score == 1.0 for c in candidates)
        """
        normalized_input = self._normalize_text(input_text)

        if normalized_input not in self._name_index:
            logger.debug(f"No exact match found for: {input_text}")
            return []

        entities = self._name_index[normalized_input]
        candidates = []

        for entity in entities:
            # Apply context filtering if provided
            if context and not self._matches_context(entity, context):
                continue

            candidate = Candidate(
                id=entity.id,
                name=entity.name,
                score=1.0,  # Exact match always gets 1.0
                source=entity.source,
                metadata={
                    "match_type": self.match_type.value,
                    "matched_name": normalized_input,
                },
            )
            candidates.append(candidate)

            if len(candidates) >= max_candidates:
                break

        logger.debug(f"ExactMatcher found {len(candidates)} candidates for: {input_text}")
        return candidates

    def _matches_context(
        self,
        entity: ReferenceEntity,
        context: ResolutionContext,
    ) -> bool:
        """
        Check if an entity matches the resolution context.

        Args:
            entity: Entity to check
            context: Resolution context

        Returns:
            bool: True if entity matches context
        """
        # Filter by vocabulary if specified
        if context.vocabulary_ids and entity.source not in context.vocabulary_ids:
            return False

        # Filter by region if specified and entity has region metadata
        if context.region and "region" in entity.metadata:
            if entity.metadata["region"] != context.region:
                return False

        return True


# =============================================================================
# Fuzzy Matcher
# =============================================================================

class FuzzyMatcher(BaseMatcher):
    """
    Levenshtein distance-based fuzzy matcher.

    This matcher uses edit distance to find approximate matches,
    normalized by string length to produce a similarity score.
    It is fully deterministic.

    Matching process:
    1. Normalize input text
    2. Calculate Levenshtein distance to all reference names
    3. Convert distance to similarity score
    4. Return candidates above minimum threshold

    Attributes:
        reference_data: List of reference entities
        min_score: Minimum score to consider a match

    Example:
        >>> matcher = FuzzyMatcher(reference_data, min_score=0.7)
        >>> # Handles typos
        >>> candidates = matcher.match("Deisel Fuel")  # typo in "Diesel"
        >>> assert candidates[0].name == "Diesel Fuel"
    """

    match_type = MatchType.FUZZY

    def __init__(
        self,
        reference_data: List[ReferenceEntity],
        min_score: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize FuzzyMatcher.

        Args:
            reference_data: List of reference entities
            min_score: Minimum similarity score (default from config)
            **kwargs: Additional configuration
        """
        super().__init__(reference_data, **kwargs)
        self.min_score = min_score or get_config().min_fuzzy_score

    def match(
        self,
        input_text: str,
        context: Optional[ResolutionContext] = None,
        max_candidates: int = 5,
    ) -> List[Candidate]:
        """
        Perform fuzzy matching using Levenshtein distance.

        Args:
            input_text: Text to match
            context: Optional resolution context
            max_candidates: Maximum number of candidates to return

        Returns:
            List of Candidate objects sorted by score (descending)

        Example:
            >>> candidates = matcher.match("Deisel Fuel")
            >>> print(candidates[0].score)  # e.g., 0.91
        """
        normalized_input = self._normalize_text(input_text)
        candidates: List[Candidate] = []

        for entity in self.reference_data:
            # Apply context filtering
            if context and not self._matches_context(entity, context):
                continue

            # Find best match among all names
            best_score = 0.0
            best_name = entity.name

            for name in entity.all_names:
                normalized_name = self._normalize_text(name)
                score = self._calculate_similarity(normalized_input, normalized_name)
                if score > best_score:
                    best_score = score
                    best_name = name

            # Only include if above minimum threshold
            if best_score >= self.min_score:
                candidate = Candidate(
                    id=entity.id,
                    name=entity.name,
                    score=best_score,
                    source=entity.source,
                    metadata={
                        "match_type": self.match_type.value,
                        "matched_name": best_name,
                        "levenshtein_score": best_score,
                    },
                )
                candidates.append(candidate)

        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)

        # Limit results
        candidates = candidates[:max_candidates]

        logger.debug(
            f"FuzzyMatcher found {len(candidates)} candidates for: {input_text}"
        )
        return candidates

    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate normalized Levenshtein similarity between two strings.

        Uses dynamic programming to compute edit distance, then
        normalizes by the maximum string length.

        Args:
            s1: First string
            s2: Second string

        Returns:
            float: Similarity score (0.0 to 1.0)

        Example:
            >>> matcher._calculate_similarity("diesel", "deisel")
            0.833...
        """
        if s1 == s2:
            return 1.0

        len_s1, len_s2 = len(s1), len(s2)

        if len_s1 == 0:
            return 0.0 if len_s2 > 0 else 1.0
        if len_s2 == 0:
            return 0.0

        # Create distance matrix
        matrix = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]

        # Initialize first row and column
        for i in range(len_s1 + 1):
            matrix[i][0] = i
        for j in range(len_s2 + 1):
            matrix[0][j] = j

        # Fill in the rest of the matrix
        for i in range(1, len_s1 + 1):
            for j in range(1, len_s2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                matrix[i][j] = min(
                    matrix[i - 1][j] + 1,      # deletion
                    matrix[i][j - 1] + 1,      # insertion
                    matrix[i - 1][j - 1] + cost,  # substitution
                )

        distance = matrix[len_s1][len_s2]
        max_len = max(len_s1, len_s2)

        # Normalize to similarity score
        similarity = 1.0 - (distance / max_len)
        return round(similarity, 6)

    def _matches_context(
        self,
        entity: ReferenceEntity,
        context: ResolutionContext,
    ) -> bool:
        """Check if entity matches context (same as ExactMatcher)."""
        if context.vocabulary_ids and entity.source not in context.vocabulary_ids:
            return False
        if context.region and "region" in entity.metadata:
            if entity.metadata["region"] != context.region:
                return False
        return True


# =============================================================================
# Semantic Matcher (Stubbed for LLM Integration)
# =============================================================================

class SemanticMatcher(BaseMatcher):
    """
    Embedding-based semantic similarity matcher.

    This matcher is STUBBED for future LLM/embedding integration.
    It always sets needs_review=True because semantic matching is
    non-deterministic and may produce hallucinated results.

    When implemented, this will:
    1. Generate embeddings for input text
    2. Compare against pre-computed reference embeddings
    3. Use cosine similarity for scoring
    4. Always flag results for human review

    Attributes:
        reference_data: List of reference entities
        embedding_model: Name of embedding model (future)
        _embeddings_cache: Cache of reference embeddings (future)

    Example:
        >>> matcher = SemanticMatcher(reference_data)
        >>> candidates = matcher.match("petroleum-based transportation fuel")
        >>> # All semantic matches require review
        >>> for c in candidates:
        ...     assert c.metadata.get("requires_review") == True
    """

    match_type = MatchType.SEMANTIC

    def __init__(
        self,
        reference_data: List[ReferenceEntity],
        embedding_model: str = "default",
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize SemanticMatcher.

        Args:
            reference_data: List of reference entities
            embedding_model: Name of embedding model to use
            embedding_function: Optional custom embedding function
            **kwargs: Additional configuration
        """
        super().__init__(reference_data, **kwargs)
        self.embedding_model = embedding_model
        self._embedding_function = embedding_function
        self._embeddings_cache: Dict[str, List[float]] = {}
        self._is_initialized = False

        logger.info(
            "SemanticMatcher initialized (STUBBED). "
            "All results will require human review."
        )

    def initialize_embeddings(self) -> None:
        """
        Initialize embeddings for reference data.

        STUBBED: In production, this would pre-compute embeddings
        for all reference entities.
        """
        logger.warning(
            "SemanticMatcher.initialize_embeddings() is STUBBED. "
            "No embeddings will be computed."
        )
        self._is_initialized = True

    def match(
        self,
        input_text: str,
        context: Optional[ResolutionContext] = None,
        max_candidates: int = 5,
    ) -> List[Candidate]:
        """
        Perform semantic matching (STUBBED).

        This is a stub implementation that returns empty results.
        When LLM integration is available, this will perform
        embedding-based similarity matching.

        IMPORTANT: All semantic matches ALWAYS require human review.

        Args:
            input_text: Text to match
            context: Optional resolution context
            max_candidates: Maximum number of candidates to return

        Returns:
            List of Candidate objects (currently empty - stubbed)

        Example:
            >>> candidates = matcher.match("renewable energy source")
            >>> # Currently returns empty list (stubbed)
            >>> assert len(candidates) == 0
        """
        logger.debug(
            f"SemanticMatcher.match() called for: {input_text} "
            "(STUBBED - returning empty results)"
        )

        # STUB: Return empty list until LLM integration is implemented
        # When implemented, this would:
        # 1. Generate embedding for input_text
        # 2. Compare against reference embeddings
        # 3. Return top candidates with similarity scores
        # 4. ALL candidates would have metadata["requires_review"] = True

        return []

    def match_with_llm_suggestion(
        self,
        input_text: str,
        context: Optional[ResolutionContext] = None,
        max_candidates: int = 5,
    ) -> List[Candidate]:
        """
        Get LLM-suggested candidates (STUBBED).

        This method is for future LLM integration where the model
        can suggest potential matches based on semantic understanding.

        IMPORTANT: All LLM suggestions ALWAYS require human review.

        Args:
            input_text: Text to match
            context: Optional resolution context
            max_candidates: Maximum number of candidates

        Returns:
            List of LLM-suggested Candidate objects (currently empty)
        """
        logger.debug(
            f"SemanticMatcher.match_with_llm_suggestion() called for: {input_text} "
            "(STUBBED - returning empty results)"
        )

        # STUB: Return empty list until LLM integration
        # When implemented, all returned candidates would have:
        # - match_type = MatchType.LLM_SUGGESTED
        # - metadata["requires_review"] = True
        # - metadata["llm_reasoning"] = <explanation from LLM>

        return []

    def _compute_embedding(self, text: str) -> List[float]:
        """
        Compute embedding for text (STUBBED).

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        if self._embedding_function:
            return self._embedding_function(text)

        # STUB: Return empty vector
        logger.warning(
            "_compute_embedding() is STUBBED. Returning empty vector."
        )
        return []

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float],
    ) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            float: Cosine similarity (0.0 to 1.0)
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


# =============================================================================
# Matcher Factory
# =============================================================================

class MatcherFactory:
    """
    Factory for creating matcher instances.

    Provides a centralized way to create and configure matchers
    based on match type.

    Example:
        >>> factory = MatcherFactory()
        >>> exact = factory.create("exact", reference_data)
        >>> fuzzy = factory.create("fuzzy", reference_data, min_score=0.75)
    """

    @staticmethod
    def create(
        match_type: str,
        reference_data: List[ReferenceEntity],
        **kwargs: Any,
    ) -> BaseMatcher:
        """
        Create a matcher instance.

        Args:
            match_type: Type of matcher ("exact", "fuzzy", "semantic")
            reference_data: Reference data for matching
            **kwargs: Matcher-specific configuration

        Returns:
            BaseMatcher: Configured matcher instance

        Raises:
            ValueError: If match_type is not recognized
        """
        match_type_lower = match_type.lower()

        if match_type_lower == "exact":
            return ExactMatcher(reference_data, **kwargs)
        elif match_type_lower == "fuzzy":
            return FuzzyMatcher(reference_data, **kwargs)
        elif match_type_lower == "semantic":
            return SemanticMatcher(reference_data, **kwargs)
        else:
            raise ValueError(
                f"Unknown match type: {match_type}. "
                f"Valid options: exact, fuzzy, semantic"
            )


__all__ = [
    "ReferenceEntity",
    "BaseMatcher",
    "ExactMatcher",
    "FuzzyMatcher",
    "SemanticMatcher",
    "MatcherFactory",
]
