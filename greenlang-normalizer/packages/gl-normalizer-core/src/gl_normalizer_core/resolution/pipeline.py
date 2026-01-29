"""
Entity Resolution Pipeline for GL-FOUND-X-003.

This module implements the main EntityResolutionPipeline class that orchestrates
multi-stage entity resolution with confidence scoring and review flagging.

Pipeline stages:
1. Exact Match - Case-insensitive exact matching (deterministic)
2. Fuzzy Match - Levenshtein distance matching (deterministic)
3. Semantic Match - Embedding-based matching (optional, always requires review)

The pipeline follows GreenLang's zero-hallucination principle:
- Core matchers (exact, fuzzy) are deterministic
- Semantic/LLM matchers always set needs_review=True
- Confidence thresholds are entity-type specific
- Margin rule flags ambiguous results for review

Example:
    >>> from gl_normalizer_core.resolution.pipeline import EntityResolutionPipeline
    >>> from gl_normalizer_core.resolution.models import ResolutionContext
    >>>
    >>> pipeline = EntityResolutionPipeline(reference_data)
    >>> result = pipeline.resolve(
    ...     input_text="Diesel Fuel",
    ...     entity_type="fuel",
    ...     context=ResolutionContext(region="US")
    ... )
    >>> if result.is_resolved and not result.needs_review:
    ...     print(f"Resolved to: {result.canonical_id}")
"""

import hashlib
import logging
import time
from typing import Dict, List, Optional, Any, Tuple

from gl_normalizer_core.errors.codes import GLNORMErrorCode
from gl_normalizer_core.resolution.models import (
    Candidate,
    MatchType,
    ResolutionResult,
    ResolutionContext,
    EntityType,
)
from gl_normalizer_core.resolution.matchers import (
    ReferenceEntity,
    ExactMatcher,
    FuzzyMatcher,
    SemanticMatcher,
    MatcherFactory,
)
from gl_normalizer_core.resolution.scorers import (
    ConfidenceScorer,
    MultiSignalScorer,
)
from gl_normalizer_core.resolution.thresholds import (
    get_config,
    ThresholdConfig,
    is_above_threshold,
    needs_margin_review,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Configuration
# =============================================================================

class PipelineConfig:
    """
    Configuration for the entity resolution pipeline.

    Attributes:
        enable_fuzzy: Enable fuzzy matching stage
        enable_semantic: Enable semantic matching stage (optional)
        max_candidates: Maximum candidates to return per stage
        require_review_on_semantic: Always require review for semantic matches
        require_review_on_llm: Always require review for LLM suggestions
        log_performance: Log performance metrics
    """

    def __init__(
        self,
        enable_fuzzy: bool = True,
        enable_semantic: bool = False,
        max_candidates: int = 5,
        require_review_on_semantic: bool = True,
        require_review_on_llm: bool = True,
        log_performance: bool = True,
    ) -> None:
        """Initialize PipelineConfig."""
        self.enable_fuzzy = enable_fuzzy
        self.enable_semantic = enable_semantic
        self.max_candidates = max_candidates
        self.require_review_on_semantic = require_review_on_semantic
        self.require_review_on_llm = require_review_on_llm
        self.log_performance = log_performance


# =============================================================================
# Entity Resolution Pipeline
# =============================================================================

class EntityResolutionPipeline:
    """
    Multi-stage entity resolution pipeline.

    This pipeline orchestrates entity resolution through multiple matching
    stages, applying confidence scoring and review flagging at each step.

    Pipeline flow:
        Input -> Exact Match -> (if no match) Fuzzy Match -> (if enabled) Semantic Match
                     |                  |                          |
                     v                  v                          v
                [Confidence Scoring] -> [Threshold Check] -> [Margin Rule]
                                                |
                                                v
                                        ResolutionResult

    Design principles (Zero-Hallucination):
    - Exact and fuzzy matchers are DETERMINISTIC
    - Semantic matchers ALWAYS set needs_review=True
    - Confidence thresholds are entity-type specific (Fuel>=0.95, Material>=0.90, Process>=0.85)
    - Margin rule: if top1 - top2 < 0.07, set needs_review=True

    Attributes:
        reference_data: List of reference entities to match against
        config: Pipeline configuration
        threshold_config: Confidence threshold configuration
        exact_matcher: Exact matching engine
        fuzzy_matcher: Fuzzy matching engine
        semantic_matcher: Semantic matching engine (optional)
        scorer: Confidence scoring engine

    Example:
        >>> pipeline = EntityResolutionPipeline(reference_data)
        >>> result = pipeline.resolve("Diesel Fuel", "fuel")
        >>> print(f"Confidence: {result.confidence}")
        >>> print(f"Needs review: {result.needs_review}")
    """

    def __init__(
        self,
        reference_data: List[ReferenceEntity],
        config: Optional[PipelineConfig] = None,
        threshold_config: Optional[ThresholdConfig] = None,
    ) -> None:
        """
        Initialize EntityResolutionPipeline.

        Args:
            reference_data: List of reference entities for matching
            config: Pipeline configuration (uses defaults if None)
            threshold_config: Threshold configuration (uses defaults if None)
        """
        self.reference_data = reference_data
        self.config = config or PipelineConfig()
        self.threshold_config = threshold_config or get_config()

        # Initialize matchers
        self.exact_matcher = ExactMatcher(reference_data)
        self.fuzzy_matcher = FuzzyMatcher(
            reference_data,
            min_score=self.threshold_config.min_fuzzy_score,
        )
        self.semantic_matcher: Optional[SemanticMatcher] = None
        if self.config.enable_semantic:
            self.semantic_matcher = SemanticMatcher(reference_data)

        # Initialize scorer
        self.scorer = ConfidenceScorer()
        self.multi_signal_scorer = MultiSignalScorer(
            base_scorer=self.scorer,
            fusion_strategy="cascade",
        )

        logger.info(
            f"EntityResolutionPipeline initialized with {len(reference_data)} "
            f"reference entities. Fuzzy={self.config.enable_fuzzy}, "
            f"Semantic={self.config.enable_semantic}"
        )

    def resolve(
        self,
        input_text: str,
        entity_type: str,
        context: Optional[ResolutionContext] = None,
    ) -> ResolutionResult:
        """
        Resolve input text to a canonical entity.

        Executes the multi-stage pipeline: exact_match -> fuzzy_match -> semantic_match.
        Applies confidence scoring and review flagging at each stage.

        Args:
            input_text: Text to resolve (e.g., "Diesel Fuel")
            entity_type: Type of entity (e.g., "fuel", "material", "process")
            context: Optional resolution context for filtering

        Returns:
            ResolutionResult containing:
            - canonical_id: Resolved entity ID (if found)
            - confidence: Confidence score (0.0-1.0)
            - needs_review: Whether human review is required
            - candidates: List of all candidates considered
            - match_type: Type of match that succeeded

        Example:
            >>> result = pipeline.resolve("Diesel Fuel", "fuel")
            >>> if result.is_resolved:
            ...     print(f"Match: {result.canonical_id} ({result.confidence:.2%})")
            ... else:
            ...     print(f"No match found. Error: {result.error_code}")
        """
        start_time = time.perf_counter()
        context = context or ResolutionContext()

        logger.debug(
            f"Starting resolution for '{input_text}' (entity_type={entity_type})"
        )

        # Track all candidates from all stages
        all_candidates: List[Candidate] = []
        signals: Dict[MatchType, float] = {}

        # Stage 1: Exact Match
        exact_candidates = self._run_exact_match(input_text, context)
        all_candidates.extend(exact_candidates)

        if exact_candidates:
            best_exact = exact_candidates[0]
            signals[MatchType.EXACT] = best_exact.score
            logger.debug(f"Exact match found: {best_exact.id} (score={best_exact.score})")

            # Exact match with score 1.0 - return immediately if above threshold
            if self._is_above_threshold(best_exact.score, entity_type):
                result = self._create_result(
                    candidates=exact_candidates,
                    match_type=MatchType.EXACT,
                    entity_type=entity_type,
                    input_text=input_text,
                    start_time=start_time,
                )
                return result

        # Stage 2: Fuzzy Match
        if self.config.enable_fuzzy:
            fuzzy_candidates = self._run_fuzzy_match(input_text, context)

            # Deduplicate candidates
            for fc in fuzzy_candidates:
                if not any(c.id == fc.id for c in all_candidates):
                    all_candidates.append(fc)

            if fuzzy_candidates:
                best_fuzzy = fuzzy_candidates[0]
                signals[MatchType.FUZZY] = best_fuzzy.score
                logger.debug(
                    f"Fuzzy match found: {best_fuzzy.id} (score={best_fuzzy.score})"
                )

        # Stage 3: Semantic Match (optional)
        if self.config.enable_semantic and self.semantic_matcher:
            semantic_candidates = self._run_semantic_match(input_text, context)

            for sc in semantic_candidates:
                if not any(c.id == sc.id for c in all_candidates):
                    all_candidates.append(sc)

            if semantic_candidates:
                best_semantic = semantic_candidates[0]
                signals[MatchType.SEMANTIC] = best_semantic.score
                logger.debug(
                    f"Semantic match found: {best_semantic.id} "
                    f"(score={best_semantic.score})"
                )

        # No candidates found
        if not all_candidates:
            return self._create_no_match_result(
                input_text=input_text,
                entity_type=entity_type,
                start_time=start_time,
            )

        # Determine best match using multi-signal fusion
        fused_score, dominant_type = self.multi_signal_scorer.fuse_signals(
            signals=signals,
            context_match=bool(context.industry_sector or context.region),
            source=all_candidates[0].source if all_candidates else None,
        )

        # Create final result
        result = self._create_result(
            candidates=all_candidates,
            match_type=dominant_type,
            entity_type=entity_type,
            input_text=input_text,
            start_time=start_time,
        )

        if self.config.log_performance:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"Resolution complete for '{input_text}': "
                f"canonical_id={result.canonical_id}, "
                f"confidence={result.confidence:.4f}, "
                f"needs_review={result.needs_review}, "
                f"elapsed_ms={elapsed_ms:.2f}"
            )

        return result

    def _run_exact_match(
        self,
        input_text: str,
        context: ResolutionContext,
    ) -> List[Candidate]:
        """
        Run the exact matching stage.

        Args:
            input_text: Text to match
            context: Resolution context

        Returns:
            List of exact match candidates
        """
        return self.exact_matcher.match(
            input_text=input_text,
            context=context,
            max_candidates=self.config.max_candidates,
        )

    def _run_fuzzy_match(
        self,
        input_text: str,
        context: ResolutionContext,
    ) -> List[Candidate]:
        """
        Run the fuzzy matching stage.

        Args:
            input_text: Text to match
            context: Resolution context

        Returns:
            List of fuzzy match candidates
        """
        return self.fuzzy_matcher.match(
            input_text=input_text,
            context=context,
            max_candidates=self.config.max_candidates,
        )

    def _run_semantic_match(
        self,
        input_text: str,
        context: ResolutionContext,
    ) -> List[Candidate]:
        """
        Run the semantic matching stage.

        Note: Semantic matches ALWAYS require human review.

        Args:
            input_text: Text to match
            context: Resolution context

        Returns:
            List of semantic match candidates
        """
        if not self.semantic_matcher:
            return []

        candidates = self.semantic_matcher.match(
            input_text=input_text,
            context=context,
            max_candidates=self.config.max_candidates,
        )

        # Mark all semantic candidates as requiring review
        for c in candidates:
            c.metadata["requires_review"] = True
            c.metadata["review_reason"] = "SEMANTIC_MATCH"

        return candidates

    def _is_above_threshold(self, score: float, entity_type: str) -> bool:
        """Check if score meets the threshold for entity type."""
        return self.threshold_config.is_above_threshold(score, entity_type)

    def _needs_margin_review(
        self,
        candidates: List[Candidate],
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if margin rule triggers review.

        Args:
            candidates: List of candidates (sorted by score)

        Returns:
            Tuple of (needs_review, reason)
        """
        if len(candidates) < 2:
            return False, None

        top = candidates[0]
        runner_up = candidates[1]

        if self.threshold_config.needs_margin_review(top.score, runner_up.score):
            margin = top.score - runner_up.score
            return True, f"MARGIN_TOO_SMALL ({margin:.4f} < {self.threshold_config.margin_threshold})"

        return False, None

    def _create_result(
        self,
        candidates: List[Candidate],
        match_type: MatchType,
        entity_type: str,
        input_text: str,
        start_time: float,
    ) -> ResolutionResult:
        """
        Create a ResolutionResult from candidates.

        Args:
            candidates: List of all candidates
            match_type: Dominant match type
            entity_type: Entity type being resolved
            input_text: Original input text
            start_time: Processing start time

        Returns:
            ResolutionResult with all fields populated
        """
        # Sort candidates by score
        sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)

        # Get best candidate
        best = sorted_candidates[0] if sorted_candidates else None

        if not best:
            return self._create_no_match_result(input_text, entity_type, start_time)

        # Calculate confidence
        confidence = best.score

        # Determine if review is needed
        needs_review = False
        review_reasons: List[str] = []

        # 1. Check threshold
        if not self._is_above_threshold(confidence, entity_type):
            needs_review = True
            threshold = self.threshold_config.get_threshold(entity_type)
            review_reasons.append(f"BELOW_THRESHOLD ({confidence:.4f} < {threshold})")

        # 2. Check margin rule
        margin_needs_review, margin_reason = self._needs_margin_review(sorted_candidates)
        if margin_needs_review:
            needs_review = True
            if margin_reason:
                review_reasons.append(margin_reason)

        # 3. Check if semantic/LLM match (always requires review)
        if match_type in (MatchType.SEMANTIC, MatchType.LLM_SUGGESTED):
            needs_review = True
            review_reasons.append(f"{match_type.value.upper()}_MATCH")

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance(input_text, best.id, confidence)

        # Build result
        result = ResolutionResult(
            canonical_id=best.id,
            confidence=confidence,
            needs_review=needs_review,
            candidates=sorted_candidates[:self.config.max_candidates],
            match_type=match_type,
            provenance_hash=provenance_hash,
        )

        # Add review reasons to metadata if any
        if review_reasons:
            for c in result.candidates:
                c.metadata["review_reasons"] = review_reasons

        return result

    def _create_no_match_result(
        self,
        input_text: str,
        entity_type: str,
        start_time: float,
    ) -> ResolutionResult:
        """
        Create a ResolutionResult when no match is found.

        Args:
            input_text: Original input text
            entity_type: Entity type being resolved
            start_time: Processing start time

        Returns:
            ResolutionResult with error information
        """
        return ResolutionResult(
            canonical_id=None,
            confidence=0.0,
            needs_review=True,
            candidates=[],
            match_type=None,
            error_code=GLNORMErrorCode.E400_REFERENCE_NOT_FOUND.value,
            error_message=f"No match found for '{input_text}' (entity_type={entity_type})",
        )

    def _calculate_provenance(
        self,
        input_text: str,
        canonical_id: str,
        confidence: float,
    ) -> str:
        """
        Calculate SHA-256 provenance hash for audit trail.

        Args:
            input_text: Original input text
            canonical_id: Resolved canonical ID
            confidence: Final confidence score

        Returns:
            SHA-256 hash string
        """
        provenance_str = f"{input_text}|{canonical_id}|{confidence:.6f}"
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def add_reference_entity(self, entity: ReferenceEntity) -> None:
        """
        Add a reference entity to the pipeline.

        Rebuilds matcher indices to include the new entity.

        Args:
            entity: Reference entity to add
        """
        self.reference_data.append(entity)

        # Rebuild matchers
        self.exact_matcher = ExactMatcher(self.reference_data)
        self.fuzzy_matcher = FuzzyMatcher(
            self.reference_data,
            min_score=self.threshold_config.min_fuzzy_score,
        )
        if self.semantic_matcher:
            self.semantic_matcher = SemanticMatcher(self.reference_data)

        logger.info(f"Added reference entity: {entity.id}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.

        Returns:
            Dict containing pipeline statistics
        """
        return {
            "reference_entity_count": len(self.reference_data),
            "fuzzy_enabled": self.config.enable_fuzzy,
            "semantic_enabled": self.config.enable_semantic,
            "threshold_config": self.threshold_config.to_dict(),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_pipeline(
    reference_data: List[ReferenceEntity],
    enable_semantic: bool = False,
) -> EntityResolutionPipeline:
    """
    Create an EntityResolutionPipeline with sensible defaults.

    Args:
        reference_data: List of reference entities
        enable_semantic: Whether to enable semantic matching

    Returns:
        Configured EntityResolutionPipeline

    Example:
        >>> pipeline = create_pipeline(reference_data)
        >>> result = pipeline.resolve("Diesel", "fuel")
    """
    config = PipelineConfig(
        enable_fuzzy=True,
        enable_semantic=enable_semantic,
    )
    return EntityResolutionPipeline(reference_data, config=config)


def quick_resolve(
    input_text: str,
    entity_type: str,
    reference_data: List[ReferenceEntity],
) -> ResolutionResult:
    """
    Quick resolution without creating a persistent pipeline.

    Useful for one-off resolutions where pipeline reuse is not needed.

    Args:
        input_text: Text to resolve
        entity_type: Entity type
        reference_data: Reference data

    Returns:
        ResolutionResult

    Example:
        >>> result = quick_resolve("Diesel", "fuel", reference_data)
    """
    pipeline = create_pipeline(reference_data, enable_semantic=False)
    return pipeline.resolve(input_text, entity_type)


__all__ = [
    "PipelineConfig",
    "EntityResolutionPipeline",
    "create_pipeline",
    "quick_resolve",
]
