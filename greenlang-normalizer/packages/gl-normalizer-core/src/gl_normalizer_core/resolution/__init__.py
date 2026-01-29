"""
Entity Resolution Pipeline for GL-FOUND-X-003.

This module provides the complete entity resolution pipeline for the GreenLang
Normalizer, implementing multi-stage matching with confidence scoring and
review flagging.

The resolution pipeline follows these principles:

1. **Multi-Stage Matching**
   - Stage 1: Exact Match (case-insensitive, deterministic)
   - Stage 2: Fuzzy Match (Levenshtein distance, deterministic)
   - Stage 3: Semantic Match (embedding-based, optional, always requires review)

2. **Entity-Type Specific Thresholds**
   - Fuel entities: >= 0.95 confidence required
   - Material entities: >= 0.90 confidence required
   - Process entities: >= 0.85 confidence required

3. **Margin Rule**
   - If (top1_score - top2_score) < 0.07, set needs_review=True
   - This flags ambiguous matches for human review

4. **Zero-Hallucination Principle**
   - Core matchers (exact, fuzzy) are deterministic
   - Semantic/LLM matchers always set needs_review=True
   - All calculations use explicit formulas, not LLM inference

Example:
    >>> from gl_normalizer_core.resolution import (
    ...     EntityResolutionPipeline,
    ...     ResolutionResult,
    ...     ReferenceEntity,
    ...     MatchType,
    ... )
    >>>
    >>> # Create reference data
    >>> reference_data = [
    ...     ReferenceEntity(id="FUEL-001", name="Diesel Fuel", source="vocab"),
    ...     ReferenceEntity(id="FUEL-002", name="Gasoline", source="vocab"),
    ... ]
    >>>
    >>> # Create pipeline and resolve
    >>> pipeline = EntityResolutionPipeline(reference_data)
    >>> result = pipeline.resolve("Diesel Fuel", "fuel")
    >>>
    >>> if result.is_resolved and not result.needs_review:
    ...     print(f"Resolved: {result.canonical_id} ({result.confidence:.2%})")
    ... elif result.needs_review:
    ...     print(f"Review required: {result.candidates}")
    ... else:
    ...     print(f"Not found: {result.error_message}")
"""

# Models
from gl_normalizer_core.resolution.models import (
    MatchType,
    Candidate,
    ResolutionResult,
    EntityType,
    ResolutionContext,
)

# Thresholds
from gl_normalizer_core.resolution.thresholds import (
    ENTITY_THRESHOLDS,
    MARGIN_THRESHOLD,
    ThresholdConfig,
    get_config,
    set_config,
    reset_config,
    get_threshold,
    is_above_threshold,
    needs_margin_review,
)

# Matchers
from gl_normalizer_core.resolution.matchers import (
    ReferenceEntity,
    BaseMatcher,
    ExactMatcher,
    FuzzyMatcher,
    SemanticMatcher,
    MatcherFactory,
)

# Scorers
from gl_normalizer_core.resolution.scorers import (
    ScoreWeights,
    ConfidenceScorer,
    MultiSignalScorer,
)

# Pipeline
from gl_normalizer_core.resolution.pipeline import (
    PipelineConfig,
    EntityResolutionPipeline,
    create_pipeline,
    quick_resolve,
)


__all__ = [
    # Models
    "MatchType",
    "Candidate",
    "ResolutionResult",
    "EntityType",
    "ResolutionContext",
    # Thresholds
    "ENTITY_THRESHOLDS",
    "MARGIN_THRESHOLD",
    "ThresholdConfig",
    "get_config",
    "set_config",
    "reset_config",
    "get_threshold",
    "is_above_threshold",
    "needs_margin_review",
    # Matchers
    "ReferenceEntity",
    "BaseMatcher",
    "ExactMatcher",
    "FuzzyMatcher",
    "SemanticMatcher",
    "MatcherFactory",
    # Scorers
    "ScoreWeights",
    "ConfidenceScorer",
    "MultiSignalScorer",
    # Pipeline
    "PipelineConfig",
    "EntityResolutionPipeline",
    "create_pipeline",
    "quick_resolve",
]
