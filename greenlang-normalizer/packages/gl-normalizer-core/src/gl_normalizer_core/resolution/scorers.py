"""
Confidence scoring for Entity Resolution Pipeline (GL-FOUND-X-003).

This module implements confidence scoring and score combination logic for
the entity resolution pipeline. It provides deterministic scoring algorithms
that combine multiple signals to produce a final confidence score.

Key features:
- Multi-signal score fusion (exact, fuzzy, semantic)
- Margin rule application for ambiguity detection
- Penalty application for lower-quality matches
- Score normalization and bounds enforcement

Example:
    >>> from gl_normalizer_core.resolution.scorers import ConfidenceScorer
    >>> scorer = ConfidenceScorer()
    >>> combined = scorer.combine_scores(exact=0.0, fuzzy=0.85, semantic=0.78)
    >>> needs_review = scorer.apply_margin_rule(top_score=0.85, runner_up=0.80)
    >>> assert needs_review == True  # margin 0.05 < 0.07
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from gl_normalizer_core.resolution.models import Candidate, MatchType
from gl_normalizer_core.resolution.thresholds import (
    get_config,
    DEFAULT_MARGIN_THRESHOLD,
    DEFAULT_FUZZY_MATCH_PENALTY,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Score Weighting Configuration
# =============================================================================

@dataclass
class ScoreWeights:
    """
    Configuration for score combination weights.

    Attributes:
        exact_weight: Weight for exact match scores
        fuzzy_weight: Weight for fuzzy match scores
        semantic_weight: Weight for semantic match scores
        context_boost: Boost factor when context matches
        source_boost: Boost factor for preferred sources
    """

    exact_weight: float = 1.0
    fuzzy_weight: float = 0.85
    semantic_weight: float = 0.75
    context_boost: float = 0.05
    source_boost: float = 0.03
    preferred_sources: List[str] = field(default_factory=lambda: ["ecoinvent", "gabi"])


# =============================================================================
# Confidence Scorer
# =============================================================================

class ConfidenceScorer:
    """
    Confidence scoring engine for entity resolution.

    This class provides methods to combine multiple match scores,
    apply penalties and boosts, and determine review requirements.

    All scoring operations are deterministic and produce consistent
    results for the same inputs.

    Attributes:
        weights: Score weighting configuration
        margin_threshold: Threshold for margin rule

    Example:
        >>> scorer = ConfidenceScorer()
        >>> score = scorer.combine_scores(exact=0.0, fuzzy=0.88)
        >>> print(f"Combined score: {score}")
    """

    def __init__(
        self,
        weights: Optional[ScoreWeights] = None,
        margin_threshold: Optional[float] = None,
    ) -> None:
        """
        Initialize ConfidenceScorer.

        Args:
            weights: Score weighting configuration (uses defaults if None)
            margin_threshold: Margin threshold override (uses config if None)
        """
        self.weights = weights or ScoreWeights()
        self._margin_threshold = margin_threshold

    @property
    def margin_threshold(self) -> float:
        """Get the margin threshold from config or override."""
        if self._margin_threshold is not None:
            return self._margin_threshold
        return get_config().margin_threshold

    def combine_scores(
        self,
        exact: float = 0.0,
        fuzzy: float = 0.0,
        semantic: float = 0.0,
        context_match: bool = False,
        source: Optional[str] = None,
    ) -> float:
        """
        Combine multiple match scores into a single confidence score.

        Uses a weighted combination with boosts for context and source matches.
        The exact match score takes priority when available.

        Args:
            exact: Exact match score (0.0-1.0)
            fuzzy: Fuzzy match score (0.0-1.0)
            semantic: Semantic match score (0.0-1.0)
            context_match: Whether context information matches
            source: Source vocabulary identifier

        Returns:
            float: Combined confidence score (0.0-1.0)

        Example:
            >>> scorer = ConfidenceScorer()
            >>> # Exact match dominates
            >>> scorer.combine_scores(exact=1.0, fuzzy=0.9)
            1.0
            >>> # Fuzzy match with context boost
            >>> scorer.combine_scores(fuzzy=0.85, context_match=True)
            0.9
        """
        # Exact match takes full priority
        if exact >= 1.0:
            return 1.0

        # Calculate weighted combination
        scores = []
        total_weight = 0.0

        if exact > 0.0:
            scores.append(exact * self.weights.exact_weight)
            total_weight += self.weights.exact_weight

        if fuzzy > 0.0:
            scores.append(fuzzy * self.weights.fuzzy_weight)
            total_weight += self.weights.fuzzy_weight

        if semantic > 0.0:
            scores.append(semantic * self.weights.semantic_weight)
            total_weight += self.weights.semantic_weight

        if not scores:
            return 0.0

        # Calculate base combined score
        combined = sum(scores) / total_weight if total_weight > 0 else 0.0

        # Apply boosts
        if context_match:
            combined += self.weights.context_boost

        if source and source.lower() in [s.lower() for s in self.weights.preferred_sources]:
            combined += self.weights.source_boost

        # Clamp to valid range
        return self._clamp_score(combined)

    def apply_fuzzy_penalty(self, score: float, penalty: Optional[float] = None) -> float:
        """
        Apply a penalty to fuzzy match scores.

        Fuzzy matches are inherently less certain than exact matches,
        so we apply a penalty to reflect this uncertainty.

        Args:
            score: Original fuzzy match score
            penalty: Penalty to apply (uses default if None)

        Returns:
            float: Penalized score

        Example:
            >>> scorer = ConfidenceScorer()
            >>> scorer.apply_fuzzy_penalty(0.90)
            0.855  # 0.90 * (1 - 0.05)
        """
        if penalty is None:
            penalty = DEFAULT_FUZZY_MATCH_PENALTY

        penalized = score * (1.0 - penalty)
        return self._clamp_score(penalized)

    def apply_margin_rule(
        self,
        top_score: float,
        runner_up: float,
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Apply the margin rule to determine if review is needed.

        If the difference between the top score and runner-up is less
        than the threshold, the result is ambiguous and requires review.

        Args:
            top_score: Score of the best candidate
            runner_up: Score of the second-best candidate
            threshold: Margin threshold (uses config if None)

        Returns:
            bool: True if review is needed (margin below threshold)

        Example:
            >>> scorer = ConfidenceScorer()
            >>> scorer.apply_margin_rule(0.92, 0.88)  # margin=0.04
            True  # 0.04 < 0.07
            >>> scorer.apply_margin_rule(0.95, 0.80)  # margin=0.15
            False  # 0.15 >= 0.07
        """
        if threshold is None:
            threshold = self.margin_threshold

        margin = top_score - runner_up
        needs_review = margin < threshold

        if needs_review:
            logger.debug(
                f"Margin rule triggered: margin={margin:.4f} < threshold={threshold:.4f}"
            )

        return needs_review

    def rank_candidates(
        self,
        candidates: List[Candidate],
        apply_penalties: bool = True,
    ) -> List[Candidate]:
        """
        Rank candidates by score in descending order.

        Optionally applies penalties based on match type.

        Args:
            candidates: List of candidates to rank
            apply_penalties: Whether to apply match-type penalties

        Returns:
            List[Candidate]: Sorted list (highest score first)

        Example:
            >>> candidates = [Candidate(..., score=0.8), Candidate(..., score=0.95)]
            >>> ranked = scorer.rank_candidates(candidates)
            >>> ranked[0].score
            0.95
        """
        if not candidates:
            return []

        # Sort by score descending
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    def calculate_confidence_for_match_type(
        self,
        raw_score: float,
        match_type: MatchType,
        has_context: bool = False,
        source: Optional[str] = None,
    ) -> Tuple[float, bool]:
        """
        Calculate final confidence and review requirement for a match.

        Takes into account the match type, raw score, and context to
        produce a final confidence score and determine if review is needed.

        Args:
            raw_score: Raw match score
            match_type: Type of match that produced the score
            has_context: Whether context information was available
            source: Source vocabulary identifier

        Returns:
            Tuple of (confidence_score, needs_review)

        Example:
            >>> confidence, needs_review = scorer.calculate_confidence_for_match_type(
            ...     raw_score=0.88,
            ...     match_type=MatchType.FUZZY,
            ...     has_context=True
            ... )
        """
        config = get_config()

        # Base confidence from match type
        if match_type == MatchType.EXACT:
            confidence = raw_score
            needs_review = False
        elif match_type == MatchType.FUZZY:
            confidence = self.apply_fuzzy_penalty(raw_score)
            needs_review = False
        elif match_type == MatchType.SEMANTIC:
            confidence = raw_score * self.weights.semantic_weight
            needs_review = config.semantic_always_review
        elif match_type == MatchType.LLM_SUGGESTED:
            confidence = raw_score * 0.7  # LLM suggestions get larger penalty
            needs_review = True
        else:
            confidence = raw_score
            needs_review = True

        # Apply context boost
        if has_context:
            confidence = min(1.0, confidence + self.weights.context_boost)

        # Apply source boost
        if source and source.lower() in [s.lower() for s in self.weights.preferred_sources]:
            confidence = min(1.0, confidence + self.weights.source_boost)

        return self._clamp_score(confidence), needs_review

    def evaluate_candidates(
        self,
        candidates: List[Candidate],
        entity_type: str,
    ) -> Tuple[Optional[Candidate], bool, str]:
        """
        Evaluate a list of candidates and determine the best match.

        Applies ranking, threshold checking, and margin rule to produce
        a final decision with review requirement.

        Args:
            candidates: List of candidates to evaluate
            entity_type: Type of entity being resolved

        Returns:
            Tuple of (best_candidate or None, needs_review, reason_code)

        Example:
            >>> best, needs_review, reason = scorer.evaluate_candidates(
            ...     candidates,
            ...     entity_type="fuel"
            ... )
            >>> if needs_review:
            ...     print(f"Review needed: {reason}")
        """
        if not candidates:
            return None, True, "NO_CANDIDATES"

        config = get_config()

        # Rank candidates
        ranked = self.rank_candidates(candidates)
        best = ranked[0]

        # Check threshold
        threshold = config.get_threshold(entity_type)
        if best.score < threshold:
            logger.debug(
                f"Best candidate below threshold: {best.score:.4f} < {threshold:.4f}"
            )
            return best, True, "BELOW_THRESHOLD"

        # Check margin rule if there's a runner-up
        if len(ranked) >= 2:
            runner_up = ranked[1]
            if self.apply_margin_rule(best.score, runner_up.score):
                return best, True, "MARGIN_TOO_SMALL"

        return best, False, "RESOLVED"

    def _clamp_score(self, score: float) -> float:
        """
        Clamp score to valid range [0.0, 1.0].

        Args:
            score: Score to clamp

        Returns:
            float: Clamped score
        """
        return round(max(0.0, min(1.0, score)), 6)


# =============================================================================
# Multi-Signal Score Fusion
# =============================================================================

class MultiSignalScorer:
    """
    Advanced scorer that fuses multiple matching signals.

    This scorer combines signals from different matching stages
    (exact, fuzzy, semantic) with configurable fusion strategies.

    Attributes:
        base_scorer: Base confidence scorer
        fusion_strategy: Strategy for combining signals

    Example:
        >>> scorer = MultiSignalScorer(fusion_strategy="weighted_max")
        >>> result = scorer.fuse_signals(signals)
    """

    FUSION_STRATEGIES = ["weighted_avg", "weighted_max", "cascade"]

    def __init__(
        self,
        base_scorer: Optional[ConfidenceScorer] = None,
        fusion_strategy: str = "cascade",
    ) -> None:
        """
        Initialize MultiSignalScorer.

        Args:
            base_scorer: Base scorer for individual calculations
            fusion_strategy: Strategy for fusing signals
                - "weighted_avg": Weighted average of all signals
                - "weighted_max": Use maximum signal with weighting
                - "cascade": Prefer exact > fuzzy > semantic
        """
        self.base_scorer = base_scorer or ConfidenceScorer()
        if fusion_strategy not in self.FUSION_STRATEGIES:
            raise ValueError(
                f"Invalid fusion strategy: {fusion_strategy}. "
                f"Valid options: {self.FUSION_STRATEGIES}"
            )
        self.fusion_strategy = fusion_strategy

    def fuse_signals(
        self,
        signals: Dict[MatchType, float],
        context_match: bool = False,
        source: Optional[str] = None,
    ) -> Tuple[float, MatchType]:
        """
        Fuse multiple matching signals into a single score.

        Args:
            signals: Dict mapping MatchType to score
            context_match: Whether context matches
            source: Source vocabulary identifier

        Returns:
            Tuple of (fused_score, dominant_match_type)

        Example:
            >>> signals = {MatchType.EXACT: 0.0, MatchType.FUZZY: 0.88}
            >>> score, match_type = scorer.fuse_signals(signals)
            >>> assert match_type == MatchType.FUZZY
        """
        if not signals:
            return 0.0, MatchType.FUZZY

        if self.fusion_strategy == "cascade":
            return self._cascade_fusion(signals, context_match, source)
        elif self.fusion_strategy == "weighted_max":
            return self._weighted_max_fusion(signals, context_match, source)
        else:  # weighted_avg
            return self._weighted_avg_fusion(signals, context_match, source)

    def _cascade_fusion(
        self,
        signals: Dict[MatchType, float],
        context_match: bool,
        source: Optional[str],
    ) -> Tuple[float, MatchType]:
        """
        Cascade fusion: prefer exact > fuzzy > semantic.

        Returns the first signal above minimum threshold in priority order.
        """
        priority_order = [MatchType.EXACT, MatchType.FUZZY, MatchType.SEMANTIC, MatchType.LLM_SUGGESTED]
        min_threshold = 0.5

        for match_type in priority_order:
            if match_type in signals and signals[match_type] >= min_threshold:
                score = signals[match_type]
                confidence, _ = self.base_scorer.calculate_confidence_for_match_type(
                    score, match_type, context_match, source
                )
                return confidence, match_type

        # Fallback to best available
        best_type = max(signals.keys(), key=lambda t: signals[t])
        score = signals[best_type]
        confidence, _ = self.base_scorer.calculate_confidence_for_match_type(
            score, best_type, context_match, source
        )
        return confidence, best_type

    def _weighted_max_fusion(
        self,
        signals: Dict[MatchType, float],
        context_match: bool,
        source: Optional[str],
    ) -> Tuple[float, MatchType]:
        """
        Weighted max fusion: use the highest weighted score.
        """
        weights = self.base_scorer.weights
        weight_map = {
            MatchType.EXACT: weights.exact_weight,
            MatchType.FUZZY: weights.fuzzy_weight,
            MatchType.SEMANTIC: weights.semantic_weight,
            MatchType.LLM_SUGGESTED: 0.6,
        }

        best_type = None
        best_weighted = 0.0

        for match_type, score in signals.items():
            weight = weight_map.get(match_type, 0.5)
            weighted_score = score * weight
            if weighted_score > best_weighted:
                best_weighted = weighted_score
                best_type = match_type

        if best_type is None:
            return 0.0, MatchType.FUZZY

        confidence, _ = self.base_scorer.calculate_confidence_for_match_type(
            signals[best_type], best_type, context_match, source
        )
        return confidence, best_type

    def _weighted_avg_fusion(
        self,
        signals: Dict[MatchType, float],
        context_match: bool,
        source: Optional[str],
    ) -> Tuple[float, MatchType]:
        """
        Weighted average fusion: combine all signals with weights.
        """
        exact = signals.get(MatchType.EXACT, 0.0)
        fuzzy = signals.get(MatchType.FUZZY, 0.0)
        semantic = signals.get(MatchType.SEMANTIC, 0.0)

        combined = self.base_scorer.combine_scores(
            exact=exact,
            fuzzy=fuzzy,
            semantic=semantic,
            context_match=context_match,
            source=source,
        )

        # Determine dominant match type
        if exact >= fuzzy and exact >= semantic:
            dominant = MatchType.EXACT
        elif fuzzy >= semantic:
            dominant = MatchType.FUZZY
        else:
            dominant = MatchType.SEMANTIC

        return combined, dominant


__all__ = [
    "ScoreWeights",
    "ConfidenceScorer",
    "MultiSignalScorer",
]
