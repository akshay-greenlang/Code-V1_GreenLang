# -*- coding: utf-8 -*-
"""
Match Classifier Engine - AGENT-DATA-011: Duplicate Detection (GL-DATA-X-014)

Classifies pairwise similarity results into MATCH, NON_MATCH, or POSSIBLE
using threshold-based classification, weighted scoring, Fellegi-Sunter
probabilistic linkage, and Otsu-like automatic threshold detection.

Zero-Hallucination Guarantees:
    - All classification uses deterministic threshold comparisons
    - Weighted scoring uses Python arithmetic only
    - Fellegi-Sunter uses log-likelihood ratios (deterministic math)
    - Auto-threshold uses Otsu-like histogram analysis
    - No ML/LLM calls in classification path
    - Provenance recorded for every classification operation

Classification Methods:
    THRESHOLD:      Score >= match_threshold -> MATCH,
                    Score >= possible_threshold -> POSSIBLE,
                    otherwise -> NON_MATCH
    FELLEGI_SUNTER: Probabilistic log-likelihood ratio model
    AUTO_THRESHOLD: Otsu-like bimodal histogram threshold detection

Example:
    >>> from greenlang.duplicate_detector.match_classifier import MatchClassifier
    >>> classifier = MatchClassifier()
    >>> result = classifier.classify_pair(
    ...     similarity_result=sim_result,
    ...     match_threshold=0.85,
    ...     possible_threshold=0.65,
    ... )
    >>> print(result.classification, result.confidence)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-011 Duplicate Detection Agent (GL-DATA-X-014)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.duplicate_detector.models import (
    FieldComparisonConfig,
    MatchClassification,
    MatchResult,
    SimilarityResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MatchClassifier",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_provenance(operation: str, data_repr: str) -> str:
    """Compute SHA-256 provenance hash for a classification operation."""
    payload = f"{operation}:{data_repr}:{_utcnow().isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MATCH_THRESHOLD: float = 0.85
_DEFAULT_POSSIBLE_THRESHOLD: float = 0.65
_DEFAULT_M_PROBABILITY: float = 0.9
_DEFAULT_U_PROBABILITY: float = 0.1
_OTSU_HISTOGRAM_BINS: int = 100


# =============================================================================
# MatchClassifier
# =============================================================================


class MatchClassifier:
    """Match classification engine for duplicate detection.

    Classifies pairwise similarity results into MATCH, NON_MATCH, or
    POSSIBLE categories using configurable thresholds and optional
    Fellegi-Sunter probabilistic scoring.

    This engine follows GreenLang's zero-hallucination principle:
    all classification decisions use deterministic threshold comparisons
    and arithmetic operations.

    Attributes:
        _stats_lock: Threading lock for stats updates.
        _invocations: Total invocation count.
        _successes: Total successful invocations.
        _failures: Total failed invocations.
        _total_duration_ms: Cumulative processing time.

    Example:
        >>> classifier = MatchClassifier()
        >>> result = classifier.classify_pair(sim_result, 0.85, 0.65)
        >>> assert result.classification in MatchClassification
    """

    def __init__(self) -> None:
        """Initialize MatchClassifier with empty statistics."""
        self._stats_lock = threading.Lock()
        self._invocations: int = 0
        self._successes: int = 0
        self._failures: int = 0
        self._total_duration_ms: float = 0.0
        self._last_invoked_at: Optional[datetime] = None
        logger.info("MatchClassifier initialized")

    # ------------------------------------------------------------------
    # Public API - Pair classification
    # ------------------------------------------------------------------

    def classify_pair(
        self,
        similarity_result: SimilarityResult,
        match_threshold: float = _DEFAULT_MATCH_THRESHOLD,
        possible_threshold: float = _DEFAULT_POSSIBLE_THRESHOLD,
        use_fellegi_sunter: bool = False,
        field_configs: Optional[List[FieldComparisonConfig]] = None,
    ) -> MatchResult:
        """Classify a single similarity result into MATCH/POSSIBLE/NON_MATCH.

        Args:
            similarity_result: Pairwise similarity scores to classify.
            match_threshold: Score threshold for MATCH classification.
            possible_threshold: Score threshold for POSSIBLE classification.
            use_fellegi_sunter: Whether to use Fellegi-Sunter scoring.
            field_configs: Field configs for Fellegi-Sunter m/u estimation.

        Returns:
            MatchResult with classification and confidence.

        Raises:
            ValueError: If thresholds are invalid.
        """
        start_time = time.monotonic()
        try:
            self._validate_thresholds(match_threshold, possible_threshold)

            score = similarity_result.overall_score

            # Optionally apply Fellegi-Sunter adjustment
            if use_fellegi_sunter:
                fs_score = self.fellegi_sunter_score(
                    similarity_result.field_scores, field_configs,
                )
                # Combine: weighted average of original and FS score
                score = 0.6 * score + 0.4 * fs_score
                score = max(0.0, min(1.0, round(score, 6)))

            classification = self._apply_thresholds(
                score, match_threshold, possible_threshold,
            )
            confidence = self._compute_confidence(
                score, classification, match_threshold, possible_threshold,
            )
            reason = self._build_decision_reason(
                classification, score, match_threshold,
                possible_threshold, use_fellegi_sunter,
            )

            provenance = _compute_provenance(
                "classify_pair",
                f"{similarity_result.record_a_id}:{similarity_result.record_b_id}"
                f":{classification.value}",
            )

            result = MatchResult(
                record_a_id=similarity_result.record_a_id,
                record_b_id=similarity_result.record_b_id,
                classification=classification,
                confidence=round(confidence, 6),
                field_scores=similarity_result.field_scores,
                overall_score=score,
                decision_reason=reason,
                provenance_hash=provenance,
            )

            self._record_success(time.monotonic() - start_time)
            logger.debug(
                "Classified pair (%s, %s): %s (%.3f)",
                similarity_result.record_a_id,
                similarity_result.record_b_id,
                classification.value,
                confidence,
            )
            return result

        except Exception as e:
            self._record_failure(time.monotonic() - start_time)
            logger.error("Classification failed: %s", e)
            raise

    def classify_batch(
        self,
        similarity_results: List[SimilarityResult],
        match_threshold: float = _DEFAULT_MATCH_THRESHOLD,
        possible_threshold: float = _DEFAULT_POSSIBLE_THRESHOLD,
        use_fellegi_sunter: bool = False,
        field_configs: Optional[List[FieldComparisonConfig]] = None,
    ) -> List[MatchResult]:
        """Classify a batch of similarity results.

        Args:
            similarity_results: List of similarity results to classify.
            match_threshold: Score threshold for MATCH classification.
            possible_threshold: Score threshold for POSSIBLE classification.
            use_fellegi_sunter: Whether to use Fellegi-Sunter scoring.
            field_configs: Field configs for Fellegi-Sunter m/u estimation.

        Returns:
            List of MatchResult instances.
        """
        if not similarity_results:
            return []

        logger.info(
            "Classifying batch of %d comparisons (thresholds: match=%.2f, "
            "possible=%.2f, fellegi_sunter=%s)",
            len(similarity_results), match_threshold,
            possible_threshold, use_fellegi_sunter,
        )

        results: List[MatchResult] = []
        for sim_result in similarity_results:
            result = self.classify_pair(
                sim_result, match_threshold, possible_threshold,
                use_fellegi_sunter, field_configs,
            )
            results.append(result)

        match_count = sum(
            1 for r in results if r.classification == MatchClassification.MATCH
        )
        possible_count = sum(
            1 for r in results if r.classification == MatchClassification.POSSIBLE
        )
        logger.info(
            "Batch classification complete: %d matches, %d possible, "
            "%d non-matches out of %d comparisons",
            match_count, possible_count,
            len(results) - match_count - possible_count,
            len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Public API - Weighted scoring
    # ------------------------------------------------------------------

    def compute_weighted_score(
        self,
        field_scores: Dict[str, float],
        field_configs: List[FieldComparisonConfig],
    ) -> float:
        """Compute a weighted overall similarity score from field scores.

        Args:
            field_scores: Per-field similarity scores.
            field_configs: Field comparison configurations with weights.

        Returns:
            Weighted overall score (0.0 to 1.0).
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for config in field_configs:
            score = field_scores.get(config.field_name, 0.0)
            weighted_sum += score * config.weight
            total_weight += config.weight

        if total_weight == 0.0:
            return 0.0

        return max(0.0, min(1.0, round(weighted_sum / total_weight, 6)))

    # ------------------------------------------------------------------
    # Public API - Fellegi-Sunter probabilistic scoring
    # ------------------------------------------------------------------

    def fellegi_sunter_score(
        self,
        field_scores: Dict[str, float],
        field_configs: Optional[List[FieldComparisonConfig]] = None,
        m_probability: float = _DEFAULT_M_PROBABILITY,
        u_probability: float = _DEFAULT_U_PROBABILITY,
    ) -> float:
        """Compute Fellegi-Sunter probabilistic linkage score.

        The Fellegi-Sunter model computes a log-likelihood ratio for
        each field agreement/disagreement. The composite weight is
        normalized to [0, 1].

        For each field:
            agreement weight   = log2(m / u)
            disagreement weight = log2((1 - m) / (1 - u))

        The total weight is the sum of agreement or disagreement
        weights for each field, normalized to [0, 1].

        Args:
            field_scores: Per-field similarity scores (0.0 to 1.0).
            field_configs: Optional field configs for per-field m/u.
            m_probability: Default probability of match agreement.
            u_probability: Default probability of random agreement.

        Returns:
            Normalized Fellegi-Sunter score (0.0 to 1.0).
        """
        if not field_scores:
            return 0.0

        # Build per-field m/u probabilities
        m_probs: Dict[str, float] = {}
        u_probs: Dict[str, float] = {}
        if field_configs:
            for config in field_configs:
                m_probs[config.field_name] = m_probability
                u_probs[config.field_name] = u_probability

        total_weight = 0.0
        max_possible_weight = 0.0
        min_possible_weight = 0.0

        for field_name, score in field_scores.items():
            m = m_probs.get(field_name, m_probability)
            u = u_probs.get(field_name, u_probability)

            # Clamp to avoid log(0)
            m = max(0.001, min(0.999, m))
            u = max(0.001, min(0.999, u))

            agree_w = math.log2(m / u)
            disagree_w = math.log2((1.0 - m) / (1.0 - u))

            # Use score as interpolation between disagree and agree
            field_weight = disagree_w + score * (agree_w - disagree_w)
            total_weight += field_weight

            max_possible_weight += agree_w
            min_possible_weight += disagree_w

        # Normalize to [0, 1]
        weight_range = max_possible_weight - min_possible_weight
        if weight_range == 0.0:
            return 0.5

        normalized = (total_weight - min_possible_weight) / weight_range
        return max(0.0, min(1.0, round(normalized, 6)))

    # ------------------------------------------------------------------
    # Public API - Auto threshold (Otsu-like)
    # ------------------------------------------------------------------

    def auto_threshold(
        self,
        similarity_results: List[SimilarityResult],
        num_bins: int = _OTSU_HISTOGRAM_BINS,
    ) -> Tuple[float, float]:
        """Detect optimal match and possible thresholds using Otsu-like analysis.

        Applies the Otsu method to find the threshold that maximizes
        inter-class variance in the bimodal distribution of overall
        similarity scores. The possible threshold is set at 75% of
        the match threshold.

        Args:
            similarity_results: Similarity results to analyze.
            num_bins: Number of histogram bins.

        Returns:
            Tuple of (match_threshold, possible_threshold).

        Raises:
            ValueError: If similarity_results is empty.
        """
        if not similarity_results:
            raise ValueError("similarity_results must not be empty")

        scores = [sr.overall_score for sr in similarity_results]
        if len(scores) < 2:
            return (_DEFAULT_MATCH_THRESHOLD, _DEFAULT_POSSIBLE_THRESHOLD)

        # Build histogram
        bin_width = 1.0 / num_bins
        histogram: List[int] = [0] * num_bins
        for score in scores:
            bin_idx = min(int(score / bin_width), num_bins - 1)
            histogram[bin_idx] += 1

        total_pixels = len(scores)
        total_sum = sum(i * histogram[i] for i in range(num_bins))

        best_threshold_bin = 0
        best_variance = -1.0
        sum_background = 0.0
        weight_background = 0

        for t in range(num_bins):
            weight_background += histogram[t]
            if weight_background == 0:
                continue

            weight_foreground = total_pixels - weight_background
            if weight_foreground == 0:
                break

            sum_background += t * histogram[t]
            mean_background = sum_background / weight_background
            mean_foreground = (total_sum - sum_background) / weight_foreground

            between_variance = (
                weight_background * weight_foreground
                * (mean_background - mean_foreground) ** 2
            )

            if between_variance > best_variance:
                best_variance = between_variance
                best_threshold_bin = t

        match_threshold = (best_threshold_bin + 0.5) * bin_width
        match_threshold = max(0.5, min(0.95, round(match_threshold, 3)))
        possible_threshold = round(match_threshold * 0.75, 3)
        possible_threshold = max(0.3, min(match_threshold - 0.05, possible_threshold))

        logger.info(
            "Auto-threshold detected: match=%.3f, possible=%.3f "
            "(from %d scores, %d bins)",
            match_threshold, possible_threshold, len(scores), num_bins,
        )
        return (match_threshold, possible_threshold)

    # ------------------------------------------------------------------
    # Public API - Classification quality evaluation
    # ------------------------------------------------------------------

    def evaluate_classification_quality(
        self,
        match_results: List[MatchResult],
    ) -> Dict[str, Any]:
        """Evaluate the quality of classification results.

        Computes distribution statistics, confidence metrics, and
        potential issues with the classification.

        Args:
            match_results: List of match results to evaluate.

        Returns:
            Dictionary with quality metrics.
        """
        if not match_results:
            return {
                "total_pairs": 0,
                "match_count": 0,
                "possible_count": 0,
                "non_match_count": 0,
                "match_rate": 0.0,
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "avg_match_score": 0.0,
                "avg_non_match_score": 0.0,
                "score_separation": 0.0,
            }

        total = len(match_results)
        match_count = sum(
            1 for r in match_results
            if r.classification == MatchClassification.MATCH
        )
        possible_count = sum(
            1 for r in match_results
            if r.classification == MatchClassification.POSSIBLE
        )
        non_match_count = total - match_count - possible_count

        confidences = [r.confidence for r in match_results]
        avg_confidence = sum(confidences) / total if total > 0 else 0.0

        match_scores = [
            r.overall_score for r in match_results
            if r.classification == MatchClassification.MATCH
        ]
        non_match_scores = [
            r.overall_score for r in match_results
            if r.classification == MatchClassification.NON_MATCH
        ]

        avg_match_score = (
            sum(match_scores) / len(match_scores) if match_scores else 0.0
        )
        avg_non_match_score = (
            sum(non_match_scores) / len(non_match_scores)
            if non_match_scores else 0.0
        )

        score_separation = avg_match_score - avg_non_match_score

        return {
            "total_pairs": total,
            "match_count": match_count,
            "possible_count": possible_count,
            "non_match_count": non_match_count,
            "match_rate": round(match_count / total, 4) if total > 0 else 0.0,
            "avg_confidence": round(avg_confidence, 4),
            "min_confidence": round(min(confidences), 4) if confidences else 0.0,
            "max_confidence": round(max(confidences), 4) if confidences else 0.0,
            "avg_match_score": round(avg_match_score, 4),
            "avg_non_match_score": round(avg_non_match_score, 4),
            "score_separation": round(score_separation, 4),
        }

    # ------------------------------------------------------------------
    # Public API - Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return current engine operational statistics."""
        with self._stats_lock:
            avg_ms = 0.0
            if self._invocations > 0:
                avg_ms = self._total_duration_ms / self._invocations
            return {
                "engine_name": "MatchClassifier",
                "invocations": self._invocations,
                "successes": self._successes,
                "failures": self._failures,
                "total_duration_ms": round(self._total_duration_ms, 3),
                "avg_duration_ms": round(avg_ms, 3),
                "last_invoked_at": (
                    self._last_invoked_at.isoformat()
                    if self._last_invoked_at else None
                ),
            }

    def reset_statistics(self) -> None:
        """Reset all operational statistics to zero."""
        with self._stats_lock:
            self._invocations = 0
            self._successes = 0
            self._failures = 0
            self._total_duration_ms = 0.0
            self._last_invoked_at = None

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _validate_thresholds(
        self,
        match_threshold: float,
        possible_threshold: float,
    ) -> None:
        """Validate classification thresholds.

        Args:
            match_threshold: Upper threshold for MATCH.
            possible_threshold: Lower threshold for POSSIBLE.

        Raises:
            ValueError: If thresholds are invalid.
        """
        if not 0.0 <= match_threshold <= 1.0:
            raise ValueError(
                f"match_threshold must be between 0.0 and 1.0, got {match_threshold}"
            )
        if not 0.0 <= possible_threshold <= 1.0:
            raise ValueError(
                f"possible_threshold must be between 0.0 and 1.0, "
                f"got {possible_threshold}"
            )
        if possible_threshold > match_threshold:
            raise ValueError(
                f"possible_threshold ({possible_threshold}) must not exceed "
                f"match_threshold ({match_threshold})"
            )

    def _apply_thresholds(
        self,
        score: float,
        match_threshold: float,
        possible_threshold: float,
    ) -> MatchClassification:
        """Apply threshold-based classification to a score.

        Args:
            score: Overall similarity score.
            match_threshold: Threshold for MATCH.
            possible_threshold: Threshold for POSSIBLE.

        Returns:
            MatchClassification enum value.
        """
        if score >= match_threshold:
            return MatchClassification.MATCH
        elif score >= possible_threshold:
            return MatchClassification.POSSIBLE
        else:
            return MatchClassification.NON_MATCH

    def _compute_confidence(
        self,
        score: float,
        classification: MatchClassification,
        match_threshold: float,
        possible_threshold: float,
    ) -> float:
        """Compute confidence in the classification decision.

        Confidence measures how clearly the score falls into its
        classification region. A score exactly at the threshold
        has lower confidence than one deeply within its region.

        Args:
            score: Overall similarity score.
            classification: The classification decision.
            match_threshold: Upper threshold.
            possible_threshold: Lower threshold.

        Returns:
            Confidence score (0.0 to 1.0).
        """
        if classification == MatchClassification.MATCH:
            # How far above match threshold (scaled to 0..1)
            margin = score - match_threshold
            max_margin = 1.0 - match_threshold
            if max_margin <= 0:
                return 1.0
            base_confidence = 0.7 + 0.3 * min(1.0, margin / max_margin)
            return min(1.0, base_confidence)

        elif classification == MatchClassification.NON_MATCH:
            # How far below possible threshold (scaled to 0..1)
            margin = possible_threshold - score
            if possible_threshold <= 0:
                return 1.0
            base_confidence = 0.7 + 0.3 * min(1.0, margin / possible_threshold)
            return min(1.0, base_confidence)

        else:
            # POSSIBLE: confidence is lower for ambiguous region
            region_size = match_threshold - possible_threshold
            if region_size <= 0:
                return 0.5
            mid = (match_threshold + possible_threshold) / 2.0
            distance_from_mid = abs(score - mid)
            # Closer to center = lower confidence
            return max(0.3, min(0.7, 0.5 + 0.2 * (distance_from_mid / (region_size / 2.0))))

    def _build_decision_reason(
        self,
        classification: MatchClassification,
        score: float,
        match_threshold: float,
        possible_threshold: float,
        use_fellegi_sunter: bool,
    ) -> str:
        """Build human-readable explanation of the classification.

        Args:
            classification: The classification decision.
            score: Overall similarity score.
            match_threshold: Upper threshold.
            possible_threshold: Lower threshold.
            use_fellegi_sunter: Whether FS scoring was used.

        Returns:
            Decision reason string.
        """
        method = "Fellegi-Sunter adjusted" if use_fellegi_sunter else "threshold"

        if classification == MatchClassification.MATCH:
            return (
                f"MATCH: score {score:.4f} >= match_threshold {match_threshold:.4f} "
                f"(method: {method})"
            )
        elif classification == MatchClassification.POSSIBLE:
            return (
                f"POSSIBLE: score {score:.4f} >= possible_threshold "
                f"{possible_threshold:.4f} but < match_threshold "
                f"{match_threshold:.4f} (method: {method})"
            )
        else:
            return (
                f"NON_MATCH: score {score:.4f} < possible_threshold "
                f"{possible_threshold:.4f} (method: {method})"
            )

    def _record_success(self, elapsed_seconds: float) -> None:
        """Record a successful invocation."""
        ms = elapsed_seconds * 1000.0
        with self._stats_lock:
            self._invocations += 1
            self._successes += 1
            self._total_duration_ms += ms
            self._last_invoked_at = _utcnow()

    def _record_failure(self, elapsed_seconds: float) -> None:
        """Record a failed invocation."""
        ms = elapsed_seconds * 1000.0
        with self._stats_lock:
            self._invocations += 1
            self._failures += 1
            self._total_duration_ms += ms
            self._last_invoked_at = _utcnow()
