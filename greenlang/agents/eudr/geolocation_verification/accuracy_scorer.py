# -*- coding: utf-8 -*-
"""
Accuracy Scoring Engine - AGENT-EUDR-002: Geolocation Verification (Feature 3)

Computes composite geolocation accuracy scores from individual verification
results (coordinate precision, polygon quality, country matching, protected
area status, deforestation verification, temporal consistency). All scoring
uses Decimal arithmetic for bit-perfect deterministic reproducibility.

Zero-Hallucination Guarantees:
    - 100% deterministic: same inputs produce identical scores and hashes
    - All arithmetic uses Decimal for bit-perfect reproducibility
    - No LLM/ML involvement in any scoring path
    - SHA-256 provenance hash on every score result
    - Complete audit trail for regulatory inspection

Score Dimensions (total: 0-100):
    - Coordinate Precision:    0-20 (20% weight)
    - Polygon Quality:         0-20 (20% weight)
    - Country Match:           0-15 (15% weight)
    - Protected Area Check:    0-15 (15% weight)
    - Deforestation Check:     0-15 (15% weight)
    - Temporal Consistency:    0-15 (15% weight)

Quality Tiers:
    - GOLD:   >= 85 (fully compliant, high confidence)
    - SILVER: >= 70 (compliant with minor issues)
    - BRONZE: >= 50 (marginal, remediation recommended)
    - FAIL:   <  50 (non-compliant, remediation required)

Performance Targets:
    - Single score calculation: <1ms
    - Batch scoring (10,000 results): <1 second

Regulatory References:
    - EUDR Article 9: Geolocation accuracy requirements
    - EUDR Article 10: Risk assessment data quality
    - EUDR Article 31: Audit trail and record retention

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-002 (Feature 3: Accuracy Scoring)
Agent ID: GL-EUDR-GEO-002
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    CoordinateValidationResult,
    DeforestationVerificationResult,
    GeolocationAccuracyScore,
    PolygonVerificationResult,
    ProtectedAreaCheckResult,
    QualityTier,
    TemporalChangeResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _to_decimal(value: float | int | str | Decimal) -> Decimal:
    """Convert a numeric value to Decimal via string for determinism."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _clamp_score(value: Decimal, max_score: Decimal) -> Decimal:
    """Clamp a score to [0, max_score] with 2-decimal precision."""
    clamped = max(Decimal("0.00"), min(max_score, value))
    return clamped.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum scores per dimension.
MAX_COORD_PRECISION_SCORE = Decimal("20.00")
MAX_POLYGON_QUALITY_SCORE = Decimal("20.00")
MAX_COUNTRY_MATCH_SCORE = Decimal("15.00")
MAX_PROTECTED_AREA_SCORE = Decimal("15.00")
MAX_DEFORESTATION_SCORE = Decimal("15.00")
MAX_TEMPORAL_SCORE = Decimal("15.00")
MAX_TOTAL_SCORE = Decimal("100.00")

#: Score precision (2 decimal places).
SCORE_PRECISION = Decimal("0.01")

#: Quality tier thresholds.
TIER_GOLD_THRESHOLD = Decimal("85.00")
TIER_SILVER_THRESHOLD = Decimal("70.00")
TIER_BRONZE_THRESHOLD = Decimal("50.00")


# ---------------------------------------------------------------------------
# Default Weights
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoringWeights:
    """Immutable weight configuration for accuracy scoring.

    All weights must sum to 1.00 for a valid configuration.
    Each weight scales the corresponding dimension's raw score
    to produce the weighted sub-score.

    Attributes:
        coordinate_precision: Weight for coordinate precision (default 0.20).
        polygon_quality: Weight for polygon topology (default 0.20).
        country_match: Weight for country matching (default 0.15).
        protected_area: Weight for protected area check (default 0.15).
        deforestation: Weight for deforestation check (default 0.15).
        temporal_consistency: Weight for temporal analysis (default 0.15).
    """

    coordinate_precision: Decimal = Decimal("0.20")
    polygon_quality: Decimal = Decimal("0.20")
    country_match: Decimal = Decimal("0.15")
    protected_area: Decimal = Decimal("0.15")
    deforestation: Decimal = Decimal("0.15")
    temporal_consistency: Decimal = Decimal("0.15")

    def __post_init__(self) -> None:
        """Validate that weights sum to 1.00."""
        total = (
            self.coordinate_precision
            + self.polygon_quality
            + self.country_match
            + self.protected_area
            + self.deforestation
            + self.temporal_consistency
        )
        if total != Decimal("1.00"):
            raise ValueError(
                f"Scoring weights must sum to 1.00, got {total}. "
                f"Weights: coord={self.coordinate_precision}, "
                f"poly={self.polygon_quality}, "
                f"country={self.country_match}, "
                f"protected={self.protected_area}, "
                f"deforest={self.deforestation}, "
                f"temporal={self.temporal_consistency}"
            )
        for name, weight in [
            ("coordinate_precision", self.coordinate_precision),
            ("polygon_quality", self.polygon_quality),
            ("country_match", self.country_match),
            ("protected_area", self.protected_area),
            ("deforestation", self.deforestation),
            ("temporal_consistency", self.temporal_consistency),
        ]:
            if weight < Decimal("0"):
                raise ValueError(f"{name} must be non-negative, got {weight}")

    def to_dict(self) -> Dict[str, str]:
        """Serialize weights to dictionary of strings."""
        return {
            "coordinate_precision": str(self.coordinate_precision),
            "polygon_quality": str(self.polygon_quality),
            "country_match": str(self.country_match),
            "protected_area": str(self.protected_area),
            "deforestation": str(self.deforestation),
            "temporal_consistency": str(self.temporal_consistency),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoringWeights":
        """Create from dictionary, converting values to Decimal."""
        kwargs: Dict[str, Decimal] = {}
        for key in [
            "coordinate_precision", "polygon_quality", "country_match",
            "protected_area", "deforestation", "temporal_consistency",
        ]:
            if key in data:
                kwargs[key] = _to_decimal(data[key])
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# AccuracyScoringEngine
# ---------------------------------------------------------------------------


class AccuracyScoringEngine:
    """Deterministic geolocation accuracy scoring engine for EUDR compliance.

    Computes composite accuracy scores from verification sub-results using
    Decimal arithmetic for bit-perfect reproducibility. Each dimension
    produces a sub-score (0 to its maximum), and the weighted sum yields
    the total score (0-100).

    All scoring is rule-based and deterministic. No LLM or ML model is
    used in any code path.

    Example::

        engine = AccuracyScoringEngine()
        score = engine.calculate_score(
            coordinate_result=coord_result,
            polygon_result=poly_result,
            protected_area_result=pa_result,
            deforestation_result=defor_result,
            temporal_result=temporal_result,
        )
        assert score.quality_tier in (QualityTier.GOLD, QualityTier.SILVER)
        assert score.provenance_hash != ""

    Attributes:
        weights: Scoring weight configuration.
    """

    def __init__(
        self,
        weights: Optional[ScoringWeights] = None,
        config: Any = None,
    ) -> None:
        """Initialize the AccuracyScoringEngine.

        Args:
            weights: Custom scoring weights. If None, default weights
                (0.20/0.20/0.15/0.15/0.15/0.15) are used.
            config: Optional GeolocationVerificationConfig instance.
                If provided and weights is None, attempts to build
                ScoringWeights from config.score_weights dictionary.
        """
        if weights is None and config is not None:
            score_weights = getattr(config, "score_weights", None)
            if score_weights and isinstance(score_weights, dict):
                try:
                    weights = ScoringWeights(
                        coordinate_precision=_to_decimal(
                            score_weights.get("precision", 0.20)
                        ),
                        polygon_quality=_to_decimal(
                            score_weights.get("polygon", 0.20)
                        ),
                        country_match=_to_decimal(
                            score_weights.get("country", 0.15)
                        ),
                        protected_area=_to_decimal(
                            score_weights.get("protected", 0.15)
                        ),
                        deforestation=_to_decimal(
                            score_weights.get("deforestation", 0.15)
                        ),
                        temporal_consistency=_to_decimal(
                            score_weights.get("temporal", 0.15)
                        ),
                    )
                except (ValueError, TypeError):
                    weights = None  # Fall back to defaults
        self.weights = weights or ScoringWeights()
        logger.info(
            "AccuracyScoringEngine initialized: weights=[coord=%.2f, "
            "poly=%.2f, country=%.2f, protected=%.2f, deforest=%.2f, "
            "temporal=%.2f]",
            self.weights.coordinate_precision,
            self.weights.polygon_quality,
            self.weights.country_match,
            self.weights.protected_area,
            self.weights.deforestation,
            self.weights.temporal_consistency,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_score(
        self,
        coordinate_result: Optional[CoordinateValidationResult] = None,
        polygon_result: Optional[PolygonVerificationResult] = None,
        protected_area_result: Optional[ProtectedAreaCheckResult] = None,
        deforestation_result: Optional[DeforestationVerificationResult] = None,
        temporal_result: Optional[TemporalChangeResult] = None,
        weights: Optional[ScoringWeights] = None,
    ) -> GeolocationAccuracyScore:
        """Calculate the composite geolocation accuracy score -- DETERMINISTIC.

        All sub-scores are computed from the verification results using
        deterministic rules. The total score is the weighted sum of all
        sub-scores, clamped to [0, 100].

        Args:
            coordinate_result: Result from CoordinateValidator.
            polygon_result: Result from PolygonTopologyVerifier.
            protected_area_result: Result from protected area check.
            deforestation_result: Result from deforestation check.
            temporal_result: Result from TemporalConsistencyAnalyzer.
            weights: Optional custom weights (overrides engine default).

        Returns:
            GeolocationAccuracyScore with all sub-scores, total, tier.
        """
        start_time = time.monotonic()
        w = weights or self.weights

        # Compute sub-scores using Decimal arithmetic
        coord_score = self._score_coordinate_precision(coordinate_result)
        poly_score = self._score_polygon_quality(polygon_result)
        country_score = self._score_country_match(coordinate_result)
        protected_score = self._score_protected_area(protected_area_result)
        defor_score = self._score_deforestation(deforestation_result)
        temporal_score = self._score_temporal_consistency(temporal_result)

        # Compute weighted total
        # Each sub-score is already on its native scale (0-max).
        # The weights represent the proportion of the total 100 points.
        # So we normalize: sub_score / max_sub * weight * 100
        total = _clamp_score(
            (coord_score / MAX_COORD_PRECISION_SCORE * w.coordinate_precision
             + poly_score / MAX_POLYGON_QUALITY_SCORE * w.polygon_quality
             + country_score / MAX_COUNTRY_MATCH_SCORE * w.country_match
             + protected_score / MAX_PROTECTED_AREA_SCORE * w.protected_area
             + defor_score / MAX_DEFORESTATION_SCORE * w.deforestation
             + temporal_score / MAX_TEMPORAL_SCORE * w.temporal_consistency
             ) * MAX_TOTAL_SCORE,
            MAX_TOTAL_SCORE,
        )

        # Determine quality tier
        quality_tier = self._determine_quality_tier(total)

        # Build result
        result = GeolocationAccuracyScore(
            total_score=total,
            coordinate_precision_score=coord_score,
            polygon_quality_score=poly_score,
            country_match_score=country_score,
            protected_area_score=protected_score,
            deforestation_score=defor_score,
            temporal_consistency_score=temporal_score,
            quality_tier=quality_tier,
            weights_used=w.to_dict(),
        )

        # Compute provenance hash
        result.provenance_hash = self._compute_result_hash(result)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Accuracy score %s: total=%.2f, tier=%s, coord=%.2f, "
            "poly=%.2f, country=%.2f, protected=%.2f, defor=%.2f, "
            "temporal=%.2f, %.3fms",
            result.score_id, total, quality_tier.value,
            coord_score, poly_score, country_score,
            protected_score, defor_score, temporal_score, elapsed_ms,
        )

        return result

    def calculate_batch_scores(
        self,
        results: List[Dict[str, Any]],
        weights: Optional[ScoringWeights] = None,
    ) -> List[GeolocationAccuracyScore]:
        """Calculate accuracy scores for a batch of verification results.

        Each entry in the list should be a dictionary with optional keys:
        'coordinate_result', 'polygon_result', 'protected_area_result',
        'deforestation_result', 'temporal_result'.

        Args:
            results: List of dictionaries containing verification results.
            weights: Optional custom weights for the batch.

        Returns:
            List of GeolocationAccuracyScore objects.
        """
        start_time = time.monotonic()

        if not results:
            logger.warning("calculate_batch_scores called with empty list")
            return []

        scores: List[GeolocationAccuracyScore] = []
        for entry in results:
            score = self.calculate_score(
                coordinate_result=entry.get("coordinate_result"),
                polygon_result=entry.get("polygon_result"),
                protected_area_result=entry.get("protected_area_result"),
                deforestation_result=entry.get("deforestation_result"),
                temporal_result=entry.get("temporal_result"),
                weights=weights,
            )
            scores.append(score)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Batch accuracy scoring: %d results, %.1fms",
            len(scores), elapsed_ms,
        )

        return scores

    def get_aggregate_statistics(
        self, scores: List[GeolocationAccuracyScore]
    ) -> Dict[str, Any]:
        """Compute aggregate statistics over a list of accuracy scores.

        Returns mean, median, standard deviation, and distribution
        by quality tier. All statistics use Decimal arithmetic.

        Args:
            scores: List of GeolocationAccuracyScore objects.

        Returns:
            Dictionary with aggregate statistics.
        """
        if not scores:
            return {
                "count": 0,
                "mean": "0.00",
                "median": "0.00",
                "std_dev": "0.00",
                "min": "0.00",
                "max": "0.00",
                "tier_distribution": {
                    "gold": 0, "silver": 0, "bronze": 0, "fail": 0,
                },
            }

        total_scores = [s.total_score for s in scores]
        n = Decimal(str(len(total_scores)))

        # Mean
        total_sum = sum(total_scores, Decimal("0.00"))
        mean = (total_sum / n).quantize(SCORE_PRECISION, rounding=ROUND_HALF_UP)

        # Sorted for median and min/max
        sorted_scores = sorted(total_scores)
        min_score = sorted_scores[0].quantize(SCORE_PRECISION)
        max_score = sorted_scores[-1].quantize(SCORE_PRECISION)

        # Median
        count = len(sorted_scores)
        if count % 2 == 0:
            median = (
                (sorted_scores[count // 2 - 1] + sorted_scores[count // 2])
                / Decimal("2")
            ).quantize(SCORE_PRECISION, rounding=ROUND_HALF_UP)
        else:
            median = sorted_scores[count // 2].quantize(SCORE_PRECISION)

        # Standard deviation
        if count > 1:
            variance = sum(
                (s - mean) ** 2 for s in total_scores
            ) / (n - Decimal("1"))
            # Decimal sqrt via Newton's method
            std_dev = self._decimal_sqrt(variance).quantize(
                SCORE_PRECISION, rounding=ROUND_HALF_UP
            )
        else:
            std_dev = Decimal("0.00")

        # Tier distribution
        tier_dist = {"gold": 0, "silver": 0, "bronze": 0, "fail": 0}
        for s in scores:
            tier_dist[s.quality_tier.value] += 1

        return {
            "count": count,
            "mean": str(mean),
            "median": str(median),
            "std_dev": str(std_dev),
            "min": str(min_score),
            "max": str(max_score),
            "tier_distribution": tier_dist,
        }

    # ------------------------------------------------------------------
    # Internal: Sub-Score Calculations
    # ------------------------------------------------------------------

    def _score_coordinate_precision(
        self, result: Optional[CoordinateValidationResult]
    ) -> Decimal:
        """Score coordinate precision (0-20 points).

        Scoring rules:
            - Base: precision_score * 12 points (max 12 from precision)
            - WGS84 valid: +3 points
            - On land: +2 points
            - No transposition: +1 point
            - No duplicate: +1 point
            - No cluster anomaly: +1 point

        If no result is provided, returns 0.

        Args:
            result: CoordinateValidationResult from validator.

        Returns:
            Decimal score in range [0, 20].
        """
        if result is None:
            return Decimal("0.00")

        score = Decimal("0.00")

        # Precision contribution (0-12 points)
        precision_raw = _to_decimal(result.precision_score) * Decimal("12")
        score += precision_raw

        # WGS84 validity (+3)
        if result.wgs84_valid:
            score += Decimal("3.00")

        # On land (+2)
        if result.is_on_land:
            score += Decimal("2.00")

        # No transposition (+1)
        if not result.transposition_detected:
            score += Decimal("1.00")

        # No duplicate (+1)
        if not result.is_duplicate:
            score += Decimal("1.00")

        # No cluster anomaly (+1)
        if not result.cluster_anomaly:
            score += Decimal("1.00")

        return _clamp_score(score, MAX_COORD_PRECISION_SCORE)

    def _score_polygon_quality(
        self, result: Optional[PolygonVerificationResult]
    ) -> Decimal:
        """Score polygon topology quality (0-20 points).

        Scoring rules:
            - Ring closed: +3 points
            - CCW winding order: +2 points
            - No self-intersection: +4 points
            - Area within tolerance: +3 points
            - Not a sliver: +2 points
            - No spike vertices: +2 points
            - Adequate vertex density: +1 point
            - Max area OK: +1 point
            - Vertex count bonus: +2 points if >= 4 unique vertices

        If no result is provided, returns 0.

        Args:
            result: PolygonVerificationResult from verifier.

        Returns:
            Decimal score in range [0, 20].
        """
        if result is None:
            return Decimal("0.00")

        score = Decimal("0.00")

        if result.ring_closed:
            score += Decimal("3.00")

        if result.winding_order_ccw:
            score += Decimal("2.00")

        if not result.has_self_intersection:
            score += Decimal("4.00")

        if result.area_within_tolerance:
            score += Decimal("3.00")

        if not result.is_sliver:
            score += Decimal("2.00")

        if not result.has_spikes:
            score += Decimal("2.00")

        if result.vertex_density_ok:
            score += Decimal("1.00")

        if result.max_area_ok:
            score += Decimal("1.00")

        # Vertex count bonus: >= 4 unique vertices for a proper polygon
        if result.vertex_count >= 4:
            score += Decimal("2.00")

        return _clamp_score(score, MAX_POLYGON_QUALITY_SCORE)

    def _score_country_match(
        self, result: Optional[CoordinateValidationResult]
    ) -> Decimal:
        """Score country matching (0-15 points).

        Scoring rules:
            - Country matches declared: +10 points
            - Country resolved (even if mismatch): +3 points
            - Elevation plausible: +2 points

        If no result is provided, returns 0.

        Args:
            result: CoordinateValidationResult from validator.

        Returns:
            Decimal score in range [0, 15].
        """
        if result is None:
            return Decimal("0.00")

        score = Decimal("0.00")

        if result.country_match:
            score += Decimal("10.00")

        if result.resolved_country is not None:
            score += Decimal("3.00")

        if result.elevation_plausible:
            score += Decimal("2.00")

        return _clamp_score(score, MAX_COUNTRY_MATCH_SCORE)

    def _score_protected_area(
        self, result: Optional[ProtectedAreaCheckResult]
    ) -> Decimal:
        """Score protected area compliance (0-15 points).

        Scoring rules:
            - Not overlapping any protected area: +15 points (full score)
            - Overlapping but < 5%: +10 points
            - Overlapping 5-25%: +5 points
            - Overlapping 25-50%: +2 points
            - Overlapping > 50%: 0 points

        If no result is provided (check not performed), returns the
        full score to avoid penalizing missing data.

        Args:
            result: ProtectedAreaCheckResult from checker.

        Returns:
            Decimal score in range [0, 15].
        """
        if result is None:
            # No check performed -- give full credit (neutral)
            return MAX_PROTECTED_AREA_SCORE

        if not result.overlaps_protected:
            return MAX_PROTECTED_AREA_SCORE

        overlap_pct = _to_decimal(result.overlap_percentage)

        if overlap_pct < Decimal("5"):
            return Decimal("10.00")
        elif overlap_pct < Decimal("25"):
            return Decimal("5.00")
        elif overlap_pct < Decimal("50"):
            return Decimal("2.00")
        else:
            return Decimal("0.00")

    def _score_deforestation(
        self, result: Optional[DeforestationVerificationResult]
    ) -> Decimal:
        """Score deforestation verification (0-15 points).

        Scoring rules:
            - No deforestation detected: +15 points (full score)
            - Deforestation detected with low confidence (<0.5): +10 points
            - Deforestation detected, confidence 0.5-0.8: +5 points
            - Deforestation detected, confidence > 0.8: +0 points
            - Alert count penalty: -1 point per alert (max -5)
            - Forest loss area penalty: -1 per 10 ha lost (max -5)

        If no result is provided, returns full score (neutral).

        Args:
            result: DeforestationVerificationResult.

        Returns:
            Decimal score in range [0, 15].
        """
        if result is None:
            return MAX_DEFORESTATION_SCORE

        if not result.deforestation_detected:
            return MAX_DEFORESTATION_SCORE

        confidence = _to_decimal(result.confidence)
        score: Decimal

        if confidence < Decimal("0.5"):
            score = Decimal("10.00")
        elif confidence < Decimal("0.8"):
            score = Decimal("5.00")
        else:
            score = Decimal("0.00")

        # Alert count penalty
        alert_penalty = min(
            _to_decimal(result.alert_count) * Decimal("1.00"),
            Decimal("5.00"),
        )
        score -= alert_penalty

        # Forest loss area penalty
        loss_penalty = min(
            _to_decimal(result.forest_loss_ha) / Decimal("10") * Decimal("1.00"),
            Decimal("5.00"),
        )
        score -= loss_penalty

        return _clamp_score(score, MAX_DEFORESTATION_SCORE)

    def _score_temporal_consistency(
        self, result: Optional[TemporalChangeResult]
    ) -> Decimal:
        """Score temporal boundary consistency (0-15 points).

        Scoring rules:
            - Consistent (no concerning changes): +15 points
            - Minor boundary change (< 5% area, < 50m shift): +12 points
            - Moderate change (5-20% area or 50-200m shift): +8 points
            - Significant change (> 20% area or > 200m shift): +3 points
            - Rapid change detected: -5 points
            - Forest encroachment detected: -5 points

        If no result is provided, returns full score (neutral).

        Args:
            result: TemporalChangeResult from analyzer.

        Returns:
            Decimal score in range [0, 15].
        """
        if result is None:
            return MAX_TEMPORAL_SCORE

        if result.is_consistent and not result.rapid_change_detected:
            return MAX_TEMPORAL_SCORE

        score = MAX_TEMPORAL_SCORE

        # Apply penalties based on boundary change
        if result.boundary_change is not None:
            area_change = abs(_to_decimal(result.boundary_change.area_change_pct))
            centroid_shift = _to_decimal(result.boundary_change.centroid_shift_m)

            if area_change > Decimal("20") or centroid_shift > Decimal("200"):
                score = Decimal("3.00")
            elif area_change > Decimal("5") or centroid_shift > Decimal("50"):
                score = Decimal("8.00")
            elif area_change > Decimal("0") or centroid_shift > Decimal("0"):
                score = Decimal("12.00")

            # Forest encroachment penalty
            if result.boundary_change.forest_encroachment:
                score -= Decimal("5.00")

        # Rapid change penalty
        if result.rapid_change_detected:
            score -= Decimal("5.00")

        return _clamp_score(score, MAX_TEMPORAL_SCORE)

    # ------------------------------------------------------------------
    # Internal: Quality Tier Classification
    # ------------------------------------------------------------------

    def _determine_quality_tier(self, total_score: Decimal) -> QualityTier:
        """Classify total score into a quality tier.

        Thresholds:
            - GOLD:   >= 85.00
            - SILVER: >= 70.00
            - BRONZE: >= 50.00
            - FAIL:   < 50.00

        Args:
            total_score: Total composite score (0-100).

        Returns:
            QualityTier classification.
        """
        if total_score >= TIER_GOLD_THRESHOLD:
            return QualityTier.GOLD
        elif total_score >= TIER_SILVER_THRESHOLD:
            return QualityTier.SILVER
        elif total_score >= TIER_BRONZE_THRESHOLD:
            return QualityTier.BRONZE
        else:
            return QualityTier.FAIL

    # ------------------------------------------------------------------
    # Internal: Decimal Math Helpers
    # ------------------------------------------------------------------

    def _decimal_sqrt(self, value: Decimal, precision: int = 20) -> Decimal:
        """Compute square root of a Decimal using Newton's method.

        Args:
            value: Non-negative Decimal value.
            precision: Number of iterations for convergence.

        Returns:
            Square root as Decimal.
        """
        if value < Decimal("0"):
            raise ValueError(f"Cannot compute sqrt of negative value: {value}")
        if value == Decimal("0"):
            return Decimal("0")

        # Initial guess
        guess = value / Decimal("2")
        for _ in range(precision):
            guess = (guess + value / guess) / Decimal("2")

        return guess

    # ------------------------------------------------------------------
    # Internal: Provenance Hash
    # ------------------------------------------------------------------

    def _compute_result_hash(
        self, result: GeolocationAccuracyScore
    ) -> str:
        """Compute SHA-256 provenance hash for an accuracy score.

        Covers all sub-scores, total, tier, and weights for deterministic
        reproducibility verification.

        Args:
            result: The accuracy score result to hash.

        Returns:
            SHA-256 hex digest.
        """
        hash_data = {
            "module_version": _MODULE_VERSION,
            "total_score": str(result.total_score),
            "coordinate_precision_score": str(result.coordinate_precision_score),
            "polygon_quality_score": str(result.polygon_quality_score),
            "country_match_score": str(result.country_match_score),
            "protected_area_score": str(result.protected_area_score),
            "deforestation_score": str(result.deforestation_score),
            "temporal_consistency_score": str(result.temporal_consistency_score),
            "quality_tier": result.quality_tier.value,
            "weights_used": result.weights_used,
        }
        return _compute_hash(hash_data)


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    "AccuracyScoringEngine",
    "ScoringWeights",
    "MAX_COORD_PRECISION_SCORE",
    "MAX_POLYGON_QUALITY_SCORE",
    "MAX_COUNTRY_MATCH_SCORE",
    "MAX_PROTECTED_AREA_SCORE",
    "MAX_DEFORESTATION_SCORE",
    "MAX_TEMPORAL_SCORE",
    "MAX_TOTAL_SCORE",
    "TIER_GOLD_THRESHOLD",
    "TIER_SILVER_THRESHOLD",
    "TIER_BRONZE_THRESHOLD",
]
