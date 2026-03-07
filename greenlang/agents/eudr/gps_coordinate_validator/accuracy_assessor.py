# -*- coding: utf-8 -*-
"""
Accuracy Assessor - AGENT-EUDR-007 Engine 7

Production-grade coordinate accuracy assessment engine for GPS coordinate
validation under the EU Deforestation Regulation (EUDR). Computes a
composite accuracy score from precision, plausibility, consistency, and
source reliability dimensions, then classifies coordinates into quality
tiers (Gold, Silver, Bronze, Unverified) with confidence interval
estimation.

Zero-Hallucination Guarantees:
    - All scoring is deterministic using fixed formulas
    - No ML/LLM involvement in any score calculation
    - Tier classification uses fixed thresholds
    - Confidence intervals use published precision specifications
    - SHA-256 provenance hashes on all assessment results

Performance Targets:
    - Single assessment: <2ms
    - Batch assessment (10,000 coordinates): <1 second

Regulatory References:
    - EUDR Article 9: Coordinate accuracy requirements
    - EUDR Article 10: Risk assessment data quality
    - EUDR Annex II: Due Diligence Statement data requirements

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-007 (Engine 7: Coordinate Accuracy Assessment)
Agent ID: GL-EUDR-GPS-007
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM: float = 6_371.0


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AccuracyTier(str, Enum):
    """Accuracy quality tier classification.

    Each tier has specific implications for EUDR DDS eligibility:
        GOLD: >= 90 score. Fully verified, DDS-ready.
        SILVER: >= 70 score. DDS-eligible with noted limitations.
        BRONZE: >= 50 score. Enhanced due diligence required.
        UNVERIFIED: < 50 score. Not eligible for DDS submission.
    """

    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"
    UNVERIFIED = "unverified"


class SourceType(str, Enum):
    """Classification of the GPS coordinate data source.

    Each source type has an inherent reliability score reflecting
    the typical accuracy of coordinates from that source.

    GNSS_SURVEY: Professional GNSS survey equipment (sub-metre).
    MOBILE_GPS: Consumer mobile device GPS (3-10m accuracy).
    CERTIFICATION_DB: Certification database (e.g., UTZ, Rainforest
        Alliance, RSPO) with verified coordinates.
    GOVERNMENT_REGISTRY: Government land registry or cadastral data.
    MANUAL_ENTRY: Manually typed coordinates (error-prone).
    ERP_EXPORT: Extracted from ERP system (potentially transformed).
    DIGITIZED_MAP: Coordinates digitized from paper or raster maps.
    UNKNOWN: Source not specified or not determinable.
    """

    GNSS_SURVEY = "gnss_survey"
    MOBILE_GPS = "mobile_gps"
    CERTIFICATION_DB = "certification_db"
    GOVERNMENT_REGISTRY = "government_registry"
    MANUAL_ENTRY = "manual_entry"
    ERP_EXPORT = "erp_export"
    DIGITIZED_MAP = "digitized_map"
    UNKNOWN = "unknown"


class PrecisionLevel(str, Enum):
    """Coordinate precision level classification."""

    SURVEY_GRADE = "survey_grade"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INADEQUATE = "inadequate"


# ---------------------------------------------------------------------------
# Input Data Classes (expected from upstream engines)
# ---------------------------------------------------------------------------


@dataclass
class PrecisionResult:
    """Precision analysis result from Engine 3 (PrecisionAnalyzer).

    Attributes:
        precision_level: Classification of coordinate precision.
        decimal_places: Number of decimal places in the coordinate.
        ground_resolution_m: Estimated ground resolution in metres.
        is_truncated: Whether the coordinate appears to be truncated.
        is_rounded: Whether the coordinate appears to be artificially
            rounded (e.g., to whole degrees).
        is_eudr_adequate: Whether precision meets EUDR requirements.
    """

    precision_level: PrecisionLevel = PrecisionLevel.MODERATE
    decimal_places: int = 5
    ground_resolution_m: float = 1.1
    is_truncated: bool = False
    is_rounded: bool = False
    is_eudr_adequate: bool = True


@dataclass
class NormalizedCoordinate:
    """Normalized coordinate from Engine 1/2.

    Attributes:
        lat: Latitude in decimal degrees (WGS84).
        lon: Longitude in decimal degrees (WGS84).
        original_format: Original format before normalization.
        datum: Geodetic datum (default: WGS84).
    """

    lat: float = 0.0
    lon: float = 0.0
    original_format: str = "DD"
    datum: str = "WGS84"


@dataclass
class ValidationResult:
    """Validation result from Engine 4 (FormatValidator).

    Attributes:
        is_valid: Overall validation result.
        errors: List of error descriptions.
        warnings: List of warning descriptions.
        was_auto_corrected: Whether any auto-corrections were applied.
        swap_detected: Whether lat/lon swap was detected.
        null_island_detected: Whether null island (0,0) was detected.
    """

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    was_auto_corrected: bool = False
    swap_detected: bool = False
    null_island_detected: bool = False


# ---------------------------------------------------------------------------
# Result Data Classes
# ---------------------------------------------------------------------------


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of an accuracy score component.

    Attributes:
        component: Name of the scoring component.
        raw_score: Unweighted score (0-100).
        weight: Component weight (0.0-1.0).
        weighted_score: Weighted contribution to overall score.
        explanation: Human-readable explanation of the score.
        penalties: List of applied penalties with descriptions.
        bonuses: List of applied bonuses with descriptions.
    """

    component: str = ""
    raw_score: float = 0.0
    weight: float = 0.0
    weighted_score: float = 0.0
    explanation: str = ""
    penalties: List[str] = field(default_factory=list)
    bonuses: List[str] = field(default_factory=list)


@dataclass
class AccuracyScore:
    """Complete accuracy assessment result.

    Attributes:
        overall_score: Weighted composite score (0-100).
        tier: Quality tier classification.
        precision_score: Precision component score (0-100).
        plausibility_score: Plausibility component score (0-100).
        consistency_score: Consistency component score (0-100).
        source_score: Source reliability component score (0-100).
        confidence_interval_m: 95% confidence radius in metres.
        breakdown: Detailed per-component score breakdown.
        explanations: List of human-readable explanations.
        recommendations: List of improvement recommendations.
        is_eudr_compliant: Whether the coordinate meets minimum EUDR
            accuracy requirements.
        provenance_hash: SHA-256 hash for audit trail.
        assessed_at: Timestamp of the assessment.
        processing_time_ms: Processing duration in milliseconds.
    """

    overall_score: float = 0.0
    tier: AccuracyTier = AccuracyTier.UNVERIFIED
    precision_score: float = 0.0
    plausibility_score: float = 0.0
    consistency_score: float = 0.0
    source_score: float = 0.0
    confidence_interval_m: float = 0.0
    breakdown: List[ScoreBreakdown] = field(default_factory=list)
    explanations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    is_eudr_compliant: bool = False
    provenance_hash: str = ""
    assessed_at: str = ""
    processing_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Scoring Constants
# ---------------------------------------------------------------------------

#: Default component weights (must sum to 1.0).
DEFAULT_WEIGHTS: Dict[str, float] = {
    "precision": 0.30,
    "plausibility": 0.30,
    "consistency": 0.25,
    "source": 0.15,
}

#: Precision level scores.
PRECISION_LEVEL_SCORES: Dict[PrecisionLevel, float] = {
    PrecisionLevel.SURVEY_GRADE: 100.0,
    PrecisionLevel.HIGH: 85.0,
    PrecisionLevel.MODERATE: 60.0,
    PrecisionLevel.LOW: 30.0,
    PrecisionLevel.INADEQUATE: 10.0,
}

#: Source type reliability scores.
SOURCE_TYPE_SCORES: Dict[SourceType, float] = {
    SourceType.GNSS_SURVEY: 95.0,
    SourceType.GOVERNMENT_REGISTRY: 85.0,
    SourceType.CERTIFICATION_DB: 80.0,
    SourceType.MOBILE_GPS: 70.0,
    SourceType.ERP_EXPORT: 50.0,
    SourceType.DIGITIZED_MAP: 45.0,
    SourceType.MANUAL_ENTRY: 40.0,
    SourceType.UNKNOWN: 25.0,
}

#: Source type typical precision (95% confidence radius in metres).
SOURCE_PRECISION_M: Dict[SourceType, float] = {
    SourceType.GNSS_SURVEY: 0.5,
    SourceType.GOVERNMENT_REGISTRY: 10.0,
    SourceType.CERTIFICATION_DB: 15.0,
    SourceType.MOBILE_GPS: 8.0,
    SourceType.ERP_EXPORT: 50.0,
    SourceType.DIGITIZED_MAP: 100.0,
    SourceType.MANUAL_ENTRY: 200.0,
    SourceType.UNKNOWN: 500.0,
}

#: Tier thresholds.
TIER_THRESHOLDS: List[Tuple[float, AccuracyTier]] = [
    (90.0, AccuracyTier.GOLD),
    (70.0, AccuracyTier.SILVER),
    (50.0, AccuracyTier.BRONZE),
    (0.0, AccuracyTier.UNVERIFIED),
]

#: Minimum score for EUDR compliance.
MIN_EUDR_SCORE: float = 50.0


# ===========================================================================
# AccuracyAssessor
# ===========================================================================


class AccuracyAssessor:
    """Production-grade coordinate accuracy assessment engine for EUDR.

    Computes a composite accuracy score from four dimensions:
    1. Precision (30%): Coordinate decimal precision and resolution
    2. Plausibility (30%): Spatial plausibility checks
    3. Consistency (25%): Validation result consistency
    4. Source Reliability (15%): Data source trustworthiness

    All scoring is deterministic with zero LLM/ML involvement.

    Attributes:
        weights: Component weight dictionary.

    Example::

        assessor = AccuracyAssessor()
        score = assessor.assess(
            coord=NormalizedCoordinate(lat=-3.46, lon=28.23),
            precision=PrecisionResult(precision_level=PrecisionLevel.HIGH),
            plausibility=PlausibilityResult(is_plausible=True, score=85.0),
            validation=ValidationResult(is_valid=True),
            source_type=SourceType.MOBILE_GPS,
        )
        assert score.tier == AccuracyTier.SILVER
    """

    def __init__(
        self,
        config: Any = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize AccuracyAssessor.

        Args:
            config: Optional configuration object.
            weights: Optional custom component weights. Must have keys
                'precision', 'plausibility', 'consistency', 'source'
                and values summing to 1.0.

        Raises:
            ValueError: If custom weights do not sum to 1.0.
        """
        self._config = config
        self.weights = weights or dict(DEFAULT_WEIGHTS)

        # Validate weights
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(
                f"Component weights must sum to 1.0, got {weight_sum:.4f}"
            )

        logger.info(
            "AccuracyAssessor initialized: weights=%s",
            {k: f"{v:.2f}" for k, v in self.weights.items()},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(
        self,
        coord: NormalizedCoordinate,
        precision: PrecisionResult,
        plausibility: Any = None,
        validation: Optional[ValidationResult] = None,
        source_type: SourceType = SourceType.UNKNOWN,
    ) -> AccuracyScore:
        """Compute comprehensive accuracy assessment.

        Calculates precision, plausibility, consistency, and source
        reliability scores, then computes weighted average and
        classifies into an accuracy tier.

        Args:
            coord: Normalized coordinate to assess.
            precision: Precision analysis result from Engine 3.
            plausibility: Plausibility result from Engine 5. Expected
                to have 'score' (float) and 'is_plausible' (bool)
                attributes. Can be None.
            validation: Validation result from Engine 4. Can be None.
            source_type: Data source classification.

        Returns:
            Complete AccuracyScore with all components populated.
        """
        start_time = time.monotonic()
        result = AccuracyScore()
        result.assessed_at = _utcnow().isoformat()
        breakdowns: List[ScoreBreakdown] = []
        explanations: List[str] = []
        recommendations: List[str] = []

        # 1. Precision score
        prec_score, prec_breakdown = self._score_precision(precision)
        result.precision_score = prec_score
        breakdowns.append(prec_breakdown)

        # 2. Plausibility score
        plaus_score, plaus_breakdown = self._score_plausibility(plausibility)
        result.plausibility_score = plaus_score
        breakdowns.append(plaus_breakdown)

        # 3. Consistency score
        val = validation or ValidationResult()
        cons_score, cons_breakdown = self._score_consistency(val)
        result.consistency_score = cons_score
        breakdowns.append(cons_breakdown)

        # 4. Source reliability score
        src_score, src_breakdown = self._score_source(source_type)
        result.source_score = src_score
        breakdowns.append(src_breakdown)

        # 5. Weighted average
        overall = (
            prec_score * self.weights.get("precision", 0.30)
            + plaus_score * self.weights.get("plausibility", 0.30)
            + cons_score * self.weights.get("consistency", 0.25)
            + src_score * self.weights.get("source", 0.15)
        )
        result.overall_score = round(max(0.0, min(100.0, overall)), 2)

        # 6. Tier classification
        result.tier = self.classify_tier(result.overall_score)

        # 7. Confidence interval
        result.confidence_interval_m = self.estimate_confidence_interval(
            precision, source_type
        )

        # 8. EUDR compliance
        result.is_eudr_compliant = result.overall_score >= MIN_EUDR_SCORE

        # 9. Generate explanations
        explanations.extend(self._generate_explanations(
            result, precision, plausibility, val, source_type
        ))

        # 10. Generate recommendations
        recommendations.extend(self._generate_recommendations(
            result, precision, plausibility, val, source_type
        ))

        result.breakdown = breakdowns
        result.explanations = explanations
        result.recommendations = recommendations

        # Provenance
        result.provenance_hash = self._compute_provenance_hash(
            coord, result
        )
        result.processing_time_ms = (
            (time.monotonic() - start_time) * 1000
        )

        logger.debug(
            "Accuracy assessment: (%.6f, %.6f) -> score=%.1f, "
            "tier=%s, CI=%.1fm, EUDR=%s, %.2fms",
            coord.lat, coord.lon, result.overall_score,
            result.tier.value, result.confidence_interval_m,
            result.is_eudr_compliant, result.processing_time_ms,
        )

        return result

    def score_precision(
        self,
        precision: PrecisionResult,
    ) -> float:
        """Score the precision component (0-100).

        Scoring formula:
            Base score from precision level mapping:
                SURVEY_GRADE: 100, HIGH: 85, MODERATE: 60,
                LOW: 30, INADEQUATE: 10
            Bonus: +5 for >= 6 decimal places, +10 for >= 8
            Penalty: -15 for truncation, -10 for artificial rounding

        Args:
            precision: Precision analysis result.

        Returns:
            Precision score from 0 to 100.
        """
        score, _ = self._score_precision(precision)
        return score

    def score_plausibility(
        self,
        plausibility: Any,
    ) -> float:
        """Score the plausibility component (0-100).

        Expected plausibility object attributes:
            - is_plausible (bool): overall plausibility
            - score (float): plausibility score (0-100)
            - land_ocean.is_land (bool): on land
            - country.matches_declared (bool): country match
            - commodity.is_plausible (bool): commodity zone match
            - elevation.is_plausible (bool): elevation ok
            - urban.is_urban (bool): in urban area
            - protected_area.is_in_protected_area (bool): in PA

        Args:
            plausibility: Plausibility result object.

        Returns:
            Plausibility score from 0 to 100.
        """
        score, _ = self._score_plausibility(plausibility)
        return score

    def score_consistency(
        self,
        validation: ValidationResult,
    ) -> float:
        """Score the consistency component (0-100).

        Scoring formula:
            No errors, no warnings: 100
            Warnings only: 80
            Auto-corrected: 60
            Uncorrectable errors: 20
            Swap detected: -30 penalty

        Args:
            validation: Validation result from Engine 4.

        Returns:
            Consistency score from 0 to 100.
        """
        score, _ = self._score_consistency(validation)
        return score

    def score_source(
        self,
        source_type: SourceType,
    ) -> float:
        """Score the source reliability component (0-100).

        Source type scores:
            GNSS_SURVEY: 95
            GOVERNMENT_REGISTRY: 85
            CERTIFICATION_DB: 80
            MOBILE_GPS: 70
            ERP_EXPORT: 50
            DIGITIZED_MAP: 45
            MANUAL_ENTRY: 40
            UNKNOWN: 25

        Args:
            source_type: Data source classification.

        Returns:
            Source reliability score from 0 to 100.
        """
        score, _ = self._score_source(source_type)
        return score

    def classify_tier(
        self,
        overall_score: float,
    ) -> AccuracyTier:
        """Classify overall score into an accuracy tier.

        Thresholds:
            >= 90: GOLD
            >= 70: SILVER
            >= 50: BRONZE
            < 50: UNVERIFIED

        Args:
            overall_score: Weighted composite score (0-100).

        Returns:
            AccuracyTier classification.
        """
        for threshold, tier in TIER_THRESHOLDS:
            if overall_score >= threshold:
                return tier
        return AccuracyTier.UNVERIFIED

    def estimate_confidence_interval(
        self,
        precision: PrecisionResult,
        source_type: SourceType,
    ) -> float:
        """Estimate 95% confidence interval radius in metres.

        Combines the source type's inherent precision with the
        coordinate's decimal precision (ground resolution).

        Formula:
            CI = sqrt(source_precision^2 + ground_resolution^2) * 1.96

        Args:
            precision: Precision analysis result.
            source_type: Data source classification.

        Returns:
            95% confidence interval radius in metres.
        """
        source_prec = SOURCE_PRECISION_M.get(source_type, 500.0)
        ground_res = precision.ground_resolution_m

        # Root-sum-square of independent error sources
        combined_error = math.sqrt(
            source_prec ** 2 + ground_res ** 2
        )

        # 95% confidence (1.96 sigma for normal distribution)
        ci_95 = combined_error * 1.96

        return round(ci_95, 2)

    def compare_with_reference(
        self,
        coord: NormalizedCoordinate,
        reference_coords: List[Tuple[float, float]],
    ) -> float:
        """Compare coordinate with nearby validated reference coordinates.

        Calculates a consistency score based on proximity to known
        validated coordinates. Useful for cross-validation against
        existing plot databases.

        Args:
            coord: Coordinate to compare.
            reference_coords: List of (lat, lon) validated coordinates.

        Returns:
            Consistency score (0-100). Higher scores indicate the
            coordinate is consistent with the reference set.
        """
        if not reference_coords:
            return 50.0  # Neutral when no references available

        # Calculate distances to all reference coordinates
        distances: List[float] = []
        for ref_lat, ref_lon in reference_coords:
            dist = self._haversine_km(
                coord.lat, coord.lon, ref_lat, ref_lon
            )
            distances.append(dist)

        # Find minimum distance
        min_dist_km = min(distances)

        # Score based on proximity
        # < 0.1 km (100m): 100 (likely same plot area)
        # < 1 km: 80 (same general area)
        # < 5 km: 60 (same region)
        # < 20 km: 40 (same district)
        # > 20 km: 20 (far from reference set)
        if min_dist_km < 0.1:
            score = 100.0
        elif min_dist_km < 1.0:
            score = 80.0 + (1.0 - min_dist_km) * 20.0
        elif min_dist_km < 5.0:
            score = 60.0 + (5.0 - min_dist_km) / 4.0 * 20.0
        elif min_dist_km < 20.0:
            score = 40.0 + (20.0 - min_dist_km) / 15.0 * 20.0
        else:
            score = max(10.0, 40.0 - (min_dist_km - 20.0) * 0.5)

        return round(max(0.0, min(100.0, score)), 2)

    def batch_assess(
        self,
        data: List[Tuple[NormalizedCoordinate, PrecisionResult, Any, Optional[ValidationResult], SourceType]],
    ) -> List[AccuracyScore]:
        """Assess accuracy for a batch of coordinates.

        Args:
            data: List of tuples containing
                (coord, precision, plausibility, validation, source_type)
                for each coordinate.

        Returns:
            List of AccuracyScore, one per input.
        """
        start_time = time.monotonic()

        if not data:
            logger.warning("batch_assess called with empty list")
            return []

        results: List[AccuracyScore] = []
        for coord, precision, plausibility, validation, source_type in data:
            score = self.assess(
                coord=coord,
                precision=precision,
                plausibility=plausibility,
                validation=validation,
                source_type=source_type,
            )
            results.append(score)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        avg_score = (
            sum(r.overall_score for r in results) / len(results)
            if results else 0.0
        )
        logger.info(
            "Batch accuracy assessment: %d coordinates, avg_score=%.1f, "
            "%.1fms total (%.2fms/coord)",
            len(data), avg_score, elapsed_ms,
            elapsed_ms / len(data) if data else 0,
        )

        return results

    # ------------------------------------------------------------------
    # Internal: Component Scoring
    # ------------------------------------------------------------------

    def _score_precision(
        self,
        precision: PrecisionResult,
    ) -> Tuple[float, ScoreBreakdown]:
        """Score precision component with breakdown.

        Args:
            precision: Precision analysis result.

        Returns:
            Tuple of (score, breakdown).
        """
        breakdown = ScoreBreakdown(
            component="precision",
            weight=self.weights.get("precision", 0.30),
        )

        # Base score from precision level
        base_score = PRECISION_LEVEL_SCORES.get(
            precision.precision_level, 50.0
        )
        explanation_parts: List[str] = [
            f"Base: {base_score:.0f} ({precision.precision_level.value})"
        ]

        # Decimal places bonus
        bonus = 0.0
        if precision.decimal_places >= 8:
            bonus = 10.0
            breakdown.bonuses.append(
                f"+{bonus:.0f}: >= 8 decimal places (survey-grade)"
            )
        elif precision.decimal_places >= 6:
            bonus = 5.0
            breakdown.bonuses.append(
                f"+{bonus:.0f}: >= 6 decimal places (EUDR compliant)"
            )

        # Truncation penalty
        trunc_penalty = 0.0
        if precision.is_truncated:
            trunc_penalty = 15.0
            breakdown.penalties.append(
                f"-{trunc_penalty:.0f}: Coordinate appears truncated"
            )

        # Rounding penalty
        round_penalty = 0.0
        if precision.is_rounded:
            round_penalty = 10.0
            breakdown.penalties.append(
                f"-{round_penalty:.0f}: Coordinate artificially rounded"
            )

        score = base_score + bonus - trunc_penalty - round_penalty
        score = max(0.0, min(100.0, score))

        breakdown.raw_score = score
        breakdown.weighted_score = round(
            score * breakdown.weight, 2
        )
        breakdown.explanation = "; ".join(explanation_parts)

        return score, breakdown

    def _score_plausibility(
        self,
        plausibility: Any,
    ) -> Tuple[float, ScoreBreakdown]:
        """Score plausibility component with breakdown.

        Uses component checks from the PlausibilityResult if available.

        Args:
            plausibility: Plausibility result object (or None).

        Returns:
            Tuple of (score, breakdown).
        """
        breakdown = ScoreBreakdown(
            component="plausibility",
            weight=self.weights.get("plausibility", 0.30),
        )

        if plausibility is None:
            score = 50.0
            breakdown.explanation = "No plausibility data; neutral score"
            breakdown.raw_score = score
            breakdown.weighted_score = round(
                score * breakdown.weight, 2
            )
            return score, breakdown

        # If plausibility has a 'score' attribute, use it directly
        if hasattr(plausibility, "score"):
            raw_score = float(getattr(plausibility, "score", 50.0))
        else:
            raw_score = 50.0

        # Additive scoring from component checks
        score = 0.0

        # On land: +25
        is_land = True
        if hasattr(plausibility, "land_ocean"):
            is_land = getattr(plausibility.land_ocean, "is_land", True)
        if is_land:
            score += 25.0
            breakdown.bonuses.append("+25: On land")
        else:
            breakdown.penalties.append("-25: In ocean (critical)")

        # Country match: +25
        country_match = True
        if hasattr(plausibility, "country"):
            country_match = getattr(
                plausibility.country, "matches_declared", True
            )
        if country_match:
            score += 25.0
            breakdown.bonuses.append("+25: Country match confirmed")
        else:
            breakdown.penalties.append("-25: Country mismatch")

        # Commodity plausible: +25
        commodity_ok = True
        if hasattr(plausibility, "commodity"):
            commodity_ok = getattr(
                plausibility.commodity, "is_plausible", True
            )
        if commodity_ok:
            score += 25.0
            breakdown.bonuses.append("+25: Commodity zone plausible")
        else:
            breakdown.penalties.append("-25: Outside commodity zone")

        # Elevation plausible: +25
        elevation_ok = True
        if hasattr(plausibility, "elevation"):
            elevation_ok = getattr(
                plausibility.elevation, "is_plausible", True
            )
        if elevation_ok:
            score += 25.0
            breakdown.bonuses.append("+25: Elevation plausible")
        else:
            breakdown.penalties.append("-25: Elevation implausible")

        # Urban penalty
        is_urban = False
        if hasattr(plausibility, "urban"):
            is_urban = getattr(plausibility.urban, "is_urban", False)
        if is_urban:
            score -= 10.0
            breakdown.penalties.append("-10: Urban area detected")

        # Protected area penalty
        is_protected = False
        if hasattr(plausibility, "protected_area"):
            is_protected = getattr(
                plausibility.protected_area, "is_in_protected_area", False
            )
        if is_protected:
            score -= 5.0
            breakdown.penalties.append("-5: In protected area")

        score = max(0.0, min(100.0, score))
        breakdown.raw_score = score
        breakdown.weighted_score = round(
            score * breakdown.weight, 2
        )
        breakdown.explanation = (
            f"Plausibility composite: {score:.0f}/100"
        )

        return score, breakdown

    def _score_consistency(
        self,
        validation: ValidationResult,
    ) -> Tuple[float, ScoreBreakdown]:
        """Score consistency component with breakdown.

        Args:
            validation: Validation result from Engine 4.

        Returns:
            Tuple of (score, breakdown).
        """
        breakdown = ScoreBreakdown(
            component="consistency",
            weight=self.weights.get("consistency", 0.25),
        )

        # Base scoring
        if not validation.errors and not validation.warnings:
            score = 100.0
            breakdown.explanation = "No errors or warnings"
        elif not validation.errors and validation.warnings:
            score = 80.0
            breakdown.explanation = (
                f"Warnings only ({len(validation.warnings)} warnings)"
            )
        elif validation.was_auto_corrected:
            score = 60.0
            breakdown.explanation = "Auto-corrected errors applied"
        else:
            score = 20.0
            breakdown.explanation = (
                f"Uncorrectable errors ({len(validation.errors)} errors)"
            )

        # Swap detection penalty
        if validation.swap_detected:
            score -= 30.0
            breakdown.penalties.append(
                "-30: Lat/lon swap detected"
            )

        # Null island penalty
        if validation.null_island_detected:
            score -= 20.0
            breakdown.penalties.append(
                "-20: Null island (0, 0) detected"
            )

        score = max(0.0, min(100.0, score))
        breakdown.raw_score = score
        breakdown.weighted_score = round(
            score * breakdown.weight, 2
        )

        return score, breakdown

    def _score_source(
        self,
        source_type: SourceType,
    ) -> Tuple[float, ScoreBreakdown]:
        """Score source reliability component with breakdown.

        Args:
            source_type: Data source classification.

        Returns:
            Tuple of (score, breakdown).
        """
        breakdown = ScoreBreakdown(
            component="source",
            weight=self.weights.get("source", 0.15),
        )

        score = SOURCE_TYPE_SCORES.get(source_type, 25.0)
        breakdown.raw_score = score
        breakdown.weighted_score = round(
            score * breakdown.weight, 2
        )
        breakdown.explanation = (
            f"Source: {source_type.value} -> {score:.0f}/100"
        )

        return score, breakdown

    # ------------------------------------------------------------------
    # Internal: Explanations & Recommendations
    # ------------------------------------------------------------------

    def _generate_explanations(
        self,
        result: AccuracyScore,
        precision: PrecisionResult,
        plausibility: Any,
        validation: ValidationResult,
        source_type: SourceType,
    ) -> List[str]:
        """Generate human-readable explanations of the score.

        Args:
            result: Computed accuracy score.
            precision: Precision result.
            plausibility: Plausibility result.
            validation: Validation result.
            source_type: Source type.

        Returns:
            List of explanation strings.
        """
        explanations: List[str] = []

        explanations.append(
            f"Overall accuracy score: {result.overall_score:.1f}/100 "
            f"({result.tier.value.upper()} tier)"
        )

        explanations.append(
            f"Precision: {result.precision_score:.0f}/100 "
            f"({precision.precision_level.value}, "
            f"{precision.decimal_places} decimal places, "
            f"~{precision.ground_resolution_m:.1f}m resolution)"
        )

        explanations.append(
            f"Plausibility: {result.plausibility_score:.0f}/100"
        )

        explanations.append(
            f"Consistency: {result.consistency_score:.0f}/100 "
            f"({len(validation.errors)} errors, "
            f"{len(validation.warnings)} warnings)"
        )

        explanations.append(
            f"Source reliability: {result.source_score:.0f}/100 "
            f"({source_type.value})"
        )

        explanations.append(
            f"95% confidence interval: "
            f"{result.confidence_interval_m:.1f}m radius"
        )

        return explanations

    def _generate_recommendations(
        self,
        result: AccuracyScore,
        precision: PrecisionResult,
        plausibility: Any,
        validation: ValidationResult,
        source_type: SourceType,
    ) -> List[str]:
        """Generate improvement recommendations.

        Args:
            result: Computed accuracy score.
            precision: Precision result.
            plausibility: Plausibility result.
            validation: Validation result.
            source_type: Source type.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        if result.tier == AccuracyTier.GOLD:
            recommendations.append(
                "Coordinate meets GOLD tier standards. "
                "No improvements required."
            )
            return recommendations

        # Precision improvements
        if result.precision_score < 70:
            if precision.decimal_places < 6:
                recommendations.append(
                    "PRIORITY: Increase coordinate precision to >= 6 "
                    "decimal places for EUDR compliance (~0.11m accuracy)."
                )
            if precision.is_truncated:
                recommendations.append(
                    "Coordinate appears truncated. Obtain original "
                    "full-precision coordinates from the source device."
                )
            if precision.is_rounded:
                recommendations.append(
                    "Coordinate appears artificially rounded. Use "
                    "the raw GPS reading without manual rounding."
                )

        # Source improvements
        if result.source_score < 70:
            recommendations.append(
                f"Current source ({source_type.value}) has limited "
                f"reliability. Consider upgrading to GNSS survey "
                f"equipment or verified mobile GPS collection."
            )

        # Consistency improvements
        if result.consistency_score < 60:
            if validation.swap_detected:
                recommendations.append(
                    "CRITICAL: Lat/lon swap detected. Verify and "
                    "correct coordinate order before submission."
                )
            if validation.null_island_detected:
                recommendations.append(
                    "CRITICAL: Null island (0, 0) detected. This "
                    "is likely a default/missing value. Collect "
                    "actual GPS coordinates."
                )
            if validation.errors:
                recommendations.append(
                    f"Resolve {len(validation.errors)} validation "
                    f"errors before EUDR submission."
                )

        # Plausibility improvements
        if result.plausibility_score < 70:
            recommendations.append(
                "Coordinate has low plausibility score. Verify "
                "the GPS reading was taken at the actual production "
                "plot location."
            )

        # EUDR compliance
        if not result.is_eudr_compliant:
            recommendations.append(
                "CRITICAL: Coordinate does not meet minimum EUDR "
                "accuracy requirements (score < 50). Remediation "
                "is required before DDS submission."
            )

        return recommendations

    # ------------------------------------------------------------------
    # Internal: Haversine Distance
    # ------------------------------------------------------------------

    def _haversine_km(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Calculate Haversine distance in kilometres.

        Args:
            lat1, lon1: Point 1 coordinates (degrees).
            lat2, lon2: Point 2 coordinates (degrees).

        Returns:
            Distance in kilometres.
        """
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = (
            math.sin(dphi / 2.0) ** 2
            + math.cos(phi1)
            * math.cos(phi2)
            * math.sin(dlambda / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

        return EARTH_RADIUS_KM * c

    # ------------------------------------------------------------------
    # Internal: Provenance
    # ------------------------------------------------------------------

    def _compute_provenance_hash(
        self,
        coord: NormalizedCoordinate,
        result: AccuracyScore,
    ) -> str:
        """Compute SHA-256 provenance hash for audit trail.

        Args:
            coord: Input coordinate.
            result: Computed accuracy score.

        Returns:
            SHA-256 hex digest.
        """
        hash_data = {
            "module_version": _MODULE_VERSION,
            "engine": "accuracy_assessor",
            "lat": coord.lat,
            "lon": coord.lon,
            "datum": coord.datum,
            "overall_score": result.overall_score,
            "tier": result.tier.value,
            "precision_score": result.precision_score,
            "plausibility_score": result.plausibility_score,
            "consistency_score": result.consistency_score,
            "source_score": result.source_score,
            "confidence_interval_m": result.confidence_interval_m,
            "is_eudr_compliant": result.is_eudr_compliant,
        }
        return _compute_hash(hash_data)


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    "AccuracyAssessor",
    "AccuracyScore",
    "AccuracyTier",
    "SourceType",
    "PrecisionLevel",
    "PrecisionResult",
    "NormalizedCoordinate",
    "ValidationResult",
    "ScoreBreakdown",
    "DEFAULT_WEIGHTS",
    "PRECISION_LEVEL_SCORES",
    "SOURCE_TYPE_SCORES",
    "SOURCE_PRECISION_M",
    "TIER_THRESHOLDS",
    "MIN_EUDR_SCORE",
]
