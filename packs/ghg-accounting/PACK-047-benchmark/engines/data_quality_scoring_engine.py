# -*- coding: utf-8 -*-
"""
DataQualityScoringEngine - PACK-047 GHG Emissions Benchmark Engine 8
====================================================================

Assesses and scores data quality for GHG benchmark inputs using the
GHG Protocol 5x5 quality matrix, PCAF 1-5 scoring, source hierarchy,
confidence interval calculation, coverage analysis, and quality-weighted
benchmark aggregation.

Calculation Methodology:
    GHG Protocol Quality Matrix (5 dimensions, 1-5 per dimension):
        Temporal:       How recent is the data?
        Geographic:     How geographically representative?
        Technological:  How technology-specific?
        Completeness:   How complete is the coverage?
        Reliability:    How reliable is the source?

    Composite Quality Score:
        Q = SUM(w_dim * score_dim) / SUM(w_dim)

        Default weights: temporal=0.25, geographic=0.20, technological=0.20,
                         completeness=0.20, reliability=0.15

    PCAF Data Quality Mapping:
        Score 1: Verified reported emissions (best)
        Score 2: Reported emissions (unverified)
        Score 3: Physical activity-based estimate
        Score 4: Economic activity-based estimate
        Score 5: Sector average estimate (worst)

    Source Hierarchy:
        verified_third_party > reported_unverified > estimated_activity >
        modelled_economic > sector_average

    Confidence Interval:
        CI_95 = result +/- Z_0.975 * sigma_quality

        Where sigma_quality = f(composite_score):
            Q=1: sigma = 0.05 (5%)
            Q=2: sigma = 0.10 (10%)
            Q=3: sigma = 0.20 (20%)
            Q=4: sigma = 0.35 (35%)
            Q=5: sigma = 0.50 (50%)

    Quality-Weighted Mean:
        mu_qw = SUM(Q_i * v_i) / SUM(Q_i)

        Where:
            Q_i = quality score for entity i (inverted: 5-Q+1 so higher=better)
            v_i = value for entity i

    Coverage Score:
        coverage = data_points_present / data_points_expected * 100

Regulatory References:
    - GHG Protocol Corporate Standard: Chapter 7 (Managing Inventory Quality)
    - IPCC 2006 Guidelines Vol 1 Ch 3: Uncertainties
    - PCAF Global GHG Accounting Standard: Data quality scoring
    - ESRS E1: Data quality disclosure requirements
    - CDP Climate Change C5-C7: Data quality indicators
    - ISO 14064-1:2018 Clause 9: Uncertainty assessment

Zero-Hallucination:
    - All quality matrices from published regulatory standards
    - All calculations use deterministic Decimal arithmetic
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-047 GHG Emissions Benchmark
Engine:  8 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0"),
) -> Decimal:
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _round2(value: Any) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _round4(value: Any) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class QualityDimension(str, Enum):
    """GHG Protocol data quality dimension."""
    TEMPORAL = "temporal"
    GEOGRAPHIC = "geographic"
    TECHNOLOGICAL = "technological"
    COMPLETENESS = "completeness"
    RELIABILITY = "reliability"


class SourceTier(str, Enum):
    """Data source hierarchy tier.

    VERIFIED:       Third-party verified/assured.
    REPORTED:       Self-reported (unverified).
    ESTIMATED:      Activity-based estimate.
    MODELLED:       Economic model estimate.
    SECTOR_AVG:     Sector average.
    """
    VERIFIED = "verified"
    REPORTED = "reported"
    ESTIMATED = "estimated"
    MODELLED = "modelled"
    SECTOR_AVG = "sector_average"


class QualityRating(str, Enum):
    """Overall quality rating."""
    HIGH = "high"
    GOOD = "good"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default dimension weights
DEFAULT_DIMENSION_WEIGHTS: Dict[str, Decimal] = {
    QualityDimension.TEMPORAL.value: Decimal("0.25"),
    QualityDimension.GEOGRAPHIC.value: Decimal("0.20"),
    QualityDimension.TECHNOLOGICAL.value: Decimal("0.20"),
    QualityDimension.COMPLETENESS.value: Decimal("0.20"),
    QualityDimension.RELIABILITY.value: Decimal("0.15"),
}

# PCAF score -> GHG Protocol composite mapping
PCAF_TO_COMPOSITE: Dict[int, Decimal] = {
    1: Decimal("1.0"),
    2: Decimal("2.0"),
    3: Decimal("3.0"),
    4: Decimal("4.0"),
    5: Decimal("5.0"),
}

# Source tier -> PCAF score mapping
SOURCE_TIER_TO_PCAF: Dict[str, int] = {
    SourceTier.VERIFIED.value: 1,
    SourceTier.REPORTED.value: 2,
    SourceTier.ESTIMATED.value: 3,
    SourceTier.MODELLED.value: 4,
    SourceTier.SECTOR_AVG.value: 5,
}

# Quality score -> uncertainty (relative, as fraction)
QUALITY_UNCERTAINTY: Dict[int, Decimal] = {
    1: Decimal("0.05"),
    2: Decimal("0.10"),
    3: Decimal("0.20"),
    4: Decimal("0.35"),
    5: Decimal("0.50"),
}

# Z-score for 95% CI
Z_95: Decimal = Decimal("1.96")

# Temporal scoring: years old -> score
TEMPORAL_SCORING: Dict[int, int] = {
    0: 1,  # Current year
    1: 1,  # 1 year old
    2: 2,  # 2 years old
    3: 3,
    4: 4,
    5: 5,  # 5+ years old
}


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class DimensionScore(BaseModel):
    """Score for a single quality dimension.

    Attributes:
        dimension:      Quality dimension.
        score:          Score (1-5, 1=best).
        justification:  Explanation for the score.
    """
    dimension: QualityDimension = Field(..., description="Dimension")
    score: int = Field(..., ge=1, le=5, description="Score (1-5)")
    justification: str = Field(default="", description="Justification")


class QualityAssessmentInput(BaseModel):
    """Input for data quality assessment.

    Attributes:
        entity_id:          Entity identifier.
        entity_name:        Entity name.
        reporting_year:     Reporting year.
        reference_year:     Reference year for temporal scoring.
        source_tier:        Data source tier.
        dimension_scores:   Pre-assigned dimension scores (optional).
        pcaf_score:         Pre-assigned PCAF score (optional).
        value:              The data value being assessed.
        scope_coverage:     Scope coverage description.
        geographic_match:   Geographic representativeness (0-1).
        technology_match:   Technology representativeness (0-1).
        completeness_pct:   Data completeness (0-100).
        dimension_weights:  Custom dimension weights (optional).
    """
    entity_id: str = Field(default="", description="Entity ID")
    entity_name: str = Field(default="", description="Entity name")
    reporting_year: int = Field(default=2024)
    reference_year: int = Field(default=2024)
    source_tier: SourceTier = Field(default=SourceTier.REPORTED)
    dimension_scores: Optional[List[DimensionScore]] = Field(default=None)
    pcaf_score: Optional[int] = Field(default=None, ge=1, le=5)
    value: Decimal = Field(default=Decimal("0"), ge=0)
    scope_coverage: str = Field(default="s1_s2")
    geographic_match: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    technology_match: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    completeness_pct: Decimal = Field(default=Decimal("50"), ge=0, le=100)
    dimension_weights: Optional[Dict[str, Decimal]] = Field(default=None)

    @field_validator("value", mode="before")
    @classmethod
    def coerce_val(cls, v: Any) -> Decimal:
        return _decimal(v)


class BatchQualityInput(BaseModel):
    """Batch quality assessment input.

    Attributes:
        assessments:        List of quality assessments.
        output_precision:   Output decimal places.
    """
    assessments: List[QualityAssessmentInput] = Field(default_factory=list)
    output_precision: int = Field(default=2, ge=0, le=6)


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class QualityScore(BaseModel):
    """Quality score result for a single entity.

    Attributes:
        entity_id:          Entity identifier.
        entity_name:        Entity name.
        dimension_scores:   Per-dimension scores.
        composite_score:    Weighted composite (1-5).
        pcaf_score:         PCAF data quality score.
        source_tier:        Data source tier.
        rating:             Quality rating.
        uncertainty_pct:    Uncertainty as percentage.
        ci_lower:           95% CI lower bound.
        ci_upper:           95% CI upper bound.
    """
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    dimension_scores: List[DimensionScore] = Field(default_factory=list)
    composite_score: Decimal = Field(default=Decimal("0"))
    pcaf_score: int = Field(default=3)
    source_tier: str = Field(default="")
    rating: QualityRating = Field(default=QualityRating.MODERATE)
    uncertainty_pct: Decimal = Field(default=Decimal("0"))
    ci_lower: Decimal = Field(default=Decimal("0"))
    ci_upper: Decimal = Field(default=Decimal("0"))


class CoverageAnalysis(BaseModel):
    """Data coverage analysis.

    Attributes:
        total_entities:         Total entities assessed.
        entities_with_data:     Entities with data present.
        coverage_pct:           Coverage percentage.
        by_source_tier:         Count by source tier.
        by_pcaf_score:          Count by PCAF score.
        avg_composite_score:    Average composite quality score.
        data_gaps:              List of identified data gaps.
    """
    total_entities: int = Field(default=0)
    entities_with_data: int = Field(default=0)
    coverage_pct: Decimal = Field(default=Decimal("0"))
    by_source_tier: Dict[str, int] = Field(default_factory=dict)
    by_pcaf_score: Dict[int, int] = Field(default_factory=dict)
    avg_composite_score: Decimal = Field(default=Decimal("0"))
    data_gaps: List[str] = Field(default_factory=list)


class QualityWeightedResult(BaseModel):
    """Quality-weighted aggregation result.

    Attributes:
        quality_weighted_mean:  Quality-weighted mean value.
        simple_mean:            Simple arithmetic mean.
        difference:             Difference between weighted and simple.
        total_entities:         Entities included.
    """
    quality_weighted_mean: Decimal = Field(default=Decimal("0"))
    simple_mean: Decimal = Field(default=Decimal("0"))
    difference: Decimal = Field(default=Decimal("0"))
    total_entities: int = Field(default=0)


class DataQualityResult(BaseModel):
    """Complete data quality assessment result.

    Attributes:
        result_id:              Unique result ID.
        entity_scores:          Per-entity quality scores.
        coverage:               Coverage analysis.
        quality_weighted:       Quality-weighted aggregation.
        overall_quality:        Overall quality rating.
        overall_composite:      Overall composite score.
        warnings:               Warnings.
        calculated_at:          Timestamp.
        processing_time_ms:     Processing time (ms).
        provenance_hash:        SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    entity_scores: List[QualityScore] = Field(default_factory=list)
    coverage: CoverageAnalysis = Field(default_factory=CoverageAnalysis)
    quality_weighted: Optional[QualityWeightedResult] = Field(default=None)
    overall_quality: QualityRating = Field(default=QualityRating.MODERATE)
    overall_composite: Decimal = Field(default=Decimal("0"))
    warnings: List[str] = Field(default_factory=list)
    calculated_at: str = Field(default="", description="Timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class DataQualityScoringEngine:
    """Assesses data quality for GHG benchmark inputs.

    Uses GHG Protocol 5x5 quality matrix, PCAF scoring, source hierarchy,
    confidence intervals, and quality-weighted aggregation.

    Guarantees:
        - Deterministic: Same inputs always produce the same output.
        - Reproducible: Full provenance tracking with SHA-256 hashes.
        - Auditable: Every dimension score documented.
        - Zero-Hallucination: No LLM in any calculation path.
    """

    def __init__(self) -> None:
        self._version = _MODULE_VERSION
        logger.info("DataQualityScoringEngine v%s initialised", self._version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, input_data: BatchQualityInput) -> DataQualityResult:
        """Assess data quality for a batch of entities.

        Args:
            input_data: Batch quality assessment input.

        Returns:
            DataQualityResult with scores, coverage, and weighted results.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []
        prec = input_data.output_precision
        prec_str = "0." + "0" * prec

        entity_scores: List[QualityScore] = []
        composites: List[Decimal] = []
        values: List[Tuple[Decimal, Decimal]] = []  # (quality_weight, value)

        by_source: Dict[str, int] = {}
        by_pcaf: Dict[int, int] = {}
        with_data = 0

        for assessment in input_data.assessments:
            score = self._assess_entity(assessment, prec_str)
            entity_scores.append(score)
            composites.append(score.composite_score)

            tier = score.source_tier
            by_source[tier] = by_source.get(tier, 0) + 1
            by_pcaf[score.pcaf_score] = by_pcaf.get(score.pcaf_score, 0) + 1

            if assessment.value > Decimal("0"):
                with_data += 1
                # Quality weight: invert PCAF (5-Q+1), so Q=1 -> weight=5
                qw = Decimal(str(6 - score.pcaf_score))
                values.append((qw, assessment.value))

        total = len(input_data.assessments)
        coverage_pct = _safe_divide(
            Decimal(str(with_data)), Decimal(str(total))
        ) * Decimal("100") if total > 0 else Decimal("0")

        avg_composite = Decimal("0")
        if composites:
            avg_composite = (
                sum(composites) / Decimal(str(len(composites)))
            ).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

        # Data gaps
        gaps: List[str] = []
        if with_data < total:
            gaps.append(f"{total - with_data} entities missing emissions data.")
        if by_pcaf.get(5, 0) > total * 0.3:
            gaps.append("Over 30% of data uses sector averages (PCAF 5).")
        if by_pcaf.get(4, 0) + by_pcaf.get(5, 0) > total * 0.5:
            gaps.append("Over 50% of data is modelled or sector average (PCAF 4-5).")

        coverage = CoverageAnalysis(
            total_entities=total,
            entities_with_data=with_data,
            coverage_pct=coverage_pct.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            by_source_tier=by_source,
            by_pcaf_score=by_pcaf,
            avg_composite_score=avg_composite,
            data_gaps=gaps,
        )

        # Quality-weighted mean
        qw_result: Optional[QualityWeightedResult] = None
        if values:
            total_qw = sum(qw for qw, _ in values)
            qw_mean = _safe_divide(
                sum(qw * v for qw, v in values), total_qw
            ).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
            simple_mean = (
                sum(v for _, v in values) / Decimal(str(len(values)))
            ).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)
            diff = (qw_mean - simple_mean).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

            qw_result = QualityWeightedResult(
                quality_weighted_mean=qw_mean,
                simple_mean=simple_mean,
                difference=diff,
                total_entities=len(values),
            )

        overall_rating = self._classify_quality(avg_composite)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = DataQualityResult(
            entity_scores=entity_scores,
            coverage=coverage,
            quality_weighted=qw_result,
            overall_quality=overall_rating,
            overall_composite=avg_composite,
            warnings=warnings,
            calculated_at=_utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def score_entity(self, assessment: QualityAssessmentInput) -> QualityScore:
        """Score a single entity.

        Args:
            assessment: Quality assessment input.

        Returns:
            QualityScore for the entity.
        """
        return self._assess_entity(assessment, "0.01")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _assess_entity(
        self, assessment: QualityAssessmentInput, prec_str: str,
    ) -> QualityScore:
        """Assess quality for a single entity."""
        # Build dimension scores
        if assessment.dimension_scores:
            dim_scores = assessment.dimension_scores
        else:
            dim_scores = self._auto_score_dimensions(assessment)

        # Composite: Q = SUM(w * score) / SUM(w)
        weights = assessment.dimension_weights or DEFAULT_DIMENSION_WEIGHTS
        numerator = Decimal("0")
        denominator = Decimal("0")

        for ds in dim_scores:
            w = weights.get(ds.dimension.value, Decimal("0.20"))
            numerator += w * Decimal(str(ds.score))
            denominator += w

        composite = _safe_divide(numerator, denominator, Decimal("3"))
        composite = composite.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

        # PCAF score
        pcaf = assessment.pcaf_score
        if pcaf is None:
            pcaf = SOURCE_TIER_TO_PCAF.get(assessment.source_tier.value, 3)

        # Uncertainty and CI
        pcaf_clamped = max(1, min(5, int(float(composite) + Decimal("0.5"))))
        uncertainty = QUALITY_UNCERTAINTY.get(pcaf_clamped, Decimal("0.20"))
        uncertainty_pct = (uncertainty * Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        ci_half = assessment.value * uncertainty * Z_95
        ci_lower = max(assessment.value - ci_half, Decimal("0")).quantize(
            Decimal(prec_str), rounding=ROUND_HALF_UP
        )
        ci_upper = (assessment.value + ci_half).quantize(
            Decimal(prec_str), rounding=ROUND_HALF_UP
        )

        rating = self._classify_quality(composite)

        return QualityScore(
            entity_id=assessment.entity_id,
            entity_name=assessment.entity_name,
            dimension_scores=dim_scores,
            composite_score=composite,
            pcaf_score=pcaf,
            source_tier=assessment.source_tier.value,
            rating=rating,
            uncertainty_pct=uncertainty_pct,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )

    def _auto_score_dimensions(
        self, assessment: QualityAssessmentInput,
    ) -> List[DimensionScore]:
        """Auto-generate dimension scores from assessment inputs."""
        scores: List[DimensionScore] = []

        # Temporal
        age = assessment.reference_year - assessment.reporting_year
        temporal = TEMPORAL_SCORING.get(min(age, 5), 5)
        scores.append(DimensionScore(
            dimension=QualityDimension.TEMPORAL,
            score=temporal,
            justification=f"Data is {age} year(s) old.",
        ))

        # Geographic
        geo = self._match_to_score(assessment.geographic_match)
        scores.append(DimensionScore(
            dimension=QualityDimension.GEOGRAPHIC,
            score=geo,
            justification=f"Geographic match: {float(assessment.geographic_match):.0%}.",
        ))

        # Technological
        tech = self._match_to_score(assessment.technology_match)
        scores.append(DimensionScore(
            dimension=QualityDimension.TECHNOLOGICAL,
            score=tech,
            justification=f"Technology match: {float(assessment.technology_match):.0%}.",
        ))

        # Completeness
        comp_pct = float(assessment.completeness_pct)
        if comp_pct >= 90:
            comp = 1
        elif comp_pct >= 70:
            comp = 2
        elif comp_pct >= 50:
            comp = 3
        elif comp_pct >= 30:
            comp = 4
        else:
            comp = 5
        scores.append(DimensionScore(
            dimension=QualityDimension.COMPLETENESS,
            score=comp,
            justification=f"Completeness: {comp_pct:.0f}%.",
        ))

        # Reliability
        reliability = SOURCE_TIER_TO_PCAF.get(assessment.source_tier.value, 3)
        scores.append(DimensionScore(
            dimension=QualityDimension.RELIABILITY,
            score=reliability,
            justification=f"Source: {assessment.source_tier.value}.",
        ))

        return scores

    def _match_to_score(self, match_value: Decimal) -> int:
        """Convert match fraction (0-1) to quality score (1-5)."""
        v = float(match_value)
        if v >= 0.9:
            return 1
        if v >= 0.7:
            return 2
        if v >= 0.5:
            return 3
        if v >= 0.3:
            return 4
        return 5

    def _classify_quality(self, composite: Decimal) -> QualityRating:
        """Classify quality rating from composite score."""
        if composite <= Decimal("1.5"):
            return QualityRating.HIGH
        if composite <= Decimal("2.5"):
            return QualityRating.GOOD
        if composite <= Decimal("3.5"):
            return QualityRating.MODERATE
        if composite <= Decimal("4.5"):
            return QualityRating.LOW
        return QualityRating.VERY_LOW

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_version(self) -> str:
        return self._version


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "QualityDimension",
    "SourceTier",
    "QualityRating",
    # Input Models
    "DimensionScore",
    "QualityAssessmentInput",
    "BatchQualityInput",
    # Output Models
    "QualityScore",
    "CoverageAnalysis",
    "QualityWeightedResult",
    "DataQualityResult",
    # Engine
    "DataQualityScoringEngine",
    # Constants
    "DEFAULT_DIMENSION_WEIGHTS",
    "QUALITY_UNCERTAINTY",
]
