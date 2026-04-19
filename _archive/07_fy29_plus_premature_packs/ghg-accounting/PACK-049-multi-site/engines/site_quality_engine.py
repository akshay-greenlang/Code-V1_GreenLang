# -*- coding: utf-8 -*-
"""
PACK-049 GHG Multi-Site Management Pack - Site Quality Engine
==============================================================

Scores the data quality of GHG submissions across a multi-site
portfolio using six weighted dimensions, maps each site to a
PCAF-equivalent quality tier (1-5), generates a visual heatmap
matrix of quality by site and scope, produces remediation plans
for underperforming facilities, and tracks quality progression
over time.

Quality Dimensions (6):
    ACCURACY:       Proportion of data from direct measurement vs estimation
    COMPLETENESS:   Proportion of required data fields filled
    CONSISTENCY:    Year-on-year variance within acceptable bounds
    TIMELINESS:     Submission relative to deadline (early, on-time, late)
    METHODOLOGY:    Tier of emission factors used (facility-specific > IPCC default)
    DOCUMENTATION:  Proportion of data points with supporting evidence

Dimension Scoring (0-5 scale):
    Each dimension produces a score from 0 (worst) to 5 (best).

Overall Score:
    overall = SUM(dim_score_i * weight_i) for i in dimensions
    where SUM(weight_i) = 1.0

PCAF Mapping:
    Score >= 4.0  ->  PCAF 1 (Audited / Verified)
    Score >= 3.0  ->  PCAF 2 (Primary / Calculated)
    Score >= 2.0  ->  PCAF 3 (Modelled / Estimated)
    Score >= 1.0  ->  PCAF 4 (Proxy / Extrapolated)
    Score <  1.0  ->  PCAF 5 (Estimated with low confidence)

Heatmap Colours:
    GREEN:  score >= 4.0
    AMBER:  score >= 3.0
    RED:    score <  3.0

Corporate Quality Score:
    If weighted_by_emissions:
        corp_score = SUM(site_score_i * site_emissions_i) / SUM(site_emissions_i)
    else:
        corp_score = SUM(site_score_i) / n

Provenance:
    SHA-256 hash on every QualityResult.

Regulatory References:
    - GHG Protocol Corporate Standard (2004, rev 2015) Ch 7 - Managing Inventory Quality
    - ISO 14064-1:2018 Clause 9 - Quality management
    - PCAF Global GHG Accounting Standard v3 (2024) - Data quality scoring
    - EU CSRD / ESRS E1 - Data quality requirements
    - ISAE 3410 para 47 - Data quality for assurance

Zero-Hallucination:
    - All quality scores computed via deterministic Decimal arithmetic
    - Thresholds and mappings are constants, not LLM-generated
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-049 GHG Multi-Site Management
Engine:  9 of 10
Status:  Production Ready
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

_ZERO = Decimal("0")
_ONE = Decimal("1")
_TWO = Decimal("2")
_THREE = Decimal("3")
_FOUR = Decimal("4")
_FIVE = Decimal("5")
_HUNDRED = Decimal("100")
_DP6 = Decimal("0.000001")
_DP2 = Decimal("0.01")

def _safe_divide(numerator: Decimal, denominator: Decimal) -> Decimal:
    """Divide with zero-guard."""
    if denominator == _ZERO:
        return _ZERO
    return (numerator / denominator).quantize(_DP6, rounding=ROUND_HALF_UP)

def _quantise(value: Decimal, precision: Decimal = _DP6) -> Decimal:
    """Quantise a Decimal value."""
    return value.quantize(precision, rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Default Weights
# ---------------------------------------------------------------------------

DEFAULT_QUALITY_WEIGHTS: Dict[str, Decimal] = {
    "ACCURACY": Decimal("0.25"),
    "COMPLETENESS": Decimal("0.25"),
    "CONSISTENCY": Decimal("0.15"),
    "TIMELINESS": Decimal("0.15"),
    "METHODOLOGY": Decimal("0.10"),
    "DOCUMENTATION": Decimal("0.10"),
}

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class QualityDimension(str, Enum):
    """Quality assessment dimensions."""
    ACCURACY = "ACCURACY"
    COMPLETENESS = "COMPLETENESS"
    CONSISTENCY = "CONSISTENCY"
    TIMELINESS = "TIMELINESS"
    METHODOLOGY = "METHODOLOGY"
    DOCUMENTATION = "DOCUMENTATION"

class HeatmapColour(str, Enum):
    """Colour codes for quality heatmap."""
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"

class PCAFTier(int, Enum):
    """PCAF data quality tiers (1=best, 5=worst)."""
    TIER_1_VERIFIED = 1
    TIER_2_CALCULATED = 2
    TIER_3_ESTIMATED = 3
    TIER_4_EXTRAPOLATED = 4
    TIER_5_PROXY = 5

class MethodologyTier(str, Enum):
    """Emission factor methodology tiers."""
    FACILITY_SPECIFIC = "FACILITY_SPECIFIC"
    NATIONAL = "NATIONAL"
    REGIONAL = "REGIONAL"
    IPCC_DEFAULT = "IPCC_DEFAULT"
    UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class DataEntry(BaseModel):
    """A single data entry for quality assessment."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entry_id: str = Field(default_factory=_new_uuid, description="Entry identifier")
    scope: str = Field(..., description="Emission scope (SCOPE_1, SCOPE_2, SCOPE_3)")
    category: str = Field("", description="Emission category within scope")
    value: Decimal = Field(_ZERO, description="Reported value (tCO2e)")
    is_measured: bool = Field(False, description="Whether value is from direct measurement")
    is_estimated: bool = Field(False, description="Whether value is estimated")
    methodology_tier: str = Field(
        MethodologyTier.UNKNOWN.value,
        description="Tier of emission factor used",
    )
    has_evidence: bool = Field(False, description="Whether supporting evidence is attached")
    prior_year_value: Optional[Decimal] = Field(
        None, description="Prior year value for consistency check"
    )
    submitted_date: Optional[date] = Field(None, description="Date of submission")
    deadline_date: Optional[date] = Field(None, description="Collection deadline")
    required: bool = Field(True, description="Whether this entry was required")
    field_count: int = Field(1, ge=1, description="Number of data fields in this entry")
    filled_field_count: int = Field(1, ge=0, description="Number of fields actually filled")

class QualityConfig(BaseModel):
    """Quality assessment configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    weights: Dict[str, Decimal] = Field(
        default_factory=lambda: dict(DEFAULT_QUALITY_WEIGHTS),
        description="Dimension weights (must sum to 1.0)",
    )
    consistency_variance_threshold: Decimal = Field(
        Decimal("0.25"), ge=_ZERO, le=_ONE,
        description="Max acceptable YoY variance for consistency (0.25 = 25%)",
    )
    timeliness_grace_days: int = Field(
        3, ge=0, description="Grace period days after deadline for full timeliness score"
    )
    min_evidence_rate: Decimal = Field(
        Decimal("0.5"), ge=_ZERO, le=_ONE,
        description="Minimum evidence attachment rate for acceptable documentation score",
    )

class DimensionScore(BaseModel):
    """Score for a single quality dimension."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dimension: str = Field(..., description="Quality dimension name")
    score: Decimal = Field(
        _ZERO, ge=_ZERO, le=_FIVE,
        description="Dimension score (0-5)",
    )
    weight: Decimal = Field(
        _ZERO, ge=_ZERO, le=_ONE,
        description="Dimension weight in overall score",
    )
    weighted_score: Decimal = Field(
        _ZERO, description="score * weight"
    )
    evidence: str = Field("", description="Description of scoring rationale")
    recommendations: List[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )

class SiteQualityAssessment(BaseModel):
    """Quality assessment for a single site."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    site_id: str = Field(..., description="Site identifier")
    period: str = Field("", description="Reporting period")
    dimension_scores: List[DimensionScore] = Field(
        default_factory=list, description="Scores per dimension"
    )
    overall_score: Decimal = Field(
        _ZERO, ge=_ZERO, le=_FIVE,
        description="Weighted overall quality score (0-5)",
    )
    pcaf_equivalent: int = Field(
        3, ge=1, le=5, description="PCAF equivalent tier (1=best, 5=worst)"
    )
    tier_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of data entries per methodology tier",
    )
    estimated_pct: Decimal = Field(
        _ZERO, ge=_ZERO, le=_HUNDRED,
        description="Percentage of data that is estimated",
    )
    improvement_actions: List[str] = Field(
        default_factory=list, description="Prioritised improvement actions"
    )

class QualityHeatmapCell(BaseModel):
    """A single cell in the quality heatmap matrix."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    site_id: str = Field(..., description="Site identifier")
    scope: str = Field(..., description="Scope identifier")
    score: Decimal = Field(
        _ZERO, ge=_ZERO, le=_FIVE,
        description="Quality score for this site/scope combination",
    )
    colour_code: str = Field(
        HeatmapColour.RED.value,
        description="Heatmap colour (GREEN, AMBER, RED)",
    )

class QualityResult(BaseModel):
    """Portfolio-wide quality assessment result."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    result_id: str = Field(default_factory=_new_uuid, description="Result identifier")
    assessments: List[SiteQualityAssessment] = Field(
        default_factory=list, description="Per-site quality assessments"
    )
    corporate_quality_score: Decimal = Field(
        _ZERO, ge=_ZERO, le=_FIVE,
        description="Corporate-level aggregate quality score",
    )
    weighted_by_emissions: bool = Field(
        False, description="Whether corporate score is emission-weighted"
    )
    heatmap: List[QualityHeatmapCell] = Field(
        default_factory=list, description="Quality heatmap cells"
    )
    improvement_priorities: List[Dict[str, Any]] = Field(
        default_factory=list, description="Prioritised portfolio improvement actions"
    )
    created_at: datetime = Field(default_factory=utcnow, description="Assessment timestamp")
    provenance_hash: str = Field("", description="SHA-256 provenance hash")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.result_id}|{self.corporate_quality_score}|"
                f"{len(self.assessments)}|{len(self.heatmap)}|"
                f"{self.weighted_by_emissions}"
            )
            for a in self.assessments:
                payload += f"|{a.site_id}={a.overall_score}"
            self.provenance_hash = _compute_hash(payload)

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SiteQualityEngine:
    """
    Scores data quality across a multi-site GHG portfolio.

    Provides:
        - Six-dimension quality scoring (accuracy, completeness,
          consistency, timeliness, methodology, documentation)
        - PCAF tier mapping
        - Portfolio quality heatmap generation
        - Emission-weighted corporate quality score
        - Remediation plan generation
        - Quality progression tracking

    All calculations use Decimal arithmetic.  Every result carries a
    SHA-256 provenance hash.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        quality_config: Optional[QualityConfig] = None,
        rounding_precision: Decimal = _DP6,
    ) -> None:
        """
        Initialise the SiteQualityEngine.

        Args:
            quality_config: Quality assessment configuration. Defaults to standard config.
            rounding_precision: Decimal quantisation precision.
        """
        self._config = quality_config or QualityConfig()
        self._precision = rounding_precision
        self._weights = self._config.weights
        logger.info(
            "SiteQualityEngine v%s initialised (weights=%s)",
            _MODULE_VERSION,
            {k: str(v) for k, v in self._weights.items()},
        )

    # ----------------------------------------------- site assessment
    def assess_site_quality(
        self,
        site_id: str,
        period: str,
        data_entries: List[DataEntry],
        quality_config: Optional[QualityConfig] = None,
    ) -> SiteQualityAssessment:
        """
        Assess data quality for a single site.

        Steps:
            1. Calculate dimension scores from data entries
            2. Compute weighted overall score
            3. Map to PCAF equivalent tier
            4. Calculate tier distribution and estimation percentage
            5. Generate improvement actions

        Args:
            site_id: Site identifier.
            period: Reporting period label.
            data_entries: List of data entries to assess.
            quality_config: Override quality config for this assessment.

        Returns:
            SiteQualityAssessment with dimension scores and overall.

        Raises:
            ValueError: If data_entries is empty.
        """
        logger.info(
            "Assessing site quality: site=%s period=%s entries=%d",
            site_id, period, len(data_entries),
        )
        if not data_entries:
            raise ValueError("data_entries must not be empty for quality assessment")

        config = quality_config or self._config
        weights = config.weights

        # Calculate dimension scores
        dimension_scores = self.calculate_dimension_scores(data_entries, weights, config)

        # Overall weighted score
        overall = sum(ds.weighted_score for ds in dimension_scores)
        overall = _quantise(min(overall, _FIVE), self._precision)

        # PCAF equivalent
        pcaf_tier = self.calculate_pcaf_equivalent(overall)

        # Tier distribution
        tier_dist: Dict[str, int] = {}
        for entry in data_entries:
            tier = entry.methodology_tier
            tier_dist[tier] = tier_dist.get(tier, 0) + 1

        # Estimation percentage
        total_entries = len(data_entries)
        estimated_count = sum(1 for e in data_entries if e.is_estimated)
        estimated_pct = _quantise(
            Decimal(str(estimated_count)) / Decimal(str(total_entries)) * _HUNDRED,
            self._precision,
        )

        # Improvement actions
        improvement_actions = self._generate_improvement_actions(dimension_scores)

        assessment = SiteQualityAssessment(
            site_id=site_id,
            period=period,
            dimension_scores=dimension_scores,
            overall_score=overall,
            pcaf_equivalent=pcaf_tier,
            tier_distribution=tier_dist,
            estimated_pct=estimated_pct,
            improvement_actions=improvement_actions,
        )

        logger.info(
            "Site quality assessed: site=%s overall=%s pcaf=%d estimated=%s%%",
            site_id, overall, pcaf_tier, estimated_pct,
        )
        return assessment

    # ----------------------------------------------- dimension scoring
    def calculate_dimension_scores(
        self,
        data_entries: List[DataEntry],
        weights: Dict[str, Decimal],
        config: Optional[QualityConfig] = None,
    ) -> List[DimensionScore]:
        """
        Calculate scores for all six quality dimensions.

        Args:
            data_entries: List of data entries.
            weights: Dimension weights.
            config: Quality configuration.

        Returns:
            List of DimensionScore, one per dimension.
        """
        config = config or self._config
        scores: List[DimensionScore] = []

        # ACCURACY: based on % direct measurement vs estimation
        accuracy = self._score_accuracy(data_entries)
        w = weights.get(QualityDimension.ACCURACY.value, Decimal("0.25"))
        scores.append(DimensionScore(
            dimension=QualityDimension.ACCURACY.value,
            score=accuracy["score"],
            weight=w,
            weighted_score=_quantise(accuracy["score"] * w, self._precision),
            evidence=accuracy["evidence"],
            recommendations=accuracy["recommendations"],
        ))

        # COMPLETENESS: based on % fields filled / required
        completeness = self._score_completeness(data_entries)
        w = weights.get(QualityDimension.COMPLETENESS.value, Decimal("0.25"))
        scores.append(DimensionScore(
            dimension=QualityDimension.COMPLETENESS.value,
            score=completeness["score"],
            weight=w,
            weighted_score=_quantise(completeness["score"] * w, self._precision),
            evidence=completeness["evidence"],
            recommendations=completeness["recommendations"],
        ))

        # CONSISTENCY: YoY variance check
        consistency = self._score_consistency(data_entries, config)
        w = weights.get(QualityDimension.CONSISTENCY.value, Decimal("0.15"))
        scores.append(DimensionScore(
            dimension=QualityDimension.CONSISTENCY.value,
            score=consistency["score"],
            weight=w,
            weighted_score=_quantise(consistency["score"] * w, self._precision),
            evidence=consistency["evidence"],
            recommendations=consistency["recommendations"],
        ))

        # TIMELINESS: submission vs deadline
        timeliness = self._score_timeliness(data_entries, config)
        w = weights.get(QualityDimension.TIMELINESS.value, Decimal("0.15"))
        scores.append(DimensionScore(
            dimension=QualityDimension.TIMELINESS.value,
            score=timeliness["score"],
            weight=w,
            weighted_score=_quantise(timeliness["score"] * w, self._precision),
            evidence=timeliness["evidence"],
            recommendations=timeliness["recommendations"],
        ))

        # METHODOLOGY: tier of emission factors used
        methodology = self._score_methodology(data_entries)
        w = weights.get(QualityDimension.METHODOLOGY.value, Decimal("0.10"))
        scores.append(DimensionScore(
            dimension=QualityDimension.METHODOLOGY.value,
            score=methodology["score"],
            weight=w,
            weighted_score=_quantise(methodology["score"] * w, self._precision),
            evidence=methodology["evidence"],
            recommendations=methodology["recommendations"],
        ))

        # DOCUMENTATION: evidence attachment rate
        documentation = self._score_documentation(data_entries, config)
        w = weights.get(QualityDimension.DOCUMENTATION.value, Decimal("0.10"))
        scores.append(DimensionScore(
            dimension=QualityDimension.DOCUMENTATION.value,
            score=documentation["score"],
            weight=w,
            weighted_score=_quantise(documentation["score"] * w, self._precision),
            evidence=documentation["evidence"],
            recommendations=documentation["recommendations"],
        ))

        return scores

    # ----------------------------------------------- PCAF mapping
    def calculate_pcaf_equivalent(self, overall_score: Decimal) -> int:
        """
        Map an overall quality score (0-5) to a PCAF tier (1-5).

        Mapping:
            Score >= 4.0 -> PCAF 1 (Verified)
            Score >= 3.0 -> PCAF 2 (Calculated)
            Score >= 2.0 -> PCAF 3 (Estimated)
            Score >= 1.0 -> PCAF 4 (Extrapolated)
            Score <  1.0 -> PCAF 5 (Proxy)

        Args:
            overall_score: Weighted overall quality score (0-5).

        Returns:
            PCAF tier integer (1-5).
        """
        if overall_score >= _FOUR:
            return PCAFTier.TIER_1_VERIFIED.value
        if overall_score >= _THREE:
            return PCAFTier.TIER_2_CALCULATED.value
        if overall_score >= _TWO:
            return PCAFTier.TIER_3_ESTIMATED.value
        if overall_score >= _ONE:
            return PCAFTier.TIER_4_EXTRAPOLATED.value
        return PCAFTier.TIER_5_PROXY.value

    # ----------------------------------------------- heatmap
    def generate_quality_heatmap(
        self,
        assessments: List[SiteQualityAssessment],
        scopes: List[str],
        data_entries_by_site: Optional[Dict[str, List[DataEntry]]] = None,
    ) -> List[QualityHeatmapCell]:
        """
        Generate a quality heatmap matrix (sites x scopes).

        If per-scope data entries are provided, calculates a scope-specific
        accuracy score.  Otherwise, uses the site overall score for all scopes.

        Colour mapping:
            GREEN: score >= 4.0
            AMBER: score >= 3.0
            RED:   score <  3.0

        Args:
            assessments: Per-site quality assessments.
            scopes: List of scopes for the heatmap columns.
            data_entries_by_site: Optional per-site data entries for scope-level scoring.

        Returns:
            List of QualityHeatmapCell, one per (site, scope) pair.
        """
        logger.info(
            "Generating quality heatmap: %d sites x %d scopes",
            len(assessments), len(scopes),
        )
        cells: List[QualityHeatmapCell] = []

        for assessment in assessments:
            for scope in scopes:
                # Attempt scope-specific scoring
                if data_entries_by_site and assessment.site_id in data_entries_by_site:
                    scope_entries = [
                        e for e in data_entries_by_site[assessment.site_id]
                        if e.scope == scope
                    ]
                    if scope_entries:
                        accuracy = self._score_accuracy(scope_entries)
                        score = accuracy["score"]
                    else:
                        score = _ZERO
                else:
                    score = assessment.overall_score

                colour = self._score_to_colour(score)
                cells.append(QualityHeatmapCell(
                    site_id=assessment.site_id,
                    scope=scope,
                    score=score,
                    colour_code=colour,
                ))

        logger.info("Heatmap generated: %d cells", len(cells))
        return cells

    # ----------------------------------------------- corporate quality
    def calculate_corporate_quality(
        self,
        assessments: List[SiteQualityAssessment],
        site_emissions: Dict[str, Decimal],
        weight_by_emissions: bool = False,
    ) -> Decimal:
        """
        Calculate the corporate-level aggregate quality score.

        If weight_by_emissions is True:
            corp = SUM(site_score * site_emissions) / SUM(site_emissions)
        Else:
            corp = SUM(site_score) / n

        Args:
            assessments: Per-site quality assessments.
            site_emissions: Mapping of site_id -> emissions (tCO2e).
            weight_by_emissions: Whether to weight by emissions.

        Returns:
            Corporate quality score (0-5).
        """
        logger.info(
            "Calculating corporate quality: %d sites, emission_weighted=%s",
            len(assessments), weight_by_emissions,
        )
        if not assessments:
            return _ZERO

        if weight_by_emissions:
            weighted_sum = _ZERO
            total_emissions = _ZERO
            for a in assessments:
                emissions = site_emissions.get(a.site_id, _ZERO)
                weighted_sum += a.overall_score * emissions
                total_emissions += emissions
            corp_score = _safe_divide(weighted_sum, total_emissions)
        else:
            total_score = sum(a.overall_score for a in assessments)
            corp_score = _safe_divide(total_score, Decimal(str(len(assessments))))

        corp_score = _quantise(min(corp_score, _FIVE), self._precision)
        logger.info("Corporate quality score: %s", corp_score)
        return corp_score

    # ----------------------------------------------- remediation plan
    def generate_remediation_plan(
        self,
        assessment: SiteQualityAssessment,
    ) -> List[Dict[str, Any]]:
        """
        Generate a prioritised remediation plan for a site.

        Priority is determined by the largest gap between actual and
        maximum score, weighted by dimension weight.

        Args:
            assessment: The site's quality assessment.

        Returns:
            List of remediation actions sorted by priority (highest first).
        """
        logger.info(
            "Generating remediation plan: site=%s overall=%s",
            assessment.site_id, assessment.overall_score,
        )
        actions: List[Dict[str, Any]] = []

        for ds in assessment.dimension_scores:
            gap = _FIVE - ds.score
            priority_score = _quantise(gap * ds.weight, self._precision)

            if gap <= _ZERO:
                continue

            action: Dict[str, Any] = {
                "dimension": ds.dimension,
                "current_score": ds.score,
                "target_score": _FIVE,
                "gap": gap,
                "weight": ds.weight,
                "priority_score": priority_score,
                "recommendations": list(ds.recommendations),
                "effort_level": self._classify_effort(gap),
            }
            actions.append(action)

        # Sort by priority score descending (largest weighted gap first)
        actions.sort(key=lambda a: a["priority_score"], reverse=True)

        return actions

    # ----------------------------------------------- quality progression
    def track_quality_progression(
        self,
        site_id: str,
        assessments_by_period: Dict[str, SiteQualityAssessment],
    ) -> Dict[str, Any]:
        """
        Track quality score changes over time for a site.

        Args:
            site_id: Site identifier.
            assessments_by_period: Mapping of period -> SiteQualityAssessment.

        Returns:
            Progression dict with trend data, direction, and change metrics.
        """
        logger.info(
            "Tracking quality progression: site=%s periods=%d",
            site_id, len(assessments_by_period),
        )
        if not assessments_by_period:
            return {
                "site_id": site_id,
                "periods": [],
                "direction": "STABLE",
                "overall_change": _ZERO,
                "dimension_trends": {},
            }

        sorted_periods = sorted(assessments_by_period.keys())
        period_data: List[Dict[str, Any]] = []

        for period in sorted_periods:
            a = assessments_by_period[period]
            period_data.append({
                "period": period,
                "overall_score": a.overall_score,
                "pcaf_tier": a.pcaf_equivalent,
                "estimated_pct": a.estimated_pct,
            })

        first_score = assessments_by_period[sorted_periods[0]].overall_score
        last_score = assessments_by_period[sorted_periods[-1]].overall_score
        overall_change = _quantise(last_score - first_score, self._precision)

        if last_score > first_score:
            direction = "IMPROVING"
        elif last_score < first_score:
            direction = "WORSENING"
        else:
            direction = "STABLE"

        # Dimension-level trends
        dimension_trends: Dict[str, Dict[str, Any]] = {}
        first_assessment = assessments_by_period[sorted_periods[0]]
        last_assessment = assessments_by_period[sorted_periods[-1]]

        first_dim_map = {ds.dimension: ds.score for ds in first_assessment.dimension_scores}
        last_dim_map = {ds.dimension: ds.score for ds in last_assessment.dimension_scores}

        for dim in first_dim_map:
            first_d = first_dim_map.get(dim, _ZERO)
            last_d = last_dim_map.get(dim, _ZERO)
            change = _quantise(last_d - first_d, self._precision)
            dimension_trends[dim] = {
                "first_score": first_d,
                "last_score": last_d,
                "change": change,
                "direction": "IMPROVING" if change > _ZERO else (
                    "WORSENING" if change < _ZERO else "STABLE"
                ),
            }

        provenance_payload = (
            f"progression|{site_id}|{len(sorted_periods)}|{first_score}|{last_score}"
        )
        provenance = _compute_hash(provenance_payload)

        return {
            "site_id": site_id,
            "periods": period_data,
            "direction": direction,
            "overall_change": overall_change,
            "first_period": sorted_periods[0],
            "last_period": sorted_periods[-1],
            "first_score": first_score,
            "last_score": last_score,
            "dimension_trends": dimension_trends,
            "provenance_hash": provenance,
        }

    # ----------------------------------------------- full portfolio assessment
    def assess_portfolio_quality(
        self,
        site_data: Dict[str, List[DataEntry]],
        period: str,
        site_emissions: Dict[str, Decimal],
        scopes: List[str],
        weight_by_emissions: bool = False,
    ) -> QualityResult:
        """
        Assess quality across the entire portfolio.

        Args:
            site_data: Mapping of site_id -> data entries.
            period: Reporting period.
            site_emissions: Mapping of site_id -> emissions.
            scopes: Scopes for heatmap generation.
            weight_by_emissions: Whether to emission-weight the corporate score.

        Returns:
            QualityResult with all assessments, heatmap, and corporate score.
        """
        logger.info(
            "Portfolio quality assessment: %d sites, period=%s",
            len(site_data), period,
        )
        assessments: List[SiteQualityAssessment] = []

        for site_id in sorted(site_data.keys()):
            entries = site_data[site_id]
            if entries:
                assessment = self.assess_site_quality(site_id, period, entries)
                assessments.append(assessment)

        # Corporate score
        corp_score = self.calculate_corporate_quality(
            assessments, site_emissions, weight_by_emissions
        )

        # Heatmap
        heatmap = self.generate_quality_heatmap(assessments, scopes, site_data)

        # Improvement priorities
        priorities: List[Dict[str, Any]] = []
        for a in assessments:
            plan = self.generate_remediation_plan(a)
            if plan:
                priorities.append({
                    "site_id": a.site_id,
                    "overall_score": a.overall_score,
                    "top_action": plan[0] if plan else None,
                    "action_count": len(plan),
                })

        priorities.sort(key=lambda p: p["overall_score"])

        result = QualityResult(
            assessments=assessments,
            corporate_quality_score=corp_score,
            weighted_by_emissions=weight_by_emissions,
            heatmap=heatmap,
            improvement_priorities=priorities,
        )

        logger.info(
            "Portfolio quality result: corp_score=%s assessments=%d heatmap=%d hash=%s",
            corp_score, len(assessments), len(heatmap), result.provenance_hash[:12],
        )
        return result

    # ---------------------------------------------------------------------------
    # Private Scoring Methods
    # ---------------------------------------------------------------------------

    def _score_accuracy(self, entries: List[DataEntry]) -> Dict[str, Any]:
        """
        Score ACCURACY dimension.

        Accuracy = (measured_count / total_count) * 5.0
        100% measured -> 5.0
        0% measured   -> 0.0
        """
        total = len(entries)
        measured = sum(1 for e in entries if e.is_measured)
        estimated = sum(1 for e in entries if e.is_estimated)

        if total == 0:
            return {"score": _ZERO, "evidence": "No entries", "recommendations": []}

        ratio = _safe_divide(Decimal(str(measured)), Decimal(str(total)))
        score = _quantise(ratio * _FIVE, self._precision)

        evidence = (
            f"{measured}/{total} entries from direct measurement "
            f"({estimated} estimated)"
        )
        recs: List[str] = []
        if score < _FOUR:
            recs.append("Increase proportion of directly measured data")
        if score < _THREE:
            recs.append("Install sub-metering for major emission sources")
            recs.append("Reduce reliance on estimation methods")

        return {"score": score, "evidence": evidence, "recommendations": recs}

    def _score_completeness(self, entries: List[DataEntry]) -> Dict[str, Any]:
        """
        Score COMPLETENESS dimension.

        Completeness = (total_filled_fields / total_required_fields) * 5.0
        """
        total_fields = sum(e.field_count for e in entries if e.required)
        filled_fields = sum(e.filled_field_count for e in entries if e.required)

        if total_fields == 0:
            return {"score": _FIVE, "evidence": "No required fields", "recommendations": []}

        ratio = _safe_divide(Decimal(str(filled_fields)), Decimal(str(total_fields)))
        score = _quantise(ratio * _FIVE, self._precision)

        missing = total_fields - filled_fields
        evidence = f"{filled_fields}/{total_fields} required fields filled ({missing} missing)"
        recs: List[str] = []
        if missing > 0:
            recs.append(f"Fill {missing} missing required data fields")
        if score < _THREE:
            recs.append("Review data collection templates for completeness")

        return {"score": score, "evidence": evidence, "recommendations": recs}

    def _score_consistency(
        self, entries: List[DataEntry], config: QualityConfig
    ) -> Dict[str, Any]:
        """
        Score CONSISTENCY dimension.

        For entries with prior_year_value:
            variance = |current - prior| / prior
            consistent if variance <= threshold

        Score = (consistent_entries / entries_with_prior) * 5.0
        Entries without prior data are excluded from the assessment.
        """
        entries_with_prior = [e for e in entries if e.prior_year_value is not None]
        if not entries_with_prior:
            return {
                "score": _THREE,
                "evidence": "No prior year data for comparison",
                "recommendations": ["Establish multi-year tracking for consistency checks"],
            }

        consistent_count = 0
        threshold = config.consistency_variance_threshold

        for e in entries_with_prior:
            prior = e.prior_year_value
            if prior is None or prior == _ZERO:
                consistent_count += 1
                continue
            variance = _safe_divide(abs(e.value - prior), abs(prior))
            if variance <= threshold:
                consistent_count += 1

        total = len(entries_with_prior)
        ratio = _safe_divide(Decimal(str(consistent_count)), Decimal(str(total)))
        score = _quantise(ratio * _FIVE, self._precision)

        inconsistent = total - consistent_count
        evidence = (
            f"{consistent_count}/{total} entries within {threshold * _HUNDRED}% "
            f"YoY variance ({inconsistent} inconsistent)"
        )
        recs: List[str] = []
        if inconsistent > 0:
            recs.append(f"Investigate {inconsistent} entries with high YoY variance")
        if score < _THREE:
            recs.append("Review methodology changes that may cause inconsistency")

        return {"score": score, "evidence": evidence, "recommendations": recs}

    def _score_timeliness(
        self, entries: List[DataEntry], config: QualityConfig
    ) -> Dict[str, Any]:
        """
        Score TIMELINESS dimension.

        For entries with both submitted_date and deadline_date:
            on_time if submitted_date <= deadline_date + grace_days

        Score = (on_time_count / entries_with_dates) * 5.0
        """
        entries_with_dates = [
            e for e in entries
            if e.submitted_date is not None and e.deadline_date is not None
        ]
        if not entries_with_dates:
            return {
                "score": _THREE,
                "evidence": "No submission/deadline dates available",
                "recommendations": ["Track submission dates against deadlines"],
            }

        grace = config.timeliness_grace_days
        on_time = 0
        for e in entries_with_dates:
            if e.submitted_date is not None and e.deadline_date is not None:
                effective_deadline = e.deadline_date + __import__("datetime").timedelta(days=grace)
                if e.submitted_date <= effective_deadline:
                    on_time += 1

        total = len(entries_with_dates)
        ratio = _safe_divide(Decimal(str(on_time)), Decimal(str(total)))
        score = _quantise(ratio * _FIVE, self._precision)

        late = total - on_time
        evidence = (
            f"{on_time}/{total} submissions on time "
            f"(grace: {grace} days, {late} late)"
        )
        recs: List[str] = []
        if late > 0:
            recs.append(f"Improve submission timeliness for {late} late entries")
        if score < _THREE:
            recs.append("Implement earlier reminder notifications")
            recs.append("Consider automated data feeds to reduce manual delays")

        return {"score": score, "evidence": evidence, "recommendations": recs}

    def _score_methodology(self, entries: List[DataEntry]) -> Dict[str, Any]:
        """
        Score METHODOLOGY dimension.

        Tier scoring:
            FACILITY_SPECIFIC -> 5
            NATIONAL          -> 4
            REGIONAL          -> 3
            IPCC_DEFAULT      -> 2
            UNKNOWN           -> 1

        Score = average of per-entry tier scores
        """
        tier_scores: Dict[str, Decimal] = {
            MethodologyTier.FACILITY_SPECIFIC.value: _FIVE,
            MethodologyTier.NATIONAL.value: _FOUR,
            MethodologyTier.REGIONAL.value: _THREE,
            MethodologyTier.IPCC_DEFAULT.value: _TWO,
            MethodologyTier.UNKNOWN.value: _ONE,
        }

        if not entries:
            return {"score": _ZERO, "evidence": "No entries", "recommendations": []}

        total_score = _ZERO
        for e in entries:
            entry_score = tier_scores.get(e.methodology_tier, _ONE)
            total_score += entry_score

        avg_score = _quantise(
            _safe_divide(total_score, Decimal(str(len(entries)))),
            self._precision,
        )

        # Distribution for evidence
        dist: Dict[str, int] = {}
        for e in entries:
            dist[e.methodology_tier] = dist.get(e.methodology_tier, 0) + 1

        evidence = "Methodology tier distribution: " + ", ".join(
            f"{k}={v}" for k, v in sorted(dist.items())
        )
        recs: List[str] = []
        if avg_score < _FOUR:
            recs.append("Upgrade emission factors to facility-specific or national sources")
        if avg_score < _THREE:
            recs.append("Replace IPCC default factors with country-specific data")
            recs.append("Engage suppliers for product-specific emission data")

        return {"score": avg_score, "evidence": evidence, "recommendations": recs}

    def _score_documentation(
        self, entries: List[DataEntry], config: QualityConfig
    ) -> Dict[str, Any]:
        """
        Score DOCUMENTATION dimension.

        Documentation = (entries_with_evidence / total_entries) * 5.0
        """
        total = len(entries)
        if total == 0:
            return {"score": _ZERO, "evidence": "No entries", "recommendations": []}

        with_evidence = sum(1 for e in entries if e.has_evidence)
        ratio = _safe_divide(Decimal(str(with_evidence)), Decimal(str(total)))
        score = _quantise(ratio * _FIVE, self._precision)

        missing_docs = total - with_evidence
        evidence = f"{with_evidence}/{total} entries have supporting evidence ({missing_docs} without)"
        recs: List[str] = []
        if ratio < config.min_evidence_rate:
            recs.append(
                f"Attach supporting evidence to at least {config.min_evidence_rate * _HUNDRED}% of entries"
            )
        if score < _THREE:
            recs.append("Implement mandatory evidence upload during data collection")
            recs.append("Create evidence checklist for data submitters")

        return {"score": score, "evidence": evidence, "recommendations": recs}

    # ---------------------------------------------------------------------------
    # Utility Methods
    # ---------------------------------------------------------------------------

    @staticmethod
    def _score_to_colour(score: Decimal) -> str:
        """Map a quality score to a heatmap colour."""
        if score >= _FOUR:
            return HeatmapColour.GREEN.value
        if score >= _THREE:
            return HeatmapColour.AMBER.value
        return HeatmapColour.RED.value

    @staticmethod
    def _classify_effort(gap: Decimal) -> str:
        """Classify remediation effort level based on score gap."""
        if gap <= _ONE:
            return "LOW"
        if gap <= _TWO:
            return "MEDIUM"
        return "HIGH"

    def _generate_improvement_actions(
        self,
        dimension_scores: List[DimensionScore],
    ) -> List[str]:
        """Collect all recommendations sorted by weighted gap (largest first)."""
        actions: List[Tuple[Decimal, str]] = []
        for ds in dimension_scores:
            gap = _FIVE - ds.score
            priority = _quantise(gap * ds.weight, self._precision)
            for rec in ds.recommendations:
                actions.append((priority, rec))

        actions.sort(key=lambda x: x[0], reverse=True)
        return [a[1] for a in actions]

# ---------------------------------------------------------------------------
# Pydantic v2 model rebuild (required with `from __future__ import annotations`)
# ---------------------------------------------------------------------------

DataEntry.model_rebuild()
QualityConfig.model_rebuild()
DimensionScore.model_rebuild()
SiteQualityAssessment.model_rebuild()
QualityHeatmapCell.model_rebuild()
QualityResult.model_rebuild()
