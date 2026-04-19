# -*- coding: utf-8 -*-
"""
GapAnalysisEngine - PACK-044 Inventory Management Engine 8
============================================================

Data quality gap identification and improvement planning engine that
assesses GHG inventory completeness, methodology tier quality, and
data gaps across all scopes and source categories. Produces
prioritised improvement recommendations using Pareto analysis to
maximise inventory accuracy improvement per unit of effort.

Calculation Methodology:
    Methodology Tier Scoring:
        tier_score = TIER_SCORES[methodology_tier]
        weighted_score = tier_score * (category_emissions / total_emissions)
        overall_quality = sum(weighted_scores) across all categories

        Tier Scores (per IPCC hierarchy):
            Tier 3 (facility-specific): 100 points
            Tier 2 (country-specific):   70 points
            Tier 1 (IPCC default):       40 points
            Estimated / proxy:           15 points
            Missing / no data:            0 points

    Data Gap Severity Classification:
        CRITICAL: Material source (>5% of total) with no data or estimates only
        HIGH:     Material source (>5%) with Tier 1 data only
        MEDIUM:   Moderate source (1-5%) with Tier 1 data
        LOW:      Minor source (<1%) with any data quality issue

    Pareto Analysis (80/20):
        Sort gaps by (severity_weight * emissions_share) descending
        Cumulative improvement potential calculated as:
            potential_i = (target_tier_score - current_tier_score)
                          * category_emissions / total_emissions

    Improvement Roadmap:
        effort_score = EFFORT_SCORES[current_tier][target_tier]
        roi_score = potential_improvement / effort_score
        Sort by roi_score descending for optimal sequencing

Regulatory References:
    - GHG Protocol Corporate Standard, Chapter 7 (Managing Inventory Quality)
    - ISO 14064-1:2018, Clause 9 (Uncertainty Assessment)
    - IPCC 2006 Guidelines, Volume 1, Chapter 3 (Uncertainties)
    - CSRD / ESRS E1 AR 46 (Data quality improvement plans)
    - CDP Climate Change Questionnaire, C6.3 (Methodology details)

Zero-Hallucination:
    - All scoring uses deterministic lookup tables from IPCC guidance
    - Severity classification based on explicit percentage thresholds
    - Pareto analysis uses deterministic sorting and accumulation
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-044 Inventory Management
Engine:  8 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Uses JSON serialization with sorted keys to guarantee reproducibility.

    Args:
        data: Data to hash -- dict, Pydantic model, or other serializable.

    Returns:
        SHA-256 hex digest string (64 characters).
    """
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
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation; Decimal("0") on failure.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)

def _round2(value: Any) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def _round4(value: Any) -> float:
    """Round to 4 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MethodologyTier(str, Enum):
    """IPCC methodology tier classification.

    TIER_3:    Facility-specific data, direct measurement, CEMS.
    TIER_2:    Country-specific or technology-specific emission factors.
    TIER_1:    IPCC default emission factors.
    ESTIMATED: Proxy data, spend-based, or expert estimates.
    MISSING:   No data available for this source category.
    """
    TIER_3 = "tier_3"
    TIER_2 = "tier_2"
    TIER_1 = "tier_1"
    ESTIMATED = "estimated"
    MISSING = "missing"

class GapSeverity(str, Enum):
    """Severity classification for identified data gaps.

    CRITICAL: Material source with no data or estimates only.
    HIGH:     Material source with Tier 1 data only.
    MEDIUM:   Moderate source with quality below target.
    LOW:      Minor source with any data quality issue.
    INFO:     Informational gap, no action needed.
    """
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class GapCategory(str, Enum):
    """Category of the identified gap.

    DATA_MISSING:       No data for a required source category.
    METHODOLOGY_LOW:    Methodology tier below target.
    EMISSION_FACTOR:    Using default/proxy emission factors.
    ACTIVITY_DATA:      Activity data quality issue.
    TEMPORAL_COVERAGE:  Incomplete time period coverage.
    SCOPE_COVERAGE:     Scope or category not fully covered.
    BOUNDARY:           Organisational boundary gap.
    DOCUMENTATION:      Missing supporting documentation.
    """
    DATA_MISSING = "data_missing"
    METHODOLOGY_LOW = "methodology_low"
    EMISSION_FACTOR = "emission_factor"
    ACTIVITY_DATA = "activity_data"
    TEMPORAL_COVERAGE = "temporal_coverage"
    SCOPE_COVERAGE = "scope_coverage"
    BOUNDARY = "boundary"
    DOCUMENTATION = "documentation"

class ImprovementEffort(str, Enum):
    """Effort level required for an improvement action.

    LOW:    Can be implemented within 1-3 months with existing resources.
    MEDIUM: Requires 3-6 months, may need additional resources.
    HIGH:   Requires 6-12 months, significant investment needed.
    VERY_HIGH: Multi-year programme, major capital or system changes.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tier quality scores (0-100) for weighted quality calculation.
# Based on IPCC 2006 Guidelines hierarchy and GHG Protocol data quality matrix.
TIER_SCORES: Dict[str, int] = {
    MethodologyTier.TIER_3.value: 100,
    MethodologyTier.TIER_2.value: 70,
    MethodologyTier.TIER_1.value: 40,
    MethodologyTier.ESTIMATED.value: 15,
    MethodologyTier.MISSING.value: 0,
}
"""Quality scores per methodology tier."""

# Severity weights for Pareto prioritisation.
SEVERITY_WEIGHTS: Dict[str, Decimal] = {
    GapSeverity.CRITICAL.value: Decimal("10"),
    GapSeverity.HIGH.value: Decimal("7"),
    GapSeverity.MEDIUM.value: Decimal("4"),
    GapSeverity.LOW.value: Decimal("2"),
    GapSeverity.INFO.value: Decimal("1"),
}
"""Severity weights for prioritisation scoring."""

# Effort scores for upgrading between tiers (lower = easier).
EFFORT_SCORES: Dict[str, Dict[str, int]] = {
    MethodologyTier.MISSING.value: {
        MethodologyTier.ESTIMATED.value: 2,
        MethodologyTier.TIER_1.value: 4,
        MethodologyTier.TIER_2.value: 7,
        MethodologyTier.TIER_3.value: 10,
    },
    MethodologyTier.ESTIMATED.value: {
        MethodologyTier.TIER_1.value: 3,
        MethodologyTier.TIER_2.value: 6,
        MethodologyTier.TIER_3.value: 9,
    },
    MethodologyTier.TIER_1.value: {
        MethodologyTier.TIER_2.value: 4,
        MethodologyTier.TIER_3.value: 8,
    },
    MethodologyTier.TIER_2.value: {
        MethodologyTier.TIER_3.value: 6,
    },
}
"""Effort scores for tier upgrades."""

# Materiality thresholds for gap severity classification.
MATERIALITY_CRITICAL_PCT: Decimal = Decimal("5.0")
"""Source is considered material (critical/high gap) above this % of total."""

MATERIALITY_MODERATE_PCT: Decimal = Decimal("1.0")
"""Source is considered moderate (medium gap) above this % of total."""

# Default target tier for improvement planning.
DEFAULT_TARGET_TIER: str = MethodologyTier.TIER_2.value
"""Default target methodology tier for improvement recommendations."""

# Improvement action templates by gap category.
IMPROVEMENT_TEMPLATES: Dict[str, str] = {
    GapCategory.DATA_MISSING.value: (
        "Establish data collection process for {source}. "
        "Start with spend-based estimation, then progress to activity-based "
        "measurement within {timeframe}."
    ),
    GapCategory.METHODOLOGY_LOW.value: (
        "Upgrade methodology for {source} from {current_tier} to {target_tier}. "
        "This requires {effort_description}."
    ),
    GapCategory.EMISSION_FACTOR.value: (
        "Replace default emission factors for {source} with "
        "supplier-specific or country-specific factors. "
        "Request data from {data_source}."
    ),
    GapCategory.ACTIVITY_DATA.value: (
        "Improve activity data quality for {source} by "
        "implementing metered measurement or automated data collection."
    ),
    GapCategory.TEMPORAL_COVERAGE.value: (
        "Close temporal gaps in {source} data. "
        "Implement monthly data collection to ensure full year coverage."
    ),
    GapCategory.SCOPE_COVERAGE.value: (
        "Extend inventory boundary to include {source}. "
        "Conduct screening assessment per GHG Protocol guidance."
    ),
    GapCategory.BOUNDARY.value: (
        "Review organisational boundary to include {source}. "
        "Assess whether operational or financial control applies."
    ),
    GapCategory.DOCUMENTATION.value: (
        "Complete documentation for {source} including methodology "
        "description, data sources, assumptions, and QA/QC procedures."
    ),
}
"""Improvement action templates."""

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class SourceCategoryAssessment(BaseModel):
    """Assessment of a single source category's data quality.

    Attributes:
        category_id: Unique category identifier.
        category_name: Human-readable category name.
        scope: Scope number (1, 2, or 3).
        scope3_category: Scope 3 category number (1-15), if applicable.
        emissions_tco2e: Emissions from this category (tCO2e).
        methodology_tier: Current methodology tier.
        has_activity_data: Whether measured activity data exists.
        has_emission_factors: Whether specific emission factors are used.
        temporal_coverage_pct: Percentage of reporting period with data.
        documentation_complete: Whether methodology documentation exists.
        last_updated: Date data was last updated.
        data_source: Description of data source.
        notes: Additional notes.
    """
    category_id: str = Field(default_factory=_new_uuid, description="Category ID")
    category_name: str = Field(..., min_length=1, description="Category name")
    scope: int = Field(..., ge=1, le=3, description="Scope (1, 2, 3)")
    scope3_category: Optional[int] = Field(
        default=None, ge=1, le=15, description="Scope 3 category (1-15)"
    )
    emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Emissions tCO2e"
    )
    methodology_tier: MethodologyTier = Field(
        default=MethodologyTier.TIER_1, description="Methodology tier"
    )
    has_activity_data: bool = Field(
        default=True, description="Has measured activity data"
    )
    has_emission_factors: bool = Field(
        default=True, description="Has specific emission factors"
    )
    temporal_coverage_pct: Decimal = Field(
        default=Decimal("100"), ge=0, le=100,
        description="Temporal coverage (%)"
    )
    documentation_complete: bool = Field(
        default=False, description="Documentation complete"
    )
    last_updated: Optional[datetime] = Field(
        default=None, description="Last update date"
    )
    data_source: str = Field(default="", description="Data source description")
    notes: str = Field(default="", description="Notes")

    @field_validator("emissions_tco2e", "temporal_coverage_pct", mode="before")
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce numeric fields to Decimal."""
        return _decimal(v)

class GapAnalysisConfig(BaseModel):
    """Configuration for gap analysis.

    Attributes:
        target_tier: Target methodology tier for improvement.
        materiality_threshold_pct: Material category threshold (%).
        include_scope3: Whether to include Scope 3 in analysis.
        pareto_threshold_pct: Pareto cumulative threshold (%).
        max_recommendations: Maximum number of recommendations.
    """
    target_tier: MethodologyTier = Field(
        default=MethodologyTier.TIER_2, description="Target tier"
    )
    materiality_threshold_pct: Decimal = Field(
        default=Decimal("1.0"), ge=0, le=100,
        description="Materiality threshold (%)"
    )
    include_scope3: bool = Field(
        default=True, description="Include Scope 3 in analysis"
    )
    pareto_threshold_pct: Decimal = Field(
        default=Decimal("80.0"), ge=0, le=100,
        description="Pareto cumulative threshold (%)"
    )
    max_recommendations: int = Field(
        default=20, ge=1, le=100, description="Max recommendations"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class DataGap(BaseModel):
    """An identified data quality gap.

    Attributes:
        gap_id: Unique gap identifier.
        category_id: Source category with the gap.
        category_name: Category name.
        scope: Affected scope.
        gap_category: Type of gap.
        severity: Severity classification.
        description: Human-readable description.
        emissions_share_pct: Category's share of total emissions.
        current_tier: Current methodology tier.
        target_tier: Target methodology tier.
        quality_score_current: Current quality score (0-100).
        quality_score_target: Target quality score (0-100).
        improvement_potential: Potential quality improvement points.
    """
    gap_id: str = Field(default_factory=_new_uuid, description="Gap ID")
    category_id: str = Field(default="", description="Category ID")
    category_name: str = Field(default="", description="Category name")
    scope: int = Field(default=1, description="Scope")
    gap_category: str = Field(default="", description="Gap category")
    severity: str = Field(default="low", description="Severity")
    description: str = Field(default="", description="Description")
    emissions_share_pct: float = Field(default=0.0, description="Emissions share %")
    current_tier: str = Field(default="", description="Current tier")
    target_tier: str = Field(default="", description="Target tier")
    quality_score_current: int = Field(default=0, description="Current score")
    quality_score_target: int = Field(default=0, description="Target score")
    improvement_potential: int = Field(default=0, description="Potential improvement")

class MethodologyGap(BaseModel):
    """A methodology-specific gap with tier details.

    Attributes:
        gap_id: Reference to parent DataGap.
        category_name: Category name.
        current_tier: Current IPCC methodology tier.
        target_tier: Recommended target tier.
        tier_gap: Number of tier levels to improve.
        estimated_effort: Estimated effort level.
        estimated_cost_range: Estimated cost range description.
        expected_uncertainty_reduction_pct: Expected uncertainty reduction.
    """
    gap_id: str = Field(default="", description="Parent gap ID")
    category_name: str = Field(default="", description="Category name")
    current_tier: str = Field(default="", description="Current tier")
    target_tier: str = Field(default="", description="Target tier")
    tier_gap: int = Field(default=0, description="Tier levels to improve")
    estimated_effort: str = Field(default="medium", description="Effort level")
    estimated_cost_range: str = Field(default="", description="Cost range")
    expected_uncertainty_reduction_pct: float = Field(
        default=0.0, description="Expected uncertainty reduction %"
    )

class ImprovementRecommendation(BaseModel):
    """A prioritised improvement recommendation.

    Attributes:
        recommendation_id: Unique recommendation identifier.
        priority_rank: Priority rank (1 = highest priority).
        gap_id: Reference to the identified gap.
        category_name: Category name.
        action: Recommended action description.
        effort: Estimated effort level.
        roi_score: Return on investment score (higher = better).
        improvement_potential_points: Quality points gained.
        emissions_impact_pct: Percentage of total emissions affected.
        timeline: Estimated implementation timeline.
        dependencies: Dependencies on other recommendations.
    """
    recommendation_id: str = Field(
        default_factory=_new_uuid, description="Recommendation ID"
    )
    priority_rank: int = Field(default=0, description="Priority rank")
    gap_id: str = Field(default="", description="Gap reference")
    category_name: str = Field(default="", description="Category name")
    action: str = Field(default="", description="Recommended action")
    effort: str = Field(default="medium", description="Effort level")
    roi_score: float = Field(default=0.0, description="ROI score")
    improvement_potential_points: int = Field(
        default=0, description="Quality points gained"
    )
    emissions_impact_pct: float = Field(
        default=0.0, description="Emissions affected %"
    )
    timeline: str = Field(default="", description="Implementation timeline")
    dependencies: List[str] = Field(
        default_factory=list, description="Dependencies"
    )

class ImprovementRoadmap(BaseModel):
    """Phased improvement roadmap.

    Attributes:
        phase_1_quick_wins: Actions achievable in 0-3 months.
        phase_2_medium_term: Actions achievable in 3-6 months.
        phase_3_long_term: Actions achievable in 6-12 months.
        total_quality_improvement: Total quality points to be gained.
        estimated_total_effort: Overall effort assessment.
        current_overall_score: Current weighted quality score.
        projected_score_after: Projected score after roadmap completion.
    """
    phase_1_quick_wins: List[str] = Field(
        default_factory=list, description="Phase 1 (0-3 months)"
    )
    phase_2_medium_term: List[str] = Field(
        default_factory=list, description="Phase 2 (3-6 months)"
    )
    phase_3_long_term: List[str] = Field(
        default_factory=list, description="Phase 3 (6-12 months)"
    )
    total_quality_improvement: int = Field(
        default=0, description="Total quality points gained"
    )
    estimated_total_effort: str = Field(
        default="medium", description="Overall effort"
    )
    current_overall_score: float = Field(
        default=0.0, description="Current quality score"
    )
    projected_score_after: float = Field(
        default=0.0, description="Projected score after improvements"
    )

class GapAnalysisResult(BaseModel):
    """Complete gap analysis result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Calculation timestamp (UTC).
        processing_time_ms: Processing time in milliseconds.
        total_emissions_tco2e: Total emissions analysed.
        categories_assessed: Number of categories assessed.
        overall_quality_score: Weighted quality score (0-100).
        gaps_identified: Total number of gaps identified.
        critical_gaps: Number of critical gaps.
        high_gaps: Number of high-severity gaps.
        data_gaps: List of all identified data gaps.
        methodology_gaps: Methodology-specific gap details.
        recommendations: Prioritised improvement recommendations.
        roadmap: Phased improvement roadmap.
        pareto_coverage_pct: Percentage of improvement covered by top items.
        methodology_notes: Methodology notes.
        provenance_hash: SHA-256 provenance hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Timestamp"
    )
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    total_emissions_tco2e: float = Field(
        default=0.0, description="Total emissions assessed"
    )
    categories_assessed: int = Field(default=0, description="Categories assessed")
    overall_quality_score: float = Field(
        default=0.0, description="Overall quality (0-100)"
    )
    gaps_identified: int = Field(default=0, description="Total gaps")
    critical_gaps: int = Field(default=0, description="Critical gaps")
    high_gaps: int = Field(default=0, description="High-severity gaps")
    data_gaps: List[DataGap] = Field(default_factory=list, description="Data gaps")
    methodology_gaps: List[MethodologyGap] = Field(
        default_factory=list, description="Methodology gaps"
    )
    recommendations: List[ImprovementRecommendation] = Field(
        default_factory=list, description="Recommendations"
    )
    roadmap: Optional[ImprovementRoadmap] = Field(
        default=None, description="Improvement roadmap"
    )
    pareto_coverage_pct: float = Field(
        default=0.0, description="Pareto coverage %"
    )
    methodology_notes: List[str] = Field(
        default_factory=list, description="Methodology notes"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Model Rebuild
# ---------------------------------------------------------------------------

SourceCategoryAssessment.model_rebuild()
GapAnalysisConfig.model_rebuild()
DataGap.model_rebuild()
MethodologyGap.model_rebuild()
ImprovementRecommendation.model_rebuild()
ImprovementRoadmap.model_rebuild()
GapAnalysisResult.model_rebuild()

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class GapAnalysisEngine:
    """GHG inventory data quality gap identification engine.

    Systematically assesses data quality across all source categories,
    identifies gaps between current and target methodology tiers,
    and produces prioritised improvement recommendations using Pareto
    analysis to maximise inventory accuracy per unit of improvement effort.

    Features:
        - Methodology tier assessment per IPCC 2006 hierarchy
        - Weighted quality scoring based on emissions contribution
        - Gap severity classification (critical/high/medium/low)
        - Pareto analysis for prioritised improvement sequencing
        - Phased improvement roadmap generation
        - ROI-based recommendation ranking

    Guarantees:
        - Deterministic: same inputs produce identical results
        - Reproducible: SHA-256 provenance hash on every result
        - Auditable: full gap-by-gap breakdown with scoring details
        - No LLM: zero hallucination risk in any calculation path

    Usage::

        engine = GapAnalysisEngine()
        categories = [
            SourceCategoryAssessment(
                category_name="Stationary Combustion",
                scope=1,
                emissions_tco2e=Decimal("5000"),
                methodology_tier=MethodologyTier.TIER_2,
            ),
        ]
        result = engine.analyse(categories)
        print(f"Quality score: {result.overall_quality_score}")
        for rec in result.recommendations[:3]:
            print(f"  [{rec.priority_rank}] {rec.action}")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[GapAnalysisConfig] = None) -> None:
        """Initialise the gap analysis engine.

        Args:
            config: Optional analysis configuration.
        """
        self._config = config or GapAnalysisConfig()
        self._notes: List[str] = []
        logger.info("GapAnalysisEngine v%s initialised.", _MODULE_VERSION)

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def analyse(
        self,
        categories: List[SourceCategoryAssessment],
        config: Optional[GapAnalysisConfig] = None,
    ) -> GapAnalysisResult:
        """Run complete gap analysis across all source categories.

        Args:
            categories: List of source category assessments.
            config: Optional configuration override.

        Returns:
            GapAnalysisResult with gaps, recommendations, and roadmap.

        Raises:
            ValueError: If categories list is empty.
        """
        t0 = time.perf_counter()
        cfg = config or self._config
        self._notes = [f"Engine version: {self.engine_version}"]

        if not categories:
            raise ValueError("At least one source category is required.")

        # Filter Scope 3 if not included
        if not cfg.include_scope3:
            categories = [c for c in categories if c.scope != 3]

        total_emissions = sum(_decimal(c.emissions_tco2e) for c in categories)

        logger.info(
            "Gap analysis: %d categories, total %.2f tCO2e",
            len(categories), float(total_emissions),
        )

        # Step 1: Calculate overall quality score
        quality_score = self._calculate_quality_score(categories, total_emissions)

        # Step 2: Identify data gaps
        data_gaps = self._identify_data_gaps(categories, total_emissions, cfg)

        # Step 3: Identify methodology gaps
        meth_gaps = self._identify_methodology_gaps(categories, cfg)

        # Step 4: Generate recommendations with Pareto analysis
        recommendations = self._generate_recommendations(
            data_gaps, meth_gaps, categories, total_emissions, cfg
        )

        # Step 5: Build improvement roadmap
        roadmap = self._build_roadmap(recommendations, quality_score)

        # Step 6: Calculate Pareto coverage
        pareto_pct = self._calculate_pareto_coverage(recommendations, cfg)

        critical_count = sum(1 for g in data_gaps if g.severity == GapSeverity.CRITICAL.value)
        high_count = sum(1 for g in data_gaps if g.severity == GapSeverity.HIGH.value)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = GapAnalysisResult(
            total_emissions_tco2e=_round4(float(total_emissions)),
            categories_assessed=len(categories),
            overall_quality_score=_round2(quality_score),
            gaps_identified=len(data_gaps),
            critical_gaps=critical_count,
            high_gaps=high_count,
            data_gaps=data_gaps,
            methodology_gaps=meth_gaps,
            recommendations=recommendations[:cfg.max_recommendations],
            roadmap=roadmap,
            pareto_coverage_pct=_round2(pareto_pct),
            methodology_notes=list(self._notes),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Gap analysis complete: score=%.1f, gaps=%d (critical=%d, high=%d), "
            "recommendations=%d, hash=%s (%.1f ms)",
            quality_score, len(data_gaps), critical_count, high_count,
            len(recommendations), result.provenance_hash[:16], elapsed_ms,
        )
        return result

    def calculate_category_quality_score(
        self,
        category: SourceCategoryAssessment,
    ) -> int:
        """Calculate quality score for a single category.

        Args:
            category: Source category assessment.

        Returns:
            Quality score (0-100).
        """
        base_score = TIER_SCORES.get(category.methodology_tier.value, 0)

        # Adjust for activity data presence
        if not category.has_activity_data:
            base_score = max(base_score - 15, 0)

        # Adjust for emission factor quality
        if not category.has_emission_factors:
            base_score = max(base_score - 10, 0)

        # Adjust for temporal coverage
        coverage = float(category.temporal_coverage_pct)
        if coverage < 100.0:
            coverage_penalty = int((100.0 - coverage) * 0.3)
            base_score = max(base_score - coverage_penalty, 0)

        # Adjust for documentation
        if not category.documentation_complete:
            base_score = max(base_score - 5, 0)

        return min(base_score, 100)

    def classify_gap_severity(
        self,
        emissions_share_pct: float,
        methodology_tier: str,
        has_data: bool,
    ) -> str:
        """Classify gap severity based on materiality and data quality.

        Args:
            emissions_share_pct: Category's share of total emissions.
            methodology_tier: Current methodology tier.
            has_data: Whether any data exists.

        Returns:
            Severity classification string.
        """
        share = _decimal(emissions_share_pct)

        if not has_data or methodology_tier == MethodologyTier.MISSING.value:
            if share >= MATERIALITY_CRITICAL_PCT:
                return GapSeverity.CRITICAL.value
            elif share >= MATERIALITY_MODERATE_PCT:
                return GapSeverity.HIGH.value
            else:
                return GapSeverity.MEDIUM.value

        if methodology_tier == MethodologyTier.ESTIMATED.value:
            if share >= MATERIALITY_CRITICAL_PCT:
                return GapSeverity.CRITICAL.value
            elif share >= MATERIALITY_MODERATE_PCT:
                return GapSeverity.HIGH.value
            else:
                return GapSeverity.LOW.value

        if methodology_tier == MethodologyTier.TIER_1.value:
            if share >= MATERIALITY_CRITICAL_PCT:
                return GapSeverity.HIGH.value
            elif share >= MATERIALITY_MODERATE_PCT:
                return GapSeverity.MEDIUM.value
            else:
                return GapSeverity.LOW.value

        if methodology_tier == MethodologyTier.TIER_2.value:
            if share >= MATERIALITY_CRITICAL_PCT:
                return GapSeverity.LOW.value
            else:
                return GapSeverity.INFO.value

        return GapSeverity.INFO.value

    # -------------------------------------------------------------------
    # Private -- Quality scoring
    # -------------------------------------------------------------------

    def _calculate_quality_score(
        self,
        categories: List[SourceCategoryAssessment],
        total_emissions: Decimal,
    ) -> float:
        """Calculate emissions-weighted overall quality score.

        Args:
            categories: Source category assessments.
            total_emissions: Total emissions across all categories.

        Returns:
            Weighted quality score (0-100).
        """
        if not categories or total_emissions == Decimal("0"):
            return 0.0

        weighted_sum = Decimal("0")

        for cat in categories:
            cat_score = Decimal(str(self.calculate_category_quality_score(cat)))
            weight = _safe_divide(_decimal(cat.emissions_tco2e), total_emissions)
            weighted_sum += cat_score * weight

        score = float(weighted_sum)
        self._notes.append(
            f"Overall quality score: {_round2(score)} (emissions-weighted, "
            f"{len(categories)} categories)"
        )
        return score

    # -------------------------------------------------------------------
    # Private -- Gap identification
    # -------------------------------------------------------------------

    def _identify_data_gaps(
        self,
        categories: List[SourceCategoryAssessment],
        total_emissions: Decimal,
        config: GapAnalysisConfig,
    ) -> List[DataGap]:
        """Identify all data quality gaps across categories.

        Args:
            categories: Source category assessments.
            total_emissions: Total emissions for share calculation.
            config: Analysis configuration.

        Returns:
            List of DataGap sorted by severity and emissions share.
        """
        gaps: List[DataGap] = []
        target_tier = config.target_tier.value
        target_score = TIER_SCORES.get(target_tier, 70)

        for cat in categories:
            emissions_share = float(
                _safe_pct(_decimal(cat.emissions_tco2e), total_emissions)
            )
            current_score = self.calculate_category_quality_score(cat)

            # Skip if already at or above target
            if current_score >= target_score:
                continue

            improvement_potential = target_score - current_score

            # Determine gap categories
            gap_categories = self._determine_gap_categories(cat)

            for gap_cat in gap_categories:
                has_data = cat.methodology_tier != MethodologyTier.MISSING
                severity = self.classify_gap_severity(
                    emissions_share, cat.methodology_tier.value, has_data
                )

                description = self._build_gap_description(
                    cat, gap_cat, severity, emissions_share
                )

                gaps.append(DataGap(
                    category_id=cat.category_id,
                    category_name=cat.category_name,
                    scope=cat.scope,
                    gap_category=gap_cat,
                    severity=severity,
                    description=description,
                    emissions_share_pct=_round2(emissions_share),
                    current_tier=cat.methodology_tier.value,
                    target_tier=target_tier,
                    quality_score_current=current_score,
                    quality_score_target=target_score,
                    improvement_potential=improvement_potential,
                ))

        # Sort by severity weight * emissions share (descending)
        gaps.sort(
            key=lambda g: (
                float(SEVERITY_WEIGHTS.get(g.severity, Decimal("1")))
                * g.emissions_share_pct
            ),
            reverse=True,
        )
        return gaps

    def _determine_gap_categories(
        self,
        cat: SourceCategoryAssessment,
    ) -> List[str]:
        """Determine which gap categories apply to a source category.

        Args:
            cat: Source category assessment.

        Returns:
            List of applicable GapCategory values.
        """
        gap_cats: List[str] = []

        if cat.methodology_tier == MethodologyTier.MISSING:
            gap_cats.append(GapCategory.DATA_MISSING.value)
            return gap_cats

        if cat.methodology_tier in (
            MethodologyTier.ESTIMATED, MethodologyTier.TIER_1
        ):
            gap_cats.append(GapCategory.METHODOLOGY_LOW.value)

        if not cat.has_activity_data:
            gap_cats.append(GapCategory.ACTIVITY_DATA.value)

        if not cat.has_emission_factors:
            gap_cats.append(GapCategory.EMISSION_FACTOR.value)

        if cat.temporal_coverage_pct < Decimal("100"):
            gap_cats.append(GapCategory.TEMPORAL_COVERAGE.value)

        if not cat.documentation_complete:
            gap_cats.append(GapCategory.DOCUMENTATION.value)

        return gap_cats if gap_cats else [GapCategory.METHODOLOGY_LOW.value]

    def _build_gap_description(
        self,
        cat: SourceCategoryAssessment,
        gap_category: str,
        severity: str,
        emissions_share: float,
    ) -> str:
        """Build a human-readable gap description.

        Args:
            cat: Source category.
            gap_category: Gap category type.
            severity: Gap severity.
            emissions_share: Category's emissions share.

        Returns:
            Description string.
        """
        scope_label = f"Scope {cat.scope}"
        if cat.scope == 3 and cat.scope3_category is not None:
            scope_label += f" Category {cat.scope3_category}"

        descriptions: Dict[str, str] = {
            GapCategory.DATA_MISSING.value: (
                f"{scope_label} - {cat.category_name}: No data available. "
                f"This category represents {_round2(emissions_share)}% of total "
                f"emissions. Severity: {severity.upper()}."
            ),
            GapCategory.METHODOLOGY_LOW.value: (
                f"{scope_label} - {cat.category_name}: Current methodology "
                f"tier is {cat.methodology_tier.value}, below target. "
                f"Represents {_round2(emissions_share)}% of total emissions."
            ),
            GapCategory.EMISSION_FACTOR.value: (
                f"{scope_label} - {cat.category_name}: Using default or proxy "
                f"emission factors. Specific factors would improve accuracy."
            ),
            GapCategory.ACTIVITY_DATA.value: (
                f"{scope_label} - {cat.category_name}: Activity data is "
                f"estimated rather than measured. Metered data recommended."
            ),
            GapCategory.TEMPORAL_COVERAGE.value: (
                f"{scope_label} - {cat.category_name}: Temporal coverage "
                f"is {_round2(float(cat.temporal_coverage_pct))}%. "
                f"Full year coverage required."
            ),
            GapCategory.DOCUMENTATION.value: (
                f"{scope_label} - {cat.category_name}: Methodology "
                f"documentation is incomplete."
            ),
        }
        return descriptions.get(
            gap_category,
            f"{scope_label} - {cat.category_name}: Data quality gap identified."
        )

    # -------------------------------------------------------------------
    # Private -- Methodology gaps
    # -------------------------------------------------------------------

    def _identify_methodology_gaps(
        self,
        categories: List[SourceCategoryAssessment],
        config: GapAnalysisConfig,
    ) -> List[MethodologyGap]:
        """Identify methodology tier gaps for each category.

        Args:
            categories: Source category assessments.
            config: Analysis configuration.

        Returns:
            List of MethodologyGap details.
        """
        target = config.target_tier.value
        target_score = TIER_SCORES.get(target, 70)
        meth_gaps: List[MethodologyGap] = []

        tier_order = [
            MethodologyTier.MISSING.value,
            MethodologyTier.ESTIMATED.value,
            MethodologyTier.TIER_1.value,
            MethodologyTier.TIER_2.value,
            MethodologyTier.TIER_3.value,
        ]

        for cat in categories:
            current = cat.methodology_tier.value
            current_score = TIER_SCORES.get(current, 0)

            if current_score >= target_score:
                continue

            # Calculate tier gap
            current_idx = tier_order.index(current) if current in tier_order else 0
            target_idx = tier_order.index(target) if target in tier_order else 3
            tier_gap = max(target_idx - current_idx, 0)

            # Effort estimation
            effort_map = EFFORT_SCORES.get(current, {})
            effort_score = effort_map.get(target, 5)
            effort_level = self._score_to_effort(effort_score)

            # Cost range estimation
            cost_range = self._estimate_cost_range(effort_level)

            # Expected uncertainty reduction
            uncertainty_reduction = self._estimate_uncertainty_reduction(
                current, target
            )

            meth_gaps.append(MethodologyGap(
                gap_id=cat.category_id,
                category_name=cat.category_name,
                current_tier=current,
                target_tier=target,
                tier_gap=tier_gap,
                estimated_effort=effort_level,
                estimated_cost_range=cost_range,
                expected_uncertainty_reduction_pct=_round2(uncertainty_reduction),
            ))

        return meth_gaps

    def _score_to_effort(self, effort_score: int) -> str:
        """Convert numeric effort score to effort level.

        Args:
            effort_score: Numeric effort score (1-10).

        Returns:
            ImprovementEffort value.
        """
        if effort_score <= 3:
            return ImprovementEffort.LOW.value
        elif effort_score <= 5:
            return ImprovementEffort.MEDIUM.value
        elif effort_score <= 8:
            return ImprovementEffort.HIGH.value
        else:
            return ImprovementEffort.VERY_HIGH.value

    def _estimate_cost_range(self, effort_level: str) -> str:
        """Estimate cost range based on effort level.

        Args:
            effort_level: Effort level string.

        Returns:
            Cost range description.
        """
        ranges: Dict[str, str] = {
            ImprovementEffort.LOW.value: "EUR 1,000 - 5,000",
            ImprovementEffort.MEDIUM.value: "EUR 5,000 - 25,000",
            ImprovementEffort.HIGH.value: "EUR 25,000 - 100,000",
            ImprovementEffort.VERY_HIGH.value: "EUR 100,000+",
        }
        return ranges.get(effort_level, "EUR 5,000 - 25,000")

    def _estimate_uncertainty_reduction(
        self, current_tier: str, target_tier: str
    ) -> float:
        """Estimate expected uncertainty reduction from tier upgrade.

        Based on IPCC 2006 typical uncertainty ranges by tier.

        Args:
            current_tier: Current methodology tier.
            target_tier: Target methodology tier.

        Returns:
            Expected uncertainty reduction in percentage points.
        """
        # Typical uncertainty (%) by tier based on IPCC guidance
        tier_uncertainties: Dict[str, float] = {
            MethodologyTier.MISSING.value: 100.0,
            MethodologyTier.ESTIMATED.value: 60.0,
            MethodologyTier.TIER_1.value: 30.0,
            MethodologyTier.TIER_2.value: 15.0,
            MethodologyTier.TIER_3.value: 5.0,
        }
        current_u = tier_uncertainties.get(current_tier, 30.0)
        target_u = tier_uncertainties.get(target_tier, 15.0)
        return max(current_u - target_u, 0.0)

    # -------------------------------------------------------------------
    # Private -- Recommendations
    # -------------------------------------------------------------------

    def _generate_recommendations(
        self,
        data_gaps: List[DataGap],
        meth_gaps: List[MethodologyGap],
        categories: List[SourceCategoryAssessment],
        total_emissions: Decimal,
        config: GapAnalysisConfig,
    ) -> List[ImprovementRecommendation]:
        """Generate prioritised improvement recommendations.

        Uses ROI scoring: potential improvement / effort to prioritise
        actions that deliver the most quality improvement per unit effort.

        Args:
            data_gaps: Identified data gaps.
            meth_gaps: Methodology gaps.
            categories: Source categories for context.
            total_emissions: Total emissions.
            config: Analysis configuration.

        Returns:
            List of recommendations sorted by ROI score (descending).
        """
        recommendations: List[ImprovementRecommendation] = []
        meth_gap_map = {g.gap_id: g for g in meth_gaps}

        # Deduplicate gaps by category_id to get one recommendation per category
        seen_categories: set = set()

        for gap in data_gaps:
            if gap.category_id in seen_categories:
                continue
            seen_categories.add(gap.category_id)

            improvement = gap.improvement_potential
            meth_gap = meth_gap_map.get(gap.category_id)

            # Calculate effort score
            effort_score_val = 5  # default
            effort_level = ImprovementEffort.MEDIUM.value
            if meth_gap:
                effort_map = EFFORT_SCORES.get(gap.current_tier, {})
                effort_score_val = effort_map.get(gap.target_tier, 5)
                effort_level = self._score_to_effort(effort_score_val)

            # Calculate ROI score
            roi = _round2(
                float(
                    _safe_divide(
                        _decimal(improvement) * _decimal(gap.emissions_share_pct),
                        _decimal(max(effort_score_val, 1)),
                    )
                )
            )

            # Build action text
            action = self._build_action_text(gap, meth_gap, effort_level)

            # Estimate timeline
            timeline = self._estimate_timeline(effort_level)

            recommendations.append(ImprovementRecommendation(
                gap_id=gap.gap_id,
                category_name=gap.category_name,
                action=action,
                effort=effort_level,
                roi_score=roi,
                improvement_potential_points=improvement,
                emissions_impact_pct=gap.emissions_share_pct,
                timeline=timeline,
            ))

        # Sort by ROI score descending
        recommendations.sort(key=lambda r: r.roi_score, reverse=True)

        # Assign priority ranks
        for rank, rec in enumerate(recommendations, start=1):
            rec.priority_rank = rank

        self._notes.append(
            f"Generated {len(recommendations)} improvement recommendations "
            f"ranked by ROI score."
        )
        return recommendations

    def _build_action_text(
        self,
        gap: DataGap,
        meth_gap: Optional[MethodologyGap],
        effort_level: str,
    ) -> str:
        """Build action text for a recommendation.

        Args:
            gap: The data gap.
            meth_gap: Optional methodology gap details.
            effort_level: Effort level.

        Returns:
            Action description string.
        """
        template = IMPROVEMENT_TEMPLATES.get(
            gap.gap_category,
            "Improve data quality for {source} from {current_tier} to {target_tier}."
        )

        effort_descriptions: Dict[str, str] = {
            ImprovementEffort.LOW.value: "minimal additional effort using existing systems",
            ImprovementEffort.MEDIUM.value: "moderate effort with some system enhancements",
            ImprovementEffort.HIGH.value: "significant investment in measurement systems",
            ImprovementEffort.VERY_HIGH.value: "major capital investment in monitoring infrastructure",
        }

        return template.format(
            source=gap.category_name,
            current_tier=gap.current_tier,
            target_tier=gap.target_tier,
            effort_description=effort_descriptions.get(effort_level, "moderate effort"),
            timeframe=self._estimate_timeline(effort_level),
            data_source="suppliers or national database",
        )

    def _estimate_timeline(self, effort_level: str) -> str:
        """Estimate implementation timeline.

        Args:
            effort_level: Effort level.

        Returns:
            Timeline description.
        """
        timelines: Dict[str, str] = {
            ImprovementEffort.LOW.value: "1-3 months",
            ImprovementEffort.MEDIUM.value: "3-6 months",
            ImprovementEffort.HIGH.value: "6-12 months",
            ImprovementEffort.VERY_HIGH.value: "12-24 months",
        }
        return timelines.get(effort_level, "3-6 months")

    # -------------------------------------------------------------------
    # Private -- Pareto analysis
    # -------------------------------------------------------------------

    def _calculate_pareto_coverage(
        self,
        recommendations: List[ImprovementRecommendation],
        config: GapAnalysisConfig,
    ) -> float:
        """Calculate Pareto coverage of top recommendations.

        Determines what percentage of total improvement potential is
        covered by the top N recommendations that collectively reach
        the Pareto threshold.

        Args:
            recommendations: Sorted recommendations.
            config: Configuration with Pareto threshold.

        Returns:
            Pareto coverage percentage.
        """
        if not recommendations:
            return 0.0

        total_potential = sum(r.improvement_potential_points for r in recommendations)
        if total_potential == 0:
            return 0.0

        threshold = float(config.pareto_threshold_pct)
        cumulative = 0.0
        items_needed = 0

        for rec in recommendations:
            cumulative += rec.improvement_potential_points
            items_needed += 1
            pct = (cumulative / total_potential) * 100.0
            if pct >= threshold:
                break

        coverage = (items_needed / len(recommendations)) * 100.0 if recommendations else 0.0
        self._notes.append(
            f"Pareto analysis: {items_needed} of {len(recommendations)} "
            f"recommendations cover {_round2(min(cumulative / total_potential * 100, 100))}% "
            f"of total improvement potential."
        )
        return min(cumulative / total_potential * 100.0, 100.0)

    # -------------------------------------------------------------------
    # Private -- Roadmap
    # -------------------------------------------------------------------

    def _build_roadmap(
        self,
        recommendations: List[ImprovementRecommendation],
        current_score: float,
    ) -> ImprovementRoadmap:
        """Build a phased improvement roadmap.

        Groups recommendations into three phases based on effort level:
            Phase 1 (Quick Wins): LOW effort, 0-3 months
            Phase 2 (Medium Term): MEDIUM effort, 3-6 months
            Phase 3 (Long Term): HIGH/VERY_HIGH effort, 6-12+ months

        Args:
            recommendations: Sorted recommendations.
            current_score: Current overall quality score.

        Returns:
            ImprovementRoadmap with phased actions.
        """
        phase_1: List[str] = []
        phase_2: List[str] = []
        phase_3: List[str] = []
        total_improvement = 0

        for rec in recommendations:
            action_text = f"[{rec.category_name}] {rec.action}"
            total_improvement += rec.improvement_potential_points

            if rec.effort == ImprovementEffort.LOW.value:
                phase_1.append(action_text)
            elif rec.effort == ImprovementEffort.MEDIUM.value:
                phase_2.append(action_text)
            else:
                phase_3.append(action_text)

        # Estimate projected score
        total_possible = sum(r.improvement_potential_points for r in recommendations)
        projected = min(current_score + (total_possible * 0.7), 100.0)

        overall_effort = ImprovementEffort.MEDIUM.value
        if len(phase_3) > len(phase_1) + len(phase_2):
            overall_effort = ImprovementEffort.HIGH.value
        elif len(phase_1) > len(phase_2) + len(phase_3):
            overall_effort = ImprovementEffort.LOW.value

        return ImprovementRoadmap(
            phase_1_quick_wins=phase_1,
            phase_2_medium_term=phase_2,
            phase_3_long_term=phase_3,
            total_quality_improvement=total_improvement,
            estimated_total_effort=overall_effort,
            current_overall_score=_round2(current_score),
            projected_score_after=_round2(projected),
        )
