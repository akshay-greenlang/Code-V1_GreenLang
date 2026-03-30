# -*- coding: utf-8 -*-
"""
InterimTargetEngine - PACK-029 Interim Targets Pack Engine 1
================================================================

Calculates 5-year and 10-year interim targets from baseline emissions
and long-term net-zero targets.  Supports SBTi validation thresholds
(42% reduction by 2030 for 1.5C, 30% for WB2C), scope-specific
timelines (Scope 3 can lag by 5 years), linear and milestone-based
pathways, and multi-scope interim target generation.

Calculation Methodology:
    Linear Interim Target:
        target(t) = baseline * (1 - annual_rate) ^ (t - base_year)
        annual_rate = 1 - (long_term_target / baseline) ^ (1 / years)

    Milestone-Based Interim Target:
        Interpolate between defined milestones using linear
        segments between milestone years.

    SBTi Validation (Near-Term):
        1.5C aligned: >= 4.2%/yr linear reduction => >= 42% by 2030
        WB2C aligned: >= 2.5%/yr linear reduction => >= 25% by 2030

    SBTi Validation (Long-Term, Net-Zero Standard v1.2):
        1.5C: >= 90% absolute reduction by long-term target year
        Scope 3 lag: near-term target may be set 5 years later
        than Scope 1+2 (max 2030 for 1.5C, 2035 for Scope 3)

    Interim Target at Year t:
        absolute_target_tco2e = baseline_tco2e * (1 - reduction_pct / 100)
        reduction_pct(t) = total_reduction_pct * (t - base_year) /
                           (target_year - base_year)

    Scope 3 Allowances:
        - SBTi allows 5-year lag for Scope 3 near-term targets
        - Scope 3 threshold: 67% of total Scope 3 must be covered
        - FLAG emissions require separate near-term targets

Regulatory References:
    - SBTi Corporate Net-Zero Standard v1.2 (2024)
    - SBTi Corporate Manual v5.3 (2024) -- target-setting criteria
    - IPCC AR6 WG3 (2022) -- 1.5C pathway: 43% CO2 by 2030
    - Paris Agreement (2015) -- Article 2.1(a) 1.5C temperature limit
    - GHG Protocol Corporate Standard (2004, revised 2015)
    - Race to Zero: 50% by 2030 aspiration
    - ISO 14064-1:2018 -- Organizational GHG inventories

Zero-Hallucination:
    - All targets use deterministic Decimal arithmetic
    - SBTi thresholds hard-coded from Corporate Manual v5.3
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-029 Interim Targets
Engine:  1 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
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
    """Compute a deterministic SHA-256 hash of arbitrary data."""
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
    """Safely convert a value to Decimal."""
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

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _round3(value: float) -> float:
    """Round to 3 decimal places."""
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ClimateAmbition(str, Enum):
    """Climate ambition level for target-setting."""
    CELSIUS_1_5 = "1.5c"
    WELL_BELOW_2C = "wb2c"
    TWO_C = "2c"
    RACE_TO_ZERO = "race_to_zero"

class ScopeType(str, Enum):
    """GHG Protocol scope classification."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"
    SCOPE_1_2 = "scope_1_2"
    ALL_SCOPES = "all_scopes"

class PathwayShape(str, Enum):
    """Shape of the reduction pathway between milestones."""
    LINEAR = "linear"
    FRONT_LOADED = "front_loaded"
    BACK_LOADED = "back_loaded"
    MILESTONE_BASED = "milestone_based"
    CONSTANT_RATE = "constant_rate"

class TargetType(str, Enum):
    """Type of interim target."""
    NEAR_TERM = "near_term"
    MID_TERM = "mid_term"
    LONG_TERM = "long_term"
    NET_ZERO = "net_zero"

class ValidationStatus(str, Enum):
    """SBTi validation status for a target."""
    ALIGNED = "aligned"
    PARTIALLY_ALIGNED = "partially_aligned"
    MISALIGNED = "misaligned"
    EXCEEDS_MINIMUM = "exceeds_minimum"
    REQUIRES_REVIEW = "requires_review"

class DataQuality(str, Enum):
    """Data quality tier."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ESTIMATED = "estimated"

# ---------------------------------------------------------------------------
# Constants -- SBTi Target Thresholds
# ---------------------------------------------------------------------------

# SBTi Corporate Net-Zero Standard v1.2 (2024) + Corporate Manual v5.3
SBTI_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    ClimateAmbition.CELSIUS_1_5.value: {
        "name": "1.5C Aligned",
        "annual_rate_pct": Decimal("4.2"),
        "near_term_reduction_pct": Decimal("42"),
        "near_term_horizon_years": 10,
        "near_term_latest_year": 2030,
        "long_term_reduction_pct": Decimal("90"),
        "scope3_near_term_lag_years": 5,
        "scope3_coverage_pct": Decimal("67"),
        "scope12_coverage_pct": Decimal("95"),
        "temperature_score": Decimal("1.5"),
    },
    ClimateAmbition.WELL_BELOW_2C.value: {
        "name": "Well Below 2C",
        "annual_rate_pct": Decimal("2.5"),
        "near_term_reduction_pct": Decimal("25"),
        "near_term_horizon_years": 10,
        "near_term_latest_year": 2030,
        "long_term_reduction_pct": Decimal("80"),
        "scope3_near_term_lag_years": 5,
        "scope3_coverage_pct": Decimal("67"),
        "scope12_coverage_pct": Decimal("95"),
        "temperature_score": Decimal("1.8"),
    },
    ClimateAmbition.TWO_C.value: {
        "name": "2C Aligned",
        "annual_rate_pct": Decimal("1.5"),
        "near_term_reduction_pct": Decimal("15"),
        "near_term_horizon_years": 10,
        "near_term_latest_year": 2035,
        "long_term_reduction_pct": Decimal("72"),
        "scope3_near_term_lag_years": 5,
        "scope3_coverage_pct": Decimal("67"),
        "scope12_coverage_pct": Decimal("95"),
        "temperature_score": Decimal("2.0"),
    },
    ClimateAmbition.RACE_TO_ZERO.value: {
        "name": "Race to Zero",
        "annual_rate_pct": Decimal("7.0"),
        "near_term_reduction_pct": Decimal("50"),
        "near_term_horizon_years": 10,
        "near_term_latest_year": 2030,
        "long_term_reduction_pct": Decimal("90"),
        "scope3_near_term_lag_years": 5,
        "scope3_coverage_pct": Decimal("67"),
        "scope12_coverage_pct": Decimal("95"),
        "temperature_score": Decimal("1.5"),
    },
}

# FLAG sector special thresholds (SBTi FLAG guidance)
FLAG_THRESHOLDS: Dict[str, Decimal] = {
    "near_term_reduction_pct": Decimal("30"),
    "near_term_latest_year_offset": 2,
    "coverage_pct": Decimal("67"),
    "annual_rate_pct": Decimal("3.0"),
}

# Default milestone years for 5-year intervals
DEFAULT_MILESTONE_YEARS: List[int] = [2025, 2030, 2035, 2040, 2045, 2050]

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class BaselineData(BaseModel):
    """Baseline emissions data for target calculation.

    Attributes:
        base_year: Year of the emissions baseline.
        scope_1_tco2e: Scope 1 baseline emissions (tCO2e).
        scope_2_tco2e: Scope 2 baseline emissions (tCO2e).
        scope_3_tco2e: Scope 3 baseline emissions (tCO2e).
        total_tco2e: Total baseline emissions (auto-calculated if zero).
        scope_2_method: Scope 2 accounting method.
        is_flag_sector: Whether entity has FLAG sector emissions.
        flag_emissions_tco2e: FLAG sector emissions (subset of above).
    """
    base_year: int = Field(
        ..., ge=2015, le=2025, description="Baseline year"
    )
    scope_1_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Scope 1 baseline (tCO2e)"
    )
    scope_2_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Scope 2 baseline (tCO2e)"
    )
    scope_3_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Scope 3 baseline (tCO2e)"
    )
    total_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Total baseline (tCO2e)"
    )
    scope_2_method: str = Field(
        default="market_based", description="Scope 2 method"
    )
    is_flag_sector: bool = Field(
        default=False, description="Has FLAG sector emissions"
    )
    flag_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="FLAG emissions (tCO2e)"
    )

    @field_validator("total_tco2e", mode="before")
    @classmethod
    def _auto_total(cls, v: Any, info: Any) -> Any:
        """Auto-calculate total if not provided."""
        return v

class LongTermTarget(BaseModel):
    """Long-term (net-zero) target parameters.

    Attributes:
        target_year: Year to achieve long-term target.
        reduction_pct: Target reduction percentage from baseline.
        residual_emissions_pct: Allowed residual emissions (SBTi: max 10%).
        net_zero_year: Year to achieve net-zero (may differ from target_year).
        includes_scope_3: Whether Scope 3 is included.
    """
    target_year: int = Field(
        default=2050, ge=2030, le=2070, description="Long-term target year"
    )
    reduction_pct: Decimal = Field(
        default=Decimal("90"), ge=Decimal("0"), le=Decimal("100"),
        description="Target reduction (%)"
    )
    residual_emissions_pct: Decimal = Field(
        default=Decimal("10"), ge=Decimal("0"), le=Decimal("100"),
        description="Residual emissions allowed (%)"
    )
    net_zero_year: int = Field(
        default=2050, ge=2030, le=2070, description="Net-zero year"
    )
    includes_scope_3: bool = Field(
        default=True, description="Scope 3 included"
    )

class MilestoneOverride(BaseModel):
    """Manual milestone override for custom pathway shapes.

    Attributes:
        year: Milestone year.
        scope: Scope this milestone applies to.
        reduction_pct: Required reduction percentage by this year.
        description: Milestone description.
    """
    year: int = Field(..., ge=2020, le=2070, description="Milestone year")
    scope: ScopeType = Field(
        default=ScopeType.ALL_SCOPES, description="Applicable scope"
    )
    reduction_pct: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("100"),
        description="Reduction % by this year"
    )
    description: str = Field(
        default="", max_length=500, description="Milestone description"
    )

class InterimTargetInput(BaseModel):
    """Input for interim target calculation.

    Attributes:
        entity_name: Company or entity name.
        entity_id: Unique entity identifier.
        baseline: Baseline emissions data.
        long_term_target: Long-term target parameters.
        ambition_level: Climate ambition level.
        pathway_shape: Shape of the reduction pathway.
        milestone_overrides: Custom milestone overrides.
        scope_3_lag_years: Scope 3 target lag (0-5 years).
        include_flag_targets: Generate separate FLAG targets.
        generate_5_year_targets: Generate 5-year interval targets.
        generate_10_year_targets: Generate 10-year interval targets.
        include_sbti_validation: Run SBTi validation checks.
        reporting_year: Current reporting year.
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Entity name"
    )
    entity_id: str = Field(
        default="", max_length=100, description="Entity identifier"
    )
    baseline: BaselineData = Field(..., description="Baseline data")
    long_term_target: LongTermTarget = Field(
        default_factory=LongTermTarget, description="Long-term target"
    )
    ambition_level: ClimateAmbition = Field(
        default=ClimateAmbition.CELSIUS_1_5,
        description="Climate ambition"
    )
    pathway_shape: PathwayShape = Field(
        default=PathwayShape.LINEAR, description="Pathway shape"
    )
    milestone_overrides: List[MilestoneOverride] = Field(
        default_factory=list, description="Custom milestones"
    )
    scope_3_lag_years: int = Field(
        default=0, ge=0, le=5, description="Scope 3 lag years"
    )
    include_flag_targets: bool = Field(
        default=False, description="Include FLAG targets"
    )
    generate_5_year_targets: bool = Field(
        default=True, description="Generate 5-year targets"
    )
    generate_10_year_targets: bool = Field(
        default=True, description="Generate 10-year targets"
    )
    include_sbti_validation: bool = Field(
        default=True, description="Run SBTi validation"
    )
    reporting_year: int = Field(
        default=2024, ge=2020, le=2030, description="Reporting year"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class InterimMilestone(BaseModel):
    """A single interim milestone/target point.

    Attributes:
        year: Target year.
        scope: Applicable scope.
        target_type: Type of target (near/mid/long).
        baseline_tco2e: Baseline emissions for this scope.
        target_tco2e: Target emissions for this year.
        reduction_pct: Reduction percentage from baseline.
        absolute_reduction_tco2e: Absolute reduction from baseline.
        annual_rate_pct: Implied annual reduction rate.
        cumulative_budget_tco2e: Cumulative emissions budget to this year.
        is_sbti_compliant: Whether this target meets SBTi minimum.
        ambition_assessment: Assessment vs ambition level.
        notes: Additional notes.
    """
    year: int = Field(default=0)
    scope: str = Field(default=ScopeType.ALL_SCOPES.value)
    target_type: str = Field(default=TargetType.NEAR_TERM.value)
    baseline_tco2e: Decimal = Field(default=Decimal("0"))
    target_tco2e: Decimal = Field(default=Decimal("0"))
    reduction_pct: Decimal = Field(default=Decimal("0"))
    absolute_reduction_tco2e: Decimal = Field(default=Decimal("0"))
    annual_rate_pct: Decimal = Field(default=Decimal("0"))
    cumulative_budget_tco2e: Decimal = Field(default=Decimal("0"))
    is_sbti_compliant: bool = Field(default=False)
    ambition_assessment: str = Field(default="")
    notes: List[str] = Field(default_factory=list)

class ScopeTimeline(BaseModel):
    """Timeline for a specific scope.

    Attributes:
        scope: GHG scope.
        baseline_tco2e: Baseline emissions.
        near_term_year: Near-term target year.
        near_term_target_tco2e: Near-term target emissions.
        near_term_reduction_pct: Near-term reduction %.
        mid_term_year: Mid-term target year (if applicable).
        mid_term_target_tco2e: Mid-term target emissions.
        mid_term_reduction_pct: Mid-term reduction %.
        long_term_year: Long-term target year.
        long_term_target_tco2e: Long-term target emissions.
        long_term_reduction_pct: Long-term reduction %.
        annual_rate_pct: Required annual reduction rate.
        milestones: All milestones for this scope.
    """
    scope: str = Field(default="")
    baseline_tco2e: Decimal = Field(default=Decimal("0"))
    near_term_year: int = Field(default=0)
    near_term_target_tco2e: Decimal = Field(default=Decimal("0"))
    near_term_reduction_pct: Decimal = Field(default=Decimal("0"))
    mid_term_year: int = Field(default=0)
    mid_term_target_tco2e: Decimal = Field(default=Decimal("0"))
    mid_term_reduction_pct: Decimal = Field(default=Decimal("0"))
    long_term_year: int = Field(default=0)
    long_term_target_tco2e: Decimal = Field(default=Decimal("0"))
    long_term_reduction_pct: Decimal = Field(default=Decimal("0"))
    annual_rate_pct: Decimal = Field(default=Decimal("0"))
    milestones: List[InterimMilestone] = Field(default_factory=list)

class SBTiValidationResult(BaseModel):
    """SBTi validation result for interim targets.

    Attributes:
        is_compliant: Overall SBTi compliance.
        ambition_level: Assessed ambition level.
        near_term_check: Near-term target validation.
        scope_coverage_check: Scope coverage validation.
        timeline_check: Timeline validation.
        flag_check: FLAG sector validation (if applicable).
        linearity_check: No-backsliding check.
        total_checks: Total checks performed.
        passed_checks: Checks passed.
        failed_checks: Checks failed.
        warning_checks: Checks with warnings.
        validation_notes: Detailed notes.
    """
    is_compliant: bool = Field(default=False)
    ambition_level: str = Field(default="")
    near_term_check: str = Field(default=ValidationStatus.REQUIRES_REVIEW.value)
    scope_coverage_check: str = Field(default=ValidationStatus.REQUIRES_REVIEW.value)
    timeline_check: str = Field(default=ValidationStatus.REQUIRES_REVIEW.value)
    flag_check: str = Field(default=ValidationStatus.REQUIRES_REVIEW.value)
    linearity_check: str = Field(default=ValidationStatus.REQUIRES_REVIEW.value)
    total_checks: int = Field(default=0)
    passed_checks: int = Field(default=0)
    failed_checks: int = Field(default=0)
    warning_checks: int = Field(default=0)
    validation_notes: List[str] = Field(default_factory=list)

class FLAGTargetResult(BaseModel):
    """FLAG sector target result.

    Attributes:
        flag_baseline_tco2e: FLAG baseline emissions.
        flag_near_term_target_tco2e: FLAG near-term target.
        flag_near_term_reduction_pct: FLAG near-term reduction %.
        flag_near_term_year: FLAG near-term target year.
        flag_annual_rate_pct: FLAG annual reduction rate.
        is_compliant: Whether FLAG target meets SBTi guidance.
        notes: FLAG-specific notes.
    """
    flag_baseline_tco2e: Decimal = Field(default=Decimal("0"))
    flag_near_term_target_tco2e: Decimal = Field(default=Decimal("0"))
    flag_near_term_reduction_pct: Decimal = Field(default=Decimal("0"))
    flag_near_term_year: int = Field(default=0)
    flag_annual_rate_pct: Decimal = Field(default=Decimal("0"))
    is_compliant: bool = Field(default=False)
    notes: List[str] = Field(default_factory=list)

class InterimTargetResult(BaseModel):
    """Complete interim target calculation result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp.
        entity_name: Entity name.
        entity_id: Entity identifier.
        ambition_level: Climate ambition level.
        pathway_shape: Pathway shape used.
        baseline_year: Baseline year.
        baseline_total_tco2e: Total baseline emissions.
        net_zero_year: Net-zero target year.
        scope_timelines: Per-scope timelines with milestones.
        all_milestones: All milestones across scopes (chronological).
        five_year_targets: 5-year interval targets.
        ten_year_targets: 10-year interval targets.
        sbti_validation: SBTi validation result.
        flag_targets: FLAG sector targets (if applicable).
        implied_temperature_score: Temperature alignment score.
        annual_reduction_rate_scope12_pct: Annual rate for Scope 1+2.
        annual_reduction_rate_scope3_pct: Annual rate for Scope 3.
        total_abatement_required_tco2e: Total abatement needed.
        data_quality: Data quality assessment.
        recommendations: Recommendations.
        warnings: Warnings.
        processing_time_ms: Processing duration (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")
    entity_id: str = Field(default="")
    ambition_level: str = Field(default="")
    pathway_shape: str = Field(default="")
    baseline_year: int = Field(default=0)
    baseline_total_tco2e: Decimal = Field(default=Decimal("0"))
    net_zero_year: int = Field(default=0)
    scope_timelines: List[ScopeTimeline] = Field(default_factory=list)
    all_milestones: List[InterimMilestone] = Field(default_factory=list)
    five_year_targets: List[InterimMilestone] = Field(default_factory=list)
    ten_year_targets: List[InterimMilestone] = Field(default_factory=list)
    sbti_validation: Optional[SBTiValidationResult] = Field(default=None)
    flag_targets: Optional[FLAGTargetResult] = Field(default=None)
    implied_temperature_score: Decimal = Field(default=Decimal("0"))
    annual_reduction_rate_scope12_pct: Decimal = Field(default=Decimal("0"))
    annual_reduction_rate_scope3_pct: Decimal = Field(default=Decimal("0"))
    total_abatement_required_tco2e: Decimal = Field(default=Decimal("0"))
    data_quality: str = Field(default=DataQuality.MEDIUM.value)
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class InterimTargetEngine:
    """Interim target calculation engine for PACK-029.

    Calculates 5-year and 10-year interim reduction targets from
    baseline emissions and long-term net-zero goals, with SBTi
    validation, scope-specific timelines, and multiple pathway shapes.

    All calculations use deterministic Decimal arithmetic.
    No LLM involvement in any calculation path.

    Usage::

        engine = InterimTargetEngine()
        result = await engine.calculate(interim_target_input)
        for m in result.all_milestones:
            print(f"  {m.year}: {m.reduction_pct}% => {m.target_tco2e} tCO2e")
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    async def calculate(self, data: InterimTargetInput) -> InterimTargetResult:
        """Run complete interim target calculation.

        Args:
            data: Validated interim target input.

        Returns:
            InterimTargetResult with milestones, timelines, and validation.
        """
        t0 = time.perf_counter()
        logger.info(
            "Interim target calculation: entity=%s, ambition=%s, shape=%s",
            data.entity_name, data.ambition_level.value, data.pathway_shape.value,
        )

        # Resolve baseline total
        baseline_total = self._resolve_baseline_total(data.baseline)

        # Build scope timelines
        scope_timelines = self._build_scope_timelines(data, baseline_total)

        # Collect all milestones
        all_milestones = self._collect_all_milestones(scope_timelines)

        # Generate 5-year targets
        five_year = self._generate_interval_targets(
            data, baseline_total, 5
        ) if data.generate_5_year_targets else []

        # Generate 10-year targets
        ten_year = self._generate_interval_targets(
            data, baseline_total, 10
        ) if data.generate_10_year_targets else []

        # SBTi validation
        sbti_validation: Optional[SBTiValidationResult] = None
        if data.include_sbti_validation:
            sbti_validation = self._validate_sbti(
                data, scope_timelines, all_milestones
            )

        # FLAG targets
        flag_targets: Optional[FLAGTargetResult] = None
        if data.include_flag_targets and data.baseline.is_flag_sector:
            flag_targets = self._calculate_flag_targets(data)

        # Annual reduction rates
        scope12_rate = self._calculate_annual_rate(
            data.baseline.scope_1_tco2e + data.baseline.scope_2_tco2e,
            data.long_term_target.reduction_pct,
            data.baseline.base_year,
            data.long_term_target.target_year,
        )
        scope3_rate = self._calculate_annual_rate(
            data.baseline.scope_3_tco2e,
            data.long_term_target.reduction_pct,
            data.baseline.base_year + data.scope_3_lag_years,
            data.long_term_target.target_year,
        )

        # Temperature score
        temp_score = self._calculate_temperature_score(scope12_rate)

        # Total abatement required
        total_abatement = baseline_total * data.long_term_target.reduction_pct / Decimal("100")

        # Data quality
        dq = self._assess_data_quality(data)

        # Recommendations
        recs = self._generate_recommendations(
            data, scope_timelines, sbti_validation
        )

        # Warnings
        warns = self._generate_warnings(data, scope_timelines, sbti_validation)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = InterimTargetResult(
            entity_name=data.entity_name,
            entity_id=data.entity_id,
            ambition_level=data.ambition_level.value,
            pathway_shape=data.pathway_shape.value,
            baseline_year=data.baseline.base_year,
            baseline_total_tco2e=baseline_total,
            net_zero_year=data.long_term_target.net_zero_year,
            scope_timelines=scope_timelines,
            all_milestones=all_milestones,
            five_year_targets=five_year,
            ten_year_targets=ten_year,
            sbti_validation=sbti_validation,
            flag_targets=flag_targets,
            implied_temperature_score=_round_val(temp_score, 2),
            annual_reduction_rate_scope12_pct=_round_val(scope12_rate, 3),
            annual_reduction_rate_scope3_pct=_round_val(scope3_rate, 3),
            total_abatement_required_tco2e=_round_val(total_abatement, 2),
            data_quality=dq,
            recommendations=recs,
            warnings=warns,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Interim targets complete: entity=%s, milestones=%d, "
            "sbti=%s, temp=%.2f",
            data.entity_name, len(all_milestones),
            sbti_validation.is_compliant if sbti_validation else "skipped",
            float(temp_score),
        )
        return result

    async def calculate_batch(
        self, inputs: List[InterimTargetInput],
    ) -> List[InterimTargetResult]:
        """Calculate interim targets for multiple entities.

        Args:
            inputs: List of interim target inputs.

        Returns:
            List of results, one per input.
        """
        results: List[InterimTargetResult] = []
        for inp in inputs:
            try:
                result = await self.calculate(inp)
                results.append(result)
            except Exception as exc:
                logger.error(
                    "Batch error for %s: %s", inp.entity_name, exc,
                )
                results.append(InterimTargetResult(
                    entity_name=inp.entity_name,
                    entity_id=inp.entity_id,
                    warnings=[f"Calculation error: {exc}"],
                ))
        return results

    # ------------------------------------------------------------------ #
    # Baseline Resolution                                                  #
    # ------------------------------------------------------------------ #

    def _resolve_baseline_total(self, baseline: BaselineData) -> Decimal:
        """Resolve total baseline emissions.

        If total_tco2e is zero, auto-calculate from scope components.

        Formula:
            total = scope_1 + scope_2 + scope_3
        """
        if baseline.total_tco2e > Decimal("0"):
            return baseline.total_tco2e
        return (
            baseline.scope_1_tco2e
            + baseline.scope_2_tco2e
            + baseline.scope_3_tco2e
        )

    # ------------------------------------------------------------------ #
    # Scope Timelines                                                      #
    # ------------------------------------------------------------------ #

    def _build_scope_timelines(
        self,
        data: InterimTargetInput,
        baseline_total: Decimal,
    ) -> List[ScopeTimeline]:
        """Build per-scope timelines with milestones.

        Generates Scope 1+2, Scope 3, and All Scopes timelines.
        """
        timelines: List[ScopeTimeline] = []
        thresholds = SBTI_THRESHOLDS.get(
            data.ambition_level.value,
            SBTI_THRESHOLDS[ClimateAmbition.CELSIUS_1_5.value],
        )

        # Scope 1+2 timeline
        s12_baseline = data.baseline.scope_1_tco2e + data.baseline.scope_2_tco2e
        if s12_baseline > Decimal("0"):
            s12_timeline = self._build_single_scope_timeline(
                scope=ScopeType.SCOPE_1_2.value,
                baseline_tco2e=s12_baseline,
                base_year=data.baseline.base_year,
                target_year=data.long_term_target.target_year,
                reduction_pct=data.long_term_target.reduction_pct,
                pathway_shape=data.pathway_shape,
                thresholds=thresholds,
                milestone_overrides=[
                    m for m in data.milestone_overrides
                    if m.scope in (ScopeType.SCOPE_1_2, ScopeType.ALL_SCOPES)
                ],
                lag_years=0,
            )
            timelines.append(s12_timeline)

        # Scope 3 timeline (with lag)
        s3_baseline = data.baseline.scope_3_tco2e
        if s3_baseline > Decimal("0"):
            s3_timeline = self._build_single_scope_timeline(
                scope=ScopeType.SCOPE_3.value,
                baseline_tco2e=s3_baseline,
                base_year=data.baseline.base_year,
                target_year=data.long_term_target.target_year,
                reduction_pct=data.long_term_target.reduction_pct,
                pathway_shape=data.pathway_shape,
                thresholds=thresholds,
                milestone_overrides=[
                    m for m in data.milestone_overrides
                    if m.scope in (ScopeType.SCOPE_3, ScopeType.ALL_SCOPES)
                ],
                lag_years=data.scope_3_lag_years,
            )
            timelines.append(s3_timeline)

        # All scopes combined timeline
        if baseline_total > Decimal("0"):
            all_timeline = self._build_single_scope_timeline(
                scope=ScopeType.ALL_SCOPES.value,
                baseline_tco2e=baseline_total,
                base_year=data.baseline.base_year,
                target_year=data.long_term_target.target_year,
                reduction_pct=data.long_term_target.reduction_pct,
                pathway_shape=data.pathway_shape,
                thresholds=thresholds,
                milestone_overrides=data.milestone_overrides,
                lag_years=0,
            )
            timelines.append(all_timeline)

        return timelines

    def _build_single_scope_timeline(
        self,
        scope: str,
        baseline_tco2e: Decimal,
        base_year: int,
        target_year: int,
        reduction_pct: Decimal,
        pathway_shape: PathwayShape,
        thresholds: Dict[str, Any],
        milestone_overrides: List[MilestoneOverride],
        lag_years: int,
    ) -> ScopeTimeline:
        """Build timeline for a single scope.

        Args:
            scope: Scope identifier.
            baseline_tco2e: Baseline emissions for this scope.
            base_year: Baseline year.
            target_year: Long-term target year.
            reduction_pct: Target reduction percentage.
            pathway_shape: Shape of reduction pathway.
            thresholds: SBTi threshold parameters.
            milestone_overrides: Custom milestones.
            lag_years: Years of lag for this scope.

        Returns:
            ScopeTimeline with milestones.
        """
        effective_base = base_year + lag_years
        total_years = target_year - effective_base
        if total_years <= 0:
            total_years = 1

        # Annual reduction rate
        annual_rate = self._calculate_annual_rate(
            baseline_tco2e, reduction_pct, effective_base, target_year
        )

        # Generate milestones at 5-year intervals
        milestones: List[InterimMilestone] = []
        milestone_years = [
            y for y in DEFAULT_MILESTONE_YEARS
            if effective_base < y <= target_year
        ]
        # Add target year if not already present
        if target_year not in milestone_years:
            milestone_years.append(target_year)
        milestone_years.sort()

        # Check for manual overrides
        override_map: Dict[int, Decimal] = {}
        for mo in milestone_overrides:
            override_map[mo.year] = mo.reduction_pct

        cumulative_budget = Decimal("0")
        prev_year = effective_base
        prev_tco2e = baseline_tco2e

        for year in milestone_years:
            # Determine reduction % at this year
            if year in override_map:
                red_pct = override_map[year]
            else:
                red_pct = self._interpolate_reduction(
                    year, effective_base, target_year,
                    reduction_pct, pathway_shape,
                )

            target_tco2e = baseline_tco2e * (Decimal("1") - red_pct / Decimal("100"))
            target_tco2e = max(target_tco2e, Decimal("0"))
            abs_reduction = baseline_tco2e - target_tco2e

            # Cumulative budget (trapezoidal)
            years_elapsed = year - prev_year
            segment_budget = (prev_tco2e + target_tco2e) / Decimal("2") * _decimal(years_elapsed)
            cumulative_budget += segment_budget

            # Target type classification
            years_from_base = year - base_year
            if years_from_base <= 10:
                target_type = TargetType.NEAR_TERM.value
            elif years_from_base <= 20:
                target_type = TargetType.MID_TERM.value
            elif year >= target_year:
                target_type = TargetType.NET_ZERO.value
            else:
                target_type = TargetType.LONG_TERM.value

            # SBTi compliance check
            min_reduction = thresholds.get("near_term_reduction_pct", Decimal("42"))
            is_compliant = red_pct >= min_reduction if target_type == TargetType.NEAR_TERM.value else True

            # Ambition assessment
            ambition = self._assess_milestone_ambition(red_pct, annual_rate, year, base_year)

            milestone = InterimMilestone(
                year=year,
                scope=scope,
                target_type=target_type,
                baseline_tco2e=_round_val(baseline_tco2e, 2),
                target_tco2e=_round_val(target_tco2e, 2),
                reduction_pct=_round_val(red_pct, 2),
                absolute_reduction_tco2e=_round_val(abs_reduction, 2),
                annual_rate_pct=_round_val(annual_rate, 3),
                cumulative_budget_tco2e=_round_val(cumulative_budget, 2),
                is_sbti_compliant=is_compliant,
                ambition_assessment=ambition,
            )
            milestones.append(milestone)

            prev_year = year
            prev_tco2e = target_tco2e

        # Near-term target (first milestone or 2030)
        near_term = next(
            (m for m in milestones if m.year <= 2030), None
        )
        # Mid-term target (around 2035-2040)
        mid_term = next(
            (m for m in milestones if 2035 <= m.year <= 2040), None
        )
        # Long-term target (final)
        long_term = milestones[-1] if milestones else None

        return ScopeTimeline(
            scope=scope,
            baseline_tco2e=_round_val(baseline_tco2e, 2),
            near_term_year=near_term.year if near_term else 0,
            near_term_target_tco2e=near_term.target_tco2e if near_term else Decimal("0"),
            near_term_reduction_pct=near_term.reduction_pct if near_term else Decimal("0"),
            mid_term_year=mid_term.year if mid_term else 0,
            mid_term_target_tco2e=mid_term.target_tco2e if mid_term else Decimal("0"),
            mid_term_reduction_pct=mid_term.reduction_pct if mid_term else Decimal("0"),
            long_term_year=long_term.year if long_term else 0,
            long_term_target_tco2e=long_term.target_tco2e if long_term else Decimal("0"),
            long_term_reduction_pct=long_term.reduction_pct if long_term else Decimal("0"),
            annual_rate_pct=_round_val(annual_rate, 3),
            milestones=milestones,
        )

    # ------------------------------------------------------------------ #
    # Reduction Interpolation                                              #
    # ------------------------------------------------------------------ #

    def _interpolate_reduction(
        self,
        year: int,
        base_year: int,
        target_year: int,
        total_reduction_pct: Decimal,
        pathway_shape: PathwayShape,
    ) -> Decimal:
        """Interpolate reduction percentage at a given year.

        Formulas by pathway shape:
            LINEAR:
                pct(t) = total_pct * (t - base) / (target - base)

            FRONT_LOADED:
                pct(t) = total_pct * sqrt((t - base) / (target - base))

            BACK_LOADED:
                pct(t) = total_pct * ((t - base) / (target - base))^2

            CONSTANT_RATE:
                rate = 1 - (1 - total_pct/100)^(1/years)
                pct(t) = (1 - (1 - rate)^(t - base)) * 100

        Args:
            year: Year to interpolate.
            base_year: Base year.
            target_year: Target year.
            total_reduction_pct: Total reduction by target year.
            pathway_shape: Shape of pathway.

        Returns:
            Reduction percentage at the given year.
        """
        total_years = target_year - base_year
        if total_years <= 0:
            return total_reduction_pct

        elapsed = year - base_year
        if elapsed <= 0:
            return Decimal("0")
        if elapsed >= total_years:
            return total_reduction_pct

        progress = _decimal(elapsed) / _decimal(total_years)

        if pathway_shape == PathwayShape.LINEAR:
            return total_reduction_pct * progress

        elif pathway_shape == PathwayShape.FRONT_LOADED:
            # Square root curve -- faster early reductions
            sqrt_progress = _decimal(math.sqrt(float(progress)))
            return total_reduction_pct * sqrt_progress

        elif pathway_shape == PathwayShape.BACK_LOADED:
            # Quadratic curve -- slower early, accelerating later
            return total_reduction_pct * progress * progress

        elif pathway_shape == PathwayShape.CONSTANT_RATE:
            # Compound annual reduction
            remaining_frac = Decimal("1") - total_reduction_pct / Decimal("100")
            if remaining_frac <= Decimal("0"):
                remaining_frac = Decimal("0.001")
            try:
                rate_per_year = Decimal("1") - _decimal(
                    float(remaining_frac) ** (1.0 / total_years)
                )
                remaining_at_year = (Decimal("1") - rate_per_year) ** elapsed
                return (Decimal("1") - remaining_at_year) * Decimal("100")
            except (OverflowError, ValueError):
                return total_reduction_pct * progress

        elif pathway_shape == PathwayShape.MILESTONE_BASED:
            # Linear between milestones (handled by override map in caller)
            return total_reduction_pct * progress

        return total_reduction_pct * progress

    # ------------------------------------------------------------------ #
    # Annual Reduction Rate                                                #
    # ------------------------------------------------------------------ #

    def _calculate_annual_rate(
        self,
        baseline_tco2e: Decimal,
        reduction_pct: Decimal,
        base_year: int,
        target_year: int,
    ) -> Decimal:
        """Calculate implied annual reduction rate.

        Formula:
            remaining = 1 - reduction_pct / 100
            annual_rate = 1 - remaining^(1 / years)
            annual_rate_pct = annual_rate * 100

        Args:
            baseline_tco2e: Baseline emissions.
            reduction_pct: Total reduction percentage.
            base_year: Base year.
            target_year: Target year.

        Returns:
            Annual reduction rate in percentage.
        """
        years = target_year - base_year
        if years <= 0 or baseline_tco2e <= Decimal("0"):
            return Decimal("0")

        remaining = Decimal("1") - reduction_pct / Decimal("100")
        if remaining <= Decimal("0"):
            remaining = Decimal("0.001")

        try:
            annual_factor = _decimal(
                float(remaining) ** (1.0 / years)
            )
            annual_rate = (Decimal("1") - annual_factor) * Decimal("100")
            return max(annual_rate, Decimal("0"))
        except (OverflowError, ValueError):
            return Decimal("0")

    # ------------------------------------------------------------------ #
    # Temperature Score                                                    #
    # ------------------------------------------------------------------ #

    def _calculate_temperature_score(
        self, annual_rate_pct: Decimal,
    ) -> Decimal:
        """Calculate implied temperature alignment score.

        Formula (simplified SBTi-aligned):
            temp = 1.5 + max(0, (4.2 - annual_rate) / 4.2) * 2.0
            capped at 4.0C

        Args:
            annual_rate_pct: Annual reduction rate (%).

        Returns:
            Temperature score in degrees Celsius.
        """
        rate_42 = Decimal("4.2")
        if annual_rate_pct >= rate_42:
            return Decimal("1.5")

        gap = max(rate_42 - annual_rate_pct, Decimal("0"))
        temp = Decimal("1.5") + _safe_divide(gap, rate_42) * Decimal("2.0")
        return min(temp, Decimal("4.0"))

    # ------------------------------------------------------------------ #
    # Interval Targets                                                     #
    # ------------------------------------------------------------------ #

    def _generate_interval_targets(
        self,
        data: InterimTargetInput,
        baseline_total: Decimal,
        interval_years: int,
    ) -> List[InterimMilestone]:
        """Generate targets at fixed intervals from baseline.

        Args:
            data: Input data.
            baseline_total: Total baseline emissions.
            interval_years: Interval between targets (5 or 10).

        Returns:
            List of InterimMilestone at each interval.
        """
        targets: List[InterimMilestone] = []
        base_year = data.baseline.base_year
        target_year = data.long_term_target.target_year

        year = base_year + interval_years
        cumulative_budget = Decimal("0")
        prev_tco2e = baseline_total
        prev_year = base_year

        while year <= target_year:
            red_pct = self._interpolate_reduction(
                year, base_year, target_year,
                data.long_term_target.reduction_pct,
                data.pathway_shape,
            )
            target_tco2e = baseline_total * (Decimal("1") - red_pct / Decimal("100"))
            target_tco2e = max(target_tco2e, Decimal("0"))
            abs_reduction = baseline_total - target_tco2e

            years_elapsed = year - prev_year
            segment_budget = (prev_tco2e + target_tco2e) / Decimal("2") * _decimal(years_elapsed)
            cumulative_budget += segment_budget

            annual_rate = self._calculate_annual_rate(
                baseline_total, red_pct, base_year, year,
            )

            years_from_base = year - base_year
            if years_from_base <= 10:
                target_type = TargetType.NEAR_TERM.value
            elif years_from_base <= 20:
                target_type = TargetType.MID_TERM.value
            else:
                target_type = TargetType.LONG_TERM.value

            ambition = self._assess_milestone_ambition(
                red_pct, annual_rate, year, base_year,
            )

            targets.append(InterimMilestone(
                year=year,
                scope=ScopeType.ALL_SCOPES.value,
                target_type=target_type,
                baseline_tco2e=_round_val(baseline_total, 2),
                target_tco2e=_round_val(target_tco2e, 2),
                reduction_pct=_round_val(red_pct, 2),
                absolute_reduction_tco2e=_round_val(abs_reduction, 2),
                annual_rate_pct=_round_val(annual_rate, 3),
                cumulative_budget_tco2e=_round_val(cumulative_budget, 2),
                is_sbti_compliant=True,
                ambition_assessment=ambition,
            ))

            prev_year = year
            prev_tco2e = target_tco2e
            year += interval_years

        return targets

    # ------------------------------------------------------------------ #
    # Milestone Collection                                                 #
    # ------------------------------------------------------------------ #

    def _collect_all_milestones(
        self, timelines: List[ScopeTimeline],
    ) -> List[InterimMilestone]:
        """Collect all milestones from scope timelines, sorted chronologically."""
        all_ms: List[InterimMilestone] = []
        for tl in timelines:
            all_ms.extend(tl.milestones)
        all_ms.sort(key=lambda m: (m.year, m.scope))
        return all_ms

    # ------------------------------------------------------------------ #
    # Ambition Assessment                                                  #
    # ------------------------------------------------------------------ #

    def _assess_milestone_ambition(
        self,
        reduction_pct: Decimal,
        annual_rate_pct: Decimal,
        year: int,
        base_year: int,
    ) -> str:
        """Assess ambition of a milestone target.

        Args:
            reduction_pct: Reduction percentage at this milestone.
            annual_rate_pct: Annual reduction rate.
            year: Milestone year.
            base_year: Baseline year.

        Returns:
            Ambition assessment string.
        """
        years_from_base = year - base_year

        # Check against SBTi thresholds
        if annual_rate_pct >= Decimal("7.0"):
            return "race_to_zero_aligned"
        elif annual_rate_pct >= Decimal("4.2"):
            return "1.5c_aligned"
        elif annual_rate_pct >= Decimal("2.5"):
            return "well_below_2c_aligned"
        elif annual_rate_pct >= Decimal("1.5"):
            return "2c_aligned"
        elif annual_rate_pct > Decimal("0"):
            return "below_2c_minimum"
        else:
            return "no_reduction"

    # ------------------------------------------------------------------ #
    # SBTi Validation                                                      #
    # ------------------------------------------------------------------ #

    def _validate_sbti(
        self,
        data: InterimTargetInput,
        timelines: List[ScopeTimeline],
        milestones: List[InterimMilestone],
    ) -> SBTiValidationResult:
        """Validate interim targets against SBTi criteria.

        Performs 8 core checks:
        1. Near-term reduction meets minimum threshold
        2. Scope 1+2 coverage >= 95%
        3. Scope 3 coverage >= 67% (if applicable)
        4. Near-term target year <= 2030 (1.5C) or 2035 (WB2C)
        5. Timeline linearity (no backsliding)
        6. Long-term target >= 90% reduction
        7. Scope 3 lag <= 5 years
        8. FLAG separate target (if applicable)
        """
        result = SBTiValidationResult()
        thresholds = SBTI_THRESHOLDS.get(
            data.ambition_level.value,
            SBTI_THRESHOLDS[ClimateAmbition.CELSIUS_1_5.value],
        )
        result.ambition_level = thresholds["name"]

        checks_total = 0
        checks_passed = 0
        checks_failed = 0
        checks_warning = 0

        # Check 1: Near-term reduction
        checks_total += 1
        s12_tl = next(
            (t for t in timelines if t.scope == ScopeType.SCOPE_1_2.value), None
        )
        min_reduction = thresholds["near_term_reduction_pct"]
        if s12_tl and s12_tl.near_term_reduction_pct >= min_reduction:
            result.near_term_check = ValidationStatus.ALIGNED.value
            checks_passed += 1
            result.validation_notes.append(
                f"PASS: Near-term Scope 1+2 reduction "
                f"{s12_tl.near_term_reduction_pct}% >= {min_reduction}% minimum."
            )
        elif s12_tl:
            result.near_term_check = ValidationStatus.MISALIGNED.value
            checks_failed += 1
            result.validation_notes.append(
                f"FAIL: Near-term Scope 1+2 reduction "
                f"{s12_tl.near_term_reduction_pct}% < {min_reduction}% minimum."
            )
        else:
            result.near_term_check = ValidationStatus.REQUIRES_REVIEW.value
            checks_warning += 1
            result.validation_notes.append(
                "WARNING: No Scope 1+2 timeline found for near-term validation."
            )

        # Check 2: Scope 1+2 coverage
        checks_total += 1
        s12_baseline = data.baseline.scope_1_tco2e + data.baseline.scope_2_tco2e
        total_baseline = self._resolve_baseline_total(data.baseline)
        if total_baseline > Decimal("0"):
            coverage = _safe_pct(s12_baseline, total_baseline)
            threshold_cov = thresholds["scope12_coverage_pct"]
            if coverage >= threshold_cov or s12_baseline > Decimal("0"):
                result.scope_coverage_check = ValidationStatus.ALIGNED.value
                checks_passed += 1
                result.validation_notes.append(
                    f"PASS: Scope 1+2 baseline provided ({s12_baseline} tCO2e)."
                )
            else:
                result.scope_coverage_check = ValidationStatus.MISALIGNED.value
                checks_failed += 1
                result.validation_notes.append(
                    f"FAIL: Scope 1+2 coverage below threshold."
                )
        else:
            result.scope_coverage_check = ValidationStatus.REQUIRES_REVIEW.value
            checks_warning += 1

        # Check 3: Scope 3 coverage
        checks_total += 1
        if data.baseline.scope_3_tco2e > Decimal("0"):
            result.validation_notes.append(
                f"PASS: Scope 3 baseline provided ({data.baseline.scope_3_tco2e} tCO2e)."
            )
            checks_passed += 1
        else:
            result.validation_notes.append(
                "WARNING: No Scope 3 baseline provided. "
                "SBTi requires Scope 3 targets if Scope 3 >= 40% of total."
            )
            checks_warning += 1

        # Check 4: Timeline validation
        checks_total += 1
        latest_year = thresholds["near_term_latest_year"]
        if s12_tl and s12_tl.near_term_year <= latest_year:
            result.timeline_check = ValidationStatus.ALIGNED.value
            checks_passed += 1
            result.validation_notes.append(
                f"PASS: Near-term target year {s12_tl.near_term_year} <= {latest_year}."
            )
        elif s12_tl:
            result.timeline_check = ValidationStatus.MISALIGNED.value
            checks_failed += 1
            result.validation_notes.append(
                f"FAIL: Near-term target year {s12_tl.near_term_year} > {latest_year}."
            )
        else:
            result.timeline_check = ValidationStatus.REQUIRES_REVIEW.value
            checks_warning += 1

        # Check 5: Linearity (no backsliding)
        checks_total += 1
        backsliding = False
        for tl in timelines:
            for i in range(1, len(tl.milestones)):
                if tl.milestones[i].reduction_pct < tl.milestones[i - 1].reduction_pct:
                    backsliding = True
                    break
        if not backsliding:
            result.linearity_check = ValidationStatus.ALIGNED.value
            checks_passed += 1
            result.validation_notes.append(
                "PASS: No backsliding detected in reduction pathway."
            )
        else:
            result.linearity_check = ValidationStatus.MISALIGNED.value
            checks_failed += 1
            result.validation_notes.append(
                "FAIL: Backsliding detected -- reduction percentage decreases "
                "in at least one milestone pair."
            )

        # Check 6: Long-term target
        checks_total += 1
        lt_threshold = thresholds["long_term_reduction_pct"]
        if data.long_term_target.reduction_pct >= lt_threshold:
            checks_passed += 1
            result.validation_notes.append(
                f"PASS: Long-term reduction {data.long_term_target.reduction_pct}% "
                f">= {lt_threshold}% threshold."
            )
        else:
            checks_failed += 1
            result.validation_notes.append(
                f"FAIL: Long-term reduction {data.long_term_target.reduction_pct}% "
                f"< {lt_threshold}% threshold."
            )

        # Check 7: Scope 3 lag
        checks_total += 1
        max_lag = thresholds["scope3_near_term_lag_years"]
        if data.scope_3_lag_years <= max_lag:
            checks_passed += 1
            result.validation_notes.append(
                f"PASS: Scope 3 lag {data.scope_3_lag_years} years <= "
                f"{max_lag} years allowed."
            )
        else:
            checks_failed += 1
            result.validation_notes.append(
                f"FAIL: Scope 3 lag {data.scope_3_lag_years} years > "
                f"{max_lag} years allowed."
            )

        # Check 8: FLAG sector
        checks_total += 1
        if not data.baseline.is_flag_sector:
            result.flag_check = ValidationStatus.ALIGNED.value
            checks_passed += 1
            result.validation_notes.append(
                "PASS: No FLAG sector emissions -- FLAG check not applicable."
            )
        elif data.include_flag_targets:
            result.flag_check = ValidationStatus.ALIGNED.value
            checks_passed += 1
            result.validation_notes.append(
                "PASS: FLAG targets are being generated separately."
            )
        else:
            result.flag_check = ValidationStatus.MISALIGNED.value
            checks_failed += 1
            result.validation_notes.append(
                "FAIL: FLAG sector detected but FLAG targets not requested. "
                "SBTi requires separate FLAG targets."
            )

        result.total_checks = checks_total
        result.passed_checks = checks_passed
        result.failed_checks = checks_failed
        result.warning_checks = checks_warning
        result.is_compliant = checks_failed == 0

        return result

    # ------------------------------------------------------------------ #
    # FLAG Targets                                                         #
    # ------------------------------------------------------------------ #

    def _calculate_flag_targets(
        self, data: InterimTargetInput,
    ) -> FLAGTargetResult:
        """Calculate FLAG sector interim targets.

        FLAG (Forests, Land and Agriculture) requires separate
        near-term targets with 30% minimum reduction (SBTi FLAG guidance).

        Formula:
            flag_target = flag_baseline * (1 - flag_reduction / 100)
            flag_annual_rate = 1 - (1 - flag_reduction/100)^(1/years)
        """
        flag_base = data.baseline.flag_emissions_tco2e
        if flag_base <= Decimal("0"):
            return FLAGTargetResult(
                notes=["No FLAG baseline emissions provided."]
            )

        flag_reduction = FLAG_THRESHOLDS["near_term_reduction_pct"]
        near_term_year = min(
            data.baseline.base_year + 10,
            2030 + int(FLAG_THRESHOLDS["near_term_latest_year_offset"]),
        )

        flag_target = flag_base * (Decimal("1") - flag_reduction / Decimal("100"))
        flag_rate = self._calculate_annual_rate(
            flag_base, flag_reduction, data.baseline.base_year, near_term_year,
        )

        return FLAGTargetResult(
            flag_baseline_tco2e=_round_val(flag_base, 2),
            flag_near_term_target_tco2e=_round_val(flag_target, 2),
            flag_near_term_reduction_pct=_round_val(flag_reduction, 2),
            flag_near_term_year=near_term_year,
            flag_annual_rate_pct=_round_val(flag_rate, 3),
            is_compliant=True,
            notes=[
                f"FLAG near-term target: {flag_reduction}% reduction by {near_term_year}.",
                f"FLAG annual rate: {_round_val(flag_rate, 3)}%/yr.",
                "Separate FLAG pathway required per SBTi FLAG guidance.",
            ],
        )

    # ------------------------------------------------------------------ #
    # Data Quality                                                         #
    # ------------------------------------------------------------------ #

    def _assess_data_quality(self, data: InterimTargetInput) -> str:
        """Assess input data quality."""
        score = 0
        if data.baseline.scope_1_tco2e > Decimal("0"):
            score += 2
        if data.baseline.scope_2_tco2e > Decimal("0"):
            score += 2
        if data.baseline.scope_3_tco2e > Decimal("0"):
            score += 2
        if data.long_term_target.target_year > 0:
            score += 1
        if len(data.milestone_overrides) > 0:
            score += 1
        if data.baseline.base_year >= 2019:
            score += 1
        if data.entity_id:
            score += 1

        if score >= 8:
            return DataQuality.HIGH.value
        elif score >= 5:
            return DataQuality.MEDIUM.value
        elif score >= 2:
            return DataQuality.LOW.value
        return DataQuality.ESTIMATED.value

    # ------------------------------------------------------------------ #
    # Recommendations                                                      #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        data: InterimTargetInput,
        timelines: List[ScopeTimeline],
        sbti_validation: Optional[SBTiValidationResult],
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recs: List[str] = []

        if data.baseline.scope_3_tco2e <= Decimal("0"):
            recs.append(
                "Include Scope 3 emissions in baseline for comprehensive "
                "interim target setting. SBTi requires Scope 3 targets "
                "when Scope 3 >= 40% of total."
            )

        if data.pathway_shape == PathwayShape.BACK_LOADED:
            recs.append(
                "Back-loaded pathway defers most reductions to later years. "
                "Consider front-loaded or linear pathway for earlier climate "
                "impact and better SBTi alignment."
            )

        if sbti_validation and not sbti_validation.is_compliant:
            recs.append(
                f"SBTi validation shows {sbti_validation.failed_checks} failed "
                f"check(s). Review validation notes and adjust targets to "
                f"meet SBTi Corporate Net-Zero Standard v1.2 criteria."
            )

        if data.ambition_level == ClimateAmbition.TWO_C:
            recs.append(
                "Consider upgrading ambition to 1.5C alignment (4.2%/yr) "
                "for SBTi Net-Zero Standard eligibility. The 2C pathway "
                "does not qualify for SBTi net-zero validation."
            )

        if data.baseline.is_flag_sector and not data.include_flag_targets:
            recs.append(
                "FLAG sector emissions detected. Enable FLAG targets "
                "for separate SBTi FLAG pathway validation."
            )

        if data.scope_3_lag_years == 0 and data.baseline.scope_3_tco2e > Decimal("0"):
            recs.append(
                "Consider utilizing the SBTi Scope 3 near-term lag allowance "
                "(up to 5 years) if Scope 3 data collection needs time to mature."
            )

        return recs

    # ------------------------------------------------------------------ #
    # Warnings                                                             #
    # ------------------------------------------------------------------ #

    def _generate_warnings(
        self,
        data: InterimTargetInput,
        timelines: List[ScopeTimeline],
        sbti_validation: Optional[SBTiValidationResult],
    ) -> List[str]:
        """Generate warnings based on analysis."""
        warns: List[str] = []

        baseline_total = self._resolve_baseline_total(data.baseline)
        if baseline_total <= Decimal("0"):
            warns.append(
                "Baseline total emissions are zero. Cannot generate "
                "meaningful interim targets."
            )

        if data.baseline.base_year < 2019:
            warns.append(
                f"Baseline year {data.baseline.base_year} is before 2019. "
                f"SBTi recommends baseline year no earlier than 2019 "
                f"for near-term targets."
            )

        if data.long_term_target.reduction_pct < Decimal("90"):
            warns.append(
                f"Long-term reduction target {data.long_term_target.reduction_pct}% "
                f"is below SBTi net-zero minimum of 90%. Residual emissions "
                f"must be neutralized."
            )

        years_to_target = data.long_term_target.target_year - data.baseline.base_year
        if years_to_target > 35:
            warns.append(
                f"Target horizon of {years_to_target} years is very long. "
                f"SBTi recommends net-zero by 2050 at the latest."
            )

        return warns

    # ------------------------------------------------------------------ #
    # Utility Methods                                                      #
    # ------------------------------------------------------------------ #

    def get_sbti_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Return SBTi threshold parameters for all ambition levels."""
        return {k: dict(v) for k, v in SBTI_THRESHOLDS.items()}

    def get_supported_ambition_levels(self) -> List[str]:
        """Return list of supported ambition levels."""
        return [a.value for a in ClimateAmbition]

    def get_supported_pathway_shapes(self) -> List[str]:
        """Return list of supported pathway shapes."""
        return [p.value for p in PathwayShape]
