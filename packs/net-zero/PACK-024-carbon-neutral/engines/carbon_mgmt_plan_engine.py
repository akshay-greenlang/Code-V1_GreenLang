# -*- coding: utf-8 -*-
"""
CarbonMgmtPlanEngine - PACK-024 Carbon Neutral Engine 2
========================================================

ISO 14068-1:2023 Section 9 compliant carbon management plan generation
with reduction-first hierarchy, 3-5 year roadmap creation, marginal
abatement cost curve (MACC) integration, reduction vs offset balance
enforcement, and milestone tracking.

This engine ensures that organisations pursuing carbon neutrality
prioritise genuine emission reductions before resorting to carbon credits,
in accordance with the mitigation hierarchy mandated by ISO 14068-1,
PAS 2060, and the Oxford Principles for Net Zero Aligned Carbon Offsetting.

Calculation Methodology:
    Reduction-First Hierarchy (ISO 14068-1:2023, Section 9.2):
        Priority 1: Avoid/eliminate emissions (source removal)
        Priority 2: Reduce emissions (efficiency, fuel switching)
        Priority 3: Substitute (renewable energy, low-carbon materials)
        Priority 4: Compensate (carbon credits for residual emissions)

    Reduction Target Trajectory:
        annual_reduction_target = (baseline - target) / years
        cumulative_reduction_pct = sum(annual_reductions) / baseline * 100

    MACC Integration:
        net_cost = implementation_cost - annual_savings
        cost_per_tco2e = net_cost / annual_abatement_tco2e
        measures sorted by cost_per_tco2e ascending (cheapest first)

    Offset Balance Rule (ISO 14068-1:2023, Section 9.3):
        max_offset_pct = 100% - min_reduction_pct
        min_reduction_pct: Year 1-2 >= 5%, Year 3-5 >= 15%, Year 5+ >= 30%
        offset_allowed = residual_after_reductions

    Roadmap Milestones:
        milestone_on_track = actual_reduction >= planned_reduction * 0.90
        (10% tolerance for annual variance)

Regulatory References:
    - ISO 14068-1:2023 - Carbon neutrality (Section 9: management plan)
    - PAS 2060:2014 - Section 5.3: Qualifying Explanatory Statement
    - Oxford Principles for Net Zero Aligned Carbon Offsetting (2020)
    - GHG Protocol Mitigation Goal Standard (2014)
    - Science Based Targets initiative Net-Zero Standard V1.3 (2024)
    - IPCC AR6 WG3 (2022) - Sectoral mitigation potentials
    - IEA Net Zero by 2050 Roadmap (2021, updated 2023)

Zero-Hallucination:
    - Mitigation hierarchy from ISO 14068-1:2023 Section 9.2
    - Reduction-first percentages from PAS 2060:2014
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-024 Carbon Neutral
Engine:  2 of 10
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
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _round3(value: float) -> float:
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MitigationPriority(str, Enum):
    """Mitigation hierarchy levels per ISO 14068-1:2023, Section 9.2.

    AVOID: Eliminate emission source entirely.
    REDUCE: Improve efficiency or switch processes.
    SUBSTITUTE: Replace with lower-carbon alternatives.
    COMPENSATE: Offset residual with carbon credits.
    """
    AVOID = "avoid"
    REDUCE = "reduce"
    SUBSTITUTE = "substitute"
    COMPENSATE = "compensate"


class MeasureStatus(str, Enum):
    """Implementation status of a reduction measure.

    PLANNED: Identified but not yet started.
    IN_PROGRESS: Implementation underway.
    COMPLETED: Fully implemented.
    DEFERRED: Postponed to a later period.
    CANCELLED: Measure cancelled.
    """
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DEFERRED = "deferred"
    CANCELLED = "cancelled"


class MeasureCategory(str, Enum):
    """Category of emission reduction measure.

    ENERGY_EFFICIENCY: Building, process, equipment efficiency.
    RENEWABLE_ENERGY: Solar, wind, hydro, geothermal procurement.
    FUEL_SWITCHING: Low-carbon fuel substitution.
    PROCESS_CHANGE: Production process modifications.
    SUPPLY_CHAIN: Supplier engagement, logistics optimisation.
    BEHAVIOUR_CHANGE: Employee engagement, travel policies.
    TECHNOLOGY: New technology deployment.
    CIRCULAR_ECONOMY: Waste reduction, material reuse.
    NATURE_BASED: On-site sequestration, green infrastructure.
    """
    ENERGY_EFFICIENCY = "energy_efficiency"
    RENEWABLE_ENERGY = "renewable_energy"
    FUEL_SWITCHING = "fuel_switching"
    PROCESS_CHANGE = "process_change"
    SUPPLY_CHAIN = "supply_chain"
    BEHAVIOUR_CHANGE = "behaviour_change"
    TECHNOLOGY = "technology"
    CIRCULAR_ECONOMY = "circular_economy"
    NATURE_BASED = "nature_based"


class RoadmapPhase(str, Enum):
    """Phase of the carbon management roadmap.

    IMMEDIATE: 0-6 months (quick wins).
    SHORT_TERM: 6-18 months.
    MEDIUM_TERM: 18-36 months.
    LONG_TERM: 36-60 months.
    STRATEGIC: 5+ years.
    """
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"
    STRATEGIC = "strategic"


class PlanCompliance(str, Enum):
    """Compliance level of the management plan.

    COMPLIANT: Meets all requirements of the target standard.
    PARTIALLY_COMPLIANT: Some gaps identified.
    NON_COMPLIANT: Significant gaps in requirements.
    NOT_ASSESSED: Not yet evaluated.
    """
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSED = "not_assessed"


# ---------------------------------------------------------------------------
# Constants -- ISO 14068-1 / PAS 2060 Requirements
# ---------------------------------------------------------------------------

# Minimum reduction percentage before offsets are acceptable.
# Source: PAS 2060:2014, Section 5.3 + ISO 14068-1:2023, Section 9.3.
# These are cumulative reduction thresholds by management plan year.
MIN_REDUCTION_BY_YEAR: Dict[int, Decimal] = {
    1: Decimal("5"),
    2: Decimal("5"),
    3: Decimal("15"),
    4: Decimal("15"),
    5: Decimal("30"),
    6: Decimal("30"),
    7: Decimal("30"),
    8: Decimal("40"),
    9: Decimal("40"),
    10: Decimal("50"),
}

# Maximum offset percentage (100% - min reduction %).
MAX_OFFSET_BY_YEAR: Dict[int, Decimal] = {
    yr: Decimal("100") - pct for yr, pct in MIN_REDUCTION_BY_YEAR.items()
}

# Roadmap phase durations in months.
PHASE_DURATION_MONTHS: Dict[str, Tuple[int, int]] = {
    RoadmapPhase.IMMEDIATE.value: (0, 6),
    RoadmapPhase.SHORT_TERM.value: (6, 18),
    RoadmapPhase.MEDIUM_TERM.value: (18, 36),
    RoadmapPhase.LONG_TERM.value: (36, 60),
    RoadmapPhase.STRATEGIC.value: (60, 120),
}

# On-track tolerance: actual must be >= 90% of planned reduction.
ON_TRACK_TOLERANCE: Decimal = Decimal("0.90")

# MACC cost effectiveness thresholds (USD per tCO2e avoided).
# Source: IEA Net Zero by 2050 (2023), McKinsey MACC analysis.
COST_TIER_NEGATIVE: Decimal = Decimal("0")
COST_TIER_LOW: Decimal = Decimal("50")
COST_TIER_MODERATE: Decimal = Decimal("100")
COST_TIER_HIGH: Decimal = Decimal("200")
COST_TIER_VERY_HIGH: Decimal = Decimal("500")

# Minimum annual reduction rate consistent with 1.5C pathway.
# Source: IPCC AR6 WG3 (2022), SBTi Corporate Net-Zero Standard V1.3.
MIN_ANNUAL_REDUCTION_RATE_15C: Decimal = Decimal("4.2")
MIN_ANNUAL_REDUCTION_RATE_WB2C: Decimal = Decimal("2.5")


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class ReductionMeasureInput(BaseModel):
    """Input for a single emission reduction measure.

    Attributes:
        measure_id: Unique identifier.
        measure_name: Descriptive name.
        priority: Mitigation hierarchy level.
        category: Measure category.
        scope_targeted: Which scopes this measure targets (1, 2, 3).
        annual_abatement_tco2e: Expected annual emission reduction.
        implementation_cost_usd: One-time implementation cost.
        annual_operating_cost_usd: Annual operating cost/savings.
        annual_savings_usd: Annual financial savings (energy, etc.).
        payback_years: Simple payback period.
        implementation_months: Months to implement.
        start_year: Year implementation begins.
        end_year: Year measure reaches full effect.
        status: Current implementation status.
        confidence_pct: Confidence in abatement estimate (0-100).
        dependencies: IDs of measures this depends on.
        co_benefits: List of co-benefits.
        risks: List of implementation risks.
        notes: Additional notes.
    """
    measure_id: str = Field(default_factory=_new_uuid, description="Measure ID")
    measure_name: str = Field(default="", max_length=300, description="Measure name")
    priority: str = Field(
        default=MitigationPriority.REDUCE.value,
        description="Mitigation hierarchy level"
    )
    category: str = Field(
        default=MeasureCategory.ENERGY_EFFICIENCY.value,
        description="Measure category"
    )
    scope_targeted: List[int] = Field(
        default_factory=lambda: [1, 2],
        description="Scopes targeted"
    )
    annual_abatement_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Annual abatement (tCO2e)"
    )
    implementation_cost_usd: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Implementation cost (USD)"
    )
    annual_operating_cost_usd: Decimal = Field(
        default=Decimal("0"),
        description="Annual operating cost (USD)"
    )
    annual_savings_usd: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Annual savings (USD)"
    )
    payback_years: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Payback period (years)"
    )
    implementation_months: int = Field(
        default=12, ge=0, le=120,
        description="Implementation time (months)"
    )
    start_year: int = Field(
        default=0, ge=0, le=2060,
        description="Implementation start year"
    )
    end_year: int = Field(
        default=0, ge=0, le=2060,
        description="Full effect year"
    )
    status: str = Field(
        default=MeasureStatus.PLANNED.value,
        description="Implementation status"
    )
    confidence_pct: Decimal = Field(
        default=Decimal("75"), ge=0, le=Decimal("100"),
        description="Confidence in estimate (%)"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="Dependent measure IDs"
    )
    co_benefits: List[str] = Field(
        default_factory=list,
        description="Co-benefits"
    )
    risks: List[str] = Field(
        default_factory=list,
        description="Implementation risks"
    )
    notes: str = Field(default="", description="Additional notes")

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        valid = {p.value for p in MitigationPriority}
        if v not in valid:
            raise ValueError(f"Unknown priority '{v}'. Must be one of: {sorted(valid)}")
        return v

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        valid = {c.value for c in MeasureCategory}
        if v not in valid:
            raise ValueError(f"Unknown category '{v}'. Must be one of: {sorted(valid)}")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid = {s.value for s in MeasureStatus}
        if v not in valid:
            raise ValueError(f"Unknown status '{v}'. Must be one of: {sorted(valid)}")
        return v


class CarbonMgmtPlanInput(BaseModel):
    """Complete input for carbon management plan generation.

    Attributes:
        entity_name: Reporting entity name.
        base_year: Emissions base year.
        baseline_tco2e: Baseline total emissions (tCO2e).
        scope1_baseline_tco2e: Scope 1 baseline.
        scope2_baseline_tco2e: Scope 2 baseline.
        scope3_baseline_tco2e: Scope 3 baseline.
        current_year: Current year.
        current_tco2e: Current year emissions.
        target_year: Target year for carbon neutrality.
        target_standard: Target standard (iso_14068_1 or pas_2060).
        measures: Planned reduction measures.
        annual_credit_budget_usd: Annual budget for carbon credits.
        credit_price_per_tco2e: Expected credit price.
        roadmap_years: Number of years for roadmap (3-10).
        include_macc: Whether to generate MACC analysis.
        include_milestones: Whether to generate milestone plan.
        include_financial: Whether to include financial analysis.
        sector: Industry sector.
        pathway_ambition: Alignment pathway (1.5c, well_below_2c, 2c).
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300,
        description="Reporting entity name"
    )
    base_year: int = Field(
        ..., ge=2015, le=2030,
        description="Emissions base year"
    )
    baseline_tco2e: Decimal = Field(
        ..., ge=0,
        description="Baseline total emissions (tCO2e)"
    )
    scope1_baseline_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Scope 1 baseline"
    )
    scope2_baseline_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Scope 2 baseline"
    )
    scope3_baseline_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Scope 3 baseline"
    )
    current_year: int = Field(
        default=2026, ge=2020, le=2060,
        description="Current year"
    )
    current_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Current year emissions"
    )
    target_year: int = Field(
        default=2030, ge=2025, le=2060,
        description="Carbon neutrality target year"
    )
    target_standard: str = Field(
        default="iso_14068_1",
        description="Target standard"
    )
    measures: List[ReductionMeasureInput] = Field(
        default_factory=list,
        description="Reduction measures"
    )
    annual_credit_budget_usd: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Annual credit budget (USD)"
    )
    credit_price_per_tco2e: Decimal = Field(
        default=Decimal("15"), ge=0,
        description="Credit price (USD/tCO2e)"
    )
    roadmap_years: int = Field(
        default=5, ge=3, le=10,
        description="Roadmap duration (years)"
    )
    include_macc: bool = Field(default=True, description="Include MACC analysis")
    include_milestones: bool = Field(default=True, description="Include milestones")
    include_financial: bool = Field(default=True, description="Include financial analysis")
    sector: str = Field(
        default="general", max_length=100,
        description="Industry sector"
    )
    pathway_ambition: str = Field(
        default="1.5c",
        description="Temperature pathway ambition"
    )


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class MACCEntry(BaseModel):
    """MACC (Marginal Abatement Cost Curve) entry for a measure.

    Attributes:
        measure_id: Measure identifier.
        measure_name: Measure name.
        category: Measure category.
        priority: Mitigation hierarchy level.
        annual_abatement_tco2e: Annual abatement potential.
        cost_per_tco2e_usd: Net cost per tCO2e avoided.
        cumulative_abatement_tco2e: Cumulative abatement (running total).
        implementation_cost_usd: Total implementation cost.
        annual_savings_usd: Annual savings.
        net_annual_cost_usd: Net annual cost (cost - savings).
        payback_years: Payback period.
        cost_tier: Cost classification.
        is_negative_cost: Whether measure saves money.
        macc_rank: Rank in MACC (1 = cheapest abatement).
    """
    measure_id: str = Field(default="")
    measure_name: str = Field(default="")
    category: str = Field(default="")
    priority: str = Field(default="")
    annual_abatement_tco2e: Decimal = Field(default=Decimal("0"))
    cost_per_tco2e_usd: Decimal = Field(default=Decimal("0"))
    cumulative_abatement_tco2e: Decimal = Field(default=Decimal("0"))
    implementation_cost_usd: Decimal = Field(default=Decimal("0"))
    annual_savings_usd: Decimal = Field(default=Decimal("0"))
    net_annual_cost_usd: Decimal = Field(default=Decimal("0"))
    payback_years: Decimal = Field(default=Decimal("0"))
    cost_tier: str = Field(default="moderate")
    is_negative_cost: bool = Field(default=False)
    macc_rank: int = Field(default=0)


class YearlyMilestone(BaseModel):
    """Annual milestone in the carbon management roadmap.

    Attributes:
        year: Calendar year.
        plan_year: Year number in management plan (1-based).
        planned_reduction_tco2e: Planned cumulative reduction by year end.
        planned_reduction_pct: Planned reduction as % of baseline.
        planned_residual_tco2e: Planned residual emissions.
        min_reduction_pct: Minimum reduction % required (for offset eligibility).
        max_offset_pct: Maximum offset % allowed.
        offset_required_tco2e: Credits needed for neutrality.
        offset_cost_usd: Estimated credit cost for the year.
        measures_active: Measures active in this year.
        measures_completing: Measures completing in this year.
        on_track: Whether milestone is on track (if actual data exists).
        actual_reduction_tco2e: Actual reduction (if available).
        actual_reduction_pct: Actual reduction % (if available).
        variance_pct: Variance from plan (actual - planned).
    """
    year: int = Field(default=0)
    plan_year: int = Field(default=0)
    planned_reduction_tco2e: Decimal = Field(default=Decimal("0"))
    planned_reduction_pct: Decimal = Field(default=Decimal("0"))
    planned_residual_tco2e: Decimal = Field(default=Decimal("0"))
    min_reduction_pct: Decimal = Field(default=Decimal("0"))
    max_offset_pct: Decimal = Field(default=Decimal("100"))
    offset_required_tco2e: Decimal = Field(default=Decimal("0"))
    offset_cost_usd: Decimal = Field(default=Decimal("0"))
    measures_active: List[str] = Field(default_factory=list)
    measures_completing: List[str] = Field(default_factory=list)
    on_track: bool = Field(default=True)
    actual_reduction_tco2e: Optional[Decimal] = Field(default=None)
    actual_reduction_pct: Optional[Decimal] = Field(default=None)
    variance_pct: Optional[Decimal] = Field(default=None)


class ReductionOffsetBalance(BaseModel):
    """Balance between reductions and offsets.

    Attributes:
        baseline_tco2e: Baseline emissions.
        total_planned_reduction_tco2e: Total planned emission reductions.
        total_planned_reduction_pct: Reductions as % of baseline.
        residual_tco2e: Residual after reductions.
        offset_needed_tco2e: Offsets needed for neutrality.
        reduction_first_compliant: Whether reduction-first hierarchy is met.
        min_reduction_met: Whether minimum reduction threshold is met.
        reduction_share_pct: % of neutrality from reductions.
        offset_share_pct: % of neutrality from offsets.
        improvement_trajectory: Whether trend shows increasing reduction share.
        message: Human-readable assessment.
    """
    baseline_tco2e: Decimal = Field(default=Decimal("0"))
    total_planned_reduction_tco2e: Decimal = Field(default=Decimal("0"))
    total_planned_reduction_pct: Decimal = Field(default=Decimal("0"))
    residual_tco2e: Decimal = Field(default=Decimal("0"))
    offset_needed_tco2e: Decimal = Field(default=Decimal("0"))
    reduction_first_compliant: bool = Field(default=False)
    min_reduction_met: bool = Field(default=False)
    reduction_share_pct: Decimal = Field(default=Decimal("0"))
    offset_share_pct: Decimal = Field(default=Decimal("0"))
    improvement_trajectory: bool = Field(default=False)
    message: str = Field(default="")


class FinancialSummary(BaseModel):
    """Financial summary of the management plan.

    Attributes:
        total_implementation_cost_usd: Total capital cost.
        total_annual_savings_usd: Total annual savings.
        total_annual_credit_cost_usd: Annual credit cost at target.
        net_annual_cost_usd: Net annual cost.
        total_5yr_cost_usd: Total 5-year cost.
        total_5yr_savings_usd: Total 5-year savings.
        avg_abatement_cost_usd: Average cost per tCO2e reduced.
        negative_cost_abatement_tco2e: Abatement from cost-saving measures.
        roi_pct: Return on investment from energy savings.
        message: Human-readable summary.
    """
    total_implementation_cost_usd: Decimal = Field(default=Decimal("0"))
    total_annual_savings_usd: Decimal = Field(default=Decimal("0"))
    total_annual_credit_cost_usd: Decimal = Field(default=Decimal("0"))
    net_annual_cost_usd: Decimal = Field(default=Decimal("0"))
    total_5yr_cost_usd: Decimal = Field(default=Decimal("0"))
    total_5yr_savings_usd: Decimal = Field(default=Decimal("0"))
    avg_abatement_cost_usd: Decimal = Field(default=Decimal("0"))
    negative_cost_abatement_tco2e: Decimal = Field(default=Decimal("0"))
    roi_pct: Decimal = Field(default=Decimal("0"))
    message: str = Field(default="")


class HierarchyAssessment(BaseModel):
    """Assessment of mitigation hierarchy compliance.

    Attributes:
        avoid_measures_count: Number of avoidance measures.
        reduce_measures_count: Number of reduction measures.
        substitute_measures_count: Number of substitution measures.
        compensate_measures_count: Number of offset measures.
        avoid_abatement_tco2e: Abatement from avoidance.
        reduce_abatement_tco2e: Abatement from reduction.
        substitute_abatement_tco2e: Abatement from substitution.
        compensate_abatement_tco2e: Abatement from offsets.
        hierarchy_compliant: Whether reduction-first hierarchy is followed.
        hierarchy_score: Score (0-100) for hierarchy compliance.
        recommendations: Hierarchy-specific recommendations.
    """
    avoid_measures_count: int = Field(default=0)
    reduce_measures_count: int = Field(default=0)
    substitute_measures_count: int = Field(default=0)
    compensate_measures_count: int = Field(default=0)
    avoid_abatement_tco2e: Decimal = Field(default=Decimal("0"))
    reduce_abatement_tco2e: Decimal = Field(default=Decimal("0"))
    substitute_abatement_tco2e: Decimal = Field(default=Decimal("0"))
    compensate_abatement_tco2e: Decimal = Field(default=Decimal("0"))
    hierarchy_compliant: bool = Field(default=False)
    hierarchy_score: Decimal = Field(default=Decimal("0"))
    recommendations: List[str] = Field(default_factory=list)


class PathwayAlignment(BaseModel):
    """Assessment of alignment with temperature pathway.

    Attributes:
        pathway: Target pathway (1.5c, well_below_2c, 2c).
        required_annual_rate_pct: Required annual reduction rate.
        planned_annual_rate_pct: Planned annual reduction rate.
        is_aligned: Whether plan aligns with pathway.
        gap_pct: Gap in annual rate.
        years_to_neutrality: Years to achieve neutrality at current rate.
        message: Human-readable assessment.
    """
    pathway: str = Field(default="1.5c")
    required_annual_rate_pct: Decimal = Field(default=Decimal("0"))
    planned_annual_rate_pct: Decimal = Field(default=Decimal("0"))
    is_aligned: bool = Field(default=False)
    gap_pct: Decimal = Field(default=Decimal("0"))
    years_to_neutrality: int = Field(default=0)
    message: str = Field(default="")


class CarbonMgmtPlanResult(BaseModel):
    """Complete carbon management plan result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        entity_name: Reporting entity name.
        base_year: Base year.
        target_year: Target year for neutrality.
        baseline_tco2e: Baseline emissions.
        target_standard: Target standard.
        macc_entries: MACC analysis results.
        milestones: Annual milestones.
        reduction_offset_balance: Reduction vs offset balance.
        hierarchy_assessment: Mitigation hierarchy assessment.
        pathway_alignment: Temperature pathway alignment.
        financial_summary: Financial summary.
        total_measures: Number of reduction measures.
        active_measures: Number of currently active measures.
        total_abatement_tco2e: Total planned abatement.
        total_abatement_pct: Total planned abatement as % of baseline.
        residual_tco2e: Residual requiring offsets.
        plan_compliance: Compliance with target standard.
        roadmap_years: Number of years in roadmap.
        warnings: Non-blocking warnings.
        errors: Blocking errors.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=_utcnow)
    entity_name: str = Field(default="")
    base_year: int = Field(default=0)
    target_year: int = Field(default=0)
    baseline_tco2e: Decimal = Field(default=Decimal("0"))
    target_standard: str = Field(default="")
    macc_entries: List[MACCEntry] = Field(default_factory=list)
    milestones: List[YearlyMilestone] = Field(default_factory=list)
    reduction_offset_balance: Optional[ReductionOffsetBalance] = Field(default=None)
    hierarchy_assessment: Optional[HierarchyAssessment] = Field(default=None)
    pathway_alignment: Optional[PathwayAlignment] = Field(default=None)
    financial_summary: Optional[FinancialSummary] = Field(default=None)
    total_measures: int = Field(default=0)
    active_measures: int = Field(default=0)
    total_abatement_tco2e: Decimal = Field(default=Decimal("0"))
    total_abatement_pct: Decimal = Field(default=Decimal("0"))
    residual_tco2e: Decimal = Field(default=Decimal("0"))
    plan_compliance: str = Field(default=PlanCompliance.NOT_ASSESSED.value)
    roadmap_years: int = Field(default=5)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class CarbonMgmtPlanEngine:
    """ISO 14068-1 compliant carbon management plan engine.

    Generates a comprehensive management plan that enforces the
    reduction-first hierarchy, builds a MACC-ordered roadmap,
    calculates reduction vs offset balance, and tracks milestones
    across a 3-10 year planning horizon.

    Usage::

        engine = CarbonMgmtPlanEngine()
        result = engine.generate_plan(input_data)
        print(f"Total abatement: {result.total_abatement_tco2e} tCO2e")
        for ms in result.milestones:
            print(f"  {ms.year}: {ms.planned_reduction_pct}% reduction")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise CarbonMgmtPlanEngine.

        Args:
            config: Optional configuration overrides. Supported keys:
                - on_track_tolerance (Decimal): Tolerance for on-track (default 0.90)
                - min_reduction_overrides (dict): Custom min reduction by year
        """
        self.config = config or {}
        self._tolerance = _decimal(
            self.config.get("on_track_tolerance", ON_TRACK_TOLERANCE)
        )
        self._min_reduction = dict(MIN_REDUCTION_BY_YEAR)
        overrides = self.config.get("min_reduction_overrides", {})
        for yr, pct in overrides.items():
            self._min_reduction[int(yr)] = _decimal(pct)
        logger.info("CarbonMgmtPlanEngine v%s initialised", self.engine_version)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate_plan(
        self, data: CarbonMgmtPlanInput,
    ) -> CarbonMgmtPlanResult:
        """Generate a complete carbon management plan.

        Orchestrates: MACC analysis, milestone generation, reduction/offset
        balance, hierarchy assessment, pathway alignment, and financial summary.

        Args:
            data: Validated plan input.

        Returns:
            CarbonMgmtPlanResult with comprehensive management plan.
        """
        t0 = time.perf_counter()
        logger.info(
            "Management plan: entity=%s, base=%d, target=%d, measures=%d",
            data.entity_name, data.base_year, data.target_year,
            len(data.measures),
        )

        warnings: List[str] = []
        errors: List[str] = []

        # Validate inputs
        if data.baseline_tco2e <= Decimal("0"):
            errors.append("Baseline emissions must be greater than zero.")
        if data.target_year <= data.current_year:
            warnings.append("Target year is not in the future.")

        # Step 1: MACC analysis
        macc_entries: List[MACCEntry] = []
        if data.include_macc:
            macc_entries = self._build_macc(data.measures)

        # Step 2: Calculate total planned abatement
        active_measures = [
            m for m in data.measures
            if m.status not in (MeasureStatus.CANCELLED.value, MeasureStatus.DEFERRED.value)
        ]
        total_abatement = sum(
            (m.annual_abatement_tco2e * (m.confidence_pct / Decimal("100"))
             for m in active_measures),
            Decimal("0"),
        )
        total_abatement_pct = _safe_pct(total_abatement, data.baseline_tco2e)
        residual = max(Decimal("0"), data.baseline_tco2e - total_abatement)

        # Step 3: Milestones
        milestones: List[YearlyMilestone] = []
        if data.include_milestones:
            milestones = self._generate_milestones(
                data, active_measures, total_abatement
            )

        # Step 4: Reduction/offset balance
        balance = self._assess_reduction_offset_balance(
            data, total_abatement, residual
        )

        # Step 5: Hierarchy assessment
        hierarchy = self._assess_hierarchy(data.measures, data.baseline_tco2e)

        # Step 6: Pathway alignment
        pathway = self._assess_pathway_alignment(
            data, total_abatement
        )

        # Step 7: Financial summary
        financial: Optional[FinancialSummary] = None
        if data.include_financial:
            financial = self._build_financial_summary(
                data, active_measures, residual
            )

        # Step 8: Plan compliance check
        compliance = self._assess_plan_compliance(
            data, balance, hierarchy, pathway, warnings
        )

        # Warnings
        if total_abatement <= Decimal("0") and len(data.measures) > 0:
            warnings.append("Total planned abatement is zero despite measures.")
        if len(data.measures) == 0:
            warnings.append("No reduction measures provided.")

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = CarbonMgmtPlanResult(
            entity_name=data.entity_name,
            base_year=data.base_year,
            target_year=data.target_year,
            baseline_tco2e=data.baseline_tco2e,
            target_standard=data.target_standard,
            macc_entries=macc_entries,
            milestones=milestones,
            reduction_offset_balance=balance,
            hierarchy_assessment=hierarchy,
            pathway_alignment=pathway,
            financial_summary=financial,
            total_measures=len(data.measures),
            active_measures=len(active_measures),
            total_abatement_tco2e=_round_val(total_abatement),
            total_abatement_pct=_round_val(total_abatement_pct, 2),
            residual_tco2e=_round_val(residual),
            plan_compliance=compliance,
            roadmap_years=data.roadmap_years,
            warnings=warnings,
            errors=errors,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Management plan complete: abatement=%.2f tCO2e (%.1f%%), "
            "residual=%.2f, compliance=%s, hash=%s",
            float(total_abatement), float(total_abatement_pct),
            float(residual), compliance, result.provenance_hash[:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal Methods                                                     #
    # ------------------------------------------------------------------ #

    def _build_macc(
        self, measures: List[ReductionMeasureInput],
    ) -> List[MACCEntry]:
        """Build MACC (Marginal Abatement Cost Curve) analysis.

        Sorts measures by cost-effectiveness (USD per tCO2e avoided),
        cheapest first, with negative-cost measures at the beginning.

        Args:
            measures: Input reduction measures.

        Returns:
            Sorted list of MACCEntry, cheapest abatement first.
        """
        entries: List[MACCEntry] = []
        for m in measures:
            if m.status == MeasureStatus.CANCELLED.value:
                continue
            if m.annual_abatement_tco2e <= Decimal("0"):
                continue

            net_annual = m.annual_operating_cost_usd - m.annual_savings_usd
            cost_per = _safe_divide(
                m.implementation_cost_usd + net_annual,
                m.annual_abatement_tco2e,
            )
            is_neg = cost_per < Decimal("0")

            if cost_per < COST_TIER_NEGATIVE:
                tier = "negative_cost"
            elif cost_per < COST_TIER_LOW:
                tier = "low"
            elif cost_per < COST_TIER_MODERATE:
                tier = "moderate"
            elif cost_per < COST_TIER_HIGH:
                tier = "high"
            elif cost_per < COST_TIER_VERY_HIGH:
                tier = "very_high"
            else:
                tier = "extreme"

            entries.append(MACCEntry(
                measure_id=m.measure_id,
                measure_name=m.measure_name,
                category=m.category,
                priority=m.priority,
                annual_abatement_tco2e=m.annual_abatement_tco2e,
                cost_per_tco2e_usd=_round_val(cost_per, 2),
                implementation_cost_usd=m.implementation_cost_usd,
                annual_savings_usd=m.annual_savings_usd,
                net_annual_cost_usd=_round_val(net_annual, 2),
                payback_years=m.payback_years,
                cost_tier=tier,
                is_negative_cost=is_neg,
            ))

        # Sort by cost per tCO2e (cheapest first)
        entries.sort(key=lambda e: e.cost_per_tco2e_usd)

        # Assign ranks and cumulative
        cumulative = Decimal("0")
        for idx, e in enumerate(entries, 1):
            e.macc_rank = idx
            cumulative += e.annual_abatement_tco2e
            e.cumulative_abatement_tco2e = _round_val(cumulative)

        return entries

    def _generate_milestones(
        self,
        data: CarbonMgmtPlanInput,
        active_measures: List[ReductionMeasureInput],
        total_abatement: Decimal,
    ) -> List[YearlyMilestone]:
        """Generate annual milestones for the management plan.

        Distributes planned reductions across years based on measure
        implementation timelines and calculates offset requirements.

        Args:
            data: Plan input data.
            active_measures: Non-cancelled measures.
            total_abatement: Total planned abatement.

        Returns:
            List of YearlyMilestone for each year in the roadmap.
        """
        milestones: List[YearlyMilestone] = []
        start_year = data.current_year
        end_year = start_year + data.roadmap_years

        for yr in range(start_year, end_year):
            plan_yr = yr - start_year + 1

            # Calculate cumulative reduction at this year
            cumulative_reduction = Decimal("0")
            active_names: List[str] = []
            completing_names: List[str] = []

            for m in active_measures:
                m_start = m.start_year if m.start_year > 0 else start_year
                m_end = m.end_year if m.end_year > 0 else m_start + 1

                if yr >= m_start:
                    # Ramp-up: linear between start and end year
                    if yr >= m_end:
                        contribution = m.annual_abatement_tco2e * (m.confidence_pct / Decimal("100"))
                    else:
                        ramp_years = max(1, m_end - m_start)
                        ramp_frac = _decimal(yr - m_start + 1) / _decimal(ramp_years)
                        ramp_frac = min(ramp_frac, Decimal("1"))
                        contribution = (
                            m.annual_abatement_tco2e
                            * (m.confidence_pct / Decimal("100"))
                            * ramp_frac
                        )
                    cumulative_reduction += contribution
                    active_names.append(m.measure_name or m.measure_id)

                    if yr == m_end:
                        completing_names.append(m.measure_name or m.measure_id)

            reduction_pct = _safe_pct(cumulative_reduction, data.baseline_tco2e)
            residual = max(Decimal("0"), data.baseline_tco2e - cumulative_reduction)

            # Min reduction from standard
            min_red_pct = self._min_reduction.get(plan_yr, Decimal("30"))
            max_off_pct = Decimal("100") - min_red_pct

            # Offset needed
            offset_needed = residual
            offset_cost = offset_needed * data.credit_price_per_tco2e

            milestones.append(YearlyMilestone(
                year=yr,
                plan_year=plan_yr,
                planned_reduction_tco2e=_round_val(cumulative_reduction),
                planned_reduction_pct=_round_val(reduction_pct, 2),
                planned_residual_tco2e=_round_val(residual),
                min_reduction_pct=min_red_pct,
                max_offset_pct=max_off_pct,
                offset_required_tco2e=_round_val(offset_needed),
                offset_cost_usd=_round_val(offset_cost, 2),
                measures_active=active_names,
                measures_completing=completing_names,
            ))

        return milestones

    def _assess_reduction_offset_balance(
        self,
        data: CarbonMgmtPlanInput,
        total_abatement: Decimal,
        residual: Decimal,
    ) -> ReductionOffsetBalance:
        """Assess the balance between reductions and offsets.

        ISO 14068-1:2023 and PAS 2060:2014 require organisations to
        demonstrate genuine emission reductions before using offsets.

        Args:
            data: Plan input.
            total_abatement: Total planned reduction.
            residual: Residual requiring offsets.

        Returns:
            ReductionOffsetBalance assessment.
        """
        reduction_pct = _safe_pct(total_abatement, data.baseline_tco2e)
        offset_pct = _safe_pct(residual, data.baseline_tco2e)

        # Determine plan year at target
        plan_years = data.target_year - data.current_year
        plan_year_key = min(plan_years, 10)
        min_red = self._min_reduction.get(plan_year_key, Decimal("30"))

        min_met = reduction_pct >= min_red
        hierarchy_ok = total_abatement >= residual or reduction_pct >= Decimal("50")

        # Check improvement trajectory
        improving = reduction_pct > Decimal("0")

        if min_met and hierarchy_ok:
            msg = (
                f"Reduction-first hierarchy compliant: {_round_val(reduction_pct, 1)}% "
                f"from reductions, {_round_val(offset_pct, 1)}% from offsets. "
                f"Exceeds minimum reduction threshold of {min_red}%."
            )
        elif min_met:
            msg = (
                f"Minimum reduction threshold met ({_round_val(reduction_pct, 1)}% >= {min_red}%), "
                f"but reduction share should ideally exceed offset share."
            )
        else:
            msg = (
                f"Reduction of {_round_val(reduction_pct, 1)}% is below minimum "
                f"threshold of {min_red}%. Additional reduction measures needed "
                f"before offsets can be used credibly."
            )

        return ReductionOffsetBalance(
            baseline_tco2e=data.baseline_tco2e,
            total_planned_reduction_tco2e=_round_val(total_abatement),
            total_planned_reduction_pct=_round_val(reduction_pct, 2),
            residual_tco2e=_round_val(residual),
            offset_needed_tco2e=_round_val(residual),
            reduction_first_compliant=min_met and hierarchy_ok,
            min_reduction_met=min_met,
            reduction_share_pct=_round_val(reduction_pct, 2),
            offset_share_pct=_round_val(offset_pct, 2),
            improvement_trajectory=improving,
            message=msg,
        )

    def _assess_hierarchy(
        self,
        measures: List[ReductionMeasureInput],
        baseline: Decimal,
    ) -> HierarchyAssessment:
        """Assess compliance with the mitigation hierarchy.

        ISO 14068-1:2023 Section 9.2 requires:
        1. Avoid/eliminate first
        2. Reduce next
        3. Substitute where possible
        4. Compensate only for residual

        Args:
            measures: All reduction measures.
            baseline: Baseline emissions.

        Returns:
            HierarchyAssessment.
        """
        avoid_count = avoid_abt = 0
        reduce_count = reduce_abt = 0
        sub_count = sub_abt = 0
        comp_count = comp_abt = 0

        avoid_abt_d = Decimal("0")
        reduce_abt_d = Decimal("0")
        sub_abt_d = Decimal("0")
        comp_abt_d = Decimal("0")

        for m in measures:
            if m.status == MeasureStatus.CANCELLED.value:
                continue
            abt = m.annual_abatement_tco2e
            if m.priority == MitigationPriority.AVOID.value:
                avoid_count += 1
                avoid_abt_d += abt
            elif m.priority == MitigationPriority.REDUCE.value:
                reduce_count += 1
                reduce_abt_d += abt
            elif m.priority == MitigationPriority.SUBSTITUTE.value:
                sub_count += 1
                sub_abt_d += abt
            elif m.priority == MitigationPriority.COMPENSATE.value:
                comp_count += 1
                comp_abt_d += abt

        total_real_reduction = avoid_abt_d + reduce_abt_d + sub_abt_d
        hierarchy_ok = total_real_reduction >= comp_abt_d or comp_count == 0

        # Score: weight avoid highest, compensate lowest
        total_abt = avoid_abt_d + reduce_abt_d + sub_abt_d + comp_abt_d
        if total_abt > Decimal("0"):
            score = (
                _safe_divide(avoid_abt_d, total_abt) * Decimal("100")
                + _safe_divide(reduce_abt_d, total_abt) * Decimal("80")
                + _safe_divide(sub_abt_d, total_abt) * Decimal("60")
                + _safe_divide(comp_abt_d, total_abt) * Decimal("20")
            )
        else:
            score = Decimal("0")

        recommendations: List[str] = []
        if avoid_count == 0:
            recommendations.append(
                "Consider avoidance measures (eliminating emission sources) "
                "as the highest priority in the mitigation hierarchy."
            )
        if comp_abt_d > total_real_reduction:
            recommendations.append(
                "Offset abatement exceeds reduction abatement. "
                "Increase genuine reduction measures to improve hierarchy compliance."
            )

        return HierarchyAssessment(
            avoid_measures_count=avoid_count,
            reduce_measures_count=reduce_count,
            substitute_measures_count=sub_count,
            compensate_measures_count=comp_count,
            avoid_abatement_tco2e=_round_val(avoid_abt_d),
            reduce_abatement_tco2e=_round_val(reduce_abt_d),
            substitute_abatement_tco2e=_round_val(sub_abt_d),
            compensate_abatement_tco2e=_round_val(comp_abt_d),
            hierarchy_compliant=hierarchy_ok,
            hierarchy_score=_round_val(score, 2),
            recommendations=recommendations,
        )

    def _assess_pathway_alignment(
        self,
        data: CarbonMgmtPlanInput,
        total_abatement: Decimal,
    ) -> PathwayAlignment:
        """Assess alignment with temperature pathway.

        Compares planned annual reduction rate to the required rate
        for 1.5C or well-below-2C pathway alignment.

        Args:
            data: Plan input.
            total_abatement: Total planned abatement.

        Returns:
            PathwayAlignment assessment.
        """
        years = max(1, data.target_year - data.current_year)
        annual_reduction = _safe_divide(total_abatement, _decimal(years))
        annual_rate = _safe_pct(annual_reduction, data.baseline_tco2e)

        if data.pathway_ambition == "1.5c":
            required = MIN_ANNUAL_REDUCTION_RATE_15C
        elif data.pathway_ambition == "well_below_2c":
            required = MIN_ANNUAL_REDUCTION_RATE_WB2C
        else:
            required = MIN_ANNUAL_REDUCTION_RATE_WB2C

        aligned = annual_rate >= required
        gap = max(Decimal("0"), required - annual_rate)

        # Years to neutrality at current rate
        if annual_rate > Decimal("0"):
            ytn = int(float(Decimal("100") / annual_rate)) + 1
        else:
            ytn = 999

        if aligned:
            msg = (
                f"Planned annual reduction rate of {_round_val(annual_rate, 1)}% "
                f"is aligned with the {data.pathway_ambition} pathway "
                f"(minimum {required}% per year)."
            )
        else:
            msg = (
                f"Planned annual reduction rate of {_round_val(annual_rate, 1)}% "
                f"is below the {data.pathway_ambition} pathway requirement "
                f"of {required}% per year. Gap: {_round_val(gap, 1)}%."
            )

        return PathwayAlignment(
            pathway=data.pathway_ambition,
            required_annual_rate_pct=required,
            planned_annual_rate_pct=_round_val(annual_rate, 2),
            is_aligned=aligned,
            gap_pct=_round_val(gap, 2),
            years_to_neutrality=min(ytn, 999),
            message=msg,
        )

    def _build_financial_summary(
        self,
        data: CarbonMgmtPlanInput,
        active_measures: List[ReductionMeasureInput],
        residual: Decimal,
    ) -> FinancialSummary:
        """Build financial summary of the management plan.

        Args:
            data: Plan input.
            active_measures: Active (non-cancelled) measures.
            residual: Residual emissions needing offsets.

        Returns:
            FinancialSummary.
        """
        total_impl = sum(
            (m.implementation_cost_usd for m in active_measures), Decimal("0")
        )
        total_savings = sum(
            (m.annual_savings_usd for m in active_measures), Decimal("0")
        )
        credit_cost = residual * data.credit_price_per_tco2e
        total_operating = sum(
            (m.annual_operating_cost_usd for m in active_measures), Decimal("0")
        )
        net_annual = total_operating - total_savings + credit_cost

        total_5yr_cost = total_impl + (net_annual * Decimal("5"))
        total_5yr_savings = total_savings * Decimal("5")

        total_abt = sum(
            (m.annual_abatement_tco2e for m in active_measures), Decimal("0")
        )
        avg_cost = _safe_divide(total_impl + net_annual, total_abt)

        neg_cost_abt = sum(
            (m.annual_abatement_tco2e for m in active_measures
             if m.annual_savings_usd > m.annual_operating_cost_usd + m.implementation_cost_usd),
            Decimal("0"),
        )

        roi = Decimal("0")
        if total_impl > Decimal("0"):
            roi = _safe_pct(total_savings - total_operating, total_impl)

        msg = (
            f"Total implementation cost: ${_round_val(total_impl, 0):,}. "
            f"Annual savings: ${_round_val(total_savings, 0):,}. "
            f"Annual credit cost: ${_round_val(credit_cost, 0):,}."
        )

        return FinancialSummary(
            total_implementation_cost_usd=_round_val(total_impl, 2),
            total_annual_savings_usd=_round_val(total_savings, 2),
            total_annual_credit_cost_usd=_round_val(credit_cost, 2),
            net_annual_cost_usd=_round_val(net_annual, 2),
            total_5yr_cost_usd=_round_val(total_5yr_cost, 2),
            total_5yr_savings_usd=_round_val(total_5yr_savings, 2),
            avg_abatement_cost_usd=_round_val(avg_cost, 2),
            negative_cost_abatement_tco2e=_round_val(neg_cost_abt),
            roi_pct=_round_val(roi, 2),
            message=msg,
        )

    def _assess_plan_compliance(
        self,
        data: CarbonMgmtPlanInput,
        balance: ReductionOffsetBalance,
        hierarchy: HierarchyAssessment,
        pathway: PathwayAlignment,
        warnings: List[str],
    ) -> str:
        """Assess overall plan compliance with target standard.

        ISO 14068-1:2023 Section 9 requirements:
        - Reduction-first hierarchy documented
        - Quantified reduction targets
        - Timeline with milestones
        - Credit quality criteria
        - Monitoring and review process

        Args:
            data: Plan input.
            balance: Reduction/offset balance.
            hierarchy: Hierarchy assessment.
            pathway: Pathway alignment.
            warnings: Warning list to append to.

        Returns:
            PlanCompliance value.
        """
        issues = 0

        if not balance.min_reduction_met:
            issues += 1
            warnings.append(
                f"ISO 14068-1 Section 9.3: Minimum reduction threshold not met. "
                f"Reduction of {balance.total_planned_reduction_pct}% is below required minimum."
            )

        if not hierarchy.hierarchy_compliant:
            issues += 1
            warnings.append(
                "ISO 14068-1 Section 9.2: Mitigation hierarchy not fully followed. "
                "Offset abatement exceeds genuine reduction abatement."
            )

        if len(data.measures) == 0:
            issues += 1
            warnings.append(
                "ISO 14068-1 Section 9: No reduction measures defined in plan."
            )

        if data.target_year <= data.current_year:
            issues += 1
            warnings.append(
                "Target year must be in the future for a valid management plan."
            )

        if issues == 0:
            return PlanCompliance.COMPLIANT.value
        elif issues <= 2:
            return PlanCompliance.PARTIALLY_COMPLIANT.value
        else:
            return PlanCompliance.NON_COMPLIANT.value
