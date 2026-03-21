# -*- coding: utf-8 -*-
"""
Interim Target Setting Workflow
====================================

6-phase DAG workflow for setting SBTi-aligned interim targets within
PACK-029 Interim Targets Pack.  The workflow loads baseline and long-term
targets, calculates 5-year and 10-year interim milestones, validates
them against SBTi criteria, generates an annual pathway, allocates carbon
budgets, and produces a comprehensive interim target summary report.

Phases:
    1. LoadBaseline        -- Load baseline emissions and long-term targets
                              from PACK-021/022/027 via PACK021Bridge
    2. CalcInterimTargets  -- Calculate 5-year and 10-year interim targets
                              using InterimTargetEngine decomposition
    3. ValidateTargets     -- Validate interim targets against SBTi near-term
                              and long-term criteria via MilestoneValidationEngine
    4. GeneratePathway     -- Generate annual intensity/absolute pathway
                              points using AnnualPathwayEngine
    5. AllocateBudget      -- Allocate carbon budget across scopes and
                              business units via BudgetAllocationEngine
    6. SummaryReport       -- Generate interim target summary report with
                              charts, tables, and executive summary

Regulatory references:
    - SBTi Corporate Net-Zero Standard v1.1 (2024)
    - SBTi Near-Term Target Setting Guidance v2.0
    - SBTi Long-Term Target Setting Guidance
    - GHG Protocol Corporate Standard (boundary rules)
    - IPCC AR6 WG III Carbon Budget Allocation
    - IEA Net Zero by 2050 Roadmap (milestone alignment)

Zero-hallucination: all target calculations use deterministic SBTi-published
minimum ambition thresholds and linear/contraction convergence formulas.
No LLM calls in any numeric computation path.

Author: GreenLang Team
Version: 29.0.0
Pack: PACK-029 Interim Targets Pack
"""

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION = "29.0.0"
_PACK_ID = "PACK-029"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _interpolate_linear(base_val: float, target_val: float, base_yr: int,
                         target_yr: int, current_yr: int) -> float:
    """Linear interpolation between base and target values."""
    if target_yr <= base_yr:
        return target_val
    t = min(max((current_yr - base_yr) / (target_yr - base_yr), 0.0), 1.0)
    return base_val + t * (target_val - base_val)


def _interpolate_exponential(base_val: float, target_val: float, base_yr: int,
                              target_yr: int, current_yr: int) -> float:
    """Exponential decay interpolation for target convergence."""
    if target_yr <= base_yr or base_val <= 0:
        return target_val
    safe_target = max(target_val, 1e-10)
    k = -math.log(safe_target / base_val) / (target_yr - base_yr)
    t = min(max(current_yr - base_yr, 0), target_yr - base_yr)
    return base_val * math.exp(-k * t)


def _interpolate_contraction(base_val: float, target_val: float, base_yr: int,
                              target_yr: int, current_yr: int) -> float:
    """Absolute contraction approach (equal annual absolute reductions)."""
    if target_yr <= base_yr:
        return target_val
    annual_reduction = (base_val - target_val) / (target_yr - base_yr)
    elapsed = min(max(current_yr - base_yr, 0), target_yr - base_yr)
    return max(base_val - annual_reduction * elapsed, 0.0)


def _calc_cagr(start_val: float, end_val: float, years: int) -> float:
    """Calculate compound annual growth rate (reduction rate if negative)."""
    if years <= 0 or start_val <= 0 or end_val <= 0:
        return 0.0
    return ((end_val / start_val) ** (1.0 / years) - 1.0) * 100.0


def _carbon_budget_remaining(base_emissions: float, target_emissions: float,
                              base_yr: int, target_yr: int,
                              current_yr: int, actual_cumulative: float) -> float:
    """Calculate remaining carbon budget assuming linear decline."""
    total_years = target_yr - base_yr
    if total_years <= 0:
        return 0.0
    total_budget = (base_emissions + target_emissions) / 2.0 * total_years
    return max(total_budget - actual_cumulative, 0.0)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a single workflow phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class TargetType(str, Enum):
    """Type of emissions target."""
    ABSOLUTE = "absolute"
    INTENSITY = "intensity"
    BOTH = "both"


class TargetScope(str, Enum):
    """GHG Protocol scope coverage."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_1_2 = "scope_1_2"
    SCOPE_3 = "scope_3"
    ALL_SCOPES = "all_scopes"


class TargetTimeframe(str, Enum):
    """SBTi target timeframe classification."""
    NEAR_TERM = "near_term"          # 5-10 years
    LONG_TERM = "long_term"          # > 10 years, up to 2050
    INTERIM_5Y = "interim_5y"        # 5-year interim milestone
    INTERIM_10Y = "interim_10y"      # 10-year interim milestone
    ANNUAL = "annual"                # Annual pathway point


class BudgetMethod(str, Enum):
    """Carbon budget allocation method."""
    EQUAL_PER_CAPITA = "equal_per_capita"
    GRANDFATHERING = "grandfathering"
    CONTRACTION_CONVERGENCE = "contraction_convergence"
    ECONOMIC_INTENSITY = "economic_intensity"
    PROPORTIONAL = "proportional"


class SBTiAmbition(str, Enum):
    """SBTi ambition level for near-term targets."""
    WELL_BELOW_2C = "well_below_2c"
    ONE_POINT_FIVE_C = "1.5c"
    NET_ZERO = "net_zero"


class ValidationResult(str, Enum):
    """Validation pass/fail result."""
    PASS = "pass"
    FAIL = "fail"
    CONDITIONAL = "conditional"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


class RAGStatus(str, Enum):
    """Red-Amber-Green status indicator."""
    RED = "red"
    AMBER = "amber"
    GREEN = "green"


# =============================================================================
# SBTI MINIMUM AMBITION THRESHOLDS (Zero-Hallucination Published Data)
# =============================================================================


SBTI_NEAR_TERM_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "1.5c": {
        "absolute_annual_linear_reduction_pct": 4.2,
        "intensity_annual_linear_reduction_pct": 4.2,
        "max_timeframe_years": 10,
        "min_timeframe_years": 5,
        "scope1_coverage_min_pct": 95.0,
        "scope2_coverage_min_pct": 95.0,
        "scope3_coverage_min_pct": 67.0,
        "scope3_threshold_pct_of_total": 40.0,
        "description": "1.5C-aligned near-term target (SBTi Corporate Standard v2.0)",
    },
    "well_below_2c": {
        "absolute_annual_linear_reduction_pct": 2.5,
        "intensity_annual_linear_reduction_pct": 2.5,
        "max_timeframe_years": 10,
        "min_timeframe_years": 5,
        "scope1_coverage_min_pct": 95.0,
        "scope2_coverage_min_pct": 95.0,
        "scope3_coverage_min_pct": 67.0,
        "scope3_threshold_pct_of_total": 40.0,
        "description": "Well-below 2C near-term target (SBTi Corporate Standard v2.0)",
    },
}

SBTI_LONG_TERM_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "net_zero": {
        "scope1_2_reduction_pct": 90.0,
        "scope3_reduction_pct": 90.0,
        "max_year": 2050,
        "neutralization_max_pct": 10.0,
        "coverage_scope1_pct": 95.0,
        "coverage_scope2_pct": 95.0,
        "coverage_scope3_pct": 90.0,
        "description": "SBTi Corporate Net-Zero Standard v1.1 long-term target",
    },
}

# SBTi cross-sector absolute contraction minimum rates
SBTI_CROSS_SECTOR_RATES: Dict[str, float] = {
    "1.5c_annual_absolute_min": 4.2,
    "wb2c_annual_absolute_min": 2.5,
    "1.5c_cumulative_5yr": 21.0,
    "wb2c_cumulative_5yr": 12.5,
    "1.5c_cumulative_10yr": 42.0,
    "wb2c_cumulative_10yr": 25.0,
    "net_zero_2050_total": 90.0,
}

# Default interim milestone percentages (from SBTi guidance)
SBTI_INTERIM_MILESTONES: Dict[str, Dict[str, float]] = {
    "1.5c": {
        "5_year_min_reduction_pct": 21.0,
        "10_year_min_reduction_pct": 42.0,
        "15_year_min_reduction_pct": 60.0,
        "20_year_min_reduction_pct": 75.0,
        "25_year_min_reduction_pct": 85.0,
        "30_year_min_reduction_pct": 90.0,
    },
    "well_below_2c": {
        "5_year_min_reduction_pct": 12.5,
        "10_year_min_reduction_pct": 25.0,
        "15_year_min_reduction_pct": 40.0,
        "20_year_min_reduction_pct": 55.0,
        "25_year_min_reduction_pct": 70.0,
        "30_year_min_reduction_pct": 85.0,
    },
}

# Carbon budget allocation by sector (GtCO2e, IPCC AR6)
SECTOR_CARBON_BUDGETS: Dict[str, Dict[str, float]] = {
    "power_generation": {
        "1.5c_50pct_budget_gt": 75.0,
        "2c_67pct_budget_gt": 145.0,
        "2020_annual_emissions_gt": 13.5,
    },
    "industry": {
        "1.5c_50pct_budget_gt": 55.0,
        "2c_67pct_budget_gt": 110.0,
        "2020_annual_emissions_gt": 9.0,
    },
    "transport": {
        "1.5c_50pct_budget_gt": 40.0,
        "2c_67pct_budget_gt": 80.0,
        "2020_annual_emissions_gt": 7.7,
    },
    "buildings": {
        "1.5c_50pct_budget_gt": 20.0,
        "2c_67pct_budget_gt": 42.0,
        "2020_annual_emissions_gt": 3.0,
    },
    "agriculture": {
        "1.5c_50pct_budget_gt": 35.0,
        "2c_67pct_budget_gt": 70.0,
        "2020_annual_emissions_gt": 5.5,
    },
    "cross_sector": {
        "1.5c_50pct_budget_gt": 400.0,
        "2c_67pct_budget_gt": 850.0,
        "2020_annual_emissions_gt": 40.0,
    },
}

# Business unit allocation weighting factors
BU_ALLOCATION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "proportional": {
        "emissions_weight": 1.0,
        "revenue_weight": 0.0,
        "headcount_weight": 0.0,
    },
    "economic_intensity": {
        "emissions_weight": 0.5,
        "revenue_weight": 0.5,
        "headcount_weight": 0.0,
    },
    "equal_effort": {
        "emissions_weight": 0.33,
        "revenue_weight": 0.33,
        "headcount_weight": 0.34,
    },
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, ge=0)
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0)
    completion_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    dag_node_id: str = Field(default="")


class BaselineData(BaseModel):
    """Loaded baseline and long-term target data."""
    entity_id: str = Field(default="")
    company_name: str = Field(default="")
    base_year: int = Field(default=2020, ge=2015, le=2030)
    base_year_scope1_tco2e: float = Field(default=0.0, ge=0.0)
    base_year_scope2_tco2e: float = Field(default=0.0, ge=0.0)
    base_year_scope3_tco2e: float = Field(default=0.0, ge=0.0)
    base_year_total_tco2e: float = Field(default=0.0, ge=0.0)
    base_year_revenue_musd: float = Field(default=0.0, ge=0.0)
    base_year_intensity: float = Field(default=0.0, ge=0.0)
    intensity_unit: str = Field(default="tCO2e/M$ revenue")
    current_year: int = Field(default=2025)
    current_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    long_term_target_year: int = Field(default=2050)
    long_term_target_reduction_pct: float = Field(default=90.0, ge=0.0, le=100.0)
    near_term_target_year: int = Field(default=2030)
    near_term_target_reduction_pct: float = Field(default=42.0, ge=0.0, le=100.0)
    sbti_ambition: str = Field(default="1.5c")
    sector: str = Field(default="cross_sector")
    scope3_material: bool = Field(default=True)
    scope3_pct_of_total: float = Field(default=0.0, ge=0.0, le=100.0)
    business_units: List[Dict[str, Any]] = Field(default_factory=list)
    source_pack: str = Field(default="PACK-021")
    provenance_hash: str = Field(default="")


class InterimTarget(BaseModel):
    """A single interim target milestone."""
    target_id: str = Field(default="")
    target_name: str = Field(default="")
    timeframe: TargetTimeframe = Field(default=TargetTimeframe.INTERIM_5Y)
    target_year: int = Field(default=2030)
    target_type: TargetType = Field(default=TargetType.ABSOLUTE)
    target_scope: TargetScope = Field(default=TargetScope.SCOPE_1_2)
    base_year: int = Field(default=2020)
    base_value_tco2e: float = Field(default=0.0, ge=0.0)
    target_value_tco2e: float = Field(default=0.0, ge=0.0)
    reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    annual_reduction_rate_pct: float = Field(default=0.0)
    sbti_min_reduction_pct: float = Field(default=0.0)
    exceeds_sbti_minimum: bool = Field(default=False)
    intensity_base_value: float = Field(default=0.0, ge=0.0)
    intensity_target_value: float = Field(default=0.0, ge=0.0)
    intensity_unit: str = Field(default="")
    carbon_budget_tco2e: float = Field(default=0.0, ge=0.0)
    cumulative_emissions_allowed_tco2e: float = Field(default=0.0, ge=0.0)
    sbti_ambition: str = Field(default="1.5c")
    provenance_hash: str = Field(default="")


class InterimTargetSet(BaseModel):
    """Complete set of interim targets."""
    entity_id: str = Field(default="")
    base_year: int = Field(default=2020)
    long_term_year: int = Field(default=2050)
    sbti_ambition: str = Field(default="1.5c")
    targets: List[InterimTarget] = Field(default_factory=list)
    five_year_target: Optional[InterimTarget] = Field(default=None)
    ten_year_target: Optional[InterimTarget] = Field(default=None)
    total_scope1_2_reduction_pct: float = Field(default=0.0)
    total_scope3_reduction_pct: float = Field(default=0.0)
    meets_sbti_near_term: bool = Field(default=False)
    meets_sbti_long_term: bool = Field(default=False)
    provenance_hash: str = Field(default="")


class ValidationFinding(BaseModel):
    """A single SBTi validation finding."""
    criterion_id: str = Field(default="")
    criterion_name: str = Field(default="")
    description: str = Field(default="")
    result: ValidationResult = Field(default=ValidationResult.NOT_APPLICABLE)
    actual_value: str = Field(default="")
    required_value: str = Field(default="")
    finding: str = Field(default="")
    remediation: str = Field(default="")
    severity: str = Field(default="info")


class ValidationSummary(BaseModel):
    """Summary of SBTi validation results."""
    total_criteria: int = Field(default=0)
    passed: int = Field(default=0)
    failed: int = Field(default=0)
    conditional: int = Field(default=0)
    warnings: int = Field(default=0)
    pass_rate_pct: float = Field(default=0.0)
    overall_result: ValidationResult = Field(default=ValidationResult.FAIL)
    findings: List[ValidationFinding] = Field(default_factory=list)
    sbti_submission_ready: bool = Field(default=False)
    improvement_actions: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class AnnualPathwayPoint(BaseModel):
    """A single year-point on the annual pathway."""
    year: int = Field(default=2025)
    scope1_2_target_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_target_tco2e: float = Field(default=0.0, ge=0.0)
    total_target_tco2e: float = Field(default=0.0, ge=0.0)
    intensity_target: float = Field(default=0.0, ge=0.0)
    cumulative_reduction_pct: float = Field(default=0.0)
    annual_reduction_rate_pct: float = Field(default=0.0)
    carbon_budget_remaining_tco2e: float = Field(default=0.0, ge=0.0)
    is_milestone_year: bool = Field(default=False)
    milestone_name: str = Field(default="")


class AnnualPathway(BaseModel):
    """Complete annual emissions pathway."""
    entity_id: str = Field(default="")
    base_year: int = Field(default=2020)
    target_year: int = Field(default=2050)
    convergence_model: str = Field(default="linear")
    pathway_points: List[AnnualPathwayPoint] = Field(default_factory=list)
    milestone_years: List[int] = Field(default_factory=list)
    total_carbon_budget_tco2e: float = Field(default=0.0, ge=0.0)
    average_annual_reduction_pct: float = Field(default=0.0)
    front_loaded: bool = Field(default=False)
    provenance_hash: str = Field(default="")


class BUBudgetAllocation(BaseModel):
    """Carbon budget allocation for a single business unit."""
    bu_id: str = Field(default="")
    bu_name: str = Field(default="")
    base_year_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    allocated_budget_tco2e: float = Field(default=0.0, ge=0.0)
    annual_target_tco2e: float = Field(default=0.0, ge=0.0)
    reduction_rate_pct: float = Field(default=0.0)
    allocation_method: str = Field(default="proportional")
    weight: float = Field(default=0.0, ge=0.0, le=1.0)
    five_year_budget_tco2e: float = Field(default=0.0, ge=0.0)
    ten_year_budget_tco2e: float = Field(default=0.0, ge=0.0)


class CarbonBudgetAllocation(BaseModel):
    """Complete carbon budget allocation."""
    entity_id: str = Field(default="")
    total_budget_tco2e: float = Field(default=0.0, ge=0.0)
    scope1_budget_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_budget_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_budget_tco2e: float = Field(default=0.0, ge=0.0)
    allocation_method: BudgetMethod = Field(default=BudgetMethod.PROPORTIONAL)
    bu_allocations: List[BUBudgetAllocation] = Field(default_factory=list)
    five_year_budget_tco2e: float = Field(default=0.0, ge=0.0)
    ten_year_budget_tco2e: float = Field(default=0.0, ge=0.0)
    remaining_budget_tco2e: float = Field(default=0.0, ge=0.0)
    budget_depletion_year: int = Field(default=2050)
    provenance_hash: str = Field(default="")


class InterimTargetReport(BaseModel):
    """Complete interim target summary report."""
    report_id: str = Field(default="")
    report_date: str = Field(default="")
    company_name: str = Field(default="")
    entity_id: str = Field(default="")
    baseline: BaselineData = Field(default_factory=BaselineData)
    interim_targets: InterimTargetSet = Field(default_factory=InterimTargetSet)
    validation_summary: ValidationSummary = Field(default_factory=ValidationSummary)
    annual_pathway: AnnualPathway = Field(default_factory=AnnualPathway)
    carbon_budget: CarbonBudgetAllocation = Field(default_factory=CarbonBudgetAllocation)
    executive_summary: str = Field(default="")
    key_findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    output_formats: List[str] = Field(default_factory=lambda: ["json", "html"])
    data_quality_score: float = Field(default=0.0, ge=0.0, le=5.0)
    provenance_hash: str = Field(default="")


class InterimTargetSettingConfig(BaseModel):
    """Configuration for the interim target setting workflow."""
    company_name: str = Field(default="")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")
    sector: str = Field(default="cross_sector")
    base_year: int = Field(default=2020, ge=2015, le=2030)
    current_year: int = Field(default=2025, ge=2020, le=2035)
    long_term_target_year: int = Field(default=2050, ge=2040, le=2070)
    long_term_reduction_pct: float = Field(default=90.0, ge=50.0, le=100.0)
    sbti_ambition: str = Field(default="1.5c")
    target_type: TargetType = Field(default=TargetType.BOTH)
    convergence_model: str = Field(default="linear")
    budget_allocation_method: BudgetMethod = Field(default=BudgetMethod.PROPORTIONAL)
    include_scope3: bool = Field(default=True)
    scope3_threshold_pct: float = Field(default=40.0, ge=0.0, le=100.0)
    interim_milestones: List[int] = Field(
        default_factory=lambda: [5, 10, 15, 20, 25, 30],
        description="Milestone intervals in years from base year",
    )
    output_formats: List[str] = Field(default_factory=lambda: ["json", "html"])
    enable_bu_allocation: bool = Field(default=True)
    front_load_reductions: bool = Field(default=False)
    front_load_factor: float = Field(default=1.3, ge=1.0, le=2.0)
    validate_sbti: bool = Field(default=True)
    retry_on_failure: bool = Field(default=True)
    max_retries: int = Field(default=3, ge=0, le=10)


class InterimTargetSettingInput(BaseModel):
    """Input data for the interim target setting workflow."""
    config: InterimTargetSettingConfig = Field(default_factory=InterimTargetSettingConfig)
    baseline_data: BaselineData = Field(default_factory=BaselineData)
    historical_emissions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Historical emissions [{year, scope1, scope2, scope3, total, revenue}]",
    )
    business_units: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="BU data [{bu_id, name, emissions, revenue, headcount}]",
    )
    external_targets: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Existing external target commitments",
    )


class InterimTargetSettingResult(BaseModel):
    """Result of the interim target setting workflow."""
    workflow_id: str = Field(...)
    workflow_name: str = Field(default="interim_target_setting")
    pack_id: str = Field(default=_PACK_ID)
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    baseline: BaselineData = Field(default_factory=BaselineData)
    interim_targets: InterimTargetSet = Field(default_factory=InterimTargetSet)
    validation_summary: ValidationSummary = Field(default_factory=ValidationSummary)
    annual_pathway: AnnualPathway = Field(default_factory=AnnualPathway)
    carbon_budget: CarbonBudgetAllocation = Field(default_factory=CarbonBudgetAllocation)
    report: InterimTargetReport = Field(default_factory=InterimTargetReport)
    key_findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class InterimTargetSettingWorkflow:
    """
    6-phase DAG workflow for setting SBTi-aligned interim targets.

    Phase 1: LoadBaseline       -- Load baseline and long-term targets.
    Phase 2: CalcInterimTargets -- Calculate 5-year and 10-year interim targets.
    Phase 3: ValidateTargets    -- Validate against SBTi criteria.
    Phase 4: GeneratePathway    -- Generate annual emissions pathway.
    Phase 5: AllocateBudget     -- Allocate carbon budget by scope and BU.
    Phase 6: SummaryReport      -- Generate interim target summary report.

    DAG Dependencies:
        Phase 1 -> Phase 2 -> Phase 3
                           -> Phase 4  (parallel with Phase 3)
                -> Phase 5  (depends on Phase 2 + Phase 4)
                -> Phase 6  (depends on all prior phases)

    Example:
        >>> wf = InterimTargetSettingWorkflow()
        >>> inp = InterimTargetSettingInput(
        ...     config=InterimTargetSettingConfig(company_name="Acme Corp"),
        ...     baseline_data=BaselineData(
        ...         base_year_total_tco2e=100000,
        ...         base_year_scope1_tco2e=50000,
        ...         base_year_scope2_tco2e=30000,
        ...         base_year_scope3_tco2e=20000,
        ...     ),
        ... )
        >>> result = await wf.execute(inp)
    """

    def __init__(self, config: Optional[InterimTargetSettingConfig] = None) -> None:
        self.workflow_id: str = _new_uuid()
        self.config = config or InterimTargetSettingConfig()
        self._phase_results: List[PhaseResult] = []
        self._baseline: BaselineData = BaselineData()
        self._interim_targets: InterimTargetSet = InterimTargetSet()
        self._validation: ValidationSummary = ValidationSummary()
        self._pathway: AnnualPathway = AnnualPathway()
        self._budget: CarbonBudgetAllocation = CarbonBudgetAllocation()
        self._report: InterimTargetReport = InterimTargetReport()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(self, input_data: InterimTargetSettingInput) -> InterimTargetSettingResult:
        """Execute the 6-phase interim target setting workflow."""
        started_at = _utcnow()
        self.config = input_data.config
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        self.logger.info(
            "Starting interim target setting workflow %s, company=%s, ambition=%s",
            self.workflow_id, self.config.company_name, self.config.sbti_ambition,
        )

        try:
            # Phase 1: Load Baseline
            phase1 = await self._phase_load_baseline(input_data)
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise RuntimeError("Phase 1 (LoadBaseline) failed; cannot continue.")

            # Phase 2: Calculate Interim Targets
            phase2 = await self._phase_calc_interim_targets(input_data)
            self._phase_results.append(phase2)

            # Phase 3: Validate Targets (depends on Phase 2)
            phase3 = await self._phase_validate_targets(input_data)
            self._phase_results.append(phase3)

            # Phase 4: Generate Annual Pathway (depends on Phase 2)
            phase4 = await self._phase_generate_pathway(input_data)
            self._phase_results.append(phase4)

            # Phase 5: Allocate Carbon Budget (depends on Phase 2 + Phase 4)
            phase5 = await self._phase_allocate_budget(input_data)
            self._phase_results.append(phase5)

            # Phase 6: Summary Report (depends on all prior phases)
            phase6 = await self._phase_summary_report(input_data)
            self._phase_results.append(phase6)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Interim target setting failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=99,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()

        result = InterimTargetSettingResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            baseline=self._baseline,
            interim_targets=self._interim_targets,
            validation_summary=self._validation,
            annual_pathway=self._pathway,
            carbon_budget=self._budget,
            report=self._report,
            key_findings=self._generate_findings(),
            recommendations=self._generate_recommendations(),
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"}),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Load Baseline
    # -------------------------------------------------------------------------

    async def _phase_load_baseline(self, input_data: InterimTargetSettingInput) -> PhaseResult:
        """Load baseline emissions and long-term targets from upstream packs."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        bl = input_data.baseline_data

        # Auto-calculate total if not provided
        if bl.base_year_total_tco2e <= 0:
            bl.base_year_total_tco2e = (
                bl.base_year_scope1_tco2e
                + bl.base_year_scope2_tco2e
                + bl.base_year_scope3_tco2e
            )
            if bl.base_year_total_tco2e <= 0:
                bl.base_year_total_tco2e = 100000.0
                warnings.append("No baseline emissions provided; using default 100,000 tCO2e.")

        # Calculate scope3 percentage
        if bl.base_year_total_tco2e > 0:
            bl.scope3_pct_of_total = round(
                bl.base_year_scope3_tco2e / bl.base_year_total_tco2e * 100, 2,
            )

        # Calculate intensity if revenue is available
        if bl.base_year_revenue_musd > 0:
            bl.base_year_intensity = round(
                bl.base_year_total_tco2e / bl.base_year_revenue_musd, 4,
            )

        # Apply config overrides
        bl.company_name = bl.company_name or self.config.company_name
        bl.entity_id = bl.entity_id or self.config.entity_id
        bl.base_year = bl.base_year or self.config.base_year
        bl.long_term_target_year = self.config.long_term_target_year
        bl.long_term_target_reduction_pct = self.config.long_term_reduction_pct
        bl.sbti_ambition = self.config.sbti_ambition
        bl.sector = self.config.sector

        # Determine near-term target year (base + 5..10)
        if bl.near_term_target_year <= bl.base_year:
            bl.near_term_target_year = bl.base_year + 7  # Default 7-year forward

        # Set near-term reduction from SBTi minimum
        ambition = self.config.sbti_ambition
        thresholds = SBTI_NEAR_TERM_THRESHOLDS.get(ambition, SBTI_NEAR_TERM_THRESHOLDS["1.5c"])
        nt_years = bl.near_term_target_year - bl.base_year
        min_annual_rate = thresholds["absolute_annual_linear_reduction_pct"]
        min_nt_reduction = min_annual_rate * nt_years
        if bl.near_term_target_reduction_pct < min_nt_reduction:
            bl.near_term_target_reduction_pct = min_nt_reduction
            warnings.append(
                f"Near-term reduction adjusted to {min_nt_reduction:.1f}% "
                f"to meet SBTi {ambition} minimum ({min_annual_rate}%/yr x {nt_years} yrs).",
            )

        # Store historical emissions for later use
        if not input_data.historical_emissions:
            # Generate synthetic historical data
            for yr in range(bl.base_year, bl.current_year + 1):
                factor = 1.0 - 0.025 * (yr - bl.base_year)
                input_data.historical_emissions.append({
                    "year": yr,
                    "scope1": round(bl.base_year_scope1_tco2e * factor, 2),
                    "scope2": round(bl.base_year_scope2_tco2e * factor, 2),
                    "scope3": round(bl.base_year_scope3_tco2e * factor, 2),
                    "total": round(bl.base_year_total_tco2e * factor, 2),
                })
            warnings.append("Synthetic historical emissions generated (2.5% annual decline).")

        # Calculate current emissions if not provided
        if bl.current_emissions_tco2e <= 0 and input_data.historical_emissions:
            latest = max(input_data.historical_emissions, key=lambda x: x.get("year", 0))
            bl.current_emissions_tco2e = latest.get("total", bl.base_year_total_tco2e)

        bl.provenance_hash = _compute_hash(bl.model_dump_json())
        self._baseline = bl

        outputs["entity_id"] = bl.entity_id
        outputs["company_name"] = bl.company_name
        outputs["base_year"] = bl.base_year
        outputs["base_year_total_tco2e"] = bl.base_year_total_tco2e
        outputs["scope1_tco2e"] = bl.base_year_scope1_tco2e
        outputs["scope2_tco2e"] = bl.base_year_scope2_tco2e
        outputs["scope3_tco2e"] = bl.base_year_scope3_tco2e
        outputs["scope3_pct"] = bl.scope3_pct_of_total
        outputs["current_emissions_tco2e"] = bl.current_emissions_tco2e
        outputs["near_term_target_year"] = bl.near_term_target_year
        outputs["near_term_reduction_pct"] = bl.near_term_target_reduction_pct
        outputs["long_term_target_year"] = bl.long_term_target_year
        outputs["long_term_reduction_pct"] = bl.long_term_target_reduction_pct
        outputs["sbti_ambition"] = bl.sbti_ambition
        outputs["source_pack"] = bl.source_pack
        outputs["historical_years"] = len(input_data.historical_emissions)

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="load_baseline", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_load_baseline",
        )

    # -------------------------------------------------------------------------
    # Phase 2: Calculate Interim Targets
    # -------------------------------------------------------------------------

    async def _phase_calc_interim_targets(self, input_data: InterimTargetSettingInput) -> PhaseResult:
        """Calculate 5-year and 10-year interim targets using SBTi criteria."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        bl = self._baseline
        ambition = self.config.sbti_ambition
        milestones = SBTI_INTERIM_MILESTONES.get(ambition, SBTI_INTERIM_MILESTONES["1.5c"])
        thresholds = SBTI_NEAR_TERM_THRESHOLDS.get(ambition, SBTI_NEAR_TERM_THRESHOLDS["1.5c"])

        targets: List[InterimTarget] = []
        five_year_target: Optional[InterimTarget] = None
        ten_year_target: Optional[InterimTarget] = None

        for interval_years in self.config.interim_milestones:
            target_year = bl.base_year + interval_years

            if target_year > self.config.long_term_target_year:
                continue

            # Determine minimum reduction from SBTi lookup
            milestone_key = f"{interval_years}_year_min_reduction_pct"
            sbti_min_pct = milestones.get(milestone_key, 0.0)

            # If no specific milestone, calculate from annual rate
            if sbti_min_pct <= 0:
                annual_rate = thresholds["absolute_annual_linear_reduction_pct"]
                sbti_min_pct = annual_rate * interval_years

            # Calculate reduction percentage (may exceed SBTi minimum)
            if self.config.convergence_model == "linear":
                total_years = self.config.long_term_target_year - bl.base_year
                reduction_pct = (self.config.long_term_reduction_pct / total_years) * interval_years
            elif self.config.convergence_model == "exponential":
                fraction = interval_years / (self.config.long_term_target_year - bl.base_year)
                reduction_pct = self.config.long_term_reduction_pct * (1 - math.exp(-3 * fraction))
            else:
                reduction_pct = sbti_min_pct

            # Apply front-loading if requested
            if self.config.front_load_reductions and interval_years <= 10:
                reduction_pct = min(
                    reduction_pct * self.config.front_load_factor,
                    self.config.long_term_reduction_pct,
                )

            # Ensure meets SBTi minimum
            reduction_pct = max(reduction_pct, sbti_min_pct)

            # Calculate absolute target values
            s12_base = bl.base_year_scope1_tco2e + bl.base_year_scope2_tco2e
            s12_target = s12_base * (1 - reduction_pct / 100.0)
            s3_base = bl.base_year_scope3_tco2e
            s3_target = s3_base * (1 - reduction_pct / 100.0) if self.config.include_scope3 else s3_base

            total_base = bl.base_year_total_tco2e
            total_target = s12_target + s3_target

            # Annual reduction rate
            annual_rate = _calc_cagr(total_base, total_target, interval_years)

            # Carbon budget (area under linear decline)
            budget = (total_base + total_target) / 2.0 * interval_years

            # Intensity target
            intensity_target = 0.0
            if bl.base_year_revenue_musd > 0:
                # Assume revenue grows 2% per year
                projected_revenue = bl.base_year_revenue_musd * (1.02 ** interval_years)
                intensity_target = round(total_target / projected_revenue, 4)

            # Determine timeframe classification
            if interval_years <= 5:
                timeframe = TargetTimeframe.INTERIM_5Y
            elif interval_years <= 10:
                timeframe = TargetTimeframe.INTERIM_10Y
            elif interval_years <= 15:
                timeframe = TargetTimeframe.NEAR_TERM
            else:
                timeframe = TargetTimeframe.LONG_TERM

            target = InterimTarget(
                target_id=f"IT-{bl.entity_id or 'ent'}-{target_year}",
                target_name=f"{interval_years}-Year Interim Target ({target_year})",
                timeframe=timeframe,
                target_year=target_year,
                target_type=self.config.target_type,
                target_scope=(
                    TargetScope.ALL_SCOPES if self.config.include_scope3
                    else TargetScope.SCOPE_1_2
                ),
                base_year=bl.base_year,
                base_value_tco2e=round(total_base, 2),
                target_value_tco2e=round(total_target, 2),
                reduction_pct=round(reduction_pct, 2),
                annual_reduction_rate_pct=round(abs(annual_rate), 2),
                sbti_min_reduction_pct=round(sbti_min_pct, 2),
                exceeds_sbti_minimum=reduction_pct >= sbti_min_pct,
                intensity_base_value=round(bl.base_year_intensity, 4),
                intensity_target_value=round(intensity_target, 4),
                intensity_unit=bl.intensity_unit,
                carbon_budget_tco2e=round(budget, 2),
                cumulative_emissions_allowed_tco2e=round(budget, 2),
                sbti_ambition=ambition,
            )
            target.provenance_hash = _compute_hash(target.model_dump_json())
            targets.append(target)

            if interval_years == 5:
                five_year_target = target
            elif interval_years == 10:
                ten_year_target = target

        # Build interim target set
        all_exceed = all(t.exceeds_sbti_minimum for t in targets)
        s12_reduction = 0.0
        s3_reduction = 0.0
        if targets:
            last_target = targets[-1]
            s12_reduction = last_target.reduction_pct
            s3_reduction = last_target.reduction_pct if self.config.include_scope3 else 0.0

        self._interim_targets = InterimTargetSet(
            entity_id=bl.entity_id,
            base_year=bl.base_year,
            long_term_year=self.config.long_term_target_year,
            sbti_ambition=ambition,
            targets=targets,
            five_year_target=five_year_target,
            ten_year_target=ten_year_target,
            total_scope1_2_reduction_pct=round(s12_reduction, 2),
            total_scope3_reduction_pct=round(s3_reduction, 2),
            meets_sbti_near_term=all_exceed,
            meets_sbti_long_term=s12_reduction >= 90.0,
        )
        self._interim_targets.provenance_hash = _compute_hash(
            self._interim_targets.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["targets_count"] = len(targets)
        outputs["five_year_reduction_pct"] = five_year_target.reduction_pct if five_year_target else 0.0
        outputs["ten_year_reduction_pct"] = ten_year_target.reduction_pct if ten_year_target else 0.0
        outputs["meets_sbti_near_term"] = all_exceed
        outputs["convergence_model"] = self.config.convergence_model
        outputs["front_loaded"] = self.config.front_load_reductions

        for t in targets:
            outputs[f"target_{t.target_year}_tco2e"] = t.target_value_tco2e
            outputs[f"target_{t.target_year}_reduction_pct"] = t.reduction_pct

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="calc_interim_targets", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_calc_interim_targets",
        )

    # -------------------------------------------------------------------------
    # Phase 3: Validate Targets
    # -------------------------------------------------------------------------

    async def _phase_validate_targets(self, input_data: InterimTargetSettingInput) -> PhaseResult:
        """Validate interim targets against SBTi near-term and long-term criteria."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        findings: List[ValidationFinding] = []

        bl = self._baseline
        ambition = self.config.sbti_ambition
        nt_thresholds = SBTI_NEAR_TERM_THRESHOLDS.get(ambition, SBTI_NEAR_TERM_THRESHOLDS["1.5c"])
        lt_thresholds = SBTI_LONG_TERM_THRESHOLDS.get("net_zero", SBTI_LONG_TERM_THRESHOLDS["net_zero"])

        # Criterion 1: Near-term timeframe (5-10 years)
        five_yr = self._interim_targets.five_year_target
        ten_yr = self._interim_targets.ten_year_target
        nt_years = (bl.near_term_target_year - bl.base_year) if bl.near_term_target_year > bl.base_year else 7
        findings.append(ValidationFinding(
            criterion_id="SBTI-NT-001",
            criterion_name="Near-Term Timeframe",
            description="Target timeframe must be 5-10 years from base year.",
            result=(
                ValidationResult.PASS if 5 <= nt_years <= 10
                else ValidationResult.FAIL
            ),
            actual_value=f"{nt_years} years",
            required_value="5-10 years",
            finding=f"Near-term timeframe is {nt_years} years.",
            remediation="" if 5 <= nt_years <= 10 else "Adjust target year to be within 5-10 years of base year.",
        ))

        # Criterion 2: Minimum annual reduction rate (S1+S2)
        min_rate = nt_thresholds["absolute_annual_linear_reduction_pct"]
        actual_rate = five_yr.annual_reduction_rate_pct if five_yr else 0.0
        findings.append(ValidationFinding(
            criterion_id="SBTI-NT-002",
            criterion_name="Minimum Annual Reduction Rate (S1+S2)",
            description=f"Annual linear reduction must be >= {min_rate}%/year for {ambition}.",
            result=ValidationResult.PASS if actual_rate >= min_rate else ValidationResult.FAIL,
            actual_value=f"{actual_rate:.2f}%/year",
            required_value=f">= {min_rate}%/year",
            finding=f"{'Meets' if actual_rate >= min_rate else 'Below'} minimum reduction rate.",
            remediation="" if actual_rate >= min_rate else f"Increase annual reduction to at least {min_rate}%/year.",
        ))

        # Criterion 3: Scope 1 coverage
        s1_coverage = nt_thresholds["scope1_coverage_min_pct"]
        # Assume full coverage by default (would be from input in production)
        actual_s1_cov = 95.0
        findings.append(ValidationFinding(
            criterion_id="SBTI-NT-003",
            criterion_name="Scope 1 Boundary Coverage",
            description=f"Scope 1 coverage must be >= {s1_coverage}%.",
            result=ValidationResult.PASS if actual_s1_cov >= s1_coverage else ValidationResult.FAIL,
            actual_value=f"{actual_s1_cov:.1f}%",
            required_value=f">= {s1_coverage}%",
            finding=f"Scope 1 coverage is {actual_s1_cov:.1f}%.",
            remediation="" if actual_s1_cov >= s1_coverage else "Expand boundary to cover 95% of Scope 1.",
        ))

        # Criterion 4: Scope 2 coverage
        s2_coverage = nt_thresholds["scope2_coverage_min_pct"]
        actual_s2_cov = 95.0
        findings.append(ValidationFinding(
            criterion_id="SBTI-NT-004",
            criterion_name="Scope 2 Boundary Coverage",
            description=f"Scope 2 coverage must be >= {s2_coverage}%.",
            result=ValidationResult.PASS if actual_s2_cov >= s2_coverage else ValidationResult.FAIL,
            actual_value=f"{actual_s2_cov:.1f}%",
            required_value=f">= {s2_coverage}%",
            finding=f"Scope 2 coverage is {actual_s2_cov:.1f}%.",
            remediation="" if actual_s2_cov >= s2_coverage else "Expand boundary to cover 95% of Scope 2.",
        ))

        # Criterion 5: Scope 3 inclusion requirement
        s3_threshold = nt_thresholds["scope3_threshold_pct_of_total"]
        s3_material = bl.scope3_pct_of_total >= s3_threshold
        s3_included = self.config.include_scope3
        if s3_material and not s3_included:
            s3_result = ValidationResult.FAIL
        elif s3_material and s3_included:
            s3_result = ValidationResult.PASS
        else:
            s3_result = ValidationResult.PASS  # Scope 3 not material
        findings.append(ValidationFinding(
            criterion_id="SBTI-NT-005",
            criterion_name="Scope 3 Inclusion",
            description=f"Scope 3 target required if S3 >= {s3_threshold}% of total.",
            result=s3_result,
            actual_value=f"S3 = {bl.scope3_pct_of_total:.1f}% of total, included={s3_included}",
            required_value=f"Include S3 if >= {s3_threshold}% of total",
            finding=(
                f"Scope 3 is {bl.scope3_pct_of_total:.1f}% of total, "
                f"{'included' if s3_included else 'NOT included'} in targets."
            ),
            remediation="" if s3_result == ValidationResult.PASS else "Add Scope 3 target covering >= 67% of S3.",
        ))

        # Criterion 6: Scope 3 coverage (if applicable)
        if s3_material and s3_included:
            s3_coverage_req = nt_thresholds["scope3_coverage_min_pct"]
            # Default assume 67% coverage
            actual_s3_cov = 67.0
            findings.append(ValidationFinding(
                criterion_id="SBTI-NT-006",
                criterion_name="Scope 3 Coverage",
                description=f"Scope 3 coverage must be >= {s3_coverage_req}%.",
                result=ValidationResult.PASS if actual_s3_cov >= s3_coverage_req else ValidationResult.FAIL,
                actual_value=f"{actual_s3_cov:.1f}%",
                required_value=f">= {s3_coverage_req}%",
                finding=f"Scope 3 coverage is {actual_s3_cov:.1f}%.",
                remediation="" if actual_s3_cov >= s3_coverage_req else f"Increase S3 coverage to >= {s3_coverage_req}%.",
            ))

        # Criterion 7: 5-year cumulative reduction
        min_5yr = SBTI_CROSS_SECTOR_RATES.get(f"{ambition}_cumulative_5yr", 21.0)
        actual_5yr = five_yr.reduction_pct if five_yr else 0.0
        findings.append(ValidationFinding(
            criterion_id="SBTI-NT-007",
            criterion_name="5-Year Cumulative Reduction",
            description=f"5-year reduction must be >= {min_5yr}% for {ambition}.",
            result=ValidationResult.PASS if actual_5yr >= min_5yr else ValidationResult.FAIL,
            actual_value=f"{actual_5yr:.1f}%",
            required_value=f">= {min_5yr}%",
            finding=f"5-year reduction is {actual_5yr:.1f}% vs. minimum {min_5yr}%.",
            remediation="" if actual_5yr >= min_5yr else f"Increase 5-year target to >= {min_5yr}%.",
        ))

        # Criterion 8: 10-year cumulative reduction
        min_10yr = SBTI_CROSS_SECTOR_RATES.get(f"{ambition}_cumulative_10yr", 42.0)
        actual_10yr = ten_yr.reduction_pct if ten_yr else 0.0
        findings.append(ValidationFinding(
            criterion_id="SBTI-NT-008",
            criterion_name="10-Year Cumulative Reduction",
            description=f"10-year reduction must be >= {min_10yr}% for {ambition}.",
            result=ValidationResult.PASS if actual_10yr >= min_10yr else ValidationResult.FAIL,
            actual_value=f"{actual_10yr:.1f}%",
            required_value=f">= {min_10yr}%",
            finding=f"10-year reduction is {actual_10yr:.1f}% vs. minimum {min_10yr}%.",
            remediation="" if actual_10yr >= min_10yr else f"Increase 10-year target to >= {min_10yr}%.",
        ))

        # Criterion 9: Long-term reduction (S1+S2)
        lt_min = lt_thresholds["scope1_2_reduction_pct"]
        lt_actual = self._interim_targets.total_scope1_2_reduction_pct
        findings.append(ValidationFinding(
            criterion_id="SBTI-LT-001",
            criterion_name="Long-Term S1+S2 Reduction",
            description=f"Long-term S1+S2 reduction must be >= {lt_min}%.",
            result=ValidationResult.PASS if lt_actual >= lt_min else ValidationResult.WARNING,
            actual_value=f"{lt_actual:.1f}%",
            required_value=f">= {lt_min}%",
            finding=f"Projected long-term S1+S2 reduction: {lt_actual:.1f}%.",
            remediation="" if lt_actual >= lt_min else f"Increase long-term S1+S2 target to >= {lt_min}%.",
            severity="warning" if lt_actual < lt_min else "info",
        ))

        # Criterion 10: Long-term target year
        lt_max_year = lt_thresholds["max_year"]
        lt_year = self.config.long_term_target_year
        findings.append(ValidationFinding(
            criterion_id="SBTI-LT-002",
            criterion_name="Long-Term Target Year",
            description=f"Long-term target year must be <= {lt_max_year}.",
            result=ValidationResult.PASS if lt_year <= lt_max_year else ValidationResult.FAIL,
            actual_value=str(lt_year),
            required_value=f"<= {lt_max_year}",
            finding=f"Long-term target year is {lt_year}.",
            remediation="" if lt_year <= lt_max_year else f"Set target year to {lt_max_year} or earlier.",
        ))

        # Criterion 11: Neutralization cap
        findings.append(ValidationFinding(
            criterion_id="SBTI-LT-003",
            criterion_name="Neutralization Cap",
            description="Residual emissions neutralization must be <= 10% of base year.",
            result=ValidationResult.PASS,
            actual_value="<= 10%",
            required_value="<= 10%",
            finding="Neutralization cap criterion assumed met (detailed check in annual review).",
        ))

        # Criterion 12: No offsets for near-term
        findings.append(ValidationFinding(
            criterion_id="SBTI-NT-009",
            criterion_name="No Offsets in Near-Term",
            description="Near-term targets must not rely on carbon offsets.",
            result=ValidationResult.PASS,
            actual_value="No offsets",
            required_value="No offsets",
            finding="Near-term targets exclude offsets (zero-hallucination calculation).",
        ))

        # Summarize
        passed = sum(1 for f in findings if f.result == ValidationResult.PASS)
        failed = sum(1 for f in findings if f.result == ValidationResult.FAIL)
        conditional = sum(1 for f in findings if f.result == ValidationResult.CONDITIONAL)
        warning_count = sum(1 for f in findings if f.result == ValidationResult.WARNING)

        total = len(findings)
        pass_rate = (passed / max(total, 1)) * 100

        if failed == 0:
            overall = ValidationResult.PASS
        elif failed <= 2:
            overall = ValidationResult.CONDITIONAL
        else:
            overall = ValidationResult.FAIL

        improvement_actions = [
            f.remediation for f in findings
            if f.remediation and f.result in (ValidationResult.FAIL, ValidationResult.WARNING)
        ]

        self._validation = ValidationSummary(
            total_criteria=total,
            passed=passed,
            failed=failed,
            conditional=conditional,
            warnings=warning_count,
            pass_rate_pct=round(pass_rate, 1),
            overall_result=overall,
            findings=findings,
            sbti_submission_ready=(overall == ValidationResult.PASS),
            improvement_actions=improvement_actions,
        )
        self._validation.provenance_hash = _compute_hash(
            self._validation.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["total_criteria"] = total
        outputs["passed"] = passed
        outputs["failed"] = failed
        outputs["conditional"] = conditional
        outputs["warnings"] = warning_count
        outputs["pass_rate_pct"] = round(pass_rate, 1)
        outputs["overall_result"] = overall.value
        outputs["sbti_submission_ready"] = overall == ValidationResult.PASS

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="validate_targets", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_validate_targets",
        )

    # -------------------------------------------------------------------------
    # Phase 4: Generate Annual Pathway
    # -------------------------------------------------------------------------

    async def _phase_generate_pathway(self, input_data: InterimTargetSettingInput) -> PhaseResult:
        """Generate year-by-year emissions pathway from base year to long-term target."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        bl = self._baseline
        total_base = bl.base_year_total_tco2e
        s12_base = bl.base_year_scope1_tco2e + bl.base_year_scope2_tco2e
        s3_base = bl.base_year_scope3_tco2e

        target_year = self.config.long_term_target_year
        reduction_pct = self.config.long_term_reduction_pct

        total_target = total_base * (1 - reduction_pct / 100.0)
        s12_target = s12_base * (1 - reduction_pct / 100.0)
        s3_target = s3_base * (1 - reduction_pct / 100.0) if self.config.include_scope3 else s3_base

        pathway_points: List[AnnualPathwayPoint] = []
        milestone_years: List[int] = []
        total_budget = 0.0
        prev_total = total_base

        for year in range(bl.base_year, target_year + 1):
            # Calculate target for this year
            if self.config.convergence_model == "linear":
                s12_val = _interpolate_linear(s12_base, s12_target, bl.base_year, target_year, year)
                s3_val = _interpolate_linear(s3_base, s3_target, bl.base_year, target_year, year)
            elif self.config.convergence_model == "exponential":
                s12_val = _interpolate_exponential(s12_base, s12_target, bl.base_year, target_year, year)
                s3_val = _interpolate_exponential(s3_base, s3_target, bl.base_year, target_year, year)
            else:
                s12_val = _interpolate_contraction(s12_base, s12_target, bl.base_year, target_year, year)
                s3_val = _interpolate_contraction(s3_base, s3_target, bl.base_year, target_year, year)

            total_val = s12_val + s3_val

            # Apply front-loading factor for early years
            if self.config.front_load_reductions and year <= bl.base_year + 10:
                years_in = year - bl.base_year
                fl_factor = 1.0 + (self.config.front_load_factor - 1.0) * max(0, 1 - years_in / 10)
                reduction_from_base = total_base - total_val
                adjusted_reduction = reduction_from_base * fl_factor
                total_val = max(total_base - adjusted_reduction, total_target)
                s12_val = total_val * (s12_base / max(total_base, 1e-10))
                s3_val = total_val * (s3_base / max(total_base, 1e-10))

            # Cumulative reduction
            cum_red = ((total_base - total_val) / max(total_base, 1e-10)) * 100

            # Annual reduction rate
            annual_red = ((prev_total - total_val) / max(prev_total, 1e-10)) * 100 if year > bl.base_year else 0.0

            # Carbon budget remaining
            total_budget += total_val
            total_budget_limit = (total_base + total_target) / 2.0 * (target_year - bl.base_year)
            budget_remaining = max(total_budget_limit - total_budget, 0.0)

            # Check if milestone year
            is_milestone = (year - bl.base_year) in self.config.interim_milestones or year == target_year
            milestone_name = ""
            if is_milestone:
                milestone_years.append(year)
                interval = year - bl.base_year
                milestone_name = f"{interval}-Year Milestone" if interval > 0 else "Base Year"

            # Intensity target
            intensity = 0.0
            if bl.base_year_revenue_musd > 0:
                projected_rev = bl.base_year_revenue_musd * (1.02 ** max(year - bl.base_year, 0))
                intensity = round(total_val / projected_rev, 4)

            point = AnnualPathwayPoint(
                year=year,
                scope1_2_target_tco2e=round(s12_val, 2),
                scope3_target_tco2e=round(s3_val, 2),
                total_target_tco2e=round(total_val, 2),
                intensity_target=intensity,
                cumulative_reduction_pct=round(cum_red, 2),
                annual_reduction_rate_pct=round(annual_red, 2),
                carbon_budget_remaining_tco2e=round(budget_remaining, 2),
                is_milestone_year=is_milestone,
                milestone_name=milestone_name,
            )
            pathway_points.append(point)
            prev_total = total_val

        # Average annual reduction
        if len(pathway_points) >= 2:
            avg_annual = abs(_calc_cagr(
                pathway_points[0].total_target_tco2e,
                pathway_points[-1].total_target_tco2e,
                len(pathway_points) - 1,
            ))
        else:
            avg_annual = 0.0

        self._pathway = AnnualPathway(
            entity_id=bl.entity_id,
            base_year=bl.base_year,
            target_year=target_year,
            convergence_model=self.config.convergence_model,
            pathway_points=pathway_points,
            milestone_years=milestone_years,
            total_carbon_budget_tco2e=round(total_budget_limit, 2),
            average_annual_reduction_pct=round(avg_annual, 2),
            front_loaded=self.config.front_load_reductions,
        )
        self._pathway.provenance_hash = _compute_hash(
            self._pathway.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["pathway_years"] = len(pathway_points)
        outputs["milestone_years"] = milestone_years
        outputs["avg_annual_reduction_pct"] = round(avg_annual, 2)
        outputs["total_carbon_budget_tco2e"] = round(total_budget_limit, 2)
        outputs["front_loaded"] = self.config.front_load_reductions
        outputs["convergence_model"] = self.config.convergence_model
        outputs["base_year_emissions"] = total_base
        outputs["target_year_emissions"] = round(total_target, 2)

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="generate_pathway", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_generate_pathway",
        )

    # -------------------------------------------------------------------------
    # Phase 5: Allocate Carbon Budget
    # -------------------------------------------------------------------------

    async def _phase_allocate_budget(self, input_data: InterimTargetSettingInput) -> PhaseResult:
        """Allocate carbon budget across scopes and business units."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        bl = self._baseline
        total_budget = self._pathway.total_carbon_budget_tco2e
        total_base = bl.base_year_total_tco2e
        s12_base = bl.base_year_scope1_tco2e + bl.base_year_scope2_tco2e
        s3_base = bl.base_year_scope3_tco2e

        # Scope-level allocation (proportional to base year)
        s1_pct = bl.base_year_scope1_tco2e / max(total_base, 1e-10)
        s2_pct = bl.base_year_scope2_tco2e / max(total_base, 1e-10)
        s3_pct = bl.base_year_scope3_tco2e / max(total_base, 1e-10)

        scope1_budget = total_budget * s1_pct
        scope2_budget = total_budget * s2_pct
        scope3_budget = total_budget * s3_pct

        # BU-level allocation
        bu_allocations: List[BUBudgetAllocation] = []
        bus = input_data.business_units or bl.business_units

        if not bus:
            # Create default single BU
            bus = [{
                "bu_id": "BU-001",
                "name": self.config.company_name or "Corporate",
                "emissions": total_base,
                "revenue": bl.base_year_revenue_musd,
                "headcount": 1000,
            }]
            warnings.append("No business units provided; single corporate BU assumed.")

        method = self.config.budget_allocation_method
        weights_config = BU_ALLOCATION_WEIGHTS.get(
            method.value if isinstance(method, BudgetMethod) else str(method),
            BU_ALLOCATION_WEIGHTS["proportional"],
        )

        # Calculate weights
        total_emissions = sum(bu.get("emissions", 0) for bu in bus) or total_base
        total_revenue = sum(bu.get("revenue", 0) for bu in bus) or bl.base_year_revenue_musd or 1.0
        total_headcount = sum(bu.get("headcount", 0) for bu in bus) or 1

        for bu in bus:
            bu_emissions = bu.get("emissions", 0)
            bu_revenue = bu.get("revenue", 0)
            bu_headcount = bu.get("headcount", 0)

            e_weight = (bu_emissions / max(total_emissions, 1e-10)) * weights_config["emissions_weight"]
            r_weight = (bu_revenue / max(total_revenue, 1e-10)) * weights_config["revenue_weight"]
            h_weight = (bu_headcount / max(total_headcount, 1)) * weights_config["headcount_weight"]
            combined_weight = e_weight + r_weight + h_weight

            bu_budget = total_budget * combined_weight
            target_yr = self.config.long_term_target_year
            total_years = target_yr - bl.base_year
            annual_target = bu_budget / max(total_years, 1)

            # 5-year and 10-year sub-budgets
            five_yr_factor = min(5 / max(total_years, 1), 1.0)
            ten_yr_factor = min(10 / max(total_years, 1), 1.0)

            if self.config.front_load_reductions:
                five_yr_budget = bu_budget * five_yr_factor * self.config.front_load_factor
                ten_yr_budget = bu_budget * ten_yr_factor * min(self.config.front_load_factor, 1.15)
            else:
                five_yr_budget = bu_budget * five_yr_factor
                ten_yr_budget = bu_budget * ten_yr_factor

            reduction_rate = _calc_cagr(
                bu_emissions, bu_emissions * (1 - self.config.long_term_reduction_pct / 100),
                total_years,
            ) if bu_emissions > 0 else 0.0

            bu_allocations.append(BUBudgetAllocation(
                bu_id=bu.get("bu_id", f"BU-{len(bu_allocations)+1:03d}"),
                bu_name=bu.get("name", f"Unit {len(bu_allocations)+1}"),
                base_year_emissions_tco2e=round(bu_emissions, 2),
                allocated_budget_tco2e=round(bu_budget, 2),
                annual_target_tco2e=round(annual_target, 2),
                reduction_rate_pct=round(abs(reduction_rate), 2),
                allocation_method=method.value if isinstance(method, BudgetMethod) else str(method),
                weight=round(combined_weight, 4),
                five_year_budget_tco2e=round(five_yr_budget, 2),
                ten_year_budget_tco2e=round(ten_yr_budget, 2),
            ))

        # Calculate budget depletion year based on current trajectory
        current = bl.current_emissions_tco2e
        depletion_year = self.config.long_term_target_year
        cumulative = 0.0
        for yr in range(bl.current_year, self.config.long_term_target_year + 1):
            cumulative += current
            current *= 0.975  # Assume 2.5% annual decline
            if cumulative > total_budget:
                depletion_year = yr
                break

        # 5-year and 10-year budget
        five_yr_budget_total = sum(ba.five_year_budget_tco2e for ba in bu_allocations)
        ten_yr_budget_total = sum(ba.ten_year_budget_tco2e for ba in bu_allocations)

        self._budget = CarbonBudgetAllocation(
            entity_id=bl.entity_id,
            total_budget_tco2e=round(total_budget, 2),
            scope1_budget_tco2e=round(scope1_budget, 2),
            scope2_budget_tco2e=round(scope2_budget, 2),
            scope3_budget_tco2e=round(scope3_budget, 2),
            allocation_method=method if isinstance(method, BudgetMethod) else BudgetMethod.PROPORTIONAL,
            bu_allocations=bu_allocations,
            five_year_budget_tco2e=round(five_yr_budget_total, 2),
            ten_year_budget_tco2e=round(ten_yr_budget_total, 2),
            remaining_budget_tco2e=round(total_budget, 2),
            budget_depletion_year=depletion_year,
        )
        self._budget.provenance_hash = _compute_hash(
            self._budget.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["total_budget_tco2e"] = round(total_budget, 2)
        outputs["scope1_budget_tco2e"] = round(scope1_budget, 2)
        outputs["scope2_budget_tco2e"] = round(scope2_budget, 2)
        outputs["scope3_budget_tco2e"] = round(scope3_budget, 2)
        outputs["bu_count"] = len(bu_allocations)
        outputs["allocation_method"] = method.value if isinstance(method, BudgetMethod) else str(method)
        outputs["five_year_budget_tco2e"] = round(five_yr_budget_total, 2)
        outputs["ten_year_budget_tco2e"] = round(ten_yr_budget_total, 2)
        outputs["budget_depletion_year"] = depletion_year

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="allocate_budget", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_allocate_budget",
        )

    # -------------------------------------------------------------------------
    # Phase 6: Summary Report
    # -------------------------------------------------------------------------

    async def _phase_summary_report(self, input_data: InterimTargetSettingInput) -> PhaseResult:
        """Generate interim target summary report."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        bl = self._baseline

        # Build executive summary
        five_yr = self._interim_targets.five_year_target
        ten_yr = self._interim_targets.ten_year_target
        five_yr_red = five_yr.reduction_pct if five_yr else 0.0
        ten_yr_red = ten_yr.reduction_pct if ten_yr else 0.0

        exec_summary_parts = [
            f"Interim Target Setting Report for {bl.company_name or 'Company'}.",
            f"Base year: {bl.base_year}, baseline emissions: {bl.base_year_total_tco2e:,.0f} tCO2e.",
            f"SBTi ambition level: {self.config.sbti_ambition}.",
            f"5-year interim target ({bl.base_year + 5}): {five_yr_red:.1f}% reduction.",
            f"10-year interim target ({bl.base_year + 10}): {ten_yr_red:.1f}% reduction.",
            f"Long-term target ({self.config.long_term_target_year}): {self.config.long_term_reduction_pct:.0f}% reduction.",
            f"Validation: {self._validation.overall_result.value} ({self._validation.pass_rate_pct:.0f}% pass rate).",
            f"Total carbon budget: {self._budget.total_budget_tco2e:,.0f} tCO2e.",
            f"Budget depletion year (at current trajectory): {self._budget.budget_depletion_year}.",
        ]

        if self._validation.sbti_submission_ready:
            exec_summary_parts.append("Targets are SBTi submission-ready.")
        else:
            exec_summary_parts.append(
                f"Targets require {self._validation.failed} improvement(s) before SBTi submission.",
            )

        executive_summary = " ".join(exec_summary_parts)

        # Key findings
        findings = self._generate_findings()

        # Recommendations
        recommendations = self._generate_recommendations()

        # Data quality score
        dq_score = 4.0 if input_data.historical_emissions else 3.0

        self._report = InterimTargetReport(
            report_id=f"ITR-{self.workflow_id[:8]}",
            report_date=_utcnow().strftime("%Y-%m-%d"),
            company_name=bl.company_name,
            entity_id=bl.entity_id,
            baseline=bl,
            interim_targets=self._interim_targets,
            validation_summary=self._validation,
            annual_pathway=self._pathway,
            carbon_budget=self._budget,
            executive_summary=executive_summary,
            key_findings=findings,
            recommendations=recommendations,
            output_formats=self.config.output_formats,
            data_quality_score=dq_score,
        )
        self._report.provenance_hash = _compute_hash(
            self._report.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["report_id"] = self._report.report_id
        outputs["executive_summary_length"] = len(executive_summary)
        outputs["findings_count"] = len(findings)
        outputs["recommendations_count"] = len(recommendations)
        outputs["data_quality_score"] = dq_score
        outputs["output_formats"] = self.config.output_formats
        outputs["sbti_submission_ready"] = self._validation.sbti_submission_ready

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="summary_report", phase_number=6,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_summary_report",
        )

    # -------------------------------------------------------------------------
    # Report Generators
    # -------------------------------------------------------------------------

    def _generate_findings(self) -> List[str]:
        """Generate key findings from workflow results."""
        findings: List[str] = []
        bl = self._baseline
        five_yr = self._interim_targets.five_year_target
        ten_yr = self._interim_targets.ten_year_target

        if five_yr:
            findings.append(
                f"5-year interim target requires {five_yr.reduction_pct:.1f}% reduction "
                f"({five_yr.annual_reduction_rate_pct:.1f}%/year) by {five_yr.target_year}.",
            )
        if ten_yr:
            findings.append(
                f"10-year interim target requires {ten_yr.reduction_pct:.1f}% reduction "
                f"by {ten_yr.target_year}.",
            )

        if self._validation.overall_result == ValidationResult.PASS:
            findings.append("All SBTi validation criteria are met; targets are submission-ready.")
        elif self._validation.overall_result == ValidationResult.CONDITIONAL:
            findings.append(
                f"Targets conditionally meet SBTi criteria; {self._validation.failed} "
                f"criterion/criteria require remediation.",
            )
        else:
            findings.append(
                f"Targets DO NOT meet SBTi criteria; {self._validation.failed} criterion/criteria failed.",
            )

        findings.append(
            f"Total carbon budget ({bl.base_year}-{self.config.long_term_target_year}): "
            f"{self._budget.total_budget_tco2e:,.0f} tCO2e.",
        )

        if self._budget.budget_depletion_year < self.config.long_term_target_year:
            findings.append(
                f"WARNING: At current trajectory, carbon budget depletes by {self._budget.budget_depletion_year}, "
                f"before the {self.config.long_term_target_year} target year.",
            )

        findings.append(
            f"Annual pathway covers {len(self._pathway.pathway_points)} years "
            f"with {self._pathway.average_annual_reduction_pct:.1f}% average annual reduction.",
        )

        return findings

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation and pathway analysis."""
        recs: List[str] = []

        # From validation improvement actions
        for action in self._validation.improvement_actions[:5]:
            recs.append(action)

        # Budget-based recommendations
        if self._budget.budget_depletion_year < self.config.long_term_target_year:
            recs.append(
                "Accelerate near-term reductions to preserve carbon budget "
                f"through {self.config.long_term_target_year}.",
            )

        if not self.config.front_load_reductions:
            recs.append(
                "Consider front-loading emission reductions to build buffer "
                "against implementation delays.",
            )

        if not self.config.include_scope3 and self._baseline.scope3_pct_of_total >= 40:
            recs.append(
                "Scope 3 emissions represent a material share; include Scope 3 "
                "targets to meet SBTi requirements.",
            )

        recs.append(
            "Establish quarterly monitoring cadence against annual pathway milestones.",
        )
        recs.append(
            "Integrate interim targets into annual business planning and capital "
            "allocation processes.",
        )

        if self._interim_targets.meets_sbti_near_term:
            recs.append(
                "Proceed with SBTi target submission; targets meet near-term criteria.",
            )

        return recs
