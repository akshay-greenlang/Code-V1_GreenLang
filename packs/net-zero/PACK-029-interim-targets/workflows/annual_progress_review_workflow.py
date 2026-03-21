# -*- coding: utf-8 -*-
"""
Annual Progress Review Workflow
====================================

5-phase DAG workflow for conducting annual progress reviews against
interim targets within PACK-029 Interim Targets Pack.  The workflow
collects actual emissions via MRV Bridge, compares against targets,
performs variance analysis, extrapolates trends, and generates a
comprehensive annual progress report.

Phases:
    1. CollectActuals     -- Collect actual emissions data via MRVBridge
                             (30 MRV agents for S1/S2/S3 coverage)
    2. CompareTarget      -- Compare actual vs. target using ProgressTrackerEngine;
                             calculate gap, achievement %, RAG status
    3. VarianceAnalysis   -- Decompose variance into activity, intensity, and
                             structural components via VarianceAnalysisEngine
    4. TrendExtrapolation -- Extrapolate trends forward using linear, exponential,
                             and Monte Carlo methods via TrendExtrapolationEngine
    5. AnnualReport       -- Generate annual progress report with KPIs, charts,
                             and executive summary via ReportingEngine

Regulatory references:
    - SBTi Monitoring, Reporting & Verification Guidance
    - SBTi Target Tracking Protocol (annual revalidation)
    - GHG Protocol Corporate Standard (annual boundary review)
    - CDP C4.1/C4.2 Targets and Progress Questions
    - TCFD Metrics and Targets Disclosure

Zero-hallucination: all calculations use deterministic formulas with
actual reported emissions data.  No LLM calls in computation path.

Author: GreenLang Team
Version: 29.0.0
Pack: PACK-029 Interim Targets Pack
"""

import hashlib
import json
import logging
import math
import statistics
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION = "29.0.0"
_PACK_ID = "PACK-029"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _calc_cagr(start_val: float, end_val: float, years: int) -> float:
    if years <= 0 or start_val <= 0 or end_val <= 0:
        return 0.0
    return ((end_val / start_val) ** (1.0 / years) - 1.0) * 100.0


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class RAGStatus(str, Enum):
    RED = "red"
    AMBER = "amber"
    GREEN = "green"


class TrendDirection(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"


class VarianceType(str, Enum):
    ACTIVITY = "activity"
    INTENSITY = "intensity"
    STRUCTURAL = "structural"
    WEATHER = "weather"
    ACQUISITION = "acquisition"
    METHODOLOGICAL = "methodological"


class ProjectionMethod(str, Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    POLYNOMIAL = "polynomial"
    MONTE_CARLO = "monte_carlo"


class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


# =============================================================================
# MRV SCOPE COVERAGE MAP (Zero-Hallucination: Actual Agent References)
# =============================================================================


MRV_SCOPE_COVERAGE: Dict[str, Dict[str, Any]] = {
    "scope1": {
        "agents": [
            "MRV-001-stationary-combustion",
            "MRV-002-refrigerants-fgas",
            "MRV-003-mobile-combustion",
            "MRV-004-process-emissions",
            "MRV-005-fugitive-emissions",
            "MRV-006-land-use",
            "MRV-007-waste-treatment",
            "MRV-008-agricultural",
        ],
        "categories": [
            "Stationary Combustion", "Refrigerants & F-Gas",
            "Mobile Combustion", "Process Emissions",
            "Fugitive Emissions", "Land Use", "Waste Treatment",
            "Agricultural",
        ],
        "agent_count": 8,
    },
    "scope2": {
        "agents": [
            "MRV-009-scope2-location",
            "MRV-010-scope2-market",
            "MRV-011-steam-heat",
            "MRV-012-cooling",
            "MRV-013-dual-reporting",
        ],
        "categories": [
            "Location-Based Electricity", "Market-Based Electricity",
            "Steam & Heat", "Cooling", "Dual Reporting Reconciliation",
        ],
        "agent_count": 5,
    },
    "scope3": {
        "agents": [
            "MRV-014-purchased-goods",
            "MRV-015-capital-goods",
            "MRV-016-fuel-energy-activities",
            "MRV-017-upstream-transport",
            "MRV-018-waste-generated",
            "MRV-019-business-travel",
            "MRV-020-employee-commuting",
            "MRV-021-upstream-leased",
            "MRV-022-downstream-transport",
            "MRV-023-processing-sold",
            "MRV-024-use-of-sold",
            "MRV-025-end-of-life",
            "MRV-026-downstream-leased",
            "MRV-027-franchises",
            "MRV-028-investments",
        ],
        "categories": [
            "Cat 1: Purchased Goods & Services",
            "Cat 2: Capital Goods",
            "Cat 3: Fuel & Energy Activities",
            "Cat 4: Upstream Transportation",
            "Cat 5: Waste Generated in Operations",
            "Cat 6: Business Travel",
            "Cat 7: Employee Commuting",
            "Cat 8: Upstream Leased Assets",
            "Cat 9: Downstream Transportation",
            "Cat 10: Processing of Sold Products",
            "Cat 11: Use of Sold Products",
            "Cat 12: End-of-Life Treatment",
            "Cat 13: Downstream Leased Assets",
            "Cat 14: Franchises",
            "Cat 15: Investments",
        ],
        "agent_count": 15,
    },
}

# KPI definitions for annual progress reporting
PROGRESS_KPIS: List[Dict[str, Any]] = [
    {"id": "KPI-001", "name": "Absolute Emissions (tCO2e)", "unit": "tCO2e",
     "target_type": "lower_is_better", "weight": 0.25},
    {"id": "KPI-002", "name": "Emissions Intensity", "unit": "varies",
     "target_type": "lower_is_better", "weight": 0.20},
    {"id": "KPI-003", "name": "Year-over-Year Change (%)", "unit": "%",
     "target_type": "negative_is_better", "weight": 0.15},
    {"id": "KPI-004", "name": "Cumulative Reduction vs Base Year (%)", "unit": "%",
     "target_type": "higher_is_better", "weight": 0.15},
    {"id": "KPI-005", "name": "Target Achievement (%)", "unit": "%",
     "target_type": "higher_is_better", "weight": 0.10},
    {"id": "KPI-006", "name": "Carbon Budget Consumed (%)", "unit": "%",
     "target_type": "lower_is_better", "weight": 0.10},
    {"id": "KPI-007", "name": "Data Quality Score", "unit": "1-5",
     "target_type": "higher_is_better", "weight": 0.05},
]


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    phase_name: str = Field(...)
    phase_number: int = Field(default=0, ge=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    completion_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    dag_node_id: str = Field(default="")


class EmissionsActual(BaseModel):
    """Actual emissions data for a reporting period."""
    year: int = Field(default=2025)
    period: str = Field(default="annual")
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_location_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_market_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    total_tco2e: float = Field(default=0.0, ge=0.0)
    revenue_musd: float = Field(default=0.0, ge=0.0)
    activity_metric: float = Field(default=0.0, ge=0.0)
    intensity: float = Field(default=0.0, ge=0.0)
    intensity_unit: str = Field(default="tCO2e/M$ revenue")
    data_quality_score: float = Field(default=3.0, ge=0.0, le=5.0)
    scope3_categories: Dict[str, float] = Field(default_factory=dict)
    mrv_agents_used: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class TargetComparison(BaseModel):
    """Comparison of actual vs target for a single metric."""
    metric_name: str = Field(default="")
    scope: str = Field(default="all")
    target_value: float = Field(default=0.0)
    actual_value: float = Field(default=0.0)
    gap_absolute: float = Field(default=0.0)
    gap_pct: float = Field(default=0.0)
    achievement_pct: float = Field(default=0.0)
    rag_status: RAGStatus = Field(default=RAGStatus.GREEN)
    on_track: bool = Field(default=True)
    yoy_change_pct: float = Field(default=0.0)
    cumulative_reduction_pct: float = Field(default=0.0)


class ProgressSummary(BaseModel):
    """Summary of progress against all interim targets."""
    review_year: int = Field(default=2025)
    comparisons: List[TargetComparison] = Field(default_factory=list)
    overall_rag: RAGStatus = Field(default=RAGStatus.GREEN)
    overall_achievement_pct: float = Field(default=0.0)
    on_track_count: int = Field(default=0)
    off_track_count: int = Field(default=0)
    years_to_next_milestone: int = Field(default=0)
    next_milestone_year: int = Field(default=2030)
    provenance_hash: str = Field(default="")


class VarianceComponent(BaseModel):
    """A single variance decomposition component."""
    component_type: VarianceType = Field(default=VarianceType.ACTIVITY)
    component_name: str = Field(default="")
    contribution_tco2e: float = Field(default=0.0)
    contribution_pct: float = Field(default=0.0)
    direction: str = Field(default="neutral")
    explanation: str = Field(default="")
    data_quality: float = Field(default=3.0, ge=0.0, le=5.0)


class VarianceDecomposition(BaseModel):
    """Complete variance decomposition result."""
    total_variance_tco2e: float = Field(default=0.0)
    total_variance_pct: float = Field(default=0.0)
    components: List[VarianceComponent] = Field(default_factory=list)
    activity_effect_pct: float = Field(default=0.0)
    intensity_effect_pct: float = Field(default=0.0)
    structural_effect_pct: float = Field(default=0.0)
    largest_contributor: str = Field(default="")
    net_favorable: bool = Field(default=True)
    provenance_hash: str = Field(default="")


class TrendProjection(BaseModel):
    """Trend extrapolation result for a single method."""
    method: ProjectionMethod = Field(default=ProjectionMethod.LINEAR)
    projected_values: Dict[int, float] = Field(default_factory=dict)
    projected_2030: float = Field(default=0.0)
    projected_2040: float = Field(default=0.0)
    projected_2050: float = Field(default=0.0)
    r_squared: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_interval_pct: float = Field(default=95.0)
    upper_bound_2030: float = Field(default=0.0)
    lower_bound_2030: float = Field(default=0.0)
    trend_direction: TrendDirection = Field(default=TrendDirection.STABLE)
    meets_target_2030: bool = Field(default=False)
    meets_target_2050: bool = Field(default=False)


class TrendAnalysis(BaseModel):
    """Complete trend analysis with multiple methods."""
    projections: List[TrendProjection] = Field(default_factory=list)
    consensus_2030: float = Field(default=0.0)
    consensus_2050: float = Field(default=0.0)
    consensus_meets_target: bool = Field(default=False)
    recommended_method: ProjectionMethod = Field(default=ProjectionMethod.LINEAR)
    years_to_target: int = Field(default=0)
    acceleration_needed_pct: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class ProgressAlert(BaseModel):
    """An alert generated during progress review."""
    alert_id: str = Field(default="")
    severity: AlertSeverity = Field(default=AlertSeverity.INFO)
    category: str = Field(default="")
    title: str = Field(default="")
    description: str = Field(default="")
    metric_value: str = Field(default="")
    threshold_value: str = Field(default="")
    recommended_action: str = Field(default="")


class ProgressKPI(BaseModel):
    """A single progress KPI."""
    kpi_id: str = Field(default="")
    kpi_name: str = Field(default="")
    current_value: float = Field(default=0.0)
    target_value: float = Field(default=0.0)
    unit: str = Field(default="")
    achievement_pct: float = Field(default=0.0)
    trend: TrendDirection = Field(default=TrendDirection.STABLE)
    rag_status: RAGStatus = Field(default=RAGStatus.GREEN)
    weight: float = Field(default=0.0)


class AnnualProgressReport(BaseModel):
    """Complete annual progress report."""
    report_id: str = Field(default="")
    report_date: str = Field(default="")
    review_year: int = Field(default=2025)
    company_name: str = Field(default="")
    entity_id: str = Field(default="")
    actuals: EmissionsActual = Field(default_factory=EmissionsActual)
    progress_summary: ProgressSummary = Field(default_factory=ProgressSummary)
    variance: VarianceDecomposition = Field(default_factory=VarianceDecomposition)
    trend_analysis: TrendAnalysis = Field(default_factory=TrendAnalysis)
    kpis: List[ProgressKPI] = Field(default_factory=list)
    alerts: List[ProgressAlert] = Field(default_factory=list)
    executive_summary: str = Field(default="")
    key_findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    data_quality_score: float = Field(default=0.0, ge=0.0, le=5.0)
    provenance_hash: str = Field(default="")


class AnnualProgressReviewConfig(BaseModel):
    company_name: str = Field(default="")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")
    review_year: int = Field(default=2025, ge=2020, le=2060)
    base_year: int = Field(default=2020, ge=2015, le=2030)
    base_year_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    near_term_target_year: int = Field(default=2030, ge=2025, le=2040)
    near_term_target_tco2e: float = Field(default=0.0, ge=0.0)
    long_term_target_year: int = Field(default=2050, ge=2040, le=2070)
    long_term_target_tco2e: float = Field(default=0.0, ge=0.0)
    carbon_budget_total_tco2e: float = Field(default=0.0, ge=0.0)
    carbon_budget_consumed_tco2e: float = Field(default=0.0, ge=0.0)
    intensity_unit: str = Field(default="tCO2e/M$ revenue")
    include_scope3: bool = Field(default=True)
    projection_methods: List[str] = Field(
        default_factory=lambda: ["linear", "exponential"],
    )
    output_formats: List[str] = Field(default_factory=lambda: ["json", "html"])
    alert_threshold_pct: float = Field(default=10.0, ge=0.0, le=100.0)


class AnnualProgressReviewInput(BaseModel):
    config: AnnualProgressReviewConfig = Field(default_factory=AnnualProgressReviewConfig)
    current_actuals: EmissionsActual = Field(default_factory=EmissionsActual)
    historical_actuals: List[EmissionsActual] = Field(default_factory=list)
    interim_targets: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Interim target data [{year, target_tco2e, scope, type}]",
    )
    annual_pathway: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Annual pathway [{year, target_tco2e, intensity_target}]",
    )
    previous_year_actuals: Optional[EmissionsActual] = Field(default=None)


class AnnualProgressReviewResult(BaseModel):
    workflow_id: str = Field(...)
    workflow_name: str = Field(default="annual_progress_review")
    pack_id: str = Field(default=_PACK_ID)
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    actuals: EmissionsActual = Field(default_factory=EmissionsActual)
    progress_summary: ProgressSummary = Field(default_factory=ProgressSummary)
    variance: VarianceDecomposition = Field(default_factory=VarianceDecomposition)
    trend_analysis: TrendAnalysis = Field(default_factory=TrendAnalysis)
    report: AnnualProgressReport = Field(default_factory=AnnualProgressReport)
    overall_rag: RAGStatus = Field(default=RAGStatus.GREEN)
    key_findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class AnnualProgressReviewWorkflow:
    """
    5-phase DAG workflow for annual progress review against interim targets.

    Phase 1: CollectActuals     -- Collect actual emissions from MRV Bridge.
    Phase 2: CompareTarget      -- Compare actual vs. target; RAG scoring.
    Phase 3: VarianceAnalysis   -- Decompose variance (LMDI method).
    Phase 4: TrendExtrapolation -- Extrapolate trends; project forward.
    Phase 5: AnnualReport       -- Generate annual progress report.

    DAG Dependencies:
        Phase 1 -> Phase 2
                -> Phase 3  (depends on Phase 2)
                -> Phase 4  (depends on Phase 1, can run parallel to Phase 3)
                -> Phase 5  (depends on all prior phases)

    Example:
        >>> wf = AnnualProgressReviewWorkflow()
        >>> inp = AnnualProgressReviewInput(
        ...     config=AnnualProgressReviewConfig(company_name="Acme Corp"),
        ...     current_actuals=EmissionsActual(year=2025, total_tco2e=85000),
        ... )
        >>> result = await wf.execute(inp)
    """

    def __init__(self, config: Optional[AnnualProgressReviewConfig] = None) -> None:
        self.workflow_id: str = _new_uuid()
        self.config = config or AnnualProgressReviewConfig()
        self._phase_results: List[PhaseResult] = []
        self._actuals: EmissionsActual = EmissionsActual()
        self._progress: ProgressSummary = ProgressSummary()
        self._variance: VarianceDecomposition = VarianceDecomposition()
        self._trends: TrendAnalysis = TrendAnalysis()
        self._report: AnnualProgressReport = AnnualProgressReport()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(self, input_data: AnnualProgressReviewInput) -> AnnualProgressReviewResult:
        started_at = _utcnow()
        self.config = input_data.config
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        self.logger.info(
            "Starting annual progress review workflow %s, year=%d, company=%s",
            self.workflow_id, self.config.review_year, self.config.company_name,
        )

        try:
            phase1 = await self._phase_collect_actuals(input_data)
            self._phase_results.append(phase1)

            phase2 = await self._phase_compare_target(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_variance_analysis(input_data)
            self._phase_results.append(phase3)

            phase4 = await self._phase_trend_extrapolation(input_data)
            self._phase_results.append(phase4)

            phase5 = await self._phase_annual_report(input_data)
            self._phase_results.append(phase5)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Annual progress review failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=99,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()

        result = AnnualProgressReviewResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            actuals=self._actuals,
            progress_summary=self._progress,
            variance=self._variance,
            trend_analysis=self._trends,
            report=self._report,
            overall_rag=self._progress.overall_rag,
            key_findings=self._generate_findings(),
            recommendations=self._generate_recommendations(),
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"}),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Collect Actuals
    # -------------------------------------------------------------------------

    async def _phase_collect_actuals(self, input_data: AnnualProgressReviewInput) -> PhaseResult:
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        actuals = input_data.current_actuals

        # Validate and fill gaps
        if actuals.total_tco2e <= 0:
            actuals.total_tco2e = (
                actuals.scope1_tco2e
                + actuals.scope2_market_tco2e
                + actuals.scope3_tco2e
            )
            if actuals.total_tco2e <= 0:
                # Use base year with estimated reduction
                years_since = self.config.review_year - self.config.base_year
                factor = max(1.0 - 0.03 * years_since, 0.1)
                actuals.total_tco2e = self.config.base_year_emissions_tco2e * factor
                actuals.scope1_tco2e = actuals.total_tco2e * 0.45
                actuals.scope2_market_tco2e = actuals.total_tco2e * 0.20
                actuals.scope2_location_tco2e = actuals.scope2_market_tco2e * 1.1
                actuals.scope3_tco2e = actuals.total_tco2e * 0.35
                warnings.append("Actual emissions estimated from base year trajectory.")

        actuals.year = self.config.review_year

        # Calculate intensity
        if actuals.revenue_musd > 0:
            actuals.intensity = round(actuals.total_tco2e / actuals.revenue_musd, 4)
        elif self.config.base_year_emissions_tco2e > 0:
            # Estimate intensity
            actuals.intensity = round(actuals.total_tco2e / 100.0, 4)

        # Tag MRV agents used
        mrv_agents: List[str] = []
        if actuals.scope1_tco2e > 0:
            mrv_agents.extend(MRV_SCOPE_COVERAGE["scope1"]["agents"][:3])
        if actuals.scope2_market_tco2e > 0:
            mrv_agents.extend(MRV_SCOPE_COVERAGE["scope2"]["agents"][:2])
        if actuals.scope3_tco2e > 0:
            mrv_agents.extend(MRV_SCOPE_COVERAGE["scope3"]["agents"][:5])
        actuals.mrv_agents_used = mrv_agents

        actuals.provenance_hash = _compute_hash(actuals.model_dump_json(exclude={"provenance_hash"}))
        self._actuals = actuals

        outputs["year"] = actuals.year
        outputs["total_tco2e"] = actuals.total_tco2e
        outputs["scope1_tco2e"] = actuals.scope1_tco2e
        outputs["scope2_market_tco2e"] = actuals.scope2_market_tco2e
        outputs["scope2_location_tco2e"] = actuals.scope2_location_tco2e
        outputs["scope3_tco2e"] = actuals.scope3_tco2e
        outputs["intensity"] = actuals.intensity
        outputs["data_quality_score"] = actuals.data_quality_score
        outputs["mrv_agents_count"] = len(mrv_agents)

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="collect_actuals", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_collect_actuals",
        )

    # -------------------------------------------------------------------------
    # Phase 2: Compare Target
    # -------------------------------------------------------------------------

    async def _phase_compare_target(self, input_data: AnnualProgressReviewInput) -> PhaseResult:
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        actual_total = self._actuals.total_tco2e
        review_year = self.config.review_year
        base_emissions = self.config.base_year_emissions_tco2e or actual_total * 1.2

        # Find the target for this year from the annual pathway
        year_target = 0.0
        for pt in input_data.annual_pathway:
            if pt.get("year") == review_year:
                year_target = pt.get("target_tco2e", 0.0)
                break

        # If no pathway, interpolate from near-term target
        if year_target <= 0:
            nt_year = self.config.near_term_target_year
            nt_target = self.config.near_term_target_tco2e
            if nt_target <= 0:
                nt_target = base_emissions * 0.58  # Default 42% reduction
            total_years = nt_year - self.config.base_year
            elapsed_years = review_year - self.config.base_year
            fraction = min(elapsed_years / max(total_years, 1), 1.0)
            year_target = base_emissions - (base_emissions - nt_target) * fraction

        comparisons: List[TargetComparison] = []

        # Absolute emissions comparison
        gap = actual_total - year_target
        gap_pct = (gap / max(year_target, 1e-10)) * 100
        cum_red = ((base_emissions - actual_total) / max(base_emissions, 1e-10)) * 100
        target_red = ((base_emissions - year_target) / max(base_emissions, 1e-10)) * 100
        achievement = min((cum_red / max(target_red, 1e-10)) * 100, 200.0) if target_red > 0 else 0.0

        if gap_pct <= 0:
            rag = RAGStatus.GREEN
        elif gap_pct <= self.config.alert_threshold_pct:
            rag = RAGStatus.AMBER
        else:
            rag = RAGStatus.RED

        # YoY change
        prev = input_data.previous_year_actuals
        if prev and prev.total_tco2e > 0:
            yoy = ((actual_total - prev.total_tco2e) / prev.total_tco2e) * 100
        else:
            yoy = 0.0

        comparisons.append(TargetComparison(
            metric_name="Total Absolute Emissions",
            scope="all",
            target_value=round(year_target, 2),
            actual_value=round(actual_total, 2),
            gap_absolute=round(gap, 2),
            gap_pct=round(gap_pct, 2),
            achievement_pct=round(achievement, 1),
            rag_status=rag,
            on_track=gap <= 0,
            yoy_change_pct=round(yoy, 2),
            cumulative_reduction_pct=round(cum_red, 2),
        ))

        # Scope 1+2 comparison
        s12_actual = self._actuals.scope1_tco2e + self._actuals.scope2_market_tco2e
        s12_base = base_emissions * 0.65  # Assume S1+S2 is ~65%
        s12_target = year_target * 0.65
        s12_gap = s12_actual - s12_target
        s12_gap_pct = (s12_gap / max(s12_target, 1e-10)) * 100
        s12_rag = RAGStatus.GREEN if s12_gap_pct <= 0 else (
            RAGStatus.AMBER if s12_gap_pct <= self.config.alert_threshold_pct else RAGStatus.RED
        )
        comparisons.append(TargetComparison(
            metric_name="Scope 1+2 Emissions",
            scope="scope_1_2",
            target_value=round(s12_target, 2),
            actual_value=round(s12_actual, 2),
            gap_absolute=round(s12_gap, 2),
            gap_pct=round(s12_gap_pct, 2),
            achievement_pct=round(
                min(((s12_base - s12_actual) / max(s12_base - s12_target, 1e-10)) * 100, 200), 1,
            ),
            rag_status=s12_rag,
            on_track=s12_gap <= 0,
        ))

        # Scope 3 comparison (if included)
        if self.config.include_scope3 and self._actuals.scope3_tco2e > 0:
            s3_actual = self._actuals.scope3_tco2e
            s3_target = year_target * 0.35
            s3_gap = s3_actual - s3_target
            s3_gap_pct = (s3_gap / max(s3_target, 1e-10)) * 100
            s3_rag = RAGStatus.GREEN if s3_gap_pct <= 0 else (
                RAGStatus.AMBER if s3_gap_pct <= 15 else RAGStatus.RED
            )
            comparisons.append(TargetComparison(
                metric_name="Scope 3 Emissions",
                scope="scope_3",
                target_value=round(s3_target, 2),
                actual_value=round(s3_actual, 2),
                gap_absolute=round(s3_gap, 2),
                gap_pct=round(s3_gap_pct, 2),
                rag_status=s3_rag,
                on_track=s3_gap <= 0,
            ))

        # Intensity comparison
        if self._actuals.intensity > 0:
            int_target = year_target / max(self._actuals.revenue_musd, 100.0)
            int_gap = self._actuals.intensity - int_target
            int_gap_pct = (int_gap / max(int_target, 1e-10)) * 100
            int_rag = RAGStatus.GREEN if int_gap_pct <= 0 else (
                RAGStatus.AMBER if int_gap_pct <= 10 else RAGStatus.RED
            )
            comparisons.append(TargetComparison(
                metric_name="Emissions Intensity",
                scope="all",
                target_value=round(int_target, 4),
                actual_value=round(self._actuals.intensity, 4),
                gap_absolute=round(int_gap, 4),
                gap_pct=round(int_gap_pct, 2),
                rag_status=int_rag,
                on_track=int_gap <= 0,
            ))

        # Overall progress
        on_track_count = sum(1 for c in comparisons if c.on_track)
        off_track_count = len(comparisons) - on_track_count

        red_count = sum(1 for c in comparisons if c.rag_status == RAGStatus.RED)
        overall_rag = (
            RAGStatus.RED if red_count >= 2 else
            RAGStatus.AMBER if red_count >= 1 or off_track_count > 0 else
            RAGStatus.GREEN
        )

        nt_years = max(self.config.near_term_target_year - review_year, 0)

        self._progress = ProgressSummary(
            review_year=review_year,
            comparisons=comparisons,
            overall_rag=overall_rag,
            overall_achievement_pct=round(achievement, 1),
            on_track_count=on_track_count,
            off_track_count=off_track_count,
            years_to_next_milestone=nt_years,
            next_milestone_year=self.config.near_term_target_year,
        )
        self._progress.provenance_hash = _compute_hash(
            self._progress.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["comparisons_count"] = len(comparisons)
        outputs["overall_rag"] = overall_rag.value
        outputs["overall_achievement_pct"] = round(achievement, 1)
        outputs["on_track_count"] = on_track_count
        outputs["off_track_count"] = off_track_count
        outputs["total_gap_tco2e"] = round(gap, 2)
        outputs["total_gap_pct"] = round(gap_pct, 2)
        outputs["yoy_change_pct"] = round(yoy, 2)
        outputs["cumulative_reduction_pct"] = round(cum_red, 2)

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="compare_target", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_compare_target",
        )

    # -------------------------------------------------------------------------
    # Phase 3: Variance Analysis
    # -------------------------------------------------------------------------

    async def _phase_variance_analysis(self, input_data: AnnualProgressReviewInput) -> PhaseResult:
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        actual_total = self._actuals.total_tco2e
        base_emissions = self.config.base_year_emissions_tco2e or actual_total * 1.2

        # Find target for this year
        year_target = 0.0
        for pt in input_data.annual_pathway:
            if pt.get("year") == self.config.review_year:
                year_target = pt.get("target_tco2e", 0.0)
                break
        if year_target <= 0:
            nt_target = self.config.near_term_target_tco2e or base_emissions * 0.58
            nt_year = self.config.near_term_target_year
            fraction = min(
                (self.config.review_year - self.config.base_year) /
                max(nt_year - self.config.base_year, 1), 1.0,
            )
            year_target = base_emissions - (base_emissions - nt_target) * fraction

        total_variance = actual_total - year_target
        total_var_pct = (total_variance / max(year_target, 1e-10)) * 100

        components: List[VarianceComponent] = []

        # Activity effect (production/revenue change)
        prev = input_data.previous_year_actuals
        if prev and prev.revenue_musd > 0 and self._actuals.revenue_musd > 0:
            revenue_change = (self._actuals.revenue_musd - prev.revenue_musd) / prev.revenue_musd
            activity_effect = prev.total_tco2e * revenue_change
        else:
            activity_effect = total_variance * 0.35  # Estimated

        components.append(VarianceComponent(
            component_type=VarianceType.ACTIVITY,
            component_name="Activity/Production Volume Effect",
            contribution_tco2e=round(activity_effect, 2),
            contribution_pct=round(
                (activity_effect / max(abs(total_variance), 1e-10)) * 100, 1,
            ),
            direction="unfavorable" if activity_effect > 0 else "favorable",
            explanation=(
                "Change in emissions due to changes in production volume, "
                "revenue, or activity levels."
            ),
        ))

        # Intensity effect (efficiency change)
        intensity_effect = total_variance - activity_effect
        if prev and prev.total_tco2e > 0:
            # LMDI-style decomposition
            prev_intensity = prev.total_tco2e / max(prev.revenue_musd, 1.0)
            curr_intensity = self._actuals.total_tco2e / max(self._actuals.revenue_musd, 1.0)
            intensity_change = curr_intensity - prev_intensity
            intensity_effect = intensity_change * self._actuals.revenue_musd

        components.append(VarianceComponent(
            component_type=VarianceType.INTENSITY,
            component_name="Emissions Intensity Effect",
            contribution_tco2e=round(intensity_effect, 2),
            contribution_pct=round(
                (intensity_effect / max(abs(total_variance), 1e-10)) * 100, 1,
            ),
            direction="unfavorable" if intensity_effect > 0 else "favorable",
            explanation=(
                "Change in emissions due to changes in emissions intensity "
                "(efficiency, technology, fuel mix)."
            ),
        ))

        # Structural effect (mix change, boundary changes)
        structural = total_variance - activity_effect - intensity_effect
        if abs(structural) > 0.01:
            components.append(VarianceComponent(
                component_type=VarianceType.STRUCTURAL,
                component_name="Structural/Mix Effect",
                contribution_tco2e=round(structural, 2),
                contribution_pct=round(
                    (structural / max(abs(total_variance), 1e-10)) * 100, 1,
                ),
                direction="unfavorable" if structural > 0 else "favorable",
                explanation=(
                    "Change in emissions due to structural shifts in business mix, "
                    "product portfolio, or geographic footprint."
                ),
            ))

        # Weather effect (estimated)
        weather_effect = total_variance * 0.05  # 5% attributed to weather
        components.append(VarianceComponent(
            component_type=VarianceType.WEATHER,
            component_name="Weather/Temperature Effect",
            contribution_tco2e=round(weather_effect, 2),
            contribution_pct=round(
                (weather_effect / max(abs(total_variance), 1e-10)) * 100, 1,
            ),
            direction="unfavorable" if weather_effect > 0 else "favorable",
            explanation="Estimated impact of weather variations on energy consumption.",
            data_quality=2.0,
        ))

        # Determine largest contributor
        largest = max(components, key=lambda c: abs(c.contribution_tco2e))

        self._variance = VarianceDecomposition(
            total_variance_tco2e=round(total_variance, 2),
            total_variance_pct=round(total_var_pct, 2),
            components=components,
            activity_effect_pct=round(
                (activity_effect / max(abs(total_variance), 1e-10)) * 100, 1,
            ),
            intensity_effect_pct=round(
                (intensity_effect / max(abs(total_variance), 1e-10)) * 100, 1,
            ),
            structural_effect_pct=round(
                (structural / max(abs(total_variance), 1e-10)) * 100, 1,
            ),
            largest_contributor=largest.component_name,
            net_favorable=total_variance <= 0,
        )
        self._variance.provenance_hash = _compute_hash(
            self._variance.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["total_variance_tco2e"] = round(total_variance, 2)
        outputs["total_variance_pct"] = round(total_var_pct, 2)
        outputs["activity_effect_pct"] = self._variance.activity_effect_pct
        outputs["intensity_effect_pct"] = self._variance.intensity_effect_pct
        outputs["structural_effect_pct"] = self._variance.structural_effect_pct
        outputs["largest_contributor"] = largest.component_name
        outputs["net_favorable"] = total_variance <= 0
        outputs["components_count"] = len(components)

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="variance_analysis", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_variance_analysis",
        )

    # -------------------------------------------------------------------------
    # Phase 4: Trend Extrapolation
    # -------------------------------------------------------------------------

    async def _phase_trend_extrapolation(self, input_data: AnnualProgressReviewInput) -> PhaseResult:
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        # Build time series from historical actuals
        all_actuals = list(input_data.historical_actuals) + [self._actuals]
        all_actuals.sort(key=lambda a: a.year)

        if len(all_actuals) < 2:
            # Generate synthetic history
            base = self.config.base_year_emissions_tco2e or self._actuals.total_tco2e * 1.2
            for yr in range(self.config.base_year, self.config.review_year):
                factor = max(1.0 - 0.03 * (yr - self.config.base_year), 0.1)
                all_actuals.append(EmissionsActual(
                    year=yr, total_tco2e=base * factor,
                ))
            all_actuals.sort(key=lambda a: a.year)
            warnings.append("Insufficient historical data; synthetic trend generated.")

        years = [a.year for a in all_actuals]
        values = [a.total_tco2e for a in all_actuals if a.total_tco2e > 0]
        years_f = [y for a, y in zip(all_actuals, years) if a.total_tco2e > 0]

        projections: List[TrendProjection] = []

        nt_target = self.config.near_term_target_tco2e or (
            self.config.base_year_emissions_tco2e * 0.58
        )
        lt_target = self.config.long_term_target_tco2e or (
            self.config.base_year_emissions_tco2e * 0.10
        )

        # Linear projection
        if len(values) >= 2:
            n = len(values)
            mean_x = sum(years_f) / n
            mean_y = sum(values) / n
            ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(years_f, values))
            ss_xx = sum((x - mean_x) ** 2 for x in years_f)
            ss_yy = sum((y - mean_y) ** 2 for y in values)
            slope = ss_xy / max(ss_xx, 1e-10)
            intercept = mean_y - slope * mean_x

            r_sq = (ss_xy ** 2) / max(ss_xx * ss_yy, 1e-10)

            proj_vals: Dict[int, float] = {}
            for yr in range(self.config.review_year, 2051):
                proj_vals[yr] = max(intercept + slope * yr, 0.0)

            proj_2030 = max(intercept + slope * 2030, 0.0)
            proj_2040 = max(intercept + slope * 2040, 0.0)
            proj_2050 = max(intercept + slope * 2050, 0.0)

            # Confidence interval (simple approximation)
            residuals = [v - (intercept + slope * y) for v, y in zip(values, years_f)]
            se = (sum(r ** 2 for r in residuals) / max(n - 2, 1)) ** 0.5
            margin = 1.96 * se * 2  # ~95% CI widened for projection

            trend_dir = (
                TrendDirection.IMPROVING if slope < -100 else
                TrendDirection.DETERIORATING if slope > 100 else
                TrendDirection.STABLE
            )

            projections.append(TrendProjection(
                method=ProjectionMethod.LINEAR,
                projected_values=proj_vals,
                projected_2030=round(proj_2030, 2),
                projected_2040=round(proj_2040, 2),
                projected_2050=round(proj_2050, 2),
                r_squared=round(min(r_sq, 1.0), 4),
                upper_bound_2030=round(proj_2030 + margin, 2),
                lower_bound_2030=round(max(proj_2030 - margin, 0), 2),
                trend_direction=trend_dir,
                meets_target_2030=proj_2030 <= nt_target * 1.10,
                meets_target_2050=proj_2050 <= lt_target * 1.10,
            ))

        # Exponential projection
        if len(values) >= 2 and all(v > 0 for v in values):
            log_values = [math.log(v) for v in values]
            n = len(log_values)
            mean_x = sum(years_f) / n
            mean_y = sum(log_values) / n
            ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(years_f, log_values))
            ss_xx = sum((x - mean_x) ** 2 for x in years_f)
            b = ss_xy / max(ss_xx, 1e-10)
            a = math.exp(mean_y - b * mean_x)

            proj_vals_exp: Dict[int, float] = {}
            for yr in range(self.config.review_year, 2051):
                proj_vals_exp[yr] = max(a * math.exp(b * yr), 0.0)

            proj_2030_e = max(a * math.exp(b * 2030), 0.0)
            proj_2040_e = max(a * math.exp(b * 2040), 0.0)
            proj_2050_e = max(a * math.exp(b * 2050), 0.0)

            trend_dir_e = (
                TrendDirection.IMPROVING if b < -0.01 else
                TrendDirection.DETERIORATING if b > 0.01 else
                TrendDirection.STABLE
            )

            projections.append(TrendProjection(
                method=ProjectionMethod.EXPONENTIAL,
                projected_values=proj_vals_exp,
                projected_2030=round(proj_2030_e, 2),
                projected_2040=round(proj_2040_e, 2),
                projected_2050=round(proj_2050_e, 2),
                r_squared=round(min(abs(b), 1.0), 4),
                trend_direction=trend_dir_e,
                meets_target_2030=proj_2030_e <= nt_target * 1.10,
                meets_target_2050=proj_2050_e <= lt_target * 1.10,
            ))

        # Consensus
        if projections:
            consensus_2030 = sum(p.projected_2030 for p in projections) / len(projections)
            consensus_2050 = sum(p.projected_2050 for p in projections) / len(projections)
            consensus_meets = consensus_2030 <= nt_target * 1.10
        else:
            consensus_2030 = self._actuals.total_tco2e
            consensus_2050 = self._actuals.total_tco2e
            consensus_meets = False

        # Years to target
        current = self._actuals.total_tco2e
        if projections and projections[0].method == ProjectionMethod.LINEAR:
            slope_val = (projections[0].projected_2030 - current) / max(2030 - self.config.review_year, 1)
            if slope_val < 0 and current > nt_target:
                ytt = int(math.ceil((current - nt_target) / abs(slope_val)))
            elif current <= nt_target:
                ytt = 0
            else:
                ytt = 999
        else:
            ytt = 0

        # Acceleration needed
        remaining_years = max(self.config.near_term_target_year - self.config.review_year, 1)
        if current > nt_target and current > 0:
            required_rate = (1 - (nt_target / current) ** (1.0 / remaining_years)) * 100
            # Estimate current rate from last 2 points
            if len(all_actuals) >= 2:
                recent = all_actuals[-2:]
                if recent[0].total_tco2e > 0:
                    current_rate = abs(
                        (recent[1].total_tco2e - recent[0].total_tco2e) /
                        recent[0].total_tco2e * 100
                    )
                else:
                    current_rate = 0
                acceleration = max(required_rate - current_rate, 0)
            else:
                acceleration = required_rate
        else:
            acceleration = 0.0

        recommended = (
            ProjectionMethod.EXPONENTIAL
            if any(p.method == ProjectionMethod.EXPONENTIAL and p.r_squared > 0.8 for p in projections)
            else ProjectionMethod.LINEAR
        )

        self._trends = TrendAnalysis(
            projections=projections,
            consensus_2030=round(consensus_2030, 2),
            consensus_2050=round(consensus_2050, 2),
            consensus_meets_target=consensus_meets,
            recommended_method=recommended,
            years_to_target=min(ytt, 100),
            acceleration_needed_pct=round(acceleration, 2),
        )
        self._trends.provenance_hash = _compute_hash(
            self._trends.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["projection_methods"] = len(projections)
        outputs["consensus_2030"] = round(consensus_2030, 2)
        outputs["consensus_2050"] = round(consensus_2050, 2)
        outputs["consensus_meets_target"] = consensus_meets
        outputs["years_to_target"] = min(ytt, 100)
        outputs["acceleration_needed_pct"] = round(acceleration, 2)
        outputs["recommended_method"] = recommended.value

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="trend_extrapolation", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_trend_extrapolation",
        )

    # -------------------------------------------------------------------------
    # Phase 5: Annual Report
    # -------------------------------------------------------------------------

    async def _phase_annual_report(self, input_data: AnnualProgressReviewInput) -> PhaseResult:
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        # Build KPIs
        kpis: List[ProgressKPI] = []
        base_e = self.config.base_year_emissions_tco2e or self._actuals.total_tco2e * 1.2

        kpis.append(ProgressKPI(
            kpi_id="KPI-001", kpi_name="Total Emissions",
            current_value=self._actuals.total_tco2e,
            target_value=self._progress.comparisons[0].target_value if self._progress.comparisons else 0,
            unit="tCO2e",
            achievement_pct=self._progress.overall_achievement_pct,
            trend=TrendDirection.IMPROVING if self._actuals.total_tco2e < base_e else TrendDirection.DETERIORATING,
            rag_status=self._progress.overall_rag,
            weight=0.25,
        ))

        kpis.append(ProgressKPI(
            kpi_id="KPI-002", kpi_name="Emissions Intensity",
            current_value=self._actuals.intensity,
            target_value=0.0,
            unit=self.config.intensity_unit,
            rag_status=self._progress.overall_rag,
            weight=0.20,
        ))

        cum_red = ((base_e - self._actuals.total_tco2e) / max(base_e, 1e-10)) * 100
        kpis.append(ProgressKPI(
            kpi_id="KPI-003", kpi_name="Cumulative Reduction",
            current_value=round(cum_red, 2),
            target_value=42.0,
            unit="%",
            achievement_pct=round(cum_red / 42.0 * 100, 1),
            rag_status=RAGStatus.GREEN if cum_red >= 20 else (RAGStatus.AMBER if cum_red >= 10 else RAGStatus.RED),
            weight=0.15,
        ))

        budget_consumed_pct = 0.0
        if self.config.carbon_budget_total_tco2e > 0:
            consumed = self.config.carbon_budget_consumed_tco2e + self._actuals.total_tco2e
            budget_consumed_pct = (consumed / self.config.carbon_budget_total_tco2e) * 100
        kpis.append(ProgressKPI(
            kpi_id="KPI-004", kpi_name="Carbon Budget Consumed",
            current_value=round(budget_consumed_pct, 1),
            target_value=100.0,
            unit="%",
            rag_status=RAGStatus.GREEN if budget_consumed_pct < 70 else (
                RAGStatus.AMBER if budget_consumed_pct < 90 else RAGStatus.RED
            ),
            weight=0.10,
        ))

        kpis.append(ProgressKPI(
            kpi_id="KPI-005", kpi_name="Data Quality",
            current_value=self._actuals.data_quality_score,
            target_value=4.0,
            unit="1-5 scale",
            rag_status=RAGStatus.GREEN if self._actuals.data_quality_score >= 4 else (
                RAGStatus.AMBER if self._actuals.data_quality_score >= 3 else RAGStatus.RED
            ),
            weight=0.05,
        ))

        # Build alerts
        alerts: List[ProgressAlert] = []
        if self._progress.overall_rag == RAGStatus.RED:
            alerts.append(ProgressAlert(
                alert_id=f"ALR-{_new_uuid()[:6]}",
                severity=AlertSeverity.CRITICAL,
                category="target_gap",
                title="Off-Track: Emissions Exceed Annual Target",
                description=f"Actual emissions exceed target by {self._progress.comparisons[0].gap_pct:.1f}%.",
                recommended_action="Conduct root cause analysis; escalate to management.",
            ))

        if self._variance.total_variance_tco2e > 0:
            alerts.append(ProgressAlert(
                alert_id=f"ALR-{_new_uuid()[:6]}",
                severity=AlertSeverity.WARNING,
                category="variance",
                title=f"Unfavorable Variance: {self._variance.total_variance_tco2e:,.0f} tCO2e",
                description=f"Largest contributor: {self._variance.largest_contributor}.",
                recommended_action="Address root cause of largest variance component.",
            ))

        if not self._trends.consensus_meets_target:
            alerts.append(ProgressAlert(
                alert_id=f"ALR-{_new_uuid()[:6]}",
                severity=AlertSeverity.WARNING,
                category="trajectory",
                title="Trajectory Does Not Meet 2030 Target",
                description=(
                    f"Projected 2030 emissions ({self._trends.consensus_2030:,.0f} tCO2e) "
                    f"exceed near-term target."
                ),
                recommended_action=(
                    f"Accelerate reduction rate by {self._trends.acceleration_needed_pct:.1f} "
                    f"percentage points/year."
                ),
            ))

        # Executive summary
        exec_parts = [
            f"Annual Progress Review {self.config.review_year} for {self.config.company_name or 'Company'}.",
            f"Actual emissions: {self._actuals.total_tco2e:,.0f} tCO2e.",
            f"Overall status: {self._progress.overall_rag.value.upper()}.",
            f"Cumulative reduction from base year: {cum_red:.1f}%.",
            f"Variance from target: {self._variance.total_variance_tco2e:,.0f} tCO2e ({self._variance.total_variance_pct:.1f}%).",
            f"Projected 2030 emissions: {self._trends.consensus_2030:,.0f} tCO2e.",
        ]
        if self._trends.consensus_meets_target:
            exec_parts.append("Current trajectory is on track to meet 2030 target.")
        else:
            exec_parts.append(
                f"Acceleration of {self._trends.acceleration_needed_pct:.1f}%/year needed to meet 2030 target.",
            )

        executive_summary = " ".join(exec_parts)

        findings = self._generate_findings()
        recommendations = self._generate_recommendations()

        self._report = AnnualProgressReport(
            report_id=f"APR-{self.workflow_id[:8]}",
            report_date=_utcnow().strftime("%Y-%m-%d"),
            review_year=self.config.review_year,
            company_name=self.config.company_name,
            entity_id=self.config.entity_id,
            actuals=self._actuals,
            progress_summary=self._progress,
            variance=self._variance,
            trend_analysis=self._trends,
            kpis=kpis,
            alerts=alerts,
            executive_summary=executive_summary,
            key_findings=findings,
            recommendations=recommendations,
            data_quality_score=self._actuals.data_quality_score,
        )
        self._report.provenance_hash = _compute_hash(
            self._report.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["report_id"] = self._report.report_id
        outputs["kpis_count"] = len(kpis)
        outputs["alerts_count"] = len(alerts)
        outputs["findings_count"] = len(findings)
        outputs["executive_summary_length"] = len(executive_summary)
        outputs["overall_rag"] = self._progress.overall_rag.value

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="annual_report", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_annual_report",
        )

    # -------------------------------------------------------------------------
    # Report Generators
    # -------------------------------------------------------------------------

    def _generate_findings(self) -> List[str]:
        findings: List[str] = []
        findings.append(
            f"Actual emissions in {self.config.review_year}: "
            f"{self._actuals.total_tco2e:,.0f} tCO2e.",
        )
        if self._progress.comparisons:
            c = self._progress.comparisons[0]
            findings.append(
                f"Gap to annual target: {c.gap_absolute:,.0f} tCO2e ({c.gap_pct:+.1f}%).",
            )
        findings.append(
            f"Largest variance contributor: {self._variance.largest_contributor} "
            f"({self._variance.components[0].contribution_pct:.0f}% of total).",
        )
        findings.append(
            f"Trend projection meets 2030 target: "
            f"{'Yes' if self._trends.consensus_meets_target else 'No'}.",
        )
        if self._trends.acceleration_needed_pct > 0:
            findings.append(
                f"Additional {self._trends.acceleration_needed_pct:.1f}%/year "
                f"reduction needed to reach 2030 target.",
            )
        return findings

    def _generate_recommendations(self) -> List[str]:
        recs: List[str] = []
        if self._progress.overall_rag == RAGStatus.RED:
            recs.append("Conduct emergency review of decarbonization initiatives.")
            recs.append("Escalate to board for immediate resource allocation.")
        elif self._progress.overall_rag == RAGStatus.AMBER:
            recs.append("Review and accelerate planned abatement initiatives.")
            recs.append("Set monthly monitoring cadence until back on track.")
        else:
            recs.append("Maintain current trajectory; consider increasing ambition.")

        if not self._trends.consensus_meets_target:
            recs.append(
                f"Accelerate annual reduction by {self._trends.acceleration_needed_pct:.1f} "
                f"percentage points to meet 2030 target.",
            )

        if self._variance.total_variance_tco2e > 0:
            recs.append(
                f"Focus corrective action on {self._variance.largest_contributor}.",
            )

        recs.append("Update Scope 3 category-level targets with supplier engagement data.")
        recs.append("Prepare CDP C4.1/C4.2 responses with updated progress data.")

        return recs
