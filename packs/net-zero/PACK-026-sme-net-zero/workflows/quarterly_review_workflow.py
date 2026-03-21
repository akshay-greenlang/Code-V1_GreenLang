# -*- coding: utf-8 -*-
"""
Quarterly Review Workflow
==============================

3-phase workflow for SME quarterly net-zero progress reviews within
PACK-026 SME Net Zero Pack.  Designed for 15-30 minute quarterly
check-ins to keep SMEs on track.

Phases:
    1. DataUpdate          -- Update energy/travel/procurement spend (5-10 min)
    2. ProgressCalculation -- Calculate progress vs. target pathway
    3. Reporting           -- Generate dashboard + board brief

Total time: 15-30 minutes.

Uses: sme_baseline_engine, cost_benefit_engine.

Zero-hallucination: all calculations deterministic.
SHA-256 provenance hashes for auditability.

Author: GreenLang Team
Version: 26.0.0
Pack: PACK-026 SME Net Zero Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION = "26.0.0"
_PACK_ID = "PACK-026"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


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
    GREEN = "green"
    AMBER = "amber"
    RED = "red"


class TrendDirection(str, Enum):
    DECREASING = "decreasing"
    FLAT = "flat"
    INCREASING = "increasing"


class Quarter(str, Enum):
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"


# =============================================================================
# EMISSION FACTOR CONSTANTS (subset for quarterly updates)
# =============================================================================

GRID_EF_KGCO2E_PER_KWH: Dict[str, float] = {
    "UK": 0.2070,
    "EU-AVG": 0.2556,
    "US-AVG": 0.3710,
    "GLOBAL": 0.4940,
}

GAS_EF_KGCO2E_PER_KWH = 0.18293

ENERGY_COST_PER_KWH_GBP: Dict[str, float] = {
    "electricity": 0.28,
    "gas": 0.08,
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, ge=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    completion_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    mobile_summary: str = Field(default="")


class QuarterlySpendUpdate(BaseModel):
    """Quarterly spend data for quick emissions update."""

    quarter: str = Field(default="Q1", description="Q1|Q2|Q3|Q4")
    year: int = Field(default=2025, ge=2020, le=2035)
    electricity_spend_gbp: float = Field(default=0.0, ge=0.0)
    electricity_kwh: float = Field(default=0.0, ge=0.0, description="If known from bills")
    gas_spend_gbp: float = Field(default=0.0, ge=0.0)
    gas_kwh: float = Field(default=0.0, ge=0.0, description="If known from bills")
    fuel_spend_gbp: float = Field(default=0.0, ge=0.0)
    travel_spend_gbp: float = Field(default=0.0, ge=0.0)
    waste_spend_gbp: float = Field(default=0.0, ge=0.0)
    procurement_spend_gbp: float = Field(default=0.0, ge=0.0)
    employee_count: int = Field(default=0, ge=0, description="Current headcount")
    notes: str = Field(default="", description="Any context for the quarter")


class QuarterlyEmissions(BaseModel):
    """Calculated quarterly emissions."""

    quarter: str = Field(default="Q1")
    year: int = Field(default=2025)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    total_tco2e: float = Field(default=0.0, ge=0.0)
    annualised_tco2e: float = Field(default=0.0, ge=0.0)


class TargetPathwayPoint(BaseModel):
    """Target pathway point for comparison."""

    year: int = Field(default=2025)
    target_tco2e: float = Field(default=0.0, ge=0.0)
    reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class ProgressMetrics(BaseModel):
    """Quarterly progress metrics vs target."""

    annualised_actual_tco2e: float = Field(default=0.0, ge=0.0)
    target_tco2e_this_year: float = Field(default=0.0, ge=0.0)
    gap_tco2e: float = Field(default=0.0, description="Positive = behind target")
    gap_pct: float = Field(default=0.0)
    rag_status: RAGStatus = Field(default=RAGStatus.RED)
    trend: TrendDirection = Field(default=TrendDirection.FLAT)
    on_track: bool = Field(default=False)
    cumulative_reduction_pct: float = Field(default=0.0)
    per_employee_tco2e: float = Field(default=0.0, ge=0.0)
    vs_previous_quarter_pct: float = Field(default=0.0)
    ytd_tco2e: float = Field(default=0.0, ge=0.0)


class QuickWinProgress(BaseModel):
    """Progress on quick win actions."""

    action_title: str = Field(default="")
    status: str = Field(default="not_started", description="not_started|in_progress|completed")
    savings_achieved_tco2e: float = Field(default=0.0, ge=0.0)
    savings_target_tco2e: float = Field(default=0.0, ge=0.0)
    completion_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class BoardBrief(BaseModel):
    """One-page board brief summary."""

    headline: str = Field(default="")
    rag_status: RAGStatus = Field(default=RAGStatus.RED)
    total_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    vs_target_pct: float = Field(default=0.0)
    key_achievements: List[str] = Field(default_factory=list)
    key_risks: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    cost_savings_ytd_gbp: float = Field(default=0.0)
    next_quarter_priorities: List[str] = Field(default_factory=list)


class QuarterlyReviewConfig(BaseModel):
    """Configuration for quarterly review workflow."""

    base_year: int = Field(default=2025, ge=2020, le=2035)
    base_year_tco2e: float = Field(default=0.0, ge=0.0)
    target_annual_reduction_pct: float = Field(default=10.0, ge=0.0, le=50.0)
    near_term_target_year: int = Field(default=2030)
    near_term_reduction_pct: float = Field(default=50.0)
    country: str = Field(default="UK")
    previous_quarters: List[QuarterlyEmissions] = Field(
        default_factory=list, description="Historical quarterly data"
    )
    active_quick_wins: List[QuickWinProgress] = Field(
        default_factory=list, description="Quick wins being tracked"
    )
    alert_threshold_pct: float = Field(default=5.0, ge=0.0, le=50.0)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class QuarterlyReviewInput(BaseModel):
    """Complete input for quarterly review workflow."""

    current_quarter: QuarterlySpendUpdate = Field(
        default_factory=QuarterlySpendUpdate,
    )
    config: QuarterlyReviewConfig = Field(
        default_factory=QuarterlyReviewConfig,
    )


class QuarterlyReviewResult(BaseModel):
    """Complete result from quarterly review workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="sme_quarterly_review")
    pack_id: str = Field(default="PACK-026")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    quarterly_emissions: QuarterlyEmissions = Field(default_factory=QuarterlyEmissions)
    progress: ProgressMetrics = Field(default_factory=ProgressMetrics)
    quick_win_progress: List[QuickWinProgress] = Field(default_factory=list)
    board_brief: BoardBrief = Field(default_factory=BoardBrief)
    trend_data: List[Dict[str, Any]] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class QuarterlyReviewWorkflow:
    """
    3-phase quarterly review workflow for SME net-zero progress tracking.

    Phase 1: Data Update (5-10 min)
        Update energy, travel, procurement spend for the quarter.

    Phase 2: Progress Calculation
        Calculate annualised emissions, compare to target pathway,
        determine RAG status and trend.

    Phase 3: Reporting
        Generate dashboard metrics and one-page board brief.

    Total time: 15-30 minutes.

    Example:
        >>> wf = QuarterlyReviewWorkflow()
        >>> inp = QuarterlyReviewInput(
        ...     current_quarter=QuarterlySpendUpdate(
        ...         quarter="Q1", year=2026,
        ...         electricity_spend_gbp=3000,
        ...         gas_spend_gbp=1000,
        ...     ),
        ...     config=QuarterlyReviewConfig(
        ...         base_year_tco2e=100.0,
        ...     ),
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.progress.rag_status in [RAGStatus.GREEN, RAGStatus.AMBER, RAGStatus.RED]
    """

    def __init__(self) -> None:
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._quarterly: QuarterlyEmissions = QuarterlyEmissions()
        self._progress: ProgressMetrics = ProgressMetrics()
        self._board_brief: BoardBrief = BoardBrief()
        self._trend_data: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(self, input_data: QuarterlyReviewInput) -> QuarterlyReviewResult:
        """Execute the 3-phase quarterly review workflow."""
        started_at = _utcnow()
        self.logger.info(
            "Starting quarterly review %s for %s %d",
            self.workflow_id,
            input_data.current_quarter.quarter,
            input_data.current_quarter.year,
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_data_update(input_data)
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise ValueError(f"DataUpdate failed: {phase1.errors}")

            phase2 = await self._phase_progress_calc(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_reporting(input_data)
            self._phase_results.append(phase3)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Quarterly review failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=99,
                status=PhaseStatus.FAILED, errors=[str(exc)],
                mobile_summary="Review failed.",
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        next_steps = self._generate_next_steps(input_data)

        result = QuarterlyReviewResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            quarterly_emissions=self._quarterly,
            progress=self._progress,
            quick_win_progress=input_data.config.active_quick_wins,
            board_brief=self._board_brief,
            trend_data=self._trend_data,
            next_steps=next_steps,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Data Update
    # -------------------------------------------------------------------------

    async def _phase_data_update(self, inp: QuarterlyReviewInput) -> PhaseResult:
        """Calculate quarterly emissions from spend/activity data."""
        started = _utcnow()
        warnings: List[str] = []
        errors: List[str] = []
        outputs: Dict[str, Any] = {}

        q = inp.current_quarter
        config = inp.config
        country = config.country or "UK"
        grid_ef = GRID_EF_KGCO2E_PER_KWH.get(country, GRID_EF_KGCO2E_PER_KWH["GLOBAL"])

        # Scope 1: Gas
        if q.gas_kwh > 0:
            gas_kwh = q.gas_kwh
        elif q.gas_spend_gbp > 0:
            gas_kwh = q.gas_spend_gbp / ENERGY_COST_PER_KWH_GBP["gas"]
        else:
            gas_kwh = 0.0
        scope1_gas = (gas_kwh * GAS_EF_KGCO2E_PER_KWH) / 1000.0

        # Scope 1: Fuel
        scope1_fuel = 0.0
        if q.fuel_spend_gbp > 0:
            fuel_kwh = q.fuel_spend_gbp / 0.10  # Approx GBP/kWh for diesel
            scope1_fuel = (fuel_kwh * 0.25301) / 1000.0

        scope1 = scope1_gas + scope1_fuel

        # Scope 2: Electricity
        if q.electricity_kwh > 0:
            elec_kwh = q.electricity_kwh
        elif q.electricity_spend_gbp > 0:
            elec_kwh = q.electricity_spend_gbp / ENERGY_COST_PER_KWH_GBP["electricity"]
        else:
            elec_kwh = 0.0
        scope2 = (elec_kwh * grid_ef) / 1000.0

        # Scope 3: Travel + Waste + Procurement + Commuting
        scope3_travel = q.travel_spend_gbp * 0.00026
        scope3_waste = q.waste_spend_gbp * 0.00058
        scope3_procurement = q.procurement_spend_gbp * 0.00042
        employee_count = q.employee_count or 1
        scope3_commuting = (employee_count * 300.0 / 4.0) / 1000.0  # Quarterly commuting

        scope3 = scope3_travel + scope3_waste + scope3_procurement + scope3_commuting
        total = scope1 + scope2 + scope3
        annualised = total * 4.0

        self._quarterly = QuarterlyEmissions(
            quarter=q.quarter,
            year=q.year,
            scope1_tco2e=round(scope1, 4),
            scope2_tco2e=round(scope2, 4),
            scope3_tco2e=round(scope3, 4),
            total_tco2e=round(total, 4),
            annualised_tco2e=round(annualised, 4),
        )

        outputs["quarter"] = q.quarter
        outputs["year"] = q.year
        outputs["scope1_tco2e"] = round(scope1, 4)
        outputs["scope2_tco2e"] = round(scope2, 4)
        outputs["scope3_tco2e"] = round(scope3, 4)
        outputs["total_tco2e"] = round(total, 4)
        outputs["annualised_tco2e"] = round(annualised, 4)

        if total <= 0:
            warnings.append("Quarterly emissions are zero; check spend data")

        total_spend = (
            q.electricity_spend_gbp + q.gas_spend_gbp + q.fuel_spend_gbp
            + q.travel_spend_gbp + q.waste_spend_gbp + q.procurement_spend_gbp
        )
        if total_spend <= 0:
            errors.append("No spend data provided for the quarter")

        elapsed = (_utcnow() - started).total_seconds()
        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        return PhaseResult(
            phase_name="data_update", phase_number=1,
            status=status, duration_seconds=round(elapsed, 4),
            completion_pct=100.0 if not errors else 0.0,
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"{q.quarter} {q.year}: {total:.1f} tCO2e ({annualised:.1f} annualised)",
        )

    # -------------------------------------------------------------------------
    # Phase 2: Progress Calculation
    # -------------------------------------------------------------------------

    async def _phase_progress_calc(self, inp: QuarterlyReviewInput) -> PhaseResult:
        """Calculate progress vs. target pathway."""
        started = _utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        config = inp.config
        base_total = config.base_year_tco2e or 1.0
        current_annualised = self._quarterly.annualised_tco2e

        # Target for this year
        years_from_base = inp.current_quarter.year - config.base_year
        if years_from_base <= 0:
            target_this_year = base_total
        else:
            nt_years = config.near_term_target_year - config.base_year
            if nt_years > 0:
                annual_reduction = config.near_term_reduction_pct / nt_years
                reduction_pct = min(annual_reduction * years_from_base, 100.0)
            else:
                reduction_pct = 0.0
            target_this_year = base_total * (1.0 - reduction_pct / 100.0)

        # Gap
        gap = current_annualised - target_this_year
        gap_pct = (gap / target_this_year * 100) if target_this_year > 0 else 0

        # RAG status
        threshold = config.alert_threshold_pct
        if gap <= 0:
            rag = RAGStatus.GREEN
        elif gap_pct <= threshold:
            rag = RAGStatus.AMBER
        else:
            rag = RAGStatus.RED

        on_track = gap <= 0

        # Cumulative reduction
        cumulative_reduction = ((base_total - current_annualised) / base_total * 100) if base_total > 0 else 0

        # Per employee
        employee_count = inp.current_quarter.employee_count or 1
        per_employee = current_annualised / employee_count

        # YTD (sum quarters in current year)
        current_year = inp.current_quarter.year
        ytd_quarters = [
            q for q in config.previous_quarters
            if q.year == current_year
        ]
        ytd = sum(q.total_tco2e for q in ytd_quarters) + self._quarterly.total_tco2e

        # Trend (compare to previous quarter)
        trend = TrendDirection.FLAT
        vs_prev = 0.0
        if config.previous_quarters:
            prev = config.previous_quarters[-1]
            if prev.total_tco2e > 0:
                vs_prev = ((self._quarterly.total_tco2e - prev.total_tco2e) / prev.total_tco2e) * 100
                if vs_prev < -2:
                    trend = TrendDirection.DECREASING
                elif vs_prev > 2:
                    trend = TrendDirection.INCREASING

        self._progress = ProgressMetrics(
            annualised_actual_tco2e=round(current_annualised, 4),
            target_tco2e_this_year=round(target_this_year, 4),
            gap_tco2e=round(gap, 4),
            gap_pct=round(gap_pct, 2),
            rag_status=rag,
            trend=trend,
            on_track=on_track,
            cumulative_reduction_pct=round(cumulative_reduction, 2),
            per_employee_tco2e=round(per_employee, 4),
            vs_previous_quarter_pct=round(vs_prev, 2),
            ytd_tco2e=round(ytd, 4),
        )

        # Build trend data
        self._trend_data = []
        for q in config.previous_quarters:
            self._trend_data.append({
                "quarter": q.quarter,
                "year": q.year,
                "actual_tco2e": q.total_tco2e,
                "annualised_tco2e": q.annualised_tco2e,
            })
        self._trend_data.append({
            "quarter": self._quarterly.quarter,
            "year": self._quarterly.year,
            "actual_tco2e": self._quarterly.total_tco2e,
            "annualised_tco2e": self._quarterly.annualised_tco2e,
            "target_tco2e": round(target_this_year, 4),
        })

        outputs["rag_status"] = rag.value
        outputs["annualised_tco2e"] = current_annualised
        outputs["target_tco2e"] = round(target_this_year, 4)
        outputs["gap_tco2e"] = round(gap, 4)
        outputs["on_track"] = on_track
        outputs["trend"] = trend.value
        outputs["cumulative_reduction_pct"] = round(cumulative_reduction, 2)
        outputs["ytd_tco2e"] = round(ytd, 4)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Progress: RAG=%s, gap=%.1f tCO2e, %s, on_track=%s",
            rag.value, gap, trend.value, on_track,
        )
        return PhaseResult(
            phase_name="progress_calculation", phase_number=2,
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            completion_pct=100.0,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"RAG: {rag.value.upper()} | {cumulative_reduction:.1f}% reduced | {'On track' if on_track else 'Behind target'}",
        )

    # -------------------------------------------------------------------------
    # Phase 3: Reporting
    # -------------------------------------------------------------------------

    async def _phase_reporting(self, inp: QuarterlyReviewInput) -> PhaseResult:
        """Generate dashboard metrics and board brief."""
        started = _utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        progress = self._progress

        # Build headline
        if progress.rag_status == RAGStatus.GREEN:
            headline = (
                f"On track: {progress.cumulative_reduction_pct:.1f}% reduction achieved. "
                f"Annualised emissions {progress.annualised_actual_tco2e:.1f} tCO2e."
            )
        elif progress.rag_status == RAGStatus.AMBER:
            headline = (
                f"Slightly behind target: {progress.gap_tco2e:.1f} tCO2e gap. "
                f"Current reduction {progress.cumulative_reduction_pct:.1f}%."
            )
        else:
            headline = (
                f"Behind target: {progress.gap_tco2e:.1f} tCO2e gap ({progress.gap_pct:.1f}%). "
                f"Corrective action required."
            )

        # Key achievements
        achievements: List[str] = []
        if progress.cumulative_reduction_pct > 0:
            achievements.append(
                f"Achieved {progress.cumulative_reduction_pct:.1f}% cumulative emission reduction"
            )
        if progress.trend == TrendDirection.DECREASING:
            achievements.append("Emissions trending downward this quarter")
        completed_wins = [
            qw for qw in inp.config.active_quick_wins
            if qw.status == "completed"
        ]
        if completed_wins:
            achievements.append(
                f"Completed {len(completed_wins)} quick win action(s)"
            )
        if not achievements:
            achievements.append("Baseline established and tracking initiated")

        # Key risks
        risks: List[str] = []
        if progress.rag_status == RAGStatus.RED:
            risks.append(
                f"Emissions {progress.gap_pct:.1f}% above target pathway"
            )
        if progress.trend == TrendDirection.INCREASING:
            risks.append("Emissions are increasing quarter-over-quarter")
        if progress.gap_tco2e > 0:
            risks.append(
                f"Annual reduction rate needs to increase to close {progress.gap_tco2e:.1f} tCO2e gap"
            )
        if not risks:
            risks.append("No significant risks identified this quarter")

        # Recommended actions
        actions: List[str] = []
        not_started_wins = [
            qw for qw in inp.config.active_quick_wins
            if qw.status == "not_started"
        ]
        if not_started_wins:
            actions.append(
                f"Start {len(not_started_wins)} pending quick win action(s): "
                + not_started_wins[0].action_title
            )
        if progress.rag_status in {RAGStatus.RED, RAGStatus.AMBER}:
            actions.append("Review and accelerate emission reduction measures")
        actions.append("Continue quarterly data collection and monitoring")

        # Priorities
        priorities: List[str] = [
            "Collect next quarter's energy and travel data",
            "Review progress on active reduction actions",
        ]
        if progress.rag_status == RAGStatus.RED:
            priorities.insert(0, "URGENT: Identify additional emission reduction opportunities")

        # Estimate cost savings (rough)
        cost_savings = max(
            (inp.config.base_year_tco2e - progress.annualised_actual_tco2e) * 50, 0
        )

        self._board_brief = BoardBrief(
            headline=headline,
            rag_status=progress.rag_status,
            total_emissions_tco2e=progress.annualised_actual_tco2e,
            vs_target_pct=round(progress.gap_pct, 2),
            key_achievements=achievements,
            key_risks=risks,
            recommended_actions=actions,
            cost_savings_ytd_gbp=round(cost_savings, 2),
            next_quarter_priorities=priorities,
        )

        outputs["headline"] = headline
        outputs["rag_status"] = progress.rag_status.value
        outputs["achievements"] = len(achievements)
        outputs["risks"] = len(risks)
        outputs["cost_savings_gbp"] = round(cost_savings, 2)

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="reporting", phase_number=3,
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            completion_pct=100.0,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=headline[:80],
        )

    # -------------------------------------------------------------------------
    # Next Steps
    # -------------------------------------------------------------------------

    def _generate_next_steps(self, inp: QuarterlyReviewInput) -> List[str]:
        steps: List[str] = []
        progress = self._progress

        steps.append("Share the board brief with your management team.")

        if progress.rag_status == RAGStatus.RED:
            steps.append(
                "URGENT: Schedule a meeting to discuss corrective actions."
            )

        # Next quarter reminder
        quarter_map = {"Q1": "Q2", "Q2": "Q3", "Q3": "Q4", "Q4": "Q1"}
        next_q = quarter_map.get(inp.current_quarter.quarter, "Q1")
        next_year = inp.current_quarter.year + (1 if next_q == "Q1" and inp.current_quarter.quarter == "Q4" else 0)
        steps.append(f"Schedule {next_q} {next_year} review.")

        not_started = [
            qw for qw in inp.config.active_quick_wins
            if qw.status == "not_started"
        ]
        if not_started:
            steps.append(f"Begin implementation of {len(not_started)} pending quick wins.")

        steps.append("Update your emissions data in the SME dashboard.")

        return steps
