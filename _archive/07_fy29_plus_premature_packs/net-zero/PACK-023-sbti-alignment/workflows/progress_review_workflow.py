# -*- coding: utf-8 -*-
"""
Progress Review Workflow
============================

5-phase workflow for annual SBTi progress review within PACK-023
SBTi Alignment Pack.  The workflow collects current-year emissions
data, calculates progress against validated targets with RAG status,
performs variance decomposition analysis, evaluates whether a base
year recalculation is triggered, and generates a comprehensive
annual progress report.

Phases:
    1. DataUpdate         -- Collect and validate current-year emissions data
    2. ProgressCalc       -- Calculate progress against targets with RAG status
    3. VarianceAnalysis   -- Decompose variance by driver (structural, activity, intensity)
    4. Recalc             -- Evaluate recalculation triggers and apply if needed
    5. Report             -- Generate annual progress report with corrective actions

Regulatory references:
    - SBTi Corporate Manual V5.3 (2024) - Progress reporting (C22)
    - SBTi Corporate Net-Zero Standard V1.3 (2024) - Annual tracking
    - GHG Protocol Corporate Standard Chapter 5 - Tracking Emissions
    - CDP Climate Change Questionnaire C4 (2024) - Progress reporting
    - CSRD ESRS E1-4 (2023) - GHG reduction progress

RAG Thresholds:
    GREEN  - actual reduction within 5% of required pathway (on track)
    AMBER  - deviation between 5% and 15% of required pathway
    RED    - deviation exceeds 15% of required pathway (critical)

Zero-hallucination: all progress calculations use deterministic
linear pathway comparison.  RAG classification uses fixed numeric
thresholds from SBTi guidance.  No LLM calls in the numeric
computation path.

Author: GreenLang Team
Version: 23.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "23.0.0"

# =============================================================================
# HELPERS
# =============================================================================

def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex

def _compute_hash(data: str) -> str:
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

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

class RAGStatus(str, Enum):
    """RAG (Red/Amber/Green) progress status."""

    GREEN = "green"
    AMBER = "amber"
    RED = "red"
    NOT_ASSESSED = "not_assessed"

class TrackingStatus(str, Enum):
    """Detailed tracking status classification."""

    ON_TRACK = "on_track"
    MINOR_DEVIATION = "minor_deviation"
    MAJOR_DEVIATION = "major_deviation"
    CRITICAL = "critical"
    NOT_STARTED = "not_started"
    AHEAD_OF_TARGET = "ahead_of_target"

class VarianceDriver(str, Enum):
    """Driver type for variance decomposition."""

    STRUCTURAL = "structural"
    ACTIVITY = "activity"
    INTENSITY = "intensity"
    METHODOLOGY = "methodology"
    EXTERNAL = "external"
    UNEXPLAINED = "unexplained"

class RecalcTriggerType(str, Enum):
    """Recalculation trigger type."""

    ACQUISITION = "acquisition"
    DIVESTITURE = "divestiture"
    MERGER = "merger"
    METHODOLOGY_CHANGE = "methodology_change"
    STRUCTURAL_CHANGE = "structural_change"
    ORGANIC_GROWTH = "organic_growth"
    EMISSION_FACTOR_UPDATE = "emission_factor_update"
    BOUNDARY_CHANGE = "boundary_change"
    NONE = "none"

class CorrectiveActionPriority(str, Enum):
    """Priority of corrective actions."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# =============================================================================
# REFERENCE DATA
# =============================================================================

# RAG threshold constants (percentage deviation from required pathway)
RAG_GREEN_THRESHOLD = 5.0     # <= 5% deviation
RAG_AMBER_THRESHOLD = 15.0    # <= 15% deviation
# > 15% deviation -> RED

# Critical overshoot: projected trajectory exceeds target by > 25%
CRITICAL_OVERSHOOT_THRESHOLD = 25.0

# Base year recalculation significance threshold (SBTi V5.3)
SIGNIFICANCE_THRESHOLD_PCT = 5.0

# Carbon budget warning: if > 60% of budget consumed before halfway
BUDGET_WARNING_PCT = 60.0
BUDGET_HALFWAY_FACTOR = 0.5

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class AnnualEmissions(BaseModel):
    """Emissions data for a single reporting year."""

    year: int = Field(..., ge=2015, le=2060)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_location_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_market_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_total_tco2e: float = Field(default=0.0, ge=0.0)
    flag_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    revenue_usd: float = Field(default=0.0, ge=0.0)
    employee_count: int = Field(default=0, ge=0)
    activity_value: float = Field(default=0.0, ge=0.0)
    is_verified: bool = Field(default=False)
    notes: str = Field(default="")

    @property
    def scope12_total(self) -> float:
        return self.scope1_tco2e + self.scope2_location_tco2e

    @property
    def total_emissions(self) -> float:
        return self.scope12_total + self.scope3_total_tco2e

class TargetReference(BaseModel):
    """Reference target for progress tracking."""

    target_id: str = Field(default="")
    target_name: str = Field(default="")
    scopes: List[str] = Field(default_factory=list,
                               description="e.g. ['scope1', 'scope2'] or ['scope3']")
    base_year: int = Field(default=2022, ge=2015)
    target_year: int = Field(default=2030, ge=2025)
    base_year_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    target_reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    target_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    annual_reduction_rate: float = Field(default=0.0, ge=0.0,
                                          description="Required annual rate (%)")
    is_intensity: bool = Field(default=False)
    intensity_metric: str = Field(default="")

class ProgressAssessment(BaseModel):
    """Progress assessment for a single target."""

    target_id: str = Field(default="")
    target_name: str = Field(default="")
    rag_status: RAGStatus = Field(default=RAGStatus.NOT_ASSESSED)
    tracking_status: TrackingStatus = Field(default=TrackingStatus.NOT_STARTED)
    required_emissions_tco2e: float = Field(default=0.0,
                                             description="Required emissions at current year on pathway")
    actual_emissions_tco2e: float = Field(default=0.0)
    gap_tco2e: float = Field(default=0.0, description="Actual - Required (positive = behind)")
    gap_pct: float = Field(default=0.0)
    actual_reduction_pct: float = Field(default=0.0,
                                         description="Actual reduction from base year")
    required_reduction_pct: float = Field(default=0.0,
                                           description="Required reduction at current year")
    actual_annual_rate: float = Field(default=0.0, description="Observed annual rate (%)")
    required_annual_rate: float = Field(default=0.0, description="Required annual rate (%)")
    years_remaining: int = Field(default=0)
    projected_target_year_emissions: float = Field(default=0.0,
                                                    description="Projected emissions at target year")
    projected_meets_target: bool = Field(default=False)
    carbon_budget_total_tco2e: float = Field(default=0.0)
    carbon_budget_used_tco2e: float = Field(default=0.0)
    carbon_budget_remaining_tco2e: float = Field(default=0.0)
    carbon_budget_used_pct: float = Field(default=0.0)

class VarianceComponent(BaseModel):
    """A single variance decomposition component."""

    driver: VarianceDriver = Field(...)
    description: str = Field(default="")
    impact_tco2e: float = Field(default=0.0)
    impact_pct: float = Field(default=0.0)
    is_controllable: bool = Field(default=True)

class VarianceResult(BaseModel):
    """Complete variance analysis for a target."""

    target_id: str = Field(default="")
    total_variance_tco2e: float = Field(default=0.0)
    total_variance_pct: float = Field(default=0.0)
    components: List[VarianceComponent] = Field(default_factory=list)
    primary_driver: VarianceDriver = Field(default=VarianceDriver.UNEXPLAINED)
    trend_direction: str = Field(default="flat",
                                  description="improving, deteriorating, or flat")
    year_over_year_change_pct: float = Field(default=0.0)

class RecalcTrigger(BaseModel):
    """A recalculation trigger assessment."""

    trigger_type: RecalcTriggerType = Field(default=RecalcTriggerType.NONE)
    description: str = Field(default="")
    impact_tco2e: float = Field(default=0.0)
    impact_pct: float = Field(default=0.0)
    exceeds_threshold: bool = Field(default=False)
    significance_threshold_pct: float = Field(default=SIGNIFICANCE_THRESHOLD_PCT)

class RecalcResult(BaseModel):
    """Recalculation evaluation result."""

    recalculation_required: bool = Field(default=False)
    triggers: List[RecalcTrigger] = Field(default_factory=list)
    original_base_emissions: float = Field(default=0.0)
    adjusted_base_emissions: float = Field(default=0.0)
    adjustment_tco2e: float = Field(default=0.0)
    adjustment_pct: float = Field(default=0.0)
    targets_adjusted: int = Field(default=0)
    notes: List[str] = Field(default_factory=list)

class CorrectiveAction(BaseModel):
    """A recommended corrective action."""

    action_id: str = Field(default="")
    target_id: str = Field(default="")
    priority: CorrectiveActionPriority = Field(default=CorrectiveActionPriority.MEDIUM)
    description: str = Field(default="")
    expected_impact_tco2e: float = Field(default=0.0)
    timeline_months: int = Field(default=6)
    category: str = Field(default="")

class ProgressReport(BaseModel):
    """Complete annual progress report."""

    report_year: int = Field(...)
    overall_rag: RAGStatus = Field(default=RAGStatus.NOT_ASSESSED)
    overall_tracking: TrackingStatus = Field(default=TrackingStatus.NOT_STARTED)
    targets_on_track: int = Field(default=0)
    targets_behind: int = Field(default=0)
    targets_ahead: int = Field(default=0)
    total_reduction_achieved_pct: float = Field(default=0.0)
    key_findings: List[str] = Field(default_factory=list)
    corrective_actions: List[CorrectiveAction] = Field(default_factory=list)
    next_review_date: str = Field(default="")

class ProgressReviewConfig(BaseModel):
    """Configuration for the progress review workflow."""

    # Emissions history
    base_year_emissions: AnnualEmissions = Field(...)
    current_year_emissions: AnnualEmissions = Field(...)
    prior_year_emissions: Optional[AnnualEmissions] = Field(None)
    historical_emissions: List[AnnualEmissions] = Field(default_factory=list)

    # Targets to track
    targets: List[TargetReference] = Field(default_factory=list)

    # Recalculation triggers
    recalc_triggers: List[RecalcTrigger] = Field(default_factory=list)

    # Activity data for variance decomposition
    base_year_activity: float = Field(default=0.0, ge=0.0)
    current_year_activity: float = Field(default=0.0, ge=0.0)
    base_year_revenue: float = Field(default=0.0, ge=0.0)
    current_year_revenue: float = Field(default=0.0, ge=0.0)

    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

class ProgressReviewResult(BaseModel):
    """Complete result from the progress review workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="progress_review")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    progress_assessments: List[ProgressAssessment] = Field(default_factory=list)
    variance_results: List[VarianceResult] = Field(default_factory=list)
    recalc_result: Optional[RecalcResult] = Field(None)
    report: Optional[ProgressReport] = Field(None)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class ProgressReviewWorkflow:
    """
    5-phase annual progress review workflow for SBTi targets.

    Collects current-year emissions data, calculates progress against
    validated targets with RAG status, performs variance decomposition
    analysis, evaluates recalculation triggers, and generates a
    comprehensive annual progress report with corrective actions.

    Zero-hallucination: all progress calculations use deterministic
    linear pathway comparison.  RAG classification uses fixed numeric
    thresholds.  No LLM calls in the numeric computation path.

    Attributes:
        workflow_id: Unique execution identifier.

    Example:
        >>> wf = ProgressReviewWorkflow()
        >>> config = ProgressReviewConfig(
        ...     base_year_emissions=AnnualEmissions(year=2022, scope1_tco2e=5000),
        ...     current_year_emissions=AnnualEmissions(year=2024, scope1_tco2e=4500),
        ...     targets=[TargetReference(target_id="NT-S12",
        ...                              base_year=2022, target_year=2030,
        ...                              base_year_emissions_tco2e=8000,
        ...                              target_reduction_pct=42.0)],
        ... )
        >>> result = await wf.execute(config)
        >>> assert result.report is not None
    """

    def __init__(self) -> None:
        """Initialise ProgressReviewWorkflow."""
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._progress: List[ProgressAssessment] = []
        self._variances: List[VarianceResult] = []
        self._recalc: Optional[RecalcResult] = None
        self._report: Optional[ProgressReport] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, config: ProgressReviewConfig) -> ProgressReviewResult:
        """
        Execute the 5-phase progress review workflow.

        Args:
            config: Progress review configuration with emissions
                history, targets, and recalculation triggers.

        Returns:
            ProgressReviewResult with assessments, variance analysis,
            recalculation evaluation, and annual report.
        """
        started_at = utcnow()
        self.logger.info(
            "Starting progress review workflow %s, base_year=%d, current_year=%d",
            self.workflow_id,
            config.base_year_emissions.year,
            config.current_year_emissions.year,
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            # Phase 1: Data Update
            phase1 = await self._phase_data_update(config)
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise ValueError("Data update phase failed; cannot proceed")

            # Phase 2: Progress Calculation
            phase2 = await self._phase_progress_calc(config)
            self._phase_results.append(phase2)

            # Phase 3: Variance Analysis
            phase3 = await self._phase_variance_analysis(config)
            self._phase_results.append(phase3)

            # Phase 4: Recalculation Evaluation
            phase4 = await self._phase_recalc(config)
            self._phase_results.append(phase4)

            # Phase 5: Report Generation
            phase5 = await self._phase_report(config)
            self._phase_results.append(phase5)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Progress review workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        result = ProgressReviewResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            progress_assessments=self._progress,
            variance_results=self._variances,
            recalc_result=self._recalc,
            report=self._report,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        self.logger.info(
            "Progress review workflow %s completed in %.2fs, targets=%d",
            self.workflow_id, elapsed, len(self._progress),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Data Update
    # -------------------------------------------------------------------------

    async def _phase_data_update(self, config: ProgressReviewConfig) -> PhaseResult:
        """Collect and validate current-year emissions data."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        base = config.base_year_emissions
        current = config.current_year_emissions

        # Validate year ordering
        if current.year <= base.year:
            errors.append(
                f"Current year ({current.year}) must be after base year ({base.year})"
            )

        # Validate emissions data
        if base.total_emissions <= 0:
            errors.append("Base year total emissions must be > 0")

        if current.total_emissions <= 0:
            warnings.append(
                "Current year total emissions are zero; data may be incomplete"
            )

        # Verification check
        if not current.is_verified:
            warnings.append(
                "Current year emissions are not verified; consider third-party verification"
            )

        # Year gap check
        years_since_base = current.year - base.year
        if years_since_base > 5:
            warnings.append(
                f"Current year is {years_since_base} years from base year; "
                "ensure base year remains valid for SBTi submission"
            )

        # Check for missing targets
        if not config.targets:
            warnings.append("No targets provided for progress tracking")

        # Summary
        outputs["base_year"] = base.year
        outputs["current_year"] = current.year
        outputs["years_elapsed"] = years_since_base
        outputs["base_scope12_tco2e"] = round(base.scope12_total, 2)
        outputs["base_scope3_tco2e"] = round(base.scope3_total_tco2e, 2)
        outputs["base_total_tco2e"] = round(base.total_emissions, 2)
        outputs["current_scope12_tco2e"] = round(current.scope12_total, 2)
        outputs["current_scope3_tco2e"] = round(current.scope3_total_tco2e, 2)
        outputs["current_total_tco2e"] = round(current.total_emissions, 2)
        outputs["targets_count"] = len(config.targets)
        outputs["historical_years"] = len(config.historical_emissions)
        outputs["current_verified"] = current.is_verified

        # Preliminary change
        if base.total_emissions > 0:
            total_change_pct = (
                (current.total_emissions - base.total_emissions)
                / base.total_emissions * 100.0
            )
            outputs["total_change_pct"] = round(total_change_pct, 2)

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Data update: base=%d (%.2f tCO2e), current=%d (%.2f tCO2e)",
            base.year, base.total_emissions, current.year, current.total_emissions,
        )
        return PhaseResult(
            phase_name="data_update",
            status=PhaseStatus.FAILED if errors else PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Progress Calculation
    # -------------------------------------------------------------------------

    async def _phase_progress_calc(self, config: ProgressReviewConfig) -> PhaseResult:
        """Calculate progress against targets with RAG status."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._progress = []

        base = config.base_year_emissions
        current = config.current_year_emissions

        for target in config.targets:
            assessment = self._assess_target(base, current, target, config)
            self._progress.append(assessment)

        # Summary statistics
        on_track = sum(1 for p in self._progress if p.tracking_status in (
            TrackingStatus.ON_TRACK, TrackingStatus.AHEAD_OF_TARGET))
        behind = sum(1 for p in self._progress if p.tracking_status in (
            TrackingStatus.MINOR_DEVIATION, TrackingStatus.MAJOR_DEVIATION,
            TrackingStatus.CRITICAL))
        ahead = sum(1 for p in self._progress if p.tracking_status == TrackingStatus.AHEAD_OF_TARGET)

        # Critical targets warning
        critical = [p for p in self._progress if p.tracking_status == TrackingStatus.CRITICAL]
        if critical:
            warnings.append(
                f"{len(critical)} target(s) in CRITICAL status: "
                f"{', '.join(p.target_id for p in critical)}"
            )

        outputs["targets_assessed"] = len(self._progress)
        outputs["on_track"] = on_track
        outputs["behind"] = behind
        outputs["ahead"] = ahead
        outputs["critical"] = len(critical)
        outputs["green_count"] = sum(1 for p in self._progress if p.rag_status == RAGStatus.GREEN)
        outputs["amber_count"] = sum(1 for p in self._progress if p.rag_status == RAGStatus.AMBER)
        outputs["red_count"] = sum(1 for p in self._progress if p.rag_status == RAGStatus.RED)

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Progress calc: %d targets, on_track=%d, behind=%d, critical=%d",
            len(self._progress), on_track, behind, len(critical),
        )
        return PhaseResult(
            phase_name="progress_calc",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _assess_target(
        self, base: AnnualEmissions, current: AnnualEmissions,
        target: TargetReference, config: ProgressReviewConfig,
    ) -> ProgressAssessment:
        """Assess progress for a single target."""
        # Determine actual emissions for target scopes
        actual = self._get_scope_emissions(current, target.scopes)
        base_actual = self._get_scope_emissions(base, target.scopes)

        # Use target's base year emissions if available
        base_emissions = target.base_year_emissions_tco2e
        if base_emissions <= 0:
            base_emissions = base_actual

        target_emissions = target.target_emissions_tco2e
        if target_emissions <= 0 and target.target_reduction_pct > 0:
            target_emissions = base_emissions * (1.0 - target.target_reduction_pct / 100.0)

        # Years
        years_elapsed = current.year - target.base_year
        total_years = target.target_year - target.base_year
        years_remaining = target.target_year - current.year

        # Required pathway: linear interpolation from base to target
        if total_years > 0:
            required_reduction_pct = (target.target_reduction_pct * years_elapsed / total_years)
            required_emissions = base_emissions * (1.0 - required_reduction_pct / 100.0)
        else:
            required_reduction_pct = target.target_reduction_pct
            required_emissions = target_emissions

        # Actual reduction
        actual_reduction_pct = 0.0
        if base_emissions > 0:
            actual_reduction_pct = (1.0 - actual / base_emissions) * 100.0

        # Gap
        gap_tco2e = actual - required_emissions
        gap_pct = 0.0
        if required_emissions > 0:
            gap_pct = abs(gap_tco2e) / required_emissions * 100.0

        # Actual annual reduction rate
        actual_annual_rate = 0.0
        if years_elapsed > 0 and base_emissions > 0 and actual > 0:
            actual_annual_rate = (1.0 - (actual / base_emissions) ** (1.0 / years_elapsed)) * 100.0

        # Required annual rate
        required_annual_rate = target.annual_reduction_rate
        if required_annual_rate <= 0 and total_years > 0:
            required_annual_rate = (target.target_reduction_pct / total_years)

        # RAG status
        if actual <= required_emissions:
            # On track or ahead
            rag = RAGStatus.GREEN
            if actual < required_emissions * 0.95:
                tracking = TrackingStatus.AHEAD_OF_TARGET
            else:
                tracking = TrackingStatus.ON_TRACK
        elif gap_pct <= RAG_GREEN_THRESHOLD:
            rag = RAGStatus.GREEN
            tracking = TrackingStatus.ON_TRACK
        elif gap_pct <= RAG_AMBER_THRESHOLD:
            rag = RAGStatus.AMBER
            tracking = TrackingStatus.MINOR_DEVIATION
        else:
            rag = RAGStatus.RED
            tracking = TrackingStatus.MAJOR_DEVIATION

        # Trajectory projection (linear extrapolation)
        projected_target_year = actual
        projected_meets = False
        if years_elapsed > 0 and base_emissions > 0:
            # Annual change rate
            annual_change = (actual - base_emissions) / years_elapsed
            if years_remaining > 0:
                projected_target_year = actual + annual_change * years_remaining
                projected_target_year = max(0.0, projected_target_year)
                projected_meets = projected_target_year <= target_emissions

            # Critical assessment
            if projected_target_year > target_emissions * (1.0 + CRITICAL_OVERSHOOT_THRESHOLD / 100.0):
                rag = RAGStatus.RED
                tracking = TrackingStatus.CRITICAL

        # Carbon budget calculation
        # Total budget = area under the pathway line from base to target
        carbon_budget_total = 0.0
        if total_years > 0:
            carbon_budget_total = (base_emissions + target_emissions) / 2.0 * total_years

        # Estimate used budget from historical data + current
        carbon_budget_used = 0.0
        all_years = sorted(
            config.historical_emissions + [current],
            key=lambda e: e.year,
        )
        for emit in all_years:
            if target.base_year < emit.year <= current.year:
                yr_emissions = self._get_scope_emissions(emit, target.scopes)
                carbon_budget_used += yr_emissions

        # If no historical data, estimate from simple average
        if carbon_budget_used <= 0 and years_elapsed > 0:
            avg_emissions = (base_emissions + actual) / 2.0
            carbon_budget_used = avg_emissions * years_elapsed

        carbon_budget_remaining = max(0.0, carbon_budget_total - carbon_budget_used)
        budget_used_pct = (carbon_budget_used / carbon_budget_total * 100.0) if carbon_budget_total > 0 else 0.0

        return ProgressAssessment(
            target_id=target.target_id,
            target_name=target.target_name,
            rag_status=rag,
            tracking_status=tracking,
            required_emissions_tco2e=round(required_emissions, 2),
            actual_emissions_tco2e=round(actual, 2),
            gap_tco2e=round(gap_tco2e, 2),
            gap_pct=round(gap_pct, 2),
            actual_reduction_pct=round(actual_reduction_pct, 2),
            required_reduction_pct=round(required_reduction_pct, 2),
            actual_annual_rate=round(actual_annual_rate, 4),
            required_annual_rate=round(required_annual_rate, 4),
            years_remaining=years_remaining,
            projected_target_year_emissions=round(projected_target_year, 2),
            projected_meets_target=projected_meets,
            carbon_budget_total_tco2e=round(carbon_budget_total, 2),
            carbon_budget_used_tco2e=round(carbon_budget_used, 2),
            carbon_budget_remaining_tco2e=round(carbon_budget_remaining, 2),
            carbon_budget_used_pct=round(budget_used_pct, 2),
        )

    def _get_scope_emissions(
        self, emissions: AnnualEmissions, scopes: List[str],
    ) -> float:
        """Extract emissions for the specified scopes."""
        total = 0.0
        for scope in scopes:
            scope_lower = scope.lower().strip()
            if scope_lower == "scope1":
                total += emissions.scope1_tco2e
            elif scope_lower == "scope2":
                total += emissions.scope2_location_tco2e
            elif scope_lower == "scope3":
                total += emissions.scope3_total_tco2e
            elif scope_lower == "flag":
                total += emissions.flag_emissions_tco2e
        if not scopes:
            total = emissions.total_emissions
        return total

    # -------------------------------------------------------------------------
    # Phase 3: Variance Analysis
    # -------------------------------------------------------------------------

    async def _phase_variance_analysis(self, config: ProgressReviewConfig) -> PhaseResult:
        """Decompose variance by driver (structural, activity, intensity)."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._variances = []

        base = config.base_year_emissions
        current = config.current_year_emissions

        for target in config.targets:
            variance = self._decompose_variance(base, current, target, config)
            self._variances.append(variance)

        # Summary
        total_var = sum(v.total_variance_tco2e for v in self._variances)
        drivers = {}
        for v in self._variances:
            for c in v.components:
                drivers[c.driver.value] = drivers.get(c.driver.value, 0.0) + c.impact_tco2e

        outputs["variance_analyses"] = len(self._variances)
        outputs["total_variance_tco2e"] = round(total_var, 2)
        outputs["driver_breakdown"] = {k: round(v, 2) for k, v in drivers.items()}

        # Identify dominant driver
        if drivers:
            dominant = max(drivers, key=lambda d: abs(drivers[d]))
            outputs["dominant_driver"] = dominant
        else:
            outputs["dominant_driver"] = "none"

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Variance analysis: %d targets, total variance=%.2f tCO2e",
            len(self._variances), total_var,
        )
        return PhaseResult(
            phase_name="variance_analysis",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _decompose_variance(
        self, base: AnnualEmissions, current: AnnualEmissions,
        target: TargetReference, config: ProgressReviewConfig,
    ) -> VarianceResult:
        """Decompose emissions variance into structural, activity, intensity drivers."""
        base_emissions = self._get_scope_emissions(base, target.scopes)
        current_emissions = self._get_scope_emissions(current, target.scopes)
        total_var = current_emissions - base_emissions

        components: List[VarianceComponent] = []

        # Activity effect (volume/production change)
        base_activity = config.base_year_activity
        current_activity = config.current_year_activity

        if base_activity > 0 and current_activity > 0:
            activity_ratio = current_activity / base_activity
            activity_effect = base_emissions * (activity_ratio - 1.0)
            components.append(VarianceComponent(
                driver=VarianceDriver.ACTIVITY,
                description=(
                    f"Activity change: {base_activity:.0f} -> {current_activity:.0f} "
                    f"({(activity_ratio - 1.0) * 100:.1f}%)"
                ),
                impact_tco2e=round(activity_effect, 2),
                impact_pct=round(
                    activity_effect / abs(total_var) * 100.0 if total_var != 0 else 0.0, 2
                ),
                is_controllable=True,
            ))

            # Intensity effect (remaining after activity)
            intensity_effect = total_var - activity_effect
            components.append(VarianceComponent(
                driver=VarianceDriver.INTENSITY,
                description="Intensity change (emissions per unit of activity)",
                impact_tco2e=round(intensity_effect, 2),
                impact_pct=round(
                    intensity_effect / abs(total_var) * 100.0 if total_var != 0 else 0.0, 2
                ),
                is_controllable=True,
            ))
        else:
            # Cannot decompose without activity data
            components.append(VarianceComponent(
                driver=VarianceDriver.UNEXPLAINED,
                description="Total variance (activity data not available for decomposition)",
                impact_tco2e=round(total_var, 2),
                impact_pct=100.0,
                is_controllable=True,
            ))

        # Add structural component from recalc triggers if any
        structural_total = 0.0
        for trigger in config.recalc_triggers:
            if trigger.trigger_type in (
                RecalcTriggerType.ACQUISITION, RecalcTriggerType.DIVESTITURE,
                RecalcTriggerType.MERGER, RecalcTriggerType.STRUCTURAL_CHANGE,
            ):
                structural_total += trigger.impact_tco2e

        if structural_total != 0:
            components.append(VarianceComponent(
                driver=VarianceDriver.STRUCTURAL,
                description="Structural changes (M&A, divestitures, reorganization)",
                impact_tco2e=round(structural_total, 2),
                impact_pct=round(
                    structural_total / abs(total_var) * 100.0 if total_var != 0 else 0.0, 2
                ),
                is_controllable=False,
            ))

        # Determine primary driver
        primary = VarianceDriver.UNEXPLAINED
        if components:
            primary_comp = max(components, key=lambda c: abs(c.impact_tco2e))
            primary = primary_comp.driver

        # Trend direction
        trend = "flat"
        if total_var < -0.01 * abs(base_emissions):
            trend = "improving"
        elif total_var > 0.01 * abs(base_emissions):
            trend = "deteriorating"

        # Year-over-year change
        yoy_change = 0.0
        if config.prior_year_emissions:
            prior_emissions = self._get_scope_emissions(config.prior_year_emissions, target.scopes)
            if prior_emissions > 0:
                yoy_change = (current_emissions - prior_emissions) / prior_emissions * 100.0

        return VarianceResult(
            target_id=target.target_id,
            total_variance_tco2e=round(total_var, 2),
            total_variance_pct=round(
                total_var / base_emissions * 100.0 if base_emissions > 0 else 0.0, 2
            ),
            components=components,
            primary_driver=primary,
            trend_direction=trend,
            year_over_year_change_pct=round(yoy_change, 2),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Recalculation Evaluation
    # -------------------------------------------------------------------------

    async def _phase_recalc(self, config: ProgressReviewConfig) -> PhaseResult:
        """Evaluate recalculation triggers and apply if needed."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        base_emissions = config.base_year_emissions.total_emissions
        recalc_required = False
        total_adjustment = 0.0
        evaluated_triggers: List[RecalcTrigger] = []
        notes: List[str] = []

        for trigger in config.recalc_triggers:
            # Calculate significance
            impact_pct = 0.0
            if base_emissions > 0:
                impact_pct = abs(trigger.impact_tco2e) / base_emissions * 100.0

            exceeds = impact_pct >= SIGNIFICANCE_THRESHOLD_PCT

            evaluated_triggers.append(RecalcTrigger(
                trigger_type=trigger.trigger_type,
                description=trigger.description,
                impact_tco2e=round(trigger.impact_tco2e, 2),
                impact_pct=round(impact_pct, 2),
                exceeds_threshold=exceeds,
                significance_threshold_pct=SIGNIFICANCE_THRESHOLD_PCT,
            ))

            if exceeds:
                recalc_required = True
                total_adjustment += trigger.impact_tco2e
                notes.append(
                    f"{trigger.trigger_type.value}: {impact_pct:.1f}% impact "
                    f"(>{SIGNIFICANCE_THRESHOLD_PCT}% threshold)"
                )

        # Calculate adjusted base
        adjusted_base = base_emissions + total_adjustment
        adjustment_pct = 0.0
        if base_emissions > 0:
            adjustment_pct = total_adjustment / base_emissions * 100.0

        # Count targets that would be adjusted
        targets_adjusted = len(config.targets) if recalc_required else 0

        if recalc_required:
            notes.append(
                f"Base year recalculation required: {base_emissions:.2f} -> "
                f"{adjusted_base:.2f} tCO2e ({adjustment_pct:+.1f}%)"
            )
            warnings.append(
                "Base year recalculation triggered; all targets will need adjustment"
            )
        else:
            notes.append("No structural changes exceed the 5% significance threshold")

        self._recalc = RecalcResult(
            recalculation_required=recalc_required,
            triggers=evaluated_triggers,
            original_base_emissions=round(base_emissions, 2),
            adjusted_base_emissions=round(adjusted_base, 2),
            adjustment_tco2e=round(total_adjustment, 2),
            adjustment_pct=round(adjustment_pct, 2),
            targets_adjusted=targets_adjusted,
            notes=notes,
        )

        outputs["recalculation_required"] = recalc_required
        outputs["triggers_evaluated"] = len(evaluated_triggers)
        outputs["triggers_exceeding_threshold"] = sum(
            1 for t in evaluated_triggers if t.exceeds_threshold
        )
        outputs["original_base_tco2e"] = round(base_emissions, 2)
        outputs["adjusted_base_tco2e"] = round(adjusted_base, 2)
        outputs["adjustment_pct"] = round(adjustment_pct, 2)
        outputs["targets_adjusted"] = targets_adjusted

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Recalc eval: required=%s, triggers=%d, adjustment=%.1f%%",
            recalc_required, len(evaluated_triggers), adjustment_pct,
        )
        return PhaseResult(
            phase_name="recalc",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Report Generation
    # -------------------------------------------------------------------------

    async def _phase_report(self, config: ProgressReviewConfig) -> PhaseResult:
        """Generate annual progress report with corrective actions."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        current_year = config.current_year_emissions.year

        # Overall RAG (worst-case across all targets)
        rag_priority = {RAGStatus.RED: 3, RAGStatus.AMBER: 2, RAGStatus.GREEN: 1, RAGStatus.NOT_ASSESSED: 0}
        overall_rag = RAGStatus.NOT_ASSESSED
        for p in self._progress:
            if rag_priority.get(p.rag_status, 0) > rag_priority.get(overall_rag, 0):
                overall_rag = p.rag_status

        # Overall tracking status
        tracking_priority = {
            TrackingStatus.CRITICAL: 5,
            TrackingStatus.MAJOR_DEVIATION: 4,
            TrackingStatus.MINOR_DEVIATION: 3,
            TrackingStatus.ON_TRACK: 2,
            TrackingStatus.AHEAD_OF_TARGET: 1,
            TrackingStatus.NOT_STARTED: 0,
        }
        overall_tracking = TrackingStatus.NOT_STARTED
        for p in self._progress:
            if tracking_priority.get(p.tracking_status, 0) > tracking_priority.get(overall_tracking, 0):
                overall_tracking = p.tracking_status

        on_track = sum(1 for p in self._progress if p.tracking_status in (
            TrackingStatus.ON_TRACK, TrackingStatus.AHEAD_OF_TARGET))
        behind = sum(1 for p in self._progress if p.tracking_status in (
            TrackingStatus.MINOR_DEVIATION, TrackingStatus.MAJOR_DEVIATION,
            TrackingStatus.CRITICAL))
        ahead = sum(1 for p in self._progress if p.tracking_status == TrackingStatus.AHEAD_OF_TARGET)

        # Total reduction achieved
        total_reduction = 0.0
        if self._progress:
            # Use the broadest target's reduction
            total_reduction = max(p.actual_reduction_pct for p in self._progress)

        # Key findings
        findings: List[str] = []
        findings.append(
            f"Overall status: {overall_rag.value.upper()} - {overall_tracking.value}"
        )
        findings.append(
            f"{on_track}/{len(self._progress)} targets on track, "
            f"{behind} behind, {ahead} ahead of target"
        )
        findings.append(
            f"Total emissions reduction from base year: {total_reduction:.1f}%"
        )

        if self._recalc and self._recalc.recalculation_required:
            findings.append(
                "BASE YEAR RECALCULATION REQUIRED due to structural changes "
                f"exceeding {SIGNIFICANCE_THRESHOLD_PCT}% threshold"
            )

        # Generate corrective actions for off-track targets
        actions: List[CorrectiveAction] = []
        action_counter = 0

        for p in self._progress:
            if p.tracking_status in (
                TrackingStatus.MINOR_DEVIATION,
                TrackingStatus.MAJOR_DEVIATION,
                TrackingStatus.CRITICAL,
            ):
                action_counter += 1

                # Priority based on status
                if p.tracking_status == TrackingStatus.CRITICAL:
                    priority = CorrectiveActionPriority.CRITICAL
                    timeline = 3
                elif p.tracking_status == TrackingStatus.MAJOR_DEVIATION:
                    priority = CorrectiveActionPriority.HIGH
                    timeline = 6
                else:
                    priority = CorrectiveActionPriority.MEDIUM
                    timeline = 12

                # Gap to close
                gap = abs(p.gap_tco2e)

                actions.append(CorrectiveAction(
                    action_id=f"CA-{action_counter:03d}",
                    target_id=p.target_id,
                    priority=priority,
                    description=(
                        f"Close {gap:.0f} tCO2e gap for target {p.target_id}. "
                        f"Currently {p.gap_pct:.1f}% behind required pathway. "
                        f"Increase annual reduction to {p.required_annual_rate:.2f}%/yr."
                    ),
                    expected_impact_tco2e=round(gap, 2),
                    timeline_months=timeline,
                    category="emissions_reduction",
                ))

                # Carbon budget warning
                if p.carbon_budget_used_pct > BUDGET_WARNING_PCT:
                    action_counter += 1
                    actions.append(CorrectiveAction(
                        action_id=f"CA-{action_counter:03d}",
                        target_id=p.target_id,
                        priority=CorrectiveActionPriority.HIGH,
                        description=(
                            f"Carbon budget alert: {p.carbon_budget_used_pct:.0f}% consumed "
                            f"with {p.years_remaining} years remaining. "
                            "Accelerate reduction efforts immediately."
                        ),
                        expected_impact_tco2e=round(p.carbon_budget_remaining_tco2e, 2),
                        timeline_months=6,
                        category="carbon_budget",
                    ))

        # Sort actions by priority
        priority_order = {
            CorrectiveActionPriority.CRITICAL: 0,
            CorrectiveActionPriority.HIGH: 1,
            CorrectiveActionPriority.MEDIUM: 2,
            CorrectiveActionPriority.LOW: 3,
        }
        actions.sort(key=lambda a: priority_order.get(a.priority, 99))

        # Next review date
        next_review = f"{current_year + 1}-03-31"

        self._report = ProgressReport(
            report_year=current_year,
            overall_rag=overall_rag,
            overall_tracking=overall_tracking,
            targets_on_track=on_track,
            targets_behind=behind,
            targets_ahead=ahead,
            total_reduction_achieved_pct=round(total_reduction, 2),
            key_findings=findings,
            corrective_actions=actions,
            next_review_date=next_review,
        )

        outputs["report_year"] = current_year
        outputs["overall_rag"] = overall_rag.value
        outputs["overall_tracking"] = overall_tracking.value
        outputs["targets_on_track"] = on_track
        outputs["targets_behind"] = behind
        outputs["targets_ahead"] = ahead
        outputs["total_reduction_pct"] = round(total_reduction, 2)
        outputs["corrective_actions_count"] = len(actions)
        outputs["critical_actions"] = sum(
            1 for a in actions if a.priority == CorrectiveActionPriority.CRITICAL
        )
        outputs["next_review_date"] = next_review

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Report: RAG=%s, on_track=%d/%d, actions=%d",
            overall_rag.value, on_track, len(self._progress), len(actions),
        )
        return PhaseResult(
            phase_name="report",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )
