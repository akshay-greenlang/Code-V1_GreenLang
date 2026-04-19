# -*- coding: utf-8 -*-
"""
Energy Savings Verification Workflow
==========================================

4-phase workflow for IPMVP-compliant energy savings measurement and
verification within PACK-031 Industrial Energy Audit Pack.

Phases:
    1. BaselinePeriodValidation    -- Verify pre-implementation baseline quality
    2. ImplementationTracking      -- Log ECM implementation dates, costs, scope
    3. PostImplementationMeasurement -- Collect post data, apply adjustments
    4. MVReportGeneration          -- IPMVP Options A-D compliant M&V report

The workflow follows GreenLang zero-hallucination principles: all savings
calculations use deterministic IPMVP formulas with baseline-adjusted
consumption comparisons. No LLM calls in the numeric computation path.

Schedule: quarterly
Estimated duration: 120 minutes

Author: GreenLang Team
Version: 31.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

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


class IPMVPOption(str, Enum):
    """IPMVP M&V option classification."""

    OPTION_A = "option_a"  # Retrofit isolation: key parameter measurement
    OPTION_B = "option_b"  # Retrofit isolation: all parameter measurement
    OPTION_C = "option_c"  # Whole facility
    OPTION_D = "option_d"  # Calibrated simulation


class ECMStatus(str, Enum):
    """ECM implementation status."""

    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"
    CANCELLED = "cancelled"


class BaselineQuality(str, Enum):
    """Baseline quality assessment level."""

    EXCELLENT = "excellent"  # R-sq >= 0.90, CV(RMSE) <= 15%
    GOOD = "good"            # R-sq >= 0.75, CV(RMSE) <= 25%
    ACCEPTABLE = "acceptable"  # R-sq >= 0.60, CV(RMSE) <= 35%
    POOR = "poor"            # Below acceptable thresholds
    INSUFFICIENT = "insufficient"  # Not enough data


class AdjustmentType(str, Enum):
    """Type of adjustment applied to baseline."""

    ROUTINE = "routine"        # Production, weather changes
    NON_ROUTINE = "non_routine"  # Equipment additions, schedule changes
    STATIC = "static"          # Factors that changed permanently


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class BaselinePeriodData(BaseModel):
    """Pre-implementation baseline period data."""

    baseline_id: str = Field(default_factory=lambda: f"bl-{uuid.uuid4().hex[:8]}")
    period_start: str = Field(default="", description="YYYY-MM-DD")
    period_end: str = Field(default="", description="YYYY-MM-DD")
    months: int = Field(default=12, ge=1, le=36, description="Baseline months")
    monthly_consumption_kwh: List[float] = Field(default_factory=list, description="Monthly kWh")
    monthly_production: List[float] = Field(default_factory=list, description="Monthly production")
    monthly_hdd: List[float] = Field(default_factory=list, description="Monthly HDD")
    monthly_cdd: List[float] = Field(default_factory=list, description="Monthly CDD")
    total_consumption_kwh: float = Field(default=0.0, ge=0.0)
    total_cost_eur: float = Field(default=0.0, ge=0.0)
    regression_r_squared: float = Field(default=0.0, ge=0.0, le=1.0)
    regression_cv_rmse_pct: float = Field(default=0.0, ge=0.0)
    model_equation: str = Field(default="", description="Regression equation")
    energy_source: str = Field(default="electricity")


class ECMImplementation(BaseModel):
    """Energy Conservation Measure implementation record."""

    ecm_id: str = Field(default_factory=lambda: f"ecm-{uuid.uuid4().hex[:8]}")
    title: str = Field(default="", description="ECM title")
    description: str = Field(default="", description="ECM description")
    ecm_type: str = Field(default="", description="retrofit|operational|behavioral")
    ipmvp_option: IPMVPOption = Field(default=IPMVPOption.OPTION_C)
    status: ECMStatus = Field(default=ECMStatus.PLANNED)
    start_date: str = Field(default="", description="Implementation start YYYY-MM-DD")
    completion_date: str = Field(default="", description="Implementation end YYYY-MM-DD")
    verification_date: str = Field(default="", description="Verification YYYY-MM-DD")
    estimated_savings_kwh: float = Field(default=0.0, ge=0.0, description="Estimated annual kWh")
    estimated_savings_eur: float = Field(default=0.0, ge=0.0, description="Estimated annual EUR")
    actual_cost_eur: float = Field(default=0.0, ge=0.0, description="Actual implementation cost")
    budgeted_cost_eur: float = Field(default=0.0, ge=0.0, description="Budgeted cost")
    measurement_boundary: str = Field(default="", description="M&V measurement boundary")
    key_parameters: List[str] = Field(default_factory=list, description="Key measurement params")
    interactive_effects: str = Field(default="", description="Interactive effects description")


class BaselineAdjustment(BaseModel):
    """Adjustment applied to baseline for post-period comparison."""

    adjustment_id: str = Field(default_factory=lambda: f"adj-{uuid.uuid4().hex[:8]}")
    adjustment_type: AdjustmentType = Field(default=AdjustmentType.ROUTINE)
    description: str = Field(default="")
    adjustment_kwh: float = Field(default=0.0, description="Adjustment in kWh (positive=increase)")
    variable: str = Field(default="", description="Variable being adjusted")
    calculation_method: str = Field(default="", description="How adjustment was calculated")


class PostImplementationData(BaseModel):
    """Post-implementation measurement period data."""

    period_start: str = Field(default="", description="YYYY-MM-DD")
    period_end: str = Field(default="", description="YYYY-MM-DD")
    months: int = Field(default=12, ge=1, le=36)
    monthly_consumption_kwh: List[float] = Field(default_factory=list)
    monthly_production: List[float] = Field(default_factory=list)
    monthly_hdd: List[float] = Field(default_factory=list)
    monthly_cdd: List[float] = Field(default_factory=list)
    total_consumption_kwh: float = Field(default=0.0, ge=0.0)
    total_cost_eur: float = Field(default=0.0, ge=0.0)
    adjustments: List[BaselineAdjustment] = Field(default_factory=list)


class MVReport(BaseModel):
    """Measurement and Verification report output."""

    report_id: str = Field(default_factory=lambda: f"mvr-{uuid.uuid4().hex[:8]}")
    ecm_id: str = Field(default="")
    ecm_title: str = Field(default="")
    ipmvp_option: IPMVPOption = Field(default=IPMVPOption.OPTION_C)
    baseline_consumption_kwh: float = Field(default=0.0, ge=0.0, description="Adjusted baseline")
    post_consumption_kwh: float = Field(default=0.0, ge=0.0, description="Post-period actual")
    gross_savings_kwh: float = Field(default=0.0, description="Baseline - Post")
    routine_adjustments_kwh: float = Field(default=0.0, description="Routine adj total")
    non_routine_adjustments_kwh: float = Field(default=0.0, description="Non-routine adj total")
    net_savings_kwh: float = Field(default=0.0, description="Net verified savings")
    net_savings_pct: float = Field(default=0.0, description="Net savings %")
    cost_savings_eur: float = Field(default=0.0, description="Monetary savings")
    co2_reduction_tonnes: float = Field(default=0.0, description="CO2 reduction")
    savings_uncertainty_pct: float = Field(default=0.0, ge=0.0, description="Savings uncertainty %")
    confidence_level_pct: float = Field(default=90.0, description="Statistical confidence %")
    meets_ipmvp: bool = Field(default=False, description="Meets IPMVP requirements")
    precision_pct: float = Field(default=0.0, description="Savings precision % at confidence")
    verification_period_months: int = Field(default=12)


class EnergySavingsVerificationInput(BaseModel):
    """Input data model for EnergySavingsVerificationWorkflow."""

    facility_id: str = Field(default="", description="Facility identifier")
    baselines: List[BaselinePeriodData] = Field(default_factory=list)
    ecms: List[ECMImplementation] = Field(default_factory=list)
    post_data: PostImplementationData = Field(default_factory=PostImplementationData)
    electricity_ef_kgco2_kwh: float = Field(default=0.385, ge=0.0)
    gas_ef_kgco2_kwh: float = Field(default=0.184, ge=0.0)
    energy_cost_eur_per_kwh: float = Field(default=0.10, ge=0.0)
    confidence_level_pct: float = Field(default=90.0, ge=80.0, le=99.0)
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class EnergySavingsVerificationResult(BaseModel):
    """Complete result from energy savings verification workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="energy_savings_verification")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    facility_id: str = Field(default="")
    mv_reports: List[MVReport] = Field(default_factory=list)
    baseline_quality: BaselineQuality = Field(default=BaselineQuality.INSUFFICIENT)
    total_gross_savings_kwh: float = Field(default=0.0)
    total_net_savings_kwh: float = Field(default=0.0)
    total_cost_savings_eur: float = Field(default=0.0)
    total_co2_reduction_tonnes: float = Field(default=0.0)
    overall_savings_pct: float = Field(default=0.0)
    ecm_count: int = Field(default=0)
    ecms_verified: int = Field(default=0)
    ipmvp_compliant: bool = Field(default=False)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# IPMVP REFERENCE CONSTANTS (Zero-Hallucination)
# =============================================================================

# IPMVP baseline quality thresholds
QUALITY_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "excellent": {"min_r_squared": 0.90, "max_cv_rmse": 15.0},
    "good": {"min_r_squared": 0.75, "max_cv_rmse": 25.0},
    "acceptable": {"min_r_squared": 0.60, "max_cv_rmse": 35.0},
}

# t-statistic lookup for 90% confidence (two-tailed, selected df values)
T_STATS_90: Dict[int, float] = {
    6: 1.943, 8: 1.860, 10: 1.812, 12: 1.782, 18: 1.734, 24: 1.711, 36: 1.690,
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class EnergySavingsVerificationWorkflow:
    """
    4-phase IPMVP-compliant energy savings verification workflow.

    Validates baseline quality, tracks ECM implementation, measures
    post-implementation consumption with adjustments, and generates
    M&V reports per IPMVP Options A-D.

    Zero-hallucination: all savings calculations use deterministic
    IPMVP formulas. No LLM calls in the numeric computation path.

    Attributes:
        workflow_id: Unique execution identifier.
        _mv_reports: M&V report outputs per ECM.
        _baseline_quality: Assessed baseline quality level.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = EnergySavingsVerificationWorkflow()
        >>> inp = EnergySavingsVerificationInput(baselines=[...], ecms=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.ipmvp_compliant is True
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize EnergySavingsVerificationWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._mv_reports: List[MVReport] = []
        self._baseline_quality: BaselineQuality = BaselineQuality.INSUFFICIENT
        self._adjusted_baseline_kwh: float = 0.0
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        input_data: Optional[EnergySavingsVerificationInput] = None,
        baselines: Optional[List[BaselinePeriodData]] = None,
        ecms: Optional[List[ECMImplementation]] = None,
    ) -> EnergySavingsVerificationResult:
        """
        Execute the 4-phase energy savings verification workflow.

        Args:
            input_data: Full input model (preferred).
            baselines: Baseline data (fallback).
            ecms: ECM records (fallback).

        Returns:
            EnergySavingsVerificationResult with M&V reports.
        """
        if input_data is None:
            input_data = EnergySavingsVerificationInput(
                baselines=baselines or [],
                ecms=ecms or [],
            )

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting savings verification workflow %s for facility=%s",
            self.workflow_id, input_data.facility_id,
        )

        self._phase_results = []
        self._mv_reports = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_baseline_validation(input_data)
            self._phase_results.append(phase1)

            phase2 = await self._phase_implementation_tracking(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_post_measurement(input_data)
            self._phase_results.append(phase3)

            phase4 = await self._phase_mv_report_generation(input_data)
            self._phase_results.append(phase4)

            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Savings verification workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        total_gross = sum(r.gross_savings_kwh for r in self._mv_reports)
        total_net = sum(r.net_savings_kwh for r in self._mv_reports)
        total_cost_sav = sum(r.cost_savings_eur for r in self._mv_reports)
        total_co2 = sum(r.co2_reduction_tonnes for r in self._mv_reports)
        verified_count = sum(1 for r in self._mv_reports if r.meets_ipmvp)

        overall_sav_pct = 0.0
        if self._adjusted_baseline_kwh > 0:
            overall_sav_pct = total_net / self._adjusted_baseline_kwh * 100.0

        result = EnergySavingsVerificationResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            facility_id=input_data.facility_id,
            mv_reports=self._mv_reports,
            baseline_quality=self._baseline_quality,
            total_gross_savings_kwh=round(total_gross, 2),
            total_net_savings_kwh=round(total_net, 2),
            total_cost_savings_eur=round(total_cost_sav, 2),
            total_co2_reduction_tonnes=round(total_co2, 4),
            overall_savings_pct=round(overall_sav_pct, 2),
            ecm_count=len(input_data.ecms),
            ecms_verified=verified_count,
            ipmvp_compliant=all(r.meets_ipmvp for r in self._mv_reports) if self._mv_reports else False,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Savings verification workflow %s completed in %.2fs net_savings=%.0f kWh",
            self.workflow_id, elapsed, total_net,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Baseline Period Validation
    # -------------------------------------------------------------------------

    async def _phase_baseline_validation(
        self, input_data: EnergySavingsVerificationInput
    ) -> PhaseResult:
        """Verify pre-implementation baseline quality per IPMVP."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        if not input_data.baselines:
            warnings.append("No baseline data provided")
            self._baseline_quality = BaselineQuality.INSUFFICIENT
            outputs["quality"] = self._baseline_quality.value
            elapsed = (datetime.utcnow() - started).total_seconds()
            return PhaseResult(
                phase_name="baseline_validation", phase_number=1,
                status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
                outputs=outputs, warnings=warnings,
                provenance_hash=self._hash_dict(outputs),
            )

        # Assess each baseline
        baseline_assessments: List[Dict[str, Any]] = []
        overall_r_sq = 0.0
        overall_cv_rmse = 0.0

        for bl in input_data.baselines:
            quality = self._assess_baseline_quality(bl)
            baseline_assessments.append({
                "baseline_id": bl.baseline_id,
                "energy_source": bl.energy_source,
                "r_squared": bl.regression_r_squared,
                "cv_rmse_pct": bl.regression_cv_rmse_pct,
                "months": bl.months,
                "quality": quality.value,
            })
            overall_r_sq = max(overall_r_sq, bl.regression_r_squared)
            overall_cv_rmse = max(overall_cv_rmse, bl.regression_cv_rmse_pct)

            # Validate data sufficiency
            if bl.months < 12:
                warnings.append(
                    f"Baseline {bl.baseline_id}: only {bl.months} months (12 recommended)"
                )
            if len(bl.monthly_consumption_kwh) < bl.months:
                warnings.append(
                    f"Baseline {bl.baseline_id}: incomplete monthly data"
                )

        # Overall quality based on primary baseline
        self._baseline_quality = self._assess_baseline_quality(input_data.baselines[0])

        outputs["baselines_assessed"] = len(input_data.baselines)
        outputs["baseline_assessments"] = baseline_assessments
        outputs["overall_quality"] = self._baseline_quality.value
        outputs["best_r_squared"] = round(overall_r_sq, 4)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 BaselineValidation: quality=%s R-sq=%.3f",
            self._baseline_quality.value, overall_r_sq,
        )
        return PhaseResult(
            phase_name="baseline_validation", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _assess_baseline_quality(self, baseline: BaselinePeriodData) -> BaselineQuality:
        """Assess baseline quality per IPMVP criteria."""
        r_sq = baseline.regression_r_squared
        cv_rmse = baseline.regression_cv_rmse_pct

        if r_sq >= 0.90 and cv_rmse <= 15.0:
            return BaselineQuality.EXCELLENT
        elif r_sq >= 0.75 and cv_rmse <= 25.0:
            return BaselineQuality.GOOD
        elif r_sq >= 0.60 and cv_rmse <= 35.0:
            return BaselineQuality.ACCEPTABLE
        elif baseline.months >= 6:
            return BaselineQuality.POOR
        else:
            return BaselineQuality.INSUFFICIENT

    # -------------------------------------------------------------------------
    # Phase 2: Implementation Tracking
    # -------------------------------------------------------------------------

    async def _phase_implementation_tracking(
        self, input_data: EnergySavingsVerificationInput
    ) -> PhaseResult:
        """Log ECM implementation dates, costs, and scope."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        ecm_summaries: List[Dict[str, Any]] = []
        total_estimated_savings = 0.0
        total_actual_cost = 0.0
        total_budgeted_cost = 0.0

        for ecm in input_data.ecms:
            # Validate ECM data completeness
            if not ecm.completion_date and ecm.status == ECMStatus.COMPLETED:
                warnings.append(f"ECM {ecm.ecm_id}: completed but no completion_date")

            if ecm.actual_cost_eur == 0 and ecm.status in (ECMStatus.COMPLETED, ECMStatus.VERIFIED):
                warnings.append(f"ECM {ecm.ecm_id}: no actual cost recorded")

            cost_variance = 0.0
            if ecm.budgeted_cost_eur > 0:
                cost_variance = ((ecm.actual_cost_eur - ecm.budgeted_cost_eur)
                                 / ecm.budgeted_cost_eur * 100.0)

            ecm_summaries.append({
                "ecm_id": ecm.ecm_id,
                "title": ecm.title,
                "status": ecm.status.value,
                "ipmvp_option": ecm.ipmvp_option.value,
                "estimated_savings_kwh": ecm.estimated_savings_kwh,
                "actual_cost_eur": ecm.actual_cost_eur,
                "cost_variance_pct": round(cost_variance, 1),
            })

            total_estimated_savings += ecm.estimated_savings_kwh
            total_actual_cost += ecm.actual_cost_eur
            total_budgeted_cost += ecm.budgeted_cost_eur

        outputs["ecm_count"] = len(input_data.ecms)
        outputs["completed_ecms"] = sum(
            1 for e in input_data.ecms if e.status in (ECMStatus.COMPLETED, ECMStatus.VERIFIED)
        )
        outputs["total_estimated_savings_kwh"] = round(total_estimated_savings, 2)
        outputs["total_actual_cost_eur"] = round(total_actual_cost, 2)
        outputs["total_budgeted_cost_eur"] = round(total_budgeted_cost, 2)
        outputs["ecm_summaries"] = ecm_summaries

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 ImplementationTracking: %d ECMs, estimated savings=%.0f kWh",
            len(input_data.ecms), total_estimated_savings,
        )
        return PhaseResult(
            phase_name="implementation_tracking", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Post-Implementation Measurement
    # -------------------------------------------------------------------------

    async def _phase_post_measurement(
        self, input_data: EnergySavingsVerificationInput
    ) -> PhaseResult:
        """Collect post data and apply routine/non-routine adjustments."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        post = input_data.post_data

        # Calculate adjusted baseline for reporting conditions
        baseline_total = sum(bl.total_consumption_kwh for bl in input_data.baselines)
        routine_adj = sum(
            a.adjustment_kwh for a in post.adjustments
            if a.adjustment_type == AdjustmentType.ROUTINE
        )
        non_routine_adj = sum(
            a.adjustment_kwh for a in post.adjustments
            if a.adjustment_type == AdjustmentType.NON_ROUTINE
        )
        static_adj = sum(
            a.adjustment_kwh for a in post.adjustments
            if a.adjustment_type == AdjustmentType.STATIC
        )

        # IPMVP Equation: Savings = (Adjusted Baseline) - (Post-Period)
        # Adjusted Baseline = Baseline + Routine Adjustments + Non-Routine Adjustments
        adjusted_baseline = baseline_total + routine_adj + non_routine_adj + static_adj
        self._adjusted_baseline_kwh = adjusted_baseline

        post_total = post.total_consumption_kwh
        gross_savings = adjusted_baseline - post_total

        if post.months < 3:
            warnings.append(f"Post-period only {post.months} months; 12 recommended")

        outputs["baseline_total_kwh"] = round(baseline_total, 2)
        outputs["routine_adjustment_kwh"] = round(routine_adj, 2)
        outputs["non_routine_adjustment_kwh"] = round(non_routine_adj, 2)
        outputs["static_adjustment_kwh"] = round(static_adj, 2)
        outputs["adjusted_baseline_kwh"] = round(adjusted_baseline, 2)
        outputs["post_total_kwh"] = round(post_total, 2)
        outputs["gross_savings_kwh"] = round(gross_savings, 2)
        outputs["adjustments_count"] = len(post.adjustments)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 PostMeasurement: adjusted_baseline=%.0f post=%.0f gross_savings=%.0f kWh",
            adjusted_baseline, post_total, gross_savings,
        )
        return PhaseResult(
            phase_name="post_measurement", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: M&V Report Generation
    # -------------------------------------------------------------------------

    async def _phase_mv_report_generation(
        self, input_data: EnergySavingsVerificationInput
    ) -> PhaseResult:
        """Generate IPMVP Options A-D compliant M&V reports."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        post = input_data.post_data

        for ecm in input_data.ecms:
            report = self._generate_mv_report(ecm, input_data, post)
            self._mv_reports.append(report)

        # If no ECMs but we have baseline and post data, generate whole-facility report
        if not input_data.ecms and input_data.baselines and post.total_consumption_kwh > 0:
            report = self._generate_whole_facility_report(input_data, post)
            self._mv_reports.append(report)

        outputs["reports_generated"] = len(self._mv_reports)
        outputs["ipmvp_compliant_count"] = sum(1 for r in self._mv_reports if r.meets_ipmvp)
        outputs["total_net_savings_kwh"] = round(
            sum(r.net_savings_kwh for r in self._mv_reports), 2
        )
        outputs["total_cost_savings_eur"] = round(
            sum(r.cost_savings_eur for r in self._mv_reports), 2
        )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 MVReportGeneration: %d reports, net_savings=%.0f kWh",
            len(self._mv_reports), outputs["total_net_savings_kwh"],
        )
        return PhaseResult(
            phase_name="mv_report_generation", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _generate_mv_report(
        self,
        ecm: ECMImplementation,
        input_data: EnergySavingsVerificationInput,
        post: PostImplementationData,
    ) -> MVReport:
        """Generate M&V report for a single ECM."""
        # Proportional allocation based on estimated savings
        total_estimated = sum(e.estimated_savings_kwh for e in input_data.ecms)
        allocation_pct = (ecm.estimated_savings_kwh / total_estimated * 100.0) \
            if total_estimated > 0 else 100.0 / max(len(input_data.ecms), 1)

        # Calculate allocated savings
        total_gross = self._adjusted_baseline_kwh - post.total_consumption_kwh
        ecm_gross = total_gross * (allocation_pct / 100.0)

        routine_adj = sum(
            a.adjustment_kwh for a in post.adjustments
            if a.adjustment_type == AdjustmentType.ROUTINE
        ) * (allocation_pct / 100.0)
        non_routine_adj = sum(
            a.adjustment_kwh for a in post.adjustments
            if a.adjustment_type == AdjustmentType.NON_ROUTINE
        ) * (allocation_pct / 100.0)

        net_savings = ecm_gross
        net_pct = (net_savings / self._adjusted_baseline_kwh * 100.0) \
            if self._adjusted_baseline_kwh > 0 else 0.0
        cost_savings = net_savings * input_data.energy_cost_eur_per_kwh
        co2_reduction = net_savings * input_data.electricity_ef_kgco2_kwh / 1000.0

        # Uncertainty analysis (simplified per ASHRAE 14)
        uncertainty = self._calculate_savings_uncertainty(
            input_data.baselines, post, input_data.confidence_level_pct
        )
        precision = uncertainty * 100.0 / max(abs(net_savings), 1.0)
        meets_ipmvp = precision <= 50.0 and self._baseline_quality != BaselineQuality.INSUFFICIENT

        return MVReport(
            ecm_id=ecm.ecm_id,
            ecm_title=ecm.title,
            ipmvp_option=ecm.ipmvp_option,
            baseline_consumption_kwh=round(self._adjusted_baseline_kwh * allocation_pct / 100.0, 2),
            post_consumption_kwh=round(post.total_consumption_kwh * allocation_pct / 100.0, 2),
            gross_savings_kwh=round(ecm_gross, 2),
            routine_adjustments_kwh=round(routine_adj, 2),
            non_routine_adjustments_kwh=round(non_routine_adj, 2),
            net_savings_kwh=round(net_savings, 2),
            net_savings_pct=round(net_pct, 2),
            cost_savings_eur=round(cost_savings, 2),
            co2_reduction_tonnes=round(co2_reduction, 4),
            savings_uncertainty_pct=round(precision, 2),
            confidence_level_pct=input_data.confidence_level_pct,
            meets_ipmvp=meets_ipmvp,
            precision_pct=round(precision, 2),
            verification_period_months=post.months,
        )

    def _generate_whole_facility_report(
        self,
        input_data: EnergySavingsVerificationInput,
        post: PostImplementationData,
    ) -> MVReport:
        """Generate whole-facility (Option C) M&V report."""
        gross = self._adjusted_baseline_kwh - post.total_consumption_kwh
        net_pct = (gross / self._adjusted_baseline_kwh * 100.0) \
            if self._adjusted_baseline_kwh > 0 else 0.0
        cost_savings = gross * input_data.energy_cost_eur_per_kwh
        co2 = gross * input_data.electricity_ef_kgco2_kwh / 1000.0

        return MVReport(
            ecm_title="Whole Facility",
            ipmvp_option=IPMVPOption.OPTION_C,
            baseline_consumption_kwh=round(self._adjusted_baseline_kwh, 2),
            post_consumption_kwh=round(post.total_consumption_kwh, 2),
            gross_savings_kwh=round(gross, 2),
            net_savings_kwh=round(gross, 2),
            net_savings_pct=round(net_pct, 2),
            cost_savings_eur=round(cost_savings, 2),
            co2_reduction_tonnes=round(co2, 4),
            meets_ipmvp=self._baseline_quality != BaselineQuality.INSUFFICIENT,
            verification_period_months=post.months,
        )

    def _calculate_savings_uncertainty(
        self,
        baselines: List[BaselinePeriodData],
        post: PostImplementationData,
        confidence_pct: float,
    ) -> float:
        """Calculate savings uncertainty per ASHRAE Guideline 14."""
        if not baselines:
            return 0.0

        bl = baselines[0]
        cv_rmse = bl.regression_cv_rmse_pct / 100.0
        n = bl.months
        m = post.months if post.months > 0 else 12

        # t-statistic for given confidence level and degrees of freedom
        df = max(n - 2, 6)
        t_stat = T_STATS_90.get(df, 1.80)

        # Fractional uncertainty (ASHRAE Guideline 14 Equation 5-2)
        uncertainty_fraction = t_stat * cv_rmse * (1.0 / m + 1.0 / n) ** 0.5
        baseline_kwh = bl.total_consumption_kwh or 1.0
        return uncertainty_fraction * baseline_kwh

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: EnergySavingsVerificationResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
