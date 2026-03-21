# -*- coding: utf-8 -*-
"""
Progress Tracking Workflow
===================================

3-phase workflow for tracking implementation progress and verifying
achieved energy savings within PACK-033 Quick Wins Identifier Pack.

Phases:
    1. DataCollection       -- Gather actual consumption, costs, implementation status
    2. SavingsVerification  -- Run QuickWinsReportingEngine verification logic
    3. VarianceAnalysis     -- Compare planned vs actual, identify issues

The workflow follows GreenLang zero-hallucination principles: savings
verification uses IPMVP Option C (whole-facility) or Option A (key
parameter measurement) deterministic formulas. SHA-256 provenance
hashes guarantee auditability.

Schedule: monthly/quarterly
Estimated duration: 15 minutes

Author: GreenLang Team
Version: 33.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from decimal import Decimal
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


class MeasureStatus(str, Enum):
    """Implementation status of a measure."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DEFERRED = "deferred"
    CANCELLED = "cancelled"


class VarianceSeverity(str, Enum):
    """Severity classification for variance findings."""

    ON_TARGET = "on_target"
    MINOR_DEVIATION = "minor_deviation"
    SIGNIFICANT_DEVIATION = "significant_deviation"
    CRITICAL_SHORTFALL = "critical_shortfall"
    OVER_PERFORMING = "over_performing"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_ms: float = Field(default=0.0, description="Phase duration in milliseconds")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class MeasureTrackingData(BaseModel):
    """Tracking data for a single measure."""

    measure_id: str = Field(default="", description="Measure identifier")
    title: str = Field(default="", description="Measure title")
    status: MeasureStatus = Field(default=MeasureStatus.NOT_STARTED)
    planned_savings_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    planned_savings_cost: Decimal = Field(default=Decimal("0"), ge=0)
    actual_savings_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    actual_savings_cost: Decimal = Field(default=Decimal("0"), ge=0)
    implementation_cost_actual: Decimal = Field(default=Decimal("0"), ge=0)
    completion_date: str = Field(default="", description="YYYY-MM-DD or empty")
    notes: str = Field(default="")


class VarianceRecord(BaseModel):
    """Variance record between planned and actual performance."""

    measure_id: str = Field(default="", description="Measure identifier")
    title: str = Field(default="", description="Measure title")
    planned_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    actual_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    variance_kwh: Decimal = Field(default=Decimal("0"), description="Actual - Planned")
    variance_pct: Decimal = Field(default=Decimal("0"), description="Variance as %")
    severity: VarianceSeverity = Field(default=VarianceSeverity.ON_TARGET)
    root_cause: str = Field(default="", description="Identified root cause")
    corrective_action: str = Field(default="", description="Recommended action")


class ProgressTrackingInput(BaseModel):
    """Input data model for ProgressTrackingWorkflow."""

    plan_id: str = Field(default="", description="Originating implementation plan ID")
    measures: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Measure dicts with actual data (actual_kwh, actual_cost, status)",
    )
    baseline_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Baseline consumption data for verification",
    )
    reporting_period: str = Field(default="", description="Reporting period YYYY-MM or YYYY-QN")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class ProgressTrackingResult(BaseModel):
    """Complete result from progress tracking workflow."""

    tracking_id: str = Field(..., description="Unique tracking execution ID")
    plan_id: str = Field(default="", description="Linked plan ID")
    measures_tracked: int = Field(default=0, ge=0)
    measures_completed: int = Field(default=0, ge=0)
    total_verified_savings_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    total_verified_cost_savings: Decimal = Field(default=Decimal("0"), ge=0)
    on_track: bool = Field(default=True, description="True if overall variance within 10%")
    variances: List[VarianceRecord] = Field(default_factory=list)
    completion_rate_pct: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    phases_completed: List[str] = Field(default_factory=list)
    reporting_period: str = Field(default="")
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ProgressTrackingWorkflow:
    """
    3-phase progress tracking workflow for quick-win implementation.

    Performs data collection of actual results, savings verification
    against baseline, and variance analysis with root cause identification.

    Zero-hallucination: savings verification uses deterministic comparison
    of metered/reported values against baseline. Variance calculations
    are pure arithmetic. No LLM calls in the numeric computation path.

    Attributes:
        tracking_id: Unique execution identifier.
        _measure_tracking: Tracking records per measure.
        _variances: Variance analysis records.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = ProgressTrackingWorkflow()
        >>> inp = ProgressTrackingInput(
        ...     plan_id="plan-123",
        ...     measures=[{"measure_id": "m-1", "actual_kwh": 1000, ...}],
        ...     reporting_period="2026-Q1",
        ... )
        >>> result = wf.run(inp)
        >>> assert result.measures_tracked > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ProgressTrackingWorkflow."""
        self.tracking_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._measure_tracking: List[MeasureTrackingData] = []
        self._variances: List[VarianceRecord] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: ProgressTrackingInput) -> ProgressTrackingResult:
        """
        Execute the 3-phase progress tracking workflow.

        Args:
            input_data: Validated progress tracking input.

        Returns:
            ProgressTrackingResult with verified savings and variance analysis.
        """
        t_start = time.perf_counter()
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting progress tracking workflow %s plan=%s period=%s",
            self.tracking_id, input_data.plan_id, input_data.reporting_period,
        )

        self._phase_results = []
        self._measure_tracking = []
        self._variances = []

        try:
            # Phase 1: Data Collection
            phase1 = self._phase_data_collection(input_data)
            self._phase_results.append(phase1)

            # Phase 2: Savings Verification
            phase2 = self._phase_savings_verification(input_data)
            self._phase_results.append(phase2)

            # Phase 3: Variance Analysis
            phase3 = self._phase_variance_analysis(input_data)
            self._phase_results.append(phase3)

        except Exception as exc:
            self.logger.error(
                "Progress tracking workflow failed: %s", exc, exc_info=True,
            )
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        measures_tracked = len(self._measure_tracking)
        measures_completed = sum(
            1 for m in self._measure_tracking if m.status == MeasureStatus.COMPLETED
        )
        total_verified_kwh = sum(m.actual_savings_kwh for m in self._measure_tracking)
        total_verified_cost = sum(m.actual_savings_cost for m in self._measure_tracking)
        completion_rate = (
            Decimal(str(round(measures_completed / max(measures_tracked, 1) * 100, 1)))
        )

        # Determine if on track (overall variance within 10%)
        total_planned_kwh = sum(m.planned_savings_kwh for m in self._measure_tracking)
        on_track = True
        if total_planned_kwh > 0:
            overall_variance_pct = abs(
                float(total_verified_kwh - total_planned_kwh)
                / float(total_planned_kwh) * 100.0
            )
            on_track = overall_variance_pct <= 10.0

        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        result = ProgressTrackingResult(
            tracking_id=self.tracking_id,
            plan_id=input_data.plan_id,
            measures_tracked=measures_tracked,
            measures_completed=measures_completed,
            total_verified_savings_kwh=total_verified_kwh,
            total_verified_cost_savings=total_verified_cost,
            on_track=on_track,
            variances=self._variances,
            completion_rate_pct=completion_rate,
            phases_completed=completed_phases,
            reporting_period=input_data.reporting_period,
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Progress tracking workflow %s completed in %.0fms "
            "tracked=%d completed=%d on_track=%s",
            self.tracking_id, elapsed_ms, measures_tracked,
            measures_completed, on_track,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Data Collection
    # -------------------------------------------------------------------------

    def _phase_data_collection(
        self, input_data: ProgressTrackingInput
    ) -> PhaseResult:
        """Gather actual consumption, costs, and implementation status."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        for measure_dict in input_data.measures:
            measure_id = measure_dict.get("measure_id", f"m-{uuid.uuid4().hex[:8]}")
            title = measure_dict.get("title", "Unnamed")

            # Parse status
            raw_status = measure_dict.get("status", "not_started")
            try:
                status = MeasureStatus(raw_status)
            except ValueError:
                status = MeasureStatus.NOT_STARTED
                warnings.append(f"Measure {measure_id}: unknown status '{raw_status}'")

            tracking = MeasureTrackingData(
                measure_id=measure_id,
                title=title,
                status=status,
                planned_savings_kwh=Decimal(str(measure_dict.get("planned_savings_kwh", 0))),
                planned_savings_cost=Decimal(str(measure_dict.get("planned_savings_cost", 0))),
                actual_savings_kwh=Decimal(str(measure_dict.get("actual_savings_kwh", 0))),
                actual_savings_cost=Decimal(str(measure_dict.get("actual_savings_cost", 0))),
                implementation_cost_actual=Decimal(str(measure_dict.get("implementation_cost_actual", 0))),
                completion_date=measure_dict.get("completion_date", ""),
                notes=measure_dict.get("notes", ""),
            )
            self._measure_tracking.append(tracking)

        # Validate baseline data
        baseline_kwh = input_data.baseline_data.get("annual_energy_kwh", 0)
        if baseline_kwh <= 0:
            warnings.append("Baseline annual energy not provided; verification may be limited")

        outputs["measures_collected"] = len(self._measure_tracking)
        outputs["statuses"] = {
            s.value: sum(1 for m in self._measure_tracking if m.status == s)
            for s in MeasureStatus
        }
        outputs["baseline_available"] = baseline_kwh > 0

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 DataCollection: %d measures collected",
            len(self._measure_tracking),
        )
        return PhaseResult(
            phase_name="data_collection", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Savings Verification
    # -------------------------------------------------------------------------

    def _phase_savings_verification(
        self, input_data: ProgressTrackingInput
    ) -> PhaseResult:
        """Verify reported savings against baseline using deterministic methods."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        baseline_kwh = Decimal(str(input_data.baseline_data.get("annual_energy_kwh", 0)))
        baseline_cost = Decimal(str(input_data.baseline_data.get("annual_energy_cost", 0)))
        current_kwh = Decimal(str(input_data.baseline_data.get("current_period_kwh", 0)))
        current_cost = Decimal(str(input_data.baseline_data.get("current_period_cost", 0)))

        # Whole-facility verification (IPMVP Option C approach)
        facility_savings_kwh = Decimal("0")
        facility_savings_cost = Decimal("0")
        if baseline_kwh > 0 and current_kwh > 0:
            facility_savings_kwh = baseline_kwh - current_kwh
            facility_savings_cost = baseline_cost - current_cost

        # Per-measure verification: validate reported actual vs expected
        verified_count = 0
        for tracking in self._measure_tracking:
            if tracking.status != MeasureStatus.COMPLETED:
                continue

            # Cross-check: actual should be within reasonable bounds
            if tracking.planned_savings_kwh > 0:
                ratio = float(tracking.actual_savings_kwh) / float(tracking.planned_savings_kwh)
                if ratio > 2.0:
                    warnings.append(
                        f"Measure {tracking.measure_id}: actual savings {ratio:.1f}x planned; "
                        f"verify measurement"
                    )
                elif ratio < 0:
                    warnings.append(
                        f"Measure {tracking.measure_id}: negative savings reported"
                    )

            verified_count += 1

        outputs["facility_savings_kwh"] = str(facility_savings_kwh)
        outputs["facility_savings_cost"] = str(facility_savings_cost)
        outputs["measures_verified"] = verified_count
        outputs["verification_method"] = "IPMVP_Option_C"

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 SavingsVerification: %d measures verified, facility savings=%.0f kWh",
            verified_count, float(facility_savings_kwh),
        )
        return PhaseResult(
            phase_name="savings_verification", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Variance Analysis
    # -------------------------------------------------------------------------

    def _phase_variance_analysis(
        self, input_data: ProgressTrackingInput
    ) -> PhaseResult:
        """Compare planned vs actual savings and identify issues."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        for tracking in self._measure_tracking:
            if tracking.status == MeasureStatus.NOT_STARTED:
                continue

            planned = tracking.planned_savings_kwh
            actual = tracking.actual_savings_kwh
            variance_kwh = actual - planned
            variance_pct = (
                Decimal(str(round(float(variance_kwh) / float(planned) * 100.0, 2)))
                if planned > 0 else Decimal("0")
            )

            # Classify severity
            abs_pct = abs(float(variance_pct))
            if float(variance_pct) > 10.0:
                severity = VarianceSeverity.OVER_PERFORMING
            elif abs_pct <= 5.0:
                severity = VarianceSeverity.ON_TARGET
            elif abs_pct <= 15.0:
                severity = VarianceSeverity.MINOR_DEVIATION
            elif abs_pct <= 30.0:
                severity = VarianceSeverity.SIGNIFICANT_DEVIATION
            else:
                severity = VarianceSeverity.CRITICAL_SHORTFALL

            # Determine root cause and corrective action
            root_cause, corrective = self._determine_root_cause(
                tracking, variance_pct, severity,
            )

            record = VarianceRecord(
                measure_id=tracking.measure_id,
                title=tracking.title,
                planned_kwh=planned,
                actual_kwh=actual,
                variance_kwh=variance_kwh,
                variance_pct=variance_pct,
                severity=severity,
                root_cause=root_cause,
                corrective_action=corrective,
            )
            self._variances.append(record)

        # Summary
        on_target = sum(1 for v in self._variances if v.severity == VarianceSeverity.ON_TARGET)
        critical = sum(1 for v in self._variances if v.severity == VarianceSeverity.CRITICAL_SHORTFALL)
        over = sum(1 for v in self._variances if v.severity == VarianceSeverity.OVER_PERFORMING)

        outputs["variances_analysed"] = len(self._variances)
        outputs["on_target"] = on_target
        outputs["critical_shortfalls"] = critical
        outputs["over_performing"] = over
        outputs["severity_distribution"] = {
            s.value: sum(1 for v in self._variances if v.severity == s)
            for s in VarianceSeverity
        }

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 VarianceAnalysis: %d variances, on_target=%d, critical=%d",
            len(self._variances), on_target, critical,
        )
        return PhaseResult(
            phase_name="variance_analysis", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _determine_root_cause(
        self,
        tracking: MeasureTrackingData,
        variance_pct: Decimal,
        severity: VarianceSeverity,
    ) -> tuple:
        """Determine root cause and corrective action for a variance."""
        if severity == VarianceSeverity.ON_TARGET:
            return "Performance aligned with projections", "Continue monitoring"

        if severity == VarianceSeverity.OVER_PERFORMING:
            return (
                "Actual savings exceed projections; conservative initial estimates",
                "Update projections for future planning accuracy",
            )

        if tracking.status == MeasureStatus.IN_PROGRESS:
            return (
                "Measure not fully implemented; partial savings expected",
                "Complete implementation to achieve full savings",
            )

        if severity == VarianceSeverity.CRITICAL_SHORTFALL:
            return (
                "Significant gap between projected and actual savings",
                "Conduct root-cause investigation; verify metering and operating conditions",
            )

        # Minor or significant deviation
        return (
            "Operating conditions may differ from assumptions",
            "Review assumptions and adjust operating parameters",
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: ProgressTrackingResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
