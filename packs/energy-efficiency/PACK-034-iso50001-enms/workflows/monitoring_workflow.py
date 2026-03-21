# -*- coding: utf-8 -*-
"""
Monitoring Workflow - ISO 50001 Clause 9.1 M&V
===================================

4-phase workflow for monitoring and measurement of energy performance
within PACK-034 ISO 50001 Energy Management System Pack.

Phases:
    1. MeteringVerification  -- Verify meter calibration, accuracy, hierarchy
    2. DataCollection        -- Collect and validate meter readings, flag issues
    3. Analysis              -- Calculate EnPIs, run CUSUM, compare baselines
    4. Reporting             -- Generate monitoring reports with recommendations

The workflow follows GreenLang zero-hallucination principles: all EnPI
calculations use validated meter data and deterministic formulas, CUSUM
analysis uses standard statistical techniques, and baseline comparisons
are pure arithmetic. SHA-256 provenance hashes guarantee auditability.

Schedule: monthly / quarterly
Estimated duration: 30 minutes

Author: GreenLang Team
Version: 34.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
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


class MonitoringPhase(str, Enum):
    """Phases of the monitoring workflow."""

    METERING_VERIFICATION = "metering_verification"
    DATA_COLLECTION = "data_collection"
    ANALYSIS = "analysis"
    REPORTING = "reporting"


class MeterStatus(str, Enum):
    """Calibration/operational status of a meter."""

    ACTIVE = "active"
    CALIBRATION_DUE = "calibration_due"
    OUT_OF_CALIBRATION = "out_of_calibration"
    FAULT = "fault"
    OFFLINE = "offline"


class CUSUMStatus(str, Enum):
    """CUSUM chart status indicators."""

    IN_CONTROL = "in_control"
    WARNING = "warning"
    OUT_OF_CONTROL = "out_of_control"
    IMPROVING = "improving"
    DETERIORATING = "deteriorating"


class DataQualityLevel(str, Enum):
    """Data quality assessment levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"


# =============================================================================
# MONITORING REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# Acceptable meter accuracy classes per ISO 50006
METER_ACCURACY_STANDARDS: Dict[str, float] = {
    "electricity_kwh": 2.0,   # +/- 2% Class 2
    "gas_m3": 2.5,            # +/- 2.5%
    "steam_kg": 3.0,          # +/- 3%
    "water_m3": 2.0,          # +/- 2%
    "thermal_kwh": 3.0,       # +/- 3%
    "compressed_air_m3": 5.0, # +/- 5%
    "DEFAULT": 3.0,
}

# Data quality scoring thresholds
DQ_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "excellent": {"completeness": 99.0, "accuracy": 98.0},
    "good": {"completeness": 95.0, "accuracy": 95.0},
    "acceptable": {"completeness": 90.0, "accuracy": 90.0},
    "poor": {"completeness": 80.0, "accuracy": 80.0},
}

# CUSUM decision interval (h parameter) in units of sigma
CUSUM_H: float = 5.0
# CUSUM reference value (k parameter) in units of sigma
CUSUM_K: float = 0.5


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


class MeterVerification(BaseModel):
    """Verification result for a single meter."""

    meter_id: str = Field(default="", description="Meter identifier")
    meter_type: str = Field(default="", description="Measurement type")
    status: MeterStatus = Field(default=MeterStatus.ACTIVE)
    accuracy_class_pct: Decimal = Field(default=Decimal("0"), ge=0)
    last_calibration_date: str = Field(default="", description="YYYY-MM-DD")
    next_calibration_date: str = Field(default="", description="YYYY-MM-DD")
    is_within_accuracy: bool = Field(default=True)
    notes: str = Field(default="")


class DataQualityAssessment(BaseModel):
    """Data quality assessment for collected data."""

    meter_id: str = Field(default="")
    readings_expected: int = Field(default=0, ge=0)
    readings_received: int = Field(default=0, ge=0)
    readings_valid: int = Field(default=0, ge=0)
    completeness_pct: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    accuracy_pct: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    quality_level: DataQualityLevel = Field(default=DataQualityLevel.GOOD)
    issues: List[str] = Field(default_factory=list)


class EnPIValue(BaseModel):
    """A calculated EnPI value for the monitoring period."""

    enpi_id: str = Field(default_factory=lambda: f"enpi-{uuid.uuid4().hex[:8]}")
    name: str = Field(default="", description="EnPI name")
    category: str = Field(default="", description="Energy end-use category")
    current_value: Decimal = Field(default=Decimal("0"))
    baseline_value: Decimal = Field(default=Decimal("0"))
    change_pct: Decimal = Field(default=Decimal("0"), description="% change vs baseline")
    unit: str = Field(default="kWh/unit")
    trend: str = Field(default="stable", description="improving|stable|deteriorating")


class CUSUMResult(BaseModel):
    """CUSUM analysis result for an EnPI."""

    enpi_id: str = Field(default="")
    cusum_plus: Decimal = Field(default=Decimal("0"), description="Upper CUSUM statistic")
    cusum_minus: Decimal = Field(default=Decimal("0"), description="Lower CUSUM statistic")
    status: CUSUMStatus = Field(default=CUSUMStatus.IN_CONTROL)
    cumulative_savings_kwh: Decimal = Field(default=Decimal("0"))
    periods_analyzed: int = Field(default=0, ge=0)
    shift_detected: bool = Field(default=False)


class AlertRecord(BaseModel):
    """An alert generated during monitoring."""

    alert_id: str = Field(default_factory=lambda: f"alert-{uuid.uuid4().hex[:8]}")
    source: str = Field(default="", description="meter|enpi|cusum|data_quality")
    severity: str = Field(default="info", description="info|warning|critical")
    message: str = Field(default="")
    recommended_action: str = Field(default="")


class MonitoringInput(BaseModel):
    """Input data model for MonitoringWorkflow."""

    enms_id: str = Field(default="", description="EnMS program identifier")
    metering_plan: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metering plan: {meters: [{meter_id, type, accuracy, ...}]}",
    )
    data_collection_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Data collection settings and raw readings",
    )
    analysis_period_start: str = Field(default="", description="Analysis period start YYYY-MM-DD")
    analysis_period_end: str = Field(default="", description="Analysis period end YYYY-MM-DD")
    baseline_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Baseline EnPI values for comparison",
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class MonitoringResult(BaseModel):
    """Complete result from monitoring workflow."""

    monitoring_id: str = Field(..., description="Unique monitoring execution ID")
    enms_id: str = Field(default="", description="EnMS program identifier")
    meter_status: List[MeterVerification] = Field(default_factory=list)
    data_quality_score: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    data_quality_assessments: List[DataQualityAssessment] = Field(default_factory=list)
    enpi_values: List[EnPIValue] = Field(default_factory=list)
    cusum_status: List[CUSUMResult] = Field(default_factory=list)
    alerts: List[AlertRecord] = Field(default_factory=list)
    report_data: Dict[str, Any] = Field(default_factory=dict)
    phases_completed: List[str] = Field(default_factory=list)
    execution_time_ms: float = Field(default=0.0)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class MonitoringWorkflow:
    """
    4-phase monitoring workflow per ISO 50001 Clause 9.1.

    Performs meter verification, data collection with quality checks,
    EnPI analysis with CUSUM monitoring, and generates monitoring
    reports with charts and recommendations.

    Zero-hallucination: EnPI calculations use validated meter data,
    CUSUM uses standard statistical formulas, and baseline comparisons
    are deterministic. No LLM calls in the numeric computation path.

    Attributes:
        monitoring_id: Unique monitoring execution identifier.
        _meter_verifications: Meter verification results.
        _quality_assessments: Data quality assessments.
        _enpi_values: Calculated EnPI values.
        _cusum_results: CUSUM analysis results.
        _alerts: Generated alerts.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = MonitoringWorkflow()
        >>> inp = MonitoringInput(
        ...     enms_id="enms-001",
        ...     metering_plan={"meters": [{"meter_id": "m-1", "type": "electricity_kwh"}]},
        ...     analysis_period_start="2026-01-01",
        ...     analysis_period_end="2026-03-31",
        ... )
        >>> result = wf.execute(inp)
        >>> assert result.data_quality_score >= 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MonitoringWorkflow."""
        self.monitoring_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._meter_verifications: List[MeterVerification] = []
        self._quality_assessments: List[DataQualityAssessment] = []
        self._enpi_values: List[EnPIValue] = []
        self._cusum_results: List[CUSUMResult] = []
        self._alerts: List[AlertRecord] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def execute(self, input_data: MonitoringInput) -> MonitoringResult:
        """
        Execute the 4-phase monitoring workflow.

        Args:
            input_data: Validated monitoring input.

        Returns:
            MonitoringResult with meter status, EnPIs, CUSUM, and alerts.
        """
        t_start = time.perf_counter()
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting monitoring workflow %s enms=%s period=%s to %s",
            self.monitoring_id, input_data.enms_id,
            input_data.analysis_period_start, input_data.analysis_period_end,
        )

        self._phase_results = []
        self._meter_verifications = []
        self._quality_assessments = []
        self._enpi_values = []
        self._cusum_results = []
        self._alerts = []

        try:
            phase1 = self._phase_metering_verification(input_data)
            self._phase_results.append(phase1)

            phase2 = self._phase_data_collection(input_data)
            self._phase_results.append(phase2)

            phase3 = self._phase_analysis(input_data)
            self._phase_results.append(phase3)

            phase4 = self._phase_reporting(input_data)
            self._phase_results.append(phase4)

        except Exception as exc:
            self.logger.error("Monitoring workflow failed: %s", exc, exc_info=True)
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Calculate overall data quality score
        dq_score = self._calculate_overall_dq_score()

        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        result = MonitoringResult(
            monitoring_id=self.monitoring_id,
            enms_id=input_data.enms_id,
            meter_status=self._meter_verifications,
            data_quality_score=dq_score,
            data_quality_assessments=self._quality_assessments,
            enpi_values=self._enpi_values,
            cusum_status=self._cusum_results,
            alerts=self._alerts,
            report_data=self._build_report_data(input_data),
            phases_completed=completed_phases,
            execution_time_ms=round(elapsed_ms, 2),
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Monitoring workflow %s completed in %.0fms meters=%d enpis=%d alerts=%d dq=%.1f",
            self.monitoring_id, elapsed_ms, len(self._meter_verifications),
            len(self._enpi_values), len(self._alerts), float(dq_score),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Metering Verification
    # -------------------------------------------------------------------------

    def _phase_metering_verification(
        self, input_data: MonitoringInput
    ) -> PhaseResult:
        """Verify meter calibration, accuracy, and hierarchy completeness."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        meters = input_data.metering_plan.get("meters", [])

        for meter_dict in meters:
            meter_id = meter_dict.get("meter_id", f"meter-{uuid.uuid4().hex[:8]}")
            meter_type = meter_dict.get("type", "electricity_kwh")
            accuracy = float(meter_dict.get("accuracy_pct", 0))
            last_cal = meter_dict.get("last_calibration", "")
            next_cal = meter_dict.get("next_calibration", "")
            raw_status = meter_dict.get("status", "active")

            # Check accuracy against standard
            standard_accuracy = METER_ACCURACY_STANDARDS.get(
                meter_type, METER_ACCURACY_STANDARDS["DEFAULT"]
            )
            is_within = accuracy <= standard_accuracy if accuracy > 0 else True

            # Determine meter status
            try:
                status = MeterStatus(raw_status)
            except ValueError:
                status = MeterStatus.ACTIVE
                warnings.append(f"Meter {meter_id}: unknown status '{raw_status}'")

            # Check if calibration is overdue
            if status == MeterStatus.CALIBRATION_DUE:
                self._alerts.append(AlertRecord(
                    source="meter",
                    severity="warning",
                    message=f"Meter {meter_id} calibration is due",
                    recommended_action=f"Schedule calibration for meter {meter_id}",
                ))

            if not is_within and accuracy > 0:
                self._alerts.append(AlertRecord(
                    source="meter",
                    severity="critical",
                    message=(
                        f"Meter {meter_id} accuracy ({accuracy}%) exceeds "
                        f"standard ({standard_accuracy}%)"
                    ),
                    recommended_action=f"Recalibrate or replace meter {meter_id}",
                ))

            verification = MeterVerification(
                meter_id=meter_id,
                meter_type=meter_type,
                status=status,
                accuracy_class_pct=Decimal(str(round(accuracy, 2))),
                last_calibration_date=last_cal,
                next_calibration_date=next_cal,
                is_within_accuracy=is_within,
            )
            self._meter_verifications.append(verification)

        # Check hierarchy completeness
        total_meters = len(self._meter_verifications)
        active_meters = sum(
            1 for m in self._meter_verifications if m.status == MeterStatus.ACTIVE
        )

        if total_meters == 0:
            warnings.append("No meters defined in metering plan")

        outputs["total_meters"] = total_meters
        outputs["active_meters"] = active_meters
        outputs["within_accuracy"] = sum(1 for m in self._meter_verifications if m.is_within_accuracy)
        outputs["calibration_alerts"] = sum(
            1 for m in self._meter_verifications if m.status == MeterStatus.CALIBRATION_DUE
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 MeteringVerification: %d meters, %d active, %d within accuracy",
            total_meters, active_meters, outputs["within_accuracy"],
        )
        return PhaseResult(
            phase_name=MonitoringPhase.METERING_VERIFICATION.value, phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Data Collection
    # -------------------------------------------------------------------------

    def _phase_data_collection(
        self, input_data: MonitoringInput
    ) -> PhaseResult:
        """Collect and validate meter readings, flag data quality issues."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        readings = input_data.data_collection_config.get("readings", {})

        for meter_id, meter_readings in readings.items():
            if not isinstance(meter_readings, list):
                warnings.append(f"Meter {meter_id}: readings not in list format")
                continue

            expected = len(meter_readings)
            valid_count = 0
            issues: List[str] = []

            for reading in meter_readings:
                value = reading.get("value")
                if value is None:
                    issues.append("missing_value")
                    continue
                try:
                    val = float(value)
                    if val < 0:
                        issues.append("negative_value")
                        continue
                    valid_count += 1
                except (ValueError, TypeError):
                    issues.append("invalid_format")

            completeness = round(valid_count / max(expected, 1) * 100.0, 1)
            accuracy = round(valid_count / max(expected, 1) * 100.0, 1)

            # Classify quality level
            quality_level = DataQualityLevel.UNUSABLE
            for level_name in ["excellent", "good", "acceptable", "poor"]:
                thresholds = DQ_THRESHOLDS[level_name]
                if completeness >= thresholds["completeness"] and accuracy >= thresholds["accuracy"]:
                    quality_level = DataQualityLevel(level_name)
                    break

            assessment = DataQualityAssessment(
                meter_id=meter_id,
                readings_expected=expected,
                readings_received=expected,
                readings_valid=valid_count,
                completeness_pct=Decimal(str(completeness)),
                accuracy_pct=Decimal(str(accuracy)),
                quality_level=quality_level,
                issues=list(set(issues)),
            )
            self._quality_assessments.append(assessment)

            if quality_level in (DataQualityLevel.POOR, DataQualityLevel.UNUSABLE):
                self._alerts.append(AlertRecord(
                    source="data_quality",
                    severity="warning",
                    message=f"Meter {meter_id}: data quality is {quality_level.value}",
                    recommended_action=f"Investigate data issues for meter {meter_id}",
                ))

        outputs["meters_assessed"] = len(self._quality_assessments)
        outputs["quality_distribution"] = {
            level.value: sum(
                1 for a in self._quality_assessments if a.quality_level == level
            )
            for level in DataQualityLevel
        }

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 DataCollection: %d meters assessed",
            len(self._quality_assessments),
        )
        return PhaseResult(
            phase_name=MonitoringPhase.DATA_COLLECTION.value, phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Analysis
    # -------------------------------------------------------------------------

    def _phase_analysis(
        self, input_data: MonitoringInput
    ) -> PhaseResult:
        """Calculate EnPIs, run CUSUM, compare against baselines."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Calculate EnPIs from collected data
        enpi_definitions = input_data.baseline_data.get("enpis", [])
        readings = input_data.data_collection_config.get("readings", {})

        for enpi_def in enpi_definitions:
            enpi_name = enpi_def.get("name", "")
            enpi_category = enpi_def.get("category", "")
            baseline_val = float(enpi_def.get("baseline_value", 0))
            meter_id = enpi_def.get("meter_id", "")
            unit = enpi_def.get("unit", "kWh/unit")

            # Sum valid readings for this meter
            meter_data = readings.get(meter_id, [])
            current_total = 0.0
            for reading in meter_data:
                try:
                    current_total += float(reading.get("value", 0))
                except (ValueError, TypeError):
                    continue

            # Calculate change
            change_pct = Decimal("0")
            if baseline_val > 0:
                change_pct = Decimal(str(round(
                    (current_total - baseline_val) / baseline_val * 100.0, 2
                )))

            # Determine trend
            trend = "stable"
            if float(change_pct) < -2.0:
                trend = "improving"
            elif float(change_pct) > 2.0:
                trend = "deteriorating"

            enpi = EnPIValue(
                name=enpi_name,
                category=enpi_category,
                current_value=Decimal(str(round(current_total, 2))),
                baseline_value=Decimal(str(round(baseline_val, 2))),
                change_pct=change_pct,
                unit=unit,
                trend=trend,
            )
            self._enpi_values.append(enpi)

            # CUSUM analysis
            cusum_data = enpi_def.get("cusum_history", [])
            cusum_result = self._run_cusum_analysis(
                enpi.enpi_id, current_total, baseline_val, cusum_data,
            )
            self._cusum_results.append(cusum_result)

            # Generate alerts for deteriorating EnPIs
            if trend == "deteriorating" and abs(float(change_pct)) > 5.0:
                self._alerts.append(AlertRecord(
                    source="enpi",
                    severity="warning",
                    message=f"EnPI '{enpi_name}' deteriorated by {change_pct}% vs baseline",
                    recommended_action=f"Investigate {enpi_category} performance decline",
                ))

        outputs["enpis_calculated"] = len(self._enpi_values)
        outputs["cusum_analyses"] = len(self._cusum_results)
        outputs["improving_enpis"] = sum(1 for e in self._enpi_values if e.trend == "improving")
        outputs["deteriorating_enpis"] = sum(1 for e in self._enpi_values if e.trend == "deteriorating")

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 Analysis: %d EnPIs, %d improving, %d deteriorating",
            len(self._enpi_values), outputs["improving_enpis"],
            outputs["deteriorating_enpis"],
        )
        return PhaseResult(
            phase_name=MonitoringPhase.ANALYSIS.value, phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _run_cusum_analysis(
        self, enpi_id: str, current_value: float,
        baseline_value: float, history: List[float]
    ) -> CUSUMResult:
        """Run CUSUM analysis for shift detection."""
        if baseline_value <= 0 or not history:
            return CUSUMResult(
                enpi_id=enpi_id,
                status=CUSUMStatus.IN_CONTROL,
                periods_analyzed=0,
            )

        # Calculate standard deviation from history
        n = len(history)
        mean_h = sum(history) / n
        if n > 1:
            std_dev = math.sqrt(sum((x - mean_h) ** 2 for x in history) / (n - 1))
        else:
            std_dev = abs(mean_h * 0.1)  # Fallback: 10% of mean

        if std_dev <= 0:
            std_dev = 1.0

        # Calculate CUSUM statistics
        k = CUSUM_K * std_dev
        h = CUSUM_H * std_dev

        cusum_plus = Decimal("0")
        cusum_minus = Decimal("0")
        shift_detected = False

        all_values = history + [current_value]
        for val in all_values:
            deviation = val - baseline_value
            cp = max(0.0, float(cusum_plus) + deviation - k)
            cm = max(0.0, float(cusum_minus) - deviation - k)
            cusum_plus = Decimal(str(round(cp, 4)))
            cusum_minus = Decimal(str(round(cm, 4)))

            if cp > h or cm > h:
                shift_detected = True

        # Determine status
        cumulative_savings = Decimal(str(round(baseline_value - current_value, 2)))
        if shift_detected:
            if current_value < baseline_value:
                status = CUSUMStatus.IMPROVING
            else:
                status = CUSUMStatus.OUT_OF_CONTROL
        elif float(cusum_plus) > h * 0.7 or float(cusum_minus) > h * 0.7:
            status = CUSUMStatus.WARNING
        else:
            status = CUSUMStatus.IN_CONTROL

        return CUSUMResult(
            enpi_id=enpi_id,
            cusum_plus=cusum_plus,
            cusum_minus=cusum_minus,
            status=status,
            cumulative_savings_kwh=cumulative_savings,
            periods_analyzed=len(all_values),
            shift_detected=shift_detected,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Reporting
    # -------------------------------------------------------------------------

    def _phase_reporting(
        self, input_data: MonitoringInput
    ) -> PhaseResult:
        """Generate monitoring reports with charts and recommendations."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Build recommendations
        recommendations: List[str] = []

        for enpi in self._enpi_values:
            if enpi.trend == "deteriorating":
                recommendations.append(
                    f"Investigate {enpi.category} performance: {enpi.name} "
                    f"deteriorated {enpi.change_pct}% vs baseline"
                )

        for cusum in self._cusum_results:
            if cusum.status == CUSUMStatus.OUT_OF_CONTROL:
                recommendations.append(
                    f"CUSUM alert: EnPI {cusum.enpi_id} shows out-of-control pattern; "
                    f"root cause analysis required"
                )

        for meter in self._meter_verifications:
            if meter.status == MeterStatus.OUT_OF_CALIBRATION:
                recommendations.append(
                    f"Meter {meter.meter_id} is out of calibration; "
                    f"schedule immediate recalibration"
                )

        if not recommendations:
            recommendations.append("All EnPIs within expected ranges; continue monitoring")

        outputs["recommendations_count"] = len(recommendations)
        outputs["recommendations"] = recommendations[:10]
        outputs["total_alerts"] = len(self._alerts)
        outputs["report_period"] = (
            f"{input_data.analysis_period_start} to {input_data.analysis_period_end}"
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 4 Reporting: %d recommendations, %d alerts",
            len(recommendations), len(self._alerts),
        )
        return PhaseResult(
            phase_name=MonitoringPhase.REPORTING.value, phase_number=4,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Supporting Methods
    # -------------------------------------------------------------------------

    def _calculate_overall_dq_score(self) -> Decimal:
        """Calculate overall data quality score from assessments."""
        if not self._quality_assessments:
            return Decimal("0")

        total_completeness = sum(
            float(a.completeness_pct) for a in self._quality_assessments
        )
        avg_completeness = total_completeness / len(self._quality_assessments)
        return Decimal(str(round(avg_completeness, 1)))

    def _build_report_data(self, input_data: MonitoringInput) -> Dict[str, Any]:
        """Build report data structure for output."""
        return {
            "period": f"{input_data.analysis_period_start} to {input_data.analysis_period_end}",
            "meters_verified": len(self._meter_verifications),
            "enpis_tracked": len(self._enpi_values),
            "alerts_generated": len(self._alerts),
            "cusum_analyses": len(self._cusum_results),
            "enpi_summary": [
                {
                    "name": e.name,
                    "current": str(e.current_value),
                    "baseline": str(e.baseline_value),
                    "change_pct": str(e.change_pct),
                    "trend": e.trend,
                }
                for e in self._enpi_values
            ],
        }

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: MonitoringResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
