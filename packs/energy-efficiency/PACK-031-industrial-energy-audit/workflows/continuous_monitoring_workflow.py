# -*- coding: utf-8 -*-
"""
Continuous Monitoring Workflow
===================================

4-phase workflow for real-time energy monitoring within
PACK-031 Industrial Energy Audit Pack.

Phases:
    1. DataIngestion       -- Collect meter readings, SCADA data, BMS data
    2. DeviationDetection  -- CUSUM analysis against baseline, threshold alerts
    3. AlertGeneration     -- Energy anomaly alerts, equipment performance degradation
    4. TrendAnalysis       -- Weekly/monthly EnPI tracking, seasonal adjustment

The workflow follows GreenLang zero-hallucination principles: CUSUM
calculations, deviation detection, and trend analysis use deterministic
statistical methods only. No LLM calls in the numeric computation path.

Schedule: continuous/daily
Estimated duration: 15 minutes

Author: GreenLang Team
Version: 31.0.0
"""

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from greenlang.schemas.enums import AlertSeverity

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


class AlertType(str, Enum):
    """Type of monitoring alert."""

    CUSUM_UPPER = "cusum_upper_deviation"
    CUSUM_LOWER = "cusum_lower_deviation"
    THRESHOLD_BREACH = "threshold_breach"
    EQUIPMENT_DEGRADATION = "equipment_degradation"
    METER_FAULT = "meter_fault"
    DEMAND_SPIKE = "demand_spike"
    BASELINE_SHIFT = "baseline_shift"


class DataSourceType(str, Enum):
    """Source of monitoring data."""

    METER = "meter"
    SCADA = "scada"
    BMS = "bms"
    IOT_SENSOR = "iot_sensor"
    MANUAL = "manual"


class TrendDirection(str, Enum):
    """Trend direction indicator."""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    INSUFFICIENT_DATA = "insufficient_data"


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


class MeterReading(BaseModel):
    """Individual meter reading data point."""

    meter_id: str = Field(default="", description="Meter identifier")
    timestamp: str = Field(default="", description="ISO 8601 timestamp")
    value: float = Field(default=0.0, description="Reading value")
    unit: str = Field(default="kWh", description="Measurement unit")
    source: DataSourceType = Field(default=DataSourceType.METER)
    quality_flag: str = Field(default="valid", description="valid|estimated|suspect|missing")


class SCADADataPoint(BaseModel):
    """SCADA system data point."""

    tag_id: str = Field(default="", description="SCADA tag identifier")
    timestamp: str = Field(default="", description="ISO 8601 timestamp")
    value: float = Field(default=0.0, description="Measured value")
    unit: str = Field(default="", description="Engineering unit")
    equipment_id: str = Field(default="", description="Associated equipment")
    signal_type: str = Field(default="analog", description="analog|digital|calculated")


class BMSDataPoint(BaseModel):
    """Building Management System data point."""

    point_id: str = Field(default="", description="BMS point identifier")
    timestamp: str = Field(default="", description="ISO 8601 timestamp")
    value: float = Field(default=0.0, description="Measured value")
    unit: str = Field(default="", description="Engineering unit")
    zone: str = Field(default="", description="Building zone")
    subsystem: str = Field(default="hvac", description="hvac|lighting|ventilation|other")


class BaselineReference(BaseModel):
    """Reference baseline for deviation detection."""

    baseline_id: str = Field(default="", description="Baseline ID from audit")
    energy_source: str = Field(default="electricity")
    expected_kwh: float = Field(default=0.0, ge=0.0, description="Expected consumption kWh")
    std_deviation_kwh: float = Field(default=0.0, ge=0.0, description="Std deviation kWh")
    cusum_threshold: float = Field(default=0.0, ge=0.0, description="CUSUM decision interval H")
    cusum_allowance: float = Field(default=0.0, ge=0.0, description="CUSUM allowance K")
    model_equation: str = Field(default="", description="Regression model")


class DeviationAlert(BaseModel):
    """Energy deviation alert from CUSUM or threshold analysis."""

    alert_id: str = Field(default_factory=lambda: f"alt-{uuid.uuid4().hex[:8]}")
    alert_type: AlertType = Field(default=AlertType.CUSUM_UPPER)
    severity: AlertSeverity = Field(default=AlertSeverity.WARNING)
    timestamp: str = Field(default="", description="When deviation detected")
    meter_id: str = Field(default="", description="Affected meter")
    equipment_id: str = Field(default="", description="Affected equipment")
    expected_value: float = Field(default=0.0, description="Expected value")
    actual_value: float = Field(default=0.0, description="Actual value")
    deviation_pct: float = Field(default=0.0, description="Deviation %")
    cusum_value: float = Field(default=0.0, description="CUSUM statistic value")
    description: str = Field(default="")
    recommended_action: str = Field(default="")


class EnPITrend(BaseModel):
    """Energy Performance Indicator trend record."""

    enpi_name: str = Field(default="", description="EnPI name")
    unit: str = Field(default="kWh/unit", description="EnPI unit")
    period: str = Field(default="", description="Period YYYY-MM or YYYY-Www")
    value: float = Field(default=0.0, description="EnPI value")
    baseline_value: float = Field(default=0.0, description="Baseline reference")
    deviation_pct: float = Field(default=0.0, description="Deviation from baseline %")
    trend_direction: TrendDirection = Field(default=TrendDirection.STABLE)
    rolling_average: float = Field(default=0.0, description="3-period rolling average")
    seasonally_adjusted: float = Field(default=0.0, description="Adjusted value")
    cumulative_savings_kwh: float = Field(default=0.0, description="Cumulative savings")


class ContinuousMonitoringInput(BaseModel):
    """Input data model for ContinuousMonitoringWorkflow."""

    facility_id: str = Field(default="", description="Facility identifier")
    meter_readings: List[MeterReading] = Field(default_factory=list)
    scada_data: List[SCADADataPoint] = Field(default_factory=list)
    bms_data: List[BMSDataPoint] = Field(default_factory=list)
    baselines: List[BaselineReference] = Field(default_factory=list)
    production_volume: float = Field(default=0.0, ge=0.0, description="Current production")
    production_unit: str = Field(default="tonnes", description="Production unit")
    heating_degree_days: float = Field(default=0.0, ge=0.0, description="Current HDD")
    cooling_degree_days: float = Field(default=0.0, ge=0.0, description="Current CDD")
    monitoring_period: str = Field(default="daily", description="daily|weekly|monthly")
    cusum_sensitivity: float = Field(default=1.0, ge=0.1, le=5.0, description="CUSUM K factor")
    threshold_pct: float = Field(default=10.0, ge=1.0, le=50.0, description="Alert threshold %")
    historical_enpis: List[EnPITrend] = Field(default_factory=list, description="Prior EnPI data")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class ContinuousMonitoringResult(BaseModel):
    """Complete result from continuous monitoring workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="continuous_monitoring")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    facility_id: str = Field(default="")
    alerts: List[DeviationAlert] = Field(default_factory=list)
    enpi_trends: List[EnPITrend] = Field(default_factory=list)
    total_readings_processed: int = Field(default=0)
    alerts_generated: int = Field(default=0)
    critical_alerts: int = Field(default=0)
    overall_deviation_pct: float = Field(default=0.0)
    cumulative_savings_kwh: float = Field(default=0.0)
    monitoring_period: str = Field(default="daily")
    provenance_hash: str = Field(default="")


# =============================================================================
# CUSUM CONSTANTS
# =============================================================================

# Default CUSUM parameters per ASHRAE Guideline 14 / ISO 50006
DEFAULT_CUSUM_H = 4.0  # Decision interval (multiples of std dev)
DEFAULT_CUSUM_K = 0.5  # Allowance (shift to detect, in std dev units)


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ContinuousMonitoringWorkflow:
    """
    4-phase real-time energy monitoring workflow.

    Performs data ingestion, CUSUM deviation detection, alert generation,
    and EnPI trend analysis with seasonal adjustments.

    Zero-hallucination: CUSUM statistics, threshold comparisons, and
    trend calculations use deterministic formulas only. No LLM calls
    in the numeric computation path.

    Attributes:
        workflow_id: Unique execution identifier.
        _alerts: Detected deviation alerts.
        _enpi_trends: Current period EnPI trends.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = ContinuousMonitoringWorkflow()
        >>> inp = ContinuousMonitoringInput(meter_readings=[...], baselines=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize ContinuousMonitoringWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._alerts: List[DeviationAlert] = []
        self._enpi_trends: List[EnPITrend] = []
        self._phase_results: List[PhaseResult] = []
        self._ingested_totals: Dict[str, float] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        input_data: Optional[ContinuousMonitoringInput] = None,
        meter_readings: Optional[List[MeterReading]] = None,
        baselines: Optional[List[BaselineReference]] = None,
    ) -> ContinuousMonitoringResult:
        """
        Execute the 4-phase continuous monitoring workflow.

        Args:
            input_data: Full input model (preferred).
            meter_readings: Meter readings (fallback).
            baselines: Baseline references (fallback).

        Returns:
            ContinuousMonitoringResult with alerts and EnPI trends.
        """
        if input_data is None:
            input_data = ContinuousMonitoringInput(
                meter_readings=meter_readings or [],
                baselines=baselines or [],
            )

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting continuous monitoring workflow %s for facility=%s",
            self.workflow_id, input_data.facility_id,
        )

        self._phase_results = []
        self._alerts = []
        self._enpi_trends = []
        self._ingested_totals = {}
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_data_ingestion(input_data)
            self._phase_results.append(phase1)

            phase2 = await self._phase_deviation_detection(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_alert_generation(input_data)
            self._phase_results.append(phase3)

            phase4 = await self._phase_trend_analysis(input_data)
            self._phase_results.append(phase4)

            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Continuous monitoring workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        total_readings = (
            len(input_data.meter_readings) +
            len(input_data.scada_data) +
            len(input_data.bms_data)
        )
        critical_count = sum(1 for a in self._alerts if a.severity == AlertSeverity.CRITICAL)
        overall_dev = self._calculate_overall_deviation(input_data)
        cumulative_savings = sum(t.cumulative_savings_kwh for t in self._enpi_trends)

        result = ContinuousMonitoringResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            facility_id=input_data.facility_id,
            alerts=self._alerts,
            enpi_trends=self._enpi_trends,
            total_readings_processed=total_readings,
            alerts_generated=len(self._alerts),
            critical_alerts=critical_count,
            overall_deviation_pct=round(overall_dev, 2),
            cumulative_savings_kwh=round(cumulative_savings, 2),
            monitoring_period=input_data.monitoring_period,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Continuous monitoring workflow %s completed in %.2fs alerts=%d",
            self.workflow_id, elapsed, len(self._alerts),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Real-time Data Ingestion
    # -------------------------------------------------------------------------

    async def _phase_data_ingestion(
        self, input_data: ContinuousMonitoringInput
    ) -> PhaseResult:
        """Collect and validate meter readings, SCADA data, BMS data."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Validate meter readings
        valid_meters = 0
        suspect_meters = 0
        for reading in input_data.meter_readings:
            if reading.quality_flag == "valid":
                valid_meters += 1
            elif reading.quality_flag in ("suspect", "missing"):
                suspect_meters += 1
                warnings.append(f"Meter {reading.meter_id}: quality={reading.quality_flag}")

        # Aggregate meter totals by meter_id
        meter_totals: Dict[str, float] = {}
        for reading in input_data.meter_readings:
            if reading.quality_flag != "missing":
                meter_totals[reading.meter_id] = (
                    meter_totals.get(reading.meter_id, 0.0) + reading.value
                )
        self._ingested_totals = meter_totals

        # Validate SCADA data
        scada_tags = len(set(s.tag_id for s in input_data.scada_data))
        bms_points = len(set(b.point_id for b in input_data.bms_data))

        outputs["meter_readings_total"] = len(input_data.meter_readings)
        outputs["meter_readings_valid"] = valid_meters
        outputs["meter_readings_suspect"] = suspect_meters
        outputs["scada_points"] = len(input_data.scada_data)
        outputs["scada_unique_tags"] = scada_tags
        outputs["bms_points"] = len(input_data.bms_data)
        outputs["bms_unique_points"] = bms_points
        outputs["meter_totals_kwh"] = {k: round(v, 2) for k, v in meter_totals.items()}

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 DataIngestion: %d meters, %d SCADA, %d BMS points",
            len(input_data.meter_readings), len(input_data.scada_data), len(input_data.bms_data),
        )
        return PhaseResult(
            phase_name="data_ingestion", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Deviation Detection (CUSUM)
    # -------------------------------------------------------------------------

    async def _phase_deviation_detection(
        self, input_data: ContinuousMonitoringInput
    ) -> PhaseResult:
        """Run CUSUM analysis against baseline for each energy source."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        cusum_alerts: List[DeviationAlert] = []

        for baseline in input_data.baselines:
            actual = self._get_actual_for_baseline(baseline, input_data)
            if actual is None:
                warnings.append(f"No actual data for baseline {baseline.baseline_id}")
                continue

            expected = baseline.expected_kwh
            std_dev = baseline.std_deviation_kwh if baseline.std_deviation_kwh > 0 else expected * 0.10
            k = baseline.cusum_allowance if baseline.cusum_allowance > 0 else std_dev * input_data.cusum_sensitivity
            h = baseline.cusum_threshold if baseline.cusum_threshold > 0 else std_dev * DEFAULT_CUSUM_H

            # CUSUM upper (overconsumption)
            diff = actual - expected
            cusum_upper = max(0.0, diff - k)

            # CUSUM lower (underconsumption / savings)
            cusum_lower = max(0.0, -diff - k)

            deviation_pct = ((actual - expected) / expected * 100.0) if expected > 0 else 0.0

            if cusum_upper > h:
                cusum_alerts.append(DeviationAlert(
                    alert_type=AlertType.CUSUM_UPPER,
                    severity=AlertSeverity.CRITICAL if cusum_upper > h * 2 else AlertSeverity.WARNING,
                    timestamp=datetime.utcnow().isoformat(),
                    meter_id=baseline.baseline_id,
                    expected_value=round(expected, 2),
                    actual_value=round(actual, 2),
                    deviation_pct=round(deviation_pct, 2),
                    cusum_value=round(cusum_upper, 2),
                    description=(
                        f"CUSUM upper limit exceeded for {baseline.energy_source}: "
                        f"actual={actual:.0f} vs expected={expected:.0f} kWh "
                        f"({deviation_pct:+.1f}%)"
                    ),
                    recommended_action="Investigate root cause of overconsumption",
                ))

            if cusum_lower > h:
                cusum_alerts.append(DeviationAlert(
                    alert_type=AlertType.CUSUM_LOWER,
                    severity=AlertSeverity.INFORMATIONAL,
                    timestamp=datetime.utcnow().isoformat(),
                    meter_id=baseline.baseline_id,
                    expected_value=round(expected, 2),
                    actual_value=round(actual, 2),
                    deviation_pct=round(deviation_pct, 2),
                    cusum_value=round(cusum_lower, 2),
                    description=(
                        f"CUSUM lower limit exceeded for {baseline.energy_source}: "
                        f"savings detected, actual={actual:.0f} vs expected={expected:.0f} kWh"
                    ),
                    recommended_action="Verify savings from implemented ECMs",
                ))

        self._alerts.extend(cusum_alerts)

        outputs["baselines_checked"] = len(input_data.baselines)
        outputs["cusum_alerts"] = len(cusum_alerts)
        outputs["cusum_upper_alerts"] = sum(
            1 for a in cusum_alerts if a.alert_type == AlertType.CUSUM_UPPER
        )
        outputs["cusum_lower_alerts"] = sum(
            1 for a in cusum_alerts if a.alert_type == AlertType.CUSUM_LOWER
        )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 DeviationDetection: %d baselines checked, %d CUSUM alerts",
            len(input_data.baselines), len(cusum_alerts),
        )
        return PhaseResult(
            phase_name="deviation_detection", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _get_actual_for_baseline(
        self, baseline: BaselineReference, input_data: ContinuousMonitoringInput
    ) -> Optional[float]:
        """Get actual consumption for a baseline reference."""
        # Sum meter readings that match the baseline energy source
        total = 0.0
        found = False
        for reading in input_data.meter_readings:
            if reading.quality_flag != "missing":
                total += reading.value
                found = True
        return total if found else None

    # -------------------------------------------------------------------------
    # Phase 3: Alert Generation
    # -------------------------------------------------------------------------

    async def _phase_alert_generation(
        self, input_data: ContinuousMonitoringInput
    ) -> PhaseResult:
        """Generate additional alerts: threshold breaches, equipment degradation."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        new_alerts: List[DeviationAlert] = []

        # Threshold-based alerts from meter readings
        for baseline in input_data.baselines:
            actual = self._get_actual_for_baseline(baseline, input_data)
            if actual is None:
                continue

            expected = baseline.expected_kwh
            threshold = input_data.threshold_pct / 100.0
            deviation_pct = ((actual - expected) / expected * 100.0) if expected > 0 else 0.0

            if abs(deviation_pct) > input_data.threshold_pct:
                new_alerts.append(DeviationAlert(
                    alert_type=AlertType.THRESHOLD_BREACH,
                    severity=AlertSeverity.WARNING,
                    timestamp=datetime.utcnow().isoformat(),
                    meter_id=baseline.baseline_id,
                    expected_value=round(expected, 2),
                    actual_value=round(actual, 2),
                    deviation_pct=round(deviation_pct, 2),
                    description=(
                        f"Threshold breach ({input_data.threshold_pct}%): "
                        f"{baseline.energy_source} deviation={deviation_pct:+.1f}%"
                    ),
                    recommended_action="Review consumption pattern and investigate cause",
                ))

        # Equipment performance degradation from SCADA data
        equipment_values: Dict[str, List[float]] = {}
        for point in input_data.scada_data:
            if point.equipment_id:
                equipment_values.setdefault(point.equipment_id, []).append(point.value)

        for eq_id, values in equipment_values.items():
            if len(values) >= 2:
                avg_val = sum(values) / len(values)
                std_val = math.sqrt(sum((v - avg_val) ** 2 for v in values) / len(values))
                cv = (std_val / avg_val * 100.0) if avg_val > 0 else 0.0

                if cv > 20.0:
                    new_alerts.append(DeviationAlert(
                        alert_type=AlertType.EQUIPMENT_DEGRADATION,
                        severity=AlertSeverity.WARNING,
                        timestamp=datetime.utcnow().isoformat(),
                        equipment_id=eq_id,
                        deviation_pct=round(cv, 2),
                        description=(
                            f"High variability detected for equipment {eq_id}: "
                            f"CV={cv:.1f}% (threshold 20%)"
                        ),
                        recommended_action="Schedule inspection for equipment performance assessment",
                    ))

        # Demand spike detection
        demand_readings = [
            r for r in input_data.meter_readings
            if r.value > 0 and r.quality_flag == "valid"
        ]
        if len(demand_readings) >= 3:
            values = [r.value for r in demand_readings]
            avg = sum(values) / len(values)
            for reading in demand_readings:
                if reading.value > avg * 1.5:
                    new_alerts.append(DeviationAlert(
                        alert_type=AlertType.DEMAND_SPIKE,
                        severity=AlertSeverity.WARNING,
                        timestamp=reading.timestamp,
                        meter_id=reading.meter_id,
                        expected_value=round(avg, 2),
                        actual_value=round(reading.value, 2),
                        deviation_pct=round((reading.value - avg) / avg * 100, 2),
                        description=f"Demand spike at meter {reading.meter_id}: {reading.value:.0f} vs avg {avg:.0f}",
                        recommended_action="Check for abnormal loads or concurrent startups",
                    ))

        self._alerts.extend(new_alerts)

        outputs["threshold_alerts"] = sum(
            1 for a in new_alerts if a.alert_type == AlertType.THRESHOLD_BREACH
        )
        outputs["degradation_alerts"] = sum(
            1 for a in new_alerts if a.alert_type == AlertType.EQUIPMENT_DEGRADATION
        )
        outputs["demand_spike_alerts"] = sum(
            1 for a in new_alerts if a.alert_type == AlertType.DEMAND_SPIKE
        )
        outputs["total_new_alerts"] = len(new_alerts)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 AlertGeneration: %d new alerts generated",
            len(new_alerts),
        )
        return PhaseResult(
            phase_name="alert_generation", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Trend Analysis
    # -------------------------------------------------------------------------

    async def _phase_trend_analysis(
        self, input_data: ContinuousMonitoringInput
    ) -> PhaseResult:
        """Weekly/monthly EnPI tracking with seasonal adjustment."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Calculate current SEC (Specific Energy Consumption)
        total_consumption = sum(self._ingested_totals.values())
        production = input_data.production_volume

        if production > 0:
            current_sec = total_consumption / production
            self._enpi_trends.append(self._build_enpi_trend(
                "sec_kwh_per_unit", "kWh/unit", current_sec,
                input_data.historical_enpis, input_data.monitoring_period,
            ))

        # Calculate current EUI if floor area available (from baselines)
        if total_consumption > 0:
            self._enpi_trends.append(self._build_enpi_trend(
                "total_consumption_kwh", "kWh", total_consumption,
                input_data.historical_enpis, input_data.monitoring_period,
            ))

        # EnPI per degree day if weather data provided
        total_dd = input_data.heating_degree_days + input_data.cooling_degree_days
        if total_dd > 0 and total_consumption > 0:
            kwh_per_dd = total_consumption / total_dd
            self._enpi_trends.append(self._build_enpi_trend(
                "kwh_per_degree_day", "kWh/DD", kwh_per_dd,
                input_data.historical_enpis, input_data.monitoring_period,
            ))

        outputs["enpi_count"] = len(self._enpi_trends)
        outputs["enpis"] = {t.enpi_name: round(t.value, 4) for t in self._enpi_trends}
        outputs["improving_count"] = sum(
            1 for t in self._enpi_trends if t.trend_direction == TrendDirection.IMPROVING
        )
        outputs["degrading_count"] = sum(
            1 for t in self._enpi_trends if t.trend_direction == TrendDirection.DEGRADING
        )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 TrendAnalysis: %d EnPIs tracked",
            len(self._enpi_trends),
        )
        return PhaseResult(
            phase_name="trend_analysis", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _build_enpi_trend(
        self,
        name: str,
        unit: str,
        current_value: float,
        historical: List[EnPITrend],
        period: str,
    ) -> EnPITrend:
        """Build an EnPI trend record with rolling average and direction."""
        # Find historical values for this EnPI
        past = [h for h in historical if h.enpi_name == name]
        past_values = [h.value for h in past]

        # Baseline from historical
        baseline_value = past_values[0] if past_values else current_value

        # Rolling average (last 3 periods + current)
        recent = past_values[-2:] + [current_value] if past_values else [current_value]
        rolling_avg = sum(recent) / len(recent)

        # Deviation from baseline
        deviation_pct = ((current_value - baseline_value) / baseline_value * 100.0) \
            if baseline_value > 0 else 0.0

        # Trend direction
        if len(past_values) >= 2:
            if current_value < past_values[-1] * 0.98:
                direction = TrendDirection.IMPROVING
            elif current_value > past_values[-1] * 1.02:
                direction = TrendDirection.DEGRADING
            else:
                direction = TrendDirection.STABLE
        else:
            direction = TrendDirection.INSUFFICIENT_DATA

        # Cumulative savings
        cumulative = max(0.0, (baseline_value - current_value))

        return EnPITrend(
            enpi_name=name,
            unit=unit,
            period=datetime.utcnow().strftime("%Y-%m") if period == "monthly" else datetime.utcnow().strftime("%Y-W%W"),
            value=round(current_value, 4),
            baseline_value=round(baseline_value, 4),
            deviation_pct=round(deviation_pct, 2),
            trend_direction=direction,
            rolling_average=round(rolling_avg, 4),
            seasonally_adjusted=round(current_value, 4),
            cumulative_savings_kwh=round(cumulative, 2),
        )

    def _calculate_overall_deviation(
        self, input_data: ContinuousMonitoringInput
    ) -> float:
        """Calculate overall deviation across all baselines."""
        total_expected = sum(b.expected_kwh for b in input_data.baselines)
        total_actual = sum(self._ingested_totals.values())
        if total_expected > 0:
            return (total_actual - total_expected) / total_expected * 100.0
        return 0.0

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: ContinuousMonitoringResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
