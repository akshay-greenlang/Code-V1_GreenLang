# -*- coding: utf-8 -*-
"""
Continuous Monitoring Workflow
===================================

3-phase workflow for ongoing energy benchmark monitoring within
PACK-035 Energy Benchmark Pack.

Phases:
    1. DataIngestion        -- Collect new meter data, validate, merge with baseline
    2. PerformanceTracking  -- CUSUM analysis, SPC rule checks, rolling EUI
    3. AlertGeneration      -- Deviation alerts, forecast regression, performance alerts

Designed for monthly/quarterly runs. Detects performance regression by
comparing rolling EUI against baseline using CUSUM and Western Electric
SPC rules. Forecasts future EUI trajectory using exponential smoothing.

The workflow follows GreenLang zero-hallucination principles: CUSUM,
SPC rules, exponential smoothing, and deviation detection use deterministic
statistical methods. No LLM calls in the numeric computation path.

Schedule: monthly/quarterly
Estimated duration: 15 minutes

Author: GreenLang Team
Version: 35.0.0
"""

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

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


class AlertSeverity(str, Enum):
    """Alert severity classification."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFORMATIONAL = "informational"


class AlertType(str, Enum):
    """Type of monitoring alert."""

    CUSUM_UPPER = "cusum_upper_deviation"
    CUSUM_LOWER = "cusum_lower_deviation"
    SPC_RULE_1 = "spc_rule_1_beyond_3sigma"
    SPC_RULE_2 = "spc_rule_2_run_of_9"
    SPC_RULE_3 = "spc_rule_3_trend_of_6"
    THRESHOLD_BREACH = "threshold_breach"
    PERFORMANCE_REGRESSION = "performance_regression"
    FORECAST_WARNING = "forecast_warning"
    DATA_QUALITY = "data_quality_issue"


class TrendDirection(str, Enum):
    """Trend direction indicator."""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    INSUFFICIENT_DATA = "insufficient_data"


class MonitoringFrequency(str, Enum):
    """Monitoring frequency."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


# =============================================================================
# CUSUM / SPC CONSTANTS
# =============================================================================

DEFAULT_CUSUM_H = 4.0  # Decision interval (multiples of std dev)
DEFAULT_CUSUM_K = 0.5  # Allowance (shift to detect, in std dev units)
SPC_SIGMA_MULTIPLIER = 3.0  # Western Electric Rule 1: beyond 3-sigma


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


class EnergyDataPoint(BaseModel):
    """Monthly energy data point for monitoring."""

    period: str = Field(default="", description="Period YYYY-MM")
    consumption_kwh: float = Field(default=0.0, ge=0.0, description="Total consumption kWh")
    floor_area_m2: float = Field(default=0.0, ge=0.0, description="Floor area for EUI calc")
    heating_degree_days: float = Field(default=0.0, ge=0.0, description="HDD for period")
    cooling_degree_days: float = Field(default=0.0, ge=0.0, description="CDD for period")
    cost: float = Field(default=0.0, ge=0.0, description="Energy cost")
    data_quality: str = Field(default="measured", description="measured|estimated|suspect")


class BaselineModel(BaseModel):
    """Reference baseline model for deviation detection."""

    baseline_id: str = Field(default="", description="Baseline identifier")
    baseline_eui: float = Field(default=0.0, ge=0.0, description="Baseline EUI kWh/m2/yr")
    monthly_expected_kwh: Dict[str, float] = Field(
        default_factory=dict, description="Expected kWh per month"
    )
    std_deviation_kwh: float = Field(default=0.0, ge=0.0, description="Std dev of baseline")
    regression_coefficients: Dict[str, float] = Field(
        default_factory=lambda: {"intercept": 0.0, "hdd_coeff": 0.0, "cdd_coeff": 0.0},
    )
    floor_area_m2: float = Field(default=0.0, ge=0.0, description="Baseline floor area")


class AlertThresholds(BaseModel):
    """Configurable alert thresholds."""

    cusum_h_factor: float = Field(default=4.0, ge=1.0, le=10.0, description="CUSUM H factor")
    cusum_k_factor: float = Field(default=0.5, ge=0.1, le=2.0, description="CUSUM K factor")
    threshold_pct: float = Field(default=10.0, ge=1.0, le=50.0, description="Threshold %")
    forecast_warning_pct: float = Field(default=15.0, ge=5.0, le=50.0, description="Forecast warning %")
    spc_enabled: bool = Field(default=True, description="Enable SPC rules")


class MonitoringAlert(BaseModel):
    """Monitoring alert record."""

    alert_id: str = Field(default_factory=lambda: f"alt-{uuid.uuid4().hex[:8]}")
    alert_type: AlertType = Field(default=AlertType.THRESHOLD_BREACH)
    severity: AlertSeverity = Field(default=AlertSeverity.WARNING)
    timestamp: str = Field(default="", description="Detection timestamp")
    period: str = Field(default="", description="Affected period")
    expected_value: float = Field(default=0.0, description="Expected value")
    actual_value: float = Field(default=0.0, description="Actual value")
    deviation_pct: float = Field(default=0.0, description="Deviation %")
    cusum_value: float = Field(default=0.0, description="CUSUM statistic")
    description: str = Field(default="")
    recommended_action: str = Field(default="")


class ForecastResult(BaseModel):
    """EUI forecast result."""

    forecast_periods: int = Field(default=3, ge=1, description="Periods ahead")
    forecast_eui: List[float] = Field(default_factory=list, description="Forecasted EUI values")
    forecast_trend: TrendDirection = Field(default=TrendDirection.STABLE)
    smoothing_alpha: float = Field(default=0.3, ge=0.0, le=1.0)


class ContinuousMonitoringInput(BaseModel):
    """Input data model for ContinuousMonitoringWorkflow."""

    facility_id: str = Field(default="", description="Facility identifier")
    new_energy_data: List[EnergyDataPoint] = Field(default_factory=list)
    baseline_model: BaselineModel = Field(default_factory=BaselineModel)
    alert_thresholds: AlertThresholds = Field(default_factory=AlertThresholds)
    historical_eui: List[float] = Field(default_factory=list, description="Historical EUI series")
    monitoring_frequency: MonitoringFrequency = Field(default=MonitoringFrequency.MONTHLY)
    cusum_state: Dict[str, float] = Field(
        default_factory=lambda: {"cusum_upper": 0.0, "cusum_lower": 0.0},
        description="Carried-forward CUSUM state",
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class ContinuousMonitoringResult(BaseModel):
    """Complete result from continuous monitoring workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="continuous_monitoring")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    facility_id: str = Field(default="")
    rolling_eui: float = Field(default=0.0, ge=0.0, description="Rolling 12-month EUI")
    cusum_status: Dict[str, float] = Field(default_factory=dict, description="CUSUM state")
    spc_alerts: List[MonitoringAlert] = Field(default_factory=list)
    forecast: Dict[str, Any] = Field(default_factory=dict)
    deviations: List[Dict[str, Any]] = Field(default_factory=list)
    total_alerts: int = Field(default=0)
    critical_alerts: int = Field(default=0)
    overall_deviation_pct: float = Field(default=0.0)
    trend_direction: str = Field(default="stable")
    duration_seconds: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ContinuousMonitoringWorkflow:
    """
    3-phase continuous energy benchmark monitoring workflow.

    Performs data ingestion and validation, CUSUM and SPC-based performance
    tracking, and automated alert generation with EUI forecasting.

    Zero-hallucination: CUSUM statistics, Western Electric SPC rules,
    exponential smoothing forecasts, and deviation calculations use
    deterministic formulas only. No LLM calls in the numeric path.

    Attributes:
        workflow_id: Unique execution identifier.
        _alerts: Generated monitoring alerts.
        _rolling_eui_values: Rolling EUI time series.
        _cusum_upper: Current CUSUM upper value.
        _cusum_lower: Current CUSUM lower value.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = ContinuousMonitoringWorkflow()
        >>> inp = ContinuousMonitoringInput(
        ...     facility_id="fac-001",
        ...     new_energy_data=[...],
        ...     baseline_model=BaselineModel(baseline_eui=150.0),
        ... )
        >>> result = wf.run(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ContinuousMonitoringWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._alerts: List[MonitoringAlert] = []
        self._rolling_eui_values: List[float] = []
        self._cusum_upper: float = 0.0
        self._cusum_lower: float = 0.0
        self._ingested_data: List[Dict[str, Any]] = []
        self._deviations: List[Dict[str, Any]] = []
        self._forecast_result: Optional[ForecastResult] = None
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: ContinuousMonitoringInput) -> ContinuousMonitoringResult:
        """
        Execute the 3-phase continuous monitoring workflow.

        Args:
            input_data: Validated continuous monitoring input.

        Returns:
            ContinuousMonitoringResult with rolling EUI, CUSUM state, alerts, forecast.
        """
        t_start = time.perf_counter()
        self.logger.info(
            "Starting continuous monitoring workflow %s for facility=%s",
            self.workflow_id, input_data.facility_id,
        )

        self._phase_results = []
        self._alerts = []
        self._ingested_data = []
        self._deviations = []
        self._forecast_result = None
        self._cusum_upper = input_data.cusum_state.get("cusum_upper", 0.0)
        self._cusum_lower = input_data.cusum_state.get("cusum_lower", 0.0)
        self._rolling_eui_values = list(input_data.historical_eui)
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = self._phase_data_ingestion(input_data)
            self._phase_results.append(phase1)

            phase2 = self._phase_performance_tracking(input_data)
            self._phase_results.append(phase2)

            phase3 = self._phase_alert_generation(input_data)
            self._phase_results.append(phase3)

            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Continuous monitoring workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = time.perf_counter() - t_start
        rolling_eui = self._rolling_eui_values[-1] if self._rolling_eui_values else 0.0
        critical_count = sum(1 for a in self._alerts if a.severity == AlertSeverity.CRITICAL)
        overall_dev = self._calculate_overall_deviation(input_data)
        trend = self._determine_overall_trend()

        result = ContinuousMonitoringResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            facility_id=input_data.facility_id,
            rolling_eui=round(rolling_eui, 2),
            cusum_status={
                "cusum_upper": round(self._cusum_upper, 4),
                "cusum_lower": round(self._cusum_lower, 4),
            },
            spc_alerts=self._alerts,
            forecast=self._forecast_result.model_dump() if self._forecast_result else {},
            deviations=self._deviations,
            total_alerts=len(self._alerts),
            critical_alerts=critical_count,
            overall_deviation_pct=round(overall_dev, 2),
            trend_direction=trend.value,
            duration_seconds=round(elapsed, 4),
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Continuous monitoring workflow %s completed in %.2fs alerts=%d trend=%s",
            self.workflow_id, elapsed, len(self._alerts), trend.value,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Data Ingestion
    # -------------------------------------------------------------------------

    def _phase_data_ingestion(
        self, input_data: ContinuousMonitoringInput
    ) -> PhaseResult:
        """Collect new meter data, validate quality, merge with baseline."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        valid_count = 0
        suspect_count = 0
        total_kwh = 0.0
        floor_area = input_data.baseline_model.floor_area_m2

        for dp in input_data.new_energy_data:
            if dp.data_quality == "suspect":
                suspect_count += 1
                warnings.append(f"Suspect data quality for period {dp.period}")
            else:
                valid_count += 1

            total_kwh += dp.consumption_kwh

            # Calculate period EUI
            area = dp.floor_area_m2 if dp.floor_area_m2 > 0 else floor_area
            period_eui = dp.consumption_kwh / area * 12.0 if area > 0 else 0.0

            self._ingested_data.append({
                "period": dp.period,
                "consumption_kwh": dp.consumption_kwh,
                "eui_annualised": round(period_eui, 2),
                "hdd": dp.heating_degree_days,
                "cdd": dp.cooling_degree_days,
                "quality": dp.data_quality,
            })

            self._rolling_eui_values.append(period_eui)

        # Keep last 36 months of rolling data
        if len(self._rolling_eui_values) > 36:
            self._rolling_eui_values = self._rolling_eui_values[-36:]

        outputs["data_points_received"] = len(input_data.new_energy_data)
        outputs["valid_count"] = valid_count
        outputs["suspect_count"] = suspect_count
        outputs["total_consumption_kwh"] = round(total_kwh, 2)
        outputs["rolling_series_length"] = len(self._rolling_eui_values)

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 1 DataIngestion: %d points, valid=%d suspect=%d",
            len(input_data.new_energy_data), valid_count, suspect_count,
        )
        return PhaseResult(
            phase_name="data_ingestion", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Performance Tracking
    # -------------------------------------------------------------------------

    def _phase_performance_tracking(
        self, input_data: ContinuousMonitoringInput
    ) -> PhaseResult:
        """CUSUM analysis, SPC rule checks, rolling EUI tracking."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        baseline = input_data.baseline_model
        thresholds = input_data.alert_thresholds

        std_dev = baseline.std_deviation_kwh if baseline.std_deviation_kwh > 0 else baseline.baseline_eui * 0.10
        k = std_dev * thresholds.cusum_k_factor
        h = std_dev * thresholds.cusum_h_factor

        cusum_alerts: List[MonitoringAlert] = []

        for dp_dict in self._ingested_data:
            period = dp_dict["period"]
            actual_kwh = dp_dict["consumption_kwh"]
            hdd = dp_dict["hdd"]
            cdd = dp_dict["cdd"]

            # Calculate expected from regression model
            coeff = baseline.regression_coefficients
            expected_kwh = (
                coeff.get("intercept", 0.0)
                + coeff.get("hdd_coeff", 0.0) * hdd
                + coeff.get("cdd_coeff", 0.0) * cdd
            )
            if expected_kwh <= 0:
                expected_kwh = baseline.monthly_expected_kwh.get(period, baseline.baseline_eui * baseline.floor_area_m2 / 12.0)

            diff = actual_kwh - expected_kwh
            deviation_pct = (diff / expected_kwh * 100.0) if expected_kwh > 0 else 0.0

            self._deviations.append({
                "period": period,
                "expected_kwh": round(expected_kwh, 2),
                "actual_kwh": round(actual_kwh, 2),
                "deviation_kwh": round(diff, 2),
                "deviation_pct": round(deviation_pct, 2),
            })

            # CUSUM update (zero-hallucination deterministic)
            self._cusum_upper = max(0.0, self._cusum_upper + diff - k)
            self._cusum_lower = max(0.0, self._cusum_lower - diff - k)

            if self._cusum_upper > h:
                cusum_alerts.append(MonitoringAlert(
                    alert_type=AlertType.CUSUM_UPPER,
                    severity=AlertSeverity.CRITICAL if self._cusum_upper > h * 2 else AlertSeverity.WARNING,
                    timestamp=datetime.utcnow().isoformat(),
                    period=period,
                    expected_value=round(expected_kwh, 2),
                    actual_value=round(actual_kwh, 2),
                    deviation_pct=round(deviation_pct, 2),
                    cusum_value=round(self._cusum_upper, 2),
                    description=(
                        f"CUSUM upper limit exceeded in {period}: "
                        f"actual={actual_kwh:.0f} vs expected={expected_kwh:.0f} kWh ({deviation_pct:+.1f}%)"
                    ),
                    recommended_action="Investigate root cause of sustained overconsumption",
                ))

            if self._cusum_lower > h:
                cusum_alerts.append(MonitoringAlert(
                    alert_type=AlertType.CUSUM_LOWER,
                    severity=AlertSeverity.INFORMATIONAL,
                    timestamp=datetime.utcnow().isoformat(),
                    period=period,
                    expected_value=round(expected_kwh, 2),
                    actual_value=round(actual_kwh, 2),
                    deviation_pct=round(deviation_pct, 2),
                    cusum_value=round(self._cusum_lower, 2),
                    description=(
                        f"CUSUM lower limit exceeded in {period}: "
                        f"sustained savings detected ({deviation_pct:+.1f}%)"
                    ),
                    recommended_action="Verify savings from implemented efficiency measures",
                ))

        # SPC rule checks on rolling EUI
        spc_alerts: List[MonitoringAlert] = []
        if thresholds.spc_enabled and len(self._rolling_eui_values) >= 9:
            spc_alerts = self._check_spc_rules(self._rolling_eui_values, baseline.baseline_eui, std_dev)

        self._alerts.extend(cusum_alerts)
        self._alerts.extend(spc_alerts)

        # Rolling 12-month EUI
        if len(self._rolling_eui_values) >= 12:
            rolling_12 = sum(self._rolling_eui_values[-12:]) / 12.0
        elif self._rolling_eui_values:
            rolling_12 = sum(self._rolling_eui_values) / len(self._rolling_eui_values)
        else:
            rolling_12 = 0.0

        outputs["cusum_upper"] = round(self._cusum_upper, 4)
        outputs["cusum_lower"] = round(self._cusum_lower, 4)
        outputs["cusum_alerts"] = len(cusum_alerts)
        outputs["spc_alerts"] = len(spc_alerts)
        outputs["rolling_12m_eui"] = round(rolling_12, 2)
        outputs["deviations_count"] = len(self._deviations)

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 2 PerformanceTracking: CUSUM alerts=%d SPC alerts=%d rolling EUI=%.1f",
            len(cusum_alerts), len(spc_alerts), rolling_12,
        )
        return PhaseResult(
            phase_name="performance_tracking", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _check_spc_rules(
        self, values: List[float], center: float, std_dev: float
    ) -> List[MonitoringAlert]:
        """Check Western Electric SPC rules (zero-hallucination)."""
        alerts: List[MonitoringAlert] = []
        sigma = std_dev if std_dev > 0 else center * 0.10

        # Rule 1: Single point beyond 3-sigma
        latest = values[-1]
        if abs(latest - center) > SPC_SIGMA_MULTIPLIER * sigma:
            alerts.append(MonitoringAlert(
                alert_type=AlertType.SPC_RULE_1,
                severity=AlertSeverity.CRITICAL,
                timestamp=datetime.utcnow().isoformat(),
                expected_value=round(center, 2),
                actual_value=round(latest, 2),
                deviation_pct=round((latest - center) / center * 100, 2) if center > 0 else 0.0,
                description=f"SPC Rule 1: EUI {latest:.1f} beyond 3-sigma ({center:.1f} +/- {3*sigma:.1f})",
                recommended_action="Immediate investigation required for extreme deviation",
            ))

        # Rule 2: 9 consecutive points on same side of center
        if len(values) >= 9:
            last_9 = values[-9:]
            above = all(v > center for v in last_9)
            below = all(v < center for v in last_9)
            if above or below:
                direction = "above" if above else "below"
                alerts.append(MonitoringAlert(
                    alert_type=AlertType.SPC_RULE_2,
                    severity=AlertSeverity.WARNING,
                    timestamp=datetime.utcnow().isoformat(),
                    expected_value=round(center, 2),
                    description=f"SPC Rule 2: 9 consecutive periods {direction} baseline ({center:.1f})",
                    recommended_action="Baseline may need recalibration; systematic shift detected",
                ))

        # Rule 3: 6 consecutive points trending in same direction
        if len(values) >= 6:
            last_6 = values[-6:]
            increasing = all(last_6[i] < last_6[i + 1] for i in range(5))
            decreasing = all(last_6[i] > last_6[i + 1] for i in range(5))
            if increasing or decreasing:
                direction = "increasing" if increasing else "decreasing"
                alerts.append(MonitoringAlert(
                    alert_type=AlertType.SPC_RULE_3,
                    severity=AlertSeverity.WARNING,
                    timestamp=datetime.utcnow().isoformat(),
                    expected_value=round(center, 2),
                    description=f"SPC Rule 3: 6 consecutive periods {direction}",
                    recommended_action=f"Investigate cause of sustained {direction} trend",
                ))

        return alerts

    # -------------------------------------------------------------------------
    # Phase 3: Alert Generation
    # -------------------------------------------------------------------------

    def _phase_alert_generation(
        self, input_data: ContinuousMonitoringInput
    ) -> PhaseResult:
        """Generate threshold, forecast, and performance regression alerts."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        new_alerts: List[MonitoringAlert] = []
        thresholds = input_data.alert_thresholds
        baseline = input_data.baseline_model

        # Threshold breach alerts
        for dev in self._deviations:
            if abs(dev["deviation_pct"]) > thresholds.threshold_pct:
                direction = "over" if dev["deviation_pct"] > 0 else "under"
                new_alerts.append(MonitoringAlert(
                    alert_type=AlertType.THRESHOLD_BREACH,
                    severity=AlertSeverity.WARNING,
                    timestamp=datetime.utcnow().isoformat(),
                    period=dev["period"],
                    expected_value=dev["expected_kwh"],
                    actual_value=dev["actual_kwh"],
                    deviation_pct=dev["deviation_pct"],
                    description=(
                        f"Threshold breach in {dev['period']}: "
                        f"{direction}-consumption by {abs(dev['deviation_pct']):.1f}%"
                    ),
                    recommended_action=f"Review {direction}-consumption cause for {dev['period']}",
                ))

        # Performance regression detection
        if len(self._rolling_eui_values) >= 6:
            recent_avg = sum(self._rolling_eui_values[-3:]) / 3.0
            older_avg = sum(self._rolling_eui_values[-6:-3]) / 3.0
            if older_avg > 0:
                regression_pct = (recent_avg - older_avg) / older_avg * 100.0
                if regression_pct > thresholds.threshold_pct:
                    new_alerts.append(MonitoringAlert(
                        alert_type=AlertType.PERFORMANCE_REGRESSION,
                        severity=AlertSeverity.CRITICAL,
                        timestamp=datetime.utcnow().isoformat(),
                        expected_value=round(older_avg, 2),
                        actual_value=round(recent_avg, 2),
                        deviation_pct=round(regression_pct, 2),
                        description=(
                            f"Performance regression: recent 3-month EUI ({recent_avg:.1f}) "
                            f"is {regression_pct:.1f}% higher than prior 3-month ({older_avg:.1f})"
                        ),
                        recommended_action="Conduct detailed investigation of recent performance decline",
                    ))

        # Exponential smoothing forecast
        self._forecast_result = self._generate_forecast(input_data)
        if self._forecast_result and self._forecast_result.forecast_eui:
            max_forecast = max(self._forecast_result.forecast_eui)
            if baseline.baseline_eui > 0:
                forecast_dev = (max_forecast - baseline.baseline_eui) / baseline.baseline_eui * 100.0
                if forecast_dev > thresholds.forecast_warning_pct:
                    new_alerts.append(MonitoringAlert(
                        alert_type=AlertType.FORECAST_WARNING,
                        severity=AlertSeverity.WARNING,
                        timestamp=datetime.utcnow().isoformat(),
                        expected_value=round(baseline.baseline_eui, 2),
                        actual_value=round(max_forecast, 2),
                        deviation_pct=round(forecast_dev, 2),
                        description=(
                            f"Forecast warning: projected EUI ({max_forecast:.1f}) exceeds "
                            f"baseline by {forecast_dev:.1f}% within {self._forecast_result.forecast_periods} periods"
                        ),
                        recommended_action="Proactive intervention recommended to prevent further degradation",
                    ))

        # Data quality alerts
        suspect_periods = [d["period"] for d in self._ingested_data if d.get("quality") == "suspect"]
        if suspect_periods:
            new_alerts.append(MonitoringAlert(
                alert_type=AlertType.DATA_QUALITY,
                severity=AlertSeverity.INFORMATIONAL,
                timestamp=datetime.utcnow().isoformat(),
                description=f"Data quality issues in periods: {', '.join(suspect_periods)}",
                recommended_action="Verify meter readings and correct suspect data",
            ))

        self._alerts.extend(new_alerts)

        outputs["threshold_alerts"] = sum(1 for a in new_alerts if a.alert_type == AlertType.THRESHOLD_BREACH)
        outputs["regression_alerts"] = sum(1 for a in new_alerts if a.alert_type == AlertType.PERFORMANCE_REGRESSION)
        outputs["forecast_alerts"] = sum(1 for a in new_alerts if a.alert_type == AlertType.FORECAST_WARNING)
        outputs["quality_alerts"] = sum(1 for a in new_alerts if a.alert_type == AlertType.DATA_QUALITY)
        outputs["total_new_alerts"] = len(new_alerts)
        outputs["total_cumulative_alerts"] = len(self._alerts)

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 3 AlertGeneration: %d new alerts, %d total",
            len(new_alerts), len(self._alerts),
        )
        return PhaseResult(
            phase_name="alert_generation", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _generate_forecast(
        self, input_data: ContinuousMonitoringInput
    ) -> Optional[ForecastResult]:
        """Generate EUI forecast using exponential smoothing (zero-hallucination)."""
        if len(self._rolling_eui_values) < 3:
            return None

        alpha = 0.3  # Smoothing factor
        values = self._rolling_eui_values
        smoothed = values[0]
        for v in values[1:]:
            smoothed = alpha * v + (1 - alpha) * smoothed

        # Trend component
        trend = 0.0
        if len(values) >= 6:
            recent = sum(values[-3:]) / 3.0
            older = sum(values[-6:-3]) / 3.0
            trend = (recent - older) / 3.0

        # Generate 3 period forecast
        forecasts = []
        for i in range(1, 4):
            forecast_val = max(0.0, smoothed + trend * i)
            forecasts.append(round(forecast_val, 2))

        # Determine forecast trend
        if trend > 0.5:
            direction = TrendDirection.DEGRADING
        elif trend < -0.5:
            direction = TrendDirection.IMPROVING
        else:
            direction = TrendDirection.STABLE

        return ForecastResult(
            forecast_periods=3,
            forecast_eui=forecasts,
            forecast_trend=direction,
            smoothing_alpha=alpha,
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _calculate_overall_deviation(self, input_data: ContinuousMonitoringInput) -> float:
        """Calculate overall deviation from baseline."""
        if not self._deviations:
            return 0.0
        avg_dev = sum(d["deviation_pct"] for d in self._deviations) / len(self._deviations)
        return avg_dev

    def _determine_overall_trend(self) -> TrendDirection:
        """Determine overall EUI trend from rolling values."""
        if len(self._rolling_eui_values) < 6:
            return TrendDirection.INSUFFICIENT_DATA
        recent = sum(self._rolling_eui_values[-3:]) / 3.0
        older = sum(self._rolling_eui_values[-6:-3]) / 3.0
        if older <= 0:
            return TrendDirection.STABLE
        change_pct = (recent - older) / older * 100.0
        if change_pct < -2.0:
            return TrendDirection.IMPROVING
        elif change_pct > 2.0:
            return TrendDirection.DEGRADING
        return TrendDirection.STABLE

    def _compute_provenance(self, result: ContinuousMonitoringResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
