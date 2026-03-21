# -*- coding: utf-8 -*-
"""
Continuous Building Monitoring Workflow
============================================

4-phase workflow for ongoing building energy performance monitoring within
PACK-032 Building Energy Assessment Pack.

Phases:
    1. DataIngestion          -- Ingest BMS, smart meter, and utility bill data
    2. PerformanceAnalysis    -- EUI tracking, deviation detection, CUSUM
    3. AnomalyDetection       -- Alerts for energy spikes, HVAC faults
    4. TrendReporting         -- Monthly/quarterly dashboards, YoY comparison

Zero-hallucination: all statistical calculations (CUSUM, deviation thresholds,
trend regression) use deterministic formulas. No LLM calls in the numeric path.

Schedule: continuous (daily/weekly/monthly cycles)
Estimated duration: 30 minutes per cycle

Author: GreenLang Team
Version: 32.0.0
"""

import hashlib
import json
import logging
import math
import statistics
import uuid
from datetime import datetime
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


class DataSourceType(str, Enum):
    """Monitoring data source types."""

    BMS = "bms"
    SMART_METER = "smart_meter"
    UTILITY_BILL = "utility_bill"
    IOT_SENSOR = "iot_sensor"
    WEATHER_API = "weather_api"
    OCCUPANCY_SENSOR = "occupancy_sensor"


class AlertSeverity(str, Enum):
    """Alert severity classification."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertCategory(str, Enum):
    """Alert categorisation."""

    ENERGY_SPIKE = "energy_spike"
    CONSUMPTION_DRIFT = "consumption_drift"
    HVAC_FAULT = "hvac_fault"
    EQUIPMENT_DEGRADATION = "equipment_degradation"
    BASELOAD_ANOMALY = "baseload_anomaly"
    TEMPERATURE_DEVIATION = "temperature_deviation"
    OCCUPANCY_MISMATCH = "occupancy_mismatch"
    METER_ERROR = "meter_error"
    CUSUM_EXCEEDANCE = "cusum_exceedance"


class TrendDirection(str, Enum):
    """Performance trend direction."""

    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"
    INSUFFICIENT_DATA = "insufficient_data"


class MonitoringFrequency(str, Enum):
    """Data monitoring frequency."""

    HALF_HOURLY = "half_hourly"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


# =============================================================================
# ZERO-HALLUCINATION REFERENCE CONSTANTS
# =============================================================================

# CUSUM control parameters -- ISO 50006 / CIBSE TM63
CUSUM_CONTROL_LIMIT_SIGMA: float = 3.0
CUSUM_WARNING_LIMIT_SIGMA: float = 2.0
CUSUM_TARGET_IMPROVEMENT_PCT: float = 0.0  # 0 = maintain baseline

# Baseload ratio thresholds by building type
BASELOAD_RATIO_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "office": {"good": 0.25, "typical": 0.40, "poor": 0.55},
    "retail": {"good": 0.30, "typical": 0.45, "poor": 0.60},
    "hospital": {"good": 0.50, "typical": 0.65, "poor": 0.80},
    "school": {"good": 0.15, "typical": 0.30, "poor": 0.45},
    "hotel": {"good": 0.35, "typical": 0.50, "poor": 0.65},
    "warehouse": {"good": 0.20, "typical": 0.35, "poor": 0.50},
    "residential": {"good": 0.20, "typical": 0.35, "poor": 0.50},
}

# Temperature setpoint ranges (C) by building type
TEMPERATURE_SETPOINTS: Dict[str, Dict[str, float]] = {
    "office": {"heating_min": 19.0, "heating_max": 22.0, "cooling_min": 23.0, "cooling_max": 26.0},
    "retail": {"heating_min": 18.0, "heating_max": 21.0, "cooling_min": 23.0, "cooling_max": 26.0},
    "hospital": {"heating_min": 20.0, "heating_max": 24.0, "cooling_min": 22.0, "cooling_max": 25.0},
    "school": {"heating_min": 18.0, "heating_max": 21.0, "cooling_min": 23.0, "cooling_max": 26.0},
    "hotel": {"heating_min": 20.0, "heating_max": 23.0, "cooling_min": 23.0, "cooling_max": 25.0},
    "warehouse": {"heating_min": 15.0, "heating_max": 18.0, "cooling_min": 25.0, "cooling_max": 28.0},
    "residential": {"heating_min": 19.0, "heating_max": 22.0, "cooling_min": 24.0, "cooling_max": 26.0},
}

# Degree day base temperatures
HDD_BASE_TEMP: float = 15.5
CDD_BASE_TEMP: float = 18.3

# EUI benchmarks (kWh/m2/yr) for deviation assessment
EUI_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "office": {"typical": 230.0, "good": 128.0, "best": 95.0},
    "retail": {"typical": 305.0, "good": 190.0, "best": 140.0},
    "hospital": {"typical": 420.0, "good": 310.0, "best": 250.0},
    "school": {"typical": 150.0, "good": 110.0, "best": 80.0},
    "hotel": {"typical": 340.0, "good": 250.0, "best": 200.0},
    "warehouse": {"typical": 120.0, "good": 85.0, "best": 55.0},
    "residential": {"typical": 170.0, "good": 120.0, "best": 80.0},
}


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
    """Individual energy consumption data point."""

    timestamp: str = Field(default="", description="ISO 8601 timestamp or YYYY-MM")
    period: str = Field(default="", description="Period identifier YYYY-MM or YYYY-MM-DD")
    source: DataSourceType = Field(default=DataSourceType.SMART_METER)
    energy_source: str = Field(default="electricity")
    consumption_kwh: float = Field(default=0.0, ge=0.0)
    demand_kw: float = Field(default=0.0, ge=0.0)
    cost_eur: float = Field(default=0.0, ge=0.0)
    temperature_c: float = Field(default=15.0, description="External temperature")
    hdd: float = Field(default=0.0, ge=0.0, description="Heating degree days")
    cdd: float = Field(default=0.0, ge=0.0, description="Cooling degree days")
    occupancy_pct: float = Field(default=100.0, ge=0.0, le=100.0)


class BMSReading(BaseModel):
    """BMS system reading."""

    point_id: str = Field(default="")
    timestamp: str = Field(default="")
    point_type: str = Field(default="", description="temperature|power|flow|status|humidity")
    value: float = Field(default=0.0)
    unit: str = Field(default="")
    zone: str = Field(default="")
    system: str = Field(default="", description="HVAC|lighting|dhw")
    quality: str = Field(default="good", description="good|suspect|bad")


class BaselineModel(BaseModel):
    """Energy baseline model for CUSUM analysis."""

    baseline_period: str = Field(default="", description="Baseline period e.g. 2024")
    baseline_kwh_per_month: float = Field(default=0.0, ge=0.0)
    baseline_kwh_per_hdd: float = Field(default=0.0, ge=0.0)
    baseline_kwh_per_cdd: float = Field(default=0.0, ge=0.0)
    baseline_intercept: float = Field(default=0.0, ge=0.0, description="Base/weather-independent kWh")
    r_squared: float = Field(default=0.0, ge=0.0, le=1.0)
    cv_rmse_pct: float = Field(default=0.0, ge=0.0)
    std_deviation: float = Field(default=0.0, ge=0.0, description="Monthly std dev of residuals")


class CUSUMResult(BaseModel):
    """CUSUM analysis result."""

    period: str = Field(default="")
    actual_kwh: float = Field(default=0.0, ge=0.0)
    expected_kwh: float = Field(default=0.0, ge=0.0)
    deviation_kwh: float = Field(default=0.0)
    cusum_kwh: float = Field(default=0.0)
    control_limit_upper: float = Field(default=0.0)
    control_limit_lower: float = Field(default=0.0)
    in_control: bool = Field(default=True)
    deviation_pct: float = Field(default=0.0)


class MonitoringAlert(BaseModel):
    """Performance monitoring alert."""

    alert_id: str = Field(default_factory=lambda: f"alert-{uuid.uuid4().hex[:8]}")
    timestamp: str = Field(default="")
    severity: AlertSeverity = Field(default=AlertSeverity.MEDIUM)
    category: AlertCategory = Field(default=AlertCategory.CONSUMPTION_DRIFT)
    title: str = Field(default="")
    description: str = Field(default="")
    affected_system: str = Field(default="")
    affected_zone: str = Field(default="")
    metric_name: str = Field(default="")
    metric_value: float = Field(default=0.0)
    threshold_value: float = Field(default=0.0)
    recommended_action: str = Field(default="")


class TrendMetric(BaseModel):
    """Performance trend metric."""

    metric_name: str = Field(default="")
    current_value: float = Field(default=0.0)
    previous_value: float = Field(default=0.0)
    change_pct: float = Field(default=0.0)
    direction: TrendDirection = Field(default=TrendDirection.STABLE)
    period_current: str = Field(default="")
    period_previous: str = Field(default="")
    yoy_change_pct: float = Field(default=0.0)


class PerformanceDashboard(BaseModel):
    """Performance summary dashboard data."""

    period: str = Field(default="")
    eui_kwh_per_sqm: float = Field(default=0.0, ge=0.0)
    total_consumption_kwh: float = Field(default=0.0, ge=0.0)
    total_cost_eur: float = Field(default=0.0, ge=0.0)
    peak_demand_kw: float = Field(default=0.0, ge=0.0)
    baseload_kw: float = Field(default=0.0, ge=0.0)
    baseload_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    hdd_total: float = Field(default=0.0, ge=0.0)
    cdd_total: float = Field(default=0.0, ge=0.0)
    weather_normalised_kwh: float = Field(default=0.0, ge=0.0)
    active_alerts: int = Field(default=0, ge=0)
    trends: List[TrendMetric] = Field(default_factory=list)


class ContinuousBuildingMonitoringInput(BaseModel):
    """Input data model for ContinuousBuildingMonitoringWorkflow."""

    building_name: str = Field(default="")
    building_type: str = Field(default="office")
    total_floor_area_sqm: float = Field(default=0.0, ge=0.0)
    baseline: BaselineModel = Field(default_factory=BaselineModel)
    current_data: List[EnergyDataPoint] = Field(default_factory=list)
    bms_readings: List[BMSReading] = Field(default_factory=list)
    previous_period_data: List[EnergyDataPoint] = Field(default_factory=list)
    previous_year_data: List[EnergyDataPoint] = Field(default_factory=list)
    monitoring_frequency: MonitoringFrequency = Field(default=MonitoringFrequency.MONTHLY)
    alert_threshold_pct: float = Field(default=10.0, ge=0.0, le=50.0)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("total_floor_area_sqm")
    @classmethod
    def validate_floor_area(cls, v: float) -> float:
        """Floor area must be positive."""
        if v <= 0:
            raise ValueError("total_floor_area_sqm must be > 0")
        return v


class ContinuousBuildingMonitoringResult(BaseModel):
    """Complete result from continuous building monitoring workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="continuous_building_monitoring")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    building_name: str = Field(default="")
    monitoring_period: str = Field(default="")
    data_points_ingested: int = Field(default=0)
    cusum_results: List[CUSUMResult] = Field(default_factory=list)
    cusum_in_control: bool = Field(default=True)
    cumulative_deviation_kwh: float = Field(default=0.0)
    alerts: List[MonitoringAlert] = Field(default_factory=list)
    critical_alerts: int = Field(default=0)
    high_alerts: int = Field(default=0)
    dashboard: Optional[PerformanceDashboard] = None
    trends: List[TrendMetric] = Field(default_factory=list)
    eui_current: float = Field(default=0.0, ge=0.0)
    benchmark_status: str = Field(default="")
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ContinuousBuildingMonitoringWorkflow:
    """
    4-phase continuous building energy monitoring workflow.

    Ingests BMS, smart meter, and utility data; performs CUSUM-based
    deviation detection; generates anomaly alerts for energy spikes and
    HVAC faults; and produces trend reports with YoY comparison.

    Zero-hallucination: all CUSUM, statistical deviation, and trend
    calculations use deterministic formulas per ISO 50006 and CIBSE TM63.

    Example:
        >>> wf = ContinuousBuildingMonitoringWorkflow()
        >>> inp = ContinuousBuildingMonitoringInput(
        ...     total_floor_area_sqm=2000,
        ...     current_data=[...],
        ...     baseline=BaselineModel(...)
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize ContinuousBuildingMonitoringWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._cusum_results: List[CUSUMResult] = []
        self._alerts: List[MonitoringAlert] = []
        self._trends: List[TrendMetric] = []
        self._dashboard: Optional[PerformanceDashboard] = None
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(
        self,
        input_data: Optional[ContinuousBuildingMonitoringInput] = None,
    ) -> ContinuousBuildingMonitoringResult:
        """Execute the 4-phase continuous monitoring workflow."""
        if input_data is None:
            raise ValueError("input_data must be provided")

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting continuous monitoring workflow %s for %s",
            self.workflow_id, input_data.building_name,
        )

        self._phase_results = []
        self._cusum_results = []
        self._alerts = []
        self._trends = []
        self._dashboard = None
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_data_ingestion(input_data)
            self._phase_results.append(phase1)

            phase2 = await self._phase_performance_analysis(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_anomaly_detection(input_data)
            self._phase_results.append(phase3)

            phase4 = await self._phase_trend_reporting(input_data)
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
        cusum_in_control = all(c.in_control for c in self._cusum_results)
        cumulative_dev = self._cusum_results[-1].cusum_kwh if self._cusum_results else 0.0
        eui = self._dashboard.eui_kwh_per_sqm if self._dashboard else 0.0
        periods = sorted({d.period for d in input_data.current_data if d.period})
        monitoring_period = f"{periods[0]} to {periods[-1]}" if len(periods) >= 2 else (periods[0] if periods else "")

        result = ContinuousBuildingMonitoringResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            building_name=input_data.building_name,
            monitoring_period=monitoring_period,
            data_points_ingested=len(input_data.current_data) + len(input_data.bms_readings),
            cusum_results=self._cusum_results,
            cusum_in_control=cusum_in_control,
            cumulative_deviation_kwh=round(cumulative_dev, 2),
            alerts=self._alerts,
            critical_alerts=sum(1 for a in self._alerts if a.severity == AlertSeverity.CRITICAL),
            high_alerts=sum(1 for a in self._alerts if a.severity == AlertSeverity.HIGH),
            dashboard=self._dashboard,
            trends=self._trends,
            eui_current=round(eui, 2),
            benchmark_status=self._assess_benchmark(eui, input_data.building_type),
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Continuous monitoring workflow %s completed in %.2fs: "
            "%d alerts, CUSUM %s, EUI=%.1f kWh/m2",
            self.workflow_id, elapsed, len(self._alerts),
            "in_control" if cusum_in_control else "OUT_OF_CONTROL", eui,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Data Ingestion
    # -------------------------------------------------------------------------

    async def _phase_data_ingestion(
        self, input_data: ContinuousBuildingMonitoringInput
    ) -> PhaseResult:
        """Ingest BMS, smart meter, and utility bill data."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Validate and count data points
        energy_points = len(input_data.current_data)
        bms_points = len(input_data.bms_readings)

        if energy_points == 0:
            warnings.append("No current energy data points provided")
        if bms_points == 0:
            warnings.append("No BMS readings provided")

        # Data quality checks
        suspect_bms = sum(1 for r in input_data.bms_readings if r.quality != "good")
        zero_consumption = sum(1 for d in input_data.current_data if d.consumption_kwh == 0)

        if suspect_bms > 0:
            warnings.append(f"{suspect_bms} BMS readings have suspect/bad quality")
        if zero_consumption > energy_points * 0.3 and energy_points > 0:
            warnings.append(f"{zero_consumption} data points have zero consumption")

        # Source breakdown
        source_counts: Dict[str, int] = {}
        for dp in input_data.current_data:
            src = dp.source.value
            source_counts[src] = source_counts.get(src, 0) + 1

        # Period range
        periods = sorted({d.period for d in input_data.current_data if d.period})
        period_range = f"{periods[0]} to {periods[-1]}" if len(periods) >= 2 else ""

        # Total consumption
        total_kwh = sum(d.consumption_kwh for d in input_data.current_data)
        total_cost = sum(d.cost_eur for d in input_data.current_data)

        outputs["energy_data_points"] = energy_points
        outputs["bms_readings"] = bms_points
        outputs["source_breakdown"] = source_counts
        outputs["period_range"] = period_range
        outputs["periods_count"] = len(periods)
        outputs["total_consumption_kwh"] = round(total_kwh, 2)
        outputs["total_cost_eur"] = round(total_cost, 2)
        outputs["suspect_bms_readings"] = suspect_bms
        outputs["zero_consumption_points"] = zero_consumption

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 DataIngestion: %d energy + %d BMS points, %.0f kWh total",
            energy_points, bms_points, total_kwh,
        )
        return PhaseResult(
            phase_name="data_ingestion", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Performance Analysis
    # -------------------------------------------------------------------------

    async def _phase_performance_analysis(
        self, input_data: ContinuousBuildingMonitoringInput
    ) -> PhaseResult:
        """EUI tracking, deviation detection, CUSUM analysis."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        baseline = input_data.baseline
        floor_area = input_data.total_floor_area_sqm

        # Group data by period for CUSUM
        period_data: Dict[str, Dict[str, float]] = {}
        for dp in input_data.current_data:
            if not dp.period:
                continue
            if dp.period not in period_data:
                period_data[dp.period] = {"kwh": 0.0, "hdd": 0.0, "cdd": 0.0, "cost": 0.0}
            period_data[dp.period]["kwh"] += dp.consumption_kwh
            period_data[dp.period]["hdd"] = max(period_data[dp.period]["hdd"], dp.hdd)
            period_data[dp.period]["cdd"] = max(period_data[dp.period]["cdd"], dp.cdd)
            period_data[dp.period]["cost"] += dp.cost_eur

        # CUSUM analysis per ISO 50006
        std_dev = baseline.std_deviation if baseline.std_deviation > 0 else (
            baseline.baseline_kwh_per_month * 0.10
        )
        control_limit = CUSUM_CONTROL_LIMIT_SIGMA * std_dev
        cumulative_sum = 0.0

        for period in sorted(period_data.keys()):
            data = period_data[period]
            actual = data["kwh"]

            # Expected from baseline regression
            expected = baseline.baseline_intercept
            if baseline.baseline_kwh_per_hdd > 0:
                expected += baseline.baseline_kwh_per_hdd * data["hdd"]
            if baseline.baseline_kwh_per_cdd > 0:
                expected += baseline.baseline_kwh_per_cdd * data["cdd"]
            if expected <= 0:
                expected = baseline.baseline_kwh_per_month

            deviation = actual - expected
            deviation_pct = (deviation / expected * 100) if expected > 0 else 0.0
            cumulative_sum += deviation
            in_control = abs(cumulative_sum) <= control_limit

            self._cusum_results.append(CUSUMResult(
                period=period,
                actual_kwh=round(actual, 2),
                expected_kwh=round(expected, 2),
                deviation_kwh=round(deviation, 2),
                cusum_kwh=round(cumulative_sum, 2),
                control_limit_upper=round(control_limit, 2),
                control_limit_lower=round(-control_limit, 2),
                in_control=in_control,
                deviation_pct=round(deviation_pct, 2),
            ))

            if not in_control:
                warnings.append(f"CUSUM out of control in period {period}: {cumulative_sum:.0f} kWh")

        # EUI tracking
        total_kwh = sum(d["kwh"] for d in period_data.values())
        months = len(period_data)
        annual_factor = 12.0 / max(months, 1)
        annualised_kwh = total_kwh * annual_factor
        eui = annualised_kwh / floor_area if floor_area > 0 else 0.0

        outputs["periods_analysed"] = months
        outputs["total_consumption_kwh"] = round(total_kwh, 2)
        outputs["annualised_kwh"] = round(annualised_kwh, 2)
        outputs["eui_kwh_per_sqm"] = round(eui, 2)
        outputs["cusum_final"] = round(cumulative_sum, 2)
        outputs["cusum_in_control"] = all(c.in_control for c in self._cusum_results)
        outputs["max_monthly_deviation_pct"] = round(
            max((abs(c.deviation_pct) for c in self._cusum_results), default=0.0), 2
        )
        outputs["control_limit_kwh"] = round(control_limit, 2)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 PerformanceAnalysis: %d periods, EUI=%.1f kWh/m2, "
            "CUSUM=%.0f kWh, control=%s",
            months, eui, cumulative_sum,
            "YES" if outputs["cusum_in_control"] else "NO",
        )
        return PhaseResult(
            phase_name="performance_analysis", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Anomaly Detection
    # -------------------------------------------------------------------------

    async def _phase_anomaly_detection(
        self, input_data: ContinuousBuildingMonitoringInput
    ) -> PhaseResult:
        """Generate alerts for energy spikes, HVAC faults, anomalies."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        threshold = input_data.alert_threshold_pct

        now_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Alert 1: CUSUM exceedance
        for cusum in self._cusum_results:
            if not cusum.in_control:
                direction = "over" if cusum.cusum_kwh > 0 else "under"
                self._alerts.append(MonitoringAlert(
                    timestamp=now_str,
                    severity=AlertSeverity.HIGH,
                    category=AlertCategory.CUSUM_EXCEEDANCE,
                    title=f"CUSUM exceedance in {cusum.period}",
                    description=(
                        f"Cumulative sum of {cusum.cusum_kwh:.0f} kWh exceeds control "
                        f"limit of {cusum.control_limit_upper:.0f} kWh ({direction}-consuming)."
                    ),
                    metric_name="cusum_kwh",
                    metric_value=cusum.cusum_kwh,
                    threshold_value=cusum.control_limit_upper,
                    recommended_action="Investigate root cause of consumption drift",
                ))

        # Alert 2: Monthly deviation above threshold
        for cusum in self._cusum_results:
            if abs(cusum.deviation_pct) > threshold:
                severity = (
                    AlertSeverity.CRITICAL if abs(cusum.deviation_pct) > threshold * 2
                    else AlertSeverity.HIGH if abs(cusum.deviation_pct) > threshold * 1.5
                    else AlertSeverity.MEDIUM
                )
                self._alerts.append(MonitoringAlert(
                    timestamp=now_str,
                    severity=severity,
                    category=AlertCategory.ENERGY_SPIKE if cusum.deviation_kwh > 0 else AlertCategory.CONSUMPTION_DRIFT,
                    title=f"Energy deviation {cusum.deviation_pct:+.1f}% in {cusum.period}",
                    description=(
                        f"Actual {cusum.actual_kwh:.0f} kWh vs expected {cusum.expected_kwh:.0f} kWh "
                        f"({cusum.deviation_pct:+.1f}% deviation)."
                    ),
                    metric_name="monthly_deviation_pct",
                    metric_value=cusum.deviation_pct,
                    threshold_value=threshold,
                    recommended_action="Review sub-metering data and operational schedules",
                ))

        # Alert 3: Baseload anomalies from BMS data
        if input_data.bms_readings:
            power_readings = [
                r for r in input_data.bms_readings
                if r.point_type == "power" and r.quality == "good"
            ]
            if power_readings:
                power_values = [r.value for r in power_readings]
                if power_values:
                    min_power = min(power_values)
                    max_power = max(power_values)
                    avg_power = statistics.mean(power_values)
                    baseload_ratio = min_power / max_power if max_power > 0 else 0.0

                    thresholds = BASELOAD_RATIO_THRESHOLDS.get(
                        input_data.building_type, BASELOAD_RATIO_THRESHOLDS["office"]
                    )
                    if baseload_ratio > thresholds["poor"]:
                        self._alerts.append(MonitoringAlert(
                            timestamp=now_str,
                            severity=AlertSeverity.MEDIUM,
                            category=AlertCategory.BASELOAD_ANOMALY,
                            title="High baseload ratio detected",
                            description=(
                                f"Baseload ratio {baseload_ratio:.2f} exceeds poor threshold "
                                f"{thresholds['poor']:.2f}. Building consumes significant energy "
                                f"out of hours."
                            ),
                            metric_name="baseload_ratio",
                            metric_value=round(baseload_ratio, 3),
                            threshold_value=thresholds["poor"],
                            recommended_action="Review out-of-hours operation and time schedules",
                        ))

        # Alert 4: Temperature deviations from BMS
        temp_readings = [
            r for r in input_data.bms_readings
            if r.point_type == "temperature" and r.quality == "good" and r.zone
        ]
        setpoints = TEMPERATURE_SETPOINTS.get(
            input_data.building_type, TEMPERATURE_SETPOINTS["office"]
        )
        zones_checked: set = set()
        for reading in temp_readings:
            if reading.zone in zones_checked:
                continue
            zones_checked.add(reading.zone)

            zone_temps = [r.value for r in temp_readings if r.zone == reading.zone]
            if zone_temps:
                avg_temp = statistics.mean(zone_temps)
                if avg_temp < setpoints["heating_min"] - 2.0:
                    self._alerts.append(MonitoringAlert(
                        timestamp=now_str,
                        severity=AlertSeverity.MEDIUM,
                        category=AlertCategory.TEMPERATURE_DEVIATION,
                        title=f"Under-heating in zone {reading.zone}",
                        description=(
                            f"Average temperature {avg_temp:.1f}C is below heating setpoint "
                            f"minimum {setpoints['heating_min']:.1f}C."
                        ),
                        affected_zone=reading.zone,
                        metric_name="zone_temperature_c",
                        metric_value=round(avg_temp, 1),
                        threshold_value=setpoints["heating_min"],
                        recommended_action="Check HVAC operation and zone scheduling",
                    ))
                elif avg_temp > setpoints["cooling_max"] + 2.0:
                    self._alerts.append(MonitoringAlert(
                        timestamp=now_str,
                        severity=AlertSeverity.MEDIUM,
                        category=AlertCategory.TEMPERATURE_DEVIATION,
                        title=f"Over-heating in zone {reading.zone}",
                        description=(
                            f"Average temperature {avg_temp:.1f}C is above cooling setpoint "
                            f"maximum {setpoints['cooling_max']:.1f}C."
                        ),
                        affected_zone=reading.zone,
                        metric_name="zone_temperature_c",
                        metric_value=round(avg_temp, 1),
                        threshold_value=setpoints["cooling_max"],
                        recommended_action="Check cooling system operation and setpoints",
                    ))

        # Alert 5: BMS quality issues
        bad_readings = sum(1 for r in input_data.bms_readings if r.quality == "bad")
        if bad_readings > len(input_data.bms_readings) * 0.05 and input_data.bms_readings:
            self._alerts.append(MonitoringAlert(
                timestamp=now_str,
                severity=AlertSeverity.LOW,
                category=AlertCategory.METER_ERROR,
                title="BMS data quality degradation",
                description=f"{bad_readings} BMS readings flagged as bad quality.",
                metric_name="bad_reading_count",
                metric_value=float(bad_readings),
                threshold_value=len(input_data.bms_readings) * 0.05,
                recommended_action="Check BMS sensor calibration and wiring",
            ))

        outputs["total_alerts"] = len(self._alerts)
        outputs["critical_alerts"] = sum(1 for a in self._alerts if a.severity == AlertSeverity.CRITICAL)
        outputs["high_alerts"] = sum(1 for a in self._alerts if a.severity == AlertSeverity.HIGH)
        outputs["medium_alerts"] = sum(1 for a in self._alerts if a.severity == AlertSeverity.MEDIUM)
        outputs["low_alerts"] = sum(1 for a in self._alerts if a.severity == AlertSeverity.LOW)
        outputs["alert_categories"] = list(set(a.category.value for a in self._alerts))

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 AnomalyDetection: %d alerts (%d critical, %d high)",
            len(self._alerts), outputs["critical_alerts"], outputs["high_alerts"],
        )
        return PhaseResult(
            phase_name="anomaly_detection", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Trend Reporting
    # -------------------------------------------------------------------------

    async def _phase_trend_reporting(
        self, input_data: ContinuousBuildingMonitoringInput
    ) -> PhaseResult:
        """Generate monthly/quarterly dashboards and YoY comparison."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        floor_area = input_data.total_floor_area_sqm

        # Current period aggregates
        total_kwh = sum(d.consumption_kwh for d in input_data.current_data)
        total_cost = sum(d.cost_eur for d in input_data.current_data)
        total_hdd = sum(d.hdd for d in input_data.current_data)
        total_cdd = sum(d.cdd for d in input_data.current_data)

        demands = [d.demand_kw for d in input_data.current_data if d.demand_kw > 0]
        peak_demand = max(demands) if demands else 0.0
        baseload = min(demands) if demands else 0.0
        baseload_ratio = baseload / peak_demand if peak_demand > 0 else 0.0

        # Weather normalisation
        baseline = input_data.baseline
        if baseline.baseline_kwh_per_hdd > 0 and total_hdd > 0:
            normalised_kwh = (
                baseline.baseline_intercept * len(set(d.period for d in input_data.current_data if d.period))
                + baseline.baseline_kwh_per_hdd * total_hdd
                + baseline.baseline_kwh_per_cdd * total_cdd
            )
        else:
            normalised_kwh = total_kwh

        months = len(set(d.period for d in input_data.current_data if d.period))
        annual_factor = 12.0 / max(months, 1)
        eui = (total_kwh * annual_factor) / floor_area if floor_area > 0 else 0.0

        periods = sorted({d.period for d in input_data.current_data if d.period})
        period_label = f"{periods[0]}-{periods[-1]}" if len(periods) >= 2 else (periods[0] if periods else "")

        # Previous period comparison
        prev_kwh = sum(d.consumption_kwh for d in input_data.previous_period_data)
        prev_cost = sum(d.cost_eur for d in input_data.previous_period_data)

        # Year-on-year comparison
        yoy_kwh = sum(d.consumption_kwh for d in input_data.previous_year_data)
        yoy_cost = sum(d.cost_eur for d in input_data.previous_year_data)

        # Build trends
        self._trends = []

        # EUI trend
        prev_eui = (prev_kwh * annual_factor) / floor_area if floor_area > 0 and prev_kwh > 0 else 0.0
        eui_change = ((eui - prev_eui) / prev_eui * 100) if prev_eui > 0 else 0.0
        yoy_eui = (yoy_kwh * annual_factor) / floor_area if floor_area > 0 and yoy_kwh > 0 else 0.0
        yoy_eui_change = ((eui - yoy_eui) / yoy_eui * 100) if yoy_eui > 0 else 0.0

        self._trends.append(TrendMetric(
            metric_name="eui_kwh_per_sqm",
            current_value=round(eui, 2),
            previous_value=round(prev_eui, 2),
            change_pct=round(eui_change, 2),
            direction=self._trend_direction(eui_change, invert=True),
            period_current=period_label,
            yoy_change_pct=round(yoy_eui_change, 2),
        ))

        # Consumption trend
        cons_change = ((total_kwh - prev_kwh) / prev_kwh * 100) if prev_kwh > 0 else 0.0
        yoy_cons_change = ((total_kwh - yoy_kwh) / yoy_kwh * 100) if yoy_kwh > 0 else 0.0
        self._trends.append(TrendMetric(
            metric_name="total_consumption_kwh",
            current_value=round(total_kwh, 2),
            previous_value=round(prev_kwh, 2),
            change_pct=round(cons_change, 2),
            direction=self._trend_direction(cons_change, invert=True),
            period_current=period_label,
            yoy_change_pct=round(yoy_cons_change, 2),
        ))

        # Cost trend
        cost_change = ((total_cost - prev_cost) / prev_cost * 100) if prev_cost > 0 else 0.0
        yoy_cost_change = ((total_cost - yoy_cost) / yoy_cost * 100) if yoy_cost > 0 else 0.0
        self._trends.append(TrendMetric(
            metric_name="total_cost_eur",
            current_value=round(total_cost, 2),
            previous_value=round(prev_cost, 2),
            change_pct=round(cost_change, 2),
            direction=self._trend_direction(cost_change, invert=True),
            period_current=period_label,
            yoy_change_pct=round(yoy_cost_change, 2),
        ))

        # Peak demand trend
        prev_demands = [d.demand_kw for d in input_data.previous_period_data if d.demand_kw > 0]
        prev_peak = max(prev_demands) if prev_demands else 0.0
        peak_change = ((peak_demand - prev_peak) / prev_peak * 100) if prev_peak > 0 else 0.0
        self._trends.append(TrendMetric(
            metric_name="peak_demand_kw",
            current_value=round(peak_demand, 2),
            previous_value=round(prev_peak, 2),
            change_pct=round(peak_change, 2),
            direction=self._trend_direction(peak_change, invert=True),
            period_current=period_label,
        ))

        self._dashboard = PerformanceDashboard(
            period=period_label,
            eui_kwh_per_sqm=round(eui, 2),
            total_consumption_kwh=round(total_kwh, 2),
            total_cost_eur=round(total_cost, 2),
            peak_demand_kw=round(peak_demand, 2),
            baseload_kw=round(baseload, 2),
            baseload_ratio=round(baseload_ratio, 3),
            hdd_total=round(total_hdd, 1),
            cdd_total=round(total_cdd, 1),
            weather_normalised_kwh=round(normalised_kwh, 2),
            active_alerts=len(self._alerts),
            trends=self._trends,
        )

        outputs["period"] = period_label
        outputs["eui_kwh_per_sqm"] = round(eui, 2)
        outputs["total_consumption_kwh"] = round(total_kwh, 2)
        outputs["total_cost_eur"] = round(total_cost, 2)
        outputs["peak_demand_kw"] = round(peak_demand, 2)
        outputs["baseload_kw"] = round(baseload, 2)
        outputs["baseload_ratio"] = round(baseload_ratio, 3)
        outputs["weather_normalised_kwh"] = round(normalised_kwh, 2)
        outputs["eui_change_vs_previous_pct"] = round(eui_change, 2)
        outputs["eui_change_yoy_pct"] = round(yoy_eui_change, 2)
        outputs["consumption_change_pct"] = round(cons_change, 2)
        outputs["cost_change_pct"] = round(cost_change, 2)
        outputs["trends_count"] = len(self._trends)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 TrendReporting: EUI=%.1f kWh/m2 (%+.1f%% vs prev, %+.1f%% YoY)",
            eui, eui_change, yoy_eui_change,
        )
        return PhaseResult(
            phase_name="trend_reporting", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _trend_direction(change_pct: float, invert: bool = False) -> TrendDirection:
        """Determine trend direction (for energy, decrease = improving)."""
        if abs(change_pct) < 2.0:
            return TrendDirection.STABLE
        if invert:
            return TrendDirection.IMPROVING if change_pct < 0 else TrendDirection.DETERIORATING
        return TrendDirection.IMPROVING if change_pct > 0 else TrendDirection.DETERIORATING

    @staticmethod
    def _assess_benchmark(eui: float, building_type: str) -> str:
        """Assess EUI against benchmarks."""
        benchmarks = EUI_BENCHMARKS.get(building_type, EUI_BENCHMARKS.get("office", {}))
        if not benchmarks or eui <= 0:
            return "insufficient_data"
        if eui <= benchmarks.get("best", 95):
            return "best_practice"
        elif eui <= benchmarks.get("good", 128):
            return "good_practice"
        elif eui <= benchmarks.get("typical", 230):
            return "typical"
        return "below_typical"

    def _compute_provenance(self, result: ContinuousBuildingMonitoringResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
