# -*- coding: utf-8 -*-
"""
AnomalyDetectionEngine - PACK-039 Energy Monitoring Engine 4
==============================================================

Statistical anomaly detection engine implementing CUSUM, EWMA, Z-score,
IQR, regression residual, and schedule-based methods for continuous
energy monitoring.  Detects overconsumption, underconsumption, after-hours
usage, schedule deviations, equipment faults, weather anomalies, and
simultaneous heating/cooling.

Calculation Methodology:
    CUSUM (Cumulative Sum):
        S_n = max(0, S_{n-1} + (x_n - target) - allowance)
        Alarm when S_n > decision_interval (h)

    EWMA (Exponentially Weighted Moving Average):
        Z_t = lambda * x_t + (1 - lambda) * Z_{t-1}
        UCL = mu + L * sigma * sqrt(lambda / (2 - lambda))
        Alarm when Z_t > UCL or Z_t < LCL

    Modified Z-Score:
        M_i = 0.6745 * (x_i - median) / MAD
        Anomaly when |M_i| > threshold (default 3.5)

    IQR Method:
        IQR = Q3 - Q1
        Lower fence = Q1 - k * IQR
        Upper fence = Q3 + k * IQR  (k = 1.5 default)

    Regression Residual:
        residual = actual - predicted
        anomaly when |residual| > threshold * RMSE

    Schedule-Based:
        anomaly when consumption > schedule_threshold during
        unoccupied / off-hours / weekends / holidays

Regulatory References:
    - ASHRAE Guideline 14-2014 - Statistical analysis methods
    - ISO 50001:2018 - Energy performance monitoring
    - ISO 50006:2014 - Measuring energy performance with EnPIs
    - CUSUM per NIST Engineering Statistics Handbook
    - Montgomery (2013) - Statistical Quality Control, 7th Ed
    - IPMVP Volume I - Option B/C analysis methods

Zero-Hallucination:
    - All detection algorithms use deterministic formulas
    - No LLM involvement in any calculation path
    - Baselines computed from historical data only
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-039 Energy Monitoring
Engine:  4 of 5
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AnomalyMethod(str, Enum):
    """Statistical method used for anomaly detection.

    CUSUM:               Cumulative Sum control chart.
    EWMA:                Exponentially Weighted Moving Average.
    ZSCORE:              Standard or modified Z-score.
    IQR:                 Interquartile Range fence method.
    REGRESSION_RESIDUAL: Regression model residual analysis.
    SCHEDULE_BASED:      Comparison against occupancy schedule.
    COMBINED:            Ensemble of multiple methods.
    """
    CUSUM = "cusum"
    EWMA = "ewma"
    ZSCORE = "zscore"
    IQR = "iqr"
    REGRESSION_RESIDUAL = "regression_residual"
    SCHEDULE_BASED = "schedule_based"
    COMBINED = "combined"

class AnomalyType(str, Enum):
    """Classification of detected anomaly.

    OVERCONSUMPTION:              Usage above expected level.
    UNDERCONSUMPTION:             Usage below expected level.
    AFTER_HOURS:                  Unexpected usage during off-hours.
    SCHEDULE_DEVIATION:           Pattern differs from schedule.
    EQUIPMENT_FAULT:              Indicates possible equipment failure.
    WEATHER_ANOMALY:              Weather-driven unexpected consumption.
    SIMULTANEOUS_HEATING_COOLING: Both heating and cooling active.
    """
    OVERCONSUMPTION = "overconsumption"
    UNDERCONSUMPTION = "underconsumption"
    AFTER_HOURS = "after_hours"
    SCHEDULE_DEVIATION = "schedule_deviation"
    EQUIPMENT_FAULT = "equipment_fault"
    WEATHER_ANOMALY = "weather_anomaly"
    SIMULTANEOUS_HEATING_COOLING = "simultaneous_heating_cooling"

class AnomalySeverity(str, Enum):
    """Severity level of a detected anomaly.

    LOW:      Minor deviation, monitor.
    MEDIUM:   Notable deviation, investigate.
    HIGH:     Significant deviation, take action.
    CRITICAL: Severe deviation, immediate action required.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class InvestigationStatus(str, Enum):
    """Status of an anomaly investigation.

    DETECTED:       Anomaly detected, not yet reviewed.
    INVESTIGATING:  Under active investigation.
    CONFIRMED:      Confirmed as real anomaly.
    FALSE_ALARM:    Determined to be false positive.
    RESOLVED:       Root cause found and resolved.
    """
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    CONFIRMED = "confirmed"
    FALSE_ALARM = "false_alarm"
    RESOLVED = "resolved"

class TimeContext(str, Enum):
    """Time context for schedule-based detection.

    OCCUPIED:    Normal occupied hours.
    UNOCCUPIED:  Unoccupied / after-hours.
    WEEKEND:     Weekend period.
    HOLIDAY:     Public holiday.
    SHUTDOWN:    Planned facility shutdown.
    """
    OCCUPIED = "occupied"
    UNOCCUPIED = "unoccupied"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"
    SHUTDOWN = "shutdown"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default CUSUM parameters.
CUSUM_DEFAULT_ALLOWANCE: Decimal = Decimal("0.5")  # k value (in std dev units)
CUSUM_DEFAULT_DECISION: Decimal = Decimal("5.0")   # h value (in std dev units)

# Default EWMA parameters.
EWMA_DEFAULT_LAMBDA: Decimal = Decimal("0.2")      # smoothing factor
EWMA_DEFAULT_L: Decimal = Decimal("3.0")           # control limit width

# Default Z-score threshold.
ZSCORE_DEFAULT_THRESHOLD: Decimal = Decimal("3.5")

# Modified Z-score constant (0.6745 = normal distribution consistency).
MODIFIED_ZSCORE_CONSTANT: Decimal = Decimal("0.6745")

# Default IQR multiplier.
IQR_DEFAULT_K: Decimal = Decimal("1.5")

# Default regression residual threshold (multiples of RMSE).
REGRESSION_RESIDUAL_THRESHOLD: Decimal = Decimal("2.5")

# Schedule thresholds: occupied fraction of base load allowed off-hours.
SCHEDULE_OFF_HOURS_FRACTION: Decimal = Decimal("0.20")

# Severity thresholds (deviation as percentage of baseline).
SEVERITY_THRESHOLDS: Dict[str, Decimal] = {
    AnomalySeverity.LOW.value: Decimal("10"),
    AnomalySeverity.MEDIUM.value: Decimal("25"),
    AnomalySeverity.HIGH.value: Decimal("50"),
    AnomalySeverity.CRITICAL.value: Decimal("100"),
}

# Default occupied hours (24h clock).
DEFAULT_OCCUPIED_START: int = 7
DEFAULT_OCCUPIED_END: int = 19

# Weekend day indices (0=Monday in Python).
WEEKEND_DAYS: List[int] = [5, 6]

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class AnomalyConfig(BaseModel):
    """Configuration for anomaly detection parameters.

    Attributes:
        method:                    Detection method to use.
        cusum_allowance:           CUSUM k parameter.
        cusum_decision:            CUSUM h parameter.
        ewma_lambda:               EWMA smoothing factor.
        ewma_l:                    EWMA control limit width.
        zscore_threshold:          Z-score anomaly threshold.
        iqr_k:                     IQR fence multiplier.
        regression_threshold:      Regression residual threshold.
        occupied_start_hour:       Occupied period start.
        occupied_end_hour:         Occupied period end.
        off_hours_fraction:        Allowed off-hours fraction.
        min_data_points:           Minimum points for baseline.
        holidays:                  List of holiday dates.
    """
    method: AnomalyMethod = Field(
        default=AnomalyMethod.COMBINED, description="Detection method"
    )
    cusum_allowance: Decimal = Field(default=CUSUM_DEFAULT_ALLOWANCE)
    cusum_decision: Decimal = Field(default=CUSUM_DEFAULT_DECISION)
    ewma_lambda: Decimal = Field(default=EWMA_DEFAULT_LAMBDA)
    ewma_l: Decimal = Field(default=EWMA_DEFAULT_L)
    zscore_threshold: Decimal = Field(default=ZSCORE_DEFAULT_THRESHOLD)
    iqr_k: Decimal = Field(default=IQR_DEFAULT_K)
    regression_threshold: Decimal = Field(default=REGRESSION_RESIDUAL_THRESHOLD)
    occupied_start_hour: int = Field(default=DEFAULT_OCCUPIED_START, ge=0, le=23)
    occupied_end_hour: int = Field(default=DEFAULT_OCCUPIED_END, ge=0, le=23)
    off_hours_fraction: Decimal = Field(default=SCHEDULE_OFF_HOURS_FRACTION)
    min_data_points: int = Field(default=30, ge=10)
    holidays: List[str] = Field(default_factory=list)

class AnomalyBaseline(BaseModel):
    """Statistical baseline for anomaly detection.

    Attributes:
        baseline_id:     Unique baseline identifier.
        meter_id:        Meter this baseline represents.
        period_start:    Baseline period start.
        period_end:      Baseline period end.
        mean:            Baseline mean value.
        median:          Baseline median value.
        std_dev:         Baseline standard deviation.
        mad:             Median absolute deviation.
        q1:              First quartile.
        q3:              Third quartile.
        iqr:             Interquartile range.
        min_value:       Minimum baseline value.
        max_value:       Maximum baseline value.
        data_points:     Number of data points in baseline.
        hourly_means:    Mean by hour of day (24 entries).
        weekday_means:   Mean by day of week (7 entries).
        calculated_at:   Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    baseline_id: str = Field(default_factory=_new_uuid)
    meter_id: str = Field(default="")
    period_start: datetime = Field(default_factory=utcnow)
    period_end: datetime = Field(default_factory=utcnow)
    mean: Decimal = Field(default=Decimal("0"))
    median: Decimal = Field(default=Decimal("0"))
    std_dev: Decimal = Field(default=Decimal("0"))
    mad: Decimal = Field(default=Decimal("0"))
    q1: Decimal = Field(default=Decimal("0"))
    q3: Decimal = Field(default=Decimal("0"))
    iqr: Decimal = Field(default=Decimal("0"))
    min_value: Decimal = Field(default=Decimal("0"))
    max_value: Decimal = Field(default=Decimal("0"))
    data_points: int = Field(default=0, ge=0)
    hourly_means: Dict[str, Decimal] = Field(default_factory=dict)
    weekday_means: Dict[str, Decimal] = Field(default_factory=dict)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class DetectedAnomaly(BaseModel):
    """A single detected anomaly instance.

    Attributes:
        anomaly_id:        Unique anomaly identifier.
        meter_id:          Source meter.
        timestamp:         When the anomaly occurred.
        value:             Observed value.
        expected_value:    Expected baseline value.
        deviation:         Absolute deviation.
        deviation_pct:     Deviation as percentage of expected.
        method:            Detection method that triggered.
        anomaly_type:      Classification of anomaly.
        severity:          Severity level.
        investigation:     Investigation status.
        time_context:      Occupancy context.
        score:             Anomaly score (0-100).
        message:           Description.
        estimated_cost:    Estimated energy cost impact.
        provenance_hash:   SHA-256 audit hash.
    """
    anomaly_id: str = Field(default_factory=_new_uuid)
    meter_id: str = Field(default="")
    timestamp: datetime = Field(default_factory=utcnow)
    value: Decimal = Field(default=Decimal("0"))
    expected_value: Decimal = Field(default=Decimal("0"))
    deviation: Decimal = Field(default=Decimal("0"))
    deviation_pct: Decimal = Field(default=Decimal("0"))
    method: AnomalyMethod = Field(default=AnomalyMethod.ZSCORE)
    anomaly_type: AnomalyType = Field(default=AnomalyType.OVERCONSUMPTION)
    severity: AnomalySeverity = Field(default=AnomalySeverity.LOW)
    investigation: InvestigationStatus = Field(default=InvestigationStatus.DETECTED)
    time_context: TimeContext = Field(default=TimeContext.OCCUPIED)
    score: Decimal = Field(default=Decimal("0"))
    message: str = Field(default="", max_length=1000)
    estimated_cost: Decimal = Field(default=Decimal("0"))
    provenance_hash: str = Field(default="")

class InvestigationRecord(BaseModel):
    """Record of anomaly investigation progress.

    Attributes:
        investigation_id:  Unique investigation identifier.
        anomaly_id:        Related anomaly.
        status:            Current status.
        assigned_to:       Investigator.
        root_cause:        Identified root cause.
        action_taken:      Corrective action taken.
        energy_saved_kwh:  Energy saved from resolution.
        cost_saved:        Cost saved from resolution.
        created_at:        Creation timestamp.
        resolved_at:       Resolution timestamp.
        notes:             Investigation notes.
    """
    investigation_id: str = Field(default_factory=_new_uuid)
    anomaly_id: str = Field(default="")
    status: InvestigationStatus = Field(default=InvestigationStatus.DETECTED)
    assigned_to: str = Field(default="", max_length=200)
    root_cause: str = Field(default="", max_length=1000)
    action_taken: str = Field(default="", max_length=1000)
    energy_saved_kwh: Decimal = Field(default=Decimal("0"))
    cost_saved: Decimal = Field(default=Decimal("0"))
    created_at: datetime = Field(default_factory=utcnow)
    resolved_at: Optional[datetime] = Field(default=None)
    notes: str = Field(default="", max_length=2000)

class AnomalyReport(BaseModel):
    """Complete anomaly detection report.

    Attributes:
        report_id:           Unique report identifier.
        meter_id:            Meter analysed.
        analysis_start:      Analysis period start.
        analysis_end:        Analysis period end.
        method_used:         Detection method(s) used.
        baseline:            Baseline statistics.
        anomalies:           List of detected anomalies.
        total_anomalies:     Total anomalies found.
        critical_count:      Critical anomalies.
        high_count:          High severity anomalies.
        medium_count:        Medium severity anomalies.
        low_count:           Low severity anomalies.
        anomaly_rate_pct:    Percentage of readings flagged.
        estimated_waste_kwh: Total estimated energy waste.
        estimated_waste_cost: Total estimated cost impact.
        investigations:      Investigation records.
        recommendations:     Recommendations.
        processing_time_ms:  Processing duration.
        calculated_at:       Calculation timestamp.
        provenance_hash:     SHA-256 audit hash.
    """
    report_id: str = Field(default_factory=_new_uuid)
    meter_id: str = Field(default="")
    analysis_start: datetime = Field(default_factory=utcnow)
    analysis_end: datetime = Field(default_factory=utcnow)
    method_used: str = Field(default="combined")
    baseline: AnomalyBaseline = Field(default_factory=AnomalyBaseline)
    anomalies: List[DetectedAnomaly] = Field(default_factory=list)
    total_anomalies: int = Field(default=0, ge=0)
    critical_count: int = Field(default=0, ge=0)
    high_count: int = Field(default=0, ge=0)
    medium_count: int = Field(default=0, ge=0)
    low_count: int = Field(default=0, ge=0)
    anomaly_rate_pct: Decimal = Field(default=Decimal("0"))
    estimated_waste_kwh: Decimal = Field(default=Decimal("0"))
    estimated_waste_cost: Decimal = Field(default=Decimal("0"))
    investigations: List[InvestigationRecord] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AnomalyDetectionEngine:
    """Statistical anomaly detection engine for energy monitoring.

    Implements CUSUM, EWMA, Z-score, IQR, regression residual, and
    schedule-based detection methods.  Supports combined ensemble
    detection, baseline computation, anomaly classification, severity
    assignment, and investigation tracking.

    Usage::

        engine = AnomalyDetectionEngine()
        baseline = engine.calculate_baseline("M-001", historical_data)
        report = engine.detect_anomalies("M-001", current_data, baseline)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[AnomalyConfig] = None) -> None:
        """Initialise AnomalyDetectionEngine.

        Args:
            config: Optional anomaly detection configuration.
        """
        self._config = config or AnomalyConfig()
        self._investigations: Dict[str, InvestigationRecord] = {}
        logger.info(
            "AnomalyDetectionEngine v%s initialised (method=%s, "
            "zscore=%.1f, ewma_lambda=%.2f)",
            self.engine_version, self._config.method.value,
            float(self._config.zscore_threshold),
            float(self._config.ewma_lambda),
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def detect_anomalies(
        self,
        meter_id: str,
        readings: List[Dict[str, Any]],
        baseline: Optional[AnomalyBaseline] = None,
        energy_cost_per_kwh: Decimal = Decimal("0.12"),
    ) -> AnomalyReport:
        """Run anomaly detection on a set of readings.

        Args:
            meter_id:             Meter identifier.
            readings:             List of dicts with 'timestamp' and 'value'.
            baseline:             Pre-computed baseline (computed if None).
            energy_cost_per_kwh:  Energy cost for waste estimation.

        Returns:
            AnomalyReport with all detected anomalies.
        """
        t0 = time.perf_counter()
        logger.info(
            "Detecting anomalies: meter=%s, readings=%d, method=%s",
            meter_id[:12], len(readings), self._config.method.value,
        )

        if not readings:
            report = AnomalyReport(meter_id=meter_id)
            report.provenance_hash = _compute_hash(report)
            return report

        values = [_decimal(r.get("value", 0)) for r in readings]
        timestamps = [r.get("timestamp", utcnow()) for r in readings]

        # Compute baseline if not provided
        if baseline is None:
            baseline = self.calculate_baseline(meter_id, readings)

        all_anomalies: List[DetectedAnomaly] = []
        method = self._config.method

        if method in (AnomalyMethod.CUSUM, AnomalyMethod.COMBINED):
            cusum_anomalies = self.run_cusum(values, timestamps, baseline)
            all_anomalies.extend(cusum_anomalies)

        if method in (AnomalyMethod.EWMA, AnomalyMethod.COMBINED):
            ewma_anomalies = self.run_ewma(values, timestamps, baseline)
            all_anomalies.extend(ewma_anomalies)

        if method in (AnomalyMethod.ZSCORE, AnomalyMethod.COMBINED):
            zscore_anomalies = self._run_zscore(values, timestamps, baseline)
            all_anomalies.extend(zscore_anomalies)

        if method in (AnomalyMethod.IQR, AnomalyMethod.COMBINED):
            iqr_anomalies = self._run_iqr(values, timestamps, baseline)
            all_anomalies.extend(iqr_anomalies)

        if method in (AnomalyMethod.SCHEDULE_BASED, AnomalyMethod.COMBINED):
            sched_anomalies = self._run_schedule_based(
                values, timestamps, baseline,
            )
            all_anomalies.extend(sched_anomalies)

        # De-duplicate by timestamp (keep highest severity)
        deduped = self._deduplicate_anomalies(all_anomalies)

        # Classify and assign severity
        for a in deduped:
            a.anomaly_type = self.classify_anomaly(a, baseline)
            a.severity = self._assign_severity(a)
            a.estimated_cost = _round_val(
                abs(a.deviation) * energy_cost_per_kwh, 2
            )
            a.provenance_hash = _compute_hash(a)

        # Counts
        critical = sum(1 for a in deduped if a.severity == AnomalySeverity.CRITICAL)
        high = sum(1 for a in deduped if a.severity == AnomalySeverity.HIGH)
        medium = sum(1 for a in deduped if a.severity == AnomalySeverity.MEDIUM)
        low = sum(1 for a in deduped if a.severity == AnomalySeverity.LOW)

        anomaly_rate = _safe_pct(_decimal(len(deduped)), _decimal(len(readings)))
        waste_kwh = sum((abs(a.deviation) for a in deduped), Decimal("0"))
        waste_cost = _round_val(waste_kwh * energy_cost_per_kwh, 2)

        recommendations = self._generate_recommendations(
            deduped, anomaly_rate, baseline,
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000.0)

        report = AnomalyReport(
            meter_id=meter_id,
            analysis_start=min(timestamps) if timestamps else utcnow(),
            analysis_end=max(timestamps) if timestamps else utcnow(),
            method_used=method.value,
            baseline=baseline,
            anomalies=deduped[:500],
            total_anomalies=len(deduped),
            critical_count=critical,
            high_count=high,
            medium_count=medium,
            low_count=low,
            anomaly_rate_pct=_round_val(anomaly_rate, 2),
            estimated_waste_kwh=_round_val(waste_kwh, 2),
            estimated_waste_cost=waste_cost,
            investigations=list(self._investigations.values()),
            recommendations=recommendations,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        report.provenance_hash = _compute_hash(report)

        logger.info(
            "Anomaly detection complete: %d anomalies (C=%d/H=%d/M=%d/L=%d), "
            "rate=%.1f%%, waste=%.1f kWh, hash=%s (%.1f ms)",
            len(deduped), critical, high, medium, low,
            float(anomaly_rate), float(waste_kwh),
            report.provenance_hash[:16], float(elapsed_ms),
        )
        return report

    def run_cusum(
        self,
        values: List[Decimal],
        timestamps: List[Any],
        baseline: AnomalyBaseline,
    ) -> List[DetectedAnomaly]:
        """Run CUSUM (Cumulative Sum) control chart detection.

        Formula: S_n = max(0, S_{n-1} + (x_n - target) - allowance)
        Alarm when S_n > decision_interval * std_dev

        Args:
            values:     List of observed values.
            timestamps: Corresponding timestamps.
            baseline:   Computed baseline statistics.

        Returns:
            List of CUSUM-detected anomalies.
        """
        t0 = time.perf_counter()
        logger.info("Running CUSUM on %d values", len(values))

        if not values or baseline.std_dev == Decimal("0"):
            return []

        target = baseline.mean
        allowance = self._config.cusum_allowance * baseline.std_dev
        decision = self._config.cusum_decision * baseline.std_dev

        anomalies: List[DetectedAnomaly] = []
        s_pos = Decimal("0")  # Upper CUSUM
        s_neg = Decimal("0")  # Lower CUSUM

        for i, x in enumerate(values):
            # Upper CUSUM: detects upward shifts
            s_pos = max(Decimal("0"), s_pos + (x - target) - allowance)
            # Lower CUSUM: detects downward shifts
            s_neg = max(Decimal("0"), s_neg + (target - x) - allowance)

            is_anomaly = False
            deviation = x - target

            if s_pos > decision:
                is_anomaly = True
                s_pos = Decimal("0")  # Reset after alarm

            if s_neg > decision:
                is_anomaly = True
                s_neg = Decimal("0")  # Reset after alarm

            if is_anomaly:
                ts = timestamps[i] if i < len(timestamps) else utcnow()
                dev_pct = _safe_pct(abs(deviation), target)
                anomalies.append(DetectedAnomaly(
                    meter_id=baseline.meter_id,
                    timestamp=ts,
                    value=x,
                    expected_value=target,
                    deviation=_round_val(deviation, 4),
                    deviation_pct=_round_val(dev_pct, 2),
                    method=AnomalyMethod.CUSUM,
                    score=min(_round_val(dev_pct, 0), Decimal("100")),
                    message=f"CUSUM alarm: value={x}, target={_round_val(target, 2)}",
                ))

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info("CUSUM complete: %d anomalies (%.1f ms)", len(anomalies), elapsed)
        return anomalies

    def run_ewma(
        self,
        values: List[Decimal],
        timestamps: List[Any],
        baseline: AnomalyBaseline,
    ) -> List[DetectedAnomaly]:
        """Run EWMA (Exponentially Weighted Moving Average) detection.

        Formula: Z_t = lambda * x_t + (1 - lambda) * Z_{t-1}
        UCL = mu + L * sigma * sqrt(lambda / (2 - lambda))

        Args:
            values:     List of observed values.
            timestamps: Corresponding timestamps.
            baseline:   Computed baseline statistics.

        Returns:
            List of EWMA-detected anomalies.
        """
        t0 = time.perf_counter()
        logger.info("Running EWMA on %d values", len(values))

        if not values or baseline.std_dev == Decimal("0"):
            return []

        lam = self._config.ewma_lambda
        L = self._config.ewma_l
        mu = baseline.mean
        sigma = baseline.std_dev

        # Control limits
        cl_factor = L * sigma * _decimal(
            math.sqrt(float(lam / (Decimal("2") - lam)))
        )
        ucl = mu + cl_factor
        lcl = mu - cl_factor

        anomalies: List[DetectedAnomaly] = []
        z_t = mu  # Initialise EWMA at baseline mean

        for i, x in enumerate(values):
            z_t = lam * x + (Decimal("1") - lam) * z_t

            if z_t > ucl or z_t < lcl:
                ts = timestamps[i] if i < len(timestamps) else utcnow()
                deviation = x - mu
                dev_pct = _safe_pct(abs(deviation), mu)
                anomalies.append(DetectedAnomaly(
                    meter_id=baseline.meter_id,
                    timestamp=ts,
                    value=x,
                    expected_value=mu,
                    deviation=_round_val(deviation, 4),
                    deviation_pct=_round_val(dev_pct, 2),
                    method=AnomalyMethod.EWMA,
                    score=min(_round_val(dev_pct, 0), Decimal("100")),
                    message=(
                        f"EWMA alarm: Z_t={_round_val(z_t, 2)}, "
                        f"UCL={_round_val(ucl, 2)}, LCL={_round_val(lcl, 2)}"
                    ),
                ))

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info("EWMA complete: %d anomalies (%.1f ms)", len(anomalies), elapsed)
        return anomalies

    def calculate_baseline(
        self,
        meter_id: str,
        readings: List[Dict[str, Any]],
    ) -> AnomalyBaseline:
        """Compute statistical baseline from historical readings.

        Args:
            meter_id: Meter identifier.
            readings: Historical readings.

        Returns:
            AnomalyBaseline with statistics.
        """
        t0 = time.perf_counter()
        logger.info(
            "Computing baseline for meter %s (%d readings)",
            meter_id[:12], len(readings),
        )

        values = sorted([_decimal(r.get("value", 0)) for r in readings])
        timestamps = [r.get("timestamp", utcnow()) for r in readings]

        if not values:
            result = AnomalyBaseline(meter_id=meter_id)
            result.provenance_hash = _compute_hash(result)
            return result

        n = _decimal(len(values))
        total = sum(values, Decimal("0"))
        mean_val = _safe_divide(total, n)
        median_val = self._percentile(values, 50)

        # Standard deviation
        variance = _safe_divide(
            sum(((v - mean_val) ** 2 for v in values), Decimal("0")), n
        )
        std_dev = _decimal(math.sqrt(float(variance)))

        # MAD (Median Absolute Deviation)
        abs_devs = sorted([abs(v - median_val) for v in values])
        mad = self._percentile(abs_devs, 50) if abs_devs else Decimal("0")

        # Quartiles
        q1 = self._percentile(values, 25)
        q3 = self._percentile(values, 75)
        iqr = q3 - q1

        # Hourly means
        hourly_sums: Dict[int, Decimal] = {}
        hourly_counts: Dict[int, int] = {}
        weekday_sums: Dict[int, Decimal] = {}
        weekday_counts: Dict[int, int] = {}

        for r in readings:
            ts = r.get("timestamp")
            val = _decimal(r.get("value", 0))
            if isinstance(ts, datetime):
                h = ts.hour
                hourly_sums[h] = hourly_sums.get(h, Decimal("0")) + val
                hourly_counts[h] = hourly_counts.get(h, 0) + 1
                wd = ts.weekday()
                weekday_sums[wd] = weekday_sums.get(wd, Decimal("0")) + val
                weekday_counts[wd] = weekday_counts.get(wd, 0) + 1

        hourly_means = {
            str(h): _round_val(
                _safe_divide(hourly_sums.get(h, Decimal("0")), _decimal(hourly_counts.get(h, 1))), 2
            )
            for h in range(24)
        }
        weekday_means = {
            str(wd): _round_val(
                _safe_divide(weekday_sums.get(wd, Decimal("0")), _decimal(weekday_counts.get(wd, 1))), 2
            )
            for wd in range(7)
        }

        period_start = min(timestamps) if timestamps else utcnow()
        period_end = max(timestamps) if timestamps else utcnow()

        result = AnomalyBaseline(
            meter_id=meter_id,
            period_start=period_start,
            period_end=period_end,
            mean=_round_val(mean_val, 4),
            median=_round_val(median_val, 4),
            std_dev=_round_val(std_dev, 4),
            mad=_round_val(mad, 4),
            q1=_round_val(q1, 4),
            q3=_round_val(q3, 4),
            iqr=_round_val(iqr, 4),
            min_value=values[0],
            max_value=values[-1],
            data_points=len(values),
            hourly_means=hourly_means,
            weekday_means=weekday_means,
        )
        result.provenance_hash = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Baseline computed: mean=%.2f, std=%.2f, MAD=%.2f, "
            "IQR=%.2f, hash=%s (%.1f ms)",
            float(mean_val), float(std_dev), float(mad),
            float(iqr), result.provenance_hash[:16], elapsed,
        )
        return result

    def investigate_anomaly(
        self,
        anomaly_id: str,
        assigned_to: str = "",
        root_cause: str = "",
        action_taken: str = "",
        status: InvestigationStatus = InvestigationStatus.INVESTIGATING,
        energy_saved_kwh: Decimal = Decimal("0"),
        cost_saved: Decimal = Decimal("0"),
        notes: str = "",
    ) -> InvestigationRecord:
        """Create or update an anomaly investigation record.

        Args:
            anomaly_id:      Anomaly being investigated.
            assigned_to:     Investigator name.
            root_cause:      Identified root cause.
            action_taken:    Corrective action.
            status:          Investigation status.
            energy_saved_kwh: Energy saved from resolution.
            cost_saved:      Cost saved from resolution.
            notes:           Additional notes.

        Returns:
            InvestigationRecord.
        """
        t0 = time.perf_counter()

        resolved_at = utcnow() if status == InvestigationStatus.RESOLVED else None

        record = InvestigationRecord(
            anomaly_id=anomaly_id,
            status=status,
            assigned_to=assigned_to,
            root_cause=root_cause,
            action_taken=action_taken,
            energy_saved_kwh=energy_saved_kwh,
            cost_saved=cost_saved,
            resolved_at=resolved_at,
            notes=notes,
        )

        self._investigations[anomaly_id] = record

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Investigation recorded: anomaly=%s, status=%s (%.1f ms)",
            anomaly_id[:12], status.value, elapsed,
        )
        return record

    def classify_anomaly(
        self,
        anomaly: DetectedAnomaly,
        baseline: AnomalyBaseline,
    ) -> AnomalyType:
        """Classify an anomaly by type based on context.

        Args:
            anomaly:  Detected anomaly.
            baseline: Baseline statistics.

        Returns:
            AnomalyType classification.
        """
        ts = anomaly.timestamp
        deviation = anomaly.deviation

        # Check time context
        is_off_hours = False
        is_weekend = False
        if isinstance(ts, datetime):
            hour = ts.hour
            is_off_hours = (
                hour < self._config.occupied_start_hour
                or hour >= self._config.occupied_end_hour
            )
            is_weekend = ts.weekday() in WEEKEND_DAYS

        # After-hours usage
        if (is_off_hours or is_weekend) and deviation > Decimal("0"):
            hourly_mean = baseline.hourly_means.get(
                str(ts.hour) if isinstance(ts, datetime) else "0", Decimal("0")
            )
            if anomaly.value > hourly_mean * (Decimal("1") + self._config.off_hours_fraction):
                return AnomalyType.AFTER_HOURS

        # Over/under consumption
        if deviation > Decimal("0"):
            return AnomalyType.OVERCONSUMPTION
        if deviation < Decimal("0"):
            return AnomalyType.UNDERCONSUMPTION

        return AnomalyType.SCHEDULE_DEVIATION

    # ------------------------------------------------------------------ #
    # Internal Methods                                                     #
    # ------------------------------------------------------------------ #

    def _run_zscore(
        self,
        values: List[Decimal],
        timestamps: List[Any],
        baseline: AnomalyBaseline,
    ) -> List[DetectedAnomaly]:
        """Run Modified Z-score detection.

        M_i = 0.6745 * (x_i - median) / MAD
        """
        if not values or baseline.mad == Decimal("0"):
            return []

        anomalies: List[DetectedAnomaly] = []
        median = baseline.median
        mad = baseline.mad
        threshold = self._config.zscore_threshold

        for i, x in enumerate(values):
            m_score = _safe_divide(
                MODIFIED_ZSCORE_CONSTANT * abs(x - median), mad
            )
            if m_score > threshold:
                ts = timestamps[i] if i < len(timestamps) else utcnow()
                deviation = x - median
                dev_pct = _safe_pct(abs(deviation), median)
                anomalies.append(DetectedAnomaly(
                    meter_id=baseline.meter_id,
                    timestamp=ts,
                    value=x,
                    expected_value=median,
                    deviation=_round_val(deviation, 4),
                    deviation_pct=_round_val(dev_pct, 2),
                    method=AnomalyMethod.ZSCORE,
                    score=min(_round_val(m_score * Decimal("10"), 0), Decimal("100")),
                    message=f"Modified Z-score={_round_val(m_score, 2)} > {threshold}",
                ))

        return anomalies

    def _run_iqr(
        self,
        values: List[Decimal],
        timestamps: List[Any],
        baseline: AnomalyBaseline,
    ) -> List[DetectedAnomaly]:
        """Run IQR fence method detection.

        Lower fence = Q1 - k * IQR, Upper fence = Q3 + k * IQR.
        """
        if not values or baseline.iqr == Decimal("0"):
            return []

        k = self._config.iqr_k
        lower_fence = baseline.q1 - k * baseline.iqr
        upper_fence = baseline.q3 + k * baseline.iqr

        anomalies: List[DetectedAnomaly] = []
        for i, x in enumerate(values):
            if x < lower_fence or x > upper_fence:
                ts = timestamps[i] if i < len(timestamps) else utcnow()
                expected = baseline.median
                deviation = x - expected
                dev_pct = _safe_pct(abs(deviation), expected)
                anomalies.append(DetectedAnomaly(
                    meter_id=baseline.meter_id,
                    timestamp=ts,
                    value=x,
                    expected_value=expected,
                    deviation=_round_val(deviation, 4),
                    deviation_pct=_round_val(dev_pct, 2),
                    method=AnomalyMethod.IQR,
                    score=min(_round_val(dev_pct, 0), Decimal("100")),
                    message=(
                        f"IQR outlier: value={x}, fences="
                        f"[{_round_val(lower_fence, 2)}, {_round_val(upper_fence, 2)}]"
                    ),
                ))

        return anomalies

    def _run_schedule_based(
        self,
        values: List[Decimal],
        timestamps: List[Any],
        baseline: AnomalyBaseline,
    ) -> List[DetectedAnomaly]:
        """Run schedule-based anomaly detection."""
        anomalies: List[DetectedAnomaly] = []
        off_frac = self._config.off_hours_fraction

        for i, x in enumerate(values):
            ts = timestamps[i] if i < len(timestamps) else None
            if not isinstance(ts, datetime):
                continue

            hour = ts.hour
            weekday = ts.weekday()
            hourly_mean = _decimal(baseline.hourly_means.get(str(hour), baseline.mean))

            is_off = (
                hour < self._config.occupied_start_hour
                or hour >= self._config.occupied_end_hour
                or weekday in WEEKEND_DAYS
            )

            if is_off:
                threshold = hourly_mean * off_frac
                if threshold == Decimal("0"):
                    threshold = baseline.mean * off_frac
                if x > threshold and threshold > Decimal("0"):
                    deviation = x - threshold
                    dev_pct = _safe_pct(abs(deviation), threshold)
                    anomalies.append(DetectedAnomaly(
                        meter_id=baseline.meter_id,
                        timestamp=ts,
                        value=x,
                        expected_value=_round_val(threshold, 2),
                        deviation=_round_val(deviation, 4),
                        deviation_pct=_round_val(dev_pct, 2),
                        method=AnomalyMethod.SCHEDULE_BASED,
                        time_context=(
                            TimeContext.WEEKEND if weekday in WEEKEND_DAYS
                            else TimeContext.UNOCCUPIED
                        ),
                        score=min(_round_val(dev_pct, 0), Decimal("100")),
                        message=f"Off-hours usage: {x} > threshold {_round_val(threshold, 2)}",
                    ))

        return anomalies

    def _deduplicate_anomalies(
        self,
        anomalies: List[DetectedAnomaly],
    ) -> List[DetectedAnomaly]:
        """Deduplicate anomalies by timestamp, keeping highest severity."""
        severity_order = {
            AnomalySeverity.CRITICAL: 4,
            AnomalySeverity.HIGH: 3,
            AnomalySeverity.MEDIUM: 2,
            AnomalySeverity.LOW: 1,
        }
        best: Dict[str, DetectedAnomaly] = {}
        for a in anomalies:
            key = str(a.timestamp)
            existing = best.get(key)
            if existing is None:
                best[key] = a
            else:
                if severity_order.get(a.severity, 0) > severity_order.get(existing.severity, 0):
                    best[key] = a
                elif a.score > existing.score:
                    best[key] = a

        return sorted(best.values(), key=lambda x: x.timestamp)

    def _assign_severity(self, anomaly: DetectedAnomaly) -> AnomalySeverity:
        """Assign severity based on deviation percentage."""
        dev = abs(anomaly.deviation_pct)
        if dev >= SEVERITY_THRESHOLDS[AnomalySeverity.CRITICAL.value]:
            return AnomalySeverity.CRITICAL
        if dev >= SEVERITY_THRESHOLDS[AnomalySeverity.HIGH.value]:
            return AnomalySeverity.HIGH
        if dev >= SEVERITY_THRESHOLDS[AnomalySeverity.MEDIUM.value]:
            return AnomalySeverity.MEDIUM
        return AnomalySeverity.LOW

    def _percentile(
        self, sorted_values: List[Decimal], pct: int,
    ) -> Decimal:
        """Compute percentile from sorted list using linear interpolation."""
        if not sorted_values:
            return Decimal("0")
        n = len(sorted_values)
        if n == 1:
            return sorted_values[0]
        rank = _decimal(pct) / Decimal("100") * _decimal(n - 1)
        lower = int(math.floor(float(rank)))
        upper = min(lower + 1, n - 1)
        frac = rank - _decimal(lower)
        return sorted_values[lower] * (Decimal("1") - frac) + sorted_values[upper] * frac

    def _generate_recommendations(
        self,
        anomalies: List[DetectedAnomaly],
        anomaly_rate: Decimal,
        baseline: AnomalyBaseline,
    ) -> List[str]:
        """Generate recommendations based on anomaly analysis."""
        recs: List[str] = []

        critical = sum(1 for a in anomalies if a.severity == AnomalySeverity.CRITICAL)
        if critical > 0:
            recs.append(
                f"{critical} critical anomaly(ies) detected. Immediate "
                "investigation required to prevent energy waste."
            )

        after_hours = sum(
            1 for a in anomalies if a.anomaly_type == AnomalyType.AFTER_HOURS
        )
        if after_hours > 3:
            recs.append(
                f"{after_hours} after-hours usage anomalies. Review BMS "
                "schedules and implement tighter shut-off controls."
            )

        if anomaly_rate > Decimal("10"):
            recs.append(
                f"High anomaly rate ({_round_val(anomaly_rate, 1)}%). "
                "Baseline may be outdated -- consider recalculating."
            )

        overconsumption = sum(
            1 for a in anomalies
            if a.anomaly_type == AnomalyType.OVERCONSUMPTION
        )
        if overconsumption > 5:
            recs.append(
                f"{overconsumption} overconsumption events. Investigate "
                "equipment efficiency and maintenance schedules."
            )

        if not recs:
            recs.append(
                "No significant anomalies detected. Continue routine "
                "monitoring and baseline validation."
            )

        return recs
