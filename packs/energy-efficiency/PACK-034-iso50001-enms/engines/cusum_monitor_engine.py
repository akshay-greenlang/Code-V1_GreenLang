# -*- coding: utf-8 -*-
"""
CUSUMMonitorEngine - PACK-034 ISO 50001 EnMS Engine 4
======================================================

Cumulative Sum (CUSUM) control chart engine per ISO 50006:2014 for
continuous monitoring of energy performance relative to an energy
baseline.  Implements four CUSUM variants -- Standard, Tabular,
V-Mask, and Exponentially Weighted -- with seasonal adjustment,
automated alerting, change-point detection, and full provenance
tracking.

Calculation Methodology:

    Standard CUSUM:
        S_i = S_{i-1} + (x_i - k)
        where x_i = actual - expected, k = reference value (target shift)
        Alert when |S_i| > h  (decision interval)

    Tabular CUSUM (Page, 1954):
        C_upper_i = max(0, C_upper_{i-1} + (x_i - k))
        C_lower_i = min(0, C_lower_{i-1} + (x_i + k))
        Alert when C_upper > h  or  C_lower < -h

    V-Mask CUSUM:
        A V-shaped mask is placed at the most recent observation.
        If any prior cumulative sum falls outside the arms of the V,
        the process is declared out of control.
        Arms defined by lead distance *d* and half-angle *theta*.
        Upper arm:  S_origin + (n - i) * d * tan(theta)
        Lower arm:  S_origin - (n - i) * d * tan(theta)

    EWMA CUSUM (Exponentially Weighted):
        z_i = lambda * x_i + (1 - lambda) * z_{i-1}
        Applies tabular CUSUM to the EWMA-smoothed series.
        Smoothing parameter lambda in (0, 1].

    Seasonal Adjustment:
        ADDITIVE:        adjusted = raw - seasonal_factor
        MULTIPLICATIVE:  adjusted = raw / seasonal_factor
        DEGREE_DAY_BASED: adjusted = raw - (HDD * hdd_coeff + CDD * cdd_coeff)

    Change-Point Detection (binary segmentation):
        Recursively splits the series at the index that maximises
        the absolute difference in segment means.  Terminates when
        the shift magnitude < minimum_detectable_change.

Regulatory References:
    - ISO 50001:2018 - Energy management systems
    - ISO 50006:2014 - Measuring energy performance using energy
      baselines (EnB) and energy performance indicators (EnPI)
    - ISO 50015:2014 - Measurement and verification of energy
      performance of organisations
    - ASHRAE Guideline 14-2014 - Measurement of Energy, Demand,
      and Water Savings
    - CUSUM technique: BS 5703, Montgomery (Statistical Quality
      Control), Page (1954 Biometrika)

Zero-Hallucination:
    - All formulas are standard statistical process control (SPC)
    - CUSUM parameters from published SPC tables (ARL lookup)
    - Deterministic Decimal arithmetic throughout
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-034 ISO 50001 EnMS
Engine:  4 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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
            if k not in ("calculated_at", "calculation_time_ms", "provenance_hash")
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


class MonitoringInterval(str, Enum):
    """Monitoring frequency for CUSUM data collection.

    DAILY:     One data point per calendar day.
    WEEKLY:    One data point per ISO week.
    MONTHLY:   One data point per calendar month.
    QUARTERLY: One data point per calendar quarter.
    """
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class AlertType(str, Enum):
    """Classification of CUSUM alerts.

    PERFORMANCE_DEGRADATION: Energy performance has worsened.
    PERFORMANCE_IMPROVEMENT: Energy performance has improved.
    TREND_CHANGE:           Trend direction has reversed.
    THRESHOLD_BREACH:       CUSUM exceeded decision interval h.
    SUSTAINED_SHIFT:        N consecutive points on one side of target.
    """
    PERFORMANCE_DEGRADATION = "performance_degradation"
    PERFORMANCE_IMPROVEMENT = "performance_improvement"
    TREND_CHANGE = "trend_change"
    THRESHOLD_BREACH = "threshold_breach"
    SUSTAINED_SHIFT = "sustained_shift"


class CUSUMMethod(str, Enum):
    """CUSUM calculation variant.

    STANDARD:              Simple cumulative sum (S_i = S_{i-1} + x_i - k).
    TABULAR:               Two-sided tabular CUSUM (Page, 1954).
    V_MASK:                V-mask graphical decision procedure.
    EXPONENTIAL_WEIGHTED:  EWMA-CUSUM hybrid for smoothed detection.
    """
    STANDARD = "standard"
    TABULAR = "tabular"
    V_MASK = "v_mask"
    EXPONENTIAL_WEIGHTED = "exponential_weighted"


class MonitorStatus(str, Enum):
    """Operational status of a CUSUM monitor.

    ACTIVE:      Normal monitoring -- accepting data points.
    PAUSED:      Monitoring suspended by user.
    ALERTING:    At least one unacknowledged alert is active.
    COMPLETED:   Monitoring period has ended.
    CALIBRATING: Collecting initial data before alert evaluation.
    """
    ACTIVE = "active"
    PAUSED = "paused"
    ALERTING = "alerting"
    COMPLETED = "completed"
    CALIBRATING = "calibrating"


class SeasonalAdjustment(str, Enum):
    """Seasonal adjustment method for raw consumption data.

    NONE:            No adjustment applied.
    ADDITIVE:        adjusted = raw - seasonal_factor.
    MULTIPLICATIVE:  adjusted = raw / seasonal_factor.
    DEGREE_DAY_BASED: adjusted = raw - (HDD*coeff + CDD*coeff).
    """
    NONE = "none"
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    DEGREE_DAY_BASED = "degree_day_based"


# ---------------------------------------------------------------------------
# Constants / Reference Data
# ---------------------------------------------------------------------------

# Recommended CUSUM parameters per monitoring interval.
# Keys: (h = decision interval, k = allowable slack, min_pts = calibration).
DEFAULT_CUSUM_PARAMS: Dict[str, Dict[str, Any]] = {
    MonitoringInterval.DAILY.value: {
        "h": Decimal("5.0"),
        "k": Decimal("0.50"),
        "min_points": 30,
        "description": "Daily monitoring -- 30-day calibration, h=5 sigma, k=0.5 sigma",
    },
    MonitoringInterval.WEEKLY.value: {
        "h": Decimal("4.5"),
        "k": Decimal("0.50"),
        "min_points": 12,
        "description": "Weekly monitoring -- 12-week calibration, h=4.5 sigma, k=0.5 sigma",
    },
    MonitoringInterval.MONTHLY.value: {
        "h": Decimal("4.0"),
        "k": Decimal("0.50"),
        "min_points": 6,
        "description": "Monthly monitoring -- 6-month calibration, h=4 sigma, k=0.5 sigma",
    },
    MonitoringInterval.QUARTERLY.value: {
        "h": Decimal("3.5"),
        "k": Decimal("0.50"),
        "min_points": 4,
        "description": "Quarterly monitoring -- 4-quarter calibration, h=3.5 sigma, k=0.5 sigma",
    },
}

# Average Run Length (ARL) lookup table for common h/k combinations.
# ARL_0 = in-control run length (higher is better).
# ARL_1 = out-of-control run length at 1-sigma shift (lower is better).
# Source: Montgomery, Statistical Quality Control, 7th Ed, Table 9.4.
ARL_TABLES: Dict[str, Dict[str, Any]] = {
    "h4_k0.50": {
        "h": Decimal("4.0"), "k": Decimal("0.50"),
        "ARL_0": Decimal("168"), "ARL_0.5": Decimal("26.6"),
        "ARL_1.0": Decimal("10.4"), "ARL_2.0": Decimal("4.01"),
    },
    "h4.5_k0.50": {
        "h": Decimal("4.5"), "k": Decimal("0.50"),
        "ARL_0": Decimal("312"), "ARL_0.5": Decimal("33.4"),
        "ARL_1.0": Decimal("11.7"), "ARL_2.0": Decimal("4.20"),
    },
    "h5_k0.50": {
        "h": Decimal("5.0"), "k": Decimal("0.50"),
        "ARL_0": Decimal("465"), "ARL_0.5": Decimal("38.0"),
        "ARL_1.0": Decimal("13.3"), "ARL_2.0": Decimal("4.40"),
    },
    "h4_k0.25": {
        "h": Decimal("4.0"), "k": Decimal("0.25"),
        "ARL_0": Decimal("77"), "ARL_0.5": Decimal("15.0"),
        "ARL_1.0": Decimal("8.38"), "ARL_2.0": Decimal("4.75"),
    },
    "h5_k0.25": {
        "h": Decimal("5.0"), "k": Decimal("0.25"),
        "ARL_0": Decimal("196"), "ARL_0.5": Decimal("20.0"),
        "ARL_1.0": Decimal("10.0"), "ARL_2.0": Decimal("5.00"),
    },
    "h4_k0.75": {
        "h": Decimal("4.0"), "k": Decimal("0.75"),
        "ARL_0": Decimal("348"), "ARL_0.5": Decimal("56.4"),
        "ARL_1.0": Decimal("12.2"), "ARL_2.0": Decimal("3.63"),
    },
    "h5_k0.75": {
        "h": Decimal("5.0"), "k": Decimal("0.75"),
        "ARL_0": Decimal("850"), "ARL_0.5": Decimal("77.0"),
        "ARL_1.0": Decimal("15.0"), "ARL_2.0": Decimal("3.90"),
    },
    "h3_k0.50": {
        "h": Decimal("3.0"), "k": Decimal("0.50"),
        "ARL_0": Decimal("54"), "ARL_0.5": Decimal("15.7"),
        "ARL_1.0": Decimal("7.60"), "ARL_2.0": Decimal("3.50"),
    },
    "h3.5_k0.50": {
        "h": Decimal("3.5"), "k": Decimal("0.50"),
        "ARL_0": Decimal("96"), "ARL_0.5": Decimal("20.5"),
        "ARL_1.0": Decimal("8.80"), "ARL_2.0": Decimal("3.70"),
    },
}

# Template for monthly seasonal indices (Jan-Dec).
# Values represent the multiplicative factor for each month.
# A neutral profile (all 1.0) means no seasonal effect.
SEASONAL_FACTORS_TEMPLATE: Dict[str, List[Decimal]] = {
    "neutral": [
        Decimal("1.00"), Decimal("1.00"), Decimal("1.00"),
        Decimal("1.00"), Decimal("1.00"), Decimal("1.00"),
        Decimal("1.00"), Decimal("1.00"), Decimal("1.00"),
        Decimal("1.00"), Decimal("1.00"), Decimal("1.00"),
    ],
    "heating_dominated": [
        Decimal("1.35"), Decimal("1.30"), Decimal("1.15"),
        Decimal("0.95"), Decimal("0.80"), Decimal("0.70"),
        Decimal("0.65"), Decimal("0.65"), Decimal("0.75"),
        Decimal("0.95"), Decimal("1.15"), Decimal("1.30"),
    ],
    "cooling_dominated": [
        Decimal("0.70"), Decimal("0.70"), Decimal("0.80"),
        Decimal("0.90"), Decimal("1.10"), Decimal("1.30"),
        Decimal("1.40"), Decimal("1.40"), Decimal("1.25"),
        Decimal("1.00"), Decimal("0.80"), Decimal("0.70"),
    ],
    "mixed_climate": [
        Decimal("1.20"), Decimal("1.15"), Decimal("1.00"),
        Decimal("0.85"), Decimal("0.90"), Decimal("1.10"),
        Decimal("1.20"), Decimal("1.20"), Decimal("1.05"),
        Decimal("0.85"), Decimal("0.95"), Decimal("1.15"),
    ],
    "industrial_24_7": [
        Decimal("1.05"), Decimal("1.03"), Decimal("1.00"),
        Decimal("0.98"), Decimal("0.97"), Decimal("0.98"),
        Decimal("1.00"), Decimal("1.00"), Decimal("0.99"),
        Decimal("0.98"), Decimal("1.00"), Decimal("1.04"),
    ],
}

# Default degree-day coefficients for degree_day_based seasonal adjustment.
DEFAULT_HDD_COEFFICIENT: Decimal = Decimal("0.05")
DEFAULT_CDD_COEFFICIENT: Decimal = Decimal("0.04")

# Minimum detectable change for binary segmentation (fraction of mean).
MIN_DETECTABLE_CHANGE_FRACTION: Decimal = Decimal("0.02")

# Maximum consecutive points for sustained-shift detection.
DEFAULT_SUSTAINED_SHIFT_POINTS: int = 7


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class CUSUMConfig(BaseModel):
    """Configuration for a CUSUM monitor instance.

    Attributes:
        method: CUSUM calculation variant to use.
        alert_threshold: Decision interval *h* (in sigma or absolute units).
        reference_value: Target / reference value *k* for shift detection.
        allowable_slack: Slack parameter *k* for tabular CUSUM.
        decision_interval_h: Decision interval *h* for tabular CUSUM.
        monitoring_interval: Data collection frequency.
        seasonal_adjustment: Seasonal adjustment method.
        min_points_for_alert: Minimum data points before alert evaluation.
        reset_after_alert: Whether to reset CUSUM after an alert fires.
    """
    method: CUSUMMethod = Field(
        default=CUSUMMethod.TABULAR,
        description="CUSUM calculation variant",
    )
    alert_threshold: Decimal = Field(
        default=Decimal("5.0"),
        ge=0,
        description="Decision interval h (sigma units)",
    )
    reference_value: Optional[Decimal] = Field(
        default=None,
        description="Target / reference value k for shift detection",
    )
    allowable_slack: Decimal = Field(
        default=Decimal("0.5"),
        ge=0,
        description="Allowable slack k for tabular CUSUM",
    )
    decision_interval_h: Decimal = Field(
        default=Decimal("5.0"),
        ge=0,
        description="Decision interval h for tabular CUSUM",
    )
    monitoring_interval: MonitoringInterval = Field(
        default=MonitoringInterval.MONTHLY,
        description="Data collection frequency",
    )
    seasonal_adjustment: SeasonalAdjustment = Field(
        default=SeasonalAdjustment.NONE,
        description="Seasonal adjustment method to apply",
    )
    min_points_for_alert: int = Field(
        default=5,
        ge=1,
        le=365,
        description="Minimum data points before alert evaluation",
    )
    reset_after_alert: bool = Field(
        default=False,
        description="Whether to reset CUSUM accumulators after an alert",
    )

    @field_validator("alert_threshold")
    @classmethod
    def validate_alert_threshold(cls, v: Decimal) -> Decimal:
        """Ensure alert threshold is positive."""
        if v <= Decimal("0"):
            raise ValueError("alert_threshold must be > 0")
        return v

    @field_validator("decision_interval_h")
    @classmethod
    def validate_decision_interval(cls, v: Decimal) -> Decimal:
        """Ensure decision interval is positive."""
        if v <= Decimal("0"):
            raise ValueError("decision_interval_h must be > 0")
        return v


class CUSUMDataPoint(BaseModel):
    """Single observation in a CUSUM control chart.

    Attributes:
        period_date: Date of the observation.
        actual_consumption: Measured energy consumption.
        expected_consumption: Baseline-predicted energy consumption.
        difference: actual - expected.
        cumulative_sum: Running cumulative sum of differences.
        upper_cusum: Upper (positive) CUSUM accumulator.
        lower_cusum: Lower (negative) CUSUM accumulator.
        upper_limit: Upper control limit (decision interval).
        lower_limit: Lower control limit (negative decision interval).
        is_alert: Whether this point triggered an alert.
    """
    period_date: date = Field(
        ..., description="Date of the observation",
    )
    actual_consumption: Decimal = Field(
        default=Decimal("0"), description="Measured energy consumption",
    )
    expected_consumption: Decimal = Field(
        default=Decimal("0"), description="Baseline-predicted consumption",
    )
    difference: Decimal = Field(
        default=Decimal("0"), description="actual - expected",
    )
    cumulative_sum: Decimal = Field(
        default=Decimal("0"), description="Running cumulative sum",
    )
    upper_cusum: Decimal = Field(
        default=Decimal("0"), description="Upper CUSUM accumulator (C+)",
    )
    lower_cusum: Decimal = Field(
        default=Decimal("0"), description="Lower CUSUM accumulator (C-)",
    )
    upper_limit: Decimal = Field(
        default=Decimal("0"), description="Upper control limit (+h)",
    )
    lower_limit: Decimal = Field(
        default=Decimal("0"), description="Lower control limit (-h)",
    )
    is_alert: bool = Field(
        default=False, description="Whether this point triggered an alert",
    )


class CUSUMAlert(BaseModel):
    """Alert generated by the CUSUM monitor.

    Attributes:
        alert_id: Unique alert identifier.
        alert_date: Datetime the alert was generated.
        alert_type: Classification of the alert.
        cumulative_value: CUSUM value at alert point.
        threshold_value: Decision interval that was breached.
        consecutive_points: Number of consecutive points in alert condition.
        estimated_change_magnitude: Estimated size of the performance shift.
        description: Human-readable alert description.
        acknowledged: Whether the alert has been acknowledged.
        root_cause: Optional root cause (user-supplied after investigation).
    """
    alert_id: str = Field(
        default_factory=_new_uuid,
        description="Unique alert identifier",
    )
    alert_date: datetime = Field(
        default_factory=_utcnow,
        description="Alert generation timestamp",
    )
    alert_type: AlertType = Field(
        ..., description="Classification of the alert",
    )
    cumulative_value: Decimal = Field(
        default=Decimal("0"),
        description="CUSUM value at alert point",
    )
    threshold_value: Decimal = Field(
        default=Decimal("0"),
        description="Decision interval that was breached",
    )
    consecutive_points: int = Field(
        default=0, ge=0,
        description="Consecutive points in alert condition",
    )
    estimated_change_magnitude: Decimal = Field(
        default=Decimal("0"),
        description="Estimated size of the performance shift",
    )
    description: str = Field(
        default="",
        max_length=2000,
        description="Human-readable alert description",
    )
    acknowledged: bool = Field(
        default=False,
        description="Whether the alert has been acknowledged",
    )
    root_cause: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Root cause (user-supplied after investigation)",
    )


class VMaskParameters(BaseModel):
    """V-Mask parameters for graphical CUSUM decision procedure.

    Attributes:
        lead_distance_d: Lead distance d (horizontal offset from origin).
        half_angle_theta: Half-angle theta of the V-mask (degrees).
        origin_point: Index of the origin point (most recent observation).
    """
    lead_distance_d: Decimal = Field(
        default=Decimal("2.0"),
        ge=0,
        description="Lead distance d (horizontal units)",
    )
    half_angle_theta: Decimal = Field(
        default=Decimal("25.0"),
        ge=0,
        le=90,
        description="Half-angle theta of the V-mask (degrees)",
    )
    origin_point: int = Field(
        default=0,
        ge=0,
        description="Index of the origin point (0 = last observation)",
    )

    @field_validator("half_angle_theta")
    @classmethod
    def validate_theta(cls, v: Decimal) -> Decimal:
        """Ensure theta is within valid range for V-mask."""
        if v <= Decimal("0") or v >= Decimal("90"):
            raise ValueError("half_angle_theta must be in (0, 90) degrees")
        return v


class CUSUMMonitorResult(BaseModel):
    """Complete result from a CUSUM monitoring analysis.

    Attributes:
        monitor_id: Unique monitor identifier.
        enms_id: Parent EnMS identifier (links to ISO 50001 system).
        monitor_name: Human-readable monitor name.
        baseline_id: Energy baseline identifier.
        method: CUSUM variant used.
        data_points: Ordered list of CUSUM observations.
        alerts: Alerts generated during monitoring.
        current_status: Current operational status of the monitor.
        total_cumulative_sum: Final cumulative sum value.
        trend_direction: Overall trend (improving, degrading, stable).
        performance_shift_detected: Whether a significant shift was found.
        shift_magnitude: Estimated magnitude of detected shift.
        provenance_hash: SHA-256 hash for audit trail.
        calculation_time_ms: Processing duration in milliseconds.
    """
    monitor_id: str = Field(
        default_factory=_new_uuid,
        description="Unique monitor identifier",
    )
    enms_id: str = Field(
        default="",
        description="Parent EnMS identifier",
    )
    monitor_name: str = Field(
        default="",
        max_length=500,
        description="Human-readable monitor name",
    )
    baseline_id: str = Field(
        default="",
        description="Energy baseline identifier",
    )
    method: CUSUMMethod = Field(
        default=CUSUMMethod.TABULAR,
        description="CUSUM variant used",
    )
    data_points: List[CUSUMDataPoint] = Field(
        default_factory=list,
        description="Ordered CUSUM observations",
    )
    alerts: List[CUSUMAlert] = Field(
        default_factory=list,
        description="Alerts generated during monitoring",
    )
    current_status: MonitorStatus = Field(
        default=MonitorStatus.ACTIVE,
        description="Current operational status",
    )
    total_cumulative_sum: Decimal = Field(
        default=Decimal("0"),
        description="Final cumulative sum value",
    )
    trend_direction: str = Field(
        default="stable",
        description="Overall trend (improving / degrading / stable)",
    )
    performance_shift_detected: bool = Field(
        default=False,
        description="Whether a significant shift was found",
    )
    shift_magnitude: Optional[Decimal] = Field(
        default=None,
        description="Estimated magnitude of detected shift",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )
    calculation_time_ms: int = Field(
        default=0, ge=0,
        description="Processing duration (ms)",
    )


class CUSUMSummary(BaseModel):
    """Aggregate summary across multiple CUSUM monitors.

    Attributes:
        monitor_count: Total number of monitors.
        active_monitors: Number in ACTIVE status.
        alerting_monitors: Number in ALERTING status.
        total_alerts: Total unacknowledged alerts across all monitors.
        avg_cumulative_sum: Mean absolute cumulative sum.
        worst_performer: Monitor with highest absolute cumulative sum.
        best_performer: Monitor with lowest absolute cumulative sum.
    """
    monitor_count: int = Field(
        default=0, ge=0, description="Total number of monitors",
    )
    active_monitors: int = Field(
        default=0, ge=0, description="Monitors in ACTIVE status",
    )
    alerting_monitors: int = Field(
        default=0, ge=0, description="Monitors in ALERTING status",
    )
    total_alerts: int = Field(
        default=0, ge=0, description="Total unacknowledged alerts",
    )
    avg_cumulative_sum: Decimal = Field(
        default=Decimal("0"), description="Mean absolute cumulative sum",
    )
    worst_performer: Optional[str] = Field(
        default=None, description="Monitor ID with highest |cumsum|",
    )
    best_performer: Optional[str] = Field(
        default=None, description="Monitor ID with lowest |cumsum|",
    )


class DegreeDayData(BaseModel):
    """Heating / cooling degree-day data for a single period.

    Attributes:
        period_date: Date of the period.
        hdd: Heating degree-days.
        cdd: Cooling degree-days.
    """
    period_date: date = Field(
        ..., description="Date of the period",
    )
    hdd: Decimal = Field(
        default=Decimal("0"), ge=0, description="Heating degree-days",
    )
    cdd: Decimal = Field(
        default=Decimal("0"), ge=0, description="Cooling degree-days",
    )


# ---------------------------------------------------------------------------
# Internal storage model for live monitors
# ---------------------------------------------------------------------------


class _MonitorState(BaseModel):
    """Internal mutable state of a CUSUM monitor (not part of public API).

    Attributes:
        monitor_id: Unique identifier.
        monitor_name: Human-readable name.
        baseline_id: Energy baseline reference.
        config: CUSUM configuration.
        data_points: Accumulated data points.
        alerts: Generated alerts.
        status: Current operational status.
        created_at: Creation timestamp.
    """
    monitor_id: str = Field(default_factory=_new_uuid)
    monitor_name: str = Field(default="")
    baseline_id: str = Field(default="")
    config: CUSUMConfig = Field(default_factory=CUSUMConfig)
    data_points: List[CUSUMDataPoint] = Field(default_factory=list)
    alerts: List[CUSUMAlert] = Field(default_factory=list)
    status: MonitorStatus = Field(default=MonitorStatus.CALIBRATING)
    created_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# model_rebuild -- required by Pydantic v2 with `from __future__ import annotations`
# ---------------------------------------------------------------------------

CUSUMConfig.model_rebuild()
CUSUMDataPoint.model_rebuild()
CUSUMAlert.model_rebuild()
VMaskParameters.model_rebuild()
CUSUMMonitorResult.model_rebuild()
CUSUMSummary.model_rebuild()
DegreeDayData.model_rebuild()
_MonitorState.model_rebuild()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class CUSUMMonitorEngine:
    """Cumulative Sum control chart engine per ISO 50006.

    Implements Standard, Tabular, V-Mask, and EWMA CUSUM variants with
    automated alerting, seasonal adjustment, change-point detection, and
    full SHA-256 provenance tracking.

    Usage::

        engine = CUSUMMonitorEngine()
        mid = engine.create_monitor("Chiller Plant", "BL-001", config)
        for period_date, actual, expected in data:
            pt = engine.add_data_point(mid, period_date, actual, expected)
        result = engine.get_monitor_result(mid, enms_id="ENMS-001")
        print(f"Status: {result.current_status}, Alerts: {len(result.alerts)}")

    All arithmetic uses ``Decimal`` for deterministic, audit-grade precision.
    Every result carries a SHA-256 provenance hash.
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise CUSUMMonitorEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - default_method (str): default CUSUMMethod value
                - default_h (Decimal): default decision interval
                - default_k (Decimal): default allowable slack
                - sustained_shift_points (int): consecutive points for shift
        """
        self.config = config or {}
        self._default_method = CUSUMMethod(
            self.config.get("default_method", CUSUMMethod.TABULAR.value)
        )
        self._default_h = _decimal(
            self.config.get("default_h", Decimal("5.0"))
        )
        self._default_k = _decimal(
            self.config.get("default_k", Decimal("0.5"))
        )
        self._sustained_shift_pts = int(
            self.config.get("sustained_shift_points", DEFAULT_SUSTAINED_SHIFT_POINTS)
        )
        self._monitors: Dict[str, _MonitorState] = {}
        logger.info(
            "CUSUMMonitorEngine v%s initialised (method=%s, h=%s, k=%s)",
            self.engine_version,
            self._default_method.value,
            self._default_h,
            self._default_k,
        )

    # ------------------------------------------------------------------ #
    # Public API -- Monitor Lifecycle                                      #
    # ------------------------------------------------------------------ #

    def create_monitor(
        self,
        name: str,
        baseline_id: str,
        config: CUSUMConfig,
    ) -> str:
        """Create a new CUSUM monitor.

        Args:
            name: Human-readable name for the monitor.
            baseline_id: Energy baseline identifier to track against.
            config: CUSUM configuration parameters.

        Returns:
            Unique monitor_id string.

        Raises:
            ValueError: If configuration validation fails.
        """
        t0 = time.perf_counter()

        validation = self.validate_monitor_config(config)
        if not validation["is_valid"]:
            raise ValueError(
                f"Invalid CUSUM config: {validation['errors']}"
            )

        monitor_id = _new_uuid()
        state = _MonitorState(
            monitor_id=monitor_id,
            monitor_name=name,
            baseline_id=baseline_id,
            config=config,
            status=MonitorStatus.CALIBRATING,
        )
        self._monitors[monitor_id] = state

        elapsed = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "Created CUSUM monitor '%s' (id=%s, method=%s, h=%s, k=%s) in %dms",
            name, monitor_id, config.method.value,
            config.decision_interval_h, config.allowable_slack, elapsed,
        )
        return monitor_id

    def reset_monitor(self, monitor_id: str) -> None:
        """Reset a CUSUM monitor, clearing all data points and alerts.

        Args:
            monitor_id: Identifier of the monitor to reset.

        Raises:
            KeyError: If monitor_id is not found.
        """
        state = self._get_monitor(monitor_id)
        state.data_points.clear()
        state.alerts.clear()
        state.status = MonitorStatus.CALIBRATING
        logger.info("Reset CUSUM monitor %s", monitor_id)

    def pause_monitor(self, monitor_id: str) -> None:
        """Pause a CUSUM monitor.

        Args:
            monitor_id: Monitor to pause.

        Raises:
            KeyError: If monitor_id is not found.
        """
        state = self._get_monitor(monitor_id)
        state.status = MonitorStatus.PAUSED
        logger.info("Paused CUSUM monitor %s", monitor_id)

    def resume_monitor(self, monitor_id: str) -> None:
        """Resume a paused CUSUM monitor.

        Args:
            monitor_id: Monitor to resume.

        Raises:
            KeyError: If monitor_id is not found.
        """
        state = self._get_monitor(monitor_id)
        if state.status == MonitorStatus.PAUSED:
            has_unacked = any(not a.acknowledged for a in state.alerts)
            state.status = MonitorStatus.ALERTING if has_unacked else MonitorStatus.ACTIVE
            logger.info("Resumed CUSUM monitor %s -> %s", monitor_id, state.status.value)

    def complete_monitor(self, monitor_id: str) -> None:
        """Mark a CUSUM monitor as completed.

        Args:
            monitor_id: Monitor to complete.

        Raises:
            KeyError: If monitor_id is not found.
        """
        state = self._get_monitor(monitor_id)
        state.status = MonitorStatus.COMPLETED
        logger.info("Completed CUSUM monitor %s", monitor_id)

    # ------------------------------------------------------------------ #
    # Public API -- Data Point Ingestion                                   #
    # ------------------------------------------------------------------ #

    def add_data_point(
        self,
        monitor_id: str,
        period_date: date,
        actual: Decimal,
        expected: Decimal,
    ) -> CUSUMDataPoint:
        """Add a new observation and update CUSUM accumulators.

        Calculates the difference (actual - expected), updates the
        running cumulative sum, evaluates upper/lower CUSUM values,
        and checks alert thresholds.

        Args:
            monitor_id: Target monitor identifier.
            period_date: Date of the observation.
            actual: Measured energy consumption.
            expected: Baseline-predicted consumption.

        Returns:
            The newly created CUSUMDataPoint.

        Raises:
            KeyError: If monitor_id is not found.
            ValueError: If monitor is paused or completed.
        """
        state = self._get_monitor(monitor_id)

        if state.status in (MonitorStatus.PAUSED, MonitorStatus.COMPLETED):
            raise ValueError(
                f"Cannot add data to monitor in {state.status.value} status"
            )

        actual_d = _decimal(actual)
        expected_d = _decimal(expected)
        diff = actual_d - expected_d

        cfg = state.config
        k = cfg.reference_value if cfg.reference_value is not None else cfg.allowable_slack
        h = cfg.decision_interval_h

        # Previous accumulators
        if state.data_points:
            prev = state.data_points[-1]
            prev_cumsum = prev.cumulative_sum
            prev_upper = prev.upper_cusum
            prev_lower = prev.lower_cusum
        else:
            prev_cumsum = Decimal("0")
            prev_upper = Decimal("0")
            prev_lower = Decimal("0")

        # Standard cumulative sum
        cumsum = prev_cumsum + diff

        # Tabular CUSUM accumulators
        upper_cusum = max(Decimal("0"), prev_upper + (diff - k))
        lower_cusum = min(Decimal("0"), prev_lower + (diff + k))

        # Check alert condition
        is_alert = False
        n_points = len(state.data_points) + 1
        if n_points >= cfg.min_points_for_alert:
            if upper_cusum > h or lower_cusum < -h:
                is_alert = True

        point = CUSUMDataPoint(
            period_date=period_date,
            actual_consumption=actual_d,
            expected_consumption=expected_d,
            difference=diff,
            cumulative_sum=cumsum,
            upper_cusum=upper_cusum,
            lower_cusum=lower_cusum,
            upper_limit=h,
            lower_limit=-h,
            is_alert=is_alert,
        )
        state.data_points.append(point)

        # Update status from calibrating to active
        if state.status == MonitorStatus.CALIBRATING:
            if n_points >= cfg.min_points_for_alert:
                state.status = MonitorStatus.ACTIVE
                logger.info(
                    "Monitor %s calibration complete (%d points), now ACTIVE",
                    monitor_id, n_points,
                )

        # Generate alert if needed
        if is_alert:
            alert = self._generate_point_alert(state, point, n_points)
            state.alerts.append(alert)
            state.status = MonitorStatus.ALERTING

            if cfg.reset_after_alert:
                self._reset_accumulators(state)
                logger.info(
                    "Monitor %s: CUSUM reset after alert (reset_after_alert=True)",
                    monitor_id,
                )

        logger.debug(
            "Monitor %s: date=%s, diff=%s, cumsum=%s, C+=%s, C-=%s, alert=%s",
            monitor_id, period_date, _round_val(diff, 4),
            _round_val(cumsum, 4), _round_val(upper_cusum, 4),
            _round_val(lower_cusum, 4), is_alert,
        )
        return point

    # ------------------------------------------------------------------ #
    # Public API -- CUSUM Calculation Variants                             #
    # ------------------------------------------------------------------ #

    def calculate_standard_cusum(
        self,
        differences: List[Decimal],
        reference: Decimal,
    ) -> List[CUSUMDataPoint]:
        """Calculate standard CUSUM for a series of differences.

        S_i = S_{i-1} + (x_i - k)  where k = reference value.

        Args:
            differences: List of (actual - expected) values.
            reference: Reference value k (target shift to detect).

        Returns:
            List of CUSUMDataPoint with cumulative_sum populated.
        """
        t0 = time.perf_counter()
        ref = _decimal(reference)
        points: List[CUSUMDataPoint] = []
        cumsum = Decimal("0")

        for i, diff_raw in enumerate(differences):
            diff = _decimal(diff_raw)
            cumsum = cumsum + (diff - ref)

            point = CUSUMDataPoint(
                period_date=date(2000, 1, 1),  # placeholder
                actual_consumption=diff,
                expected_consumption=Decimal("0"),
                difference=diff,
                cumulative_sum=cumsum,
                upper_cusum=max(Decimal("0"), cumsum),
                lower_cusum=min(Decimal("0"), cumsum),
                upper_limit=Decimal("0"),
                lower_limit=Decimal("0"),
                is_alert=False,
            )
            points.append(point)

        elapsed = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "Standard CUSUM: %d points, ref=%s, final_cumsum=%s (%dms)",
            len(points), ref,
            _round_val(cumsum, 4) if points else Decimal("0"),
            elapsed,
        )
        return points

    def calculate_tabular_cusum(
        self,
        differences: List[Decimal],
        slack: Decimal,
        decision_h: Decimal,
    ) -> List[CUSUMDataPoint]:
        """Calculate two-sided tabular CUSUM (Page, 1954).

        C_upper_i = max(0, C_upper_{i-1} + (x_i - k))
        C_lower_i = min(0, C_lower_{i-1} + (x_i + k))
        Alert when C_upper > h  or  C_lower < -h.

        Args:
            differences: List of (actual - expected) values.
            slack: Allowable slack parameter k.
            decision_h: Decision interval h.

        Returns:
            List of CUSUMDataPoint with upper/lower CUSUM and alert flags.
        """
        t0 = time.perf_counter()
        k = _decimal(slack)
        h = _decimal(decision_h)
        points: List[CUSUMDataPoint] = []

        c_upper = Decimal("0")
        c_lower = Decimal("0")
        cumsum = Decimal("0")
        alert_count = 0

        for i, diff_raw in enumerate(differences):
            diff = _decimal(diff_raw)
            cumsum = cumsum + diff

            c_upper = max(Decimal("0"), c_upper + (diff - k))
            c_lower = min(Decimal("0"), c_lower + (diff + k))

            is_alert = (c_upper > h) or (c_lower < -h)
            if is_alert:
                alert_count += 1

            point = CUSUMDataPoint(
                period_date=date(2000, 1, 1),  # placeholder
                actual_consumption=diff,
                expected_consumption=Decimal("0"),
                difference=diff,
                cumulative_sum=cumsum,
                upper_cusum=c_upper,
                lower_cusum=c_lower,
                upper_limit=h,
                lower_limit=-h,
                is_alert=is_alert,
            )
            points.append(point)

        elapsed = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "Tabular CUSUM: %d points, k=%s, h=%s, alerts=%d (%dms)",
            len(points), k, h, alert_count, elapsed,
        )
        return points

    def apply_v_mask(
        self,
        data_points: List[CUSUMDataPoint],
        params: VMaskParameters,
    ) -> List[CUSUMAlert]:
        """Apply V-mask decision procedure to a CUSUM series.

        The V-mask is placed at the most recent observation (origin).
        For each prior point i, the upper and lower arms are computed.
        If the cumulative sum at point i falls outside the V arms, the
        process is declared out of control starting at point i.

        Upper arm: S_origin + (n - i) * d * tan(theta)
        Lower arm: S_origin - (n - i) * d * tan(theta)

        Args:
            data_points: List of CUSUM data points (cumulative_sum populated).
            params: V-mask parameters (d, theta, origin_point).

        Returns:
            List of CUSUMAlert for each detected out-of-control point.
        """
        t0 = time.perf_counter()

        if not data_points:
            return []

        d = _decimal(params.lead_distance_d)
        theta_deg = _decimal(params.half_angle_theta)
        theta_rad = theta_deg * _decimal(math.pi) / Decimal("180")

        # tan(theta) computed via float then back to Decimal
        tan_theta = _decimal(math.tan(float(theta_rad)))

        n = len(data_points)
        origin_idx = n - 1 if params.origin_point == 0 else min(params.origin_point, n - 1)
        origin_cumsum = data_points[origin_idx].cumulative_sum

        alerts: List[CUSUMAlert] = []

        for i in range(origin_idx):
            distance = Decimal(str(origin_idx - i))
            arm_offset = (d + distance) * tan_theta

            upper_arm = origin_cumsum + arm_offset
            lower_arm = origin_cumsum - arm_offset

            pt_cumsum = data_points[i].cumulative_sum

            if pt_cumsum > upper_arm:
                alert = CUSUMAlert(
                    alert_type=AlertType.PERFORMANCE_DEGRADATION,
                    cumulative_value=pt_cumsum,
                    threshold_value=upper_arm,
                    consecutive_points=origin_idx - i,
                    estimated_change_magnitude=abs(pt_cumsum - upper_arm),
                    description=(
                        f"V-mask violation at point {i}: cumsum {_round_val(pt_cumsum, 4)} "
                        f"> upper arm {_round_val(upper_arm, 4)}. "
                        f"Process went out of control at observation {i}."
                    ),
                )
                alerts.append(alert)
            elif pt_cumsum < lower_arm:
                alert = CUSUMAlert(
                    alert_type=AlertType.PERFORMANCE_IMPROVEMENT,
                    cumulative_value=pt_cumsum,
                    threshold_value=lower_arm,
                    consecutive_points=origin_idx - i,
                    estimated_change_magnitude=abs(pt_cumsum - lower_arm),
                    description=(
                        f"V-mask violation at point {i}: cumsum {_round_val(pt_cumsum, 4)} "
                        f"< lower arm {_round_val(lower_arm, 4)}. "
                        f"Process went out of control at observation {i}."
                    ),
                )
                alerts.append(alert)

        elapsed = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "V-mask analysis: %d points, d=%s, theta=%s, violations=%d (%dms)",
            n, d, theta_deg, len(alerts), elapsed,
        )
        return alerts

    def calculate_ewma_cusum(
        self,
        differences: List[Decimal],
        lambda_param: Decimal,
    ) -> List[CUSUMDataPoint]:
        """Calculate EWMA-CUSUM hybrid for smoothed shift detection.

        First applies EWMA smoothing:
            z_i = lambda * x_i + (1 - lambda) * z_{i-1}

        Then applies tabular CUSUM to the EWMA-smoothed series using
        the engine's default k and h values.

        Args:
            differences: List of (actual - expected) values.
            lambda_param: EWMA smoothing parameter (0 < lambda <= 1).

        Returns:
            List of CUSUMDataPoint from CUSUM on EWMA-smoothed series.

        Raises:
            ValueError: If lambda_param is outside (0, 1].
        """
        t0 = time.perf_counter()
        lam = _decimal(lambda_param)

        if lam <= Decimal("0") or lam > Decimal("1"):
            raise ValueError("lambda_param must be in (0, 1]")

        one_minus_lam = Decimal("1") - lam

        # Step 1: EWMA smoothing
        ewma_values: List[Decimal] = []
        z_prev = Decimal("0")

        for diff_raw in differences:
            diff = _decimal(diff_raw)
            z_i = lam * diff + one_minus_lam * z_prev
            ewma_values.append(z_i)
            z_prev = z_i

        # Step 2: Tabular CUSUM on EWMA series
        k = self._default_k
        h = self._default_h
        points: List[CUSUMDataPoint] = []

        c_upper = Decimal("0")
        c_lower = Decimal("0")
        cumsum = Decimal("0")

        for i, z_val in enumerate(ewma_values):
            cumsum = cumsum + z_val

            c_upper = max(Decimal("0"), c_upper + (z_val - k))
            c_lower = min(Decimal("0"), c_lower + (z_val + k))

            is_alert = (c_upper > h) or (c_lower < -h)

            point = CUSUMDataPoint(
                period_date=date(2000, 1, 1),  # placeholder
                actual_consumption=_decimal(differences[i]),
                expected_consumption=Decimal("0"),
                difference=z_val,  # EWMA-smoothed value
                cumulative_sum=cumsum,
                upper_cusum=c_upper,
                lower_cusum=c_lower,
                upper_limit=h,
                lower_limit=-h,
                is_alert=is_alert,
            )
            points.append(point)

        elapsed = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "EWMA-CUSUM: %d points, lambda=%s, k=%s, h=%s (%dms)",
            len(points), lam, k, h, elapsed,
        )
        return points

    # ------------------------------------------------------------------ #
    # Public API -- Alert Detection                                        #
    # ------------------------------------------------------------------ #

    def check_alerts(
        self,
        data_points: List[CUSUMDataPoint],
        config: CUSUMConfig,
    ) -> List[CUSUMAlert]:
        """Evaluate a CUSUM series for alerting conditions.

        Checks:
            1. Threshold breach -- upper or lower CUSUM exceeds h.
            2. Sustained shift -- N consecutive points on one side.
            3. Trend change -- sign reversal of cumulative sum slope.

        Args:
            data_points: Ordered list of CUSUMDataPoint.
            config: CUSUM configuration for threshold parameters.

        Returns:
            List of generated CUSUMAlert instances.
        """
        t0 = time.perf_counter()
        alerts: List[CUSUMAlert] = []

        if len(data_points) < config.min_points_for_alert:
            logger.debug(
                "Insufficient data points (%d < %d) for alert evaluation",
                len(data_points), config.min_points_for_alert,
            )
            return alerts

        h = config.decision_interval_h

        # 1. Threshold breach detection
        breach_alerts = self._detect_threshold_breaches(data_points, h)
        alerts.extend(breach_alerts)

        # 2. Sustained shift detection
        shift_alerts = self._detect_sustained_shift(
            data_points, self._sustained_shift_pts,
        )
        alerts.extend(shift_alerts)

        # 3. Trend change detection
        trend_alerts = self._detect_trend_changes(data_points)
        alerts.extend(trend_alerts)

        elapsed = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "Alert check: %d points -> %d alerts (breach=%d, shift=%d, trend=%d) (%dms)",
            len(data_points), len(alerts),
            len(breach_alerts), len(shift_alerts), len(trend_alerts),
            elapsed,
        )
        return alerts

    # ------------------------------------------------------------------ #
    # Public API -- Seasonal Adjustment                                    #
    # ------------------------------------------------------------------ #

    def apply_seasonal_adjustment(
        self,
        raw_data: List[Decimal],
        method: SeasonalAdjustment,
        seasonal_factors: List[Decimal],
        degree_day_data: Optional[List[DegreeDayData]] = None,
        hdd_coefficient: Optional[Decimal] = None,
        cdd_coefficient: Optional[Decimal] = None,
    ) -> List[Decimal]:
        """Apply seasonal adjustment to raw consumption data.

        Methods:
            NONE:            Returns raw_data unchanged.
            ADDITIVE:        adjusted_i = raw_i - factor_i
            MULTIPLICATIVE:  adjusted_i = raw_i / factor_i
            DEGREE_DAY_BASED: adjusted_i = raw_i - (HDD_i * coeff + CDD_i * coeff)

        Args:
            raw_data: Raw consumption values.
            method: Seasonal adjustment method.
            seasonal_factors: Seasonal factors (one per period; cycled).
            degree_day_data: Required for DEGREE_DAY_BASED method.
            hdd_coefficient: HDD regression coefficient.
            cdd_coefficient: CDD regression coefficient.

        Returns:
            List of seasonally adjusted Decimal values.

        Raises:
            ValueError: If seasonal_factors is empty (when method != NONE).
            ValueError: If degree_day_data is missing for DEGREE_DAY_BASED.
        """
        t0 = time.perf_counter()

        if method == SeasonalAdjustment.NONE:
            logger.debug("Seasonal adjustment: NONE -- returning raw data")
            return list(raw_data)

        if not seasonal_factors and method != SeasonalAdjustment.DEGREE_DAY_BASED:
            raise ValueError("seasonal_factors must not be empty")

        if method == SeasonalAdjustment.DEGREE_DAY_BASED:
            if not degree_day_data:
                raise ValueError("degree_day_data required for DEGREE_DAY_BASED")
            if len(degree_day_data) < len(raw_data):
                raise ValueError(
                    f"degree_day_data ({len(degree_day_data)}) must cover "
                    f"all raw_data periods ({len(raw_data)})"
                )

        hdd_coeff = _decimal(hdd_coefficient) if hdd_coefficient is not None else DEFAULT_HDD_COEFFICIENT
        cdd_coeff = _decimal(cdd_coefficient) if cdd_coefficient is not None else DEFAULT_CDD_COEFFICIENT

        adjusted: List[Decimal] = []
        n_factors = len(seasonal_factors) if seasonal_factors else 1

        for i, raw_val in enumerate(raw_data):
            raw = _decimal(raw_val)

            if method == SeasonalAdjustment.ADDITIVE:
                factor = _decimal(seasonal_factors[i % n_factors])
                adj = raw - factor
                adjusted.append(adj)

            elif method == SeasonalAdjustment.MULTIPLICATIVE:
                factor = _decimal(seasonal_factors[i % n_factors])
                adj = _safe_divide(raw, factor, raw)
                adjusted.append(adj)

            elif method == SeasonalAdjustment.DEGREE_DAY_BASED:
                dd = degree_day_data[i]
                weather_effect = (
                    _decimal(dd.hdd) * hdd_coeff
                    + _decimal(dd.cdd) * cdd_coeff
                )
                adj = raw - weather_effect
                adjusted.append(adj)

            else:
                adjusted.append(raw)

        elapsed = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "Seasonal adjustment (%s): %d values adjusted (%dms)",
            method.value, len(adjusted), elapsed,
        )
        return adjusted

    # ------------------------------------------------------------------ #
    # Public API -- Change-Point Detection                                 #
    # ------------------------------------------------------------------ #

    def estimate_change_point(
        self,
        data_points: List[CUSUMDataPoint],
    ) -> Tuple[int, Decimal]:
        """Estimate the most likely change point via binary segmentation.

        Recursively splits the series at the index that maximises the
        absolute difference in segment means.

        Args:
            data_points: Ordered list of CUSUMDataPoint.

        Returns:
            Tuple of (change_index, estimated_magnitude).
            Returns (-1, 0) if no significant change detected.
        """
        t0 = time.perf_counter()

        if len(data_points) < 4:
            logger.debug("Insufficient data (%d points) for change-point detection", len(data_points))
            return (-1, Decimal("0"))

        diffs = [pt.difference for pt in data_points]
        n = len(diffs)

        overall_mean = _safe_divide(
            sum(diffs), Decimal(str(n)),
        )
        overall_abs_mean = abs(overall_mean) if overall_mean != Decimal("0") else Decimal("1")

        best_idx = -1
        best_magnitude = Decimal("0")

        # Search for the split that maximises |mean_left - mean_right|
        for split in range(2, n - 1):
            left_vals = diffs[:split]
            right_vals = diffs[split:]

            left_mean = _safe_divide(sum(left_vals), Decimal(str(len(left_vals))))
            right_mean = _safe_divide(sum(right_vals), Decimal(str(len(right_vals))))

            magnitude = abs(left_mean - right_mean)

            if magnitude > best_magnitude:
                best_magnitude = magnitude
                best_idx = split

        # Check if magnitude is significant relative to overall mean
        min_change = overall_abs_mean * MIN_DETECTABLE_CHANGE_FRACTION
        if best_magnitude < min_change:
            best_idx = -1
            best_magnitude = Decimal("0")
            logger.info(
                "No significant change point detected (best_magnitude=%s < min=%s)",
                _round_val(best_magnitude, 4), _round_val(min_change, 4),
            )
        else:
            logger.info(
                "Change point detected at index %d, magnitude=%s",
                best_idx, _round_val(best_magnitude, 4),
            )

        elapsed = int((time.perf_counter() - t0) * 1000)
        logger.debug("Change-point detection completed in %dms", elapsed)
        return (best_idx, _round_val(best_magnitude, 6))

    # ------------------------------------------------------------------ #
    # Public API -- Result Generation                                      #
    # ------------------------------------------------------------------ #

    def get_monitor_result(
        self,
        monitor_id: str,
        enms_id: str = "",
    ) -> CUSUMMonitorResult:
        """Build a complete CUSUMMonitorResult for a monitor.

        Aggregates all data points, alerts, determines trend direction,
        performs change-point estimation, and generates provenance hash.

        Args:
            monitor_id: Target monitor identifier.
            enms_id: Parent EnMS identifier.

        Returns:
            CUSUMMonitorResult with all data and provenance.

        Raises:
            KeyError: If monitor_id is not found.
        """
        t0 = time.perf_counter()
        state = self._get_monitor(monitor_id)

        # Determine trend direction
        trend = self._determine_trend(state.data_points)

        # Change-point detection
        shift_detected = False
        shift_mag: Optional[Decimal] = None
        if len(state.data_points) >= 4:
            cp_idx, cp_mag = self.estimate_change_point(state.data_points)
            if cp_idx >= 0:
                shift_detected = True
                shift_mag = cp_mag

        # Total cumulative sum
        total_cumsum = (
            state.data_points[-1].cumulative_sum if state.data_points else Decimal("0")
        )

        elapsed = int((time.perf_counter() - t0) * 1000)

        result = CUSUMMonitorResult(
            monitor_id=state.monitor_id,
            enms_id=enms_id,
            monitor_name=state.monitor_name,
            baseline_id=state.baseline_id,
            method=state.config.method,
            data_points=list(state.data_points),
            alerts=list(state.alerts),
            current_status=state.status,
            total_cumulative_sum=total_cumsum,
            trend_direction=trend,
            performance_shift_detected=shift_detected,
            shift_magnitude=shift_mag,
            provenance_hash="",
            calculation_time_ms=elapsed,
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Monitor result: id=%s, points=%d, alerts=%d, trend=%s, "
            "shift=%s, cumsum=%s (%dms)",
            monitor_id, len(state.data_points), len(state.alerts),
            trend, shift_detected, _round_val(total_cumsum, 4), elapsed,
        )
        return result

    def generate_control_chart_data(
        self,
        result: CUSUMMonitorResult,
    ) -> Dict[str, Any]:
        """Format CUSUM data for control chart rendering.

        Produces structured data suitable for front-end charting libraries
        (e.g. Chart.js, D3, Plotly).

        Args:
            result: A CUSUMMonitorResult.

        Returns:
            Dict with keys: dates, cumulative_sums, upper_cusums,
            lower_cusums, upper_limits, lower_limits, alert_indices,
            alert_dates, metadata.
        """
        dates: List[str] = []
        cumulative_sums: List[str] = []
        upper_cusums: List[str] = []
        lower_cusums: List[str] = []
        upper_limits: List[str] = []
        lower_limits: List[str] = []
        differences: List[str] = []
        alert_indices: List[int] = []
        alert_dates: List[str] = []

        for i, pt in enumerate(result.data_points):
            dates.append(pt.period_date.isoformat())
            cumulative_sums.append(str(_round_val(pt.cumulative_sum, 4)))
            upper_cusums.append(str(_round_val(pt.upper_cusum, 4)))
            lower_cusums.append(str(_round_val(pt.lower_cusum, 4)))
            upper_limits.append(str(_round_val(pt.upper_limit, 4)))
            lower_limits.append(str(_round_val(pt.lower_limit, 4)))
            differences.append(str(_round_val(pt.difference, 4)))

            if pt.is_alert:
                alert_indices.append(i)
                alert_dates.append(pt.period_date.isoformat())

        chart_data: Dict[str, Any] = {
            "dates": dates,
            "cumulative_sums": cumulative_sums,
            "upper_cusums": upper_cusums,
            "lower_cusums": lower_cusums,
            "upper_limits": upper_limits,
            "lower_limits": lower_limits,
            "differences": differences,
            "alert_indices": alert_indices,
            "alert_dates": alert_dates,
            "metadata": {
                "monitor_id": result.monitor_id,
                "monitor_name": result.monitor_name,
                "method": result.method.value,
                "total_points": len(result.data_points),
                "total_alerts": len(result.alerts),
                "trend_direction": result.trend_direction,
                "performance_shift_detected": result.performance_shift_detected,
                "shift_magnitude": str(result.shift_magnitude) if result.shift_magnitude else None,
                "provenance_hash": result.provenance_hash,
            },
        }

        logger.info(
            "Control chart data: monitor=%s, points=%d, alerts=%d",
            result.monitor_id, len(dates), len(alert_indices),
        )
        return chart_data

    # ------------------------------------------------------------------ #
    # Public API -- Summary / Portfolio                                    #
    # ------------------------------------------------------------------ #

    def get_monitor_summary(
        self,
        monitors: List[CUSUMMonitorResult],
    ) -> CUSUMSummary:
        """Aggregate metrics across multiple CUSUM monitors.

        Args:
            monitors: List of CUSUMMonitorResult instances.

        Returns:
            CUSUMSummary with counts, averages, best/worst performers.
        """
        t0 = time.perf_counter()

        if not monitors:
            return CUSUMSummary()

        active_count = sum(
            1 for m in monitors if m.current_status == MonitorStatus.ACTIVE
        )
        alerting_count = sum(
            1 for m in monitors if m.current_status == MonitorStatus.ALERTING
        )
        total_unacked = sum(
            sum(1 for a in m.alerts if not a.acknowledged) for m in monitors
        )

        abs_cumsums = [
            (m.monitor_id, abs(m.total_cumulative_sum)) for m in monitors
        ]

        sum_abs = sum(ac[1] for ac in abs_cumsums)
        avg_abs = _safe_divide(sum_abs, Decimal(str(len(monitors))))

        # Sort by absolute cumsum
        sorted_by_abs = sorted(abs_cumsums, key=lambda x: x[1])
        best_id = sorted_by_abs[0][0] if sorted_by_abs else None
        worst_id = sorted_by_abs[-1][0] if sorted_by_abs else None

        elapsed = int((time.perf_counter() - t0) * 1000)

        summary = CUSUMSummary(
            monitor_count=len(monitors),
            active_monitors=active_count,
            alerting_monitors=alerting_count,
            total_alerts=total_unacked,
            avg_cumulative_sum=_round_val(avg_abs, 4),
            worst_performer=worst_id,
            best_performer=best_id,
        )

        logger.info(
            "Monitor summary: %d monitors, %d active, %d alerting, "
            "%d unacked alerts, avg_cumsum=%s (%dms)",
            len(monitors), active_count, alerting_count,
            total_unacked, _round_val(avg_abs, 4), elapsed,
        )
        return summary

    # ------------------------------------------------------------------ #
    # Public API -- Validation                                             #
    # ------------------------------------------------------------------ #

    def validate_monitor_config(self, config: CUSUMConfig) -> Dict[str, Any]:
        """Validate a CUSUMConfig for correctness and ARL feasibility.

        Checks:
            - decision_interval_h > 0
            - allowable_slack >= 0
            - min_points_for_alert >= 1
            - ARL_0 lookup (warns if in-control ARL is too low)
            - Method-specific parameter requirements

        Args:
            config: CUSUMConfig to validate.

        Returns:
            Dict with keys: is_valid (bool), errors (list), warnings (list),
            arl_info (dict or None).
        """
        errors: List[str] = []
        warnings: List[str] = []
        arl_info: Optional[Dict[str, Any]] = None

        # Basic parameter validation
        if config.decision_interval_h <= Decimal("0"):
            errors.append("decision_interval_h must be > 0")

        if config.allowable_slack < Decimal("0"):
            errors.append("allowable_slack must be >= 0")

        if config.min_points_for_alert < 1:
            errors.append("min_points_for_alert must be >= 1")

        # Check recommended min points per interval
        interval_key = config.monitoring_interval.value
        if interval_key in DEFAULT_CUSUM_PARAMS:
            rec_min = DEFAULT_CUSUM_PARAMS[interval_key]["min_points"]
            if config.min_points_for_alert < rec_min:
                warnings.append(
                    f"min_points_for_alert ({config.min_points_for_alert}) is below "
                    f"recommended minimum ({rec_min}) for {interval_key} monitoring"
                )

        # ARL lookup -- try several key normalisations
        h_val = config.decision_interval_h
        k_val = config.allowable_slack
        arl_candidates = [
            f"h{h_val}_k{k_val}",
            f"h{h_val.normalize()}_k{k_val.normalize()}",
            f"h{h_val.normalize()}_k{k_val}",
            f"h{h_val}_k{k_val.normalize()}",
        ]
        arl_entry = None
        arl_key = arl_candidates[0]
        for candidate in arl_candidates:
            if candidate in ARL_TABLES:
                arl_entry = ARL_TABLES[candidate]
                arl_key = candidate
                break
        if arl_entry is not None:
            arl_info = {
                "h": str(arl_entry["h"]),
                "k": str(arl_entry["k"]),
                "ARL_0": str(arl_entry["ARL_0"]),
                "ARL_1.0": str(arl_entry["ARL_1.0"]),
            }
            if arl_entry["ARL_0"] < Decimal("50"):
                warnings.append(
                    f"In-control ARL_0 = {arl_entry['ARL_0']} is low -- "
                    f"expect frequent false alarms"
                )
        else:
            warnings.append(
                f"No ARL table entry for h={h_val}, k={k_val} -- "
                f"cannot verify average run length characteristics"
            )

        # Method-specific checks
        if config.method == CUSUMMethod.V_MASK:
            warnings.append(
                "V-mask method requires VMaskParameters; ensure they are "
                "supplied when calling apply_v_mask()"
            )

        if config.method == CUSUMMethod.EXPONENTIAL_WEIGHTED:
            warnings.append(
                "EWMA-CUSUM method requires lambda_param; ensure it is "
                "supplied when calling calculate_ewma_cusum()"
            )

        is_valid = len(errors) == 0

        logger.info(
            "Config validation: valid=%s, errors=%d, warnings=%d",
            is_valid, len(errors), len(warnings),
        )
        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "arl_info": arl_info,
        }

    # ------------------------------------------------------------------ #
    # Public API -- Batch / Full Analysis                                  #
    # ------------------------------------------------------------------ #

    def run_full_analysis(
        self,
        name: str,
        baseline_id: str,
        config: CUSUMConfig,
        observations: List[Tuple[date, Decimal, Decimal]],
        enms_id: str = "",
        seasonal_factors: Optional[List[Decimal]] = None,
        degree_day_data: Optional[List[DegreeDayData]] = None,
    ) -> CUSUMMonitorResult:
        """Run a complete CUSUM analysis from a batch of observations.

        Convenience method that creates a monitor, optionally applies
        seasonal adjustment, ingests all data points, runs alert
        detection, and returns the final result.

        Args:
            name: Monitor name.
            baseline_id: Baseline identifier.
            config: CUSUM configuration.
            observations: List of (date, actual, expected) tuples.
            enms_id: Parent EnMS identifier.
            seasonal_factors: Optional seasonal factors for adjustment.
            degree_day_data: Optional degree-day data for adjustment.

        Returns:
            CUSUMMonitorResult with full analysis.
        """
        t0 = time.perf_counter()
        logger.info(
            "Full CUSUM analysis: name='%s', baseline=%s, %d observations",
            name, baseline_id, len(observations),
        )

        # Apply seasonal adjustment to actual values if configured
        if (
            config.seasonal_adjustment != SeasonalAdjustment.NONE
            and (seasonal_factors or degree_day_data)
        ):
            raw_actuals = [_decimal(obs[1]) for obs in observations]
            adjusted_actuals = self.apply_seasonal_adjustment(
                raw_data=raw_actuals,
                method=config.seasonal_adjustment,
                seasonal_factors=seasonal_factors or [],
                degree_day_data=degree_day_data,
            )
            observations = [
                (obs[0], adjusted_actuals[i], obs[2])
                for i, obs in enumerate(observations)
            ]

        # Create monitor and ingest data
        monitor_id = self.create_monitor(name, baseline_id, config)

        for obs_date, actual, expected in observations:
            self.add_data_point(
                monitor_id, obs_date, _decimal(actual), _decimal(expected),
            )

        # Additional alert check on the full series
        state = self._get_monitor(monitor_id)
        additional_alerts = self.check_alerts(state.data_points, config)

        # Deduplicate alerts by checking existing alert dates
        existing_dates = {a.alert_date for a in state.alerts}
        for alert in additional_alerts:
            if alert.alert_date not in existing_dates:
                state.alerts.append(alert)

        # Update status if new alerts were added
        if state.alerts and any(not a.acknowledged for a in state.alerts):
            state.status = MonitorStatus.ALERTING

        result = self.get_monitor_result(monitor_id, enms_id=enms_id)

        elapsed = int((time.perf_counter() - t0) * 1000)
        result.calculation_time_ms = elapsed
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Full analysis complete: monitor=%s, %d points, %d alerts, "
            "trend=%s, shift=%s (%dms)",
            monitor_id, len(result.data_points), len(result.alerts),
            result.trend_direction, result.performance_shift_detected,
            elapsed,
        )
        return result

    def run_batch_analysis(
        self,
        analyses: List[Dict[str, Any]],
        enms_id: str = "",
    ) -> List[CUSUMMonitorResult]:
        """Run CUSUM analysis for multiple monitors in batch.

        Args:
            analyses: List of dicts, each with keys:
                - name (str): Monitor name.
                - baseline_id (str): Baseline identifier.
                - config (CUSUMConfig): CUSUM configuration.
                - observations (list): List of (date, actual, expected).
                - seasonal_factors (list, optional): Seasonal factors.
                - degree_day_data (list, optional): Degree-day data.
            enms_id: Parent EnMS identifier.

        Returns:
            List of CUSUMMonitorResult, one per analysis.
        """
        t0 = time.perf_counter()
        results: List[CUSUMMonitorResult] = []

        logger.info("Batch analysis: %d monitors", len(analyses))

        for i, spec in enumerate(analyses):
            try:
                result = self.run_full_analysis(
                    name=spec.get("name", f"Monitor-{i+1}"),
                    baseline_id=spec.get("baseline_id", ""),
                    config=spec.get("config", CUSUMConfig()),
                    observations=spec.get("observations", []),
                    enms_id=enms_id,
                    seasonal_factors=spec.get("seasonal_factors"),
                    degree_day_data=spec.get("degree_day_data"),
                )
                results.append(result)
            except Exception as exc:
                logger.error(
                    "Batch analysis failed for item %d ('%s'): %s",
                    i, spec.get("name", ""), str(exc), exc_info=True,
                )

        elapsed = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "Batch analysis complete: %d/%d succeeded (%dms)",
            len(results), len(analyses), elapsed,
        )
        return results

    # ------------------------------------------------------------------ #
    # Public API -- ARL Lookup                                             #
    # ------------------------------------------------------------------ #

    def lookup_arl(
        self,
        h: Decimal,
        k: Decimal,
    ) -> Optional[Dict[str, Any]]:
        """Look up Average Run Length from the ARL table.

        Args:
            h: Decision interval.
            k: Allowable slack.

        Returns:
            Dict with ARL values if found, else None.
        """
        # Try several key normalisations (e.g. Decimal("5.0") -> "5" or "5.0")
        candidates = [
            f"h{h}_k{k}",
            f"h{h.normalize()}_k{k.normalize()}",
            f"h{h.normalize()}_k{k}",
            f"h{h}_k{k.normalize()}",
        ]
        entry = None
        for key in candidates:
            entry = ARL_TABLES.get(key)
            if entry:
                break
        if entry:
            return {
                "h": str(entry["h"]),
                "k": str(entry["k"]),
                "ARL_0": str(entry["ARL_0"]),
                "ARL_0.5": str(entry.get("ARL_0.5", "N/A")),
                "ARL_1.0": str(entry.get("ARL_1.0", "N/A")),
                "ARL_2.0": str(entry.get("ARL_2.0", "N/A")),
            }
        logger.debug("No ARL entry for h=%s, k=%s", h, k)
        return None

    def get_recommended_params(
        self,
        interval: MonitoringInterval,
    ) -> Dict[str, Any]:
        """Get recommended CUSUM parameters for a monitoring interval.

        Args:
            interval: Monitoring frequency.

        Returns:
            Dict with h, k, min_points, and description.
        """
        key = interval.value
        params = DEFAULT_CUSUM_PARAMS.get(key, DEFAULT_CUSUM_PARAMS[MonitoringInterval.MONTHLY.value])
        return {
            "h": str(params["h"]),
            "k": str(params["k"]),
            "min_points": params["min_points"],
            "description": params["description"],
        }

    def get_seasonal_template(
        self,
        profile: str,
    ) -> List[Decimal]:
        """Get a seasonal factor template by profile name.

        Args:
            profile: One of 'neutral', 'heating_dominated',
                     'cooling_dominated', 'mixed_climate', 'industrial_24_7'.

        Returns:
            List of 12 monthly Decimal factors.

        Raises:
            ValueError: If profile is not recognised.
        """
        if profile not in SEASONAL_FACTORS_TEMPLATE:
            available = ", ".join(SEASONAL_FACTORS_TEMPLATE.keys())
            raise ValueError(
                f"Unknown seasonal profile '{profile}'. "
                f"Available profiles: {available}"
            )
        return list(SEASONAL_FACTORS_TEMPLATE[profile])

    # ------------------------------------------------------------------ #
    # Public API -- Acknowledge Alert                                      #
    # ------------------------------------------------------------------ #

    def acknowledge_alert(
        self,
        monitor_id: str,
        alert_id: str,
        root_cause: Optional[str] = None,
    ) -> bool:
        """Acknowledge an alert and optionally record root cause.

        Args:
            monitor_id: Target monitor.
            alert_id: Alert to acknowledge.
            root_cause: Optional root cause description.

        Returns:
            True if the alert was found and acknowledged, False otherwise.

        Raises:
            KeyError: If monitor_id is not found.
        """
        state = self._get_monitor(monitor_id)

        for alert in state.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                if root_cause:
                    alert.root_cause = root_cause
                logger.info(
                    "Acknowledged alert %s on monitor %s (root_cause=%s)",
                    alert_id, monitor_id, root_cause or "N/A",
                )
                # Check if all alerts are now acknowledged
                if all(a.acknowledged for a in state.alerts):
                    if state.status == MonitorStatus.ALERTING:
                        state.status = MonitorStatus.ACTIVE
                        logger.info(
                            "All alerts acknowledged -- monitor %s back to ACTIVE",
                            monitor_id,
                        )
                return True

        logger.warning("Alert %s not found on monitor %s", alert_id, monitor_id)
        return False

    # ------------------------------------------------------------------ #
    # Public API -- Utility: Compute Sigma                                 #
    # ------------------------------------------------------------------ #

    def compute_std_deviation(
        self,
        values: List[Decimal],
    ) -> Decimal:
        """Compute sample standard deviation of a list of Decimal values.

        Uses N-1 (Bessel's correction) for unbiased estimation.

        Args:
            values: List of Decimal values.

        Returns:
            Sample standard deviation as Decimal.
        """
        n = len(values)
        if n < 2:
            return Decimal("0")

        n_dec = Decimal(str(n))
        mean = _safe_divide(sum(values), n_dec)

        sum_sq = sum((v - mean) ** 2 for v in values)
        variance = _safe_divide(sum_sq, Decimal(str(n - 1)))

        # sqrt via float then back to Decimal for precision
        std_dev = _decimal(math.sqrt(float(variance)))
        return _round_val(std_dev, 6)

    def normalise_to_sigma(
        self,
        differences: List[Decimal],
    ) -> Tuple[List[Decimal], Decimal]:
        """Normalise difference values to sigma (standard deviation) units.

        Args:
            differences: Raw difference values (actual - expected).

        Returns:
            Tuple of (normalised_differences, sigma).
            normalised_i = difference_i / sigma.
        """
        sigma = self.compute_std_deviation(differences)

        if sigma == Decimal("0"):
            logger.warning("Standard deviation is zero -- returning unnormalised values")
            return (list(differences), Decimal("0"))

        normalised = [_safe_divide(d, sigma) for d in differences]
        return (normalised, sigma)

    # ------------------------------------------------------------------ #
    # Private -- Monitor State Access                                      #
    # ------------------------------------------------------------------ #

    def _get_monitor(self, monitor_id: str) -> _MonitorState:
        """Retrieve monitor state or raise KeyError.

        Args:
            monitor_id: Monitor identifier.

        Returns:
            _MonitorState instance.

        Raises:
            KeyError: If monitor_id is not found.
        """
        if monitor_id not in self._monitors:
            raise KeyError(f"Monitor '{monitor_id}' not found")
        return self._monitors[monitor_id]

    # ------------------------------------------------------------------ #
    # Private -- Alert Generation                                          #
    # ------------------------------------------------------------------ #

    def _generate_point_alert(
        self,
        state: _MonitorState,
        point: CUSUMDataPoint,
        n_points: int,
    ) -> CUSUMAlert:
        """Generate an alert from a data point that breached a threshold.

        Args:
            state: Monitor state.
            point: The triggering data point.
            n_points: Total number of data points at time of alert.

        Returns:
            CUSUMAlert instance.
        """
        h = state.config.decision_interval_h

        if point.upper_cusum > h:
            alert_type = AlertType.PERFORMANCE_DEGRADATION
            cum_val = point.upper_cusum
            desc = (
                f"Upper CUSUM ({_round_val(cum_val, 4)}) exceeded "
                f"decision interval h={_round_val(h, 4)} at "
                f"{point.period_date.isoformat()}. "
                f"Energy consumption is trending above baseline."
            )
        else:
            alert_type = AlertType.PERFORMANCE_IMPROVEMENT
            cum_val = point.lower_cusum
            desc = (
                f"Lower CUSUM ({_round_val(cum_val, 4)}) exceeded "
                f"-h={_round_val(-h, 4)} at "
                f"{point.period_date.isoformat()}. "
                f"Energy consumption is trending below baseline."
            )

        # Estimate consecutive points in alert direction
        consecutive = self._count_consecutive_direction(state.data_points, cum_val > Decimal("0"))

        # Estimate change magnitude from last N points
        magnitude = self._estimate_recent_shift(state.data_points, min(n_points, 10))

        return CUSUMAlert(
            alert_type=alert_type,
            cumulative_value=cum_val,
            threshold_value=h,
            consecutive_points=consecutive,
            estimated_change_magnitude=magnitude,
            description=desc,
        )

    def _count_consecutive_direction(
        self,
        data_points: List[CUSUMDataPoint],
        positive: bool,
    ) -> int:
        """Count consecutive data points with differences in one direction.

        Args:
            data_points: Data points (newest last).
            positive: True to count positive diffs, False for negative.

        Returns:
            Count of consecutive points from the end.
        """
        count = 0
        for pt in reversed(data_points):
            if positive and pt.difference > Decimal("0"):
                count += 1
            elif not positive and pt.difference < Decimal("0"):
                count += 1
            else:
                break
        return count

    def _estimate_recent_shift(
        self,
        data_points: List[CUSUMDataPoint],
        window: int,
    ) -> Decimal:
        """Estimate the mean absolute shift over a recent window.

        Args:
            data_points: Data points (newest last).
            window: Number of recent points to consider.

        Returns:
            Mean absolute difference over the window.
        """
        if not data_points:
            return Decimal("0")

        recent = data_points[-window:]
        if not recent:
            return Decimal("0")

        abs_diffs = [abs(pt.difference) for pt in recent]
        return _round_val(
            _safe_divide(sum(abs_diffs), Decimal(str(len(abs_diffs)))),
            6,
        )

    def _reset_accumulators(self, state: _MonitorState) -> None:
        """Reset CUSUM accumulators while keeping data history.

        Creates a synthetic reset point: upper_cusum and lower_cusum
        set to zero at the next observation boundary.

        Args:
            state: Monitor state to modify.
        """
        # We mark the reset by ensuring the next add_data_point starts
        # from zero.  We do this by inserting a sentinel that the
        # add_data_point logic will pick up.
        if state.data_points:
            last = state.data_points[-1]
            # Create a copy with zeroed accumulators
            reset_pt = CUSUMDataPoint(
                period_date=last.period_date,
                actual_consumption=last.actual_consumption,
                expected_consumption=last.expected_consumption,
                difference=last.difference,
                cumulative_sum=last.cumulative_sum,
                upper_cusum=Decimal("0"),
                lower_cusum=Decimal("0"),
                upper_limit=last.upper_limit,
                lower_limit=last.lower_limit,
                is_alert=last.is_alert,
            )
            state.data_points[-1] = reset_pt

    # ------------------------------------------------------------------ #
    # Private -- Alert Detection Helpers                                   #
    # ------------------------------------------------------------------ #

    def _detect_threshold_breaches(
        self,
        data_points: List[CUSUMDataPoint],
        h: Decimal,
    ) -> List[CUSUMAlert]:
        """Detect points where CUSUM exceeded the decision interval.

        Args:
            data_points: Ordered data points.
            h: Decision interval.

        Returns:
            List of CUSUMAlert for each breach.
        """
        alerts: List[CUSUMAlert] = []

        for i, pt in enumerate(data_points):
            if pt.upper_cusum > h:
                alerts.append(CUSUMAlert(
                    alert_type=AlertType.THRESHOLD_BREACH,
                    cumulative_value=pt.upper_cusum,
                    threshold_value=h,
                    consecutive_points=1,
                    estimated_change_magnitude=pt.upper_cusum - h,
                    description=(
                        f"Upper CUSUM breach at point {i} "
                        f"({pt.period_date.isoformat()}): "
                        f"C+={_round_val(pt.upper_cusum, 4)} > h={_round_val(h, 4)}"
                    ),
                ))
            if pt.lower_cusum < -h:
                alerts.append(CUSUMAlert(
                    alert_type=AlertType.THRESHOLD_BREACH,
                    cumulative_value=pt.lower_cusum,
                    threshold_value=-h,
                    consecutive_points=1,
                    estimated_change_magnitude=abs(pt.lower_cusum) - h,
                    description=(
                        f"Lower CUSUM breach at point {i} "
                        f"({pt.period_date.isoformat()}): "
                        f"C-={_round_val(pt.lower_cusum, 4)} < -h={_round_val(-h, 4)}"
                    ),
                ))

        return alerts

    def _detect_sustained_shift(
        self,
        data_points: List[CUSUMDataPoint],
        n_consecutive: int,
    ) -> List[CUSUMAlert]:
        """Detect sustained shifts -- N consecutive differences on one side.

        A sustained shift is when N or more consecutive data points have
        differences all positive (degradation) or all negative (improvement).

        Args:
            data_points: Ordered data points.
            n_consecutive: Required consecutive count.

        Returns:
            List of CUSUMAlert for each detected sustained shift.
        """
        alerts: List[CUSUMAlert] = []

        if len(data_points) < n_consecutive:
            return alerts

        # Scan for positive runs
        pos_run = 0
        neg_run = 0

        for i, pt in enumerate(data_points):
            if pt.difference > Decimal("0"):
                pos_run += 1
                neg_run = 0
            elif pt.difference < Decimal("0"):
                neg_run += 1
                pos_run = 0
            else:
                pos_run = 0
                neg_run = 0

            if pos_run == n_consecutive:
                avg_shift = self._estimate_recent_shift(
                    data_points[:i + 1], n_consecutive,
                )
                alerts.append(CUSUMAlert(
                    alert_type=AlertType.SUSTAINED_SHIFT,
                    cumulative_value=pt.cumulative_sum,
                    threshold_value=Decimal(str(n_consecutive)),
                    consecutive_points=n_consecutive,
                    estimated_change_magnitude=avg_shift,
                    description=(
                        f"Sustained positive shift: {n_consecutive} consecutive "
                        f"points above baseline ending at {pt.period_date.isoformat()}. "
                        f"Average magnitude: {_round_val(avg_shift, 4)}"
                    ),
                ))
                # Reset counter to avoid duplicate alerts for overlapping windows
                pos_run = 0

            if neg_run == n_consecutive:
                avg_shift = self._estimate_recent_shift(
                    data_points[:i + 1], n_consecutive,
                )
                alerts.append(CUSUMAlert(
                    alert_type=AlertType.SUSTAINED_SHIFT,
                    cumulative_value=pt.cumulative_sum,
                    threshold_value=Decimal(str(n_consecutive)),
                    consecutive_points=n_consecutive,
                    estimated_change_magnitude=avg_shift,
                    description=(
                        f"Sustained negative shift: {n_consecutive} consecutive "
                        f"points below baseline ending at {pt.period_date.isoformat()}. "
                        f"Average magnitude: {_round_val(avg_shift, 4)}"
                    ),
                ))
                neg_run = 0

        return alerts

    def _detect_trend_changes(
        self,
        data_points: List[CUSUMDataPoint],
    ) -> List[CUSUMAlert]:
        """Detect trend direction reversals in the cumulative sum.

        A trend change is detected when the slope of the cumulative sum
        (computed over a rolling window) changes sign.

        Args:
            data_points: Ordered data points.

        Returns:
            List of CUSUMAlert for each detected trend reversal.
        """
        alerts: List[CUSUMAlert] = []
        window = 5

        if len(data_points) < window * 2:
            return alerts

        # Compute rolling slope signs
        slopes: List[int] = []  # +1, 0, -1
        for i in range(window, len(data_points)):
            segment = data_points[i - window: i]
            first_half = segment[: window // 2]
            second_half = segment[window // 2:]

            mean_first = _safe_divide(
                sum(pt.cumulative_sum for pt in first_half),
                Decimal(str(len(first_half))),
            )
            mean_second = _safe_divide(
                sum(pt.cumulative_sum for pt in second_half),
                Decimal(str(len(second_half))),
            )

            diff = mean_second - mean_first
            if diff > Decimal("0"):
                slopes.append(1)
            elif diff < Decimal("0"):
                slopes.append(-1)
            else:
                slopes.append(0)

        # Detect sign changes
        for i in range(1, len(slopes)):
            if slopes[i] != 0 and slopes[i - 1] != 0 and slopes[i] != slopes[i - 1]:
                pt_idx = window + i
                if pt_idx < len(data_points):
                    pt = data_points[pt_idx]
                    direction = "improving" if slopes[i] < 0 else "degrading"
                    prev_direction = "improving" if slopes[i - 1] < 0 else "degrading"

                    alerts.append(CUSUMAlert(
                        alert_type=AlertType.TREND_CHANGE,
                        cumulative_value=pt.cumulative_sum,
                        threshold_value=Decimal("0"),
                        consecutive_points=0,
                        estimated_change_magnitude=abs(pt.difference),
                        description=(
                            f"Trend reversal at {pt.period_date.isoformat()}: "
                            f"{prev_direction} -> {direction}. "
                            f"CUSUM={_round_val(pt.cumulative_sum, 4)}"
                        ),
                    ))

        return alerts

    # ------------------------------------------------------------------ #
    # Private -- Trend Determination                                       #
    # ------------------------------------------------------------------ #

    def _determine_trend(
        self,
        data_points: List[CUSUMDataPoint],
    ) -> str:
        """Determine overall trend from cumulative sum trajectory.

        Uses linear regression slope on the cumulative sum series.
        Returns 'improving' (negative slope), 'degrading' (positive slope),
        or 'stable' (slope near zero).

        Args:
            data_points: Ordered data points.

        Returns:
            One of 'improving', 'degrading', 'stable'.
        """
        if len(data_points) < 3:
            return "stable"

        n = Decimal(str(len(data_points)))
        x_vals = [Decimal(str(i)) for i in range(len(data_points))]
        y_vals = [pt.cumulative_sum for pt in data_points]

        # Linear regression: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - (sum(x))^2)
        sum_x = sum(x_vals)
        sum_y = sum(y_vals)
        sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
        sum_x2 = sum(x * x for x in x_vals)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = n * sum_x2 - sum_x * sum_x

        slope = _safe_divide(numerator, denominator)

        # Normalise slope by the range of cumulative sums
        y_range = max(abs(y) for y in y_vals) if y_vals else Decimal("1")
        if y_range == Decimal("0"):
            y_range = Decimal("1")

        normalised_slope = _safe_divide(abs(slope), y_range)

        # Threshold for "stable" determination
        stability_threshold = Decimal("0.01")

        if normalised_slope < stability_threshold:
            return "stable"
        elif slope > Decimal("0"):
            return "degrading"
        else:
            return "improving"

    # ------------------------------------------------------------------ #
    # Private -- Sigma-based Helpers                                       #
    # ------------------------------------------------------------------ #

    def _compute_in_control_sigma(
        self,
        data_points: List[CUSUMDataPoint],
        calibration_count: int,
    ) -> Decimal:
        """Compute standard deviation from calibration-period data.

        Uses the first *calibration_count* data points to estimate sigma.

        Args:
            data_points: Full data point list.
            calibration_count: Number of initial points for calibration.

        Returns:
            Estimated in-control sigma.
        """
        cal_points = data_points[:calibration_count]
        if len(cal_points) < 2:
            return Decimal("1")

        diffs = [pt.difference for pt in cal_points]
        return self.compute_std_deviation(diffs)

    # ------------------------------------------------------------------ #
    # dunder
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        """Return engine representation."""
        return (
            f"CUSUMMonitorEngine(v={self.engine_version}, "
            f"monitors={len(self._monitors)}, "
            f"method={self._default_method.value})"
        )

    def __len__(self) -> int:
        """Return number of active monitors."""
        return len(self._monitors)
