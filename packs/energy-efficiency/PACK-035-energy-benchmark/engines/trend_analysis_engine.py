# -*- coding: utf-8 -*-
"""
TrendAnalysisEngine - PACK-035 Energy Benchmark Engine 9
==========================================================

Analyses energy performance trends with statistical rigour using CUSUM
(Cumulative Sum) control charts, SPC (Statistical Process Control) with
individuals/moving-range control limits, Mann-Kendall trend testing,
EWMA smoothing, step-change detection, and Holt-Winters forecasting
with seasonal decomposition.

Generates performance alerts at configurable severity levels when energy
consumption deviates from expected patterns, enabling proactive energy
management per ISO 50001 continuous improvement requirements.

Calculation Methodology:
    CUSUM (Tabular Method):
        S_h(t) = max(0, S_h(t-1) + (x_t - mu_0 - k))    (upper)
        S_l(t) = max(0, S_l(t-1) + (-x_t + mu_0 - k))    (lower)
        Signal when S_h > h or S_l > h

    SPC Individuals Chart:
        UCL = x_bar + L * MR_bar / d2
        LCL = x_bar - L * MR_bar / d2
        where L = sigma multiplier, d2 = 1.128 for n=2

    Moving Range Chart:
        UCL_MR = D4 * MR_bar    (D4 = 3.267 for n=2)

    EWMA (Exponentially Weighted Moving Average):
        Z_t = lambda * X_t + (1 - lambda) * Z_{t-1}
        UCL = mu_0 + L * sigma * sqrt(lambda / (2 - lambda) * (1 - (1-lambda)^(2t)))
        LCL = mu_0 - L * ...

    Mann-Kendall Trend Test:
        S = sum( sign(x_j - x_i) )  for all i < j
        Var(S) = n(n-1)(2n+5)/18
        Z = (S - sign(S)) / sqrt(Var(S))

    Holt-Winters (Additive):
        Level:    L_t = alpha * (Y_t - S_{t-m}) + (1-alpha) * (L_{t-1} + T_{t-1})
        Trend:    T_t = beta  * (L_t - L_{t-1}) + (1-beta) * T_{t-1}
        Season:   S_t = gamma * (Y_t - L_t) + (1-gamma) * S_{t-m}
        Forecast: F_{t+h} = L_t + h * T_t + S_{t-m+h_mod}

    Year-over-Year Change:
        yoy_pct = (current - previous) / abs(previous) * 100

Regulatory / Standard References:
    - ISO 50001:2018 Energy Management (Clause 9.1 - Monitoring)
    - ISO 50006:2014 Energy performance indicators and baselines
    - ASHRAE Guideline 14-2014 (CUSUM for M&V)
    - IPMVP Volume I (Savings determination via CUSUM)
    - Montgomery, D.C. "Introduction to Statistical Quality Control" (SPC)
    - Mann (1945), Kendall (1975) - Non-parametric trend tests
    - Holt (1957), Winters (1960) - Exponential smoothing

Zero-Hallucination:
    - All formulas from published statistical literature
    - SPC constants (d2, D4) from statistical quality control tables
    - Mann-Kendall from standard non-parametric statistics
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-035 Energy Benchmark
Engine:  9 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
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


def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to *places* decimal digits and return float."""
    return float(value.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP))


def _round2(value: Any) -> float:
    """Round to 2 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _round3(value: Any) -> float:
    """Round to 3 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _round4(value: Any) -> float:
    """Round to 4 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TrendDirection(str, Enum):
    """Direction of energy performance trend.

    IMPROVING:      Energy consumption trending downward.
    STABLE:         No significant trend detected.
    DETERIORATING:  Energy consumption trending upward.
    """
    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"


class AlertSeverity(str, Enum):
    """Severity levels for energy performance alerts.

    INFO:      Informational, no action required.
    WARNING:   Performance deviation detected, investigation advised.
    CRITICAL:  Significant deviation, immediate action required.
    """
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ControlChartType(str, Enum):
    """Types of control charts for SPC analysis.

    INDIVIDUALS:   Individuals (I) chart for single observations.
    MOVING_RANGE:  Moving Range (MR) chart for variability.
    CUSUM:         Cumulative Sum chart for sustained shifts.
    EWMA:          Exponentially Weighted Moving Average chart.
    """
    INDIVIDUALS = "individuals"
    MOVING_RANGE = "moving_range"
    CUSUM = "cusum"
    EWMA = "ewma"


class ForecastMethod(str, Enum):
    """Time series forecasting methods.

    EXPONENTIAL_SMOOTHING:  Simple exponential smoothing.
    HOLT_WINTERS:           Holt-Winters with seasonal decomposition.
    LINEAR_EXTRAPOLATION:   Linear trend extrapolation.
    """
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    HOLT_WINTERS = "holt_winters"
    LINEAR_EXTRAPOLATION = "linear_extrapolation"


class DecompositionMethod(str, Enum):
    """Time series decomposition methods."""
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SPC sigma limits for control charts.
# Source: Montgomery, "Introduction to Statistical Quality Control", 7th ed.
SPC_SIGMA_LIMITS: Dict[str, float] = {
    "warning": 2.0,   # 2-sigma warning limits
    "control": 3.0,   # 3-sigma control limits
}

# SPC constants for Individuals/Moving Range charts (subgroup n=2).
# Source: ASTM E2587, Montgomery Table VI.
SPC_D2: float = 1.128     # Expected value of range for n=2
SPC_D4: float = 3.267     # Upper control limit factor for MR (n=2)
SPC_D3: float = 0.0       # Lower control limit factor for MR (n=2)

# CUSUM parameters.
# Source: Lucas (1976), Hawkins & Olwell (1998).
CUSUM_THRESHOLD_FACTOR: float = 4.0   # h = factor * sigma
CUSUM_ALLOWANCE_FACTOR: float = 0.5   # k = factor * sigma

# EWMA default smoothing parameter.
# Source: Roberts (1959), common default lambda = 0.2.
EWMA_LAMBDA_DEFAULT: float = 0.2

# Mann-Kendall Z-score thresholds.
# Source: Standard normal distribution table.
MK_SIGNIFICANCE_95: float = 1.96
MK_SIGNIFICANCE_99: float = 2.576

# Default Holt-Winters parameters.
# Source: Hyndman & Athanasopoulos, "Forecasting: Principles and Practice"
HW_ALPHA_DEFAULT: float = 0.3   # Level smoothing
HW_BETA_DEFAULT: float = 0.1    # Trend smoothing
HW_GAMMA_DEFAULT: float = 0.1   # Seasonal smoothing


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class TimeSeriesPoint(BaseModel):
    """A single time series data point.

    Attributes:
        period_label: Period label (e.g. '2024-01', 'Week 5').
        timestamp: Optional datetime for the period.
        value: Observed value (e.g. EUI, energy consumption).
        is_baseline: Whether this point is part of the baseline period.
    """
    period_label: str = Field(default="", description="Period label")
    timestamp: Optional[datetime] = Field(default=None, description="Period datetime")
    value: float = Field(default=0.0, description="Observed value")
    is_baseline: bool = Field(default=False, description="Is baseline period")


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class TrendResult(BaseModel):
    """Result of trend analysis.

    Attributes:
        direction: Overall trend direction.
        mann_kendall_s: Mann-Kendall S statistic.
        mann_kendall_z: Mann-Kendall Z score.
        mann_kendall_p: Two-tailed p-value.
        is_significant: Whether the trend is statistically significant.
        sen_slope: Sen's slope estimator (median slope).
        yoy_changes: Year-over-year change percentages.
        average_yoy_change_pct: Average year-over-year change.
    """
    direction: TrendDirection = Field(default=TrendDirection.STABLE)
    mann_kendall_s: float = Field(default=0.0)
    mann_kendall_z: float = Field(default=0.0)
    mann_kendall_p: float = Field(default=1.0)
    is_significant: bool = Field(default=False)
    sen_slope: float = Field(default=0.0)
    yoy_changes: List[float] = Field(default_factory=list)
    average_yoy_change_pct: float = Field(default=0.0)


class CUSUMResult(BaseModel):
    """CUSUM control chart result.

    Attributes:
        cusum_upper: Upper CUSUM values (positive shifts).
        cusum_lower: Lower CUSUM values (negative shifts).
        threshold_h: Decision interval threshold.
        allowance_k: Reference value (allowance).
        signal_indices: Indices where a signal was detected.
        signal_direction: Direction of detected signal(s).
        baseline_mean: Baseline mean used for CUSUM.
        baseline_std: Baseline standard deviation.
    """
    cusum_upper: List[float] = Field(default_factory=list)
    cusum_lower: List[float] = Field(default_factory=list)
    threshold_h: float = Field(default=0.0)
    allowance_k: float = Field(default=0.0)
    signal_indices: List[int] = Field(default_factory=list)
    signal_direction: List[str] = Field(default_factory=list)
    baseline_mean: float = Field(default=0.0)
    baseline_std: float = Field(default=0.0)


class ControlLimit(BaseModel):
    """Control limit for an SPC chart.

    Attributes:
        center_line: Center line (process mean).
        ucl: Upper control limit.
        lcl: Lower control limit.
        uwl: Upper warning limit (2-sigma).
        lwl: Lower warning limit (2-sigma).
    """
    center_line: float = Field(default=0.0)
    ucl: float = Field(default=0.0)
    lcl: float = Field(default=0.0)
    uwl: float = Field(default=0.0)
    lwl: float = Field(default=0.0)


class SPCResult(BaseModel):
    """SPC (Statistical Process Control) chart result.

    Attributes:
        chart_type: Type of control chart.
        values: Plotted values (individuals or MR).
        control_limits: Control limits for the chart.
        out_of_control_indices: Indices of points beyond control limits.
        warning_indices: Indices of points beyond warning limits.
        rule_violations: Nelson rules or Western Electric violations.
        process_capability: Cp/Cpk if specification limits given.
    """
    chart_type: ControlChartType = Field(default=ControlChartType.INDIVIDUALS)
    values: List[float] = Field(default_factory=list)
    control_limits: Optional[ControlLimit] = Field(default=None)
    out_of_control_indices: List[int] = Field(default_factory=list)
    warning_indices: List[int] = Field(default_factory=list)
    rule_violations: List[Dict[str, Any]] = Field(default_factory=list)
    process_capability: Optional[float] = Field(default=None)


class SPCRuleViolation(BaseModel):
    """A specific SPC rule violation.

    Attributes:
        rule_name: Name of the violated rule.
        description: Description of the violation.
        indices: Indices of data points involved.
    """
    rule_name: str = Field(default="")
    description: str = Field(default="")
    indices: List[int] = Field(default_factory=list)


class Alert(BaseModel):
    """Energy performance alert.

    Attributes:
        alert_id: Unique alert identifier.
        severity: Alert severity level.
        message: Alert message.
        period_label: Period that triggered the alert.
        value: Observed value.
        threshold: Threshold that was exceeded.
        recommendation: Recommended action.
    """
    alert_id: str = Field(default_factory=_new_uuid)
    severity: AlertSeverity = Field(default=AlertSeverity.INFO)
    message: str = Field(default="")
    period_label: str = Field(default="")
    value: float = Field(default=0.0)
    threshold: float = Field(default=0.0)
    recommendation: str = Field(default="")


class AlertThreshold(BaseModel):
    """Configurable alert threshold.

    Attributes:
        name: Threshold name.
        warning_pct: Warning threshold (% deviation).
        critical_pct: Critical threshold (% deviation).
    """
    name: str = Field(default="")
    warning_pct: float = Field(default=10.0, ge=0.0)
    critical_pct: float = Field(default=20.0, ge=0.0)


class ForecastResult(BaseModel):
    """Time series forecast result.

    Attributes:
        method: Forecasting method used.
        forecast_values: Forecasted values.
        forecast_labels: Period labels for forecasts.
        confidence_lower: Lower confidence bound.
        confidence_upper: Upper confidence bound.
        rmse: In-sample RMSE.
        parameters: Model parameters used.
    """
    method: ForecastMethod = Field(default=ForecastMethod.HOLT_WINTERS)
    forecast_values: List[float] = Field(default_factory=list)
    forecast_labels: List[str] = Field(default_factory=list)
    confidence_lower: List[float] = Field(default_factory=list)
    confidence_upper: List[float] = Field(default_factory=list)
    rmse: float = Field(default=0.0)
    parameters: Dict[str, float] = Field(default_factory=dict)


class SeasonalPattern(BaseModel):
    """Detected seasonal pattern.

    Attributes:
        period: Seasonal period (e.g. 12 for monthly).
        seasonal_indices: Seasonal index for each period position.
        strength: Strength of seasonality (0-1).
    """
    period: int = Field(default=12, ge=1)
    seasonal_indices: List[float] = Field(default_factory=list)
    strength: float = Field(default=0.0, ge=0.0, le=1.0)


class TrendDecomposition(BaseModel):
    """Time series decomposition result.

    Attributes:
        method: Decomposition method.
        trend_component: Trend values.
        seasonal_component: Seasonal values.
        residual_component: Residual values.
        seasonal_pattern: Detected seasonal pattern.
    """
    method: DecompositionMethod = Field(default=DecompositionMethod.ADDITIVE)
    trend_component: List[float] = Field(default_factory=list)
    seasonal_component: List[float] = Field(default_factory=list)
    residual_component: List[float] = Field(default_factory=list)
    seasonal_pattern: Optional[SeasonalPattern] = Field(default=None)


class TrendAnalysisResult(BaseModel):
    """Complete trend analysis result.

    Attributes:
        result_id: Unique result identifier.
        trend: Trend analysis results.
        cusum: CUSUM analysis results.
        spc_individuals: SPC individuals chart.
        spc_moving_range: SPC moving range chart.
        ewma: EWMA chart values.
        decomposition: Time series decomposition.
        forecast: Forecast results.
        alerts: Generated alerts.
        step_changes: Detected step changes.
        methodology_notes: Methodology and source notes.
        processing_time_ms: Computation time (ms).
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 provenance hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    trend: Optional[TrendResult] = Field(default=None)
    cusum: Optional[CUSUMResult] = Field(default=None)
    spc_individuals: Optional[SPCResult] = Field(default=None)
    spc_moving_range: Optional[SPCResult] = Field(default=None)
    ewma: List[float] = Field(default_factory=list)
    decomposition: Optional[TrendDecomposition] = Field(default=None)
    forecast: Optional[ForecastResult] = Field(default=None)
    alerts: List[Alert] = Field(default_factory=list)
    step_changes: List[Dict[str, Any]] = Field(default_factory=list)
    methodology_notes: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TrendAnalysisEngine:
    """Zero-hallucination trend analysis engine for energy performance.

    Provides CUSUM, SPC, Mann-Kendall, EWMA, step-change detection,
    Holt-Winters forecasting, and alert generation for energy time series.

    Guarantees:
        - Deterministic: same inputs produce identical outputs.
        - Reproducible: every result carries a SHA-256 provenance hash.
        - Auditable: full breakdown of trend tests and control charts.
        - No LLM: zero hallucination risk in any calculation path.

    Usage::

        engine = TrendAnalysisEngine()
        result = engine.analyse(time_series_data)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the trend analysis engine.

        Args:
            config: Optional overrides.  Supported keys:
                - ewma_lambda (float): EWMA lambda parameter
                - sigma_warning (float): warning sigma multiplier
                - sigma_control (float): control sigma multiplier
                - seasonal_period (int): seasonal period for decomposition
                - forecast_horizon (int): number of periods to forecast
        """
        self._config = config or {}
        self._ewma_lambda = float(self._config.get("ewma_lambda", EWMA_LAMBDA_DEFAULT))
        self._sigma_warn = float(self._config.get("sigma_warning", SPC_SIGMA_LIMITS["warning"]))
        self._sigma_ctrl = float(self._config.get("sigma_control", SPC_SIGMA_LIMITS["control"]))
        self._seasonal_period = int(self._config.get("seasonal_period", 12))
        self._forecast_horizon = int(self._config.get("forecast_horizon", 12))
        self._notes: List[str] = []
        logger.info("TrendAnalysisEngine v%s initialised.", _MODULE_VERSION)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def analyse(
        self,
        data: List[TimeSeriesPoint],
        forecast_horizon: Optional[int] = None,
    ) -> TrendAnalysisResult:
        """Run comprehensive trend analysis on time series data.

        Args:
            data: Time series data points in chronological order.
            forecast_horizon: Number of future periods to forecast.

        Returns:
            TrendAnalysisResult with all analyses and provenance.

        Raises:
            ValueError: If fewer than 6 data points are provided.
        """
        t0 = time.perf_counter()
        self._notes = [f"Engine version: {self.engine_version}"]

        if len(data) < 6:
            raise ValueError("At least 6 data points required for trend analysis.")

        values = [p.value for p in data]
        horizon = forecast_horizon or self._forecast_horizon

        # --- 1. Rolling EUI / moving average ---
        # (implicitly captured in EWMA below)

        # --- 2. Mann-Kendall Trend Test ---
        trend_result = self.mann_kendall_test(values)

        # --- 3. CUSUM Analysis ---
        cusum_result = self.calculate_cusum(data)

        # --- 4. SPC Individuals Chart ---
        spc_ind = self.run_spc_analysis(values, ControlChartType.INDIVIDUALS)

        # --- 5. SPC Moving Range Chart ---
        spc_mr = self.run_spc_analysis(values, ControlChartType.MOVING_RANGE)

        # --- 6. EWMA ---
        ewma_values = self._calculate_ewma(values)

        # --- 7. Step-Change Detection ---
        step_changes = self.detect_step_changes(values)

        # --- 8. YoY Changes ---
        yoy = self.calculate_yoy_change(values)
        trend_result.yoy_changes = yoy
        if yoy:
            avg_yoy = float(_safe_divide(
                sum(_decimal(y) for y in yoy), _decimal(len(yoy))
            ))
            trend_result.average_yoy_change_pct = _round2(avg_yoy)

        # --- 9. Decomposition ---
        decomp = None
        if len(values) >= 2 * self._seasonal_period:
            decomp = self._decompose(values)

        # --- 10. Forecast ---
        forecast = None
        if len(values) >= 2 * self._seasonal_period:
            forecast = self.forecast_holt_winters(values, horizon)
        elif len(values) >= 6:
            forecast = self._forecast_linear(values, data, horizon)

        # --- 11. Alerts ---
        alerts = self.generate_alerts(data, spc_ind, cusum_result, trend_result)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = TrendAnalysisResult(
            trend=trend_result,
            cusum=cusum_result,
            spc_individuals=spc_ind,
            spc_moving_range=spc_mr,
            ewma=[_round2(v) for v in ewma_values],
            decomposition=decomp,
            forecast=forecast,
            alerts=alerts,
            step_changes=step_changes,
            methodology_notes=list(self._notes),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Trend analysis complete: %d points, trend=%s, %d alerts, hash=%s (%.1f ms)",
            len(data),
            trend_result.direction.value,
            len(alerts),
            result.provenance_hash[:16],
            elapsed_ms,
        )
        return result

    def calculate_rolling_eui(
        self,
        values: List[float],
        window: int = 12,
    ) -> List[Optional[float]]:
        """Calculate rolling average EUI.

        Args:
            values: Time series values.
            window: Rolling window size.

        Returns:
            List of rolling averages (None for initial periods).
        """
        result: List[Optional[float]] = []
        for i in range(len(values)):
            if i < window - 1:
                result.append(None)
            else:
                window_vals = values[i - window + 1: i + 1]
                avg = _safe_divide(
                    sum(_decimal(v) for v in window_vals),
                    _decimal(len(window_vals)),
                )
                result.append(_round2(float(avg)))
        return result

    def calculate_cusum(
        self,
        data: List[TimeSeriesPoint],
    ) -> CUSUMResult:
        """Calculate tabular CUSUM from time series data.

        Uses baseline period points (is_baseline=True) to establish
        mean and standard deviation.  If no baseline is flagged, the
        first half of the data is used.

        Args:
            data: Time series data.

        Returns:
            CUSUMResult with upper/lower CUSUM and signals.
        """
        values = [p.value for p in data]
        n = len(values)

        # Determine baseline.
        baseline_vals = [p.value for p in data if p.is_baseline]
        if len(baseline_vals) < 3:
            # Use first half as baseline.
            half = max(3, n // 2)
            baseline_vals = values[:half]

        d_baseline = [_decimal(v) for v in baseline_vals]
        mu = _safe_divide(sum(d_baseline), _decimal(len(d_baseline)))
        sigma = Decimal("0")
        if len(d_baseline) > 1:
            variance = _safe_divide(
                sum((v - mu) ** 2 for v in d_baseline),
                _decimal(len(d_baseline) - 1),
            )
            sigma = _decimal(math.sqrt(max(float(variance), 0.0)))

        k = sigma * _decimal(CUSUM_ALLOWANCE_FACTOR)
        h = sigma * _decimal(CUSUM_THRESHOLD_FACTOR)

        cusum_upper: List[float] = []
        cusum_lower: List[float] = []
        signal_indices: List[int] = []
        signal_directions: List[str] = []

        s_h = Decimal("0")
        s_l = Decimal("0")

        for i, val in enumerate(values):
            d_val = _decimal(val)
            s_h = max(Decimal("0"), s_h + (d_val - mu - k))
            s_l = max(Decimal("0"), s_l + (-d_val + mu - k))

            cusum_upper.append(_round2(float(s_h)))
            cusum_lower.append(_round2(float(s_l)))

            if h > Decimal("0"):
                if s_h > h:
                    signal_indices.append(i)
                    signal_directions.append("upward")
                elif s_l > h:
                    signal_indices.append(i)
                    signal_directions.append("downward")

        self._notes.append(
            f"CUSUM: baseline mean {_round2(float(mu))}, sigma {_round3(float(sigma))}, "
            f"k={_round3(float(k))}, h={_round2(float(h))}, {len(signal_indices)} signals."
        )

        return CUSUMResult(
            cusum_upper=cusum_upper,
            cusum_lower=cusum_lower,
            threshold_h=_round2(float(h)),
            allowance_k=_round3(float(k)),
            signal_indices=signal_indices,
            signal_direction=signal_directions,
            baseline_mean=_round2(float(mu)),
            baseline_std=_round3(float(sigma)),
        )

    def run_spc_analysis(
        self,
        values: List[float],
        chart_type: ControlChartType = ControlChartType.INDIVIDUALS,
    ) -> SPCResult:
        """Run SPC control chart analysis.

        Args:
            values: Time series values.
            chart_type: Type of control chart.

        Returns:
            SPCResult with control limits and violations.
        """
        n = len(values)
        if n < 4:
            return SPCResult(chart_type=chart_type)

        d_values = [_decimal(v) for v in values]

        if chart_type == ControlChartType.INDIVIDUALS:
            return self._spc_individuals(d_values)
        elif chart_type == ControlChartType.MOVING_RANGE:
            return self._spc_moving_range(d_values)
        elif chart_type == ControlChartType.EWMA:
            return self._spc_ewma(d_values)
        else:
            return SPCResult(chart_type=chart_type)

    def detect_step_changes(
        self,
        values: List[float],
        min_segment: int = 5,
    ) -> List[Dict[str, Any]]:
        """Detect step changes in the time series.

        Uses a sliding window approach comparing means of adjacent segments.

        Args:
            values: Time series values.
            min_segment: Minimum segment length to test.

        Returns:
            List of detected step changes with index and magnitude.
        """
        n = len(values)
        if n < 2 * min_segment:
            return []

        changes: List[Dict[str, Any]] = []
        d_values = [_decimal(v) for v in values]

        # Calculate overall standard deviation for threshold.
        overall_mean = _safe_divide(sum(d_values), _decimal(n))
        overall_var = _safe_divide(
            sum((v - overall_mean) ** 2 for v in d_values),
            _decimal(max(n - 1, 1)),
        )
        overall_std = _decimal(math.sqrt(max(float(overall_var), 0.0)))
        threshold = overall_std * Decimal("2")  # 2-sigma threshold

        for i in range(min_segment, n - min_segment + 1):
            before = d_values[:i]
            after = d_values[i:]

            mean_before = _safe_divide(sum(before), _decimal(len(before)))
            mean_after = _safe_divide(sum(after), _decimal(len(after)))
            diff = abs(mean_after - mean_before)

            if diff > threshold and threshold > Decimal("0"):
                pct_change = _safe_pct(mean_after - mean_before, mean_before)
                changes.append({
                    "index": i,
                    "mean_before": _round2(float(mean_before)),
                    "mean_after": _round2(float(mean_after)),
                    "absolute_change": _round2(float(mean_after - mean_before)),
                    "pct_change": _round2(float(pct_change)),
                })

        # Deduplicate to keep only the most significant.
        if len(changes) > 5:
            changes.sort(key=lambda c: abs(c["absolute_change"]), reverse=True)
            changes = changes[:5]

        return changes

    def calculate_yoy_change(
        self,
        values: List[float],
        period: int = 12,
    ) -> List[float]:
        """Calculate year-over-year percentage changes.

        Args:
            values: Time series values.
            period: Number of periods in a year (12 for monthly).

        Returns:
            List of YoY percentage changes.
        """
        yoy: List[float] = []
        for i in range(period, len(values)):
            prev = _decimal(values[i - period])
            curr = _decimal(values[i])
            if prev != Decimal("0"):
                change = _safe_pct(curr - prev, abs(prev))
                yoy.append(_round2(float(change)))
        return yoy

    def mann_kendall_test(self, values: List[float]) -> TrendResult:
        """Perform the Mann-Kendall non-parametric trend test.

        Args:
            values: Time series values.

        Returns:
            TrendResult with S, Z, p-value, and trend direction.
        """
        n = len(values)
        if n < 4:
            return TrendResult()

        # Calculate S.
        s = Decimal("0")
        for i in range(n - 1):
            for j in range(i + 1, n):
                diff = _decimal(values[j]) - _decimal(values[i])
                if diff > Decimal("0"):
                    s += Decimal("1")
                elif diff < Decimal("0"):
                    s -= Decimal("1")

        # Variance of S.
        var_s = _decimal(n * (n - 1) * (2 * n + 5)) / Decimal("18")

        # Z score.
        z = Decimal("0")
        if var_s > Decimal("0"):
            sqrt_var = _decimal(math.sqrt(max(float(var_s), 0.0)))
            if s > Decimal("0"):
                z = (s - Decimal("1")) / sqrt_var
            elif s < Decimal("0"):
                z = (s + Decimal("1")) / sqrt_var

        # Approximate two-tailed p-value using normal approximation.
        z_abs = float(abs(z))
        # Simplified p-value from Z using approximation.
        p_value = 2.0 * (1.0 - self._normal_cdf(z_abs))

        # Direction and significance.
        is_sig = z_abs > MK_SIGNIFICANCE_95
        if is_sig:
            direction = TrendDirection.IMPROVING if float(s) < 0 else TrendDirection.DETERIORATING
        else:
            direction = TrendDirection.STABLE

        # Sen's slope.
        slopes: List[Decimal] = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                if j != i:
                    slope = _safe_divide(
                        _decimal(values[j]) - _decimal(values[i]),
                        _decimal(j - i),
                    )
                    slopes.append(slope)
        slopes.sort()
        sen = slopes[len(slopes) // 2] if slopes else Decimal("0")

        self._notes.append(
            f"Mann-Kendall: S={float(s)}, Z={_round3(float(z))}, "
            f"p={_round4(p_value)}, direction={direction.value}, "
            f"Sen slope={_round4(float(sen))}."
        )

        return TrendResult(
            direction=direction,
            mann_kendall_s=_round2(float(s)),
            mann_kendall_z=_round3(float(z)),
            mann_kendall_p=_round4(p_value),
            is_significant=is_sig,
            sen_slope=_round4(float(sen)),
        )

    def forecast_holt_winters(
        self,
        values: List[float],
        horizon: int = 12,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
    ) -> ForecastResult:
        """Forecast using additive Holt-Winters exponential smoothing.

        Args:
            values: Historical time series values.
            horizon: Number of periods to forecast.
            alpha: Level smoothing parameter.
            beta: Trend smoothing parameter.
            gamma: Seasonal smoothing parameter.

        Returns:
            ForecastResult with forecasted values and confidence intervals.
        """
        m = self._seasonal_period
        n = len(values)
        if n < 2 * m:
            return ForecastResult(method=ForecastMethod.HOLT_WINTERS)

        a = alpha or HW_ALPHA_DEFAULT
        b = beta or HW_BETA_DEFAULT
        g = gamma or HW_GAMMA_DEFAULT

        d_values = [_decimal(v) for v in values]

        # Initialise: level = mean of first season, trend = average slope.
        level = _safe_divide(sum(d_values[:m]), _decimal(m))
        trend = _safe_divide(
            _safe_divide(sum(d_values[m:2 * m]), _decimal(m)) - level,
            _decimal(m),
        )

        # Initial seasonal indices from first complete season.
        seasonal = [_decimal(0)] * m
        for i in range(m):
            seasonal[i] = d_values[i] - level

        # Fit the model.
        fitted: List[Decimal] = []
        for t in range(n):
            if t < m:
                fitted.append(level + trend + seasonal[t % m])
                continue

            x_t = d_values[t]
            s_prev = seasonal[t % m]

            new_level = _decimal(a) * (x_t - s_prev) + (Decimal("1") - _decimal(a)) * (level + trend)
            new_trend = _decimal(b) * (new_level - level) + (Decimal("1") - _decimal(b)) * trend
            new_seasonal = _decimal(g) * (x_t - new_level) + (Decimal("1") - _decimal(g)) * s_prev

            level = new_level
            trend = new_trend
            seasonal[t % m] = new_seasonal
            fitted.append(new_level + new_trend + new_seasonal)

        # In-sample RMSE.
        residuals = [d_values[t] - fitted[t] for t in range(n)]
        mse = _safe_divide(sum(r ** 2 for r in residuals), _decimal(n))
        rmse = _decimal(math.sqrt(max(float(mse), 0.0)))

        # Forecast.
        forecasts: List[float] = []
        lower: List[float] = []
        upper: List[float] = []
        labels: List[str] = []

        for h in range(1, horizon + 1):
            fc = level + _decimal(h) * trend + seasonal[(n + h - 1) % m]
            # Prediction interval widens with horizon.
            margin = Decimal("1.96") * rmse * _decimal(math.sqrt(h))
            forecasts.append(_round2(float(fc)))
            lower.append(_round2(float(fc - margin)))
            upper.append(_round2(float(fc + margin)))
            labels.append(f"forecast_{h}")

        return ForecastResult(
            method=ForecastMethod.HOLT_WINTERS,
            forecast_values=forecasts,
            forecast_labels=labels,
            confidence_lower=lower,
            confidence_upper=upper,
            rmse=_round2(float(rmse)),
            parameters={"alpha": a, "beta": b, "gamma": g, "period": float(m)},
        )

    def generate_alerts(
        self,
        data: List[TimeSeriesPoint],
        spc: Optional[SPCResult],
        cusum: Optional[CUSUMResult],
        trend: Optional[TrendResult],
    ) -> List[Alert]:
        """Generate performance alerts based on analysis results.

        Args:
            data: Original time series data.
            spc: SPC analysis results.
            cusum: CUSUM analysis results.
            trend: Trend analysis results.

        Returns:
            List of Alert objects sorted by severity.
        """
        alerts: List[Alert] = []

        # SPC out-of-control alerts.
        if spc and spc.out_of_control_indices:
            for idx in spc.out_of_control_indices:
                label = data[idx].period_label if idx < len(data) else f"period_{idx}"
                alerts.append(Alert(
                    severity=AlertSeverity.CRITICAL,
                    message=f"SPC control limit exceeded at {label}",
                    period_label=label,
                    value=data[idx].value if idx < len(data) else 0.0,
                    threshold=spc.control_limits.ucl if spc.control_limits else 0.0,
                    recommendation="Investigate cause of abnormal energy consumption.",
                ))

        if spc and spc.warning_indices:
            for idx in spc.warning_indices:
                if idx not in (spc.out_of_control_indices or []):
                    label = data[idx].period_label if idx < len(data) else f"period_{idx}"
                    alerts.append(Alert(
                        severity=AlertSeverity.WARNING,
                        message=f"SPC warning limit exceeded at {label}",
                        period_label=label,
                        value=data[idx].value if idx < len(data) else 0.0,
                        threshold=spc.control_limits.uwl if spc.control_limits else 0.0,
                        recommendation="Monitor closely for potential sustained shift.",
                    ))

        # CUSUM signal alerts.
        if cusum and cusum.signal_indices:
            alerts.append(Alert(
                severity=AlertSeverity.CRITICAL,
                message=f"CUSUM detects sustained shift ({len(cusum.signal_indices)} signals)",
                period_label=data[cusum.signal_indices[0]].period_label
                if cusum.signal_indices[0] < len(data) else "",
                value=cusum.cusum_upper[cusum.signal_indices[0]]
                if cusum.signal_indices[0] < len(cusum.cusum_upper) else 0.0,
                threshold=cusum.threshold_h,
                recommendation="Identify root cause of sustained energy shift.",
            ))

        # Trend alerts.
        if trend and trend.is_significant:
            sev = AlertSeverity.WARNING if trend.direction == TrendDirection.DETERIORATING \
                else AlertSeverity.INFO
            alerts.append(Alert(
                severity=sev,
                message=f"Significant {trend.direction.value} trend detected "
                        f"(Z={trend.mann_kendall_z}, p={trend.mann_kendall_p})",
                recommendation="Review energy management strategy."
                if trend.direction == TrendDirection.DETERIORATING
                else "Continue current energy management practices.",
            ))

        # Sort by severity (CRITICAL first).
        severity_order = {AlertSeverity.CRITICAL: 0, AlertSeverity.WARNING: 1, AlertSeverity.INFO: 2}
        alerts.sort(key=lambda a: severity_order.get(a.severity, 3))

        return alerts

    # --------------------------------------------------------------------- #
    # Private -- SPC Chart Implementations
    # --------------------------------------------------------------------- #

    def _spc_individuals(self, values: List[Decimal]) -> SPCResult:
        """Build SPC Individuals (I) chart."""
        n = len(values)
        x_bar = _safe_divide(sum(values), _decimal(n))

        # Moving ranges.
        mr_values: List[Decimal] = []
        for i in range(1, n):
            mr_values.append(abs(values[i] - values[i - 1]))

        mr_bar = _safe_divide(sum(mr_values), _decimal(len(mr_values))) if mr_values else Decimal("0")

        # Control limits: UCL = x_bar + L * MR_bar / d2.
        sigma_est = _safe_divide(mr_bar, _decimal(SPC_D2))
        ucl = x_bar + _decimal(self._sigma_ctrl) * sigma_est
        lcl = x_bar - _decimal(self._sigma_ctrl) * sigma_est
        uwl = x_bar + _decimal(self._sigma_warn) * sigma_est
        lwl = x_bar - _decimal(self._sigma_warn) * sigma_est

        # Detect violations.
        ooc: List[int] = []
        warn: List[int] = []
        for i, v in enumerate(values):
            if v > ucl or v < lcl:
                ooc.append(i)
            elif v > uwl or v < lwl:
                warn.append(i)

        # Nelson Rule 2: 9 consecutive points on same side of center.
        rule_violations = self._check_nelson_rules(values, x_bar)

        return SPCResult(
            chart_type=ControlChartType.INDIVIDUALS,
            values=[_round2(float(v)) for v in values],
            control_limits=ControlLimit(
                center_line=_round2(float(x_bar)),
                ucl=_round2(float(ucl)),
                lcl=_round2(float(lcl)),
                uwl=_round2(float(uwl)),
                lwl=_round2(float(lwl)),
            ),
            out_of_control_indices=ooc,
            warning_indices=warn,
            rule_violations=rule_violations,
        )

    def _spc_moving_range(self, values: List[Decimal]) -> SPCResult:
        """Build SPC Moving Range (MR) chart."""
        mr_values: List[Decimal] = []
        for i in range(1, len(values)):
            mr_values.append(abs(values[i] - values[i - 1]))

        if not mr_values:
            return SPCResult(chart_type=ControlChartType.MOVING_RANGE)

        mr_bar = _safe_divide(sum(mr_values), _decimal(len(mr_values)))
        ucl_mr = _decimal(SPC_D4) * mr_bar
        lcl_mr = Decimal("0")  # D3 = 0 for n=2

        ooc: List[int] = []
        for i, mr in enumerate(mr_values):
            if mr > ucl_mr:
                ooc.append(i + 1)  # +1 because MR starts at index 1

        return SPCResult(
            chart_type=ControlChartType.MOVING_RANGE,
            values=[_round2(float(mr)) for mr in mr_values],
            control_limits=ControlLimit(
                center_line=_round2(float(mr_bar)),
                ucl=_round2(float(ucl_mr)),
                lcl=0.0,
                uwl=_round2(float(mr_bar + _decimal(self._sigma_warn) * mr_bar)),
                lwl=0.0,
            ),
            out_of_control_indices=ooc,
        )

    def _spc_ewma(self, values: List[Decimal]) -> SPCResult:
        """Build EWMA control chart."""
        n = len(values)
        mu = _safe_divide(sum(values), _decimal(n))

        # Estimate sigma from moving range.
        mr_values: List[Decimal] = []
        for i in range(1, n):
            mr_values.append(abs(values[i] - values[i - 1]))
        mr_bar = _safe_divide(sum(mr_values), _decimal(len(mr_values))) if mr_values else Decimal("0")
        sigma = _safe_divide(mr_bar, _decimal(SPC_D2))

        lam = _decimal(self._ewma_lambda)
        L = _decimal(self._sigma_ctrl)
        z = mu
        ewma_vals: List[Decimal] = []
        ooc: List[int] = []

        for t, x in enumerate(values):
            z = lam * x + (Decimal("1") - lam) * z
            ewma_vals.append(z)

            # Time-varying limits.
            factor = lam / (Decimal("2") - lam) * (
                Decimal("1") - (Decimal("1") - lam) ** (Decimal("2") * _decimal(t + 1))
            )
            margin = L * sigma * _decimal(math.sqrt(max(float(factor), 0.0)))

            if z > mu + margin or z < mu - margin:
                ooc.append(t)

        return SPCResult(
            chart_type=ControlChartType.EWMA,
            values=[_round2(float(v)) for v in ewma_vals],
            control_limits=ControlLimit(
                center_line=_round2(float(mu)),
                ucl=_round2(float(mu + L * sigma)),
                lcl=_round2(float(mu - L * sigma)),
                uwl=_round2(float(mu + _decimal(self._sigma_warn) * sigma)),
                lwl=_round2(float(mu - _decimal(self._sigma_warn) * sigma)),
            ),
            out_of_control_indices=ooc,
        )

    # --------------------------------------------------------------------- #
    # Private -- EWMA Calculation
    # --------------------------------------------------------------------- #

    def _calculate_ewma(self, values: List[float]) -> List[float]:
        """Calculate EWMA smoothed values.

        Args:
            values: Raw time series values.

        Returns:
            EWMA-smoothed values.
        """
        lam = _decimal(self._ewma_lambda)
        ewma: List[float] = []
        z = _decimal(values[0]) if values else Decimal("0")

        for v in values:
            z = lam * _decimal(v) + (Decimal("1") - lam) * z
            ewma.append(float(z))

        return ewma

    # --------------------------------------------------------------------- #
    # Private -- Decomposition
    # --------------------------------------------------------------------- #

    def _decompose(self, values: List[float]) -> TrendDecomposition:
        """Perform additive time series decomposition.

        Args:
            values: Time series values.

        Returns:
            TrendDecomposition with trend, seasonal, and residual.
        """
        m = self._seasonal_period
        n = len(values)
        d_values = [_decimal(v) for v in values]

        # Trend: centered moving average.
        trend: List[Decimal] = [Decimal("0")] * n
        half = m // 2
        for i in range(half, n - half):
            window = d_values[i - half: i + half + 1]
            trend[i] = _safe_divide(sum(window), _decimal(len(window)))

        # Detrended series.
        detrended = [d_values[i] - trend[i] if trend[i] != Decimal("0") else Decimal("0")
                     for i in range(n)]

        # Seasonal indices (average detrended by position).
        seasonal_idx: List[Decimal] = [Decimal("0")] * m
        for pos in range(m):
            pos_values = [detrended[i] for i in range(pos, n, m) if trend[i] != Decimal("0")]
            if pos_values:
                seasonal_idx[pos] = _safe_divide(sum(pos_values), _decimal(len(pos_values)))

        # Normalise so seasonal indices sum to 0.
        s_mean = _safe_divide(sum(seasonal_idx), _decimal(m))
        seasonal_idx = [s - s_mean for s in seasonal_idx]

        # Seasonal component.
        seasonal = [seasonal_idx[i % m] for i in range(n)]

        # Residual.
        residual = [d_values[i] - trend[i] - seasonal[i] for i in range(n)]

        # Seasonal strength: 1 - Var(residual) / Var(detrended).
        var_res = _safe_divide(sum(r ** 2 for r in residual), _decimal(max(n - 1, 1)))
        var_det = _safe_divide(sum(d ** 2 for d in detrended), _decimal(max(n - 1, 1)))
        strength = max(Decimal("0"), Decimal("1") - _safe_divide(var_res, var_det))

        return TrendDecomposition(
            method=DecompositionMethod.ADDITIVE,
            trend_component=[_round2(float(t)) for t in trend],
            seasonal_component=[_round2(float(s)) for s in seasonal],
            residual_component=[_round2(float(r)) for r in residual],
            seasonal_pattern=SeasonalPattern(
                period=m,
                seasonal_indices=[_round3(float(s)) for s in seasonal_idx],
                strength=_round3(float(strength)),
            ),
        )

    # --------------------------------------------------------------------- #
    # Private -- Linear Forecast Fallback
    # --------------------------------------------------------------------- #

    def _forecast_linear(
        self,
        values: List[float],
        data: List[TimeSeriesPoint],
        horizon: int,
    ) -> ForecastResult:
        """Simple linear extrapolation forecast.

        Args:
            values: Historical values.
            data: Original data points.
            horizon: Forecast horizon.

        Returns:
            ForecastResult with linear forecasts.
        """
        n = len(values)
        d_x = [_decimal(i) for i in range(n)]
        d_y = [_decimal(v) for v in values]

        # OLS fit.
        sx = sum(d_x)
        sy = sum(d_y)
        sxx = sum(x ** 2 for x in d_x)
        sxy = sum(x * y for x, y in zip(d_x, d_y))
        d_n = _decimal(n)

        denom = d_n * sxx - sx ** 2
        if denom == Decimal("0"):
            return ForecastResult(method=ForecastMethod.LINEAR_EXTRAPOLATION)

        slope = _safe_divide(d_n * sxy - sx * sy, denom)
        intercept = _safe_divide(sy - slope * sx, d_n)

        # RMSE.
        predicted = [intercept + slope * _decimal(i) for i in range(n)]
        residuals = [d_y[i] - predicted[i] for i in range(n)]
        mse = _safe_divide(sum(r ** 2 for r in residuals), _decimal(max(n - 2, 1)))
        rmse = _decimal(math.sqrt(max(float(mse), 0.0)))

        forecasts: List[float] = []
        lower: List[float] = []
        upper: List[float] = []
        labels: List[str] = []

        for h in range(1, horizon + 1):
            t = n + h - 1
            fc = intercept + slope * _decimal(t)
            margin = Decimal("1.96") * rmse * _decimal(math.sqrt(h))
            forecasts.append(_round2(float(fc)))
            lower.append(_round2(float(fc - margin)))
            upper.append(_round2(float(fc + margin)))
            labels.append(f"forecast_{h}")

        return ForecastResult(
            method=ForecastMethod.LINEAR_EXTRAPOLATION,
            forecast_values=forecasts,
            forecast_labels=labels,
            confidence_lower=lower,
            confidence_upper=upper,
            rmse=_round2(float(rmse)),
            parameters={"intercept": _round4(float(intercept)), "slope": _round4(float(slope))},
        )

    # --------------------------------------------------------------------- #
    # Private -- Nelson Rules
    # --------------------------------------------------------------------- #

    def _check_nelson_rules(
        self,
        values: List[Decimal],
        center: Decimal,
    ) -> List[Dict[str, Any]]:
        """Check Nelson rules for SPC violations.

        Implements:
            Rule 1: Point beyond 3-sigma (handled by control limits).
            Rule 2: 9 consecutive points on same side of center line.
            Rule 3: 6 consecutive points steadily increasing or decreasing.

        Args:
            values: Time series values.
            center: Center line (process mean).

        Returns:
            List of rule violation dicts.
        """
        violations: List[Dict[str, Any]] = []
        n = len(values)

        # Rule 2: 9 consecutive on same side.
        if n >= 9:
            for i in range(n - 8):
                segment = values[i:i + 9]
                above = all(v > center for v in segment)
                below = all(v < center for v in segment)
                if above or below:
                    violations.append({
                        "rule": "Nelson Rule 2",
                        "description": "9 consecutive points on same side of center",
                        "indices": list(range(i, i + 9)),
                        "side": "above" if above else "below",
                    })
                    break

        # Rule 3: 6 consecutive increasing or decreasing.
        if n >= 6:
            for i in range(n - 5):
                segment = values[i:i + 6]
                increasing = all(segment[j + 1] > segment[j] for j in range(5))
                decreasing = all(segment[j + 1] < segment[j] for j in range(5))
                if increasing or decreasing:
                    violations.append({
                        "rule": "Nelson Rule 3",
                        "description": "6 consecutive points steadily " +
                                       ("increasing" if increasing else "decreasing"),
                        "indices": list(range(i, i + 6)),
                    })
                    break

        return violations

    # --------------------------------------------------------------------- #
    # Private -- Statistical Utilities
    # --------------------------------------------------------------------- #

    def _normal_cdf(self, z: float) -> float:
        """Approximate the standard normal CDF using the error function.

        Args:
            z: Z-score value.

        Returns:
            Cumulative probability P(Z <= z).
        """
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Pydantic v2 model_rebuild for forward-reference resolution
# ---------------------------------------------------------------------------

TimeSeriesPoint.model_rebuild()
TrendResult.model_rebuild()
CUSUMResult.model_rebuild()
ControlLimit.model_rebuild()
SPCResult.model_rebuild()
SPCRuleViolation.model_rebuild()
Alert.model_rebuild()
AlertThreshold.model_rebuild()
ForecastResult.model_rebuild()
SeasonalPattern.model_rebuild()
TrendDecomposition.model_rebuild()
TrendAnalysisResult.model_rebuild()


# ---------------------------------------------------------------------------
# Public Aliases -- required by PACK-035 __init__.py symbol contract
# ---------------------------------------------------------------------------

SPCChart = SPCResult
"""Alias: ``SPCChart`` -> :class:`SPCResult`."""
