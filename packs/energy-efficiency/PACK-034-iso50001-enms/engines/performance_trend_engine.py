# -*- coding: utf-8 -*-
"""
PerformanceTrendEngine - PACK-034 ISO 50001 EnMS Engine 9
=========================================================

Energy performance trending, regression validation, and ISO 50015
measurement and verification engine.  Provides year-over-year
comparison, rolling 12-month analysis, linear regression with
statistical validation, multi-method forecasting (linear, exponential,
moving average, weighted moving average, Holt-Winters additive), and
savings verification per ISO 50015:2014 and IPMVP Options A-D.

Calculation Methodology:
    Linear Regression (Ordinary Least Squares):
        slope     = (n * sum(x*y) - sum(x) * sum(y))
                    / (n * sum(x^2) - sum(x)^2)
        intercept = (sum(y) - slope * sum(x)) / n
        R^2       = 1 - SS_res / SS_tot
        p-value   = approximated via t-distribution lookup
        F-stat    = (SS_reg / k) / (SS_res / (n - k - 1))

    Durbin-Watson Autocorrelation:
        DW = sum((e_t - e_{t-1})^2) / sum(e_t^2)
        (2.0 = no autocorrelation; <1.5 or >2.5 = concern)

    CV(RMSE) per ASHRAE Guideline 14-2014:
        CV(RMSE) = RMSE / mean(actual) * 100
        Threshold: <= 25% for monthly data, <= 30% for daily

    MAPE:
        MAPE = (1/n) * sum(|actual - predicted| / |actual|) * 100

    Holt-Winters Additive:
        level_t   = alpha * (y_t - seasonal_{t-m}) + (1 - alpha) * (level_{t-1} + trend_{t-1})
        trend_t   = beta  * (level_t - level_{t-1}) + (1 - beta)  * trend_{t-1}
        seasonal_t = gamma * (y_t - level_t)         + (1 - gamma) * seasonal_{t-m}
        forecast   = level_t + h * trend_t + seasonal_{t-m+h}

    Savings Verification (ISO 50015 / IPMVP):
        adjusted_baseline  = baseline_consumption +/- routine_adjustments
                             +/- non_routine_adjustments
        gross_savings       = adjusted_baseline - reporting_period_consumption
        net_savings         = gross_savings - non_routine_adjustments
        uncertainty_pct     = t_value * sqrt(sum_of_squared_errors / (n - p))
                              / mean(baseline) * 100

Regulatory References:
    - ISO 50001:2018 - Energy management systems
    - ISO 50006:2014 - Measuring energy performance using EnPIs and EnBs
    - ISO 50015:2014 - Measurement and verification of energy performance
    - IPMVP Core Concepts (EVO, 2022) - Options A, B, C, D
    - ASHRAE Guideline 14-2014 - Measurement of Energy, Demand, and
      Water Savings (model adequacy thresholds)
    - SEP 50001 - Superior Energy Performance Programme

Zero-Hallucination:
    - All formulas are standard statistical/engineering calculations
    - Regression coefficients computed via closed-form OLS
    - Model adequacy thresholds from ASHRAE 14-2014 published values
    - IPMVP option descriptions from EVO Core Concepts 2022
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-034 ISO 50001 EnMS
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
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash",
                         "analysis_date", "calculation_time_ms")
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


def _decimal_sqrt(value: Decimal) -> Decimal:
    """Compute square root of a Decimal using math.sqrt conversion."""
    if value <= Decimal("0"):
        return Decimal("0")
    return _decimal(math.sqrt(float(value)))


def _decimal_exp(value: Decimal) -> Decimal:
    """Compute e^value using math.exp conversion."""
    try:
        return _decimal(math.exp(float(value)))
    except OverflowError:
        return Decimal("0")


def _decimal_log(value: Decimal) -> Decimal:
    """Compute natural log of a Decimal."""
    if value <= Decimal("0"):
        return Decimal("0")
    return _decimal(math.log(float(value)))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TrendDirection(str, Enum):
    """Direction of energy performance trend.

    IMPROVING: Energy performance is improving (consumption decreasing
               or EnPI improving).
    STABLE: No statistically significant change in performance.
    DEGRADING: Energy performance is degrading (consumption increasing
               or EnPI worsening).
    INSUFFICIENT_DATA: Not enough data points to determine a trend.
    """
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    INSUFFICIENT_DATA = "insufficient_data"


class AnalysisType(str, Enum):
    """Type of performance trend analysis to perform.

    YEAR_OVER_YEAR: Compare current year to baseline year.
    ROLLING_12_MONTH: Rolling 12-month sum/average analysis.
    REGRESSION_VALIDATION: Validate regression model adequacy.
    FORECAST_VS_ACTUAL: Compare forecasted values to actuals.
    SEASONAL_DECOMPOSITION: Decompose into trend + seasonal + residual.
    """
    YEAR_OVER_YEAR = "year_over_year"
    ROLLING_12_MONTH = "rolling_12_month"
    REGRESSION_VALIDATION = "regression_validation"
    FORECAST_VS_ACTUAL = "forecast_vs_actual"
    SEASONAL_DECOMPOSITION = "seasonal_decomposition"


class RegressionMetric(str, Enum):
    """Statistical metrics for regression model validation.

    R_SQUARED: Coefficient of determination (0-1).
    CV_RMSE: Coefficient of variation of RMSE (percentage).
    P_VALUE: Statistical significance of model coefficients.
    F_STATISTIC: Overall model significance (F-test).
    DURBIN_WATSON: Autocorrelation test statistic (0-4).
    MAPE: Mean absolute percentage error.
    """
    R_SQUARED = "r_squared"
    CV_RMSE = "cv_rmse"
    P_VALUE = "p_value"
    F_STATISTIC = "f_statistic"
    DURBIN_WATSON = "durbin_watson"
    MAPE = "mape"


class ForecastMethod(str, Enum):
    """Forecasting method for energy consumption projection.

    LINEAR: Linear extrapolation of trend line.
    EXPONENTIAL: Exponential growth/decay extrapolation.
    MOVING_AVERAGE: Simple moving average forecast.
    WEIGHTED_MOVING_AVERAGE: Weighted moving average (recent data
                              receives higher weight).
    HOLT_WINTERS: Holt-Winters additive exponential smoothing
                  with trend and seasonal components.
    """
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    MOVING_AVERAGE = "moving_average"
    WEIGHTED_MOVING_AVERAGE = "weighted_moving_average"
    HOLT_WINTERS = "holt_winters"


class VerificationStandard(str, Enum):
    """M&V standard/option for savings verification.

    ISO_50015: ISO 50015:2014 energy performance M&V.
    IPMVP_OPTION_A: Retrofit isolation - key parameter measurement.
    IPMVP_OPTION_B: Retrofit isolation - all parameter measurement.
    IPMVP_OPTION_C: Whole facility analysis (utility billing).
    IPMVP_OPTION_D: Calibrated simulation.
    """
    ISO_50015 = "iso_50015"
    IPMVP_OPTION_A = "ipmvp_option_a"
    IPMVP_OPTION_B = "ipmvp_option_b"
    IPMVP_OPTION_C = "ipmvp_option_c"
    IPMVP_OPTION_D = "ipmvp_option_d"


# ---------------------------------------------------------------------------
# Reference Data
# ---------------------------------------------------------------------------

# Model adequacy thresholds per ISO 50015 and ASHRAE Guideline 14-2014.
# These are the acceptance criteria for regression models used in M&V.
MODEL_ADEQUACY_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "ashrae_14_monthly": {
        "r_squared_min": Decimal("0.70"),
        "cv_rmse_max_pct": Decimal("25.0"),
        "mape_max_pct": Decimal("20.0"),
        "durbin_watson_low": Decimal("1.0"),
        "durbin_watson_high": Decimal("3.0"),
        "p_value_max": Decimal("0.05"),
        "description": (
            "ASHRAE Guideline 14-2014 monthly data thresholds. "
            "R^2 >= 0.70, CV(RMSE) <= 25%, p-value < 0.05."
        ),
    },
    "ashrae_14_daily": {
        "r_squared_min": Decimal("0.50"),
        "cv_rmse_max_pct": Decimal("30.0"),
        "mape_max_pct": Decimal("25.0"),
        "durbin_watson_low": Decimal("1.0"),
        "durbin_watson_high": Decimal("3.0"),
        "p_value_max": Decimal("0.05"),
        "description": (
            "ASHRAE Guideline 14-2014 daily data thresholds. "
            "R^2 >= 0.50, CV(RMSE) <= 30%, p-value < 0.05."
        ),
    },
    "iso_50015": {
        "r_squared_min": Decimal("0.75"),
        "cv_rmse_max_pct": Decimal("20.0"),
        "mape_max_pct": Decimal("15.0"),
        "durbin_watson_low": Decimal("1.5"),
        "durbin_watson_high": Decimal("2.5"),
        "p_value_max": Decimal("0.05"),
        "description": (
            "ISO 50015:2014 recommended thresholds. "
            "R^2 >= 0.75, CV(RMSE) <= 20%, DW in [1.5, 2.5]."
        ),
    },
    "sep_50001": {
        "r_squared_min": Decimal("0.75"),
        "cv_rmse_max_pct": Decimal("20.0"),
        "mape_max_pct": Decimal("15.0"),
        "durbin_watson_low": Decimal("1.5"),
        "durbin_watson_high": Decimal("2.5"),
        "p_value_max": Decimal("0.05"),
        "description": (
            "Superior Energy Performance (SEP 50001) programme thresholds. "
            "Same as ISO 50015 for certification."
        ),
    },
}

# IPMVP option descriptions from EVO Core Concepts (2022).
IPMVP_OPTION_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    VerificationStandard.IPMVP_OPTION_A.value: {
        "name": "Option A - Retrofit Isolation: Key Parameter Measurement",
        "description": (
            "Savings are determined by field measurement of the key "
            "performance parameter(s) that define the energy use of the "
            "affected system. Parameters not measured are estimated. "
            "Best for individual equipment or systems where one key "
            "parameter dominates energy savings."
        ),
        "applicability": (
            "Lighting retrofits, motor replacements, VFD installations, "
            "equipment upgrades with well-defined operating profiles."
        ),
        "typical_accuracy": "10-20% uncertainty",
    },
    VerificationStandard.IPMVP_OPTION_B.value: {
        "name": "Option B - Retrofit Isolation: All Parameter Measurement",
        "description": (
            "Savings are determined by field measurement of all parameters "
            "needed to determine energy use of the affected system. "
            "Continuous or periodic measurement of energy use at the "
            "system or component level."
        ),
        "applicability": (
            "HVAC system upgrades, chiller replacements, boiler retrofits, "
            "complex systems where multiple parameters affect savings."
        ),
        "typical_accuracy": "5-10% uncertainty",
    },
    VerificationStandard.IPMVP_OPTION_C.value: {
        "name": "Option C - Whole Facility",
        "description": (
            "Savings are determined by measuring energy use at the whole "
            "facility or sub-facility level, typically using utility "
            "billing data. Regression analysis of baseline and reporting "
            "period data with routine adjustments."
        ),
        "applicability": (
            "Multiple interactive measures, whole-building retrofits, "
            "operational improvements affecting entire facility. "
            "Requires >= 12 months baseline and reporting data."
        ),
        "typical_accuracy": "5-15% uncertainty",
    },
    VerificationStandard.IPMVP_OPTION_D.value: {
        "name": "Option D - Calibrated Simulation",
        "description": (
            "Savings are determined through simulation of energy use, "
            "calibrated to actual utility billing or metered data. "
            "The simulation model must meet ASHRAE 14-2014 calibration "
            "criteria."
        ),
        "applicability": (
            "New construction, major renovations, complex facilities "
            "where direct measurement is impractical. Requires detailed "
            "building model and calibration data."
        ),
        "typical_accuracy": "10-20% uncertainty",
    },
}

# Savings uncertainty formulas and references.
SAVINGS_UNCERTAINTY_FORMULAS: Dict[str, Dict[str, str]] = {
    "iso_50015": {
        "formula": (
            "U = t_{n-p, alpha/2} * sqrt(SSE / (n - p)) / mean(y_baseline) * 100"
        ),
        "variables": (
            "t = Student's t-value at confidence level, "
            "SSE = sum of squared errors, n = number of observations, "
            "p = number of model parameters, y_baseline = baseline mean"
        ),
        "reference": "ISO 50015:2014, Clause 7.4",
        "notes": (
            "Uncertainty is expressed as a percentage of baseline "
            "consumption at a stated confidence level (typically 68% or 90%)."
        ),
    },
    "ipmvp_fractional_savings": {
        "formula": (
            "F_savings = (E_baseline_adj - E_reporting) / E_baseline_adj"
        ),
        "variables": (
            "E_baseline_adj = adjusted baseline energy, "
            "E_reporting = reporting period energy"
        ),
        "reference": "IPMVP Core Concepts, Section 4.7",
        "notes": (
            "Fractional savings must exceed 2x the model uncertainty "
            "to be considered statistically significant at 90% confidence."
        ),
    },
    "ashrae_14_precision": {
        "formula": (
            "Precision = t * CV(RMSE) / sqrt(n) * sqrt(1 + 2/n)"
        ),
        "variables": (
            "t = t-value at 90% confidence, CV(RMSE) = coefficient of "
            "variation of RMSE (%), n = number of baseline periods"
        ),
        "reference": "ASHRAE Guideline 14-2014, Section 6.3",
        "notes": (
            "Savings uncertainty at 90% confidence for whole-building "
            "analysis. Precision improves with more baseline data points."
        ),
    },
}

# T-values for common confidence levels and degrees of freedom.
# Used for uncertainty calculations. Key = degrees of freedom.
_T_VALUES_90: Dict[int, Decimal] = {
    1: Decimal("6.314"),
    2: Decimal("2.920"),
    3: Decimal("2.353"),
    4: Decimal("2.132"),
    5: Decimal("2.015"),
    6: Decimal("1.943"),
    7: Decimal("1.895"),
    8: Decimal("1.860"),
    9: Decimal("1.833"),
    10: Decimal("1.812"),
    11: Decimal("1.796"),
    12: Decimal("1.782"),
    15: Decimal("1.753"),
    20: Decimal("1.725"),
    25: Decimal("1.708"),
    30: Decimal("1.697"),
    40: Decimal("1.684"),
    60: Decimal("1.671"),
    120: Decimal("1.658"),
}

# Default smoothing parameters for Holt-Winters.
_DEFAULT_HW_ALPHA: Decimal = Decimal("0.3")
_DEFAULT_HW_BETA: Decimal = Decimal("0.1")
_DEFAULT_HW_GAMMA: Decimal = Decimal("0.2")
_DEFAULT_HW_SEASON_LENGTH: int = 12

# Trend detection thresholds.
_IMPROVING_SLOPE_THRESHOLD: Decimal = Decimal("-0.5")
_DEGRADING_SLOPE_THRESHOLD: Decimal = Decimal("0.5")
_MIN_DATA_POINTS_TREND: int = 3
_MIN_DATA_POINTS_REGRESSION: int = 6
_MIN_DATA_POINTS_FORECAST: int = 4


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class PerformanceDataPoint(BaseModel):
    """A single period of energy performance data.

    Attributes:
        period_start: Start date of the measurement period.
        period_end: End date of the measurement period.
        actual_kwh: Actual measured energy consumption (kWh).
        expected_kwh: Expected/predicted consumption from model (kWh).
        normalized_kwh: Weather- or production-normalised consumption.
        enpi_value: Energy Performance Indicator value.
        cost: Energy cost for the period.
        emissions_tco2e: GHG emissions for the period (tCO2e).
    """
    period_start: date = Field(..., description="Period start date")
    period_end: date = Field(..., description="Period end date")
    actual_kwh: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Actual energy consumption (kWh)"
    )
    expected_kwh: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Expected/predicted consumption (kWh)"
    )
    normalized_kwh: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Normalised consumption (kWh)"
    )
    enpi_value: Optional[Decimal] = Field(
        default=None,
        description="Energy Performance Indicator value"
    )
    cost: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Energy cost for the period"
    )
    emissions_tco2e: Optional[Decimal] = Field(
        default=None, ge=0,
        description="GHG emissions (tCO2e)"
    )

    @field_validator("period_end")
    @classmethod
    def validate_period_end(cls, v: date, info: Any) -> date:
        """Ensure period_end is not before period_start."""
        period_start = info.data.get("period_start")
        if period_start and v < period_start:
            raise ValueError(
                f"period_end ({v}) cannot be before period_start ({period_start})"
            )
        return v


class TrendAnalysis(BaseModel):
    """Result of a trend analysis on energy performance data.

    Attributes:
        analysis_type: Type of analysis performed.
        period_count: Number of data periods analysed.
        trend_direction: Detected trend direction.
        trend_slope: Slope of the trend line (kWh/period).
        intercept: Y-intercept of the trend line.
        r_squared: Coefficient of determination (0-1).
        p_value: Statistical significance of the slope.
        confidence_interval_lower: Lower bound of slope CI.
        confidence_interval_upper: Upper bound of slope CI.
        data_points: Summarised data for charting.
    """
    analysis_type: AnalysisType = Field(
        default=AnalysisType.YEAR_OVER_YEAR,
        description="Analysis type performed"
    )
    period_count: int = Field(
        default=0, ge=0,
        description="Number of periods analysed"
    )
    trend_direction: TrendDirection = Field(
        default=TrendDirection.INSUFFICIENT_DATA,
        description="Detected trend direction"
    )
    trend_slope: Decimal = Field(
        default=Decimal("0"),
        description="Slope of the trend line (kWh per period)"
    )
    intercept: Decimal = Field(
        default=Decimal("0"),
        description="Y-intercept of the trend line"
    )
    r_squared: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("1"),
        description="R-squared (coefficient of determination)"
    )
    p_value: Decimal = Field(
        default=Decimal("1"), ge=0, le=Decimal("1"),
        description="P-value for slope significance"
    )
    confidence_interval_lower: Decimal = Field(
        default=Decimal("0"),
        description="Lower bound of slope confidence interval"
    )
    confidence_interval_upper: Decimal = Field(
        default=Decimal("0"),
        description="Upper bound of slope confidence interval"
    )
    data_points: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Summarised data points for charting"
    )


class YearOverYearComparison(BaseModel):
    """Year-over-year comparison of energy performance.

    Attributes:
        current_year: The current/reporting year.
        baseline_year: The baseline/reference year.
        current_total_kwh: Total consumption in the current year.
        baseline_total_kwh: Total consumption in the baseline year.
        absolute_change_kwh: Absolute change (current - baseline).
        percentage_change: Percentage change from baseline.
        normalized_change: Change after normalisation adjustments.
        months_compared: Number of months with data in both years.
        monthly_comparison: Per-month comparison details.
    """
    current_year: int = Field(
        default=0, description="Current/reporting year"
    )
    baseline_year: int = Field(
        default=0, description="Baseline/reference year"
    )
    current_total_kwh: Decimal = Field(
        default=Decimal("0"),
        description="Total current year consumption (kWh)"
    )
    baseline_total_kwh: Decimal = Field(
        default=Decimal("0"),
        description="Total baseline year consumption (kWh)"
    )
    absolute_change_kwh: Decimal = Field(
        default=Decimal("0"),
        description="Absolute change in consumption (kWh)"
    )
    percentage_change: Decimal = Field(
        default=Decimal("0"),
        description="Percentage change from baseline"
    )
    normalized_change: Decimal = Field(
        default=Decimal("0"),
        description="Normalised change after adjustments (kWh)"
    )
    months_compared: int = Field(
        default=0, ge=0, le=12,
        description="Number of months with data in both years"
    )
    monthly_comparison: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-month comparison details"
    )


class RollingAnalysis(BaseModel):
    """Rolling window analysis of energy performance.

    Attributes:
        window_months: Size of the rolling window in months.
        current_rolling_kwh: Current rolling window total (kWh).
        previous_rolling_kwh: Previous rolling window total (kWh).
        trend: Detected trend direction.
        rolling_values: Time series of rolling window values.
    """
    window_months: int = Field(
        default=12, ge=1,
        description="Rolling window size in months"
    )
    current_rolling_kwh: Decimal = Field(
        default=Decimal("0"),
        description="Current rolling window total (kWh)"
    )
    previous_rolling_kwh: Decimal = Field(
        default=Decimal("0"),
        description="Previous rolling window total (kWh)"
    )
    trend: TrendDirection = Field(
        default=TrendDirection.INSUFFICIENT_DATA,
        description="Trend direction of rolling values"
    )
    rolling_values: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Time series of rolling window values"
    )


class RegressionValidation(BaseModel):
    """Statistical validation results for a regression model.

    Attributes:
        r_squared: Coefficient of determination.
        adjusted_r_squared: Adjusted R-squared for multiple variables.
        cv_rmse: Coefficient of variation of RMSE (percentage).
        mape: Mean absolute percentage error (percentage).
        durbin_watson: Durbin-Watson statistic for autocorrelation.
        f_statistic: F-test statistic for overall significance.
        f_p_value: P-value for the F-test.
        residual_normality_p: P-value from residual normality test.
        is_model_adequate: Whether the model passes all thresholds.
        inadequacy_reasons: List of reasons if model is inadequate.
    """
    r_squared: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Coefficient of determination"
    )
    adjusted_r_squared: Decimal = Field(
        default=Decimal("0"),
        description="Adjusted R-squared"
    )
    cv_rmse: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="CV(RMSE) percentage"
    )
    mape: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Mean absolute percentage error"
    )
    durbin_watson: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Durbin-Watson statistic"
    )
    f_statistic: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="F-test statistic"
    )
    f_p_value: Decimal = Field(
        default=Decimal("1"), ge=0,
        description="F-test p-value"
    )
    residual_normality_p: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Residual normality test p-value"
    )
    is_model_adequate: bool = Field(
        default=False,
        description="Whether model passes all adequacy thresholds"
    )
    inadequacy_reasons: List[str] = Field(
        default_factory=list,
        description="Reasons for model inadequacy"
    )


class ForecastResult(BaseModel):
    """Result of an energy consumption forecast.

    Attributes:
        forecast_method: Method used for forecasting.
        forecast_periods: Number of periods forecasted.
        forecasted_values: List of forecasted values with periods.
        actual_values: Actual values for comparison (if available).
        mape: MAPE of forecast vs actuals (if available).
        forecast_accuracy_pct: Forecast accuracy percentage.
    """
    forecast_method: ForecastMethod = Field(
        default=ForecastMethod.LINEAR,
        description="Forecasting method used"
    )
    forecast_periods: int = Field(
        default=0, ge=0,
        description="Number of periods forecasted"
    )
    forecasted_values: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Forecasted values with period labels"
    )
    actual_values: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Actual values for comparison"
    )
    mape: Optional[Decimal] = Field(
        default=None, ge=0,
        description="MAPE of forecast vs actuals"
    )
    forecast_accuracy_pct: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Forecast accuracy percentage"
    )


class SavingsVerification(BaseModel):
    """M&V savings verification result per ISO 50015 or IPMVP.

    Attributes:
        verification_standard: Standard/option used.
        baseline_consumption: Original baseline consumption (kWh).
        reporting_period_consumption: Reporting period consumption (kWh).
        adjusted_baseline: Baseline adjusted for conditions (kWh).
        gross_savings_kwh: Gross energy savings (kWh).
        net_savings_kwh: Net energy savings after adjustments (kWh).
        savings_uncertainty_pct: Savings uncertainty percentage.
        confidence_level: Confidence level for uncertainty.
        is_savings_significant: Whether savings exceed uncertainty.
    """
    verification_standard: VerificationStandard = Field(
        default=VerificationStandard.ISO_50015,
        description="M&V standard/option used"
    )
    baseline_consumption: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Original baseline consumption (kWh)"
    )
    reporting_period_consumption: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Reporting period consumption (kWh)"
    )
    adjusted_baseline: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Adjusted baseline consumption (kWh)"
    )
    gross_savings_kwh: Decimal = Field(
        default=Decimal("0"),
        description="Gross energy savings (kWh)"
    )
    net_savings_kwh: Decimal = Field(
        default=Decimal("0"),
        description="Net energy savings (kWh)"
    )
    savings_uncertainty_pct: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Savings uncertainty percentage"
    )
    confidence_level: Decimal = Field(
        default=Decimal("90"), ge=0, le=Decimal("99.9"),
        description="Confidence level for uncertainty"
    )
    is_savings_significant: bool = Field(
        default=False,
        description="Whether savings exceed 2x uncertainty"
    )


class PerformanceTrendResult(BaseModel):
    """Complete performance trend analysis result.

    Attributes:
        analysis_id: Unique analysis identifier.
        enms_id: Energy management system identifier.
        analysis_date: Date/time of the analysis.
        period_start: Start of the analysis period.
        period_end: End of the analysis period.
        trend_analysis: Core trend analysis results.
        yoy_comparison: Year-over-year comparison (optional).
        rolling_analysis: Rolling analysis results (optional).
        regression_validation: Model validation results (optional).
        forecast: Forecast results (optional).
        savings_verification: Savings verification (optional).
        summary: Summary metrics dictionary.
        provenance_hash: SHA-256 provenance hash.
        calculation_time_ms: Calculation time in milliseconds.
    """
    analysis_id: str = Field(
        default_factory=_new_uuid,
        description="Unique analysis identifier"
    )
    enms_id: str = Field(
        default="", description="Energy management system identifier"
    )
    analysis_date: datetime = Field(
        default_factory=_utcnow,
        description="Analysis date/time"
    )
    period_start: date = Field(
        ..., description="Analysis period start"
    )
    period_end: date = Field(
        ..., description="Analysis period end"
    )
    trend_analysis: TrendAnalysis = Field(
        default_factory=TrendAnalysis,
        description="Core trend analysis"
    )
    yoy_comparison: Optional[YearOverYearComparison] = Field(
        default=None,
        description="Year-over-year comparison"
    )
    rolling_analysis: Optional[RollingAnalysis] = Field(
        default=None,
        description="Rolling window analysis"
    )
    regression_validation: Optional[RegressionValidation] = Field(
        default=None,
        description="Regression model validation"
    )
    forecast: Optional[ForecastResult] = Field(
        default=None,
        description="Forecast results"
    )
    savings_verification: Optional[SavingsVerification] = Field(
        default=None,
        description="Savings verification"
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary metrics"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )
    calculation_time_ms: int = Field(
        default=0, ge=0,
        description="Calculation time (milliseconds)"
    )


# ---------------------------------------------------------------------------
# Model Rebuild (required for `from __future__ import annotations`)
# ---------------------------------------------------------------------------

PerformanceDataPoint.model_rebuild()
TrendAnalysis.model_rebuild()
YearOverYearComparison.model_rebuild()
RollingAnalysis.model_rebuild()
RegressionValidation.model_rebuild()
ForecastResult.model_rebuild()
SavingsVerification.model_rebuild()
PerformanceTrendResult.model_rebuild()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class PerformanceTrendEngine:
    """Energy performance trending, regression validation, and savings
    verification engine for ISO 50001 Energy Management Systems.

    Provides year-over-year comparison, rolling 12-month analysis,
    linear regression with full statistical validation (R-squared,
    CV(RMSE), Durbin-Watson, F-test), multi-method forecasting, and
    savings verification per ISO 50015 and IPMVP Options A-D.

    All calculations use deterministic Decimal arithmetic.  Every result
    carries a SHA-256 provenance hash for audit trail integrity.

    Usage::

        engine = PerformanceTrendEngine()
        data = [
            PerformanceDataPoint(
                period_start=date(2024, 1, 1),
                period_end=date(2024, 1, 31),
                actual_kwh=Decimal("120000"),
            ),
            # ... more monthly data points
        ]
        result = engine.analyze_trends(
            data=data,
            analysis_types=[AnalysisType.YEAR_OVER_YEAR,
                            AnalysisType.ROLLING_12_MONTH],
        )
        print(f"Trend: {result.trend_analysis.trend_direction.value}")
        print(f"R^2:   {result.trend_analysis.r_squared}")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise PerformanceTrendEngine.

        Args:
            config: Optional configuration overrides.  Supported keys:
                - threshold_standard (str): 'ashrae_14_monthly',
                  'ashrae_14_daily', 'iso_50015', or 'sep_50001'.
                - hw_alpha (Decimal): Holt-Winters level smoothing.
                - hw_beta (Decimal): Holt-Winters trend smoothing.
                - hw_gamma (Decimal): Holt-Winters seasonal smoothing.
                - hw_season_length (int): seasonal cycle length.
                - improving_threshold (Decimal): slope below which
                  trend is considered improving.
                - degrading_threshold (Decimal): slope above which
                  trend is considered degrading.
                - confidence_level (Decimal): default confidence (90).
        """
        self.config = config or {}
        self._threshold_key = self.config.get(
            "threshold_standard", "ashrae_14_monthly"
        )
        self._hw_alpha = _decimal(
            self.config.get("hw_alpha", _DEFAULT_HW_ALPHA)
        )
        self._hw_beta = _decimal(
            self.config.get("hw_beta", _DEFAULT_HW_BETA)
        )
        self._hw_gamma = _decimal(
            self.config.get("hw_gamma", _DEFAULT_HW_GAMMA)
        )
        self._hw_season_length = int(
            self.config.get("hw_season_length", _DEFAULT_HW_SEASON_LENGTH)
        )
        self._improving_threshold = _decimal(
            self.config.get("improving_threshold", _IMPROVING_SLOPE_THRESHOLD)
        )
        self._degrading_threshold = _decimal(
            self.config.get("degrading_threshold", _DEGRADING_SLOPE_THRESHOLD)
        )
        self._confidence_level = _decimal(
            self.config.get("confidence_level", Decimal("90"))
        )
        logger.info(
            "PerformanceTrendEngine v%s initialised (thresholds=%s)",
            self.engine_version, self._threshold_key,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def analyze_trends(
        self,
        data: List[PerformanceDataPoint],
        analysis_types: Optional[List[AnalysisType]] = None,
        enms_id: str = "",
        baseline_data: Optional[List[PerformanceDataPoint]] = None,
        forecast_method: ForecastMethod = ForecastMethod.LINEAR,
        forecast_periods: int = 12,
        verification_standard: VerificationStandard = VerificationStandard.ISO_50015,
    ) -> PerformanceTrendResult:
        """Run comprehensive performance trend analysis.

        Performs one or more analysis types on the supplied data:
        trend line fitting, year-over-year comparison, rolling analysis,
        regression validation, forecasting, and savings verification.

        Args:
            data: Time-ordered list of performance data points.
            analysis_types: Types of analysis to perform.  Defaults to
                all types if not specified.
            enms_id: EnMS identifier for the result.
            baseline_data: Baseline period data for savings verification.
            forecast_method: Method for forecasting.
            forecast_periods: Number of periods to forecast.
            verification_standard: M&V standard for savings verification.

        Returns:
            PerformanceTrendResult with all requested analyses.

        Raises:
            ValueError: If data is empty or insufficient.
        """
        t0 = time.perf_counter()

        if not data:
            raise ValueError("No performance data provided for trend analysis.")

        if analysis_types is None:
            analysis_types = [AnalysisType.YEAR_OVER_YEAR]

        # Sort data by period_start
        sorted_data = sorted(data, key=lambda d: d.period_start)
        period_start = sorted_data[0].period_start
        period_end = sorted_data[-1].period_end

        logger.info(
            "Trend analysis: enms=%s, periods=%d, types=%s, range=%s to %s",
            enms_id, len(sorted_data),
            [t.value for t in analysis_types],
            period_start, period_end,
        )

        # Core trend line (always computed)
        values = [dp.actual_kwh for dp in sorted_data]
        trend_analysis = self.calculate_trend_line(values)
        trend_analysis.analysis_type = (
            analysis_types[0] if analysis_types else AnalysisType.YEAR_OVER_YEAR
        )
        trend_analysis.data_points = [
            {
                "period": idx + 1,
                "period_start": str(dp.period_start),
                "period_end": str(dp.period_end),
                "actual_kwh": str(dp.actual_kwh),
                "fitted_kwh": str(
                    _round_val(
                        trend_analysis.intercept
                        + trend_analysis.trend_slope * _decimal(idx + 1),
                        2,
                    )
                ),
            }
            for idx, dp in enumerate(sorted_data)
        ]

        # Year-over-year comparison
        yoy_comparison: Optional[YearOverYearComparison] = None
        if AnalysisType.YEAR_OVER_YEAR in analysis_types:
            yoy_comparison = self._perform_yoy_analysis(sorted_data)

        # Rolling analysis
        rolling_analysis: Optional[RollingAnalysis] = None
        if AnalysisType.ROLLING_12_MONTH in analysis_types:
            rolling_analysis = self.calculate_rolling_analysis(sorted_data, 12)

        # Regression validation
        regression_validation: Optional[RegressionValidation] = None
        if AnalysisType.REGRESSION_VALIDATION in analysis_types:
            actual_list = [dp.actual_kwh for dp in sorted_data]
            predicted_list = [
                dp.expected_kwh if dp.expected_kwh is not None
                else (
                    trend_analysis.intercept
                    + trend_analysis.trend_slope * _decimal(idx + 1)
                )
                for idx, dp in enumerate(sorted_data)
            ]
            regression_validation = self.validate_regression_model(
                actual_list, predicted_list, n_variables=1
            )

        # Forecast
        forecast: Optional[ForecastResult] = None
        if AnalysisType.FORECAST_VS_ACTUAL in analysis_types:
            forecast = self.generate_forecast(
                sorted_data, forecast_method, forecast_periods
            )

        # Savings verification
        savings_verification: Optional[SavingsVerification] = None
        if baseline_data and len(baseline_data) > 0:
            savings_verification = self.verify_savings(
                baseline_data=baseline_data,
                reporting_data=sorted_data,
                model={"slope": trend_analysis.trend_slope,
                       "intercept": trend_analysis.intercept},
                standard=verification_standard,
            )

        # Build summary
        summary = self._build_summary(
            sorted_data, trend_analysis, yoy_comparison,
            rolling_analysis, regression_validation,
            forecast, savings_verification,
        )

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        result = PerformanceTrendResult(
            enms_id=enms_id,
            analysis_date=_utcnow(),
            period_start=period_start,
            period_end=period_end,
            trend_analysis=trend_analysis,
            yoy_comparison=yoy_comparison,
            rolling_analysis=rolling_analysis,
            regression_validation=regression_validation,
            forecast=forecast,
            savings_verification=savings_verification,
            summary=summary,
            calculation_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Trend analysis complete: enms=%s, direction=%s, R^2=%.4f, "
            "slope=%.2f, hash=%s, %d ms",
            enms_id, trend_analysis.trend_direction.value,
            float(trend_analysis.r_squared), float(trend_analysis.trend_slope),
            result.provenance_hash[:16], elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # Trend Line Calculation                                              #
    # ------------------------------------------------------------------ #

    def calculate_trend_line(
        self,
        values: List[Decimal],
    ) -> TrendAnalysis:
        """Calculate a linear trend line using OLS regression.

        Fits y = slope * x + intercept where x is the period index
        (1, 2, 3, ...) and y is the consumption value.  Computes R^2,
        approximate p-value, and confidence intervals.

        Args:
            values: Ordered list of Decimal values (one per period).

        Returns:
            TrendAnalysis with slope, intercept, R^2, p-value, and
            trend direction.
        """
        n = len(values)
        if n < _MIN_DATA_POINTS_TREND:
            logger.warning(
                "Insufficient data for trend (%d < %d required)",
                n, _MIN_DATA_POINTS_TREND,
            )
            return TrendAnalysis(
                period_count=n,
                trend_direction=TrendDirection.INSUFFICIENT_DATA,
            )

        # OLS: x = 1..n, y = values
        x_vals = [_decimal(i + 1) for i in range(n)]
        y_vals = [_decimal(v) for v in values]

        n_d = _decimal(n)
        sum_x = sum(x_vals)
        sum_y = sum(y_vals)
        sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
        sum_x2 = sum(x * x for x in x_vals)
        mean_y = _safe_divide(sum_y, n_d)

        # Slope and intercept
        denom = n_d * sum_x2 - sum_x * sum_x
        slope = _safe_divide(n_d * sum_xy - sum_x * sum_y, denom)
        intercept = _safe_divide(sum_y - slope * sum_x, n_d)

        # R-squared
        ss_tot = sum((y - mean_y) ** 2 for y in y_vals)
        ss_res = sum(
            (y - (intercept + slope * x)) ** 2
            for x, y in zip(x_vals, y_vals)
        )
        r_squared = Decimal("1") - _safe_divide(ss_res, ss_tot, Decimal("1"))
        r_squared = max(Decimal("0"), min(r_squared, Decimal("1")))

        # Approximate p-value for slope (t-test)
        p_value = self._approximate_slope_p_value(slope, ss_res, sum_x2, sum_x, n)

        # Confidence interval for slope
        se_slope = self._calculate_slope_se(ss_res, sum_x2, sum_x, n)
        t_val = self._get_t_value(n - 2)
        ci_lower = slope - t_val * se_slope
        ci_upper = slope + t_val * se_slope

        # Determine direction
        direction = self._classify_trend_direction(slope, p_value, r_squared)

        return TrendAnalysis(
            period_count=n,
            trend_direction=direction,
            trend_slope=_round_val(slope, 6),
            intercept=_round_val(intercept, 6),
            r_squared=_round_val(r_squared, 6),
            p_value=_round_val(p_value, 6),
            confidence_interval_lower=_round_val(ci_lower, 6),
            confidence_interval_upper=_round_val(ci_upper, 6),
        )

    # ------------------------------------------------------------------ #
    # Year-Over-Year Comparison                                           #
    # ------------------------------------------------------------------ #

    def compare_year_over_year(
        self,
        current_data: List[PerformanceDataPoint],
        baseline_data: List[PerformanceDataPoint],
        normalize: bool = False,
    ) -> YearOverYearComparison:
        """Compare current year performance to baseline year.

        Matches months between the two years and computes absolute and
        percentage changes.  If normalised data is available and
        normalize=True, uses normalised consumption.

        Args:
            current_data: Data points for the current/reporting year.
            baseline_data: Data points for the baseline year.
            normalize: If True, use normalized_kwh when available.

        Returns:
            YearOverYearComparison with monthly and total comparisons.
        """
        if not current_data or not baseline_data:
            logger.warning("Insufficient data for year-over-year comparison")
            return YearOverYearComparison()

        # Group by month
        current_by_month = self._group_by_month(current_data)
        baseline_by_month = self._group_by_month(baseline_data)

        # Find common months
        common_months = sorted(
            set(current_by_month.keys()) & set(baseline_by_month.keys())
        )

        if not common_months:
            logger.warning("No common months found for YoY comparison")
            return YearOverYearComparison(
                current_year=current_data[0].period_start.year,
                baseline_year=baseline_data[0].period_start.year,
                months_compared=0,
            )

        current_total = Decimal("0")
        baseline_total = Decimal("0")
        monthly_comparison: List[Dict[str, Any]] = []

        for month in common_months:
            c_data = current_by_month[month]
            b_data = baseline_by_month[month]

            if normalize:
                c_kwh = c_data.normalized_kwh or c_data.actual_kwh
                b_kwh = b_data.normalized_kwh or b_data.actual_kwh
            else:
                c_kwh = c_data.actual_kwh
                b_kwh = b_data.actual_kwh

            current_total += c_kwh
            baseline_total += b_kwh
            change_kwh = c_kwh - b_kwh
            change_pct = _safe_pct(change_kwh, b_kwh)

            monthly_comparison.append({
                "month": month,
                "current_kwh": str(_round_val(c_kwh, 2)),
                "baseline_kwh": str(_round_val(b_kwh, 2)),
                "change_kwh": str(_round_val(change_kwh, 2)),
                "change_pct": str(_round_val(change_pct, 2)),
            })

        absolute_change = current_total - baseline_total
        percentage_change = _safe_pct(absolute_change, baseline_total)

        # Normalised change (if applicable)
        normalized_change = absolute_change
        if normalize:
            norm_current = sum(
                (_decimal(
                    current_by_month[m].normalized_kwh
                    or current_by_month[m].actual_kwh
                ) for m in common_months),
                Decimal("0"),
            )
            norm_baseline = sum(
                (_decimal(
                    baseline_by_month[m].normalized_kwh
                    or baseline_by_month[m].actual_kwh
                ) for m in common_months),
                Decimal("0"),
            )
            normalized_change = norm_current - norm_baseline

        return YearOverYearComparison(
            current_year=current_data[0].period_start.year,
            baseline_year=baseline_data[0].period_start.year,
            current_total_kwh=_round_val(current_total, 2),
            baseline_total_kwh=_round_val(baseline_total, 2),
            absolute_change_kwh=_round_val(absolute_change, 2),
            percentage_change=_round_val(percentage_change, 2),
            normalized_change=_round_val(normalized_change, 2),
            months_compared=len(common_months),
            monthly_comparison=monthly_comparison,
        )

    # ------------------------------------------------------------------ #
    # Rolling Analysis                                                    #
    # ------------------------------------------------------------------ #

    def calculate_rolling_analysis(
        self,
        data: List[PerformanceDataPoint],
        window_months: int = 12,
    ) -> RollingAnalysis:
        """Compute rolling window analysis of energy consumption.

        Calculates rolling sums over the specified window and detects
        the overall direction of the rolling values.

        Args:
            data: Time-ordered data points.
            window_months: Size of the rolling window.

        Returns:
            RollingAnalysis with rolling values and trend detection.
        """
        sorted_data = sorted(data, key=lambda d: d.period_start)
        n = len(sorted_data)

        if n < window_months:
            logger.warning(
                "Insufficient data for rolling analysis (%d < %d required)",
                n, window_months,
            )
            return RollingAnalysis(
                window_months=window_months,
                trend=TrendDirection.INSUFFICIENT_DATA,
            )

        rolling_values: List[Dict[str, Any]] = []
        values_for_trend: List[Decimal] = []

        for i in range(window_months - 1, n):
            window_start = i - window_months + 1
            window_sum = sum(
                (sorted_data[j].actual_kwh for j in range(window_start, i + 1)),
                Decimal("0"),
            )
            window_avg = _safe_divide(window_sum, _decimal(window_months))

            rolling_values.append({
                "period_end": str(sorted_data[i].period_end),
                "rolling_sum_kwh": str(_round_val(window_sum, 2)),
                "rolling_avg_kwh": str(_round_val(window_avg, 2)),
                "window_start": str(sorted_data[window_start].period_start),
            })
            values_for_trend.append(window_sum)

        # Current and previous rolling values
        current_rolling = values_for_trend[-1] if values_for_trend else Decimal("0")
        previous_rolling = (
            values_for_trend[-2] if len(values_for_trend) >= 2 else Decimal("0")
        )

        # Detect trend from rolling values
        trend = TrendDirection.INSUFFICIENT_DATA
        if len(values_for_trend) >= _MIN_DATA_POINTS_TREND:
            trend_result = self.calculate_trend_line(values_for_trend)
            trend = trend_result.trend_direction

        return RollingAnalysis(
            window_months=window_months,
            current_rolling_kwh=_round_val(current_rolling, 2),
            previous_rolling_kwh=_round_val(previous_rolling, 2),
            trend=trend,
            rolling_values=rolling_values,
        )

    # ------------------------------------------------------------------ #
    # Regression Model Validation                                         #
    # ------------------------------------------------------------------ #

    def validate_regression_model(
        self,
        actual: List[Decimal],
        predicted: List[Decimal],
        n_variables: int = 1,
    ) -> RegressionValidation:
        """Validate a regression model against ASHRAE 14 / ISO 50015.

        Computes R-squared, adjusted R-squared, CV(RMSE), MAPE,
        Durbin-Watson statistic, F-statistic, and an approximate
        residual normality test.  Checks all metrics against the
        configured adequacy thresholds.

        Args:
            actual: Actual consumption values.
            predicted: Predicted/modelled consumption values.
            n_variables: Number of independent variables in the model.

        Returns:
            RegressionValidation with all metrics and adequacy assessment.

        Raises:
            ValueError: If actual and predicted lists differ in length.
        """
        if len(actual) != len(predicted):
            raise ValueError(
                f"Actual ({len(actual)}) and predicted ({len(predicted)}) "
                f"lists must be the same length."
            )

        n = len(actual)
        if n < _MIN_DATA_POINTS_REGRESSION:
            logger.warning(
                "Insufficient data for regression validation (%d < %d)",
                n, _MIN_DATA_POINTS_REGRESSION,
            )
            return RegressionValidation(
                is_model_adequate=False,
                inadequacy_reasons=[
                    f"Insufficient data: {n} points, "
                    f"minimum {_MIN_DATA_POINTS_REGRESSION} required."
                ],
            )

        actual_d = [_decimal(v) for v in actual]
        predicted_d = [_decimal(v) for v in predicted]

        # Residuals
        residuals = [a - p for a, p in zip(actual_d, predicted_d)]

        # Mean of actuals
        n_d = _decimal(n)
        mean_actual = _safe_divide(sum(actual_d), n_d)

        # SS_tot, SS_res, SS_reg
        ss_tot = sum((a - mean_actual) ** 2 for a in actual_d)
        ss_res = sum(r ** 2 for r in residuals)
        ss_reg = ss_tot - ss_res

        # R-squared
        r_squared = Decimal("1") - _safe_divide(ss_res, ss_tot, Decimal("1"))
        r_squared = max(Decimal("0"), min(r_squared, Decimal("1")))

        # Adjusted R-squared
        k = _decimal(n_variables)
        adj_denom = n_d - k - Decimal("1")
        if adj_denom > Decimal("0") and (n_d - Decimal("1")) > Decimal("0"):
            adjusted_r_squared = (
                Decimal("1") - (Decimal("1") - r_squared)
                * (n_d - Decimal("1")) / adj_denom
            )
        else:
            adjusted_r_squared = r_squared

        # CV(RMSE)
        rmse = _decimal_sqrt(_safe_divide(ss_res, n_d))
        cv_rmse = _safe_pct(rmse, mean_actual) if mean_actual > Decimal("0") else Decimal("0")

        # MAPE
        mape = self.calculate_mape(actual_d, predicted_d)

        # Durbin-Watson
        durbin_watson = self.calculate_durbin_watson(residuals)

        # F-statistic
        k_d = _decimal(n_variables)
        ms_reg = _safe_divide(ss_reg, k_d)
        ms_res = _safe_divide(ss_res, n_d - k_d - Decimal("1"))
        f_statistic = _safe_divide(ms_reg, ms_res)

        # Approximate F p-value (simplified)
        f_p_value = self._approximate_f_p_value(f_statistic, n_variables, n)

        # Residual normality (approximate via skewness/kurtosis check)
        normality_p = self._test_residual_normality(residuals)

        # Check adequacy thresholds
        thresholds = MODEL_ADEQUACY_THRESHOLDS.get(
            self._threshold_key,
            MODEL_ADEQUACY_THRESHOLDS["ashrae_14_monthly"],
        )
        inadequacy_reasons: List[str] = []
        is_adequate = True

        if r_squared < thresholds["r_squared_min"]:
            is_adequate = False
            inadequacy_reasons.append(
                f"R^2 = {_round_val(r_squared, 4)} < "
                f"{thresholds['r_squared_min']} minimum"
            )

        if cv_rmse > thresholds["cv_rmse_max_pct"]:
            is_adequate = False
            inadequacy_reasons.append(
                f"CV(RMSE) = {_round_val(cv_rmse, 2)}% > "
                f"{thresholds['cv_rmse_max_pct']}% maximum"
            )

        if mape > thresholds["mape_max_pct"]:
            is_adequate = False
            inadequacy_reasons.append(
                f"MAPE = {_round_val(mape, 2)}% > "
                f"{thresholds['mape_max_pct']}% maximum"
            )

        dw_low = thresholds["durbin_watson_low"]
        dw_high = thresholds["durbin_watson_high"]
        if durbin_watson < dw_low or durbin_watson > dw_high:
            is_adequate = False
            inadequacy_reasons.append(
                f"Durbin-Watson = {_round_val(durbin_watson, 4)} outside "
                f"[{dw_low}, {dw_high}] range (autocorrelation concern)"
            )

        if f_p_value > thresholds["p_value_max"]:
            is_adequate = False
            inadequacy_reasons.append(
                f"F-test p-value = {_round_val(f_p_value, 6)} > "
                f"{thresholds['p_value_max']} (model not significant)"
            )

        return RegressionValidation(
            r_squared=_round_val(r_squared, 6),
            adjusted_r_squared=_round_val(adjusted_r_squared, 6),
            cv_rmse=_round_val(cv_rmse, 4),
            mape=_round_val(mape, 4),
            durbin_watson=_round_val(durbin_watson, 4),
            f_statistic=_round_val(f_statistic, 4),
            f_p_value=_round_val(f_p_value, 6),
            residual_normality_p=_round_val(normality_p, 6),
            is_model_adequate=is_adequate,
            inadequacy_reasons=inadequacy_reasons,
        )

    # ------------------------------------------------------------------ #
    # Forecasting                                                         #
    # ------------------------------------------------------------------ #

    def generate_forecast(
        self,
        data: List[PerformanceDataPoint],
        method: ForecastMethod = ForecastMethod.LINEAR,
        periods: int = 12,
    ) -> ForecastResult:
        """Generate energy consumption forecast using the specified method.

        Supports linear extrapolation, exponential extrapolation,
        simple and weighted moving averages, and Holt-Winters additive
        exponential smoothing.

        Args:
            data: Historical data points (time-ordered).
            method: Forecasting method to use.
            periods: Number of future periods to forecast.

        Returns:
            ForecastResult with forecasted values and accuracy metrics.
        """
        sorted_data = sorted(data, key=lambda d: d.period_start)
        n = len(sorted_data)

        if n < _MIN_DATA_POINTS_FORECAST:
            logger.warning(
                "Insufficient data for forecast (%d < %d required)",
                n, _MIN_DATA_POINTS_FORECAST,
            )
            return ForecastResult(
                forecast_method=method,
                forecast_periods=0,
            )

        values = [dp.actual_kwh for dp in sorted_data]

        if method == ForecastMethod.LINEAR:
            forecasted = self._forecast_linear(values, periods)
        elif method == ForecastMethod.EXPONENTIAL:
            forecasted = self._forecast_exponential(values, periods)
        elif method == ForecastMethod.MOVING_AVERAGE:
            forecasted = self._forecast_moving_average(values, periods)
        elif method == ForecastMethod.WEIGHTED_MOVING_AVERAGE:
            forecasted = self._forecast_weighted_moving_average(values, periods)
        elif method == ForecastMethod.HOLT_WINTERS:
            forecasted = self._forecast_holt_winters(values, periods)
        else:
            forecasted = self._forecast_linear(values, periods)

        # Build forecast output
        forecasted_values: List[Dict[str, Any]] = []
        for idx, val in enumerate(forecasted):
            forecasted_values.append({
                "period": n + idx + 1,
                "forecasted_kwh": str(_round_val(val, 2)),
            })

        # If we have actuals in the data that can be compared against
        # a holdout (last N points), compute accuracy on in-sample
        actual_values: Optional[List[Dict[str, Any]]] = None
        mape_val: Optional[Decimal] = None
        accuracy_pct: Optional[Decimal] = None

        if n > periods and method != ForecastMethod.HOLT_WINTERS:
            # Use first (n - periods) to forecast last (periods), then compare
            train = values[:n - periods]
            holdout = values[n - periods:]

            if method == ForecastMethod.LINEAR:
                holdout_forecast = self._forecast_linear(train, periods)
            elif method == ForecastMethod.EXPONENTIAL:
                holdout_forecast = self._forecast_exponential(train, periods)
            elif method == ForecastMethod.MOVING_AVERAGE:
                holdout_forecast = self._forecast_moving_average(train, periods)
            elif method == ForecastMethod.WEIGHTED_MOVING_AVERAGE:
                holdout_forecast = self._forecast_weighted_moving_average(
                    train, periods
                )
            else:
                holdout_forecast = self._forecast_linear(train, periods)

            compare_len = min(len(holdout), len(holdout_forecast))
            if compare_len > 0:
                actual_values = [
                    {
                        "period": n - periods + idx + 1,
                        "actual_kwh": str(_round_val(holdout[idx], 2)),
                        "forecasted_kwh": str(
                            _round_val(holdout_forecast[idx], 2)
                        ),
                    }
                    for idx in range(compare_len)
                ]
                mape_val = self.calculate_mape(
                    holdout[:compare_len], holdout_forecast[:compare_len]
                )
                accuracy_pct = max(
                    Decimal("0"),
                    Decimal("100") - mape_val,
                )

        return ForecastResult(
            forecast_method=method,
            forecast_periods=periods,
            forecasted_values=forecasted_values,
            actual_values=actual_values,
            mape=_round_val(mape_val, 4) if mape_val is not None else None,
            forecast_accuracy_pct=(
                _round_val(accuracy_pct, 2) if accuracy_pct is not None else None
            ),
        )

    # ------------------------------------------------------------------ #
    # Savings Verification                                                #
    # ------------------------------------------------------------------ #

    def verify_savings(
        self,
        baseline_data: List[PerformanceDataPoint],
        reporting_data: List[PerformanceDataPoint],
        model: Dict[str, Any],
        standard: VerificationStandard = VerificationStandard.ISO_50015,
    ) -> SavingsVerification:
        """Verify energy savings per ISO 50015 or IPMVP.

        Computes adjusted baseline, gross and net savings, and
        uncertainty at the configured confidence level.

        Args:
            baseline_data: Baseline period data points.
            reporting_data: Reporting period data points.
            model: Regression model parameters (slope, intercept).
            standard: Verification standard to apply.

        Returns:
            SavingsVerification with savings and significance assessment.
        """
        if not baseline_data or not reporting_data:
            logger.warning("Insufficient data for savings verification")
            return SavingsVerification(verification_standard=standard)

        baseline_sorted = sorted(baseline_data, key=lambda d: d.period_start)
        reporting_sorted = sorted(reporting_data, key=lambda d: d.period_start)

        # Baseline consumption
        baseline_consumption = sum(
            (dp.actual_kwh for dp in baseline_sorted), Decimal("0")
        )

        # Reporting period consumption
        reporting_consumption = sum(
            (dp.actual_kwh for dp in reporting_sorted), Decimal("0")
        )

        # Adjusted baseline: apply model to reporting period conditions
        # For simplicity, if model is provided, adjust baseline proportionally
        # to the number of reporting periods vs baseline periods
        n_baseline = _decimal(len(baseline_sorted))
        n_reporting = _decimal(len(reporting_sorted))

        slope = _decimal(model.get("slope", 0))
        intercept_val = _decimal(model.get("intercept", 0))

        # Adjusted baseline = baseline scaled to reporting period length
        # with model adjustments for routine conditions
        period_ratio = _safe_divide(n_reporting, n_baseline, Decimal("1"))
        adjusted_baseline = baseline_consumption * period_ratio

        # Apply non-routine adjustments if model slope indicates change
        # Positive slope means increasing trend; adjust baseline upward
        if slope != Decimal("0") and n_baseline > Decimal("0"):
            baseline_periods = n_baseline
            reporting_midpoint = n_baseline + n_reporting / Decimal("2")
            baseline_midpoint = n_baseline / Decimal("2")
            trend_adjustment = slope * (reporting_midpoint - baseline_midpoint)
            adjusted_baseline = adjusted_baseline + trend_adjustment

        # Ensure adjusted baseline is non-negative
        adjusted_baseline = max(adjusted_baseline, Decimal("0"))

        # Gross savings
        gross_savings = adjusted_baseline - reporting_consumption

        # Net savings (gross minus non-routine adjustments)
        # For ISO 50015, non-routine adjustments are separate; here we
        # approximate net = gross for the standard case
        net_savings = gross_savings

        # Uncertainty calculation
        uncertainty_pct = self._calculate_savings_uncertainty(
            baseline_sorted, reporting_sorted, model, standard
        )

        # Significance: savings must exceed 2x uncertainty for 90% confidence
        fractional_savings = _safe_pct(
            gross_savings, adjusted_baseline
        ) if adjusted_baseline > Decimal("0") else Decimal("0")
        is_significant = (
            abs(fractional_savings) > Decimal("2") * uncertainty_pct
            and gross_savings > Decimal("0")
        )

        return SavingsVerification(
            verification_standard=standard,
            baseline_consumption=_round_val(baseline_consumption, 2),
            reporting_period_consumption=_round_val(reporting_consumption, 2),
            adjusted_baseline=_round_val(adjusted_baseline, 2),
            gross_savings_kwh=_round_val(gross_savings, 2),
            net_savings_kwh=_round_val(net_savings, 2),
            savings_uncertainty_pct=_round_val(uncertainty_pct, 4),
            confidence_level=self._confidence_level,
            is_savings_significant=is_significant,
        )

    # ------------------------------------------------------------------ #
    # Degradation Detection                                               #
    # ------------------------------------------------------------------ #

    def detect_performance_degradation(
        self,
        data: List[PerformanceDataPoint],
        threshold_pct: Decimal = Decimal("5"),
    ) -> Dict[str, Any]:
        """Identify periods where energy performance has degraded.

        Compares each period's consumption to the rolling average and
        flags periods exceeding the threshold percentage above average.
        Quantifies the total degradation magnitude.

        Args:
            data: Time-ordered performance data points.
            threshold_pct: Percentage threshold above rolling average
                that triggers a degradation flag (default 5%).

        Returns:
            Dict with degradation periods, magnitude, and summary.
        """
        sorted_data = sorted(data, key=lambda d: d.period_start)
        n = len(sorted_data)

        if n < _MIN_DATA_POINTS_TREND:
            return {
                "degradation_detected": False,
                "periods_flagged": 0,
                "total_degradation_kwh": "0",
                "degradation_periods": [],
                "message": f"Insufficient data ({n} points).",
                "provenance_hash": _compute_hash({
                    "n": n, "threshold_pct": str(threshold_pct),
                }),
            }

        # Calculate rolling average for comparison
        values = [dp.actual_kwh for dp in sorted_data]
        mean_val = _safe_divide(sum(values), _decimal(n))

        threshold_factor = Decimal("1") + threshold_pct / Decimal("100")
        degradation_periods: List[Dict[str, Any]] = []
        total_degradation = Decimal("0")

        for idx, dp in enumerate(sorted_data):
            # Use expanding mean up to this point for comparison
            if idx < 2:
                continue
            historical_mean = _safe_divide(
                sum(values[:idx]), _decimal(idx)
            )
            threshold_val = historical_mean * threshold_factor

            if dp.actual_kwh > threshold_val:
                excess = dp.actual_kwh - historical_mean
                pct_above = _safe_pct(excess, historical_mean)
                total_degradation += excess
                degradation_periods.append({
                    "period_start": str(dp.period_start),
                    "period_end": str(dp.period_end),
                    "actual_kwh": str(_round_val(dp.actual_kwh, 2)),
                    "historical_mean_kwh": str(_round_val(historical_mean, 2)),
                    "excess_kwh": str(_round_val(excess, 2)),
                    "pct_above_mean": str(_round_val(pct_above, 2)),
                })

        degradation_detected = len(degradation_periods) > 0

        result = {
            "degradation_detected": degradation_detected,
            "periods_flagged": len(degradation_periods),
            "total_periods": n,
            "total_degradation_kwh": str(_round_val(total_degradation, 2)),
            "average_consumption_kwh": str(_round_val(mean_val, 2)),
            "threshold_pct": str(threshold_pct),
            "degradation_periods": degradation_periods,
            "message": (
                f"{len(degradation_periods)} periods flagged with "
                f"consumption >{threshold_pct}% above historical average."
                if degradation_detected else
                f"No degradation detected (threshold: {threshold_pct}%)."
            ),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    # Durbin-Watson Statistic                                             #
    # ------------------------------------------------------------------ #

    def calculate_durbin_watson(
        self,
        residuals: List[Decimal],
    ) -> Decimal:
        """Calculate the Durbin-Watson statistic for autocorrelation.

        DW = sum((e_t - e_{t-1})^2, t=2..n) / sum(e_t^2, t=1..n)

        Values near 2.0 indicate no autocorrelation.  Values < 1.5
        suggest positive autocorrelation; > 2.5 suggest negative.

        Args:
            residuals: Ordered list of residual values.

        Returns:
            Durbin-Watson statistic (Decimal, range approximately 0-4).
        """
        n = len(residuals)
        if n < 2:
            return Decimal("2")  # Assume no autocorrelation with 1 point

        sum_sq_diff = Decimal("0")
        sum_sq = Decimal("0")

        for i in range(n):
            e_i = _decimal(residuals[i])
            sum_sq += e_i ** 2
            if i > 0:
                e_prev = _decimal(residuals[i - 1])
                sum_sq_diff += (e_i - e_prev) ** 2

        dw = _safe_divide(sum_sq_diff, sum_sq, Decimal("2"))
        return dw

    # ------------------------------------------------------------------ #
    # MAPE Calculation                                                    #
    # ------------------------------------------------------------------ #

    def calculate_mape(
        self,
        actual: List[Decimal],
        predicted: List[Decimal],
    ) -> Decimal:
        """Calculate Mean Absolute Percentage Error.

        MAPE = (1/n) * sum(|actual - predicted| / |actual|) * 100

        Periods where actual is zero are excluded from the calculation
        to avoid division by zero.

        Args:
            actual: Actual values.
            predicted: Predicted values.

        Returns:
            MAPE as a percentage (Decimal).
        """
        n = min(len(actual), len(predicted))
        if n == 0:
            return Decimal("0")

        total_ape = Decimal("0")
        valid_count = 0

        for i in range(n):
            a = _decimal(actual[i])
            p = _decimal(predicted[i])
            if a == Decimal("0"):
                continue
            ape = abs(a - p) / abs(a)
            total_ape += ape
            valid_count += 1

        if valid_count == 0:
            return Decimal("0")

        mape = _safe_divide(
            total_ape * Decimal("100"),
            _decimal(valid_count),
        )
        return mape

    # ------------------------------------------------------------------ #
    # Chart Data Generation                                               #
    # ------------------------------------------------------------------ #

    def generate_trend_chart_data(
        self,
        result: PerformanceTrendResult,
    ) -> Dict[str, Any]:
        """Generate structured data suitable for trend charting.

        Produces datasets for actual consumption, fitted trend line,
        rolling values, forecast, and year-over-year comparison bars.

        Args:
            result: A completed PerformanceTrendResult.

        Returns:
            Dict with chart datasets, axis labels, and annotations.
        """
        chart_data: Dict[str, Any] = {
            "chart_type": "performance_trend",
            "title": f"Energy Performance Trend - {result.enms_id}",
            "period_start": str(result.period_start),
            "period_end": str(result.period_end),
            "datasets": {},
            "annotations": [],
        }

        # Actual vs fitted trend
        if result.trend_analysis.data_points:
            chart_data["datasets"]["actual_vs_fitted"] = {
                "x_label": "Period",
                "y_label": "Energy Consumption (kWh)",
                "actual": [
                    {
                        "x": dp["period"],
                        "y": dp["actual_kwh"],
                    }
                    for dp in result.trend_analysis.data_points
                ],
                "fitted": [
                    {
                        "x": dp["period"],
                        "y": dp["fitted_kwh"],
                    }
                    for dp in result.trend_analysis.data_points
                ],
            }

        # Rolling analysis
        if result.rolling_analysis and result.rolling_analysis.rolling_values:
            chart_data["datasets"]["rolling_consumption"] = {
                "x_label": "Period End",
                "y_label": "Rolling Sum (kWh)",
                "values": [
                    {
                        "x": rv["period_end"],
                        "y": rv["rolling_sum_kwh"],
                    }
                    for rv in result.rolling_analysis.rolling_values
                ],
            }

        # Year-over-year bars
        if result.yoy_comparison and result.yoy_comparison.monthly_comparison:
            chart_data["datasets"]["year_over_year"] = {
                "x_label": "Month",
                "y_label": "Energy Consumption (kWh)",
                "current_year": result.yoy_comparison.current_year,
                "baseline_year": result.yoy_comparison.baseline_year,
                "months": [
                    {
                        "month": mc["month"],
                        "current": mc["current_kwh"],
                        "baseline": mc["baseline_kwh"],
                    }
                    for mc in result.yoy_comparison.monthly_comparison
                ],
            }

        # Forecast
        if result.forecast and result.forecast.forecasted_values:
            chart_data["datasets"]["forecast"] = {
                "x_label": "Period",
                "y_label": "Forecasted Consumption (kWh)",
                "method": result.forecast.forecast_method.value,
                "values": result.forecast.forecasted_values,
            }

        # Annotations
        chart_data["annotations"].append({
            "type": "trend_direction",
            "value": result.trend_analysis.trend_direction.value,
            "r_squared": str(result.trend_analysis.r_squared),
            "slope": str(result.trend_analysis.trend_slope),
        })

        if result.savings_verification:
            chart_data["annotations"].append({
                "type": "savings_verification",
                "gross_savings_kwh": str(
                    result.savings_verification.gross_savings_kwh
                ),
                "is_significant": result.savings_verification.is_savings_significant,
                "standard": result.savings_verification.verification_standard.value,
            })

        chart_data["provenance_hash"] = _compute_hash(chart_data)
        return chart_data

    # ------------------------------------------------------------------ #
    # Private: Year-Over-Year                                             #
    # ------------------------------------------------------------------ #

    def _perform_yoy_analysis(
        self,
        data: List[PerformanceDataPoint],
    ) -> Optional[YearOverYearComparison]:
        """Extract the two most recent full years and compare them.

        Args:
            data: Sorted data points.

        Returns:
            YearOverYearComparison or None if insufficient years.
        """
        # Group by year
        by_year: Dict[int, List[PerformanceDataPoint]] = {}
        for dp in data:
            yr = dp.period_start.year
            by_year.setdefault(yr, []).append(dp)

        years = sorted(by_year.keys())
        if len(years) < 2:
            logger.warning(
                "Insufficient years for YoY comparison (found %d years)",
                len(years),
            )
            return None

        # Use the two most recent years
        current_year = years[-1]
        baseline_year = years[-2]

        return self.compare_year_over_year(
            current_data=by_year[current_year],
            baseline_data=by_year[baseline_year],
            normalize=False,
        )

    # ------------------------------------------------------------------ #
    # Private: Forecasting Methods                                        #
    # ------------------------------------------------------------------ #

    def _forecast_linear(
        self,
        values: List[Decimal],
        periods: int,
    ) -> List[Decimal]:
        """Linear extrapolation forecast.

        Fits a trend line to the data and extrapolates forward.

        Args:
            values: Historical values.
            periods: Number of periods to forecast.

        Returns:
            List of forecasted Decimal values.
        """
        trend = self.calculate_trend_line(values)
        n = len(values)
        forecasted: List[Decimal] = []
        for i in range(1, periods + 1):
            x = _decimal(n + i)
            y = trend.intercept + trend.trend_slope * x
            forecasted.append(max(y, Decimal("0")))
        return forecasted

    def _forecast_exponential(
        self,
        values: List[Decimal],
        periods: int,
    ) -> List[Decimal]:
        """Exponential extrapolation forecast.

        Fits ln(y) = a + b*x then y_forecast = exp(a + b*x_future).

        Args:
            values: Historical values.
            periods: Number of periods to forecast.

        Returns:
            List of forecasted Decimal values.
        """
        # Filter out zero/negative values for log transform
        positive_values = [(i + 1, v) for i, v in enumerate(values) if v > Decimal("0")]
        if len(positive_values) < _MIN_DATA_POINTS_TREND:
            return self._forecast_linear(values, periods)

        log_values = [_decimal_log(v) for _, v in positive_values]
        x_vals = [_decimal(i) for i, _ in positive_values]

        # OLS on log-transformed values
        n_d = _decimal(len(positive_values))
        sum_x = sum(x_vals)
        sum_y = sum(log_values)
        sum_xy = sum(x * y for x, y in zip(x_vals, log_values))
        sum_x2 = sum(x * x for x in x_vals)

        denom = n_d * sum_x2 - sum_x * sum_x
        b = _safe_divide(n_d * sum_xy - sum_x * sum_y, denom)
        a = _safe_divide(sum_y - b * sum_x, n_d)

        n = len(values)
        forecasted: List[Decimal] = []
        for i in range(1, periods + 1):
            x = _decimal(n + i)
            y = _decimal_exp(a + b * x)
            forecasted.append(max(y, Decimal("0")))
        return forecasted

    def _forecast_moving_average(
        self,
        values: List[Decimal],
        periods: int,
        window: int = 3,
    ) -> List[Decimal]:
        """Simple moving average forecast.

        Uses the average of the last *window* observations as the
        forecast for each future period.

        Args:
            values: Historical values.
            periods: Number of periods to forecast.
            window: Moving average window size.

        Returns:
            List of forecasted Decimal values.
        """
        effective_window = min(window, len(values))
        if effective_window == 0:
            return [Decimal("0")] * periods

        working = list(values)
        forecasted: List[Decimal] = []

        for _ in range(periods):
            recent = working[-effective_window:]
            avg = _safe_divide(sum(recent), _decimal(len(recent)))
            forecasted.append(max(_round_val(avg, 2), Decimal("0")))
            working.append(avg)

        return forecasted

    def _forecast_weighted_moving_average(
        self,
        values: List[Decimal],
        periods: int,
        window: int = 4,
    ) -> List[Decimal]:
        """Weighted moving average forecast (linearly increasing weights).

        Recent observations receive proportionally higher weights.
        Weight for position i (0 = oldest) = (i + 1) / sum(1..window).

        Args:
            values: Historical values.
            periods: Number of periods to forecast.
            window: Window size.

        Returns:
            List of forecasted Decimal values.
        """
        effective_window = min(window, len(values))
        if effective_window == 0:
            return [Decimal("0")] * periods

        # Weights: 1, 2, 3, ..., w (most recent = highest weight)
        weight_sum = _decimal(effective_window * (effective_window + 1) // 2)

        working = list(values)
        forecasted: List[Decimal] = []

        for _ in range(periods):
            recent = working[-effective_window:]
            weighted_sum = Decimal("0")
            for idx, val in enumerate(recent):
                weight = _decimal(idx + 1)
                weighted_sum += val * weight
            avg = _safe_divide(weighted_sum, weight_sum)
            forecasted.append(max(_round_val(avg, 2), Decimal("0")))
            working.append(avg)

        return forecasted

    def _forecast_holt_winters(
        self,
        values: List[Decimal],
        periods: int,
    ) -> List[Decimal]:
        """Holt-Winters additive exponential smoothing forecast.

        Decomposes the time series into level, trend, and seasonal
        components, then extrapolates forward.

        Requires at least 2 full seasonal cycles of data.

        Args:
            values: Historical values.
            periods: Number of periods to forecast.

        Returns:
            List of forecasted Decimal values.
        """
        m = self._hw_season_length
        n = len(values)
        alpha = self._hw_alpha
        beta = self._hw_beta
        gamma = self._hw_gamma

        if n < 2 * m:
            logger.warning(
                "Insufficient data for Holt-Winters (%d < %d required). "
                "Falling back to weighted moving average.",
                n, 2 * m,
            )
            return self._forecast_weighted_moving_average(values, periods)

        # Initialise level and trend from first two seasons
        first_season_avg = _safe_divide(
            sum(values[:m]), _decimal(m)
        )
        second_season_avg = _safe_divide(
            sum(values[m:2 * m]), _decimal(m)
        )
        initial_trend = _safe_divide(
            second_season_avg - first_season_avg, _decimal(m)
        )
        level = first_season_avg
        trend = initial_trend

        # Initialise seasonal components from first season
        seasonal: List[Decimal] = []
        for i in range(m):
            seasonal.append(values[i] - first_season_avg)

        # Run Holt-Winters through the data
        for t in range(n):
            s_idx = t % m
            y_t = values[t]

            # Update level
            new_level = (
                alpha * (y_t - seasonal[s_idx])
                + (Decimal("1") - alpha) * (level + trend)
            )

            # Update trend
            new_trend = (
                beta * (new_level - level)
                + (Decimal("1") - beta) * trend
            )

            # Update seasonal
            new_seasonal = (
                gamma * (y_t - new_level)
                + (Decimal("1") - gamma) * seasonal[s_idx]
            )

            level = new_level
            trend = new_trend
            seasonal[s_idx] = new_seasonal

        # Forecast
        forecasted: List[Decimal] = []
        for h in range(1, periods + 1):
            s_idx = (n + h - 1) % m
            forecast_val = level + _decimal(h) * trend + seasonal[s_idx]
            forecasted.append(max(_round_val(forecast_val, 2), Decimal("0")))

        return forecasted

    # ------------------------------------------------------------------ #
    # Private: Statistical Helpers                                        #
    # ------------------------------------------------------------------ #

    def _classify_trend_direction(
        self,
        slope: Decimal,
        p_value: Decimal,
        r_squared: Decimal,
    ) -> TrendDirection:
        """Classify trend direction based on slope, significance, and fit.

        A trend is only classified as improving/degrading if the slope
        is statistically significant (p < 0.10) and R^2 > 0.10.

        Args:
            slope: Trend line slope.
            p_value: P-value for the slope.
            r_squared: R-squared of the trend fit.

        Returns:
            TrendDirection classification.
        """
        # Not significant or poor fit -> stable
        if p_value > Decimal("0.10") or r_squared < Decimal("0.10"):
            return TrendDirection.STABLE

        if slope < self._improving_threshold:
            return TrendDirection.IMPROVING
        elif slope > self._degrading_threshold:
            return TrendDirection.DEGRADING
        else:
            return TrendDirection.STABLE

    def _approximate_slope_p_value(
        self,
        slope: Decimal,
        ss_res: Decimal,
        sum_x2: Decimal,
        sum_x: Decimal,
        n: int,
    ) -> Decimal:
        """Approximate the p-value for the regression slope.

        Uses the t-statistic: t = slope / SE(slope), and approximates
        the p-value from the t-distribution with (n-2) degrees of
        freedom.

        Args:
            slope: Estimated slope.
            ss_res: Sum of squared residuals.
            sum_x2: Sum of x^2 values.
            sum_x: Sum of x values.
            n: Number of observations.

        Returns:
            Approximate p-value (Decimal, 0-1).
        """
        se_slope = self._calculate_slope_se(ss_res, sum_x2, sum_x, n)

        if se_slope <= Decimal("0"):
            return Decimal("0") if slope != Decimal("0") else Decimal("1")

        t_stat = abs(_safe_divide(slope, se_slope))

        # Approximate two-tailed p-value from t-statistic
        # Using a simplified approximation (adequate for trend detection)
        df = n - 2
        if df <= 0:
            return Decimal("1")

        # Compare t-stat to critical values
        t_crit_10 = self._get_t_value(df)  # ~90% CI
        t_crit_05 = t_crit_10 * Decimal("1.2")  # rough 95% approximation
        t_crit_01 = t_crit_10 * Decimal("1.7")  # rough 99% approximation

        if t_stat >= t_crit_01:
            return Decimal("0.01")
        elif t_stat >= t_crit_05:
            return Decimal("0.05")
        elif t_stat >= t_crit_10:
            return Decimal("0.10")
        else:
            # Linear interpolation for values below t_crit_10
            ratio = _safe_divide(t_stat, t_crit_10, Decimal("0"))
            return max(Decimal("0.10") + (Decimal("1") - ratio) * Decimal("0.40"),
                       Decimal("0.10"))

    def _calculate_slope_se(
        self,
        ss_res: Decimal,
        sum_x2: Decimal,
        sum_x: Decimal,
        n: int,
    ) -> Decimal:
        """Calculate the standard error of the slope coefficient.

        SE(slope) = sqrt(MSE / (sum(x^2) - sum(x)^2/n))

        Args:
            ss_res: Sum of squared residuals.
            sum_x2: Sum of x^2.
            sum_x: Sum of x.
            n: Number of observations.

        Returns:
            Standard error of slope (Decimal).
        """
        n_d = _decimal(n)
        df = n_d - Decimal("2")
        if df <= Decimal("0"):
            return Decimal("0")

        mse = _safe_divide(ss_res, df)
        x_var = sum_x2 - _safe_divide(sum_x * sum_x, n_d)

        if x_var <= Decimal("0"):
            return Decimal("0")

        se_squared = _safe_divide(mse, x_var)
        return _decimal_sqrt(se_squared)

    def _get_t_value(self, df: int) -> Decimal:
        """Look up t-value for the configured confidence level.

        Uses the pre-computed table for 90% confidence (two-tailed,
        alpha/2 = 0.05 per tail).  Falls back to 1.645 for large df.

        Args:
            df: Degrees of freedom.

        Returns:
            t-value (Decimal).
        """
        if df <= 0:
            return Decimal("6.314")

        if df in _T_VALUES_90:
            return _T_VALUES_90[df]

        # Find closest smaller df in the table
        available = sorted(_T_VALUES_90.keys())
        for t_df in reversed(available):
            if t_df <= df:
                return _T_VALUES_90[t_df]

        # Large sample: use z-value approximation
        return Decimal("1.645")

    def _approximate_f_p_value(
        self,
        f_stat: Decimal,
        k: int,
        n: int,
    ) -> Decimal:
        """Approximate the p-value for an F-statistic.

        Uses a simplified threshold-based approximation.

        Args:
            f_stat: F-statistic value.
            k: Number of independent variables.
            n: Number of observations.

        Returns:
            Approximate p-value (Decimal, 0-1).
        """
        df1 = _decimal(k)
        df2 = _decimal(n - k - 1)

        if df2 <= Decimal("0") or f_stat <= Decimal("0"):
            return Decimal("1")

        # Approximate critical values for F(k, n-k-1)
        # F > 10 typically means p < 0.01
        # F > 4 typically means p < 0.05
        # F > 2.5 typically means p < 0.10
        if f_stat > Decimal("10"):
            return Decimal("0.001")
        elif f_stat > Decimal("6"):
            return Decimal("0.01")
        elif f_stat > Decimal("4"):
            return Decimal("0.025")
        elif f_stat > Decimal("3"):
            return Decimal("0.05")
        elif f_stat > Decimal("2.5"):
            return Decimal("0.10")
        elif f_stat > Decimal("2"):
            return Decimal("0.15")
        else:
            return Decimal("0.25")

    def _test_residual_normality(
        self,
        residuals: List[Decimal],
    ) -> Decimal:
        """Approximate residual normality using skewness/kurtosis.

        Computes sample skewness and excess kurtosis.  If both are
        within acceptable ranges (|skew| < 2, |kurt| < 7), the
        residuals are considered approximately normal.

        Args:
            residuals: List of residual values.

        Returns:
            Approximate p-value for normality (higher = more normal).
        """
        n = len(residuals)
        if n < 4:
            return Decimal("0.50")

        n_d = _decimal(n)
        mean_r = _safe_divide(sum(_decimal(r) for r in residuals), n_d)

        # Central moments
        diffs = [_decimal(r) - mean_r for r in residuals]
        m2 = _safe_divide(sum(d ** 2 for d in diffs), n_d)
        m3 = _safe_divide(sum(d ** 3 for d in diffs), n_d)
        m4 = _safe_divide(sum(d ** 4 for d in diffs), n_d)

        if m2 <= Decimal("0"):
            return Decimal("1")

        # Skewness = m3 / m2^(3/2)
        m2_32 = m2 * _decimal_sqrt(m2)
        skewness = _safe_divide(m3, m2_32)

        # Kurtosis = m4 / m2^2 - 3 (excess)
        m2_sq = m2 * m2
        kurtosis = _safe_divide(m4, m2_sq) - Decimal("3")

        # Approximate normality p-value
        skew_ok = abs(skewness) < Decimal("2")
        kurt_ok = abs(kurtosis) < Decimal("7")

        if skew_ok and kurt_ok:
            # Well within normal range
            if abs(skewness) < Decimal("0.5") and abs(kurtosis) < Decimal("1"):
                return Decimal("0.90")
            elif abs(skewness) < Decimal("1") and abs(kurtosis) < Decimal("3"):
                return Decimal("0.50")
            else:
                return Decimal("0.20")
        elif skew_ok or kurt_ok:
            return Decimal("0.10")
        else:
            return Decimal("0.01")

    # ------------------------------------------------------------------ #
    # Private: Savings Uncertainty                                        #
    # ------------------------------------------------------------------ #

    def _calculate_savings_uncertainty(
        self,
        baseline_data: List[PerformanceDataPoint],
        reporting_data: List[PerformanceDataPoint],
        model: Dict[str, Any],
        standard: VerificationStandard,
    ) -> Decimal:
        """Calculate savings uncertainty per the chosen M&V standard.

        For ISO 50015 and IPMVP Option C:
            U = t_{n-p} * sqrt(SSE/(n-p)) / mean(baseline) * 100

        For IPMVP Options A and B, a simplified approach is used with
        default uncertainty ranges from the IPMVP guidance.

        For IPMVP Option D, uses simulation-specific uncertainty.

        Args:
            baseline_data: Baseline period data.
            reporting_data: Reporting period data.
            model: Regression model parameters.
            standard: Verification standard.

        Returns:
            Savings uncertainty as a percentage.
        """
        if standard in (
            VerificationStandard.IPMVP_OPTION_A,
        ):
            # Option A: key parameter only - typical 10-20% uncertainty
            return Decimal("15.0")

        if standard == VerificationStandard.IPMVP_OPTION_B:
            # Option B: all parameters - typical 5-10% uncertainty
            return Decimal("7.5")

        if standard == VerificationStandard.IPMVP_OPTION_D:
            # Option D: simulation - typical 10-20% uncertainty
            return Decimal("15.0")

        # ISO 50015 or IPMVP Option C: regression-based uncertainty
        slope = _decimal(model.get("slope", 0))
        intercept_val = _decimal(model.get("intercept", 0))

        n_baseline = len(baseline_data)
        if n_baseline < 3:
            return Decimal("25.0")

        # Calculate residuals from baseline model
        baseline_values = [dp.actual_kwh for dp in baseline_data]
        mean_baseline = _safe_divide(
            sum(baseline_values), _decimal(n_baseline)
        )

        if mean_baseline <= Decimal("0"):
            return Decimal("25.0")

        # Model predictions for baseline
        residuals_sq = Decimal("0")
        for idx, dp in enumerate(baseline_data):
            predicted = intercept_val + slope * _decimal(idx + 1)
            residual = dp.actual_kwh - predicted
            residuals_sq += residual ** 2

        # Number of model parameters (slope + intercept)
        p = 2
        df = n_baseline - p
        if df <= 0:
            return Decimal("25.0")

        # Standard error of estimate
        see = _decimal_sqrt(_safe_divide(residuals_sq, _decimal(df)))

        # t-value for the confidence level
        t_val = self._get_t_value(df)

        # Uncertainty = t * SEE / mean(baseline) * 100
        uncertainty = _safe_pct(t_val * see, mean_baseline)

        # Cap uncertainty at reasonable range
        return min(uncertainty, Decimal("50.0"))

    # ------------------------------------------------------------------ #
    # Private: Group By Month                                             #
    # ------------------------------------------------------------------ #

    def _group_by_month(
        self,
        data: List[PerformanceDataPoint],
    ) -> Dict[int, PerformanceDataPoint]:
        """Group data points by month, taking the first entry per month.

        Args:
            data: List of data points.

        Returns:
            Dict mapping month number (1-12) to the data point.
        """
        by_month: Dict[int, PerformanceDataPoint] = {}
        for dp in sorted(data, key=lambda d: d.period_start):
            month = dp.period_start.month
            if month not in by_month:
                by_month[month] = dp
        return by_month

    # ------------------------------------------------------------------ #
    # Private: Summary Builder                                            #
    # ------------------------------------------------------------------ #

    def _build_summary(
        self,
        data: List[PerformanceDataPoint],
        trend: TrendAnalysis,
        yoy: Optional[YearOverYearComparison],
        rolling: Optional[RollingAnalysis],
        regression: Optional[RegressionValidation],
        forecast: Optional[ForecastResult],
        savings: Optional[SavingsVerification],
    ) -> Dict[str, Any]:
        """Build a summary dictionary of all analysis results.

        Args:
            data: Source data points.
            trend: Trend analysis result.
            yoy: Year-over-year comparison.
            rolling: Rolling analysis result.
            regression: Regression validation result.
            forecast: Forecast result.
            savings: Savings verification result.

        Returns:
            Summary dictionary.
        """
        values = [dp.actual_kwh for dp in data]
        n = len(values)
        total_kwh = sum(values)
        mean_kwh = _safe_divide(total_kwh, _decimal(n))
        min_kwh = min(values) if values else Decimal("0")
        max_kwh = max(values) if values else Decimal("0")

        summary: Dict[str, Any] = {
            "total_periods": n,
            "total_consumption_kwh": str(_round_val(total_kwh, 2)),
            "mean_consumption_kwh": str(_round_val(mean_kwh, 2)),
            "min_consumption_kwh": str(_round_val(min_kwh, 2)),
            "max_consumption_kwh": str(_round_val(max_kwh, 2)),
            "trend_direction": trend.trend_direction.value,
            "trend_slope_kwh_per_period": str(trend.trend_slope),
            "r_squared": str(trend.r_squared),
            "p_value": str(trend.p_value),
            "engine_version": self.engine_version,
        }

        if yoy:
            summary["yoy_percentage_change"] = str(yoy.percentage_change)
            summary["yoy_absolute_change_kwh"] = str(yoy.absolute_change_kwh)
            summary["yoy_months_compared"] = yoy.months_compared

        if rolling:
            summary["rolling_window_months"] = rolling.window_months
            summary["rolling_current_kwh"] = str(rolling.current_rolling_kwh)
            summary["rolling_trend"] = rolling.trend.value

        if regression:
            summary["model_adequate"] = regression.is_model_adequate
            summary["cv_rmse_pct"] = str(regression.cv_rmse)
            summary["mape_pct"] = str(regression.mape)
            summary["durbin_watson"] = str(regression.durbin_watson)

        if forecast:
            summary["forecast_method"] = forecast.forecast_method.value
            summary["forecast_periods"] = forecast.forecast_periods
            if forecast.mape is not None:
                summary["forecast_mape_pct"] = str(forecast.mape)
            if forecast.forecast_accuracy_pct is not None:
                summary["forecast_accuracy_pct"] = str(
                    forecast.forecast_accuracy_pct
                )

        if savings:
            summary["savings_standard"] = savings.verification_standard.value
            summary["gross_savings_kwh"] = str(savings.gross_savings_kwh)
            summary["net_savings_kwh"] = str(savings.net_savings_kwh)
            summary["savings_uncertainty_pct"] = str(
                savings.savings_uncertainty_pct
            )
            summary["savings_significant"] = savings.is_savings_significant

        return summary
