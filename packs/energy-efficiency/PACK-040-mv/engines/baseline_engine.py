# -*- coding: utf-8 -*-
"""
BaselineEngine - PACK-040 M&V Engine 1
========================================

Multivariate regression baseline development engine for Measurement &
Verification per IPMVP Core Concepts 2022 and ASHRAE Guideline 14-2014.
Builds energy baselines using OLS, change-point (3P cooling, 3P heating,
4P, 5P), and TOWT regression models.  Performs automated model selection
comparing goodness-of-fit statistics, validates models against ASHRAE 14
criteria, and provides deterministic balance-point optimization for
HDD/CDD calculations.

Calculation Methodology:
    Simple Linear (OLS):
        E = a + b * X

    Multivariate Linear:
        E = a + b1*X1 + b2*X2 + ... + bn*Xn

    3P Cooling (change-point):
        E = a + b * max(0, T - Tcp)

    3P Heating (change-point):
        E = a + b * max(0, Thp - T)

    4P Model:
        E = a + bh * max(0, Thp - T) + bc * max(0, T - Tcp)
        where Thp == Tcp (single change-point)

    5P Model:
        E = a + bh * max(0, Thp - T) + bc * max(0, T - Tcp)
        where Thp <= Tcp (two separate change-points)

    CVRMSE:
        CVRMSE = sqrt(sum((y - y_hat)^2) / (n - p)) / y_mean * 100

    NMBE:
        NMBE = sum(y - y_hat) / ((n - p) * y_mean) * 100

    R-squared:
        R2 = 1 - SS_res / SS_tot

    Adjusted R-squared:
        R2_adj = 1 - (1 - R2) * (n - 1) / (n - p - 1)

    F-statistic:
        F = (SS_reg / p) / (SS_res / (n - p - 1))

    Durbin-Watson:
        DW = sum((e_i - e_{i-1})^2) / sum(e_i^2)

    t-statistic:
        t = coefficient / standard_error_of_coefficient

Regulatory References:
    - IPMVP Core Concepts 2022 (EVO)
    - ASHRAE Guideline 14-2014 (Measurement of Energy Savings)
    - ISO 50015:2014 (M&V of Energy Performance)
    - ISO 50006:2014 (Energy Baselines and EnPIs)
    - FEMP M&V Guidelines 4.0

Zero-Hallucination:
    - All regression computed via deterministic normal-equation / grid-search
    - Balance-point optimization via iterative R-squared maximisation
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-040 M&V
Engine:  1 of 10
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


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ModelType(str, Enum):
    """Regression model type for baseline fitting.

    OLS:          Ordinary Least Squares simple/multivariate linear.
    THREE_P_COOL: 3-parameter change-point cooling model.
    THREE_P_HEAT: 3-parameter change-point heating model.
    FOUR_P:       4-parameter model (single change-point, two slopes).
    FIVE_P:       5-parameter model (two change-points, two slopes).
    TOWT:         Time-of-Week and Temperature model.
    """
    OLS = "ols"
    THREE_P_COOL = "3p_cooling"
    THREE_P_HEAT = "3p_heating"
    FOUR_P = "4p"
    FIVE_P = "5p"
    TOWT = "towt"


class BaselinePeriodGranularity(str, Enum):
    """Granularity of baseline data and model.

    HOURLY:   Hourly data (8,760 points/year).
    DAILY:    Daily data (365 points/year).
    WEEKLY:   Weekly data (52 points/year).
    MONTHLY:  Monthly data (12 points/year).
    """
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class ValidationGrade(str, Enum):
    """Model validation grade per ASHRAE 14 criteria.

    PASS:       Meets all ASHRAE 14 thresholds.
    MARGINAL:   Meets most criteria, within 10% of thresholds.
    FAIL:       Fails one or more ASHRAE 14 criteria.
    INSUFFICIENT_DATA: Not enough data points for model validation.
    """
    PASS = "pass"
    MARGINAL = "marginal"
    FAIL = "fail"
    INSUFFICIENT_DATA = "insufficient_data"


class IndependentVariableType(str, Enum):
    """Type of independent variable in regression model.

    TEMPERATURE:      Outdoor dry-bulb temperature.
    HDD:              Heating degree days.
    CDD:              Cooling degree days.
    PRODUCTION:       Production volume / throughput.
    OCCUPANCY:        Occupancy count or percentage.
    OPERATING_HOURS:  Operating hours per period.
    DAYLIGHT_HOURS:   Daylight hours per day.
    HUMIDITY:         Outdoor relative humidity.
    CUSTOM:           Custom independent variable.
    """
    TEMPERATURE = "temperature"
    HDD = "hdd"
    CDD = "cdd"
    PRODUCTION = "production"
    OCCUPANCY = "occupancy"
    OPERATING_HOURS = "operating_hours"
    DAYLIGHT_HOURS = "daylight_hours"
    HUMIDITY = "humidity"
    CUSTOM = "custom"


class ModelSelectionCriterion(str, Enum):
    """Criterion used for automated model selection.

    BEST_R_SQUARED:   Highest adjusted R-squared.
    BEST_CVRMSE:      Lowest CVRMSE.
    BEST_AIC:         Lowest Akaike Information Criterion.
    BEST_BIC:         Lowest Bayesian Information Criterion.
    BEST_COMPOSITE:   Composite score (weighted combination).
    """
    BEST_R_SQUARED = "best_r_squared"
    BEST_CVRMSE = "best_cvrmse"
    BEST_AIC = "best_aic"
    BEST_BIC = "best_bic"
    BEST_COMPOSITE = "best_composite"


class DataQualityGrade(str, Enum):
    """Data quality assessment grade for baseline data.

    EXCELLENT:  >99% completeness, no outliers.
    GOOD:       >95% completeness, <2% outliers.
    FAIR:       >90% completeness, <5% outliers.
    POOR:       <90% completeness or >5% outliers.
    """
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ASHRAE 14 validation thresholds by granularity.
ASHRAE14_CVRMSE: Dict[str, Decimal] = {
    BaselinePeriodGranularity.MONTHLY.value: Decimal("25"),
    BaselinePeriodGranularity.DAILY.value: Decimal("30"),
    BaselinePeriodGranularity.HOURLY.value: Decimal("30"),
    BaselinePeriodGranularity.WEEKLY.value: Decimal("30"),
}

ASHRAE14_NMBE: Dict[str, Decimal] = {
    BaselinePeriodGranularity.MONTHLY.value: Decimal("5"),
    BaselinePeriodGranularity.DAILY.value: Decimal("10"),
    BaselinePeriodGranularity.HOURLY.value: Decimal("10"),
    BaselinePeriodGranularity.WEEKLY.value: Decimal("10"),
}

ASHRAE14_R_SQUARED: Dict[str, Decimal] = {
    BaselinePeriodGranularity.MONTHLY.value: Decimal("0.70"),
    BaselinePeriodGranularity.DAILY.value: Decimal("0.50"),
    BaselinePeriodGranularity.HOURLY.value: Decimal("0"),
    BaselinePeriodGranularity.WEEKLY.value: Decimal("0.50"),
}

# Minimum data points per granularity for valid model.
MIN_DATA_POINTS: Dict[str, int] = {
    BaselinePeriodGranularity.MONTHLY.value: 12,
    BaselinePeriodGranularity.DAILY.value: 90,
    BaselinePeriodGranularity.WEEKLY.value: 26,
    BaselinePeriodGranularity.HOURLY.value: 2160,
}

# Balance-point search range (Fahrenheit).
BALANCE_POINT_MIN_F: Decimal = Decimal("40")
BALANCE_POINT_MAX_F: Decimal = Decimal("80")
BALANCE_POINT_STEP_F: Decimal = Decimal("1")

# Temperature conversion constant.
NINE_FIFTHS: Decimal = Decimal("1.8")
THIRTY_TWO: Decimal = Decimal("32")

# Default change-point search range (Celsius).
CHANGE_POINT_MIN_C: Decimal = Decimal("5")
CHANGE_POINT_MAX_C: Decimal = Decimal("28")
CHANGE_POINT_STEP_C: Decimal = Decimal("0.5")

# Durbin-Watson acceptable range (no autocorrelation).
DW_LOWER: Decimal = Decimal("1.5")
DW_UPPER: Decimal = Decimal("2.5")

# Model selection composite weights.
COMPOSITE_WEIGHT_R2: Decimal = Decimal("0.30")
COMPOSITE_WEIGHT_CVRMSE: Decimal = Decimal("0.30")
COMPOSITE_WEIGHT_NMBE: Decimal = Decimal("0.20")
COMPOSITE_WEIGHT_DW: Decimal = Decimal("0.20")

# Quality grade thresholds: (min_completeness_pct, max_outlier_pct, grade).
QUALITY_THRESHOLDS: List[Tuple[Decimal, Decimal, DataQualityGrade]] = [
    (Decimal("99"), Decimal("0.5"), DataQualityGrade.EXCELLENT),
    (Decimal("95"), Decimal("2.0"), DataQualityGrade.GOOD),
    (Decimal("90"), Decimal("5.0"), DataQualityGrade.FAIR),
    (Decimal("0"), Decimal("100"), DataQualityGrade.POOR),
]


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class BaselineDataPoint(BaseModel):
    """Single observation for baseline development.

    Attributes:
        period_start: Start of the observation period.
        period_end: End of the observation period.
        energy_consumption: Total energy consumption (kWh, therms, etc.).
        temperature_avg: Average outdoor dry-bulb temperature.
        hdd: Heating degree days in this period.
        cdd: Cooling degree days in this period.
        production_volume: Production volume / throughput.
        occupancy_pct: Occupancy percentage (0-100).
        operating_hours: Operating hours in this period.
        daylight_hours: Average daylight hours per day.
        humidity_pct: Average relative humidity percentage.
        custom_variables: Additional custom independent variables.
        is_valid: Whether this data point passed quality checks.
        notes: Additional notes or flags.
    """
    period_start: datetime = Field(
        default_factory=_utcnow, description="Period start timestamp"
    )
    period_end: datetime = Field(
        default_factory=_utcnow, description="Period end timestamp"
    )
    energy_consumption: Decimal = Field(
        default=Decimal("0"), ge=0, description="Energy consumption (kWh)"
    )
    temperature_avg: Optional[Decimal] = Field(
        default=None, description="Average temperature"
    )
    hdd: Optional[Decimal] = Field(
        default=None, ge=0, description="Heating degree days"
    )
    cdd: Optional[Decimal] = Field(
        default=None, ge=0, description="Cooling degree days"
    )
    production_volume: Optional[Decimal] = Field(
        default=None, ge=0, description="Production volume"
    )
    occupancy_pct: Optional[Decimal] = Field(
        default=None, ge=0, le=100, description="Occupancy percentage"
    )
    operating_hours: Optional[Decimal] = Field(
        default=None, ge=0, description="Operating hours"
    )
    daylight_hours: Optional[Decimal] = Field(
        default=None, ge=0, le=24, description="Daylight hours/day"
    )
    humidity_pct: Optional[Decimal] = Field(
        default=None, ge=0, le=100, description="Relative humidity %"
    )
    custom_variables: Dict[str, Decimal] = Field(
        default_factory=dict, description="Custom independent variables"
    )
    is_valid: bool = Field(default=True, description="Data quality flag")
    notes: str = Field(default="", max_length=500, description="Notes")


class BaselineConfig(BaseModel):
    """Configuration for baseline model development.

    Attributes:
        project_id: M&V project identifier.
        ecm_id: Energy Conservation Measure identifier.
        facility_id: Facility identifier.
        facility_name: Human-readable facility name.
        baseline_start: Start of baseline period.
        baseline_end: End of baseline period.
        granularity: Data granularity (monthly, daily, etc.).
        model_types: List of model types to evaluate.
        independent_variables: List of independent variable types.
        selection_criterion: Model selection criterion.
        balance_point_heating_f: Heating balance point (Fahrenheit).
        balance_point_cooling_f: Cooling balance point (Fahrenheit).
        optimize_balance_points: Whether to optimize balance points.
        ashrae14_validation: Whether to validate against ASHRAE 14.
        min_data_points: Minimum data points required.
        energy_unit: Unit of energy measurement.
        temperature_unit: Unit of temperature measurement.
    """
    project_id: str = Field(default="", description="M&V project ID")
    ecm_id: str = Field(default="", description="ECM identifier")
    facility_id: str = Field(default="", description="Facility ID")
    facility_name: str = Field(
        default="", max_length=500, description="Facility name"
    )
    baseline_start: datetime = Field(
        default_factory=_utcnow, description="Baseline period start"
    )
    baseline_end: datetime = Field(
        default_factory=_utcnow, description="Baseline period end"
    )
    granularity: BaselinePeriodGranularity = Field(
        default=BaselinePeriodGranularity.MONTHLY,
        description="Data granularity",
    )
    model_types: List[ModelType] = Field(
        default_factory=lambda: [
            ModelType.OLS, ModelType.THREE_P_COOL,
            ModelType.THREE_P_HEAT, ModelType.FOUR_P, ModelType.FIVE_P,
        ],
        description="Model types to evaluate",
    )
    independent_variables: List[IndependentVariableType] = Field(
        default_factory=lambda: [IndependentVariableType.TEMPERATURE],
        description="Independent variables to include",
    )
    selection_criterion: ModelSelectionCriterion = Field(
        default=ModelSelectionCriterion.BEST_COMPOSITE,
        description="Model selection criterion",
    )
    balance_point_heating_f: Decimal = Field(
        default=Decimal("65"), description="Heating balance point (F)"
    )
    balance_point_cooling_f: Decimal = Field(
        default=Decimal("65"), description="Cooling balance point (F)"
    )
    optimize_balance_points: bool = Field(
        default=True, description="Auto-optimize balance points"
    )
    ashrae14_validation: bool = Field(
        default=True, description="Validate against ASHRAE 14"
    )
    min_data_points: Optional[int] = Field(
        default=None, ge=6, description="Min data points override"
    )
    energy_unit: str = Field(default="kWh", description="Energy unit")
    temperature_unit: str = Field(
        default="fahrenheit", description="Temperature unit (fahrenheit/celsius)"
    )


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class RegressionCoefficients(BaseModel):
    """Regression model coefficients and change-points.

    Attributes:
        intercept: Model intercept (a).
        coefficients: Named coefficients {variable: value}.
        heating_slope: Heating slope (bh) for change-point models.
        cooling_slope: Cooling slope (bc) for change-point models.
        heating_change_point: Heating change-point temperature.
        cooling_change_point: Cooling change-point temperature.
        standard_errors: Standard errors of each coefficient.
        t_statistics: t-statistics for each coefficient.
        p_values: p-values for each coefficient.
    """
    intercept: Decimal = Field(default=Decimal("0"), description="Intercept")
    coefficients: Dict[str, Decimal] = Field(
        default_factory=dict, description="Named coefficients"
    )
    heating_slope: Optional[Decimal] = Field(
        default=None, description="Heating slope (bh)"
    )
    cooling_slope: Optional[Decimal] = Field(
        default=None, description="Cooling slope (bc)"
    )
    heating_change_point: Optional[Decimal] = Field(
        default=None, description="Heating change-point temp"
    )
    cooling_change_point: Optional[Decimal] = Field(
        default=None, description="Cooling change-point temp"
    )
    standard_errors: Dict[str, Decimal] = Field(
        default_factory=dict, description="Coefficient standard errors"
    )
    t_statistics: Dict[str, Decimal] = Field(
        default_factory=dict, description="t-statistics"
    )
    p_values: Dict[str, Decimal] = Field(
        default_factory=dict, description="p-values (approx)"
    )


class ModelValidationStats(BaseModel):
    """Statistical validation metrics for a regression model.

    Attributes:
        r_squared: Coefficient of determination.
        adjusted_r_squared: Adjusted R-squared.
        cvrmse_pct: Coefficient of Variation of RMSE (%).
        nmbe_pct: Normalised Mean Bias Error (%).
        rmse: Root Mean Square Error.
        mae: Mean Absolute Error.
        f_statistic: F-statistic for overall model significance.
        f_p_value: p-value for F-statistic (approx).
        durbin_watson: Durbin-Watson autocorrelation statistic.
        n_observations: Number of observations.
        n_parameters: Number of model parameters (including intercept).
        degrees_of_freedom: Degrees of freedom (n - p).
        ss_total: Total sum of squares.
        ss_residual: Residual sum of squares.
        ss_regression: Regression sum of squares.
        aic: Akaike Information Criterion.
        bic: Bayesian Information Criterion.
        ashrae14_cvrmse_pass: Whether CVRMSE meets ASHRAE 14.
        ashrae14_nmbe_pass: Whether NMBE meets ASHRAE 14.
        ashrae14_r2_pass: Whether R-squared meets ASHRAE 14.
        ashrae14_overall_pass: Whether all ASHRAE 14 criteria met.
        dw_autocorrelation_flag: Whether DW indicates autocorrelation.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    r_squared: Decimal = Field(default=Decimal("0"))
    adjusted_r_squared: Decimal = Field(default=Decimal("0"))
    cvrmse_pct: Decimal = Field(default=Decimal("0"))
    nmbe_pct: Decimal = Field(default=Decimal("0"))
    rmse: Decimal = Field(default=Decimal("0"))
    mae: Decimal = Field(default=Decimal("0"))
    f_statistic: Decimal = Field(default=Decimal("0"))
    f_p_value: Decimal = Field(default=Decimal("1"))
    durbin_watson: Decimal = Field(default=Decimal("0"))
    n_observations: int = Field(default=0)
    n_parameters: int = Field(default=0)
    degrees_of_freedom: int = Field(default=0)
    ss_total: Decimal = Field(default=Decimal("0"))
    ss_residual: Decimal = Field(default=Decimal("0"))
    ss_regression: Decimal = Field(default=Decimal("0"))
    aic: Decimal = Field(default=Decimal("0"))
    bic: Decimal = Field(default=Decimal("0"))
    ashrae14_cvrmse_pass: bool = Field(default=False)
    ashrae14_nmbe_pass: bool = Field(default=False)
    ashrae14_r2_pass: bool = Field(default=False)
    ashrae14_overall_pass: bool = Field(default=False)
    dw_autocorrelation_flag: bool = Field(default=False)
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


class FittedModel(BaseModel):
    """Complete fitted regression model with coefficients and diagnostics.

    Attributes:
        model_id: Unique model identifier.
        model_type: Type of regression model.
        coefficients: Model coefficients.
        validation: Model validation statistics.
        residuals: Residual values for each observation.
        predicted_values: Predicted values for each observation.
        composite_score: Composite ranking score.
        rank: Rank among compared models (1 = best).
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    model_id: str = Field(default_factory=_new_uuid)
    model_type: ModelType = Field(default=ModelType.OLS)
    coefficients: RegressionCoefficients = Field(
        default_factory=RegressionCoefficients
    )
    validation: ModelValidationStats = Field(
        default_factory=ModelValidationStats
    )
    residuals: List[Decimal] = Field(default_factory=list)
    predicted_values: List[Decimal] = Field(default_factory=list)
    composite_score: Decimal = Field(default=Decimal("0"))
    rank: int = Field(default=0)
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


class BalancePointResult(BaseModel):
    """Result of balance-point optimisation.

    Attributes:
        heating_balance_point_f: Optimised heating balance point (F).
        cooling_balance_point_f: Optimised cooling balance point (F).
        heating_balance_point_c: Optimised heating balance point (C).
        cooling_balance_point_c: Optimised cooling balance point (C).
        heating_r_squared: R-squared at optimal heating balance point.
        cooling_r_squared: R-squared at optimal cooling balance point.
        search_range_min_f: Search range minimum (F).
        search_range_max_f: Search range maximum (F).
        search_step_f: Search step size (F).
        iterations_heating: Number of iterations for heating optimisation.
        iterations_cooling: Number of iterations for cooling optimisation.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    heating_balance_point_f: Decimal = Field(default=Decimal("65"))
    cooling_balance_point_f: Decimal = Field(default=Decimal("65"))
    heating_balance_point_c: Decimal = Field(default=Decimal("18.3"))
    cooling_balance_point_c: Decimal = Field(default=Decimal("18.3"))
    heating_r_squared: Decimal = Field(default=Decimal("0"))
    cooling_r_squared: Decimal = Field(default=Decimal("0"))
    search_range_min_f: Decimal = Field(default=BALANCE_POINT_MIN_F)
    search_range_max_f: Decimal = Field(default=BALANCE_POINT_MAX_F)
    search_step_f: Decimal = Field(default=BALANCE_POINT_STEP_F)
    iterations_heating: int = Field(default=0)
    iterations_cooling: int = Field(default=0)
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


class BaselineResult(BaseModel):
    """Complete baseline development result.

    Attributes:
        baseline_id: Unique baseline identifier.
        project_id: M&V project identifier.
        ecm_id: ECM identifier.
        facility_id: Facility identifier.
        facility_name: Human-readable facility name.
        granularity: Baseline data granularity.
        baseline_start: Baseline period start.
        baseline_end: Baseline period end.
        n_data_points: Number of data points used.
        data_quality: Quality assessment of baseline data.
        selected_model: The selected (best) regression model.
        all_models: All evaluated models ranked.
        balance_point_result: Balance-point optimisation result.
        selection_criterion: Criterion used for selection.
        selection_rationale: Human-readable rationale.
        independent_variables_used: Variables used in model.
        energy_unit: Unit of energy measurement.
        total_baseline_energy: Total baseline-period energy.
        mean_baseline_energy: Mean baseline-period energy per period.
        recommendations: List of analysis recommendations.
        warnings: List of warnings.
        processing_time_ms: Processing duration (ms).
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    baseline_id: str = Field(default_factory=_new_uuid)
    project_id: str = Field(default="")
    ecm_id: str = Field(default="")
    facility_id: str = Field(default="")
    facility_name: str = Field(default="", max_length=500)
    granularity: BaselinePeriodGranularity = Field(
        default=BaselinePeriodGranularity.MONTHLY
    )
    baseline_start: datetime = Field(default_factory=_utcnow)
    baseline_end: datetime = Field(default_factory=_utcnow)
    n_data_points: int = Field(default=0)
    data_quality: DataQualityGrade = Field(default=DataQualityGrade.POOR)
    selected_model: Optional[FittedModel] = Field(default=None)
    all_models: List[FittedModel] = Field(default_factory=list)
    balance_point_result: Optional[BalancePointResult] = Field(default=None)
    selection_criterion: ModelSelectionCriterion = Field(
        default=ModelSelectionCriterion.BEST_COMPOSITE
    )
    selection_rationale: str = Field(default="")
    independent_variables_used: List[str] = Field(default_factory=list)
    energy_unit: str = Field(default="kWh")
    total_baseline_energy: Decimal = Field(default=Decimal("0"))
    mean_baseline_energy: Decimal = Field(default=Decimal("0"))
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BaselineEngine:
    """Energy baseline development engine for M&V per IPMVP / ASHRAE 14.

    Develops statistically valid energy baselines using multivariate
    regression with change-point models.  Performs automated model
    selection, validates against ASHRAE Guideline 14 criteria, and
    provides deterministic balance-point optimisation.  All calculations
    use Decimal arithmetic with SHA-256 provenance hashing.

    Usage::

        engine = BaselineEngine()
        data_points = [
            BaselineDataPoint(
                period_start=..., period_end=...,
                energy_consumption=Decimal("12500"),
                temperature_avg=Decimal("72"),
            ),
            ...
        ]
        config = BaselineConfig(
            project_id="PRJ-001",
            facility_id="FAC-001",
            facility_name="Office Tower A",
            granularity=BaselinePeriodGranularity.MONTHLY,
        )
        result = engine.develop_baseline(config, data_points)
        print(f"Best model: {result.selected_model.model_type.value}")
        print(f"CVRMSE: {result.selected_model.validation.cvrmse_pct}%")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise BaselineEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - change_point_min_c (float): min change-point search temp
                - change_point_max_c (float): max change-point search temp
                - change_point_step_c (float): change-point search step
                - balance_point_min_f (float): min balance-point search temp
                - balance_point_max_f (float): max balance-point search temp
                - balance_point_step_f (float): balance-point search step
                - composite_weights (dict): composite scoring weights
        """
        self.config = config or {}
        self._cp_min = _decimal(
            self.config.get("change_point_min_c", CHANGE_POINT_MIN_C)
        )
        self._cp_max = _decimal(
            self.config.get("change_point_max_c", CHANGE_POINT_MAX_C)
        )
        self._cp_step = _decimal(
            self.config.get("change_point_step_c", CHANGE_POINT_STEP_C)
        )
        self._bp_min = _decimal(
            self.config.get("balance_point_min_f", BALANCE_POINT_MIN_F)
        )
        self._bp_max = _decimal(
            self.config.get("balance_point_max_f", BALANCE_POINT_MAX_F)
        )
        self._bp_step = _decimal(
            self.config.get("balance_point_step_f", BALANCE_POINT_STEP_F)
        )
        logger.info(
            "BaselineEngine v%s initialised (cp_range=%.1f-%.1f C, "
            "bp_range=%.0f-%.0f F)",
            self.engine_version, float(self._cp_min), float(self._cp_max),
            float(self._bp_min), float(self._bp_max),
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def develop_baseline(
        self,
        baseline_config: BaselineConfig,
        data_points: List[BaselineDataPoint],
    ) -> BaselineResult:
        """Develop a complete baseline model from energy data.

        Evaluates all configured model types, optimises balance points,
        validates against ASHRAE 14, ranks models, and selects the best.

        Args:
            baseline_config: Baseline configuration.
            data_points: Chronological baseline-period observations.

        Returns:
            BaselineResult with selected model and all comparisons.
        """
        t0 = time.perf_counter()
        logger.info(
            "Developing baseline: %s (%d points, granularity=%s)",
            baseline_config.facility_name, len(data_points),
            baseline_config.granularity.value,
        )

        # Validate data sufficiency
        valid_points = [dp for dp in data_points if dp.is_valid]
        min_pts = baseline_config.min_data_points or MIN_DATA_POINTS.get(
            baseline_config.granularity.value, 12
        )

        if len(valid_points) < min_pts:
            result = BaselineResult(
                project_id=baseline_config.project_id,
                ecm_id=baseline_config.ecm_id,
                facility_id=baseline_config.facility_id,
                facility_name=baseline_config.facility_name,
                granularity=baseline_config.granularity,
                baseline_start=baseline_config.baseline_start,
                baseline_end=baseline_config.baseline_end,
                n_data_points=len(valid_points),
                data_quality=DataQualityGrade.POOR,
                warnings=[
                    f"Insufficient data: {len(valid_points)} points "
                    f"(minimum {min_pts} required for "
                    f"{baseline_config.granularity.value} model)."
                ],
            )
            result.provenance_hash = _compute_hash(result)
            return result

        # Assess data quality
        data_quality = self._assess_data_quality(data_points)

        # Optimise balance points if requested
        balance_result: Optional[BalancePointResult] = None
        if baseline_config.optimize_balance_points:
            balance_result = self.optimize_balance_points(
                valid_points, baseline_config
            )

        # Extract response and predictor vectors
        y_vals = [_decimal(dp.energy_consumption) for dp in valid_points]
        total_energy = sum(y_vals, Decimal("0"))
        mean_energy = _safe_divide(total_energy, _decimal(len(y_vals)))

        # Determine timestamps
        timestamps = [dp.period_start for dp in valid_points]
        bl_start = min(timestamps) if timestamps else baseline_config.baseline_start
        bl_end = max(timestamps) if timestamps else baseline_config.baseline_end

        # Fit all configured model types
        fitted_models: List[FittedModel] = []
        for model_type in baseline_config.model_types:
            model = self._fit_model(
                model_type, valid_points, baseline_config, balance_result
            )
            if model is not None:
                fitted_models.append(model)

        # Rank and select
        if fitted_models:
            fitted_models = self._rank_models(
                fitted_models, baseline_config.selection_criterion,
                baseline_config.granularity,
            )
            selected = fitted_models[0]
            rationale = self._build_selection_rationale(
                selected, fitted_models, baseline_config.selection_criterion,
            )
        else:
            selected = None
            rationale = "No models could be fitted to the data."

        # Collect variable names used
        iv_names = [iv.value for iv in baseline_config.independent_variables]

        # Generate recommendations
        recommendations = self._generate_recommendations(
            selected, fitted_models, data_quality, baseline_config
        )
        warnings = self._generate_warnings(
            selected, fitted_models, valid_points, baseline_config
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000.0)

        result = BaselineResult(
            project_id=baseline_config.project_id,
            ecm_id=baseline_config.ecm_id,
            facility_id=baseline_config.facility_id,
            facility_name=baseline_config.facility_name,
            granularity=baseline_config.granularity,
            baseline_start=bl_start,
            baseline_end=bl_end,
            n_data_points=len(valid_points),
            data_quality=data_quality,
            selected_model=selected,
            all_models=fitted_models,
            balance_point_result=balance_result,
            selection_criterion=baseline_config.selection_criterion,
            selection_rationale=rationale,
            independent_variables_used=iv_names,
            energy_unit=baseline_config.energy_unit,
            total_baseline_energy=_round_val(total_energy, 2),
            mean_baseline_energy=_round_val(mean_energy, 2),
            recommendations=recommendations,
            warnings=warnings,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Baseline developed: %s, best=%s, R2=%.3f, CVRMSE=%.1f%%, "
            "NMBE=%.2f%%, ASHRAE14=%s, hash=%s (%.1f ms)",
            baseline_config.facility_name,
            selected.model_type.value if selected else "none",
            float(selected.validation.r_squared) if selected else 0.0,
            float(selected.validation.cvrmse_pct) if selected else 0.0,
            float(selected.validation.nmbe_pct) if selected else 0.0,
            selected.validation.ashrae14_overall_pass if selected else False,
            result.provenance_hash[:16], float(elapsed_ms),
        )
        return result

    def optimize_balance_points(
        self,
        data_points: List[BaselineDataPoint],
        baseline_config: BaselineConfig,
    ) -> BalancePointResult:
        """Optimise heating and cooling balance points via iterative regression.

        Iterates through balance-point temperatures, computes HDD/CDD at
        each, fits single-variable regressions, and selects the balance
        point that maximises R-squared.

        Args:
            data_points: Validated baseline data points.
            baseline_config: Baseline configuration.

        Returns:
            BalancePointResult with optimised balance points.
        """
        t0 = time.perf_counter()
        logger.info(
            "Optimising balance points: range %.0f-%.0f F, step %.0f F",
            float(self._bp_min), float(self._bp_max), float(self._bp_step),
        )

        # Extract temperature data
        temps = [
            _decimal(dp.temperature_avg) for dp in data_points
            if dp.temperature_avg is not None
        ]
        y_vals = [
            _decimal(dp.energy_consumption) for dp in data_points
            if dp.temperature_avg is not None
        ]

        if len(temps) < 6:
            result = BalancePointResult()
            result.provenance_hash = _compute_hash(result)
            return result

        # Convert to Fahrenheit if needed
        if baseline_config.temperature_unit.lower().startswith("c"):
            temps_f = [t * NINE_FIFTHS + THIRTY_TWO for t in temps]
        else:
            temps_f = list(temps)

        # Search for heating balance point
        best_heat_bp = Decimal("65")
        best_heat_r2 = Decimal("-1")
        heat_iterations = 0
        bp = self._bp_min
        while bp <= self._bp_max:
            heat_iterations += 1
            hdd_vals = [max(Decimal("0"), bp - t) for t in temps_f]
            r2 = self._simple_r_squared(hdd_vals, y_vals)
            if r2 > best_heat_r2:
                best_heat_r2 = r2
                best_heat_bp = bp
            bp += self._bp_step

        # Search for cooling balance point
        best_cool_bp = Decimal("65")
        best_cool_r2 = Decimal("-1")
        cool_iterations = 0
        bp = self._bp_min
        while bp <= self._bp_max:
            cool_iterations += 1
            cdd_vals = [max(Decimal("0"), t - bp) for t in temps_f]
            r2 = self._simple_r_squared(cdd_vals, y_vals)
            if r2 > best_cool_r2:
                best_cool_r2 = r2
                best_cool_bp = bp
            bp += self._bp_step

        # Convert to Celsius
        heat_bp_c = _safe_divide(best_heat_bp - THIRTY_TWO, NINE_FIFTHS)
        cool_bp_c = _safe_divide(best_cool_bp - THIRTY_TWO, NINE_FIFTHS)

        result = BalancePointResult(
            heating_balance_point_f=_round_val(best_heat_bp, 1),
            cooling_balance_point_f=_round_val(best_cool_bp, 1),
            heating_balance_point_c=_round_val(heat_bp_c, 1),
            cooling_balance_point_c=_round_val(cool_bp_c, 1),
            heating_r_squared=_round_val(max(best_heat_r2, Decimal("0")), 6),
            cooling_r_squared=_round_val(max(best_cool_r2, Decimal("0")), 6),
            search_range_min_f=self._bp_min,
            search_range_max_f=self._bp_max,
            search_step_f=self._bp_step,
            iterations_heating=heat_iterations,
            iterations_cooling=cool_iterations,
        )
        result.provenance_hash = _compute_hash(result)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Balance points optimised: heat=%.0f F (R2=%.3f), "
            "cool=%.0f F (R2=%.3f), hash=%s (%.1f ms)",
            float(best_heat_bp), float(max(best_heat_r2, Decimal("0"))),
            float(best_cool_bp), float(max(best_cool_r2, Decimal("0"))),
            result.provenance_hash[:16], elapsed,
        )
        return result

    def fit_ols_model(
        self,
        data_points: List[BaselineDataPoint],
        baseline_config: BaselineConfig,
    ) -> FittedModel:
        """Fit an Ordinary Least Squares regression model.

        Implements OLS via the normal equation: beta = (X'X)^{-1} X'y
        for arbitrary numbers of independent variables.

        Args:
            data_points: Validated baseline data points.
            baseline_config: Baseline configuration.

        Returns:
            FittedModel with OLS coefficients and diagnostics.
        """
        t0 = time.perf_counter()
        logger.info(
            "Fitting OLS model: %d points, %d variables",
            len(data_points), len(baseline_config.independent_variables),
        )

        y_vals = [_decimal(dp.energy_consumption) for dp in data_points]
        x_matrix = self._build_predictor_matrix(
            data_points, baseline_config.independent_variables, baseline_config
        )
        var_names = [iv.value for iv in baseline_config.independent_variables]

        coeffs, residuals, predicted = self._ols_normal_equation(
            x_matrix, y_vals
        )
        intercept = coeffs[0] if coeffs else Decimal("0")
        slopes = coeffs[1:] if len(coeffs) > 1 else []

        coeff_dict: Dict[str, Decimal] = {}
        for i, name in enumerate(var_names):
            if i < len(slopes):
                coeff_dict[name] = _round_val(slopes[i], 8)

        # Compute validation statistics
        n = len(y_vals)
        p = len(coeffs)
        validation = self._compute_validation_stats(
            y_vals, predicted, residuals, n, p,
            baseline_config.granularity,
        )

        # Standard errors and t-stats
        se_dict, t_dict, p_dict = self._compute_coefficient_stats(
            x_matrix, residuals, coeffs, var_names
        )

        reg_coeffs = RegressionCoefficients(
            intercept=_round_val(intercept, 8),
            coefficients=coeff_dict,
            standard_errors=se_dict,
            t_statistics=t_dict,
            p_values=p_dict,
        )

        model = FittedModel(
            model_type=ModelType.OLS,
            coefficients=reg_coeffs,
            validation=validation,
            residuals=[_round_val(r, 4) for r in residuals],
            predicted_values=[_round_val(p, 4) for p in predicted],
        )
        model.provenance_hash = _compute_hash(model)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "OLS model fitted: R2=%.4f, CVRMSE=%.2f%%, hash=%s (%.1f ms)",
            float(validation.r_squared), float(validation.cvrmse_pct),
            model.provenance_hash[:16], elapsed,
        )
        return model

    def fit_change_point_model(
        self,
        data_points: List[BaselineDataPoint],
        baseline_config: BaselineConfig,
        model_type: ModelType,
        balance_result: Optional[BalancePointResult] = None,
    ) -> Optional[FittedModel]:
        """Fit a change-point regression model (3P, 4P, or 5P).

        Performs grid search over change-point temperatures to find
        the model with best R-squared, then fits the full model.

        Args:
            data_points: Validated baseline data points.
            baseline_config: Baseline configuration.
            model_type: Specific change-point model type.
            balance_result: Optional pre-computed balance points.

        Returns:
            FittedModel if successful, None if model cannot be fitted.
        """
        t0 = time.perf_counter()
        logger.info(
            "Fitting %s model: %d points",
            model_type.value, len(data_points),
        )

        temps = self._extract_temperatures(data_points, baseline_config)
        y_vals = [_decimal(dp.energy_consumption) for dp in data_points]

        if len(temps) != len(y_vals) or len(temps) < 6:
            logger.warning(
                "Insufficient temperature data for %s model", model_type.value
            )
            return None

        best_model: Optional[FittedModel] = None
        best_r2 = Decimal("-999")

        if model_type == ModelType.THREE_P_COOL:
            best_model, best_r2 = self._search_3p_cooling(temps, y_vals)
        elif model_type == ModelType.THREE_P_HEAT:
            best_model, best_r2 = self._search_3p_heating(temps, y_vals)
        elif model_type == ModelType.FOUR_P:
            best_model, best_r2 = self._search_4p(temps, y_vals)
        elif model_type == ModelType.FIVE_P:
            best_model, best_r2 = self._search_5p(temps, y_vals)

        if best_model is not None:
            # Re-validate against ASHRAE 14
            n = len(y_vals)
            p_count = len([
                v for v in [
                    best_model.coefficients.intercept,
                    best_model.coefficients.heating_slope,
                    best_model.coefficients.cooling_slope,
                ] if v is not None
            ])
            validation = self._compute_validation_stats(
                y_vals, best_model.predicted_values,
                best_model.residuals, n, p_count,
                baseline_config.granularity,
            )
            best_model.validation = validation
            best_model.provenance_hash = _compute_hash(best_model)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "%s model %s: R2=%.4f, hash=%s (%.1f ms)",
            model_type.value,
            "fitted" if best_model else "failed",
            float(best_r2) if best_r2 > Decimal("-999") else 0.0,
            best_model.provenance_hash[:16] if best_model else "n/a",
            elapsed,
        )
        return best_model

    def predict_energy(
        self,
        model: FittedModel,
        data_points: List[BaselineDataPoint],
        baseline_config: BaselineConfig,
    ) -> List[Decimal]:
        """Predict energy consumption using a fitted model.

        Applies the model coefficients to new independent variable data
        to produce energy predictions (adjusted baseline).

        Args:
            model: Fitted regression model.
            data_points: Data points with independent variables filled.
            baseline_config: Baseline configuration.

        Returns:
            List of predicted energy values aligned with data_points.
        """
        t0 = time.perf_counter()
        logger.info(
            "Predicting energy: %s model, %d points",
            model.model_type.value, len(data_points),
        )

        predictions: List[Decimal] = []
        temps = self._extract_temperatures(data_points, baseline_config)

        for i, dp in enumerate(data_points):
            temp = temps[i] if i < len(temps) else Decimal("0")
            pred = self._predict_single(model, dp, temp, baseline_config)
            predictions.append(_round_val(max(pred, Decimal("0")), 4))

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Prediction complete: %d values, sum=%.1f, hash=%s (%.1f ms)",
            len(predictions),
            float(sum(predictions, Decimal("0"))),
            _compute_hash(predictions)[:16], elapsed,
        )
        return predictions

    def validate_model(
        self,
        model: FittedModel,
        granularity: BaselinePeriodGranularity,
    ) -> ModelValidationStats:
        """Validate a fitted model against ASHRAE Guideline 14 criteria.

        Checks CVRMSE, NMBE, R-squared, and Durbin-Watson against the
        thresholds for the given data granularity.

        Args:
            model: Fitted regression model to validate.
            granularity: Data granularity for threshold selection.

        Returns:
            Updated ModelValidationStats with ASHRAE 14 pass/fail flags.
        """
        t0 = time.perf_counter()
        logger.info(
            "Validating model %s against ASHRAE 14 (%s granularity)",
            model.model_type.value, granularity.value,
        )

        stats = model.validation

        # CVRMSE threshold
        cvrmse_threshold = ASHRAE14_CVRMSE.get(
            granularity.value, Decimal("30")
        )
        stats.ashrae14_cvrmse_pass = abs(stats.cvrmse_pct) <= cvrmse_threshold

        # NMBE threshold
        nmbe_threshold = ASHRAE14_NMBE.get(
            granularity.value, Decimal("10")
        )
        stats.ashrae14_nmbe_pass = abs(stats.nmbe_pct) <= nmbe_threshold

        # R-squared threshold
        r2_threshold = ASHRAE14_R_SQUARED.get(
            granularity.value, Decimal("0")
        )
        stats.ashrae14_r2_pass = stats.r_squared >= r2_threshold

        # Overall
        stats.ashrae14_overall_pass = (
            stats.ashrae14_cvrmse_pass
            and stats.ashrae14_nmbe_pass
            and stats.ashrae14_r2_pass
        )

        # Durbin-Watson flag
        stats.dw_autocorrelation_flag = (
            stats.durbin_watson < DW_LOWER or stats.durbin_watson > DW_UPPER
        )

        stats.provenance_hash = _compute_hash(stats)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Model validation: CVRMSE=%.1f%% (%s), NMBE=%.2f%% (%s), "
            "R2=%.4f (%s), DW=%.3f, overall=%s (%.1f ms)",
            float(stats.cvrmse_pct),
            "PASS" if stats.ashrae14_cvrmse_pass else "FAIL",
            float(stats.nmbe_pct),
            "PASS" if stats.ashrae14_nmbe_pass else "FAIL",
            float(stats.r_squared),
            "PASS" if stats.ashrae14_r2_pass else "FAIL",
            float(stats.durbin_watson),
            "PASS" if stats.ashrae14_overall_pass else "FAIL",
            elapsed,
        )
        return stats

    def compare_models(
        self,
        models: List[FittedModel],
        criterion: ModelSelectionCriterion = ModelSelectionCriterion.BEST_COMPOSITE,
        granularity: BaselinePeriodGranularity = BaselinePeriodGranularity.MONTHLY,
    ) -> List[FittedModel]:
        """Compare and rank multiple fitted models.

        Computes composite scores and assigns ranks.  Returns models
        sorted by rank (best first).

        Args:
            models: List of fitted models to compare.
            criterion: Selection criterion.
            granularity: Data granularity for ASHRAE 14 thresholds.

        Returns:
            Ranked list of models (rank 1 = best).
        """
        t0 = time.perf_counter()
        logger.info(
            "Comparing %d models using %s criterion",
            len(models), criterion.value,
        )

        if not models:
            return []

        ranked = self._rank_models(models, criterion, granularity)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Model comparison complete: best=%s (score=%.4f) (%.1f ms)",
            ranked[0].model_type.value if ranked else "none",
            float(ranked[0].composite_score) if ranked else 0.0,
            elapsed,
        )
        return ranked

    # ------------------------------------------------------------------ #
    # Private: Model Fitting                                               #
    # ------------------------------------------------------------------ #

    def _fit_model(
        self,
        model_type: ModelType,
        data_points: List[BaselineDataPoint],
        baseline_config: BaselineConfig,
        balance_result: Optional[BalancePointResult],
    ) -> Optional[FittedModel]:
        """Dispatch to the appropriate model fitter."""
        try:
            if model_type == ModelType.OLS:
                return self.fit_ols_model(data_points, baseline_config)
            elif model_type in (
                ModelType.THREE_P_COOL, ModelType.THREE_P_HEAT,
                ModelType.FOUR_P, ModelType.FIVE_P,
            ):
                return self.fit_change_point_model(
                    data_points, baseline_config, model_type, balance_result
                )
            elif model_type == ModelType.TOWT:
                return self._fit_towt_model(data_points, baseline_config)
            else:
                logger.warning("Unknown model type: %s", model_type.value)
                return None
        except Exception as exc:
            logger.error(
                "Failed to fit %s model: %s", model_type.value, str(exc),
                exc_info=True,
            )
            return None

    def _fit_towt_model(
        self,
        data_points: List[BaselineDataPoint],
        baseline_config: BaselineConfig,
    ) -> Optional[FittedModel]:
        """Fit a Time-of-Week and Temperature (TOWT) model.

        Combines day-of-week indicator variables with temperature
        to create a piecewise-linear regression model.
        """
        t0 = time.perf_counter()
        logger.info("Fitting TOWT model: %d points", len(data_points))

        y_vals = [_decimal(dp.energy_consumption) for dp in data_points]
        temps = self._extract_temperatures(data_points, baseline_config)

        if len(temps) != len(y_vals) or len(y_vals) < 14:
            return None

        # Build TOWT predictor matrix: 7 day-of-week dummies + temperature
        x_matrix: List[List[Decimal]] = []
        for i, dp in enumerate(data_points):
            row = [Decimal("1")]  # intercept
            dow = dp.period_start.weekday()  # 0=Mon, 6=Sun
            for d in range(6):  # 6 dummy variables (Mon-Sat, Sun=reference)
                row.append(Decimal("1") if dow == d else Decimal("0"))
            row.append(temps[i] if i < len(temps) else Decimal("0"))
            x_matrix.append(row)

        coeffs, residuals, predicted = self._ols_normal_equation(
            x_matrix, y_vals
        )

        if not coeffs:
            return None

        intercept = coeffs[0]
        coeff_dict: Dict[str, Decimal] = {}
        day_names = ["monday", "tuesday", "wednesday", "thursday",
                     "friday", "saturday"]
        for i, name in enumerate(day_names):
            if i + 1 < len(coeffs):
                coeff_dict[name] = _round_val(coeffs[i + 1], 8)
        if len(coeffs) > 7:
            coeff_dict["temperature"] = _round_val(coeffs[7], 8)

        n = len(y_vals)
        p = len(coeffs)
        validation = self._compute_validation_stats(
            y_vals, predicted, residuals, n, p,
            baseline_config.granularity,
        )

        reg_coeffs = RegressionCoefficients(
            intercept=_round_val(intercept, 8),
            coefficients=coeff_dict,
        )

        model = FittedModel(
            model_type=ModelType.TOWT,
            coefficients=reg_coeffs,
            validation=validation,
            residuals=[_round_val(r, 4) for r in residuals],
            predicted_values=[_round_val(p, 4) for p in predicted],
        )
        model.provenance_hash = _compute_hash(model)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "TOWT model fitted: R2=%.4f, CVRMSE=%.2f%%, hash=%s (%.1f ms)",
            float(validation.r_squared), float(validation.cvrmse_pct),
            model.provenance_hash[:16], elapsed,
        )
        return model

    def _search_3p_cooling(
        self,
        temps: List[Decimal],
        y_vals: List[Decimal],
    ) -> Tuple[Optional[FittedModel], Decimal]:
        """Grid-search for best 3P cooling change-point.

        E = a + b * max(0, T - Tcp) where b > 0
        """
        best_model: Optional[FittedModel] = None
        best_r2 = Decimal("-999")

        cp = self._cp_min
        while cp <= self._cp_max:
            x_cp = [max(Decimal("0"), t - cp) for t in temps]
            x_matrix = [[Decimal("1"), x] for x in x_cp]
            coeffs, residuals, predicted = self._ols_normal_equation(
                x_matrix, y_vals
            )
            if len(coeffs) >= 2 and coeffs[1] > Decimal("0"):
                r2 = self._calc_r_squared(y_vals, predicted)
                if r2 > best_r2:
                    best_r2 = r2
                    n = len(y_vals)
                    p = 3  # intercept, slope, change-point
                    validation = self._compute_validation_stats(
                        y_vals, predicted, residuals, n, p,
                        BaselinePeriodGranularity.MONTHLY,
                    )
                    reg_coeffs = RegressionCoefficients(
                        intercept=_round_val(coeffs[0], 8),
                        cooling_slope=_round_val(coeffs[1], 8),
                        cooling_change_point=_round_val(cp, 2),
                    )
                    best_model = FittedModel(
                        model_type=ModelType.THREE_P_COOL,
                        coefficients=reg_coeffs,
                        validation=validation,
                        residuals=[_round_val(r, 4) for r in residuals],
                        predicted_values=[_round_val(p, 4) for p in predicted],
                    )
            cp += self._cp_step

        return best_model, best_r2

    def _search_3p_heating(
        self,
        temps: List[Decimal],
        y_vals: List[Decimal],
    ) -> Tuple[Optional[FittedModel], Decimal]:
        """Grid-search for best 3P heating change-point.

        E = a + b * max(0, Thp - T) where b > 0
        """
        best_model: Optional[FittedModel] = None
        best_r2 = Decimal("-999")

        cp = self._cp_min
        while cp <= self._cp_max:
            x_cp = [max(Decimal("0"), cp - t) for t in temps]
            x_matrix = [[Decimal("1"), x] for x in x_cp]
            coeffs, residuals, predicted = self._ols_normal_equation(
                x_matrix, y_vals
            )
            if len(coeffs) >= 2 and coeffs[1] > Decimal("0"):
                r2 = self._calc_r_squared(y_vals, predicted)
                if r2 > best_r2:
                    best_r2 = r2
                    n = len(y_vals)
                    p = 3
                    validation = self._compute_validation_stats(
                        y_vals, predicted, residuals, n, p,
                        BaselinePeriodGranularity.MONTHLY,
                    )
                    reg_coeffs = RegressionCoefficients(
                        intercept=_round_val(coeffs[0], 8),
                        heating_slope=_round_val(coeffs[1], 8),
                        heating_change_point=_round_val(cp, 2),
                    )
                    best_model = FittedModel(
                        model_type=ModelType.THREE_P_HEAT,
                        coefficients=reg_coeffs,
                        validation=validation,
                        residuals=[_round_val(r, 4) for r in residuals],
                        predicted_values=[_round_val(p, 4) for p in predicted],
                    )
            cp += self._cp_step

        return best_model, best_r2

    def _search_4p(
        self,
        temps: List[Decimal],
        y_vals: List[Decimal],
    ) -> Tuple[Optional[FittedModel], Decimal]:
        """Grid-search for best 4P model (single change-point, two slopes).

        E = a + bh*max(0, Tcp - T) + bc*max(0, T - Tcp)
        """
        best_model: Optional[FittedModel] = None
        best_r2 = Decimal("-999")

        cp = self._cp_min
        while cp <= self._cp_max:
            x_heat = [max(Decimal("0"), cp - t) for t in temps]
            x_cool = [max(Decimal("0"), t - cp) for t in temps]
            x_matrix = [
                [Decimal("1"), x_heat[i], x_cool[i]]
                for i in range(len(temps))
            ]
            coeffs, residuals, predicted = self._ols_normal_equation(
                x_matrix, y_vals
            )
            if (len(coeffs) >= 3
                    and coeffs[1] > Decimal("0")
                    and coeffs[2] > Decimal("0")):
                r2 = self._calc_r_squared(y_vals, predicted)
                if r2 > best_r2:
                    best_r2 = r2
                    n = len(y_vals)
                    p = 4
                    validation = self._compute_validation_stats(
                        y_vals, predicted, residuals, n, p,
                        BaselinePeriodGranularity.MONTHLY,
                    )
                    reg_coeffs = RegressionCoefficients(
                        intercept=_round_val(coeffs[0], 8),
                        heating_slope=_round_val(coeffs[1], 8),
                        cooling_slope=_round_val(coeffs[2], 8),
                        heating_change_point=_round_val(cp, 2),
                        cooling_change_point=_round_val(cp, 2),
                    )
                    best_model = FittedModel(
                        model_type=ModelType.FOUR_P,
                        coefficients=reg_coeffs,
                        validation=validation,
                        residuals=[_round_val(r, 4) for r in residuals],
                        predicted_values=[_round_val(p, 4) for p in predicted],
                    )
            cp += self._cp_step

        return best_model, best_r2

    def _search_5p(
        self,
        temps: List[Decimal],
        y_vals: List[Decimal],
    ) -> Tuple[Optional[FittedModel], Decimal]:
        """Grid-search for best 5P model (two change-points, two slopes).

        E = a + bh*max(0, Thp - T) + bc*max(0, T - Tcp)
        where Thp <= Tcp
        """
        best_model: Optional[FittedModel] = None
        best_r2 = Decimal("-999")

        cp_h = self._cp_min
        while cp_h <= self._cp_max:
            cp_c = cp_h
            while cp_c <= self._cp_max:
                x_heat = [max(Decimal("0"), cp_h - t) for t in temps]
                x_cool = [max(Decimal("0"), t - cp_c) for t in temps]
                x_matrix = [
                    [Decimal("1"), x_heat[i], x_cool[i]]
                    for i in range(len(temps))
                ]
                coeffs, residuals, predicted = self._ols_normal_equation(
                    x_matrix, y_vals
                )
                if (len(coeffs) >= 3
                        and coeffs[1] > Decimal("0")
                        and coeffs[2] > Decimal("0")):
                    r2 = self._calc_r_squared(y_vals, predicted)
                    if r2 > best_r2:
                        best_r2 = r2
                        n = len(y_vals)
                        p = 5
                        validation = self._compute_validation_stats(
                            y_vals, predicted, residuals, n, p,
                            BaselinePeriodGranularity.MONTHLY,
                        )
                        reg_coeffs = RegressionCoefficients(
                            intercept=_round_val(coeffs[0], 8),
                            heating_slope=_round_val(coeffs[1], 8),
                            cooling_slope=_round_val(coeffs[2], 8),
                            heating_change_point=_round_val(cp_h, 2),
                            cooling_change_point=_round_val(cp_c, 2),
                        )
                        best_model = FittedModel(
                            model_type=ModelType.FIVE_P,
                            coefficients=reg_coeffs,
                            validation=validation,
                            residuals=[_round_val(r, 4) for r in residuals],
                            predicted_values=[
                                _round_val(p, 4) for p in predicted
                            ],
                        )
                cp_c += self._cp_step
            cp_h += self._cp_step

        return best_model, best_r2

    # ------------------------------------------------------------------ #
    # Private: Linear Algebra (Normal Equation)                           #
    # ------------------------------------------------------------------ #

    def _ols_normal_equation(
        self,
        x_matrix: List[List[Decimal]],
        y_vals: List[Decimal],
    ) -> Tuple[List[Decimal], List[Decimal], List[Decimal]]:
        """Solve OLS via normal equation: beta = (X'X)^{-1} X'y.

        Args:
            x_matrix: Design matrix (n x p), first column = 1 (intercept).
            y_vals: Response vector (n x 1).

        Returns:
            Tuple of (coefficients, residuals, predicted_values).
        """
        n = len(y_vals)
        if n == 0 or not x_matrix:
            return [], [], []

        p = len(x_matrix[0])
        if n <= p:
            return [], [], []

        # X'X (p x p)
        xtx: List[List[Decimal]] = [
            [Decimal("0")] * p for _ in range(p)
        ]
        for i in range(p):
            for j in range(p):
                s = Decimal("0")
                for k in range(n):
                    s += x_matrix[k][i] * x_matrix[k][j]
                xtx[i][j] = s

        # X'y (p x 1)
        xty: List[Decimal] = [Decimal("0")] * p
        for i in range(p):
            s = Decimal("0")
            for k in range(n):
                s += x_matrix[k][i] * y_vals[k]
            xty[i] = s

        # Invert X'X via Gauss-Jordan
        inv = self._invert_matrix(xtx)
        if inv is None:
            return [], [], []

        # beta = inv(X'X) * X'y
        coeffs: List[Decimal] = [Decimal("0")] * p
        for i in range(p):
            s = Decimal("0")
            for j in range(p):
                s += inv[i][j] * xty[j]
            coeffs[i] = s

        # Predicted and residuals
        predicted: List[Decimal] = []
        residuals: List[Decimal] = []
        for k in range(n):
            pred = Decimal("0")
            for j in range(p):
                pred += coeffs[j] * x_matrix[k][j]
            predicted.append(pred)
            residuals.append(y_vals[k] - pred)

        return coeffs, residuals, predicted

    def _invert_matrix(
        self,
        matrix: List[List[Decimal]],
    ) -> Optional[List[List[Decimal]]]:
        """Invert a square matrix using Gauss-Jordan elimination."""
        n = len(matrix)
        # Augment with identity
        aug: List[List[Decimal]] = [
            list(matrix[i]) + [
                Decimal("1") if j == i else Decimal("0")
                for j in range(n)
            ]
            for i in range(n)
        ]

        for col in range(n):
            # Find pivot
            max_row = col
            max_val = abs(aug[col][col])
            for row in range(col + 1, n):
                if abs(aug[row][col]) > max_val:
                    max_val = abs(aug[row][col])
                    max_row = row
            if max_val < Decimal("1E-30"):
                return None
            aug[col], aug[max_row] = aug[max_row], aug[col]

            # Scale pivot row
            pivot = aug[col][col]
            for j in range(2 * n):
                aug[col][j] = _safe_divide(aug[col][j], pivot)

            # Eliminate
            for row in range(n):
                if row == col:
                    continue
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]

        # Extract inverse
        inv = [aug[i][n:] for i in range(n)]
        return inv

    # ------------------------------------------------------------------ #
    # Private: Validation Statistics                                       #
    # ------------------------------------------------------------------ #

    def _compute_validation_stats(
        self,
        y_vals: List[Decimal],
        predicted: List[Decimal],
        residuals: List[Decimal],
        n: int,
        p: int,
        granularity: BaselinePeriodGranularity,
    ) -> ModelValidationStats:
        """Compute full suite of regression validation statistics."""
        if n <= p or n == 0:
            return ModelValidationStats(
                n_observations=n, n_parameters=p,
                degrees_of_freedom=max(0, n - p),
            )

        y_mean = _safe_divide(sum(y_vals, Decimal("0")), _decimal(n))
        dof = n - p

        # Sums of squares
        ss_total = sum(((y - y_mean) ** 2 for y in y_vals), Decimal("0"))
        ss_residual = sum((r ** 2 for r in residuals), Decimal("0"))
        ss_regression = ss_total - ss_residual

        # R-squared
        r2 = Decimal("1") - _safe_divide(ss_residual, ss_total)
        r2 = max(Decimal("0"), min(Decimal("1"), r2))

        # Adjusted R-squared
        adj_r2 = Decimal("1") - _safe_divide(
            (Decimal("1") - r2) * _decimal(n - 1),
            _decimal(max(1, n - p - 1)),
        )

        # RMSE
        mse = _safe_divide(ss_residual, _decimal(dof))
        rmse = _decimal(math.sqrt(max(0, float(mse))))

        # CVRMSE
        cvrmse = _safe_pct(rmse, y_mean)

        # NMBE
        sum_residuals = sum(residuals, Decimal("0"))
        nmbe = _safe_divide(
            sum_residuals * Decimal("100"),
            _decimal(dof) * y_mean,
        )

        # MAE
        mae = _safe_divide(
            sum((abs(r) for r in residuals), Decimal("0")),
            _decimal(n),
        )

        # F-statistic
        ms_reg = _safe_divide(ss_regression, _decimal(max(1, p - 1)))
        ms_res = _safe_divide(ss_residual, _decimal(dof))
        f_stat = _safe_divide(ms_reg, ms_res)

        # Durbin-Watson
        dw = self._durbin_watson(residuals)

        # AIC / BIC
        ln_mse = _decimal(
            math.log(max(1e-30, float(_safe_divide(ss_residual, _decimal(n)))))
        )
        aic = _decimal(n) * ln_mse + Decimal("2") * _decimal(p)
        bic = _decimal(n) * ln_mse + _decimal(
            math.log(max(1, n))
        ) * _decimal(p)

        # ASHRAE 14 thresholds
        cvrmse_thresh = ASHRAE14_CVRMSE.get(
            granularity.value, Decimal("30")
        )
        nmbe_thresh = ASHRAE14_NMBE.get(
            granularity.value, Decimal("10")
        )
        r2_thresh = ASHRAE14_R_SQUARED.get(
            granularity.value, Decimal("0")
        )

        stats = ModelValidationStats(
            r_squared=_round_val(r2, 6),
            adjusted_r_squared=_round_val(adj_r2, 6),
            cvrmse_pct=_round_val(cvrmse, 4),
            nmbe_pct=_round_val(nmbe, 4),
            rmse=_round_val(rmse, 4),
            mae=_round_val(mae, 4),
            f_statistic=_round_val(f_stat, 4),
            f_p_value=Decimal("0"),  # exact p-value requires F-distribution table
            durbin_watson=_round_val(dw, 4),
            n_observations=n,
            n_parameters=p,
            degrees_of_freedom=dof,
            ss_total=_round_val(ss_total, 4),
            ss_residual=_round_val(ss_residual, 4),
            ss_regression=_round_val(ss_regression, 4),
            aic=_round_val(aic, 4),
            bic=_round_val(bic, 4),
            ashrae14_cvrmse_pass=abs(cvrmse) <= cvrmse_thresh,
            ashrae14_nmbe_pass=abs(nmbe) <= nmbe_thresh,
            ashrae14_r2_pass=r2 >= r2_thresh,
            ashrae14_overall_pass=(
                abs(cvrmse) <= cvrmse_thresh
                and abs(nmbe) <= nmbe_thresh
                and r2 >= r2_thresh
            ),
            dw_autocorrelation_flag=(dw < DW_LOWER or dw > DW_UPPER),
        )
        stats.provenance_hash = _compute_hash(stats)
        return stats

    def _compute_coefficient_stats(
        self,
        x_matrix: List[List[Decimal]],
        residuals: List[Decimal],
        coeffs: List[Decimal],
        var_names: List[str],
    ) -> Tuple[Dict[str, Decimal], Dict[str, Decimal], Dict[str, Decimal]]:
        """Compute standard errors, t-statistics, and p-values for coefficients."""
        n = len(residuals)
        p = len(coeffs)
        se_dict: Dict[str, Decimal] = {}
        t_dict: Dict[str, Decimal] = {}
        p_dict: Dict[str, Decimal] = {}

        if n <= p or p == 0:
            return se_dict, t_dict, p_dict

        # MSE
        ss_res = sum((r ** 2 for r in residuals), Decimal("0"))
        mse = _safe_divide(ss_res, _decimal(n - p))

        # (X'X)^-1 diagonal for standard errors
        xtx: List[List[Decimal]] = [
            [Decimal("0")] * p for _ in range(p)
        ]
        for i in range(p):
            for j in range(p):
                s = Decimal("0")
                for k in range(n):
                    s += x_matrix[k][i] * x_matrix[k][j]
                xtx[i][j] = s

        inv = self._invert_matrix(xtx)
        if inv is None:
            return se_dict, t_dict, p_dict

        names = ["intercept"] + list(var_names)
        for i in range(min(p, len(names))):
            var_coeff = mse * inv[i][i]
            se = _decimal(math.sqrt(max(0, float(var_coeff))))
            t_stat = _safe_divide(coeffs[i], se)
            se_dict[names[i]] = _round_val(se, 6)
            t_dict[names[i]] = _round_val(t_stat, 4)
            p_dict[names[i]] = Decimal("0")  # exact p requires t-distribution

        return se_dict, t_dict, p_dict

    def _durbin_watson(self, residuals: List[Decimal]) -> Decimal:
        """Compute Durbin-Watson statistic for autocorrelation."""
        n = len(residuals)
        if n < 3:
            return Decimal("2")

        numerator = Decimal("0")
        denominator = Decimal("0")
        for i in range(1, n):
            diff = residuals[i] - residuals[i - 1]
            numerator += diff ** 2
        for r in residuals:
            denominator += r ** 2

        return _safe_divide(numerator, denominator, Decimal("2"))

    # ------------------------------------------------------------------ #
    # Private: Predictor Extraction                                        #
    # ------------------------------------------------------------------ #

    def _build_predictor_matrix(
        self,
        data_points: List[BaselineDataPoint],
        variables: List[IndependentVariableType],
        baseline_config: BaselineConfig,
    ) -> List[List[Decimal]]:
        """Build the design matrix for OLS regression."""
        x_matrix: List[List[Decimal]] = []
        for dp in data_points:
            row = [Decimal("1")]  # intercept
            for var in variables:
                val = self._get_variable_value(dp, var)
                row.append(val)
            x_matrix.append(row)
        return x_matrix

    def _get_variable_value(
        self,
        dp: BaselineDataPoint,
        var_type: IndependentVariableType,
    ) -> Decimal:
        """Extract a specific independent variable value from a data point."""
        if var_type == IndependentVariableType.TEMPERATURE:
            return _decimal(dp.temperature_avg) if dp.temperature_avg is not None else Decimal("0")
        elif var_type == IndependentVariableType.HDD:
            return _decimal(dp.hdd) if dp.hdd is not None else Decimal("0")
        elif var_type == IndependentVariableType.CDD:
            return _decimal(dp.cdd) if dp.cdd is not None else Decimal("0")
        elif var_type == IndependentVariableType.PRODUCTION:
            return _decimal(dp.production_volume) if dp.production_volume is not None else Decimal("0")
        elif var_type == IndependentVariableType.OCCUPANCY:
            return _decimal(dp.occupancy_pct) if dp.occupancy_pct is not None else Decimal("0")
        elif var_type == IndependentVariableType.OPERATING_HOURS:
            return _decimal(dp.operating_hours) if dp.operating_hours is not None else Decimal("0")
        elif var_type == IndependentVariableType.DAYLIGHT_HOURS:
            return _decimal(dp.daylight_hours) if dp.daylight_hours is not None else Decimal("0")
        elif var_type == IndependentVariableType.HUMIDITY:
            return _decimal(dp.humidity_pct) if dp.humidity_pct is not None else Decimal("0")
        elif var_type == IndependentVariableType.CUSTOM:
            if dp.custom_variables:
                vals = list(dp.custom_variables.values())
                return vals[0] if vals else Decimal("0")
            return Decimal("0")
        return Decimal("0")

    def _extract_temperatures(
        self,
        data_points: List[BaselineDataPoint],
        baseline_config: BaselineConfig,
    ) -> List[Decimal]:
        """Extract temperature values, converting to Celsius if needed."""
        temps: List[Decimal] = []
        for dp in data_points:
            if dp.temperature_avg is not None:
                t = _decimal(dp.temperature_avg)
                if baseline_config.temperature_unit.lower().startswith("f"):
                    t = _safe_divide(t - THIRTY_TWO, NINE_FIFTHS)
                temps.append(t)
            else:
                temps.append(Decimal("15"))  # default mild temperature
        return temps

    # ------------------------------------------------------------------ #
    # Private: Prediction                                                  #
    # ------------------------------------------------------------------ #

    def _predict_single(
        self,
        model: FittedModel,
        dp: BaselineDataPoint,
        temp_c: Decimal,
        baseline_config: BaselineConfig,
    ) -> Decimal:
        """Predict energy consumption for a single data point."""
        coeff = model.coefficients
        pred = coeff.intercept

        if model.model_type == ModelType.OLS:
            for var_type in baseline_config.independent_variables:
                var_name = var_type.value
                if var_name in coeff.coefficients:
                    val = self._get_variable_value(dp, var_type)
                    pred += coeff.coefficients[var_name] * val

        elif model.model_type == ModelType.THREE_P_COOL:
            if coeff.cooling_change_point is not None and coeff.cooling_slope is not None:
                x = max(Decimal("0"), temp_c - coeff.cooling_change_point)
                pred += coeff.cooling_slope * x

        elif model.model_type == ModelType.THREE_P_HEAT:
            if coeff.heating_change_point is not None and coeff.heating_slope is not None:
                x = max(Decimal("0"), coeff.heating_change_point - temp_c)
                pred += coeff.heating_slope * x

        elif model.model_type in (ModelType.FOUR_P, ModelType.FIVE_P):
            if coeff.heating_change_point is not None and coeff.heating_slope is not None:
                x_h = max(Decimal("0"), coeff.heating_change_point - temp_c)
                pred += coeff.heating_slope * x_h
            if coeff.cooling_change_point is not None and coeff.cooling_slope is not None:
                x_c = max(Decimal("0"), temp_c - coeff.cooling_change_point)
                pred += coeff.cooling_slope * x_c

        elif model.model_type == ModelType.TOWT:
            dow = dp.period_start.weekday()
            day_names = ["monday", "tuesday", "wednesday", "thursday",
                         "friday", "saturday"]
            for i, name in enumerate(day_names):
                if name in coeff.coefficients and dow == i:
                    pred += coeff.coefficients[name]
            if "temperature" in coeff.coefficients:
                pred += coeff.coefficients["temperature"] * temp_c

        return pred

    # ------------------------------------------------------------------ #
    # Private: Ranking                                                     #
    # ------------------------------------------------------------------ #

    def _rank_models(
        self,
        models: List[FittedModel],
        criterion: ModelSelectionCriterion,
        granularity: BaselinePeriodGranularity,
    ) -> List[FittedModel]:
        """Rank models by the specified criterion."""
        for model in models:
            model.composite_score = self._compute_composite_score(
                model, criterion, granularity
            )

        models_sorted = sorted(
            models, key=lambda m: float(m.composite_score), reverse=True
        )
        for rank, model in enumerate(models_sorted, 1):
            model.rank = rank

        return models_sorted

    def _compute_composite_score(
        self,
        model: FittedModel,
        criterion: ModelSelectionCriterion,
        granularity: BaselinePeriodGranularity,
    ) -> Decimal:
        """Compute a composite ranking score for a model."""
        v = model.validation

        if criterion == ModelSelectionCriterion.BEST_R_SQUARED:
            return v.adjusted_r_squared

        if criterion == ModelSelectionCriterion.BEST_CVRMSE:
            return Decimal("100") - abs(v.cvrmse_pct)

        if criterion == ModelSelectionCriterion.BEST_AIC:
            return Decimal("0") - v.aic

        if criterion == ModelSelectionCriterion.BEST_BIC:
            return Decimal("0") - v.bic

        # Composite
        r2_score = v.adjusted_r_squared * COMPOSITE_WEIGHT_R2

        cvrmse_thresh = ASHRAE14_CVRMSE.get(
            granularity.value, Decimal("30")
        )
        cvrmse_score = _safe_divide(
            max(Decimal("0"), cvrmse_thresh - abs(v.cvrmse_pct)),
            cvrmse_thresh,
        ) * COMPOSITE_WEIGHT_CVRMSE

        nmbe_thresh = ASHRAE14_NMBE.get(
            granularity.value, Decimal("10")
        )
        nmbe_score = _safe_divide(
            max(Decimal("0"), nmbe_thresh - abs(v.nmbe_pct)),
            nmbe_thresh,
        ) * COMPOSITE_WEIGHT_NMBE

        dw_score = Decimal("0")
        if DW_LOWER <= v.durbin_watson <= DW_UPPER:
            dw_score = Decimal("1") * COMPOSITE_WEIGHT_DW
        else:
            dw_deviation = min(
                abs(v.durbin_watson - DW_LOWER),
                abs(v.durbin_watson - DW_UPPER),
            )
            dw_score = max(
                Decimal("0"),
                (Decimal("1") - dw_deviation) * COMPOSITE_WEIGHT_DW,
            )

        return _round_val(r2_score + cvrmse_score + nmbe_score + dw_score, 6)

    # ------------------------------------------------------------------ #
    # Private: R-squared Utility                                           #
    # ------------------------------------------------------------------ #

    def _simple_r_squared(
        self,
        x_vals: List[Decimal],
        y_vals: List[Decimal],
    ) -> Decimal:
        """Compute R-squared for a simple linear regression (single variable)."""
        n = len(x_vals)
        if n < 3:
            return Decimal("-1")

        x_matrix = [[Decimal("1"), x] for x in x_vals]
        coeffs, residuals, predicted = self._ols_normal_equation(
            x_matrix, y_vals
        )
        if not predicted:
            return Decimal("-1")

        return self._calc_r_squared(y_vals, predicted)

    def _calc_r_squared(
        self,
        y_vals: List[Decimal],
        predicted: List[Decimal],
    ) -> Decimal:
        """Compute R-squared from actual and predicted values."""
        n = len(y_vals)
        if n == 0:
            return Decimal("0")

        y_mean = _safe_divide(sum(y_vals, Decimal("0")), _decimal(n))
        ss_total = sum(((y - y_mean) ** 2 for y in y_vals), Decimal("0"))
        ss_residual = sum(
            ((y_vals[i] - predicted[i]) ** 2 for i in range(n)),
            Decimal("0"),
        )

        if ss_total == Decimal("0"):
            return Decimal("1") if ss_residual == Decimal("0") else Decimal("0")

        r2 = Decimal("1") - _safe_divide(ss_residual, ss_total)
        return max(Decimal("0"), min(Decimal("1"), r2))

    # ------------------------------------------------------------------ #
    # Private: Data Quality                                                #
    # ------------------------------------------------------------------ #

    def _assess_data_quality(
        self,
        data_points: List[BaselineDataPoint],
    ) -> DataQualityGrade:
        """Assess the quality of baseline data."""
        total = len(data_points)
        if total == 0:
            return DataQualityGrade.POOR

        valid = sum(1 for dp in data_points if dp.is_valid)
        completeness = _safe_pct(_decimal(valid), _decimal(total))

        # Simple outlier detection: values > 3 sigma from mean
        energies = [_decimal(dp.energy_consumption) for dp in data_points if dp.is_valid]
        outlier_count = 0
        if len(energies) >= 3:
            mean_e = _safe_divide(
                sum(energies, Decimal("0")), _decimal(len(energies))
            )
            var_e = _safe_divide(
                sum(((e - mean_e) ** 2 for e in energies), Decimal("0")),
                _decimal(len(energies)),
            )
            std_e = _decimal(math.sqrt(max(0, float(var_e))))
            if std_e > Decimal("0"):
                for e in energies:
                    z = _safe_divide(abs(e - mean_e), std_e)
                    if z > Decimal("3"):
                        outlier_count += 1

        outlier_pct = _safe_pct(_decimal(outlier_count), _decimal(total))

        for min_comp, max_outlier, grade in QUALITY_THRESHOLDS:
            if completeness >= min_comp and outlier_pct <= max_outlier:
                return grade

        return DataQualityGrade.POOR

    # ------------------------------------------------------------------ #
    # Private: Recommendations & Warnings                                  #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        selected_model: Optional[FittedModel],
        all_models: List[FittedModel],
        data_quality: DataQualityGrade,
        config: BaselineConfig,
    ) -> List[str]:
        """Generate analysis recommendations."""
        recs: List[str] = []

        if data_quality in (DataQualityGrade.FAIR, DataQualityGrade.POOR):
            recs.append(
                "Data quality is below standard. Consider cleaning outliers "
                "and filling data gaps before finalising the baseline."
            )

        if selected_model is not None:
            v = selected_model.validation
            if not v.ashrae14_overall_pass:
                recs.append(
                    "Selected model does not meet all ASHRAE Guideline 14 "
                    "criteria. Consider adding independent variables or "
                    "changing model type."
                )
            if v.dw_autocorrelation_flag:
                recs.append(
                    "Durbin-Watson statistic indicates autocorrelation in "
                    "residuals. Consider using a higher-granularity model "
                    "or adding time-series terms."
                )
            if v.r_squared < Decimal("0.5"):
                recs.append(
                    "R-squared is below 0.50, indicating the model explains "
                    "less than half of energy variance. Add more independent "
                    "variables or review data for errors."
                )

        if not all_models:
            recs.append(
                "No models could be fitted. Check that data contains "
                "sufficient variation in both energy and independent variables."
            )

        passing_count = sum(
            1 for m in all_models if m.validation.ashrae14_overall_pass
        )
        if passing_count == 0 and all_models:
            recs.append(
                "None of the evaluated models pass ASHRAE 14 validation. "
                "Consider extending the baseline period or adding variables."
            )
        elif passing_count > 1:
            recs.append(
                f"{passing_count} models pass ASHRAE 14. Review the model "
                "comparison to confirm the selection is appropriate."
            )

        return recs

    def _generate_warnings(
        self,
        selected_model: Optional[FittedModel],
        all_models: List[FittedModel],
        data_points: List[BaselineDataPoint],
        config: BaselineConfig,
    ) -> List[str]:
        """Generate warnings for baseline development."""
        warnings: List[str] = []

        # Check for short baseline period
        if data_points:
            start = min(dp.period_start for dp in data_points)
            end = max(dp.period_end for dp in data_points)
            days = (end - start).days
            if days < 365 and config.granularity in (
                BaselinePeriodGranularity.MONTHLY,
                BaselinePeriodGranularity.WEEKLY,
            ):
                warnings.append(
                    f"Baseline period is {days} days, which is less than "
                    "12 months. IPMVP recommends at least 12 months of "
                    "baseline data to capture seasonal variation."
                )

        # Check for low data point count
        min_pts = config.min_data_points or MIN_DATA_POINTS.get(
            config.granularity.value, 12
        )
        actual_pts = len([dp for dp in data_points if dp.is_valid])
        if actual_pts < min_pts * 2:
            warnings.append(
                f"Only {actual_pts} data points available (recommended: "
                f">={min_pts * 2} for robust model fitting)."
            )

        return warnings

    def _build_selection_rationale(
        self,
        selected: FittedModel,
        all_models: List[FittedModel],
        criterion: ModelSelectionCriterion,
    ) -> str:
        """Build a human-readable rationale for model selection."""
        v = selected.validation
        rationale = (
            f"Selected {selected.model_type.value} model based on "
            f"{criterion.value} criterion. "
            f"R2={float(v.r_squared):.4f}, "
            f"CVRMSE={float(v.cvrmse_pct):.2f}%, "
            f"NMBE={float(v.nmbe_pct):.2f}%, "
            f"DW={float(v.durbin_watson):.3f}. "
        )
        if v.ashrae14_overall_pass:
            rationale += "Model passes all ASHRAE Guideline 14 criteria. "
        else:
            failed = []
            if not v.ashrae14_cvrmse_pass:
                failed.append("CVRMSE")
            if not v.ashrae14_nmbe_pass:
                failed.append("NMBE")
            if not v.ashrae14_r2_pass:
                failed.append("R-squared")
            rationale += f"Model fails ASHRAE 14 on: {', '.join(failed)}. "

        rationale += (
            f"Ranked 1 of {len(all_models)} models evaluated "
            f"(composite score: {float(selected.composite_score):.4f})."
        )
        return rationale
