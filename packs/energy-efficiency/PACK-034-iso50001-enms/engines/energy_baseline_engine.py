# -*- coding: utf-8 -*-
"""
EnergyBaselineEngine - PACK-034 ISO 50001 EnMS Engine 2
========================================================

Establishes Energy Baselines (EnB) per ISO 50006:2014 and ISO 50047:2016.
Provides regression-based baseline modelling (simple mean, single-variable,
multi-variable, degree-day, and 3P/4P/5P change-point models), model
adequacy assessment, outlier detection, baseline normalisation, baseline
adjustment, and multi-model comparison with AIC/BIC selection.

Calculation Methodology:
    Simple Mean:
        predicted = mean(energy_kwh)

    Single-Variable OLS Regression:
        E = b0 + b1 * X
        where b1 = sum((Xi - Xm)(Yi - Ym)) / sum((Xi - Xm)^2)
              b0 = Ym - b1 * Xm

    Multi-Variable OLS Regression:
        E = b0 + b1*X1 + b2*X2 + ... + bn*Xn
        (Normal equations: B = (X'X)^-1 * X'Y via Gauss elimination)

    Degree-Day Model:
        E = b0 + b1 * HDD  (heating)
        E = b0 + b1 * CDD  (cooling)
        E = b0 + b1 * HDD + b2 * CDD  (combined)

    3-Parameter Change-Point (3P):
        E = b0 + b1 * max(X - CP, 0)   (heating)
        E = b0 + b1 * max(CP - X, 0)   (cooling)

    4-Parameter Change-Point (4P):
        E = b0 + b1 * max(CP - X, 0) + b2 * max(X - CP, 0)

    5-Parameter Change-Point (5P):
        E = b0 + b1 * max(CPh - X, 0) + b2 * max(X - CPc, 0)

    Model Adequacy (ISO 50006):
        R^2 = 1 - SS_res / SS_tot
        CV(RMSE) = RMSE / Y_mean * 100
        F-statistic = (SS_reg / p) / (SS_res / (n - p - 1))
        AIC = n * ln(SS_res / n) + 2 * (p + 1)
        BIC = n * ln(SS_res / n) + (p + 1) * ln(n)

    Normalised Consumption:
        E_norm = E_actual + (E_baseline_conditions - E_actual_conditions)

    Shapiro-Wilk Approximation:
        W_approx = 1 - (skewness^2 + (kurtosis-3)^2 / 4) / (6 * n)
        (lightweight proxy for residual normality check)

Regulatory References:
    - ISO 50001:2018 - Energy management systems
    - ISO 50006:2014 - Baseline/indicators measurement and verification
    - ISO 50047:2016 - Determination of energy savings
    - ISO 50015:2014 - Measurement and verification
    - ASHRAE Guideline 14-2014 - Measurement of energy savings
    - IPMVP Core Concepts (EVO, 2022)
    - ASHRAE RP-1050: Inverse modelling toolkit

Zero-Hallucination:
    - All formulas are standard statistical / engineering calculations
    - Regression via analytical OLS (normal equations), no ML models
    - Model selection via published AIC/BIC/CV(RMSE) criteria
    - ISO 50006 adequacy thresholds from published standard
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-034 ISO 50001 EnMS
Engine:  2 of 10
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
    try:
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)
    except (InvalidOperation, OverflowError):
        return Decimal("0")


def _decimal_sqrt(value: Decimal) -> Decimal:
    """Compute square root of a Decimal via float intermediary.

    Suitable for statistical calculations where float-level precision
    is acceptable before the final result is stored as Decimal.
    """
    if value < Decimal("0"):
        return Decimal("0")
    return _decimal(math.sqrt(float(value)))


def _decimal_ln(value: Decimal) -> Decimal:
    """Compute natural logarithm of a Decimal via float intermediary."""
    fv = float(value)
    if fv <= 0:
        return Decimal("0")
    return _decimal(math.log(fv))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BaselineModelType(str, Enum):
    """Energy baseline regression model types per ISO 50006.

    SIMPLE_MEAN:       Flat average (no relevant variables).
    SINGLE_VARIABLE:   Simple OLS with one relevant variable.
    MULTI_VARIABLE:    Multi-variable OLS regression.
    DEGREE_DAY:        Degree-day (HDD/CDD) regression model.
    CHANGE_POINT_3P:   3-parameter change-point model.
    CHANGE_POINT_4P:   4-parameter change-point model.
    CHANGE_POINT_5P:   5-parameter change-point model.
    """
    SIMPLE_MEAN = "simple_mean"
    SINGLE_VARIABLE = "single_variable"
    MULTI_VARIABLE = "multi_variable"
    DEGREE_DAY = "degree_day"
    CHANGE_POINT_3P = "change_point_3p"
    CHANGE_POINT_4P = "change_point_4p"
    CHANGE_POINT_5P = "change_point_5p"


class BaselineStatus(str, Enum):
    """Lifecycle status of an energy baseline.

    DRAFT:        Initial creation, not yet reviewed.
    UNDER_REVIEW: Submitted for management review.
    APPROVED:     Accepted for use in EnPI calculations.
    ADJUSTED:     Modified due to a qualifying trigger event.
    SUPERSEDED:   Replaced by a newer approved baseline.
    """
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    ADJUSTED = "adjusted"
    SUPERSEDED = "superseded"


class VariableType(str, Enum):
    """Classification of baseline variables per ISO 50006.

    STATIC_FACTOR:     Conditions that do not change during baseline period.
    RELEVANT_VARIABLE: Factors that routinely affect energy consumption.
    ENERGY_DRIVER:     Primary driver of energy use (e.g. production output).
    """
    STATIC_FACTOR = "static_factor"
    RELEVANT_VARIABLE = "relevant_variable"
    ENERGY_DRIVER = "energy_driver"


class AdjustmentTrigger(str, Enum):
    """Events that trigger a baseline adjustment per ISO 50006 clause 6.5.

    PROCESS_CHANGE:          Significant change in production process.
    BOUNDARY_CHANGE:         Change in organisational / physical boundary.
    EQUIPMENT_CHANGE:        Major equipment addition or removal.
    METHODOLOGY_CHANGE:      Change in measurement or calculation method.
    DATA_ERROR_CORRECTION:   Correction of a discovered data error.
    """
    PROCESS_CHANGE = "process_change"
    BOUNDARY_CHANGE = "boundary_change"
    EQUIPMENT_CHANGE = "equipment_change"
    METHODOLOGY_CHANGE = "methodology_change"
    DATA_ERROR_CORRECTION = "data_error_correction"


class DataGranularity(str, Enum):
    """Temporal granularity of baseline energy data.

    HOURLY:  Hourly interval data.
    DAILY:   Daily aggregated data.
    WEEKLY:  Weekly aggregated data.
    MONTHLY: Monthly aggregated data (most common for ISO 50006).
    ANNUAL:  Annual aggregated data.
    """
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ANNUAL = "annual"


# ---------------------------------------------------------------------------
# Constants / Reference Data
# ---------------------------------------------------------------------------

# Minimum data points required per model type (ISO 50006 guidance).
MINIMUM_DATA_REQUIREMENTS: Dict[str, int] = {
    BaselineModelType.SIMPLE_MEAN.value: 12,
    BaselineModelType.SINGLE_VARIABLE.value: 12,
    BaselineModelType.MULTI_VARIABLE.value: 24,
    BaselineModelType.DEGREE_DAY.value: 12,
    BaselineModelType.CHANGE_POINT_3P.value: 12,
    BaselineModelType.CHANGE_POINT_4P.value: 18,
    BaselineModelType.CHANGE_POINT_5P.value: 24,
}

# Model adequacy criteria per ISO 50006 / ASHRAE 14-2014.
MODEL_ADEQUACY_CRITERIA: Dict[str, Dict[str, Decimal]] = {
    "excellent": {
        "min_r_squared": Decimal("0.90"),
        "max_cv_rmse": Decimal("15.0"),
        "max_p_value": Decimal("0.01"),
    },
    "good": {
        "min_r_squared": Decimal("0.75"),
        "max_cv_rmse": Decimal("25.0"),
        "max_p_value": Decimal("0.05"),
    },
    "acceptable": {
        "min_r_squared": Decimal("0.50"),
        "max_cv_rmse": Decimal("35.0"),
        "max_p_value": Decimal("0.10"),
    },
    "poor": {
        "min_r_squared": Decimal("0.00"),
        "max_cv_rmse": Decimal("100.0"),
        "max_p_value": Decimal("1.00"),
    },
}

# ISO 50006 guideline references for audit trail.
ISO_50006_GUIDELINES: Dict[str, str] = {
    "clause_4.1": "General requirements for baselines and EnPIs",
    "clause_4.2": "Types of EnPIs (measured value, ratio, model, estimate)",
    "clause_4.3": "Relevant variables and static factors",
    "clause_5.1": "Establishing the energy baseline period",
    "clause_5.2": "Data requirements for baseline establishment",
    "clause_5.3": "Statistical analysis and model fitting",
    "clause_5.4": "Model validation and adequacy testing",
    "clause_6.1": "Normalisation of energy consumption",
    "clause_6.2": "Comparison of actual vs expected consumption",
    "clause_6.3": "Quantification of energy performance improvement",
    "clause_6.4": "Uncertainty in energy savings determination",
    "clause_6.5": "Conditions requiring baseline adjustment",
    "clause_6.6": "Documentation and record-keeping requirements",
    "annex_a": "Examples of baseline models and EnPI types",
    "annex_b": "Statistical methods for model evaluation",
    "annex_c": "Worked examples of baseline adjustment",
    "ashrae_14_cv_rmse": "CV(RMSE) <= 25% for monthly models (ASHRAE 14-2014)",
    "ashrae_14_nmbe": "NMBE <= +/- 5% for calibrated models (ASHRAE 14-2014)",
    "ipmvp_option_c": "Whole facility measurement - regression baseline",
    "ipmvp_option_d": "Calibrated simulation baseline",
}

# Granularity to typical number of data points per year.
GRANULARITY_POINTS_PER_YEAR: Dict[str, int] = {
    DataGranularity.HOURLY.value: 8760,
    DataGranularity.DAILY.value: 365,
    DataGranularity.WEEKLY.value: 52,
    DataGranularity.MONTHLY.value: 12,
    DataGranularity.ANNUAL.value: 1,
}

# CV(RMSE) thresholds by granularity (ASHRAE 14-2014).
CV_RMSE_THRESHOLDS: Dict[str, Decimal] = {
    DataGranularity.HOURLY.value: Decimal("30.0"),
    DataGranularity.DAILY.value: Decimal("25.0"),
    DataGranularity.WEEKLY.value: Decimal("25.0"),
    DataGranularity.MONTHLY.value: Decimal("25.0"),
    DataGranularity.ANNUAL.value: Decimal("20.0"),
}


# ---------------------------------------------------------------------------
# Pydantic Models -- Data / Configuration
# ---------------------------------------------------------------------------


class BaselineDataPoint(BaseModel):
    """A single observation in the baseline dataset.

    Attributes:
        period_start: Start date of the measurement period.
        period_end: End date of the measurement period.
        energy_kwh: Total energy consumption in kWh for the period.
        variables: Dictionary of relevant-variable name to observed value.
    """
    period_start: date = Field(..., description="Period start date")
    period_end: date = Field(..., description="Period end date")
    energy_kwh: Decimal = Field(
        ..., ge=0, description="Energy consumption (kWh)"
    )
    variables: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Relevant variable values for this period",
    )

    @field_validator("energy_kwh", mode="before")
    @classmethod
    def coerce_energy(cls, v: Any) -> Any:
        """Coerce energy value to Decimal."""
        if not isinstance(v, Decimal):
            return _decimal(v)
        return v

    @field_validator("variables", mode="before")
    @classmethod
    def coerce_variables(cls, v: Any) -> Any:
        """Coerce variable values to Decimal."""
        if isinstance(v, dict):
            return {k: _decimal(val) for k, val in v.items()}
        return v


class RegressionCoefficient(BaseModel):
    """A single regression coefficient with significance statistics.

    Attributes:
        variable_name: Name of the independent variable (or 'intercept').
        coefficient: Estimated coefficient value.
        standard_error: Standard error of the coefficient.
        t_statistic: t-statistic for significance testing.
        p_value: p-value for the null hypothesis test.
        is_significant: Whether the coefficient is statistically significant.
    """
    variable_name: str = Field(..., description="Variable name")
    coefficient: Decimal = Field(
        default=Decimal("0"), description="Coefficient value"
    )
    standard_error: Decimal = Field(
        default=Decimal("0"), ge=0, description="Standard error"
    )
    t_statistic: Decimal = Field(
        default=Decimal("0"), description="t-statistic"
    )
    p_value: Decimal = Field(
        default=Decimal("1"), ge=0, le=Decimal("1"), description="p-value"
    )
    is_significant: bool = Field(
        default=False, description="Significant at alpha level"
    )


class RegressionModel(BaseModel):
    """Complete regression model specification and fit statistics.

    Attributes:
        model_type: Type of regression model.
        intercept: Intercept (b0) coefficient.
        coefficients: List of variable coefficients with significance.
        r_squared: Coefficient of determination.
        adjusted_r_squared: Adjusted R-squared.
        cv_rmse: Coefficient of variation of RMSE (percentage).
        f_statistic: Overall model F-statistic.
        f_p_value: p-value of the F-statistic.
        n_observations: Number of observations used.
        degrees_of_freedom: Residual degrees of freedom.
        aic: Akaike Information Criterion.
        bic: Bayesian Information Criterion.
    """
    model_type: BaselineModelType = Field(
        ..., description="Regression model type"
    )
    intercept: Decimal = Field(
        default=Decimal("0"), description="Intercept coefficient"
    )
    coefficients: List[RegressionCoefficient] = Field(
        default_factory=list, description="Variable coefficients"
    )
    r_squared: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("1"),
        description="Coefficient of determination",
    )
    adjusted_r_squared: Decimal = Field(
        default=Decimal("0"), description="Adjusted R-squared"
    )
    cv_rmse: Decimal = Field(
        default=Decimal("0"), ge=0, description="CV(RMSE) percentage"
    )
    f_statistic: Decimal = Field(
        default=Decimal("0"), ge=0, description="F-statistic"
    )
    f_p_value: Decimal = Field(
        default=Decimal("1"), ge=0, le=Decimal("1"),
        description="F-statistic p-value",
    )
    n_observations: int = Field(
        default=0, ge=0, description="Number of observations"
    )
    degrees_of_freedom: int = Field(
        default=0, ge=0, description="Residual degrees of freedom"
    )
    aic: Decimal = Field(default=Decimal("0"), description="AIC")
    bic: Decimal = Field(default=Decimal("0"), description="BIC")

    @field_validator("model_type", mode="before")
    @classmethod
    def validate_model_type(cls, v: Any) -> Any:
        """Accept string values for BaselineModelType."""
        if isinstance(v, str):
            valid = {t.value for t in BaselineModelType}
            if v not in valid:
                raise ValueError(
                    f"Unknown model type '{v}'. Must be one of: {sorted(valid)}"
                )
        return v


class StaticFactor(BaseModel):
    """A static factor (unchanging condition) during the baseline period.

    Attributes:
        name: Factor name (e.g. 'floor_area', 'operating_shifts').
        value: Numeric value of the factor.
        unit: Unit of measurement.
        description: Human-readable description.
        verification_date: Date the factor value was last verified.
    """
    name: str = Field(..., max_length=200, description="Factor name")
    value: Decimal = Field(..., description="Factor value")
    unit: str = Field(default="", max_length=50, description="Unit")
    description: str = Field(
        default="", max_length=500, description="Description"
    )
    verification_date: date = Field(
        ..., description="Last verification date"
    )

    @field_validator("value", mode="before")
    @classmethod
    def coerce_value(cls, v: Any) -> Any:
        """Coerce value to Decimal."""
        if not isinstance(v, Decimal):
            return _decimal(v)
        return v


class RelevantVariable(BaseModel):
    """A relevant variable that routinely affects energy consumption.

    Attributes:
        name: Variable name (e.g. 'production_volume', 'hdd_18').
        variable_type: Classification of this variable.
        unit: Unit of measurement.
        description: Human-readable description.
        data_source: Source system or data feed for this variable.
        min_value: Observed minimum value during baseline period.
        max_value: Observed maximum value during baseline period.
        mean_value: Mean value during baseline period.
    """
    name: str = Field(..., max_length=200, description="Variable name")
    variable_type: VariableType = Field(
        default=VariableType.RELEVANT_VARIABLE,
        description="Variable classification",
    )
    unit: str = Field(default="", max_length=50, description="Unit")
    description: str = Field(
        default="", max_length=500, description="Description"
    )
    data_source: str = Field(
        default="", max_length=200, description="Data source"
    )
    min_value: Decimal = Field(
        default=Decimal("0"), description="Observed minimum"
    )
    max_value: Decimal = Field(
        default=Decimal("0"), description="Observed maximum"
    )
    mean_value: Decimal = Field(
        default=Decimal("0"), description="Observed mean"
    )

    @field_validator("variable_type", mode="before")
    @classmethod
    def validate_variable_type(cls, v: Any) -> Any:
        """Accept string values for VariableType."""
        if isinstance(v, str):
            valid = {t.value for t in VariableType}
            if v not in valid:
                raise ValueError(
                    f"Unknown variable type '{v}'. Must be one of: {sorted(valid)}"
                )
        return v


class BaselineConfig(BaseModel):
    """Configuration parameters for baseline establishment.

    Attributes:
        model_type: Type of regression model to fit.
        min_data_points: Minimum number of data points required.
        significance_level: Alpha level for significance testing.
        max_cv_rmse: Maximum acceptable CV(RMSE) percentage.
        min_r_squared: Minimum acceptable R-squared.
        outlier_threshold: Standard deviations for outlier detection.
    """
    model_type: BaselineModelType = Field(
        default=BaselineModelType.SINGLE_VARIABLE,
        description="Model type to fit",
    )
    min_data_points: int = Field(
        default=12, ge=3, le=8760,
        description="Minimum data points required",
    )
    significance_level: Decimal = Field(
        default=Decimal("0.05"), gt=0, lt=Decimal("1"),
        description="Significance level (alpha)",
    )
    max_cv_rmse: Decimal = Field(
        default=Decimal("25.0"), gt=0, le=Decimal("100"),
        description="Maximum acceptable CV(RMSE) %",
    )
    min_r_squared: Decimal = Field(
        default=Decimal("0.75"), ge=0, le=Decimal("1"),
        description="Minimum acceptable R-squared",
    )
    outlier_threshold: Decimal = Field(
        default=Decimal("3.0"), gt=0, le=Decimal("10"),
        description="Outlier z-score threshold",
    )

    @field_validator("model_type", mode="before")
    @classmethod
    def validate_model_type(cls, v: Any) -> Any:
        """Accept string values for BaselineModelType."""
        if isinstance(v, str):
            valid = {t.value for t in BaselineModelType}
            if v not in valid:
                raise ValueError(
                    f"Unknown model type '{v}'. Must be one of: {sorted(valid)}"
                )
        return v


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class BaselineResult(BaseModel):
    """Complete energy baseline establishment result.

    Attributes:
        baseline_id: Unique identifier for this baseline.
        enms_id: Parent EnMS entity identifier.
        baseline_name: Human-readable name.
        period_start: First date of baseline period.
        period_end: Last date of baseline period.
        energy_type: Type of energy (e.g. 'electricity', 'natural_gas').
        total_energy_kwh: Total energy consumption during baseline period.
        model: Fitted regression model.
        relevant_variables: List of relevant variables used.
        static_factors: List of static factors documented.
        data_points: Cleaned data points used for fitting.
        residuals: List of model residuals (actual - predicted).
        confidence_intervals: Model-level confidence interval bounds.
        status: Current baseline lifecycle status.
        calculated_at: Timestamp of calculation.
        processing_time_ms: Duration of calculation in milliseconds.
        provenance_hash: SHA-256 audit hash.
    """
    baseline_id: str = Field(
        default_factory=_new_uuid, description="Unique baseline ID"
    )
    enms_id: str = Field(default="", description="Parent EnMS ID")
    baseline_name: str = Field(
        default="", max_length=500, description="Baseline name"
    )
    period_start: date = Field(..., description="Baseline period start")
    period_end: date = Field(..., description="Baseline period end")
    energy_type: str = Field(
        default="electricity", max_length=100,
        description="Type of energy",
    )
    total_energy_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total baseline energy (kWh)"
    )
    model: RegressionModel = Field(
        ..., description="Fitted regression model"
    )
    relevant_variables: List[RelevantVariable] = Field(
        default_factory=list, description="Relevant variables"
    )
    static_factors: List[StaticFactor] = Field(
        default_factory=list, description="Static factors"
    )
    data_points: List[BaselineDataPoint] = Field(
        default_factory=list, description="Cleaned data points"
    )
    residuals: List[Decimal] = Field(
        default_factory=list, description="Model residuals"
    )
    confidence_intervals: Dict[str, Any] = Field(
        default_factory=dict, description="Confidence interval bounds"
    )
    status: BaselineStatus = Field(
        default=BaselineStatus.DRAFT, description="Baseline status"
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Calculation timestamp"
    )
    processing_time_ms: int = Field(
        default=0, ge=0, description="Processing time (ms)"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )


class BaselineAdjustment(BaseModel):
    """Record of an adjustment made to an approved baseline.

    Attributes:
        adjustment_id: Unique adjustment identifier.
        baseline_id: Identifier of the adjusted baseline.
        trigger: Event that triggered the adjustment.
        description: Explanation of the adjustment.
        old_value: Previous total energy or coefficient value.
        new_value: Updated value after adjustment.
        adjustment_date: Date the adjustment was applied.
        approved_by: Person or role that approved the adjustment.
        provenance_hash: SHA-256 audit hash.
    """
    adjustment_id: str = Field(
        default_factory=_new_uuid, description="Unique adjustment ID"
    )
    baseline_id: str = Field(..., description="Adjusted baseline ID")
    trigger: AdjustmentTrigger = Field(
        ..., description="Adjustment trigger event"
    )
    description: str = Field(
        default="", max_length=2000, description="Adjustment description"
    )
    old_value: Decimal = Field(
        default=Decimal("0"), description="Previous value"
    )
    new_value: Decimal = Field(
        default=Decimal("0"), description="Updated value"
    )
    adjustment_date: date = Field(
        ..., description="Adjustment application date"
    )
    approved_by: str = Field(
        default="", max_length=200, description="Approver"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    @field_validator("trigger", mode="before")
    @classmethod
    def validate_trigger(cls, v: Any) -> Any:
        """Accept string values for AdjustmentTrigger."""
        if isinstance(v, str):
            valid = {t.value for t in AdjustmentTrigger}
            if v not in valid:
                raise ValueError(
                    f"Unknown trigger '{v}'. Must be one of: {sorted(valid)}"
                )
        return v


class BaselineValidation(BaseModel):
    """Validation result for a baseline model.

    Attributes:
        is_adequate: Whether the model passes all adequacy criteria.
        r_squared_pass: Whether R-squared meets the threshold.
        cv_rmse_pass: Whether CV(RMSE) meets the threshold.
        significant_variables_pass: Whether all variables are significant.
        residual_normality_pass: Whether residuals appear normally distributed.
        data_sufficiency_pass: Whether minimum data points are met.
        messages: List of validation messages / warnings.
        adequacy_rating: Overall rating (excellent/good/acceptable/poor).
        provenance_hash: SHA-256 audit hash.
    """
    is_adequate: bool = Field(default=False, description="Overall adequacy")
    r_squared_pass: bool = Field(default=False, description="R^2 check")
    cv_rmse_pass: bool = Field(default=False, description="CV(RMSE) check")
    significant_variables_pass: bool = Field(
        default=False, description="Variable significance check"
    )
    residual_normality_pass: bool = Field(
        default=False, description="Residual normality check"
    )
    data_sufficiency_pass: bool = Field(
        default=False, description="Data sufficiency check"
    )
    messages: List[str] = Field(
        default_factory=list, description="Validation messages"
    )
    adequacy_rating: str = Field(
        default="poor", description="Adequacy rating"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )


class ModelComparison(BaseModel):
    """Result of comparing multiple candidate baseline models.

    Attributes:
        models: List of candidate models with identifiers.
        best_model_index: Index of the recommended model.
        best_model_reason: Reason for recommendation.
        comparison_matrix: Metric-by-metric comparison.
        provenance_hash: SHA-256 audit hash.
    """
    models: List[Dict[str, Any]] = Field(
        default_factory=list, description="Candidate models"
    )
    best_model_index: int = Field(
        default=0, ge=0, description="Best model index"
    )
    best_model_reason: str = Field(
        default="", description="Recommendation rationale"
    )
    comparison_matrix: Dict[str, Any] = Field(
        default_factory=dict, description="Metric comparison matrix"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class EnergyBaselineEngine:
    """Energy Baseline (EnB) establishment engine per ISO 50006.

    Fits regression-based baseline models to historical energy data,
    evaluates model adequacy, detects outliers, normalises consumption,
    supports baseline adjustments, and compares candidate models via
    AIC/BIC selection.

    Usage::

        engine = EnergyBaselineEngine()
        result = engine.establish_baseline(data_points, config, metadata)
        print(f"R-squared: {result.model.r_squared}")
        print(f"CV(RMSE): {result.model.cv_rmse}%")

    All arithmetic uses ``Decimal`` for deterministic, audit-grade precision.
    Every result carries a SHA-256 provenance hash.
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise EnergyBaselineEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - min_r_squared (Decimal): default minimum R-squared
                - max_cv_rmse (Decimal): default maximum CV(RMSE) %
                - significance_level (Decimal): default alpha
                - outlier_method (str): 'zscore' or 'iqr'
        """
        self.config = config or {}
        self._default_min_r_squared = _decimal(
            self.config.get("min_r_squared", Decimal("0.75"))
        )
        self._default_max_cv_rmse = _decimal(
            self.config.get("max_cv_rmse", Decimal("25.0"))
        )
        self._default_significance = _decimal(
            self.config.get("significance_level", Decimal("0.05"))
        )
        self._default_outlier_method: str = str(
            self.config.get("outlier_method", "zscore")
        )
        logger.info(
            "EnergyBaselineEngine v%s initialised (min_r2=%.2f, max_cv=%.1f, "
            "alpha=%.2f, outlier=%s)",
            self.engine_version,
            float(self._default_min_r_squared),
            float(self._default_max_cv_rmse),
            float(self._default_significance),
            self._default_outlier_method,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def establish_baseline(
        self,
        data_points: List[BaselineDataPoint],
        config: BaselineConfig,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BaselineResult:
        """Establish an energy baseline from historical data.

        Workflow:
            1. Validate minimum data point count.
            2. Detect and remove outliers.
            3. Fit regression model per ``config.model_type``.
            4. Calculate R-squared, CV(RMSE), F-statistic.
            5. Compute residuals and confidence intervals.
            6. Assemble baseline result with provenance hash.

        Args:
            data_points: Historical energy data observations.
            config: Baseline configuration parameters.
            metadata: Optional metadata (enms_id, name, energy_type, etc.).

        Returns:
            BaselineResult with fitted model and statistics.

        Raises:
            ValueError: If data points are insufficient or invalid.
        """
        t0 = time.perf_counter()
        metadata = metadata or {}
        logger.info(
            "Establishing baseline: model=%s, points=%d",
            config.model_type.value, len(data_points),
        )

        # Step 1 -- Validate data sufficiency
        min_required = max(
            config.min_data_points,
            MINIMUM_DATA_REQUIREMENTS.get(
                config.model_type.value, config.min_data_points
            ),
        )
        if len(data_points) < min_required:
            raise ValueError(
                f"Insufficient data: {len(data_points)} points provided, "
                f"{min_required} required for {config.model_type.value} model"
            )

        # Step 2 -- Sort by period_start
        sorted_points = sorted(data_points, key=lambda dp: dp.period_start)

        # Step 3 -- Detect outliers
        outlier_indices = self.detect_outliers(
            sorted_points,
            method=self._default_outlier_method,
            threshold=config.outlier_threshold,
        )
        clean_points = [
            dp for i, dp in enumerate(sorted_points) if i not in outlier_indices
        ]
        if outlier_indices:
            logger.warning(
                "Removed %d outlier(s) from %d data points",
                len(outlier_indices), len(sorted_points),
            )

        # Post-outlier data sufficiency check
        if len(clean_points) < min_required:
            raise ValueError(
                f"After outlier removal only {len(clean_points)} points remain; "
                f"{min_required} required for {config.model_type.value} model"
            )

        # Step 4 -- Fit model
        model = self._fit_model(clean_points, config)

        # Step 5 -- Compute residuals
        residuals = self._compute_residuals(clean_points, model)

        # Step 6 -- Confidence intervals
        confidence_intervals = self._compute_confidence_intervals(
            model, residuals, config.significance_level
        )

        # Step 7 -- Build relevant variable metadata
        rel_vars = self._extract_relevant_variables(clean_points, model)

        # Step 8 -- Compute total energy
        total_energy = sum(
            (dp.energy_kwh for dp in clean_points), Decimal("0")
        )

        # Step 9 -- Assemble result
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        result = BaselineResult(
            baseline_id=metadata.get("baseline_id", _new_uuid()),
            enms_id=metadata.get("enms_id", ""),
            baseline_name=metadata.get("baseline_name", ""),
            period_start=clean_points[0].period_start,
            period_end=clean_points[-1].period_end,
            energy_type=metadata.get("energy_type", "electricity"),
            total_energy_kwh=_round_val(total_energy, 2),
            model=model,
            relevant_variables=rel_vars,
            static_factors=metadata.get("static_factors", []),
            data_points=clean_points,
            residuals=[_round_val(r, 4) for r in residuals],
            confidence_intervals=confidence_intervals,
            status=BaselineStatus.DRAFT,
            calculated_at=_utcnow(),
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Baseline established: id=%s, R2=%.4f, CV(RMSE)=%.2f%%, "
            "n=%d, hash=%s (%.0f ms)",
            result.baseline_id,
            float(model.r_squared),
            float(model.cv_rmse),
            model.n_observations,
            result.provenance_hash[:16],
            elapsed_ms,
        )
        return result

    def fit_simple_mean(
        self,
        data: List[BaselineDataPoint],
    ) -> RegressionModel:
        """Fit a simple-mean (flat average) baseline model.

        The predicted value for every period is the arithmetic mean of
        all observed energy values.  R-squared is zero by definition.

        Args:
            data: Baseline data points.

        Returns:
            RegressionModel with intercept = mean and no coefficients.
        """
        n = len(data)
        if n == 0:
            raise ValueError("Cannot fit model with zero data points")

        energies = [dp.energy_kwh for dp in data]
        total = sum(energies, Decimal("0"))
        mean_val = _safe_divide(total, _decimal(n))

        # SS_tot
        ss_tot = sum(
            ((e - mean_val) ** 2 for e in energies), Decimal("0")
        )

        # For simple mean: predicted = mean, so SS_res == SS_tot, R^2 = 0
        predicted = [mean_val] * n
        cv_rmse = self.calculate_cv_rmse(energies, predicted)

        # AIC / BIC (p = 0 parameters, only intercept)
        ss_res = ss_tot
        aic = self._compute_aic(ss_res, n, 0)
        bic = self._compute_bic(ss_res, n, 0)

        model = RegressionModel(
            model_type=BaselineModelType.SIMPLE_MEAN,
            intercept=_round_val(mean_val, 4),
            coefficients=[],
            r_squared=Decimal("0"),
            adjusted_r_squared=Decimal("0"),
            cv_rmse=_round_val(cv_rmse, 4),
            f_statistic=Decimal("0"),
            f_p_value=Decimal("1"),
            n_observations=n,
            degrees_of_freedom=n - 1,
            aic=_round_val(aic, 4),
            bic=_round_val(bic, 4),
        )
        logger.debug(
            "Simple mean model: intercept=%.2f, CV(RMSE)=%.2f%%",
            float(mean_val), float(cv_rmse),
        )
        return model

    def fit_single_variable(
        self,
        data: List[BaselineDataPoint],
        variable: str,
    ) -> RegressionModel:
        """Fit a single-variable OLS regression model.

        E = b0 + b1 * X

        Uses the analytical OLS solution:
            b1 = sum((Xi-Xm)(Yi-Ym)) / sum((Xi-Xm)^2)
            b0 = Ym - b1 * Xm

        Args:
            data: Baseline data points.
            variable: Name of the relevant variable to use.

        Returns:
            RegressionModel with one coefficient.
        """
        n = len(data)
        if n < 3:
            raise ValueError(
                f"Need >= 3 data points for single-variable OLS, got {n}"
            )

        y_vals = [dp.energy_kwh for dp in data]
        x_vals = [dp.variables.get(variable, Decimal("0")) for dp in data]

        y_mean = _safe_divide(
            sum(y_vals, Decimal("0")), _decimal(n)
        )
        x_mean = _safe_divide(
            sum(x_vals, Decimal("0")), _decimal(n)
        )

        # Sums for OLS
        sxy = sum(
            ((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals)),
            Decimal("0"),
        )
        sxx = sum(
            ((x - x_mean) ** 2 for x in x_vals), Decimal("0"),
        )

        b1 = _safe_divide(sxy, sxx)
        b0 = y_mean - b1 * x_mean

        # Predicted and residuals
        predicted = [b0 + b1 * x for x in x_vals]
        ss_res = sum(
            ((y - yp) ** 2 for y, yp in zip(y_vals, predicted)), Decimal("0")
        )
        ss_tot = sum(
            ((y - y_mean) ** 2 for y in y_vals), Decimal("0")
        )

        r_squared = self.calculate_r_squared(y_vals, predicted)
        adj_r_squared = self._adjusted_r_squared(r_squared, n, 1)
        cv_rmse = self.calculate_cv_rmse(y_vals, predicted)

        # Standard error of b1
        mse = _safe_divide(ss_res, _decimal(n - 2))
        se_b1 = _decimal_sqrt(_safe_divide(mse, sxx))

        # t-statistic and p-value approximation for b1
        # When SE is zero (perfect fit), the coefficient is infinitely
        # significant -- set p_value to 0 rather than the degenerate 1.
        if se_b1 == Decimal("0") and b1 != Decimal("0"):
            t_stat = Decimal("9999")
            p_val = Decimal("0")
        else:
            t_stat = _safe_divide(b1, se_b1)
            p_val = self._approx_t_p_value(t_stat, n - 2)

        is_sig = p_val < self._default_significance

        # Standard error of b0
        x_sq_mean = _safe_divide(
            sum((x ** 2 for x in x_vals), Decimal("0")), _decimal(n)
        )
        se_b0 = _decimal_sqrt(mse * _safe_divide(x_sq_mean, sxx))
        t_stat_b0 = _safe_divide(b0, se_b0) if se_b0 != Decimal("0") else Decimal("0")
        p_val_b0 = self._approx_t_p_value(t_stat_b0, n - 2)

        # F-statistic
        f_stat, f_p = self._compute_f_statistic(ss_tot - ss_res, ss_res, 1, n)

        # AIC / BIC
        aic = self._compute_aic(ss_res, n, 1)
        bic = self._compute_bic(ss_res, n, 1)

        coeff = RegressionCoefficient(
            variable_name=variable,
            coefficient=_round_val(b1, 6),
            standard_error=_round_val(se_b1, 6),
            t_statistic=_round_val(t_stat, 4),
            p_value=_round_val(p_val, 6),
            is_significant=is_sig,
        )

        model = RegressionModel(
            model_type=BaselineModelType.SINGLE_VARIABLE,
            intercept=_round_val(b0, 4),
            coefficients=[coeff],
            r_squared=_round_val(r_squared, 6),
            adjusted_r_squared=_round_val(adj_r_squared, 6),
            cv_rmse=_round_val(cv_rmse, 4),
            f_statistic=_round_val(f_stat, 4),
            f_p_value=_round_val(f_p, 6),
            n_observations=n,
            degrees_of_freedom=n - 2,
            aic=_round_val(aic, 4),
            bic=_round_val(bic, 4),
        )
        logger.debug(
            "Single-variable model (%s): b0=%.2f, b1=%.4f, R2=%.4f, "
            "CV(RMSE)=%.2f%%",
            variable, float(b0), float(b1), float(r_squared), float(cv_rmse),
        )
        return model

    def fit_multi_variable(
        self,
        data: List[BaselineDataPoint],
        variables: List[str],
    ) -> RegressionModel:
        """Fit a multi-variable OLS regression model.

        E = b0 + b1*X1 + b2*X2 + ... + bn*Xn

        Solves the normal equations (X'X)B = X'Y via Gauss elimination
        using pure Decimal arithmetic.

        Args:
            data: Baseline data points.
            variables: List of relevant variable names.

        Returns:
            RegressionModel with multiple coefficients.
        """
        n = len(data)
        p = len(variables)
        if n < p + 2:
            raise ValueError(
                f"Need >= {p + 2} data points for {p}-variable OLS, got {n}"
            )
        if p == 0:
            raise ValueError("At least one variable required")

        # Build matrices
        y_vec = [dp.energy_kwh for dp in data]
        x_matrix = self._build_design_matrix(data, variables)

        # Normal equations: (X'X) B = X'Y
        xtx = self._matrix_multiply_transpose(x_matrix, x_matrix, n, p + 1)
        xty = self._matrix_vector_multiply_transpose(x_matrix, y_vec, n, p + 1)

        # Solve via Gauss elimination
        beta = self._gauss_solve(xtx, xty, p + 1)

        # Predicted values and residuals
        predicted = self._matrix_vector_multiply(x_matrix, beta, n, p + 1)
        y_mean = _safe_divide(sum(y_vec, Decimal("0")), _decimal(n))

        ss_res = sum(
            ((y - yp) ** 2 for y, yp in zip(y_vec, predicted)), Decimal("0")
        )
        ss_tot = sum(
            ((y - y_mean) ** 2 for y in y_vec), Decimal("0")
        )
        ss_reg = ss_tot - ss_res

        r_squared = self.calculate_r_squared(y_vec, predicted)
        adj_r_squared = self._adjusted_r_squared(r_squared, n, p)
        cv_rmse = self.calculate_cv_rmse(y_vec, predicted)

        # MSE and coefficient standard errors
        mse = _safe_divide(ss_res, _decimal(n - p - 1))
        xtx_inv = self._matrix_inverse(xtx, p + 1)
        se_list = self._coefficient_std_errors(xtx_inv, mse, p + 1)

        # Build coefficients
        coefficients: List[RegressionCoefficient] = []
        for j, var_name in enumerate(variables):
            bj = beta[j + 1]
            se_j = se_list[j + 1]
            if se_j == Decimal("0") and bj != Decimal("0"):
                t_j = Decimal("9999")
                p_j = Decimal("0")
            else:
                t_j = _safe_divide(bj, se_j)
                p_j = self._approx_t_p_value(t_j, n - p - 1)
            is_sig = p_j < self._default_significance
            coefficients.append(RegressionCoefficient(
                variable_name=var_name,
                coefficient=_round_val(bj, 6),
                standard_error=_round_val(se_j, 6),
                t_statistic=_round_val(t_j, 4),
                p_value=_round_val(p_j, 6),
                is_significant=is_sig,
            ))

        # F-statistic
        f_stat, f_p = self._compute_f_statistic(ss_reg, ss_res, p, n)

        # AIC / BIC
        aic = self._compute_aic(ss_res, n, p)
        bic = self._compute_bic(ss_res, n, p)

        model = RegressionModel(
            model_type=BaselineModelType.MULTI_VARIABLE,
            intercept=_round_val(beta[0], 4),
            coefficients=coefficients,
            r_squared=_round_val(r_squared, 6),
            adjusted_r_squared=_round_val(adj_r_squared, 6),
            cv_rmse=_round_val(cv_rmse, 4),
            f_statistic=_round_val(f_stat, 4),
            f_p_value=_round_val(f_p, 6),
            n_observations=n,
            degrees_of_freedom=n - p - 1,
            aic=_round_val(aic, 4),
            bic=_round_val(bic, 4),
        )
        logger.debug(
            "Multi-variable model (%s): R2=%.4f, adj_R2=%.4f, "
            "CV(RMSE)=%.2f%%, F=%.2f",
            ", ".join(variables), float(r_squared), float(adj_r_squared),
            float(cv_rmse), float(f_stat),
        )
        return model

    def calculate_r_squared(
        self,
        actual: List[Decimal],
        predicted: List[Decimal],
    ) -> Decimal:
        """Calculate the coefficient of determination (R-squared).

        R^2 = 1 - SS_res / SS_tot

        Args:
            actual: Observed energy values.
            predicted: Model-predicted energy values.

        Returns:
            R-squared value between 0 and 1.
        """
        n = len(actual)
        if n == 0:
            return Decimal("0")

        mean_actual = _safe_divide(
            sum(actual, Decimal("0")), _decimal(n)
        )
        ss_res = sum(
            ((a - p) ** 2 for a, p in zip(actual, predicted)), Decimal("0")
        )
        ss_tot = sum(
            ((a - mean_actual) ** 2 for a in actual), Decimal("0")
        )

        if ss_tot == Decimal("0"):
            return Decimal("1") if ss_res == Decimal("0") else Decimal("0")

        r2 = Decimal("1") - _safe_divide(ss_res, ss_tot)
        # Clamp to [0, 1]
        return max(Decimal("0"), min(Decimal("1"), r2))

    def calculate_cv_rmse(
        self,
        actual: List[Decimal],
        predicted: List[Decimal],
    ) -> Decimal:
        """Calculate the Coefficient of Variation of RMSE.

        CV(RMSE) = RMSE / mean(actual) * 100

        Args:
            actual: Observed energy values.
            predicted: Model-predicted energy values.

        Returns:
            CV(RMSE) as a percentage.
        """
        n = len(actual)
        if n == 0:
            return Decimal("0")

        mean_actual = _safe_divide(
            sum(actual, Decimal("0")), _decimal(n)
        )
        mse = _safe_divide(
            sum(
                ((a - p) ** 2 for a, p in zip(actual, predicted)),
                Decimal("0"),
            ),
            _decimal(n),
        )
        rmse = _decimal_sqrt(mse)
        return _safe_divide(rmse * Decimal("100"), mean_actual)

    def calculate_f_statistic(
        self,
        model: RegressionModel,
        data: List[BaselineDataPoint],
    ) -> Tuple[Decimal, Decimal]:
        """Calculate the F-statistic for overall model significance.

        F = (SS_reg / p) / (SS_res / (n - p - 1))

        Args:
            model: Fitted regression model.
            data: Baseline data points used in fitting.

        Returns:
            Tuple of (F-statistic, p-value).
        """
        n = len(data)
        p = len(model.coefficients)

        y_vals = [dp.energy_kwh for dp in data]
        predicted = [
            self.calculate_expected_consumption(model, dp.variables)
            for dp in data
        ]
        y_mean = _safe_divide(sum(y_vals, Decimal("0")), _decimal(n))

        ss_res = sum(
            ((y - yp) ** 2 for y, yp in zip(y_vals, predicted)), Decimal("0")
        )
        ss_tot = sum(
            ((y - y_mean) ** 2 for y in y_vals), Decimal("0")
        )
        ss_reg = ss_tot - ss_res

        return self._compute_f_statistic(ss_reg, ss_res, p, n)

    def detect_outliers(
        self,
        data: List[BaselineDataPoint],
        method: str = "zscore",
        threshold: Decimal = Decimal("3.0"),
    ) -> List[int]:
        """Detect outlier data points by energy consumption.

        Supports z-score and IQR methods.

        Args:
            data: Baseline data points to evaluate.
            method: Detection method ('zscore' or 'iqr').
            threshold: z-score threshold or IQR multiplier.

        Returns:
            List of indices flagged as outliers.
        """
        if len(data) < 4:
            return []

        energies = [dp.energy_kwh for dp in data]

        if method == "iqr":
            return self._detect_outliers_iqr(energies, threshold)
        else:
            return self._detect_outliers_zscore(energies, threshold)

    def validate_baseline(
        self,
        result: BaselineResult,
        config: Optional[BaselineConfig] = None,
    ) -> BaselineValidation:
        """Validate a baseline model against adequacy criteria.

        Checks:
            1. R-squared >= configured minimum.
            2. CV(RMSE) <= configured maximum.
            3. All regression coefficients are significant.
            4. Residuals are approximately normally distributed.
            5. Sufficient data points for the model type.

        Args:
            result: Established baseline result to validate.
            config: Optional config with thresholds; uses engine defaults
                    if not provided.

        Returns:
            BaselineValidation with pass/fail for each criterion.
        """
        t0 = time.perf_counter()
        cfg = config or BaselineConfig()
        messages: List[str] = []
        model = result.model

        # Check 1 -- R-squared
        r2_threshold = cfg.min_r_squared
        r2_pass = model.r_squared >= r2_threshold
        if not r2_pass:
            messages.append(
                f"R-squared {model.r_squared} < threshold {r2_threshold}"
            )
        else:
            messages.append(
                f"R-squared {model.r_squared} >= threshold {r2_threshold} -- PASS"
            )

        # Check 2 -- CV(RMSE)
        cv_threshold = cfg.max_cv_rmse
        cv_pass = model.cv_rmse <= cv_threshold
        if not cv_pass:
            messages.append(
                f"CV(RMSE) {model.cv_rmse}% > threshold {cv_threshold}%"
            )
        else:
            messages.append(
                f"CV(RMSE) {model.cv_rmse}% <= threshold {cv_threshold}% -- PASS"
            )

        # Check 3 -- Significant variables
        if model.coefficients:
            all_sig = all(c.is_significant for c in model.coefficients)
            insig_vars = [
                c.variable_name for c in model.coefficients
                if not c.is_significant
            ]
            if not all_sig:
                messages.append(
                    f"Non-significant variables: {', '.join(insig_vars)}"
                )
            else:
                messages.append("All coefficients statistically significant -- PASS")
        else:
            all_sig = True
            messages.append("No coefficients to test (simple mean model)")
        sig_pass = all_sig

        # Check 4 -- Residual normality (Shapiro-Wilk approximation)
        norm_pass = self._check_residual_normality(result.residuals)
        if norm_pass:
            messages.append("Residuals pass normality check -- PASS")
        else:
            messages.append(
                "Residuals may not be normally distributed (Shapiro-Wilk proxy)"
            )

        # Check 5 -- Data sufficiency
        min_required = MINIMUM_DATA_REQUIREMENTS.get(
            model.model_type.value,
            cfg.min_data_points,
        )
        data_pass = model.n_observations >= min_required
        if not data_pass:
            messages.append(
                f"Observations {model.n_observations} < minimum {min_required}"
            )
        else:
            messages.append(
                f"Observations {model.n_observations} >= minimum {min_required} -- PASS"
            )

        # Overall adequacy
        # Simple mean models are exempt from R-squared check
        if model.model_type == BaselineModelType.SIMPLE_MEAN:
            is_adequate = cv_pass and data_pass
        else:
            is_adequate = r2_pass and cv_pass and sig_pass and data_pass

        # Adequacy rating
        rating = self._determine_adequacy_rating(model)

        validation = BaselineValidation(
            is_adequate=is_adequate,
            r_squared_pass=r2_pass,
            cv_rmse_pass=cv_pass,
            significant_variables_pass=sig_pass,
            residual_normality_pass=norm_pass,
            data_sufficiency_pass=data_pass,
            messages=messages,
            adequacy_rating=rating,
        )
        validation.provenance_hash = _compute_hash(validation)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Baseline validation: adequate=%s, rating=%s, R2_pass=%s, "
            "CV_pass=%s, sig_pass=%s (%.1f ms)",
            is_adequate, rating, r2_pass, cv_pass, sig_pass, elapsed_ms,
        )
        return validation

    def normalize_consumption(
        self,
        actual_kwh: Decimal,
        model: RegressionModel,
        baseline_conditions: Dict[str, Decimal],
        actual_conditions: Dict[str, Decimal],
    ) -> Decimal:
        """Normalise actual consumption to baseline conditions.

        E_norm = E_actual + (E_baseline_conditions - E_actual_conditions)

        This removes the effect of changes in relevant variables so that
        energy performance can be compared on a like-for-like basis.

        Args:
            actual_kwh: Actual measured energy consumption (kWh).
            model: Fitted baseline regression model.
            baseline_conditions: Relevant variable values at baseline.
            actual_conditions: Relevant variable values during actual period.

        Returns:
            Normalised energy consumption (kWh).
        """
        e_baseline = self.calculate_expected_consumption(
            model, baseline_conditions
        )
        e_actual_cond = self.calculate_expected_consumption(
            model, actual_conditions
        )
        adjustment = e_baseline - e_actual_cond
        normalised = actual_kwh + adjustment
        logger.debug(
            "Normalised: actual=%.2f, adjustment=%.2f, normalised=%.2f",
            float(actual_kwh), float(adjustment), float(normalised),
        )
        return _round_val(normalised, 2)

    def calculate_expected_consumption(
        self,
        model: RegressionModel,
        conditions: Dict[str, Decimal],
    ) -> Decimal:
        """Calculate expected energy consumption from a model and conditions.

        Evaluates: E = intercept + sum(coeff_i * condition_i)

        For change-point models the appropriate max(0, ...) transformation
        is applied to the condition values before multiplication.

        Args:
            model: Fitted regression model.
            conditions: Variable name -> value mapping.

        Returns:
            Expected energy consumption (kWh).
        """
        result = model.intercept

        for coeff in model.coefficients:
            var_name = coeff.variable_name
            raw_val = conditions.get(var_name, Decimal("0"))

            # Apply change-point transformation if applicable
            transformed = self._apply_cp_transform(
                raw_val, var_name, model, conditions
            )
            result = result + coeff.coefficient * transformed

        return _round_val(max(result, Decimal("0")), 2)

    def adjust_baseline(
        self,
        baseline: BaselineResult,
        trigger: AdjustmentTrigger,
        new_data: Dict[str, Any],
    ) -> BaselineResult:
        """Adjust an existing baseline due to a qualifying trigger event.

        Creates a new baseline result with adjusted model parameters and
        records the adjustment reason.

        Args:
            baseline: Current approved baseline to adjust.
            trigger: Event triggering the adjustment.
            new_data: Adjustment data.  Supported keys:
                - 'data_points': new or replacement data points
                - 'config': new BaselineConfig for refitting
                - 'description': textual explanation
                - 'approved_by': approver identifier
                - 'scale_factor': multiplicative adjustment factor
                - 'additive_offset': additive adjustment (kWh)

        Returns:
            New BaselineResult with status ADJUSTED.
        """
        t0 = time.perf_counter()
        logger.info(
            "Adjusting baseline %s: trigger=%s",
            baseline.baseline_id, trigger.value,
        )

        old_total = baseline.total_energy_kwh

        # Option A: Refit with new data
        if "data_points" in new_data and "config" in new_data:
            new_points = new_data["data_points"]
            new_config = new_data["config"]
            metadata = {
                "baseline_id": _new_uuid(),
                "enms_id": baseline.enms_id,
                "baseline_name": f"{baseline.baseline_name} (adjusted)",
                "energy_type": baseline.energy_type,
                "static_factors": [
                    sf.model_dump() if hasattr(sf, "model_dump") else sf
                    for sf in baseline.static_factors
                ],
            }
            adjusted = self.establish_baseline(new_points, new_config, metadata)
            adjusted.status = BaselineStatus.ADJUSTED
            return adjusted

        # Option B: Apply scale factor
        scale = _decimal(new_data.get("scale_factor", "1"))
        offset = _decimal(new_data.get("additive_offset", "0"))

        # Create adjusted model with scaled intercept
        adj_intercept = baseline.model.intercept * scale + offset
        adj_coefficients = []
        for coeff in baseline.model.coefficients:
            adj_coeff = RegressionCoefficient(
                variable_name=coeff.variable_name,
                coefficient=_round_val(coeff.coefficient * scale, 6),
                standard_error=coeff.standard_error,
                t_statistic=coeff.t_statistic,
                p_value=coeff.p_value,
                is_significant=coeff.is_significant,
            )
            adj_coefficients.append(adj_coeff)

        adj_model = RegressionModel(
            model_type=baseline.model.model_type,
            intercept=_round_val(adj_intercept, 4),
            coefficients=adj_coefficients,
            r_squared=baseline.model.r_squared,
            adjusted_r_squared=baseline.model.adjusted_r_squared,
            cv_rmse=baseline.model.cv_rmse,
            f_statistic=baseline.model.f_statistic,
            f_p_value=baseline.model.f_p_value,
            n_observations=baseline.model.n_observations,
            degrees_of_freedom=baseline.model.degrees_of_freedom,
            aic=baseline.model.aic,
            bic=baseline.model.bic,
        )

        new_total = _round_val(old_total * scale + offset * _decimal(
            baseline.model.n_observations
        ), 2)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        adjusted = BaselineResult(
            baseline_id=_new_uuid(),
            enms_id=baseline.enms_id,
            baseline_name=f"{baseline.baseline_name} (adjusted)",
            period_start=baseline.period_start,
            period_end=baseline.period_end,
            energy_type=baseline.energy_type,
            total_energy_kwh=new_total,
            model=adj_model,
            relevant_variables=baseline.relevant_variables,
            static_factors=baseline.static_factors,
            data_points=baseline.data_points,
            residuals=baseline.residuals,
            confidence_intervals=baseline.confidence_intervals,
            status=BaselineStatus.ADJUSTED,
            calculated_at=_utcnow(),
            processing_time_ms=elapsed_ms,
        )
        adjusted.provenance_hash = _compute_hash(adjusted)

        # Create adjustment record
        adjustment_record = BaselineAdjustment(
            baseline_id=baseline.baseline_id,
            trigger=trigger,
            description=new_data.get("description", ""),
            old_value=old_total,
            new_value=new_total,
            adjustment_date=date.today(),
            approved_by=new_data.get("approved_by", ""),
        )
        adjustment_record.provenance_hash = _compute_hash(adjustment_record)

        logger.info(
            "Baseline adjusted: old_id=%s, new_id=%s, trigger=%s, "
            "old_total=%.2f, new_total=%.2f, hash=%s",
            baseline.baseline_id, adjusted.baseline_id,
            trigger.value, float(old_total), float(new_total),
            adjusted.provenance_hash[:16],
        )
        return adjusted

    def compare_models(
        self,
        models: List[RegressionModel],
    ) -> ModelComparison:
        """Compare multiple candidate baseline models and recommend the best.

        Selection criteria (weighted):
            1. Lowest AIC (primary information criterion).
            2. Lowest BIC (penalises complexity more).
            3. Highest adjusted R-squared.
            4. Lowest CV(RMSE).

        Args:
            models: List of candidate RegressionModel objects.

        Returns:
            ModelComparison with ranking and recommendation.
        """
        t0 = time.perf_counter()
        if not models:
            return ModelComparison(
                models=[],
                best_model_index=0,
                best_model_reason="No models provided",
            )

        if len(models) == 1:
            entry = self._model_summary(models[0], 0)
            result = ModelComparison(
                models=[entry],
                best_model_index=0,
                best_model_reason="Only one model provided",
            )
            result.provenance_hash = _compute_hash(result)
            return result

        # Build comparison data
        entries: List[Dict[str, Any]] = []
        for idx, m in enumerate(models):
            entries.append(self._model_summary(m, idx))

        # Score each model (lower is better for AIC/BIC/CV; higher for adj R2)
        scores: List[Decimal] = []
        for idx, m in enumerate(models):
            # Rank-based scoring: best gets 0, worst gets len-1
            aic_rank = self._rank_position(
                [mm.aic for mm in models], m.aic, ascending=True
            )
            bic_rank = self._rank_position(
                [mm.bic for mm in models], m.bic, ascending=True
            )
            r2_rank = self._rank_position(
                [mm.adjusted_r_squared for mm in models],
                m.adjusted_r_squared, ascending=False,
            )
            cv_rank = self._rank_position(
                [mm.cv_rmse for mm in models], m.cv_rmse, ascending=True
            )
            # Weighted total (AIC=3, BIC=2, adj_R2=3, CV(RMSE)=2)
            total = (
                _decimal(aic_rank) * Decimal("3")
                + _decimal(bic_rank) * Decimal("2")
                + _decimal(r2_rank) * Decimal("3")
                + _decimal(cv_rank) * Decimal("2")
            )
            scores.append(total)

        best_idx = int(min(range(len(scores)), key=lambda i: scores[i]))
        best_model = models[best_idx]

        reason_parts = [
            f"Model {best_idx} ({best_model.model_type.value}) selected",
            f"AIC={best_model.aic}",
            f"BIC={best_model.bic}",
            f"adj_R2={best_model.adjusted_r_squared}",
            f"CV(RMSE)={best_model.cv_rmse}%",
        ]
        reason = "; ".join(reason_parts)

        # Comparison matrix
        matrix: Dict[str, Any] = {
            "aic": [str(m.aic) for m in models],
            "bic": [str(m.bic) for m in models],
            "r_squared": [str(m.r_squared) for m in models],
            "adjusted_r_squared": [str(m.adjusted_r_squared) for m in models],
            "cv_rmse": [str(m.cv_rmse) for m in models],
            "f_statistic": [str(m.f_statistic) for m in models],
            "scores": [str(s) for s in scores],
        }

        result = ModelComparison(
            models=entries,
            best_model_index=best_idx,
            best_model_reason=reason,
            comparison_matrix=matrix,
        )
        result.provenance_hash = _compute_hash(result)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Model comparison: %d candidates, best=%d (%s), score=%.1f (%.1f ms)",
            len(models), best_idx, best_model.model_type.value,
            float(scores[best_idx]), elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal - Model Fitting                                             #
    # ------------------------------------------------------------------ #

    def _fit_model(
        self,
        data: List[BaselineDataPoint],
        config: BaselineConfig,
    ) -> RegressionModel:
        """Dispatch to the appropriate fitting method based on model_type.

        Args:
            data: Clean baseline data points.
            config: Baseline configuration.

        Returns:
            Fitted RegressionModel.
        """
        model_type = config.model_type

        if model_type == BaselineModelType.SIMPLE_MEAN:
            return self.fit_simple_mean(data)

        if model_type == BaselineModelType.SINGLE_VARIABLE:
            variables = self._infer_variables(data)
            if not variables:
                logger.warning(
                    "No variables found; falling back to simple_mean"
                )
                return self.fit_simple_mean(data)
            return self.fit_single_variable(data, variables[0])

        if model_type == BaselineModelType.MULTI_VARIABLE:
            variables = self._infer_variables(data)
            if len(variables) < 2:
                logger.warning(
                    "Fewer than 2 variables; falling back to single_variable"
                )
                if variables:
                    return self.fit_single_variable(data, variables[0])
                return self.fit_simple_mean(data)
            return self.fit_multi_variable(data, variables)

        if model_type == BaselineModelType.DEGREE_DAY:
            return self._fit_degree_day(data)

        if model_type == BaselineModelType.CHANGE_POINT_3P:
            return self._fit_change_point_3p(data)

        if model_type == BaselineModelType.CHANGE_POINT_4P:
            return self._fit_change_point_4p(data)

        if model_type == BaselineModelType.CHANGE_POINT_5P:
            return self._fit_change_point_5p(data)

        raise ValueError(f"Unsupported model type: {model_type.value}")

    def _fit_degree_day(
        self,
        data: List[BaselineDataPoint],
    ) -> RegressionModel:
        """Fit a degree-day regression model.

        Looks for 'hdd' and/or 'cdd' variables.  Fits single or
        multi-variable model depending on availability.

        Args:
            data: Baseline data points.

        Returns:
            RegressionModel configured as DEGREE_DAY type.
        """
        variables = self._infer_variables(data)
        dd_vars = [
            v for v in variables
            if v.lower() in ("hdd", "cdd", "hdd_18", "cdd_18",
                             "hdd_65", "cdd_65", "heating_degree_days",
                             "cooling_degree_days")
        ]

        if not dd_vars:
            logger.warning(
                "No degree-day variables found; using first available variable"
            )
            dd_vars = variables[:1] if variables else []

        if len(dd_vars) == 0:
            return self.fit_simple_mean(data)
        elif len(dd_vars) == 1:
            model = self.fit_single_variable(data, dd_vars[0])
        else:
            model = self.fit_multi_variable(data, dd_vars[:2])

        # Override model_type to DEGREE_DAY
        model.model_type = BaselineModelType.DEGREE_DAY
        return model

    def _fit_change_point_3p(
        self,
        data: List[BaselineDataPoint],
    ) -> RegressionModel:
        """Fit a 3-parameter change-point model.

        E = b0 + b1 * max(X - CP, 0)

        Searches for the optimal change point (CP) by iterating over
        candidate values between the 10th and 90th percentile of X.

        Args:
            data: Baseline data points.

        Returns:
            RegressionModel configured as CHANGE_POINT_3P type.
        """
        variables = self._infer_variables(data)
        if not variables:
            return self.fit_simple_mean(data)
        var_name = variables[0]

        x_vals = sorted([
            dp.variables.get(var_name, Decimal("0")) for dp in data
        ])
        n = len(x_vals)

        # Candidate change points: 10th to 90th percentile
        low_idx = max(0, int(n * 0.1))
        high_idx = min(n - 1, int(n * 0.9))
        candidates = x_vals[low_idx:high_idx + 1]

        # Remove duplicates and ensure reasonable set
        candidates = sorted(set(candidates))
        if not candidates:
            candidates = [x_vals[n // 2]]

        best_model: Optional[RegressionModel] = None
        best_r2 = Decimal("-1")

        for cp in candidates:
            # Transform: max(X - CP, 0)
            transformed = self._create_cp_data(data, var_name, cp, "heating")
            try:
                model = self.fit_single_variable(transformed, f"{var_name}_cp")
                if model.r_squared > best_r2:
                    best_r2 = model.r_squared
                    best_model = model
            except (ValueError, ZeroDivisionError):
                continue

        if best_model is None:
            return self.fit_single_variable(data, var_name)

        best_model.model_type = BaselineModelType.CHANGE_POINT_3P
        return best_model

    def _fit_change_point_4p(
        self,
        data: List[BaselineDataPoint],
    ) -> RegressionModel:
        """Fit a 4-parameter change-point model.

        E = b0 + b1 * max(CP - X, 0) + b2 * max(X - CP, 0)

        Args:
            data: Baseline data points.

        Returns:
            RegressionModel configured as CHANGE_POINT_4P type.
        """
        variables = self._infer_variables(data)
        if not variables:
            return self.fit_simple_mean(data)
        var_name = variables[0]

        x_vals = sorted([
            dp.variables.get(var_name, Decimal("0")) for dp in data
        ])
        n = len(x_vals)

        low_idx = max(0, int(n * 0.15))
        high_idx = min(n - 1, int(n * 0.85))
        candidates = sorted(set(x_vals[low_idx:high_idx + 1]))
        if not candidates:
            candidates = [x_vals[n // 2]]

        best_model: Optional[RegressionModel] = None
        best_r2 = Decimal("-1")

        for cp in candidates:
            transformed = self._create_cp4_data(data, var_name, cp)
            try:
                model = self.fit_multi_variable(
                    transformed,
                    [f"{var_name}_heat", f"{var_name}_cool"],
                )
                if model.r_squared > best_r2:
                    best_r2 = model.r_squared
                    best_model = model
            except (ValueError, ZeroDivisionError):
                continue

        if best_model is None:
            return self.fit_single_variable(data, var_name)

        best_model.model_type = BaselineModelType.CHANGE_POINT_4P
        return best_model

    def _fit_change_point_5p(
        self,
        data: List[BaselineDataPoint],
    ) -> RegressionModel:
        """Fit a 5-parameter change-point model.

        E = b0 + b1 * max(CPh - X, 0) + b2 * max(X - CPc, 0)

        Two separate change points (heating CPh and cooling CPc).

        Args:
            data: Baseline data points.

        Returns:
            RegressionModel configured as CHANGE_POINT_5P type.
        """
        variables = self._infer_variables(data)
        if not variables:
            return self.fit_simple_mean(data)
        var_name = variables[0]

        x_vals = sorted([
            dp.variables.get(var_name, Decimal("0")) for dp in data
        ])
        n = len(x_vals)

        low_idx = max(0, int(n * 0.15))
        high_idx = min(n - 1, int(n * 0.85))
        mid_idx = n // 2
        heat_candidates = sorted(set(x_vals[low_idx:mid_idx]))
        cool_candidates = sorted(set(x_vals[mid_idx:high_idx + 1]))

        if not heat_candidates:
            heat_candidates = [x_vals[low_idx]]
        if not cool_candidates:
            cool_candidates = [x_vals[high_idx]]

        best_model: Optional[RegressionModel] = None
        best_r2 = Decimal("-1")

        # Grid search over (CPh, CPc) combinations
        # Limit to avoid combinatorial explosion
        step_h = max(1, len(heat_candidates) // 5)
        step_c = max(1, len(cool_candidates) // 5)
        sampled_heat = heat_candidates[::step_h]
        sampled_cool = cool_candidates[::step_c]

        for cp_h in sampled_heat:
            for cp_c in sampled_cool:
                if cp_h >= cp_c:
                    continue
                transformed = self._create_cp5_data(
                    data, var_name, cp_h, cp_c
                )
                try:
                    model = self.fit_multi_variable(
                        transformed,
                        [f"{var_name}_heat", f"{var_name}_cool"],
                    )
                    if model.r_squared > best_r2:
                        best_r2 = model.r_squared
                        best_model = model
                except (ValueError, ZeroDivisionError):
                    continue

        if best_model is None:
            return self._fit_change_point_4p(data)

        best_model.model_type = BaselineModelType.CHANGE_POINT_5P
        return best_model

    # ------------------------------------------------------------------ #
    # Internal - Change-Point Data Transformations                         #
    # ------------------------------------------------------------------ #

    def _create_cp_data(
        self,
        data: List[BaselineDataPoint],
        var_name: str,
        change_point: Decimal,
        mode: str = "heating",
    ) -> List[BaselineDataPoint]:
        """Create transformed data for 3P change-point model.

        Args:
            data: Original data points.
            var_name: Variable name to transform.
            change_point: Change-point value.
            mode: 'heating' uses max(X-CP,0), 'cooling' uses max(CP-X,0).

        Returns:
            New data points with transformed variable.
        """
        result: List[BaselineDataPoint] = []
        cp_var = f"{var_name}_cp"
        for dp in data:
            x = dp.variables.get(var_name, Decimal("0"))
            if mode == "heating":
                transformed = max(x - change_point, Decimal("0"))
            else:
                transformed = max(change_point - x, Decimal("0"))
            new_vars = dict(dp.variables)
            new_vars[cp_var] = transformed
            result.append(BaselineDataPoint(
                period_start=dp.period_start,
                period_end=dp.period_end,
                energy_kwh=dp.energy_kwh,
                variables=new_vars,
            ))
        return result

    def _create_cp4_data(
        self,
        data: List[BaselineDataPoint],
        var_name: str,
        change_point: Decimal,
    ) -> List[BaselineDataPoint]:
        """Create transformed data for 4P change-point model.

        Adds two variables: max(CP-X, 0) and max(X-CP, 0).

        Args:
            data: Original data points.
            var_name: Variable name to transform.
            change_point: Change-point value.

        Returns:
            New data points with two transformed variables.
        """
        result: List[BaselineDataPoint] = []
        heat_var = f"{var_name}_heat"
        cool_var = f"{var_name}_cool"
        for dp in data:
            x = dp.variables.get(var_name, Decimal("0"))
            new_vars = dict(dp.variables)
            new_vars[heat_var] = max(change_point - x, Decimal("0"))
            new_vars[cool_var] = max(x - change_point, Decimal("0"))
            result.append(BaselineDataPoint(
                period_start=dp.period_start,
                period_end=dp.period_end,
                energy_kwh=dp.energy_kwh,
                variables=new_vars,
            ))
        return result

    def _create_cp5_data(
        self,
        data: List[BaselineDataPoint],
        var_name: str,
        cp_heat: Decimal,
        cp_cool: Decimal,
    ) -> List[BaselineDataPoint]:
        """Create transformed data for 5P change-point model.

        Adds two variables: max(CPh-X, 0) and max(X-CPc, 0).

        Args:
            data: Original data points.
            var_name: Variable name to transform.
            cp_heat: Heating change-point.
            cp_cool: Cooling change-point (must be > cp_heat).

        Returns:
            New data points with two transformed variables.
        """
        result: List[BaselineDataPoint] = []
        heat_var = f"{var_name}_heat"
        cool_var = f"{var_name}_cool"
        for dp in data:
            x = dp.variables.get(var_name, Decimal("0"))
            new_vars = dict(dp.variables)
            new_vars[heat_var] = max(cp_heat - x, Decimal("0"))
            new_vars[cool_var] = max(x - cp_cool, Decimal("0"))
            result.append(BaselineDataPoint(
                period_start=dp.period_start,
                period_end=dp.period_end,
                energy_kwh=dp.energy_kwh,
                variables=new_vars,
            ))
        return result

    def _apply_cp_transform(
        self,
        raw_val: Decimal,
        var_name: str,
        model: RegressionModel,
        conditions: Dict[str, Decimal],
    ) -> Decimal:
        """Apply change-point transformation when calculating expected E.

        For non-change-point models, returns the raw value unchanged.

        Args:
            raw_val: Raw variable value from conditions.
            var_name: Variable name.
            model: Regression model.
            conditions: Full conditions dictionary.

        Returns:
            Transformed value.
        """
        # If the variable name already contains _cp, _heat, _cool
        # the transformation was already applied in conditions
        if any(suffix in var_name for suffix in ("_cp", "_heat", "_cool")):
            return raw_val
        return raw_val

    # ------------------------------------------------------------------ #
    # Internal - Linear Algebra (Decimal-Based)                            #
    # ------------------------------------------------------------------ #

    def _build_design_matrix(
        self,
        data: List[BaselineDataPoint],
        variables: List[str],
    ) -> List[List[Decimal]]:
        """Build the design matrix X with intercept column.

        Returns an n x (p+1) matrix where the first column is all 1s.

        Args:
            data: Baseline data points.
            variables: Variable names for columns.

        Returns:
            Design matrix as list of rows.
        """
        matrix: List[List[Decimal]] = []
        for dp in data:
            row = [Decimal("1")]  # intercept
            for var in variables:
                row.append(dp.variables.get(var, Decimal("0")))
            matrix.append(row)
        return matrix

    def _matrix_multiply_transpose(
        self,
        a: List[List[Decimal]],
        b: List[List[Decimal]],
        n: int,
        cols: int,
    ) -> List[List[Decimal]]:
        """Compute A' * B where A and B are n x cols matrices.

        Result is a cols x cols matrix.

        Args:
            a: First matrix (n x cols).
            b: Second matrix (n x cols).
            n: Number of rows.
            cols: Number of columns.

        Returns:
            Result matrix (cols x cols).
        """
        result: List[List[Decimal]] = [
            [Decimal("0")] * cols for _ in range(cols)
        ]
        for i in range(cols):
            for j in range(cols):
                s = Decimal("0")
                for k in range(n):
                    s += a[k][i] * b[k][j]
                result[i][j] = s
        return result

    def _matrix_vector_multiply_transpose(
        self,
        a: List[List[Decimal]],
        y: List[Decimal],
        n: int,
        cols: int,
    ) -> List[Decimal]:
        """Compute A' * y where A is n x cols and y is n x 1.

        Args:
            a: Matrix (n x cols).
            y: Vector of length n.
            n: Number of rows.
            cols: Number of columns.

        Returns:
            Result vector of length cols.
        """
        result: List[Decimal] = []
        for j in range(cols):
            s = Decimal("0")
            for i in range(n):
                s += a[i][j] * y[i]
            result.append(s)
        return result

    def _matrix_vector_multiply(
        self,
        a: List[List[Decimal]],
        b: List[Decimal],
        n: int,
        cols: int,
    ) -> List[Decimal]:
        """Compute A * b where A is n x cols and b is cols x 1.

        Args:
            a: Matrix (n x cols).
            b: Vector of length cols.
            n: Number of rows.
            cols: Number of columns.

        Returns:
            Result vector of length n.
        """
        result: List[Decimal] = []
        for i in range(n):
            s = Decimal("0")
            for j in range(cols):
                s += a[i][j] * b[j]
            result.append(s)
        return result

    def _gauss_solve(
        self,
        a_matrix: List[List[Decimal]],
        b_vector: List[Decimal],
        size: int,
    ) -> List[Decimal]:
        """Solve a linear system Ax = b via Gauss elimination with pivoting.

        Args:
            a_matrix: Coefficient matrix (size x size).
            b_vector: Right-hand-side vector of length size.
            size: Dimension of the system.

        Returns:
            Solution vector of length size.
        """
        # Create augmented matrix [A|b]
        aug: List[List[Decimal]] = []
        for i in range(size):
            row = list(a_matrix[i]) + [b_vector[i]]
            aug.append(row)

        # Forward elimination with partial pivoting
        for col in range(size):
            # Find pivot
            max_val = abs(aug[col][col])
            max_row = col
            for row in range(col + 1, size):
                if abs(aug[row][col]) > max_val:
                    max_val = abs(aug[row][col])
                    max_row = row
            aug[col], aug[max_row] = aug[max_row], aug[col]

            pivot = aug[col][col]
            if pivot == Decimal("0"):
                # Singular matrix - use small value to prevent crash
                pivot = Decimal("1E-20")
                aug[col][col] = pivot

            for row in range(col + 1, size):
                factor = aug[row][col] / pivot
                for j in range(col, size + 1):
                    aug[row][j] -= factor * aug[col][j]

        # Back substitution
        x = [Decimal("0")] * size
        for i in range(size - 1, -1, -1):
            s = aug[i][size]
            for j in range(i + 1, size):
                s -= aug[i][j] * x[j]
            x[i] = _safe_divide(s, aug[i][i])

        return x

    def _matrix_inverse(
        self,
        matrix: List[List[Decimal]],
        size: int,
    ) -> List[List[Decimal]]:
        """Compute the inverse of a square matrix via Gauss-Jordan.

        Args:
            matrix: Square matrix (size x size).
            size: Dimension.

        Returns:
            Inverse matrix (size x size).
        """
        # Augment with identity
        aug: List[List[Decimal]] = []
        for i in range(size):
            row = list(matrix[i]) + [
                Decimal("1") if j == i else Decimal("0")
                for j in range(size)
            ]
            aug.append(row)

        # Gauss-Jordan elimination
        for col in range(size):
            # Pivot
            max_val = abs(aug[col][col])
            max_row = col
            for row in range(col + 1, size):
                if abs(aug[row][col]) > max_val:
                    max_val = abs(aug[row][col])
                    max_row = row
            aug[col], aug[max_row] = aug[max_row], aug[col]

            pivot = aug[col][col]
            if pivot == Decimal("0"):
                pivot = Decimal("1E-20")

            # Scale pivot row
            for j in range(2 * size):
                aug[col][j] = aug[col][j] / pivot

            # Eliminate column in all other rows
            for row in range(size):
                if row == col:
                    continue
                factor = aug[row][col]
                for j in range(2 * size):
                    aug[row][j] -= factor * aug[col][j]

        # Extract inverse
        inv: List[List[Decimal]] = []
        for i in range(size):
            inv.append(aug[i][size:])
        return inv

    def _coefficient_std_errors(
        self,
        xtx_inv: List[List[Decimal]],
        mse: Decimal,
        size: int,
    ) -> List[Decimal]:
        """Compute standard errors of regression coefficients.

        SE(bj) = sqrt(MSE * (X'X)^-1[j,j])

        Args:
            xtx_inv: Inverse of X'X matrix.
            mse: Mean squared error.
            size: Number of coefficients (including intercept).

        Returns:
            List of standard errors.
        """
        se_list: List[Decimal] = []
        for j in range(size):
            var_bj = mse * xtx_inv[j][j]
            se_list.append(_decimal_sqrt(max(var_bj, Decimal("0"))))
        return se_list

    # ------------------------------------------------------------------ #
    # Internal - Statistics                                                #
    # ------------------------------------------------------------------ #

    def _adjusted_r_squared(
        self,
        r_squared: Decimal,
        n: int,
        p: int,
    ) -> Decimal:
        """Calculate adjusted R-squared.

        adj_R^2 = 1 - (1 - R^2) * (n - 1) / (n - p - 1)

        Args:
            r_squared: Unadjusted R-squared.
            n: Number of observations.
            p: Number of predictor variables.

        Returns:
            Adjusted R-squared.
        """
        denom = n - p - 1
        if denom <= 0:
            return r_squared
        adj = Decimal("1") - (
            (Decimal("1") - r_squared) * _decimal(n - 1) / _decimal(denom)
        )
        return max(adj, Decimal("0"))

    def _compute_f_statistic(
        self,
        ss_reg: Decimal,
        ss_res: Decimal,
        p: int,
        n: int,
    ) -> Tuple[Decimal, Decimal]:
        """Calculate F-statistic and approximate p-value.

        F = (SS_reg / p) / (SS_res / (n - p - 1))

        Args:
            ss_reg: Regression sum of squares.
            ss_res: Residual sum of squares.
            p: Number of predictors.
            n: Number of observations.

        Returns:
            Tuple of (F-statistic, approximate p-value).
        """
        if p == 0 or n <= p + 1:
            return Decimal("0"), Decimal("1")

        ms_reg = _safe_divide(ss_reg, _decimal(p))
        ms_res = _safe_divide(ss_res, _decimal(n - p - 1))
        f_stat = _safe_divide(ms_reg, ms_res)

        # Approximate p-value using F-distribution CDF approximation
        f_p = self._approx_f_p_value(f_stat, p, n - p - 1)
        return f_stat, f_p

    def _compute_aic(
        self,
        ss_res: Decimal,
        n: int,
        p: int,
    ) -> Decimal:
        """Compute Akaike Information Criterion.

        AIC = n * ln(SS_res / n) + 2 * (p + 1)

        Args:
            ss_res: Residual sum of squares.
            n: Number of observations.
            p: Number of predictor variables.

        Returns:
            AIC value.
        """
        if n == 0:
            return Decimal("0")
        ratio = _safe_divide(ss_res, _decimal(n), Decimal("1E-20"))
        return _decimal(n) * _decimal_ln(ratio) + Decimal("2") * _decimal(p + 1)

    def _compute_bic(
        self,
        ss_res: Decimal,
        n: int,
        p: int,
    ) -> Decimal:
        """Compute Bayesian Information Criterion.

        BIC = n * ln(SS_res / n) + (p + 1) * ln(n)

        Args:
            ss_res: Residual sum of squares.
            n: Number of observations.
            p: Number of predictor variables.

        Returns:
            BIC value.
        """
        if n == 0:
            return Decimal("0")
        ratio = _safe_divide(ss_res, _decimal(n), Decimal("1E-20"))
        return (
            _decimal(n) * _decimal_ln(ratio)
            + _decimal(p + 1) * _decimal_ln(_decimal(n))
        )

    def _approx_t_p_value(
        self,
        t_stat: Decimal,
        df: int,
    ) -> Decimal:
        """Approximate two-sided p-value from t-statistic.

        Uses a rational approximation to the Student-t CDF based on
        the normal approximation with Cornish-Fisher correction.

        Args:
            t_stat: Observed t-statistic.
            df: Degrees of freedom.

        Returns:
            Approximate two-sided p-value.
        """
        if df <= 0:
            return Decimal("1")

        t_abs = float(abs(t_stat))
        d = float(df)

        # Approximation: p ~ 2 * (1 - Phi(t * sqrt(d/(d + t^2 * correction))))
        # For large df this converges to the normal distribution.
        correction = 1.0 - 1.0 / (4.0 * d) + 1.0 / (32.0 * d * d)
        z = t_abs * math.sqrt(d / (d + t_abs * t_abs)) * correction

        # Standard normal CDF approximation (Abramowitz & Stegun 26.2.17)
        p_one_tail = self._normal_cdf_complement(z)
        p_two_tail = 2.0 * p_one_tail

        result = _decimal(min(max(p_two_tail, 0.0), 1.0))
        return result

    def _approx_f_p_value(
        self,
        f_stat: Decimal,
        df1: int,
        df2: int,
    ) -> Decimal:
        """Approximate p-value from F-statistic.

        Uses a transformation to the normal distribution for large df.

        Args:
            f_stat: Observed F-statistic.
            df1: Numerator degrees of freedom.
            df2: Denominator degrees of freedom.

        Returns:
            Approximate p-value.
        """
        if df1 <= 0 or df2 <= 0:
            return Decimal("1")

        f = float(f_stat)
        d1 = float(df1)
        d2 = float(df2)

        if f <= 0:
            return Decimal("1")

        # Fisher's transformation to approximately normal
        try:
            x = d2 / (d2 + d1 * f)
            # Beta-distribution approximation via normal
            a = d2 / 2.0
            b = d1 / 2.0
            mu = a / (a + b)
            sigma = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.0)))
            z = (x - mu) / sigma if sigma > 0 else 0.0
            p_val = self._normal_cdf_complement(abs(z))
        except (ValueError, ZeroDivisionError, OverflowError):
            p_val = 0.5

        return _decimal(min(max(p_val, 0.0), 1.0))

    @staticmethod
    def _normal_cdf_complement(z: float) -> float:
        """Compute 1 - Phi(z) for the standard normal distribution.

        Uses the Abramowitz & Stegun approximation (formula 26.2.17)
        with maximum error < 7.5e-8.

        Args:
            z: Standard normal z-value (should be >= 0).

        Returns:
            Right-tail probability.
        """
        if z < 0:
            return 1.0 - EnergyBaselineEngine._normal_cdf_complement(-z)
        if z > 8.0:
            return 0.0

        # Constants from A&S 26.2.17
        p = 0.2316419
        b1 = 0.319381530
        b2 = -0.356563782
        b3 = 1.781477937
        b4 = -1.821255978
        b5 = 1.330274429

        t = 1.0 / (1.0 + p * z)
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t

        phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        q = phi * (b1 * t + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5)
        return max(0.0, min(1.0, q))

    def _check_residual_normality(
        self,
        residuals: List[Decimal],
    ) -> bool:
        """Check residual normality using a Shapiro-Wilk proxy.

        Uses skewness and kurtosis to approximate the Shapiro-Wilk W
        statistic.  The model passes if W_approx >= 0.90.

        W_approx = 1 - (skew^2 + (kurt-3)^2 / 4) / (6 * n)

        Args:
            residuals: List of model residuals.

        Returns:
            True if residuals appear approximately normal.
        """
        n = len(residuals)
        if n < 8:
            # Too few points for meaningful normality test
            return True

        # Compute mean
        r_float = [float(r) for r in residuals]
        mean_r = sum(r_float) / n

        # Compute central moments
        m2 = sum((r - mean_r) ** 2 for r in r_float) / n
        m3 = sum((r - mean_r) ** 3 for r in r_float) / n
        m4 = sum((r - mean_r) ** 4 for r in r_float) / n

        if m2 == 0:
            return True

        sd = math.sqrt(m2)
        skewness = m3 / (sd ** 3) if sd > 0 else 0.0
        kurtosis = m4 / (sd ** 4) if sd > 0 else 3.0

        # Approximate W statistic
        w_approx = 1.0 - (
            skewness ** 2 + (kurtosis - 3.0) ** 2 / 4.0
        ) / (6.0 * n)

        return w_approx >= 0.90

    # ------------------------------------------------------------------ #
    # Internal - Outlier Detection                                         #
    # ------------------------------------------------------------------ #

    def _detect_outliers_zscore(
        self,
        energies: List[Decimal],
        threshold: Decimal,
    ) -> List[int]:
        """Detect outliers using z-score method.

        Args:
            energies: List of energy values.
            threshold: z-score threshold (typically 3.0).

        Returns:
            Indices of outlier observations.
        """
        n = len(energies)
        if n < 4:
            return []

        mean_e = _safe_divide(sum(energies, Decimal("0")), _decimal(n))
        variance = _safe_divide(
            sum(((e - mean_e) ** 2 for e in energies), Decimal("0")),
            _decimal(n - 1),
        )
        std_dev = _decimal_sqrt(variance)

        if std_dev == Decimal("0"):
            return []

        outliers: List[int] = []
        for idx, e in enumerate(energies):
            z = abs(e - mean_e) / std_dev
            if z > threshold:
                outliers.append(idx)

        return outliers

    def _detect_outliers_iqr(
        self,
        energies: List[Decimal],
        multiplier: Decimal,
    ) -> List[int]:
        """Detect outliers using IQR method.

        Args:
            energies: List of energy values.
            multiplier: IQR multiplier (typically 1.5 or 3.0).

        Returns:
            Indices of outlier observations.
        """
        n = len(energies)
        if n < 4:
            return []

        sorted_e = sorted(energies)
        q1_idx = n // 4
        q3_idx = (3 * n) // 4
        q1 = sorted_e[q1_idx]
        q3 = sorted_e[q3_idx]
        iqr = q3 - q1

        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        outliers: List[int] = []
        for idx, e in enumerate(energies):
            if e < lower or e > upper:
                outliers.append(idx)

        return outliers

    # ------------------------------------------------------------------ #
    # Internal - Residuals & Confidence Intervals                          #
    # ------------------------------------------------------------------ #

    def _compute_residuals(
        self,
        data: List[BaselineDataPoint],
        model: RegressionModel,
    ) -> List[Decimal]:
        """Compute residuals (actual - predicted) for each data point.

        Args:
            data: Baseline data points.
            model: Fitted regression model.

        Returns:
            List of residuals.
        """
        residuals: List[Decimal] = []
        for dp in data:
            predicted = self.calculate_expected_consumption(model, dp.variables)
            residuals.append(dp.energy_kwh - predicted)
        return residuals

    def _compute_confidence_intervals(
        self,
        model: RegressionModel,
        residuals: List[Decimal],
        alpha: Decimal,
    ) -> Dict[str, Any]:
        """Compute confidence intervals for model coefficients.

        Uses the t-distribution approximation.

        Args:
            model: Fitted regression model.
            residuals: Model residuals.
            alpha: Significance level (e.g. 0.05 for 95% CI).

        Returns:
            Dictionary with confidence interval information.
        """
        n = len(residuals)
        if n <= 2:
            return {"note": "Insufficient data for confidence intervals"}

        # Approximate t-critical value for (1 - alpha/2) quantile
        # Using a simple approximation for common alpha values
        t_crit = self._approx_t_critical(alpha, model.degrees_of_freedom)

        intervals: Dict[str, Any] = {
            "alpha": str(alpha),
            "confidence_level": str(Decimal("1") - alpha),
            "t_critical": str(_round_val(t_crit, 4)),
            "intercept": {
                "value": str(model.intercept),
                "lower": str(_round_val(
                    model.intercept - t_crit * Decimal("1"), 4
                )),
                "upper": str(_round_val(
                    model.intercept + t_crit * Decimal("1"), 4
                )),
            },
            "coefficients": {},
        }

        for coeff in model.coefficients:
            margin = t_crit * coeff.standard_error
            intervals["coefficients"][coeff.variable_name] = {
                "value": str(coeff.coefficient),
                "standard_error": str(coeff.standard_error),
                "lower": str(_round_val(coeff.coefficient - margin, 6)),
                "upper": str(_round_val(coeff.coefficient + margin, 6)),
            }

        # Model-level prediction interval statistics
        mse = _safe_divide(
            sum(((r ** 2) for r in residuals), Decimal("0")),
            _decimal(max(n - len(model.coefficients) - 1, 1)),
        )
        rmse = _decimal_sqrt(mse)
        intervals["prediction"] = {
            "rmse": str(_round_val(rmse, 4)),
            "mse": str(_round_val(mse, 4)),
        }

        return intervals

    def _approx_t_critical(
        self,
        alpha: Decimal,
        df: int,
    ) -> Decimal:
        """Approximate t-critical value for (1 - alpha/2) quantile.

        Uses a lookup table for common values and interpolation for others.

        Args:
            alpha: Significance level.
            df: Degrees of freedom.

        Returns:
            Approximate t-critical value.
        """
        if df <= 0:
            return Decimal("2.0")

        # Common t-critical values (two-tailed)
        # Format: {alpha: {df: t_crit}} for small df, then asymptotic
        alpha_f = float(alpha)

        # For alpha = 0.05 (95% CI)
        if alpha_f <= 0.01:
            # 99% confidence
            if df >= 120:
                return Decimal("2.576")
            elif df >= 30:
                return Decimal("2.750")
            elif df >= 10:
                return Decimal("3.169")
            else:
                return Decimal("3.500")
        elif alpha_f <= 0.05:
            # 95% confidence
            if df >= 120:
                return Decimal("1.960")
            elif df >= 30:
                return Decimal("2.042")
            elif df >= 10:
                return Decimal("2.228")
            else:
                return Decimal("2.571")
        elif alpha_f <= 0.10:
            # 90% confidence
            if df >= 120:
                return Decimal("1.645")
            elif df >= 30:
                return Decimal("1.697")
            elif df >= 10:
                return Decimal("1.812")
            else:
                return Decimal("2.015")
        else:
            return Decimal("1.645")

    # ------------------------------------------------------------------ #
    # Internal - Utility                                                   #
    # ------------------------------------------------------------------ #

    def _infer_variables(
        self,
        data: List[BaselineDataPoint],
    ) -> List[str]:
        """Infer available variable names from the data points.

        Args:
            data: Baseline data points.

        Returns:
            Sorted list of variable names present in the data.
        """
        if not data:
            return []

        # Use the union of all variable names across all data points
        all_vars: set[str] = set()
        for dp in data:
            all_vars.update(dp.variables.keys())

        return sorted(all_vars)

    def _extract_relevant_variables(
        self,
        data: List[BaselineDataPoint],
        model: RegressionModel,
    ) -> List[RelevantVariable]:
        """Build RelevantVariable metadata from data and fitted model.

        Args:
            data: Baseline data points.
            model: Fitted regression model.

        Returns:
            List of RelevantVariable objects.
        """
        var_names = self._infer_variables(data)
        model_var_names = {c.variable_name for c in model.coefficients}

        result: List[RelevantVariable] = []
        for var_name in var_names:
            values = [
                dp.variables.get(var_name, Decimal("0")) for dp in data
            ]
            if not values:
                continue

            n = len(values)
            min_v = min(values)
            max_v = max(values)
            mean_v = _safe_divide(sum(values, Decimal("0")), _decimal(n))

            var_type = (
                VariableType.ENERGY_DRIVER
                if var_name in model_var_names
                else VariableType.RELEVANT_VARIABLE
            )

            result.append(RelevantVariable(
                name=var_name,
                variable_type=var_type,
                unit="",
                description=f"Variable '{var_name}' from baseline data",
                data_source="baseline_dataset",
                min_value=_round_val(min_v, 4),
                max_value=_round_val(max_v, 4),
                mean_value=_round_val(mean_v, 4),
            ))

        return result

    def _determine_adequacy_rating(
        self,
        model: RegressionModel,
    ) -> str:
        """Determine the overall adequacy rating for a model.

        Args:
            model: Fitted regression model.

        Returns:
            Rating string: 'excellent', 'good', 'acceptable', or 'poor'.
        """
        for rating in ("excellent", "good", "acceptable"):
            criteria = MODEL_ADEQUACY_CRITERIA[rating]
            r2_ok = model.r_squared >= criteria["min_r_squared"]
            cv_ok = model.cv_rmse <= criteria["max_cv_rmse"]
            if r2_ok and cv_ok:
                return rating
        return "poor"

    def _model_summary(
        self,
        model: RegressionModel,
        index: int,
    ) -> Dict[str, Any]:
        """Create a summary dictionary for a model in comparison.

        Args:
            model: Regression model.
            index: Model index in the comparison list.

        Returns:
            Summary dictionary.
        """
        return {
            "index": index,
            "model_type": model.model_type.value,
            "r_squared": str(model.r_squared),
            "adjusted_r_squared": str(model.adjusted_r_squared),
            "cv_rmse": str(model.cv_rmse),
            "aic": str(model.aic),
            "bic": str(model.bic),
            "f_statistic": str(model.f_statistic),
            "f_p_value": str(model.f_p_value),
            "n_observations": model.n_observations,
            "n_coefficients": len(model.coefficients),
            "adequacy_rating": self._determine_adequacy_rating(model),
        }

    @staticmethod
    def _rank_position(
        values: List[Decimal],
        target: Decimal,
        ascending: bool = True,
    ) -> int:
        """Determine the rank position of a target in a list of values.

        Args:
            values: List of values to rank against.
            target: Target value.
            ascending: If True, lower values get better (lower) rank.

        Returns:
            Rank position (0-based, 0 = best).
        """
        if ascending:
            sorted_vals = sorted(values)
        else:
            sorted_vals = sorted(values, reverse=True)

        for rank, val in enumerate(sorted_vals):
            if val == target:
                return rank
        return len(values) - 1
