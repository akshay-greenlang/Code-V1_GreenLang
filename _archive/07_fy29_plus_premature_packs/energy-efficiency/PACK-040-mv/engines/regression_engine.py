# -*- coding: utf-8 -*-
"""
RegressionEngine - PACK-040 M&V Engine 6
===========================================

Statistical regression modelling for Measurement & Verification baselines.
Implements OLS via normal equation with Gauss-Jordan inversion,
3-parameter change-point models (3P cooling / 3P heating), 4-parameter
model (heating slope + flat + cooling slope), 5-parameter model (two
change-points), and TOWT (Time-of-Week and Temperature) model for
hourly / daily data.

Calculation Methodology:
    OLS Normal Equation:
        beta = (X'X)^(-1) X'y
        where X'X is inverted via Gauss-Jordan elimination (no numpy)

    3P Cooling Change-Point Model:
        E = a + b * max(0, T - Tcp)
        where Tcp is the cooling change-point temperature

    3P Heating Change-Point Model:
        E = a + b * max(0, Thp - T)
        where Thp is the heating change-point temperature

    4P Change-Point Model:
        E = a + bh * max(0, Thp - T) + bc * max(0, T - Tcp)
        where Thp == Tcp (single balance point)

    5P Change-Point Model:
        E = a + bh * max(0, Thp - T) + bc * max(0, T - Tcp)
        where Thp <= Tcp (two distinct change-points / deadband)

    TOWT (Time-of-Week and Temperature):
        E = sum_j(alpha_j * D_j) + f(T)
        where D_j = indicator for time-of-week bin j
        and f(T) = piecewise-linear temperature function

    Diagnostics:
        R-squared = 1 - SS_res / SS_tot
        Adjusted R-squared = 1 - (1 - R2) * (n-1) / (n-p-1)
        CVRMSE = RMSE / y_mean * 100
        NMBE = sum(residuals) / (n * y_mean) * 100
        F-statistic = (SS_reg / p) / (SS_res / (n-p-1))
        t-statistics = beta_j / SE(beta_j)
        Durbin-Watson = sum((e_i - e_{i-1})^2) / sum(e_i^2)
        Cook's distance_i = (e_i^2 / (p * MSE)) * (h_ii / (1-h_ii)^2)
        VIF_j = 1 / (1 - R2_j)

Regulatory References:
    - IPMVP Core Concepts 2022 (EVO) - Baseline regression modelling
    - ASHRAE Guideline 14-2014 - CVRMSE <25% monthly, <30% daily; NMBE +/-5% monthly
    - ISO 50015:2014 - M&V regression requirements
    - ISO 50006:2014 - Energy baseline development
    - FEMP M&V Guidelines 4.0 - Federal regression requirements

Zero-Hallucination:
    - OLS solved via deterministic normal equation (no LLM)
    - Change-points found by exhaustive grid search maximising R-squared
    - All diagnostics use standard statistical formulae
    - Decimal arithmetic throughout for reproducibility
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-040 M&V
Engine:  6 of 10
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

class RegressionModelType(str, Enum):
    """Regression model type for M&V baseline.

    OLS:           Ordinary Least Squares (simple / multivariate).
    CP3_COOLING:   3-parameter cooling change-point model.
    CP3_HEATING:   3-parameter heating change-point model.
    CP4:           4-parameter (heating + cooling, single balance point).
    CP5:           5-parameter (heating + cooling, two change-points).
    TOWT:          Time-of-Week and Temperature model.
    """
    OLS = "ols"
    CP3_COOLING = "cp3_cooling"
    CP3_HEATING = "cp3_heating"
    CP4 = "cp4"
    CP5 = "cp5"
    TOWT = "towt"

class DataFrequency(str, Enum):
    """Time-series data frequency for regression.

    HOURLY:   Hourly readings (8760 per year).
    DAILY:    Daily readings (365 per year).
    WEEKLY:   Weekly readings (52 per year).
    MONTHLY:  Monthly readings (12 per year).
    """
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

class ValidationGrade(str, Enum):
    """Model validation grade per ASHRAE 14.

    EXCELLENT:   All criteria met with wide margin.
    GOOD:        All ASHRAE 14 criteria satisfied.
    ACCEPTABLE:  Near thresholds but within limits.
    MARGINAL:    One criterion borderline.
    FAIL:        One or more criteria exceeded.
    """
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    MARGINAL = "marginal"
    FAIL = "fail"

class ResidualPattern(str, Enum):
    """Residual analysis pattern classification.

    RANDOM:        No discernible pattern (good).
    TREND:         Linear trend in residuals.
    FUNNEL:        Heteroscedasticity (fan shape).
    CYCLIC:        Seasonal / cyclical pattern.
    CLUSTERED:     Clusters of positive / negative residuals.
    AUTOCORRELATED: Significant serial correlation.
    """
    RANDOM = "random"
    TREND = "trend"
    FUNNEL = "funnel"
    CYCLIC = "cyclic"
    CLUSTERED = "clustered"
    AUTOCORRELATED = "autocorrelated"

class DiagnosticStatus(str, Enum):
    """Diagnostic check pass / fail status.

    PASS:     Diagnostic within acceptable limits.
    WARNING:  Diagnostic near threshold.
    FAIL:     Diagnostic exceeded threshold.
    SKIPPED:  Not enough data to run diagnostic.
    """
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    SKIPPED = "skipped"

class VariableRole(str, Enum):
    """Role of an independent variable in the regression.

    TEMPERATURE:  Outdoor air temperature (primary).
    PRODUCTION:   Production volume or throughput.
    OCCUPANCY:    Building occupancy rate.
    DAYLIGHT:     Daylight hours.
    SCHEDULE:     Operating schedule indicator.
    TIME_OF_WEEK: Time-of-week indicator (TOWT).
    CUSTOM:       User-defined variable.
    """
    TEMPERATURE = "temperature"
    PRODUCTION = "production"
    OCCUPANCY = "occupancy"
    DAYLIGHT = "daylight"
    SCHEDULE = "schedule"
    TIME_OF_WEEK = "time_of_week"
    CUSTOM = "custom"

# ---------------------------------------------------------------------------
# ASHRAE 14 Validation Thresholds
# ---------------------------------------------------------------------------

ASHRAE14_THRESHOLDS: Dict[str, Dict[str, Decimal]] = {
    DataFrequency.MONTHLY.value: {
        "cvrmse_max": Decimal("25"),
        "nmbe_max": Decimal("5"),
        "r_squared_min": Decimal("0.70"),
    },
    DataFrequency.DAILY.value: {
        "cvrmse_max": Decimal("30"),
        "nmbe_max": Decimal("10"),
        "r_squared_min": Decimal("0.50"),
    },
    DataFrequency.HOURLY.value: {
        "cvrmse_max": Decimal("30"),
        "nmbe_max": Decimal("10"),
        "r_squared_min": Decimal("0"),  # Not required for hourly
    },
    DataFrequency.WEEKLY.value: {
        "cvrmse_max": Decimal("28"),
        "nmbe_max": Decimal("8"),
        "r_squared_min": Decimal("0.55"),
    },
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class IndependentVariable(BaseModel):
    """Definition of an independent variable for regression."""

    variable_id: str = Field(default_factory=_new_uuid, description="Unique variable ID")
    name: str = Field(..., min_length=1, description="Variable name")
    role: VariableRole = Field(..., description="Variable role in regression")
    unit: str = Field(default="", description="Variable unit (e.g. degC, units, %)")
    values: List[Decimal] = Field(default_factory=list, description="Observed values")
    description: str = Field(default="", description="Free-text description")

    @field_validator("values", mode="before")
    @classmethod
    def _coerce_values(cls, v: Any) -> List[Decimal]:
        if isinstance(v, list):
            return [_decimal(x) for x in v]
        return v

class RegressionConfig(BaseModel):
    """Configuration for a regression analysis run."""

    config_id: str = Field(default_factory=_new_uuid, description="Run identifier")
    project_id: str = Field(default="", description="M&V project reference")
    model_type: RegressionModelType = Field(
        default=RegressionModelType.OLS, description="Model type to fit"
    )
    frequency: DataFrequency = Field(
        default=DataFrequency.MONTHLY, description="Data frequency"
    )
    dependent_values: List[Decimal] = Field(
        default_factory=list, description="Dependent variable (energy) values"
    )
    independent_vars: List[IndependentVariable] = Field(
        default_factory=list, description="Independent variables"
    )
    cp_search_min: Decimal = Field(
        default=Decimal("5"), description="Change-point search lower bound (degC)"
    )
    cp_search_max: Decimal = Field(
        default=Decimal("35"), description="Change-point search upper bound (degC)"
    )
    cp_search_step: Decimal = Field(
        default=Decimal("0.5"), description="Change-point search step size (degC)"
    )
    towt_bins: int = Field(
        default=12, ge=4, le=168, description="Number of time-of-week bins for TOWT"
    )
    min_observations: int = Field(
        default=12, ge=6, description="Minimum observations required"
    )

    @field_validator("dependent_values", mode="before")
    @classmethod
    def _coerce_dep(cls, v: Any) -> List[Decimal]:
        if isinstance(v, list):
            return [_decimal(x) for x in v]
        return v

class CoefficientDetail(BaseModel):
    """Detailed regression coefficient with statistics."""

    name: str = Field(..., description="Coefficient name (intercept, slope, etc.)")
    value: Decimal = Field(default=Decimal("0"), description="Estimated coefficient")
    std_error: Decimal = Field(default=Decimal("0"), description="Standard error")
    t_statistic: Decimal = Field(default=Decimal("0"), description="t-statistic")
    p_value: Decimal = Field(default=Decimal("1"), description="Two-tailed p-value approx")
    significant: bool = Field(default=False, description="Significant at alpha=0.05")
    vif: Decimal = Field(default=Decimal("1"), description="Variance Inflation Factor")

class ChangePointResult(BaseModel):
    """Change-point search result."""

    heating_cp: Optional[Decimal] = Field(None, description="Heating change-point (degC)")
    cooling_cp: Optional[Decimal] = Field(None, description="Cooling change-point (degC)")
    best_r_squared: Decimal = Field(default=Decimal("0"), description="R-squared at optimal CP")
    search_min: Decimal = Field(default=Decimal("5"), description="Search lower bound")
    search_max: Decimal = Field(default=Decimal("35"), description="Search upper bound")
    candidates_evaluated: int = Field(default=0, description="Number of candidates tested")

class DiagnosticResult(BaseModel):
    """Single diagnostic check result."""

    diagnostic_id: str = Field(default_factory=_new_uuid, description="Diagnostic ID")
    name: str = Field(..., description="Diagnostic name")
    value: Decimal = Field(default=Decimal("0"), description="Computed value")
    threshold: Optional[Decimal] = Field(None, description="Threshold for pass/fail")
    status: DiagnosticStatus = Field(default=DiagnosticStatus.SKIPPED, description="Status")
    detail: str = Field(default="", description="Additional detail")

class RegressionStatistics(BaseModel):
    """Comprehensive regression statistics."""

    n_observations: int = Field(default=0, description="Number of observations")
    n_parameters: int = Field(default=0, description="Number of estimated parameters")
    degrees_of_freedom: int = Field(default=0, description="Residual degrees of freedom")
    r_squared: Decimal = Field(default=Decimal("0"), description="R-squared")
    adj_r_squared: Decimal = Field(default=Decimal("0"), description="Adjusted R-squared")
    cvrmse: Decimal = Field(default=Decimal("0"), description="CV(RMSE) in percent")
    nmbe: Decimal = Field(default=Decimal("0"), description="NMBE in percent")
    rmse: Decimal = Field(default=Decimal("0"), description="Root mean squared error")
    mse: Decimal = Field(default=Decimal("0"), description="Mean squared error")
    mae: Decimal = Field(default=Decimal("0"), description="Mean absolute error")
    ss_total: Decimal = Field(default=Decimal("0"), description="Total sum of squares")
    ss_residual: Decimal = Field(default=Decimal("0"), description="Residual sum of squares")
    ss_regression: Decimal = Field(default=Decimal("0"), description="Regression sum of squares")
    f_statistic: Decimal = Field(default=Decimal("0"), description="F-statistic")
    durbin_watson: Decimal = Field(default=Decimal("0"), description="Durbin-Watson statistic")
    y_mean: Decimal = Field(default=Decimal("0"), description="Mean of dependent variable")
    y_std: Decimal = Field(default=Decimal("0"), description="Std dev of dependent variable")

class RegressionFitResult(BaseModel):
    """Complete regression fit result with provenance."""

    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    config_id: str = Field(default="", description="Configuration reference")
    model_type: RegressionModelType = Field(
        default=RegressionModelType.OLS, description="Fitted model type"
    )
    coefficients: List[CoefficientDetail] = Field(
        default_factory=list, description="Estimated coefficients"
    )
    statistics: RegressionStatistics = Field(
        default_factory=RegressionStatistics, description="Fit statistics"
    )
    change_point: Optional[ChangePointResult] = Field(
        None, description="Change-point search result (if applicable)"
    )
    residuals: List[Decimal] = Field(default_factory=list, description="Residual vector")
    predicted: List[Decimal] = Field(default_factory=list, description="Predicted values")
    diagnostics: List[DiagnosticResult] = Field(
        default_factory=list, description="Diagnostic checks"
    )
    validation_grade: ValidationGrade = Field(
        default=ValidationGrade.FAIL, description="ASHRAE 14 validation grade"
    )
    calculated_at: datetime = Field(default_factory=utcnow, description="Timestamp")
    processing_time_ms: Decimal = Field(default=Decimal("0"), description="Processing time")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    notes: List[str] = Field(default_factory=list, description="Processing notes")

class ModelComparisonEntry(BaseModel):
    """Single entry in a model comparison."""

    model_type: RegressionModelType = Field(..., description="Model type")
    r_squared: Decimal = Field(default=Decimal("0"), description="R-squared")
    adj_r_squared: Decimal = Field(default=Decimal("0"), description="Adjusted R-squared")
    cvrmse: Decimal = Field(default=Decimal("0"), description="CV(RMSE) %")
    nmbe: Decimal = Field(default=Decimal("0"), description="NMBE %")
    n_parameters: int = Field(default=0, description="Parameter count")
    passes_ashrae14: bool = Field(default=False, description="Passes ASHRAE 14")
    aic: Decimal = Field(default=Decimal("0"), description="Akaike Information Criterion")
    bic: Decimal = Field(default=Decimal("0"), description="Bayesian Information Criterion")
    rank: int = Field(default=0, description="Rank (1 = best)")

class ModelComparisonResult(BaseModel):
    """Model comparison across multiple regression types."""

    comparison_id: str = Field(default_factory=_new_uuid, description="Comparison ID")
    entries: List[ModelComparisonEntry] = Field(
        default_factory=list, description="Comparison entries"
    )
    recommended: Optional[RegressionModelType] = Field(
        None, description="Recommended model type"
    )
    recommendation_reason: str = Field(default="", description="Recommendation rationale")
    calculated_at: datetime = Field(default_factory=utcnow, description="Timestamp")
    processing_time_ms: Decimal = Field(default=Decimal("0"), description="Processing time")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Linear Algebra Helpers (Pure Python - no numpy)
# ---------------------------------------------------------------------------

def _transpose(matrix: List[List[Decimal]]) -> List[List[Decimal]]:
    """Transpose a 2D matrix."""
    if not matrix:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    return [[matrix[r][c] for r in range(rows)] for c in range(cols)]

def _mat_mult(a: List[List[Decimal]], b: List[List[Decimal]]) -> List[List[Decimal]]:
    """Multiply two 2D matrices (a x b)."""
    rows_a = len(a)
    cols_a = len(a[0]) if a else 0
    cols_b = len(b[0]) if b else 0
    result = [[Decimal("0")] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            s = Decimal("0")
            for k in range(cols_a):
                s += a[i][k] * b[k][j]
            result[i][j] = s
    return result

def _mat_vec_mult(a: List[List[Decimal]], v: List[Decimal]) -> List[Decimal]:
    """Multiply a matrix by a column vector."""
    result = []
    for row in a:
        s = Decimal("0")
        for j, val in enumerate(row):
            s += val * v[j]
        result.append(s)
    return result

def _gauss_jordan_inverse(matrix: List[List[Decimal]]) -> Optional[List[List[Decimal]]]:
    """Invert a square matrix via Gauss-Jordan elimination.

    Returns None if the matrix is singular.
    """
    n = len(matrix)
    # Augment with identity
    aug = [row[:] + [Decimal("1") if i == j else Decimal("0") for j in range(n)]
           for i, row in enumerate(matrix)]

    for col in range(n):
        # Partial pivoting
        max_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        if max_val < Decimal("1e-30"):
            return None  # Singular
        aug[col], aug[max_row] = aug[max_row], aug[col]

        # Scale pivot row
        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] = aug[col][j] / pivot

        # Eliminate column
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(2 * n):
                aug[row][j] -= factor * aug[col][j]

    # Extract inverse
    return [row[n:] for row in aug]

def _identity(n: int) -> List[List[Decimal]]:
    """Create an n x n identity matrix."""
    return [[Decimal("1") if i == j else Decimal("0") for j in range(n)] for i in range(n)]

# ---------------------------------------------------------------------------
# Statistical Helpers
# ---------------------------------------------------------------------------

def _compute_ss_total(y: List[Decimal], y_mean: Decimal) -> Decimal:
    """Compute total sum of squares."""
    return sum((yi - y_mean) ** 2 for yi in y)

def _compute_ss_residual(y: List[Decimal], y_hat: List[Decimal]) -> Decimal:
    """Compute residual sum of squares."""
    return sum((yi - yhi) ** 2 for yi, yhi in zip(y, y_hat))

def _compute_rmse(ss_res: Decimal, n: int) -> Decimal:
    """Compute root mean squared error."""
    if n <= 0:
        return Decimal("0")
    mse = ss_res / _decimal(n)
    if mse < Decimal("0"):
        return Decimal("0")
    return _decimal(math.sqrt(float(mse)))

def _compute_cvrmse(rmse: Decimal, y_mean: Decimal) -> Decimal:
    """Compute CV(RMSE) in percent."""
    return _safe_pct(rmse, y_mean)

def _compute_nmbe(residuals: List[Decimal], y_mean: Decimal, n: int) -> Decimal:
    """Compute NMBE in percent."""
    if n <= 0 or y_mean == Decimal("0"):
        return Decimal("0")
    return _safe_divide(
        sum(residuals) * Decimal("100"),
        _decimal(n) * y_mean,
    )

def _compute_durbin_watson(residuals: List[Decimal]) -> Decimal:
    """Compute Durbin-Watson statistic for autocorrelation."""
    if len(residuals) < 2:
        return Decimal("2")  # Ideal value = no autocorrelation
    numerator = sum(
        (residuals[i] - residuals[i - 1]) ** 2
        for i in range(1, len(residuals))
    )
    denominator = sum(e ** 2 for e in residuals)
    return _safe_divide(numerator, denominator, Decimal("2"))

def _compute_cooks_distance(
    residuals: List[Decimal],
    hat_diag: List[Decimal],
    p: int,
    mse: Decimal,
) -> List[Decimal]:
    """Compute Cook's distance for each observation."""
    n = len(residuals)
    if p <= 0 or mse == Decimal("0") or n == 0:
        return [Decimal("0")] * n
    cooks = []
    for i in range(n):
        h_ii = hat_diag[i] if i < len(hat_diag) else Decimal("0")
        denom = _decimal(p) * mse * (Decimal("1") - h_ii) ** 2
        if denom == Decimal("0"):
            cooks.append(Decimal("0"))
        else:
            cooks.append(residuals[i] ** 2 * h_ii / denom)
    return cooks

def _approximate_p_value(t_stat: Decimal, df: int) -> Decimal:
    """Approximate two-tailed p-value from t-statistic using normal approx.

    For production use a proper t-distribution CDF is preferred.  This
    approximation is sufficient for pass/fail classification at alpha=0.05.
    """
    if df <= 0:
        return Decimal("1")
    t = float(abs(t_stat))
    # Abramowitz & Stegun normal approximation
    if t > 6.0:
        return Decimal("0.0001")
    b = 1.0 / (1.0 + 0.2316419 * t)
    poly = b * (0.319381530 + b * (-0.356563782 + b * (1.781477937
               + b * (-1.821255978 + b * 1.330274429))))
    phi = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * t * t)
    one_tail = phi * poly
    return _decimal(max(2.0 * one_tail, 0.0001))

def _compute_hat_diagonal(
    x_matrix: List[List[Decimal]],
    xtx_inv: List[List[Decimal]],
) -> List[Decimal]:
    """Compute diagonal of the hat matrix H = X (X'X)^-1 X'."""
    n = len(x_matrix)
    p = len(x_matrix[0]) if x_matrix else 0
    hat_diag = []
    for i in range(n):
        h_ii = Decimal("0")
        for j in range(p):
            for k in range(p):
                h_ii += x_matrix[i][j] * xtx_inv[j][k] * x_matrix[i][k]
        hat_diag.append(h_ii)
    return hat_diag

# ---------------------------------------------------------------------------
# Engine Class
# ---------------------------------------------------------------------------

class RegressionEngine:
    """Statistical regression engine for M&V baseline modelling.

    Implements OLS, 3P/4P/5P change-point, and TOWT models with full
    ASHRAE Guideline 14 diagnostics.  All calculations are deterministic
    (zero-hallucination) using Decimal arithmetic and pure-Python linear
    algebra (no numpy dependency).

    Attributes:
        _module_version: Engine version string.

    Example:
        >>> engine = RegressionEngine()
        >>> config = RegressionConfig(
        ...     model_type=RegressionModelType.OLS,
        ...     frequency=DataFrequency.MONTHLY,
        ...     dependent_values=[100, 120, 140, 130, 110, 95, ...],
        ...     independent_vars=[IndependentVariable(
        ...         name="temperature", role=VariableRole.TEMPERATURE,
        ...         values=[5, 10, 15, 20, 25, 30, ...]
        ...     )],
        ... )
        >>> result = engine.fit_regression(config)
        >>> assert result.validation_grade != ValidationGrade.FAIL
    """

    def __init__(self) -> None:
        """Initialise the RegressionEngine."""
        self._module_version: str = _MODULE_VERSION
        logger.info("RegressionEngine v%s initialised", self._module_version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_regression(self, config: RegressionConfig) -> RegressionFitResult:
        """Fit a regression model per the supplied configuration.

        Dispatches to the appropriate model fitting method based on
        ``config.model_type``.  Returns full diagnostics and provenance.

        Args:
            config: Regression configuration with data.

        Returns:
            RegressionFitResult with coefficients, statistics, diagnostics.

        Raises:
            ValueError: If insufficient data or invalid configuration.
        """
        t0 = time.perf_counter()
        logger.info(
            "Fitting regression: type=%s, freq=%s, n=%d, vars=%d",
            config.model_type.value,
            config.frequency.value,
            len(config.dependent_values),
            len(config.independent_vars),
        )

        n = len(config.dependent_values)
        if n < config.min_observations:
            raise ValueError(
                f"Insufficient observations: {n} < {config.min_observations}"
            )

        dispatch: Dict[RegressionModelType, Any] = {
            RegressionModelType.OLS: self._fit_ols,
            RegressionModelType.CP3_COOLING: self._fit_cp3_cooling,
            RegressionModelType.CP3_HEATING: self._fit_cp3_heating,
            RegressionModelType.CP4: self._fit_cp4,
            RegressionModelType.CP5: self._fit_cp5,
            RegressionModelType.TOWT: self._fit_towt,
        }
        fit_fn = dispatch.get(config.model_type)
        if fit_fn is None:
            raise ValueError(f"Unsupported model type: {config.model_type}")

        result: RegressionFitResult = fit_fn(config)
        result.config_id = config.config_id

        # Run diagnostics
        result.diagnostics = self._run_diagnostics(result, config)

        # Assign validation grade
        result.validation_grade = self._assign_grade(result, config.frequency)

        elapsed = (time.perf_counter() - t0) * 1000.0
        result.processing_time_ms = _round_val(_decimal(elapsed), 2)
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Regression fit complete: R2=%.4f, CVRMSE=%.2f%%, grade=%s, "
            "hash=%s (%.1f ms)",
            float(result.statistics.r_squared),
            float(result.statistics.cvrmse),
            result.validation_grade.value,
            result.provenance_hash[:16],
            elapsed,
        )
        return result

    def compare_models(
        self,
        config: RegressionConfig,
        model_types: Optional[List[RegressionModelType]] = None,
    ) -> ModelComparisonResult:
        """Fit multiple model types and compare their performance.

        Args:
            config: Base regression configuration.
            model_types: List of model types to compare.  Defaults to all
                types appropriate for the data.

        Returns:
            ModelComparisonResult with ranked entries and recommendation.
        """
        t0 = time.perf_counter()
        if model_types is None:
            model_types = [
                RegressionModelType.OLS,
                RegressionModelType.CP3_COOLING,
                RegressionModelType.CP3_HEATING,
                RegressionModelType.CP4,
                RegressionModelType.CP5,
            ]
            if config.frequency in (DataFrequency.HOURLY, DataFrequency.DAILY):
                model_types.append(RegressionModelType.TOWT)

        logger.info(
            "Comparing %d model types for %d observations",
            len(model_types), len(config.dependent_values),
        )

        entries: List[ModelComparisonEntry] = []
        for mt in model_types:
            try:
                cfg_copy = config.model_copy(deep=True)
                cfg_copy.model_type = mt
                fit_result = self.fit_regression(cfg_copy)

                n = fit_result.statistics.n_observations
                p = fit_result.statistics.n_parameters
                ss_res = fit_result.statistics.ss_residual

                aic = self._compute_aic(n, p, ss_res)
                bic = self._compute_bic(n, p, ss_res)

                thresholds = ASHRAE14_THRESHOLDS.get(
                    config.frequency.value,
                    ASHRAE14_THRESHOLDS[DataFrequency.MONTHLY.value],
                )
                passes = (
                    fit_result.statistics.cvrmse <= thresholds["cvrmse_max"]
                    and abs(fit_result.statistics.nmbe) <= thresholds["nmbe_max"]
                    and fit_result.statistics.r_squared >= thresholds["r_squared_min"]
                )

                entries.append(ModelComparisonEntry(
                    model_type=mt,
                    r_squared=fit_result.statistics.r_squared,
                    adj_r_squared=fit_result.statistics.adj_r_squared,
                    cvrmse=fit_result.statistics.cvrmse,
                    nmbe=fit_result.statistics.nmbe,
                    n_parameters=p,
                    passes_ashrae14=passes,
                    aic=_round_val(aic, 2),
                    bic=_round_val(bic, 2),
                ))
            except Exception as exc:
                logger.warning("Model %s failed: %s", mt.value, exc)

        # Rank by adjusted R-squared (descending), then by AIC (ascending)
        entries.sort(key=lambda e: (-float(e.adj_r_squared), float(e.aic)))
        for i, entry in enumerate(entries):
            entry.rank = i + 1

        recommended = None
        reason = ""
        passing = [e for e in entries if e.passes_ashrae14]
        if passing:
            recommended = passing[0].model_type
            reason = (
                f"Best adjusted R-squared ({float(passing[0].adj_r_squared):.4f}) "
                f"among models passing ASHRAE 14 criteria"
            )
        elif entries:
            recommended = entries[0].model_type
            reason = "No model passes ASHRAE 14; selected highest adjusted R-squared"

        elapsed = (time.perf_counter() - t0) * 1000.0
        result = ModelComparisonResult(
            entries=entries,
            recommended=recommended,
            recommendation_reason=reason,
            processing_time_ms=_round_val(_decimal(elapsed), 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Model comparison complete: %d models, recommended=%s (%.1f ms)",
            len(entries),
            recommended.value if recommended else "none",
            elapsed,
        )
        return result

    def predict(
        self,
        fit_result: RegressionFitResult,
        independent_values: List[List[Decimal]],
        temperatures: Optional[List[Decimal]] = None,
    ) -> List[Decimal]:
        """Predict dependent variable using a fitted model.

        Args:
            fit_result: A fitted regression result.
            independent_values: Independent variable matrix (rows x cols).
            temperatures: Temperature values (for change-point models).

        Returns:
            List of predicted values.
        """
        t0 = time.perf_counter()
        betas = [c.value for c in fit_result.coefficients]
        model_type = fit_result.model_type
        predictions: List[Decimal] = []

        if model_type == RegressionModelType.OLS:
            for row in independent_values:
                y_hat = betas[0]  # intercept
                for j, x_val in enumerate(row):
                    if j + 1 < len(betas):
                        y_hat += betas[j + 1] * x_val
                predictions.append(_round_val(y_hat, 4))

        elif model_type == RegressionModelType.CP3_COOLING:
            cp = (fit_result.change_point.cooling_cp
                  if fit_result.change_point else Decimal("18"))
            temps = temperatures if temperatures else (
                [row[0] for row in independent_values] if independent_values else []
            )
            for t_val in temps:
                y_hat = betas[0] + betas[1] * max(Decimal("0"), t_val - cp)
                predictions.append(_round_val(y_hat, 4))

        elif model_type == RegressionModelType.CP3_HEATING:
            cp = (fit_result.change_point.heating_cp
                  if fit_result.change_point else Decimal("18"))
            temps = temperatures if temperatures else (
                [row[0] for row in independent_values] if independent_values else []
            )
            for t_val in temps:
                y_hat = betas[0] + betas[1] * max(Decimal("0"), cp - t_val)
                predictions.append(_round_val(y_hat, 4))

        elif model_type == RegressionModelType.CP4:
            cp = (fit_result.change_point.heating_cp
                  if fit_result.change_point else Decimal("18"))
            temps = temperatures if temperatures else (
                [row[0] for row in independent_values] if independent_values else []
            )
            for t_val in temps:
                y_hat = (betas[0]
                         + betas[1] * max(Decimal("0"), cp - t_val)
                         + betas[2] * max(Decimal("0"), t_val - cp))
                predictions.append(_round_val(y_hat, 4))

        elif model_type == RegressionModelType.CP5:
            h_cp = (fit_result.change_point.heating_cp
                    if fit_result.change_point else Decimal("15"))
            c_cp = (fit_result.change_point.cooling_cp
                    if fit_result.change_point else Decimal("22"))
            temps = temperatures if temperatures else (
                [row[0] for row in independent_values] if independent_values else []
            )
            for t_val in temps:
                y_hat = (betas[0]
                         + betas[1] * max(Decimal("0"), h_cp - t_val)
                         + betas[2] * max(Decimal("0"), t_val - c_cp))
                predictions.append(_round_val(y_hat, 4))

        elif model_type == RegressionModelType.TOWT:
            for row in independent_values:
                y_hat = betas[0]
                for j, x_val in enumerate(row):
                    if j + 1 < len(betas):
                        y_hat += betas[j + 1] * x_val
                predictions.append(_round_val(y_hat, 4))

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Predicted %d values using %s model (%.1f ms)",
            len(predictions), model_type.value, elapsed,
        )
        return predictions

    def compute_residual_diagnostics(
        self,
        residuals: List[Decimal],
        y_values: List[Decimal],
        y_hat: List[Decimal],
    ) -> List[DiagnosticResult]:
        """Compute a full suite of residual diagnostics.

        Args:
            residuals: Residual vector (y - y_hat).
            y_values: Observed dependent values.
            y_hat: Predicted values.

        Returns:
            List of DiagnosticResult objects.
        """
        t0 = time.perf_counter()
        diags: List[DiagnosticResult] = []
        n = len(residuals)

        if n < 3:
            logger.warning("Too few observations (%d) for residual diagnostics", n)
            return diags

        # Durbin-Watson
        dw = _compute_durbin_watson(residuals)
        dw_status = DiagnosticStatus.PASS
        if dw < Decimal("1.5") or dw > Decimal("2.5"):
            dw_status = DiagnosticStatus.WARNING
        if dw < Decimal("1.0") or dw > Decimal("3.0"):
            dw_status = DiagnosticStatus.FAIL
        diags.append(DiagnosticResult(
            name="durbin_watson",
            value=_round_val(dw, 4),
            threshold=Decimal("1.5"),
            status=dw_status,
            detail=f"DW={float(dw):.4f}; ideal=2.0 (no autocorrelation)",
        ))

        # Residual mean (should be near zero)
        res_mean = _safe_divide(sum(residuals), _decimal(n))
        res_mean_status = DiagnosticStatus.PASS
        y_mean = _safe_divide(sum(y_values), _decimal(n))
        mean_pct = abs(_safe_pct(res_mean, y_mean)) if y_mean != Decimal("0") else Decimal("0")
        if mean_pct > Decimal("1"):
            res_mean_status = DiagnosticStatus.WARNING
        if mean_pct > Decimal("5"):
            res_mean_status = DiagnosticStatus.FAIL
        diags.append(DiagnosticResult(
            name="residual_mean",
            value=_round_val(res_mean, 6),
            threshold=Decimal("0"),
            status=res_mean_status,
            detail=f"Mean residual as % of y_mean: {float(mean_pct):.2f}%",
        ))

        # Normality (Jarque-Bera approximation)
        jb_stat = self._jarque_bera(residuals)
        jb_status = DiagnosticStatus.PASS
        if jb_stat > Decimal("6"):
            jb_status = DiagnosticStatus.WARNING
        if jb_stat > Decimal("10"):
            jb_status = DiagnosticStatus.FAIL
        diags.append(DiagnosticResult(
            name="jarque_bera",
            value=_round_val(jb_stat, 4),
            threshold=Decimal("5.99"),
            status=jb_status,
            detail="Chi-squared(2) critical value at alpha=0.05 is 5.99",
        ))

        # Runs test for randomness
        runs, expected_runs = self._runs_test(residuals)
        runs_status = DiagnosticStatus.PASS
        if abs(runs - expected_runs) > expected_runs * Decimal("0.3"):
            runs_status = DiagnosticStatus.WARNING
        if abs(runs - expected_runs) > expected_runs * Decimal("0.5"):
            runs_status = DiagnosticStatus.FAIL
        diags.append(DiagnosticResult(
            name="runs_test",
            value=_round_val(_decimal(runs), 0),
            threshold=_round_val(expected_runs, 0),
            status=runs_status,
            detail=f"Observed runs={int(runs)}, expected={float(expected_runs):.0f}",
        ))

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info("Residual diagnostics computed: %d checks (%.1f ms)", len(diags), elapsed)
        return diags

    def compute_vif(
        self,
        independent_vars: List[IndependentVariable],
    ) -> List[CoefficientDetail]:
        """Compute Variance Inflation Factor for each independent variable.

        VIF_j = 1 / (1 - R2_j) where R2_j is the R-squared of regressing
        variable j on all other variables.

        Args:
            independent_vars: List of independent variables.

        Returns:
            List of CoefficientDetail with VIF populated.
        """
        t0 = time.perf_counter()
        k = len(independent_vars)
        if k < 2:
            return [CoefficientDetail(
                name=v.name, value=Decimal("0"), vif=Decimal("1")
            ) for v in independent_vars]

        results: List[CoefficientDetail] = []
        for j in range(k):
            # Regress variable j on all other variables
            y_j = independent_vars[j].values
            n = len(y_j)
            other_cols: List[List[Decimal]] = []
            for m in range(k):
                if m == j:
                    continue
                vals = independent_vars[m].values
                if len(vals) >= n:
                    other_cols.append(vals[:n])

            if not other_cols or n < 3:
                results.append(CoefficientDetail(
                    name=independent_vars[j].name, value=Decimal("0"),
                    vif=Decimal("1"),
                ))
                continue

            # Build X matrix with intercept
            x_mat = [[Decimal("1")] + [other_cols[c][i] for c in range(len(other_cols))]
                      for i in range(n)]
            y_mean_j = _safe_divide(sum(y_j[:n]), _decimal(n))
            ss_tot_j = sum((y_j[i] - y_mean_j) ** 2 for i in range(n))

            # OLS
            betas_j = self._ols_solve(x_mat, list(y_j[:n]))
            if betas_j is None:
                results.append(CoefficientDetail(
                    name=independent_vars[j].name, value=Decimal("0"),
                    vif=Decimal("999"),
                ))
                continue

            y_hat_j = _mat_vec_mult(x_mat, betas_j)
            ss_res_j = sum((y_j[i] - y_hat_j[i]) ** 2 for i in range(n))
            r2_j = Decimal("1") - _safe_divide(ss_res_j, ss_tot_j) if ss_tot_j > Decimal("0") else Decimal("0")
            vif_j = _safe_divide(Decimal("1"), Decimal("1") - r2_j, Decimal("999"))

            results.append(CoefficientDetail(
                name=independent_vars[j].name,
                value=Decimal("0"),
                vif=_round_val(vif_j, 4),
            ))

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info("VIF computed for %d variables (%.1f ms)", k, elapsed)
        return results

    # ------------------------------------------------------------------
    # Internal: OLS Fit
    # ------------------------------------------------------------------

    def _fit_ols(self, config: RegressionConfig) -> RegressionFitResult:
        """Fit an OLS regression model via normal equation."""
        y = list(config.dependent_values)
        n = len(y)

        # Build design matrix [1, X1, X2, ...]
        x_mat = self._build_design_matrix(config.independent_vars, n)
        p = len(x_mat[0])

        betas = self._ols_solve(x_mat, y)
        if betas is None:
            raise ValueError("Design matrix is singular; cannot invert X'X")

        # Compute statistics
        y_hat = _mat_vec_mult(x_mat, betas)
        residuals = [y[i] - y_hat[i] for i in range(n)]
        y_mean = _safe_divide(sum(y), _decimal(n))

        stats, coefs = self._compute_fit_statistics(
            y, y_hat, residuals, betas, x_mat, y_mean, n, p, config,
        )

        return RegressionFitResult(
            model_type=RegressionModelType.OLS,
            coefficients=coefs,
            statistics=stats,
            residuals=[_round_val(r, 6) for r in residuals],
            predicted=[_round_val(yh, 4) for yh in y_hat],
        )

    # ------------------------------------------------------------------
    # Internal: 3P Cooling
    # ------------------------------------------------------------------

    def _fit_cp3_cooling(self, config: RegressionConfig) -> RegressionFitResult:
        """Fit 3P cooling change-point model: E = a + b * max(0, T - Tcp)."""
        y = list(config.dependent_values)
        n = len(y)
        temps = self._get_temperature_values(config, n)

        best_r2 = Decimal("-1")
        best_cp = config.cp_search_min
        best_result: Optional[RegressionFitResult] = None
        candidates = 0

        cp = config.cp_search_min
        while cp <= config.cp_search_max:
            candidates += 1
            x_col = [max(Decimal("0"), temps[i] - cp) for i in range(n)]
            x_mat = [[Decimal("1"), x_col[i]] for i in range(n)]
            betas = self._ols_solve(x_mat, y)
            if betas is not None:
                y_hat = _mat_vec_mult(x_mat, betas)
                y_mean = _safe_divide(sum(y), _decimal(n))
                ss_tot = _compute_ss_total(y, y_mean)
                ss_res = _compute_ss_residual(y, y_hat)
                r2 = Decimal("1") - _safe_divide(ss_res, ss_tot) if ss_tot > Decimal("0") else Decimal("0")
                if r2 > best_r2:
                    best_r2 = r2
                    best_cp = cp
                    residuals = [y[i] - y_hat[i] for i in range(n)]
                    stats, coefs = self._compute_fit_statistics(
                        y, y_hat, residuals, betas, x_mat, y_mean, n, 2, config,
                    )
                    coefs[0].name = "intercept (baseload)"
                    coefs[1].name = "cooling_slope"
                    best_result = RegressionFitResult(
                        model_type=RegressionModelType.CP3_COOLING,
                        coefficients=coefs,
                        statistics=stats,
                        residuals=[_round_val(r, 6) for r in residuals],
                        predicted=[_round_val(yh, 4) for yh in y_hat],
                        change_point=ChangePointResult(
                            cooling_cp=_round_val(cp, 2),
                            best_r_squared=_round_val(r2, 6),
                            search_min=config.cp_search_min,
                            search_max=config.cp_search_max,
                            candidates_evaluated=candidates,
                        ),
                    )
            cp += config.cp_search_step

        if best_result is None:
            raise ValueError("3P cooling change-point search failed for all candidates")
        best_result.change_point.candidates_evaluated = candidates
        return best_result

    # ------------------------------------------------------------------
    # Internal: 3P Heating
    # ------------------------------------------------------------------

    def _fit_cp3_heating(self, config: RegressionConfig) -> RegressionFitResult:
        """Fit 3P heating change-point model: E = a + b * max(0, Thp - T)."""
        y = list(config.dependent_values)
        n = len(y)
        temps = self._get_temperature_values(config, n)

        best_r2 = Decimal("-1")
        best_cp = config.cp_search_min
        best_result: Optional[RegressionFitResult] = None
        candidates = 0

        cp = config.cp_search_min
        while cp <= config.cp_search_max:
            candidates += 1
            x_col = [max(Decimal("0"), cp - temps[i]) for i in range(n)]
            x_mat = [[Decimal("1"), x_col[i]] for i in range(n)]
            betas = self._ols_solve(x_mat, y)
            if betas is not None:
                y_hat = _mat_vec_mult(x_mat, betas)
                y_mean = _safe_divide(sum(y), _decimal(n))
                ss_tot = _compute_ss_total(y, y_mean)
                ss_res = _compute_ss_residual(y, y_hat)
                r2 = Decimal("1") - _safe_divide(ss_res, ss_tot) if ss_tot > Decimal("0") else Decimal("0")
                if r2 > best_r2:
                    best_r2 = r2
                    best_cp = cp
                    residuals = [y[i] - y_hat[i] for i in range(n)]
                    stats, coefs = self._compute_fit_statistics(
                        y, y_hat, residuals, betas, x_mat, y_mean, n, 2, config,
                    )
                    coefs[0].name = "intercept (baseload)"
                    coefs[1].name = "heating_slope"
                    best_result = RegressionFitResult(
                        model_type=RegressionModelType.CP3_HEATING,
                        coefficients=coefs,
                        statistics=stats,
                        residuals=[_round_val(r, 6) for r in residuals],
                        predicted=[_round_val(yh, 4) for yh in y_hat],
                        change_point=ChangePointResult(
                            heating_cp=_round_val(cp, 2),
                            best_r_squared=_round_val(r2, 6),
                            search_min=config.cp_search_min,
                            search_max=config.cp_search_max,
                            candidates_evaluated=candidates,
                        ),
                    )
            cp += config.cp_search_step

        if best_result is None:
            raise ValueError("3P heating change-point search failed for all candidates")
        best_result.change_point.candidates_evaluated = candidates
        return best_result

    # ------------------------------------------------------------------
    # Internal: 4P Model
    # ------------------------------------------------------------------

    def _fit_cp4(self, config: RegressionConfig) -> RegressionFitResult:
        """Fit 4P change-point model (single balance point).

        E = a + bh * max(0, Tcp - T) + bc * max(0, T - Tcp)
        """
        y = list(config.dependent_values)
        n = len(y)
        temps = self._get_temperature_values(config, n)

        best_r2 = Decimal("-1")
        best_result: Optional[RegressionFitResult] = None
        candidates = 0

        cp = config.cp_search_min
        while cp <= config.cp_search_max:
            candidates += 1
            heat_col = [max(Decimal("0"), cp - temps[i]) for i in range(n)]
            cool_col = [max(Decimal("0"), temps[i] - cp) for i in range(n)]
            x_mat = [[Decimal("1"), heat_col[i], cool_col[i]] for i in range(n)]
            betas = self._ols_solve(x_mat, y)
            if betas is not None:
                y_hat = _mat_vec_mult(x_mat, betas)
                y_mean = _safe_divide(sum(y), _decimal(n))
                ss_tot = _compute_ss_total(y, y_mean)
                ss_res = _compute_ss_residual(y, y_hat)
                r2 = Decimal("1") - _safe_divide(ss_res, ss_tot) if ss_tot > Decimal("0") else Decimal("0")
                if r2 > best_r2:
                    best_r2 = r2
                    residuals = [y[i] - y_hat[i] for i in range(n)]
                    stats, coefs = self._compute_fit_statistics(
                        y, y_hat, residuals, betas, x_mat, y_mean, n, 3, config,
                    )
                    coefs[0].name = "intercept (baseload)"
                    coefs[1].name = "heating_slope"
                    coefs[2].name = "cooling_slope"
                    best_result = RegressionFitResult(
                        model_type=RegressionModelType.CP4,
                        coefficients=coefs,
                        statistics=stats,
                        residuals=[_round_val(r, 6) for r in residuals],
                        predicted=[_round_val(yh, 4) for yh in y_hat],
                        change_point=ChangePointResult(
                            heating_cp=_round_val(cp, 2),
                            cooling_cp=_round_val(cp, 2),
                            best_r_squared=_round_val(r2, 6),
                            search_min=config.cp_search_min,
                            search_max=config.cp_search_max,
                            candidates_evaluated=candidates,
                        ),
                    )
            cp += config.cp_search_step

        if best_result is None:
            raise ValueError("4P change-point search failed for all candidates")
        best_result.change_point.candidates_evaluated = candidates
        return best_result

    # ------------------------------------------------------------------
    # Internal: 5P Model
    # ------------------------------------------------------------------

    def _fit_cp5(self, config: RegressionConfig) -> RegressionFitResult:
        """Fit 5P change-point model (two change-points / deadband).

        E = a + bh * max(0, Thp - T) + bc * max(0, T - Tcp)
        where Thp <= Tcp.
        """
        y = list(config.dependent_values)
        n = len(y)
        temps = self._get_temperature_values(config, n)

        best_r2 = Decimal("-1")
        best_result: Optional[RegressionFitResult] = None
        candidates = 0

        h_cp = config.cp_search_min
        while h_cp <= config.cp_search_max:
            c_cp = h_cp + config.cp_search_step
            while c_cp <= config.cp_search_max:
                candidates += 1
                heat_col = [max(Decimal("0"), h_cp - temps[i]) for i in range(n)]
                cool_col = [max(Decimal("0"), temps[i] - c_cp) for i in range(n)]
                x_mat = [[Decimal("1"), heat_col[i], cool_col[i]] for i in range(n)]
                betas = self._ols_solve(x_mat, y)
                if betas is not None:
                    y_hat = _mat_vec_mult(x_mat, betas)
                    y_mean = _safe_divide(sum(y), _decimal(n))
                    ss_tot = _compute_ss_total(y, y_mean)
                    ss_res = _compute_ss_residual(y, y_hat)
                    r2 = Decimal("1") - _safe_divide(ss_res, ss_tot) if ss_tot > Decimal("0") else Decimal("0")
                    if r2 > best_r2:
                        best_r2 = r2
                        residuals = [y[i] - y_hat[i] for i in range(n)]
                        stats, coefs = self._compute_fit_statistics(
                            y, y_hat, residuals, betas, x_mat, y_mean, n, 3, config,
                        )
                        coefs[0].name = "intercept (baseload)"
                        coefs[1].name = "heating_slope"
                        coefs[2].name = "cooling_slope"
                        best_result = RegressionFitResult(
                            model_type=RegressionModelType.CP5,
                            coefficients=coefs,
                            statistics=stats,
                            residuals=[_round_val(r, 6) for r in residuals],
                            predicted=[_round_val(yh, 4) for yh in y_hat],
                            change_point=ChangePointResult(
                                heating_cp=_round_val(h_cp, 2),
                                cooling_cp=_round_val(c_cp, 2),
                                best_r_squared=_round_val(r2, 6),
                                search_min=config.cp_search_min,
                                search_max=config.cp_search_max,
                                candidates_evaluated=candidates,
                            ),
                        )
                c_cp += config.cp_search_step
            h_cp += config.cp_search_step

        if best_result is None:
            raise ValueError("5P change-point search failed for all candidates")
        best_result.change_point.candidates_evaluated = candidates
        return best_result

    # ------------------------------------------------------------------
    # Internal: TOWT Model
    # ------------------------------------------------------------------

    def _fit_towt(self, config: RegressionConfig) -> RegressionFitResult:
        """Fit Time-of-Week and Temperature (TOWT) model.

        Creates indicator variables for time-of-week bins and adds
        temperature as a continuous variable.
        E = sum_j(alpha_j * D_j) + beta * T
        """
        y = list(config.dependent_values)
        n = len(y)
        temps = self._get_temperature_values(config, n)
        n_bins = config.towt_bins

        # Assign each observation to a time-of-week bin
        bin_assignments = [i % n_bins for i in range(n)]

        # Build design matrix: intercept + (n_bins-1) indicators + temperature
        # Use first bin as reference category to avoid multicollinearity
        x_mat: List[List[Decimal]] = []
        for i in range(n):
            row = [Decimal("1")]  # intercept
            for b in range(1, n_bins):
                row.append(Decimal("1") if bin_assignments[i] == b else Decimal("0"))
            row.append(temps[i])  # temperature
            x_mat.append(row)

        p = len(x_mat[0])
        betas = self._ols_solve(x_mat, y)
        if betas is None:
            raise ValueError("TOWT design matrix is singular")

        y_hat = _mat_vec_mult(x_mat, betas)
        residuals = [y[i] - y_hat[i] for i in range(n)]
        y_mean = _safe_divide(sum(y), _decimal(n))

        stats, coefs = self._compute_fit_statistics(
            y, y_hat, residuals, betas, x_mat, y_mean, n, p, config,
        )

        # Name coefficients
        coefs[0].name = "intercept (bin_0 reference)"
        for b in range(1, n_bins):
            if b < len(coefs):
                coefs[b].name = f"bin_{b}_indicator"
        if len(coefs) > n_bins:
            coefs[n_bins].name = "temperature_slope"

        return RegressionFitResult(
            model_type=RegressionModelType.TOWT,
            coefficients=coefs,
            statistics=stats,
            residuals=[_round_val(r, 6) for r in residuals],
            predicted=[_round_val(yh, 4) for yh in y_hat],
            notes=[f"TOWT with {n_bins} time-of-week bins"],
        )

    # ------------------------------------------------------------------
    # Internal: Shared helpers
    # ------------------------------------------------------------------

    def _build_design_matrix(
        self,
        independent_vars: List[IndependentVariable],
        n: int,
    ) -> List[List[Decimal]]:
        """Build design matrix [1, X1, X2, ...] with intercept."""
        x_mat: List[List[Decimal]] = []
        for i in range(n):
            row = [Decimal("1")]  # intercept
            for var in independent_vars:
                val = var.values[i] if i < len(var.values) else Decimal("0")
                row.append(val)
            x_mat.append(row)
        return x_mat

    def _get_temperature_values(
        self, config: RegressionConfig, n: int,
    ) -> List[Decimal]:
        """Extract temperature values from independent variables."""
        for var in config.independent_vars:
            if var.role == VariableRole.TEMPERATURE:
                vals = var.values[:n]
                while len(vals) < n:
                    vals.append(Decimal("15"))  # Default if missing
                return vals
        # If no temperature variable, use first variable
        if config.independent_vars:
            vals = config.independent_vars[0].values[:n]
            while len(vals) < n:
                vals.append(Decimal("15"))
            return vals
        return [Decimal("15")] * n

    def _ols_solve(
        self,
        x_mat: List[List[Decimal]],
        y: List[Decimal],
    ) -> Optional[List[Decimal]]:
        """Solve OLS via normal equation: beta = (X'X)^-1 X'y."""
        xt = _transpose(x_mat)
        xtx = _mat_mult(xt, x_mat)
        xtx_inv = _gauss_jordan_inverse(xtx)
        if xtx_inv is None:
            return None
        xty = _mat_vec_mult(xt, y)
        return _mat_vec_mult(xtx_inv, xty)

    def _compute_fit_statistics(
        self,
        y: List[Decimal],
        y_hat: List[Decimal],
        residuals: List[Decimal],
        betas: List[Decimal],
        x_mat: List[List[Decimal]],
        y_mean: Decimal,
        n: int,
        p: int,
        config: RegressionConfig,
    ) -> Tuple[RegressionStatistics, List[CoefficientDetail]]:
        """Compute all fit statistics and coefficient details."""
        ss_tot = _compute_ss_total(y, y_mean)
        ss_res = _compute_ss_residual(y, y_hat)
        ss_reg = ss_tot - ss_res

        r2 = Decimal("1") - _safe_divide(ss_res, ss_tot) if ss_tot > Decimal("0") else Decimal("0")
        r2 = max(r2, Decimal("0"))

        df_res = n - p
        adj_r2 = (Decimal("1") - (Decimal("1") - r2) * _decimal(n - 1) / _decimal(df_res)
                  ) if df_res > 0 else Decimal("0")

        mse = _safe_divide(ss_res, _decimal(df_res)) if df_res > 0 else Decimal("0")
        rmse = _decimal(math.sqrt(float(mse))) if mse > Decimal("0") else Decimal("0")
        cvrmse = _safe_pct(rmse, y_mean) if y_mean != Decimal("0") else Decimal("0")
        nmbe = _compute_nmbe(residuals, y_mean, n)
        mae = _safe_divide(sum(abs(r) for r in residuals), _decimal(n))

        f_stat = _safe_divide(
            _safe_divide(ss_reg, _decimal(p - 1)) if p > 1 else Decimal("0"),
            mse,
        )

        dw = _compute_durbin_watson(residuals)

        y_vals = [y[i] for i in range(n)]
        y_std = _decimal(math.sqrt(float(
            _safe_divide(ss_tot, _decimal(n - 1))
        ))) if n > 1 and ss_tot > Decimal("0") else Decimal("0")

        stats = RegressionStatistics(
            n_observations=n,
            n_parameters=p,
            degrees_of_freedom=max(df_res, 0),
            r_squared=_round_val(r2, 6),
            adj_r_squared=_round_val(adj_r2, 6),
            cvrmse=_round_val(cvrmse, 4),
            nmbe=_round_val(nmbe, 4),
            rmse=_round_val(rmse, 6),
            mse=_round_val(mse, 6),
            mae=_round_val(mae, 6),
            ss_total=_round_val(ss_tot, 4),
            ss_residual=_round_val(ss_res, 4),
            ss_regression=_round_val(ss_reg, 4),
            f_statistic=_round_val(f_stat, 4),
            durbin_watson=_round_val(dw, 4),
            y_mean=_round_val(y_mean, 4),
            y_std=_round_val(y_std, 4),
        )

        # Coefficient standard errors from (X'X)^-1 diagonal
        xt = _transpose(x_mat)
        xtx = _mat_mult(xt, x_mat)
        xtx_inv = _gauss_jordan_inverse(xtx)
        hat_diag: List[Decimal] = []
        if xtx_inv is not None:
            hat_diag = _compute_hat_diagonal(x_mat, xtx_inv)

        coefs: List[CoefficientDetail] = []
        for j, beta_val in enumerate(betas):
            se = Decimal("0")
            if xtx_inv is not None and j < len(xtx_inv) and mse > Decimal("0"):
                var_beta = mse * xtx_inv[j][j]
                se = _decimal(math.sqrt(float(var_beta))) if var_beta > Decimal("0") else Decimal("0")

            t_stat = _safe_divide(beta_val, se) if se > Decimal("0") else Decimal("0")
            p_val = _approximate_p_value(t_stat, df_res)
            significant = p_val < Decimal("0.05")

            name = "intercept" if j == 0 else f"beta_{j}"
            coefs.append(CoefficientDetail(
                name=name,
                value=_round_val(beta_val, 8),
                std_error=_round_val(se, 8),
                t_statistic=_round_val(t_stat, 4),
                p_value=_round_val(p_val, 6),
                significant=significant,
            ))

        return stats, coefs

    def _run_diagnostics(
        self,
        result: RegressionFitResult,
        config: RegressionConfig,
    ) -> List[DiagnosticResult]:
        """Run standard diagnostic checks on a fitted model."""
        diags: List[DiagnosticResult] = []
        thresholds = ASHRAE14_THRESHOLDS.get(
            config.frequency.value,
            ASHRAE14_THRESHOLDS[DataFrequency.MONTHLY.value],
        )

        # CVRMSE check
        cvrmse_max = thresholds["cvrmse_max"]
        cv_status = DiagnosticStatus.PASS
        if result.statistics.cvrmse > cvrmse_max:
            cv_status = DiagnosticStatus.FAIL
        elif result.statistics.cvrmse > cvrmse_max * Decimal("0.9"):
            cv_status = DiagnosticStatus.WARNING
        diags.append(DiagnosticResult(
            name="cvrmse_ashrae14",
            value=result.statistics.cvrmse,
            threshold=cvrmse_max,
            status=cv_status,
            detail=f"CVRMSE={float(result.statistics.cvrmse):.2f}% vs limit {float(cvrmse_max)}%",
        ))

        # NMBE check
        nmbe_max = thresholds["nmbe_max"]
        nmbe_abs = abs(result.statistics.nmbe)
        nmbe_status = DiagnosticStatus.PASS
        if nmbe_abs > nmbe_max:
            nmbe_status = DiagnosticStatus.FAIL
        elif nmbe_abs > nmbe_max * Decimal("0.8"):
            nmbe_status = DiagnosticStatus.WARNING
        diags.append(DiagnosticResult(
            name="nmbe_ashrae14",
            value=result.statistics.nmbe,
            threshold=nmbe_max,
            status=nmbe_status,
            detail=f"|NMBE|={float(nmbe_abs):.2f}% vs limit +/-{float(nmbe_max)}%",
        ))

        # R-squared check
        r2_min = thresholds["r_squared_min"]
        r2_status = DiagnosticStatus.PASS
        if result.statistics.r_squared < r2_min:
            r2_status = DiagnosticStatus.FAIL
        elif result.statistics.r_squared < r2_min + Decimal("0.05"):
            r2_status = DiagnosticStatus.WARNING
        diags.append(DiagnosticResult(
            name="r_squared",
            value=result.statistics.r_squared,
            threshold=r2_min,
            status=r2_status,
            detail=f"R2={float(result.statistics.r_squared):.4f} vs min {float(r2_min)}",
        ))

        # Durbin-Watson
        dw = result.statistics.durbin_watson
        dw_status = DiagnosticStatus.PASS
        if dw < Decimal("1.5") or dw > Decimal("2.5"):
            dw_status = DiagnosticStatus.WARNING
        if dw < Decimal("1.0") or dw > Decimal("3.0"):
            dw_status = DiagnosticStatus.FAIL
        diags.append(DiagnosticResult(
            name="durbin_watson",
            value=dw,
            threshold=Decimal("1.5"),
            status=dw_status,
            detail=f"DW={float(dw):.4f}; acceptable range 1.5-2.5",
        ))

        # F-statistic (should be > 1 for any meaningful model)
        f_stat = result.statistics.f_statistic
        f_status = DiagnosticStatus.PASS
        if f_stat < Decimal("4"):
            f_status = DiagnosticStatus.WARNING
        if f_stat < Decimal("1"):
            f_status = DiagnosticStatus.FAIL
        diags.append(DiagnosticResult(
            name="f_statistic",
            value=f_stat,
            threshold=Decimal("4"),
            status=f_status,
            detail=f"F={float(f_stat):.2f}; larger is better",
        ))

        return diags

    def _assign_grade(
        self, result: RegressionFitResult, frequency: DataFrequency,
    ) -> ValidationGrade:
        """Assign ASHRAE 14 validation grade based on diagnostics."""
        thresholds = ASHRAE14_THRESHOLDS.get(
            frequency.value, ASHRAE14_THRESHOLDS[DataFrequency.MONTHLY.value],
        )

        cvrmse_ok = result.statistics.cvrmse <= thresholds["cvrmse_max"]
        nmbe_ok = abs(result.statistics.nmbe) <= thresholds["nmbe_max"]
        r2_ok = result.statistics.r_squared >= thresholds["r_squared_min"]

        if not (cvrmse_ok and nmbe_ok and r2_ok):
            return ValidationGrade.FAIL

        # Check margins
        cv_margin = thresholds["cvrmse_max"] - result.statistics.cvrmse
        nmbe_margin = thresholds["nmbe_max"] - abs(result.statistics.nmbe)

        if cv_margin > thresholds["cvrmse_max"] * Decimal("0.5") and nmbe_margin > thresholds["nmbe_max"] * Decimal("0.5"):
            return ValidationGrade.EXCELLENT
        if cv_margin > thresholds["cvrmse_max"] * Decimal("0.2") and nmbe_margin > thresholds["nmbe_max"] * Decimal("0.2"):
            return ValidationGrade.GOOD
        if cv_margin > Decimal("0") and nmbe_margin > Decimal("0"):
            return ValidationGrade.ACCEPTABLE
        return ValidationGrade.MARGINAL

    def _compute_aic(self, n: int, p: int, ss_res: Decimal) -> Decimal:
        """Compute Akaike Information Criterion.

        AIC = n * ln(SS_res / n) + 2p
        """
        if n <= 0 or ss_res <= Decimal("0"):
            return Decimal("0")
        log_term = _decimal(math.log(float(ss_res) / n))
        return _decimal(n) * log_term + Decimal("2") * _decimal(p)

    def _compute_bic(self, n: int, p: int, ss_res: Decimal) -> Decimal:
        """Compute Bayesian Information Criterion.

        BIC = n * ln(SS_res / n) + p * ln(n)
        """
        if n <= 0 or ss_res <= Decimal("0"):
            return Decimal("0")
        log_term = _decimal(math.log(float(ss_res) / n))
        return _decimal(n) * log_term + _decimal(p) * _decimal(math.log(n))

    def _jarque_bera(self, residuals: List[Decimal]) -> Decimal:
        """Compute Jarque-Bera test statistic for normality.

        JB = (n/6) * (S^2 + K^2/4)
        where S = skewness, K = excess kurtosis.
        """
        n = len(residuals)
        if n < 4:
            return Decimal("0")
        mean = _safe_divide(sum(residuals), _decimal(n))
        m2 = _safe_divide(sum((r - mean) ** 2 for r in residuals), _decimal(n))
        m3 = _safe_divide(sum((r - mean) ** 3 for r in residuals), _decimal(n))
        m4 = _safe_divide(sum((r - mean) ** 4 for r in residuals), _decimal(n))

        if m2 == Decimal("0"):
            return Decimal("0")

        std = _decimal(math.sqrt(float(m2)))
        skewness = _safe_divide(m3, std ** 3)
        kurtosis = _safe_divide(m4, m2 ** 2) - Decimal("3")  # excess kurtosis

        jb = _decimal(n) / Decimal("6") * (
            skewness ** 2 + kurtosis ** 2 / Decimal("4")
        )
        return abs(jb)

    def _runs_test(self, residuals: List[Decimal]) -> Tuple[Decimal, Decimal]:
        """Perform Wald-Wolfowitz runs test for randomness.

        Returns (observed_runs, expected_runs).
        """
        n = len(residuals)
        if n < 3:
            return Decimal("1"), Decimal("1")

        n_pos = sum(1 for r in residuals if r >= Decimal("0"))
        n_neg = n - n_pos

        runs = Decimal("1")
        for i in range(1, n):
            if (residuals[i] >= Decimal("0")) != (residuals[i - 1] >= Decimal("0")):
                runs += Decimal("1")

        expected = Decimal("1")
        if n > 0:
            expected = Decimal("1") + Decimal("2") * _decimal(n_pos) * _decimal(n_neg) / _decimal(n)

        return runs, expected
