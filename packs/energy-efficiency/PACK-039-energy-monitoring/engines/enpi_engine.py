# -*- coding: utf-8 -*-
"""
EnPIEngine - PACK-039 Energy Monitoring Engine 5
==================================================

ISO 50001 Energy Performance Indicator (EnPI) engine with regression
normalization, CUSUM savings tracking, and statistical significance
testing for energy management systems.

Calculation Methodology:
    Simple Ratio EnPI:
        EnPI = energy_consumption / relevant_variable
        Example: kWh / m2, kWh / unit_produced

    Regression EnPI (multivariate):
        y_hat = b0 + b1*HDD + b2*CDD + b3*production + ...
        Coefficients from Ordinary Least Squares (OLS).

    CUSUM Savings Tracking:
        savings_i = predicted_i - actual_i
        CUSUM_n = SUM(savings_1 ... savings_n)

    Model Statistics:
        R^2 = 1 - SS_res / SS_tot
        Adjusted R^2 = 1 - (1 - R^2) * (n - 1) / (n - p - 1)
        RMSE = sqrt(SS_res / n)
        CVRMSE = RMSE / y_mean * 100
        ASHRAE 14: CVRMSE < 25% monthly, < 30% daily

    F-test (overall model significance):
        F = (SS_reg / p) / (SS_res / (n - p - 1))

    t-test (coefficient significance):
        t = b_j / SE(b_j)

    Confidence Interval:
        b_j +/- t_alpha/2 * SE(b_j)

    Performance Rating:
        significantly_improved: savings > 2 * std_error
        improved: savings > 0 (statistically significant)
        stable: no significant change
        declined: consumption increased significantly

Regulatory References:
    - ISO 50001:2018 - Energy management systems
    - ISO 50006:2014 - Measuring energy performance using EnPIs
    - ISO 50015:2014 - M&V of energy performance
    - ASHRAE Guideline 14-2014 - CVRMSE thresholds
    - IPMVP Volume I - Option C whole-facility regression
    - Superior Energy Performance (SEP) programme
    - ANSI/MSE 50021:2014 - Assessment of EnMS

Zero-Hallucination:
    - All regression uses closed-form OLS (normal equation)
    - No LLM involvement in any calculation path
    - Statistical tests use deterministic formulas
    - CUSUM tracks cumulative differences only
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-039 Energy Monitoring
Engine:  5 of 5
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

class EnPIType(str, Enum):
    """Type of Energy Performance Indicator methodology.

    SIMPLE_RATIO:      Simple ratio (energy / variable).
    REGRESSION:        Multivariate regression model.
    STATISTICAL_MODEL: Advanced statistical model.
    CUSUM_BASED:       CUSUM-derived performance tracking.
    """
    SIMPLE_RATIO = "simple_ratio"
    REGRESSION = "regression"
    STATISTICAL_MODEL = "statistical_model"
    CUSUM_BASED = "cusum_based"

class RelevantVariable(str, Enum):
    """ISO 50006 relevant variables that affect energy consumption.

    HDD:               Heating degree days.
    CDD:               Cooling degree days.
    PRODUCTION_VOLUME: Production output volume.
    OCCUPANCY:         Building occupancy rate.
    OPERATING_HOURS:   Equipment operating hours.
    FLOOR_AREA:        Conditioned floor area.
    """
    HDD = "hdd"
    CDD = "cdd"
    PRODUCTION_VOLUME = "production_volume"
    OCCUPANCY = "occupancy"
    OPERATING_HOURS = "operating_hours"
    FLOOR_AREA = "floor_area"

class BaselineStatus(str, Enum):
    """Status of an energy baseline per ISO 50001.

    DRAFT:       Baseline under development.
    ACTIVE:      Currently active baseline.
    EXPIRED:     Baseline validity period has expired.
    SUPERSEDED:  Replaced by a newer baseline.
    """
    DRAFT = "draft"
    ACTIVE = "active"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"

class SignificanceLevel(str, Enum):
    """Statistical significance level for hypothesis testing.

    ALPHA_001: 99.9% confidence (p < 0.001).
    ALPHA_005: 95% confidence (p < 0.05).
    ALPHA_010: 90% confidence (p < 0.10).
    """
    ALPHA_001 = "0.001"
    ALPHA_005 = "0.05"
    ALPHA_010 = "0.10"

class PerformanceRating(str, Enum):
    """Energy performance rating relative to baseline.

    SIGNIFICANTLY_IMPROVED: Statistically significant improvement.
    IMPROVED:               Improvement detected.
    STABLE:                 No significant change.
    DECLINED:               Performance has declined.
    SIGNIFICANTLY_DECLINED: Statistically significant decline.
    """
    SIGNIFICANTLY_IMPROVED = "significantly_improved"
    IMPROVED = "improved"
    STABLE = "stable"
    DECLINED = "declined"
    SIGNIFICANTLY_DECLINED = "significantly_declined"

class RegressionQuality(str, Enum):
    """Quality classification of a regression model.

    EXCELLENT:   R^2 >= 0.90 and CVRMSE <= 15%.
    GOOD:        R^2 >= 0.75 and CVRMSE <= 25%.
    ACCEPTABLE:  R^2 >= 0.50 and CVRMSE <= 30%.
    POOR:        Below acceptable thresholds.
    """
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ASHRAE Guideline 14 CVRMSE thresholds.
CVRMSE_MONTHLY_THRESHOLD: Decimal = Decimal("25")  # %
CVRMSE_DAILY_THRESHOLD: Decimal = Decimal("30")    # %

# Minimum data points for regression.
MIN_REGRESSION_POINTS: int = 12

# t-distribution critical values (approximate for df >= 30).
T_CRITICAL: Dict[str, Decimal] = {
    "0.001": Decimal("3.646"),
    "0.01": Decimal("2.750"),
    "0.05": Decimal("2.042"),
    "0.10": Decimal("1.697"),
}

# F-distribution critical values (approximate for p=2, df2>=30).
F_CRITICAL: Dict[str, Decimal] = {
    "0.001": Decimal("8.77"),
    "0.01": Decimal("5.39"),
    "0.05": Decimal("3.32"),
    "0.10": Decimal("2.49"),
}

# Regression quality thresholds.
REGRESSION_QUALITY_THRESHOLDS: List[Tuple[Decimal, Decimal, RegressionQuality]] = [
    (Decimal("0.90"), Decimal("15"), RegressionQuality.EXCELLENT),
    (Decimal("0.75"), Decimal("25"), RegressionQuality.GOOD),
    (Decimal("0.50"), Decimal("30"), RegressionQuality.ACCEPTABLE),
    (Decimal("0"), Decimal("100"), RegressionQuality.POOR),
]

# Default base temperature for degree days.
BASE_TEMP_HEATING_C: Decimal = Decimal("18")
BASE_TEMP_COOLING_C: Decimal = Decimal("18")

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class EnPIDefinition(BaseModel):
    """Definition of an Energy Performance Indicator.

    Attributes:
        enpi_id:             Unique EnPI identifier.
        name:                EnPI name.
        description:         EnPI description.
        enpi_type:           Calculation methodology.
        energy_scope:        What energy is measured (electricity, gas, total).
        unit:                EnPI unit (e.g., kWh/m2).
        numerator_unit:      Numerator unit.
        denominator_unit:    Denominator unit.
        relevant_variables:  List of relevant variables.
        target_value:        Target EnPI value.
        boundary:            Measurement boundary description.
        frequency:           Calculation frequency (monthly/daily).
        created_at:          Creation timestamp.
    """
    enpi_id: str = Field(default_factory=_new_uuid)
    name: str = Field(default="", max_length=200)
    description: str = Field(default="", max_length=1000)
    enpi_type: EnPIType = Field(default=EnPIType.SIMPLE_RATIO)
    energy_scope: str = Field(default="electricity", max_length=100)
    unit: str = Field(default="kWh/unit", max_length=50)
    numerator_unit: str = Field(default="kWh", max_length=50)
    denominator_unit: str = Field(default="unit", max_length=50)
    relevant_variables: List[RelevantVariable] = Field(default_factory=list)
    target_value: Optional[Decimal] = Field(default=None)
    boundary: str = Field(default="whole_facility", max_length=200)
    frequency: str = Field(default="monthly", max_length=20)
    created_at: datetime = Field(default_factory=utcnow)

class RegressionModel(BaseModel):
    """Regression model parameters and statistics.

    Attributes:
        intercept:          b0 coefficient.
        coefficients:       Dict of variable -> coefficient value.
        r_squared:          R-squared (coefficient of determination).
        adjusted_r_squared: Adjusted R-squared.
        rmse:               Root Mean Square Error.
        cvrmse:             Coefficient of Variation of RMSE (%).
        f_statistic:        F-statistic for overall significance.
        f_p_value_approx:   Approximate F-test p-value category.
        t_statistics:       t-statistics for each coefficient.
        std_errors:         Standard errors for each coefficient.
        n_observations:     Number of observations.
        n_predictors:       Number of predictors.
        ss_total:           Total sum of squares.
        ss_regression:      Regression sum of squares.
        ss_residual:        Residual sum of squares.
        quality:            Model quality classification.
        meets_ashrae14:     Whether model meets ASHRAE 14 criteria.
    """
    intercept: Decimal = Field(default=Decimal("0"))
    coefficients: Dict[str, Decimal] = Field(default_factory=dict)
    r_squared: Decimal = Field(default=Decimal("0"))
    adjusted_r_squared: Decimal = Field(default=Decimal("0"))
    rmse: Decimal = Field(default=Decimal("0"))
    cvrmse: Decimal = Field(default=Decimal("0"))
    f_statistic: Decimal = Field(default=Decimal("0"))
    f_p_value_approx: str = Field(default="not_significant")
    t_statistics: Dict[str, Decimal] = Field(default_factory=dict)
    std_errors: Dict[str, Decimal] = Field(default_factory=dict)
    n_observations: int = Field(default=0, ge=0)
    n_predictors: int = Field(default=0, ge=0)
    ss_total: Decimal = Field(default=Decimal("0"))
    ss_regression: Decimal = Field(default=Decimal("0"))
    ss_residual: Decimal = Field(default=Decimal("0"))
    quality: RegressionQuality = Field(default=RegressionQuality.POOR)
    meets_ashrae14: bool = Field(default=False)

class EnergyBaseline(BaseModel):
    """ISO 50001 energy baseline definition.

    Attributes:
        baseline_id:       Unique baseline identifier.
        name:              Baseline name.
        period_start:      Baseline period start.
        period_end:        Baseline period end.
        status:            Baseline status.
        total_energy_kwh:  Total energy in baseline period.
        regression_model:  Regression model if applicable.
        enpi_baseline_value: Baseline EnPI value.
        relevant_variables: Variables used.
        data_points:       Number of data points.
        notes:             Baseline notes.
        created_at:        Creation timestamp.
        provenance_hash:   SHA-256 audit hash.
    """
    baseline_id: str = Field(default_factory=_new_uuid)
    name: str = Field(default="", max_length=200)
    period_start: datetime = Field(default_factory=utcnow)
    period_end: datetime = Field(default_factory=utcnow)
    status: BaselineStatus = Field(default=BaselineStatus.DRAFT)
    total_energy_kwh: Decimal = Field(default=Decimal("0"))
    regression_model: Optional[RegressionModel] = Field(default=None)
    enpi_baseline_value: Decimal = Field(default=Decimal("0"))
    relevant_variables: List[str] = Field(default_factory=list)
    data_points: int = Field(default=0, ge=0)
    notes: str = Field(default="", max_length=2000)
    created_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class EnPIValue(BaseModel):
    """A single calculated EnPI value for a reporting period.

    Attributes:
        value_id:           Unique value identifier.
        enpi_id:            EnPI this value belongs to.
        period_start:       Reporting period start.
        period_end:         Reporting period end.
        actual_energy_kwh:  Actual energy consumed.
        predicted_energy_kwh: Baseline-predicted energy.
        enpi_value:         Calculated EnPI value.
        savings_kwh:        Energy savings (predicted - actual).
        savings_pct:        Savings as percentage.
        variable_values:    Values of relevant variables.
        provenance_hash:    SHA-256 audit hash.
    """
    value_id: str = Field(default_factory=_new_uuid)
    enpi_id: str = Field(default="")
    period_start: datetime = Field(default_factory=utcnow)
    period_end: datetime = Field(default_factory=utcnow)
    actual_energy_kwh: Decimal = Field(default=Decimal("0"))
    predicted_energy_kwh: Decimal = Field(default=Decimal("0"))
    enpi_value: Decimal = Field(default=Decimal("0"))
    savings_kwh: Decimal = Field(default=Decimal("0"))
    savings_pct: Decimal = Field(default=Decimal("0"))
    variable_values: Dict[str, Decimal] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")

class CUSUMTracker(BaseModel):
    """CUSUM (Cumulative Sum) savings tracker.

    Attributes:
        tracker_id:          Unique tracker identifier.
        enpi_id:             EnPI being tracked.
        baseline_id:         Baseline used.
        cumulative_savings:  Running cumulative savings (kWh).
        period_savings:      List of per-period savings.
        total_periods:       Number of periods tracked.
        trend:               Savings trend direction.
        performance_rating:  Overall performance rating.
        std_error:           Standard error of savings.
        is_significant:      Whether savings are statistically significant.
        calculated_at:       Calculation timestamp.
        provenance_hash:     SHA-256 audit hash.
    """
    tracker_id: str = Field(default_factory=_new_uuid)
    enpi_id: str = Field(default="")
    baseline_id: str = Field(default="")
    cumulative_savings: Decimal = Field(default=Decimal("0"))
    period_savings: List[Dict[str, Decimal]] = Field(default_factory=list)
    total_periods: int = Field(default=0, ge=0)
    trend: str = Field(default="stable", max_length=50)
    performance_rating: PerformanceRating = Field(
        default=PerformanceRating.STABLE
    )
    std_error: Decimal = Field(default=Decimal("0"))
    is_significant: bool = Field(default=False)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class EnPIResult(BaseModel):
    """Complete EnPI analysis result.

    Attributes:
        result_id:           Unique result identifier.
        enpi_definition:     EnPI definition.
        baseline:            Energy baseline.
        current_values:      List of current period EnPI values.
        cusum_tracker:       CUSUM savings tracker.
        regression_model:    Regression model (if applicable).
        performance_rating:  Overall performance rating.
        total_savings_kwh:   Total savings across all periods.
        total_savings_pct:   Overall savings percentage.
        total_savings_cost:  Total cost savings.
        recommendations:     List of recommendations.
        processing_time_ms:  Processing duration.
        calculated_at:       Calculation timestamp.
        provenance_hash:     SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    enpi_definition: EnPIDefinition = Field(default_factory=EnPIDefinition)
    baseline: EnergyBaseline = Field(default_factory=EnergyBaseline)
    current_values: List[EnPIValue] = Field(default_factory=list)
    cusum_tracker: CUSUMTracker = Field(default_factory=CUSUMTracker)
    regression_model: Optional[RegressionModel] = Field(default=None)
    performance_rating: PerformanceRating = Field(
        default=PerformanceRating.STABLE
    )
    total_savings_kwh: Decimal = Field(default=Decimal("0"))
    total_savings_pct: Decimal = Field(default=Decimal("0"))
    total_savings_cost: Decimal = Field(default=Decimal("0"))
    recommendations: List[str] = Field(default_factory=list)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class EnPIEngine:
    """ISO 50001 Energy Performance Indicator engine.

    Implements regression-based EnPI calculation, CUSUM savings tracking,
    statistical significance testing, and baseline management per
    ISO 50001/50006/50015.

    Usage::

        engine = EnPIEngine()
        definition = engine.define_enpi("EUI", "kWh/m2",
            relevant_variables=[RelevantVariable.HDD, RelevantVariable.CDD])
        baseline = engine.build_baseline(definition, baseline_data)
        result = engine.calculate_enpi(definition, baseline, current_data)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise EnPIEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - significance_level (str): alpha for hypothesis tests
                - cvrmse_threshold (float): override ASHRAE 14 threshold
                - energy_cost_per_kwh (float): cost for savings calc
                - base_temp_heating (float): HDD base temperature
                - base_temp_cooling (float): CDD base temperature
        """
        self.config = config or {}
        self._alpha = self.config.get("significance_level", "0.05")
        self._cvrmse_threshold = _decimal(
            self.config.get("cvrmse_threshold", CVRMSE_MONTHLY_THRESHOLD)
        )
        self._energy_cost = _decimal(
            self.config.get("energy_cost_per_kwh", "0.12")
        )
        self._base_temp_heat = _decimal(
            self.config.get("base_temp_heating", BASE_TEMP_HEATING_C)
        )
        self._base_temp_cool = _decimal(
            self.config.get("base_temp_cooling", BASE_TEMP_COOLING_C)
        )
        self._definitions: Dict[str, EnPIDefinition] = {}
        self._baselines: Dict[str, EnergyBaseline] = {}
        logger.info(
            "EnPIEngine v%s initialised (alpha=%s, cvrmse_threshold=%s%%)",
            self.engine_version, self._alpha, str(self._cvrmse_threshold),
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def define_enpi(
        self,
        name: str,
        unit: str,
        enpi_type: EnPIType = EnPIType.REGRESSION,
        relevant_variables: Optional[List[RelevantVariable]] = None,
        energy_scope: str = "electricity",
        target_value: Optional[Decimal] = None,
        boundary: str = "whole_facility",
        frequency: str = "monthly",
        description: str = "",
    ) -> EnPIDefinition:
        """Define a new Energy Performance Indicator.

        Args:
            name:                EnPI name.
            unit:                EnPI unit (e.g., kWh/m2).
            enpi_type:           Calculation methodology.
            relevant_variables:  ISO 50006 relevant variables.
            energy_scope:        Energy scope.
            target_value:        Target EnPI value.
            boundary:            Measurement boundary.
            frequency:           Calculation frequency.
            description:         Description.

        Returns:
            EnPIDefinition.
        """
        t0 = time.perf_counter()
        logger.info("Defining EnPI: %s (%s)", name, unit)

        parts = unit.split("/")
        num_unit = parts[0].strip() if parts else "kWh"
        den_unit = parts[1].strip() if len(parts) > 1 else "unit"

        definition = EnPIDefinition(
            name=name,
            description=description or f"Energy Performance Indicator: {name}",
            enpi_type=enpi_type,
            energy_scope=energy_scope,
            unit=unit,
            numerator_unit=num_unit,
            denominator_unit=den_unit,
            relevant_variables=relevant_variables or [],
            target_value=target_value,
            boundary=boundary,
            frequency=frequency,
        )

        self._definitions[definition.enpi_id] = definition

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "EnPI defined: %s (id=%s, type=%s) (%.1f ms)",
            name, definition.enpi_id[:12], enpi_type.value, elapsed,
        )
        return definition

    def build_baseline(
        self,
        definition: EnPIDefinition,
        data: List[Dict[str, Any]],
        name: str = "Baseline",
    ) -> EnergyBaseline:
        """Build an energy baseline from historical data.

        For regression EnPIs, computes OLS regression model.
        For simple ratio EnPIs, computes average ratio.

        Args:
            definition: EnPI definition.
            data:       List of dicts with 'energy_kwh' and variable values.
            name:       Baseline name.

        Returns:
            EnergyBaseline with regression model if applicable.
        """
        t0 = time.perf_counter()
        logger.info(
            "Building baseline for %s (%d data points)",
            definition.name, len(data),
        )

        if not data:
            result = EnergyBaseline(name=name, status=BaselineStatus.DRAFT)
            result.provenance_hash = _compute_hash(result)
            return result

        total_energy = sum(
            (_decimal(d.get("energy_kwh", 0)) for d in data), Decimal("0")
        )

        regression_model = None
        enpi_baseline_value = Decimal("0")
        var_names: List[str] = [v.value for v in definition.relevant_variables]

        if definition.enpi_type == EnPIType.REGRESSION and var_names:
            regression_model = self._fit_regression(data, var_names)
            enpi_baseline_value = _safe_divide(
                total_energy, _decimal(len(data))
            )
        else:
            # Simple ratio
            denom_total = Decimal("0")
            for d in data:
                for var in var_names:
                    denom_total += _decimal(d.get(var, 0))
            if not var_names:
                denom_total = _decimal(len(data))
            enpi_baseline_value = _safe_divide(total_energy, denom_total)

        timestamps = [d.get("timestamp", utcnow()) for d in data]
        valid_ts = [t for t in timestamps if isinstance(t, datetime)]
        period_start = min(valid_ts) if valid_ts else utcnow()
        period_end = max(valid_ts) if valid_ts else utcnow()

        baseline = EnergyBaseline(
            name=name,
            period_start=period_start,
            period_end=period_end,
            status=BaselineStatus.ACTIVE,
            total_energy_kwh=_round_val(total_energy, 2),
            regression_model=regression_model,
            enpi_baseline_value=_round_val(enpi_baseline_value, 4),
            relevant_variables=var_names,
            data_points=len(data),
        )
        baseline.provenance_hash = _compute_hash(baseline)

        self._baselines[baseline.baseline_id] = baseline

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Baseline built: %s, energy=%s kWh, EnPI=%s, "
            "R2=%s, hash=%s (%.1f ms)",
            name, str(_round_val(total_energy, 0)),
            str(_round_val(enpi_baseline_value, 4)),
            str(regression_model.r_squared) if regression_model else "N/A",
            baseline.provenance_hash[:16], elapsed,
        )
        return baseline

    def calculate_enpi(
        self,
        definition: EnPIDefinition,
        baseline: EnergyBaseline,
        current_data: List[Dict[str, Any]],
    ) -> EnPIResult:
        """Calculate EnPI values for current reporting periods.

        Args:
            definition:   EnPI definition.
            baseline:     Energy baseline.
            current_data: List of current period data points.

        Returns:
            EnPIResult with values, CUSUM tracker, and recommendations.
        """
        t0 = time.perf_counter()
        logger.info(
            "Calculating EnPI %s for %d periods",
            definition.name, len(current_data),
        )

        if not current_data:
            result = EnPIResult(
                enpi_definition=definition,
                baseline=baseline,
            )
            result.provenance_hash = _compute_hash(result)
            return result

        var_names = [v.value for v in definition.relevant_variables]
        enpi_values: List[EnPIValue] = []
        cumulative_savings = Decimal("0")
        period_savings_list: List[Dict[str, Decimal]] = []

        for d in current_data:
            actual = _decimal(d.get("energy_kwh", 0))
            period_start = d.get("period_start", d.get("timestamp", utcnow()))
            period_end = d.get("period_end", period_start)

            # Predict baseline energy
            predicted = self._predict_energy(baseline, d, var_names)

            # Calculate EnPI value
            if definition.enpi_type == EnPIType.SIMPLE_RATIO:
                denom = Decimal("0")
                for var in var_names:
                    denom += _decimal(d.get(var, 0))
                if denom == Decimal("0"):
                    denom = Decimal("1")
                enpi_val = _safe_divide(actual, denom)
            else:
                enpi_val = actual  # For regression, actual value is the EnPI

            # Savings
            savings = predicted - actual
            savings_pct = _safe_pct(savings, predicted)
            cumulative_savings += savings

            variable_values = {
                var: _decimal(d.get(var, 0)) for var in var_names
            }

            ev = EnPIValue(
                enpi_id=definition.enpi_id,
                period_start=period_start,
                period_end=period_end,
                actual_energy_kwh=_round_val(actual, 2),
                predicted_energy_kwh=_round_val(predicted, 2),
                enpi_value=_round_val(enpi_val, 4),
                savings_kwh=_round_val(savings, 2),
                savings_pct=_round_val(savings_pct, 2),
                variable_values=variable_values,
            )
            ev.provenance_hash = _compute_hash(ev)
            enpi_values.append(ev)

            period_savings_list.append({
                "period": str(period_start),
                "savings_kwh": _round_val(savings, 2),
                "cumulative_kwh": _round_val(cumulative_savings, 2),
            })

        # Build CUSUM tracker
        cusum_tracker = self.track_cusum(
            definition.enpi_id,
            baseline.baseline_id,
            period_savings_list,
            baseline.regression_model,
        )

        # Overall savings
        total_actual = sum((ev.actual_energy_kwh for ev in enpi_values), Decimal("0"))
        total_predicted = sum((ev.predicted_energy_kwh for ev in enpi_values), Decimal("0"))
        total_savings = total_predicted - total_actual
        total_savings_pct = _safe_pct(total_savings, total_predicted)
        total_savings_cost = _round_val(total_savings * self._energy_cost, 2)

        # Performance rating
        performance = cusum_tracker.performance_rating

        # Recommendations
        recommendations = self._generate_recommendations(
            definition, baseline, enpi_values, cusum_tracker,
            total_savings_pct,
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000.0)

        result = EnPIResult(
            enpi_definition=definition,
            baseline=baseline,
            current_values=enpi_values,
            cusum_tracker=cusum_tracker,
            regression_model=baseline.regression_model,
            performance_rating=performance,
            total_savings_kwh=_round_val(total_savings, 2),
            total_savings_pct=_round_val(total_savings_pct, 2),
            total_savings_cost=total_savings_cost,
            recommendations=recommendations,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "EnPI calculated: %s, savings=%s kWh (%.1f%%), rating=%s, "
            "hash=%s (%.1f ms)",
            definition.name, str(_round_val(total_savings, 0)),
            float(total_savings_pct), performance.value,
            result.provenance_hash[:16], float(elapsed_ms),
        )
        return result

    def track_cusum(
        self,
        enpi_id: str,
        baseline_id: str,
        period_savings: List[Dict[str, Decimal]],
        regression_model: Optional[RegressionModel] = None,
    ) -> CUSUMTracker:
        """Track cumulative savings using CUSUM methodology.

        CUSUM_n = SUM(savings_1 ... savings_n)

        Args:
            enpi_id:          EnPI identifier.
            baseline_id:      Baseline identifier.
            period_savings:   Per-period savings records.
            regression_model: Regression model for std error.

        Returns:
            CUSUMTracker with cumulative savings and significance.
        """
        t0 = time.perf_counter()
        logger.info("Tracking CUSUM for EnPI %s", enpi_id[:12])

        if not period_savings:
            result = CUSUMTracker(enpi_id=enpi_id, baseline_id=baseline_id)
            result.provenance_hash = _compute_hash(result)
            return result

        cumulative = Decimal("0")
        savings_values: List[Decimal] = []

        for ps in period_savings:
            s = _decimal(ps.get("savings_kwh", 0))
            cumulative += s
            savings_values.append(s)

        # Standard error of savings
        std_error = Decimal("0")
        if regression_model and regression_model.rmse > Decimal("0"):
            n = _decimal(len(period_savings))
            std_error = regression_model.rmse * _decimal(math.sqrt(float(n)))
        elif len(savings_values) > 1:
            mean_s = _safe_divide(
                sum(savings_values, Decimal("0")), _decimal(len(savings_values))
            )
            variance = _safe_divide(
                sum(((s - mean_s) ** 2 for s in savings_values), Decimal("0")),
                _decimal(len(savings_values) - 1),
            )
            std_error = _decimal(math.sqrt(float(variance)))

        # Significance test
        t_critical = T_CRITICAL.get(self._alpha, Decimal("2.042"))
        is_significant = False
        if std_error > Decimal("0"):
            t_stat = _safe_divide(abs(cumulative), std_error)
            is_significant = t_stat > t_critical

        # Trend determination
        if len(savings_values) >= 3:
            recent = savings_values[-3:]
            if all(s > Decimal("0") for s in recent):
                trend = "improving"
            elif all(s < Decimal("0") for s in recent):
                trend = "declining"
            else:
                trend = "variable"
        else:
            trend = "insufficient_data"

        # Performance rating
        rating = self._rate_performance(cumulative, std_error, is_significant)

        tracker = CUSUMTracker(
            enpi_id=enpi_id,
            baseline_id=baseline_id,
            cumulative_savings=_round_val(cumulative, 2),
            period_savings=period_savings,
            total_periods=len(period_savings),
            trend=trend,
            performance_rating=rating,
            std_error=_round_val(std_error, 2),
            is_significant=is_significant,
        )
        tracker.provenance_hash = _compute_hash(tracker)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "CUSUM tracked: cumulative=%s kWh, significant=%s, "
            "rating=%s, hash=%s (%.1f ms)",
            str(_round_val(cumulative, 0)), str(is_significant),
            rating.value, tracker.provenance_hash[:16], elapsed,
        )
        return tracker

    def test_significance(
        self,
        savings_values: List[Decimal],
        alpha: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Test statistical significance of energy savings.

        Performs a one-sample t-test: H0: mean_savings = 0.

        Args:
            savings_values: List of per-period savings.
            alpha:          Significance level override.

        Returns:
            Dict with t-statistic, critical value, and conclusion.
        """
        t0 = time.perf_counter()
        sig_level = alpha or self._alpha

        if len(savings_values) < 2:
            return {
                "t_statistic": "0",
                "t_critical": str(T_CRITICAL.get(sig_level, Decimal("2.042"))),
                "is_significant": False,
                "conclusion": "Insufficient data for significance test.",
                "n": len(savings_values),
            }

        n = _decimal(len(savings_values))
        mean_s = _safe_divide(sum(savings_values, Decimal("0")), n)
        variance = _safe_divide(
            sum(((s - mean_s) ** 2 for s in savings_values), Decimal("0")),
            n - Decimal("1"),
        )
        std_dev = _decimal(math.sqrt(float(variance)))
        std_error = _safe_divide(std_dev, _decimal(math.sqrt(float(n))))

        t_stat = _safe_divide(abs(mean_s), std_error)
        t_crit = T_CRITICAL.get(sig_level, Decimal("2.042"))
        is_significant = t_stat > t_crit

        if is_significant and mean_s > Decimal("0"):
            conclusion = (
                f"Savings are statistically significant at alpha={sig_level}. "
                f"Mean savings = {_round_val(mean_s, 2)} kWh/period."
            )
        elif is_significant and mean_s < Decimal("0"):
            conclusion = (
                f"Performance decline is statistically significant at alpha={sig_level}. "
                f"Mean increase = {_round_val(abs(mean_s), 2)} kWh/period."
            )
        else:
            conclusion = (
                f"No statistically significant change at alpha={sig_level}. "
                f"t={_round_val(t_stat, 3)} < t_crit={t_crit}."
            )

        elapsed = (time.perf_counter() - t0) * 1000.0
        result = {
            "t_statistic": str(_round_val(t_stat, 4)),
            "t_critical": str(t_crit),
            "is_significant": is_significant,
            "mean_savings": str(_round_val(mean_s, 2)),
            "std_error": str(_round_val(std_error, 4)),
            "n": len(savings_values),
            "alpha": sig_level,
            "conclusion": conclusion,
            "processing_time_ms": round(elapsed, 2),
        }

        logger.info(
            "Significance test: t=%.3f, critical=%.3f, significant=%s (%.1f ms)",
            float(t_stat), float(t_crit), str(is_significant), elapsed,
        )
        return result

    def update_baseline(
        self,
        baseline_id: str,
        new_data: List[Dict[str, Any]],
        reason: str = "periodic_review",
    ) -> Optional[EnergyBaseline]:
        """Update an existing baseline with new data.

        Supersedes the old baseline and creates a new active one.

        Args:
            baseline_id: ID of baseline to update.
            new_data:    New baseline data.
            reason:      Reason for baseline update.

        Returns:
            New EnergyBaseline or None if original not found.
        """
        t0 = time.perf_counter()
        old_baseline = self._baselines.get(baseline_id)
        if old_baseline is None:
            logger.error("Baseline %s not found", baseline_id[:12])
            return None

        # Supersede old baseline
        old_baseline.status = BaselineStatus.SUPERSEDED

        # Find the original definition
        definition = None
        for d in self._definitions.values():
            if old_baseline.relevant_variables:
                definition = d
                break

        if definition is None:
            # Create a basic definition
            definition = EnPIDefinition(
                name="Updated",
                enpi_type=EnPIType.REGRESSION if old_baseline.regression_model else EnPIType.SIMPLE_RATIO,
                relevant_variables=[
                    RelevantVariable(v) for v in old_baseline.relevant_variables
                    if v in [rv.value for rv in RelevantVariable]
                ],
            )

        new_baseline = self.build_baseline(
            definition, new_data,
            name=f"{old_baseline.name} (updated: {reason})",
        )

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Baseline updated: %s -> %s, reason=%s (%.1f ms)",
            baseline_id[:12], new_baseline.baseline_id[:12],
            reason, elapsed,
        )
        return new_baseline

    # ------------------------------------------------------------------ #
    # Internal Methods                                                     #
    # ------------------------------------------------------------------ #

    def _fit_regression(
        self,
        data: List[Dict[str, Any]],
        variable_names: List[str],
    ) -> RegressionModel:
        """Fit an OLS regression model using the normal equation.

        Normal equation: beta = (X^T X)^{-1} X^T y
        For simplicity, implements up to 3 variables using direct algebra.

        Args:
            data:           Training data.
            variable_names: Names of predictor variables.

        Returns:
            RegressionModel with coefficients and statistics.
        """
        t0 = time.perf_counter()
        n = len(data)
        p = len(variable_names)

        if n < MIN_REGRESSION_POINTS or p == 0:
            return RegressionModel(n_observations=n, n_predictors=p)

        # Extract y and X values as floats for numeric computation
        y_vals = [float(_decimal(d.get("energy_kwh", 0))) for d in data]
        x_matrix = []
        for d in data:
            row = [float(_decimal(d.get(var, 0))) for var in variable_names]
            x_matrix.append(row)

        y_mean = sum(y_vals) / n

        # Build X with intercept column: [1, x1, x2, ...]
        x_aug = [[1.0] + row for row in x_matrix]
        cols = p + 1  # Including intercept

        # Normal equation: beta = (X^T X)^-1 X^T y
        # Compute X^T X
        xtx = [[0.0] * cols for _ in range(cols)]
        for i in range(cols):
            for j in range(cols):
                for k in range(n):
                    xtx[i][j] += x_aug[k][i] * x_aug[k][j]

        # Compute X^T y
        xty = [0.0] * cols
        for i in range(cols):
            for k in range(n):
                xty[i] += x_aug[k][i] * y_vals[k]

        # Invert X^T X using Gauss-Jordan elimination
        inv = self._invert_matrix(xtx)
        if inv is None:
            logger.warning("Singular matrix - regression failed")
            return RegressionModel(n_observations=n, n_predictors=p)

        # Compute beta = inv(X^T X) * X^T y
        beta = [0.0] * cols
        for i in range(cols):
            for j in range(cols):
                beta[i] += inv[i][j] * xty[j]

        # Predictions and residuals
        y_hat = []
        for row in x_aug:
            pred = sum(beta[j] * row[j] for j in range(cols))
            y_hat.append(pred)

        residuals = [y_vals[i] - y_hat[i] for i in range(n)]

        # Sum of squares
        ss_total = sum((y - y_mean) ** 2 for y in y_vals)
        ss_residual = sum(r ** 2 for r in residuals)
        ss_regression = ss_total - ss_residual

        # R-squared
        r_squared = 1.0 - (ss_residual / ss_total) if ss_total > 0 else 0.0
        adj_r_squared = (
            1.0 - (1.0 - r_squared) * (n - 1) / (n - p - 1)
            if (n - p - 1) > 0 else 0.0
        )

        # RMSE and CVRMSE
        mse = ss_residual / n if n > 0 else 0.0
        rmse = math.sqrt(mse)
        cvrmse = (rmse / abs(y_mean) * 100) if y_mean != 0 else 0.0

        # F-statistic
        ms_reg = ss_regression / p if p > 0 else 0.0
        ms_res = ss_residual / (n - p - 1) if (n - p - 1) > 0 else 0.0
        f_stat = ms_reg / ms_res if ms_res > 0 else 0.0

        # F-test p-value category
        f_crit_005 = float(F_CRITICAL.get("0.05", Decimal("3.32")))
        f_crit_001 = float(F_CRITICAL.get("0.01", Decimal("5.39")))
        if f_stat > float(F_CRITICAL.get("0.001", Decimal("8.77"))):
            f_p = "p<0.001"
        elif f_stat > f_crit_001:
            f_p = "p<0.01"
        elif f_stat > f_crit_005:
            f_p = "p<0.05"
        else:
            f_p = "not_significant"

        # Standard errors and t-statistics for coefficients
        std_errors: Dict[str, Decimal] = {}
        t_stats: Dict[str, Decimal] = {}
        coefficients: Dict[str, Decimal] = {}

        for j, var_name in enumerate(variable_names):
            coef_idx = j + 1  # Skip intercept
            coefficients[var_name] = _round_val(_decimal(beta[coef_idx]), 6)
            se = math.sqrt(abs(ms_res * inv[coef_idx][coef_idx])) if ms_res > 0 else 0.0
            std_errors[var_name] = _round_val(_decimal(se), 6)
            t_val = beta[coef_idx] / se if se > 0 else 0.0
            t_stats[var_name] = _round_val(_decimal(t_val), 4)

        # Model quality
        quality = RegressionQuality.POOR
        for r2_thresh, cv_thresh, q in REGRESSION_QUALITY_THRESHOLDS:
            if _decimal(r_squared) >= r2_thresh and _decimal(cvrmse) <= cv_thresh:
                quality = q
                break

        meets_ashrae = _decimal(cvrmse) <= self._cvrmse_threshold

        model = RegressionModel(
            intercept=_round_val(_decimal(beta[0]), 6),
            coefficients=coefficients,
            r_squared=_round_val(_decimal(r_squared), 6),
            adjusted_r_squared=_round_val(_decimal(adj_r_squared), 6),
            rmse=_round_val(_decimal(rmse), 4),
            cvrmse=_round_val(_decimal(cvrmse), 2),
            f_statistic=_round_val(_decimal(f_stat), 4),
            f_p_value_approx=f_p,
            t_statistics=t_stats,
            std_errors=std_errors,
            n_observations=n,
            n_predictors=p,
            ss_total=_round_val(_decimal(ss_total), 2),
            ss_regression=_round_val(_decimal(ss_regression), 2),
            ss_residual=_round_val(_decimal(ss_residual), 2),
            quality=quality,
            meets_ashrae14=meets_ashrae,
        )

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Regression fitted: R2=%.4f, adj_R2=%.4f, CVRMSE=%.2f%%, "
            "F=%.2f (%s), quality=%s, ASHRAE14=%s (%.1f ms)",
            r_squared, adj_r_squared, cvrmse, f_stat, f_p,
            quality.value, str(meets_ashrae), elapsed,
        )
        return model

    def _invert_matrix(
        self, matrix: List[List[float]],
    ) -> Optional[List[List[float]]]:
        """Invert a square matrix using Gauss-Jordan elimination.

        Args:
            matrix: Square matrix to invert.

        Returns:
            Inverted matrix or None if singular.
        """
        n = len(matrix)
        # Create augmented matrix [A | I]
        aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
               for i, row in enumerate(matrix)]

        for col in range(n):
            # Find pivot
            max_row = col
            for row in range(col + 1, n):
                if abs(aug[row][col]) > abs(aug[max_row][col]):
                    max_row = row
            aug[col], aug[max_row] = aug[max_row], aug[col]

            pivot = aug[col][col]
            if abs(pivot) < 1e-12:
                return None

            # Scale pivot row
            for j in range(2 * n):
                aug[col][j] /= pivot

            # Eliminate column
            for row in range(n):
                if row == col:
                    continue
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]

        # Extract inverse
        return [row[n:] for row in aug]

    def _predict_energy(
        self,
        baseline: EnergyBaseline,
        data_point: Dict[str, Any],
        variable_names: List[str],
    ) -> Decimal:
        """Predict energy consumption using the baseline model.

        Args:
            baseline:       Energy baseline.
            data_point:     Current period data.
            variable_names: Variable names.

        Returns:
            Predicted energy consumption (kWh).
        """
        model = baseline.regression_model
        if model is None or not model.coefficients:
            return baseline.enpi_baseline_value

        predicted = float(model.intercept)
        for var in variable_names:
            coef = float(model.coefficients.get(var, Decimal("0")))
            value = float(_decimal(data_point.get(var, 0)))
            predicted += coef * value

        return _round_val(_decimal(max(predicted, 0)), 2)

    def _rate_performance(
        self,
        cumulative_savings: Decimal,
        std_error: Decimal,
        is_significant: bool,
    ) -> PerformanceRating:
        """Rate energy performance based on CUSUM savings.

        Args:
            cumulative_savings: Total cumulative savings.
            std_error:          Standard error of savings.
            is_significant:     Whether savings are statistically significant.

        Returns:
            PerformanceRating.
        """
        if not is_significant:
            return PerformanceRating.STABLE

        if cumulative_savings > Decimal("0"):
            if std_error > Decimal("0"):
                ratio = _safe_divide(cumulative_savings, std_error)
                if ratio > Decimal("2"):
                    return PerformanceRating.SIGNIFICANTLY_IMPROVED
            return PerformanceRating.IMPROVED
        else:
            if std_error > Decimal("0"):
                ratio = _safe_divide(abs(cumulative_savings), std_error)
                if ratio > Decimal("2"):
                    return PerformanceRating.SIGNIFICANTLY_DECLINED
            return PerformanceRating.DECLINED

    def _generate_recommendations(
        self,
        definition: EnPIDefinition,
        baseline: EnergyBaseline,
        values: List[EnPIValue],
        cusum: CUSUMTracker,
        savings_pct: Decimal,
    ) -> List[str]:
        """Generate recommendations based on EnPI analysis."""
        recs: List[str] = []

        # Check regression model quality
        model = baseline.regression_model
        if model and not model.meets_ashrae14:
            recs.append(
                f"Regression model CVRMSE ({model.cvrmse}%) exceeds ASHRAE "
                f"Guideline 14 threshold ({self._cvrmse_threshold}%). "
                "Consider adding relevant variables or more data points."
            )

        if model and model.quality == RegressionQuality.POOR:
            recs.append(
                "Regression model quality is POOR (R2 < 0.50). "
                "Review measurement boundary and variable selection."
            )

        # Performance assessment
        if cusum.performance_rating == PerformanceRating.SIGNIFICANTLY_DECLINED:
            recs.append(
                "Energy performance has significantly declined. "
                "Investigate equipment degradation, operational changes, "
                "or baseline validity."
            )

        if cusum.performance_rating == PerformanceRating.DECLINED:
            recs.append(
                "Energy performance has declined. Review maintenance "
                "schedules and operational procedures."
            )

        if savings_pct < Decimal("0") and abs(savings_pct) > Decimal("5"):
            recs.append(
                f"Energy consumption is {_round_val(abs(savings_pct), 1)}% "
                "above baseline. Conduct energy audit to identify causes."
            )

        if cusum.trend == "declining":
            recs.append(
                "Savings trend is declining. Verify that energy "
                "conservation measures are still in place."
            )

        # Target check
        if definition.target_value is not None and values:
            latest = values[-1].enpi_value
            if latest > definition.target_value:
                recs.append(
                    f"Current EnPI ({_round_val(latest, 2)}) exceeds target "
                    f"({_round_val(definition.target_value, 2)}). "
                    "Intensify energy management actions."
                )

        # Baseline age check
        if baseline.data_points < MIN_REGRESSION_POINTS * 2:
            recs.append(
                "Baseline has limited data points. Collect more data "
                "to improve regression model confidence."
            )

        if not recs:
            recs.append(
                "Energy performance is on track. Continue monitoring "
                "and validating baseline assumptions per ISO 50001."
            )

        return recs
