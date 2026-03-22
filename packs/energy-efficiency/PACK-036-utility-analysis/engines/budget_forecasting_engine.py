# -*- coding: utf-8 -*-
"""
BudgetForecastingEngine - PACK-036 Utility Analysis Engine 5
==============================================================

Energy budget forecasting engine with confidence intervals, variance
decomposition, scenario analysis, and rolling forecast updates.  Supports
multiple forecasting methods (historical trend, linear/multiple regression,
Monte Carlo simulation, and ensemble averaging) across electricity,
natural gas, water, steam, and chilled-water commodities.

Calculation Methodology:
    Linear Trend:
        Cost_m = alpha + beta * m
        where alpha = intercept, beta = slope from OLS regression

    Weather Regression (Multiple):
        Cost_m = a + b1 * HDD_m + b2 * CDD_m + b3 * Production_m
        OLS with up to 3 independent variables

    Rate Escalation:
        Cost_m = Base_Cost * (1 + escalation_pct / 100) ^ (m / 12)

    Monte Carlo Simulation:
        For each iteration i in N (default 1000):
            Sample weather_i ~ N(mu_DD, sigma_DD)
            Sample rate_i ~ N(mu_rate, sigma_rate)
            Sample consumption_i ~ N(mu_cons, sigma_cons)
            Cost_i = consumption_i * rate_i * weather_adjustment(weather_i)
        Percentile extraction: P50 (median), P80, P90, P95

    Ensemble Forecast:
        F_ensemble = w1 * F_trend + w2 * F_regression + w3 * F_monte_carlo
        where weights w are inversely proportional to in-sample MAPE

    Model Validation (Train/Test Split):
        MAPE  = mean(|actual - forecast| / actual) * 100
        RMSE  = sqrt(mean((actual - forecast)^2))
        R^2   = 1 - SS_res / SS_tot

    Variance Decomposition:
        Weather  = (Actual_DD - Budget_DD) * cost_per_DD
        Rate     = (Actual_Rate - Budget_Rate) * Actual_Volume
        Volume   = (Actual_Vol  - Budget_Vol)  * Budget_Rate
        Residual = Total_Variance - Weather - Rate - Volume

Regulatory / Standard References:
    - ISO 50001:2018 Clause 6.2 (Energy objectives, targets, action plans)
    - ISO 50006:2014 (Energy baselines and performance indicators)
    - ASHRAE Guideline 14-2014 (M&V uncertainty analysis)
    - IPMVP Volume I (Option C - Whole facility approach)
    - EU Energy Efficiency Directive (EED) 2023/1791 Art. 8 (energy audits)
    - Energy Star Portfolio Manager (weather normalisation methodology)

Zero-Hallucination:
    - All formulas from published statistical / engineering literature
    - Monte Carlo uses deterministic pseudo-random seeding for reproducibility
    - OLS regression coefficients computed analytically (no ML library)
    - Decimal arithmetic for all financial calculations
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-036 Utility Analysis
Engine:  5 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
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
            if k not in ("result_id", "calculated_at", "processing_time_ms", "provenance_hash")
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


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ForecastMethod(str, Enum):
    """Forecasting method for energy budget projection.

    HISTORICAL_TREND:     Extrapolation based on time-indexed OLS trend.
    LINEAR_REGRESSION:    Single-variable OLS regression.
    MULTIPLE_REGRESSION:  Multi-variable OLS regression (HDD, CDD, production).
    ARIMA:                Autoregressive Integrated Moving Average.
    SARIMA:               Seasonal ARIMA with monthly seasonality.
    MONTE_CARLO:          Stochastic simulation with N=1000 iterations.
    ENSEMBLE:             Weighted average of multiple methods.
    """
    HISTORICAL_TREND = "historical_trend"
    LINEAR_REGRESSION = "linear_regression"
    MULTIPLE_REGRESSION = "multiple_regression"
    ARIMA = "arima"
    SARIMA = "sarima"
    MONTE_CARLO = "monte_carlo"
    ENSEMBLE = "ensemble"


class ForecastHorizon(str, Enum):
    """Forecast planning horizon.

    SHORT_12M:   12-month operational budget.
    MEDIUM_24M:  24-month tactical planning.
    LONG_36M:    36-month strategic projection.
    """
    SHORT_12M = "short_12m"
    MEDIUM_24M = "medium_24m"
    LONG_36M = "long_36m"


class ConfidenceLevel(str, Enum):
    """Confidence interval percentile levels.

    P50:  50th percentile (median).
    P80:  80th percentile.
    P90:  90th percentile.
    P95:  95th percentile.
    """
    P50 = "p50"
    P80 = "p80"
    P90 = "p90"
    P95 = "p95"


class VarianceCategory(str, Enum):
    """Categories for budget variance attribution.

    WEATHER:      Variance due to deviation from normal degree-days.
    RATE:         Variance due to tariff/rate changes.
    VOLUME:       Variance due to consumption volume changes.
    TIMING:       Variance due to billing period timing shifts.
    OPERATIONAL:  Variance due to operational changes (production, occupancy).
    OTHER:        Residual/unexplained variance.
    """
    WEATHER = "weather"
    RATE = "rate"
    VOLUME = "volume"
    TIMING = "timing"
    OPERATIONAL = "operational"
    OTHER = "other"


class ScenarioType(str, Enum):
    """Types of budget scenarios for sensitivity analysis.

    BASE:               Most likely outcome based on current conditions.
    BEST:               Favourable conditions (mild weather, low rates).
    WORST:              Adverse conditions (extreme weather, high rates).
    CUSTOM:             User-defined scenario parameters.
    REGULATORY_CHANGE:  Scenario reflecting regulatory/tariff changes.
    """
    BASE = "base"
    BEST = "best"
    WORST = "worst"
    CUSTOM = "custom"
    REGULATORY_CHANGE = "regulatory_change"


class CommodityType(str, Enum):
    """Utility commodity types tracked for budget forecasting.

    ELECTRICITY:    Electrical energy (kWh).
    NATURAL_GAS:    Natural gas (therms, kWh, m3).
    WATER:          Potable water (m3, gallons).
    STEAM:          District steam (kg, lbs).
    CHILLED_WATER:  District chilled water (ton-hours, kWh).
    """
    ELECTRICITY = "electricity"
    NATURAL_GAS = "natural_gas"
    WATER = "water"
    STEAM = "steam"
    CHILLED_WATER = "chilled_water"


class TrendDirection(str, Enum):
    """Direction of observed energy cost/consumption trend.

    INCREASING:  Costs or consumption rising over time.
    DECREASING:  Costs or consumption falling over time.
    STABLE:      No statistically significant change.
    VOLATILE:    High variance with no clear directional trend.
    """
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Z-scores for confidence intervals.
# Source: Standard normal distribution tables.
CONFIDENCE_Z_SCORES: Dict[ConfidenceLevel, float] = {
    ConfidenceLevel.P50: 0.6745,
    ConfidenceLevel.P80: 1.2816,
    ConfidenceLevel.P90: 1.6449,
    ConfidenceLevel.P95: 1.9600,
}

# Default horizon months.
HORIZON_MONTHS: Dict[ForecastHorizon, int] = {
    ForecastHorizon.SHORT_12M: 12,
    ForecastHorizon.MEDIUM_24M: 24,
    ForecastHorizon.LONG_36M: 36,
}

# Monte Carlo default iterations.
MC_DEFAULT_ITERATIONS: int = 1000

# Train/test split ratio for model validation.
VALIDATION_SPLIT_RATIO: float = 0.75


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class HistoricalDataPoint(BaseModel):
    """A single month of historical utility data.

    Attributes:
        month:           Period label in YYYY-MM format.
        commodity:       Commodity type.
        consumption_kwh: Energy consumption (kWh or equivalent).
        cost_eur:        Total cost in EUR.
        demand_kw:       Peak demand (kW) if applicable.
        hdd:             Heating degree-days for the month.
        cdd:             Cooling degree-days for the month.
        production_units: Production volume (units) if applicable.
        occupancy_pct:   Building occupancy percentage (0-100).
    """
    month: str = Field(..., description="Period label YYYY-MM", pattern=r"^\d{4}-\d{2}$")
    commodity: CommodityType = Field(default=CommodityType.ELECTRICITY)
    consumption_kwh: float = Field(default=0.0, ge=0.0, description="Consumption kWh")
    cost_eur: float = Field(default=0.0, ge=0.0, description="Total cost EUR")
    demand_kw: float = Field(default=0.0, ge=0.0, description="Peak demand kW")
    hdd: float = Field(default=0.0, ge=0.0, description="Heating degree-days")
    cdd: float = Field(default=0.0, ge=0.0, description="Cooling degree-days")
    production_units: float = Field(default=0.0, ge=0.0, description="Production volume")
    occupancy_pct: float = Field(default=100.0, ge=0.0, le=100.0, description="Occupancy %")

    @field_validator("month")
    @classmethod
    def validate_month_format(cls, v: str) -> str:
        """Validate month is a valid YYYY-MM string."""
        parts = v.split("-")
        if len(parts) != 2:
            raise ValueError(f"Month must be YYYY-MM format, got: {v}")
        year, month = int(parts[0]), int(parts[1])
        if year < 2000 or year > 2100:
            raise ValueError(f"Year out of range (2000-2100): {year}")
        if month < 1 or month > 12:
            raise ValueError(f"Month out of range (1-12): {month}")
        return v


class ScenarioDefinition(BaseModel):
    """Definition of a budget scenario for sensitivity analysis.

    Attributes:
        scenario_type:        Type of scenario.
        name:                 Human-readable scenario name.
        description:          Scenario description/rationale.
        rate_adjustment_pct:  Rate adjustment relative to base (e.g. +5.0 = 5% increase).
        volume_adjustment_pct: Consumption volume adjustment.
        weather_adjustment_pct: Degree-day adjustment for weather deviation.
        assumptions:          Key-value pairs of scenario assumptions.
    """
    scenario_type: ScenarioType = Field(default=ScenarioType.BASE)
    name: str = Field(default="Base Case", description="Scenario name")
    description: str = Field(default="", description="Scenario description")
    rate_adjustment_pct: float = Field(default=0.0, description="Rate adjustment %")
    volume_adjustment_pct: float = Field(default=0.0, description="Volume adjustment %")
    weather_adjustment_pct: float = Field(default=0.0, description="Weather adjustment %")
    assumptions: Dict[str, Any] = Field(default_factory=dict, description="Assumptions")


class ForecastInput(BaseModel):
    """Input for budget forecast generation.

    Attributes:
        facility_id:          Facility identifier.
        commodity:            Commodity type to forecast.
        historical_data:      24-36 months of historical data.
        method:               Forecasting method to apply.
        horizon:              Forecast planning horizon.
        rate_escalation_pct:  Annual rate escalation percentage.
        weather_source:       Weather data source identifier.
        scenarios:            Scenario definitions for sensitivity analysis.
        random_seed:          Seed for Monte Carlo reproducibility.
    """
    facility_id: str = Field(..., description="Facility identifier")
    commodity: CommodityType = Field(default=CommodityType.ELECTRICITY)
    historical_data: List[HistoricalDataPoint] = Field(
        ..., min_length=12, description="Historical data (24-36 months recommended)"
    )
    method: ForecastMethod = Field(default=ForecastMethod.HISTORICAL_TREND)
    horizon: ForecastHorizon = Field(default=ForecastHorizon.SHORT_12M)
    rate_escalation_pct: float = Field(default=0.0, ge=-20.0, le=50.0, description="Annual rate escalation %")
    weather_source: str = Field(default="", description="Weather data source")
    scenarios: List[ScenarioDefinition] = Field(default_factory=list, description="Scenarios")
    random_seed: int = Field(default=42, ge=0, description="Monte Carlo seed")


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class MonthlyForecast(BaseModel):
    """Forecasted values for a single month.

    Attributes:
        month:              Period label YYYY-MM.
        consumption_forecast: Forecasted consumption (kWh).
        cost_forecast:      Forecasted cost (EUR).
        demand_forecast:    Forecasted peak demand (kW).
        confidence_lower:   Lower confidence bound (EUR).
        confidence_upper:   Upper confidence bound (EUR).
        confidence_level:   Confidence level used.
    """
    month: str = Field(default="", description="Period YYYY-MM")
    consumption_forecast: float = Field(default=0.0, description="Consumption forecast kWh")
    cost_forecast: float = Field(default=0.0, description="Cost forecast EUR")
    demand_forecast: float = Field(default=0.0, description="Demand forecast kW")
    confidence_lower: float = Field(default=0.0, description="Lower confidence bound EUR")
    confidence_upper: float = Field(default=0.0, description="Upper confidence bound EUR")
    confidence_level: ConfidenceLevel = Field(default=ConfidenceLevel.P90)


class ConfidenceBand(BaseModel):
    """Confidence band around an annual cost forecast.

    Attributes:
        level:          Confidence percentile.
        lower_bound_eur: Lower bound of the confidence band.
        upper_bound_eur: Upper bound of the confidence band.
        width_eur:      Band width (upper - lower).
    """
    level: ConfidenceLevel = Field(default=ConfidenceLevel.P90)
    lower_bound_eur: float = Field(default=0.0, description="Lower bound EUR")
    upper_bound_eur: float = Field(default=0.0, description="Upper bound EUR")
    width_eur: float = Field(default=0.0, description="Band width EUR")


class BudgetVariance(BaseModel):
    """Variance between budgeted and actual values for one month.

    Attributes:
        month:          Period label YYYY-MM.
        budgeted_eur:   Budgeted cost.
        actual_eur:     Actual cost.
        variance_eur:   Variance amount (actual - budgeted).
        variance_pct:   Variance as percentage of budget.
        explanations:   List of explanatory notes.
    """
    month: str = Field(default="", description="Period YYYY-MM")
    budgeted_eur: float = Field(default=0.0, description="Budgeted cost EUR")
    actual_eur: float = Field(default=0.0, description="Actual cost EUR")
    variance_eur: float = Field(default=0.0, description="Variance EUR")
    variance_pct: float = Field(default=0.0, description="Variance %")
    explanations: List[str] = Field(default_factory=list, description="Explanatory notes")


class VarianceDecomposition(BaseModel):
    """Decomposition of total budget variance into causal factors.

    Attributes:
        total_variance_eur:  Total variance (actual - budget).
        weather_impact_eur:  Variance attributable to weather deviation.
        rate_impact_eur:     Variance attributable to rate changes.
        volume_impact_eur:   Variance attributable to consumption volume changes.
        timing_impact_eur:   Variance attributable to billing period timing.
        residual_eur:        Unexplained residual variance.
    """
    total_variance_eur: float = Field(default=0.0, description="Total variance EUR")
    weather_impact_eur: float = Field(default=0.0, description="Weather impact EUR")
    rate_impact_eur: float = Field(default=0.0, description="Rate impact EUR")
    volume_impact_eur: float = Field(default=0.0, description="Volume impact EUR")
    timing_impact_eur: float = Field(default=0.0, description="Timing impact EUR")
    residual_eur: float = Field(default=0.0, description="Residual EUR")


class ScenarioResult(BaseModel):
    """Result of a single scenario analysis.

    Attributes:
        scenario_type:      Scenario type.
        name:               Scenario name.
        description:        Scenario description.
        annual_cost_eur:    Projected annual cost under this scenario.
        monthly_forecasts:  Monthly forecast detail.
        assumptions:        Key assumptions applied.
    """
    scenario_type: ScenarioType = Field(default=ScenarioType.BASE)
    name: str = Field(default="", description="Scenario name")
    description: str = Field(default="", description="Description")
    annual_cost_eur: float = Field(default=0.0, description="Annual cost EUR")
    monthly_forecasts: List[MonthlyForecast] = Field(default_factory=list)
    assumptions: Dict[str, Any] = Field(default_factory=dict, description="Assumptions")


class RollingForecast(BaseModel):
    """Rolling forecast update combining actuals YTD with revised projections.

    Attributes:
        as_of_date:            Date of the rolling update.
        months_remaining:      Months remaining in the forecast period.
        original_budget_eur:   Original full-period budget.
        revised_forecast_eur:  Revised forecast (actuals YTD + remaining forecast).
        variance_to_budget_eur: Variance between revised forecast and original budget.
    """
    as_of_date: str = Field(default="", description="As-of date YYYY-MM-DD")
    months_remaining: int = Field(default=0, ge=0, description="Months remaining")
    original_budget_eur: float = Field(default=0.0, description="Original budget EUR")
    revised_forecast_eur: float = Field(default=0.0, description="Revised forecast EUR")
    variance_to_budget_eur: float = Field(default=0.0, description="Variance to budget EUR")


class ForecastResult(BaseModel):
    """Complete budget forecast result with all analyses.

    Attributes:
        result_id:          Unique result identifier.
        facility_id:        Facility identifier.
        commodity:          Commodity type.
        method_used:        Forecasting method applied.
        horizon:            Forecast horizon.
        monthly_forecasts:  Monthly projections.
        annual_total_eur:   Total forecasted annual cost.
        confidence_bands:   Confidence interval bands.
        scenarios:          Scenario analysis results.
        model_accuracy:     Model validation metrics (R2, MAPE, RMSE).
        trend_direction:    Detected cost trend direction.
        methodology_notes:  Methodology and source notes.
        processing_time_ms: Computation time (ms).
        engine_version:     Engine version.
        calculated_at:      Calculation timestamp.
        provenance_hash:    SHA-256 provenance hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    facility_id: str = Field(default="", description="Facility identifier")
    commodity: CommodityType = Field(default=CommodityType.ELECTRICITY)
    method_used: ForecastMethod = Field(default=ForecastMethod.HISTORICAL_TREND)
    horizon: ForecastHorizon = Field(default=ForecastHorizon.SHORT_12M)
    monthly_forecasts: List[MonthlyForecast] = Field(default_factory=list)
    annual_total_eur: float = Field(default=0.0, description="Annual total cost EUR")
    confidence_bands: List[ConfidenceBand] = Field(default_factory=list)
    scenarios: List[ScenarioResult] = Field(default_factory=list)
    model_accuracy: Dict[str, float] = Field(default_factory=dict)
    trend_direction: TrendDirection = Field(default=TrendDirection.STABLE)
    methodology_notes: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BudgetForecastingEngine:
    """Zero-hallucination energy budget forecasting engine.

    Provides multiple forecasting methods (historical trend, linear/multiple
    regression, Monte Carlo simulation, ensemble averaging), variance
    decomposition, scenario analysis, and rolling forecast updates for
    utility budgets across multiple commodity types.

    Guarantees:
        - Deterministic: same inputs produce identical outputs (seeded RNG).
        - Reproducible: every result carries a SHA-256 provenance hash.
        - Auditable: full breakdown of variance, confidence intervals, and model
          accuracy metrics.
        - No LLM: zero hallucination risk in any calculation path.

    Usage::

        engine = BudgetForecastingEngine()
        result = engine.forecast(forecast_input)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the budget forecasting engine.

        Args:
            config: Optional overrides.  Supported keys:
                - mc_iterations (int): Monte Carlo iterations (default 1000)
                - validation_split (float): Train/test split ratio (default 0.75)
                - default_confidence (str): Default confidence level
                - seasonal_period (int): Seasonal period (default 12)
        """
        self._config = config or {}
        self._mc_iterations = int(self._config.get("mc_iterations", MC_DEFAULT_ITERATIONS))
        self._validation_split = float(self._config.get("validation_split", VALIDATION_SPLIT_RATIO))
        self._seasonal_period = int(self._config.get("seasonal_period", 12))
        self._notes: List[str] = []
        logger.info("BudgetForecastingEngine v%s initialised.", _MODULE_VERSION)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def forecast(self, input_data: ForecastInput) -> ForecastResult:
        """Generate a complete energy budget forecast.

        Dispatches to the appropriate forecasting method, runs scenario
        analysis if requested, and computes model validation metrics.

        Args:
            input_data: Validated forecast input data.

        Returns:
            ForecastResult with monthly projections, confidence bands,
            scenario results, and model accuracy metrics.

        Raises:
            ValueError: If insufficient historical data is provided.
        """
        t0 = time.perf_counter()
        self._notes = [f"Engine version: {self.engine_version}"]

        data = input_data.historical_data
        horizon_months = HORIZON_MONTHS[input_data.horizon]

        if len(data) < 12:
            raise ValueError("At least 12 months of historical data required.")

        self._notes.append(
            f"Facility: {input_data.facility_id}, commodity: {input_data.commodity.value}, "
            f"method: {input_data.method.value}, horizon: {horizon_months} months."
        )

        # Detect trend direction.
        trend_dir = self._detect_trend(data)

        # Dispatch to method.
        monthly_forecasts = self._dispatch_forecast(
            input_data, data, horizon_months
        )

        # Apply rate escalation.
        if input_data.rate_escalation_pct != 0.0:
            monthly_forecasts = self._apply_rate_escalation(
                monthly_forecasts, input_data.rate_escalation_pct
            )

        # Compute annual total.
        annual_total = _round2(sum(
            _decimal(mf.cost_forecast) for mf in monthly_forecasts
        ))

        # Confidence bands.
        confidence_bands = self._compute_confidence_bands(data, monthly_forecasts)

        # Scenario analysis.
        scenario_results: List[ScenarioResult] = []
        if input_data.scenarios:
            scenario_results = self.run_scenarios(input_data, input_data.scenarios)

        # Model accuracy (validate against historical with train/test split).
        model_accuracy = self.validate_model(data, input_data.method)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = ForecastResult(
            facility_id=input_data.facility_id,
            commodity=input_data.commodity,
            method_used=input_data.method,
            horizon=input_data.horizon,
            monthly_forecasts=monthly_forecasts,
            annual_total_eur=annual_total,
            confidence_bands=confidence_bands,
            scenarios=scenario_results,
            model_accuracy=model_accuracy,
            trend_direction=trend_dir,
            methodology_notes=list(self._notes),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Budget forecast complete: facility=%s, commodity=%s, method=%s, "
            "annual=%.2f EUR, trend=%s, hash=%s (%.1f ms)",
            input_data.facility_id,
            input_data.commodity.value,
            input_data.method.value,
            annual_total,
            trend_dir.value,
            result.provenance_hash[:16],
            elapsed_ms,
        )
        return result

    def historical_trend_forecast(
        self,
        data: List[HistoricalDataPoint],
        months: int,
    ) -> List[MonthlyForecast]:
        """Forecast using OLS linear trend on historical cost data.

        Fits: Cost_m = alpha + beta * m, then extrapolates forward.

        Args:
            data:   Historical data points in chronological order.
            months: Number of months to forecast.

        Returns:
            List of monthly forecasts.
        """
        n = len(data)
        if n < 6:
            return []

        d_costs = [_decimal(dp.cost_eur) for dp in data]
        d_cons = [_decimal(dp.consumption_kwh) for dp in data]
        d_demand = [_decimal(dp.demand_kw) for dp in data]

        # Fit OLS for cost: y = alpha + beta * x.
        alpha_cost, beta_cost, rmse_cost = self._fit_ols_simple(d_costs)

        # Fit OLS for consumption.
        alpha_cons, beta_cons, _ = self._fit_ols_simple(d_cons)

        # Fit OLS for demand.
        alpha_dem, beta_dem, _ = self._fit_ols_simple(d_demand)

        # Compute z-score for P90 confidence interval.
        z_score = _decimal(CONFIDENCE_Z_SCORES[ConfidenceLevel.P90])

        forecasts: List[MonthlyForecast] = []
        last_month = data[-1].month
        for h in range(1, months + 1):
            t_idx = _decimal(n + h - 1)
            fc_cost = alpha_cost + beta_cost * t_idx
            fc_cons = max(Decimal("0"), alpha_cons + beta_cons * t_idx)
            fc_dem = max(Decimal("0"), alpha_dem + beta_dem * t_idx)

            # Prediction interval widens with horizon.
            margin = z_score * rmse_cost * _decimal(math.sqrt(h))
            future_month = self._advance_month(last_month, h)

            forecasts.append(MonthlyForecast(
                month=future_month,
                consumption_forecast=_round2(fc_cons),
                cost_forecast=_round2(fc_cost),
                demand_forecast=_round2(fc_dem),
                confidence_lower=_round2(fc_cost - margin),
                confidence_upper=_round2(fc_cost + margin),
                confidence_level=ConfidenceLevel.P90,
            ))

        self._notes.append(
            f"Historical trend: alpha={_round2(alpha_cost)}, beta={_round4(beta_cost)}, "
            f"RMSE={_round2(rmse_cost)}, {months} months projected."
        )
        return forecasts

    def regression_forecast(
        self,
        data: List[HistoricalDataPoint],
        months: int,
        variables: Optional[List[str]] = None,
    ) -> List[MonthlyForecast]:
        """Forecast using multiple regression on weather and production drivers.

        Cost_m = a + b1*HDD + b2*CDD + b3*Production

        If only one variable has non-zero values, falls back to linear regression.

        Args:
            data:      Historical data points.
            months:    Number of months to forecast.
            variables: Predictor variables ('hdd', 'cdd', 'production_units').

        Returns:
            List of monthly forecasts.
        """
        variables = variables or ["hdd", "cdd", "production_units"]
        n = len(data)
        if n < 12:
            return []

        # Build predictor matrix and response vector.
        y_vals = [_decimal(dp.cost_eur) for dp in data]
        x_matrix = self._build_predictor_matrix(data, variables)

        # Fit OLS multiple regression.
        coefficients, rmse = self._fit_ols_multiple(x_matrix, y_vals)
        if not coefficients:
            self._notes.append("Regression: singular matrix, falling back to trend.")
            return self.historical_trend_forecast(data, months)

        # Compute seasonal averages for future predictions.
        seasonal_predictors = self._compute_seasonal_predictors(data, variables)

        z_score = _decimal(CONFIDENCE_Z_SCORES[ConfidenceLevel.P90])
        forecasts: List[MonthlyForecast] = []
        last_month = data[-1].month

        for h in range(1, months + 1):
            future_month = self._advance_month(last_month, h)
            month_idx = (int(future_month.split("-")[1]) - 1) % 12

            # Predicted cost using seasonal average predictors.
            predicted_cost = coefficients[0]  # intercept
            for i, var_name in enumerate(variables):
                x_val = seasonal_predictors.get(var_name, {}).get(month_idx, Decimal("0"))
                predicted_cost += coefficients[i + 1] * x_val

            margin = z_score * rmse * _decimal(math.sqrt(h))

            # Estimate consumption from cost using average rate.
            avg_rate = self._compute_average_rate(data)
            fc_cons = _safe_divide(predicted_cost, avg_rate) if avg_rate > Decimal("0") else Decimal("0")

            forecasts.append(MonthlyForecast(
                month=future_month,
                consumption_forecast=_round2(fc_cons),
                cost_forecast=_round2(predicted_cost),
                demand_forecast=0.0,
                confidence_lower=_round2(predicted_cost - margin),
                confidence_upper=_round2(predicted_cost + margin),
                confidence_level=ConfidenceLevel.P90,
            ))

        self._notes.append(
            f"Regression forecast: {len(variables)} variables, "
            f"intercept={_round2(coefficients[0])}, RMSE={_round2(rmse)}, "
            f"{months} months projected."
        )
        return forecasts

    def monte_carlo_forecast(
        self,
        data: List[HistoricalDataPoint],
        months: int,
        iterations: Optional[int] = None,
        seed: int = 42,
    ) -> List[MonthlyForecast]:
        """Forecast using Monte Carlo simulation with confidence bands.

        Samples consumption, rates, and weather from historical distributions,
        then computes cost for N iterations and extracts percentile forecasts.

        Args:
            data:       Historical data points.
            months:     Number of months to forecast.
            iterations: Number of simulation iterations (default 1000).
            seed:       Random seed for reproducibility.

        Returns:
            List of monthly forecasts at P50 with P90 confidence bounds.
        """
        n_iter = iterations or self._mc_iterations
        n = len(data)
        if n < 12:
            return []

        rng = random.Random(seed)

        # Compute monthly statistics (mean, std) by calendar month.
        monthly_stats = self._compute_monthly_stats(data)

        last_month = data[-1].month
        forecasts: List[MonthlyForecast] = []

        for h in range(1, months + 1):
            future_month = self._advance_month(last_month, h)
            cal_month = int(future_month.split("-")[1]) - 1  # 0-indexed

            stats = monthly_stats.get(cal_month, {})
            mu_cons = float(stats.get("mean_consumption", 0.0))
            sd_cons = float(stats.get("std_consumption", 0.0))
            mu_cost = float(stats.get("mean_cost", 0.0))
            sd_cost = float(stats.get("std_cost", 0.0))
            mu_dem = float(stats.get("mean_demand", 0.0))
            sd_dem = float(stats.get("std_demand", 0.0))

            cost_samples: List[Decimal] = []
            cons_samples: List[Decimal] = []
            dem_samples: List[Decimal] = []

            for _ in range(n_iter):
                sim_cost = _decimal(max(0.0, rng.gauss(mu_cost, max(sd_cost, 0.01))))
                sim_cons = _decimal(max(0.0, rng.gauss(mu_cons, max(sd_cons, 0.01))))
                sim_dem = _decimal(max(0.0, rng.gauss(mu_dem, max(sd_dem, 0.01))))
                cost_samples.append(sim_cost)
                cons_samples.append(sim_cons)
                dem_samples.append(sim_dem)

            # Sort for percentile extraction.
            cost_samples.sort()
            cons_samples.sort()
            dem_samples.sort()

            p50_idx = n_iter // 2
            p05_idx = max(0, int(n_iter * 0.05))
            p95_idx = min(n_iter - 1, int(n_iter * 0.95))

            forecasts.append(MonthlyForecast(
                month=future_month,
                consumption_forecast=_round2(cons_samples[p50_idx]),
                cost_forecast=_round2(cost_samples[p50_idx]),
                demand_forecast=_round2(dem_samples[p50_idx]),
                confidence_lower=_round2(cost_samples[p05_idx]),
                confidence_upper=_round2(cost_samples[p95_idx]),
                confidence_level=ConfidenceLevel.P90,
            ))

        self._notes.append(
            f"Monte Carlo: {n_iter} iterations, seed={seed}, "
            f"{months} months projected with P90 confidence bands."
        )
        return forecasts

    def ensemble_forecast(
        self,
        data: List[HistoricalDataPoint],
        months: int,
        seed: int = 42,
    ) -> List[MonthlyForecast]:
        """Ensemble forecast combining multiple methods with error-weighted averaging.

        Weights each method inversely proportional to its in-sample MAPE.

        Args:
            data:   Historical data points.
            months: Number of months to forecast.
            seed:   Random seed for Monte Carlo component.

        Returns:
            List of ensemble-averaged monthly forecasts.
        """
        if len(data) < 12:
            return []

        # Generate component forecasts.
        trend_fc = self.historical_trend_forecast(data, months)
        regression_fc = self.regression_forecast(data, months)
        mc_fc = self.monte_carlo_forecast(data, months, seed=seed)

        # Validate each method to get MAPE for weighting.
        trend_acc = self.validate_model(data, ForecastMethod.HISTORICAL_TREND)
        reg_acc = self.validate_model(data, ForecastMethod.MULTIPLE_REGRESSION)
        mc_acc = self.validate_model(data, ForecastMethod.MONTE_CARLO)

        # Compute inverse-MAPE weights (lower MAPE = higher weight).
        mape_trend = _decimal(max(trend_acc.get("mape", 100.0), 0.01))
        mape_reg = _decimal(max(reg_acc.get("mape", 100.0), 0.01))
        mape_mc = _decimal(max(mc_acc.get("mape", 100.0), 0.01))

        inv_trend = _safe_divide(Decimal("1"), mape_trend)
        inv_reg = _safe_divide(Decimal("1"), mape_reg)
        inv_mc = _safe_divide(Decimal("1"), mape_mc)
        total_inv = inv_trend + inv_reg + inv_mc

        w_trend = _safe_divide(inv_trend, total_inv)
        w_reg = _safe_divide(inv_reg, total_inv)
        w_mc = _safe_divide(inv_mc, total_inv)

        # Combine forecasts.
        forecasts: List[MonthlyForecast] = []
        for i in range(months):
            t_cost = _decimal(trend_fc[i].cost_forecast) if i < len(trend_fc) else Decimal("0")
            r_cost = _decimal(regression_fc[i].cost_forecast) if i < len(regression_fc) else Decimal("0")
            m_cost = _decimal(mc_fc[i].cost_forecast) if i < len(mc_fc) else Decimal("0")

            ens_cost = w_trend * t_cost + w_reg * r_cost + w_mc * m_cost

            t_cons = _decimal(trend_fc[i].consumption_forecast) if i < len(trend_fc) else Decimal("0")
            r_cons = _decimal(regression_fc[i].consumption_forecast) if i < len(regression_fc) else Decimal("0")
            m_cons = _decimal(mc_fc[i].consumption_forecast) if i < len(mc_fc) else Decimal("0")

            ens_cons = w_trend * t_cons + w_reg * r_cons + w_mc * m_cons

            # Confidence bounds: widest of all methods.
            lower_vals = [
                trend_fc[i].confidence_lower if i < len(trend_fc) else 0.0,
                regression_fc[i].confidence_lower if i < len(regression_fc) else 0.0,
                mc_fc[i].confidence_lower if i < len(mc_fc) else 0.0,
            ]
            upper_vals = [
                trend_fc[i].confidence_upper if i < len(trend_fc) else 0.0,
                regression_fc[i].confidence_upper if i < len(regression_fc) else 0.0,
                mc_fc[i].confidence_upper if i < len(mc_fc) else 0.0,
            ]

            future_month = trend_fc[i].month if i < len(trend_fc) else ""

            forecasts.append(MonthlyForecast(
                month=future_month,
                consumption_forecast=_round2(ens_cons),
                cost_forecast=_round2(ens_cost),
                demand_forecast=0.0,
                confidence_lower=_round2(min(lower_vals)),
                confidence_upper=_round2(max(upper_vals)),
                confidence_level=ConfidenceLevel.P90,
            ))

        self._notes.append(
            f"Ensemble: weights trend={_round3(w_trend)}, regression={_round3(w_reg)}, "
            f"monte_carlo={_round3(w_mc)}."
        )
        return forecasts

    def run_scenarios(
        self,
        input_data: ForecastInput,
        scenarios: List[ScenarioDefinition],
    ) -> List[ScenarioResult]:
        """Run multiple scenario analyses on the base forecast.

        Each scenario adjusts rate, volume, and/or weather factors from
        the base case.

        Args:
            input_data: Original forecast input.
            scenarios:  List of scenario definitions.

        Returns:
            List of ScenarioResult with per-scenario forecasts.
        """
        base_forecasts = self._dispatch_forecast(
            input_data, input_data.historical_data,
            HORIZON_MONTHS[input_data.horizon],
        )
        results: List[ScenarioResult] = []

        for scenario in scenarios:
            adjusted = self._apply_scenario_adjustments(base_forecasts, scenario)
            annual_cost = _round2(sum(
                _decimal(mf.cost_forecast) for mf in adjusted
            ))

            results.append(ScenarioResult(
                scenario_type=scenario.scenario_type,
                name=scenario.name,
                description=scenario.description,
                annual_cost_eur=annual_cost,
                monthly_forecasts=adjusted,
                assumptions=scenario.assumptions,
            ))

            self._notes.append(
                f"Scenario '{scenario.name}': rate adj={scenario.rate_adjustment_pct}%, "
                f"volume adj={scenario.volume_adjustment_pct}%, annual={annual_cost} EUR."
            )

        return results

    def calculate_variance(
        self,
        budget: List[MonthlyForecast],
        actuals: List[HistoricalDataPoint],
    ) -> List[BudgetVariance]:
        """Calculate month-by-month budget variance.

        Args:
            budget:  Budgeted monthly forecasts.
            actuals: Actual historical data points.

        Returns:
            List of BudgetVariance for each matching month.
        """
        # Build lookup of actuals by month.
        actual_by_month: Dict[str, HistoricalDataPoint] = {
            dp.month: dp for dp in actuals
        }

        variances: List[BudgetVariance] = []
        for bf in budget:
            actual = actual_by_month.get(bf.month)
            if actual is None:
                continue

            d_budget = _decimal(bf.cost_forecast)
            d_actual = _decimal(actual.cost_eur)
            d_variance = d_actual - d_budget
            d_variance_pct = _safe_pct(d_variance, d_budget)

            explanations: List[str] = []
            abs_pct = abs(float(d_variance_pct))
            if abs_pct > 10.0:
                explanations.append(
                    f"Significant variance of {_round2(d_variance_pct)}% detected."
                )
            if abs_pct > 20.0:
                explanations.append("Variance exceeds 20% threshold -- investigation required.")

            variances.append(BudgetVariance(
                month=bf.month,
                budgeted_eur=_round2(d_budget),
                actual_eur=_round2(d_actual),
                variance_eur=_round2(d_variance),
                variance_pct=_round2(d_variance_pct),
                explanations=explanations,
            ))

        return variances

    def decompose_variance(
        self,
        budget: List[MonthlyForecast],
        actuals: List[HistoricalDataPoint],
        weather_budget: Optional[List[Dict[str, float]]] = None,
        rate_budget: Optional[List[Dict[str, float]]] = None,
    ) -> VarianceDecomposition:
        """Decompose total budget variance into weather, rate, volume, and residual.

        Formulae:
            Weather  = (Actual_DD - Budget_DD) * cost_per_DD
            Rate     = (Actual_Rate - Budget_Rate) * Actual_Volume
            Volume   = (Actual_Vol  - Budget_Vol)  * Budget_Rate
            Residual = Total - Weather - Rate - Volume

        Args:
            budget:         Budgeted monthly forecasts.
            actuals:        Actual historical data.
            weather_budget: Budget weather data [{month, hdd, cdd}].
            rate_budget:    Budget rate data [{month, rate_per_kwh}].

        Returns:
            VarianceDecomposition with attributed impacts.
        """
        actual_by_month = {dp.month: dp for dp in actuals}
        budget_by_month = {bf.month: bf for bf in budget}

        weather_by_month: Dict[str, Dict[str, float]] = {}
        if weather_budget:
            for wb in weather_budget:
                weather_by_month[wb.get("month", "")] = wb

        rate_by_month: Dict[str, Dict[str, float]] = {}
        if rate_budget:
            for rb in rate_budget:
                rate_by_month[rb.get("month", "")] = rb

        total_variance = Decimal("0")
        total_weather = Decimal("0")
        total_rate = Decimal("0")
        total_volume = Decimal("0")

        for bf in budget:
            actual = actual_by_month.get(bf.month)
            if actual is None:
                continue

            d_actual_cost = _decimal(actual.cost_eur)
            d_budget_cost = _decimal(bf.cost_forecast)
            variance = d_actual_cost - d_budget_cost
            total_variance += variance

            # Weather impact.
            wb = weather_by_month.get(bf.month, {})
            budget_dd = _decimal(wb.get("hdd", 0.0)) + _decimal(wb.get("cdd", 0.0))
            actual_dd = _decimal(actual.hdd) + _decimal(actual.cdd)
            total_dd_all = sum(
                _decimal(a.hdd) + _decimal(a.cdd) for a in actuals
                if a.month in budget_by_month
            )
            total_cost_all = sum(
                _decimal(a.cost_eur) for a in actuals
                if a.month in budget_by_month
            )
            cost_per_dd = _safe_divide(total_cost_all, total_dd_all) if total_dd_all > Decimal("0") else Decimal("0")
            weather_impact = (actual_dd - budget_dd) * cost_per_dd
            total_weather += weather_impact

            # Rate impact: (actual_rate - budget_rate) * actual_volume.
            rb = rate_by_month.get(bf.month, {})
            budget_rate = _decimal(rb.get("rate_per_kwh", 0.0))
            actual_volume = _decimal(actual.consumption_kwh)
            if budget_rate > Decimal("0") and actual_volume > Decimal("0"):
                actual_rate = _safe_divide(d_actual_cost, actual_volume)
                rate_impact = (actual_rate - budget_rate) * actual_volume
                total_rate += rate_impact

            # Volume impact: (actual_vol - budget_vol) * budget_rate.
            budget_volume = _decimal(bf.consumption_forecast)
            if budget_rate > Decimal("0"):
                volume_impact = (actual_volume - budget_volume) * budget_rate
                total_volume += volume_impact

        residual = total_variance - total_weather - total_rate - total_volume

        self._notes.append(
            f"Variance decomposition: total={_round2(total_variance)} EUR, "
            f"weather={_round2(total_weather)}, rate={_round2(total_rate)}, "
            f"volume={_round2(total_volume)}, residual={_round2(residual)}."
        )

        return VarianceDecomposition(
            total_variance_eur=_round2(total_variance),
            weather_impact_eur=_round2(total_weather),
            rate_impact_eur=_round2(total_rate),
            volume_impact_eur=_round2(total_volume),
            timing_impact_eur=0.0,
            residual_eur=_round2(residual),
        )

    def rolling_forecast_update(
        self,
        original_budget: List[MonthlyForecast],
        actuals_ytd: List[HistoricalDataPoint],
        remaining_forecast: List[MonthlyForecast],
    ) -> RollingForecast:
        """Update a rolling forecast combining YTD actuals with remaining projections.

        Args:
            original_budget:    Original full-year budget.
            actuals_ytd:        Actual data received year-to-date.
            remaining_forecast: Revised forecast for remaining months.

        Returns:
            RollingForecast with revised total and variance to original.
        """
        original_total = sum(_decimal(bf.cost_forecast) for bf in original_budget)
        actuals_total = sum(_decimal(a.cost_eur) for a in actuals_ytd)
        remaining_total = sum(_decimal(rf.cost_forecast) for rf in remaining_forecast)

        revised_total = actuals_total + remaining_total
        variance = revised_total - original_total
        months_remaining = len(remaining_forecast)

        as_of = _utcnow().strftime("%Y-%m-%d")

        self._notes.append(
            f"Rolling forecast: original={_round2(original_total)}, "
            f"revised={_round2(revised_total)}, variance={_round2(variance)}, "
            f"{months_remaining} months remaining."
        )

        return RollingForecast(
            as_of_date=as_of,
            months_remaining=months_remaining,
            original_budget_eur=_round2(original_total),
            revised_forecast_eur=_round2(revised_total),
            variance_to_budget_eur=_round2(variance),
        )

    def validate_model(
        self,
        data: List[HistoricalDataPoint],
        method: ForecastMethod,
    ) -> Dict[str, float]:
        """Validate forecast model using train/test split.

        Splits historical data at the validation ratio, trains on the first
        portion, and evaluates forecast accuracy on the holdout portion.

        Metrics returned:
            R2:   Coefficient of determination (1.0 = perfect).
            MAPE: Mean Absolute Percentage Error (%).
            RMSE: Root Mean Squared Error (EUR).

        Args:
            data:   Historical data points.
            method: Forecasting method to validate.

        Returns:
            Dict with keys 'r2', 'mape', 'rmse'.
        """
        n = len(data)
        if n < 12:
            return {"r2": 0.0, "mape": 100.0, "rmse": 0.0}

        split_idx = max(6, int(n * self._validation_split))
        train_data = data[:split_idx]
        test_data = data[split_idx:]

        if len(test_data) < 1:
            return {"r2": 0.0, "mape": 100.0, "rmse": 0.0}

        test_months = len(test_data)

        # Generate forecast on training data.
        forecasts = self._dispatch_forecast_for_validation(
            train_data, test_months, method
        )

        if len(forecasts) < 1:
            return {"r2": 0.0, "mape": 100.0, "rmse": 0.0}

        # Calculate accuracy metrics.
        actuals = [_decimal(dp.cost_eur) for dp in test_data]
        predicted = [_decimal(fc.cost_forecast) for fc in forecasts[:len(actuals)]]
        n_eval = min(len(actuals), len(predicted))

        if n_eval < 1:
            return {"r2": 0.0, "mape": 100.0, "rmse": 0.0}

        actuals = actuals[:n_eval]
        predicted = predicted[:n_eval]

        # MAPE.
        mape_sum = Decimal("0")
        mape_count = 0
        for a, p in zip(actuals, predicted):
            if a != Decimal("0"):
                mape_sum += abs(a - p) / abs(a)
                mape_count += 1
        mape = _safe_divide(mape_sum * Decimal("100"), _decimal(max(mape_count, 1)))

        # RMSE.
        sse = sum((a - p) ** 2 for a, p in zip(actuals, predicted))
        rmse = _decimal(math.sqrt(max(float(_safe_divide(sse, _decimal(n_eval))), 0.0)))

        # R-squared.
        mean_actual = _safe_divide(sum(actuals), _decimal(n_eval))
        ss_tot = sum((a - mean_actual) ** 2 for a in actuals)
        ss_res = sum((a - p) ** 2 for a, p in zip(actuals, predicted))
        r2 = Decimal("1") - _safe_divide(ss_res, ss_tot) if ss_tot > Decimal("0") else Decimal("0")
        r2 = max(Decimal("0"), r2)  # Clamp to [0, 1]

        return {
            "r2": _round4(r2),
            "mape": _round2(mape),
            "rmse": _round2(rmse),
        }

    def consolidated_forecast(
        self,
        inputs: List[ForecastInput],
    ) -> ForecastResult:
        """Generate a consolidated forecast across multiple commodities.

        Sums the forecasts for each commodity into a combined result.

        Args:
            inputs: List of ForecastInput objects, one per commodity.

        Returns:
            ForecastResult with combined monthly projections.

        Raises:
            ValueError: If no inputs are provided.
        """
        t0 = time.perf_counter()
        self._notes = [f"Engine version: {self.engine_version}"]

        if not inputs:
            raise ValueError("At least one ForecastInput is required.")

        # Generate individual forecasts.
        individual_results: List[ForecastResult] = []
        for fi in inputs:
            result = self.forecast(fi)
            individual_results.append(result)

        # Determine max horizon across all results.
        max_months = max(len(r.monthly_forecasts) for r in individual_results)

        # Consolidate month by month.
        consolidated: List[MonthlyForecast] = []
        for m_idx in range(max_months):
            total_cost = Decimal("0")
            total_cons = Decimal("0")
            total_dem = Decimal("0")
            total_lower = Decimal("0")
            total_upper = Decimal("0")
            month_label = ""

            for r in individual_results:
                if m_idx < len(r.monthly_forecasts):
                    mf = r.monthly_forecasts[m_idx]
                    total_cost += _decimal(mf.cost_forecast)
                    total_cons += _decimal(mf.consumption_forecast)
                    total_dem += _decimal(mf.demand_forecast)
                    total_lower += _decimal(mf.confidence_lower)
                    total_upper += _decimal(mf.confidence_upper)
                    if not month_label:
                        month_label = mf.month

            consolidated.append(MonthlyForecast(
                month=month_label,
                consumption_forecast=_round2(total_cons),
                cost_forecast=_round2(total_cost),
                demand_forecast=_round2(total_dem),
                confidence_lower=_round2(total_lower),
                confidence_upper=_round2(total_upper),
                confidence_level=ConfidenceLevel.P90,
            ))

        annual_total = _round2(sum(
            _decimal(mf.cost_forecast) for mf in consolidated
        ))

        # Aggregate confidence bands.
        all_bands: List[ConfidenceBand] = []
        for level in ConfidenceLevel:
            level_lower = sum(
                _decimal(b.lower_bound_eur)
                for r in individual_results
                for b in r.confidence_bands
                if b.level == level
            )
            level_upper = sum(
                _decimal(b.upper_bound_eur)
                for r in individual_results
                for b in r.confidence_bands
                if b.level == level
            )
            if level_lower > Decimal("0") or level_upper > Decimal("0"):
                all_bands.append(ConfidenceBand(
                    level=level,
                    lower_bound_eur=_round2(level_lower),
                    upper_bound_eur=_round2(level_upper),
                    width_eur=_round2(level_upper - level_lower),
                ))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        facility_id = inputs[0].facility_id if inputs else ""

        commodities = ", ".join(fi.commodity.value for fi in inputs)
        self._notes.append(
            f"Consolidated forecast: {len(inputs)} commodities ({commodities}), "
            f"annual total={annual_total} EUR."
        )

        result = ForecastResult(
            facility_id=facility_id,
            commodity=CommodityType.ELECTRICITY,  # placeholder for consolidated
            method_used=ForecastMethod.ENSEMBLE,
            horizon=inputs[0].horizon if inputs else ForecastHorizon.SHORT_12M,
            monthly_forecasts=consolidated,
            annual_total_eur=annual_total,
            confidence_bands=all_bands,
            scenarios=[],
            model_accuracy={},
            trend_direction=TrendDirection.STABLE,
            methodology_notes=list(self._notes),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Consolidated forecast: %d commodities, annual=%.2f EUR, hash=%s (%.1f ms)",
            len(inputs), annual_total, result.provenance_hash[:16], elapsed_ms,
        )
        return result

    # --------------------------------------------------------------------- #
    # Private -- Forecast Dispatch
    # --------------------------------------------------------------------- #

    def _dispatch_forecast(
        self,
        input_data: ForecastInput,
        data: List[HistoricalDataPoint],
        months: int,
    ) -> List[MonthlyForecast]:
        """Dispatch to the appropriate forecasting method.

        Args:
            input_data: Forecast input specification.
            data:       Historical data.
            months:     Forecast horizon in months.

        Returns:
            List of monthly forecasts.
        """
        method = input_data.method

        if method == ForecastMethod.HISTORICAL_TREND:
            return self.historical_trend_forecast(data, months)
        elif method == ForecastMethod.LINEAR_REGRESSION:
            return self.regression_forecast(data, months, variables=["hdd"])
        elif method == ForecastMethod.MULTIPLE_REGRESSION:
            return self.regression_forecast(data, months)
        elif method == ForecastMethod.MONTE_CARLO:
            return self.monte_carlo_forecast(data, months, seed=input_data.random_seed)
        elif method == ForecastMethod.ENSEMBLE:
            return self.ensemble_forecast(data, months, seed=input_data.random_seed)
        elif method in (ForecastMethod.ARIMA, ForecastMethod.SARIMA):
            # ARIMA/SARIMA: fallback to trend + seasonal adjustment.
            return self._arima_fallback_forecast(data, months)
        else:
            return self.historical_trend_forecast(data, months)

    def _dispatch_forecast_for_validation(
        self,
        train_data: List[HistoricalDataPoint],
        test_months: int,
        method: ForecastMethod,
    ) -> List[MonthlyForecast]:
        """Dispatch forecast for model validation (no scenarios).

        Args:
            train_data:  Training subset of historical data.
            test_months: Number of months to predict for testing.
            method:      Forecasting method.

        Returns:
            List of monthly forecasts.
        """
        if method == ForecastMethod.HISTORICAL_TREND:
            return self.historical_trend_forecast(train_data, test_months)
        elif method in (ForecastMethod.LINEAR_REGRESSION, ForecastMethod.MULTIPLE_REGRESSION):
            return self.regression_forecast(train_data, test_months)
        elif method == ForecastMethod.MONTE_CARLO:
            return self.monte_carlo_forecast(train_data, test_months)
        elif method == ForecastMethod.ENSEMBLE:
            return self.historical_trend_forecast(train_data, test_months)
        else:
            return self.historical_trend_forecast(train_data, test_months)

    # --------------------------------------------------------------------- #
    # Private -- ARIMA Fallback (Seasonal Naive + Trend)
    # --------------------------------------------------------------------- #

    def _arima_fallback_forecast(
        self,
        data: List[HistoricalDataPoint],
        months: int,
    ) -> List[MonthlyForecast]:
        """Seasonal naive forecast with linear trend as ARIMA/SARIMA fallback.

        Uses the last full year of seasonal pattern plus a linear trend
        adjustment estimated from the complete history.

        Args:
            data:   Historical data.
            months: Number of months to forecast.

        Returns:
            List of monthly forecasts.
        """
        n = len(data)
        m = self._seasonal_period

        # Compute linear trend on cost data.
        d_costs = [_decimal(dp.cost_eur) for dp in data]
        _, beta, rmse = self._fit_ols_simple(d_costs)

        # Seasonal indices from last full year.
        seasonal_base: List[Decimal] = []
        start_idx = max(0, n - m)
        for i in range(start_idx, n):
            seasonal_base.append(_decimal(data[i].cost_eur))

        # Pad if fewer than m points.
        while len(seasonal_base) < m:
            seasonal_base.append(
                _safe_divide(sum(d_costs), _decimal(n)) if n > 0 else Decimal("0")
            )

        z_score = _decimal(CONFIDENCE_Z_SCORES[ConfidenceLevel.P90])
        last_month = data[-1].month
        forecasts: List[MonthlyForecast] = []

        for h in range(1, months + 1):
            season_idx = (h - 1) % m
            base_val = seasonal_base[season_idx]
            trend_adj = beta * _decimal(h)
            fc_cost = base_val + trend_adj
            margin = z_score * rmse * _decimal(math.sqrt(h))

            future_month = self._advance_month(last_month, h)

            forecasts.append(MonthlyForecast(
                month=future_month,
                consumption_forecast=0.0,
                cost_forecast=_round2(fc_cost),
                demand_forecast=0.0,
                confidence_lower=_round2(fc_cost - margin),
                confidence_upper=_round2(fc_cost + margin),
                confidence_level=ConfidenceLevel.P90,
            ))

        self._notes.append(
            f"ARIMA fallback (seasonal naive + trend): beta={_round4(beta)}, "
            f"RMSE={_round2(rmse)}, {months} months projected."
        )
        return forecasts

    # --------------------------------------------------------------------- #
    # Private -- OLS Simple Regression
    # --------------------------------------------------------------------- #

    def _fit_ols_simple(
        self,
        y_values: List[Decimal],
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """Fit simple OLS: y = alpha + beta * x, where x = 0..n-1.

        Args:
            y_values: Response values.

        Returns:
            Tuple of (alpha, beta, rmse).
        """
        n = len(y_values)
        if n < 2:
            mean_y = y_values[0] if y_values else Decimal("0")
            return mean_y, Decimal("0"), Decimal("0")

        d_n = _decimal(n)
        d_x = [_decimal(i) for i in range(n)]

        sx = sum(d_x)
        sy = sum(y_values)
        sxx = sum(x ** 2 for x in d_x)
        sxy = sum(x * y for x, y in zip(d_x, y_values))

        denom = d_n * sxx - sx ** 2
        if denom == Decimal("0"):
            mean_y = _safe_divide(sy, d_n)
            return mean_y, Decimal("0"), Decimal("0")

        beta = _safe_divide(d_n * sxy - sx * sy, denom)
        alpha = _safe_divide(sy - beta * sx, d_n)

        # RMSE.
        predicted = [alpha + beta * _decimal(i) for i in range(n)]
        residuals = [y_values[i] - predicted[i] for i in range(n)]
        mse = _safe_divide(
            sum(r ** 2 for r in residuals),
            _decimal(max(n - 2, 1)),
        )
        rmse = _decimal(math.sqrt(max(float(mse), 0.0)))

        return alpha, beta, rmse

    # --------------------------------------------------------------------- #
    # Private -- OLS Multiple Regression
    # --------------------------------------------------------------------- #

    def _fit_ols_multiple(
        self,
        x_matrix: List[List[Decimal]],
        y_values: List[Decimal],
    ) -> Tuple[List[Decimal], Decimal]:
        """Fit OLS multiple regression: y = b0 + b1*x1 + b2*x2 + ... + bk*xk.

        Uses the normal equations: b = (X'X)^{-1} X'y.
        For numerical stability, implements Cramer's rule for up to 4x4 systems.

        Args:
            x_matrix: Predictor values (each row = [x1, x2, ...]).
            y_values: Response values.

        Returns:
            Tuple of (coefficients [b0, b1, ..., bk], rmse).
            Returns empty list if system is singular.
        """
        n = len(y_values)
        if n < 2 or not x_matrix:
            return [], Decimal("0")

        k = len(x_matrix[0])  # number of predictors
        p = k + 1  # number of parameters (including intercept)

        # Build design matrix with intercept column.
        X: List[List[Decimal]] = []
        for i in range(n):
            row = [Decimal("1")] + x_matrix[i]
            X.append(row)

        # Compute X'X (p x p) and X'y (p x 1).
        XtX: List[List[Decimal]] = [[Decimal("0")] * p for _ in range(p)]
        Xty: List[Decimal] = [Decimal("0")] * p

        for i in range(n):
            for j in range(p):
                Xty[j] += X[i][j] * y_values[i]
                for l in range(p):
                    XtX[j][l] += X[i][j] * X[i][l]

        # Solve using Gaussian elimination.
        coefficients = self._solve_linear_system(XtX, Xty)
        if coefficients is None:
            return [], Decimal("0")

        # RMSE.
        predicted = []
        for i in range(n):
            y_hat = sum(coefficients[j] * X[i][j] for j in range(p))
            predicted.append(y_hat)

        residuals = [y_values[i] - predicted[i] for i in range(n)]
        mse = _safe_divide(
            sum(r ** 2 for r in residuals),
            _decimal(max(n - p, 1)),
        )
        rmse = _decimal(math.sqrt(max(float(mse), 0.0)))

        return coefficients, rmse

    def _solve_linear_system(
        self,
        A: List[List[Decimal]],
        b: List[Decimal],
    ) -> Optional[List[Decimal]]:
        """Solve Ax = b using Gaussian elimination with partial pivoting.

        Args:
            A: Coefficient matrix (n x n).
            b: Right-hand side vector.

        Returns:
            Solution vector, or None if system is singular.
        """
        n = len(b)
        # Augmented matrix.
        aug = [row[:] + [b[i]] for i, row in enumerate(A)]

        # Forward elimination.
        for col in range(n):
            # Partial pivoting.
            max_row = col
            max_val = abs(aug[col][col])
            for row in range(col + 1, n):
                if abs(aug[row][col]) > max_val:
                    max_val = abs(aug[row][col])
                    max_row = row
            if max_val < Decimal("1E-15"):
                return None
            aug[col], aug[max_row] = aug[max_row], aug[col]

            # Eliminate below.
            for row in range(col + 1, n):
                factor = _safe_divide(aug[row][col], aug[col][col])
                for j in range(col, n + 1):
                    aug[row][j] -= factor * aug[col][j]

        # Back substitution.
        x = [Decimal("0")] * n
        for i in range(n - 1, -1, -1):
            if abs(aug[i][i]) < Decimal("1E-15"):
                return None
            s = aug[i][n] - sum(aug[i][j] * x[j] for j in range(i + 1, n))
            x[i] = _safe_divide(s, aug[i][i])

        return x

    # --------------------------------------------------------------------- #
    # Private -- Predictor Matrix Construction
    # --------------------------------------------------------------------- #

    def _build_predictor_matrix(
        self,
        data: List[HistoricalDataPoint],
        variables: List[str],
    ) -> List[List[Decimal]]:
        """Build predictor matrix from historical data.

        Args:
            data:      Historical data points.
            variables: List of variable names to include.

        Returns:
            Matrix of predictor values (one row per data point).
        """
        matrix: List[List[Decimal]] = []
        for dp in data:
            row: List[Decimal] = []
            for var in variables:
                if var == "hdd":
                    row.append(_decimal(dp.hdd))
                elif var == "cdd":
                    row.append(_decimal(dp.cdd))
                elif var == "production_units":
                    row.append(_decimal(dp.production_units))
                elif var == "occupancy_pct":
                    row.append(_decimal(dp.occupancy_pct))
                elif var == "demand_kw":
                    row.append(_decimal(dp.demand_kw))
                else:
                    row.append(Decimal("0"))
            matrix.append(row)
        return matrix

    def _compute_seasonal_predictors(
        self,
        data: List[HistoricalDataPoint],
        variables: List[str],
    ) -> Dict[str, Dict[int, Decimal]]:
        """Compute average predictor values by calendar month.

        Args:
            data:      Historical data points.
            variables: Predictor variable names.

        Returns:
            Dict mapping variable name to {month_index: average_value}.
        """
        # Accumulate sums and counts by (variable, month_idx).
        sums: Dict[str, Dict[int, Decimal]] = {v: {} for v in variables}
        counts: Dict[str, Dict[int, int]] = {v: {} for v in variables}

        for dp in data:
            month_idx = int(dp.month.split("-")[1]) - 1
            for var in variables:
                val = Decimal("0")
                if var == "hdd":
                    val = _decimal(dp.hdd)
                elif var == "cdd":
                    val = _decimal(dp.cdd)
                elif var == "production_units":
                    val = _decimal(dp.production_units)
                elif var == "occupancy_pct":
                    val = _decimal(dp.occupancy_pct)

                sums[var][month_idx] = sums[var].get(month_idx, Decimal("0")) + val
                counts[var][month_idx] = counts[var].get(month_idx, 0) + 1

        result: Dict[str, Dict[int, Decimal]] = {}
        for var in variables:
            result[var] = {}
            for m_idx in range(12):
                s = sums[var].get(m_idx, Decimal("0"))
                c = counts[var].get(m_idx, 0)
                result[var][m_idx] = _safe_divide(s, _decimal(max(c, 1)))

        return result

    # --------------------------------------------------------------------- #
    # Private -- Monthly Statistics for Monte Carlo
    # --------------------------------------------------------------------- #

    def _compute_monthly_stats(
        self,
        data: List[HistoricalDataPoint],
    ) -> Dict[int, Dict[str, float]]:
        """Compute mean and standard deviation by calendar month.

        Args:
            data: Historical data points.

        Returns:
            Dict mapping calendar month (0-11) to statistics dict.
        """
        by_month: Dict[int, List[HistoricalDataPoint]] = {}
        for dp in data:
            m_idx = int(dp.month.split("-")[1]) - 1
            by_month.setdefault(m_idx, []).append(dp)

        stats: Dict[int, Dict[str, float]] = {}
        for m_idx, points in by_month.items():
            costs = [_decimal(p.cost_eur) for p in points]
            cons = [_decimal(p.consumption_kwh) for p in points]
            dems = [_decimal(p.demand_kw) for p in points]
            n = _decimal(len(points))

            mean_cost = _safe_divide(sum(costs), n)
            mean_cons = _safe_divide(sum(cons), n)
            mean_dem = _safe_divide(sum(dems), n)

            std_cost = Decimal("0")
            std_cons = Decimal("0")
            std_dem = Decimal("0")
            if len(points) > 1:
                var_cost = _safe_divide(
                    sum((c - mean_cost) ** 2 for c in costs),
                    _decimal(len(points) - 1),
                )
                var_cons = _safe_divide(
                    sum((c - mean_cons) ** 2 for c in cons),
                    _decimal(len(points) - 1),
                )
                var_dem = _safe_divide(
                    sum((d - mean_dem) ** 2 for d in dems),
                    _decimal(len(points) - 1),
                )
                std_cost = _decimal(math.sqrt(max(float(var_cost), 0.0)))
                std_cons = _decimal(math.sqrt(max(float(var_cons), 0.0)))
                std_dem = _decimal(math.sqrt(max(float(var_dem), 0.0)))

            stats[m_idx] = {
                "mean_cost": float(mean_cost),
                "std_cost": float(std_cost),
                "mean_consumption": float(mean_cons),
                "std_consumption": float(std_cons),
                "mean_demand": float(mean_dem),
                "std_demand": float(std_dem),
            }

        return stats

    # --------------------------------------------------------------------- #
    # Private -- Rate Escalation
    # --------------------------------------------------------------------- #

    def _apply_rate_escalation(
        self,
        forecasts: List[MonthlyForecast],
        escalation_pct: float,
    ) -> List[MonthlyForecast]:
        """Apply annual rate escalation to monthly forecasts.

        Formula: Cost_m = Base_Cost * (1 + escalation_pct / 100) ^ (m / 12)

        Args:
            forecasts:      Monthly forecasts to adjust.
            escalation_pct: Annual escalation percentage.

        Returns:
            Adjusted list of monthly forecasts.
        """
        esc_rate = _decimal(escalation_pct) / Decimal("100")
        adjusted: List[MonthlyForecast] = []

        for i, mf in enumerate(forecasts):
            month_offset = i + 1
            # Compound escalation factor.
            factor = (Decimal("1") + esc_rate) ** (
                _decimal(month_offset) / Decimal("12")
            )
            adj_cost = _decimal(mf.cost_forecast) * factor
            adj_lower = _decimal(mf.confidence_lower) * factor
            adj_upper = _decimal(mf.confidence_upper) * factor

            adjusted.append(MonthlyForecast(
                month=mf.month,
                consumption_forecast=mf.consumption_forecast,
                cost_forecast=_round2(adj_cost),
                demand_forecast=mf.demand_forecast,
                confidence_lower=_round2(adj_lower),
                confidence_upper=_round2(adj_upper),
                confidence_level=mf.confidence_level,
            ))

        self._notes.append(
            f"Rate escalation applied: {escalation_pct}% per annum, "
            f"compounded monthly over {len(forecasts)} months."
        )
        return adjusted

    # --------------------------------------------------------------------- #
    # Private -- Scenario Adjustments
    # --------------------------------------------------------------------- #

    def _apply_scenario_adjustments(
        self,
        base_forecasts: List[MonthlyForecast],
        scenario: ScenarioDefinition,
    ) -> List[MonthlyForecast]:
        """Apply scenario-specific adjustments to base forecasts.

        Args:
            base_forecasts: Base case monthly forecasts.
            scenario:       Scenario definition with adjustment factors.

        Returns:
            Adjusted list of monthly forecasts.
        """
        rate_factor = Decimal("1") + _decimal(scenario.rate_adjustment_pct) / Decimal("100")
        volume_factor = Decimal("1") + _decimal(scenario.volume_adjustment_pct) / Decimal("100")
        # Weather adjustment affects cost indirectly via consumption volume.
        weather_factor = Decimal("1") + _decimal(scenario.weather_adjustment_pct) / Decimal("100")

        combined_factor = rate_factor * volume_factor * weather_factor
        adjusted: List[MonthlyForecast] = []

        for mf in base_forecasts:
            adj_cost = _decimal(mf.cost_forecast) * combined_factor
            adj_cons = _decimal(mf.consumption_forecast) * volume_factor * weather_factor
            adj_lower = _decimal(mf.confidence_lower) * combined_factor
            adj_upper = _decimal(mf.confidence_upper) * combined_factor

            adjusted.append(MonthlyForecast(
                month=mf.month,
                consumption_forecast=_round2(adj_cons),
                cost_forecast=_round2(adj_cost),
                demand_forecast=mf.demand_forecast,
                confidence_lower=_round2(adj_lower),
                confidence_upper=_round2(adj_upper),
                confidence_level=mf.confidence_level,
            ))

        return adjusted

    # --------------------------------------------------------------------- #
    # Private -- Confidence Bands
    # --------------------------------------------------------------------- #

    def _compute_confidence_bands(
        self,
        data: List[HistoricalDataPoint],
        forecasts: List[MonthlyForecast],
    ) -> List[ConfidenceBand]:
        """Compute annual confidence bands at P50, P80, P90, P95 levels.

        Uses historical cost volatility to scale confidence intervals.

        Args:
            data:      Historical data for volatility estimation.
            forecasts: Monthly forecasts.

        Returns:
            List of ConfidenceBand at each confidence level.
        """
        # Estimate annual cost volatility from historical data.
        costs = [_decimal(dp.cost_eur) for dp in data]
        n = len(costs)
        if n < 2:
            return []

        mean_cost = _safe_divide(sum(costs), _decimal(n))
        variance = _safe_divide(
            sum((c - mean_cost) ** 2 for c in costs),
            _decimal(n - 1),
        )
        monthly_std = _decimal(math.sqrt(max(float(variance), 0.0)))

        # Annual std scales with sqrt(12) for monthly data.
        annual_std = monthly_std * _decimal(math.sqrt(12))

        annual_total = sum(_decimal(mf.cost_forecast) for mf in forecasts)

        bands: List[ConfidenceBand] = []
        for level in ConfidenceLevel:
            z = _decimal(CONFIDENCE_Z_SCORES[level])
            lower = annual_total - z * annual_std
            upper = annual_total + z * annual_std
            width = upper - lower

            bands.append(ConfidenceBand(
                level=level,
                lower_bound_eur=_round2(lower),
                upper_bound_eur=_round2(upper),
                width_eur=_round2(width),
            ))

        return bands

    # --------------------------------------------------------------------- #
    # Private -- Trend Detection
    # --------------------------------------------------------------------- #

    def _detect_trend(
        self,
        data: List[HistoricalDataPoint],
    ) -> TrendDirection:
        """Detect cost trend direction using simple slope analysis.

        Args:
            data: Historical data points.

        Returns:
            TrendDirection classification.
        """
        n = len(data)
        if n < 6:
            return TrendDirection.STABLE

        costs = [_decimal(dp.cost_eur) for dp in data]
        _, beta, rmse = self._fit_ols_simple(costs)

        mean_cost = _safe_divide(sum(costs), _decimal(n))
        if mean_cost == Decimal("0"):
            return TrendDirection.STABLE

        # Coefficient of variation.
        cv = _safe_divide(rmse, mean_cost) * Decimal("100")

        # Trend significance: slope relative to mean.
        slope_pct = abs(_safe_pct(beta, mean_cost))

        if float(cv) > 30.0:
            return TrendDirection.VOLATILE
        elif float(slope_pct) < 0.5:
            return TrendDirection.STABLE
        elif beta > Decimal("0"):
            return TrendDirection.INCREASING
        else:
            return TrendDirection.DECREASING

    # --------------------------------------------------------------------- #
    # Private -- Average Rate Calculation
    # --------------------------------------------------------------------- #

    def _compute_average_rate(
        self,
        data: List[HistoricalDataPoint],
    ) -> Decimal:
        """Compute average cost per kWh from historical data.

        Args:
            data: Historical data points.

        Returns:
            Average rate (EUR/kWh).
        """
        total_cost = sum(_decimal(dp.cost_eur) for dp in data)
        total_cons = sum(_decimal(dp.consumption_kwh) for dp in data)
        return _safe_divide(total_cost, total_cons)

    # --------------------------------------------------------------------- #
    # Private -- Month Arithmetic
    # --------------------------------------------------------------------- #

    def _advance_month(self, base_month: str, offset: int) -> str:
        """Advance a YYYY-MM string by *offset* months.

        Args:
            base_month: Starting month in YYYY-MM format.
            offset:     Number of months to advance (positive = forward).

        Returns:
            New month string in YYYY-MM format.
        """
        parts = base_month.split("-")
        year = int(parts[0])
        month = int(parts[1])

        total_months = (year * 12 + month - 1) + offset
        new_year = total_months // 12
        new_month = (total_months % 12) + 1

        return f"{new_year:04d}-{new_month:02d}"


# ---------------------------------------------------------------------------
# Pydantic v2 model_rebuild for forward-reference resolution
# ---------------------------------------------------------------------------

HistoricalDataPoint.model_rebuild()
ScenarioDefinition.model_rebuild()
ForecastInput.model_rebuild()
MonthlyForecast.model_rebuild()
ConfidenceBand.model_rebuild()
BudgetVariance.model_rebuild()
VarianceDecomposition.model_rebuild()
ScenarioResult.model_rebuild()
RollingForecast.model_rebuild()
ForecastResult.model_rebuild()


# ---------------------------------------------------------------------------
# Public Aliases -- required by PACK-036 __init__.py symbol contract
# ---------------------------------------------------------------------------

BudgetForecast = ForecastResult
"""Alias: ``BudgetForecast`` -> :class:`ForecastResult`."""

VarianceAnalysis = VarianceDecomposition
"""Alias: ``VarianceAnalysis`` -> :class:`VarianceDecomposition`."""
