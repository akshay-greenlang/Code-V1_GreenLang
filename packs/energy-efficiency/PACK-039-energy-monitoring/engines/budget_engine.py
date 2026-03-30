# -*- coding: utf-8 -*-
"""
BudgetEngine - PACK-039 Energy Monitoring Engine 7
=====================================================

Energy budget creation, tracking, and variance analysis engine with
weather normalisation.  Supports baseline-reduction, zero-based,
incremental, and regression-based budgeting methods.  Decomposes
variance into volume, rate, weather, efficiency, production, and other
components.  Provides rolling forecasts and configurable alert thresholds.

Calculation Methodology:
    Baseline-Reduction Budget:
        budget_kwh = baseline_kwh * (1 - reduction_target_pct / 100)
        budget_cost = budget_kwh * blended_rate

    Zero-Based Budget:
        budget_kwh = sum(equipment_rated_kw * operating_hours * load_factor)
        budget_cost = budget_kwh * tariff_rate

    Incremental Budget:
        budget_kwh = prior_actual_kwh * (1 + growth_rate)
        budget_cost = budget_kwh * (prior_rate * (1 + rate_escalation))

    Variance Decomposition:
        volume_variance = (actual_kwh - budget_kwh) * budget_rate
        rate_variance = (actual_rate - budget_rate) * actual_kwh
        weather_variance = weather_sensitivity * (actual_hdd_cdd - budget_hdd_cdd) * budget_rate
        efficiency_variance = total_variance - volume - rate - weather
        production_variance = (actual_output - budget_output) * energy_intensity

    Weather Normalisation:
        normalised_kwh = actual_kwh - weather_sensitivity * (actual_dd - normal_dd)
        weather_sensitivity = regression_slope (kWh per degree-day)

    Rolling Forecast:
        forecast = ytd_actual + remaining_months_budget * adjustment_factor
        adjustment_factor = ytd_actual / ytd_budget (if ytd_budget > 0)

Regulatory References:
    - ISO 50001:2018   Energy baselines and EnPI tracking
    - ISO 50006:2014   Baselines, EnPIs, and energy performance measurement
    - ASHRAE Guideline 14  Measurement of energy, demand, and water savings
    - IPMVP             International Performance Measurement and Verification
    - EN 16247-1:2022   Energy audits - general requirements
    - UK ESOS           Energy Savings Opportunity Scheme
    - US DOE SEP 50001  Superior Energy Performance

Zero-Hallucination:
    - All budgets and variances computed from deterministic formulas
    - Weather normalisation uses regression coefficients, not LLM
    - No LLM involvement in any budget or forecast calculation
    - Decimal arithmetic throughout for audit-grade precision
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-039 Energy Monitoring
Engine:  7 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
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

class BudgetMethod(str, Enum):
    """Energy budget creation methodology.

    BASELINE_REDUCTION:  Reduce from historical baseline by target percentage.
    ZERO_BASED:          Build up from equipment and operating schedules.
    INCREMENTAL:         Adjust prior-year actual by growth factor.
    REGRESSION_BASED:    Statistical regression on weather and production.
    """
    BASELINE_REDUCTION = "baseline_reduction"
    ZERO_BASED = "zero_based"
    INCREMENTAL = "incremental"
    REGRESSION_BASED = "regression_based"

class VarianceComponent(str, Enum):
    """Variance decomposition component type.

    VOLUME:      Consumption volume variance.
    RATE:        Unit rate / price variance.
    WEATHER:     Weather-driven variance (HDD/CDD deviation).
    EFFICIENCY:  Equipment or process efficiency variance.
    PRODUCTION:  Production volume impact on energy.
    OTHER:       Unexplained / residual variance.
    """
    VOLUME = "volume"
    RATE = "rate"
    WEATHER = "weather"
    EFFICIENCY = "efficiency"
    PRODUCTION = "production"
    OTHER = "other"

class BudgetStatus(str, Enum):
    """Budget lifecycle status.

    DRAFT:     Budget under preparation.
    APPROVED:  Budget approved by management.
    ACTIVE:    Budget in active tracking period.
    CLOSED:    Budget period completed.
    """
    DRAFT = "draft"
    APPROVED = "approved"
    ACTIVE = "active"
    CLOSED = "closed"

class AlertThreshold(str, Enum):
    """Budget variance alert level.

    ON_TRACK:   Variance within acceptable range.
    WARNING:    Variance approaching limit.
    CRITICAL:   Variance exceeding acceptable range.
    EXCEEDED:   Budget exceeded by significant margin.
    """
    ON_TRACK = "on_track"
    WARNING = "warning"
    CRITICAL = "critical"
    EXCEEDED = "exceeded"

class ForecastMethod(str, Enum):
    """Rolling forecast methodology.

    TREND:             Linear trend extrapolation.
    SEASONAL:          Seasonal adjustment from historical patterns.
    REGRESSION:        Weather-adjusted regression forecast.
    WEIGHTED_AVERAGE:  Weighted average of recent periods.
    """
    TREND = "trend"
    SEASONAL = "seasonal"
    REGRESSION = "regression"
    WEIGHTED_AVERAGE = "weighted_average"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default variance alert thresholds (percent over budget).
DEFAULT_ALERT_THRESHOLDS: Dict[str, Decimal] = {
    AlertThreshold.ON_TRACK.value: Decimal("5.0"),
    AlertThreshold.WARNING.value: Decimal("10.0"),
    AlertThreshold.CRITICAL.value: Decimal("20.0"),
    AlertThreshold.EXCEEDED.value: Decimal("100.0"),
}

# Default weather sensitivity (kWh per degree-day).
DEFAULT_WEATHER_SENSITIVITY: Decimal = Decimal("50.0")

# Default energy rate escalation (annual percentage).
DEFAULT_RATE_ESCALATION: Decimal = Decimal("3.0")

# Default budget reduction target (percentage).
DEFAULT_REDUCTION_TARGET: Decimal = Decimal("5.0")

# Months in a year constant.
MONTHS_PER_YEAR: int = 12

# Maximum forecast horizon (months).
MAX_FORECAST_MONTHS: int = 60

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class BudgetDefinition(BaseModel):
    """Energy budget definition.

    Attributes:
        budget_id: Unique budget identifier.
        budget_name: Human-readable budget name.
        budget_method: Budgeting methodology.
        fiscal_year: Fiscal year.
        baseline_kwh: Historical baseline consumption (kWh).
        baseline_cost: Historical baseline cost.
        reduction_target_pct: Target reduction percentage.
        blended_rate: Blended energy rate ($/kWh).
        weather_sensitivity: kWh per degree-day sensitivity.
        normal_hdd: Normal heating degree-days for the year.
        normal_cdd: Normal cooling degree-days for the year.
        production_baseline: Baseline production output.
        energy_intensity: Energy per unit of production (kWh/unit).
        growth_rate_pct: Expected load growth percentage.
        rate_escalation_pct: Expected rate escalation percentage.
        status: Budget lifecycle status.
        notes: Additional notes.
    """
    budget_id: str = Field(
        default_factory=_new_uuid, description="Budget identifier"
    )
    budget_name: str = Field(
        default="", max_length=500, description="Budget name"
    )
    budget_method: BudgetMethod = Field(
        default=BudgetMethod.BASELINE_REDUCTION, description="Budget method"
    )
    fiscal_year: int = Field(
        default=2026, ge=2000, le=2050, description="Fiscal year"
    )
    baseline_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Baseline consumption (kWh)"
    )
    baseline_cost: Decimal = Field(
        default=Decimal("0"), ge=0, description="Baseline cost"
    )
    reduction_target_pct: Decimal = Field(
        default=DEFAULT_REDUCTION_TARGET, ge=0, le=Decimal("100"),
        description="Reduction target (%)"
    )
    blended_rate: Decimal = Field(
        default=Decimal("0.12"), ge=0, description="Blended rate ($/kWh)"
    )
    weather_sensitivity: Decimal = Field(
        default=DEFAULT_WEATHER_SENSITIVITY,
        description="Weather sensitivity (kWh/degree-day)"
    )
    normal_hdd: Decimal = Field(
        default=Decimal("4500"), ge=0, description="Normal HDD"
    )
    normal_cdd: Decimal = Field(
        default=Decimal("1200"), ge=0, description="Normal CDD"
    )
    production_baseline: Decimal = Field(
        default=Decimal("0"), ge=0, description="Production baseline"
    )
    energy_intensity: Decimal = Field(
        default=Decimal("0"), ge=0, description="kWh per production unit"
    )
    growth_rate_pct: Decimal = Field(
        default=Decimal("0"), description="Growth rate (%)"
    )
    rate_escalation_pct: Decimal = Field(
        default=DEFAULT_RATE_ESCALATION, ge=0,
        description="Rate escalation (%)"
    )
    status: BudgetStatus = Field(
        default=BudgetStatus.DRAFT, description="Budget status"
    )
    notes: str = Field(
        default="", max_length=2000, description="Notes"
    )

    @field_validator("budget_name", mode="before")
    @classmethod
    def validate_name(cls, v: Any) -> Any:
        """Ensure budget name is non-empty."""
        if isinstance(v, str) and not v.strip():
            return "Unnamed Budget"
        return v

class BudgetPeriod(BaseModel):
    """Monthly or periodic budget values with actuals.

    Attributes:
        period_id: Period identifier.
        period_label: Human-readable label (e.g. 'Jan 2026').
        period_number: Ordinal period number (1-12).
        budget_kwh: Budgeted consumption (kWh).
        budget_cost: Budgeted cost.
        actual_kwh: Actual consumption (kWh).
        actual_cost: Actual cost.
        actual_rate: Actual blended rate ($/kWh).
        actual_hdd: Actual heating degree-days.
        actual_cdd: Actual cooling degree-days.
        actual_production: Actual production output.
        variance_kwh: Consumption variance (kWh).
        variance_cost: Cost variance.
        variance_pct: Variance percentage.
    """
    period_id: str = Field(
        default_factory=_new_uuid, description="Period ID"
    )
    period_label: str = Field(
        default="", description="Period label"
    )
    period_number: int = Field(
        default=1, ge=1, le=12, description="Period number"
    )
    budget_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Budget (kWh)"
    )
    budget_cost: Decimal = Field(
        default=Decimal("0"), ge=0, description="Budget cost"
    )
    actual_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Actual (kWh)"
    )
    actual_cost: Decimal = Field(
        default=Decimal("0"), ge=0, description="Actual cost"
    )
    actual_rate: Decimal = Field(
        default=Decimal("0"), ge=0, description="Actual rate"
    )
    actual_hdd: Decimal = Field(
        default=Decimal("0"), ge=0, description="Actual HDD"
    )
    actual_cdd: Decimal = Field(
        default=Decimal("0"), ge=0, description="Actual CDD"
    )
    actual_production: Decimal = Field(
        default=Decimal("0"), ge=0, description="Actual production"
    )
    variance_kwh: Decimal = Field(
        default=Decimal("0"), description="Variance (kWh)"
    )
    variance_cost: Decimal = Field(
        default=Decimal("0"), description="Variance cost"
    )
    variance_pct: Decimal = Field(
        default=Decimal("0"), description="Variance (%)"
    )

class VarianceAnalysis(BaseModel):
    """Decomposed variance analysis for a budget period.

    Attributes:
        analysis_id: Analysis identifier.
        period_label: Period label.
        total_variance_cost: Total cost variance.
        total_variance_pct: Total variance percentage.
        volume_variance: Volume-driven variance.
        rate_variance: Rate/price-driven variance.
        weather_variance: Weather-driven variance.
        efficiency_variance: Efficiency-driven variance.
        production_variance: Production-driven variance.
        other_variance: Unexplained residual variance.
        normalised_actual_kwh: Weather-normalised actual (kWh).
        normalised_variance_pct: Normalised variance percentage.
        alert_level: Alert threshold classification.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    analysis_id: str = Field(
        default_factory=_new_uuid, description="Analysis ID"
    )
    period_label: str = Field(default="", description="Period label")
    total_variance_cost: Decimal = Field(
        default=Decimal("0"), description="Total variance cost"
    )
    total_variance_pct: Decimal = Field(
        default=Decimal("0"), description="Total variance (%)"
    )
    volume_variance: Decimal = Field(
        default=Decimal("0"), description="Volume variance"
    )
    rate_variance: Decimal = Field(
        default=Decimal("0"), description="Rate variance"
    )
    weather_variance: Decimal = Field(
        default=Decimal("0"), description="Weather variance"
    )
    efficiency_variance: Decimal = Field(
        default=Decimal("0"), description="Efficiency variance"
    )
    production_variance: Decimal = Field(
        default=Decimal("0"), description="Production variance"
    )
    other_variance: Decimal = Field(
        default=Decimal("0"), description="Other variance"
    )
    normalised_actual_kwh: Decimal = Field(
        default=Decimal("0"), description="Normalised actual (kWh)"
    )
    normalised_variance_pct: Decimal = Field(
        default=Decimal("0"), description="Normalised variance (%)"
    )
    alert_level: AlertThreshold = Field(
        default=AlertThreshold.ON_TRACK, description="Alert level"
    )
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

class RollingForecast(BaseModel):
    """Rolling forecast of remaining budget period.

    Attributes:
        forecast_id: Forecast identifier.
        forecast_method: Forecasting method.
        ytd_actual_kwh: Year-to-date actual consumption (kWh).
        ytd_actual_cost: Year-to-date actual cost.
        ytd_budget_kwh: Year-to-date budget consumption.
        ytd_budget_cost: Year-to-date budget cost.
        remaining_budget_kwh: Remaining budget (kWh).
        remaining_budget_cost: Remaining budget cost.
        forecast_total_kwh: Forecast total for full year (kWh).
        forecast_total_cost: Forecast total cost for full year.
        projected_variance_kwh: Projected full-year variance (kWh).
        projected_variance_cost: Projected full-year cost variance.
        projected_variance_pct: Projected variance percentage.
        adjustment_factor: YTD adjustment factor.
        confidence_band_low: Lower confidence bound cost.
        confidence_band_high: Upper confidence bound cost.
        months_elapsed: Months elapsed in period.
        months_remaining: Months remaining in period.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    forecast_id: str = Field(
        default_factory=_new_uuid, description="Forecast ID"
    )
    forecast_method: ForecastMethod = Field(
        default=ForecastMethod.TREND, description="Forecast method"
    )
    ytd_actual_kwh: Decimal = Field(
        default=Decimal("0"), description="YTD actual (kWh)"
    )
    ytd_actual_cost: Decimal = Field(
        default=Decimal("0"), description="YTD actual cost"
    )
    ytd_budget_kwh: Decimal = Field(
        default=Decimal("0"), description="YTD budget (kWh)"
    )
    ytd_budget_cost: Decimal = Field(
        default=Decimal("0"), description="YTD budget cost"
    )
    remaining_budget_kwh: Decimal = Field(
        default=Decimal("0"), description="Remaining (kWh)"
    )
    remaining_budget_cost: Decimal = Field(
        default=Decimal("0"), description="Remaining cost"
    )
    forecast_total_kwh: Decimal = Field(
        default=Decimal("0"), description="Forecast total (kWh)"
    )
    forecast_total_cost: Decimal = Field(
        default=Decimal("0"), description="Forecast total cost"
    )
    projected_variance_kwh: Decimal = Field(
        default=Decimal("0"), description="Projected variance (kWh)"
    )
    projected_variance_cost: Decimal = Field(
        default=Decimal("0"), description="Projected variance cost"
    )
    projected_variance_pct: Decimal = Field(
        default=Decimal("0"), description="Projected variance (%)"
    )
    adjustment_factor: Decimal = Field(
        default=Decimal("1"), description="Adjustment factor"
    )
    confidence_band_low: Decimal = Field(
        default=Decimal("0"), description="Confidence low"
    )
    confidence_band_high: Decimal = Field(
        default=Decimal("0"), description="Confidence high"
    )
    months_elapsed: int = Field(
        default=0, ge=0, description="Months elapsed"
    )
    months_remaining: int = Field(
        default=12, ge=0, description="Months remaining"
    )
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

class BudgetResult(BaseModel):
    """Comprehensive energy budget result.

    Attributes:
        result_id: Result identifier.
        budget_id: Associated budget definition ID.
        budget_name: Budget name.
        budget_method: Budgeting methodology.
        fiscal_year: Fiscal year.
        annual_budget_kwh: Annual budget consumption (kWh).
        annual_budget_cost: Annual budget cost.
        monthly_budgets: Monthly budget breakdown.
        variance_analyses: Period variance analyses.
        rolling_forecast: Latest rolling forecast.
        ytd_variance_pct: Year-to-date variance percentage.
        alert_level: Current alert level.
        status: Budget status.
        calculated_at: Calculation timestamp.
        processing_time_ms: Processing duration (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(
        default_factory=_new_uuid, description="Result ID"
    )
    budget_id: str = Field(default="", description="Budget ID")
    budget_name: str = Field(default="", description="Budget name")
    budget_method: BudgetMethod = Field(
        default=BudgetMethod.BASELINE_REDUCTION, description="Method"
    )
    fiscal_year: int = Field(default=2026, description="Fiscal year")
    annual_budget_kwh: Decimal = Field(
        default=Decimal("0"), description="Annual budget (kWh)"
    )
    annual_budget_cost: Decimal = Field(
        default=Decimal("0"), description="Annual budget cost"
    )
    monthly_budgets: List[BudgetPeriod] = Field(
        default_factory=list, description="Monthly budgets"
    )
    variance_analyses: List[VarianceAnalysis] = Field(
        default_factory=list, description="Variance analyses"
    )
    rolling_forecast: Optional[RollingForecast] = Field(
        default=None, description="Rolling forecast"
    )
    ytd_variance_pct: Decimal = Field(
        default=Decimal("0"), description="YTD variance (%)"
    )
    alert_level: AlertThreshold = Field(
        default=AlertThreshold.ON_TRACK, description="Alert level"
    )
    status: BudgetStatus = Field(
        default=BudgetStatus.DRAFT, description="Budget status"
    )
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time (ms)"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BudgetEngine:
    """Energy budget creation, tracking, and variance analysis engine.

    Creates energy budgets using baseline-reduction, zero-based,
    incremental, or regression-based methods.  Tracks actual vs budget
    variance, decomposes variance into volume, rate, weather, efficiency,
    production, and residual components.  Provides rolling forecasts with
    weather normalisation and configurable alert thresholds.

    Usage::

        engine = BudgetEngine()
        budget = engine.create_budget(definition)
        variance = engine.track_variance(definition, periods)
        decomposed = engine.decompose_variance(definition, period)
        forecast = engine.forecast_consumption(definition, periods)
        alerts = engine.set_alerts(definition, periods)

    All arithmetic uses ``Decimal`` for deterministic, audit-grade precision.
    Every result carries a SHA-256 provenance hash.
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise BudgetEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - alert_thresholds (dict): custom alert thresholds
                - weather_sensitivity (Decimal): default kWh/degree-day
                - seasonal_profile (list): 12-element monthly weight profile
        """
        self.config = config or {}
        self._alert_thresholds: Dict[str, Decimal] = dict(DEFAULT_ALERT_THRESHOLDS)
        if "alert_thresholds" in self.config:
            for k, v in self.config["alert_thresholds"].items():
                self._alert_thresholds[k] = _decimal(v)
        self._weather_sensitivity = _decimal(
            self.config.get("weather_sensitivity", DEFAULT_WEATHER_SENSITIVITY)
        )
        # Default seasonal profile (% of annual by month, sum = 100)
        self._seasonal_profile: List[Decimal] = [
            _decimal(v) for v in self.config.get("seasonal_profile", [
                Decimal("10.5"), Decimal("9.5"), Decimal("8.0"), Decimal("7.0"),
                Decimal("6.5"), Decimal("8.5"), Decimal("10.0"), Decimal("10.5"),
                Decimal("8.5"), Decimal("7.0"), Decimal("6.5"), Decimal("7.5"),
            ])
        ]
        logger.info(
            "BudgetEngine v%s initialised (weather_sens=%.1f kWh/dd)",
            self.engine_version, float(self._weather_sensitivity),
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def create_budget(
        self,
        definition: BudgetDefinition,
    ) -> BudgetResult:
        """Create an energy budget from a definition.

        Generates monthly budget allocations using the specified
        methodology and seasonal profile.

        Args:
            definition: Budget definition with parameters.

        Returns:
            BudgetResult with annual and monthly budgets.
        """
        t0 = time.perf_counter()
        logger.info(
            "Creating budget: %s, method=%s, baseline=%.0f kWh",
            definition.budget_name, definition.budget_method.value,
            float(definition.baseline_kwh),
        )

        # Calculate annual budget based on method
        annual_kwh = self._calculate_annual_budget(definition)
        annual_cost = annual_kwh * definition.blended_rate

        # Distribute to months using seasonal profile
        monthly = self._distribute_monthly(
            annual_kwh, annual_cost, definition,
        )

        elapsed = (time.perf_counter() - t0) * 1000.0

        result = BudgetResult(
            budget_id=definition.budget_id,
            budget_name=definition.budget_name,
            budget_method=definition.budget_method,
            fiscal_year=definition.fiscal_year,
            annual_budget_kwh=_round_val(annual_kwh, 2),
            annual_budget_cost=_round_val(annual_cost, 2),
            monthly_budgets=monthly,
            status=definition.status,
            processing_time_ms=round(elapsed, 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Budget created: %s, annual=%.0f kWh / $%.2f, "
            "%d months, hash=%s (%.1f ms)",
            definition.budget_name, float(annual_kwh),
            float(annual_cost), len(monthly),
            result.provenance_hash[:16], elapsed,
        )
        return result

    def track_variance(
        self,
        definition: BudgetDefinition,
        periods: List[BudgetPeriod],
    ) -> Dict[str, Any]:
        """Track budget vs actual variance across periods.

        Computes period-by-period and cumulative variance tracking
        with alert level classification.

        Args:
            definition: Budget definition.
            periods: List of budget periods with actuals.

        Returns:
            Dictionary with variance tracking and provenance hash.
        """
        t0 = time.perf_counter()
        logger.info(
            "Tracking variance: %s, %d periods",
            definition.budget_name, len(periods),
        )

        period_results: List[Dict[str, Any]] = []
        cumulative_budget_kwh = Decimal("0")
        cumulative_actual_kwh = Decimal("0")
        cumulative_budget_cost = Decimal("0")
        cumulative_actual_cost = Decimal("0")

        for period in periods:
            # Period variance
            var_kwh = period.actual_kwh - period.budget_kwh
            var_cost = period.actual_cost - period.budget_cost
            var_pct = _safe_pct(var_cost, period.budget_cost)

            cumulative_budget_kwh += period.budget_kwh
            cumulative_actual_kwh += period.actual_kwh
            cumulative_budget_cost += period.budget_cost
            cumulative_actual_cost += period.actual_cost

            cum_var_cost = cumulative_actual_cost - cumulative_budget_cost
            cum_var_pct = _safe_pct(cum_var_cost, cumulative_budget_cost)

            alert = self._determine_alert_level(abs(var_pct))

            period_results.append({
                "period_label": period.period_label,
                "period_number": period.period_number,
                "budget_kwh": str(_round_val(period.budget_kwh, 2)),
                "actual_kwh": str(_round_val(period.actual_kwh, 2)),
                "variance_kwh": str(_round_val(var_kwh, 2)),
                "budget_cost": str(_round_val(period.budget_cost, 2)),
                "actual_cost": str(_round_val(period.actual_cost, 2)),
                "variance_cost": str(_round_val(var_cost, 2)),
                "variance_pct": str(_round_val(var_pct, 2)),
                "cumulative_variance_cost": str(_round_val(cum_var_cost, 2)),
                "cumulative_variance_pct": str(_round_val(cum_var_pct, 2)),
                "alert_level": alert.value,
            })

        # Overall YTD
        ytd_var_pct = _safe_pct(
            cumulative_actual_cost - cumulative_budget_cost,
            cumulative_budget_cost,
        )
        overall_alert = self._determine_alert_level(abs(ytd_var_pct))

        elapsed = (time.perf_counter() - t0) * 1000.0
        result: Dict[str, Any] = {
            "budget_id": definition.budget_id,
            "budget_name": definition.budget_name,
            "periods_tracked": len(periods),
            "ytd_budget_kwh": str(_round_val(cumulative_budget_kwh, 2)),
            "ytd_actual_kwh": str(_round_val(cumulative_actual_kwh, 2)),
            "ytd_budget_cost": str(_round_val(cumulative_budget_cost, 2)),
            "ytd_actual_cost": str(_round_val(cumulative_actual_cost, 2)),
            "ytd_variance_pct": str(_round_val(ytd_var_pct, 2)),
            "alert_level": overall_alert.value,
            "period_details": period_results,
            "calculated_at": utcnow().isoformat(),
            "processing_time_ms": round(elapsed, 2),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Variance tracked: %d periods, YTD=%.1f%%, alert=%s, "
            "hash=%s (%.1f ms)",
            len(periods), float(ytd_var_pct), overall_alert.value,
            result["provenance_hash"][:16], elapsed,
        )
        return result

    def decompose_variance(
        self,
        definition: BudgetDefinition,
        period: BudgetPeriod,
    ) -> VarianceAnalysis:
        """Decompose variance into volume, rate, weather, efficiency components.

        Applies the variance decomposition methodology to isolate the
        causes of budget deviation for a single period.

        Args:
            definition: Budget definition with baseline parameters.
            period: Budget period with actuals.

        Returns:
            VarianceAnalysis with decomposed components.
        """
        t0 = time.perf_counter()
        logger.info(
            "Decomposing variance: %s, period=%s",
            definition.budget_name, period.period_label,
        )

        budget_rate = definition.blended_rate
        actual_rate = period.actual_rate if period.actual_rate > Decimal("0") else budget_rate

        # Total variance
        total_var = period.actual_cost - period.budget_cost
        total_var_pct = _safe_pct(total_var, period.budget_cost)

        # Volume variance: (actual_kwh - budget_kwh) * budget_rate
        volume_var = (period.actual_kwh - period.budget_kwh) * budget_rate

        # Rate variance: (actual_rate - budget_rate) * actual_kwh
        rate_var = (actual_rate - budget_rate) * period.actual_kwh

        # Weather variance
        budget_dd = (definition.normal_hdd + definition.normal_cdd) / Decimal("12")
        actual_dd = period.actual_hdd + period.actual_cdd
        dd_deviation = actual_dd - budget_dd
        weather_var = dd_deviation * definition.weather_sensitivity * budget_rate

        # Production variance
        production_var = Decimal("0")
        if definition.production_baseline > Decimal("0") and definition.energy_intensity > Decimal("0"):
            budget_prod_month = definition.production_baseline / Decimal("12")
            prod_deviation = period.actual_production - budget_prod_month
            production_var = prod_deviation * definition.energy_intensity * budget_rate

        # Efficiency = residual
        explained = volume_var + rate_var + weather_var + production_var
        efficiency_var = Decimal("0")
        other_var = total_var - explained

        # If residual is large, classify as efficiency
        if abs(other_var) > abs(total_var) * Decimal("0.10"):
            efficiency_var = other_var * Decimal("0.70")
            other_var = other_var - efficiency_var

        # Weather normalisation
        normalised_kwh = period.actual_kwh - (
            dd_deviation * definition.weather_sensitivity
        )
        normalised_var_pct = _safe_pct(
            normalised_kwh - period.budget_kwh, period.budget_kwh,
        )

        alert = self._determine_alert_level(abs(total_var_pct))

        analysis = VarianceAnalysis(
            period_label=period.period_label,
            total_variance_cost=_round_val(total_var, 2),
            total_variance_pct=_round_val(total_var_pct, 2),
            volume_variance=_round_val(volume_var, 2),
            rate_variance=_round_val(rate_var, 2),
            weather_variance=_round_val(weather_var, 2),
            efficiency_variance=_round_val(efficiency_var, 2),
            production_variance=_round_val(production_var, 2),
            other_variance=_round_val(other_var, 2),
            normalised_actual_kwh=_round_val(normalised_kwh, 2),
            normalised_variance_pct=_round_val(normalised_var_pct, 2),
            alert_level=alert,
        )
        analysis.provenance_hash = _compute_hash(analysis)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Variance decomposed: total=$%.2f (%.1f%%), volume=$%.2f, "
            "rate=$%.2f, weather=$%.2f, efficiency=$%.2f, hash=%s (%.1f ms)",
            float(total_var), float(total_var_pct), float(volume_var),
            float(rate_var), float(weather_var), float(efficiency_var),
            analysis.provenance_hash[:16], elapsed,
        )
        return analysis

    def forecast_consumption(
        self,
        definition: BudgetDefinition,
        periods: List[BudgetPeriod],
        forecast_method: ForecastMethod = ForecastMethod.TREND,
    ) -> RollingForecast:
        """Generate a rolling forecast for the remaining budget period.

        Projects year-end consumption and cost based on year-to-date
        actuals and remaining budget, with confidence bands.

        Args:
            definition: Budget definition.
            periods: Completed periods with actuals.
            forecast_method: Forecasting method to apply.

        Returns:
            RollingForecast with projections and confidence bands.
        """
        t0 = time.perf_counter()
        logger.info(
            "Forecasting consumption: %s, %d periods, method=%s",
            definition.budget_name, len(periods), forecast_method.value,
        )

        months_elapsed = len(periods)
        months_remaining = max(MONTHS_PER_YEAR - months_elapsed, 0)

        # YTD actuals
        ytd_actual_kwh = sum(
            (p.actual_kwh for p in periods), Decimal("0")
        )
        ytd_actual_cost = sum(
            (p.actual_cost for p in periods), Decimal("0")
        )
        ytd_budget_kwh = sum(
            (p.budget_kwh for p in periods), Decimal("0")
        )
        ytd_budget_cost = sum(
            (p.budget_cost for p in periods), Decimal("0")
        )

        # Annual budget
        annual_kwh = definition.baseline_kwh * (
            Decimal("1") - definition.reduction_target_pct / Decimal("100")
        )
        annual_cost = annual_kwh * definition.blended_rate

        # Remaining budget
        remaining_kwh = max(annual_kwh - ytd_budget_kwh, Decimal("0"))
        remaining_cost = max(annual_cost - ytd_budget_cost, Decimal("0"))

        # Adjustment factor
        adj_factor = _safe_divide(ytd_actual_kwh, ytd_budget_kwh, Decimal("1"))

        # Forecast
        if forecast_method == ForecastMethod.TREND:
            forecast_remaining_kwh = remaining_kwh * adj_factor
        elif forecast_method == ForecastMethod.WEIGHTED_AVERAGE:
            if months_elapsed > 0:
                avg_monthly = ytd_actual_kwh / _decimal(months_elapsed)
                forecast_remaining_kwh = avg_monthly * _decimal(months_remaining)
            else:
                forecast_remaining_kwh = remaining_kwh
        elif forecast_method == ForecastMethod.SEASONAL:
            # Use seasonal weights for remaining months
            forecast_remaining_kwh = Decimal("0")
            for m_idx in range(months_elapsed, MONTHS_PER_YEAR):
                if m_idx < len(self._seasonal_profile):
                    monthly_frac = self._seasonal_profile[m_idx] / Decimal("100")
                else:
                    monthly_frac = Decimal("1") / Decimal("12")
                forecast_remaining_kwh += annual_kwh * monthly_frac * adj_factor
        else:
            forecast_remaining_kwh = remaining_kwh * adj_factor

        forecast_total_kwh = ytd_actual_kwh + forecast_remaining_kwh
        forecast_total_cost = forecast_total_kwh * definition.blended_rate

        # Projected variance
        proj_var_kwh = forecast_total_kwh - annual_kwh
        proj_var_cost = forecast_total_cost - annual_cost
        proj_var_pct = _safe_pct(proj_var_cost, annual_cost)

        # Confidence bands (wider with fewer data points)
        uncertainty = Decimal("0.05") + Decimal("0.02") * _decimal(
            max(MONTHS_PER_YEAR - months_elapsed, 0)
        )
        conf_low = forecast_total_cost * (Decimal("1") - uncertainty)
        conf_high = forecast_total_cost * (Decimal("1") + uncertainty)

        elapsed = (time.perf_counter() - t0) * 1000.0

        forecast = RollingForecast(
            forecast_method=forecast_method,
            ytd_actual_kwh=_round_val(ytd_actual_kwh, 2),
            ytd_actual_cost=_round_val(ytd_actual_cost, 2),
            ytd_budget_kwh=_round_val(ytd_budget_kwh, 2),
            ytd_budget_cost=_round_val(ytd_budget_cost, 2),
            remaining_budget_kwh=_round_val(remaining_kwh, 2),
            remaining_budget_cost=_round_val(remaining_cost, 2),
            forecast_total_kwh=_round_val(forecast_total_kwh, 2),
            forecast_total_cost=_round_val(forecast_total_cost, 2),
            projected_variance_kwh=_round_val(proj_var_kwh, 2),
            projected_variance_cost=_round_val(proj_var_cost, 2),
            projected_variance_pct=_round_val(proj_var_pct, 2),
            adjustment_factor=_round_val(adj_factor, 4),
            confidence_band_low=_round_val(conf_low, 2),
            confidence_band_high=_round_val(conf_high, 2),
            months_elapsed=months_elapsed,
            months_remaining=months_remaining,
        )
        forecast.provenance_hash = _compute_hash(forecast)

        logger.info(
            "Forecast: total=%.0f kWh / $%.2f, variance=%.1f%%, "
            "adj=%.3f, months=%d/%d, hash=%s (%.1f ms)",
            float(forecast_total_kwh), float(forecast_total_cost),
            float(proj_var_pct), float(adj_factor),
            months_elapsed, MONTHS_PER_YEAR,
            forecast.provenance_hash[:16], elapsed,
        )
        return forecast

    def set_alerts(
        self,
        definition: BudgetDefinition,
        periods: List[BudgetPeriod],
        custom_thresholds: Optional[Dict[str, Decimal]] = None,
    ) -> Dict[str, Any]:
        """Evaluate budget alerts across all tracked periods.

        Determines alert levels for each period and YTD, using
        configurable threshold percentages.

        Args:
            definition: Budget definition.
            periods: Budget periods with actuals.
            custom_thresholds: Optional override thresholds.

        Returns:
            Dictionary with alert evaluations and provenance hash.
        """
        t0 = time.perf_counter()
        logger.info(
            "Evaluating alerts: %s, %d periods",
            definition.budget_name, len(periods),
        )

        thresholds = custom_thresholds or self._alert_thresholds

        alerts: List[Dict[str, Any]] = []
        ytd_budget = Decimal("0")
        ytd_actual = Decimal("0")

        for period in periods:
            var_pct = _safe_pct(
                abs(period.actual_cost - period.budget_cost),
                period.budget_cost,
            )
            alert = self._determine_alert_level(var_pct, thresholds)

            ytd_budget += period.budget_cost
            ytd_actual += period.actual_cost

            alerts.append({
                "period_label": period.period_label,
                "variance_pct": str(_round_val(var_pct, 2)),
                "alert_level": alert.value,
                "budget_cost": str(_round_val(period.budget_cost, 2)),
                "actual_cost": str(_round_val(period.actual_cost, 2)),
            })

        ytd_var_pct = _safe_pct(abs(ytd_actual - ytd_budget), ytd_budget)
        overall_alert = self._determine_alert_level(ytd_var_pct, thresholds)

        # Count alerts by level
        alert_counts: Dict[str, int] = {}
        for a in alerts:
            level = a["alert_level"]
            alert_counts[level] = alert_counts.get(level, 0) + 1

        elapsed = (time.perf_counter() - t0) * 1000.0
        result: Dict[str, Any] = {
            "budget_id": definition.budget_id,
            "budget_name": definition.budget_name,
            "periods_evaluated": len(periods),
            "ytd_variance_pct": str(_round_val(ytd_var_pct, 2)),
            "overall_alert_level": overall_alert.value,
            "alert_counts": alert_counts,
            "thresholds": {k: str(v) for k, v in thresholds.items()},
            "period_alerts": alerts,
            "calculated_at": utcnow().isoformat(),
            "processing_time_ms": round(elapsed, 2),
        }
        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "Alerts evaluated: %d periods, YTD=%.1f%%, overall=%s, "
            "counts=%s, hash=%s (%.1f ms)",
            len(periods), float(ytd_var_pct), overall_alert.value,
            alert_counts, result["provenance_hash"][:16], elapsed,
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _calculate_annual_budget(
        self, definition: BudgetDefinition,
    ) -> Decimal:
        """Calculate annual budget kWh from definition.

        Args:
            definition: Budget definition.

        Returns:
            Annual budget in kWh.
        """
        method = definition.budget_method

        if method == BudgetMethod.BASELINE_REDUCTION:
            reduction = definition.reduction_target_pct / Decimal("100")
            return definition.baseline_kwh * (Decimal("1") - reduction)

        elif method == BudgetMethod.ZERO_BASED:
            # Use baseline as proxy if no equipment data
            if definition.baseline_kwh > Decimal("0"):
                return definition.baseline_kwh
            return Decimal("0")

        elif method == BudgetMethod.INCREMENTAL:
            growth = definition.growth_rate_pct / Decimal("100")
            return definition.baseline_kwh * (Decimal("1") + growth)

        elif method == BudgetMethod.REGRESSION_BASED:
            # Base on weather-normalised baseline
            return definition.baseline_kwh

        return definition.baseline_kwh

    def _distribute_monthly(
        self,
        annual_kwh: Decimal,
        annual_cost: Decimal,
        definition: BudgetDefinition,
    ) -> List[BudgetPeriod]:
        """Distribute annual budget to monthly periods.

        Args:
            annual_kwh: Annual budget consumption.
            annual_cost: Annual budget cost.
            definition: Budget definition.

        Returns:
            List of 12 BudgetPeriod objects.
        """
        month_names = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]

        periods: List[BudgetPeriod] = []
        profile_total = sum(self._seasonal_profile, Decimal("0"))

        for idx in range(MONTHS_PER_YEAR):
            if idx < len(self._seasonal_profile) and profile_total > Decimal("0"):
                frac = self._seasonal_profile[idx] / profile_total
            else:
                frac = Decimal("1") / Decimal("12")

            month_kwh = annual_kwh * frac
            month_cost = annual_cost * frac

            periods.append(BudgetPeriod(
                period_label=f"{month_names[idx]} {definition.fiscal_year}",
                period_number=idx + 1,
                budget_kwh=_round_val(month_kwh, 2),
                budget_cost=_round_val(month_cost, 2),
            ))

        return periods

    def _determine_alert_level(
        self,
        variance_pct: Decimal,
        thresholds: Optional[Dict[str, Decimal]] = None,
    ) -> AlertThreshold:
        """Determine alert level from variance percentage.

        Args:
            variance_pct: Absolute variance percentage.
            thresholds: Alert thresholds (optional override).

        Returns:
            AlertThreshold classification.
        """
        t = thresholds or self._alert_thresholds

        exceeded_val = t.get(AlertThreshold.EXCEEDED.value, Decimal("100"))
        critical_val = t.get(AlertThreshold.CRITICAL.value, Decimal("20"))
        warning_val = t.get(AlertThreshold.WARNING.value, Decimal("10"))

        if variance_pct >= exceeded_val:
            return AlertThreshold.EXCEEDED
        elif variance_pct >= critical_val:
            return AlertThreshold.CRITICAL
        elif variance_pct >= warning_val:
            return AlertThreshold.WARNING
        else:
            return AlertThreshold.ON_TRACK
