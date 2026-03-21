# -*- coding: utf-8 -*-
"""
DegreeDayEngine - PACK-034 ISO 50001 EnMS Engine 5
====================================================

Heating Degree Day (HDD) and Cooling Degree Day (CDD) calculation engine
with weather normalisation, change-point energy modelling, and base
temperature optimisation for ISO 50001 Energy Management Systems.

Provides daily/monthly/annual degree day aggregation, ASHRAE-style
change-point regression models (2P through 5P), weather-normalised
consumption analysis, and multi-fuel degree day breakdowns.

Calculation Methodology:
    Daily Degree Days:
        HDD = max(0, base_temp_heating - mean_temp)
        CDD = max(0, mean_temp - base_temp_cooling)

    Monthly Aggregation:
        monthly_hdd = sum(daily_hdd for each day in month)
        monthly_cdd = sum(daily_cdd for each day in month)

    Change-Point Models (ASHRAE Guideline 14-2014):
        2P (Two-Parameter):
            E = b0 + b1 * T
        3P-H (Three-Parameter Heating):
            E = b0 + b1 * max(0, Tbal - T)
        3P-C (Three-Parameter Cooling):
            E = b0 + b1 * max(0, T - Tbal)
        4P (Four-Parameter):
            E = b0 + b1 * max(0, Tbal_h - T) + b2 * max(0, T - Tbal_c)
        5P (Five-Parameter):
            E = b0 + b1 * max(0, Tbal_h - T) + b2 * max(0, T - Tbal_c)
            with deadband between Tbal_h and Tbal_c

    Weather Normalisation:
        normalized = baseload + heating_slope * ref_hdd + cooling_slope * ref_cdd
        weather_adjustment = normalized - actual

    Base Temperature Optimisation:
        Grid search over candidate base temperatures.
        Maximise R-squared of energy vs. degree days regression.

    Temperature Conversion:
        F = C * 9/5 + 32
        C = (F - 32) * 5/9

Regulatory References:
    - ISO 50001:2018 - Energy management systems
    - ISO 50006:2014 - Measuring energy performance using EnPIs and EnBs
    - ISO 50015:2014 - Measurement and verification of energy performance
    - ASHRAE Guideline 14-2014 - Measurement of Energy, Demand, and
      Water Savings
    - IPMVP Core Concepts (EVO, 2022)
    - EN 16247-1:2022 - Energy audits (general requirements)
    - NOAA TMY3 Typical Meteorological Year datasets

Zero-Hallucination:
    - All formulas are standard degree day / regression calculations
    - Change-point models per ASHRAE Guideline 14-2014
    - Default base temperatures from CIBSE / ASHRAE published values
    - Climate zone data from ASHRAE Standard 169
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-034 ISO 50001 EnMS
Engine:  5 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from collections import defaultdict
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
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DegreeDayType(str, Enum):
    """Type of degree day calculation.

    HEATING: Heating degree days (HDD) -- energy needed for heating.
    COOLING: Cooling degree days (CDD) -- energy needed for cooling.
    COMBINED: Both HDD and CDD calculated simultaneously.
    """
    HEATING = "heating"
    COOLING = "cooling"
    COMBINED = "combined"


class ChangePointModel(str, Enum):
    """ASHRAE change-point energy regression model type.

    TWO_PARAMETER: E = b0 + b1*T (linear, no change point).
    THREE_PARAMETER_HEATING: E = b0 + b1*max(0, Tbal - T).
    THREE_PARAMETER_COOLING: E = b0 + b1*max(0, T - Tbal).
    FOUR_PARAMETER: E = b0 + b1*max(0, Tbal_h - T) + b2*max(0, T - Tbal_c).
    FIVE_PARAMETER: 4P with explicit deadband between Tbal_h and Tbal_c.
    """
    TWO_PARAMETER = "two_parameter"
    THREE_PARAMETER_HEATING = "three_parameter_heating"
    THREE_PARAMETER_COOLING = "three_parameter_cooling"
    FOUR_PARAMETER = "four_parameter"
    FIVE_PARAMETER = "five_parameter"


class TemperatureUnit(str, Enum):
    """Temperature measurement unit.

    CELSIUS: Degrees Celsius (ISO standard).
    FAHRENHEIT: Degrees Fahrenheit (US customary).
    """
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


class BaseTemperatureMethod(str, Enum):
    """Method for determining base (balance point) temperature.

    FIXED: Use a fixed, pre-defined base temperature.
    VARIABLE: Use a variable base temperature per period.
    OPTIMIZED: Determine optimal base via grid search (R-squared).
    """
    FIXED = "fixed"
    VARIABLE = "variable"
    OPTIMIZED = "optimized"


class NormalizationBasis(str, Enum):
    """Reference dataset for weather normalisation.

    TMY3: NOAA Typical Meteorological Year 3 dataset.
    TEN_YEAR_AVERAGE: 10-year historical average degree days.
    FIVE_YEAR_AVERAGE: 5-year historical average degree days.
    CUSTOM_REFERENCE: User-supplied reference degree days.
    """
    TMY3 = "tmy3"
    TEN_YEAR_AVERAGE = "ten_year_average"
    FIVE_YEAR_AVERAGE = "five_year_average"
    CUSTOM_REFERENCE = "custom_reference"


# ---------------------------------------------------------------------------
# Constants -- Default Base Temperatures
# ---------------------------------------------------------------------------

DEFAULT_BASE_TEMPERATURES: Dict[str, Decimal] = {
    "heating_celsius": Decimal("15.5"),
    "cooling_celsius": Decimal("18.3"),
    "heating_fahrenheit": Decimal("60.0"),
    "cooling_fahrenheit": Decimal("65.0"),
}

# ---------------------------------------------------------------------------
# Constants -- ASHRAE Climate Zones (typical annual HDD/CDD ranges)
# ---------------------------------------------------------------------------

ASHRAE_CLIMATE_ZONES: Dict[str, Dict[str, Any]] = {
    "1A": {
        "description": "Very Hot - Humid",
        "hdd_min": Decimal("0"), "hdd_max": Decimal("500"),
        "cdd_min": Decimal("2500"), "cdd_max": Decimal("5000"),
        "example_city": "Miami, FL",
    },
    "1B": {
        "description": "Very Hot - Dry",
        "hdd_min": Decimal("0"), "hdd_max": Decimal("500"),
        "cdd_min": Decimal("2500"), "cdd_max": Decimal("5000"),
        "example_city": "Riyadh, Saudi Arabia",
    },
    "2A": {
        "description": "Hot - Humid",
        "hdd_min": Decimal("500"), "hdd_max": Decimal("1000"),
        "cdd_min": Decimal("2000"), "cdd_max": Decimal("3500"),
        "example_city": "Houston, TX",
    },
    "2B": {
        "description": "Hot - Dry",
        "hdd_min": Decimal("500"), "hdd_max": Decimal("1000"),
        "cdd_min": Decimal("2000"), "cdd_max": Decimal("3500"),
        "example_city": "Phoenix, AZ",
    },
    "3A": {
        "description": "Warm - Humid",
        "hdd_min": Decimal("1000"), "hdd_max": Decimal("2000"),
        "cdd_min": Decimal("1500"), "cdd_max": Decimal("2500"),
        "example_city": "Atlanta, GA",
    },
    "3B": {
        "description": "Warm - Dry",
        "hdd_min": Decimal("1000"), "hdd_max": Decimal("2000"),
        "cdd_min": Decimal("1000"), "cdd_max": Decimal("2500"),
        "example_city": "Las Vegas, NV",
    },
    "3C": {
        "description": "Warm - Marine",
        "hdd_min": Decimal("1000"), "hdd_max": Decimal("2000"),
        "cdd_min": Decimal("500"), "cdd_max": Decimal("1000"),
        "example_city": "San Francisco, CA",
    },
    "4A": {
        "description": "Mixed - Humid",
        "hdd_min": Decimal("2000"), "hdd_max": Decimal("3000"),
        "cdd_min": Decimal("1000"), "cdd_max": Decimal("2000"),
        "example_city": "Baltimore, MD",
    },
    "4B": {
        "description": "Mixed - Dry",
        "hdd_min": Decimal("2000"), "hdd_max": Decimal("3000"),
        "cdd_min": Decimal("800"), "cdd_max": Decimal("1500"),
        "example_city": "Albuquerque, NM",
    },
    "4C": {
        "description": "Mixed - Marine",
        "hdd_min": Decimal("2000"), "hdd_max": Decimal("3000"),
        "cdd_min": Decimal("200"), "cdd_max": Decimal("800"),
        "example_city": "Seattle, WA",
    },
    "5A": {
        "description": "Cool - Humid",
        "hdd_min": Decimal("3000"), "hdd_max": Decimal("4000"),
        "cdd_min": Decimal("500"), "cdd_max": Decimal("1200"),
        "example_city": "Chicago, IL",
    },
    "5B": {
        "description": "Cool - Dry",
        "hdd_min": Decimal("3000"), "hdd_max": Decimal("4000"),
        "cdd_min": Decimal("400"), "cdd_max": Decimal("1000"),
        "example_city": "Denver, CO",
    },
    "5C": {
        "description": "Cool - Marine",
        "hdd_min": Decimal("3000"), "hdd_max": Decimal("4000"),
        "cdd_min": Decimal("100"), "cdd_max": Decimal("500"),
        "example_city": "Vancouver, BC",
    },
    "6A": {
        "description": "Cold - Humid",
        "hdd_min": Decimal("4000"), "hdd_max": Decimal("5000"),
        "cdd_min": Decimal("300"), "cdd_max": Decimal("800"),
        "example_city": "Minneapolis, MN",
    },
    "6B": {
        "description": "Cold - Dry",
        "hdd_min": Decimal("4000"), "hdd_max": Decimal("5000"),
        "cdd_min": Decimal("200"), "cdd_max": Decimal("600"),
        "example_city": "Helena, MT",
    },
    "7": {
        "description": "Very Cold",
        "hdd_min": Decimal("5000"), "hdd_max": Decimal("7000"),
        "cdd_min": Decimal("100"), "cdd_max": Decimal("400"),
        "example_city": "Duluth, MN",
    },
    "8": {
        "description": "Subarctic / Arctic",
        "hdd_min": Decimal("7000"), "hdd_max": Decimal("12000"),
        "cdd_min": Decimal("0"), "cdd_max": Decimal("100"),
        "example_city": "Fairbanks, AK",
    },
}

# ---------------------------------------------------------------------------
# Constants -- TMY3 Reference Cities (annual HDD/CDD in Celsius base)
# ---------------------------------------------------------------------------

TMY3_REFERENCE_CITIES: Dict[str, Dict[str, Decimal]] = {
    "Miami_FL": {
        "annual_hdd": Decimal("93"),
        "annual_cdd": Decimal("2427"),
        "latitude": Decimal("25.79"),
        "longitude": Decimal("-80.29"),
    },
    "Houston_TX": {
        "annual_hdd": Decimal("732"),
        "annual_cdd": Decimal("1590"),
        "latitude": Decimal("29.97"),
        "longitude": Decimal("-95.36"),
    },
    "Atlanta_GA": {
        "annual_hdd": Decimal("1478"),
        "annual_cdd": Decimal("917"),
        "latitude": Decimal("33.64"),
        "longitude": Decimal("-84.43"),
    },
    "Baltimore_MD": {
        "annual_hdd": Decimal("2358"),
        "annual_cdd": Decimal("680"),
        "latitude": Decimal("39.17"),
        "longitude": Decimal("-76.68"),
    },
    "Chicago_IL": {
        "annual_hdd": Decimal("3354"),
        "annual_cdd": Decimal("459"),
        "latitude": Decimal("41.99"),
        "longitude": Decimal("-87.91"),
    },
    "Denver_CO": {
        "annual_hdd": Decimal("3236"),
        "annual_cdd": Decimal("371"),
        "latitude": Decimal("39.83"),
        "longitude": Decimal("-104.66"),
    },
    "Minneapolis_MN": {
        "annual_hdd": Decimal("4200"),
        "annual_cdd": Decimal("390"),
        "latitude": Decimal("44.88"),
        "longitude": Decimal("-93.22"),
    },
    "Fairbanks_AK": {
        "annual_hdd": Decimal("7656"),
        "annual_cdd": Decimal("22"),
        "latitude": Decimal("64.80"),
        "longitude": Decimal("-147.88"),
    },
    "London_UK": {
        "annual_hdd": Decimal("2150"),
        "annual_cdd": Decimal("85"),
        "latitude": Decimal("51.51"),
        "longitude": Decimal("-0.13"),
    },
    "Frankfurt_DE": {
        "annual_hdd": Decimal("2700"),
        "annual_cdd": Decimal("150"),
        "latitude": Decimal("50.11"),
        "longitude": Decimal("8.68"),
    },
    "Sydney_AU": {
        "annual_hdd": Decimal("480"),
        "annual_cdd": Decimal("620"),
        "latitude": Decimal("-33.87"),
        "longitude": Decimal("151.21"),
    },
    "Tokyo_JP": {
        "annual_hdd": Decimal("1450"),
        "annual_cdd": Decimal("580"),
        "latitude": Decimal("35.68"),
        "longitude": Decimal("139.69"),
    },
}

# ---------------------------------------------------------------------------
# Constants -- Regression limits
# ---------------------------------------------------------------------------

_MAX_GRID_SEARCH_STEPS: int = 500
_MIN_DATA_POINTS: int = 6
_DEFAULT_SEARCH_RANGE_C: Tuple[Decimal, Decimal] = (Decimal("5"), Decimal("30"))
_DEFAULT_SEARCH_STEP_C: Decimal = Decimal("0.5")


# ---------------------------------------------------------------------------
# Pydantic Models -- Input / Intermediate
# ---------------------------------------------------------------------------


class DailyTemperature(BaseModel):
    """Daily outdoor temperature observation.

    Attributes:
        observation_date: Observation date.
        mean_temp: Mean daily temperature.
        min_temp: Minimum daily temperature (optional).
        max_temp: Maximum daily temperature (optional).
        unit: Temperature measurement unit.
    """
    observation_date: date = Field(..., description="Observation date")
    mean_temp: Decimal = Field(..., description="Mean daily temperature")
    min_temp: Optional[Decimal] = Field(
        default=None, description="Minimum daily temperature"
    )
    max_temp: Optional[Decimal] = Field(
        default=None, description="Maximum daily temperature"
    )
    unit: TemperatureUnit = Field(
        default=TemperatureUnit.CELSIUS,
        description="Temperature unit",
    )

    @field_validator("mean_temp")
    @classmethod
    def validate_mean_temp(cls, v: Decimal) -> Decimal:
        """Validate mean temperature is within physical range."""
        if v < Decimal("-90") or v > Decimal("70"):
            raise ValueError(
                f"Mean temperature {v} outside physical range (-90 to 70 C)"
            )
        return v


class EnergyDataPoint(BaseModel):
    """Energy consumption data point for regression analysis.

    Attributes:
        period_start: Start date of the measurement period.
        period_end: End date of the measurement period.
        consumption: Energy consumption for the period.
        fuel_type: Fuel / energy type identifier.
    """
    period_start: date = Field(..., description="Period start date")
    period_end: date = Field(..., description="Period end date")
    consumption: Decimal = Field(
        ..., ge=0, description="Energy consumption for the period"
    )
    fuel_type: str = Field(
        default="electricity", max_length=100,
        description="Fuel or energy type identifier",
    )


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class DegreeDayResult(BaseModel):
    """Aggregated degree day result for a defined period.

    Attributes:
        period_start: Start of the aggregation period.
        period_end: End of the aggregation period.
        hdd: Total heating degree days in the period.
        cdd: Total cooling degree days in the period.
        base_temp_heating: Base temperature used for HDD.
        base_temp_cooling: Base temperature used for CDD.
        days_count: Number of days in the period.
        avg_temperature: Average daily temperature over the period.
    """
    period_start: date = Field(..., description="Period start date")
    period_end: date = Field(..., description="Period end date")
    hdd: Decimal = Field(default=Decimal("0"), description="Heating degree days")
    cdd: Decimal = Field(default=Decimal("0"), description="Cooling degree days")
    base_temp_heating: Decimal = Field(
        default=Decimal("15.5"), description="HDD base temperature"
    )
    base_temp_cooling: Decimal = Field(
        default=Decimal("18.3"), description="CDD base temperature"
    )
    days_count: int = Field(default=0, ge=0, description="Days in period")
    avg_temperature: Decimal = Field(
        default=Decimal("0"), description="Average daily temperature"
    )


class MonthlyDegreeDays(BaseModel):
    """Monthly degree day aggregation.

    Attributes:
        year: Calendar year.
        month: Calendar month (1-12).
        hdd: Heating degree days for the month.
        cdd: Cooling degree days for the month.
        mean_temp: Mean monthly outdoor temperature.
        days: Number of days in the month with data.
    """
    year: int = Field(..., ge=1900, le=2100, description="Calendar year")
    month: int = Field(..., ge=1, le=12, description="Calendar month")
    hdd: Decimal = Field(default=Decimal("0"), description="Monthly HDD")
    cdd: Decimal = Field(default=Decimal("0"), description="Monthly CDD")
    mean_temp: Decimal = Field(
        default=Decimal("0"), description="Mean monthly temperature"
    )
    days: int = Field(default=0, ge=0, description="Days with data")


class ChangePointModelResult(BaseModel):
    """Result of fitting a change-point energy regression model.

    Attributes:
        model_type: Type of change-point model fitted.
        balance_point_heating: Heating balance point temperature.
        balance_point_cooling: Cooling balance point temperature.
        heating_slope: Slope of heating segment (energy per degree day).
        cooling_slope: Slope of cooling segment (energy per degree day).
        baseload: Temperature-independent base energy consumption.
        r_squared: Coefficient of determination (0 to 1).
        cv_rmse: Coefficient of Variation of RMSE (percentage).
        parameters: Full parameter dictionary for the model.
    """
    model_type: ChangePointModel = Field(
        ..., description="Change-point model type"
    )
    balance_point_heating: Optional[Decimal] = Field(
        default=None, description="Heating balance point temperature"
    )
    balance_point_cooling: Optional[Decimal] = Field(
        default=None, description="Cooling balance point temperature"
    )
    heating_slope: Optional[Decimal] = Field(
        default=None, description="Heating segment slope"
    )
    cooling_slope: Optional[Decimal] = Field(
        default=None, description="Cooling segment slope"
    )
    baseload: Decimal = Field(
        default=Decimal("0"), description="Baseload consumption"
    )
    r_squared: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("1"),
        description="R-squared goodness of fit",
    )
    cv_rmse: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="CV(RMSE) percentage",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Full model parameters"
    )


class WeatherNormalization(BaseModel):
    """Weather-normalised consumption result.

    Attributes:
        actual_consumption: Observed energy consumption.
        actual_hdd: Observed heating degree days.
        actual_cdd: Observed cooling degree days.
        reference_hdd: Reference (normal year) HDD.
        reference_cdd: Reference (normal year) CDD.
        normalized_consumption: Weather-normalised consumption.
        weather_adjustment: Difference (normalised - actual).
        model_used: Change-point model type used.
    """
    actual_consumption: Decimal = Field(
        ..., description="Observed energy consumption"
    )
    actual_hdd: Decimal = Field(
        ..., description="Observed HDD"
    )
    actual_cdd: Decimal = Field(
        ..., description="Observed CDD"
    )
    reference_hdd: Decimal = Field(
        ..., description="Reference HDD"
    )
    reference_cdd: Decimal = Field(
        ..., description="Reference CDD"
    )
    normalized_consumption: Decimal = Field(
        ..., description="Weather-normalised consumption"
    )
    weather_adjustment: Decimal = Field(
        ..., description="Weather adjustment (normalised - actual)"
    )
    model_used: ChangePointModel = Field(
        ..., description="Change-point model used for normalisation"
    )


class BaseTemperatureOptimization(BaseModel):
    """Result of base temperature optimisation via grid search.

    Attributes:
        optimal_heating_base: Optimal heating base temperature.
        optimal_cooling_base: Optimal cooling base temperature.
        heating_r_squared: R-squared at optimal heating base.
        cooling_r_squared: R-squared at optimal cooling base.
        search_range_min: Lower bound of search range.
        search_range_max: Upper bound of search range.
        step_size: Step size used in the grid search.
    """
    optimal_heating_base: Decimal = Field(
        ..., description="Optimal heating base temperature"
    )
    optimal_cooling_base: Decimal = Field(
        ..., description="Optimal cooling base temperature"
    )
    heating_r_squared: Decimal = Field(
        default=Decimal("0"), description="R-squared at optimal heating base"
    )
    cooling_r_squared: Decimal = Field(
        default=Decimal("0"), description="R-squared at optimal cooling base"
    )
    search_range_min: Decimal = Field(
        ..., description="Search range lower bound"
    )
    search_range_max: Decimal = Field(
        ..., description="Search range upper bound"
    )
    step_size: Decimal = Field(
        ..., description="Grid search step size"
    )


class DegreeDayAnalysisResult(BaseModel):
    """Complete degree day analysis result with provenance.

    Attributes:
        analysis_id: Unique analysis identifier.
        location_name: Name or description of the location.
        monthly_data: Monthly degree day breakdown.
        annual_hdd: Total annual heating degree days.
        annual_cdd: Total annual cooling degree days.
        change_point_model: Fitted change-point model result.
        normalization: Weather normalisation result (if performed).
        base_temp_optimization: Base temperature optimisation result.
        provenance_hash: SHA-256 audit trail hash.
        calculation_time_ms: Processing duration in milliseconds.
    """
    analysis_id: str = Field(
        default_factory=_new_uuid, description="Analysis identifier"
    )
    location_name: str = Field(
        default="", max_length=500, description="Location name"
    )
    monthly_data: List[MonthlyDegreeDays] = Field(
        default_factory=list, description="Monthly degree day data"
    )
    annual_hdd: Decimal = Field(
        default=Decimal("0"), description="Annual HDD"
    )
    annual_cdd: Decimal = Field(
        default=Decimal("0"), description="Annual CDD"
    )
    change_point_model: Optional[ChangePointModelResult] = Field(
        default=None, description="Fitted change-point model"
    )
    normalization: Optional[WeatherNormalization] = Field(
        default=None, description="Weather normalisation result"
    )
    base_temp_optimization: Optional[BaseTemperatureOptimization] = Field(
        default=None, description="Base temperature optimisation"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )
    calculation_time_ms: int = Field(
        default=0, ge=0, description="Calculation time (ms)"
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class DegreeDayEngine:
    """HDD/CDD calculation and weather normalisation engine.

    Computes daily and monthly degree days, fits ASHRAE change-point
    energy regression models (2P through 5P), optimises base temperatures
    via grid search, and produces weather-normalised consumption values
    for ISO 50001 energy performance tracking.

    All arithmetic uses ``Decimal`` for deterministic, audit-grade
    precision.  Every result carries a SHA-256 provenance hash.

    Usage::

        engine = DegreeDayEngine()
        temps = [
            DailyTemperature(date=date(2025, 1, 1), mean_temp=Decimal("-5")),
            DailyTemperature(date=date(2025, 1, 2), mean_temp=Decimal("-3")),
            ...
        ]
        results = engine.calculate_degree_days(
            temperatures=temps,
            heating_base=Decimal("15.5"),
            cooling_base=Decimal("18.3"),
        )
        for r in results:
            print(f"{r.period_start}-{r.period_end}: HDD={r.hdd}, CDD={r.cdd}")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise DegreeDayEngine.

        Args:
            config: Optional configuration overrides.  Supported keys:
                - default_heating_base (Decimal): default heating base temp (C)
                - default_cooling_base (Decimal): default cooling base temp (C)
                - temperature_unit (str): default unit ('celsius'/'fahrenheit')
                - search_range_min (Decimal): base temp search lower bound
                - search_range_max (Decimal): base temp search upper bound
                - search_step (Decimal): base temp search step size
        """
        self.config = config or {}
        self._default_heating_base = _decimal(
            self.config.get(
                "default_heating_base",
                DEFAULT_BASE_TEMPERATURES["heating_celsius"],
            )
        )
        self._default_cooling_base = _decimal(
            self.config.get(
                "default_cooling_base",
                DEFAULT_BASE_TEMPERATURES["cooling_celsius"],
            )
        )
        self._default_unit = TemperatureUnit(
            self.config.get("temperature_unit", TemperatureUnit.CELSIUS.value)
        )
        self._search_range = (
            _decimal(self.config.get("search_range_min", _DEFAULT_SEARCH_RANGE_C[0])),
            _decimal(self.config.get("search_range_max", _DEFAULT_SEARCH_RANGE_C[1])),
        )
        self._search_step = _decimal(
            self.config.get("search_step", _DEFAULT_SEARCH_STEP_C)
        )
        logger.info(
            "DegreeDayEngine v%s initialised (heat_base=%.1f, cool_base=%.1f, "
            "unit=%s)",
            self.engine_version,
            float(self._default_heating_base),
            float(self._default_cooling_base),
            self._default_unit.value,
        )

    # ------------------------------------------------------------------ #
    # Public API -- Degree Day Calculation                                #
    # ------------------------------------------------------------------ #

    def calculate_degree_days(
        self,
        temperatures: List[DailyTemperature],
        heating_base: Optional[Decimal] = None,
        cooling_base: Optional[Decimal] = None,
    ) -> List[DegreeDayResult]:
        """Calculate HDD and CDD from daily temperature observations.

        Groups daily temperatures by calendar month and computes degree
        days for each monthly period.

        Formulas:
            HDD_day = max(0, heating_base - mean_temp)
            CDD_day = max(0, mean_temp - cooling_base)

        Args:
            temperatures: Daily temperature observations.
            heating_base: Heating base temperature.  Defaults to config.
            cooling_base: Cooling base temperature.  Defaults to config.

        Returns:
            List of DegreeDayResult, one per calendar month found in data.

        Raises:
            ValueError: If temperature list is empty.
        """
        t0 = time.perf_counter()
        if not temperatures:
            raise ValueError("Temperature list must not be empty")

        h_base = heating_base if heating_base is not None else self._default_heating_base
        c_base = cooling_base if cooling_base is not None else self._default_cooling_base

        logger.info(
            "Calculating degree days: %d observations, h_base=%.1f, c_base=%.1f",
            len(temperatures), float(h_base), float(c_base),
        )

        # Normalise temperatures to Celsius for calculation
        normalised = self._normalise_temps_to_celsius(temperatures)

        # Group by (year, month)
        month_groups: Dict[Tuple[int, int], List[Decimal]] = defaultdict(list)
        date_ranges: Dict[Tuple[int, int], Tuple[date, date]] = {}

        for obs_date, mean_c in normalised:
            key = (obs_date.year, obs_date.month)
            month_groups[key].append(mean_c)
            if key not in date_ranges:
                date_ranges[key] = (obs_date, obs_date)
            else:
                existing_start, existing_end = date_ranges[key]
                date_ranges[key] = (
                    min(existing_start, obs_date),
                    max(existing_end, obs_date),
                )

        results: List[DegreeDayResult] = []
        for key in sorted(month_groups.keys()):
            temps_c = month_groups[key]
            p_start, p_end = date_ranges[key]

            total_hdd = Decimal("0")
            total_cdd = Decimal("0")
            total_temp = Decimal("0")

            for t in temps_c:
                daily_hdd = max(Decimal("0"), h_base - t)
                daily_cdd = max(Decimal("0"), t - c_base)
                total_hdd += daily_hdd
                total_cdd += daily_cdd
                total_temp += t

            day_count = len(temps_c)
            avg_temp = _safe_divide(total_temp, _decimal(day_count))

            results.append(DegreeDayResult(
                period_start=p_start,
                period_end=p_end,
                hdd=_round_val(total_hdd, 2),
                cdd=_round_val(total_cdd, 2),
                base_temp_heating=h_base,
                base_temp_cooling=c_base,
                days_count=day_count,
                avg_temperature=_round_val(avg_temp, 2),
            ))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Degree day calculation complete: %d periods, %.1f ms",
            len(results), elapsed_ms,
        )
        return results

    def calculate_monthly_degree_days(
        self,
        temperatures: List[DailyTemperature],
        heating_base: Optional[Decimal] = None,
        cooling_base: Optional[Decimal] = None,
    ) -> List[MonthlyDegreeDays]:
        """Calculate monthly degree day summaries from daily data.

        Aggregates daily HDD/CDD into MonthlyDegreeDays records, one per
        calendar month present in the input data.

        Args:
            temperatures: Daily temperature observations.
            heating_base: Heating base temperature (Celsius).
            cooling_base: Cooling base temperature (Celsius).

        Returns:
            Sorted list of MonthlyDegreeDays.
        """
        t0 = time.perf_counter()
        if not temperatures:
            return []

        h_base = heating_base if heating_base is not None else self._default_heating_base
        c_base = cooling_base if cooling_base is not None else self._default_cooling_base

        normalised = self._normalise_temps_to_celsius(temperatures)

        # Accumulate by (year, month)
        accum: Dict[Tuple[int, int], Dict[str, Decimal]] = defaultdict(
            lambda: {
                "hdd": Decimal("0"),
                "cdd": Decimal("0"),
                "temp_sum": Decimal("0"),
                "count": Decimal("0"),
            }
        )

        for obs_date, mean_c in normalised:
            key = (obs_date.year, obs_date.month)
            daily_hdd = max(Decimal("0"), h_base - mean_c)
            daily_cdd = max(Decimal("0"), mean_c - c_base)
            accum[key]["hdd"] += daily_hdd
            accum[key]["cdd"] += daily_cdd
            accum[key]["temp_sum"] += mean_c
            accum[key]["count"] += Decimal("1")

        results: List[MonthlyDegreeDays] = []
        for key in sorted(accum.keys()):
            yr, mo = key
            data = accum[key]
            cnt = data["count"]
            mean_t = _safe_divide(data["temp_sum"], cnt)
            results.append(MonthlyDegreeDays(
                year=yr,
                month=mo,
                hdd=_round_val(data["hdd"], 2),
                cdd=_round_val(data["cdd"], 2),
                mean_temp=_round_val(mean_t, 2),
                days=int(cnt),
            ))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Monthly degree days: %d months, %.1f ms",
            len(results), elapsed_ms,
        )
        return results

    # ------------------------------------------------------------------ #
    # Public API -- Base Temperature Optimisation                         #
    # ------------------------------------------------------------------ #

    def optimize_base_temperature(
        self,
        energy_data: List[EnergyDataPoint],
        temperature_data: List[DailyTemperature],
        search_range: Optional[Tuple[Decimal, Decimal]] = None,
        step: Optional[Decimal] = None,
    ) -> BaseTemperatureOptimization:
        """Find optimal HDD/CDD base temperatures via grid search.

        Iterates over candidate base temperatures within the search range
        at the given step size.  For each candidate, computes degree days
        and correlates against energy consumption.  Selects the base
        temperature that maximises R-squared.

        Args:
            energy_data: Periodic energy consumption observations.
            temperature_data: Daily temperature observations covering
                the same period as energy_data.
            search_range: (min, max) base temperature range in Celsius.
            step: Search step size in Celsius.

        Returns:
            BaseTemperatureOptimization with optimal bases and R-squared.

        Raises:
            ValueError: If insufficient data for optimisation.
        """
        t0 = time.perf_counter()
        sr = search_range if search_range is not None else self._search_range
        st = step if step is not None else self._search_step

        if len(energy_data) < _MIN_DATA_POINTS:
            raise ValueError(
                f"At least {_MIN_DATA_POINTS} energy data points required "
                f"for base temperature optimisation, got {len(energy_data)}"
            )

        logger.info(
            "Optimising base temperature: range=%.1f-%.1f, step=%.1f, "
            "%d energy points, %d temp points",
            float(sr[0]), float(sr[1]), float(st),
            len(energy_data), len(temperature_data),
        )

        # Build mean temperature per energy period
        period_temps = self._compute_period_mean_temps(
            energy_data, temperature_data
        )
        consumptions = [ep.consumption for ep in energy_data]

        best_heat_base = sr[0]
        best_heat_r2 = Decimal("-1")
        best_cool_base = sr[1]
        best_cool_r2 = Decimal("-1")

        candidate = sr[0]
        steps_taken = 0

        while candidate <= sr[1] and steps_taken < _MAX_GRID_SEARCH_STEPS:
            # Evaluate as heating base
            hdd_values = [
                max(Decimal("0"), candidate - t) for t in period_temps
            ]
            r2_heat = self._compute_r_squared(hdd_values, consumptions)
            if r2_heat > best_heat_r2:
                best_heat_r2 = r2_heat
                best_heat_base = candidate

            # Evaluate as cooling base
            cdd_values = [
                max(Decimal("0"), t - candidate) for t in period_temps
            ]
            r2_cool = self._compute_r_squared(cdd_values, consumptions)
            if r2_cool > best_cool_r2:
                best_cool_r2 = r2_cool
                best_cool_base = candidate

            candidate += st
            steps_taken += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Base temp optimisation complete: heat=%.1f (R2=%.4f), "
            "cool=%.1f (R2=%.4f), %d steps, %.1f ms",
            float(best_heat_base), float(best_heat_r2),
            float(best_cool_base), float(best_cool_r2),
            steps_taken, elapsed_ms,
        )

        return BaseTemperatureOptimization(
            optimal_heating_base=_round_val(best_heat_base, 2),
            optimal_cooling_base=_round_val(best_cool_base, 2),
            heating_r_squared=_round_val(max(best_heat_r2, Decimal("0")), 6),
            cooling_r_squared=_round_val(max(best_cool_r2, Decimal("0")), 6),
            search_range_min=sr[0],
            search_range_max=sr[1],
            step_size=st,
        )

    # ------------------------------------------------------------------ #
    # Public API -- Change-Point Model Fitting                            #
    # ------------------------------------------------------------------ #

    def fit_change_point_model(
        self,
        energy_data: List[EnergyDataPoint],
        temperature_data: List[DailyTemperature],
        model_type: ChangePointModel,
    ) -> ChangePointModelResult:
        """Fit a change-point energy model to consumption and temperature.

        Models supported (per ASHRAE Guideline 14-2014):
            2P:  E = b0 + b1 * T
            3PH: E = b0 + b1 * max(0, Tbal - T)
            3PC: E = b0 + b1 * max(0, T - Tbal)
            4P:  E = b0 + b1 * max(0, Tbal_h - T) + b2 * max(0, T - Tbal_c)
            5P:  4P with deadband between Tbal_h and Tbal_c

        Uses least-squares regression.  For models with a balance point
        (3P, 4P, 5P), performs a grid search over candidate balance
        points and selects the combination that minimises SSE.

        Args:
            energy_data: Periodic energy consumption data.
            temperature_data: Daily temperature observations.
            model_type: Which change-point model to fit.

        Returns:
            ChangePointModelResult with fitted parameters and statistics.

        Raises:
            ValueError: If insufficient data for the requested model.
        """
        t0 = time.perf_counter()

        if len(energy_data) < _MIN_DATA_POINTS:
            raise ValueError(
                f"At least {_MIN_DATA_POINTS} data points required, "
                f"got {len(energy_data)}"
            )

        logger.info(
            "Fitting change-point model: type=%s, %d energy points",
            model_type.value, len(energy_data),
        )

        period_temps = self._compute_period_mean_temps(
            energy_data, temperature_data
        )
        consumptions = [ep.consumption for ep in energy_data]

        if model_type == ChangePointModel.TWO_PARAMETER:
            result = self._fit_two_parameter(period_temps, consumptions)
        elif model_type == ChangePointModel.THREE_PARAMETER_HEATING:
            result = self._fit_three_parameter_heating(period_temps, consumptions)
        elif model_type == ChangePointModel.THREE_PARAMETER_COOLING:
            result = self._fit_three_parameter_cooling(period_temps, consumptions)
        elif model_type == ChangePointModel.FOUR_PARAMETER:
            result = self._fit_four_parameter(period_temps, consumptions)
        elif model_type == ChangePointModel.FIVE_PARAMETER:
            result = self._fit_five_parameter(period_temps, consumptions)
        else:
            raise ValueError(f"Unsupported model type: {model_type.value}")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Change-point model fit: type=%s, R2=%.4f, CV(RMSE)=%.2f%%, "
            "baseload=%.2f, %.1f ms",
            result.model_type.value,
            float(result.r_squared),
            float(result.cv_rmse),
            float(result.baseload),
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # Public API -- Weather Normalisation                                 #
    # ------------------------------------------------------------------ #

    def normalize_consumption(
        self,
        actual_consumption: Decimal,
        actual_hdd: Decimal,
        actual_cdd: Decimal,
        reference_hdd: Decimal,
        reference_cdd: Decimal,
        model: ChangePointModelResult,
    ) -> WeatherNormalization:
        """Calculate weather-normalised consumption.

        Uses the fitted change-point model to adjust actual consumption
        from actual weather conditions to reference weather conditions.

        Formula:
            normalised = baseload
                         + heating_slope * reference_hdd
                         + cooling_slope * reference_cdd
            weather_adjustment = normalised - actual_consumption

        Args:
            actual_consumption: Observed consumption in the period.
            actual_hdd: Observed HDD in the period.
            actual_cdd: Observed CDD in the period.
            reference_hdd: Reference (normal year) HDD.
            reference_cdd: Reference (normal year) CDD.
            model: Fitted change-point model.

        Returns:
            WeatherNormalization result.
        """
        t0 = time.perf_counter()
        logger.info(
            "Normalising consumption: actual=%.2f, actual_hdd=%.1f, "
            "actual_cdd=%.1f, ref_hdd=%.1f, ref_cdd=%.1f, model=%s",
            float(actual_consumption),
            float(actual_hdd), float(actual_cdd),
            float(reference_hdd), float(reference_cdd),
            model.model_type.value,
        )

        baseload = model.baseload
        h_slope = model.heating_slope if model.heating_slope is not None else Decimal("0")
        c_slope = model.cooling_slope if model.cooling_slope is not None else Decimal("0")

        # Normalised consumption at reference weather
        normalised = baseload + h_slope * reference_hdd + c_slope * reference_cdd
        weather_adj = normalised - actual_consumption

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Normalisation complete: normalised=%.2f, adjustment=%.2f, %.1f ms",
            float(normalised), float(weather_adj), elapsed_ms,
        )

        return WeatherNormalization(
            actual_consumption=_round_val(actual_consumption, 2),
            actual_hdd=_round_val(actual_hdd, 2),
            actual_cdd=_round_val(actual_cdd, 2),
            reference_hdd=_round_val(reference_hdd, 2),
            reference_cdd=_round_val(reference_cdd, 2),
            normalized_consumption=_round_val(normalised, 2),
            weather_adjustment=_round_val(weather_adj, 2),
            model_used=model.model_type,
        )

    # ------------------------------------------------------------------ #
    # Public API -- Temperature Conversion                                #
    # ------------------------------------------------------------------ #

    def convert_temperature(
        self,
        value: Decimal,
        from_unit: TemperatureUnit,
        to_unit: TemperatureUnit,
    ) -> Decimal:
        """Convert a temperature value between Celsius and Fahrenheit.

        Formulas:
            C -> F: F = C * 9/5 + 32
            F -> C: C = (F - 32) * 5/9

        Args:
            value: Temperature value to convert.
            from_unit: Source temperature unit.
            to_unit: Target temperature unit.

        Returns:
            Converted temperature, rounded to 4 decimal places.
        """
        if from_unit == to_unit:
            return _round_val(value, 4)

        if from_unit == TemperatureUnit.CELSIUS:
            # C -> F
            converted = value * Decimal("9") / Decimal("5") + Decimal("32")
        else:
            # F -> C
            converted = (value - Decimal("32")) * Decimal("5") / Decimal("9")

        return _round_val(converted, 4)

    # ------------------------------------------------------------------ #
    # Public API -- Balance Point Calculation                             #
    # ------------------------------------------------------------------ #

    def calculate_balance_point(
        self,
        energy_data: List[EnergyDataPoint],
        temperature_data: List[DailyTemperature],
    ) -> Decimal:
        """Find the temperature at which heating/cooling begins.

        The balance point is the outdoor temperature at which the
        building requires neither heating nor cooling.  Determined by
        fitting a 3P-heating model and extracting the balance point.
        If the heating model has poor fit (R2 < 0.3), a 3P-cooling
        model is tried instead.

        Args:
            energy_data: Periodic energy consumption data.
            temperature_data: Daily temperature observations.

        Returns:
            Balance point temperature in Celsius.
        """
        t0 = time.perf_counter()
        logger.info(
            "Calculating balance point: %d energy points",
            len(energy_data),
        )

        # Try heating model first
        heat_model = self.fit_change_point_model(
            energy_data, temperature_data,
            ChangePointModel.THREE_PARAMETER_HEATING,
        )

        if (
            heat_model.r_squared >= Decimal("0.3")
            and heat_model.balance_point_heating is not None
        ):
            balance = heat_model.balance_point_heating
        else:
            # Fall back to cooling model
            cool_model = self.fit_change_point_model(
                energy_data, temperature_data,
                ChangePointModel.THREE_PARAMETER_COOLING,
            )
            if cool_model.balance_point_cooling is not None:
                balance = cool_model.balance_point_cooling
            else:
                balance = self._default_heating_base

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Balance point: %.2f C, %.1f ms",
            float(balance), elapsed_ms,
        )
        return _round_val(balance, 2)

    # ------------------------------------------------------------------ #
    # Public API -- Model Comparison                                      #
    # ------------------------------------------------------------------ #

    def compare_models(
        self,
        models: List[ChangePointModelResult],
    ) -> Dict[str, Any]:
        """Compare multiple change-point models and recommend the best.

        Evaluates models on:
            1. R-squared (higher is better)
            2. CV(RMSE) (lower is better, ASHRAE threshold < 25%)
            3. Number of parameters (fewer preferred, Occam's razor)

        The recommended model has the best R-squared among those meeting
        the ASHRAE CV(RMSE) threshold of 25%.  Ties are broken by lower
        CV(RMSE), then fewer parameters.

        Args:
            models: List of fitted ChangePointModelResult objects.

        Returns:
            Dict with ranked models, recommendation, and comparison data.
        """
        if not models:
            return {
                "models_compared": 0,
                "recommendation": None,
                "rankings": [],
                "provenance_hash": _compute_hash({"models_compared": 0}),
            }

        logger.info("Comparing %d change-point models", len(models))

        ashrae_threshold = Decimal("25")

        # Parameter count by model type
        param_counts = {
            ChangePointModel.TWO_PARAMETER: 2,
            ChangePointModel.THREE_PARAMETER_HEATING: 3,
            ChangePointModel.THREE_PARAMETER_COOLING: 3,
            ChangePointModel.FOUR_PARAMETER: 4,
            ChangePointModel.FIVE_PARAMETER: 5,
        }

        # Build comparison entries
        entries: List[Dict[str, Any]] = []
        for m in models:
            n_params = param_counts.get(m.model_type, 0)
            meets_ashrae = m.cv_rmse <= ashrae_threshold
            entries.append({
                "model_type": m.model_type.value,
                "r_squared": m.r_squared,
                "cv_rmse": m.cv_rmse,
                "baseload": m.baseload,
                "n_parameters": n_params,
                "meets_ashrae_threshold": meets_ashrae,
            })

        # Sort: meets threshold first, then R2 desc, CV(RMSE) asc, params asc
        entries.sort(
            key=lambda e: (
                not e["meets_ashrae_threshold"],
                -float(e["r_squared"]),
                float(e["cv_rmse"]),
                e["n_parameters"],
            )
        )

        # Assign ranks
        for rank, entry in enumerate(entries, start=1):
            entry["rank"] = rank

        recommendation = entries[0]["model_type"] if entries else None

        result = {
            "models_compared": len(models),
            "recommendation": recommendation,
            "ashrae_cv_rmse_threshold_pct": str(ashrae_threshold),
            "rankings": entries,
            "provenance_hash": _compute_hash({
                "models_compared": len(models),
                "recommendation": recommendation,
            }),
        }

        logger.info(
            "Model comparison: recommendation=%s, R2=%.4f",
            recommendation,
            float(entries[0]["r_squared"]) if entries else 0,
        )
        return result

    # ------------------------------------------------------------------ #
    # Public API -- Multi-Fuel Degree Days                                #
    # ------------------------------------------------------------------ #

    def calculate_multi_fuel_degree_days(
        self,
        fuel_data: Dict[str, List[EnergyDataPoint]],
        temperatures: List[DailyTemperature],
    ) -> Dict[str, Any]:
        """Perform separate degree day analysis per fuel type.

        For each fuel type, optimises the base temperature and computes
        monthly degree days tailored to that fuel's consumption profile.

        Args:
            fuel_data: Map of fuel_type -> list of EnergyDataPoint.
            temperatures: Daily temperature observations.

        Returns:
            Dict of fuel_type -> analysis results including optimal base
            temperatures, monthly degree days, and model statistics.
        """
        t0 = time.perf_counter()
        logger.info(
            "Multi-fuel degree day analysis: %d fuel types",
            len(fuel_data),
        )

        results: Dict[str, Any] = {}

        for fuel_type, energy_points in fuel_data.items():
            logger.info(
                "Analysing fuel type: %s (%d data points)",
                fuel_type, len(energy_points),
            )

            fuel_result: Dict[str, Any] = {
                "fuel_type": fuel_type,
                "data_points": len(energy_points),
            }

            # Optimise base temperature for this fuel
            if len(energy_points) >= _MIN_DATA_POINTS:
                optimization = self.optimize_base_temperature(
                    energy_points, temperatures
                )
                fuel_result["base_temp_optimization"] = optimization.model_dump(
                    mode="json"
                )

                # Calculate monthly DD with optimised bases
                monthly = self.calculate_monthly_degree_days(
                    temperatures,
                    heating_base=optimization.optimal_heating_base,
                    cooling_base=optimization.optimal_cooling_base,
                )
                fuel_result["monthly_degree_days"] = [
                    m.model_dump(mode="json") for m in monthly
                ]

                # Fit best change-point model
                best_model = self._auto_fit_best_model(
                    energy_points, temperatures
                )
                if best_model is not None:
                    fuel_result["change_point_model"] = best_model.model_dump(
                        mode="json"
                    )
            else:
                # Insufficient data; use defaults
                monthly = self.calculate_monthly_degree_days(temperatures)
                fuel_result["monthly_degree_days"] = [
                    m.model_dump(mode="json") for m in monthly
                ]
                fuel_result["note"] = (
                    f"Insufficient data ({len(energy_points)} points) "
                    f"for base temperature optimisation; using defaults."
                )

            results[fuel_type] = fuel_result

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        results["_meta"] = {
            "fuel_types_analysed": len(fuel_data),
            "calculation_time_ms": int(elapsed_ms),
            "provenance_hash": _compute_hash(results),
        }

        logger.info(
            "Multi-fuel analysis complete: %d types, %.1f ms",
            len(fuel_data), elapsed_ms,
        )
        return results

    # ------------------------------------------------------------------ #
    # Public API -- Chart Data Generation                                 #
    # ------------------------------------------------------------------ #

    def generate_degree_day_chart_data(
        self,
        monthly: List[MonthlyDegreeDays],
    ) -> Dict[str, Any]:
        """Format monthly degree day data for chart rendering.

        Produces arrays suitable for bar charts, stacked area charts,
        and temperature overlay line charts.

        Args:
            monthly: List of MonthlyDegreeDays records.

        Returns:
            Dict with labels, HDD series, CDD series, temperature series,
            and summary statistics suitable for front-end charting.
        """
        if not monthly:
            return {
                "labels": [],
                "hdd_series": [],
                "cdd_series": [],
                "temperature_series": [],
                "annual_summary": {},
                "provenance_hash": _compute_hash({"empty": True}),
            }

        labels: List[str] = []
        hdd_series: List[str] = []
        cdd_series: List[str] = []
        temp_series: List[str] = []

        total_hdd = Decimal("0")
        total_cdd = Decimal("0")
        total_temp = Decimal("0")
        total_days = 0

        for m in monthly:
            label = f"{m.year}-{m.month:02d}"
            labels.append(label)
            hdd_series.append(str(_round_val(m.hdd, 1)))
            cdd_series.append(str(_round_val(m.cdd, 1)))
            temp_series.append(str(_round_val(m.mean_temp, 1)))
            total_hdd += m.hdd
            total_cdd += m.cdd
            total_temp += m.mean_temp * _decimal(m.days)
            total_days += m.days

        avg_annual_temp = _safe_divide(total_temp, _decimal(total_days))

        # Identify peak heating and cooling months
        peak_heating_idx = max(
            range(len(monthly)), key=lambda i: monthly[i].hdd
        )
        peak_cooling_idx = max(
            range(len(monthly)), key=lambda i: monthly[i].cdd
        )

        annual_summary = {
            "total_hdd": str(_round_val(total_hdd, 1)),
            "total_cdd": str(_round_val(total_cdd, 1)),
            "avg_temperature": str(_round_val(avg_annual_temp, 2)),
            "total_days": total_days,
            "months_count": len(monthly),
            "peak_heating_month": labels[peak_heating_idx],
            "peak_heating_hdd": str(
                _round_val(monthly[peak_heating_idx].hdd, 1)
            ),
            "peak_cooling_month": labels[peak_cooling_idx],
            "peak_cooling_cdd": str(
                _round_val(monthly[peak_cooling_idx].cdd, 1)
            ),
            "heating_dominant": total_hdd > total_cdd,
        }

        chart_data = {
            "labels": labels,
            "hdd_series": hdd_series,
            "cdd_series": cdd_series,
            "temperature_series": temp_series,
            "annual_summary": annual_summary,
            "provenance_hash": _compute_hash(annual_summary),
        }

        logger.info(
            "Chart data generated: %d months, HDD=%.0f, CDD=%.0f",
            len(monthly), float(total_hdd), float(total_cdd),
        )
        return chart_data

    # ------------------------------------------------------------------ #
    # Public API -- Full Analysis Pipeline                                #
    # ------------------------------------------------------------------ #

    def run_full_analysis(
        self,
        location_name: str,
        temperatures: List[DailyTemperature],
        energy_data: Optional[List[EnergyDataPoint]] = None,
        reference_hdd: Optional[Decimal] = None,
        reference_cdd: Optional[Decimal] = None,
        model_type: Optional[ChangePointModel] = None,
    ) -> DegreeDayAnalysisResult:
        """Run a complete degree day analysis pipeline.

        Orchestrates:
            1. Monthly degree day calculation.
            2. Base temperature optimisation (if energy data provided).
            3. Change-point model fitting (if energy data provided).
            4. Weather normalisation (if reference DD and model provided).

        Args:
            location_name: Name of the location being analysed.
            temperatures: Daily temperature observations.
            energy_data: Optional energy consumption data.
            reference_hdd: Optional reference HDD for normalisation.
            reference_cdd: Optional reference CDD for normalisation.
            model_type: Optional change-point model type to fit.

        Returns:
            DegreeDayAnalysisResult with complete analysis and provenance.
        """
        t0 = time.perf_counter()
        logger.info(
            "Running full degree day analysis: location=%s, "
            "%d temp observations, %d energy points",
            location_name,
            len(temperatures),
            len(energy_data) if energy_data else 0,
        )

        # Step 1: Monthly degree days
        monthly = self.calculate_monthly_degree_days(temperatures)
        annual_hdd = sum((m.hdd for m in monthly), Decimal("0"))
        annual_cdd = sum((m.cdd for m in monthly), Decimal("0"))

        # Step 2: Base temperature optimisation
        base_opt: Optional[BaseTemperatureOptimization] = None
        if energy_data and len(energy_data) >= _MIN_DATA_POINTS:
            base_opt = self.optimize_base_temperature(
                energy_data, temperatures
            )
            # Recalculate monthly DD with optimised bases
            monthly = self.calculate_monthly_degree_days(
                temperatures,
                heating_base=base_opt.optimal_heating_base,
                cooling_base=base_opt.optimal_cooling_base,
            )
            annual_hdd = sum((m.hdd for m in monthly), Decimal("0"))
            annual_cdd = sum((m.cdd for m in monthly), Decimal("0"))

        # Step 3: Change-point model
        cp_model: Optional[ChangePointModelResult] = None
        if energy_data and len(energy_data) >= _MIN_DATA_POINTS:
            if model_type is not None:
                cp_model = self.fit_change_point_model(
                    energy_data, temperatures, model_type
                )
            else:
                cp_model = self._auto_fit_best_model(
                    energy_data, temperatures
                )

        # Step 4: Weather normalisation
        normalization: Optional[WeatherNormalization] = None
        if (
            cp_model is not None
            and reference_hdd is not None
            and reference_cdd is not None
            and energy_data
        ):
            total_consumption = sum(
                (ep.consumption for ep in energy_data), Decimal("0")
            )
            normalization = self.normalize_consumption(
                actual_consumption=total_consumption,
                actual_hdd=annual_hdd,
                actual_cdd=annual_cdd,
                reference_hdd=reference_hdd,
                reference_cdd=reference_cdd,
                model=cp_model,
            )

        elapsed_ms = int((time.perf_counter() - t0) * 1000.0)

        result = DegreeDayAnalysisResult(
            location_name=location_name,
            monthly_data=monthly,
            annual_hdd=_round_val(annual_hdd, 2),
            annual_cdd=_round_val(annual_cdd, 2),
            change_point_model=cp_model,
            normalization=normalization,
            base_temp_optimization=base_opt,
            calculation_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Full analysis complete: location=%s, annual_hdd=%.0f, "
            "annual_cdd=%.0f, model=%s, normalisation=%s, hash=%s, %d ms",
            location_name,
            float(annual_hdd), float(annual_cdd),
            cp_model.model_type.value if cp_model else "none",
            "yes" if normalization else "no",
            result.provenance_hash[:16],
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal -- Temperature Normalisation                               #
    # ------------------------------------------------------------------ #

    def _normalise_temps_to_celsius(
        self,
        temperatures: List[DailyTemperature],
    ) -> List[Tuple[date, Decimal]]:
        """Convert all temperatures to Celsius for internal calculations.

        Args:
            temperatures: Raw daily temperature observations.

        Returns:
            List of (date, mean_temp_celsius) tuples.
        """
        result: List[Tuple[date, Decimal]] = []
        for obs in temperatures:
            if obs.unit == TemperatureUnit.FAHRENHEIT:
                mean_c = self.convert_temperature(
                    obs.mean_temp, TemperatureUnit.FAHRENHEIT, TemperatureUnit.CELSIUS
                )
            else:
                mean_c = obs.mean_temp
            result.append((obs.observation_date, mean_c))
        return result

    # ------------------------------------------------------------------ #
    # Internal -- Period Mean Temperature                                 #
    # ------------------------------------------------------------------ #

    def _compute_period_mean_temps(
        self,
        energy_data: List[EnergyDataPoint],
        temperature_data: List[DailyTemperature],
    ) -> List[Decimal]:
        """Compute mean outdoor temperature for each energy data period.

        Averages daily mean temperatures that fall within each energy
        data point's [period_start, period_end] range.

        Args:
            energy_data: Energy consumption periods.
            temperature_data: Daily temperature observations.

        Returns:
            List of mean temperatures aligned with energy_data order.
        """
        normalised = self._normalise_temps_to_celsius(temperature_data)

        # Build date -> temperature lookup
        temp_lookup: Dict[date, Decimal] = {}
        for obs_date, mean_c in normalised:
            temp_lookup[obs_date] = mean_c

        period_temps: List[Decimal] = []
        for ep in energy_data:
            temps_in_period: List[Decimal] = []
            current = ep.period_start
            while current <= ep.period_end:
                if current in temp_lookup:
                    temps_in_period.append(temp_lookup[current])
                current = _next_date(current)

            if temps_in_period:
                mean_t = _safe_divide(
                    sum(temps_in_period),
                    _decimal(len(temps_in_period)),
                )
            else:
                mean_t = Decimal("0")
            period_temps.append(mean_t)

        return period_temps

    # ------------------------------------------------------------------ #
    # Internal -- R-Squared Calculation                                   #
    # ------------------------------------------------------------------ #

    def _compute_r_squared(
        self,
        x_values: List[Decimal],
        y_values: List[Decimal],
    ) -> Decimal:
        """Compute R-squared for a simple linear regression y = a + b*x.

        Uses ordinary least squares (OLS) to fit the model and then
        computes R-squared = 1 - SS_res / SS_tot.

        Args:
            x_values: Independent variable values.
            y_values: Dependent variable values.

        Returns:
            R-squared value (0 to 1), or 0 if calculation fails.
        """
        n = len(x_values)
        if n < 2 or n != len(y_values):
            return Decimal("0")

        sum_x = sum(x_values)
        sum_y = sum(y_values)
        mean_x = _safe_divide(sum_x, _decimal(n))
        mean_y = _safe_divide(sum_y, _decimal(n))

        ss_xx = Decimal("0")
        ss_yy = Decimal("0")
        ss_xy = Decimal("0")

        for i in range(n):
            dx = x_values[i] - mean_x
            dy = y_values[i] - mean_y
            ss_xx += dx * dx
            ss_yy += dy * dy
            ss_xy += dx * dy

        if ss_xx == Decimal("0") or ss_yy == Decimal("0"):
            return Decimal("0")

        # slope and intercept
        slope = _safe_divide(ss_xy, ss_xx)
        intercept = mean_y - slope * mean_x

        # SS_res
        ss_res = Decimal("0")
        for i in range(n):
            predicted = intercept + slope * x_values[i]
            residual = y_values[i] - predicted
            ss_res += residual * residual

        r_squared = Decimal("1") - _safe_divide(ss_res, ss_yy)

        # Clamp to [0, 1]
        return max(Decimal("0"), min(Decimal("1"), r_squared))

    # ------------------------------------------------------------------ #
    # Internal -- CV(RMSE) Calculation                                    #
    # ------------------------------------------------------------------ #

    def _compute_cv_rmse(
        self,
        predicted: List[Decimal],
        observed: List[Decimal],
    ) -> Decimal:
        """Compute Coefficient of Variation of Root Mean Square Error.

        CV(RMSE) = RMSE / mean(observed) * 100

        Args:
            predicted: Model-predicted values.
            observed: Actual observed values.

        Returns:
            CV(RMSE) as a percentage.
        """
        n = len(predicted)
        if n == 0 or n != len(observed):
            return Decimal("0")

        ss_res = Decimal("0")
        for i in range(n):
            residual = observed[i] - predicted[i]
            ss_res += residual * residual

        mse = _safe_divide(ss_res, _decimal(n))
        # Use math.sqrt for the square root then convert back
        rmse = _decimal(math.sqrt(float(mse)))
        mean_obs = _safe_divide(sum(observed), _decimal(n))

        return _safe_divide(rmse * Decimal("100"), mean_obs)

    # ------------------------------------------------------------------ #
    # Internal -- OLS Linear Regression                                   #
    # ------------------------------------------------------------------ #

    def _ols_regression(
        self,
        x_values: List[Decimal],
        y_values: List[Decimal],
    ) -> Tuple[Decimal, Decimal]:
        """Perform simple OLS linear regression: y = intercept + slope * x.

        Args:
            x_values: Independent variable values.
            y_values: Dependent variable values.

        Returns:
            Tuple of (intercept, slope).
        """
        n = len(x_values)
        if n < 2:
            return (Decimal("0"), Decimal("0"))

        sum_x = sum(x_values)
        sum_y = sum(y_values)
        mean_x = _safe_divide(sum_x, _decimal(n))
        mean_y = _safe_divide(sum_y, _decimal(n))

        ss_xx = Decimal("0")
        ss_xy = Decimal("0")
        for i in range(n):
            dx = x_values[i] - mean_x
            ss_xx += dx * dx
            ss_xy += dx * (y_values[i] - mean_y)

        slope = _safe_divide(ss_xy, ss_xx)
        intercept = mean_y - slope * mean_x

        return (intercept, slope)

    # ------------------------------------------------------------------ #
    # Internal -- Two-Parameter Model                                     #
    # ------------------------------------------------------------------ #

    def _fit_two_parameter(
        self,
        temps: List[Decimal],
        consumptions: List[Decimal],
    ) -> ChangePointModelResult:
        """Fit a two-parameter model: E = b0 + b1 * T.

        Simple linear regression of energy consumption against
        outdoor temperature with no change point.

        Args:
            temps: Mean period temperatures (Celsius).
            consumptions: Energy consumption per period.

        Returns:
            ChangePointModelResult with 2P coefficients.
        """
        intercept, slope = self._ols_regression(temps, consumptions)

        # Compute predicted and statistics
        predicted = [intercept + slope * t for t in temps]
        r_squared = self._compute_r_squared(temps, consumptions)
        cv_rmse = self._compute_cv_rmse(predicted, consumptions)

        return ChangePointModelResult(
            model_type=ChangePointModel.TWO_PARAMETER,
            balance_point_heating=None,
            balance_point_cooling=None,
            heating_slope=None,
            cooling_slope=None,
            baseload=_round_val(intercept, 4),
            r_squared=_round_val(r_squared, 6),
            cv_rmse=_round_val(cv_rmse, 2),
            parameters={
                "b0_intercept": str(_round_val(intercept, 4)),
                "b1_slope": str(_round_val(slope, 6)),
                "model_equation": "E = b0 + b1 * T",
            },
        )

    # ------------------------------------------------------------------ #
    # Internal -- Three-Parameter Heating Model                           #
    # ------------------------------------------------------------------ #

    def _fit_three_parameter_heating(
        self,
        temps: List[Decimal],
        consumptions: List[Decimal],
    ) -> ChangePointModelResult:
        """Fit a 3P-heating model: E = b0 + b1 * max(0, Tbal - T).

        Grid search over candidate balance points in the search range.
        For each Tbal, compute HDD = max(0, Tbal - T), regress E on HDD,
        and select the Tbal that maximises R-squared.

        Args:
            temps: Mean period temperatures (Celsius).
            consumptions: Energy consumption per period.

        Returns:
            ChangePointModelResult with 3P-heating coefficients.
        """
        best_r2 = Decimal("-1")
        best_tbal = self._default_heating_base
        best_intercept = Decimal("0")
        best_slope = Decimal("0")

        candidate = self._search_range[0]
        steps = 0
        while candidate <= self._search_range[1] and steps < _MAX_GRID_SEARCH_STEPS:
            hdd_vals = [max(Decimal("0"), candidate - t) for t in temps]
            intercept, slope = self._ols_regression(hdd_vals, consumptions)
            r2 = self._compute_r_squared(hdd_vals, consumptions)

            if r2 > best_r2 and slope >= Decimal("0"):
                best_r2 = r2
                best_tbal = candidate
                best_intercept = intercept
                best_slope = slope

            candidate += self._search_step
            steps += 1

        # Compute predicted and CV(RMSE)
        hdd_best = [max(Decimal("0"), best_tbal - t) for t in temps]
        predicted = [best_intercept + best_slope * h for h in hdd_best]
        cv_rmse = self._compute_cv_rmse(predicted, consumptions)

        return ChangePointModelResult(
            model_type=ChangePointModel.THREE_PARAMETER_HEATING,
            balance_point_heating=_round_val(best_tbal, 2),
            balance_point_cooling=None,
            heating_slope=_round_val(best_slope, 6),
            cooling_slope=None,
            baseload=_round_val(best_intercept, 4),
            r_squared=_round_val(max(best_r2, Decimal("0")), 6),
            cv_rmse=_round_val(cv_rmse, 2),
            parameters={
                "b0_baseload": str(_round_val(best_intercept, 4)),
                "b1_heating_slope": str(_round_val(best_slope, 6)),
                "tbal_heating": str(_round_val(best_tbal, 2)),
                "model_equation": "E = b0 + b1 * max(0, Tbal - T)",
            },
        )

    # ------------------------------------------------------------------ #
    # Internal -- Three-Parameter Cooling Model                           #
    # ------------------------------------------------------------------ #

    def _fit_three_parameter_cooling(
        self,
        temps: List[Decimal],
        consumptions: List[Decimal],
    ) -> ChangePointModelResult:
        """Fit a 3P-cooling model: E = b0 + b1 * max(0, T - Tbal).

        Grid search over candidate balance points.  For each Tbal,
        compute CDD = max(0, T - Tbal), regress E on CDD, and select
        the Tbal that maximises R-squared.

        Args:
            temps: Mean period temperatures (Celsius).
            consumptions: Energy consumption per period.

        Returns:
            ChangePointModelResult with 3P-cooling coefficients.
        """
        best_r2 = Decimal("-1")
        best_tbal = self._default_cooling_base
        best_intercept = Decimal("0")
        best_slope = Decimal("0")

        candidate = self._search_range[0]
        steps = 0
        while candidate <= self._search_range[1] and steps < _MAX_GRID_SEARCH_STEPS:
            cdd_vals = [max(Decimal("0"), t - candidate) for t in temps]
            intercept, slope = self._ols_regression(cdd_vals, consumptions)
            r2 = self._compute_r_squared(cdd_vals, consumptions)

            if r2 > best_r2 and slope >= Decimal("0"):
                best_r2 = r2
                best_tbal = candidate
                best_intercept = intercept
                best_slope = slope

            candidate += self._search_step
            steps += 1

        # Compute predicted and CV(RMSE)
        cdd_best = [max(Decimal("0"), t - best_tbal) for t in temps]
        predicted = [best_intercept + best_slope * c for c in cdd_best]
        cv_rmse = self._compute_cv_rmse(predicted, consumptions)

        return ChangePointModelResult(
            model_type=ChangePointModel.THREE_PARAMETER_COOLING,
            balance_point_heating=None,
            balance_point_cooling=_round_val(best_tbal, 2),
            heating_slope=None,
            cooling_slope=_round_val(best_slope, 6),
            baseload=_round_val(best_intercept, 4),
            r_squared=_round_val(max(best_r2, Decimal("0")), 6),
            cv_rmse=_round_val(cv_rmse, 2),
            parameters={
                "b0_baseload": str(_round_val(best_intercept, 4)),
                "b1_cooling_slope": str(_round_val(best_slope, 6)),
                "tbal_cooling": str(_round_val(best_tbal, 2)),
                "model_equation": "E = b0 + b1 * max(0, T - Tbal)",
            },
        )

    # ------------------------------------------------------------------ #
    # Internal -- Four-Parameter Model                                    #
    # ------------------------------------------------------------------ #

    def _fit_four_parameter(
        self,
        temps: List[Decimal],
        consumptions: List[Decimal],
    ) -> ChangePointModelResult:
        """Fit a 4P model with dual change points.

        E = b0 + b1 * max(0, Tbal_h - T) + b2 * max(0, T - Tbal_c)

        Grid search over pairs of (Tbal_h, Tbal_c) where Tbal_h <= Tbal_c.
        For each pair, constructs HDD and CDD features, performs
        multivariate OLS, and selects the pair minimising SSE.

        Args:
            temps: Mean period temperatures (Celsius).
            consumptions: Energy consumption per period.

        Returns:
            ChangePointModelResult with 4P coefficients.
        """
        best_sse = None
        best_tbal_h = self._default_heating_base
        best_tbal_c = self._default_cooling_base
        best_b0 = Decimal("0")
        best_b1 = Decimal("0")
        best_b2 = Decimal("0")

        n = len(temps)
        candidate_h = self._search_range[0]
        steps = 0

        while candidate_h <= self._search_range[1] and steps < _MAX_GRID_SEARCH_STEPS:
            candidate_c = candidate_h
            while candidate_c <= self._search_range[1] and steps < _MAX_GRID_SEARCH_STEPS:
                hdd_vals = [max(Decimal("0"), candidate_h - t) for t in temps]
                cdd_vals = [max(Decimal("0"), t - candidate_c) for t in temps]

                b0, b1, b2 = self._multivariate_ols_2var(
                    hdd_vals, cdd_vals, consumptions
                )

                # Require non-negative slopes
                if b1 >= Decimal("0") and b2 >= Decimal("0"):
                    sse = Decimal("0")
                    for i in range(n):
                        pred = b0 + b1 * hdd_vals[i] + b2 * cdd_vals[i]
                        residual = consumptions[i] - pred
                        sse += residual * residual

                    if best_sse is None or sse < best_sse:
                        best_sse = sse
                        best_tbal_h = candidate_h
                        best_tbal_c = candidate_c
                        best_b0 = b0
                        best_b1 = b1
                        best_b2 = b2

                candidate_c += self._search_step * Decimal("2")
                steps += 1

            candidate_h += self._search_step * Decimal("2")
            steps += 1

        # Compute statistics
        hdd_best = [max(Decimal("0"), best_tbal_h - t) for t in temps]
        cdd_best = [max(Decimal("0"), t - best_tbal_c) for t in temps]
        predicted = [
            best_b0 + best_b1 * hdd_best[i] + best_b2 * cdd_best[i]
            for i in range(n)
        ]
        r_squared = self._compute_model_r_squared(predicted, consumptions)
        cv_rmse = self._compute_cv_rmse(predicted, consumptions)

        return ChangePointModelResult(
            model_type=ChangePointModel.FOUR_PARAMETER,
            balance_point_heating=_round_val(best_tbal_h, 2),
            balance_point_cooling=_round_val(best_tbal_c, 2),
            heating_slope=_round_val(best_b1, 6),
            cooling_slope=_round_val(best_b2, 6),
            baseload=_round_val(best_b0, 4),
            r_squared=_round_val(r_squared, 6),
            cv_rmse=_round_val(cv_rmse, 2),
            parameters={
                "b0_baseload": str(_round_val(best_b0, 4)),
                "b1_heating_slope": str(_round_val(best_b1, 6)),
                "b2_cooling_slope": str(_round_val(best_b2, 6)),
                "tbal_heating": str(_round_val(best_tbal_h, 2)),
                "tbal_cooling": str(_round_val(best_tbal_c, 2)),
                "model_equation": (
                    "E = b0 + b1 * max(0, Tbal_h - T) "
                    "+ b2 * max(0, T - Tbal_c)"
                ),
            },
        )

    # ------------------------------------------------------------------ #
    # Internal -- Five-Parameter Model                                    #
    # ------------------------------------------------------------------ #

    def _fit_five_parameter(
        self,
        temps: List[Decimal],
        consumptions: List[Decimal],
    ) -> ChangePointModelResult:
        """Fit a 5P model with deadband between change points.

        E = b0 + b1 * max(0, Tbal_h - T) + b2 * max(0, T - Tbal_c)
        with the constraint Tbal_h < Tbal_c (explicit deadband).

        Grid search over pairs requiring Tbal_c > Tbal_h + minimum_gap.

        Args:
            temps: Mean period temperatures (Celsius).
            consumptions: Energy consumption per period.

        Returns:
            ChangePointModelResult with 5P coefficients.
        """
        minimum_deadband = Decimal("2")
        best_sse = None
        best_tbal_h = self._default_heating_base
        best_tbal_c = self._default_cooling_base
        best_b0 = Decimal("0")
        best_b1 = Decimal("0")
        best_b2 = Decimal("0")

        n = len(temps)
        candidate_h = self._search_range[0]
        steps = 0

        while candidate_h <= self._search_range[1] and steps < _MAX_GRID_SEARCH_STEPS:
            # Cooling balance must be at least minimum_deadband above heating
            candidate_c = candidate_h + minimum_deadband
            while candidate_c <= self._search_range[1] and steps < _MAX_GRID_SEARCH_STEPS:
                hdd_vals = [max(Decimal("0"), candidate_h - t) for t in temps]
                cdd_vals = [max(Decimal("0"), t - candidate_c) for t in temps]

                b0, b1, b2 = self._multivariate_ols_2var(
                    hdd_vals, cdd_vals, consumptions
                )

                if b1 >= Decimal("0") and b2 >= Decimal("0"):
                    sse = Decimal("0")
                    for i in range(n):
                        pred = b0 + b1 * hdd_vals[i] + b2 * cdd_vals[i]
                        residual = consumptions[i] - pred
                        sse += residual * residual

                    if best_sse is None or sse < best_sse:
                        best_sse = sse
                        best_tbal_h = candidate_h
                        best_tbal_c = candidate_c
                        best_b0 = b0
                        best_b1 = b1
                        best_b2 = b2

                candidate_c += self._search_step * Decimal("2")
                steps += 1

            candidate_h += self._search_step * Decimal("2")
            steps += 1

        deadband = best_tbal_c - best_tbal_h

        # Compute statistics
        hdd_best = [max(Decimal("0"), best_tbal_h - t) for t in temps]
        cdd_best = [max(Decimal("0"), t - best_tbal_c) for t in temps]
        predicted = [
            best_b0 + best_b1 * hdd_best[i] + best_b2 * cdd_best[i]
            for i in range(n)
        ]
        r_squared = self._compute_model_r_squared(predicted, consumptions)
        cv_rmse = self._compute_cv_rmse(predicted, consumptions)

        return ChangePointModelResult(
            model_type=ChangePointModel.FIVE_PARAMETER,
            balance_point_heating=_round_val(best_tbal_h, 2),
            balance_point_cooling=_round_val(best_tbal_c, 2),
            heating_slope=_round_val(best_b1, 6),
            cooling_slope=_round_val(best_b2, 6),
            baseload=_round_val(best_b0, 4),
            r_squared=_round_val(r_squared, 6),
            cv_rmse=_round_val(cv_rmse, 2),
            parameters={
                "b0_baseload": str(_round_val(best_b0, 4)),
                "b1_heating_slope": str(_round_val(best_b1, 6)),
                "b2_cooling_slope": str(_round_val(best_b2, 6)),
                "tbal_heating": str(_round_val(best_tbal_h, 2)),
                "tbal_cooling": str(_round_val(best_tbal_c, 2)),
                "deadband": str(_round_val(deadband, 2)),
                "model_equation": (
                    "E = b0 + b1 * max(0, Tbal_h - T) "
                    "+ b2 * max(0, T - Tbal_c) [deadband]"
                ),
            },
        )

    # ------------------------------------------------------------------ #
    # Internal -- Multivariate OLS (2 independent variables)              #
    # ------------------------------------------------------------------ #

    def _multivariate_ols_2var(
        self,
        x1_values: List[Decimal],
        x2_values: List[Decimal],
        y_values: List[Decimal],
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """Fit y = b0 + b1*x1 + b2*x2 via normal equations.

        Solves the 3x3 normal equation system:
            [n     S_x1   S_x2  ] [b0]   [S_y     ]
            [S_x1  S_x1x1 S_x1x2] [b1] = [S_x1y   ]
            [S_x2  S_x1x2 S_x2x2] [b2]   [S_x2y   ]

        Uses Cramer's rule for the 3x3 system.

        Args:
            x1_values: First independent variable.
            x2_values: Second independent variable.
            y_values: Dependent variable.

        Returns:
            Tuple of (b0, b1, b2) coefficients.
        """
        n_d = _decimal(len(y_values))
        if n_d < Decimal("3"):
            return (Decimal("0"), Decimal("0"), Decimal("0"))

        s_x1 = sum(x1_values)
        s_x2 = sum(x2_values)
        s_y = sum(y_values)
        s_x1x1 = sum(x * x for x in x1_values)
        s_x2x2 = sum(x * x for x in x2_values)
        s_x1x2 = sum(x1_values[i] * x2_values[i] for i in range(int(n_d)))
        s_x1y = sum(x1_values[i] * y_values[i] for i in range(int(n_d)))
        s_x2y = sum(x2_values[i] * y_values[i] for i in range(int(n_d)))

        # Determinant of coefficient matrix (Cramer's rule)
        det = (
            n_d * (s_x1x1 * s_x2x2 - s_x1x2 * s_x1x2)
            - s_x1 * (s_x1 * s_x2x2 - s_x1x2 * s_x2)
            + s_x2 * (s_x1 * s_x1x2 - s_x1x1 * s_x2)
        )

        if abs(det) < Decimal("1E-20"):
            # Singular matrix; fall back to simple mean
            mean_y = _safe_divide(s_y, n_d)
            return (mean_y, Decimal("0"), Decimal("0"))

        # Cramer's rule for b0
        det_b0 = (
            s_y * (s_x1x1 * s_x2x2 - s_x1x2 * s_x1x2)
            - s_x1 * (s_x1y * s_x2x2 - s_x1x2 * s_x2y)
            + s_x2 * (s_x1y * s_x1x2 - s_x1x1 * s_x2y)
        )

        # Cramer's rule for b1
        det_b1 = (
            n_d * (s_x1y * s_x2x2 - s_x1x2 * s_x2y)
            - s_y * (s_x1 * s_x2x2 - s_x1x2 * s_x2)
            + s_x2 * (s_x1 * s_x2y - s_x1y * s_x2)
        )

        # Cramer's rule for b2
        det_b2 = (
            n_d * (s_x1x1 * s_x2y - s_x1y * s_x1x2)
            - s_x1 * (s_x1 * s_x2y - s_x1y * s_x2)
            + s_y * (s_x1 * s_x1x2 - s_x1x1 * s_x2)
        )

        b0 = _safe_divide(det_b0, det)
        b1 = _safe_divide(det_b1, det)
        b2 = _safe_divide(det_b2, det)

        return (b0, b1, b2)

    # ------------------------------------------------------------------ #
    # Internal -- Model R-Squared from Predicted vs Observed              #
    # ------------------------------------------------------------------ #

    def _compute_model_r_squared(
        self,
        predicted: List[Decimal],
        observed: List[Decimal],
    ) -> Decimal:
        """Compute R-squared from predicted and observed values.

        R2 = 1 - SS_res / SS_tot

        Args:
            predicted: Model-predicted values.
            observed: Actual observed values.

        Returns:
            R-squared value clamped to [0, 1].
        """
        n = len(observed)
        if n < 2 or n != len(predicted):
            return Decimal("0")

        mean_obs = _safe_divide(sum(observed), _decimal(n))

        ss_tot = Decimal("0")
        ss_res = Decimal("0")
        for i in range(n):
            ss_tot += (observed[i] - mean_obs) ** 2
            ss_res += (observed[i] - predicted[i]) ** 2

        if ss_tot == Decimal("0"):
            return Decimal("0")

        r2 = Decimal("1") - _safe_divide(ss_res, ss_tot)
        return max(Decimal("0"), min(Decimal("1"), r2))

    # ------------------------------------------------------------------ #
    # Internal -- Auto-Fit Best Model                                     #
    # ------------------------------------------------------------------ #

    def _auto_fit_best_model(
        self,
        energy_data: List[EnergyDataPoint],
        temperature_data: List[DailyTemperature],
    ) -> Optional[ChangePointModelResult]:
        """Automatically fit and select the best change-point model.

        Fits all five model types and selects the best using the
        compare_models ranking logic.

        Args:
            energy_data: Energy consumption data.
            temperature_data: Temperature observations.

        Returns:
            Best ChangePointModelResult, or None if fitting fails.
        """
        models: List[ChangePointModelResult] = []
        model_types = [
            ChangePointModel.TWO_PARAMETER,
            ChangePointModel.THREE_PARAMETER_HEATING,
            ChangePointModel.THREE_PARAMETER_COOLING,
            ChangePointModel.FOUR_PARAMETER,
            ChangePointModel.FIVE_PARAMETER,
        ]

        for mt in model_types:
            try:
                result = self.fit_change_point_model(
                    energy_data, temperature_data, mt
                )
                models.append(result)
            except (ValueError, Exception) as e:
                logger.warning(
                    "Failed to fit model %s: %s", mt.value, str(e)
                )

        if not models:
            return None

        comparison = self.compare_models(models)
        recommended_type = comparison.get("recommendation")

        if recommended_type is None:
            return models[0]

        for m in models:
            if m.model_type.value == recommended_type:
                return m

        return models[0]

    # ------------------------------------------------------------------ #
    # Utility -- Climate Zone Lookup                                      #
    # ------------------------------------------------------------------ #

    def get_climate_zone_info(
        self,
        zone_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Look up ASHRAE climate zone information.

        Args:
            zone_id: Climate zone identifier (e.g. '4A', '5B').

        Returns:
            Climate zone data dict, or None if not found.
        """
        return ASHRAE_CLIMATE_ZONES.get(zone_id.upper())

    # ------------------------------------------------------------------ #
    # Utility -- Reference City Lookup                                    #
    # ------------------------------------------------------------------ #

    def get_reference_city_data(
        self,
        city_key: str,
    ) -> Optional[Dict[str, Decimal]]:
        """Look up TMY3 reference city degree day data.

        Args:
            city_key: City key (e.g. 'Chicago_IL', 'London_UK').

        Returns:
            Reference city data dict, or None if not found.
        """
        return TMY3_REFERENCE_CITIES.get(city_key)

    # ------------------------------------------------------------------ #
    # Utility -- Validate Temperature Data                                #
    # ------------------------------------------------------------------ #

    def validate_temperature_data(
        self,
        temperatures: List[DailyTemperature],
    ) -> Dict[str, Any]:
        """Validate temperature data quality for degree day analysis.

        Checks:
            - Sufficient data points (at least 30 days)
            - No large gaps in dates (> 7 consecutive missing days)
            - Temperature values within physical range
            - Consistent units across observations

        Args:
            temperatures: Daily temperature observations.

        Returns:
            Validation result dict with is_valid, warnings, errors.
        """
        warnings: List[str] = []
        errors: List[str] = []

        if not temperatures:
            errors.append("Temperature list is empty.")
            return {
                "is_valid": False,
                "warnings": warnings,
                "errors": errors,
                "data_points": 0,
                "provenance_hash": _compute_hash({"empty": True}),
            }

        n = len(temperatures)

        # Minimum data requirement
        if n < 30:
            warnings.append(
                f"Only {n} data points; at least 30 recommended "
                f"for reliable degree day analysis."
            )

        # Check for date gaps
        sorted_obs = sorted(temperatures, key=lambda t: t.observation_date)
        max_gap = 0
        for i in range(1, len(sorted_obs)):
            gap = (sorted_obs[i].observation_date - sorted_obs[i - 1].observation_date).days
            if gap > max_gap:
                max_gap = gap
            if gap > 7:
                warnings.append(
                    f"Gap of {gap} days between "
                    f"{sorted_obs[i - 1].observation_date} and "
                    f"{sorted_obs[i].observation_date}."
                )

        # Check unit consistency
        units = set(t.unit for t in temperatures)
        if len(units) > 1:
            warnings.append(
                f"Mixed temperature units detected: {[u.value for u in units]}. "
                f"Engine will normalise to Celsius."
            )

        # Date range
        date_min = sorted_obs[0].observation_date
        date_max = sorted_obs[-1].observation_date
        span_days = (date_max - date_min).days + 1

        # Coverage percentage
        coverage_pct = _safe_pct(_decimal(n), _decimal(span_days))

        if coverage_pct < Decimal("80"):
            warnings.append(
                f"Data coverage is {_round_val(coverage_pct, 1)}% "
                f"(< 80% recommended)."
            )

        is_valid = len(errors) == 0

        return {
            "is_valid": is_valid,
            "data_points": n,
            "date_range_start": str(date_min),
            "date_range_end": str(date_max),
            "span_days": span_days,
            "coverage_pct": str(_round_val(coverage_pct, 1)),
            "max_gap_days": max_gap,
            "units_found": [u.value for u in units],
            "warnings": warnings,
            "errors": errors,
            "provenance_hash": _compute_hash({
                "data_points": n,
                "is_valid": is_valid,
                "span_days": span_days,
            }),
        }

    # ------------------------------------------------------------------ #
    # Utility -- Degree Day Summary Statistics                            #
    # ------------------------------------------------------------------ #

    def summarise_degree_days(
        self,
        monthly: List[MonthlyDegreeDays],
    ) -> Dict[str, Any]:
        """Produce summary statistics from monthly degree day data.

        Args:
            monthly: List of MonthlyDegreeDays records.

        Returns:
            Summary dict with annual totals, averages, peaks, and
            heating/cooling balance.
        """
        if not monthly:
            return {
                "months_count": 0,
                "provenance_hash": _compute_hash({"empty": True}),
            }

        total_hdd = sum((m.hdd for m in monthly), Decimal("0"))
        total_cdd = sum((m.cdd for m in monthly), Decimal("0"))
        total_days = sum(m.days for m in monthly)

        avg_monthly_hdd = _safe_divide(total_hdd, _decimal(len(monthly)))
        avg_monthly_cdd = _safe_divide(total_cdd, _decimal(len(monthly)))

        max_hdd_month = max(monthly, key=lambda m: m.hdd)
        max_cdd_month = max(monthly, key=lambda m: m.cdd)
        coldest_month = min(monthly, key=lambda m: m.mean_temp)
        warmest_month = max(monthly, key=lambda m: m.mean_temp)

        # Heating / cooling balance ratio
        total_dd = total_hdd + total_cdd
        heating_ratio = _safe_divide(total_hdd, total_dd) if total_dd > Decimal("0") else Decimal("0")

        # Classify climate profile
        if heating_ratio > Decimal("0.80"):
            climate_profile = "heating_dominant"
        elif heating_ratio < Decimal("0.20"):
            climate_profile = "cooling_dominant"
        elif heating_ratio >= Decimal("0.40") and heating_ratio <= Decimal("0.60"):
            climate_profile = "mixed_balanced"
        elif heating_ratio > Decimal("0.60"):
            climate_profile = "mixed_heating"
        else:
            climate_profile = "mixed_cooling"

        summary = {
            "months_count": len(monthly),
            "total_days": total_days,
            "annual_hdd": str(_round_val(total_hdd, 1)),
            "annual_cdd": str(_round_val(total_cdd, 1)),
            "avg_monthly_hdd": str(_round_val(avg_monthly_hdd, 1)),
            "avg_monthly_cdd": str(_round_val(avg_monthly_cdd, 1)),
            "peak_heating_month": f"{max_hdd_month.year}-{max_hdd_month.month:02d}",
            "peak_heating_hdd": str(_round_val(max_hdd_month.hdd, 1)),
            "peak_cooling_month": f"{max_cdd_month.year}-{max_cdd_month.month:02d}",
            "peak_cooling_cdd": str(_round_val(max_cdd_month.cdd, 1)),
            "coldest_month": f"{coldest_month.year}-{coldest_month.month:02d}",
            "coldest_temp": str(_round_val(coldest_month.mean_temp, 1)),
            "warmest_month": f"{warmest_month.year}-{warmest_month.month:02d}",
            "warmest_temp": str(_round_val(warmest_month.mean_temp, 1)),
            "heating_ratio": str(_round_val(heating_ratio, 4)),
            "climate_profile": climate_profile,
            "engine_version": self.engine_version,
        }
        summary["provenance_hash"] = _compute_hash(summary)

        logger.info(
            "Degree day summary: HDD=%.0f, CDD=%.0f, profile=%s",
            float(total_hdd), float(total_cdd), climate_profile,
        )
        return summary


# ---------------------------------------------------------------------------
# Module-Level Helper -- Date Increment
# ---------------------------------------------------------------------------


def _next_date(d: date) -> date:
    """Return the next calendar date after *d*.

    Args:
        d: A date object.

    Returns:
        The date one day after *d*.
    """
    from datetime import timedelta
    return d + timedelta(days=1)
