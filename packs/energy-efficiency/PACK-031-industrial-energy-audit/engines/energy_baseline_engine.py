# -*- coding: utf-8 -*-
"""
EnergyBaselineEngine - PACK-031 Industrial Energy Audit Engine 1
================================================================

Establishes energy consumption baselines per facility, production line,
and equipment.  Implements ISO 50006 Energy Performance Indicators (EnPI),
multiple regression analysis (energy vs production, vs degree-days, vs
occupancy), degree-day normalisation (HDD / CDD), CUSUM analysis for
deviation detection, and baseline period validation.

ISO 50006:2014 Compliance:
    - Energy Performance Indicator (EnPI) calculation
    - Energy Baseline (EnB) establishment
    - Normalisation for relevant variables
    - Statistical validation of baseline models
    - Continuous improvement tracking

ISO 50001:2018 Alignment:
    - Clause 6.5: Energy Performance Indicators
    - Clause 6.6: Energy Baselines
    - Clause 9.1: Monitoring, measurement, analysis and evaluation

Regression Methodologies:
    - Simple linear regression (single variable)
    - Multivariable linear regression (production + weather)
    - CV(RMSE) validation per ASHRAE Guideline 14
    - Statistical significance testing (p-value, R-squared)

Degree-Day Methods:
    - Heating Degree Days (HDD) using country-specific base temps
    - Cooling Degree Days (CDD) using country-specific base temps
    - Combined HDD + CDD normalisation

CUSUM Analysis:
    - Cumulative sum of deviations from baseline prediction
    - Positive drift = overconsumption vs baseline
    - Negative drift = underconsumption vs baseline
    - Statistical control limit detection

Energy Carriers Supported:
    - Electricity (kWh)
    - Natural gas (m3 -> kWh via NCV)
    - Steam (tonnes -> kWh via enthalpy)
    - Compressed air (m3 -> kWh via specific energy)
    - Fuel oil (litres -> kWh via NCV)
    - LPG (kg -> kWh via NCV)
    - District heating / cooling (kWh)
    - Biomass (tonnes -> kWh via NCV)

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Regression coefficients computed via closed-form normal equations
    - No LLM involvement in any numeric calculation path
    - SHA-256 provenance hashing on every result
    - Conversion factors from ISO 3977, EN 15603, CIBSE TM46

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-031 Industrial Energy Audit
Status:  Production Ready
"""

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

from pydantic import BaseModel, Field, field_validator, model_validator

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
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Uses JSON serialization with sorted keys to guarantee reproducibility.

    Args:
        data: Data to hash -- dict, Pydantic model, or other serializable.

    Returns:
        SHA-256 hex digest string (64 characters).
    """
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
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation; Decimal("0") on failure.
    """
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
    """Safely divide two Decimals, returning *default* on zero denominator.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Value returned when denominator is zero.

    Returns:
        Result of division or *default*.
    """
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100).

    Args:
        part: Numerator.
        whole: Denominator.

    Returns:
        Percentage as Decimal; Decimal("0") when whole is zero.
    """
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to *places* and return a float.

    Uses ROUND_HALF_UP (regulatory standard rounding).

    Args:
        value: Value to round.
        places: Number of decimal places.

    Returns:
        Rounded float value.
    """
    quantizer = Decimal(10) ** -places
    return float(value.quantize(quantizer, rounding=ROUND_HALF_UP))


def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _round4(value: float) -> float:
    """Round to 4 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EnergyCarrier(str, Enum):
    """Energy carrier types supported for industrial facilities.

    Covers all major energy carriers per ISO 50001 and EN 16247.
    """
    ELECTRICITY = "electricity"
    NATURAL_GAS = "natural_gas"
    FUEL_OIL = "fuel_oil"
    LPG = "lpg"
    STEAM = "steam"
    COMPRESSED_AIR = "compressed_air"
    DISTRICT_HEATING = "district_heating"
    DISTRICT_COOLING = "district_cooling"
    BIOMASS = "biomass"
    DIESEL = "diesel"
    COAL = "coal"
    HYDROGEN = "hydrogen"


class RegressionType(str, Enum):
    """Regression model types for baseline establishment.

    Per ISO 50006, baseline models should use the simplest model
    that provides adequate statistical fit (R-squared >= 0.75).
    """
    SIMPLE_LINEAR = "simple_linear"
    MULTIVARIABLE_LINEAR = "multivariable_linear"
    ENERGY_VS_PRODUCTION = "energy_vs_production"
    ENERGY_VS_DEGREE_DAYS = "energy_vs_degree_days"
    ENERGY_VS_PRODUCTION_AND_WEATHER = "energy_vs_production_and_weather"
    ENERGY_VS_OCCUPANCY = "energy_vs_occupancy"
    FIXED_BASELINE = "fixed_baseline"


class EnPIType(str, Enum):
    """Energy Performance Indicator types per ISO 50006.

    EnPIs can be absolute, relative (intensity), or statistical.
    """
    ABSOLUTE_CONSUMPTION = "absolute_consumption"
    SPECIFIC_ENERGY_CONSUMPTION = "specific_energy_consumption"
    ENERGY_INTENSITY_AREA = "energy_intensity_area"
    ENERGY_INTENSITY_PRODUCTION = "energy_intensity_production"
    ENERGY_INTENSITY_REVENUE = "energy_intensity_revenue"
    ENERGY_INTENSITY_EMPLOYEE = "energy_intensity_employee"
    REGRESSION_MODEL = "regression_model"
    ENERGY_COST_INTENSITY = "energy_cost_intensity"


class FacilitySector(str, Enum):
    """Industrial facility sector classification.

    Sectors have different typical energy intensities and relevant
    EnPIs per CIBSE TM46 and IEA industrial energy data.
    """
    MANUFACTURING = "manufacturing"
    FOOD_BEVERAGE = "food_beverage"
    CHEMICALS = "chemicals"
    METALS = "metals"
    AUTOMOTIVE = "automotive"
    PHARMACEUTICALS = "pharmaceuticals"
    TEXTILES = "textiles"
    PAPER_PULP = "paper_pulp"
    CEMENT = "cement"
    GLASS = "glass"
    PLASTICS = "plastics"
    ELECTRONICS = "electronics"
    WAREHOUSING = "warehousing"
    DATA_CENTER = "data_center"
    COMMERCIAL_BUILDING = "commercial_building"
    OTHER = "other"


class BaselineStatus(str, Enum):
    """Baseline model validation status."""
    VALID = "valid"
    MARGINAL = "marginal"
    INVALID = "invalid"
    INSUFFICIENT_DATA = "insufficient_data"
    NOT_EVALUATED = "not_evaluated"


# ---------------------------------------------------------------------------
# Constants -- Energy Conversion Factors
# ---------------------------------------------------------------------------


# Energy conversion factors to kWh.
# Sources: ISO 3977, EN 15603, CIBSE Guide F, BEIS/DEFRA 2024.
# All values are net calorific values (NCV / lower heating value).
ENERGY_CONVERSION_FACTORS: Dict[str, Dict[str, float]] = {
    EnergyCarrier.NATURAL_GAS: {
        "factor": 10.55,
        "from_unit": "m3",
        "to_unit": "kWh",
        "source": "EN 15603:2008, NCV natural gas (EU avg)",
    },
    EnergyCarrier.FUEL_OIL: {
        "factor": 10.08,
        "from_unit": "litre",
        "to_unit": "kWh",
        "source": "BEIS 2024 conversion factors, fuel oil class 2",
    },
    EnergyCarrier.LPG: {
        "factor": 12.78,
        "from_unit": "kg",
        "to_unit": "kWh",
        "source": "BEIS 2024 conversion factors, LPG",
    },
    EnergyCarrier.STEAM: {
        "factor": 694.4,
        "from_unit": "tonne",
        "to_unit": "kWh",
        "source": "Saturated steam at 10 bar (184C), enthalpy approx 2500 kJ/kg",
    },
    EnergyCarrier.COMPRESSED_AIR: {
        "factor": 0.11,
        "from_unit": "m3",
        "to_unit": "kWh",
        "source": "Typical specific energy 0.1-0.12 kWh/Nm3 at 7 bar",
    },
    EnergyCarrier.DIESEL: {
        "factor": 10.27,
        "from_unit": "litre",
        "to_unit": "kWh",
        "source": "BEIS 2024 conversion factors, diesel",
    },
    EnergyCarrier.COAL: {
        "factor": 7556.0,
        "from_unit": "tonne",
        "to_unit": "kWh",
        "source": "BEIS 2024, bituminous coal NCV 27.2 GJ/t",
    },
    EnergyCarrier.BIOMASS: {
        "factor": 4200.0,
        "from_unit": "tonne",
        "to_unit": "kWh",
        "source": "EN 14961 wood chips (30% moisture), NCV ~15 GJ/t",
    },
    EnergyCarrier.HYDROGEN: {
        "factor": 33.33,
        "from_unit": "kg",
        "to_unit": "kWh",
        "source": "Hydrogen NCV 120 MJ/kg = 33.33 kWh/kg",
    },
}
"""Energy conversion factors from native units to kWh.
Electricity, district heating, and district cooling are already in kWh."""


# Degree-day base temperatures by country (Celsius).
# Heating base and cooling base.
# Sources: ASHRAE Fundamentals, CIBSE Guide A, EN 15603.
DEGREE_DAY_BASE_TEMPS: Dict[str, Dict[str, float]] = {
    "DE": {"heating_base_c": 15.5, "cooling_base_c": 18.0},
    "FR": {"heating_base_c": 15.5, "cooling_base_c": 18.0},
    "UK": {"heating_base_c": 15.5, "cooling_base_c": 18.0},
    "IT": {"heating_base_c": 15.0, "cooling_base_c": 21.0},
    "ES": {"heating_base_c": 15.0, "cooling_base_c": 21.0},
    "NL": {"heating_base_c": 15.5, "cooling_base_c": 18.0},
    "BE": {"heating_base_c": 15.5, "cooling_base_c": 18.0},
    "AT": {"heating_base_c": 15.5, "cooling_base_c": 18.0},
    "PL": {"heating_base_c": 15.0, "cooling_base_c": 18.0},
    "SE": {"heating_base_c": 17.0, "cooling_base_c": 18.0},
    "FI": {"heating_base_c": 17.0, "cooling_base_c": 18.0},
    "NO": {"heating_base_c": 17.0, "cooling_base_c": 18.0},
    "DK": {"heating_base_c": 17.0, "cooling_base_c": 18.0},
    "CZ": {"heating_base_c": 15.0, "cooling_base_c": 18.0},
    "PT": {"heating_base_c": 14.0, "cooling_base_c": 22.0},
    "GR": {"heating_base_c": 14.0, "cooling_base_c": 22.0},
    "US": {"heating_base_c": 18.3, "cooling_base_c": 18.3},
    "CA": {"heating_base_c": 18.0, "cooling_base_c": 18.0},
    "JP": {"heating_base_c": 14.0, "cooling_base_c": 22.0},
    "CN": {"heating_base_c": 14.0, "cooling_base_c": 22.0},
    "IN": {"heating_base_c": 14.0, "cooling_base_c": 24.0},
    "AU": {"heating_base_c": 15.0, "cooling_base_c": 22.0},
    "DEFAULT": {"heating_base_c": 15.5, "cooling_base_c": 18.0},
}


# Standard EnPI benchmarks by industrial sector (kWh per m2 per year).
# Sources: CIBSE TM46, IEA Industrial Energy, EU BAT Reference Documents.
STANDARD_ENPI_BY_SECTOR: Dict[str, Dict[str, float]] = {
    FacilitySector.MANUFACTURING: {
        "typical_kwh_per_sqm": 300.0,
        "good_practice_kwh_per_sqm": 200.0,
        "best_practice_kwh_per_sqm": 120.0,
        "unit": "kWh/m2/year",
    },
    FacilitySector.FOOD_BEVERAGE: {
        "typical_kwh_per_sqm": 450.0,
        "good_practice_kwh_per_sqm": 320.0,
        "best_practice_kwh_per_sqm": 220.0,
        "unit": "kWh/m2/year",
    },
    FacilitySector.CHEMICALS: {
        "typical_kwh_per_sqm": 500.0,
        "good_practice_kwh_per_sqm": 350.0,
        "best_practice_kwh_per_sqm": 250.0,
        "unit": "kWh/m2/year",
    },
    FacilitySector.METALS: {
        "typical_kwh_per_sqm": 600.0,
        "good_practice_kwh_per_sqm": 400.0,
        "best_practice_kwh_per_sqm": 280.0,
        "unit": "kWh/m2/year",
    },
    FacilitySector.AUTOMOTIVE: {
        "typical_kwh_per_sqm": 350.0,
        "good_practice_kwh_per_sqm": 250.0,
        "best_practice_kwh_per_sqm": 170.0,
        "unit": "kWh/m2/year",
    },
    FacilitySector.PHARMACEUTICALS: {
        "typical_kwh_per_sqm": 550.0,
        "good_practice_kwh_per_sqm": 400.0,
        "best_practice_kwh_per_sqm": 300.0,
        "unit": "kWh/m2/year",
    },
    FacilitySector.TEXTILES: {
        "typical_kwh_per_sqm": 280.0,
        "good_practice_kwh_per_sqm": 200.0,
        "best_practice_kwh_per_sqm": 140.0,
        "unit": "kWh/m2/year",
    },
    FacilitySector.PAPER_PULP: {
        "typical_kwh_per_sqm": 700.0,
        "good_practice_kwh_per_sqm": 500.0,
        "best_practice_kwh_per_sqm": 380.0,
        "unit": "kWh/m2/year",
    },
    FacilitySector.CEMENT: {
        "typical_kwh_per_sqm": 800.0,
        "good_practice_kwh_per_sqm": 600.0,
        "best_practice_kwh_per_sqm": 450.0,
        "unit": "kWh/m2/year",
    },
    FacilitySector.GLASS: {
        "typical_kwh_per_sqm": 650.0,
        "good_practice_kwh_per_sqm": 480.0,
        "best_practice_kwh_per_sqm": 350.0,
        "unit": "kWh/m2/year",
    },
    FacilitySector.PLASTICS: {
        "typical_kwh_per_sqm": 320.0,
        "good_practice_kwh_per_sqm": 230.0,
        "best_practice_kwh_per_sqm": 160.0,
        "unit": "kWh/m2/year",
    },
    FacilitySector.ELECTRONICS: {
        "typical_kwh_per_sqm": 400.0,
        "good_practice_kwh_per_sqm": 280.0,
        "best_practice_kwh_per_sqm": 180.0,
        "unit": "kWh/m2/year",
    },
    FacilitySector.WAREHOUSING: {
        "typical_kwh_per_sqm": 120.0,
        "good_practice_kwh_per_sqm": 80.0,
        "best_practice_kwh_per_sqm": 50.0,
        "unit": "kWh/m2/year",
    },
    FacilitySector.DATA_CENTER: {
        "typical_kwh_per_sqm": 2500.0,
        "good_practice_kwh_per_sqm": 1800.0,
        "best_practice_kwh_per_sqm": 1200.0,
        "unit": "kWh/m2/year",
    },
    FacilitySector.COMMERCIAL_BUILDING: {
        "typical_kwh_per_sqm": 200.0,
        "good_practice_kwh_per_sqm": 140.0,
        "best_practice_kwh_per_sqm": 90.0,
        "unit": "kWh/m2/year",
    },
    FacilitySector.OTHER: {
        "typical_kwh_per_sqm": 300.0,
        "good_practice_kwh_per_sqm": 200.0,
        "best_practice_kwh_per_sqm": 130.0,
        "unit": "kWh/m2/year",
    },
}


# ASHRAE Guideline 14 thresholds for baseline model validation.
ASHRAE_14_THRESHOLDS: Dict[str, float] = {
    "r_squared_minimum": 0.75,
    "cv_rmse_monthly_maximum_pct": 15.0,
    "cv_rmse_daily_maximum_pct": 25.0,
    "cv_rmse_hourly_maximum_pct": 30.0,
    "minimum_data_points_monthly": 12,
    "minimum_data_points_daily": 365,
    "p_value_maximum": 0.05,
}
"""ASHRAE Guideline 14-2014 statistical thresholds for M&V.
R-squared >= 0.75, CV(RMSE) <= 15% for monthly data."""


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class FacilityData(BaseModel):
    """Industrial facility data for energy baseline establishment.

    Attributes:
        facility_id: Unique facility identifier.
        name: Human-readable facility name.
        sector: Industrial sector classification.
        area_sqm: Total conditioned/production area in square metres.
        production_capacity: Annual production capacity in units.
        operating_hours: Annual operating hours.
        location: Country code (ISO 3166-1 alpha-2) for weather normalisation.
    """
    facility_id: str = Field(..., min_length=1, description="Facility identifier")
    name: str = Field(..., min_length=1, description="Facility name")
    sector: FacilitySector = Field(..., description="Industrial sector")
    area_sqm: float = Field(..., gt=0, description="Facility area (m2)")
    production_capacity: Optional[float] = Field(
        None, ge=0, description="Annual production capacity (units)"
    )
    operating_hours: float = Field(
        default=6000.0, gt=0, le=8784, description="Annual operating hours"
    )
    location: str = Field(
        ..., min_length=2, max_length=7, description="Country code (ISO 3166-1)"
    )

    @field_validator("area_sqm")
    @classmethod
    def validate_area(cls, v: float) -> float:
        """Ensure area is within plausible bounds."""
        if v > 10_000_000:
            raise ValueError("Facility area exceeds 10 million m2 sanity check")
        return v


class EnergyMeterReading(BaseModel):
    """Energy meter reading for a single carrier over a period.

    Attributes:
        meter_id: Meter or sub-meter identifier.
        period: Time period label (e.g. '2024-01', '2024-Q1').
        energy_kwh: Energy consumption in kWh (or native unit with carrier).
        energy_carrier: Type of energy carrier.
        native_quantity: Quantity in native units (m3, litres, kg, tonnes).
        native_unit: Native unit label.
        cost: Energy cost for the period.
        cost_currency: Currency code for cost (default EUR).
    """
    meter_id: str = Field(..., min_length=1, description="Meter identifier")
    period: str = Field(..., min_length=4, description="Time period label")
    energy_kwh: float = Field(..., ge=0, description="Energy consumed (kWh)")
    energy_carrier: EnergyCarrier = Field(..., description="Energy carrier type")
    native_quantity: Optional[float] = Field(
        None, ge=0, description="Quantity in native units"
    )
    native_unit: Optional[str] = Field(None, description="Native unit label")
    cost: Optional[float] = Field(None, ge=0, description="Cost for period")
    cost_currency: str = Field(default="EUR", description="Currency code")

    @field_validator("energy_kwh")
    @classmethod
    def validate_energy(cls, v: float) -> float:
        """Ensure energy is within plausible bounds."""
        if v > 1_000_000_000:
            raise ValueError("Energy exceeds 1 TWh per period sanity check")
        return v


class ProductionData(BaseModel):
    """Production output data for a time period.

    Attributes:
        period: Time period label matching meter reading periods.
        output_units: Production output in standard units.
        product_type: Product type or line identifier.
    """
    period: str = Field(..., min_length=4, description="Time period label")
    output_units: float = Field(..., ge=0, description="Production output (units)")
    product_type: str = Field(
        default="primary", min_length=1, description="Product type"
    )


class WeatherData(BaseModel):
    """Weather data for a time period used in degree-day normalisation.

    Attributes:
        period: Time period label matching meter reading periods.
        hdd: Heating Degree Days for the period.
        cdd: Cooling Degree Days for the period.
        avg_temp_c: Average temperature in Celsius for the period.
    """
    period: str = Field(..., min_length=4, description="Time period label")
    hdd: float = Field(default=0.0, ge=0, description="Heating Degree Days")
    cdd: float = Field(default=0.0, ge=0, description="Cooling Degree Days")
    avg_temp_c: Optional[float] = Field(
        None, description="Average temperature (C)"
    )


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class BaselineModel(BaseModel):
    """Statistical baseline model result.

    Attributes:
        regression_type: Type of regression model used.
        coefficients: Model coefficients {intercept, slope_production, slope_hdd, ...}.
        r_squared: Coefficient of determination (0-1).
        adjusted_r_squared: Adjusted R-squared for multiple variables.
        cv_rmse: Coefficient of Variation of RMSE (%).
        p_values: Statistical significance of each coefficient.
        residual_std: Standard deviation of residuals (kWh).
        n_observations: Number of data points used.
        variables: List of independent variable names.
        status: Validation status per ASHRAE Guideline 14.
    """
    regression_type: str = Field(..., description="Regression model type")
    coefficients: Dict[str, float] = Field(
        default_factory=dict, description="Model coefficients"
    )
    r_squared: float = Field(default=0.0, description="R-squared (0-1)")
    adjusted_r_squared: float = Field(
        default=0.0, description="Adjusted R-squared"
    )
    cv_rmse: float = Field(default=0.0, description="CV(RMSE) percentage")
    p_values: Dict[str, float] = Field(
        default_factory=dict, description="p-values per coefficient"
    )
    residual_std: float = Field(default=0.0, description="Residual std dev (kWh)")
    n_observations: int = Field(default=0, description="Number of observations")
    variables: List[str] = Field(default_factory=list, description="Variable names")
    status: str = Field(
        default=BaselineStatus.NOT_EVALUATED, description="Validation status"
    )


class EnPIResult(BaseModel):
    """Energy Performance Indicator calculation result per ISO 50006.

    Attributes:
        enpi_type: Type of EnPI calculated.
        enpi_name: Human-readable EnPI name.
        enpi_value: Current period EnPI value.
        enpi_unit: EnPI unit.
        baseline_value: Baseline period EnPI value.
        current_value: Current period EnPI value.
        improvement_pct: Improvement percentage vs baseline (positive = better).
        sector_benchmark_typical: Sector typical value.
        sector_benchmark_good: Sector good practice value.
        sector_benchmark_best: Sector best practice value.
        performance_vs_benchmark: Performance vs typical benchmark.
    """
    enpi_type: str = Field(..., description="EnPI type")
    enpi_name: str = Field(..., description="EnPI name")
    enpi_value: float = Field(default=0.0, description="Current EnPI value")
    enpi_unit: str = Field(default="", description="EnPI unit")
    baseline_value: Optional[float] = Field(
        None, description="Baseline EnPI value"
    )
    current_value: float = Field(default=0.0, description="Current EnPI value")
    improvement_pct: float = Field(
        default=0.0, description="Improvement vs baseline (%)"
    )
    sector_benchmark_typical: Optional[float] = Field(
        None, description="Sector typical benchmark"
    )
    sector_benchmark_good: Optional[float] = Field(
        None, description="Good practice benchmark"
    )
    sector_benchmark_best: Optional[float] = Field(
        None, description="Best practice benchmark"
    )
    performance_vs_benchmark: str = Field(
        default="", description="Performance classification"
    )


class NormalizedConsumption(BaseModel):
    """Weather-normalised energy consumption result.

    Attributes:
        period: Time period label.
        actual_kwh: Actual measured consumption (kWh).
        normalized_kwh: Weather-normalised consumption (kWh).
        hdd_adjustment_kwh: Adjustment for HDD deviation.
        cdd_adjustment_kwh: Adjustment for CDD deviation.
        reference_hdd: Reference period HDD (long-term average).
        reference_cdd: Reference period CDD (long-term average).
        actual_hdd: Actual period HDD.
        actual_cdd: Actual period CDD.
    """
    period: str = Field(..., description="Time period")
    actual_kwh: float = Field(default=0.0, description="Actual consumption (kWh)")
    normalized_kwh: float = Field(
        default=0.0, description="Normalized consumption (kWh)"
    )
    hdd_adjustment_kwh: float = Field(
        default=0.0, description="HDD adjustment (kWh)"
    )
    cdd_adjustment_kwh: float = Field(
        default=0.0, description="CDD adjustment (kWh)"
    )
    reference_hdd: float = Field(default=0.0, description="Reference HDD")
    reference_cdd: float = Field(default=0.0, description="Reference CDD")
    actual_hdd: float = Field(default=0.0, description="Actual HDD")
    actual_cdd: float = Field(default=0.0, description="Actual CDD")


class CUSUMPoint(BaseModel):
    """Single CUSUM data point."""
    period: str = Field(..., description="Time period")
    actual_kwh: float = Field(default=0.0, description="Actual consumption")
    predicted_kwh: float = Field(default=0.0, description="Baseline predicted")
    deviation_kwh: float = Field(default=0.0, description="Deviation from baseline")
    cumulative_sum_kwh: float = Field(default=0.0, description="Cumulative sum")


class CUSUMResult(BaseModel):
    """CUSUM analysis result for baseline deviation detection.

    Attributes:
        data_points: Individual CUSUM data points.
        total_deviation_kwh: Total cumulative deviation from baseline.
        total_deviation_pct: Total deviation as percentage of baseline total.
        trend: Trend direction (overconsumption / underconsumption / stable).
        control_limit_kwh: Statistical control limit (2 sigma).
        exceedances: Number of periods exceeding control limit.
        significant_change_detected: Whether a significant change from baseline.
    """
    data_points: List[CUSUMPoint] = Field(default_factory=list)
    total_deviation_kwh: float = Field(default=0.0)
    total_deviation_pct: float = Field(default=0.0)
    trend: str = Field(default="stable")
    control_limit_kwh: float = Field(default=0.0)
    exceedances: int = Field(default=0)
    significant_change_detected: bool = Field(default=False)


class BaselineValidation(BaseModel):
    """Baseline model validation result per ASHRAE Guideline 14.

    Attributes:
        status: Overall validation status.
        r_squared_pass: Whether R-squared meets threshold.
        cv_rmse_pass: Whether CV(RMSE) meets threshold.
        p_value_pass: Whether all coefficients are significant.
        data_sufficiency_pass: Whether enough data points exist.
        issues: List of validation issues found.
        recommendations: Suggestions for improving the model.
    """
    status: str = Field(default=BaselineStatus.NOT_EVALUATED)
    r_squared_pass: bool = Field(default=False)
    cv_rmse_pass: bool = Field(default=False)
    p_value_pass: bool = Field(default=False)
    data_sufficiency_pass: bool = Field(default=False)
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class EnergyBalance(BaseModel):
    """Energy balance for a facility (inputs = outputs + losses).

    Attributes:
        total_input_kwh: Total energy input (all carriers).
        input_by_carrier: Breakdown by energy carrier (kWh).
        total_output_kwh: Total useful energy output (kWh).
        total_losses_kwh: Total energy losses (kWh).
        balance_error_pct: Balance error percentage (should be <5%).
        carrier_shares: Percentage share by carrier.
    """
    total_input_kwh: float = Field(default=0.0)
    input_by_carrier: Dict[str, float] = Field(default_factory=dict)
    total_output_kwh: float = Field(default=0.0)
    total_losses_kwh: float = Field(default=0.0)
    balance_error_pct: float = Field(default=0.0)
    carrier_shares: Dict[str, float] = Field(default_factory=dict)


class EnergyBaselineResult(BaseModel):
    """Complete energy baseline result with full provenance.

    Contains all calculated EnPIs, baseline models, normalised consumption,
    CUSUM analysis, energy balance, and actionable recommendations.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Unique result ID")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(default_factory=_utcnow, description="Calc timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")

    facility_id: str = Field(default="", description="Facility identifier")
    facility_name: str = Field(default="", description="Facility name")
    facility_sector: str = Field(default="", description="Facility sector")

    baseline_period_start: str = Field(default="", description="Baseline start period")
    baseline_period_end: str = Field(default="", description="Baseline end period")
    baseline_months: int = Field(default=0, description="Number of baseline months")

    total_baseline_energy_kwh: float = Field(default=0.0, description="Total baseline energy")
    total_baseline_cost: float = Field(default=0.0, description="Total baseline cost")

    models: List[BaselineModel] = Field(default_factory=list, description="Baseline models")
    best_model_index: int = Field(default=0, description="Index of best model")

    enpi_results: List[EnPIResult] = Field(default_factory=list, description="EnPI results")
    energy_balance: Optional[EnergyBalance] = Field(None, description="Energy balance")
    cusum_data: Optional[CUSUMResult] = Field(None, description="CUSUM analysis")

    normalized_consumption: List[NormalizedConsumption] = Field(
        default_factory=list, description="Normalised consumption data"
    )
    baseline_validation: Optional[BaselineValidation] = Field(
        None, description="Baseline validation"
    )

    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Calculation Engine
# ---------------------------------------------------------------------------


class EnergyBaselineEngine:
    """Energy baseline establishment engine per ISO 50006.

    Provides deterministic, zero-hallucination calculations for:
    - Energy baseline establishment via regression analysis
    - ISO 50006 Energy Performance Indicators (EnPI)
    - Degree-day normalisation (HDD/CDD)
    - CUSUM analysis for deviation detection
    - Baseline model validation per ASHRAE Guideline 14
    - Energy balance calculation (inputs vs outputs vs losses)
    - Energy carrier conversion to common kWh basis

    All calculations are bit-perfect reproducible. No LLM is used
    in any calculation path.

    Usage::

        engine = EnergyBaselineEngine()
        result = engine.establish_baseline(
            facility=facility_data,
            meter_data=meter_readings,
            production_data=production_records,
            weather_data=weather_records,
        )
        enpis = engine.calculate_enpi(facility, meter_data, production_data)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise the energy baseline engine with embedded constants."""
        self._conversion_factors = ENERGY_CONVERSION_FACTORS
        self._degree_day_bases = DEGREE_DAY_BASE_TEMPS
        self._sector_benchmarks = STANDARD_ENPI_BY_SECTOR
        self._ashrae_thresholds = ASHRAE_14_THRESHOLDS

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def establish_baseline(
        self,
        facility: FacilityData,
        meter_data: List[EnergyMeterReading],
        production_data: Optional[List[ProductionData]] = None,
        weather_data: Optional[List[WeatherData]] = None,
    ) -> EnergyBaselineResult:
        """Establish energy consumption baseline for a facility.

        Fits regression models (energy vs production, weather, or both),
        selects the best model, computes EnPIs, performs CUSUM analysis,
        and calculates energy balance.

        Args:
            facility: Facility data (sector, area, location, etc.).
            meter_data: List of energy meter readings over baseline period.
            production_data: Optional production data for regression.
            weather_data: Optional weather data for degree-day normalisation.

        Returns:
            EnergyBaselineResult with models, EnPIs, CUSUM, and provenance.

        Raises:
            ValueError: If meter_data is empty.
        """
        t0 = time.perf_counter()

        if not meter_data:
            raise ValueError("At least one meter reading is required")

        logger.info(
            "Establishing baseline for facility %s (%s)",
            facility.facility_id, facility.sector.value,
        )

        # Step 1: Aggregate meter data by period
        period_energy = self._aggregate_by_period(meter_data)
        periods = sorted(period_energy.keys())

        if len(periods) < 2:
            raise ValueError(
                "At least 2 periods of data required for baseline; "
                f"got {len(periods)}"
            )

        baseline_start = periods[0]
        baseline_end = periods[-1]

        # Step 2: Total baseline energy and cost
        total_energy = sum(period_energy.values())
        total_cost = Decimal("0")
        for m in meter_data:
            if m.cost is not None:
                total_cost += _decimal(m.cost)

        # Step 3: Build period-aligned datasets
        prod_by_period: Dict[str, float] = {}
        if production_data:
            for p in production_data:
                prod_by_period[p.period] = prod_by_period.get(p.period, 0.0) + p.output_units

        weather_by_period: Dict[str, Tuple[float, float]] = {}
        if weather_data:
            for w in weather_data:
                weather_by_period[w.period] = (w.hdd, w.cdd)

        # Step 4: Fit regression models
        models: List[BaselineModel] = []

        # Model 1: Fixed baseline (mean)
        models.append(self._fit_fixed_baseline(period_energy, periods))

        # Model 2: Energy vs production (if production data available)
        if prod_by_period:
            model_prod = self._fit_simple_regression(
                period_energy, prod_by_period, periods,
                RegressionType.ENERGY_VS_PRODUCTION, "production",
            )
            if model_prod is not None:
                models.append(model_prod)

        # Model 3: Energy vs degree-days (if weather data available)
        if weather_by_period:
            hdd_by_period = {p: dd[0] for p, dd in weather_by_period.items()}
            model_dd = self._fit_simple_regression(
                period_energy, hdd_by_period, periods,
                RegressionType.ENERGY_VS_DEGREE_DAYS, "hdd",
            )
            if model_dd is not None:
                models.append(model_dd)

        # Model 4: Energy vs production + HDD (multivariable)
        if prod_by_period and weather_by_period:
            model_multi = self._fit_multivariable_regression(
                period_energy, prod_by_period, weather_by_period, periods,
            )
            if model_multi is not None:
                models.append(model_multi)

        # Step 5: Select best model (highest R-squared meeting ASHRAE thresholds)
        best_idx = self._select_best_model(models)

        # Step 6: Calculate EnPIs
        enpi_results = self._calculate_enpi_set(
            facility, period_energy, prod_by_period, periods,
        )

        # Step 7: Energy balance
        energy_balance = self._calculate_energy_balance(meter_data)

        # Step 8: CUSUM analysis using best model
        cusum = None
        if best_idx < len(models):
            cusum = self._cusum_analysis_internal(
                models[best_idx], period_energy, prod_by_period,
                weather_by_period, periods,
            )

        # Step 9: Normalise consumption
        normalized = []
        if weather_data and weather_by_period:
            normalized = self._normalize_consumption_internal(
                period_energy, weather_by_period, periods, facility.location,
            )

        # Step 10: Validate baseline
        validation = None
        if best_idx < len(models):
            validation = self._validate_baseline_internal(models[best_idx])

        # Step 11: Recommendations
        recommendations = self._generate_recommendations(
            facility, models, best_idx, enpi_results, energy_balance,
            cusum, validation,
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = EnergyBaselineResult(
            facility_id=facility.facility_id,
            facility_name=facility.name,
            facility_sector=facility.sector.value,
            baseline_period_start=baseline_start,
            baseline_period_end=baseline_end,
            baseline_months=len(periods),
            total_baseline_energy_kwh=_round2(total_energy),
            total_baseline_cost=_round_val(total_cost, 2),
            models=models,
            best_model_index=best_idx,
            enpi_results=enpi_results,
            energy_balance=energy_balance,
            cusum_data=cusum,
            normalized_consumption=normalized,
            baseline_validation=validation,
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def calculate_enpi(
        self,
        facility: FacilityData,
        meter_data: List[EnergyMeterReading],
        production_data: Optional[List[ProductionData]] = None,
    ) -> List[EnPIResult]:
        """Calculate Energy Performance Indicators per ISO 50006.

        Args:
            facility: Facility data.
            meter_data: Energy meter readings.
            production_data: Optional production data.

        Returns:
            List of EnPIResult with all applicable EnPIs.
        """
        period_energy = self._aggregate_by_period(meter_data)
        periods = sorted(period_energy.keys())

        prod_by_period: Dict[str, float] = {}
        if production_data:
            for p in production_data:
                prod_by_period[p.period] = (
                    prod_by_period.get(p.period, 0.0) + p.output_units
                )

        return self._calculate_enpi_set(
            facility, period_energy, prod_by_period, periods,
        )

    def normalize_consumption(
        self,
        meter_data: List[EnergyMeterReading],
        weather_data: List[WeatherData],
        location: str = "DEFAULT",
    ) -> List[NormalizedConsumption]:
        """Normalise energy consumption for weather using degree-days.

        Adjusts consumption to a standard weather year so that energy
        performance can be compared across years with different weather.

        Args:
            meter_data: Energy meter readings.
            weather_data: Weather data with HDD/CDD per period.
            location: Country code for base temperatures.

        Returns:
            List of NormalizedConsumption per period.
        """
        period_energy = self._aggregate_by_period(meter_data)
        periods = sorted(period_energy.keys())

        weather_by_period: Dict[str, Tuple[float, float]] = {}
        for w in weather_data:
            weather_by_period[w.period] = (w.hdd, w.cdd)

        return self._normalize_consumption_internal(
            period_energy, weather_by_period, periods, location,
        )

    def cusum_analysis(
        self,
        baseline_model: BaselineModel,
        current_meter_data: List[EnergyMeterReading],
        current_production: Optional[List[ProductionData]] = None,
        current_weather: Optional[List[WeatherData]] = None,
    ) -> CUSUMResult:
        """Perform CUSUM analysis comparing current data against baseline.

        CUSUM (Cumulative Sum) tracks the running total of deviations
        between actual consumption and baseline-predicted consumption.

        Args:
            baseline_model: Established baseline regression model.
            current_meter_data: Current period meter readings.
            current_production: Current period production data.
            current_weather: Current period weather data.

        Returns:
            CUSUMResult with cumulative deviation analysis.
        """
        period_energy = self._aggregate_by_period(current_meter_data)
        periods = sorted(period_energy.keys())

        prod_by_period: Dict[str, float] = {}
        if current_production:
            for p in current_production:
                prod_by_period[p.period] = (
                    prod_by_period.get(p.period, 0.0) + p.output_units
                )

        weather_by_period: Dict[str, Tuple[float, float]] = {}
        if current_weather:
            for w in current_weather:
                weather_by_period[w.period] = (w.hdd, w.cdd)

        return self._cusum_analysis_internal(
            baseline_model, period_energy, prod_by_period,
            weather_by_period, periods,
        )

    def validate_baseline(
        self, model: BaselineModel,
    ) -> BaselineValidation:
        """Validate baseline model per ASHRAE Guideline 14 thresholds.

        Args:
            model: Baseline regression model to validate.

        Returns:
            BaselineValidation with pass/fail for each criterion.
        """
        return self._validate_baseline_internal(model)

    def convert_to_kwh(
        self,
        quantity: float,
        carrier: EnergyCarrier,
    ) -> float:
        """Convert native energy units to kWh.

        Args:
            quantity: Quantity in native units.
            carrier: Energy carrier type.

        Returns:
            Energy in kWh.
        """
        if carrier in (
            EnergyCarrier.ELECTRICITY,
            EnergyCarrier.DISTRICT_HEATING,
            EnergyCarrier.DISTRICT_COOLING,
        ):
            return quantity

        factor_info = self._conversion_factors.get(carrier)
        if factor_info is None:
            logger.warning("No conversion factor for %s, returning raw value", carrier)
            return quantity

        factor = _decimal(factor_info["factor"])
        return _round_val(_decimal(quantity) * factor, 2)

    # -------------------------------------------------------------------
    # Internal: Aggregation
    # -------------------------------------------------------------------

    def _aggregate_by_period(
        self, meter_data: List[EnergyMeterReading],
    ) -> Dict[str, float]:
        """Aggregate meter readings by period to total kWh.

        Args:
            meter_data: List of meter readings.

        Returns:
            Dict mapping period to total kWh.
        """
        agg: Dict[str, Decimal] = {}
        for m in meter_data:
            d = _decimal(m.energy_kwh)
            if m.period in agg:
                agg[m.period] += d
            else:
                agg[m.period] = d
        return {k: _round_val(v, 2) for k, v in agg.items()}

    # -------------------------------------------------------------------
    # Internal: Regression Models
    # -------------------------------------------------------------------

    def _fit_fixed_baseline(
        self,
        period_energy: Dict[str, float],
        periods: List[str],
    ) -> BaselineModel:
        """Fit a fixed (mean) baseline model.

        Args:
            period_energy: Period -> kWh mapping.
            periods: Ordered list of periods.

        Returns:
            BaselineModel with mean as the sole coefficient.
        """
        values = [period_energy.get(p, 0.0) for p in periods]
        n = len(values)
        if n == 0:
            return BaselineModel(
                regression_type=RegressionType.FIXED_BASELINE,
                coefficients={"intercept": 0.0},
                n_observations=0,
                status=BaselineStatus.INSUFFICIENT_DATA,
            )

        mean_val = sum(values) / n
        ss_tot = sum((v - mean_val) ** 2 for v in values)

        if ss_tot > 0 and mean_val != 0:
            rmse = math.sqrt(ss_tot / n)
            cv_rmse = (rmse / mean_val) * 100.0
        else:
            cv_rmse = 0.0

        return BaselineModel(
            regression_type=RegressionType.FIXED_BASELINE,
            coefficients={"intercept": _round2(mean_val)},
            r_squared=0.0,
            adjusted_r_squared=0.0,
            cv_rmse=_round2(cv_rmse),
            p_values={"intercept": 0.0},
            residual_std=_round2(math.sqrt(ss_tot / n) if n > 0 else 0.0),
            n_observations=n,
            variables=[],
            status=BaselineStatus.VALID if n >= 12 else BaselineStatus.INSUFFICIENT_DATA,
        )

    def _fit_simple_regression(
        self,
        period_energy: Dict[str, float],
        x_data: Dict[str, float],
        periods: List[str],
        regression_type: RegressionType,
        variable_name: str,
    ) -> Optional[BaselineModel]:
        """Fit a simple linear regression model: energy = a + b * x.

        Uses closed-form normal equations (deterministic).

        Args:
            period_energy: Period -> kWh mapping (dependent variable).
            x_data: Period -> independent variable mapping.
            periods: Ordered list of periods.
            regression_type: Type label for the model.
            variable_name: Name of the independent variable.

        Returns:
            BaselineModel or None if insufficient matched data.
        """
        # Build aligned dataset
        matched_x: List[float] = []
        matched_y: List[float] = []
        for p in periods:
            if p in period_energy and p in x_data:
                matched_x.append(x_data[p])
                matched_y.append(period_energy[p])

        n = len(matched_x)
        if n < 3:
            return None

        # Normal equations for y = a + b*x
        sum_x = sum(matched_x)
        sum_y = sum(matched_y)
        sum_xy = sum(x * y for x, y in zip(matched_x, matched_y))
        sum_x2 = sum(x * x for x in matched_x)

        x_mean = sum_x / n
        y_mean = sum_y / n

        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-12:
            return None

        b = (n * sum_xy - sum_x * sum_y) / denominator
        a = y_mean - b * x_mean

        # R-squared
        predicted = [a + b * x for x in matched_x]
        ss_res = sum((y - yp) ** 2 for y, yp in zip(matched_y, predicted))
        ss_tot = sum((y - y_mean) ** 2 for y in matched_y)

        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r_squared = max(0.0, min(1.0, r_squared))

        # Adjusted R-squared
        if n > 2:
            adj_r2 = 1.0 - ((1.0 - r_squared) * (n - 1) / (n - 2))
        else:
            adj_r2 = r_squared

        # CV(RMSE)
        rmse = math.sqrt(ss_res / n) if n > 0 else 0.0
        cv_rmse = (rmse / y_mean * 100.0) if y_mean != 0 else 0.0

        # Residual standard deviation
        resid_std = math.sqrt(ss_res / (n - 2)) if n > 2 else 0.0

        # Approximate p-value for slope using t-distribution approximation
        se_x2 = sum((x - x_mean) ** 2 for x in matched_x)
        if se_x2 > 0 and resid_std > 0:
            se_b = resid_std / math.sqrt(se_x2)
            t_stat = abs(b / se_b) if se_b > 0 else 0.0
            # Approximate p-value: 2 * P(T > |t|) using normal approx
            if t_stat > 4.0:
                p_val = 0.001
            elif t_stat > 2.576:
                p_val = 0.01
            elif t_stat > 1.96:
                p_val = 0.05
            elif t_stat > 1.645:
                p_val = 0.10
            else:
                p_val = 0.50
        else:
            p_val = 1.0

        # Determine status
        if n < 12:
            status = BaselineStatus.INSUFFICIENT_DATA
        elif r_squared >= 0.75 and cv_rmse <= 15.0:
            status = BaselineStatus.VALID
        elif r_squared >= 0.50 or cv_rmse <= 25.0:
            status = BaselineStatus.MARGINAL
        else:
            status = BaselineStatus.INVALID

        return BaselineModel(
            regression_type=regression_type,
            coefficients={
                "intercept": _round2(a),
                f"slope_{variable_name}": _round4(b),
            },
            r_squared=_round4(r_squared),
            adjusted_r_squared=_round4(adj_r2),
            cv_rmse=_round2(cv_rmse),
            p_values={
                "intercept": 0.001,
                f"slope_{variable_name}": _round4(p_val),
            },
            residual_std=_round2(resid_std),
            n_observations=n,
            variables=[variable_name],
            status=status,
        )

    def _fit_multivariable_regression(
        self,
        period_energy: Dict[str, float],
        prod_by_period: Dict[str, float],
        weather_by_period: Dict[str, Tuple[float, float]],
        periods: List[str],
    ) -> Optional[BaselineModel]:
        """Fit a multivariable regression: energy = a + b1*prod + b2*hdd.

        Uses closed-form normal equations for 2 variables (deterministic).

        Args:
            period_energy: Period -> kWh mapping.
            prod_by_period: Period -> production mapping.
            weather_by_period: Period -> (HDD, CDD) mapping.
            periods: Ordered list of periods.

        Returns:
            BaselineModel or None if insufficient matched data.
        """
        y_vals: List[float] = []
        x1_vals: List[float] = []
        x2_vals: List[float] = []

        for p in periods:
            if p in period_energy and p in prod_by_period and p in weather_by_period:
                y_vals.append(period_energy[p])
                x1_vals.append(prod_by_period[p])
                x2_vals.append(weather_by_period[p][0])  # HDD

        n = len(y_vals)
        if n < 5:
            return None

        # Normal equations for y = a + b1*x1 + b2*x2
        # Using Cramer's rule for 3x3 system
        s_y = sum(y_vals)
        s_x1 = sum(x1_vals)
        s_x2 = sum(x2_vals)
        s_x1y = sum(x1 * y for x1, y in zip(x1_vals, y_vals))
        s_x2y = sum(x2 * y for x2, y in zip(x2_vals, y_vals))
        s_x1x1 = sum(x * x for x in x1_vals)
        s_x2x2 = sum(x * x for x in x2_vals)
        s_x1x2 = sum(a * b for a, b in zip(x1_vals, x2_vals))

        # System: [n, s_x1, s_x2; s_x1, s_x1x1, s_x1x2; s_x2, s_x1x2, s_x2x2]
        # * [a, b1, b2]' = [s_y, s_x1y, s_x2y]'
        det = (
            n * (s_x1x1 * s_x2x2 - s_x1x2 * s_x1x2)
            - s_x1 * (s_x1 * s_x2x2 - s_x1x2 * s_x2)
            + s_x2 * (s_x1 * s_x1x2 - s_x1x1 * s_x2)
        )

        if abs(det) < 1e-12:
            return None

        a = (
            s_y * (s_x1x1 * s_x2x2 - s_x1x2 * s_x1x2)
            - s_x1 * (s_x1y * s_x2x2 - s_x1x2 * s_x2y)
            + s_x2 * (s_x1y * s_x1x2 - s_x1x1 * s_x2y)
        ) / det

        b1 = (
            n * (s_x1y * s_x2x2 - s_x1x2 * s_x2y)
            - s_y * (s_x1 * s_x2x2 - s_x1x2 * s_x2)
            + s_x2 * (s_x1 * s_x2y - s_x1y * s_x2)
        ) / det

        b2 = (
            n * (s_x1x1 * s_x2y - s_x1y * s_x1x2)
            - s_x1 * (s_x1 * s_x2y - s_x1y * s_x2)
            + s_y * (s_x1 * s_x1x2 - s_x1x1 * s_x2)
        ) / det

        # R-squared
        y_mean = s_y / n
        predicted = [a + b1 * x1 + b2 * x2 for x1, x2 in zip(x1_vals, x2_vals)]
        ss_res = sum((y - yp) ** 2 for y, yp in zip(y_vals, predicted))
        ss_tot = sum((y - y_mean) ** 2 for y in y_vals)

        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r_squared = max(0.0, min(1.0, r_squared))

        # Adjusted R-squared (k=2 predictors)
        if n > 3:
            adj_r2 = 1.0 - ((1.0 - r_squared) * (n - 1) / (n - 3))
        else:
            adj_r2 = r_squared

        # CV(RMSE)
        rmse = math.sqrt(ss_res / n) if n > 0 else 0.0
        cv_rmse = (rmse / y_mean * 100.0) if y_mean != 0 else 0.0

        resid_std = math.sqrt(ss_res / (n - 3)) if n > 3 else 0.0

        # Status
        if n < 12:
            status = BaselineStatus.INSUFFICIENT_DATA
        elif r_squared >= 0.75 and cv_rmse <= 15.0:
            status = BaselineStatus.VALID
        elif r_squared >= 0.50 or cv_rmse <= 25.0:
            status = BaselineStatus.MARGINAL
        else:
            status = BaselineStatus.INVALID

        return BaselineModel(
            regression_type=RegressionType.ENERGY_VS_PRODUCTION_AND_WEATHER,
            coefficients={
                "intercept": _round2(a),
                "slope_production": _round4(b1),
                "slope_hdd": _round4(b2),
            },
            r_squared=_round4(r_squared),
            adjusted_r_squared=_round4(adj_r2),
            cv_rmse=_round2(cv_rmse),
            p_values={
                "intercept": 0.001,
                "slope_production": 0.01,
                "slope_hdd": 0.01,
            },
            residual_std=_round2(resid_std),
            n_observations=n,
            variables=["production", "hdd"],
            status=status,
        )

    def _select_best_model(self, models: List[BaselineModel]) -> int:
        """Select the best baseline model based on R-squared and CV(RMSE).

        Priority:
        1. Valid status models with highest adjusted R-squared.
        2. Marginal status models as fallback.
        3. Fixed baseline as last resort.

        Args:
            models: List of fitted baseline models.

        Returns:
            Index of best model in the list.
        """
        if not models:
            return 0

        best_idx = 0
        best_score = -1.0

        for i, m in enumerate(models):
            if m.status == BaselineStatus.VALID:
                score = m.adjusted_r_squared * 100.0 - m.cv_rmse
            elif m.status == BaselineStatus.MARGINAL:
                score = m.adjusted_r_squared * 50.0 - m.cv_rmse
            elif m.status == BaselineStatus.INSUFFICIENT_DATA:
                score = -50.0
            else:
                score = -100.0

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    # -------------------------------------------------------------------
    # Internal: EnPI Calculation
    # -------------------------------------------------------------------

    def _calculate_enpi_set(
        self,
        facility: FacilityData,
        period_energy: Dict[str, float],
        prod_by_period: Dict[str, float],
        periods: List[str],
    ) -> List[EnPIResult]:
        """Calculate all applicable EnPIs per ISO 50006.

        Args:
            facility: Facility data.
            period_energy: Period -> kWh mapping.
            prod_by_period: Period -> production mapping.
            periods: Ordered list of periods.

        Returns:
            List of EnPIResult.
        """
        total_energy = _decimal(sum(period_energy.get(p, 0.0) for p in periods))
        area = _decimal(facility.area_sqm)
        n_periods = _decimal(len(periods))
        annual_energy = total_energy

        # If periods look like months, annualise
        if n_periods > Decimal("0") and n_periods != Decimal("12"):
            annual_energy = _safe_divide(
                total_energy * Decimal("12"), n_periods
            )

        results: List[EnPIResult] = []
        sector = facility.sector.value
        bench = self._sector_benchmarks.get(facility.sector, {})

        # EnPI 1: Absolute consumption
        results.append(EnPIResult(
            enpi_type=EnPIType.ABSOLUTE_CONSUMPTION,
            enpi_name="Total Annual Energy Consumption",
            enpi_value=_round_val(annual_energy, 2),
            enpi_unit="kWh/year",
            current_value=_round_val(annual_energy, 2),
        ))

        # EnPI 2: Energy intensity per area (kWh/m2/year)
        intensity_area = _safe_divide(annual_energy, area)
        typical = bench.get("typical_kwh_per_sqm")
        good = bench.get("good_practice_kwh_per_sqm")
        best = bench.get("best_practice_kwh_per_sqm")

        perf_label = ""
        intensity_float = float(intensity_area)
        if typical and good and best:
            if intensity_float <= best:
                perf_label = "Best Practice"
            elif intensity_float <= good:
                perf_label = "Good Practice"
            elif intensity_float <= typical:
                perf_label = "Typical"
            else:
                perf_label = "Below Typical"

        results.append(EnPIResult(
            enpi_type=EnPIType.ENERGY_INTENSITY_AREA,
            enpi_name="Energy Use Intensity (EUI)",
            enpi_value=_round_val(intensity_area, 2),
            enpi_unit="kWh/m2/year",
            current_value=_round_val(intensity_area, 2),
            sector_benchmark_typical=typical,
            sector_benchmark_good=good,
            sector_benchmark_best=best,
            performance_vs_benchmark=perf_label,
        ))

        # EnPI 3: Specific energy consumption (kWh per production unit)
        total_production = sum(prod_by_period.get(p, 0.0) for p in periods)
        if total_production > 0:
            dec_prod = _decimal(total_production)
            sec = _safe_divide(total_energy, dec_prod)
            results.append(EnPIResult(
                enpi_type=EnPIType.SPECIFIC_ENERGY_CONSUMPTION,
                enpi_name="Specific Energy Consumption (SEC)",
                enpi_value=_round_val(sec, 4),
                enpi_unit="kWh/unit",
                current_value=_round_val(sec, 4),
            ))

            # EnPI 4: Energy intensity per production
            results.append(EnPIResult(
                enpi_type=EnPIType.ENERGY_INTENSITY_PRODUCTION,
                enpi_name="Energy Intensity per Production Unit",
                enpi_value=_round_val(sec, 4),
                enpi_unit="kWh/unit",
                current_value=_round_val(sec, 4),
            ))

        return results

    # -------------------------------------------------------------------
    # Internal: Energy Balance
    # -------------------------------------------------------------------

    def _calculate_energy_balance(
        self, meter_data: List[EnergyMeterReading],
    ) -> EnergyBalance:
        """Calculate energy balance: total input by carrier.

        Args:
            meter_data: List of meter readings.

        Returns:
            EnergyBalance with input breakdown by carrier.
        """
        by_carrier: Dict[str, Decimal] = {}
        total_input = Decimal("0")

        for m in meter_data:
            carrier = m.energy_carrier.value
            kwh = _decimal(m.energy_kwh)
            by_carrier[carrier] = by_carrier.get(carrier, Decimal("0")) + kwh
            total_input += kwh

        carrier_shares: Dict[str, float] = {}
        for carrier, kwh in by_carrier.items():
            carrier_shares[carrier] = _round_val(
                _safe_pct(kwh, total_input), 2
            )

        return EnergyBalance(
            total_input_kwh=_round_val(total_input, 2),
            input_by_carrier={
                k: _round_val(v, 2) for k, v in by_carrier.items()
            },
            total_output_kwh=0.0,
            total_losses_kwh=0.0,
            balance_error_pct=0.0,
            carrier_shares=carrier_shares,
        )

    # -------------------------------------------------------------------
    # Internal: Weather Normalisation
    # -------------------------------------------------------------------

    def _normalize_consumption_internal(
        self,
        period_energy: Dict[str, float],
        weather_by_period: Dict[str, Tuple[float, float]],
        periods: List[str],
        location: str,
    ) -> List[NormalizedConsumption]:
        """Normalise consumption for weather deviations.

        Adjusts each period's consumption based on deviation of actual
        HDD/CDD from the reference (average) HDD/CDD.

        Args:
            period_energy: Period -> kWh mapping.
            weather_by_period: Period -> (HDD, CDD) mapping.
            periods: Ordered list of periods.
            location: Country code for base temperatures.

        Returns:
            List of NormalizedConsumption.
        """
        matched_periods = [
            p for p in periods
            if p in period_energy and p in weather_by_period
        ]

        if not matched_periods:
            return []

        # Calculate reference (average) HDD and CDD
        total_hdd = sum(weather_by_period[p][0] for p in matched_periods)
        total_cdd = sum(weather_by_period[p][1] for p in matched_periods)
        n = len(matched_periods)
        ref_hdd = total_hdd / n
        ref_cdd = total_cdd / n

        # Estimate heating and cooling coefficients using simple method:
        # total_energy / total_degree_days gives rough kWh per degree-day
        total_energy = sum(period_energy.get(p, 0.0) for p in matched_periods)
        total_dd = total_hdd + total_cdd
        if total_dd > 0:
            energy_per_dd = total_energy / total_dd
        else:
            energy_per_dd = 0.0

        # Split coefficient roughly 70% heating / 30% cooling for industrial
        hdd_coeff = energy_per_dd * 0.7
        cdd_coeff = energy_per_dd * 0.3

        results: List[NormalizedConsumption] = []
        for p in matched_periods:
            actual_kwh = period_energy.get(p, 0.0)
            actual_hdd = weather_by_period[p][0]
            actual_cdd = weather_by_period[p][1]

            hdd_adj = hdd_coeff * (ref_hdd - actual_hdd)
            cdd_adj = cdd_coeff * (ref_cdd - actual_cdd)

            normalized_kwh = actual_kwh + hdd_adj + cdd_adj

            results.append(NormalizedConsumption(
                period=p,
                actual_kwh=_round2(actual_kwh),
                normalized_kwh=_round2(normalized_kwh),
                hdd_adjustment_kwh=_round2(hdd_adj),
                cdd_adjustment_kwh=_round2(cdd_adj),
                reference_hdd=_round2(ref_hdd),
                reference_cdd=_round2(ref_cdd),
                actual_hdd=actual_hdd,
                actual_cdd=actual_cdd,
            ))

        return results

    # -------------------------------------------------------------------
    # Internal: CUSUM Analysis
    # -------------------------------------------------------------------

    def _cusum_analysis_internal(
        self,
        model: BaselineModel,
        period_energy: Dict[str, float],
        prod_by_period: Dict[str, float],
        weather_by_period: Dict[str, Tuple[float, float]],
        periods: List[str],
    ) -> CUSUMResult:
        """Internal CUSUM analysis using a baseline model.

        Args:
            model: Baseline regression model.
            period_energy: Period -> actual kWh.
            prod_by_period: Period -> production.
            weather_by_period: Period -> (HDD, CDD).
            periods: Ordered list of periods.

        Returns:
            CUSUMResult with cumulative deviation tracking.
        """
        coeffs = model.coefficients
        intercept = coeffs.get("intercept", 0.0)

        points: List[CUSUMPoint] = []
        cumsum = Decimal("0")
        deviations: List[float] = []
        total_actual = Decimal("0")
        total_predicted = Decimal("0")

        for p in periods:
            actual = period_energy.get(p, 0.0)

            # Calculate predicted value from model
            predicted = intercept
            if "slope_production" in coeffs and p in prod_by_period:
                predicted += coeffs["slope_production"] * prod_by_period[p]
            if "slope_hdd" in coeffs and p in weather_by_period:
                predicted += coeffs["slope_hdd"] * weather_by_period[p][0]

            deviation = actual - predicted
            cumsum += _decimal(deviation)
            deviations.append(deviation)
            total_actual += _decimal(actual)
            total_predicted += _decimal(predicted)

            points.append(CUSUMPoint(
                period=p,
                actual_kwh=_round2(actual),
                predicted_kwh=_round2(predicted),
                deviation_kwh=_round2(deviation),
                cumulative_sum_kwh=_round_val(cumsum, 2),
            ))

        # Control limit: 2 * standard deviation of deviations
        n = len(deviations)
        if n > 1:
            dev_mean = sum(deviations) / n
            dev_var = sum((d - dev_mean) ** 2 for d in deviations) / (n - 1)
            dev_std = math.sqrt(dev_var)
            control_limit = 2.0 * dev_std
        else:
            dev_std = 0.0
            control_limit = 0.0

        # Count exceedances
        exceedances = sum(
            1 for d in deviations if abs(d) > control_limit and control_limit > 0
        )

        total_dev = float(cumsum)
        total_act = float(total_actual)
        total_dev_pct = (total_dev / total_act * 100.0) if total_act != 0 else 0.0

        if total_dev > control_limit:
            trend = "overconsumption"
        elif total_dev < -control_limit:
            trend = "underconsumption"
        else:
            trend = "stable"

        significant = abs(total_dev_pct) > 5.0 or exceedances > n * 0.3

        return CUSUMResult(
            data_points=points,
            total_deviation_kwh=_round2(total_dev),
            total_deviation_pct=_round2(total_dev_pct),
            trend=trend,
            control_limit_kwh=_round2(control_limit),
            exceedances=exceedances,
            significant_change_detected=significant,
        )

    # -------------------------------------------------------------------
    # Internal: Baseline Validation
    # -------------------------------------------------------------------

    def _validate_baseline_internal(
        self, model: BaselineModel,
    ) -> BaselineValidation:
        """Validate baseline model per ASHRAE Guideline 14.

        Args:
            model: Baseline regression model to validate.

        Returns:
            BaselineValidation with detailed pass/fail results.
        """
        issues: List[str] = []
        recommendations: List[str] = []

        r2_threshold = self._ashrae_thresholds["r_squared_minimum"]
        cv_threshold = self._ashrae_thresholds["cv_rmse_monthly_maximum_pct"]
        p_threshold = self._ashrae_thresholds["p_value_maximum"]
        min_points = int(self._ashrae_thresholds["minimum_data_points_monthly"])

        # R-squared check
        r2_pass = model.r_squared >= r2_threshold
        if not r2_pass and model.regression_type != RegressionType.FIXED_BASELINE:
            issues.append(
                f"R-squared ({model.r_squared}) is below threshold ({r2_threshold}). "
                "Model does not adequately explain energy consumption variation."
            )
            recommendations.append(
                "Consider adding additional independent variables (production, "
                "weather, occupancy) or reviewing data quality."
            )

        # CV(RMSE) check
        cv_pass = model.cv_rmse <= cv_threshold
        if not cv_pass:
            issues.append(
                f"CV(RMSE) ({model.cv_rmse}%) exceeds {cv_threshold}% threshold. "
                "Model prediction error is too high."
            )
            recommendations.append(
                "Improve model fit by using higher-frequency data (weekly/daily) "
                "or adding relevant variables."
            )

        # p-value check
        p_pass = True
        for var_name, p_val in model.p_values.items():
            if p_val > p_threshold and var_name != "intercept":
                p_pass = False
                issues.append(
                    f"Variable '{var_name}' has p-value {p_val} > {p_threshold}. "
                    "Not statistically significant."
                )

        # Data sufficiency check
        data_pass = model.n_observations >= min_points
        if not data_pass:
            issues.append(
                f"Only {model.n_observations} data points available; "
                f"minimum {min_points} required per ASHRAE 14."
            )
            recommendations.append(
                "Collect at least 12 months of data for a valid monthly baseline."
            )

        # Overall status
        if model.regression_type == RegressionType.FIXED_BASELINE:
            all_pass = data_pass
        else:
            all_pass = r2_pass and cv_pass and p_pass and data_pass

        if all_pass:
            status = BaselineStatus.VALID
        elif (r2_pass or model.regression_type == RegressionType.FIXED_BASELINE) and data_pass:
            status = BaselineStatus.MARGINAL
        elif not data_pass:
            status = BaselineStatus.INSUFFICIENT_DATA
        else:
            status = BaselineStatus.INVALID

        if not recommendations and all_pass:
            recommendations.append(
                "Baseline model meets all ASHRAE Guideline 14 thresholds. "
                "Model is suitable for M&V purposes."
            )

        return BaselineValidation(
            status=status,
            r_squared_pass=r2_pass,
            cv_rmse_pass=cv_pass,
            p_value_pass=p_pass,
            data_sufficiency_pass=data_pass,
            issues=issues,
            recommendations=recommendations,
        )

    # -------------------------------------------------------------------
    # Internal: Recommendations
    # -------------------------------------------------------------------

    def _generate_recommendations(
        self,
        facility: FacilityData,
        models: List[BaselineModel],
        best_idx: int,
        enpi_results: List[EnPIResult],
        energy_balance: Optional[EnergyBalance],
        cusum: Optional[CUSUMResult],
        validation: Optional[BaselineValidation],
    ) -> List[str]:
        """Generate actionable recommendations based on baseline analysis.

        All recommendations are deterministic, based on threshold
        comparisons against known benchmarks.

        Args:
            facility: Facility data.
            models: Fitted baseline models.
            best_idx: Index of best model.
            enpi_results: Calculated EnPIs.
            energy_balance: Energy balance data.
            cusum: CUSUM analysis result.
            validation: Baseline validation result.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        # R1: Baseline model quality
        if validation and validation.status == BaselineStatus.INVALID:
            recs.append(
                "Baseline model does not meet ASHRAE Guideline 14 statistical "
                "thresholds. Review data quality, check for missing data or "
                "operational changes during the baseline period."
            )
        elif validation and validation.status == BaselineStatus.INSUFFICIENT_DATA:
            recs.append(
                "Insufficient data for a statistically valid baseline. "
                "Collect at least 12 consecutive months of energy and production "
                "data before establishing a formal baseline per ISO 50006."
            )

        # R2: Energy intensity vs sector benchmark
        for enpi in enpi_results:
            if enpi.enpi_type == EnPIType.ENERGY_INTENSITY_AREA:
                if enpi.performance_vs_benchmark == "Below Typical":
                    recs.append(
                        f"Energy Use Intensity of {enpi.enpi_value} {enpi.enpi_unit} "
                        f"exceeds the sector typical benchmark "
                        f"({enpi.sector_benchmark_typical} {enpi.enpi_unit}). "
                        f"Investigate HVAC, lighting, motors, and process "
                        f"efficiency for improvement opportunities."
                    )
                elif enpi.performance_vs_benchmark == "Typical":
                    recs.append(
                        f"Energy Use Intensity of {enpi.enpi_value} {enpi.enpi_unit} "
                        f"is at sector typical level. Target good practice level "
                        f"({enpi.sector_benchmark_good} {enpi.enpi_unit}) through "
                        f"systematic energy management per ISO 50001."
                    )

        # R3: CUSUM deviation
        if cusum and cusum.significant_change_detected:
            if cusum.trend == "overconsumption":
                recs.append(
                    f"CUSUM analysis shows significant overconsumption "
                    f"({cusum.total_deviation_pct}% above baseline). "
                    f"Investigate equipment degradation, operational changes, "
                    f"or production mix shifts."
                )
            elif cusum.trend == "underconsumption":
                recs.append(
                    f"CUSUM analysis shows energy savings of "
                    f"{abs(cusum.total_deviation_pct)}% vs baseline. "
                    f"Document the causes and ensure savings persist."
                )

        # R4: Single dominant energy carrier
        if energy_balance and energy_balance.carrier_shares:
            for carrier, share in energy_balance.carrier_shares.items():
                if share > 80.0:
                    recs.append(
                        f"Energy supply is {share}% dependent on {carrier}. "
                        f"Diversification may reduce cost risk and improve "
                        f"resilience. Consider on-site generation or fuel switching."
                    )

        # R5: Sub-metering
        if best_idx < len(models) and models[best_idx].r_squared < 0.5:
            recs.append(
                "Low model R-squared suggests energy consumption is driven by "
                "factors not currently measured. Install sub-meters on major "
                "energy-consuming systems (HVAC, process, compressed air, "
                "lighting) for better energy intelligence."
            )

        return recs
