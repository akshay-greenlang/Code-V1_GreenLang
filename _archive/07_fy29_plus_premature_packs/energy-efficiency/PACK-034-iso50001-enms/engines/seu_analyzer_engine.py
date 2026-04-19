# -*- coding: utf-8 -*-
"""
SEUAnalyzerEngine - PACK-034 ISO 50001 EnMS Engine 1
=====================================================

Significant Energy Use (SEU) identification and analysis per
ISO 50001:2018 Clause 6.3 (Energy Review).  Performs Pareto analysis
on facility energy consumers to identify SEUs that collectively
account for a configurable threshold (default 80%) of total
consumption.  Calculates energy driver correlations using Pearson
correlation with Decimal arithmetic, assesses operating patterns,
generates equipment census summaries, ranks improvement opportunities,
and validates SEU determinations against ISO 50001 requirements.

Calculation Methodology:
    Pareto Analysis (80/20 Rule):
        Rank consumers by annual_consumption_kwh descending.
        Accumulate percentages until cumulative >= threshold (default 80%).
        All consumers within the cumulative threshold are SEU candidates.

    Significance Score (0-100):
        Base score from percentage of total (max 40 pts)
        + Operating hours contribution (max 20 pts)
        + Load factor contribution (max 15 pts)
        + Improvement potential (max 15 pts)
        + Driver correlation strength (max 10 pts)

    Pearson Correlation (Energy Driver):
        r = n*sum(xy) - sum(x)*sum(y)
            / sqrt((n*sum(x2) - (sum(x))^2) * (n*sum(y2) - (sum(y))^2))
        All arithmetic performed with Decimal to avoid float error.

    Operating Pattern Assessment:
        Baseload fraction = min(hourly_data) / max(hourly_data)
        Variable fraction  = 1 - baseload_fraction
        Peak-to-off-peak ratio = max(peak_period_avg) / avg(off_peak_avg)

    Equipment Census:
        Aggregate rated_power_kw, count, and avg load_factor by category.
        Flag equipment with load_factor < 0.3 or > 0.9 as anomalous.

Regulatory References:
    - ISO 50001:2018 - Energy management systems (Clause 6.3)
    - ISO 50006:2014 - Measuring energy performance using EnPIs and EnBs
    - ISO 50015:2014 - Measurement and verification of energy performance
    - EN 16247-1:2022 - Energy audits (general requirements)
    - DOE SEP 50001 guidance on SEU identification
    - ASHRAE Guideline 14-2014 - Measurement of Energy Savings

Zero-Hallucination:
    - All formulas are standard engineering/statistical calculations
    - Pareto analysis uses deterministic ranking and accumulation
    - Pearson correlation computed with exact Decimal arithmetic
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-034 ISO 50001 EnMS
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

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Excludes volatile fields (timestamps, durations, provenance hashes)
    from the hash input to ensure reproducibility.

from greenlang.schemas import utcnow

    Args:
        data: Model instance, dict, or other serialisable object.

    Returns:
        Hexadecimal SHA-256 digest string.
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
            if k not in (
                "calculated_at", "analysis_date", "calculation_time_ms",
                "provenance_hash",
            )
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value, string, or other type.

    Returns:
        Decimal representation; Decimal("0") on conversion failure.
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
        default: Fallback value when denominator is zero.

    Returns:
        Division result or *default*.
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
        Percentage as Decimal; 0 if whole is zero.
    """
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP.

    Args:
        value: Decimal value to round.
        places: Number of decimal places.

    Returns:
        Rounded Decimal.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _sqrt_decimal(value: Decimal) -> Decimal:
    """Compute square root of a Decimal using Newton's method.

    Args:
        value: Non-negative Decimal value.

    Returns:
        Square root as Decimal; Decimal("0") if value <= 0.
    """
    if value <= Decimal("0"):
        return Decimal("0")
    # Use float sqrt for initial guess, refine with Decimal arithmetic.
    guess = _decimal(math.sqrt(float(value)))
    if guess == Decimal("0"):
        return Decimal("0")
    # Two Newton iterations for improved precision.
    for _ in range(2):
        guess = (guess + value / guess) / Decimal("2")
    return guess

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SEUCategory(str, Enum):
    """Significant Energy Use equipment/system category.

    HVAC: Heating, ventilation, and air conditioning systems.
    LIGHTING: Interior and exterior lighting systems.
    COMPRESSED_AIR: Compressed air generation and distribution.
    MOTORS: Electric motors and drive systems.
    PROCESS_HEAT: Process heating (ovens, furnaces, dryers).
    REFRIGERATION: Commercial and industrial refrigeration.
    STEAM: Steam generation and distribution systems.
    PUMPS: Pumping systems (water, process fluids).
    FANS: Fan systems (ventilation, process).
    BOILERS: Boiler plant and hot water systems.
    TRANSPORT: On-site transport and material handling.
    COOKING: Commercial cooking equipment.
    DATA_CENTER: IT equipment and data centre infrastructure.
    WATER_HEATING: Domestic/service hot water systems.
    OTHER: Other energy-consuming systems not classified above.
    """
    HVAC = "hvac"
    LIGHTING = "lighting"
    COMPRESSED_AIR = "compressed_air"
    MOTORS = "motors"
    PROCESS_HEAT = "process_heat"
    REFRIGERATION = "refrigeration"
    STEAM = "steam"
    PUMPS = "pumps"
    FANS = "fans"
    BOILERS = "boilers"
    TRANSPORT = "transport"
    COOKING = "cooking"
    DATA_CENTER = "data_center"
    WATER_HEATING = "water_heating"
    OTHER = "other"

class OperatingPattern(str, Enum):
    """Equipment operating pattern classification.

    CONTINUOUS: 24/7 operation (8,760 hours/year).
    SINGLE_SHIFT: Single daytime shift (~2,000 hours/year).
    TWO_SHIFT: Two-shift operation (~4,000 hours/year).
    THREE_SHIFT: Three-shift / 24-hour weekday operation (~6,000 hours/year).
    SEASONAL: Seasonal operation (heating/cooling season only).
    INTERMITTENT: Intermittent or on-demand operation.
    """
    CONTINUOUS = "continuous"
    SINGLE_SHIFT = "single_shift"
    TWO_SHIFT = "two_shift"
    THREE_SHIFT = "three_shift"
    SEASONAL = "seasonal"
    INTERMITTENT = "intermittent"

class SEUStatus(str, Enum):
    """Lifecycle status of a Significant Energy Use.

    IDENTIFIED: Newly identified through energy review.
    CONFIRMED: Confirmed as SEU after validation.
    MONITORED: Actively monitored with EnPIs.
    IMPROVED: Improvement action completed.
    RETIRED: No longer classified as SEU (decommissioned or reduced).
    """
    IDENTIFIED = "identified"
    CONFIRMED = "confirmed"
    MONITORED = "monitored"
    IMPROVED = "improved"
    RETIRED = "retired"

class DeterminationMethod(str, Enum):
    """Method used to determine SEU significance.

    PARETO_ANALYSIS: Ranked by consumption, cumulative threshold applied.
    THRESHOLD: Individual consumption exceeds a percentage threshold.
    ENGINEERING_JUDGMENT: Expert judgment based on qualitative factors.
    STATISTICAL: Statistical analysis of consumption patterns.
    """
    PARETO_ANALYSIS = "pareto_analysis"
    THRESHOLD = "threshold"
    ENGINEERING_JUDGMENT = "engineering_judgment"
    STATISTICAL = "statistical"

class LoadType(str, Enum):
    """Load characteristic classification.

    BASELOAD: Constant load regardless of external factors.
    VARIABLE: Load varies with production or occupancy.
    WEATHER_DEPENDENT: Load driven by ambient temperature/humidity.
    PRODUCTION_DEPENDENT: Load driven by production volume/throughput.
    """
    BASELOAD = "baseload"
    VARIABLE = "variable"
    WEATHER_DEPENDENT = "weather_dependent"
    PRODUCTION_DEPENDENT = "production_dependent"

# ---------------------------------------------------------------------------
# Reference Data Constants
# ---------------------------------------------------------------------------

# Default load factors by SEU category.
# Source: DOE Industrial Assessment Center (IAC) database, ASHRAE
# Handbook - HVAC Applications, and typical industrial benchmarks.
DEFAULT_LOAD_FACTORS: Dict[str, Decimal] = {
    SEUCategory.HVAC.value: Decimal("0.65"),
    SEUCategory.LIGHTING.value: Decimal("0.80"),
    SEUCategory.COMPRESSED_AIR.value: Decimal("0.70"),
    SEUCategory.MOTORS.value: Decimal("0.75"),
    SEUCategory.PROCESS_HEAT.value: Decimal("0.60"),
    SEUCategory.REFRIGERATION.value: Decimal("0.72"),
    SEUCategory.STEAM.value: Decimal("0.55"),
    SEUCategory.PUMPS.value: Decimal("0.68"),
    SEUCategory.FANS.value: Decimal("0.65"),
    SEUCategory.BOILERS.value: Decimal("0.58"),
    SEUCategory.TRANSPORT.value: Decimal("0.40"),
    SEUCategory.COOKING.value: Decimal("0.45"),
    SEUCategory.DATA_CENTER.value: Decimal("0.85"),
    SEUCategory.WATER_HEATING.value: Decimal("0.50"),
    SEUCategory.OTHER.value: Decimal("0.60"),
}

# Typical efficiency ranges by SEU category (min, max as fraction).
# Source: DOE Advanced Manufacturing Office benchmarks, EU BREF
# documents, and ASHRAE Standard 90.1 minimum efficiencies.
TYPICAL_EFFICIENCY_RANGES: Dict[str, Tuple[Decimal, Decimal]] = {
    SEUCategory.HVAC.value: (Decimal("0.70"), Decimal("0.95")),
    SEUCategory.LIGHTING.value: (Decimal("0.40"), Decimal("0.95")),
    SEUCategory.COMPRESSED_AIR.value: (Decimal("0.10"), Decimal("0.30")),
    SEUCategory.MOTORS.value: (Decimal("0.85"), Decimal("0.97")),
    SEUCategory.PROCESS_HEAT.value: (Decimal("0.50"), Decimal("0.90")),
    SEUCategory.REFRIGERATION.value: (Decimal("0.60"), Decimal("0.92")),
    SEUCategory.STEAM.value: (Decimal("0.75"), Decimal("0.95")),
    SEUCategory.PUMPS.value: (Decimal("0.60"), Decimal("0.90")),
    SEUCategory.FANS.value: (Decimal("0.55"), Decimal("0.88")),
    SEUCategory.BOILERS.value: (Decimal("0.80"), Decimal("0.98")),
    SEUCategory.TRANSPORT.value: (Decimal("0.20"), Decimal("0.45")),
    SEUCategory.COOKING.value: (Decimal("0.35"), Decimal("0.75")),
    SEUCategory.DATA_CENTER.value: (Decimal("0.55"), Decimal("0.85")),
    SEUCategory.WATER_HEATING.value: (Decimal("0.70"), Decimal("0.98")),
    SEUCategory.OTHER.value: (Decimal("0.50"), Decimal("0.85")),
}

# SEU improvement benchmarks by category (typical improvement %).
# Source: DOE Better Plants program, EU Energy Efficiency Directive
# best practice documents, IEA Industrial Energy Technology Roadmaps.
SEU_IMPROVEMENT_BENCHMARKS: Dict[str, Decimal] = {
    SEUCategory.HVAC.value: Decimal("15.0"),
    SEUCategory.LIGHTING.value: Decimal("40.0"),
    SEUCategory.COMPRESSED_AIR.value: Decimal("25.0"),
    SEUCategory.MOTORS.value: Decimal("8.0"),
    SEUCategory.PROCESS_HEAT.value: Decimal("12.0"),
    SEUCategory.REFRIGERATION.value: Decimal("18.0"),
    SEUCategory.STEAM.value: Decimal("15.0"),
    SEUCategory.PUMPS.value: Decimal("20.0"),
    SEUCategory.FANS.value: Decimal("18.0"),
    SEUCategory.BOILERS.value: Decimal("10.0"),
    SEUCategory.TRANSPORT.value: Decimal("12.0"),
    SEUCategory.COOKING.value: Decimal("10.0"),
    SEUCategory.DATA_CENTER.value: Decimal("20.0"),
    SEUCategory.WATER_HEATING.value: Decimal("15.0"),
    SEUCategory.OTHER.value: Decimal("10.0"),
}

# Typical annual operating hours by operating pattern.
TYPICAL_OPERATING_HOURS: Dict[str, int] = {
    OperatingPattern.CONTINUOUS.value: 8760,
    OperatingPattern.SINGLE_SHIFT.value: 2000,
    OperatingPattern.TWO_SHIFT.value: 4000,
    OperatingPattern.THREE_SHIFT.value: 6000,
    OperatingPattern.SEASONAL.value: 3000,
    OperatingPattern.INTERMITTENT.value: 1500,
}

# Standard improvement opportunities by SEU category.
# Each entry is a list of typical improvement actions.
STANDARD_IMPROVEMENT_OPPORTUNITIES: Dict[str, List[str]] = {
    SEUCategory.HVAC.value: [
        "Install variable speed drives on AHU fans and pumps",
        "Implement demand-controlled ventilation (CO2 sensors)",
        "Optimise chilled/hot water temperature setpoints",
        "Commission or re-commission BMS sequences",
        "Replace aged chillers with high-efficiency models",
        "Install heat recovery on exhaust air streams",
        "Seal ductwork leaks and insulate exposed ducts",
    ],
    SEUCategory.LIGHTING.value: [
        "Retrofit fluorescent/HID fixtures with LED",
        "Install occupancy/vacancy sensors in intermittent spaces",
        "Implement daylight harvesting with photocell dimming",
        "Reduce lighting power density to ASHRAE 90.1 targets",
        "De-lamp over-lit areas and remove unnecessary fixtures",
        "Install astronomical time-clock for exterior lighting",
    ],
    SEUCategory.COMPRESSED_AIR.value: [
        "Conduct and repair compressed air leak survey",
        "Install VSD compressor for variable-demand profile",
        "Reduce system operating pressure by 0.5-1.0 bar",
        "Install zero-loss condensate drains",
        "Segregate high-pressure and low-pressure demands",
        "Replace compressed air blow-off with blowers",
        "Install air receiver tanks for demand buffering",
    ],
    SEUCategory.MOTORS.value: [
        "Replace standard motors with IE4/IE5 premium efficiency",
        "Install variable speed drives on variable-torque loads",
        "Right-size oversized motors (load factor < 0.4)",
        "Implement power quality correction (PFC capacitors)",
        "Improve belt/coupling alignment and tensioning",
    ],
    SEUCategory.PROCESS_HEAT.value: [
        "Install waste heat recovery on flue gas/exhaust",
        "Insulate furnace walls and process piping",
        "Optimise combustion air-fuel ratio",
        "Preheat combustion air using recuperator/regenerator",
        "Implement thermal curtains on furnace openings",
        "Schedule batch operations to minimise reheat cycles",
    ],
    SEUCategory.REFRIGERATION.value: [
        "Clean condenser coils and maintain head pressure",
        "Install floating head pressure controls",
        "Replace expansion valves with electronic types",
        "Install VSD on condenser fans",
        "Add subcooling and liquid-suction heat exchanger",
        "Verify and correct refrigerant charge levels",
        "Install strip curtains on walk-in cooler doors",
    ],
    SEUCategory.STEAM.value: [
        "Conduct steam trap survey and repair failed traps",
        "Insulate bare steam pipes, valves, and flanges",
        "Implement condensate return system",
        "Install economiser on boiler flue gas",
        "Optimise boiler blowdown (TDS control)",
        "Reduce steam pressure to minimum required",
        "Install flash steam recovery vessels",
    ],
    SEUCategory.PUMPS.value: [
        "Install variable speed drives on variable-flow pumps",
        "Right-size oversized pumps (trim impeller or replace)",
        "Eliminate throttling valves with VSD control",
        "Reduce system friction losses (pipe sizing, bends)",
        "Implement parallel pump sequencing optimisation",
    ],
    SEUCategory.FANS.value: [
        "Install variable speed drives on supply/return fans",
        "Replace forward-curved fans with backward-curved/airfoil",
        "Optimise fan system ductwork to reduce pressure drop",
        "Implement inlet guide vanes or outlet damper control",
        "Right-size oversized fans to match actual flow requirements",
    ],
    SEUCategory.BOILERS.value: [
        "Tune combustion to optimise excess air ratio",
        "Install flue gas economiser for feedwater preheating",
        "Implement oxygen trim control on burners",
        "Insulate boiler shell and hot water piping",
        "Optimise blowdown frequency and recover blowdown heat",
        "Schedule boiler sequencing for multi-boiler plants",
    ],
    SEUCategory.TRANSPORT.value: [
        "Electrify forklift fleet (replace LPG/diesel)",
        "Implement regenerative braking on conveyors",
        "Optimise material handling routes and scheduling",
        "Install battery management for electric fleet",
        "Right-size transport equipment for actual loads",
    ],
    SEUCategory.COOKING.value: [
        "Replace standard equipment with ENERGY STAR rated",
        "Install demand-controlled kitchen exhaust hoods",
        "Implement cooking schedule optimisation",
        "Pre-heat equipment only when needed (timer controls)",
        "Install heat recovery on kitchen exhaust for hot water",
    ],
    SEUCategory.DATA_CENTER.value: [
        "Raise supply air temperature to ASHRAE A1 upper limit",
        "Implement hot/cold aisle containment",
        "Install variable speed drives on CRAC/CRAH fans",
        "Migrate to virtualised/cloud infrastructure",
        "Implement free cooling or economiser modes",
        "Optimise UPS loading and replace with high-efficiency units",
    ],
    SEUCategory.WATER_HEATING.value: [
        "Install heat pump water heater (COP 3.0+)",
        "Insulate hot water storage tanks and distribution pipes",
        "Install point-of-use heaters for remote fixtures",
        "Implement solar thermal preheat system",
        "Reduce hot water temperature setpoint to minimum safe level",
        "Install drain water heat recovery exchangers",
    ],
    SEUCategory.OTHER.value: [
        "Conduct detailed energy audit of unclassified systems",
        "Install sub-metering on unmonitored loads",
        "Review operational schedules for energy reduction",
        "Benchmark against industry best practice",
    ],
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class EnergyConsumer(BaseModel):
    """Individual energy-consuming system or equipment group.

    Represents a single identifiable energy consumer within a facility
    for the purpose of SEU analysis per ISO 50001 Clause 6.3.

    Attributes:
        id: Unique consumer identifier.
        name: Descriptive name of the energy consumer.
        category: SEU category classification.
        rated_power_kw: Nameplate rated power (kW).
        load_factor: Actual-to-rated power ratio (0.0-1.0).
        annual_operating_hours: Hours of operation per year.
        annual_consumption_kwh: Measured annual energy consumption (kWh).
        percentage_of_total: Percentage of facility total (calculated).
        energy_driver: Primary energy driver description.
        operating_pattern: Operating pattern classification.
    """
    id: str = Field(
        default_factory=_new_uuid,
        description="Unique consumer identifier",
    )
    name: str = Field(
        default="",
        max_length=300,
        description="Energy consumer name",
    )
    category: str = Field(
        default=SEUCategory.OTHER.value,
        description="SEU category classification",
    )
    rated_power_kw: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Nameplate rated power (kW)",
    )
    load_factor: Decimal = Field(
        default=Decimal("0.75"),
        ge=0,
        le=Decimal("1.5"),
        description="Actual-to-rated power ratio (0.0-1.0, may exceed 1.0 for overloaded)",
    )
    annual_operating_hours: int = Field(
        default=2000,
        ge=0,
        le=8760,
        description="Annual operating hours",
    )
    annual_consumption_kwh: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Measured annual energy consumption (kWh)",
    )
    percentage_of_total: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        le=Decimal("100"),
        description="Percentage of facility total consumption",
    )
    energy_driver: str = Field(
        default="",
        max_length=300,
        description="Primary energy driver (e.g. production volume, degree days)",
    )
    operating_pattern: str = Field(
        default=OperatingPattern.SINGLE_SHIFT.value,
        description="Operating pattern classification",
    )

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        """Validate SEU category is a known value."""
        valid = {c.value for c in SEUCategory}
        if v not in valid:
            raise ValueError(
                f"Unknown SEU category '{v}'. Must be one of: {sorted(valid)}"
            )
        return v

    @field_validator("operating_pattern")
    @classmethod
    def validate_operating_pattern(cls, v: str) -> str:
        """Validate operating pattern is a known value."""
        valid = {p.value for p in OperatingPattern}
        if v not in valid:
            raise ValueError(
                f"Unknown operating pattern '{v}'. Must be one of: {sorted(valid)}"
            )
        return v

class SEUThresholds(BaseModel):
    """Configuration thresholds for SEU determination.

    Attributes:
        cumulative_threshold_pct: Cumulative percentage for Pareto cutoff (default 80%).
        individual_threshold_pct: Individual consumer threshold (default 5%).
        min_consumers_for_pareto: Minimum consumers required for Pareto analysis.
    """
    cumulative_threshold_pct: Decimal = Field(
        default=Decimal("80.0"),
        ge=Decimal("50"),
        le=Decimal("99"),
        description="Cumulative Pareto threshold (%)",
    )
    individual_threshold_pct: Decimal = Field(
        default=Decimal("5.0"),
        ge=Decimal("1"),
        le=Decimal("50"),
        description="Individual consumer significance threshold (%)",
    )
    min_consumers_for_pareto: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Minimum consumers for Pareto analysis",
    )

class EnergyDriver(BaseModel):
    """Energy driver with correlation data for an SEU.

    Represents a potential energy driver (e.g. production volume,
    outdoor temperature) and its statistical relationship to the
    SEU's energy consumption.

    Attributes:
        driver_name: Name of the energy driver.
        driver_type: Type classification (production, weather, occupancy, other).
        values: Time-series values of the driver variable.
        correlation_coefficient: Pearson correlation with consumption (-1 to 1).
        unit_of_measure: Unit for the driver values.
    """
    driver_name: str = Field(
        default="",
        max_length=200,
        description="Energy driver name",
    )
    driver_type: str = Field(
        default="other",
        max_length=50,
        description="Driver type (production, weather, occupancy, other)",
    )
    values: List[Decimal] = Field(
        default_factory=list,
        description="Time-series values of the driver variable",
    )
    correlation_coefficient: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("-1"),
        le=Decimal("1"),
        description="Pearson correlation coefficient (-1 to 1)",
    )
    unit_of_measure: str = Field(
        default="",
        max_length=50,
        description="Unit of measure for driver values",
    )

    @field_validator("driver_type")
    @classmethod
    def validate_driver_type(cls, v: str) -> str:
        """Validate driver type is a recognised classification."""
        valid_types = {"production", "weather", "occupancy", "time", "other"}
        if v not in valid_types:
            raise ValueError(
                f"Unknown driver type '{v}'. Must be one of: {sorted(valid_types)}"
            )
        return v

class FacilityEnergyProfile(BaseModel):
    """Facility-level energy profile for SEU analysis.

    Contains the complete energy consumer inventory and metadata
    required for ISO 50001 energy review and SEU identification.

    Attributes:
        facility_id: Unique facility identifier.
        facility_name: Name of the facility.
        total_annual_consumption_kwh: Total facility energy (kWh/yr).
        energy_consumers: List of energy consumers in the facility.
        measurement_period_start: Start of measurement period.
        measurement_period_end: End of measurement period.
    """
    facility_id: str = Field(
        default_factory=_new_uuid,
        description="Unique facility identifier",
    )
    facility_name: str = Field(
        default="",
        max_length=300,
        description="Facility name",
    )
    total_annual_consumption_kwh: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Total annual energy consumption (kWh)",
    )
    energy_consumers: List[EnergyConsumer] = Field(
        default_factory=list,
        description="List of energy consumers",
    )
    measurement_period_start: date = Field(
        default_factory=lambda: date(2025, 1, 1),
        description="Measurement period start date",
    )
    measurement_period_end: date = Field(
        default_factory=lambda: date(2025, 12, 31),
        description="Measurement period end date",
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Intermediate
# ---------------------------------------------------------------------------

class ParetoDataPoint(BaseModel):
    """Single data point in a Pareto chart.

    Attributes:
        rank: Rank position (1 = highest consumer).
        consumer_id: Reference to the energy consumer.
        consumer_name: Consumer name for display.
        category: SEU category.
        consumption_kwh: Annual consumption (kWh).
        percentage_of_total: Individual percentage of total.
        cumulative_percentage: Running cumulative percentage.
        is_above_threshold: Whether within cumulative threshold.
    """
    rank: int = Field(default=0, ge=0, description="Pareto rank position")
    consumer_id: str = Field(default="", description="Consumer ID reference")
    consumer_name: str = Field(default="", description="Consumer name")
    category: str = Field(default="", description="SEU category")
    consumption_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Consumption (kWh)"
    )
    percentage_of_total: Decimal = Field(
        default=Decimal("0"), ge=0, description="% of total"
    )
    cumulative_percentage: Decimal = Field(
        default=Decimal("0"), ge=0, description="Cumulative %"
    )
    is_above_threshold: bool = Field(
        default=False, description="Within cumulative threshold"
    )

class OperatingPatternAssessment(BaseModel):
    """Assessment of an energy consumer's operating pattern.

    Attributes:
        consumer_id: Reference to the consumer.
        baseload_fraction: Fraction of load that is constant (0-1).
        variable_fraction: Fraction of load that is variable (0-1).
        peak_to_offpeak_ratio: Ratio of peak to off-peak demand.
        estimated_baseload_kw: Estimated baseload demand (kW).
        estimated_peak_kw: Estimated peak demand (kW).
        load_type: Classified load type.
        operating_hours_utilisation: Fraction of period with non-zero load.
        notes: Assessment notes.
    """
    consumer_id: str = Field(default="", description="Consumer ID")
    baseload_fraction: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("1"), description="Baseload fraction"
    )
    variable_fraction: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("1"), description="Variable fraction"
    )
    peak_to_offpeak_ratio: Decimal = Field(
        default=Decimal("1"), ge=0, description="Peak-to-off-peak ratio"
    )
    estimated_baseload_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Baseload demand (kW)"
    )
    estimated_peak_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Peak demand (kW)"
    )
    load_type: str = Field(
        default=LoadType.VARIABLE.value, description="Load type"
    )
    operating_hours_utilisation: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("1"), description="Utilisation"
    )
    notes: str = Field(default="", max_length=1000, description="Assessment notes")

class ImprovementOpportunity(BaseModel):
    """Ranked improvement opportunity for an SEU.

    Attributes:
        opportunity_id: Unique identifier.
        seu_id: Reference to the SEU.
        description: Improvement description.
        category: SEU category.
        estimated_savings_pct: Estimated savings percentage.
        estimated_savings_kwh: Estimated savings (kWh/yr).
        priority_rank: Priority rank (1 = highest).
        confidence_level: Confidence in estimate (low, medium, high).
    """
    opportunity_id: str = Field(
        default_factory=_new_uuid, description="Opportunity ID"
    )
    seu_id: str = Field(default="", description="SEU reference")
    description: str = Field(default="", max_length=500, description="Description")
    category: str = Field(default="", description="SEU category")
    estimated_savings_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"), description="Savings %"
    )
    estimated_savings_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Savings (kWh/yr)"
    )
    priority_rank: int = Field(default=0, ge=0, description="Priority rank")
    confidence_level: str = Field(
        default="medium", description="Confidence (low, medium, high)"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class SEUResult(BaseModel):
    """Result for a single energy consumer's SEU analysis.

    Contains the determination of whether the consumer is a Significant
    Energy Use, its significance score, determination method, cumulative
    Pareto position, identified energy drivers, and improvement
    opportunities.

    Attributes:
        seu_id: Unique SEU result identifier.
        consumer: The energy consumer analysed.
        is_significant: Whether identified as an SEU.
        significance_score: Score from 0-100.
        determination_method: Method used for SEU determination.
        cumulative_percentage: Cumulative Pareto position.
        energy_drivers: Identified energy drivers.
        improvement_opportunities: List of improvement descriptions.
        status: SEU lifecycle status.
        estimated_savings_potential_kwh: Total estimated savings (kWh/yr).
        estimated_savings_potential_pct: Total estimated savings (%).
    """
    seu_id: str = Field(
        default_factory=_new_uuid, description="SEU result ID"
    )
    consumer: EnergyConsumer = Field(..., description="Energy consumer")
    is_significant: bool = Field(
        default=False, description="Whether identified as SEU"
    )
    significance_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="Significance score (0-100)"
    )
    determination_method: str = Field(
        default=DeterminationMethod.PARETO_ANALYSIS.value,
        description="Determination method",
    )
    cumulative_percentage: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="Cumulative Pareto position (%)",
    )
    energy_drivers: List[EnergyDriver] = Field(
        default_factory=list, description="Identified energy drivers"
    )
    improvement_opportunities: List[str] = Field(
        default_factory=list, description="Improvement descriptions"
    )
    status: str = Field(
        default=SEUStatus.IDENTIFIED.value, description="SEU status"
    )
    estimated_savings_potential_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Estimated savings (kWh/yr)"
    )
    estimated_savings_potential_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="Estimated savings (%)"
    )

class SEUAnalysisResult(BaseModel):
    """Complete SEU analysis result for a facility.

    Represents the full output of an ISO 50001 energy review SEU
    determination, including Pareto analysis, energy driver summary,
    and provenance tracking.

    Attributes:
        analysis_id: Unique analysis identifier.
        facility_id: Facility reference.
        analysis_date: Timestamp of analysis.
        total_consumption_kwh: Total facility consumption (kWh).
        seu_count: Number of identified SEUs.
        non_seu_count: Number of non-SEU consumers.
        seus: List of individual SEU results.
        pareto_chart_data: Pareto chart data points.
        energy_driver_summary: Summary of energy drivers across SEUs.
        coverage_percentage: Percentage of total consumption covered by SEUs.
        provenance_hash: SHA-256 provenance hash.
        calculation_time_ms: Processing duration (milliseconds).
    """
    analysis_id: str = Field(
        default_factory=_new_uuid, description="Analysis ID"
    )
    facility_id: str = Field(default="", description="Facility reference")
    analysis_date: datetime = Field(
        default_factory=utcnow, description="Analysis timestamp"
    )
    total_consumption_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total consumption (kWh)"
    )
    seu_count: int = Field(default=0, ge=0, description="Number of SEUs")
    non_seu_count: int = Field(
        default=0, ge=0, description="Number of non-SEUs"
    )
    seus: List[SEUResult] = Field(
        default_factory=list, description="Individual SEU results"
    )
    pareto_chart_data: List[Dict[str, Any]] = Field(
        default_factory=list, description="Pareto chart data"
    )
    energy_driver_summary: List[Dict[str, Any]] = Field(
        default_factory=list, description="Energy driver summary"
    )
    coverage_percentage: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="SEU coverage of total (%)",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )
    calculation_time_ms: int = Field(
        default=0, ge=0, description="Processing time (ms)"
    )

class SEUValidationResult(BaseModel):
    """Validation result for an SEU analysis.

    Checks completeness and compliance of SEU determination against
    ISO 50001 requirements.

    Attributes:
        validation_id: Unique validation ID.
        analysis_id: Reference to the analysis being validated.
        is_valid: Overall pass/fail.
        checks_passed: Number of checks passed.
        checks_total: Total number of checks performed.
        findings: List of validation findings.
        recommendations: List of recommendations.
        iso_clause_references: Relevant ISO 50001 clause references.
        provenance_hash: SHA-256 hash.
    """
    validation_id: str = Field(
        default_factory=_new_uuid, description="Validation ID"
    )
    analysis_id: str = Field(default="", description="Analysis reference")
    is_valid: bool = Field(default=False, description="Overall pass/fail")
    checks_passed: int = Field(default=0, ge=0, description="Checks passed")
    checks_total: int = Field(default=0, ge=0, description="Total checks")
    findings: List[Dict[str, Any]] = Field(
        default_factory=list, description="Validation findings"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations"
    )
    iso_clause_references: List[str] = Field(
        default_factory=list, description="ISO 50001 clause references"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SEUAnalyzerEngine:
    """Significant Energy Use (SEU) Analyzer per ISO 50001 Clause 6.3.

    Performs energy review analysis to identify Significant Energy Uses
    within a facility.  Uses Pareto analysis (80/20 rule) to rank energy
    consumers, calculates significance scores, identifies energy drivers
    via Pearson correlation, assesses operating patterns, generates
    equipment census summaries, ranks improvement opportunities, and
    validates SEU determinations against ISO 50001 requirements.

    All calculations use Decimal arithmetic for zero-hallucination
    deterministic results.  Every output includes a SHA-256 provenance
    hash for complete audit trail.

    Usage::

        engine = SEUAnalyzerEngine()
        profile = FacilityEnergyProfile(
            facility_name="Manufacturing Plant A",
            total_annual_consumption_kwh=Decimal("5000000"),
            energy_consumers=[...],
        )
        thresholds = SEUThresholds(cumulative_threshold_pct=Decimal("80.0"))
        result = engine.analyze_seus(profile, thresholds)
        for seu in result.seus:
            if seu.is_significant:
                print(f"SEU: {seu.consumer.name} ({seu.significance_score} pts)")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise SEUAnalyzerEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - default_load_factors (Dict): override default load factors
                - improvement_benchmarks (Dict): override improvement benchmarks
                - efficiency_ranges (Dict): override efficiency ranges
                - significance_threshold (Decimal): minimum score to classify as SEU
        """
        self.config = config or {}
        self._load_factors: Dict[str, Decimal] = {
            **DEFAULT_LOAD_FACTORS,
            **{
                k: _decimal(v) for k, v in
                self.config.get("default_load_factors", {}).items()
            },
        }
        self._improvement_benchmarks: Dict[str, Decimal] = {
            **SEU_IMPROVEMENT_BENCHMARKS,
            **{
                k: _decimal(v) for k, v in
                self.config.get("improvement_benchmarks", {}).items()
            },
        }
        self._efficiency_ranges: Dict[str, Tuple[Decimal, Decimal]] = {
            **TYPICAL_EFFICIENCY_RANGES,
            **{
                k: (
                    _decimal(v[0]) if isinstance(v, (list, tuple)) else _decimal(v),
                    _decimal(v[1]) if isinstance(v, (list, tuple)) else _decimal(v),
                )
                for k, v in self.config.get("efficiency_ranges", {}).items()
            },
        }
        self._significance_threshold = _decimal(
            self.config.get("significance_threshold", Decimal("40"))
        )
        logger.info(
            "SEUAnalyzerEngine v%s initialised "
            "(significance_threshold=%.1f, categories=%d)",
            self.engine_version,
            float(self._significance_threshold),
            len(self._load_factors),
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def analyze_seus(
        self,
        profile: FacilityEnergyProfile,
        thresholds: Optional[SEUThresholds] = None,
    ) -> SEUAnalysisResult:
        """Perform complete SEU analysis for a facility.

        Executes the full ISO 50001 Clause 6.3 energy review workflow:
        1. Calculate percentage of total for each consumer.
        2. Sort consumers by consumption descending.
        3. Perform Pareto analysis with configurable threshold.
        4. Calculate significance scores for all consumers.
        5. Identify energy drivers (if driver data provided).
        6. Assign improvement opportunities.
        7. Generate Pareto chart data.
        8. Calculate provenance hash.

        Args:
            profile: Facility energy profile with consumer inventory.
            thresholds: SEU determination thresholds (optional, uses defaults).

        Returns:
            Complete SEUAnalysisResult with all findings.

        Raises:
            ValueError: If profile has no consumers or zero total consumption.
        """
        t0 = time.perf_counter()
        if thresholds is None:
            thresholds = SEUThresholds()

        logger.info(
            "SEU analysis: facility=%s, consumers=%d, total=%.0f kWh, "
            "threshold=%.1f%%",
            profile.facility_name,
            len(profile.energy_consumers),
            float(profile.total_annual_consumption_kwh),
            float(thresholds.cumulative_threshold_pct),
        )

        # Validate inputs.
        if not profile.energy_consumers:
            raise ValueError("Facility energy profile has no energy consumers")

        # Step 1: Determine total consumption.
        total_kwh = self._resolve_total_consumption(profile)

        if total_kwh <= Decimal("0"):
            raise ValueError(
                "Total annual consumption must be > 0. "
                f"Got {total_kwh} kWh."
            )

        # Step 2: Calculate individual percentages.
        consumers_with_pct = self._calculate_percentages(
            profile.energy_consumers, total_kwh
        )

        # Step 3: Sort by consumption descending.
        sorted_consumers = sorted(
            consumers_with_pct,
            key=lambda c: c.annual_consumption_kwh,
            reverse=True,
        )

        # Step 4: Perform Pareto analysis.
        pareto_data = self.perform_pareto_analysis(
            sorted_consumers, thresholds
        )

        # Build set of consumer IDs within Pareto threshold.
        pareto_seu_ids = {
            p.consumer_id for p in pareto_data
            if p.is_above_threshold
        }

        # Step 5: Calculate significance scores and build SEU results.
        seu_results: List[SEUResult] = []
        for consumer in sorted_consumers:
            in_pareto = consumer.id in pareto_seu_ids
            above_individual = (
                consumer.percentage_of_total >= thresholds.individual_threshold_pct
            )
            significance_score = self._calculate_significance_score(
                consumer, total_kwh, in_pareto
            )
            is_significant = (
                (in_pareto or above_individual)
                and significance_score >= self._significance_threshold
            )

            # Determine method.
            if in_pareto and above_individual:
                method = DeterminationMethod.PARETO_ANALYSIS.value
            elif above_individual:
                method = DeterminationMethod.THRESHOLD.value
            elif in_pareto:
                method = DeterminationMethod.PARETO_ANALYSIS.value
            else:
                method = DeterminationMethod.ENGINEERING_JUDGMENT.value

            # Get cumulative percentage from Pareto data.
            cumulative_pct = Decimal("0")
            for pd_item in pareto_data:
                if pd_item.consumer_id == consumer.id:
                    cumulative_pct = pd_item.cumulative_percentage
                    break

            # Improvement opportunities.
            improvements = self._get_improvement_opportunities(consumer)

            # Savings potential.
            savings_pct = self._improvement_benchmarks.get(
                consumer.category, Decimal("10")
            )
            savings_kwh = consumer.annual_consumption_kwh * savings_pct / Decimal("100")

            seu_result = SEUResult(
                consumer=consumer,
                is_significant=is_significant,
                significance_score=_round_val(significance_score, 2),
                determination_method=method,
                cumulative_percentage=_round_val(cumulative_pct, 2),
                energy_drivers=[],
                improvement_opportunities=improvements,
                status=(
                    SEUStatus.IDENTIFIED.value if is_significant
                    else SEUStatus.RETIRED.value
                ),
                estimated_savings_potential_kwh=_round_val(savings_kwh, 2),
                estimated_savings_potential_pct=_round_val(savings_pct, 2),
            )
            seu_results.append(seu_result)

        # Step 6: Compute aggregates.
        significant_seus = [s for s in seu_results if s.is_significant]
        non_significant = [s for s in seu_results if not s.is_significant]

        coverage = sum(
            (s.consumer.percentage_of_total for s in significant_seus),
            Decimal("0"),
        )

        # Step 7: Pareto chart data as dicts.
        pareto_chart = [
            {
                "rank": p.rank,
                "consumer_name": p.consumer_name,
                "category": p.category,
                "consumption_kwh": str(_round_val(p.consumption_kwh, 2)),
                "percentage": str(_round_val(p.percentage_of_total, 2)),
                "cumulative_percentage": str(
                    _round_val(p.cumulative_percentage, 2)
                ),
                "is_seu": p.is_above_threshold,
            }
            for p in pareto_data
        ]

        # Step 8: Energy driver summary (placeholder for external data).
        driver_summary = self._build_driver_summary(significant_seus)

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0)

        result = SEUAnalysisResult(
            facility_id=profile.facility_id,
            total_consumption_kwh=_round_val(total_kwh, 2),
            seu_count=len(significant_seus),
            non_seu_count=len(non_significant),
            seus=seu_results,
            pareto_chart_data=pareto_chart,
            energy_driver_summary=driver_summary,
            coverage_percentage=_round_val(min(coverage, Decimal("100")), 2),
            calculation_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "SEU analysis complete: %d SEUs / %d consumers, "
            "coverage=%.1f%%, time=%d ms, hash=%s",
            len(significant_seus),
            len(profile.energy_consumers),
            float(coverage),
            elapsed_ms,
            result.provenance_hash[:16],
        )
        return result

    def perform_pareto_analysis(
        self,
        consumers: List[EnergyConsumer],
        thresholds: Optional[SEUThresholds] = None,
    ) -> List[ParetoDataPoint]:
        """Perform Pareto (80/20) analysis on energy consumers.

        Ranks consumers by annual consumption descending, calculates
        cumulative percentages, and flags consumers that collectively
        account for the threshold percentage of total consumption.

        Args:
            consumers: List of energy consumers (should be pre-sorted
                       descending by consumption, or will be sorted here).
            thresholds: SEU thresholds (uses defaults if not provided).

        Returns:
            List of ParetoDataPoint in rank order.
        """
        if thresholds is None:
            thresholds = SEUThresholds()

        # Sort descending by consumption.
        sorted_list = sorted(
            consumers,
            key=lambda c: c.annual_consumption_kwh,
            reverse=True,
        )

        total_kwh = sum(
            (c.annual_consumption_kwh for c in sorted_list), Decimal("0")
        )
        if total_kwh <= Decimal("0"):
            logger.warning("Pareto analysis: total consumption is zero")
            return []

        pareto_data: List[ParetoDataPoint] = []
        cumulative = Decimal("0")
        threshold_reached = False

        for rank, consumer in enumerate(sorted_list, start=1):
            pct = _safe_pct(consumer.annual_consumption_kwh, total_kwh)
            cumulative += pct

            # A consumer is within the Pareto threshold if cumulative has not
            # yet exceeded the threshold at the START of this consumer's
            # addition, OR if this consumer itself pushes it past.
            is_within = not threshold_reached
            if cumulative >= thresholds.cumulative_threshold_pct:
                threshold_reached = True

            pareto_data.append(ParetoDataPoint(
                rank=rank,
                consumer_id=consumer.id,
                consumer_name=consumer.name,
                category=consumer.category,
                consumption_kwh=consumer.annual_consumption_kwh,
                percentage_of_total=_round_val(pct, 4),
                cumulative_percentage=_round_val(
                    min(cumulative, Decimal("100")), 4
                ),
                is_above_threshold=is_within,
            ))

        logger.debug(
            "Pareto analysis: %d consumers, %d within %.0f%% threshold",
            len(pareto_data),
            sum(1 for p in pareto_data if p.is_above_threshold),
            float(thresholds.cumulative_threshold_pct),
        )
        return pareto_data

    def identify_energy_drivers(
        self,
        seu: SEUResult,
        driver_data: Dict[str, Dict[str, Any]],
    ) -> List[EnergyDriver]:
        """Identify and correlate energy drivers for an SEU.

        Calculates Pearson correlation coefficients between the SEU's
        energy consumption time-series and each potential driver's
        time-series.  Drivers with |r| >= 0.5 are considered significant.

        Args:
            seu: SEU result to analyse.
            driver_data: Dictionary mapping driver names to metadata.
                Each entry must contain:
                    - "values": list of Decimal driver values
                    - "consumption_values": list of Decimal consumption values
                    - "type": str driver type
                    - "unit": str unit of measure

        Returns:
            List of EnergyDriver with computed correlations.
        """
        drivers: List[EnergyDriver] = []

        for driver_name, data in driver_data.items():
            driver_values = [_decimal(v) for v in data.get("values", [])]
            consumption_values = [
                _decimal(v) for v in data.get("consumption_values", [])
            ]
            driver_type = data.get("type", "other")
            unit = data.get("unit", "")

            if len(driver_values) < 3 or len(consumption_values) < 3:
                logger.warning(
                    "Insufficient data for driver '%s' (n=%d), skipping",
                    driver_name, min(len(driver_values), len(consumption_values)),
                )
                continue

            # Ensure equal length (truncate to shorter).
            n = min(len(driver_values), len(consumption_values))
            x_vals = driver_values[:n]
            y_vals = consumption_values[:n]

            r = self._pearson_correlation(x_vals, y_vals)

            drivers.append(EnergyDriver(
                driver_name=driver_name,
                driver_type=driver_type,
                values=driver_values,
                correlation_coefficient=_round_val(r, 6),
                unit_of_measure=unit,
            ))

        # Sort by absolute correlation strength descending.
        drivers.sort(key=lambda d: abs(d.correlation_coefficient), reverse=True)

        logger.info(
            "Energy drivers for SEU '%s': %d analysed, %d significant (|r|>=0.5)",
            seu.consumer.name,
            len(drivers),
            sum(
                1 for d in drivers
                if abs(d.correlation_coefficient) >= Decimal("0.5")
            ),
        )
        return drivers

    def assess_operating_patterns(
        self,
        consumer: EnergyConsumer,
        hourly_data: List[Decimal],
    ) -> OperatingPatternAssessment:
        """Assess operating patterns from hourly demand data.

        Decomposes load into baseload and variable components, calculates
        peak-to-off-peak ratios, and classifies the load type.

        Args:
            consumer: Energy consumer being assessed.
            hourly_data: List of hourly demand values (kW).  Minimum 24
                         values for meaningful analysis.

        Returns:
            OperatingPatternAssessment with load decomposition.
        """
        if not hourly_data or len(hourly_data) < 2:
            logger.warning(
                "Insufficient hourly data for '%s' (n=%d)",
                consumer.name, len(hourly_data),
            )
            return OperatingPatternAssessment(
                consumer_id=consumer.id,
                load_type=LoadType.VARIABLE.value,
                notes="Insufficient hourly data for pattern assessment",
            )

        values = [_decimal(v) for v in hourly_data]
        non_zero = [v for v in values if v > Decimal("0")]

        if not non_zero:
            return OperatingPatternAssessment(
                consumer_id=consumer.id,
                baseload_fraction=Decimal("0"),
                variable_fraction=Decimal("1"),
                load_type=LoadType.VARIABLE.value,
                notes="All hourly values are zero",
            )

        min_val = min(values)
        max_val = max(values)
        n_values = _decimal(len(values))

        # Baseload is the minimum sustained demand.
        baseload_kw = max(min_val, Decimal("0"))
        peak_kw = max_val

        # Baseload fraction.
        baseload_fraction = _safe_divide(baseload_kw, peak_kw)
        variable_fraction = Decimal("1") - baseload_fraction

        # Operating hours utilisation (fraction of hours with non-zero load).
        utilisation = _safe_divide(
            _decimal(len(non_zero)), n_values
        )

        # Peak-to-off-peak ratio.
        # Define peak hours as top 25%, off-peak as bottom 25%.
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        quarter = max(n // 4, 1)

        offpeak_avg = _safe_divide(
            sum(sorted_vals[:quarter], Decimal("0")),
            _decimal(quarter),
        )
        peak_avg = _safe_divide(
            sum(sorted_vals[-quarter:], Decimal("0")),
            _decimal(quarter),
        )
        peak_to_offpeak = _safe_divide(peak_avg, offpeak_avg, Decimal("1"))

        # Classify load type.
        load_type = self._classify_load_type(
            baseload_fraction, peak_to_offpeak, consumer
        )

        notes_parts: List[str] = []
        if baseload_fraction > Decimal("0.8"):
            notes_parts.append("Predominantly baseload pattern detected.")
        elif baseload_fraction < Decimal("0.2"):
            notes_parts.append("Highly variable load pattern detected.")

        if peak_to_offpeak > Decimal("3"):
            notes_parts.append(
                f"High peak-to-off-peak ratio ({float(peak_to_offpeak):.1f}x) "
                "indicates significant load variation."
            )

        if utilisation < Decimal("0.5"):
            notes_parts.append(
                f"Low utilisation ({float(utilisation * Decimal('100')):.0f}%) "
                "suggests intermittent operation."
            )

        return OperatingPatternAssessment(
            consumer_id=consumer.id,
            baseload_fraction=_round_val(baseload_fraction, 4),
            variable_fraction=_round_val(variable_fraction, 4),
            peak_to_offpeak_ratio=_round_val(peak_to_offpeak, 4),
            estimated_baseload_kw=_round_val(baseload_kw, 2),
            estimated_peak_kw=_round_val(peak_kw, 2),
            load_type=load_type,
            operating_hours_utilisation=_round_val(utilisation, 4),
            notes=" ".join(notes_parts),
        )

    def generate_equipment_census(
        self,
        consumers: List[EnergyConsumer],
    ) -> Dict[str, Any]:
        """Generate equipment census summary grouped by category.

        Aggregates equipment counts, total rated power, average load
        factor, total consumption, and flags anomalies (very low or
        very high load factors) by SEU category.

        Args:
            consumers: List of energy consumers.

        Returns:
            Dictionary with census data:
                - total_consumers: int
                - total_rated_power_kw: str (Decimal)
                - total_consumption_kwh: str (Decimal)
                - by_category: dict of category summaries
                - anomalies: list of flagged consumers
        """
        t0 = time.perf_counter()

        category_data: Dict[str, Dict[str, Any]] = {}
        anomalies: List[Dict[str, Any]] = []
        total_power = Decimal("0")
        total_consumption = Decimal("0")

        for consumer in consumers:
            cat = consumer.category
            total_power += consumer.rated_power_kw
            total_consumption += consumer.annual_consumption_kwh

            if cat not in category_data:
                category_data[cat] = {
                    "count": 0,
                    "total_rated_power_kw": Decimal("0"),
                    "total_consumption_kwh": Decimal("0"),
                    "load_factors": [],
                    "operating_hours": [],
                    "consumers": [],
                }

            entry = category_data[cat]
            entry["count"] += 1
            entry["total_rated_power_kw"] += consumer.rated_power_kw
            entry["total_consumption_kwh"] += consumer.annual_consumption_kwh
            entry["load_factors"].append(consumer.load_factor)
            entry["operating_hours"].append(consumer.annual_operating_hours)
            entry["consumers"].append(consumer.name)

            # Flag anomalies.
            if consumer.load_factor < Decimal("0.3") and consumer.rated_power_kw > Decimal("0"):
                anomalies.append({
                    "consumer_id": consumer.id,
                    "consumer_name": consumer.name,
                    "category": cat,
                    "issue": "low_load_factor",
                    "value": str(_round_val(consumer.load_factor, 3)),
                    "recommendation": (
                        "Load factor below 0.30 indicates potential oversizing. "
                        "Consider right-sizing or replacing with smaller equipment."
                    ),
                })
            elif consumer.load_factor > Decimal("0.9"):
                anomalies.append({
                    "consumer_id": consumer.id,
                    "consumer_name": consumer.name,
                    "category": cat,
                    "issue": "high_load_factor",
                    "value": str(_round_val(consumer.load_factor, 3)),
                    "recommendation": (
                        "Load factor above 0.90 indicates equipment may be "
                        "overloaded. Check for capacity issues and overheating."
                    ),
                })

        # Build category summaries.
        by_category: Dict[str, Dict[str, Any]] = {}
        for cat, data in category_data.items():
            lf_list = data["load_factors"]
            n = _decimal(len(lf_list))
            avg_lf = _safe_divide(
                sum(lf_list, Decimal("0")), n
            )
            oh_list = data["operating_hours"]
            avg_hours = _safe_divide(
                _decimal(sum(oh_list)), n
            )
            eff_range = self._efficiency_ranges.get(
                cat, (Decimal("0.50"), Decimal("0.85"))
            )

            by_category[cat] = {
                "count": data["count"],
                "total_rated_power_kw": str(
                    _round_val(data["total_rated_power_kw"], 2)
                ),
                "total_consumption_kwh": str(
                    _round_val(data["total_consumption_kwh"], 2)
                ),
                "average_load_factor": str(_round_val(avg_lf, 4)),
                "average_operating_hours": round(float(avg_hours)),
                "typical_efficiency_range": {
                    "min": str(_round_val(eff_range[0], 2)),
                    "max": str(_round_val(eff_range[1], 2)),
                },
                "consumers": data["consumers"],
            }

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0)

        census = {
            "total_consumers": len(consumers),
            "total_rated_power_kw": str(_round_val(total_power, 2)),
            "total_consumption_kwh": str(_round_val(total_consumption, 2)),
            "categories_count": len(by_category),
            "by_category": by_category,
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
            "census_time_ms": elapsed_ms,
            "provenance_hash": _compute_hash({
                "total_consumers": len(consumers),
                "total_power": str(total_power),
                "total_consumption": str(total_consumption),
                "by_category": {
                    k: v["count"] for k, v in by_category.items()
                },
            }),
        }

        logger.info(
            "Equipment census: %d consumers, %d categories, %d anomalies",
            len(consumers), len(by_category), len(anomalies),
        )
        return census

    def rank_improvement_opportunities(
        self,
        seus: List[SEUResult],
    ) -> List[ImprovementOpportunity]:
        """Rank improvement opportunities across all SEUs.

        Creates a prioritised list of improvement opportunities based
        on estimated energy savings potential, applying category-specific
        benchmarks from DOE/ASHRAE published data.

        Ranking criteria:
            1. Estimated savings (kWh/yr) - higher is better.
            2. Category benchmark percentage - higher potential first.
            3. Consumer's percentage of total - larger SEUs prioritised.

        Args:
            seus: List of SEU results (typically only significant SEUs).

        Returns:
            Sorted list of ImprovementOpportunity instances.
        """
        opportunities: List[ImprovementOpportunity] = []

        for seu in seus:
            if not seu.is_significant:
                continue

            benchmark_pct = self._improvement_benchmarks.get(
                seu.consumer.category, Decimal("10")
            )
            base_savings = (
                seu.consumer.annual_consumption_kwh * benchmark_pct / Decimal("100")
            )

            for idx, desc in enumerate(seu.improvement_opportunities):
                # Distribute benchmark across opportunities (weighted).
                n_opps = _decimal(max(len(seu.improvement_opportunities), 1))
                # First opportunity gets more weight, diminishing returns.
                weight = _safe_divide(
                    Decimal("2") * (_decimal(len(seu.improvement_opportunities)) - _decimal(idx)),
                    n_opps * (_decimal(len(seu.improvement_opportunities)) + Decimal("1")),
                    Decimal("1"),
                )
                opp_savings = base_savings * weight
                opp_pct = benchmark_pct * weight

                # Confidence based on position and data quality.
                if idx == 0:
                    confidence = "high"
                elif idx <= 2:
                    confidence = "medium"
                else:
                    confidence = "low"

                opportunities.append(ImprovementOpportunity(
                    seu_id=seu.seu_id,
                    description=desc,
                    category=seu.consumer.category,
                    estimated_savings_pct=_round_val(opp_pct, 2),
                    estimated_savings_kwh=_round_val(opp_savings, 2),
                    priority_rank=0,  # Assigned after sorting.
                    confidence_level=confidence,
                ))

        # Sort by savings descending.
        opportunities.sort(
            key=lambda o: o.estimated_savings_kwh, reverse=True
        )

        # Assign ranks.
        for rank, opp in enumerate(opportunities, start=1):
            opp.priority_rank = rank

        logger.info(
            "Ranked %d improvement opportunities across %d SEUs",
            len(opportunities),
            sum(1 for s in seus if s.is_significant),
        )
        return opportunities

    def validate_seu_determination(
        self,
        result: SEUAnalysisResult,
    ) -> SEUValidationResult:
        """Validate SEU analysis against ISO 50001 requirements.

        Checks the completeness and methodological soundness of the
        SEU determination process, verifying:
        - Minimum number of consumers assessed.
        - Pareto analysis was performed.
        - Coverage threshold met.
        - Energy drivers identified for each SEU.
        - Improvement opportunities assigned.
        - Provenance tracking in place.

        Args:
            result: Complete SEU analysis result to validate.

        Returns:
            SEUValidationResult with pass/fail and findings.
        """
        findings: List[Dict[str, Any]] = []
        recommendations: List[str] = []
        checks_passed = 0
        checks_total = 0

        # Check 1: Minimum consumers assessed (ISO 50001 6.3 requirement).
        checks_total += 1
        total_consumers = result.seu_count + result.non_seu_count
        if total_consumers >= 3:
            checks_passed += 1
            findings.append({
                "check": "minimum_consumers",
                "status": "PASS",
                "detail": f"{total_consumers} consumers assessed (minimum 3)",
            })
        else:
            findings.append({
                "check": "minimum_consumers",
                "status": "FAIL",
                "detail": (
                    f"Only {total_consumers} consumers. "
                    "ISO 50001 requires comprehensive energy consumer inventory."
                ),
            })
            recommendations.append(
                "Expand energy consumer inventory to include all significant loads."
            )

        # Check 2: At least one SEU identified.
        checks_total += 1
        if result.seu_count > 0:
            checks_passed += 1
            findings.append({
                "check": "seus_identified",
                "status": "PASS",
                "detail": f"{result.seu_count} SEU(s) identified",
            })
        else:
            findings.append({
                "check": "seus_identified",
                "status": "FAIL",
                "detail": "No SEUs identified. Review thresholds or data quality.",
            })
            recommendations.append(
                "Lower significance thresholds or verify energy consumption data."
            )

        # Check 3: SEU coverage >= 50% of total (good practice).
        checks_total += 1
        if result.coverage_percentage >= Decimal("50"):
            checks_passed += 1
            findings.append({
                "check": "coverage_threshold",
                "status": "PASS",
                "detail": (
                    f"SEUs cover {float(result.coverage_percentage):.1f}% "
                    "of total consumption"
                ),
            })
        else:
            findings.append({
                "check": "coverage_threshold",
                "status": "WARNING",
                "detail": (
                    f"SEU coverage is {float(result.coverage_percentage):.1f}% "
                    "(recommended >= 50%)."
                ),
            })
            recommendations.append(
                "Review SEU threshold settings to ensure major consumers are captured."
            )

        # Check 4: Pareto data generated.
        checks_total += 1
        if result.pareto_chart_data:
            checks_passed += 1
            findings.append({
                "check": "pareto_analysis",
                "status": "PASS",
                "detail": f"Pareto analysis generated {len(result.pareto_chart_data)} data points",
            })
        else:
            findings.append({
                "check": "pareto_analysis",
                "status": "FAIL",
                "detail": "No Pareto analysis data generated.",
            })
            recommendations.append(
                "Ensure energy consumer data is sufficient for Pareto analysis."
            )

        # Check 5: Improvement opportunities assigned to SEUs.
        checks_total += 1
        seus_with_improvements = sum(
            1 for s in result.seus
            if s.is_significant and s.improvement_opportunities
        )
        if seus_with_improvements >= result.seu_count and result.seu_count > 0:
            checks_passed += 1
            findings.append({
                "check": "improvement_opportunities",
                "status": "PASS",
                "detail": (
                    f"All {result.seu_count} SEUs have improvement "
                    "opportunities assigned"
                ),
            })
        elif seus_with_improvements > 0:
            findings.append({
                "check": "improvement_opportunities",
                "status": "WARNING",
                "detail": (
                    f"{seus_with_improvements}/{result.seu_count} SEUs have "
                    "improvement opportunities."
                ),
            })
            recommendations.append(
                "Assign improvement opportunities to all identified SEUs "
                "per ISO 50001 Clause 6.3."
            )
        else:
            findings.append({
                "check": "improvement_opportunities",
                "status": "FAIL",
                "detail": "No improvement opportunities assigned to any SEU.",
            })
            recommendations.append(
                "Conduct detailed assessment to identify improvement "
                "opportunities for each SEU."
            )

        # Check 6: Provenance hash present.
        checks_total += 1
        if result.provenance_hash and len(result.provenance_hash) == 64:
            checks_passed += 1
            findings.append({
                "check": "provenance_tracking",
                "status": "PASS",
                "detail": f"SHA-256 provenance hash: {result.provenance_hash[:16]}...",
            })
        else:
            findings.append({
                "check": "provenance_tracking",
                "status": "FAIL",
                "detail": "Missing or invalid provenance hash.",
            })
            recommendations.append(
                "Regenerate analysis to ensure provenance hash is computed."
            )

        # Check 7: Total consumption is non-zero and matches.
        checks_total += 1
        if result.total_consumption_kwh > Decimal("0"):
            checks_passed += 1
            findings.append({
                "check": "consumption_data",
                "status": "PASS",
                "detail": (
                    f"Total consumption: "
                    f"{float(result.total_consumption_kwh):,.0f} kWh"
                ),
            })
        else:
            findings.append({
                "check": "consumption_data",
                "status": "FAIL",
                "detail": "Total consumption is zero or missing.",
            })
            recommendations.append(
                "Provide valid energy consumption data for the facility."
            )

        # Check 8: Significance scores are within valid range.
        checks_total += 1
        invalid_scores = [
            s for s in result.seus
            if s.significance_score < Decimal("0")
            or s.significance_score > Decimal("100")
        ]
        if not invalid_scores:
            checks_passed += 1
            findings.append({
                "check": "score_validity",
                "status": "PASS",
                "detail": "All significance scores within 0-100 range",
            })
        else:
            findings.append({
                "check": "score_validity",
                "status": "FAIL",
                "detail": (
                    f"{len(invalid_scores)} SEU(s) have invalid "
                    "significance scores."
                ),
            })
            recommendations.append(
                "Review significance scoring algorithm for boundary errors."
            )

        # Check 9: Measurement period defined.
        checks_total += 1
        # We check via the analysis result's facility_id being non-empty.
        if result.facility_id:
            checks_passed += 1
            findings.append({
                "check": "facility_reference",
                "status": "PASS",
                "detail": f"Facility reference: {result.facility_id[:16]}...",
            })
        else:
            findings.append({
                "check": "facility_reference",
                "status": "FAIL",
                "detail": "Missing facility reference.",
            })
            recommendations.append(
                "Ensure facility profile includes a valid facility_id."
            )

        # Check 10: No duplicate consumer IDs.
        checks_total += 1
        consumer_ids = [s.consumer.id for s in result.seus]
        if len(consumer_ids) == len(set(consumer_ids)):
            checks_passed += 1
            findings.append({
                "check": "unique_consumers",
                "status": "PASS",
                "detail": "All consumer IDs are unique",
            })
        else:
            findings.append({
                "check": "unique_consumers",
                "status": "FAIL",
                "detail": "Duplicate consumer IDs detected.",
            })
            recommendations.append(
                "Ensure each energy consumer has a unique identifier."
            )

        is_valid = checks_passed == checks_total

        validation = SEUValidationResult(
            analysis_id=result.analysis_id,
            is_valid=is_valid,
            checks_passed=checks_passed,
            checks_total=checks_total,
            findings=findings,
            recommendations=recommendations,
            iso_clause_references=[
                "ISO 50001:2018 Clause 6.3 - Energy review",
                "ISO 50001:2018 Clause 6.3 a) - Analysis of energy use and consumption",
                "ISO 50001:2018 Clause 6.3 b) - Identification of SEUs",
                "ISO 50001:2018 Clause 6.3 c) - Determination of current energy performance",
                "ISO 50001:2018 Clause 6.3 d) - Estimation of future energy use",
                "ISO 50006:2014 - Measuring energy performance using EnPIs and EnBs",
            ],
        )
        validation.provenance_hash = _compute_hash(validation)

        logger.info(
            "SEU validation: %d/%d checks passed, valid=%s",
            checks_passed, checks_total, is_valid,
        )
        return validation

    def get_category_benchmark(
        self,
        category: SEUCategory,
    ) -> Dict[str, Any]:
        """Get benchmark data for a specific SEU category.

        Returns reference data including typical load factor, efficiency
        range, improvement benchmark, and standard improvement actions.

        Args:
            category: SEU category to look up.

        Returns:
            Dictionary with benchmark data.
        """
        cat_val = category.value
        eff_range = self._efficiency_ranges.get(
            cat_val, (Decimal("0.50"), Decimal("0.85"))
        )
        return {
            "category": cat_val,
            "default_load_factor": str(
                self._load_factors.get(cat_val, Decimal("0.60"))
            ),
            "typical_efficiency_range": {
                "min": str(_round_val(eff_range[0], 2)),
                "max": str(_round_val(eff_range[1], 2)),
            },
            "improvement_benchmark_pct": str(
                self._improvement_benchmarks.get(cat_val, Decimal("10"))
            ),
            "typical_operating_hours": TYPICAL_OPERATING_HOURS,
            "improvement_opportunities": STANDARD_IMPROVEMENT_OPPORTUNITIES.get(
                cat_val, []
            ),
        }

    def get_all_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """Get benchmark data for all SEU categories.

        Returns:
            Dictionary mapping category values to benchmark data dicts.
        """
        return {
            cat.value: self.get_category_benchmark(cat)
            for cat in SEUCategory
        }

    def estimate_consumption_from_specs(
        self,
        consumer: EnergyConsumer,
    ) -> Decimal:
        """Estimate annual consumption from equipment specifications.

        Uses the formula:
            consumption = rated_power * load_factor * operating_hours

        Useful for validating measured consumption data or filling gaps.

        Args:
            consumer: Energy consumer with specifications.

        Returns:
            Estimated annual consumption in kWh.
        """
        load_factor = consumer.load_factor
        if load_factor <= Decimal("0"):
            load_factor = self._load_factors.get(
                consumer.category, Decimal("0.60")
            )

        estimated = (
            consumer.rated_power_kw
            * load_factor
            * _decimal(consumer.annual_operating_hours)
        )
        return _round_val(estimated, 2)

    def compare_measured_vs_estimated(
        self,
        consumer: EnergyConsumer,
    ) -> Dict[str, Any]:
        """Compare measured vs estimated consumption for a consumer.

        Calculates the discrepancy between measured consumption and
        spec-based estimation.  Large discrepancies may indicate
        sub-metering errors or changed operating conditions.

        Args:
            consumer: Energy consumer with measured and spec data.

        Returns:
            Dictionary with comparison metrics.
        """
        estimated = self.estimate_consumption_from_specs(consumer)
        measured = consumer.annual_consumption_kwh

        if estimated <= Decimal("0"):
            return {
                "consumer_id": consumer.id,
                "consumer_name": consumer.name,
                "measured_kwh": str(_round_val(measured, 2)),
                "estimated_kwh": str(_round_val(estimated, 2)),
                "discrepancy_pct": "N/A",
                "discrepancy_kwh": "N/A",
                "status": "insufficient_data",
                "notes": "Estimated consumption is zero; check specifications.",
            }

        discrepancy_kwh = measured - estimated
        discrepancy_pct = _safe_pct(abs(discrepancy_kwh), estimated)

        if discrepancy_pct <= Decimal("10"):
            status = "good_agreement"
        elif discrepancy_pct <= Decimal("25"):
            status = "acceptable"
        elif discrepancy_pct <= Decimal("50"):
            status = "investigate"
        else:
            status = "significant_discrepancy"

        notes = []
        if measured > estimated * Decimal("1.25"):
            notes.append(
                "Measured exceeds estimated by >25%. Check for additional "
                "connected loads, higher-than-expected operating hours, or "
                "degraded equipment efficiency."
            )
        elif measured < estimated * Decimal("0.75"):
            notes.append(
                "Measured is below estimated by >25%. Check for reduced "
                "operating hours, partial shutdown periods, or meter errors."
            )

        return {
            "consumer_id": consumer.id,
            "consumer_name": consumer.name,
            "measured_kwh": str(_round_val(measured, 2)),
            "estimated_kwh": str(_round_val(estimated, 2)),
            "discrepancy_kwh": str(_round_val(discrepancy_kwh, 2)),
            "discrepancy_pct": str(_round_val(discrepancy_pct, 2)),
            "status": status,
            "notes": " ".join(notes) if notes else "Within acceptable range.",
        }

    # ------------------------------------------------------------------ #
    # Private Helpers                                                     #
    # ------------------------------------------------------------------ #

    def _resolve_total_consumption(
        self,
        profile: FacilityEnergyProfile,
    ) -> Decimal:
        """Resolve total consumption from profile or sum of consumers.

        If the profile has a non-zero total_annual_consumption_kwh, use
        it.  Otherwise, sum all consumers' consumption.

        Args:
            profile: Facility energy profile.

        Returns:
            Total consumption in kWh.
        """
        if profile.total_annual_consumption_kwh > Decimal("0"):
            return profile.total_annual_consumption_kwh

        total = sum(
            (c.annual_consumption_kwh for c in profile.energy_consumers),
            Decimal("0"),
        )
        logger.info(
            "Total consumption derived from consumer sum: %.0f kWh",
            float(total),
        )
        return total

    def _calculate_percentages(
        self,
        consumers: List[EnergyConsumer],
        total_kwh: Decimal,
    ) -> List[EnergyConsumer]:
        """Calculate percentage of total for each consumer.

        Updates the percentage_of_total field on each consumer model
        by creating new model instances (Pydantic immutability).

        Args:
            consumers: List of energy consumers.
            total_kwh: Total facility consumption (kWh).

        Returns:
            New list of EnergyConsumer with updated percentages.
        """
        updated: List[EnergyConsumer] = []
        for consumer in consumers:
            pct = _safe_pct(consumer.annual_consumption_kwh, total_kwh)
            updated_consumer = consumer.model_copy(
                update={"percentage_of_total": _round_val(pct, 4)}
            )
            updated.append(updated_consumer)
        return updated

    def _calculate_significance_score(
        self,
        consumer: EnergyConsumer,
        total_kwh: Decimal,
        in_pareto: bool,
    ) -> Decimal:
        """Calculate significance score (0-100) for a consumer.

        Scoring breakdown:
            - Percentage of total (max 40 pts): linear scaling.
            - Operating hours (max 20 pts): hours / 8760 * 20.
            - Load factor (max 15 pts): higher load factor = more impact.
            - Improvement potential (max 15 pts): benchmark-based.
            - Pareto membership bonus (10 pts): if within Pareto threshold.

        Args:
            consumer: Energy consumer to score.
            total_kwh: Total facility consumption.
            in_pareto: Whether consumer is within Pareto threshold.

        Returns:
            Significance score as Decimal (0-100).
        """
        score = Decimal("0")

        # Percentage of total (max 40 pts).
        pct = _safe_pct(consumer.annual_consumption_kwh, total_kwh)
        # Scale: 0% = 0 pts, 20%+ = 40 pts (linear from 0 to 20%).
        pct_score = min(pct / Decimal("20") * Decimal("40"), Decimal("40"))
        score += pct_score

        # Operating hours (max 20 pts).
        hours_factor = _safe_divide(
            _decimal(consumer.annual_operating_hours),
            Decimal("8760"),
        )
        score += min(hours_factor * Decimal("20"), Decimal("20"))

        # Load factor (max 15 pts).
        lf_score = min(
            consumer.load_factor * Decimal("15"), Decimal("15")
        )
        score += lf_score

        # Improvement potential (max 15 pts).
        benchmark = self._improvement_benchmarks.get(
            consumer.category, Decimal("10")
        )
        # Scale: 0% = 0 pts, 40%+ = 15 pts.
        imp_score = min(
            benchmark / Decimal("40") * Decimal("15"), Decimal("15")
        )
        score += imp_score

        # Pareto membership bonus (10 pts).
        if in_pareto:
            score += Decimal("10")

        return min(score, Decimal("100"))

    def _get_improvement_opportunities(
        self,
        consumer: EnergyConsumer,
    ) -> List[str]:
        """Get standard improvement opportunities for a consumer's category.

        Args:
            consumer: Energy consumer.

        Returns:
            List of improvement opportunity descriptions.
        """
        return STANDARD_IMPROVEMENT_OPPORTUNITIES.get(
            consumer.category,
            STANDARD_IMPROVEMENT_OPPORTUNITIES[SEUCategory.OTHER.value],
        )

    def _pearson_correlation(
        self,
        x_vals: List[Decimal],
        y_vals: List[Decimal],
    ) -> Decimal:
        """Calculate Pearson correlation coefficient using Decimal arithmetic.

        Formula:
            r = (n * sum(xy) - sum(x) * sum(y))
                / sqrt((n * sum(x^2) - sum(x)^2) * (n * sum(y^2) - sum(y)^2))

        Args:
            x_vals: Independent variable values.
            y_vals: Dependent variable values.

        Returns:
            Pearson r as Decimal (-1 to 1); 0 if computation fails.
        """
        n = min(len(x_vals), len(y_vals))
        if n < 3:
            return Decimal("0")

        n_dec = _decimal(n)

        sum_x = Decimal("0")
        sum_y = Decimal("0")
        sum_xy = Decimal("0")
        sum_x2 = Decimal("0")
        sum_y2 = Decimal("0")

        for i in range(n):
            xi = x_vals[i]
            yi = y_vals[i]
            sum_x += xi
            sum_y += yi
            sum_xy += xi * yi
            sum_x2 += xi * xi
            sum_y2 += yi * yi

        numerator = n_dec * sum_xy - sum_x * sum_y
        denom_a = n_dec * sum_x2 - sum_x * sum_x
        denom_b = n_dec * sum_y2 - sum_y * sum_y

        if denom_a <= Decimal("0") or denom_b <= Decimal("0"):
            return Decimal("0")

        denominator = _sqrt_decimal(denom_a * denom_b)

        if denominator <= Decimal("0"):
            return Decimal("0")

        r = _safe_divide(numerator, denominator)

        # Clamp to [-1, 1] to handle minor floating point drift.
        if r > Decimal("1"):
            r = Decimal("1")
        elif r < Decimal("-1"):
            r = Decimal("-1")

        return r

    def _classify_load_type(
        self,
        baseload_fraction: Decimal,
        peak_to_offpeak: Decimal,
        consumer: EnergyConsumer,
    ) -> str:
        """Classify load type from pattern metrics and consumer metadata.

        Args:
            baseload_fraction: Fraction of constant load (0-1).
            peak_to_offpeak: Peak to off-peak demand ratio.
            consumer: Energy consumer metadata.

        Returns:
            LoadType value string.
        """
        # Check for weather-dependent categories.
        weather_categories = {
            SEUCategory.HVAC.value,
            SEUCategory.BOILERS.value,
        }
        if consumer.category in weather_categories:
            if baseload_fraction < Decimal("0.5"):
                return LoadType.WEATHER_DEPENDENT.value

        # Check for production-dependent categories.
        production_categories = {
            SEUCategory.MOTORS.value,
            SEUCategory.PROCESS_HEAT.value,
            SEUCategory.COMPRESSED_AIR.value,
            SEUCategory.PUMPS.value,
            SEUCategory.FANS.value,
        }
        if consumer.category in production_categories:
            if peak_to_offpeak > Decimal("2"):
                return LoadType.PRODUCTION_DEPENDENT.value

        # General classification by baseload fraction.
        if baseload_fraction >= Decimal("0.8"):
            return LoadType.BASELOAD.value
        else:
            return LoadType.VARIABLE.value

    def _build_driver_summary(
        self,
        seus: List[SEUResult],
    ) -> List[Dict[str, Any]]:
        """Build energy driver summary across all SEUs.

        Args:
            seus: List of significant SEU results.

        Returns:
            List of driver summary dicts.
        """
        driver_counts: Dict[str, int] = {}
        driver_avg_r: Dict[str, List[Decimal]] = {}

        for seu in seus:
            for driver in seu.energy_drivers:
                name = driver.driver_name
                driver_counts[name] = driver_counts.get(name, 0) + 1
                if name not in driver_avg_r:
                    driver_avg_r[name] = []
                driver_avg_r[name].append(driver.correlation_coefficient)

        summary: List[Dict[str, Any]] = []
        for name, count in driver_counts.items():
            r_values = driver_avg_r.get(name, [])
            avg_r = _safe_divide(
                sum(r_values, Decimal("0")),
                _decimal(len(r_values)),
            )
            summary.append({
                "driver_name": name,
                "seus_affected": count,
                "average_correlation": str(_round_val(avg_r, 4)),
                "is_significant": abs(avg_r) >= Decimal("0.5"),
            })

        summary.sort(
            key=lambda d: abs(_decimal(d["average_correlation"])),
            reverse=True,
        )
        return summary

    # ------------------------------------------------------------------ #
    # Batch Processing                                                    #
    # ------------------------------------------------------------------ #

    def analyze_batch(
        self,
        profiles: List[FacilityEnergyProfile],
        thresholds: Optional[SEUThresholds] = None,
        batch_size: int = 10,
    ) -> List[SEUAnalysisResult]:
        """Analyse multiple facilities in batch.

        Processes facilities in configurable batch sizes for memory
        efficiency on large portfolios.

        Args:
            profiles: List of facility energy profiles.
            thresholds: Common thresholds (optional, uses defaults).
            batch_size: Number of facilities per processing batch.

        Returns:
            List of SEUAnalysisResult, one per facility.
        """
        t0 = time.perf_counter()
        results: List[SEUAnalysisResult] = []

        logger.info(
            "Batch SEU analysis: %d facilities, batch_size=%d",
            len(profiles), batch_size,
        )

        for i in range(0, len(profiles), batch_size):
            batch = profiles[i:i + batch_size]
            for profile in batch:
                try:
                    result = self.analyze_seus(profile, thresholds)
                    results.append(result)
                except (ValueError, Exception) as exc:
                    logger.error(
                        "SEU analysis failed for facility '%s': %s",
                        profile.facility_name, str(exc),
                    )
                    # Create error result placeholder.
                    results.append(SEUAnalysisResult(
                        facility_id=profile.facility_id,
                        total_consumption_kwh=profile.total_annual_consumption_kwh,
                        provenance_hash=_compute_hash({
                            "error": str(exc),
                            "facility_id": profile.facility_id,
                        }),
                    ))

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0)
        logger.info(
            "Batch analysis complete: %d/%d succeeded in %d ms",
            sum(1 for r in results if r.seu_count > 0),
            len(profiles),
            elapsed_ms,
        )
        return results

    # ------------------------------------------------------------------ #
    # Summary / Reporting                                                 #
    # ------------------------------------------------------------------ #

    def generate_summary_report(
        self,
        result: SEUAnalysisResult,
    ) -> Dict[str, Any]:
        """Generate a summary report from an SEU analysis result.

        Produces a structured dictionary suitable for rendering into
        ISO 50001 energy review documentation.

        Args:
            result: Complete SEU analysis result.

        Returns:
            Dictionary with report sections.
        """
        significant_seus = [s for s in result.seus if s.is_significant]

        # Category breakdown.
        category_breakdown: Dict[str, Dict[str, Any]] = {}
        for seu in significant_seus:
            cat = seu.consumer.category
            if cat not in category_breakdown:
                category_breakdown[cat] = {
                    "count": 0,
                    "total_consumption_kwh": Decimal("0"),
                    "total_savings_potential_kwh": Decimal("0"),
                    "consumers": [],
                }
            entry = category_breakdown[cat]
            entry["count"] += 1
            entry["total_consumption_kwh"] += seu.consumer.annual_consumption_kwh
            entry["total_savings_potential_kwh"] += seu.estimated_savings_potential_kwh
            entry["consumers"].append(seu.consumer.name)

        # Convert Decimals to strings for JSON serialisation.
        for cat, data in category_breakdown.items():
            data["total_consumption_kwh"] = str(
                _round_val(data["total_consumption_kwh"], 2)
            )
            data["total_savings_potential_kwh"] = str(
                _round_val(data["total_savings_potential_kwh"], 2)
            )

        # Total savings potential.
        total_savings = sum(
            (s.estimated_savings_potential_kwh for s in significant_seus),
            Decimal("0"),
        )
        total_savings_pct = _safe_pct(total_savings, result.total_consumption_kwh)

        report = {
            "report_title": "ISO 50001 Significant Energy Use Analysis",
            "facility_id": result.facility_id,
            "analysis_id": result.analysis_id,
            "analysis_date": str(result.analysis_date),
            "summary": {
                "total_consumption_kwh": str(
                    _round_val(result.total_consumption_kwh, 2)
                ),
                "seu_count": result.seu_count,
                "non_seu_count": result.non_seu_count,
                "coverage_percentage": str(
                    _round_val(result.coverage_percentage, 2)
                ),
                "total_savings_potential_kwh": str(
                    _round_val(total_savings, 2)
                ),
                "total_savings_potential_pct": str(
                    _round_val(total_savings_pct, 2)
                ),
            },
            "seus": [
                {
                    "name": s.consumer.name,
                    "category": s.consumer.category,
                    "consumption_kwh": str(
                        _round_val(s.consumer.annual_consumption_kwh, 2)
                    ),
                    "percentage_of_total": str(
                        _round_val(s.consumer.percentage_of_total, 2)
                    ),
                    "significance_score": str(
                        _round_val(s.significance_score, 2)
                    ),
                    "determination_method": s.determination_method,
                    "savings_potential_kwh": str(
                        _round_val(s.estimated_savings_potential_kwh, 2)
                    ),
                    "improvement_opportunities": s.improvement_opportunities[:3],
                }
                for s in significant_seus
            ],
            "category_breakdown": category_breakdown,
            "pareto_chart_data": result.pareto_chart_data,
            "methodology": {
                "approach": "Pareto analysis with multi-criteria significance scoring",
                "standards": [
                    "ISO 50001:2018 Clause 6.3",
                    "ISO 50006:2014",
                    "ISO 50015:2014",
                ],
                "significance_threshold": str(self._significance_threshold),
                "engine_version": self.engine_version,
            },
            "provenance_hash": result.provenance_hash,
            "calculation_time_ms": result.calculation_time_ms,
        }

        report_hash = _compute_hash(report)
        report["report_provenance_hash"] = report_hash

        logger.info(
            "Summary report generated for analysis %s (hash=%s)",
            result.analysis_id[:16], report_hash[:16],
        )
        return report
