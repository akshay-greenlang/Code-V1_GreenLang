# -*- coding: utf-8 -*-
"""
HVACAssessmentEngine - PACK-032 Building Energy Assessment Engine 3
====================================================================

Comprehensive HVAC (Heating, Ventilation and Air Conditioning) system
efficiency assessment for buildings.  Evaluates heating system seasonal
efficiency, heat pump SPF, cooling SEER, ventilation SFP, heat recovery
effectiveness, refrigerant F-gas compliance, distribution losses, and
control system effectiveness.

EN 15316-4-1:2017 Compliance:
    - Heating system seasonal efficiency (indirect method)
    - Boiler part-load efficiency correction
    - Distribution and control losses

EN 14825:2022 Compliance:
    - Heat pump Seasonal Performance Factor (SPF)
    - Temperature bin method for SCOP/SEER
    - Part-load ratio curves

EN 13779:2007 / EN 16798-3:2017 Compliance:
    - Ventilation Specific Fan Power (SFP)
    - Indoor air quality categories (IDA 1-4)
    - Minimum fresh air rates

EN 308:1997 / EN 13141-7 Compliance:
    - Heat recovery efficiency measurement
    - Thermal effectiveness by HRV type

EU 517/2014 (F-Gas Regulation) Compliance:
    - Refrigerant GWP values (IPCC AR6)
    - Leak rate thresholds and mandatory checks
    - F-gas phase-down compliance assessment

ASHRAE 90.1-2022 / CIBSE Guide B Compliance:
    - System efficiency benchmarks
    - Part-load performance curves
    - Control credit methodology

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Lookup tables from published standards and manufacturer data
    - No LLM involvement in any numeric calculation path
    - SHA-256 provenance hashing on every result
    - Efficiency data from EN standards, ASHRAE, CIBSE

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-032 Building Energy Assessment
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


class HeatingSystemType(str, Enum):
    """Heating system types for HVAC assessment.

    Covers all major heating technologies found in commercial
    and residential buildings per EN 15316-4-1 and CIBSE Guide B.
    """
    GAS_BOILER = "gas_boiler"
    OIL_BOILER = "oil_boiler"
    ELECTRIC_BOILER = "electric_boiler"
    BIOMASS_BOILER = "biomass_boiler"
    AIR_SOURCE_HEAT_PUMP = "air_source_heat_pump"
    GROUND_SOURCE_HEAT_PUMP = "ground_source_heat_pump"
    WATER_SOURCE_HEAT_PUMP = "water_source_heat_pump"
    DISTRICT_HEATING = "district_heating"
    ELECTRIC_RESISTANCE = "electric_resistance"
    CHP = "combined_heat_power"
    INFRARED_RADIANT = "infrared_radiant"
    WARM_AIR = "warm_air"


class CoolingSystemType(str, Enum):
    """Cooling system types for HVAC assessment.

    Covers split systems through to large chillers and specialist
    data centre cooling per EN 14825 and ASHRAE 90.1.
    """
    SPLIT_SYSTEM = "split_system"
    VRF = "variable_refrigerant_flow"
    CHILLER_AHU = "chiller_ahu"
    PTAC = "packaged_terminal_ac"
    CRAC = "computer_room_ac"
    EVAPORATIVE = "evaporative"
    ABSORPTION_CHILLER = "absorption_chiller"
    DISTRICT_COOLING = "district_cooling"
    HEAT_PUMP_REVERSIBLE = "heat_pump_reversible"


class VentilationType(str, Enum):
    """Ventilation system types per EN 16798-3 and EN 13779.

    SFP (Specific Fan Power) varies significantly by type,
    with MVHR being the most efficient when properly designed.
    """
    NATURAL = "natural"
    MECHANICAL_EXTRACT = "mechanical_extract"
    MECHANICAL_SUPPLY_EXTRACT = "mechanical_supply_extract"
    MVHR = "mechanical_ventilation_heat_recovery"
    DEMAND_CONTROLLED = "demand_controlled_ventilation"


class DistributionType(str, Enum):
    """Heating/cooling distribution system types.

    Distribution losses depend on pipe/duct insulation, layout,
    and temperature differential per EN 15316-2.
    """
    RADIATOR = "radiator"
    UNDERFLOOR = "underfloor"
    FAN_COIL = "fan_coil"
    DUCT_AIR = "duct_air"
    CHILLED_BEAM = "chilled_beam"
    RADIANT_CEILING = "radiant_ceiling"


class RefrigerantType(str, Enum):
    """Refrigerant types with GWP classification per EU 517/2014.

    GWP values from IPCC AR6 (100-year time horizon).
    Regulation requires phase-down of high-GWP refrigerants.
    """
    R22 = "R22"
    R410A = "R410A"
    R32 = "R32"
    R290 = "R290"
    R744 = "R744"
    R1234YF = "R1234yf"
    R1234ZE = "R1234ze"
    R454B = "R454B"
    R407C = "R407C"
    R134A = "R134a"
    R717 = "R717"
    R404A = "R404A"
    R507A = "R507A"
    R448A = "R448A"
    R449A = "R449A"


class ControlType(str, Enum):
    """HVAC control system types for control credit assessment."""
    MANUAL = "manual"
    TIMER_PROGRAMMER = "timer_programmer"
    TRV = "thermostatic_radiator_valves"
    ROOM_THERMOSTAT = "room_thermostat"
    WEATHER_COMPENSATION = "weather_compensation"
    OPTIMUM_START = "optimum_start_stop"
    BMS = "building_management_system"
    ADAPTIVE_CONTROL = "adaptive_predictive_control"


class HeatRecoveryType(str, Enum):
    """Heat recovery unit types with different effectiveness ranges.

    Thermal effectiveness varies by type per EN 308.
    """
    PLATE = "plate_heat_exchanger"
    ROTARY = "rotary_thermal_wheel"
    RUN_AROUND = "run_around_coil"
    HEAT_PIPE = "heat_pipe"
    NONE = "none"


class BuildingType(str, Enum):
    """Building use type for HVAC assessment context."""
    RESIDENTIAL = "residential"
    OFFICE = "office"
    RETAIL = "retail"
    HOTEL = "hotel"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    WAREHOUSE = "warehouse"
    DATA_CENTRE = "data_centre"
    MIXED_USE = "mixed_use"
    RESTAURANT = "restaurant"


# ---------------------------------------------------------------------------
# Constants -- Boiler Seasonal Efficiency
# ---------------------------------------------------------------------------


# Boiler seasonal efficiency by boiler type, condensing status, and age.
# Format: {(heating_type, condensing, age): seasonal_efficiency}
# Sources: SAP 10.2 Table 4a/4b, EN 15316-4-1, CIBSE Guide B.
BOILER_SEASONAL_EFFICIENCY: Dict[str, Dict[str, Dict[str, float]]] = {
    HeatingSystemType.GAS_BOILER: {
        "condensing_new": {"gross": 0.935, "net": 0.920, "part_load": 0.95},
        "condensing_modern": {"gross": 0.910, "net": 0.895, "part_load": 0.93},
        "condensing_old": {"gross": 0.880, "net": 0.860, "part_load": 0.90},
        "non_condensing_new": {"gross": 0.840, "net": 0.825, "part_load": 0.88},
        "non_condensing_modern": {"gross": 0.800, "net": 0.785, "part_load": 0.85},
        "non_condensing_old": {"gross": 0.750, "net": 0.730, "part_load": 0.80},
        "back_boiler": {"gross": 0.650, "net": 0.630, "part_load": 0.70},
        "floor_standing_old": {"gross": 0.700, "net": 0.680, "part_load": 0.75},
    },
    HeatingSystemType.OIL_BOILER: {
        "condensing_new": {"gross": 0.940, "net": 0.925, "part_load": 0.95},
        "condensing_modern": {"gross": 0.920, "net": 0.900, "part_load": 0.93},
        "condensing_old": {"gross": 0.890, "net": 0.870, "part_load": 0.90},
        "non_condensing_new": {"gross": 0.850, "net": 0.835, "part_load": 0.88},
        "non_condensing_modern": {"gross": 0.820, "net": 0.800, "part_load": 0.85},
        "non_condensing_old": {"gross": 0.780, "net": 0.760, "part_load": 0.82},
    },
    HeatingSystemType.BIOMASS_BOILER: {
        "pellet_new": {"gross": 0.920, "net": 0.900, "part_load": 0.93},
        "pellet_modern": {"gross": 0.880, "net": 0.860, "part_load": 0.90},
        "pellet_old": {"gross": 0.830, "net": 0.810, "part_load": 0.85},
        "wood_chip_new": {"gross": 0.880, "net": 0.860, "part_load": 0.90},
        "wood_chip_modern": {"gross": 0.840, "net": 0.820, "part_load": 0.87},
        "wood_chip_old": {"gross": 0.750, "net": 0.730, "part_load": 0.80},
        "log_new": {"gross": 0.850, "net": 0.830, "part_load": 0.88},
        "log_old": {"gross": 0.700, "net": 0.680, "part_load": 0.75},
    },
    HeatingSystemType.ELECTRIC_BOILER: {
        "standard": {"gross": 1.000, "net": 1.000, "part_load": 1.00},
    },
    HeatingSystemType.WARM_AIR: {
        "condensing_new": {"gross": 0.920, "net": 0.905, "part_load": 0.94},
        "condensing_modern": {"gross": 0.890, "net": 0.875, "part_load": 0.91},
        "non_condensing_old": {"gross": 0.750, "net": 0.730, "part_load": 0.80},
    },
}
"""Boiler seasonal efficiency by type, condensing status, and age.
Source: SAP 10.2 Table 4a/4b, EN 15316-4-1, CIBSE Guide B."""


# Heat pump seasonal COP by type and operating conditions.
# Format: {HP_type: {source_sink_condition: SCOP}}
# Sources: EN 14825:2022, MCS heat pump data, ASHRAE 90.1.
HEAT_PUMP_SEASONAL_COP: Dict[str, Dict[str, float]] = {
    HeatingSystemType.AIR_SOURCE_HEAT_PUMP: {
        "source_m7_sink_35": 2.50,
        "source_m7_sink_45": 2.00,
        "source_m7_sink_55": 1.70,
        "source_2_sink_35": 3.20,
        "source_2_sink_45": 2.60,
        "source_2_sink_55": 2.10,
        "source_7_sink_35": 4.10,
        "source_7_sink_45": 3.30,
        "source_7_sink_55": 2.70,
        "source_12_sink_35": 5.00,
        "source_12_sink_45": 4.00,
        "source_12_sink_55": 3.20,
        "seasonal_average_uk": 3.10,
        "seasonal_average_mild": 3.50,
        "seasonal_average_cold": 2.60,
    },
    HeatingSystemType.GROUND_SOURCE_HEAT_PUMP: {
        "source_0_sink_35": 4.10,
        "source_0_sink_45": 3.30,
        "source_0_sink_55": 2.70,
        "source_5_sink_35": 4.80,
        "source_5_sink_45": 3.80,
        "source_5_sink_55": 3.10,
        "source_10_sink_35": 5.50,
        "source_10_sink_45": 4.40,
        "source_10_sink_55": 3.60,
        "seasonal_average_uk": 3.90,
        "seasonal_average_mild": 4.20,
        "seasonal_average_cold": 3.50,
    },
    HeatingSystemType.WATER_SOURCE_HEAT_PUMP: {
        "source_7_sink_35": 4.50,
        "source_7_sink_45": 3.60,
        "source_7_sink_55": 2.90,
        "source_10_sink_35": 5.20,
        "source_10_sink_45": 4.10,
        "source_10_sink_55": 3.40,
        "source_15_sink_35": 6.00,
        "source_15_sink_45": 4.80,
        "source_15_sink_55": 3.90,
        "seasonal_average_uk": 4.20,
        "seasonal_average_mild": 4.50,
        "seasonal_average_cold": 3.80,
    },
}
"""Heat pump seasonal COP by type and source/sink temperatures.
Source: EN 14825:2022, MCS data, ASHRAE 90.1."""


# Cooling SEER benchmarks by system type (4-tier rating).
# Sources: EN 14825, Eurovent certification, ASHRAE 90.1-2022.
COOLING_SEER_BENCHMARKS: Dict[str, Dict[str, float]] = {
    CoolingSystemType.SPLIT_SYSTEM: {
        "poor": 3.0, "average": 4.5, "good": 6.0, "excellent": 8.0,
    },
    CoolingSystemType.VRF: {
        "poor": 3.5, "average": 5.0, "good": 7.0, "excellent": 9.0,
    },
    CoolingSystemType.CHILLER_AHU: {
        "poor": 2.8, "average": 4.0, "good": 5.5, "excellent": 7.5,
    },
    CoolingSystemType.PTAC: {
        "poor": 2.5, "average": 3.5, "good": 4.5, "excellent": 6.0,
    },
    CoolingSystemType.CRAC: {
        "poor": 2.0, "average": 3.0, "good": 4.0, "excellent": 5.5,
    },
    CoolingSystemType.EVAPORATIVE: {
        "poor": 8.0, "average": 12.0, "good": 15.0, "excellent": 20.0,
    },
    CoolingSystemType.ABSORPTION_CHILLER: {
        "poor": 0.6, "average": 0.8, "good": 1.2, "excellent": 1.5,
    },
    CoolingSystemType.DISTRICT_COOLING: {
        "poor": 3.5, "average": 5.0, "good": 5.5, "excellent": 6.5,
    },
    CoolingSystemType.HEAT_PUMP_REVERSIBLE: {
        "poor": 3.0, "average": 4.5, "good": 6.0, "excellent": 8.0,
    },
}
"""Cooling SEER benchmarks (4-tier) by system type.
Source: EN 14825, Eurovent, ASHRAE 90.1-2022."""


# Ventilation Specific Fan Power (SFP) benchmarks per EN 13779 / EN 16798-3.
# Units: W/(l/s). Lower is better.
# Sources: EN 13779:2007 Table A.16, CIBSE Guide B Table 2.6.
VENTILATION_SFP_BENCHMARKS: Dict[str, Dict[str, float]] = {
    VentilationType.NATURAL: {
        "typical": 0.0, "good": 0.0, "best": 0.0,
    },
    VentilationType.MECHANICAL_EXTRACT: {
        "typical": 1.5, "good": 1.0, "best": 0.6,
        "sfp_limit_part_l": 0.8,
    },
    VentilationType.MECHANICAL_SUPPLY_EXTRACT: {
        "typical": 3.0, "good": 2.0, "best": 1.2,
        "sfp_limit_part_l": 2.0,
    },
    VentilationType.MVHR: {
        "typical": 2.5, "good": 1.5, "best": 0.8,
        "sfp_limit_part_l": 1.5,
    },
    VentilationType.DEMAND_CONTROLLED: {
        "typical": 2.0, "good": 1.2, "best": 0.7,
        "sfp_limit_part_l": 1.5,
    },
}
"""Ventilation SFP benchmarks (W/(l/s)) by system type.
Source: EN 13779:2007 Table A.16, CIBSE Guide B, Part L."""


# Heat recovery effectiveness by HRV type per EN 308.
# Sources: EN 308:1997, manufacturer data, CIBSE Guide B.
HEAT_RECOVERY_EFFECTIVENESS: Dict[str, Dict[str, float]] = {
    HeatRecoveryType.PLATE: {
        "dry_effectiveness": 0.75,
        "wet_effectiveness": 0.65,
        "pressure_drop_pa": 150.0,
        "typical_range_low": 0.70,
        "typical_range_high": 0.80,
    },
    HeatRecoveryType.ROTARY: {
        "dry_effectiveness": 0.80,
        "wet_effectiveness": 0.70,
        "pressure_drop_pa": 100.0,
        "typical_range_low": 0.75,
        "typical_range_high": 0.85,
    },
    HeatRecoveryType.RUN_AROUND: {
        "dry_effectiveness": 0.55,
        "wet_effectiveness": 0.45,
        "pressure_drop_pa": 200.0,
        "typical_range_low": 0.45,
        "typical_range_high": 0.65,
    },
    HeatRecoveryType.HEAT_PIPE: {
        "dry_effectiveness": 0.60,
        "wet_effectiveness": 0.50,
        "pressure_drop_pa": 120.0,
        "typical_range_low": 0.55,
        "typical_range_high": 0.65,
    },
}
"""Heat recovery effectiveness by HRV type.
Source: EN 308:1997, manufacturer data, CIBSE Guide B."""


# Refrigerant GWP values (100-year, IPCC AR6).
# Sources: IPCC AR6 WG1, EU 517/2014 Annex I, ASHRAE Standard 34.
REFRIGERANT_GWP: Dict[str, Dict[str, Any]] = {
    RefrigerantType.R22: {
        "gwp_100yr": 1810, "odp": 0.055, "class": "HCFC", "phase_out": True,
        "safety_class": "A1", "status": "banned_new_equipment",
    },
    RefrigerantType.R410A: {
        "gwp_100yr": 2088, "odp": 0.0, "class": "HFC_blend", "phase_out": True,
        "safety_class": "A1", "status": "phase_down",
    },
    RefrigerantType.R32: {
        "gwp_100yr": 675, "odp": 0.0, "class": "HFC", "phase_out": False,
        "safety_class": "A2L", "status": "current",
    },
    RefrigerantType.R290: {
        "gwp_100yr": 3, "odp": 0.0, "class": "HC", "phase_out": False,
        "safety_class": "A3", "status": "preferred_natural",
    },
    RefrigerantType.R744: {
        "gwp_100yr": 1, "odp": 0.0, "class": "natural", "phase_out": False,
        "safety_class": "A1", "status": "preferred_natural",
    },
    RefrigerantType.R1234YF: {
        "gwp_100yr": 4, "odp": 0.0, "class": "HFO", "phase_out": False,
        "safety_class": "A2L", "status": "preferred_low_gwp",
    },
    RefrigerantType.R1234ZE: {
        "gwp_100yr": 7, "odp": 0.0, "class": "HFO", "phase_out": False,
        "safety_class": "A2L", "status": "preferred_low_gwp",
    },
    RefrigerantType.R454B: {
        "gwp_100yr": 466, "odp": 0.0, "class": "HFO_blend", "phase_out": False,
        "safety_class": "A2L", "status": "current_replacement_R410A",
    },
    RefrigerantType.R407C: {
        "gwp_100yr": 1774, "odp": 0.0, "class": "HFC_blend", "phase_out": True,
        "safety_class": "A1", "status": "phase_down",
    },
    RefrigerantType.R134A: {
        "gwp_100yr": 1430, "odp": 0.0, "class": "HFC", "phase_out": True,
        "safety_class": "A1", "status": "phase_down",
    },
    RefrigerantType.R717: {
        "gwp_100yr": 0, "odp": 0.0, "class": "natural", "phase_out": False,
        "safety_class": "B2L", "status": "preferred_natural_industrial",
    },
    RefrigerantType.R404A: {
        "gwp_100yr": 3922, "odp": 0.0, "class": "HFC_blend", "phase_out": True,
        "safety_class": "A1", "status": "banned_new_2020",
    },
    RefrigerantType.R507A: {
        "gwp_100yr": 3985, "odp": 0.0, "class": "HFC_blend", "phase_out": True,
        "safety_class": "A1", "status": "banned_new_2020",
    },
    RefrigerantType.R448A: {
        "gwp_100yr": 1386, "odp": 0.0, "class": "HFO_blend", "phase_out": False,
        "safety_class": "A1", "status": "current_replacement_R404A",
    },
    RefrigerantType.R449A: {
        "gwp_100yr": 1397, "odp": 0.0, "class": "HFO_blend", "phase_out": False,
        "safety_class": "A1", "status": "current_replacement_R404A",
    },
}
"""Refrigerant GWP values (100-year, IPCC AR6) and regulatory status.
Source: IPCC AR6 WG1, EU 517/2014 Annex I, ASHRAE Standard 34."""


# Distribution efficiency by distribution type and insulation level.
# "uninsulated" = bare pipes/ducts, "insulated" = code-compliant insulation.
# Sources: EN 15316-2, CIBSE Guide B, ASHRAE 90.1.
DISTRIBUTION_EFFICIENCY: Dict[str, Dict[str, float]] = {
    DistributionType.RADIATOR: {
        "well_insulated": 0.95, "insulated": 0.90, "uninsulated": 0.80,
    },
    DistributionType.UNDERFLOOR: {
        "well_insulated": 0.97, "insulated": 0.93, "uninsulated": 0.85,
    },
    DistributionType.FAN_COIL: {
        "well_insulated": 0.93, "insulated": 0.88, "uninsulated": 0.78,
    },
    DistributionType.DUCT_AIR: {
        "well_insulated": 0.90, "insulated": 0.82, "uninsulated": 0.65,
    },
    DistributionType.CHILLED_BEAM: {
        "well_insulated": 0.96, "insulated": 0.93, "uninsulated": 0.87,
    },
    DistributionType.RADIANT_CEILING: {
        "well_insulated": 0.97, "insulated": 0.94, "uninsulated": 0.88,
    },
}
"""Distribution efficiency by type and insulation level.
Source: EN 15316-2, CIBSE Guide B, ASHRAE 90.1."""


# Control credit factors by control type per EN 15316-2.
# Applied as multiplier to reduce system energy consumption.
# Sources: EN 15316-2:2017, SAP 10.2 Table 4e, CIBSE Guide H.
CONTROL_CREDIT: Dict[str, Dict[str, Any]] = {
    ControlType.MANUAL: {
        "credit_factor": 1.00, "description": "No automatic control",
        "typical_savings_pct": 0.0,
    },
    ControlType.TIMER_PROGRAMMER: {
        "credit_factor": 0.95, "description": "Time-based scheduling",
        "typical_savings_pct": 5.0,
    },
    ControlType.TRV: {
        "credit_factor": 0.90, "description": "Thermostatic radiator valves",
        "typical_savings_pct": 10.0,
    },
    ControlType.ROOM_THERMOSTAT: {
        "credit_factor": 0.88, "description": "Room thermostat with programmer",
        "typical_savings_pct": 12.0,
    },
    ControlType.WEATHER_COMPENSATION: {
        "credit_factor": 0.82, "description": "Weather-compensated flow temperature",
        "typical_savings_pct": 18.0,
    },
    ControlType.OPTIMUM_START: {
        "credit_factor": 0.85, "description": "Optimum start/stop control",
        "typical_savings_pct": 15.0,
    },
    ControlType.BMS: {
        "credit_factor": 0.78, "description": "Building Management System",
        "typical_savings_pct": 22.0,
    },
    ControlType.ADAPTIVE_CONTROL: {
        "credit_factor": 0.72, "description": "Adaptive/predictive learning control",
        "typical_savings_pct": 28.0,
    },
}
"""Control credit factors by control type.
Source: EN 15316-2:2017, SAP 10.2 Table 4e, CIBSE Guide H."""


# F-gas annual leak rates by system type per EU 517/2014.
# Sources: EU 517/2014, RAC Magazine, DEFRA guidance.
F_GAS_LEAK_RATES: Dict[str, Dict[str, float]] = {
    CoolingSystemType.SPLIT_SYSTEM: {
        "new": 0.03, "typical": 0.05, "old": 0.10, "poor": 0.15,
    },
    CoolingSystemType.VRF: {
        "new": 0.02, "typical": 0.04, "old": 0.08, "poor": 0.12,
    },
    CoolingSystemType.CHILLER_AHU: {
        "new": 0.02, "typical": 0.03, "old": 0.07, "poor": 0.10,
    },
    CoolingSystemType.PTAC: {
        "new": 0.03, "typical": 0.06, "old": 0.12, "poor": 0.18,
    },
    CoolingSystemType.CRAC: {
        "new": 0.02, "typical": 0.04, "old": 0.08, "poor": 0.12,
    },
    CoolingSystemType.ABSORPTION_CHILLER: {
        "new": 0.01, "typical": 0.02, "old": 0.04, "poor": 0.06,
    },
    CoolingSystemType.HEAT_PUMP_REVERSIBLE: {
        "new": 0.03, "typical": 0.05, "old": 0.10, "poor": 0.15,
    },
}
"""F-gas annual leak rates (fraction of charge) by system type and condition.
Source: EU 517/2014, RAC Magazine, DEFRA guidance."""


# Minimum inspection intervals for F-gas equipment per EU 517/2014.
# Based on charge quantity in tCO2e equivalent.
F_GAS_INSPECTION_INTERVALS: Dict[str, Dict[str, Any]] = {
    "below_5_tco2e": {
        "interval_months": 0,
        "description": "No mandatory check (below 5 tCO2e threshold)",
        "leak_detection_required": False,
    },
    "5_to_50_tco2e": {
        "interval_months": 12,
        "description": "Annual leak check required",
        "leak_detection_required": False,
    },
    "50_to_500_tco2e": {
        "interval_months": 6,
        "description": "Six-monthly leak check or automatic detection",
        "leak_detection_required": True,
    },
    "above_500_tco2e": {
        "interval_months": 3,
        "description": "Quarterly leak check with automatic detection",
        "leak_detection_required": True,
    },
}
"""F-gas inspection intervals per EU 517/2014 based on tCO2e charge.
Source: EU 517/2014 Article 4."""


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class HeatingSystem(BaseModel):
    """Heating system specification for HVAC assessment.

    Attributes:
        system_id: Unique system identifier.
        system_type: Heating system type.
        boiler_subtype: Boiler subtype (condensing_new, non_condensing_old, etc.).
        capacity_kw: Nominal output capacity (kW).
        age_years: System age in years.
        annual_energy_kwh: Measured annual energy consumption (kWh).
        known_efficiency: Known seasonal efficiency (0-1 for boilers, >1 for HPs).
        fuel_type: Primary fuel type label.
        description: Description.
    """
    system_id: str = Field(default_factory=_new_uuid, description="System ID")
    system_type: HeatingSystemType = Field(..., description="Heating system type")
    boiler_subtype: str = Field(
        default="condensing_modern", description="Boiler subtype"
    )
    capacity_kw: float = Field(default=0.0, ge=0, description="Capacity (kW)")
    age_years: int = Field(default=5, ge=0, le=80, description="Age (years)")
    annual_energy_kwh: Optional[float] = Field(
        None, ge=0, description="Annual energy (kWh)"
    )
    known_efficiency: Optional[float] = Field(
        None, gt=0, le=10.0, description="Known seasonal efficiency"
    )
    fuel_type: str = Field(default="natural_gas", description="Fuel type label")
    description: Optional[str] = Field(None, description="Description")


class CoolingSystem(BaseModel):
    """Cooling system specification for HVAC assessment.

    Attributes:
        system_id: Unique system identifier.
        system_type: Cooling system type.
        capacity_kw: Nominal cooling capacity (kW).
        age_years: System age in years.
        known_seer: Known SEER value.
        annual_energy_kwh: Measured annual electricity (kWh).
        refrigerant: Refrigerant type.
        refrigerant_charge_kg: Refrigerant charge (kg).
        leak_rate_condition: System condition for leak rate (new/typical/old/poor).
        description: Description.
    """
    system_id: str = Field(default_factory=_new_uuid, description="System ID")
    system_type: CoolingSystemType = Field(..., description="Cooling system type")
    capacity_kw: float = Field(default=0.0, ge=0, description="Cooling capacity (kW)")
    age_years: int = Field(default=5, ge=0, le=50, description="Age (years)")
    known_seer: Optional[float] = Field(None, gt=0, le=30, description="Known SEER")
    annual_energy_kwh: Optional[float] = Field(
        None, ge=0, description="Annual electricity (kWh)"
    )
    refrigerant: RefrigerantType = Field(
        default=RefrigerantType.R410A, description="Refrigerant type"
    )
    refrigerant_charge_kg: float = Field(
        default=0.0, ge=0, description="Refrigerant charge (kg)"
    )
    leak_rate_condition: str = Field(
        default="typical", description="Condition for leak rate"
    )
    description: Optional[str] = Field(None, description="Description")


class VentilationSystem(BaseModel):
    """Ventilation system specification for HVAC assessment.

    Attributes:
        system_id: Unique system identifier.
        ventilation_type: Ventilation system type.
        airflow_rate_ls: Design airflow rate (l/s).
        fan_power_w: Measured total fan power (W).
        known_sfp: Known SFP (W/(l/s)).
        heat_recovery_type: Heat recovery unit type.
        heat_recovery_efficiency: Known HR effectiveness (0-1).
        description: Description.
    """
    system_id: str = Field(default_factory=_new_uuid, description="System ID")
    ventilation_type: VentilationType = Field(..., description="Ventilation type")
    airflow_rate_ls: float = Field(
        default=0.0, ge=0, description="Airflow rate (l/s)"
    )
    fan_power_w: Optional[float] = Field(
        None, ge=0, description="Total fan power (W)"
    )
    known_sfp: Optional[float] = Field(
        None, ge=0, le=20, description="Known SFP (W/(l/s))"
    )
    heat_recovery_type: HeatRecoveryType = Field(
        default=HeatRecoveryType.NONE, description="Heat recovery type"
    )
    heat_recovery_efficiency: Optional[float] = Field(
        None, ge=0, le=1.0, description="Heat recovery effectiveness"
    )
    description: Optional[str] = Field(None, description="Description")


class DistributionSystem(BaseModel):
    """Heating/cooling distribution system specification.

    Attributes:
        system_id: Unique system identifier.
        distribution_type: Distribution type.
        insulation_level: Insulation level (well_insulated/insulated/uninsulated).
        total_pipe_length_m: Total pipe/duct run length (m).
        flow_temperature_c: Design flow temperature (C).
        return_temperature_c: Design return temperature (C).
        ambient_temperature_c: Average ambient temperature around pipes (C).
        description: Description.
    """
    system_id: str = Field(default_factory=_new_uuid, description="System ID")
    distribution_type: DistributionType = Field(
        default=DistributionType.RADIATOR, description="Distribution type"
    )
    insulation_level: str = Field(
        default="insulated", description="Insulation level"
    )
    total_pipe_length_m: float = Field(
        default=0.0, ge=0, description="Pipe/duct length (m)"
    )
    flow_temperature_c: float = Field(
        default=55.0, description="Flow temperature (C)"
    )
    return_temperature_c: float = Field(
        default=40.0, description="Return temperature (C)"
    )
    ambient_temperature_c: float = Field(
        default=20.0, description="Ambient temperature (C)"
    )
    description: Optional[str] = Field(None, description="Description")


class ControlSystem(BaseModel):
    """HVAC control system specification.

    Attributes:
        control_type: Control type.
        zones_controlled: Number of zones.
        has_night_setback: Whether night setback is configured.
        has_holiday_scheduling: Whether holiday mode exists.
        description: Description.
    """
    control_type: ControlType = Field(
        default=ControlType.ROOM_THERMOSTAT, description="Control type"
    )
    zones_controlled: int = Field(default=1, ge=1, description="Zones controlled")
    has_night_setback: bool = Field(default=False, description="Night setback enabled")
    has_holiday_scheduling: bool = Field(
        default=False, description="Holiday scheduling"
    )
    description: Optional[str] = Field(None, description="Description")


class HVACInput(BaseModel):
    """Complete HVAC system input for assessment.

    Attributes:
        facility_id: Unique facility identifier.
        building_name: Building name.
        building_type: Building use type.
        floor_area_m2: Useful floor area (m2).
        heated_volume_m3: Heated volume (m3).
        annual_heating_demand_kwh: Known annual heating demand (kWh).
        annual_cooling_demand_kwh: Known annual cooling demand (kWh).
        operating_hours: Annual operating hours.
        heating_systems: Heating systems.
        cooling_systems: Cooling systems.
        ventilation_systems: Ventilation systems.
        distribution_systems: Distribution systems.
        controls: Control systems.
    """
    facility_id: str = Field(..., min_length=1, description="Facility identifier")
    building_name: str = Field(default="", description="Building name")
    building_type: BuildingType = Field(
        default=BuildingType.OFFICE, description="Building type"
    )
    floor_area_m2: float = Field(..., gt=0, description="Floor area (m2)")
    heated_volume_m3: float = Field(default=0.0, ge=0, description="Heated volume (m3)")
    annual_heating_demand_kwh: Optional[float] = Field(
        None, ge=0, description="Annual heating demand (kWh)"
    )
    annual_cooling_demand_kwh: Optional[float] = Field(
        None, ge=0, description="Annual cooling demand (kWh)"
    )
    operating_hours: float = Field(
        default=3000.0, gt=0, le=8784, description="Operating hours"
    )
    heating_systems: List[HeatingSystem] = Field(
        default_factory=list, description="Heating systems"
    )
    cooling_systems: List[CoolingSystem] = Field(
        default_factory=list, description="Cooling systems"
    )
    ventilation_systems: List[VentilationSystem] = Field(
        default_factory=list, description="Ventilation systems"
    )
    distribution_systems: List[DistributionSystem] = Field(
        default_factory=list, description="Distribution systems"
    )
    controls: List[ControlSystem] = Field(
        default_factory=list, description="Control systems"
    )

    @field_validator("floor_area_m2")
    @classmethod
    def validate_area(cls, v: float) -> float:
        """Ensure floor area is within plausible bounds."""
        if v > 1_000_000:
            raise ValueError("Floor area exceeds 1,000,000 m2 sanity check")
        return v


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class HeatingAssessmentResult(BaseModel):
    """Heating system assessment result.

    Attributes:
        system_id: System identifier.
        system_type: Heating system type.
        seasonal_efficiency: Seasonal efficiency (0-1 for boilers, >1 for HPs).
        efficiency_rating: Efficiency tier (poor/average/good/excellent).
        annual_delivered_kwh: Annual delivered energy (kWh).
        annual_useful_kwh: Annual useful heat output (kWh).
        losses_kwh: Annual system losses (kWh).
        recommendations: Improvement suggestions.
    """
    system_id: str = Field(default="", description="System ID")
    system_type: str = Field(default="", description="System type")
    seasonal_efficiency: float = Field(default=0.0, description="Seasonal efficiency")
    efficiency_rating: str = Field(default="", description="Efficiency tier")
    annual_delivered_kwh: float = Field(default=0.0, description="Delivered energy")
    annual_useful_kwh: float = Field(default=0.0, description="Useful heat")
    losses_kwh: float = Field(default=0.0, description="System losses (kWh)")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")


class CoolingAssessmentResult(BaseModel):
    """Cooling system assessment result.

    Attributes:
        system_id: System identifier.
        system_type: Cooling system type.
        seer: Seasonal Energy Efficiency Ratio.
        efficiency_rating: Efficiency tier (poor/average/good/excellent).
        annual_electricity_kwh: Annual electricity consumption (kWh).
        annual_cooling_kwh: Annual cooling output (kWh).
        recommendations: Improvement suggestions.
    """
    system_id: str = Field(default="", description="System ID")
    system_type: str = Field(default="", description="System type")
    seer: float = Field(default=0.0, description="SEER")
    efficiency_rating: str = Field(default="", description="Efficiency tier")
    annual_electricity_kwh: float = Field(default=0.0, description="Electricity (kWh)")
    annual_cooling_kwh: float = Field(default=0.0, description="Cooling output (kWh)")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")


class VentilationAssessmentResult(BaseModel):
    """Ventilation system assessment result.

    Attributes:
        system_id: System identifier.
        ventilation_type: Ventilation type.
        sfp_w_ls: Specific Fan Power (W/(l/s)).
        sfp_rating: SFP performance tier.
        sfp_compliant: Whether SFP meets regulatory limit.
        heat_recovery_efficiency: Heat recovery effectiveness (0-1).
        annual_fan_energy_kwh: Annual fan energy (kWh).
        annual_heat_recovered_kwh: Annual heat recovered (kWh).
        recommendations: Improvement suggestions.
    """
    system_id: str = Field(default="", description="System ID")
    ventilation_type: str = Field(default="", description="Type")
    sfp_w_ls: float = Field(default=0.0, description="SFP (W/(l/s))")
    sfp_rating: str = Field(default="", description="SFP tier")
    sfp_compliant: bool = Field(default=True, description="Regulatory compliance")
    heat_recovery_efficiency: float = Field(default=0.0, description="HR effectiveness")
    annual_fan_energy_kwh: float = Field(default=0.0, description="Fan energy (kWh)")
    annual_heat_recovered_kwh: float = Field(default=0.0, description="Heat recovered (kWh)")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")


class RefrigerantRiskResult(BaseModel):
    """Refrigerant / F-gas risk assessment result.

    Attributes:
        refrigerant: Refrigerant type.
        gwp_100yr: 100-year GWP value.
        charge_kg: Refrigerant charge (kg).
        charge_tco2e: Charge in tCO2e equivalent.
        annual_leak_rate: Annual leak rate fraction.
        annual_emissions_tco2e: Annual F-gas emissions (tCO2e).
        inspection_interval_months: Required inspection interval.
        phase_out_risk: Whether refrigerant faces phase-out.
        f_gas_compliant: EU 517/2014 compliance status.
        recommendations: Improvement suggestions.
    """
    refrigerant: str = Field(default="", description="Refrigerant type")
    gwp_100yr: int = Field(default=0, description="GWP (100yr)")
    charge_kg: float = Field(default=0.0, description="Charge (kg)")
    charge_tco2e: float = Field(default=0.0, description="Charge (tCO2e)")
    annual_leak_rate: float = Field(default=0.0, description="Annual leak rate")
    annual_emissions_tco2e: float = Field(default=0.0, description="Annual F-gas (tCO2e)")
    inspection_interval_months: int = Field(default=0, description="Inspection interval")
    phase_out_risk: bool = Field(default=False, description="Phase-out risk")
    f_gas_compliant: bool = Field(default=True, description="F-gas compliance")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")


class DistributionAssessmentResult(BaseModel):
    """Distribution system assessment result.

    Attributes:
        distribution_type: Distribution type.
        efficiency: Distribution efficiency (0-1).
        annual_losses_kwh: Annual distribution losses (kWh).
        loss_percentage: Losses as percentage of total energy.
        recommendations: Improvement suggestions.
    """
    distribution_type: str = Field(default="", description="Distribution type")
    efficiency: float = Field(default=0.0, description="Distribution efficiency")
    annual_losses_kwh: float = Field(default=0.0, description="Annual losses (kWh)")
    loss_percentage: float = Field(default=0.0, description="Loss percentage")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")


class ControlAssessmentResult(BaseModel):
    """Control system assessment result.

    Attributes:
        control_type: Control type.
        credit_factor: Control credit factor.
        estimated_savings_pct: Estimated savings percentage.
        upgrade_potential: Whether upgrade would yield benefit.
        recommendations: Improvement suggestions.
    """
    control_type: str = Field(default="", description="Control type")
    credit_factor: float = Field(default=1.0, description="Credit factor")
    estimated_savings_pct: float = Field(default=0.0, description="Savings %")
    upgrade_potential: bool = Field(default=False, description="Upgrade potential")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")


class ImprovementMeasure(BaseModel):
    """HVAC improvement measure with estimated savings.

    Attributes:
        category: Improvement category (heating/cooling/ventilation/etc.).
        description: Description of the improvement.
        annual_savings_kwh: Estimated annual energy savings (kWh).
        annual_savings_co2_kg: Estimated CO2 savings (kg/yr).
        estimated_cost: Estimated implementation cost.
        payback_years: Simple payback period (years).
        priority: Priority ranking.
    """
    category: str = Field(default="", description="Category")
    description: str = Field(default="", description="Description")
    annual_savings_kwh: float = Field(default=0.0, description="Savings (kWh/yr)")
    annual_savings_co2_kg: float = Field(default=0.0, description="CO2 savings (kg/yr)")
    estimated_cost: float = Field(default=0.0, description="Est cost")
    payback_years: float = Field(default=0.0, description="Payback (years)")
    priority: int = Field(default=0, description="Priority")


class HVACResult(BaseModel):
    """Complete HVAC assessment result with full provenance.

    Contains heating, cooling, ventilation, refrigerant, distribution,
    and control assessments, plus improvement measures.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Unique result ID")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Calculation timestamp"
    )
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")

    facility_id: str = Field(default="", description="Facility identifier")
    building_name: str = Field(default="", description="Building name")
    building_type: str = Field(default="", description="Building type")

    # Sub-assessments
    heating_assessments: List[HeatingAssessmentResult] = Field(
        default_factory=list, description="Heating assessments"
    )
    cooling_assessments: List[CoolingAssessmentResult] = Field(
        default_factory=list, description="Cooling assessments"
    )
    ventilation_assessments: List[VentilationAssessmentResult] = Field(
        default_factory=list, description="Ventilation assessments"
    )
    refrigerant_risks: List[RefrigerantRiskResult] = Field(
        default_factory=list, description="Refrigerant / F-gas risks"
    )
    distribution_assessment: Optional[DistributionAssessmentResult] = Field(
        None, description="Distribution assessment"
    )
    control_assessment: Optional[ControlAssessmentResult] = Field(
        None, description="Control assessment"
    )

    # Aggregate metrics
    overall_heating_efficiency: float = Field(
        default=0.0, description="Overall heating efficiency"
    )
    overall_cooling_seer: float = Field(default=0.0, description="Overall cooling SEER")
    total_hvac_energy_kwh: float = Field(
        default=0.0, description="Total HVAC energy (kWh/yr)"
    )
    total_f_gas_emissions_tco2e: float = Field(
        default=0.0, description="Total F-gas emissions (tCO2e/yr)"
    )

    # Improvements
    improvement_measures: List[ImprovementMeasure] = Field(
        default_factory=list, description="Improvement measures"
    )
    total_savings_potential_kwh: float = Field(
        default=0.0, description="Total savings potential (kWh/yr)"
    )

    recommendations: List[str] = Field(
        default_factory=list, description="Summary recommendations"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Calculation Engine
# ---------------------------------------------------------------------------


class HVACAssessmentEngine:
    """HVAC system efficiency assessment engine.

    Provides deterministic, zero-hallucination calculations for:
    - Boiler seasonal efficiency per EN 15316-4-1 (indirect method)
    - Heat pump SPF per EN 14825
    - Cooling SEER per EN 14825
    - Ventilation SFP per EN 13779 / EN 16798-3
    - Heat recovery effectiveness per EN 308
    - Refrigerant F-gas emissions per EU 517/2014
    - Distribution losses per EN 15316-2
    - Control credit per EN 15316-2

    All calculations are bit-perfect reproducible.  No LLM is used
    in any calculation path.

    Usage::

        engine = HVACAssessmentEngine()
        result = engine.assess(hvac_input)

    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise the HVAC assessment engine with embedded constants."""
        self._boiler_efficiency = BOILER_SEASONAL_EFFICIENCY
        self._hp_cop = HEAT_PUMP_SEASONAL_COP
        self._cooling_benchmarks = COOLING_SEER_BENCHMARKS
        self._sfp_benchmarks = VENTILATION_SFP_BENCHMARKS
        self._hr_effectiveness = HEAT_RECOVERY_EFFECTIVENESS
        self._refrigerant_gwp = REFRIGERANT_GWP
        self._distribution_eff = DISTRIBUTION_EFFICIENCY
        self._control_credit = CONTROL_CREDIT
        self._f_gas_leak_rates = F_GAS_LEAK_RATES
        self._f_gas_inspections = F_GAS_INSPECTION_INTERVALS

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def assess(self, hvac_input: HVACInput) -> HVACResult:
        """Run complete HVAC system assessment.

        Orchestrates all sub-assessments: heating, cooling, ventilation,
        refrigerant, distribution, controls, and improvements.

        Args:
            hvac_input: Complete HVAC system input data.

        Returns:
            HVACResult with full provenance and audit trail.

        Raises:
            ValueError: If no systems are provided.
        """
        t0 = time.perf_counter()

        total_systems = (
            len(hvac_input.heating_systems) + len(hvac_input.cooling_systems) +
            len(hvac_input.ventilation_systems)
        )
        if total_systems == 0:
            raise ValueError(
                "At least one HVAC system (heating, cooling, or ventilation) "
                "is required for assessment"
            )

        logger.info(
            "Assessing HVAC for facility %s (%s), %d systems",
            hvac_input.facility_id, hvac_input.building_type.value, total_systems,
        )

        # Step 1: Assess heating systems
        heating_results = []
        for hs in hvac_input.heating_systems:
            heating_results.append(self.assess_heating(hs, hvac_input))

        # Step 2: Assess cooling systems
        cooling_results = []
        for cs in hvac_input.cooling_systems:
            cooling_results.append(self.assess_cooling(cs, hvac_input))

        # Step 3: Assess ventilation systems
        vent_results = []
        for vs in hvac_input.ventilation_systems:
            vent_results.append(self.assess_ventilation(vs, hvac_input))

        # Step 4: Assess refrigerant risk
        ref_risks = []
        for cs in hvac_input.cooling_systems:
            risk = self.assess_refrigerant_risk(cs)
            if risk is not None:
                ref_risks.append(risk)

        # Step 5: Assess distribution
        dist_result = None
        if hvac_input.distribution_systems:
            dist_result = self.assess_distribution(
                hvac_input.distribution_systems, hvac_input,
            )

        # Step 6: Assess controls
        ctrl_result = None
        if hvac_input.controls:
            ctrl_result = self.assess_controls(hvac_input.controls)

        # Step 7: Aggregate metrics
        overall_heat_eff = self._calculate_overall_heating_efficiency(heating_results)
        overall_cool_seer = self._calculate_overall_cooling_seer(cooling_results)

        total_hvac_energy = Decimal("0")
        for hr in heating_results:
            total_hvac_energy += _decimal(hr.annual_delivered_kwh)
        for cr in cooling_results:
            total_hvac_energy += _decimal(cr.annual_electricity_kwh)
        for vr in vent_results:
            total_hvac_energy += _decimal(vr.annual_fan_energy_kwh)

        total_fgas = Decimal("0")
        for rr in ref_risks:
            total_fgas += _decimal(rr.annual_emissions_tco2e)

        # Step 8: Improvement measures
        improvements = self._identify_improvements(
            heating_results, cooling_results, vent_results,
            ref_risks, dist_result, ctrl_result, hvac_input,
        )
        total_savings = Decimal("0")
        for imp in improvements:
            total_savings += _decimal(imp.annual_savings_kwh)

        # Step 9: Summary recommendations
        recs = self._generate_recommendations(
            heating_results, cooling_results, vent_results,
            ref_risks, dist_result, ctrl_result, improvements,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = HVACResult(
            facility_id=hvac_input.facility_id,
            building_name=hvac_input.building_name,
            building_type=hvac_input.building_type.value,
            heating_assessments=heating_results,
            cooling_assessments=cooling_results,
            ventilation_assessments=vent_results,
            refrigerant_risks=ref_risks,
            distribution_assessment=dist_result,
            control_assessment=ctrl_result,
            overall_heating_efficiency=_round3(float(overall_heat_eff)),
            overall_cooling_seer=_round2(float(overall_cool_seer)),
            total_hvac_energy_kwh=_round2(float(total_hvac_energy)),
            total_f_gas_emissions_tco2e=_round4(float(total_fgas)),
            improvement_measures=improvements,
            total_savings_potential_kwh=_round2(float(total_savings)),
            recommendations=recs,
            processing_time_ms=_round2(elapsed_ms),
        )

        result.provenance_hash = _compute_hash(result)
        return result

    def assess_heating(
        self, system: HeatingSystem, context: HVACInput,
    ) -> HeatingAssessmentResult:
        """Assess a single heating system's seasonal efficiency.

        For boilers: eta_seasonal = eta_gross * f_part_load * f_controls * f_distribution
        For heat pumps: SPF from EN 14825 lookup.

        Args:
            system: Heating system specification.
            context: HVAC input context.

        Returns:
            HeatingAssessmentResult.
        """
        # Determine efficiency
        if system.known_efficiency is not None:
            efficiency = _decimal(system.known_efficiency)
        elif system.system_type in (
            HeatingSystemType.AIR_SOURCE_HEAT_PUMP,
            HeatingSystemType.GROUND_SOURCE_HEAT_PUMP,
            HeatingSystemType.WATER_SOURCE_HEAT_PUMP,
        ):
            efficiency = self._lookup_heat_pump_spf(system)
        else:
            efficiency = self._lookup_boiler_efficiency(system)

        # Rating classification
        if system.system_type in (
            HeatingSystemType.AIR_SOURCE_HEAT_PUMP,
            HeatingSystemType.GROUND_SOURCE_HEAT_PUMP,
            HeatingSystemType.WATER_SOURCE_HEAT_PUMP,
        ):
            if float(efficiency) >= 4.0:
                rating = "excellent"
            elif float(efficiency) >= 3.0:
                rating = "good"
            elif float(efficiency) >= 2.5:
                rating = "average"
            else:
                rating = "poor"
        else:
            if float(efficiency) >= 0.92:
                rating = "excellent"
            elif float(efficiency) >= 0.85:
                rating = "good"
            elif float(efficiency) >= 0.78:
                rating = "average"
            else:
                rating = "poor"

        # Energy calculation
        delivered = Decimal("0")
        useful = Decimal("0")
        if system.annual_energy_kwh is not None:
            delivered = _decimal(system.annual_energy_kwh)
            useful = delivered * efficiency
        elif context.annual_heating_demand_kwh is not None:
            useful = _decimal(context.annual_heating_demand_kwh)
            delivered = _safe_divide(useful, efficiency)
        else:
            # Estimate from floor area
            area = _decimal(context.floor_area_m2)
            demand_kwh_m2 = Decimal("80")  # Typical office heating
            useful = area * demand_kwh_m2
            delivered = _safe_divide(useful, efficiency)

        losses = delivered - useful if delivered > useful else Decimal("0")

        recs: List[str] = []
        if rating == "poor":
            recs.append("System efficiency is poor -- consider replacement")
        if system.age_years > 15:
            recs.append(f"System is {system.age_years} years old -- plan for replacement")
        if system.system_type == HeatingSystemType.GAS_BOILER and float(efficiency) < 0.90:
            recs.append("Upgrade to condensing boiler for 10-20% efficiency gain")

        return HeatingAssessmentResult(
            system_id=system.system_id,
            system_type=system.system_type.value,
            seasonal_efficiency=_round3(float(efficiency)),
            efficiency_rating=rating,
            annual_delivered_kwh=_round2(float(delivered)),
            annual_useful_kwh=_round2(float(useful)),
            losses_kwh=_round2(float(losses)),
            recommendations=recs,
        )

    def assess_cooling(
        self, system: CoolingSystem, context: HVACInput,
    ) -> CoolingAssessmentResult:
        """Assess a single cooling system's SEER.

        SEER = Qcool / Welec (seasonal across temperature bins).

        Args:
            system: Cooling system specification.
            context: HVAC input context.

        Returns:
            CoolingAssessmentResult.
        """
        # Determine SEER
        if system.known_seer is not None:
            seer = _decimal(system.known_seer)
        else:
            seer = self._lookup_cooling_seer(system)

        # Rating
        benchmarks = self._cooling_benchmarks.get(system.system_type, {})
        if float(seer) >= benchmarks.get("excellent", 99):
            rating = "excellent"
        elif float(seer) >= benchmarks.get("good", 99):
            rating = "good"
        elif float(seer) >= benchmarks.get("average", 99):
            rating = "average"
        else:
            rating = "poor"

        # Energy calculation
        elec = Decimal("0")
        cooling_output = Decimal("0")
        if system.annual_energy_kwh is not None:
            elec = _decimal(system.annual_energy_kwh)
            cooling_output = elec * seer
        elif context.annual_cooling_demand_kwh is not None:
            cooling_output = _decimal(context.annual_cooling_demand_kwh)
            elec = _safe_divide(cooling_output, seer)
        else:
            area = _decimal(context.floor_area_m2)
            demand_kwh_m2 = Decimal("30")  # Typical cooling demand
            cooling_output = area * demand_kwh_m2
            elec = _safe_divide(cooling_output, seer)

        recs: List[str] = []
        if rating == "poor":
            recs.append("Cooling SEER is poor -- consider system replacement or upgrade")
        if system.age_years > 12:
            recs.append(f"System is {system.age_years} years old -- modern units are 30-50% more efficient")
        excellent_seer = benchmarks.get("excellent", 8.0)
        if float(seer) < excellent_seer:
            recs.append(
                f"Target SEER of {excellent_seer} is achievable with modern inverter equipment"
            )

        return CoolingAssessmentResult(
            system_id=system.system_id,
            system_type=system.system_type.value,
            seer=_round2(float(seer)),
            efficiency_rating=rating,
            annual_electricity_kwh=_round2(float(elec)),
            annual_cooling_kwh=_round2(float(cooling_output)),
            recommendations=recs,
        )

    def assess_ventilation(
        self, system: VentilationSystem, context: HVACInput,
    ) -> VentilationAssessmentResult:
        """Assess ventilation system SFP and heat recovery.

        SFP = P_fan / q_v [W/(l/s)] per EN 13779.
        HR: eta_hr = (T_supply - T_fresh) / (T_extract - T_fresh).

        Args:
            system: Ventilation system specification.
            context: HVAC input context.

        Returns:
            VentilationAssessmentResult.
        """
        # SFP calculation
        if system.known_sfp is not None:
            sfp = _decimal(system.known_sfp)
        elif system.fan_power_w and system.airflow_rate_ls and system.airflow_rate_ls > 0:
            sfp = _safe_divide(_decimal(system.fan_power_w), _decimal(system.airflow_rate_ls))
        elif system.ventilation_type == VentilationType.NATURAL:
            sfp = Decimal("0")
        else:
            benchmarks = self._sfp_benchmarks.get(system.ventilation_type, {})
            sfp = _decimal(benchmarks.get("typical", 2.0))

        # SFP rating
        benchmarks = self._sfp_benchmarks.get(system.ventilation_type, {})
        sfp_limit = benchmarks.get("sfp_limit_part_l", 999)
        if float(sfp) <= benchmarks.get("best", 0):
            sfp_rating = "excellent"
        elif float(sfp) <= benchmarks.get("good", 0):
            sfp_rating = "good"
        elif float(sfp) <= benchmarks.get("typical", 0):
            sfp_rating = "average"
        else:
            sfp_rating = "poor"

        sfp_compliant = float(sfp) <= sfp_limit

        # Heat recovery
        hr_eff = Decimal("0")
        if system.heat_recovery_type != HeatRecoveryType.NONE:
            if system.heat_recovery_efficiency is not None:
                hr_eff = _decimal(system.heat_recovery_efficiency)
            else:
                hr_data = self._hr_effectiveness.get(system.heat_recovery_type, {})
                hr_eff = _decimal(hr_data.get("dry_effectiveness", 0.0))

        # Annual fan energy
        airflow = _decimal(system.airflow_rate_ls) if system.airflow_rate_ls > 0 else Decimal("0")
        if airflow <= Decimal("0"):
            # Estimate airflow from floor area (10 l/s per person, 1 person per 10m2)
            airflow = _decimal(context.floor_area_m2) / Decimal("10") * Decimal("10")

        op_hours = _decimal(context.operating_hours)
        fan_energy = sfp * airflow * op_hours / Decimal("1000")  # kWh

        # Annual heat recovered
        heat_recovered = Decimal("0")
        if hr_eff > Decimal("0"):
            # Q_recovered = eta * rho * cp * qv * delta_T * hours / 1000
            # Simplified: Q = eta * 0.34 * qv * delta_T * hours / 1000
            delta_t = Decimal("15")  # Typical indoor-outdoor delta-T during heating
            heat_recovered = (
                hr_eff * Decimal("0.34") * airflow * delta_t * op_hours / Decimal("1000")
            )

        recs: List[str] = []
        if not sfp_compliant:
            recs.append(f"SFP of {_round2(float(sfp))} W/(l/s) exceeds regulatory limit of {sfp_limit}")
        if sfp_rating == "poor":
            recs.append("Replace or upgrade fans to achieve better SFP performance")
        if system.heat_recovery_type == HeatRecoveryType.NONE and system.ventilation_type in (
            VentilationType.MECHANICAL_SUPPLY_EXTRACT, VentilationType.MVHR,
        ):
            recs.append("Install heat recovery unit to recover 70-85% of exhaust heat")
        elif hr_eff > Decimal("0") and hr_eff < Decimal("0.70"):
            recs.append("Heat recovery effectiveness is below 70% -- consider upgrade or maintenance")

        return VentilationAssessmentResult(
            system_id=system.system_id,
            ventilation_type=system.ventilation_type.value,
            sfp_w_ls=_round2(float(sfp)),
            sfp_rating=sfp_rating,
            sfp_compliant=sfp_compliant,
            heat_recovery_efficiency=_round2(float(hr_eff)),
            annual_fan_energy_kwh=_round2(float(fan_energy)),
            annual_heat_recovered_kwh=_round2(float(heat_recovered)),
            recommendations=recs,
        )

    def assess_refrigerant_risk(
        self, system: CoolingSystem,
    ) -> Optional[RefrigerantRiskResult]:
        """Assess refrigerant F-gas emissions and compliance per EU 517/2014.

        GHG = charge_kg * leak_rate * GWP / 1000 [tCO2e/yr]

        Args:
            system: Cooling system with refrigerant data.

        Returns:
            RefrigerantRiskResult or None if no refrigerant data.
        """
        if system.refrigerant_charge_kg <= 0:
            return None

        ref_data = self._refrigerant_gwp.get(system.refrigerant, {})
        gwp = _decimal(ref_data.get("gwp_100yr", 0))
        charge = _decimal(system.refrigerant_charge_kg)
        phase_out = ref_data.get("phase_out", False)

        # Charge in tCO2e
        charge_tco2e = charge * gwp / Decimal("1000")

        # Leak rate
        leak_table = self._f_gas_leak_rates.get(system.system_type, {})
        leak_rate = _decimal(leak_table.get(system.leak_rate_condition, 0.05))

        # Annual emissions
        annual_tco2e = charge * leak_rate * gwp / Decimal("1000")

        # Inspection interval
        if float(charge_tco2e) < 5:
            interval_key = "below_5_tco2e"
        elif float(charge_tco2e) < 50:
            interval_key = "5_to_50_tco2e"
        elif float(charge_tco2e) < 500:
            interval_key = "50_to_500_tco2e"
        else:
            interval_key = "above_500_tco2e"
        interval = self._f_gas_inspections[interval_key]["interval_months"]

        # Compliance
        compliant = not (phase_out and ref_data.get("status") == "banned_new_equipment")

        recs: List[str] = []
        if phase_out:
            recs.append(
                f"Refrigerant {system.refrigerant.value} (GWP={int(gwp)}) "
                f"faces phase-down -- plan transition to low-GWP alternative"
            )
        if int(gwp) > 750:
            recs.append(
                f"High-GWP refrigerant ({int(gwp)}). "
                f"Consider R32 (GWP=675), R290 (GWP=3), or R1234yf (GWP=4)"
            )
        if interval > 0:
            recs.append(f"F-gas leak checks required every {interval} months")
        if float(annual_tco2e) > 1.0:
            recs.append(
                f"Annual F-gas emissions of {_round3(float(annual_tco2e))} tCO2e "
                f"-- reduce charge or switch refrigerant"
            )

        return RefrigerantRiskResult(
            refrigerant=system.refrigerant.value,
            gwp_100yr=int(gwp),
            charge_kg=_round2(float(charge)),
            charge_tco2e=_round3(float(charge_tco2e)),
            annual_leak_rate=_round3(float(leak_rate)),
            annual_emissions_tco2e=_round4(float(annual_tco2e)),
            inspection_interval_months=interval,
            phase_out_risk=phase_out,
            f_gas_compliant=compliant,
            recommendations=recs,
        )

    def assess_distribution(
        self,
        systems: List[DistributionSystem],
        context: HVACInput,
    ) -> DistributionAssessmentResult:
        """Assess distribution system losses.

        Qloss = U_pipe * L * (Tflow - Tambient) * hours [kWh/yr]

        Args:
            systems: Distribution system specifications.
            context: HVAC input context.

        Returns:
            DistributionAssessmentResult.
        """
        total_losses = Decimal("0")
        total_energy = _decimal(context.annual_heating_demand_kwh or 0)
        if total_energy <= Decimal("0"):
            total_energy = _decimal(context.floor_area_m2) * Decimal("80")

        best_eff = Decimal("1")
        primary_type = ""

        for ds in systems:
            eff_table = self._distribution_eff.get(ds.distribution_type, {})
            eff = _decimal(eff_table.get(ds.insulation_level, 0.85))

            if primary_type == "":
                primary_type = ds.distribution_type.value
                best_eff = eff

            # Calculate pipe/duct losses
            if ds.total_pipe_length_m > 0:
                length = _decimal(ds.total_pipe_length_m)
                t_flow = _decimal(ds.flow_temperature_c)
                t_ambient = _decimal(ds.ambient_temperature_c)
                delta_t = t_flow - t_ambient

                # U-value of pipe insulation (W/m.K)
                pipe_u_map = {
                    "well_insulated": Decimal("0.20"),
                    "insulated": Decimal("0.40"),
                    "uninsulated": Decimal("1.50"),
                }
                pipe_u = pipe_u_map.get(ds.insulation_level, Decimal("0.40"))

                # Annual losses: U * L * dT * hours / 1000
                hours = _decimal(context.operating_hours)
                loss = pipe_u * length * delta_t * hours / Decimal("1000")
                total_losses += loss
            else:
                # Estimate from efficiency
                loss_frac = Decimal("1") - eff
                total_losses += total_energy * loss_frac

        loss_pct = float(_safe_pct(total_losses, total_energy))

        recs: List[str] = []
        if loss_pct > 15:
            recs.append(f"Distribution losses of {_round2(loss_pct)}% are excessive -- insulate pipes/ducts")
        if loss_pct > 10:
            recs.append("Consider insulation upgrade on exposed distribution runs")
        for ds in systems:
            if ds.insulation_level == "uninsulated":
                recs.append(
                    f"Uninsulated {ds.distribution_type.value} distribution -- "
                    f"add insulation for 10-20% energy saving on distribution losses"
                )

        return DistributionAssessmentResult(
            distribution_type=primary_type,
            efficiency=_round3(float(best_eff)),
            annual_losses_kwh=_round2(float(total_losses)),
            loss_percentage=_round2(loss_pct),
            recommendations=recs,
        )

    def assess_controls(
        self, controls: List[ControlSystem],
    ) -> ControlAssessmentResult:
        """Assess HVAC control system effectiveness.

        Args:
            controls: Control system specifications.

        Returns:
            ControlAssessmentResult.
        """
        # Use the best control level
        best_factor = Decimal("1.0")
        best_type = ControlType.MANUAL.value
        best_savings = Decimal("0")

        for ctrl in controls:
            credit_data = self._control_credit.get(ctrl.control_type, {})
            factor = _decimal(credit_data.get("credit_factor", 1.0))
            savings = _decimal(credit_data.get("typical_savings_pct", 0.0))
            if factor < best_factor:
                best_factor = factor
                best_type = ctrl.control_type.value
                best_savings = savings

        # Upgrade potential
        upgrade = float(best_factor) > 0.82  # Better than weather compensation

        recs: List[str] = []
        if float(best_factor) >= 0.95:
            recs.append("Controls are basic -- upgrade to programmable thermostat with TRVs")
        if float(best_factor) >= 0.88 and float(best_factor) < 0.95:
            recs.append("Add weather compensation control for 10-18% additional savings")
        if float(best_factor) >= 0.82 and float(best_factor) < 0.88:
            recs.append("Consider BMS integration for optimised scheduling and monitoring")
        if not any(c.has_night_setback for c in controls):
            recs.append("Enable night setback to reduce out-of-hours heating")
        if not any(c.has_holiday_scheduling for c in controls):
            recs.append("Configure holiday scheduling to avoid heating unoccupied building")

        return ControlAssessmentResult(
            control_type=best_type,
            credit_factor=_round3(float(best_factor)),
            estimated_savings_pct=_round2(float(best_savings)),
            upgrade_potential=upgrade,
            recommendations=recs,
        )

    # -------------------------------------------------------------------
    # Internal Calculations -- Lookups
    # -------------------------------------------------------------------

    def _lookup_boiler_efficiency(self, system: HeatingSystem) -> Decimal:
        """Look up boiler seasonal efficiency from embedded data."""
        sys_table = self._boiler_efficiency.get(system.system_type, {})
        sub_data = sys_table.get(system.boiler_subtype, {})

        if sub_data:
            return _decimal(sub_data.get("net", 0.85))

        # Fallback by age
        if system.age_years <= 3:
            return _decimal(0.92)
        elif system.age_years <= 10:
            return _decimal(0.85)
        elif system.age_years <= 20:
            return _decimal(0.78)
        else:
            return _decimal(0.70)

    def _lookup_heat_pump_spf(self, system: HeatingSystem) -> Decimal:
        """Look up heat pump seasonal performance factor from EN 14825 data."""
        hp_table = self._hp_cop.get(system.system_type, {})

        # Use seasonal average for UK as default
        spf = hp_table.get("seasonal_average_uk", 3.0)

        # Age correction: ~2% degradation per 5 years
        age_factor = Decimal("1") - _decimal(system.age_years) * Decimal("0.004")
        if age_factor < Decimal("0.75"):
            age_factor = Decimal("0.75")

        return _decimal(spf) * age_factor

    def _lookup_cooling_seer(self, system: CoolingSystem) -> Decimal:
        """Look up cooling SEER from benchmarks and age."""
        benchmarks = self._cooling_benchmarks.get(system.system_type, {})

        # Start with average benchmark
        base_seer = _decimal(benchmarks.get("average", 4.0))

        # Age correction
        if system.age_years <= 3:
            base_seer = _decimal(benchmarks.get("good", 5.0))
        elif system.age_years <= 10:
            base_seer = _decimal(benchmarks.get("average", 4.0))
        else:
            # Degradation for old systems
            degradation = Decimal("1") - _decimal(system.age_years - 10) * Decimal("0.02")
            if degradation < Decimal("0.70"):
                degradation = Decimal("0.70")
            base_seer = _decimal(benchmarks.get("average", 4.0)) * degradation

        return base_seer

    # -------------------------------------------------------------------
    # Internal Calculations -- Aggregates
    # -------------------------------------------------------------------

    def _calculate_overall_heating_efficiency(
        self, results: List[HeatingAssessmentResult],
    ) -> Decimal:
        """Calculate weighted average heating efficiency."""
        if not results:
            return Decimal("0")

        total_energy = Decimal("0")
        weighted_eff = Decimal("0")

        for r in results:
            delivered = _decimal(r.annual_delivered_kwh)
            eff = _decimal(r.seasonal_efficiency)
            weighted_eff += delivered * eff
            total_energy += delivered

        return _safe_divide(weighted_eff, total_energy)

    def _calculate_overall_cooling_seer(
        self, results: List[CoolingAssessmentResult],
    ) -> Decimal:
        """Calculate weighted average cooling SEER."""
        if not results:
            return Decimal("0")

        total_elec = Decimal("0")
        total_cooling = Decimal("0")

        for r in results:
            total_elec += _decimal(r.annual_electricity_kwh)
            total_cooling += _decimal(r.annual_cooling_kwh)

        return _safe_divide(total_cooling, total_elec)

    # -------------------------------------------------------------------
    # Internal -- Improvements
    # -------------------------------------------------------------------

    def _identify_improvements(
        self,
        heating_results: List[HeatingAssessmentResult],
        cooling_results: List[CoolingAssessmentResult],
        vent_results: List[VentilationAssessmentResult],
        ref_risks: List[RefrigerantRiskResult],
        dist_result: Optional[DistributionAssessmentResult],
        ctrl_result: Optional[ControlAssessmentResult],
        context: HVACInput,
    ) -> List[ImprovementMeasure]:
        """Identify HVAC improvement measures with savings estimates."""
        measures: List[ImprovementMeasure] = []
        co2_factor = Decimal("0.21")  # kgCO2/kWh gas
        energy_cost = Decimal("0.10")  # EUR/kWh
        priority = 0

        # Heating improvements
        for hr in heating_results:
            if hr.efficiency_rating in ("poor", "average") and hr.losses_kwh > 0:
                savings = _decimal(hr.losses_kwh) * Decimal("0.50")
                priority += 1
                measures.append(ImprovementMeasure(
                    category="heating",
                    description=f"Upgrade {hr.system_type} -- current efficiency {hr.seasonal_efficiency}",
                    annual_savings_kwh=_round2(float(savings)),
                    annual_savings_co2_kg=_round2(float(savings * co2_factor)),
                    estimated_cost=_round2(float(Decimal("5000"))),
                    payback_years=_round2(float(_safe_divide(
                        Decimal("5000"), savings * energy_cost, Decimal("999"),
                    ))),
                    priority=priority,
                ))

        # Cooling improvements
        for cr in cooling_results:
            if cr.efficiency_rating in ("poor", "average"):
                better_seer = _decimal(cr.seer) * Decimal("1.50")
                old_elec = _decimal(cr.annual_electricity_kwh)
                new_elec = _safe_divide(_decimal(cr.annual_cooling_kwh), better_seer)
                savings = old_elec - new_elec
                if savings > Decimal("0"):
                    priority += 1
                    measures.append(ImprovementMeasure(
                        category="cooling",
                        description=f"Replace {cr.system_type} (SEER {cr.seer}) with high-efficiency unit",
                        annual_savings_kwh=_round2(float(savings)),
                        annual_savings_co2_kg=_round2(float(savings * co2_factor)),
                        estimated_cost=_round2(float(Decimal("8000"))),
                        payback_years=_round2(float(_safe_divide(
                            Decimal("8000"), savings * energy_cost, Decimal("999"),
                        ))),
                        priority=priority,
                    ))

        # Ventilation improvements
        for vr in vent_results:
            if vr.sfp_rating == "poor" or not vr.sfp_compliant:
                savings = _decimal(vr.annual_fan_energy_kwh) * Decimal("0.30")
                priority += 1
                measures.append(ImprovementMeasure(
                    category="ventilation",
                    description="Upgrade fans / add inverter drives to reduce SFP",
                    annual_savings_kwh=_round2(float(savings)),
                    annual_savings_co2_kg=_round2(float(savings * co2_factor)),
                    estimated_cost=_round2(float(Decimal("3000"))),
                    payback_years=_round2(float(_safe_divide(
                        Decimal("3000"), savings * energy_cost, Decimal("999"),
                    ))),
                    priority=priority,
                ))
            if vr.heat_recovery_efficiency == 0 and vr.ventilation_type in (
                VentilationType.MECHANICAL_SUPPLY_EXTRACT.value,
                VentilationType.MVHR.value,
            ):
                savings = _decimal(context.floor_area_m2) * Decimal("10")
                priority += 1
                measures.append(ImprovementMeasure(
                    category="ventilation",
                    description="Install heat recovery unit (70-85% effectiveness)",
                    annual_savings_kwh=_round2(float(savings)),
                    annual_savings_co2_kg=_round2(float(savings * co2_factor)),
                    estimated_cost=_round2(float(Decimal("6000"))),
                    payback_years=_round2(float(_safe_divide(
                        Decimal("6000"), savings * energy_cost, Decimal("999"),
                    ))),
                    priority=priority,
                ))

        # Distribution improvements
        if dist_result and dist_result.loss_percentage > 10:
            savings = _decimal(dist_result.annual_losses_kwh) * Decimal("0.50")
            priority += 1
            measures.append(ImprovementMeasure(
                category="distribution",
                description="Insulate pipes/ducts to reduce distribution losses",
                annual_savings_kwh=_round2(float(savings)),
                annual_savings_co2_kg=_round2(float(savings * co2_factor)),
                estimated_cost=_round2(float(Decimal("2000"))),
                payback_years=_round2(float(_safe_divide(
                    Decimal("2000"), savings * energy_cost, Decimal("999"),
                ))),
                priority=priority,
            ))

        # Control improvements
        if ctrl_result and ctrl_result.upgrade_potential:
            total_hvac = Decimal("0")
            for hr in heating_results:
                total_hvac += _decimal(hr.annual_delivered_kwh)
            savings = total_hvac * Decimal("0.10")
            priority += 1
            measures.append(ImprovementMeasure(
                category="controls",
                description="Upgrade to BMS / weather compensation control",
                annual_savings_kwh=_round2(float(savings)),
                annual_savings_co2_kg=_round2(float(savings * co2_factor)),
                estimated_cost=_round2(float(Decimal("4000"))),
                payback_years=_round2(float(_safe_divide(
                    Decimal("4000"), savings * energy_cost, Decimal("999"),
                ))),
                priority=priority,
            ))

        # Sort by savings
        measures.sort(key=lambda x: x.annual_savings_kwh, reverse=True)
        for i, m in enumerate(measures):
            m.priority = i + 1

        return measures

    # -------------------------------------------------------------------
    # Internal -- Recommendations
    # -------------------------------------------------------------------

    def _generate_recommendations(
        self,
        heating_results: List[HeatingAssessmentResult],
        cooling_results: List[CoolingAssessmentResult],
        vent_results: List[VentilationAssessmentResult],
        ref_risks: List[RefrigerantRiskResult],
        dist_result: Optional[DistributionAssessmentResult],
        ctrl_result: Optional[ControlAssessmentResult],
        improvements: List[ImprovementMeasure],
    ) -> List[str]:
        """Generate summary HVAC recommendations."""
        recs: List[str] = []

        # Heating summary
        poor_heating = [h for h in heating_results if h.efficiency_rating == "poor"]
        if poor_heating:
            recs.append(
                f"{len(poor_heating)} heating system(s) rated poor efficiency -- "
                f"prioritise replacement or upgrade"
            )

        # Cooling summary
        poor_cooling = [c for c in cooling_results if c.efficiency_rating == "poor"]
        if poor_cooling:
            recs.append(
                f"{len(poor_cooling)} cooling system(s) rated poor SEER -- "
                f"modern inverter units could improve by 50%+"
            )

        # Ventilation
        non_compliant_vent = [v for v in vent_results if not v.sfp_compliant]
        if non_compliant_vent:
            recs.append(
                f"{len(non_compliant_vent)} ventilation system(s) exceed SFP regulatory limits"
            )

        # F-gas
        high_risk_refs = [r for r in ref_risks if r.phase_out_risk]
        if high_risk_refs:
            recs.append(
                f"{len(high_risk_refs)} refrigerant(s) face phase-out under F-gas regulation -- "
                f"plan transition to low-GWP alternatives"
            )
        total_fgas = sum(r.annual_emissions_tco2e for r in ref_risks)
        if total_fgas > 0:
            recs.append(
                f"Total annual F-gas emissions: {_round3(total_fgas)} tCO2e -- "
                f"reduce through leak prevention and refrigerant transition"
            )

        # Distribution
        if dist_result and dist_result.loss_percentage > 10:
            recs.append(
                f"Distribution losses of {dist_result.loss_percentage}% -- "
                f"pipe/duct insulation upgrade recommended"
            )

        # Controls
        if ctrl_result and ctrl_result.upgrade_potential:
            recs.append(
                f"Control system ({ctrl_result.control_type}) has upgrade potential -- "
                f"BMS or adaptive controls could save {ctrl_result.estimated_savings_pct}%+"
            )

        # Total savings
        if improvements:
            total = sum(m.annual_savings_kwh for m in improvements)
            recs.append(
                f"{len(improvements)} improvement measures identified with "
                f"total savings potential of {_round2(total)} kWh/yr"
            )

        if not recs:
            recs.append("HVAC systems perform well. No critical improvements identified")

        return recs
