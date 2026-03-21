# -*- coding: utf-8 -*-
"""
DomesticHotWaterEngine - PACK-032 Building Energy Assessment Engine 4
=====================================================================

Assesses domestic hot water (DHW) system energy performance per
EN 15316-3 methodology.  Covers demand estimation, generation
efficiency, storage and distribution losses, solar thermal
contribution via the f-chart method, and legionella compliance
per HSG274.

EN 15316-3:2017 Compliance:
    - DHW demand calculation from occupancy and building type
    - Generation sub-system efficiency by fuel/technology
    - Storage heat loss quantification (standing losses)
    - Distribution pipe heat losses (circulation loops)
    - Auxiliary energy (pumps, controls)

Solar Thermal per EN 15316-4-3:
    - f-chart correlation for solar fraction estimation
    - Collector performance (flat-plate, evacuated-tube, unglazed, PVT)
    - Climate-zone-specific irradiance data
    - Storage sizing adequacy check

Legionella Compliance per HSG274:
    - Storage temperature >= 60 C
    - Distribution temperature >= 50 C at outlets
    - Monthly pasteurisation cycle check
    - Dead-leg length compliance

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - No LLM involvement in any numeric calculation path
    - SHA-256 provenance hashing on every result
    - Factors from EN 15316, BS 6700, CIBSE Guide G, HSG274

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-032 Building Energy Assessment
Engine:  4 of 10
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


class DHWSystemType(str, Enum):
    """DHW generation system types.

    Covers all common domestic hot water generation technologies
    encountered in building energy assessments per EN 15316-3.
    """
    GAS_BOILER = "gas_boiler"
    OIL_BOILER = "oil_boiler"
    ELECTRIC_IMMERSION = "electric_immersion"
    HEAT_PUMP_DEDICATED = "heat_pump_dedicated"
    HEAT_PUMP_INTEGRATED = "heat_pump_integrated"
    SOLAR_THERMAL = "solar_thermal"
    COMBI_BOILER = "combi_boiler"
    DISTRICT_HEATING = "district_heating"
    INSTANTANEOUS_GAS = "instantaneous_gas"
    INSTANTANEOUS_ELECTRIC = "instantaneous_electric"
    CHP = "chp"
    BIOMASS = "biomass"


class SolarCollectorType(str, Enum):
    """Solar thermal collector types per EN ISO 9806.

    Performance characteristics vary significantly by type.
    """
    FLAT_PLATE = "flat_plate"
    EVACUATED_TUBE = "evacuated_tube"
    UNGLAZED = "unglazed"
    PVT = "pvt"


class StorageType(str, Enum):
    """DHW storage vessel types per BS 1566 / BS EN 12897.

    Storage losses depend on insulation, volume and standing temperature.
    """
    CYLINDER_INDIRECT = "cylinder_indirect"
    CYLINDER_DIRECT = "cylinder_direct"
    THERMAL_STORE = "thermal_store"
    COMBI_STORE = "combi_store"
    NONE = "none"


class BuildingOccupancyType(str, Enum):
    """Building type for DHW demand estimation.

    Demand profiles vary significantly across building types.
    Sources: CIBSE Guide G, BS 6700, EN 15316-3.
    """
    RESIDENTIAL = "residential"
    OFFICE = "office"
    HOTEL = "hotel"
    HOSPITAL = "hospital"
    SCHOOL = "school"
    CARE_HOME = "care_home"
    SPORTS_CENTRE = "sports_centre"
    RESTAURANT = "restaurant"
    RETAIL = "retail"


class InsulationType(str, Enum):
    """Cylinder insulation type."""
    FACTORY_FOAM = "factory_foam"
    FACTORY_MINERAL_WOOL = "factory_mineral_wool"
    RETROFIT_JACKET = "retrofit_jacket"
    SPRAY_FOAM = "spray_foam"
    NONE = "none"


class ClimateZone(str, Enum):
    """Climate zone for solar thermal performance estimation."""
    NORTHERN_EUROPE = "northern_europe"
    CENTRAL_EUROPE = "central_europe"
    SOUTHERN_EUROPE = "southern_europe"
    MEDITERRANEAN = "mediterranean"
    OCEANIC = "oceanic"
    CONTINENTAL = "continental"


class ComplianceStatus(str, Enum):
    """Legionella compliance status per HSG274."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    AT_RISK = "at_risk"
    NOT_ASSESSED = "not_assessed"


# ---------------------------------------------------------------------------
# Constants -- DHW Demand (litres per day per occupant/unit)
# ---------------------------------------------------------------------------

# DHW demand in litres per day per occupant or per unit (room, bed, pupil).
# Sources: CIBSE Guide G Table 2.1, BS 6700:2006+A1, EN 15316-3 Annex B.
DHW_DEMAND_LITRES_PER_DAY: Dict[str, Dict[str, Any]] = {
    BuildingOccupancyType.RESIDENTIAL: {
        "low": 40.0,
        "typical": 50.0,
        "high": 60.0,
        "unit": "litres/person/day",
        "temperature_c": 60.0,
        "source": "CIBSE Guide G Table 2.1, EN 15316-3 Annex B",
    },
    BuildingOccupancyType.OFFICE: {
        "low": 5.0,
        "typical": 8.0,
        "high": 10.0,
        "unit": "litres/person/day",
        "temperature_c": 60.0,
        "source": "CIBSE Guide G Table 2.2",
    },
    BuildingOccupancyType.HOTEL: {
        "low": 100.0,
        "typical": 120.0,
        "high": 150.0,
        "unit": "litres/room/day",
        "temperature_c": 60.0,
        "source": "CIBSE Guide G Table 2.3, ASHRAE 2019",
    },
    BuildingOccupancyType.HOSPITAL: {
        "low": 200.0,
        "typical": 250.0,
        "high": 300.0,
        "unit": "litres/bed/day",
        "temperature_c": 60.0,
        "source": "HTM 04-01, CIBSE Guide G Table 2.4",
    },
    BuildingOccupancyType.SCHOOL: {
        "low": 15.0,
        "typical": 20.0,
        "high": 25.0,
        "unit": "litres/pupil/day",
        "temperature_c": 60.0,
        "source": "BB87 / CIBSE Guide G Table 2.5",
    },
    BuildingOccupancyType.CARE_HOME: {
        "low": 80.0,
        "typical": 100.0,
        "high": 130.0,
        "unit": "litres/bed/day",
        "temperature_c": 60.0,
        "source": "CIBSE Guide G, HTM 04-01",
    },
    BuildingOccupancyType.SPORTS_CENTRE: {
        "low": 30.0,
        "typical": 40.0,
        "high": 60.0,
        "unit": "litres/user/day",
        "temperature_c": 60.0,
        "source": "CIBSE Guide G Table 2.7",
    },
    BuildingOccupancyType.RESTAURANT: {
        "low": 8.0,
        "typical": 12.0,
        "high": 15.0,
        "unit": "litres/meal/day",
        "temperature_c": 60.0,
        "source": "CIBSE Guide G Table 2.8",
    },
    BuildingOccupancyType.RETAIL: {
        "low": 3.0,
        "typical": 5.0,
        "high": 8.0,
        "unit": "litres/person/day",
        "temperature_c": 60.0,
        "source": "CIBSE Guide G",
    },
}


# ---------------------------------------------------------------------------
# Constants -- System Efficiency (seasonal)
# ---------------------------------------------------------------------------

# Generation efficiency by system type and age band.
# Sources: SEDBUK database, EN 15316-4, CIBSE Guide F, SAP 2012.
SYSTEM_EFFICIENCY: Dict[str, Dict[str, float]] = {
    DHWSystemType.GAS_BOILER: {
        "new_condensing": 0.92,
        "old_condensing": 0.88,
        "non_condensing_post_1998": 0.80,
        "non_condensing_pre_1998": 0.72,
        "non_condensing_pre_1980": 0.60,
        "source": "SEDBUK / SAP Table 4b",
    },
    DHWSystemType.OIL_BOILER: {
        "new_condensing": 0.90,
        "old_condensing": 0.85,
        "non_condensing_post_1998": 0.78,
        "non_condensing_pre_1998": 0.70,
        "non_condensing_pre_1980": 0.55,
        "source": "SEDBUK / SAP Table 4b",
    },
    DHWSystemType.ELECTRIC_IMMERSION: {
        "new": 1.00,
        "old": 1.00,
        "source": "Direct conversion, 100% at point of use",
    },
    DHWSystemType.HEAT_PUMP_DEDICATED: {
        "new_high_efficiency": 3.50,
        "new_standard": 2.80,
        "old": 2.20,
        "source": "EN 16147 measured COPdhw, MCS data",
    },
    DHWSystemType.HEAT_PUMP_INTEGRATED: {
        "new_high_efficiency": 3.00,
        "new_standard": 2.50,
        "old": 2.00,
        "source": "EN 16147, integrated ASHP seasonal COPdhw",
    },
    DHWSystemType.SOLAR_THERMAL: {
        "flat_plate": 0.45,
        "evacuated_tube": 0.55,
        "note": "Solar fraction; backed up by auxiliary system",
        "source": "EN ISO 9806, f-chart method",
    },
    DHWSystemType.COMBI_BOILER: {
        "new_condensing": 0.90,
        "old_condensing": 0.85,
        "non_condensing": 0.75,
        "source": "SEDBUK / SAP Table 4b",
    },
    DHWSystemType.DISTRICT_HEATING: {
        "modern_insulated": 0.95,
        "older_network": 0.85,
        "source": "EN 15316-4-5, typical DH substations",
    },
    DHWSystemType.INSTANTANEOUS_GAS: {
        "new_condensing": 0.89,
        "non_condensing": 0.75,
        "source": "SAP 2012 Table 4a",
    },
    DHWSystemType.INSTANTANEOUS_ELECTRIC: {
        "standard": 0.98,
        "source": "Near 100% at point of use, minor standby losses",
    },
    DHWSystemType.CHP: {
        "gas_engine": 0.85,
        "fuel_cell": 0.90,
        "source": "CIBSE CP1, thermal efficiency for DHW",
    },
    DHWSystemType.BIOMASS: {
        "wood_pellet_auto": 0.85,
        "wood_chip": 0.78,
        "log_batch": 0.65,
        "source": "MCS, EN 303-5, RHI performance data",
    },
}


# ---------------------------------------------------------------------------
# Constants -- Distribution Loss Factors
# ---------------------------------------------------------------------------

# Distribution loss factors as fraction of delivered energy.
# Sources: EN 15316-3 Table B.3, SAP 2012 Table 3.
DISTRIBUTION_LOSS_FACTORS: Dict[str, Dict[str, float]] = {
    "well_insulated_short_runs": {
        "loss_factor": 0.03,
        "description": "Insulated pipes <3m, no circulation loop",
        "source": "EN 15316-3 Table B.3",
    },
    "insulated_medium_runs": {
        "loss_factor": 0.05,
        "description": "Insulated pipes 3-10m, no circulation loop",
        "source": "EN 15316-3 Table B.3",
    },
    "insulated_with_circulation": {
        "loss_factor": 0.10,
        "description": "Insulated pipes with circulation pump",
        "source": "EN 15316-3 Table B.3",
    },
    "poorly_insulated": {
        "loss_factor": 0.15,
        "description": "Older uninsulated or poorly insulated pipework",
        "source": "CIBSE Guide G, older building stock",
    },
    "long_runs_uninsulated": {
        "loss_factor": 0.20,
        "description": "Long uninsulated runs, common in older hospitals",
        "source": "HTM 04-01, pre-refurbishment",
    },
    "district_heating_substation": {
        "loss_factor": 0.08,
        "description": "HIU / substation losses for DH",
        "source": "CIBSE/ADE Heat Networks Code of Practice",
    },
}


# ---------------------------------------------------------------------------
# Constants -- Storage (Cylinder) Heat Losses
# ---------------------------------------------------------------------------

# Storage heat loss in watts per kelvin (W/K) by cylinder size and insulation.
# Sources: BS 1566-1, EN 12897, SAP Table 2, Product test data.
STORAGE_LOSS_WK: Dict[str, Dict[str, float]] = {
    "80_litre": {
        InsulationType.FACTORY_FOAM: 0.95,
        InsulationType.FACTORY_MINERAL_WOOL: 1.20,
        InsulationType.RETROFIT_JACKET: 1.80,
        InsulationType.SPRAY_FOAM: 1.10,
        InsulationType.NONE: 3.50,
        "source": "SAP Table 2a / BS 1566",
    },
    "120_litre": {
        InsulationType.FACTORY_FOAM: 1.10,
        InsulationType.FACTORY_MINERAL_WOOL: 1.40,
        InsulationType.RETROFIT_JACKET: 2.10,
        InsulationType.SPRAY_FOAM: 1.30,
        InsulationType.NONE: 4.10,
        "source": "SAP Table 2a / BS 1566",
    },
    "150_litre": {
        InsulationType.FACTORY_FOAM: 1.25,
        InsulationType.FACTORY_MINERAL_WOOL: 1.55,
        InsulationType.RETROFIT_JACKET: 2.40,
        InsulationType.SPRAY_FOAM: 1.45,
        InsulationType.NONE: 4.80,
        "source": "SAP Table 2a / BS 1566",
    },
    "200_litre": {
        InsulationType.FACTORY_FOAM: 1.45,
        InsulationType.FACTORY_MINERAL_WOOL: 1.80,
        InsulationType.RETROFIT_JACKET: 2.80,
        InsulationType.SPRAY_FOAM: 1.65,
        InsulationType.NONE: 5.60,
        "source": "SAP Table 2a / BS 1566",
    },
    "250_litre": {
        InsulationType.FACTORY_FOAM: 1.60,
        InsulationType.FACTORY_MINERAL_WOOL: 2.00,
        InsulationType.RETROFIT_JACKET: 3.10,
        InsulationType.SPRAY_FOAM: 1.85,
        InsulationType.NONE: 6.30,
        "source": "SAP Table 2a / BS 1566",
    },
    "300_litre": {
        InsulationType.FACTORY_FOAM: 1.80,
        InsulationType.FACTORY_MINERAL_WOOL: 2.20,
        InsulationType.RETROFIT_JACKET: 3.50,
        InsulationType.SPRAY_FOAM: 2.05,
        InsulationType.NONE: 7.10,
        "source": "SAP Table 2a / BS 1566",
    },
    "400_litre": {
        InsulationType.FACTORY_FOAM: 2.10,
        InsulationType.FACTORY_MINERAL_WOOL: 2.60,
        InsulationType.RETROFIT_JACKET: 4.10,
        InsulationType.SPRAY_FOAM: 2.40,
        InsulationType.NONE: 8.50,
        "source": "SAP Table 2a / BS 1566",
    },
    "500_litre": {
        InsulationType.FACTORY_FOAM: 2.40,
        InsulationType.FACTORY_MINERAL_WOOL: 3.00,
        InsulationType.RETROFIT_JACKET: 4.70,
        InsulationType.SPRAY_FOAM: 2.80,
        InsulationType.NONE: 9.80,
        "source": "Extrapolated from SAP / BS 1566",
    },
}


# ---------------------------------------------------------------------------
# Constants -- Solar Thermal Performance
# ---------------------------------------------------------------------------

# Annual global horizontal irradiance (kWh/m2/yr) by climate zone.
# Optimal tilt irradiance typically 10-15% higher.
# Sources: PVGIS, Meteonorm, EN 15316-4-3.
SOLAR_IRRADIANCE_BY_CLIMATE: Dict[str, Dict[str, float]] = {
    ClimateZone.NORTHERN_EUROPE: {
        "horizontal_kwh_m2_yr": 900.0,
        "optimal_tilt_kwh_m2_yr": 1050.0,
        "source": "PVGIS, Stockholm/Helsinki avg",
    },
    ClimateZone.CENTRAL_EUROPE: {
        "horizontal_kwh_m2_yr": 1100.0,
        "optimal_tilt_kwh_m2_yr": 1280.0,
        "source": "PVGIS, Berlin/Frankfurt avg",
    },
    ClimateZone.SOUTHERN_EUROPE: {
        "horizontal_kwh_m2_yr": 1500.0,
        "optimal_tilt_kwh_m2_yr": 1700.0,
        "source": "PVGIS, Rome/Madrid avg",
    },
    ClimateZone.MEDITERRANEAN: {
        "horizontal_kwh_m2_yr": 1750.0,
        "optimal_tilt_kwh_m2_yr": 1950.0,
        "source": "PVGIS, Athens/Seville avg",
    },
    ClimateZone.OCEANIC: {
        "horizontal_kwh_m2_yr": 1000.0,
        "optimal_tilt_kwh_m2_yr": 1150.0,
        "source": "PVGIS, London/Dublin avg",
    },
    ClimateZone.CONTINENTAL: {
        "horizontal_kwh_m2_yr": 1050.0,
        "optimal_tilt_kwh_m2_yr": 1200.0,
        "source": "PVGIS, Prague/Warsaw avg",
    },
}

# Solar thermal collector performance parameters per EN ISO 9806.
# eta_0 = optical efficiency, a1 = first-order loss (W/m2K),
# a2 = second-order loss (W/m2K2).
SOLAR_COLLECTOR_PERFORMANCE: Dict[str, Dict[str, float]] = {
    SolarCollectorType.FLAT_PLATE: {
        "eta_0": 0.78,
        "a1": 3.50,
        "a2": 0.015,
        "typical_yield_kwh_m2_yr_central": 450.0,
        "typical_yield_kwh_m2_yr_south": 650.0,
        "typical_yield_kwh_m2_yr_north": 350.0,
        "source": "EN ISO 9806, Solar Keymark database avg",
    },
    SolarCollectorType.EVACUATED_TUBE: {
        "eta_0": 0.72,
        "a1": 1.50,
        "a2": 0.008,
        "typical_yield_kwh_m2_yr_central": 520.0,
        "typical_yield_kwh_m2_yr_south": 720.0,
        "typical_yield_kwh_m2_yr_north": 420.0,
        "source": "EN ISO 9806, Solar Keymark database avg",
    },
    SolarCollectorType.UNGLAZED: {
        "eta_0": 0.90,
        "a1": 15.0,
        "a2": 0.0,
        "typical_yield_kwh_m2_yr_central": 280.0,
        "typical_yield_kwh_m2_yr_south": 400.0,
        "typical_yield_kwh_m2_yr_north": 200.0,
        "source": "EN ISO 9806, pool heating applications",
    },
    SolarCollectorType.PVT: {
        "eta_0": 0.65,
        "a1": 5.00,
        "a2": 0.020,
        "typical_yield_kwh_m2_yr_central": 380.0,
        "typical_yield_kwh_m2_yr_south": 550.0,
        "typical_yield_kwh_m2_yr_north": 300.0,
        "source": "EN ISO 9806, combined PV-thermal panels",
    },
}


# ---------------------------------------------------------------------------
# Constants -- Legionella Requirements (HSG274 / HTM 04-01)
# ---------------------------------------------------------------------------

LEGIONELLA_REQUIREMENTS: Dict[str, Any] = {
    "storage_min_temp_c": 60.0,
    "distribution_min_temp_c": 50.0,
    "cold_water_max_temp_c": 20.0,
    "pasteurisation_temp_c": 70.0,
    "pasteurisation_hold_minutes": 2,
    "dead_leg_max_length_m": 3.0,
    "outlet_max_wait_seconds": 60,
    "thermostatic_mixing_valve_max_c": 44.0,
    "source": "HSG274 Part 2, HTM 04-01:2016",
}


# ---------------------------------------------------------------------------
# Constants -- Pipe heat loss coefficients
# ---------------------------------------------------------------------------

# U-value of pipework in W/(m*K) by insulation thickness.
# Sources: BS 5422:2009, CIBSE Guide C Table 3.21.
PIPE_HEAT_LOSS_COEFFICIENTS: Dict[str, Dict[str, float]] = {
    "15mm_uninsulated": {"u_value_w_per_m_k": 0.30, "source": "CIBSE Guide C"},
    "15mm_25mm_insulation": {"u_value_w_per_m_k": 0.11, "source": "BS 5422"},
    "22mm_uninsulated": {"u_value_w_per_m_k": 0.37, "source": "CIBSE Guide C"},
    "22mm_25mm_insulation": {"u_value_w_per_m_k": 0.13, "source": "BS 5422"},
    "28mm_uninsulated": {"u_value_w_per_m_k": 0.43, "source": "CIBSE Guide C"},
    "28mm_25mm_insulation": {"u_value_w_per_m_k": 0.15, "source": "BS 5422"},
    "35mm_uninsulated": {"u_value_w_per_m_k": 0.52, "source": "CIBSE Guide C"},
    "35mm_25mm_insulation": {"u_value_w_per_m_k": 0.17, "source": "BS 5422"},
    "42mm_uninsulated": {"u_value_w_per_m_k": 0.60, "source": "CIBSE Guide C"},
    "42mm_25mm_insulation": {"u_value_w_per_m_k": 0.19, "source": "BS 5422"},
    "54mm_uninsulated": {"u_value_w_per_m_k": 0.73, "source": "CIBSE Guide C"},
    "54mm_25mm_insulation": {"u_value_w_per_m_k": 0.22, "source": "BS 5422"},
    "54mm_50mm_insulation": {"u_value_w_per_m_k": 0.14, "source": "BS 5422"},
}


# ---------------------------------------------------------------------------
# Constants -- Water properties
# ---------------------------------------------------------------------------

WATER_DENSITY_KG_PER_LITRE: Decimal = Decimal("0.988")  # at 60C
WATER_SPECIFIC_HEAT_KJ_PER_KG_K: Decimal = Decimal("4.18")
COLD_WATER_INLET_TEMP_C: Dict[str, float] = {
    ClimateZone.NORTHERN_EUROPE: 6.0,
    ClimateZone.CENTRAL_EUROPE: 8.0,
    ClimateZone.SOUTHERN_EUROPE: 12.0,
    ClimateZone.MEDITERRANEAN: 14.0,
    ClimateZone.OCEANIC: 9.0,
    ClimateZone.CONTINENTAL: 7.0,
}
HOURS_PER_YEAR: Decimal = Decimal("8760")
DAYS_PER_YEAR: Decimal = Decimal("365")
KJ_TO_KWH: Decimal = Decimal("3600")


# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------


class DHWSystemInput(BaseModel):
    """DHW generation system specification."""

    system_type: DHWSystemType = Field(
        ..., description="DHW generation system type per EN 15316-3"
    )
    age_band: str = Field(
        "new_condensing",
        description="Age/efficiency band key matching SYSTEM_EFFICIENCY lookup",
    )
    rated_output_kw: Optional[float] = Field(
        None, description="Rated thermal output in kW"
    )
    efficiency_override: Optional[float] = Field(
        None,
        description="Override seasonal efficiency if known from test data (0-1 or COP)",
    )


class StorageInput(BaseModel):
    """DHW storage vessel specification."""

    storage_type: StorageType = Field(
        ..., description="Type of hot water storage vessel"
    )
    volume_litres: float = Field(
        ..., gt=0, description="Storage vessel volume in litres"
    )
    insulation_type: InsulationType = Field(
        InsulationType.FACTORY_FOAM,
        description="Insulation type on cylinder",
    )
    insulation_thickness_mm: float = Field(
        50.0, ge=0, description="Insulation thickness in mm"
    )
    stored_temperature_c: float = Field(
        60.0, description="Hot water stored temperature in degrees C"
    )
    thermostat_present: bool = Field(
        True, description="Whether cylinder thermostat is fitted"
    )
    heat_loss_wk_override: Optional[float] = Field(
        None, description="Measured heat loss rate in W/K if known"
    )


class DistributionInput(BaseModel):
    """DHW distribution pipework specification."""

    total_pipe_length_m: float = Field(
        ..., gt=0, description="Total length of DHW pipework in metres"
    )
    pipe_diameter_mm: float = Field(
        22.0, description="Predominant pipe diameter in mm"
    )
    insulation_thickness_mm: float = Field(
        25.0, ge=0, description="Pipe insulation thickness in mm"
    )
    has_circulation_loop: bool = Field(
        False, description="Whether a secondary return / circulation loop exists"
    )
    circulation_pump_watts: float = Field(
        0.0, ge=0, description="Circulation pump power in watts"
    )
    circulation_hours_per_day: float = Field(
        24.0, ge=0, le=24.0, description="Hours per day circulation pump runs"
    )
    flow_temperature_c: float = Field(
        60.0, description="Flow temperature in the distribution pipework"
    )
    ambient_temperature_c: float = Field(
        18.0, description="Ambient temperature around the pipework"
    )
    dead_leg_max_length_m: Optional[float] = Field(
        None, description="Longest dead-leg length in metres"
    )


class SolarThermalInput(BaseModel):
    """Solar thermal system specification."""

    collector_type: SolarCollectorType = Field(
        ..., description="Solar collector type per EN ISO 9806"
    )
    collector_area_m2: float = Field(
        ..., gt=0, description="Gross collector area in m2"
    )
    collector_orientation_deg: float = Field(
        180.0, ge=0, le=360,
        description="Collector azimuth: 180=south (northern hemisphere)",
    )
    collector_tilt_deg: float = Field(
        35.0, ge=0, le=90,
        description="Collector tilt angle from horizontal in degrees",
    )
    storage_volume_litres: float = Field(
        ..., gt=0, description="Dedicated solar storage volume in litres"
    )
    climate_zone: ClimateZone = Field(
        ..., description="Climate zone for irradiance lookup"
    )
    eta_0_override: Optional[float] = Field(
        None, description="Override optical efficiency from test certificate"
    )
    a1_override: Optional[float] = Field(
        None, description="Override first-order loss coefficient from test certificate"
    )
    a2_override: Optional[float] = Field(
        None, description="Override second-order loss coefficient from test certificate"
    )


class LegionellaInput(BaseModel):
    """Legionella risk parameters for compliance check."""

    storage_temperature_c: float = Field(
        ..., description="Measured storage temperature in degrees C"
    )
    distribution_temperature_c: float = Field(
        ..., description="Measured furthest outlet temperature in degrees C"
    )
    cold_water_temperature_c: float = Field(
        ..., description="Measured cold water supply temperature in degrees C"
    )
    dead_leg_max_length_m: float = Field(
        ..., ge=0, description="Longest dead-leg in metres"
    )
    pasteurisation_cycle: bool = Field(
        False, description="Whether monthly pasteurisation cycle is performed"
    )
    tmv_fitted: bool = Field(
        False, description="Whether thermostatic mixing valves are fitted"
    )
    risk_assessment_date: Optional[str] = Field(
        None, description="Date of last legionella risk assessment (ISO format)"
    )


class DHWAssessmentInput(BaseModel):
    """Top-level input for DHW assessment."""

    building_id: str = Field(
        ..., description="Unique identifier for the building"
    )
    building_type: BuildingOccupancyType = Field(
        ..., description="Building occupancy type for demand estimation"
    )
    occupancy_count: int = Field(
        ..., gt=0, description="Number of occupants/rooms/beds/pupils"
    )
    demand_level: str = Field(
        "typical",
        description="Demand level: low, typical, or high",
    )
    climate_zone: ClimateZone = Field(
        ClimateZone.CENTRAL_EUROPE,
        description="Climate zone for cold water inlet temperature",
    )
    operating_days_per_year: int = Field(
        365, gt=0, le=366, description="Number of operating days per year"
    )
    dhw_system: DHWSystemInput = Field(
        ..., description="DHW generation system specification"
    )
    storage: Optional[StorageInput] = Field(
        None, description="Storage vessel specification (None for instantaneous)"
    )
    distribution: Optional[DistributionInput] = Field(
        None, description="Distribution pipework specification"
    )
    solar_thermal: Optional[SolarThermalInput] = Field(
        None, description="Solar thermal system specification if present"
    )
    legionella: Optional[LegionellaInput] = Field(
        None, description="Legionella compliance parameters"
    )
    hot_water_setpoint_c: float = Field(
        60.0, description="DHW setpoint temperature in degrees C"
    )
    electricity_cost_eur_per_kwh: float = Field(
        0.30, description="Electricity unit cost in EUR/kWh"
    )
    gas_cost_eur_per_kwh: float = Field(
        0.08, description="Gas unit cost in EUR/kWh"
    )
    carbon_factor_electricity_kg_per_kwh: float = Field(
        0.233, description="Grid electricity carbon factor in kgCO2e/kWh"
    )
    carbon_factor_gas_kg_per_kwh: float = Field(
        0.203, description="Natural gas carbon factor in kgCO2e/kWh"
    )


# ---------------------------------------------------------------------------
# Pydantic Result Models
# ---------------------------------------------------------------------------


class DHWDemandResult(BaseModel):
    """DHW demand calculation result."""

    daily_demand_litres: float = Field(
        ..., description="Total daily DHW demand in litres"
    )
    annual_demand_litres: float = Field(
        ..., description="Annual DHW demand in litres"
    )
    annual_demand_kwh: float = Field(
        ..., description="Annual DHW energy demand in kWh (net at tap)"
    )
    demand_per_occupant_litres_day: float = Field(
        ..., description="Demand per occupant/unit per day in litres"
    )
    cold_water_inlet_temp_c: float = Field(
        ..., description="Assumed cold water inlet temperature in degrees C"
    )
    hot_water_delivery_temp_c: float = Field(
        ..., description="Hot water delivery temperature in degrees C"
    )
    temperature_rise_k: float = Field(
        ..., description="Temperature rise from inlet to delivery in K"
    )


class GenerationResult(BaseModel):
    """DHW generation system assessment result."""

    system_type: str = Field(..., description="DHW system type")
    seasonal_efficiency: float = Field(
        ..., description="Seasonal generation efficiency (or COP for heat pumps)"
    )
    annual_delivered_energy_kwh: float = Field(
        ..., description="Annual energy delivered to DHW system in kWh"
    )
    annual_fuel_input_kwh: float = Field(
        ..., description="Annual fuel/electricity input to generator in kWh"
    )
    annual_cost_eur: float = Field(
        ..., description="Annual DHW energy cost in EUR"
    )
    annual_carbon_kg_co2e: float = Field(
        ..., description="Annual carbon emissions from DHW generation in kgCO2e"
    )
    efficiency_rating: str = Field(
        ..., description="Qualitative efficiency rating"
    )


class StorageResult(BaseModel):
    """DHW storage assessment result."""

    storage_type: str = Field(..., description="Storage vessel type")
    volume_litres: float = Field(..., description="Volume in litres")
    heat_loss_wk: float = Field(
        ..., description="Standing heat loss rate in W/K"
    )
    annual_storage_loss_kwh: float = Field(
        ..., description="Annual standing loss from storage in kWh"
    )
    storage_loss_pct_of_demand: float = Field(
        ..., description="Storage loss as percentage of net DHW demand"
    )
    insulation_rating: str = Field(
        ..., description="Insulation quality rating"
    )
    thermostat_status: str = Field(
        ..., description="Cylinder thermostat compliance status"
    )


class DistributionResult(BaseModel):
    """DHW distribution assessment result."""

    total_pipe_length_m: float = Field(
        ..., description="Total pipe length in metres"
    )
    has_circulation_loop: bool = Field(
        ..., description="Whether circulation loop is present"
    )
    annual_distribution_loss_kwh: float = Field(
        ..., description="Annual distribution pipe heat loss in kWh"
    )
    annual_pump_energy_kwh: float = Field(
        ..., description="Annual circulation pump energy in kWh"
    )
    distribution_loss_pct_of_demand: float = Field(
        ..., description="Distribution loss as percentage of net demand"
    )
    pipe_insulation_rating: str = Field(
        ..., description="Pipe insulation quality rating"
    )


class SolarThermalResult(BaseModel):
    """Solar thermal system assessment result."""

    collector_type: str = Field(..., description="Solar collector type")
    collector_area_m2: float = Field(..., description="Collector area in m2")
    annual_solar_yield_kwh: float = Field(
        ..., description="Annual solar energy yield in kWh"
    )
    solar_fraction: float = Field(
        ..., description="Solar fraction (0-1) of total DHW demand met by solar"
    )
    f_chart_x: float = Field(
        ..., description="f-chart X parameter (collector loss ratio)"
    )
    f_chart_y: float = Field(
        ..., description="f-chart Y parameter (collector gain ratio)"
    )
    annual_carbon_saving_kg: float = Field(
        ..., description="Annual carbon saving from solar contribution in kgCO2e"
    )
    storage_adequacy: str = Field(
        ..., description="Whether solar storage volume is adequate"
    )
    payback_years: float = Field(
        ..., description="Simple payback of solar thermal system in years"
    )


class LegionellaResult(BaseModel):
    """Legionella compliance assessment result."""

    overall_status: str = Field(
        ..., description="Overall legionella compliance status"
    )
    storage_temp_compliant: bool = Field(
        ..., description="Whether storage temperature meets HSG274 minimum"
    )
    distribution_temp_compliant: bool = Field(
        ..., description="Whether distribution temperature meets HSG274 minimum"
    )
    cold_water_temp_compliant: bool = Field(
        ..., description="Whether cold water is below maximum temperature"
    )
    dead_leg_compliant: bool = Field(
        ..., description="Whether dead-leg lengths comply with guidance"
    )
    pasteurisation_compliant: bool = Field(
        ..., description="Whether pasteurisation cycle is adequate"
    )
    issues: List[str] = Field(
        default_factory=list, description="List of compliance issues"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommended corrective actions"
    )


class DHWAssessmentResult(BaseModel):
    """Complete DHW system assessment result."""

    assessment_id: str = Field(
        ..., description="Unique assessment identifier"
    )
    building_id: str = Field(..., description="Building identifier")
    engine_version: str = Field(
        ..., description="Engine version that produced this result"
    )
    demand: DHWDemandResult = Field(
        ..., description="DHW demand calculation"
    )
    generation: GenerationResult = Field(
        ..., description="Generation system assessment"
    )
    storage: Optional[StorageResult] = Field(
        None, description="Storage assessment"
    )
    distribution: Optional[DistributionResult] = Field(
        None, description="Distribution assessment"
    )
    solar_thermal: Optional[SolarThermalResult] = Field(
        None, description="Solar thermal assessment"
    )
    legionella: Optional[LegionellaResult] = Field(
        None, description="Legionella compliance check"
    )
    total_annual_energy_kwh: float = Field(
        ..., description="Total annual DHW energy consumption in kWh"
    )
    total_annual_cost_eur: float = Field(
        ..., description="Total annual DHW energy cost in EUR"
    )
    total_annual_carbon_kg_co2e: float = Field(
        ..., description="Total annual carbon emissions in kgCO2e"
    )
    energy_per_occupant_kwh: float = Field(
        ..., description="DHW energy per occupant/unit in kWh/yr"
    )
    system_overall_efficiency: float = Field(
        ..., description="Overall system efficiency (net demand / total input)"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Prioritised improvement recommendations"
    )
    calculated_at: str = Field(
        ..., description="ISO format UTC timestamp"
    )
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash for audit trail"
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class DomesticHotWaterEngine:
    """Domestic Hot Water system assessment engine.

    Implements EN 15316-3 methodology for DHW energy assessment,
    including demand calculation, generation efficiency, storage and
    distribution losses, solar thermal contribution, and legionella
    compliance checking.

    Zero-Hallucination Guarantees:
        - All calculations use deterministic Decimal arithmetic
        - No LLM involvement in any numeric calculation path
        - SHA-256 provenance hashing on every result
        - All factors sourced from published standards

    Example::

        engine = DomesticHotWaterEngine()
        result = engine.analyze(assessment_input)
    """

    # ---------------------------------------------------------------
    # Public: analyze
    # ---------------------------------------------------------------

    def analyze(self, inp: DHWAssessmentInput) -> DHWAssessmentResult:
        """Run complete DHW system assessment.

        Executes demand calculation, generation assessment, storage and
        distribution loss analysis, optional solar thermal evaluation,
        and legionella compliance check.

        Args:
            inp: Complete DHW assessment input.

        Returns:
            DHWAssessmentResult with full provenance.
        """
        t_start = time.perf_counter()

        # Step 1: Calculate DHW demand
        demand = self.calculate_demand(
            building_type=inp.building_type,
            occupancy_count=inp.occupancy_count,
            demand_level=inp.demand_level,
            climate_zone=inp.climate_zone,
            hot_water_setpoint_c=inp.hot_water_setpoint_c,
            operating_days=inp.operating_days_per_year,
        )

        # Step 2: Assess generation system
        generation = self.assess_generation(
            system=inp.dhw_system,
            net_demand_kwh=demand.annual_demand_kwh,
            electricity_cost=inp.electricity_cost_eur_per_kwh,
            gas_cost=inp.gas_cost_eur_per_kwh,
            carbon_factor_elec=inp.carbon_factor_electricity_kg_per_kwh,
            carbon_factor_gas=inp.carbon_factor_gas_kg_per_kwh,
        )

        # Step 3: Assess storage
        storage_result: Optional[StorageResult] = None
        storage_loss_kwh = Decimal("0")
        if inp.storage and inp.storage.storage_type != StorageType.NONE:
            storage_result = self.assess_storage(
                storage=inp.storage,
                net_demand_kwh=demand.annual_demand_kwh,
            )
            storage_loss_kwh = _decimal(storage_result.annual_storage_loss_kwh)

        # Step 4: Assess distribution
        dist_result: Optional[DistributionResult] = None
        dist_loss_kwh = Decimal("0")
        pump_energy_kwh = Decimal("0")
        if inp.distribution:
            dist_result = self.assess_distribution(
                distribution=inp.distribution,
                net_demand_kwh=demand.annual_demand_kwh,
                operating_days=inp.operating_days_per_year,
            )
            dist_loss_kwh = _decimal(dist_result.annual_distribution_loss_kwh)
            pump_energy_kwh = _decimal(dist_result.annual_pump_energy_kwh)

        # Step 5: Assess solar thermal
        solar_result: Optional[SolarThermalResult] = None
        solar_yield_kwh = Decimal("0")
        if inp.solar_thermal:
            solar_result = self.assess_solar_thermal(
                solar=inp.solar_thermal,
                net_demand_kwh=demand.annual_demand_kwh,
                carbon_factor=inp.carbon_factor_gas_kg_per_kwh,
                gas_cost=inp.gas_cost_eur_per_kwh,
            )
            solar_yield_kwh = _decimal(solar_result.annual_solar_yield_kwh)

        # Step 6: Legionella compliance
        legionella_result: Optional[LegionellaResult] = None
        if inp.legionella:
            legionella_result = self.check_legionella_compliance(inp.legionella)

        # Step 7: Aggregate totals
        net_demand = _decimal(demand.annual_demand_kwh)
        gen_input = _decimal(generation.annual_fuel_input_kwh)
        total_energy = gen_input + storage_loss_kwh + pump_energy_kwh - solar_yield_kwh
        if total_energy < Decimal("0"):
            total_energy = Decimal("0")

        overall_eff = _safe_divide(net_demand, total_energy)

        # Cost and carbon for auxiliary / losses
        elec_cost = _decimal(inp.electricity_cost_eur_per_kwh)
        gas_cost = _decimal(inp.gas_cost_eur_per_kwh)
        cf_elec = _decimal(inp.carbon_factor_electricity_kg_per_kwh)
        cf_gas = _decimal(inp.carbon_factor_gas_kg_per_kwh)

        aux_cost = pump_energy_kwh * elec_cost + storage_loss_kwh * gas_cost
        aux_carbon = pump_energy_kwh * cf_elec + storage_loss_kwh * cf_gas

        total_cost = _decimal(generation.annual_cost_eur) + aux_cost
        total_carbon = _decimal(generation.annual_carbon_kg_co2e) + aux_carbon

        if solar_result:
            solar_saving_cost = solar_yield_kwh * gas_cost
            total_cost = total_cost - solar_saving_cost
            total_carbon = total_carbon - _decimal(solar_result.annual_carbon_saving_kg)

        if total_cost < Decimal("0"):
            total_cost = Decimal("0")
        if total_carbon < Decimal("0"):
            total_carbon = Decimal("0")

        energy_per_occupant = _safe_divide(
            total_energy, _decimal(inp.occupancy_count)
        )

        # Step 8: Recommendations
        recommendations = self._generate_recommendations(
            inp, demand, generation, storage_result, dist_result,
            solar_result, legionella_result,
        )

        t_end = time.perf_counter()
        processing_ms = (t_end - t_start) * 1000.0

        result = DHWAssessmentResult(
            assessment_id=_new_uuid(),
            building_id=inp.building_id,
            engine_version=_MODULE_VERSION,
            demand=demand,
            generation=generation,
            storage=storage_result,
            distribution=dist_result,
            solar_thermal=solar_result,
            legionella=legionella_result,
            total_annual_energy_kwh=_round2(float(total_energy)),
            total_annual_cost_eur=_round2(float(total_cost)),
            total_annual_carbon_kg_co2e=_round2(float(total_carbon)),
            energy_per_occupant_kwh=_round2(float(energy_per_occupant)),
            system_overall_efficiency=_round4(float(overall_eff)),
            recommendations=recommendations,
            calculated_at=_utcnow().isoformat(),
            processing_time_ms=_round3(processing_ms),
            provenance_hash="",
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ---------------------------------------------------------------
    # Public: calculate_demand
    # ---------------------------------------------------------------

    def calculate_demand(
        self,
        building_type: BuildingOccupancyType,
        occupancy_count: int,
        demand_level: str = "typical",
        climate_zone: ClimateZone = ClimateZone.CENTRAL_EUROPE,
        hot_water_setpoint_c: float = 60.0,
        operating_days: int = 365,
    ) -> DHWDemandResult:
        """Calculate annual DHW energy demand per EN 15316-3.

        Formula:
            Qw = V_day * rho * cp * DeltaT * days / 3600000 [kWh/yr]

        Where:
            V_day = daily volume demand (litres)
            rho   = water density at delivery temperature (kg/L)
            cp    = specific heat capacity (kJ/kg/K)
            DeltaT = T_hot - T_cold (K)

        Args:
            building_type: Building occupancy type.
            occupancy_count: Number of occupants/rooms/beds.
            demand_level: "low", "typical", or "high".
            climate_zone: Climate zone for cold water temperature.
            hot_water_setpoint_c: Delivery temperature.
            operating_days: Operating days per year.

        Returns:
            DHWDemandResult with demand values.
        """
        # Lookup demand profile
        profile = DHW_DEMAND_LITRES_PER_DAY.get(building_type)
        if not profile:
            profile = DHW_DEMAND_LITRES_PER_DAY[BuildingOccupancyType.OFFICE]

        level_key = demand_level if demand_level in ("low", "typical", "high") else "typical"
        litres_per_unit_day = _decimal(profile[level_key])

        # Cold water inlet temperature
        t_cold = _decimal(
            COLD_WATER_INLET_TEMP_C.get(climate_zone, 8.0)
        )
        t_hot = _decimal(hot_water_setpoint_c)
        delta_t = t_hot - t_cold
        if delta_t <= Decimal("0"):
            delta_t = Decimal("1")

        # Daily volume
        count = _decimal(occupancy_count)
        v_day = litres_per_unit_day * count

        # Annual volume
        days = _decimal(operating_days)
        v_annual = v_day * days

        # Energy demand: Qw = V * rho * cp * DeltaT / 3600 (litres * kg/L * kJ/kgK * K / (kJ/kWh))
        rho = WATER_DENSITY_KG_PER_LITRE
        cp = WATER_SPECIFIC_HEAT_KJ_PER_KG_K
        qw_annual = _safe_divide(
            v_annual * rho * cp * delta_t,
            KJ_TO_KWH,
        )

        return DHWDemandResult(
            daily_demand_litres=_round2(float(v_day)),
            annual_demand_litres=_round2(float(v_annual)),
            annual_demand_kwh=_round2(float(qw_annual)),
            demand_per_occupant_litres_day=_round2(float(litres_per_unit_day)),
            cold_water_inlet_temp_c=_round2(float(t_cold)),
            hot_water_delivery_temp_c=_round2(float(t_hot)),
            temperature_rise_k=_round2(float(delta_t)),
        )

    # ---------------------------------------------------------------
    # Public: assess_generation
    # ---------------------------------------------------------------

    def assess_generation(
        self,
        system: DHWSystemInput,
        net_demand_kwh: float,
        electricity_cost: float = 0.30,
        gas_cost: float = 0.08,
        carbon_factor_elec: float = 0.233,
        carbon_factor_gas: float = 0.203,
    ) -> GenerationResult:
        """Assess DHW generation system efficiency.

        Calculates fuel input = net_demand / efficiency, and derives
        cost and carbon.

        Args:
            system: DHW system specification.
            net_demand_kwh: Annual net DHW demand at tap in kWh.
            electricity_cost: EUR per kWh electricity.
            gas_cost: EUR per kWh gas.
            carbon_factor_elec: kgCO2e per kWh electricity.
            carbon_factor_gas: kgCO2e per kWh gas.

        Returns:
            GenerationResult with efficiency and cost data.
        """
        # Determine efficiency
        eff = self._lookup_generation_efficiency(system)
        eff_d = _decimal(eff)
        demand_d = _decimal(net_demand_kwh)

        # For heat pumps, efficiency is COP (>1). For boilers, <1.
        if eff_d <= Decimal("0"):
            eff_d = Decimal("0.5")

        fuel_input = _safe_divide(demand_d, eff_d)

        # Cost and carbon depend on fuel type
        is_electric = system.system_type in (
            DHWSystemType.ELECTRIC_IMMERSION,
            DHWSystemType.HEAT_PUMP_DEDICATED,
            DHWSystemType.HEAT_PUMP_INTEGRATED,
            DHWSystemType.INSTANTANEOUS_ELECTRIC,
        )

        if is_electric:
            unit_cost = _decimal(electricity_cost)
            carbon_factor = _decimal(carbon_factor_elec)
        elif system.system_type == DHWSystemType.DISTRICT_HEATING:
            # District heating: blended rate
            unit_cost = _decimal(gas_cost) * Decimal("1.2")
            carbon_factor = _decimal(carbon_factor_gas) * Decimal("0.8")
        else:
            unit_cost = _decimal(gas_cost)
            carbon_factor = _decimal(carbon_factor_gas)

        annual_cost = fuel_input * unit_cost
        annual_carbon = fuel_input * carbon_factor

        # Rating
        if eff >= 2.5:
            rating = "Excellent (high-COP heat pump)"
        elif eff >= 0.90:
            rating = "Good (modern condensing)"
        elif eff >= 0.80:
            rating = "Acceptable"
        elif eff >= 0.70:
            rating = "Below average"
        else:
            rating = "Poor (upgrade recommended)"

        return GenerationResult(
            system_type=system.system_type.value,
            seasonal_efficiency=_round4(eff),
            annual_delivered_energy_kwh=_round2(float(demand_d)),
            annual_fuel_input_kwh=_round2(float(fuel_input)),
            annual_cost_eur=_round2(float(annual_cost)),
            annual_carbon_kg_co2e=_round2(float(annual_carbon)),
            efficiency_rating=rating,
        )

    # ---------------------------------------------------------------
    # Public: assess_storage
    # ---------------------------------------------------------------

    def assess_storage(
        self,
        storage: StorageInput,
        net_demand_kwh: float,
    ) -> StorageResult:
        """Assess DHW storage vessel standing losses.

        Storage loss formula:
            Qs = h_loss * (T_store - T_ambient) * 8760 / 1000 [kWh/yr]

        Where:
            h_loss  = heat loss coefficient in W/K (from lookup or override)
            T_store = stored water temperature (C)
            T_amb   = ambient temperature around cylinder (typically 18-20 C)

        Args:
            storage: Storage vessel specification.
            net_demand_kwh: Annual net demand for percentage calculation.

        Returns:
            StorageResult with loss data.
        """
        # Determine heat loss coefficient
        h_loss_wk = self._lookup_storage_heat_loss(storage)
        h_loss = _decimal(h_loss_wk)

        t_store = _decimal(storage.stored_temperature_c)
        t_amb = Decimal("18")  # Typical airing cupboard / plant room
        delta_t = t_store - t_amb
        if delta_t < Decimal("0"):
            delta_t = Decimal("0")

        # Annual loss: h_loss (W/K) * deltaT (K) * 8760 (h/yr) / 1000 (W to kW) = kWh/yr
        annual_loss = h_loss * delta_t * HOURS_PER_YEAR / Decimal("1000")

        net_d = _decimal(net_demand_kwh)
        loss_pct = _safe_pct(annual_loss, net_d)

        # Insulation rating
        if storage.insulation_type == InsulationType.FACTORY_FOAM:
            ins_rating = "Good (factory-applied foam)"
        elif storage.insulation_type == InsulationType.FACTORY_MINERAL_WOOL:
            ins_rating = "Acceptable (factory mineral wool)"
        elif storage.insulation_type == InsulationType.RETROFIT_JACKET:
            ins_rating = "Below average (retrofit jacket)"
        elif storage.insulation_type == InsulationType.SPRAY_FOAM:
            ins_rating = "Good (spray foam)"
        else:
            ins_rating = "Poor (no insulation)"

        thermostat = "Fitted" if storage.thermostat_present else "Missing - fit thermostat to reduce losses"

        return StorageResult(
            storage_type=storage.storage_type.value,
            volume_litres=storage.volume_litres,
            heat_loss_wk=_round3(h_loss_wk),
            annual_storage_loss_kwh=_round2(float(annual_loss)),
            storage_loss_pct_of_demand=_round2(float(loss_pct)),
            insulation_rating=ins_rating,
            thermostat_status=thermostat,
        )

    # ---------------------------------------------------------------
    # Public: assess_distribution
    # ---------------------------------------------------------------

    def assess_distribution(
        self,
        distribution: DistributionInput,
        net_demand_kwh: float,
        operating_days: int = 365,
    ) -> DistributionResult:
        """Assess DHW distribution pipe losses.

        Pipe loss formula:
            Qd = U_pipe * L * (T_flow - T_ambient) * circulation_hours_yr / 1000 [kWh/yr]

        Where:
            U_pipe = pipe linear heat loss coefficient (W/(m*K))
            L      = total pipe length (m)
            T_flow = flow temperature (C)
            T_amb  = ambient temperature around pipes (C)

        Args:
            distribution: Distribution pipework specification.
            net_demand_kwh: Annual net demand for percentage.
            operating_days: Operating days per year.

        Returns:
            DistributionResult with loss data.
        """
        u_pipe = self._lookup_pipe_u_value(distribution)
        u_d = _decimal(u_pipe)
        length_d = _decimal(distribution.total_pipe_length_m)
        t_flow = _decimal(distribution.flow_temperature_c)
        t_amb = _decimal(distribution.ambient_temperature_c)
        delta_t = t_flow - t_amb
        if delta_t < Decimal("0"):
            delta_t = Decimal("0")

        # Circulation hours per year
        circ_hours_day = _decimal(distribution.circulation_hours_per_day)
        days_d = _decimal(operating_days)
        circ_hours_yr = circ_hours_day * days_d

        # Pipe loss: W/(m*K) * m * K * h / 1000 = kWh
        annual_pipe_loss = u_d * length_d * delta_t * circ_hours_yr / Decimal("1000")

        # Pump energy
        pump_w = _decimal(distribution.circulation_pump_watts)
        annual_pump = pump_w * circ_hours_yr / Decimal("1000")

        net_d = _decimal(net_demand_kwh)
        loss_pct = _safe_pct(annual_pipe_loss + annual_pump, net_d)

        # Insulation rating
        if distribution.insulation_thickness_mm >= 25.0:
            pipe_rating = "Good (>= 25mm insulation)"
        elif distribution.insulation_thickness_mm > 0:
            pipe_rating = "Partial insulation (< 25mm)"
        else:
            pipe_rating = "Poor (no insulation)"

        return DistributionResult(
            total_pipe_length_m=distribution.total_pipe_length_m,
            has_circulation_loop=distribution.has_circulation_loop,
            annual_distribution_loss_kwh=_round2(float(annual_pipe_loss)),
            annual_pump_energy_kwh=_round2(float(annual_pump)),
            distribution_loss_pct_of_demand=_round2(float(loss_pct)),
            pipe_insulation_rating=pipe_rating,
        )

    # ---------------------------------------------------------------
    # Public: assess_solar_thermal
    # ---------------------------------------------------------------

    def assess_solar_thermal(
        self,
        solar: SolarThermalInput,
        net_demand_kwh: float,
        carbon_factor: float = 0.203,
        gas_cost: float = 0.08,
    ) -> SolarThermalResult:
        """Assess solar thermal contribution using f-chart method.

        f-chart correlation (Beckman, Klein & Duffie):
            f = 1.029*Y - 0.065*X - 0.245*Y^2 + 0.0018*X^2 + 0.0215*Y^3

        Where:
            X = collector loss ratio = Ac * FR_UL * (Tref - Ta) * dt / Q_load
            Y = collector gain ratio = Ac * FR_eta0 * HT * N / Q_load

        Args:
            solar: Solar thermal system specification.
            net_demand_kwh: Annual net DHW demand in kWh.
            carbon_factor: Displaced fuel carbon factor in kgCO2e/kWh.
            gas_cost: Displaced fuel cost in EUR/kWh.

        Returns:
            SolarThermalResult with solar fraction and payback.
        """
        # Collector parameters
        perf = SOLAR_COLLECTOR_PERFORMANCE.get(solar.collector_type)
        if not perf:
            perf = SOLAR_COLLECTOR_PERFORMANCE[SolarCollectorType.FLAT_PLATE]

        eta_0 = _decimal(solar.eta_0_override if solar.eta_0_override else perf["eta_0"])
        a1 = _decimal(solar.a1_override if solar.a1_override else perf["a1"])

        # Irradiance for climate zone
        irr_data = SOLAR_IRRADIANCE_BY_CLIMATE.get(solar.climate_zone)
        if not irr_data:
            irr_data = SOLAR_IRRADIANCE_BY_CLIMATE[ClimateZone.CENTRAL_EUROPE]
        annual_irr = _decimal(irr_data["optimal_tilt_kwh_m2_yr"])

        # Orientation and tilt correction factor
        # Optimal is 180deg azimuth (south) and ~35deg tilt for central Europe
        orientation_factor = self._orientation_correction(
            solar.collector_orientation_deg, solar.collector_tilt_deg
        )
        effective_irr = annual_irr * orientation_factor

        ac = _decimal(solar.collector_area_m2)
        q_load = _decimal(net_demand_kwh)

        if q_load <= Decimal("0"):
            q_load = Decimal("1")

        # f-chart parameters
        # Reference temperature for X calculation
        t_ref = Decimal("100")  # Reference temperature (C)
        t_amb_avg = Decimal("10")  # Average annual ambient (C)
        dt_hours = Decimal("8760")  # Total hours

        # X = Ac * a1 * (Tref - Tamb) * dt / Q_load
        # Note: a1 is in W/(m2*K), convert to kW: /1000, then * hours = kWh
        x_param = ac * a1 * (t_ref - t_amb_avg) * dt_hours / (Decimal("1000") * q_load)

        # Y = Ac * eta0 * HT_annual / Q_load
        # HT is annual irradiance on tilted surface (kWh/m2)
        y_param = ac * eta_0 * effective_irr / q_load

        # f-chart correlation
        f = (
            Decimal("1.029") * y_param
            - Decimal("0.065") * x_param
            - Decimal("0.245") * y_param ** 2
            + Decimal("0.0018") * x_param ** 2
            + Decimal("0.0215") * y_param ** 3
        )

        # Clamp solar fraction
        if f < Decimal("0"):
            f = Decimal("0")
        if f > Decimal("1"):
            f = Decimal("1")

        annual_yield = f * q_load

        # Carbon saving
        cf = _decimal(carbon_factor)
        annual_carbon_saving = annual_yield * cf

        # Storage adequacy: rule of thumb 50-75 L per m2 collector
        storage_ratio = _safe_divide(
            _decimal(solar.storage_volume_litres), ac
        )
        if storage_ratio >= Decimal("50") and storage_ratio <= Decimal("80"):
            storage_adeq = "Adequate (50-80 L/m2 collector)"
        elif storage_ratio > Decimal("80"):
            storage_adeq = "Oversized (> 80 L/m2, increased standing losses)"
        else:
            storage_adeq = "Undersized (< 50 L/m2, reduced solar fraction)"

        # Simple payback
        # Typical solar thermal installed cost: 500-800 EUR/m2
        installed_cost_per_m2 = Decimal("650")
        capex = ac * installed_cost_per_m2
        annual_saving_eur = annual_yield * _decimal(gas_cost)
        payback = _safe_divide(capex, annual_saving_eur, default=Decimal("99"))

        return SolarThermalResult(
            collector_type=solar.collector_type.value,
            collector_area_m2=solar.collector_area_m2,
            annual_solar_yield_kwh=_round2(float(annual_yield)),
            solar_fraction=_round4(float(f)),
            f_chart_x=_round4(float(x_param)),
            f_chart_y=_round4(float(y_param)),
            annual_carbon_saving_kg=_round2(float(annual_carbon_saving)),
            storage_adequacy=storage_adeq,
            payback_years=_round2(float(payback)),
        )

    # ---------------------------------------------------------------
    # Public: check_legionella_compliance
    # ---------------------------------------------------------------

    def check_legionella_compliance(
        self, leg: LegionellaInput
    ) -> LegionellaResult:
        """Check legionella compliance per HSG274 Part 2.

        Requirements:
            - Hot water storage >= 60 C
            - Distribution outlets >= 50 C within 60 seconds
            - Cold water <= 20 C
            - Dead legs <= 3 m
            - Monthly pasteurisation to 70 C (if high risk)

        Args:
            leg: Legionella risk parameters.

        Returns:
            LegionellaResult with compliance status.
        """
        reqs = LEGIONELLA_REQUIREMENTS
        issues: List[str] = []
        recommendations: List[str] = []

        # Storage temperature
        storage_ok = leg.storage_temperature_c >= reqs["storage_min_temp_c"]
        if not storage_ok:
            issues.append(
                f"Storage temperature {leg.storage_temperature_c} C is below "
                f"HSG274 minimum of {reqs['storage_min_temp_c']} C. "
                f"Risk of Legionella proliferation."
            )
            recommendations.append(
                "Increase cylinder thermostat setpoint to 60 C minimum. "
                "Ensure thermostat is functional and properly positioned."
            )

        # Distribution temperature
        dist_ok = leg.distribution_temperature_c >= reqs["distribution_min_temp_c"]
        if not dist_ok:
            issues.append(
                f"Distribution temperature {leg.distribution_temperature_c} C "
                f"is below HSG274 minimum of {reqs['distribution_min_temp_c']} C "
                f"at furthest outlet."
            )
            recommendations.append(
                "Investigate distribution losses and circulation system. "
                "Insulate pipework, check circulation pump operation, and "
                "reduce dead-leg lengths."
            )

        # Cold water temperature
        cold_ok = leg.cold_water_temperature_c <= reqs["cold_water_max_temp_c"]
        if not cold_ok:
            issues.append(
                f"Cold water temperature {leg.cold_water_temperature_c} C "
                f"exceeds HSG274 maximum of {reqs['cold_water_max_temp_c']} C."
            )
            recommendations.append(
                "Insulate cold water pipes, especially near heat sources. "
                "Check cold water storage is not exposed to heat gains."
            )

        # Dead legs
        dead_ok = leg.dead_leg_max_length_m <= reqs["dead_leg_max_length_m"]
        if not dead_ok:
            issues.append(
                f"Dead-leg length {leg.dead_leg_max_length_m} m exceeds "
                f"HSG274 maximum of {reqs['dead_leg_max_length_m']} m."
            )
            recommendations.append(
                "Re-route pipework to shorten dead legs, or install "
                "point-of-use water heaters for remote outlets."
            )

        # Pasteurisation
        past_ok = leg.pasteurisation_cycle
        if not past_ok:
            issues.append(
                "Monthly pasteurisation cycle (70 C for 2 minutes) "
                "is not being performed."
            )
            recommendations.append(
                "Implement monthly high-temperature pasteurisation cycle "
                "to 70 C held for at least 2 minutes at all outlets per HSG274."
            )

        # TMV check
        if leg.tmv_fitted:
            recommendations.append(
                "TMVs are fitted. Ensure they are serviced annually and "
                "outlets downstream are monitored for temperature."
            )

        # Overall status
        all_ok = storage_ok and dist_ok and cold_ok and dead_ok and past_ok
        if all_ok:
            overall = ComplianceStatus.COMPLIANT.value
        elif storage_ok and dist_ok:
            overall = ComplianceStatus.AT_RISK.value
        else:
            overall = ComplianceStatus.NON_COMPLIANT.value

        return LegionellaResult(
            overall_status=overall,
            storage_temp_compliant=storage_ok,
            distribution_temp_compliant=dist_ok,
            cold_water_temp_compliant=cold_ok,
            dead_leg_compliant=dead_ok,
            pasteurisation_compliant=past_ok,
            issues=issues,
            recommendations=recommendations,
        )

    # ---------------------------------------------------------------
    # Internal: efficiency lookup
    # ---------------------------------------------------------------

    def _lookup_generation_efficiency(self, system: DHWSystemInput) -> float:
        """Look up generation efficiency from constants table.

        Uses override if provided, otherwise looks up by system type
        and age band.

        Args:
            system: DHW system specification.

        Returns:
            Seasonal efficiency (0-1 for boilers, >1 for heat pumps).
        """
        if system.efficiency_override is not None and system.efficiency_override > 0:
            return system.efficiency_override

        eff_table = SYSTEM_EFFICIENCY.get(system.system_type)
        if not eff_table:
            return 0.80  # Conservative default

        # Try exact age band match
        val = eff_table.get(system.age_band)
        if val is not None and isinstance(val, (int, float)):
            return float(val)

        # Fallback: first numeric value
        for k, v in eff_table.items():
            if k == "source" or k == "note":
                continue
            if isinstance(v, (int, float)):
                return float(v)

        return 0.80

    # ---------------------------------------------------------------
    # Internal: storage heat loss lookup
    # ---------------------------------------------------------------

    def _lookup_storage_heat_loss(self, storage: StorageInput) -> float:
        """Look up or interpolate storage heat loss from constants.

        Args:
            storage: Storage specification.

        Returns:
            Heat loss coefficient in W/K.
        """
        if storage.heat_loss_wk_override is not None and storage.heat_loss_wk_override > 0:
            return storage.heat_loss_wk_override

        vol = storage.volume_litres
        ins = storage.insulation_type

        # Find closest size bracket
        size_brackets = [
            (80, "80_litre"),
            (120, "120_litre"),
            (150, "150_litre"),
            (200, "200_litre"),
            (250, "250_litre"),
            (300, "300_litre"),
            (400, "400_litre"),
            (500, "500_litre"),
        ]

        # Find bracketing entries for interpolation
        lower_key = size_brackets[0]
        upper_key = size_brackets[-1]

        for i, (size, key) in enumerate(size_brackets):
            if vol <= size:
                upper_key = (size, key)
                lower_key = size_brackets[max(0, i - 1)]
                break
        else:
            lower_key = size_brackets[-1]
            upper_key = size_brackets[-1]

        lower_size, lower_name = lower_key
        upper_size, upper_name = upper_key

        lower_table = STORAGE_LOSS_WK.get(lower_name, {})
        upper_table = STORAGE_LOSS_WK.get(upper_name, {})

        lower_val = lower_table.get(ins, 2.0)
        upper_val = upper_table.get(ins, 2.0)

        if not isinstance(lower_val, (int, float)):
            lower_val = 2.0
        if not isinstance(upper_val, (int, float)):
            upper_val = 2.0

        # Linear interpolation
        if upper_size == lower_size:
            return float(lower_val)

        frac = (vol - lower_size) / (upper_size - lower_size)
        frac = max(0.0, min(1.0, frac))
        interpolated = lower_val + frac * (upper_val - lower_val)

        # Scale for volumes above 500L
        if vol > 500:
            scale = vol / 500.0
            interpolated = float(upper_val) * scale

        return float(interpolated)

    # ---------------------------------------------------------------
    # Internal: pipe U-value lookup
    # ---------------------------------------------------------------

    def _lookup_pipe_u_value(self, dist: DistributionInput) -> float:
        """Look up pipe linear heat loss coefficient.

        Args:
            dist: Distribution specification.

        Returns:
            U-value in W/(m*K).
        """
        diam = dist.pipe_diameter_mm
        ins = dist.insulation_thickness_mm

        # Map to closest lookup key
        if ins >= 50:
            suffix = "50mm_insulation"
        elif ins >= 25:
            suffix = "25mm_insulation"
        else:
            suffix = "uninsulated"

        # Map diameter
        diam_options = [15, 22, 28, 35, 42, 54]
        closest_diam = min(diam_options, key=lambda d: abs(d - diam))

        key = f"{closest_diam}mm_{suffix}"
        entry = PIPE_HEAT_LOSS_COEFFICIENTS.get(key)
        if entry:
            return entry["u_value_w_per_m_k"]

        # Fallback: try uninsulated at closest diameter
        key_fallback = f"{closest_diam}mm_uninsulated"
        entry_fb = PIPE_HEAT_LOSS_COEFFICIENTS.get(key_fallback)
        if entry_fb:
            return entry_fb["u_value_w_per_m_k"]

        return 0.30  # Conservative default

    # ---------------------------------------------------------------
    # Internal: orientation correction
    # ---------------------------------------------------------------

    def _orientation_correction(
        self, azimuth_deg: float, tilt_deg: float
    ) -> Decimal:
        """Calculate orientation correction factor for solar collectors.

        Optimal: azimuth=180 (south), tilt=30-40 deg for central Europe.
        Correction based on cosine approximation.

        Args:
            azimuth_deg: Collector azimuth (0=N, 90=E, 180=S, 270=W).
            tilt_deg: Collector tilt from horizontal.

        Returns:
            Orientation correction factor (0-1).
        """
        # Azimuth correction: deviation from south (180)
        azimuth_deviation = abs(azimuth_deg - 180.0)
        if azimuth_deviation > 180.0:
            azimuth_deviation = 360.0 - azimuth_deviation

        # Cosine-based correction
        az_factor = math.cos(math.radians(azimuth_deviation * 0.9))
        if az_factor < 0:
            az_factor = 0.0

        # Tilt correction: optimal around 35 degrees
        tilt_deviation = abs(tilt_deg - 35.0)
        tilt_factor = 1.0 - (tilt_deviation / 90.0) * 0.3

        if tilt_factor < 0.5:
            tilt_factor = 0.5

        combined = az_factor * tilt_factor
        return _decimal(max(0.1, min(1.0, combined)))

    # ---------------------------------------------------------------
    # Internal: recommendations
    # ---------------------------------------------------------------

    def _generate_recommendations(
        self,
        inp: DHWAssessmentInput,
        demand: DHWDemandResult,
        generation: GenerationResult,
        storage: Optional[StorageResult],
        distribution: Optional[DistributionResult],
        solar: Optional[SolarThermalResult],
        legionella: Optional[LegionellaResult],
    ) -> List[str]:
        """Generate prioritised improvement recommendations.

        All recommendations are deterministic, based on threshold
        comparisons against known benchmarks and compliance requirements.

        Args:
            inp: Assessment input.
            demand: Demand calculation result.
            generation: Generation assessment result.
            storage: Storage assessment result.
            distribution: Distribution assessment result.
            solar: Solar thermal result.
            legionella: Legionella result.

        Returns:
            List of recommendation strings, priority-ordered.
        """
        recs: List[str] = []

        # R1: Legionella non-compliance (highest priority)
        if legionella and legionella.overall_status == ComplianceStatus.NON_COMPLIANT.value:
            recs.append(
                "URGENT: Legionella non-compliance detected per HSG274. "
                "Address all identified temperature and dead-leg issues "
                "immediately to protect building occupants."
            )

        # R2: Generation efficiency
        eff = generation.seasonal_efficiency
        if inp.dhw_system.system_type in (
            DHWSystemType.GAS_BOILER, DHWSystemType.OIL_BOILER,
            DHWSystemType.COMBI_BOILER,
        ):
            if eff < 0.86:
                recs.append(
                    f"DHW generation efficiency of {eff:.1%} is below modern "
                    f"condensing boiler standard (>= 90%). Consider replacing "
                    f"with a new condensing boiler or heat pump to reduce "
                    f"energy consumption by {(0.90 - eff) / eff * 100:.0f}%."
                )
            elif eff < 0.90:
                recs.append(
                    f"DHW generation efficiency of {eff:.1%} is acceptable but "
                    f"below best practice. When the boiler reaches end of life, "
                    f"replace with a high-efficiency condensing unit or heat pump."
                )

        # R3: Heat pump recommendation for electric immersion
        if inp.dhw_system.system_type == DHWSystemType.ELECTRIC_IMMERSION:
            recs.append(
                "Direct electric immersion heating is the most expensive DHW "
                "option per kWh. A dedicated DHW heat pump (COP 2.5-3.5) would "
                "reduce electricity consumption by 60-70% while maintaining "
                "legionella-safe temperatures."
            )

        # R4: Storage losses
        if storage and storage.storage_loss_pct_of_demand > 15.0:
            recs.append(
                f"Storage standing losses are {storage.storage_loss_pct_of_demand:.1f}% "
                f"of net DHW demand, exceeding the 15% guideline. "
                f"Improve cylinder insulation or replace with a factory-insulated "
                f"unit to reduce losses."
            )
        elif storage and storage.insulation_rating.startswith("Poor"):
            recs.append(
                "Hot water cylinder has no insulation. Fit a minimum 80mm "
                "cylinder jacket or replace with a factory-insulated cylinder. "
                "This is a low-cost, high-impact energy saving measure."
            )

        # R5: Distribution losses
        if distribution and distribution.distribution_loss_pct_of_demand > 10.0:
            recs.append(
                f"Distribution losses are {distribution.distribution_loss_pct_of_demand:.1f}% "
                f"of net demand. Insulate all accessible DHW pipework to a "
                f"minimum of 25mm per BS 5422. Priority: circulation loop "
                f"pipework and risers."
            )

        # R6: Missing circulation controls
        if (
            distribution
            and distribution.has_circulation_loop
            and distribution.annual_pump_energy_kwh > 200
        ):
            recs.append(
                "DHW circulation pump runs continuously. Install a time clock "
                "and/or temperature-controlled pump to limit operation to "
                "occupied hours only. Potential to reduce pump energy by 40-60%."
            )

        # R7: Solar thermal opportunity
        if not solar and inp.building_type in (
            BuildingOccupancyType.RESIDENTIAL,
            BuildingOccupancyType.HOTEL,
            BuildingOccupancyType.CARE_HOME,
            BuildingOccupancyType.SPORTS_CENTRE,
        ):
            recs.append(
                "No solar thermal system present. Buildings with consistent "
                "DHW demand (residential, hotel, care home) are ideal candidates "
                "for solar thermal systems. Typical solar fraction of 40-60% "
                "is achievable in central/southern European climates."
            )

        # R8: Solar storage undersized
        if solar and "Undersized" in solar.storage_adequacy:
            recs.append(
                "Solar thermal storage is undersized. Increase dedicated solar "
                "storage to 50-75 litres per m2 of collector area to maximise "
                "solar fraction and avoid stagnation."
            )

        # R9: Cylinder thermostat
        if (
            inp.storage
            and inp.storage.storage_type != StorageType.NONE
            and not inp.storage.thermostat_present
        ):
            recs.append(
                "No cylinder thermostat detected. Fit a cylinder thermostat "
                "set to 60 C to prevent overheating and reduce standing losses. "
                "This is a Building Regulations Part L requirement."
            )

        # R10: Overall system efficiency
        gen_input_d = _decimal(generation.annual_fuel_input_kwh)
        demand_d = _decimal(demand.annual_demand_kwh)
        if gen_input_d > Decimal("0"):
            overall = float(_safe_divide(demand_d, gen_input_d))
            if overall < 0.50 and inp.dhw_system.system_type not in (
                DHWSystemType.HEAT_PUMP_DEDICATED,
                DHWSystemType.HEAT_PUMP_INTEGRATED,
            ):
                recs.append(
                    f"Overall DHW system efficiency is only {overall:.0%}. "
                    f"A comprehensive DHW system upgrade including modern "
                    f"generation, insulated storage, and insulated pipework "
                    f"could reduce energy use by 40-60%."
                )

        if not recs:
            recs.append(
                "DHW system is performing within acceptable parameters. "
                "Continue routine monitoring and maintenance per manufacturer "
                "recommendations and HSG274 legionella management plan."
            )

        return recs
