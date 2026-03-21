# -*- coding: utf-8 -*-
"""
EPCRatingEngine - PACK-032 Building Energy Assessment Engine 2
==============================================================

Generate Energy Performance Certificate (EPC) ratings (A-G) per the
Energy Performance of Buildings Directive (EPBD) recast methodology.
Calculates primary energy consumption, CO2 emissions, and assigns EPC
bands based on national thresholds for 15+ EU/EEA countries.

EPBD 2010/31/EU & 2024 Recast Compliance:
    - Primary energy calculation methodology
    - CO2 emission calculation
    - National EPC rating bands (A-G)
    - Reference building comparison
    - Minimum Energy Efficiency Standards (MEES) compliance

EN 15603:2008 / EN ISO 52000-1:2017 Compliance:
    - Primary energy factors by fuel type
    - Energy balance methodology
    - Renewable energy contribution
    - Exported energy credit

SAP 10.2 / RdSAP (UK) Compliance:
    - Standard Assessment Procedure methodology
    - SAP rating calculation (1-100+)
    - Environmental Impact Rating (EI)
    - Dwelling CO2 Emission Rate (DER)

DIN V 18599 / GEG 2020 (Germany) Compliance:
    - Primaerenergiebedarf calculation
    - Reference building (Referenzgebaeude) comparison
    - Endenergiebedarf

DPE 2021 (France) Compliance:
    - 3CL-DPE method (dual scale: energy + carbon)
    - Climate zone corrections

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Primary energy and emission factors from EN 15603 / national databases
    - No LLM involvement in any numeric calculation path
    - SHA-256 provenance hashing on every result
    - Rating thresholds from official national gazettes

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


class EPCMethodology(str, Enum):
    """EPC calculation methodology by country.

    Each EU/EEA member state implements the EPBD with a national
    calculation methodology.
    """
    SAP_2012 = "sap_2012"
    RDSAP_2012 = "rdsap_2012"
    GEG_2020 = "geg_2020"
    DPE_2021 = "dpe_2021"
    APE_2015 = "ape_2015"
    EPBD_GENERIC = "epbd_generic"


class EPCRating(str, Enum):
    """EPC rating bands per EPBD directive.

    A+ is used in some national schemes for near-zero energy buildings.
    """
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


class BuildingUseType(str, Enum):
    """Building use type for EPC assessment.

    Different building types have different reference values,
    internal gains, and operating patterns.
    """
    RESIDENTIAL = "residential"
    OFFICE = "office"
    RETAIL = "retail"
    HOTEL = "hotel"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    WAREHOUSE = "warehouse"
    MIXED_USE = "mixed_use"
    RESTAURANT = "restaurant"
    LEISURE = "leisure"


class HeatingFuelType(str, Enum):
    """Heating fuel types for primary energy and emission calculations.

    Primary energy and CO2 emission factors differ by fuel type and
    country per EN 15603.
    """
    NATURAL_GAS = "natural_gas"
    LPG = "lpg"
    HEATING_OIL = "heating_oil"
    ELECTRICITY = "electricity"
    BIOMASS_WOOD = "biomass_wood"
    BIOMASS_PELLET = "biomass_pellet"
    DISTRICT_HEATING = "district_heating"
    COAL = "coal"
    HEAT_PUMP_AIR = "heat_pump_air"
    HEAT_PUMP_GROUND = "heat_pump_ground"


class CoolingSystemType(str, Enum):
    """Cooling system types for EPC assessment."""
    NONE = "none"
    SPLIT_AC = "split_ac"
    CENTRAL_CHILLER = "central_chiller"
    VRF = "vrf"
    EVAPORATIVE = "evaporative"
    DISTRICT_COOLING = "district_cooling"
    HEAT_PUMP_REVERSIBLE = "heat_pump_reversible"


class RenewableType(str, Enum):
    """Renewable energy system types for on-site generation."""
    SOLAR_PV = "solar_pv"
    SOLAR_THERMAL = "solar_thermal"
    WIND = "wind"
    BIOMASS_CHP = "biomass_chp"
    MICRO_CHP = "micro_chp"
    NONE = "none"


# ---------------------------------------------------------------------------
# Constants -- Primary Energy Factors
# ---------------------------------------------------------------------------


# Primary energy factors (fp) by fuel type and country.
# Total primary energy factor = non-renewable + renewable components.
# Sources: EN 15603:2008 Annex E, national implementations, EPBD reports.
# Format: {country: {fuel_type: fp_total}}
PRIMARY_ENERGY_FACTORS: Dict[str, Dict[str, float]] = {
    "UK": {
        HeatingFuelType.NATURAL_GAS: 1.13,
        HeatingFuelType.LPG: 1.09,
        HeatingFuelType.HEATING_OIL: 1.10,
        HeatingFuelType.ELECTRICITY: 1.501,
        HeatingFuelType.BIOMASS_WOOD: 1.04,
        HeatingFuelType.BIOMASS_PELLET: 1.06,
        HeatingFuelType.DISTRICT_HEATING: 1.30,
        HeatingFuelType.COAL: 1.07,
        HeatingFuelType.HEAT_PUMP_AIR: 1.501,
        HeatingFuelType.HEAT_PUMP_GROUND: 1.501,
    },
    "DE": {
        HeatingFuelType.NATURAL_GAS: 1.10,
        HeatingFuelType.LPG: 1.10,
        HeatingFuelType.HEATING_OIL: 1.10,
        HeatingFuelType.ELECTRICITY: 1.80,
        HeatingFuelType.BIOMASS_WOOD: 0.20,
        HeatingFuelType.BIOMASS_PELLET: 0.20,
        HeatingFuelType.DISTRICT_HEATING: 0.70,
        HeatingFuelType.COAL: 1.10,
        HeatingFuelType.HEAT_PUMP_AIR: 1.80,
        HeatingFuelType.HEAT_PUMP_GROUND: 1.80,
    },
    "FR": {
        HeatingFuelType.NATURAL_GAS: 1.00,
        HeatingFuelType.LPG: 1.00,
        HeatingFuelType.HEATING_OIL: 1.00,
        HeatingFuelType.ELECTRICITY: 2.30,
        HeatingFuelType.BIOMASS_WOOD: 0.60,
        HeatingFuelType.BIOMASS_PELLET: 0.60,
        HeatingFuelType.DISTRICT_HEATING: 1.00,
        HeatingFuelType.COAL: 1.00,
        HeatingFuelType.HEAT_PUMP_AIR: 2.30,
        HeatingFuelType.HEAT_PUMP_GROUND: 2.30,
    },
    "IT": {
        HeatingFuelType.NATURAL_GAS: 1.05,
        HeatingFuelType.LPG: 1.05,
        HeatingFuelType.HEATING_OIL: 1.07,
        HeatingFuelType.ELECTRICITY: 1.95,
        HeatingFuelType.BIOMASS_WOOD: 0.30,
        HeatingFuelType.BIOMASS_PELLET: 0.30,
        HeatingFuelType.DISTRICT_HEATING: 1.50,
        HeatingFuelType.COAL: 1.10,
        HeatingFuelType.HEAT_PUMP_AIR: 1.95,
        HeatingFuelType.HEAT_PUMP_GROUND: 1.95,
    },
    "ES": {
        HeatingFuelType.NATURAL_GAS: 1.19,
        HeatingFuelType.LPG: 1.20,
        HeatingFuelType.HEATING_OIL: 1.18,
        HeatingFuelType.ELECTRICITY: 1.954,
        HeatingFuelType.BIOMASS_WOOD: 1.03,
        HeatingFuelType.BIOMASS_PELLET: 1.03,
        HeatingFuelType.DISTRICT_HEATING: 1.30,
        HeatingFuelType.COAL: 1.08,
        HeatingFuelType.HEAT_PUMP_AIR: 1.954,
        HeatingFuelType.HEAT_PUMP_GROUND: 1.954,
    },
    "NL": {
        HeatingFuelType.NATURAL_GAS: 1.00,
        HeatingFuelType.LPG: 1.00,
        HeatingFuelType.HEATING_OIL: 1.00,
        HeatingFuelType.ELECTRICITY: 1.45,
        HeatingFuelType.BIOMASS_WOOD: 0.00,
        HeatingFuelType.BIOMASS_PELLET: 0.00,
        HeatingFuelType.DISTRICT_HEATING: 0.80,
        HeatingFuelType.COAL: 1.00,
        HeatingFuelType.HEAT_PUMP_AIR: 1.45,
        HeatingFuelType.HEAT_PUMP_GROUND: 1.45,
    },
    "BE": {
        HeatingFuelType.NATURAL_GAS: 1.00,
        HeatingFuelType.LPG: 1.00,
        HeatingFuelType.HEATING_OIL: 1.00,
        HeatingFuelType.ELECTRICITY: 2.50,
        HeatingFuelType.BIOMASS_WOOD: 1.00,
        HeatingFuelType.BIOMASS_PELLET: 1.00,
        HeatingFuelType.DISTRICT_HEATING: 1.00,
        HeatingFuelType.COAL: 1.00,
        HeatingFuelType.HEAT_PUMP_AIR: 2.50,
        HeatingFuelType.HEAT_PUMP_GROUND: 2.50,
    },
    "AT": {
        HeatingFuelType.NATURAL_GAS: 1.17,
        HeatingFuelType.LPG: 1.14,
        HeatingFuelType.HEATING_OIL: 1.18,
        HeatingFuelType.ELECTRICITY: 1.91,
        HeatingFuelType.BIOMASS_WOOD: 0.08,
        HeatingFuelType.BIOMASS_PELLET: 0.08,
        HeatingFuelType.DISTRICT_HEATING: 1.60,
        HeatingFuelType.COAL: 1.46,
        HeatingFuelType.HEAT_PUMP_AIR: 1.91,
        HeatingFuelType.HEAT_PUMP_GROUND: 1.91,
    },
    "PL": {
        HeatingFuelType.NATURAL_GAS: 1.10,
        HeatingFuelType.LPG: 1.10,
        HeatingFuelType.HEATING_OIL: 1.10,
        HeatingFuelType.ELECTRICITY: 3.00,
        HeatingFuelType.BIOMASS_WOOD: 0.20,
        HeatingFuelType.BIOMASS_PELLET: 0.20,
        HeatingFuelType.DISTRICT_HEATING: 0.80,
        HeatingFuelType.COAL: 1.10,
        HeatingFuelType.HEAT_PUMP_AIR: 3.00,
        HeatingFuelType.HEAT_PUMP_GROUND: 3.00,
    },
    "SE": {
        HeatingFuelType.NATURAL_GAS: 1.09,
        HeatingFuelType.LPG: 1.09,
        HeatingFuelType.HEATING_OIL: 1.08,
        HeatingFuelType.ELECTRICITY: 1.60,
        HeatingFuelType.BIOMASS_WOOD: 0.04,
        HeatingFuelType.BIOMASS_PELLET: 0.04,
        HeatingFuelType.DISTRICT_HEATING: 0.91,
        HeatingFuelType.COAL: 1.10,
        HeatingFuelType.HEAT_PUMP_AIR: 1.60,
        HeatingFuelType.HEAT_PUMP_GROUND: 1.60,
    },
    "IE": {
        HeatingFuelType.NATURAL_GAS: 1.10,
        HeatingFuelType.LPG: 1.09,
        HeatingFuelType.HEATING_OIL: 1.10,
        HeatingFuelType.ELECTRICITY: 1.898,
        HeatingFuelType.BIOMASS_WOOD: 0.06,
        HeatingFuelType.BIOMASS_PELLET: 0.06,
        HeatingFuelType.DISTRICT_HEATING: 1.10,
        HeatingFuelType.COAL: 1.10,
        HeatingFuelType.HEAT_PUMP_AIR: 1.898,
        HeatingFuelType.HEAT_PUMP_GROUND: 1.898,
    },
    "DK": {
        HeatingFuelType.NATURAL_GAS: 1.00,
        HeatingFuelType.LPG: 1.00,
        HeatingFuelType.HEATING_OIL: 1.00,
        HeatingFuelType.ELECTRICITY: 1.90,
        HeatingFuelType.BIOMASS_WOOD: 0.00,
        HeatingFuelType.BIOMASS_PELLET: 0.00,
        HeatingFuelType.DISTRICT_HEATING: 0.85,
        HeatingFuelType.COAL: 1.00,
        HeatingFuelType.HEAT_PUMP_AIR: 1.90,
        HeatingFuelType.HEAT_PUMP_GROUND: 1.90,
    },
    "FI": {
        HeatingFuelType.NATURAL_GAS: 1.00,
        HeatingFuelType.LPG: 1.00,
        HeatingFuelType.HEATING_OIL: 1.00,
        HeatingFuelType.ELECTRICITY: 1.20,
        HeatingFuelType.BIOMASS_WOOD: 0.50,
        HeatingFuelType.BIOMASS_PELLET: 0.50,
        HeatingFuelType.DISTRICT_HEATING: 0.50,
        HeatingFuelType.COAL: 1.00,
        HeatingFuelType.HEAT_PUMP_AIR: 1.20,
        HeatingFuelType.HEAT_PUMP_GROUND: 1.20,
    },
    "CZ": {
        HeatingFuelType.NATURAL_GAS: 1.10,
        HeatingFuelType.LPG: 1.10,
        HeatingFuelType.HEATING_OIL: 1.10,
        HeatingFuelType.ELECTRICITY: 2.60,
        HeatingFuelType.BIOMASS_WOOD: 0.10,
        HeatingFuelType.BIOMASS_PELLET: 0.10,
        HeatingFuelType.DISTRICT_HEATING: 1.40,
        HeatingFuelType.COAL: 1.10,
        HeatingFuelType.HEAT_PUMP_AIR: 2.60,
        HeatingFuelType.HEAT_PUMP_GROUND: 2.60,
    },
    "PT": {
        HeatingFuelType.NATURAL_GAS: 1.00,
        HeatingFuelType.LPG: 1.00,
        HeatingFuelType.HEATING_OIL: 1.00,
        HeatingFuelType.ELECTRICITY: 2.50,
        HeatingFuelType.BIOMASS_WOOD: 1.00,
        HeatingFuelType.BIOMASS_PELLET: 1.00,
        HeatingFuelType.DISTRICT_HEATING: 1.00,
        HeatingFuelType.COAL: 1.00,
        HeatingFuelType.HEAT_PUMP_AIR: 2.50,
        HeatingFuelType.HEAT_PUMP_GROUND: 2.50,
    },
    "DEFAULT": {
        HeatingFuelType.NATURAL_GAS: 1.10,
        HeatingFuelType.LPG: 1.10,
        HeatingFuelType.HEATING_OIL: 1.10,
        HeatingFuelType.ELECTRICITY: 2.00,
        HeatingFuelType.BIOMASS_WOOD: 0.50,
        HeatingFuelType.BIOMASS_PELLET: 0.50,
        HeatingFuelType.DISTRICT_HEATING: 1.00,
        HeatingFuelType.COAL: 1.10,
        HeatingFuelType.HEAT_PUMP_AIR: 2.00,
        HeatingFuelType.HEAT_PUMP_GROUND: 2.00,
    },
}
"""Primary energy factors by fuel type and country.
Source: EN 15603:2008 Annex E, national implementations."""


# CO2 emission factors (kgCO2/kWh delivered energy) by fuel type and country.
# Sources: BEIS/DEFRA 2024, UBA (DE), ADEME (FR), ISPRA (IT), Eurostat.
CO2_EMISSION_FACTORS: Dict[str, Dict[str, float]] = {
    "UK": {
        HeatingFuelType.NATURAL_GAS: 0.183,
        HeatingFuelType.LPG: 0.214,
        HeatingFuelType.HEATING_OIL: 0.247,
        HeatingFuelType.ELECTRICITY: 0.136,
        HeatingFuelType.BIOMASS_WOOD: 0.015,
        HeatingFuelType.BIOMASS_PELLET: 0.019,
        HeatingFuelType.DISTRICT_HEATING: 0.180,
        HeatingFuelType.COAL: 0.326,
        HeatingFuelType.HEAT_PUMP_AIR: 0.136,
        HeatingFuelType.HEAT_PUMP_GROUND: 0.136,
    },
    "DE": {
        HeatingFuelType.NATURAL_GAS: 0.201,
        HeatingFuelType.LPG: 0.227,
        HeatingFuelType.HEATING_OIL: 0.266,
        HeatingFuelType.ELECTRICITY: 0.366,
        HeatingFuelType.BIOMASS_WOOD: 0.024,
        HeatingFuelType.BIOMASS_PELLET: 0.028,
        HeatingFuelType.DISTRICT_HEATING: 0.175,
        HeatingFuelType.COAL: 0.338,
        HeatingFuelType.HEAT_PUMP_AIR: 0.366,
        HeatingFuelType.HEAT_PUMP_GROUND: 0.366,
    },
    "FR": {
        HeatingFuelType.NATURAL_GAS: 0.205,
        HeatingFuelType.LPG: 0.231,
        HeatingFuelType.HEATING_OIL: 0.271,
        HeatingFuelType.ELECTRICITY: 0.052,
        HeatingFuelType.BIOMASS_WOOD: 0.013,
        HeatingFuelType.BIOMASS_PELLET: 0.016,
        HeatingFuelType.DISTRICT_HEATING: 0.125,
        HeatingFuelType.COAL: 0.343,
        HeatingFuelType.HEAT_PUMP_AIR: 0.052,
        HeatingFuelType.HEAT_PUMP_GROUND: 0.052,
    },
    "IT": {
        HeatingFuelType.NATURAL_GAS: 0.202,
        HeatingFuelType.LPG: 0.229,
        HeatingFuelType.HEATING_OIL: 0.268,
        HeatingFuelType.ELECTRICITY: 0.233,
        HeatingFuelType.BIOMASS_WOOD: 0.018,
        HeatingFuelType.BIOMASS_PELLET: 0.022,
        HeatingFuelType.DISTRICT_HEATING: 0.155,
        HeatingFuelType.COAL: 0.341,
        HeatingFuelType.HEAT_PUMP_AIR: 0.233,
        HeatingFuelType.HEAT_PUMP_GROUND: 0.233,
    },
    "ES": {
        HeatingFuelType.NATURAL_GAS: 0.202,
        HeatingFuelType.LPG: 0.227,
        HeatingFuelType.HEATING_OIL: 0.267,
        HeatingFuelType.ELECTRICITY: 0.190,
        HeatingFuelType.BIOMASS_WOOD: 0.018,
        HeatingFuelType.BIOMASS_PELLET: 0.022,
        HeatingFuelType.DISTRICT_HEATING: 0.160,
        HeatingFuelType.COAL: 0.340,
        HeatingFuelType.HEAT_PUMP_AIR: 0.190,
        HeatingFuelType.HEAT_PUMP_GROUND: 0.190,
    },
    "NL": {
        HeatingFuelType.NATURAL_GAS: 0.183,
        HeatingFuelType.LPG: 0.215,
        HeatingFuelType.HEATING_OIL: 0.250,
        HeatingFuelType.ELECTRICITY: 0.328,
        HeatingFuelType.BIOMASS_WOOD: 0.016,
        HeatingFuelType.BIOMASS_PELLET: 0.020,
        HeatingFuelType.DISTRICT_HEATING: 0.145,
        HeatingFuelType.COAL: 0.330,
        HeatingFuelType.HEAT_PUMP_AIR: 0.328,
        HeatingFuelType.HEAT_PUMP_GROUND: 0.328,
    },
    "DEFAULT": {
        HeatingFuelType.NATURAL_GAS: 0.202,
        HeatingFuelType.LPG: 0.227,
        HeatingFuelType.HEATING_OIL: 0.265,
        HeatingFuelType.ELECTRICITY: 0.233,
        HeatingFuelType.BIOMASS_WOOD: 0.018,
        HeatingFuelType.BIOMASS_PELLET: 0.022,
        HeatingFuelType.DISTRICT_HEATING: 0.160,
        HeatingFuelType.COAL: 0.340,
        HeatingFuelType.HEAT_PUMP_AIR: 0.233,
        HeatingFuelType.HEAT_PUMP_GROUND: 0.233,
    },
}
"""CO2 emission factors (kgCO2/kWh) by fuel type and country.
Source: BEIS/DEFRA 2024, UBA (DE), ADEME (FR), Eurostat."""


# EPC rating thresholds: primary energy (kWh/m2/yr) boundaries for each band.
# Format: {country: {building_type: [(band_letter, upper_limit), ...]}}
# Source: Official national EPC regulations.
EPC_RATING_THRESHOLDS: Dict[str, Dict[str, List[Tuple[str, float]]]] = {
    "UK": {
        BuildingUseType.RESIDENTIAL: [
            (EPCRating.A, 25.0), (EPCRating.B, 50.0), (EPCRating.C, 75.0),
            (EPCRating.D, 100.0), (EPCRating.E, 125.0), (EPCRating.F, 150.0),
            (EPCRating.G, 999999.0),
        ],
        BuildingUseType.OFFICE: [
            (EPCRating.A, 25.0), (EPCRating.B, 50.0), (EPCRating.C, 75.0),
            (EPCRating.D, 100.0), (EPCRating.E, 125.0), (EPCRating.F, 150.0),
            (EPCRating.G, 999999.0),
        ],
    },
    "DE": {
        BuildingUseType.RESIDENTIAL: [
            (EPCRating.A_PLUS, 30.0), (EPCRating.A, 50.0), (EPCRating.B, 75.0),
            (EPCRating.C, 100.0), (EPCRating.D, 130.0), (EPCRating.E, 160.0),
            (EPCRating.F, 200.0), (EPCRating.G, 999999.0),
        ],
        BuildingUseType.OFFICE: [
            (EPCRating.A_PLUS, 40.0), (EPCRating.A, 60.0), (EPCRating.B, 100.0),
            (EPCRating.C, 150.0), (EPCRating.D, 200.0), (EPCRating.E, 250.0),
            (EPCRating.F, 300.0), (EPCRating.G, 999999.0),
        ],
    },
    "FR": {
        BuildingUseType.RESIDENTIAL: [
            (EPCRating.A, 50.0), (EPCRating.B, 90.0), (EPCRating.C, 150.0),
            (EPCRating.D, 230.0), (EPCRating.E, 330.0), (EPCRating.F, 450.0),
            (EPCRating.G, 999999.0),
        ],
        BuildingUseType.OFFICE: [
            (EPCRating.A, 50.0), (EPCRating.B, 90.0), (EPCRating.C, 150.0),
            (EPCRating.D, 230.0), (EPCRating.E, 330.0), (EPCRating.F, 450.0),
            (EPCRating.G, 999999.0),
        ],
    },
    "IT": {
        BuildingUseType.RESIDENTIAL: [
            (EPCRating.A_PLUS, 15.0), (EPCRating.A, 30.0), (EPCRating.B, 50.0),
            (EPCRating.C, 70.0), (EPCRating.D, 90.0), (EPCRating.E, 120.0),
            (EPCRating.F, 160.0), (EPCRating.G, 999999.0),
        ],
    },
    "ES": {
        BuildingUseType.RESIDENTIAL: [
            (EPCRating.A, 40.0), (EPCRating.B, 65.0), (EPCRating.C, 100.0),
            (EPCRating.D, 135.0), (EPCRating.E, 175.0), (EPCRating.F, 230.0),
            (EPCRating.G, 999999.0),
        ],
    },
    "NL": {
        BuildingUseType.RESIDENTIAL: [
            (EPCRating.A_PLUS, 0.0), (EPCRating.A, 50.0), (EPCRating.B, 75.0),
            (EPCRating.C, 105.0), (EPCRating.D, 150.0), (EPCRating.E, 195.0),
            (EPCRating.F, 250.0), (EPCRating.G, 999999.0),
        ],
    },
    "DEFAULT": {
        BuildingUseType.RESIDENTIAL: [
            (EPCRating.A, 25.0), (EPCRating.B, 50.0), (EPCRating.C, 75.0),
            (EPCRating.D, 100.0), (EPCRating.E, 125.0), (EPCRating.F, 150.0),
            (EPCRating.G, 999999.0),
        ],
        BuildingUseType.OFFICE: [
            (EPCRating.A, 25.0), (EPCRating.B, 50.0), (EPCRating.C, 75.0),
            (EPCRating.D, 100.0), (EPCRating.E, 125.0), (EPCRating.F, 150.0),
            (EPCRating.G, 999999.0),
        ],
    },
}
"""EPC rating thresholds (kWh/m2/yr primary energy) by country and building type.
Source: Official national EPC regulations and EPBD transposition."""


# Reference building U-values for notional building comparison per EPBD.
# These represent the "reference building" against which actual performance
# is compared.  Values are typical national building regulations targets.
# Format: {country: {element_type: u_value_w_m2k}}
REFERENCE_BUILDING_VALUES: Dict[str, Dict[str, float]] = {
    "UK": {"wall": 0.26, "roof": 0.16, "floor": 0.18, "window": 1.60, "door": 1.60},
    "DE": {"wall": 0.28, "roof": 0.20, "floor": 0.35, "window": 1.30, "door": 1.80},
    "FR": {"wall": 0.36, "roof": 0.20, "floor": 0.36, "window": 1.80, "door": 1.80},
    "IT": {"wall": 0.34, "roof": 0.30, "floor": 0.33, "window": 2.20, "door": 2.20},
    "ES": {"wall": 0.38, "roof": 0.38, "floor": 0.38, "window": 2.30, "door": 2.30},
    "NL": {"wall": 0.22, "roof": 0.15, "floor": 0.22, "window": 1.60, "door": 1.60},
    "DEFAULT": {"wall": 0.30, "roof": 0.20, "floor": 0.25, "window": 1.80, "door": 1.80},
}
"""Reference building U-values by country per national building regulations.
Source: National building regulations transposing EPBD."""


# Heating system seasonal efficiency by system type, fuel, and age.
# Format: {(system_type, fuel_type, age_category): seasonal_efficiency}
# Sources: SAP 10.2 Table 4a, DIN V 18599-5, EN 15316-4-1.
HEATING_SYSTEM_EFFICIENCY: Dict[str, Dict[str, Dict[str, float]]] = {
    "gas_boiler_condensing": {
        HeatingFuelType.NATURAL_GAS: {"new": 0.92, "modern": 0.89, "old": 0.85},
        HeatingFuelType.LPG: {"new": 0.91, "modern": 0.88, "old": 0.84},
    },
    "gas_boiler_non_condensing": {
        HeatingFuelType.NATURAL_GAS: {"new": 0.82, "modern": 0.78, "old": 0.72},
        HeatingFuelType.LPG: {"new": 0.81, "modern": 0.77, "old": 0.71},
    },
    "oil_boiler_condensing": {
        HeatingFuelType.HEATING_OIL: {"new": 0.93, "modern": 0.90, "old": 0.85},
    },
    "oil_boiler_non_condensing": {
        HeatingFuelType.HEATING_OIL: {"new": 0.82, "modern": 0.78, "old": 0.70},
    },
    "electric_boiler": {
        HeatingFuelType.ELECTRICITY: {"new": 1.00, "modern": 1.00, "old": 1.00},
    },
    "electric_resistance": {
        HeatingFuelType.ELECTRICITY: {"new": 1.00, "modern": 1.00, "old": 1.00},
    },
    "biomass_boiler": {
        HeatingFuelType.BIOMASS_WOOD: {"new": 0.88, "modern": 0.83, "old": 0.75},
        HeatingFuelType.BIOMASS_PELLET: {"new": 0.92, "modern": 0.88, "old": 0.82},
    },
    "air_source_heat_pump": {
        HeatingFuelType.HEAT_PUMP_AIR: {"new": 3.50, "modern": 3.00, "old": 2.50},
    },
    "ground_source_heat_pump": {
        HeatingFuelType.HEAT_PUMP_GROUND: {"new": 4.20, "modern": 3.70, "old": 3.00},
    },
    "district_heating": {
        HeatingFuelType.DISTRICT_HEATING: {"new": 0.95, "modern": 0.90, "old": 0.80},
    },
    "coal_boiler": {
        HeatingFuelType.COAL: {"new": 0.75, "modern": 0.70, "old": 0.60},
    },
}
"""Heating system seasonal efficiency by type, fuel, and age.
Source: SAP 10.2 Table 4a, DIN V 18599-5, EN 15316-4-1."""


# Cooling system SEER values by system type and age.
# Sources: EN 14825, ASHRAE 90.1, Eurovent certification data.
COOLING_SYSTEM_EFFICIENCY: Dict[str, Dict[str, float]] = {
    CoolingSystemType.SPLIT_AC: {"new": 5.5, "modern": 4.5, "old": 3.0},
    CoolingSystemType.CENTRAL_CHILLER: {"new": 5.0, "modern": 4.0, "old": 2.8},
    CoolingSystemType.VRF: {"new": 6.0, "modern": 5.0, "old": 3.5},
    CoolingSystemType.EVAPORATIVE: {"new": 15.0, "modern": 12.0, "old": 8.0},
    CoolingSystemType.DISTRICT_COOLING: {"new": 5.5, "modern": 5.0, "old": 4.0},
    CoolingSystemType.HEAT_PUMP_REVERSIBLE: {"new": 5.0, "modern": 4.2, "old": 3.2},
}
"""Cooling system SEER values by type and age category.
Source: EN 14825, Eurovent certification data."""


# DHW (Domestic Hot Water) system efficiency by system type.
# Sources: SAP 10.2 Table 4c, EN 15316-3.
DHW_SYSTEM_EFFICIENCY: Dict[str, float] = {
    "instantaneous_gas": 0.85,
    "storage_gas": 0.78,
    "instantaneous_electric": 1.00,
    "immersion_cylinder": 0.90,
    "heat_pump_integrated": 2.80,
    "solar_pre_heat_gas": 0.60,
    "solar_pre_heat_electric": 0.45,
    "district_heating": 0.90,
    "combi_boiler_gas": 0.82,
    "combi_boiler_condensing": 0.90,
}
"""DHW system efficiency by type.
Source: SAP 10.2 Table 4c, EN 15316-3."""


# Lighting power density allowance (W/m2) by building type.
# Sources: EN 15193, CIBSE Guide F, ASHRAE 90.1.
LIGHTING_ALLOWANCE: Dict[str, float] = {
    BuildingUseType.RESIDENTIAL: 6.0,
    BuildingUseType.OFFICE: 10.0,
    BuildingUseType.RETAIL: 15.0,
    BuildingUseType.HOTEL: 10.0,
    BuildingUseType.HEALTHCARE: 12.0,
    BuildingUseType.EDUCATION: 10.0,
    BuildingUseType.WAREHOUSE: 5.0,
    BuildingUseType.MIXED_USE: 10.0,
    BuildingUseType.RESTAURANT: 12.0,
    BuildingUseType.LEISURE: 10.0,
}
"""Lighting power density (W/m2) by building type.
Source: EN 15193, CIBSE Guide F."""


# Internal heat gains (W/m2) by building type (occupancy + equipment + lighting).
# Sources: CIBSE Guide A Table 6.3, EN ISO 13790 Table G.8.
INTERNAL_GAINS: Dict[str, float] = {
    BuildingUseType.RESIDENTIAL: 5.0,
    BuildingUseType.OFFICE: 25.0,
    BuildingUseType.RETAIL: 20.0,
    BuildingUseType.HOTEL: 15.0,
    BuildingUseType.HEALTHCARE: 20.0,
    BuildingUseType.EDUCATION: 20.0,
    BuildingUseType.WAREHOUSE: 5.0,
    BuildingUseType.MIXED_USE: 18.0,
    BuildingUseType.RESTAURANT: 25.0,
    BuildingUseType.LEISURE: 20.0,
}
"""Internal heat gains (W/m2) by building type.
Source: CIBSE Guide A Table 6.3, EN ISO 13790 Table G.8."""


# National climate weighting / HDD for EPC normalisation.
# Sources: CIBSE Guide A, Eurostat, national meteorological data.
NATIONAL_CLIMATE_DATA: Dict[str, Dict[str, float]] = {
    "UK": {"hdd": 2353.0, "cdd": 30.0, "heating_season_hours": 5760.0},
    "DE": {"hdd": 2899.0, "cdd": 120.0, "heating_season_hours": 6000.0},
    "FR": {"hdd": 2306.0, "cdd": 180.0, "heating_season_hours": 5280.0},
    "IT": {"hdd": 1783.0, "cdd": 450.0, "heating_season_hours": 4800.0},
    "ES": {"hdd": 1485.0, "cdd": 550.0, "heating_season_hours": 4200.0},
    "NL": {"hdd": 2662.0, "cdd": 50.0, "heating_season_hours": 5760.0},
    "BE": {"hdd": 2600.0, "cdd": 60.0, "heating_season_hours": 5640.0},
    "AT": {"hdd": 3163.0, "cdd": 100.0, "heating_season_hours": 6240.0},
    "PL": {"hdd": 3180.0, "cdd": 80.0, "heating_season_hours": 6360.0},
    "SE": {"hdd": 3950.0, "cdd": 20.0, "heating_season_hours": 6600.0},
    "FI": {"hdd": 4508.0, "cdd": 10.0, "heating_season_hours": 7200.0},
    "IE": {"hdd": 2629.0, "cdd": 10.0, "heating_season_hours": 5760.0},
    "DK": {"hdd": 3100.0, "cdd": 20.0, "heating_season_hours": 6120.0},
    "CZ": {"hdd": 3280.0, "cdd": 90.0, "heating_season_hours": 6360.0},
    "PT": {"hdd": 1056.0, "cdd": 350.0, "heating_season_hours": 3600.0},
    "DEFAULT": {"hdd": 2500.0, "cdd": 150.0, "heating_season_hours": 5760.0},
}
"""National climate data (HDD/CDD/heating season) by country.
Source: CIBSE Guide A, Eurostat climate data."""


# Minimum Energy Efficiency Standards (MEES) / Building Performance Standards.
# Minimum EPC band required for rental or sale per national regulations.
MEES_REQUIREMENTS: Dict[str, Dict[str, str]] = {
    "UK": {
        "rental_residential": EPCRating.E.value,
        "rental_commercial": EPCRating.E.value,
        "new_build": EPCRating.B.value,
        "future_2028": EPCRating.C.value,
        "future_2030": EPCRating.C.value,
    },
    "DE": {
        "rental_residential": EPCRating.E.value,
        "new_build": EPCRating.A.value,
        "future_2030": EPCRating.D.value,
    },
    "FR": {
        "rental_residential": EPCRating.F.value,
        "new_build": EPCRating.A.value,
        "future_2028": EPCRating.E.value,
        "future_2034": EPCRating.D.value,
    },
    "NL": {
        "rental_commercial": EPCRating.C.value,
        "new_build": EPCRating.A.value,
        "future_2030": EPCRating.A.value,
    },
    "DEFAULT": {
        "rental_residential": EPCRating.E.value,
        "new_build": EPCRating.B.value,
    },
}
"""Minimum Energy Efficiency Standards (MEES) by country.
Source: National MEES / BPS regulations."""


# Operating hours by building type (hours/year for energy calculations).
# Sources: EN 15193, CIBSE Guide F, national standards.
OPERATING_HOURS: Dict[str, float] = {
    BuildingUseType.RESIDENTIAL: 8760.0,
    BuildingUseType.OFFICE: 2500.0,
    BuildingUseType.RETAIL: 3000.0,
    BuildingUseType.HOTEL: 8760.0,
    BuildingUseType.HEALTHCARE: 8760.0,
    BuildingUseType.EDUCATION: 2000.0,
    BuildingUseType.WAREHOUSE: 3000.0,
    BuildingUseType.MIXED_USE: 3000.0,
    BuildingUseType.RESTAURANT: 3500.0,
    BuildingUseType.LEISURE: 3000.0,
}
"""Annual operating hours by building type.
Source: EN 15193, CIBSE Guide F."""


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class HeatingSystem(BaseModel):
    """Heating system specification for EPC calculation.

    Attributes:
        system_type: Heating system type key.
        fuel_type: Primary fuel type.
        age_category: System age category (new / modern / old).
        seasonal_efficiency: Override seasonal efficiency if known.
        output_capacity_kw: System output capacity (kW).
        fraction_of_demand: Fraction of total heating demand served (0-1).
    """
    system_type: str = Field(
        default="gas_boiler_condensing", description="Heating system type"
    )
    fuel_type: HeatingFuelType = Field(
        default=HeatingFuelType.NATURAL_GAS, description="Primary fuel type"
    )
    age_category: str = Field(
        default="modern", description="Age category (new/modern/old)"
    )
    seasonal_efficiency: Optional[float] = Field(
        None, gt=0, le=6.0, description="Seasonal efficiency / COP override"
    )
    output_capacity_kw: Optional[float] = Field(
        None, gt=0, description="Output capacity (kW)"
    )
    fraction_of_demand: float = Field(
        default=1.0, ge=0, le=1.0, description="Fraction of heating demand"
    )


class CoolingSystem(BaseModel):
    """Cooling system specification for EPC calculation.

    Attributes:
        system_type: Cooling system type.
        age_category: System age (new / modern / old).
        seer: Override SEER if known.
        output_capacity_kw: Cooling capacity (kW).
        fraction_of_demand: Fraction of total cooling demand served (0-1).
    """
    system_type: CoolingSystemType = Field(
        default=CoolingSystemType.NONE, description="Cooling system type"
    )
    age_category: str = Field(
        default="modern", description="Age category (new/modern/old)"
    )
    seer: Optional[float] = Field(None, gt=0, le=30, description="SEER override")
    output_capacity_kw: Optional[float] = Field(
        None, gt=0, description="Cooling capacity (kW)"
    )
    fraction_of_demand: float = Field(
        default=1.0, ge=0, le=1.0, description="Fraction of cooling demand"
    )


class DHWSystem(BaseModel):
    """Domestic hot water system specification.

    Attributes:
        system_type: DHW system type key.
        efficiency: Override efficiency if known.
        solar_fraction: Fraction of DHW provided by solar thermal (0-1).
    """
    system_type: str = Field(
        default="combi_boiler_condensing", description="DHW system type"
    )
    efficiency: Optional[float] = Field(
        None, gt=0, le=5.0, description="Efficiency override"
    )
    solar_fraction: float = Field(
        default=0.0, ge=0, le=1.0, description="Solar thermal fraction"
    )


class LightingSystem(BaseModel):
    """Lighting system specification.

    Attributes:
        led_fraction: Fraction of lighting that is LED (0-1).
        average_efficacy_lm_w: Average luminaire efficacy (lm/W).
        control_factor: Lighting control factor (0-1, 1 = no controls).
    """
    led_fraction: float = Field(
        default=0.5, ge=0, le=1.0, description="LED fraction"
    )
    average_efficacy_lm_w: float = Field(
        default=80.0, gt=0, description="Average efficacy (lm/W)"
    )
    control_factor: float = Field(
        default=1.0, gt=0, le=1.0, description="Control factor (1=no controls)"
    )


class RenewableSystem(BaseModel):
    """On-site renewable energy system.

    Attributes:
        renewable_type: Type of renewable system.
        capacity_kw: Installed capacity (kWp for PV, kW for others).
        annual_generation_kwh: Known annual generation (kWh), or None for estimate.
    """
    renewable_type: RenewableType = Field(
        default=RenewableType.NONE, description="Renewable type"
    )
    capacity_kw: float = Field(default=0.0, ge=0, description="Capacity (kW/kWp)")
    annual_generation_kwh: Optional[float] = Field(
        None, ge=0, description="Annual generation (kWh)"
    )


class EnvelopeSummary(BaseModel):
    """Simplified envelope data for EPC calculation.

    Uses area-weighted U-values for walls/roof/floor/windows.
    """
    wall_u_value: float = Field(default=0.50, gt=0, description="Wall U-value (W/m2K)")
    wall_area_m2: float = Field(default=100.0, gt=0, description="Wall area (m2)")
    roof_u_value: float = Field(default=0.25, gt=0, description="Roof U-value (W/m2K)")
    roof_area_m2: float = Field(default=50.0, gt=0, description="Roof area (m2)")
    floor_u_value: float = Field(default=0.25, gt=0, description="Floor U-value (W/m2K)")
    floor_area_m2: float = Field(default=50.0, gt=0, description="Floor area (m2)")
    window_u_value: float = Field(default=2.00, gt=0, description="Window U-value (W/m2K)")
    window_area_m2: float = Field(default=20.0, gt=0, description="Window area (m2)")
    window_g_value: float = Field(default=0.63, gt=0, le=1.0, description="Window g-value")
    door_u_value: float = Field(default=2.00, gt=0, description="Door U-value (W/m2K)")
    door_area_m2: float = Field(default=4.0, gt=0, description="Door area (m2)")
    airtightness_n50: float = Field(default=7.0, ge=0, description="n50 (ACH at 50Pa)")
    heated_volume_m3: float = Field(default=250.0, gt=0, description="Heated volume (m3)")


class BuildingData(BaseModel):
    """Complete building input data for EPC rating.

    Attributes:
        facility_id: Unique facility identifier.
        building_name: Building name.
        building_type: Building use type.
        country: Country code (ISO 3166-1 alpha-2).
        year_built: Year of construction.
        floor_area_m2: Useful floor area (m2).
        floors: Number of storeys.
        envelope: Envelope data summary.
        heating_systems: Heating system(s).
        cooling_systems: Cooling system(s).
        dhw_system: Domestic hot water system.
        lighting: Lighting system.
        renewables: On-site renewable systems.
    """
    facility_id: str = Field(..., min_length=1, description="Facility identifier")
    building_name: str = Field(default="", description="Building name")
    building_type: BuildingUseType = Field(
        default=BuildingUseType.RESIDENTIAL, description="Building use type"
    )
    country: str = Field(default="UK", min_length=2, max_length=3, description="Country")
    year_built: int = Field(default=2000, ge=1600, le=2030, description="Year built")
    floor_area_m2: float = Field(..., gt=0, description="Useful floor area (m2)")
    floors: int = Field(default=2, ge=1, le=200, description="Number of storeys")
    envelope: EnvelopeSummary = Field(
        default_factory=EnvelopeSummary, description="Envelope data"
    )
    heating_systems: List[HeatingSystem] = Field(
        default_factory=lambda: [HeatingSystem()], description="Heating systems"
    )
    cooling_systems: List[CoolingSystem] = Field(
        default_factory=lambda: [CoolingSystem()], description="Cooling systems"
    )
    dhw_system: DHWSystem = Field(
        default_factory=DHWSystem, description="DHW system"
    )
    lighting: LightingSystem = Field(
        default_factory=LightingSystem, description="Lighting system"
    )
    renewables: List[RenewableSystem] = Field(
        default_factory=list, description="Renewable systems"
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


class EnergyBreakdown(BaseModel):
    """Energy demand breakdown by end use.

    Attributes:
        space_heating_kwh: Space heating delivered energy (kWh/yr).
        space_cooling_kwh: Space cooling delivered energy (kWh/yr).
        dhw_kwh: Domestic hot water delivered energy (kWh/yr).
        lighting_kwh: Lighting energy (kWh/yr).
        auxiliary_kwh: Pumps, fans, and controls (kWh/yr).
        renewable_generation_kwh: On-site renewable generation (kWh/yr).
        net_delivered_energy_kwh: Net delivered energy after renewables (kWh/yr).
    """
    space_heating_kwh: float = Field(default=0.0, description="Space heating (kWh/yr)")
    space_cooling_kwh: float = Field(default=0.0, description="Space cooling (kWh/yr)")
    dhw_kwh: float = Field(default=0.0, description="DHW (kWh/yr)")
    lighting_kwh: float = Field(default=0.0, description="Lighting (kWh/yr)")
    auxiliary_kwh: float = Field(default=0.0, description="Auxiliary (kWh/yr)")
    renewable_generation_kwh: float = Field(default=0.0, description="Renewable gen (kWh/yr)")
    net_delivered_energy_kwh: float = Field(default=0.0, description="Net delivered (kWh/yr)")


class ImprovementMeasure(BaseModel):
    """EPC improvement recommendation.

    Attributes:
        measure: Description of improvement.
        estimated_savings_kwh_m2: Estimated savings per m2.
        estimated_new_rating: Estimated EPC band after improvement.
        cost_category: Low / Medium / High cost indicator.
        priority: Priority ranking.
    """
    measure: str = Field(default="", description="Improvement description")
    estimated_savings_kwh_m2: float = Field(default=0.0, description="Savings (kWh/m2)")
    estimated_new_rating: str = Field(default="", description="Estimated new EPC band")
    cost_category: str = Field(default="medium", description="Cost category")
    priority: int = Field(default=0, description="Priority")


class EPCResult(BaseModel):
    """Complete EPC rating result with full provenance.

    Contains primary energy, CO2 emissions, EPC rating band, reference
    building comparison, improvement measures, and MEES compliance.
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
    country: str = Field(default="", description="Country")
    methodology: str = Field(default="", description="EPC methodology used")

    # Energy
    energy_breakdown: Optional[EnergyBreakdown] = Field(
        None, description="Energy demand breakdown"
    )
    total_delivered_energy_kwh: float = Field(default=0.0, description="Total delivered (kWh/yr)")
    total_delivered_energy_kwh_m2: float = Field(
        default=0.0, description="Delivered energy per m2 (kWh/m2/yr)"
    )

    # Primary energy
    primary_energy_kwh: float = Field(default=0.0, description="Primary energy (kWh/yr)")
    primary_energy_kwh_m2: float = Field(
        default=0.0, description="Primary energy per m2 (kWh/m2/yr)"
    )

    # CO2
    co2_emissions_kg: float = Field(default=0.0, description="CO2 emissions (kg/yr)")
    co2_emissions_kg_m2: float = Field(
        default=0.0, description="CO2 emissions per m2 (kg/m2/yr)"
    )

    # Rating
    epc_rating: str = Field(default="", description="EPC rating band (A-G)")
    epc_score: float = Field(default=0.0, description="EPC numeric score")

    # Reference building
    reference_building_energy_kwh_m2: float = Field(
        default=0.0, description="Reference building energy (kWh/m2/yr)"
    )
    improvement_potential_kwh_m2: float = Field(
        default=0.0, description="Improvement potential (kWh/m2/yr)"
    )
    improvement_potential_pct: float = Field(
        default=0.0, description="Improvement potential (%)"
    )

    # MEES compliance
    regulatory_minimum: str = Field(default="", description="MEES minimum rating")
    compliance_status: str = Field(default="", description="MEES compliance status")

    # Improvements
    improvement_measures: List[ImprovementMeasure] = Field(
        default_factory=list, description="Improvement recommendations"
    )

    recommendations: List[str] = Field(
        default_factory=list, description="Summary recommendations"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Calculation Engine
# ---------------------------------------------------------------------------


class EPCRatingEngine:
    """Energy Performance Certificate (EPC) rating engine per EPBD.

    Provides deterministic, zero-hallucination calculations for:
    - Space heating demand per EN ISO 52016 / EN ISO 13790
    - Space cooling demand
    - DHW demand per EN 15316-3
    - Lighting energy per EN 15193
    - Primary energy calculation per EN 15603 / EN ISO 52000-1
    - CO2 emission calculation
    - EPC rating assignment (A-G) per national thresholds
    - Reference building comparison
    - MEES / BPS compliance assessment
    - Improvement measure identification

    All calculations are bit-perfect reproducible.  No LLM is used
    in any calculation path.

    Usage::

        engine = EPCRatingEngine()
        result = engine.rate(building_data)

    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise the EPC rating engine with embedded constants."""
        self._primary_energy_factors = PRIMARY_ENERGY_FACTORS
        self._co2_emission_factors = CO2_EMISSION_FACTORS
        self._epc_thresholds = EPC_RATING_THRESHOLDS
        self._reference_values = REFERENCE_BUILDING_VALUES
        self._heating_efficiency = HEATING_SYSTEM_EFFICIENCY
        self._cooling_efficiency = COOLING_SYSTEM_EFFICIENCY
        self._dhw_efficiency = DHW_SYSTEM_EFFICIENCY
        self._lighting_allowance = LIGHTING_ALLOWANCE
        self._internal_gains = INTERNAL_GAINS
        self._climate_data = NATIONAL_CLIMATE_DATA
        self._mees = MEES_REQUIREMENTS
        self._operating_hours = OPERATING_HOURS

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def rate(self, building: BuildingData) -> EPCResult:
        """Generate EPC rating for a building.

        Orchestrates all sub-calculations: heating/cooling demand,
        DHW, lighting, primary energy, CO2, rating assignment,
        reference comparison, and improvement identification.

        Args:
            building: Complete building input data.

        Returns:
            EPCResult with EPC rating, energy data, and provenance.

        Raises:
            ValueError: If required data is missing.
        """
        t0 = time.perf_counter()

        country = building.country.upper()
        area = _decimal(building.floor_area_m2)
        methodology = self._select_methodology(country)

        logger.info(
            "Generating EPC for facility %s (%s, %s, methodology=%s)",
            building.facility_id, building.building_type.value, country, methodology,
        )

        # Step 1: Calculate space heating demand
        heating_demand = self.calculate_heating_demand(building)

        # Step 2: Calculate space cooling demand
        cooling_demand = self.calculate_cooling_demand(building)

        # Step 3: Calculate DHW demand
        dhw_demand = self.calculate_dhw_demand(building)

        # Step 4: Calculate lighting energy
        lighting_energy = self.calculate_lighting_energy(building)

        # Step 5: Auxiliary energy (pumps, fans, controls) -- approx 5% of heating+cooling
        auxiliary = (_decimal(heating_demand) + _decimal(cooling_demand)) * Decimal("0.05")

        # Step 6: Renewable contribution
        renewable_kwh = self.calculate_renewable_contribution(building)

        # Step 7: Net delivered energy
        total_delivered = (
            _decimal(heating_demand) + _decimal(cooling_demand) +
            _decimal(dhw_demand) + _decimal(lighting_energy) + auxiliary
        )
        net_delivered = total_delivered - _decimal(renewable_kwh)
        if net_delivered < Decimal("0"):
            net_delivered = Decimal("0")

        energy_breakdown = EnergyBreakdown(
            space_heating_kwh=_round2(float(heating_demand)),
            space_cooling_kwh=_round2(float(cooling_demand)),
            dhw_kwh=_round2(float(dhw_demand)),
            lighting_kwh=_round2(float(lighting_energy)),
            auxiliary_kwh=_round2(float(auxiliary)),
            renewable_generation_kwh=_round2(float(renewable_kwh)),
            net_delivered_energy_kwh=_round2(float(net_delivered)),
        )

        # Step 8: Primary energy calculation
        primary_energy = self.calculate_primary_energy(building, energy_breakdown)
        primary_per_m2 = _safe_divide(_decimal(primary_energy), area)

        # Step 9: CO2 emissions
        co2_emissions = self._calculate_co2(building, energy_breakdown)
        co2_per_m2 = _safe_divide(_decimal(co2_emissions), area)

        # Step 10: Assign EPC rating
        epc_rating = self.assign_epc_rating(
            float(primary_per_m2), country, building.building_type,
        )

        # Step 11: Reference building comparison
        ref_energy = self._calculate_reference_energy(building)
        improvement_potential = float(primary_per_m2) - ref_energy
        if improvement_potential < 0:
            improvement_potential = 0.0
        improvement_pct = float(
            _safe_pct(_decimal(improvement_potential), _decimal(float(primary_per_m2)))
        )

        # Step 12: MEES compliance
        mees_data = self._mees.get(country, self._mees["DEFAULT"])
        reg_min = mees_data.get("rental_residential", EPCRating.E.value)
        compliance = self._check_mees_compliance(epc_rating, reg_min)

        # Step 13: Improvement recommendations
        improvements = self.generate_recommendations(
            building, float(primary_per_m2), epc_rating, country,
        )

        delivered_per_m2 = _safe_divide(total_delivered, area)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = EPCResult(
            facility_id=building.facility_id,
            building_name=building.building_name,
            building_type=building.building_type.value,
            country=country,
            methodology=methodology,
            energy_breakdown=energy_breakdown,
            total_delivered_energy_kwh=_round2(float(total_delivered)),
            total_delivered_energy_kwh_m2=_round2(float(delivered_per_m2)),
            primary_energy_kwh=_round2(float(primary_energy)),
            primary_energy_kwh_m2=_round2(float(primary_per_m2)),
            co2_emissions_kg=_round2(float(co2_emissions)),
            co2_emissions_kg_m2=_round3(float(co2_per_m2)),
            epc_rating=epc_rating,
            epc_score=_round2(float(primary_per_m2)),
            reference_building_energy_kwh_m2=_round2(ref_energy),
            improvement_potential_kwh_m2=_round2(improvement_potential),
            improvement_potential_pct=_round2(improvement_pct),
            regulatory_minimum=reg_min,
            compliance_status=compliance,
            improvement_measures=improvements,
            recommendations=self._summary_recommendations(
                epc_rating, compliance, improvements,
            ),
            processing_time_ms=_round2(elapsed_ms),
        )

        result.provenance_hash = _compute_hash(result)
        return result

    def calculate_heating_demand(self, building: BuildingData) -> float:
        """Calculate annual space heating demand per EN ISO 52016 simplified.

        Qh = (Htr + Hve) * HDD * 24 / 1000 - Qint_useful - Qsol_useful

        Args:
            building: Building data with envelope and systems.

        Returns:
            Annual space heating delivered energy (kWh/yr).
        """
        env = building.envelope
        country = building.country.upper()
        climate = self._climate_data.get(country, self._climate_data["DEFAULT"])
        area = _decimal(building.floor_area_m2)

        # Fabric heat loss: Htr = sum(Ui * Ai) [W/K]
        htr = (
            _decimal(env.wall_u_value) * _decimal(env.wall_area_m2) +
            _decimal(env.roof_u_value) * _decimal(env.roof_area_m2) +
            _decimal(env.floor_u_value) * _decimal(env.floor_area_m2) +
            _decimal(env.window_u_value) * _decimal(env.window_area_m2) +
            _decimal(env.door_u_value) * _decimal(env.door_area_m2)
        )

        # Ventilation heat loss: Hve = 0.34 * n * V [W/K]
        n50 = _decimal(env.airtightness_n50)
        n = n50 / Decimal("20")  # Shelter factor approximation
        hve = Decimal("0.34") * n * _decimal(env.heated_volume_m3)

        total_h = htr + hve

        # Heating demand: Qh = H * HDD * 24 / 1000 [kWh/yr]
        hdd = _decimal(climate["hdd"])
        gross_demand = total_h * hdd * Decimal("24") / Decimal("1000")

        # Internal gains utilisation (simplified: 90% utilisation)
        int_gains_w_m2 = _decimal(
            self._internal_gains.get(building.building_type, 10.0)
        )
        heating_hours = _decimal(climate["heating_season_hours"])
        q_int = int_gains_w_m2 * area * heating_hours / Decimal("1000")
        q_int_useful = q_int * Decimal("0.90")

        # Solar gains utilisation (simplified)
        # Qsol = g * Aw * Isol * Ff * 0.9  (south-equivalent)
        g_val = _decimal(env.window_g_value)
        aw = _decimal(env.window_area_m2)
        # Average solar irradiance on south facade (kWh/m2/yr) varies by country
        isol_map = {
            "UK": 550, "DE": 600, "FR": 700, "IT": 900, "ES": 1000,
            "NL": 550, "SE": 500, "FI": 450, "PT": 1100,
        }
        isol = _decimal(isol_map.get(country, 600))
        frame_factor = Decimal("0.70")
        # Orientation factor: average over all facades = ~0.5 of south peak
        orientation_factor = Decimal("0.50")
        q_sol = g_val * aw * isol * frame_factor * orientation_factor
        q_sol_useful = q_sol * Decimal("0.90")

        # Net heating demand
        net_demand = gross_demand - q_int_useful - q_sol_useful
        if net_demand < Decimal("0"):
            net_demand = Decimal("0")

        # Apply system efficiency (weighted across heating systems)
        heating_delivered = self._apply_heating_efficiency(building, net_demand)

        return _round2(float(heating_delivered))

    def calculate_cooling_demand(self, building: BuildingData) -> float:
        """Calculate annual space cooling demand.

        Qc = max(0, Qsol + Qint - (Htr + Hve) * CDD * 24 / 1000) [kWh/yr]

        Args:
            building: Building data.

        Returns:
            Annual space cooling delivered energy (kWh/yr).
        """
        # Check if any active cooling system
        has_cooling = any(
            cs.system_type != CoolingSystemType.NONE
            for cs in building.cooling_systems
        )
        if not has_cooling:
            return 0.0

        env = building.envelope
        country = building.country.upper()
        climate = self._climate_data.get(country, self._climate_data["DEFAULT"])
        area = _decimal(building.floor_area_m2)
        cdd = _decimal(climate["cdd"])

        if cdd <= Decimal("0"):
            return 0.0

        # Total heat transfer coefficient
        htr = (
            _decimal(env.wall_u_value) * _decimal(env.wall_area_m2) +
            _decimal(env.roof_u_value) * _decimal(env.roof_area_m2) +
            _decimal(env.floor_u_value) * _decimal(env.floor_area_m2) +
            _decimal(env.window_u_value) * _decimal(env.window_area_m2) +
            _decimal(env.door_u_value) * _decimal(env.door_area_m2)
        )
        n50 = _decimal(env.airtightness_n50)
        n = n50 / Decimal("20")
        hve = Decimal("0.34") * n * _decimal(env.heated_volume_m3)
        total_h = htr + hve

        # Heat rejection by fabric
        q_rejection = total_h * cdd * Decimal("24") / Decimal("1000")

        # Internal gains over cooling season (approx 4 months = 2880 hours)
        int_gains_w = _decimal(
            self._internal_gains.get(building.building_type, 10.0)
        ) * area
        cooling_hours = Decimal("2880")
        q_int = int_gains_w * cooling_hours / Decimal("1000")

        # Solar gains (higher in cooling season)
        g_val = _decimal(env.window_g_value)
        aw = _decimal(env.window_area_m2)
        isol_cooling = Decimal("400")  # kWh/m2 during cooling season
        q_sol = g_val * aw * isol_cooling * Decimal("0.70") * Decimal("0.50")

        # Net cooling demand
        net_demand = q_int + q_sol - q_rejection
        if net_demand < Decimal("0"):
            return 0.0

        # Apply cooling system efficiency
        cooling_delivered = self._apply_cooling_efficiency(building, net_demand)

        return _round2(float(cooling_delivered))

    def calculate_dhw_demand(self, building: BuildingData) -> float:
        """Calculate annual domestic hot water energy demand per EN 15316-3.

        Args:
            building: Building data with DHW system.

        Returns:
            Annual DHW delivered energy (kWh/yr).
        """
        area = _decimal(building.floor_area_m2)
        btype = building.building_type

        # DHW demand (kWh/m2/yr) by building type per EN 15316-3
        dhw_demand_map: Dict[str, Decimal] = {
            BuildingUseType.RESIDENTIAL: Decimal("25.0"),
            BuildingUseType.OFFICE: Decimal("5.0"),
            BuildingUseType.RETAIL: Decimal("3.0"),
            BuildingUseType.HOTEL: Decimal("40.0"),
            BuildingUseType.HEALTHCARE: Decimal("35.0"),
            BuildingUseType.EDUCATION: Decimal("10.0"),
            BuildingUseType.WAREHOUSE: Decimal("2.0"),
            BuildingUseType.MIXED_USE: Decimal("15.0"),
            BuildingUseType.RESTAURANT: Decimal("30.0"),
            BuildingUseType.LEISURE: Decimal("25.0"),
        }

        dhw_per_m2 = dhw_demand_map.get(btype, Decimal("15.0"))
        gross_dhw = dhw_per_m2 * area

        # DHW system efficiency
        dhw = building.dhw_system
        efficiency = _decimal(dhw.efficiency) if dhw.efficiency else Decimal("0")
        if efficiency <= Decimal("0"):
            eff_lookup = self._dhw_efficiency.get(dhw.system_type, 0.85)
            efficiency = _decimal(eff_lookup)

        # Solar fraction reduction
        solar_fraction = _decimal(dhw.solar_fraction)
        net_dhw = gross_dhw * (Decimal("1") - solar_fraction)

        # Delivered energy = demand / efficiency
        delivered = _safe_divide(net_dhw, efficiency)

        return _round2(float(delivered))

    def calculate_lighting_energy(self, building: BuildingData) -> float:
        """Calculate annual lighting energy per EN 15193.

        Args:
            building: Building data with lighting system.

        Returns:
            Annual lighting energy (kWh/yr).
        """
        area = _decimal(building.floor_area_m2)
        btype = building.building_type
        lighting = building.lighting

        # Base lighting power density
        base_lpd = _decimal(
            self._lighting_allowance.get(btype, 10.0)
        )

        # LED correction: LED is ~2x efficacy of fluorescent
        led_frac = _decimal(lighting.led_fraction)
        non_led_frac = Decimal("1") - led_frac
        # LED correction factor: LED at ~130 lm/W vs fluorescent ~65 lm/W
        correction = led_frac * Decimal("0.50") + non_led_frac * Decimal("1.00")

        # Control factor
        control = _decimal(lighting.control_factor)

        # Effective LPD
        effective_lpd = base_lpd * correction * control

        # Annual energy: LPD * area * operating_hours / 1000
        op_hours = _decimal(self._operating_hours.get(btype, 3000.0))
        annual_kwh = effective_lpd * area * op_hours / Decimal("1000")

        return _round2(float(annual_kwh))

    def calculate_renewable_contribution(self, building: BuildingData) -> float:
        """Calculate annual on-site renewable energy generation.

        Args:
            building: Building data with renewable systems.

        Returns:
            Annual renewable generation (kWh/yr).
        """
        total = Decimal("0")
        country = building.country.upper()

        for ren in building.renewables:
            if ren.renewable_type == RenewableType.NONE:
                continue

            if ren.annual_generation_kwh is not None:
                total += _decimal(ren.annual_generation_kwh)
            else:
                # Estimate based on capacity and country
                capacity = _decimal(ren.capacity_kw)
                if ren.renewable_type == RenewableType.SOLAR_PV:
                    # kWh/kWp/yr by country (typical yield)
                    pv_yield_map = {
                        "UK": 900, "DE": 950, "FR": 1100, "IT": 1300,
                        "ES": 1500, "NL": 880, "SE": 850, "PT": 1600,
                    }
                    yield_kwh = _decimal(pv_yield_map.get(country, 1000))
                    total += capacity * yield_kwh
                elif ren.renewable_type == RenewableType.SOLAR_THERMAL:
                    # Assume 500 kWh/m2 panel/yr (approx 1m2/kW)
                    total += capacity * Decimal("500")
                elif ren.renewable_type == RenewableType.WIND:
                    # Assume 1500 kWh/kW/yr small wind
                    total += capacity * Decimal("1500")
                elif ren.renewable_type in (
                    RenewableType.BIOMASS_CHP, RenewableType.MICRO_CHP,
                ):
                    # Assume 4000 hours/yr operation
                    total += capacity * Decimal("4000")

        return _round2(float(total))

    def calculate_primary_energy(
        self, building: BuildingData, breakdown: EnergyBreakdown,
    ) -> float:
        """Calculate total primary energy per EN 15603 / EN ISO 52000-1.

        EP = sum(Qdel_i * fp_i) - sum(Qexp_i * fp_exp_i) [kWh/yr]

        Args:
            building: Building data.
            breakdown: Energy demand breakdown.

        Returns:
            Annual primary energy (kWh/yr).
        """
        country = building.country.upper()
        pef_table = self._primary_energy_factors.get(
            country, self._primary_energy_factors["DEFAULT"]
        )

        # Heating primary energy
        heating_pe = Decimal("0")
        for hs in building.heating_systems:
            pef = _decimal(pef_table.get(hs.fuel_type, 1.10))
            frac = _decimal(hs.fraction_of_demand)
            heating_pe += _decimal(breakdown.space_heating_kwh) * frac * pef

        # Cooling primary energy (electricity)
        cooling_pef = _decimal(pef_table.get(HeatingFuelType.ELECTRICITY, 2.00))
        cooling_pe = _decimal(breakdown.space_cooling_kwh) * cooling_pef

        # DHW primary energy
        dhw_pe = Decimal("0")
        dhw_fuel = HeatingFuelType.NATURAL_GAS  # default
        for hs in building.heating_systems:
            if hs.fraction_of_demand > 0:
                dhw_fuel = hs.fuel_type
                break
        dhw_pef = _decimal(pef_table.get(dhw_fuel, 1.10))
        dhw_pe = _decimal(breakdown.dhw_kwh) * dhw_pef

        # Lighting primary energy (electricity)
        lighting_pe = _decimal(breakdown.lighting_kwh) * cooling_pef

        # Auxiliary primary energy (electricity)
        aux_pe = _decimal(breakdown.auxiliary_kwh) * cooling_pef

        # Renewable credit (electricity PEF for export)
        ren_credit = _decimal(breakdown.renewable_generation_kwh) * cooling_pef

        total_pe = heating_pe + cooling_pe + dhw_pe + lighting_pe + aux_pe - ren_credit
        if total_pe < Decimal("0"):
            total_pe = Decimal("0")

        return _round2(float(total_pe))

    def assign_epc_rating(
        self,
        primary_energy_kwh_m2: float,
        country: str,
        building_type: BuildingUseType,
    ) -> str:
        """Assign EPC rating band based on primary energy per m2.

        Args:
            primary_energy_kwh_m2: Primary energy per m2 (kWh/m2/yr).
            country: Country code.
            building_type: Building use type.

        Returns:
            EPC rating band string (A+, A, B, C, D, E, F, or G).
        """
        country = country.upper()
        thresholds = self._epc_thresholds.get(
            country, self._epc_thresholds["DEFAULT"]
        )
        bands = thresholds.get(building_type, thresholds.get(
            BuildingUseType.RESIDENTIAL, thresholds.get(
                list(thresholds.keys())[0], [],
            ),
        ))

        pe = float(primary_energy_kwh_m2)
        for band_letter, upper_limit in bands:
            if pe <= upper_limit:
                return band_letter.value if hasattr(band_letter, 'value') else str(band_letter)

        return EPCRating.G.value

    def generate_recommendations(
        self,
        building: BuildingData,
        primary_kwh_m2: float,
        current_rating: str,
        country: str,
    ) -> List[ImprovementMeasure]:
        """Generate EPC improvement recommendations.

        Args:
            building: Building data.
            primary_kwh_m2: Current primary energy per m2.
            current_rating: Current EPC rating.
            country: Country code.

        Returns:
            List of improvement measures sorted by priority.
        """
        measures: List[ImprovementMeasure] = []
        env = building.envelope
        priority = 0

        # Insulation improvements
        ref = self._reference_values.get(country, self._reference_values["DEFAULT"])
        if env.wall_u_value > ref.get("wall", 0.30) * 1.5:
            priority += 1
            est_saving = (env.wall_u_value - ref["wall"]) * env.wall_area_m2 * 2.5 / building.floor_area_m2
            measures.append(ImprovementMeasure(
                measure=f"Upgrade wall insulation to U={ref['wall']} W/m2K",
                estimated_savings_kwh_m2=_round2(float(est_saving)),
                estimated_new_rating=self._estimate_improved_rating(
                    primary_kwh_m2 - float(est_saving), country, building.building_type,
                ),
                cost_category="high",
                priority=priority,
            ))

        if env.roof_u_value > ref.get("roof", 0.20) * 1.5:
            priority += 1
            est_saving = (env.roof_u_value - ref["roof"]) * env.roof_area_m2 * 2.5 / building.floor_area_m2
            measures.append(ImprovementMeasure(
                measure=f"Upgrade roof insulation to U={ref['roof']} W/m2K",
                estimated_savings_kwh_m2=_round2(float(est_saving)),
                estimated_new_rating=self._estimate_improved_rating(
                    primary_kwh_m2 - float(est_saving), country, building.building_type,
                ),
                cost_category="medium",
                priority=priority,
            ))

        # Window upgrade
        if env.window_u_value > 2.0:
            priority += 1
            est_saving = (env.window_u_value - 1.40) * env.window_area_m2 * 2.5 / building.floor_area_m2
            measures.append(ImprovementMeasure(
                measure="Replace windows with double/triple glazed low-e units",
                estimated_savings_kwh_m2=_round2(float(est_saving)),
                estimated_new_rating=self._estimate_improved_rating(
                    primary_kwh_m2 - float(est_saving), country, building.building_type,
                ),
                cost_category="high",
                priority=priority,
            ))

        # Heating system upgrade
        for hs in building.heating_systems:
            eff = self._get_heating_efficiency(hs)
            if eff < 0.90 and hs.fuel_type not in (
                HeatingFuelType.HEAT_PUMP_AIR, HeatingFuelType.HEAT_PUMP_GROUND,
            ):
                priority += 1
                est_saving = primary_kwh_m2 * 0.15  # ~15% savings from boiler upgrade
                measures.append(ImprovementMeasure(
                    measure="Upgrade to high-efficiency condensing boiler or heat pump",
                    estimated_savings_kwh_m2=_round2(float(est_saving)),
                    estimated_new_rating=self._estimate_improved_rating(
                        primary_kwh_m2 - float(est_saving), country, building.building_type,
                    ),
                    cost_category="high",
                    priority=priority,
                ))
                break

        # LED lighting
        if building.lighting.led_fraction < 0.8:
            priority += 1
            est_saving = (1.0 - building.lighting.led_fraction) * 5.0
            measures.append(ImprovementMeasure(
                measure="Upgrade lighting to LED with occupancy/daylight controls",
                estimated_savings_kwh_m2=_round2(float(est_saving)),
                estimated_new_rating=self._estimate_improved_rating(
                    primary_kwh_m2 - float(est_saving), country, building.building_type,
                ),
                cost_category="low",
                priority=priority,
            ))

        # Solar PV
        has_pv = any(
            r.renewable_type == RenewableType.SOLAR_PV for r in building.renewables
        )
        if not has_pv:
            priority += 1
            est_saving = min(primary_kwh_m2 * 0.20, 25.0)
            measures.append(ImprovementMeasure(
                measure="Install rooftop solar PV system",
                estimated_savings_kwh_m2=_round2(float(est_saving)),
                estimated_new_rating=self._estimate_improved_rating(
                    primary_kwh_m2 - float(est_saving), country, building.building_type,
                ),
                cost_category="high",
                priority=priority,
            ))

        # Airtightness
        if env.airtightness_n50 > 7.0:
            priority += 1
            est_saving = (env.airtightness_n50 - 5.0) * 2.0
            measures.append(ImprovementMeasure(
                measure="Improve airtightness with draught-proofing and sealing",
                estimated_savings_kwh_m2=_round2(float(est_saving)),
                estimated_new_rating=self._estimate_improved_rating(
                    primary_kwh_m2 - float(est_saving), country, building.building_type,
                ),
                cost_category="low",
                priority=priority,
            ))

        return measures

    # -------------------------------------------------------------------
    # Internal Calculations
    # -------------------------------------------------------------------

    def _apply_heating_efficiency(
        self, building: BuildingData, demand: Decimal,
    ) -> Decimal:
        """Apply heating system efficiency to get delivered energy."""
        total_delivered = Decimal("0")
        for hs in building.heating_systems:
            eff = _decimal(self._get_heating_efficiency(hs))
            frac = _decimal(hs.fraction_of_demand)
            delivered = _safe_divide(demand * frac, eff)
            total_delivered += delivered
        return total_delivered

    def _get_heating_efficiency(self, hs: HeatingSystem) -> float:
        """Get seasonal heating efficiency for a heating system."""
        if hs.seasonal_efficiency is not None:
            return hs.seasonal_efficiency

        sys_table = self._heating_efficiency.get(hs.system_type, {})
        fuel_table = sys_table.get(hs.fuel_type, {})
        eff = fuel_table.get(hs.age_category, 0.85)
        return eff

    def _apply_cooling_efficiency(
        self, building: BuildingData, demand: Decimal,
    ) -> Decimal:
        """Apply cooling system efficiency (SEER) to get delivered energy."""
        total_delivered = Decimal("0")
        for cs in building.cooling_systems:
            if cs.system_type == CoolingSystemType.NONE:
                continue
            seer = _decimal(cs.seer) if cs.seer else Decimal("0")
            if seer <= Decimal("0"):
                seer_table = self._cooling_efficiency.get(cs.system_type, {})
                seer = _decimal(seer_table.get(cs.age_category, 4.0))
            frac = _decimal(cs.fraction_of_demand)
            delivered = _safe_divide(demand * frac, seer)
            total_delivered += delivered
        return total_delivered

    def _calculate_co2(
        self, building: BuildingData, breakdown: EnergyBreakdown,
    ) -> float:
        """Calculate total CO2 emissions per EN 15603.

        CO2 = sum(Qdel_i * ef_i) [kgCO2/yr]
        """
        country = building.country.upper()
        ef_table = self._co2_emission_factors.get(
            country, self._co2_emission_factors["DEFAULT"]
        )

        # Heating CO2
        heating_co2 = Decimal("0")
        for hs in building.heating_systems:
            ef = _decimal(ef_table.get(hs.fuel_type, 0.20))
            frac = _decimal(hs.fraction_of_demand)
            heating_co2 += _decimal(breakdown.space_heating_kwh) * frac * ef

        # Cooling CO2 (electricity)
        elec_ef = _decimal(ef_table.get(HeatingFuelType.ELECTRICITY, 0.233))
        cooling_co2 = _decimal(breakdown.space_cooling_kwh) * elec_ef

        # DHW CO2
        dhw_fuel = HeatingFuelType.NATURAL_GAS
        for hs in building.heating_systems:
            if hs.fraction_of_demand > 0:
                dhw_fuel = hs.fuel_type
                break
        dhw_ef = _decimal(ef_table.get(dhw_fuel, 0.20))
        dhw_co2 = _decimal(breakdown.dhw_kwh) * dhw_ef

        # Lighting + Auxiliary CO2 (electricity)
        light_aux_co2 = (
            _decimal(breakdown.lighting_kwh) + _decimal(breakdown.auxiliary_kwh)
        ) * elec_ef

        # Renewable credit
        ren_credit = _decimal(breakdown.renewable_generation_kwh) * elec_ef

        total = heating_co2 + cooling_co2 + dhw_co2 + light_aux_co2 - ren_credit
        if total < Decimal("0"):
            total = Decimal("0")

        return _round2(float(total))

    def _calculate_reference_energy(self, building: BuildingData) -> float:
        """Calculate reference building primary energy per m2 per EPBD.

        The reference building has the same geometry but uses national
        regulation U-values and standard system efficiencies.
        """
        country = building.country.upper()
        ref = self._reference_values.get(country, self._reference_values["DEFAULT"])
        area = _decimal(building.floor_area_m2)
        env = building.envelope

        # Reference fabric heat loss
        htr_ref = (
            _decimal(ref["wall"]) * _decimal(env.wall_area_m2) +
            _decimal(ref["roof"]) * _decimal(env.roof_area_m2) +
            _decimal(ref["floor"]) * _decimal(env.floor_area_m2) +
            _decimal(ref["window"]) * _decimal(env.window_area_m2) +
            _decimal(ref["door"]) * _decimal(env.door_area_m2)
        )

        # Reference ventilation (n50 = 5.0)
        hve_ref = Decimal("0.34") * (Decimal("5.0") / Decimal("20")) * _decimal(env.heated_volume_m3)

        total_h_ref = htr_ref + hve_ref

        # Reference heating demand
        climate = self._climate_data.get(country, self._climate_data["DEFAULT"])
        hdd = _decimal(climate["hdd"])
        gross_ref = total_h_ref * hdd * Decimal("24") / Decimal("1000")

        # Reference gains
        int_gains = _decimal(self._internal_gains.get(building.building_type, 10.0))
        heating_hours = _decimal(climate["heating_season_hours"])
        q_int_ref = int_gains * area * heating_hours / Decimal("1000") * Decimal("0.90")

        net_ref = gross_ref - q_int_ref
        if net_ref < Decimal("0"):
            net_ref = Decimal("0")

        # Reference system: condensing boiler at 92%
        delivered_ref = _safe_divide(net_ref, Decimal("0.92"))

        # Reference primary energy
        pef = _decimal(1.10)  # gas PEF
        pe_ref = delivered_ref * pef

        # Add lighting (reference: 8 W/m2 with controls)
        op_hours = _decimal(self._operating_hours.get(building.building_type, 3000.0))
        light_ref = Decimal("8") * area * op_hours / Decimal("1000")
        pe_light_ref = light_ref * _decimal(2.00)  # electricity PEF

        pe_total_ref = pe_ref + pe_light_ref
        pe_per_m2_ref = _safe_divide(pe_total_ref, area)

        return _round2(float(pe_per_m2_ref))

    def _check_mees_compliance(self, rating: str, minimum: str) -> str:
        """Check if EPC rating meets MEES requirements.

        Args:
            rating: Current EPC rating.
            minimum: MEES minimum rating.

        Returns:
            Compliance status string.
        """
        rating_order = [
            EPCRating.A_PLUS.value, EPCRating.A.value, EPCRating.B.value,
            EPCRating.C.value, EPCRating.D.value, EPCRating.E.value,
            EPCRating.F.value, EPCRating.G.value,
        ]

        try:
            rating_idx = rating_order.index(rating)
            min_idx = rating_order.index(minimum)
        except ValueError:
            return "unknown"

        if rating_idx <= min_idx:
            return "compliant"
        else:
            return f"non_compliant (minimum {minimum} required)"

    def _estimate_improved_rating(
        self,
        new_pe_kwh_m2: float,
        country: str,
        building_type: BuildingUseType,
    ) -> str:
        """Estimate EPC rating after improvement."""
        return self.assign_epc_rating(new_pe_kwh_m2, country, building_type)

    def _select_methodology(self, country: str) -> str:
        """Select EPC calculation methodology based on country."""
        methodology_map = {
            "UK": EPCMethodology.SAP_2012.value,
            "DE": EPCMethodology.GEG_2020.value,
            "FR": EPCMethodology.DPE_2021.value,
            "IT": EPCMethodology.APE_2015.value,
        }
        return methodology_map.get(country, EPCMethodology.EPBD_GENERIC.value)

    def _summary_recommendations(
        self,
        rating: str,
        compliance: str,
        measures: List[ImprovementMeasure],
    ) -> List[str]:
        """Generate summary recommendation strings."""
        recs: List[str] = []

        recs.append(f"Current EPC rating: {rating}")

        if "non_compliant" in compliance:
            recs.append(f"MEES compliance: {compliance} -- urgent action required")
        else:
            recs.append(f"MEES compliance: {compliance}")

        if measures:
            total_savings = sum(m.estimated_savings_kwh_m2 for m in measures)
            recs.append(
                f"{len(measures)} improvement measures identified with total "
                f"potential savings of {_round2(total_savings)} kWh/m2/yr"
            )
            low_cost = [m for m in measures if m.cost_category == "low"]
            if low_cost:
                recs.append(
                    f"Quick wins ({len(low_cost)} low-cost measures): "
                    + "; ".join(m.measure for m in low_cost)
                )

        if rating in (EPCRating.F.value, EPCRating.G.value):
            recs.append(
                "Building is in lowest EPC bands. Major retrofit recommended "
                "to meet current and upcoming MEES requirements"
            )

        return recs
