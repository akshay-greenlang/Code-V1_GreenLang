# -*- coding: utf-8 -*-
"""
WholeLifeCarbonEngine - PACK-032 Building Energy Assessment Engine 10
=====================================================================

Whole life carbon (WLC) assessment per EN 15978 lifecycle stages A1-D.
Calculates embodied carbon (A1-A3), transport (A4), construction (A5),
operational energy (B6) and water (B7), maintenance/replacement (B2-B5),
end-of-life (C1-C4), and benefits beyond the system boundary (Module D).
Compares against RIBA 2030, LETI, GLA, and DGNB carbon budgets.

Calculation Methodology:
    Embodied Carbon (A1-A3):
        EC = sum( mass_i * ECF_i )  [kgCO2e]
        where ECF_i = embodied carbon factor from ICE Database / EPD

    Transport to Site (A4):
        EC_A4 = sum( mass_i * distance_i * TEF_mode )  [kgCO2e]

    Construction (A5):
        EC_A5 = 0.03-0.05 * EC_A1A3  (typically 3-5% of product stage)

    Operational Energy (B6):
        OC = sum( E_annual * EF_grid_year,  year=1..study_period )
        Using grid decarbonisation projections

    Replacement (B4):
        EC_B4 = sum( EC_material * (ceil(study_period / lifetime) - 1) )

    End of Life (C1-C4):
        EC_C = 0.01-0.03 * EC_A1A3  (typically 1-3% of product stage)

    Module D (Beyond Lifecycle):
        Credits for recycling, reuse, energy recovery

    Whole Life Carbon:
        WLC = A1-A5 + B1-B7 + C1-C4 + D  [kgCO2e/m2]

Regulatory References:
    - EN 15978:2011 - Sustainability of construction works
    - EN 15804:2012+A2:2019 - Environmental Product Declarations
    - RICS Whole Life Carbon Assessment (2017)
    - RIBA 2030 Climate Challenge targets
    - LETI Climate Emergency Design Guide (2020)
    - GLA Whole Life Carbon Assessment (2022)
    - Level(s) framework (EU indicator 1.2)

Zero-Hallucination:
    - Embodied carbon factors from ICE Database v3.0 (Bath)
    - Grid decarbonisation from national energy projections
    - All calculations deterministic Decimal arithmetic
    - No LLM in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-032 Building Energy Assessment
Engine:  10 of 10
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

def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to *places* and return a float."""
    quantizer = Decimal(10) ** -places
    return float(value.quantize(quantizer, rounding=ROUND_HALF_UP))

def _round1(value: float) -> float:
    """Round to 1 decimal place using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))

def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

def _round4(value: float) -> float:
    """Round to 4 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

def _ceil_div(a: int, b: int) -> int:
    """Ceiling integer division."""
    if b <= 0:
        return 0
    return (a + b - 1) // b

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LifecycleStage(str, Enum):
    """EN 15978 lifecycle stages for whole life carbon assessment."""
    A1_RAW_MATERIAL = "A1_raw_material"
    A2_TRANSPORT = "A2_transport"
    A3_MANUFACTURING = "A3_manufacturing"
    A4_TRANSPORT_TO_SITE = "A4_transport_to_site"
    A5_CONSTRUCTION = "A5_construction"
    B1_USE = "B1_use"
    B2_MAINTENANCE = "B2_maintenance"
    B3_REPAIR = "B3_repair"
    B4_REPLACEMENT = "B4_replacement"
    B5_REFURBISHMENT = "B5_refurbishment"
    B6_OPERATIONAL_ENERGY = "B6_operational_energy"
    B7_OPERATIONAL_WATER = "B7_operational_water"
    C1_DECONSTRUCTION = "C1_deconstruction"
    C2_TRANSPORT_TO_DISPOSAL = "C2_transport_to_disposal"
    C3_WASTE_PROCESSING = "C3_waste_processing"
    C4_DISPOSAL = "C4_disposal"
    D_BENEFITS_BEYOND = "D_benefits_beyond"

class MaterialCategory(str, Enum):
    """Material categories for embodied carbon assessment."""
    CONCRETE = "concrete"
    STEEL_STRUCTURAL = "steel_structural"
    STEEL_REBAR = "steel_rebar"
    ALUMINIUM = "aluminium"
    TIMBER_SOFTWOOD = "timber_softwood"
    TIMBER_HARDWOOD = "timber_hardwood"
    TIMBER_CLT = "timber_CLT"
    TIMBER_GLULAM = "timber_glulam"
    BRICK = "brick"
    BLOCK_CONCRETE = "block_concrete"
    GLASS = "glass"
    INSULATION_MINERAL_WOOL = "insulation_mineral_wool"
    INSULATION_EPS = "insulation_EPS"
    INSULATION_XPS = "insulation_XPS"
    INSULATION_PIR = "insulation_PIR"
    PLASTERBOARD = "plasterboard"
    COPPER = "copper"
    PVC = "PVC"
    BITUMEN = "bitumen"
    STONE = "stone"
    AGGREGATE = "aggregate"

class CarbonTarget(str, Enum):
    """Carbon budget target standards for comparison."""
    RIBA_2030 = "RIBA_2030"
    LETI_2020 = "LETI_2020"
    GLA_2022 = "GLA_2022"
    RICS_2017 = "RICS_2017"
    DGNB = "DGNB"

class TransportMode(str, Enum):
    """Transport mode for material delivery."""
    TRUCK_RIGID = "truck_rigid"
    TRUCK_ARTICULATED = "truck_articulated"
    RAIL = "rail"
    SHIP_CONTAINER = "ship_container"
    SHIP_BULK = "ship_bulk"
    VAN = "van"

class BuildingTypeWLC(str, Enum):
    """Building typology for carbon budget comparison."""
    RESIDENTIAL = "residential"
    OFFICE = "office"
    SCHOOL = "school"
    HOSPITAL = "hospital"
    RETAIL = "retail"
    HOTEL = "hotel"
    INDUSTRIAL = "industrial"
    MIXED_USE = "mixed_use"

# ---------------------------------------------------------------------------
# Constants -- Embodied Carbon Factors (kgCO2e per unit)
# ---------------------------------------------------------------------------
# Source: ICE Database v3.0 (University of Bath), National EPD databases
# unit: per_kg, per_m2, or per_m3

EMBODIED_CARBON_FACTORS: Dict[str, Dict[str, str]] = {
    # Concrete by strength class (kgCO2e per m3)
    "concrete_C20_25": {"factor": "150", "unit": "per_m3", "density_kg_m3": "2350", "source": "ICE v3"},
    "concrete_C25_30": {"factor": "190", "unit": "per_m3", "density_kg_m3": "2350", "source": "ICE v3"},
    "concrete_C30_37": {"factor": "240", "unit": "per_m3", "density_kg_m3": "2400", "source": "ICE v3"},
    "concrete_C32_40": {"factor": "280", "unit": "per_m3", "density_kg_m3": "2400", "source": "ICE v3"},
    "concrete_C40_50": {"factor": "330", "unit": "per_m3", "density_kg_m3": "2450", "source": "ICE v3"},
    "concrete_C50_60": {"factor": "400", "unit": "per_m3", "density_kg_m3": "2450", "source": "ICE v3"},
    "concrete_C28_35_30pct_GGBS": {"factor": "170", "unit": "per_m3", "density_kg_m3": "2400", "source": "ICE v3"},
    "concrete_C28_35_50pct_GGBS": {"factor": "140", "unit": "per_m3", "density_kg_m3": "2400", "source": "ICE v3"},
    # Steel (kgCO2e per kg)
    "steel_structural": {"factor": "1.55", "unit": "per_kg", "source": "ICE v3"},
    "steel_structural_recycled_60pct": {"factor": "0.96", "unit": "per_kg", "source": "ICE v3"},
    "steel_rebar": {"factor": "1.99", "unit": "per_kg", "source": "ICE v3"},
    "steel_rebar_recycled_97pct": {"factor": "0.49", "unit": "per_kg", "source": "ICE v3"},
    "steel_stainless": {"factor": "6.15", "unit": "per_kg", "source": "ICE v3"},
    "steel_cold_formed": {"factor": "2.44", "unit": "per_kg", "source": "ICE v3"},
    # Aluminium (kgCO2e per kg)
    "aluminium_primary": {"factor": "9.16", "unit": "per_kg", "source": "ICE v3"},
    "aluminium_recycled": {"factor": "1.81", "unit": "per_kg", "source": "ICE v3"},
    "aluminium_extruded": {"factor": "10.60", "unit": "per_kg", "source": "ICE v3"},
    "aluminium_cast": {"factor": "11.50", "unit": "per_kg", "source": "ICE v3"},
    # Timber (kgCO2e per kg) -- note negative values include biogenic carbon
    "timber_softwood_general": {"factor": "0.31", "unit": "per_kg", "biogenic": "-1.63", "source": "ICE v3"},
    "timber_hardwood_general": {"factor": "0.41", "unit": "per_kg", "biogenic": "-1.63", "source": "ICE v3"},
    "timber_CLT": {"factor": "0.42", "unit": "per_kg", "biogenic": "-1.63", "density_kg_m3": "480", "source": "ICE v3"},
    "timber_glulam": {"factor": "0.51", "unit": "per_kg", "biogenic": "-1.63", "density_kg_m3": "460", "source": "ICE v3"},
    "timber_plywood": {"factor": "0.68", "unit": "per_kg", "biogenic": "-1.63", "source": "ICE v3"},
    "timber_OSB": {"factor": "0.45", "unit": "per_kg", "biogenic": "-1.63", "source": "ICE v3"},
    "timber_MDF": {"factor": "0.72", "unit": "per_kg", "biogenic": "-1.50", "source": "ICE v3"},
    # Masonry (kgCO2e per kg)
    "brick_clay": {"factor": "0.24", "unit": "per_kg", "density_kg_m3": "1900", "source": "ICE v3"},
    "brick_concrete": {"factor": "0.09", "unit": "per_kg", "density_kg_m3": "2100", "source": "ICE v3"},
    "block_concrete_dense": {"factor": "0.10", "unit": "per_kg", "density_kg_m3": "2000", "source": "ICE v3"},
    "block_concrete_lightweight": {"factor": "0.28", "unit": "per_kg", "density_kg_m3": "1400", "source": "ICE v3"},
    "block_AAC": {"factor": "0.34", "unit": "per_kg", "density_kg_m3": "600", "source": "ICE v3"},
    # Glass (kgCO2e per m2 for typical thickness)
    "glass_single_4mm": {"factor": "10.0", "unit": "per_m2", "source": "ICE v3"},
    "glass_double_4_16_4": {"factor": "20.0", "unit": "per_m2", "source": "ICE v3"},
    "glass_triple_4_12_4_12_4": {"factor": "30.0", "unit": "per_m2", "source": "ICE v3"},
    "glass_low_e_double": {"factor": "22.0", "unit": "per_m2", "source": "ICE v3"},
    # Insulation (kgCO2e per kg)
    "insulation_mineral_wool": {"factor": "1.28", "unit": "per_kg", "density_kg_m3": "30", "source": "ICE v3"},
    "insulation_glass_wool": {"factor": "1.35", "unit": "per_kg", "density_kg_m3": "25", "source": "ICE v3"},
    "insulation_EPS": {"factor": "3.29", "unit": "per_kg", "density_kg_m3": "20", "source": "ICE v3"},
    "insulation_XPS": {"factor": "3.48", "unit": "per_kg", "density_kg_m3": "35", "source": "ICE v3"},
    "insulation_PIR": {"factor": "3.44", "unit": "per_kg", "density_kg_m3": "32", "source": "ICE v3"},
    "insulation_cellulose": {"factor": "0.18", "unit": "per_kg", "density_kg_m3": "50", "source": "ICE v3"},
    "insulation_wood_fibre": {"factor": "0.98", "unit": "per_kg", "density_kg_m3": "160", "biogenic": "-1.63", "source": "ICE v3"},
    "insulation_cork": {"factor": "0.19", "unit": "per_kg", "density_kg_m3": "120", "biogenic": "-1.50", "source": "ICE v3"},
    # Other materials (kgCO2e per kg)
    "plasterboard": {"factor": "0.39", "unit": "per_kg", "density_kg_m3": "830", "source": "ICE v3"},
    "plaster_gypsum": {"factor": "0.12", "unit": "per_kg", "source": "ICE v3"},
    "copper_pipe": {"factor": "2.71", "unit": "per_kg", "source": "ICE v3"},
    "copper_sheet": {"factor": "3.03", "unit": "per_kg", "source": "ICE v3"},
    "PVC_pipe": {"factor": "3.10", "unit": "per_kg", "source": "ICE v3"},
    "PVC_window_frame": {"factor": "3.19", "unit": "per_kg", "source": "ICE v3"},
    "bitumen_roofing": {"factor": "0.50", "unit": "per_kg", "source": "ICE v3"},
    "bitumen_waterproofing": {"factor": "0.48", "unit": "per_kg", "source": "ICE v3"},
    "stone_natural": {"factor": "0.06", "unit": "per_kg", "source": "ICE v3"},
    "stone_granite": {"factor": "0.70", "unit": "per_kg", "source": "ICE v3"},
    "aggregate_general": {"factor": "0.005", "unit": "per_kg", "source": "ICE v3"},
    "aggregate_recycled": {"factor": "0.004", "unit": "per_kg", "source": "ICE v3"},
    "mortar_general": {"factor": "0.19", "unit": "per_kg", "source": "ICE v3"},
    "ceramic_tiles": {"factor": "0.78", "unit": "per_kg", "source": "ICE v3"},
    "carpet_synthetic": {"factor": "5.43", "unit": "per_kg", "source": "ICE v3"},
    "paint_water_based": {"factor": "2.42", "unit": "per_kg", "source": "ICE v3"},
}

# ---------------------------------------------------------------------------
# Material Lifetime -- expected replacement cycle in years
# ---------------------------------------------------------------------------

MATERIAL_LIFETIME: Dict[str, int] = {
    "concrete": 100,
    "steel_structural": 100,
    "steel_rebar": 100,
    "aluminium": 60,
    "timber_softwood": 60,
    "timber_hardwood": 80,
    "timber_CLT": 60,
    "timber_glulam": 60,
    "brick": 100,
    "block_concrete": 100,
    "glass_double": 30,
    "glass_triple": 30,
    "insulation_mineral_wool": 60,
    "insulation_EPS": 50,
    "insulation_XPS": 50,
    "insulation_PIR": 50,
    "plasterboard": 30,
    "copper": 50,
    "PVC": 35,
    "bitumen_roofing": 25,
    "stone": 100,
    "ceramic_tiles": 40,
    "carpet": 10,
    "paint": 8,
    "mechanical_systems": 20,
    "electrical_systems": 25,
    "lifts": 25,
    "solar_PV": 25,
    "window_frame_timber": 30,
    "window_frame_aluminium": 40,
    "window_frame_PVC": 30,
    "flat_roof_membrane": 25,
    "pitched_roof_tiles": 60,
}

# ---------------------------------------------------------------------------
# Transport Emission Factors (kgCO2e per tonne-km)
# ---------------------------------------------------------------------------
# Source: DEFRA 2024 conversion factors, ecoinvent 3.9

TRANSPORT_EMISSION_FACTORS: Dict[str, str] = {
    "truck_rigid": "0.170",
    "truck_articulated": "0.089",
    "rail": "0.028",
    "ship_container": "0.016",
    "ship_bulk": "0.005",
    "van": "0.280",
}

# ---------------------------------------------------------------------------
# Carbon Budgets by Building Type (kgCO2e/m2 GIA)
# ---------------------------------------------------------------------------
# Upfront embodied (A1-A5), whole life (A-C), and with Module D

CARBON_BUDGETS: Dict[str, Dict[str, Dict[str, str]]] = {
    "RIBA_2030": {
        "residential": {"upfront_A1A5": "300", "whole_life_AC": "625", "with_D": "500"},
        "office": {"upfront_A1A5": "350", "whole_life_AC": "750", "with_D": "600"},
        "school": {"upfront_A1A5": "400", "whole_life_AC": "800", "with_D": "650"},
        "hospital": {"upfront_A1A5": "500", "whole_life_AC": "1000", "with_D": "850"},
        "retail": {"upfront_A1A5": "350", "whole_life_AC": "700", "with_D": "575"},
        "hotel": {"upfront_A1A5": "400", "whole_life_AC": "800", "with_D": "650"},
        "industrial": {"upfront_A1A5": "250", "whole_life_AC": "550", "with_D": "450"},
        "mixed_use": {"upfront_A1A5": "350", "whole_life_AC": "725", "with_D": "600"},
    },
    "LETI_2020": {
        "residential": {"upfront_A1A5": "300", "whole_life_AC": "500", "with_D": "400"},
        "office": {"upfront_A1A5": "350", "whole_life_AC": "600", "with_D": "500"},
        "school": {"upfront_A1A5": "350", "whole_life_AC": "600", "with_D": "500"},
        "hospital": {"upfront_A1A5": "450", "whole_life_AC": "800", "with_D": "700"},
        "retail": {"upfront_A1A5": "300", "whole_life_AC": "550", "with_D": "450"},
        "hotel": {"upfront_A1A5": "350", "whole_life_AC": "650", "with_D": "550"},
        "industrial": {"upfront_A1A5": "200", "whole_life_AC": "400", "with_D": "350"},
        "mixed_use": {"upfront_A1A5": "325", "whole_life_AC": "575", "with_D": "475"},
    },
    "GLA_2022": {
        "residential": {"upfront_A1A5": "300", "whole_life_AC": "475", "with_D": "400"},
        "office": {"upfront_A1A5": "350", "whole_life_AC": "550", "with_D": "475"},
        "school": {"upfront_A1A5": "350", "whole_life_AC": "550", "with_D": "475"},
        "hospital": {"upfront_A1A5": "450", "whole_life_AC": "750", "with_D": "650"},
        "retail": {"upfront_A1A5": "300", "whole_life_AC": "500", "with_D": "425"},
        "hotel": {"upfront_A1A5": "350", "whole_life_AC": "600", "with_D": "525"},
        "industrial": {"upfront_A1A5": "200", "whole_life_AC": "375", "with_D": "325"},
        "mixed_use": {"upfront_A1A5": "325", "whole_life_AC": "530", "with_D": "460"},
    },
    "RICS_2017": {
        "residential": {"upfront_A1A5": "400", "whole_life_AC": "800", "with_D": "700"},
        "office": {"upfront_A1A5": "500", "whole_life_AC": "1000", "with_D": "900"},
        "school": {"upfront_A1A5": "450", "whole_life_AC": "900", "with_D": "800"},
        "hospital": {"upfront_A1A5": "600", "whole_life_AC": "1200", "with_D": "1100"},
        "retail": {"upfront_A1A5": "400", "whole_life_AC": "850", "with_D": "750"},
        "hotel": {"upfront_A1A5": "500", "whole_life_AC": "1000", "with_D": "900"},
        "industrial": {"upfront_A1A5": "300", "whole_life_AC": "650", "with_D": "575"},
        "mixed_use": {"upfront_A1A5": "450", "whole_life_AC": "925", "with_D": "825"},
    },
    "DGNB": {
        "residential": {"upfront_A1A5": "320", "whole_life_AC": "600", "with_D": "480"},
        "office": {"upfront_A1A5": "380", "whole_life_AC": "700", "with_D": "560"},
        "school": {"upfront_A1A5": "370", "whole_life_AC": "680", "with_D": "540"},
        "hospital": {"upfront_A1A5": "480", "whole_life_AC": "900", "with_D": "740"},
        "retail": {"upfront_A1A5": "340", "whole_life_AC": "620", "with_D": "500"},
        "hotel": {"upfront_A1A5": "390", "whole_life_AC": "720", "with_D": "580"},
        "industrial": {"upfront_A1A5": "240", "whole_life_AC": "460", "with_D": "380"},
        "mixed_use": {"upfront_A1A5": "360", "whole_life_AC": "660", "with_D": "530"},
    },
}

# ---------------------------------------------------------------------------
# Grid Decarbonisation Projections (kgCO2/kWh) 2025-2070
# ---------------------------------------------------------------------------
# Source: National grid forecasts, IEA WEO 2024

GRID_DECARBONISATION: Dict[str, Dict[str, str]] = {
    "UK": {
        "2025": "0.212", "2030": "0.130", "2035": "0.060", "2040": "0.030",
        "2045": "0.015", "2050": "0.010", "2055": "0.008", "2060": "0.005",
        "2065": "0.003", "2070": "0.002",
    },
    "IE": {
        "2025": "0.296", "2030": "0.180", "2035": "0.100", "2040": "0.060",
        "2045": "0.035", "2050": "0.020", "2055": "0.015", "2060": "0.010",
        "2065": "0.008", "2070": "0.005",
    },
    "DE": {
        "2025": "0.366", "2030": "0.240", "2035": "0.150", "2040": "0.080",
        "2045": "0.045", "2050": "0.025", "2055": "0.018", "2060": "0.012",
        "2065": "0.008", "2070": "0.005",
    },
    "FR": {
        "2025": "0.052", "2030": "0.040", "2035": "0.030", "2040": "0.022",
        "2045": "0.016", "2050": "0.012", "2055": "0.010", "2060": "0.008",
        "2065": "0.006", "2070": "0.004",
    },
    "NL": {
        "2025": "0.328", "2030": "0.200", "2035": "0.110", "2040": "0.060",
        "2045": "0.035", "2050": "0.020", "2055": "0.014", "2060": "0.010",
        "2065": "0.007", "2070": "0.005",
    },
    "DK": {
        "2025": "0.112", "2030": "0.060", "2035": "0.025", "2040": "0.012",
        "2045": "0.008", "2050": "0.005", "2055": "0.004", "2060": "0.003",
        "2065": "0.002", "2070": "0.001",
    },
    "SE": {
        "2025": "0.013", "2030": "0.010", "2035": "0.008", "2040": "0.006",
        "2045": "0.005", "2050": "0.004", "2055": "0.003", "2060": "0.002",
        "2065": "0.002", "2070": "0.001",
    },
    "ES": {
        "2025": "0.151", "2030": "0.100", "2035": "0.060", "2040": "0.035",
        "2045": "0.020", "2050": "0.012", "2055": "0.009", "2060": "0.007",
        "2065": "0.005", "2070": "0.003",
    },
    "IT": {
        "2025": "0.257", "2030": "0.170", "2035": "0.100", "2040": "0.055",
        "2045": "0.030", "2050": "0.018", "2055": "0.013", "2060": "0.010",
        "2065": "0.007", "2070": "0.005",
    },
    "PL": {
        "2025": "0.623", "2030": "0.450", "2035": "0.300", "2040": "0.180",
        "2045": "0.100", "2050": "0.060", "2055": "0.040", "2060": "0.025",
        "2065": "0.016", "2070": "0.010",
    },
    "EU_AVG": {
        "2025": "0.230", "2030": "0.155", "2035": "0.095", "2040": "0.055",
        "2045": "0.032", "2050": "0.020", "2055": "0.014", "2060": "0.010",
        "2065": "0.007", "2070": "0.005",
    },
}

# ---------------------------------------------------------------------------
# Biogenic Carbon Factors (kgCO2e sequestered per kg of product)
# ---------------------------------------------------------------------------
# Negative values = carbon stored in timber/bio-based products

BIOGENIC_CARBON_FACTORS: Dict[str, str] = {
    "timber_softwood": "-1.63",
    "timber_hardwood": "-1.63",
    "timber_CLT": "-1.63",
    "timber_glulam": "-1.63",
    "timber_plywood": "-1.63",
    "timber_OSB": "-1.63",
    "timber_MDF": "-1.50",
    "insulation_cellulose": "-1.50",
    "insulation_wood_fibre": "-1.63",
    "insulation_cork": "-1.50",
    "straw_bale": "-1.40",
    "hemp_lime": "-0.30",
}

# ---------------------------------------------------------------------------
# Construction Stage Factors (A5 as % of A1-A3)
# ---------------------------------------------------------------------------

CONSTRUCTION_WASTE_FACTORS: Dict[str, str] = {
    "concrete": "0.05",
    "steel": "0.03",
    "aluminium": "0.05",
    "timber": "0.10",
    "masonry": "0.05",
    "glass": "0.03",
    "insulation": "0.05",
    "finishes": "0.08",
    "MEP": "0.03",
    "default": "0.05",
}

# ---------------------------------------------------------------------------
# End of Life Factors (C1-C4 as proportion of A1-A3)
# ---------------------------------------------------------------------------

EOL_FACTORS: Dict[str, Dict[str, str]] = {
    "concrete": {"C1_pct": "0.005", "C2_pct": "0.010", "C3_pct": "0.005", "C4_pct": "0.005"},
    "steel": {"C1_pct": "0.003", "C2_pct": "0.008", "C3_pct": "0.010", "C4_pct": "0.002"},
    "aluminium": {"C1_pct": "0.003", "C2_pct": "0.008", "C3_pct": "0.010", "C4_pct": "0.002"},
    "timber": {"C1_pct": "0.005", "C2_pct": "0.010", "C3_pct": "0.005", "C4_pct": "0.010"},
    "masonry": {"C1_pct": "0.005", "C2_pct": "0.012", "C3_pct": "0.005", "C4_pct": "0.008"},
    "glass": {"C1_pct": "0.005", "C2_pct": "0.010", "C3_pct": "0.008", "C4_pct": "0.005"},
    "insulation": {"C1_pct": "0.003", "C2_pct": "0.008", "C3_pct": "0.005", "C4_pct": "0.010"},
    "finishes": {"C1_pct": "0.005", "C2_pct": "0.010", "C3_pct": "0.005", "C4_pct": "0.010"},
    "default": {"C1_pct": "0.005", "C2_pct": "0.010", "C3_pct": "0.005", "C4_pct": "0.005"},
}

# ---------------------------------------------------------------------------
# Module D Credits (recycling/reuse credit as proportion of A1-A3)
# ---------------------------------------------------------------------------

MODULE_D_CREDITS: Dict[str, str] = {
    "steel_structural": "-0.35",
    "steel_rebar": "-0.35",
    "aluminium": "-0.40",
    "concrete": "-0.02",
    "timber": "-0.10",
    "copper": "-0.30",
    "glass": "-0.05",
    "brick": "-0.01",
    "insulation_mineral_wool": "-0.05",
    "PVC": "-0.10",
    "default": "-0.02",
}

# Operational water factor (kgCO2e per m3 of water)
WATER_CARBON_FACTOR: str = "0.344"

# Typical operational water use by building type (m3/m2/yr)
WATER_USE_INTENSITY: Dict[str, str] = {
    "residential": "1.2",
    "office": "0.6",
    "school": "0.5",
    "hospital": "2.0",
    "retail": "0.4",
    "hotel": "1.5",
    "industrial": "0.8",
    "mixed_use": "0.7",
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class MaterialInput(BaseModel):
    """A single material entry for embodied carbon calculation."""
    material_id: str = Field(..., description="Key in EMBODIED_CARBON_FACTORS or custom")
    material_category: str = Field(..., description="Broad material category")
    description: Optional[str] = None
    quantity: float = Field(..., gt=0, description="Quantity in the specified unit")
    unit: str = Field(default="kg", description="kg, m2, or m3")
    transport_distance_km: float = Field(default=50.0, ge=0, description="Distance to site km")
    transport_mode: str = Field(default="truck_articulated", description="Transport mode")
    custom_ecf: Optional[float] = Field(None, ge=0, description="Override embodied carbon factor")
    expected_lifetime_years: Optional[int] = Field(None, ge=1, le=200)
    include_biogenic: bool = Field(default=False, description="Include biogenic carbon")
    epd_reference: Optional[str] = None

class WholeLifeCarbonInput(BaseModel):
    """Full input for the WholeLifeCarbonEngine."""
    building_id: str = Field(..., min_length=1)
    building_type: str = Field(default="office")
    country_code: str = Field(default="IE")
    gross_internal_area_m2: float = Field(..., gt=0, description="GIA in m2")
    study_period_years: int = Field(default=60, ge=15, le=120, description="Study period")
    start_year: int = Field(default=2025, ge=2020, le=2050)
    annual_energy_kwh_m2: float = Field(default=0.0, ge=0, description="Annual energy kWh/m2 for B6")
    annual_water_m3_m2: Optional[float] = Field(None, ge=0, description="Annual water m3/m2 for B7")
    materials: List[MaterialInput] = Field(..., min_length=1)
    construction_waste_pct: Optional[float] = Field(None, ge=0, le=0.3, description="A5 waste factor override")
    target_standards: List[str] = Field(default=["RIBA_2030", "LETI_2020"], description="Target standards for comparison")
    include_biogenic: bool = Field(default=False, description="Include biogenic carbon globally")
    include_module_d: bool = Field(default=True, description="Include Module D credits")

    @field_validator("building_type")
    @classmethod
    def validate_building_type(cls, v: str) -> str:
        valid = [bt.value for bt in BuildingTypeWLC]
        if v not in valid:
            raise ValueError(f"building_type must be one of {valid}")
        return v

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class MaterialEmbodiedResult(BaseModel):
    """Embodied carbon result for a single material."""
    material_id: str
    material_category: str
    description: str
    quantity: float
    unit: str
    ecf_value: float
    ecf_source: str
    embodied_carbon_A1A3_kgCO2e: float
    biogenic_carbon_kgCO2e: float
    transport_A4_kgCO2e: float
    replacement_B4_kgCO2e: float
    n_replacements: int
    eol_C1C4_kgCO2e: float
    module_d_kgCO2e: float

class LifecycleStageResult(BaseModel):
    """Carbon totals by EN 15978 lifecycle stage."""
    stage: str
    stage_name: str
    total_kgCO2e: float
    per_m2_kgCO2e: float
    pct_of_whole_life: float

class TargetComparison(BaseModel):
    """Comparison against a carbon budget target."""
    standard: str
    building_type: str
    target_upfront_A1A5_kgCO2e_m2: float
    target_whole_life_AC_kgCO2e_m2: float
    target_with_D_kgCO2e_m2: float
    actual_upfront_A1A5_kgCO2e_m2: float
    actual_whole_life_AC_kgCO2e_m2: float
    actual_with_D_kgCO2e_m2: float
    upfront_compliant: bool
    whole_life_compliant: bool
    with_D_compliant: bool
    upfront_margin_pct: float
    whole_life_margin_pct: float

class TopMaterialContributor(BaseModel):
    """Top contributing material to embodied carbon."""
    material_id: str
    material_category: str
    embodied_A1A3_kgCO2e: float
    pct_of_total: float

class MaterialSubstitution(BaseModel):
    """Analysis of a material substitution opportunity."""
    original_material_id: str
    alternative_material_id: str
    original_A1A3_kgCO2e: float
    alternative_A1A3_kgCO2e: float
    savings_kgCO2e: float
    savings_pct: float
    cost_implication: str
    feasibility: str
    notes: str

class SensitivityScenario(BaseModel):
    """Result of a sensitivity analysis scenario."""
    scenario_name: str
    parameter_varied: str
    variation_pct: float
    wlc_ac_kgCO2e: float
    wlc_ac_per_m2: float
    delta_from_base_pct: float

class DesignRecommendation(BaseModel):
    """Design recommendation for carbon reduction."""
    priority: int
    category: str
    recommendation: str
    estimated_savings_pct: float
    lifecycle_stage_affected: str

class WholeLifeCarbonResult(BaseModel):
    """Complete output of the WholeLifeCarbonEngine."""
    assessment_id: str
    building_id: str
    building_type: str
    country_code: str
    gross_internal_area_m2: float
    study_period_years: int
    start_year: int

    # Material results
    material_results: List[MaterialEmbodiedResult]

    # Lifecycle stage breakdown
    stage_results: List[LifecycleStageResult]

    # Totals
    total_A1A3_kgCO2e: float
    total_A4_kgCO2e: float
    total_A5_kgCO2e: float
    total_upfront_A1A5_kgCO2e: float
    total_B1B5_kgCO2e: float
    total_B6_kgCO2e: float
    total_B7_kgCO2e: float
    total_C1C4_kgCO2e: float
    total_D_kgCO2e: float
    total_biogenic_kgCO2e: float

    # Whole life
    whole_life_AC_kgCO2e: float
    whole_life_ACD_kgCO2e: float
    whole_life_AC_per_m2: float
    whole_life_ACD_per_m2: float
    whole_life_AC_per_m2_per_yr: float

    # Target comparisons
    target_comparisons: List[TargetComparison]

    # Top contributors
    top_contributors: List[TopMaterialContributor]

    # Substitutions and sensitivity
    substitution_opportunities: List[MaterialSubstitution]
    sensitivity_scenarios: List[SensitivityScenario]
    design_recommendations: List[DesignRecommendation]

    # Metadata
    engine_version: str = _MODULE_VERSION
    calculated_at: str
    processing_time_ms: float
    provenance_hash: str

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class WholeLifeCarbonEngine:
    """
    Whole life carbon assessment engine per EN 15978.

    Calculates embodied, operational, and end-of-life carbon across all
    EN 15978 lifecycle stages (A1-D), compares against RIBA 2030, LETI,
    GLA, RICS, and DGNB carbon budgets.

    Zero-Hallucination Guarantee:
        - ECFs from ICE Database v3.0 (deterministic lookup)
        - Grid decarbonisation from published national projections
        - All calculations deterministic Decimal arithmetic
        - No LLM in any calculation path
        - SHA-256 provenance hash on every result
    """

    # ------------------------------------------------------------------ #
    # calculate_embodied_carbon
    # ------------------------------------------------------------------ #

    def calculate_embodied_carbon(
        self,
        material: MaterialInput,
        study_period: int,
        include_biogenic: bool,
    ) -> MaterialEmbodiedResult:
        """Calculate embodied carbon (A1-A3) for a single material.

        EC = quantity * ECF (with unit conversion as needed)

        Args:
            material: Material specification.
            study_period: Study period years for replacement calc.
            include_biogenic: Whether to include biogenic carbon.

        Returns:
            Embodied carbon result for this material.
        """
        ecf_data = EMBODIED_CARBON_FACTORS.get(material.material_id)
        if ecf_data is None and material.custom_ecf is None:
            raise ValueError(f"Unknown material_id '{material.material_id}' and no custom_ecf provided")

        # Get ECF
        if material.custom_ecf is not None:
            ecf = _decimal(material.custom_ecf)
            ecf_source = "custom_EPD"
        else:
            ecf = _decimal(ecf_data["factor"])
            ecf_source = ecf_data.get("source", "ICE v3")

        quantity = _decimal(material.quantity)

        # Unit conversion: if ECF is per_m3 and input is m3, use directly
        # If ECF is per_kg and input is kg, use directly
        # If ECF is per_m3 but input is kg, convert via density
        if ecf_data and material.custom_ecf is None:
            ecf_unit = ecf_data.get("unit", "per_kg")
            density = _decimal(ecf_data.get("density_kg_m3", "1"))

            if ecf_unit == "per_m3" and material.unit == "m3":
                ec_a1a3 = quantity * ecf
            elif ecf_unit == "per_m3" and material.unit == "kg":
                vol = _safe_divide(quantity, density)
                ec_a1a3 = vol * ecf
            elif ecf_unit == "per_m2" and material.unit == "m2":
                ec_a1a3 = quantity * ecf
            elif ecf_unit == "per_kg" and material.unit == "kg":
                ec_a1a3 = quantity * ecf
            elif ecf_unit == "per_kg" and material.unit == "m3":
                mass = quantity * density
                ec_a1a3 = mass * ecf
            else:
                ec_a1a3 = quantity * ecf
        else:
            ec_a1a3 = quantity * ecf

        # Biogenic carbon
        biogenic = Decimal("0")
        if include_biogenic or material.include_biogenic:
            bio_factor_str = None
            if ecf_data:
                bio_factor_str = ecf_data.get("biogenic")
            if bio_factor_str is None:
                bio_factor_str = BIOGENIC_CARBON_FACTORS.get(material.material_category, "0")
            bio_factor = _decimal(bio_factor_str)
            # Convert quantity to kg for biogenic
            if material.unit == "kg":
                mass_kg = quantity
            elif material.unit == "m3" and ecf_data:
                mass_kg = quantity * _decimal(ecf_data.get("density_kg_m3", "1"))
            else:
                mass_kg = quantity
            biogenic = mass_kg * bio_factor

        # Transport (A4)
        transport = self.calculate_transport_carbon(material)

        # Replacement (B4)
        lifetime = material.expected_lifetime_years or MATERIAL_LIFETIME.get(material.material_category, 60)
        n_replacements = max(0, _ceil_div(study_period, lifetime) - 1)
        replacement_carbon = ec_a1a3 * _decimal(n_replacements)

        # End of life (C1-C4)
        eol = self._calculate_eol(material.material_category, ec_a1a3)

        # Module D
        module_d = self._calculate_module_d(material.material_category, ec_a1a3)

        return MaterialEmbodiedResult(
            material_id=material.material_id,
            material_category=material.material_category,
            description=material.description or material.material_id,
            quantity=_round2(float(quantity)),
            unit=material.unit,
            ecf_value=_round4(float(ecf)),
            ecf_source=ecf_source,
            embodied_carbon_A1A3_kgCO2e=_round2(float(ec_a1a3)),
            biogenic_carbon_kgCO2e=_round2(float(biogenic)),
            transport_A4_kgCO2e=_round2(float(transport)),
            replacement_B4_kgCO2e=_round2(float(replacement_carbon)),
            n_replacements=n_replacements,
            eol_C1C4_kgCO2e=_round2(float(eol)),
            module_d_kgCO2e=_round2(float(module_d)),
        )

    # ------------------------------------------------------------------ #
    # calculate_transport_carbon
    # ------------------------------------------------------------------ #

    def calculate_transport_carbon(self, material: MaterialInput) -> Decimal:
        """Calculate transport emissions (A4).

        EC_A4 = mass_tonnes * distance_km * TEF_mode

        Args:
            material: Material with transport details.

        Returns:
            Transport carbon kgCO2e.
        """
        quantity = _decimal(material.quantity)
        distance = _decimal(material.transport_distance_km)
        tef = _decimal(TRANSPORT_EMISSION_FACTORS.get(material.transport_mode, "0.089"))

        # Convert to tonnes
        ecf_data = EMBODIED_CARBON_FACTORS.get(material.material_id)
        if material.unit == "kg":
            mass_tonnes = quantity / Decimal("1000")
        elif material.unit == "m3":
            density = _decimal(ecf_data.get("density_kg_m3", "2400") if ecf_data else "2400")
            mass_tonnes = quantity * density / Decimal("1000")
        elif material.unit == "m2":
            # Approximate mass per m2 -- use typical thickness assumptions
            mass_tonnes = quantity * Decimal("10") / Decimal("1000")  # ~10 kg/m2 default
        else:
            mass_tonnes = quantity / Decimal("1000")

        return mass_tonnes * distance * tef

    # ------------------------------------------------------------------ #
    # calculate_construction_carbon
    # ------------------------------------------------------------------ #

    def calculate_construction_carbon(
        self,
        material_results: List[MaterialEmbodiedResult],
        override_pct: Optional[float] = None,
    ) -> Decimal:
        """Calculate construction stage emissions (A5).

        A5 = sum( A1A3_i * waste_factor_i ) for each material category.
        Includes site energy and material wastage.

        Args:
            material_results: Embodied carbon results per material.
            override_pct: Override waste percentage for all.

        Returns:
            Total A5 carbon kgCO2e.
        """
        total = Decimal("0")
        for mr in material_results:
            ec = _decimal(mr.embodied_carbon_A1A3_kgCO2e)
            if override_pct is not None:
                factor = _decimal(override_pct)
            else:
                factor = _decimal(CONSTRUCTION_WASTE_FACTORS.get(
                    mr.material_category,
                    CONSTRUCTION_WASTE_FACTORS["default"],
                ))
            total += ec * factor
        return total

    # ------------------------------------------------------------------ #
    # calculate_operational_carbon
    # ------------------------------------------------------------------ #

    def calculate_operational_carbon(
        self,
        annual_energy_kwh_m2: Decimal,
        floor_area_m2: Decimal,
        country_code: str,
        start_year: int,
        study_period: int,
    ) -> Decimal:
        """Calculate operational energy carbon (B6) over study period.

        Uses grid decarbonisation projections for future years.
        OC = sum( E_annual * EF_grid_year, year=start..start+period )

        Args:
            annual_energy_kwh_m2: Annual energy intensity kWh/m2.
            floor_area_m2: GIA m2.
            country_code: ISO country code.
            start_year: Construction start year.
            study_period: Study period years.

        Returns:
            Total B6 carbon kgCO2e.
        """
        projections = GRID_DECARBONISATION.get(
            country_code, GRID_DECARBONISATION["EU_AVG"]
        )
        annual_energy = annual_energy_kwh_m2 * floor_area_m2
        total = Decimal("0")

        for yr_offset in range(1, study_period + 1):
            year = start_year + yr_offset
            year_str = str(year)

            # Find closest projection year
            ef = self._interpolate_grid_ef(projections, year)
            total += annual_energy * ef

        return total

    def _interpolate_grid_ef(
        self,
        projections: Dict[str, str],
        year: int,
    ) -> Decimal:
        """Interpolate grid emission factor for a given year.

        Linear interpolation between available data points.

        Args:
            projections: Year -> kgCO2/kWh mapping.
            year: Target year.

        Returns:
            Interpolated grid emission factor.
        """
        years_available = sorted([int(y) for y in projections.keys()])
        if not years_available:
            return _decimal("0.230")

        if year <= years_available[0]:
            return _decimal(projections[str(years_available[0])])
        if year >= years_available[-1]:
            return _decimal(projections[str(years_available[-1])])

        # Find bracketing years
        for i in range(len(years_available) - 1):
            y1 = years_available[i]
            y2 = years_available[i + 1]
            if y1 <= year <= y2:
                ef1 = _decimal(projections[str(y1)])
                ef2 = _decimal(projections[str(y2)])
                frac = _safe_divide(_decimal(year - y1), _decimal(y2 - y1))
                return ef1 + frac * (ef2 - ef1)

        return _decimal(projections[str(years_available[-1])])

    # ------------------------------------------------------------------ #
    # calculate_replacement_carbon (B4 -- aggregated)
    # ------------------------------------------------------------------ #

    def calculate_replacement_carbon(
        self,
        material_results: List[MaterialEmbodiedResult],
    ) -> Decimal:
        """Aggregate replacement carbon (B4) from material results."""
        return sum(_decimal(mr.replacement_B4_kgCO2e) for mr in material_results)

    # ------------------------------------------------------------------ #
    # calculate_end_of_life (C1-C4)
    # ------------------------------------------------------------------ #

    def _calculate_eol(
        self,
        category: str,
        ec_a1a3: Decimal,
    ) -> Decimal:
        """Calculate end-of-life carbon (C1-C4) for a material.

        Typically 1-3% of A1-A3 depending on material.

        Args:
            category: Material category.
            ec_a1a3: Product stage carbon kgCO2e.

        Returns:
            End of life carbon kgCO2e.
        """
        factors = EOL_FACTORS.get(category, EOL_FACTORS["default"])
        c1 = ec_a1a3 * _decimal(factors["C1_pct"])
        c2 = ec_a1a3 * _decimal(factors["C2_pct"])
        c3 = ec_a1a3 * _decimal(factors["C3_pct"])
        c4 = ec_a1a3 * _decimal(factors["C4_pct"])
        return c1 + c2 + c3 + c4

    def calculate_end_of_life(
        self,
        material_results: List[MaterialEmbodiedResult],
    ) -> Decimal:
        """Aggregate end-of-life carbon (C1-C4) from material results."""
        return sum(_decimal(mr.eol_C1C4_kgCO2e) for mr in material_results)

    # ------------------------------------------------------------------ #
    # calculate_module_d
    # ------------------------------------------------------------------ #

    def _calculate_module_d(
        self,
        category: str,
        ec_a1a3: Decimal,
    ) -> Decimal:
        """Calculate Module D credits for a material.

        Credits from recycling, reuse, or energy recovery.

        Args:
            category: Material category.
            ec_a1a3: Product stage carbon kgCO2e.

        Returns:
            Module D carbon credit kgCO2e (negative).
        """
        credit_str = MODULE_D_CREDITS.get(category, MODULE_D_CREDITS["default"])
        return ec_a1a3 * _decimal(credit_str)

    def calculate_module_d(
        self,
        material_results: List[MaterialEmbodiedResult],
    ) -> Decimal:
        """Aggregate Module D credits from material results."""
        return sum(_decimal(mr.module_d_kgCO2e) for mr in material_results)

    # ------------------------------------------------------------------ #
    # calculate_whole_life
    # ------------------------------------------------------------------ #

    def calculate_whole_life(
        self,
        a1a3: Decimal,
        a4: Decimal,
        a5: Decimal,
        b1b5: Decimal,
        b6: Decimal,
        b7: Decimal,
        c1c4: Decimal,
        d: Decimal,
    ) -> Tuple[Decimal, Decimal]:
        """Calculate whole life carbon totals.

        WLC_AC = A1-A5 + B1-B7 + C1-C4
        WLC_ACD = WLC_AC + D

        Returns:
            Tuple of (WLC_AC, WLC_ACD).
        """
        wlc_ac = a1a3 + a4 + a5 + b1b5 + b6 + b7 + c1c4
        wlc_acd = wlc_ac + d
        return wlc_ac, wlc_acd

    # ------------------------------------------------------------------ #
    # compare_targets
    # ------------------------------------------------------------------ #

    def compare_targets(
        self,
        upfront_a1a5_per_m2: Decimal,
        wlc_ac_per_m2: Decimal,
        wlc_acd_per_m2: Decimal,
        building_type: str,
        target_standards: List[str],
    ) -> List[TargetComparison]:
        """Compare actual carbon against target budgets.

        Args:
            upfront_a1a5_per_m2: Actual upfront carbon kgCO2e/m2.
            wlc_ac_per_m2: Actual whole life A-C kgCO2e/m2.
            wlc_acd_per_m2: Actual whole life A-D kgCO2e/m2.
            building_type: Building typology.
            target_standards: Standards to compare against.

        Returns:
            List of target comparisons.
        """
        comparisons: List[TargetComparison] = []

        for standard in target_standards:
            budgets = CARBON_BUDGETS.get(standard, {})
            bt_budget = budgets.get(building_type, budgets.get("mixed_use", {}))
            if not bt_budget:
                continue

            t_up = _decimal(bt_budget.get("upfront_A1A5", "999"))
            t_wl = _decimal(bt_budget.get("whole_life_AC", "999"))
            t_wd = _decimal(bt_budget.get("with_D", "999"))

            up_margin = _safe_pct(t_up - upfront_a1a5_per_m2, t_up)
            wl_margin = _safe_pct(t_wl - wlc_ac_per_m2, t_wl)

            comparisons.append(TargetComparison(
                standard=standard,
                building_type=building_type,
                target_upfront_A1A5_kgCO2e_m2=_round1(float(t_up)),
                target_whole_life_AC_kgCO2e_m2=_round1(float(t_wl)),
                target_with_D_kgCO2e_m2=_round1(float(t_wd)),
                actual_upfront_A1A5_kgCO2e_m2=_round1(float(upfront_a1a5_per_m2)),
                actual_whole_life_AC_kgCO2e_m2=_round1(float(wlc_ac_per_m2)),
                actual_with_D_kgCO2e_m2=_round1(float(wlc_acd_per_m2)),
                upfront_compliant=upfront_a1a5_per_m2 <= t_up,
                whole_life_compliant=wlc_ac_per_m2 <= t_wl,
                with_D_compliant=wlc_acd_per_m2 <= t_wd,
                upfront_margin_pct=_round1(float(up_margin)),
                whole_life_margin_pct=_round1(float(wl_margin)),
            ))

        return comparisons

    # ------------------------------------------------------------------ #
    # Material substitution analysis
    # ------------------------------------------------------------------ #

    # Mapping of common materials to lower-carbon alternatives
    SUBSTITUTION_MAP: Dict[str, List[Dict[str, str]]] = {
        "concrete_C30_37": [
            {"alt": "concrete_C28_35_30pct_GGBS", "cost": "similar", "feasibility": "high", "notes": "30% GGBS replacement -- widely available, no structural compromise"},
            {"alt": "concrete_C28_35_50pct_GGBS", "cost": "similar", "feasibility": "medium", "notes": "50% GGBS -- check supply and early strength requirements"},
        ],
        "concrete_C32_40": [
            {"alt": "concrete_C28_35_30pct_GGBS", "cost": "similar", "feasibility": "high", "notes": "Reduce strength class where structurally possible, use GGBS"},
        ],
        "concrete_C40_50": [
            {"alt": "concrete_C28_35_50pct_GGBS", "cost": "similar", "feasibility": "medium", "notes": "Optimise mix design -- reduce strength where possible, maximise SCM"},
        ],
        "steel_structural": [
            {"alt": "steel_structural_recycled_60pct", "cost": "similar", "feasibility": "high", "notes": "Specify minimum 60% recycled content in procurement"},
            {"alt": "timber_glulam", "cost": "premium_10pct", "feasibility": "medium", "notes": "Glulam structure where fire and span allow"},
        ],
        "steel_rebar": [
            {"alt": "steel_rebar_recycled_97pct", "cost": "similar", "feasibility": "high", "notes": "EAF rebar with 97% recycled content -- widely available"},
        ],
        "aluminium_primary": [
            {"alt": "aluminium_recycled", "cost": "similar", "feasibility": "high", "notes": "Specify recycled aluminium content minimum 75%"},
        ],
        "aluminium_extruded": [
            {"alt": "aluminium_recycled", "cost": "similar", "feasibility": "medium", "notes": "Source post-consumer recycled extrusions where available"},
        ],
        "insulation_EPS": [
            {"alt": "insulation_mineral_wool", "cost": "similar", "feasibility": "high", "notes": "Mineral wool has lower embodied carbon; check thermal conductivity"},
            {"alt": "insulation_cellulose", "cost": "similar", "feasibility": "medium", "notes": "Cellulose (recycled paper) -- very low ECF, suitable for timber frames"},
        ],
        "insulation_XPS": [
            {"alt": "insulation_mineral_wool", "cost": "similar", "feasibility": "high", "notes": "Replace where moisture resistance is not critical"},
            {"alt": "insulation_cork", "cost": "premium_15pct", "feasibility": "medium", "notes": "Cork -- natural, carbon-negative when biogenic counted"},
        ],
        "insulation_PIR": [
            {"alt": "insulation_wood_fibre", "cost": "premium_10pct", "feasibility": "medium", "notes": "Wood fibre boards -- carbon negative with biogenic, good hygrothermal"},
        ],
        "glass_triple_4_12_4_12_4": [
            {"alt": "glass_low_e_double", "cost": "saving_15pct", "feasibility": "high", "notes": "Low-E double may suffice where triple not mandated"},
        ],
        "brick_clay": [
            {"alt": "timber_CLT", "cost": "premium_10pct", "feasibility": "medium", "notes": "CLT panels where structurally appropriate"},
        ],
        "PVC_window_frame": [
            {"alt": "timber_softwood_general", "cost": "premium_5pct", "feasibility": "high", "notes": "Timber window frames -- lower EC and biogenic carbon store"},
        ],
        "carpet_synthetic": [
            {"alt": "ceramic_tiles", "cost": "similar", "feasibility": "high", "notes": "Ceramic tiles last 4x longer, reducing B4 replacement carbon"},
        ],
    }

    def analyze_substitutions(
        self,
        material_results: List[MaterialEmbodiedResult],
    ) -> List[MaterialSubstitution]:
        """Identify material substitution opportunities.

        Checks each material against the substitution map and calculates
        potential embodied carbon savings.

        Args:
            material_results: Current material embodied carbon results.

        Returns:
            List of substitution opportunities sorted by savings.
        """
        subs: List[MaterialSubstitution] = []

        for mr in material_results:
            alternatives = self.SUBSTITUTION_MAP.get(mr.material_id, [])
            for alt in alternatives:
                alt_id = alt["alt"]
                alt_data = EMBODIED_CARBON_FACTORS.get(alt_id)
                if alt_data is None:
                    continue

                alt_ecf = _decimal(alt_data["factor"])
                orig_ecf = _decimal(mr.ecf_value)
                quantity = _decimal(mr.quantity)

                # Recalculate A1-A3 for alternative
                alt_unit = alt_data.get("unit", "per_kg")
                if alt_unit == mr.unit or alt_unit == "per_kg":
                    alt_ec = quantity * alt_ecf
                elif alt_unit == "per_m3" and mr.unit == "m3":
                    alt_ec = quantity * alt_ecf
                else:
                    alt_ec = quantity * alt_ecf

                orig_ec = _decimal(mr.embodied_carbon_A1A3_kgCO2e)
                savings = orig_ec - alt_ec
                savings_pct = _safe_pct(savings, orig_ec) if orig_ec > Decimal("0") else Decimal("0")

                if savings > Decimal("0"):
                    subs.append(MaterialSubstitution(
                        original_material_id=mr.material_id,
                        alternative_material_id=alt_id,
                        original_A1A3_kgCO2e=_round2(float(orig_ec)),
                        alternative_A1A3_kgCO2e=_round2(float(alt_ec)),
                        savings_kgCO2e=_round2(float(savings)),
                        savings_pct=_round1(float(savings_pct)),
                        cost_implication=alt.get("cost", "unknown"),
                        feasibility=alt.get("feasibility", "medium"),
                        notes=alt.get("notes", ""),
                    ))

        # Sort by savings descending
        subs.sort(key=lambda x: x.savings_kgCO2e, reverse=True)
        return subs

    # ------------------------------------------------------------------ #
    # Sensitivity analysis
    # ------------------------------------------------------------------ #

    def run_sensitivity_analysis(
        self,
        base_a1a3: Decimal,
        base_a4: Decimal,
        base_a5: Decimal,
        base_b1b5: Decimal,
        base_b6: Decimal,
        base_b7: Decimal,
        base_c1c4: Decimal,
        base_d: Decimal,
        gia: Decimal,
    ) -> List[SensitivityScenario]:
        """Run sensitivity analysis on key parameters.

        Varies each major lifecycle stage by +/-20% to determine which
        parameters most affect the whole life carbon total.

        Args:
            base_*: Baseline values for each lifecycle stage.
            gia: Gross internal area m2.

        Returns:
            List of sensitivity scenarios.
        """
        base_wlc_ac = base_a1a3 + base_a4 + base_a5 + base_b1b5 + base_b6 + base_b7 + base_c1c4
        base_per_m2 = _safe_divide(base_wlc_ac, gia)
        scenarios: List[SensitivityScenario] = []

        variations = [
            ("A1-A3 +20%", "A1-A3 Product Stage", Decimal("0.20"), base_a1a3),
            ("A1-A3 -20%", "A1-A3 Product Stage", Decimal("-0.20"), base_a1a3),
            ("B6 +20%", "B6 Operational Energy", Decimal("0.20"), base_b6),
            ("B6 -20%", "B6 Operational Energy", Decimal("-0.20"), base_b6),
            ("B4 +50%", "B4 Replacement (lifetime shorter)", Decimal("0.50"), base_b1b5),
            ("B4 -30%", "B4 Replacement (lifetime longer)", Decimal("-0.30"), base_b1b5),
            ("A4 +50%", "A4 Transport (longer distance)", Decimal("0.50"), base_a4),
            ("Grid decarb faster", "B6 Operational Energy", Decimal("-0.30"), base_b6),
            ("Grid decarb slower", "B6 Operational Energy", Decimal("0.30"), base_b6),
            ("C1-C4 +50%", "C1-C4 End of Life", Decimal("0.50"), base_c1c4),
        ]

        for name, param, var_pct, base_val in variations:
            delta = base_val * var_pct
            new_wlc = base_wlc_ac + delta
            new_per_m2 = _safe_divide(new_wlc, gia)
            delta_pct = _safe_pct(new_wlc - base_wlc_ac, base_wlc_ac)

            scenarios.append(SensitivityScenario(
                scenario_name=name,
                parameter_varied=param,
                variation_pct=_round1(float(var_pct * Decimal("100"))),
                wlc_ac_kgCO2e=_round2(float(new_wlc)),
                wlc_ac_per_m2=_round2(float(new_per_m2)),
                delta_from_base_pct=_round2(float(delta_pct)),
            ))

        return scenarios

    # ------------------------------------------------------------------ #
    # Design recommendations
    # ------------------------------------------------------------------ #

    def generate_recommendations(
        self,
        stage_results: List[LifecycleStageResult],
        material_results: List[MaterialEmbodiedResult],
        target_comparisons: List[TargetComparison],
        substitutions: List[MaterialSubstitution],
    ) -> List[DesignRecommendation]:
        """Generate prioritised design recommendations for carbon reduction.

        Analyses the assessment results and identifies the most impactful
        opportunities for reducing whole life carbon.

        Args:
            stage_results: Carbon by lifecycle stage.
            material_results: Per-material results.
            target_comparisons: Budget comparison results.
            substitutions: Substitution opportunities.

        Returns:
            Prioritised list of design recommendations.
        """
        recs: List[DesignRecommendation] = []
        priority = 1

        # Check if over budget
        over_budget = any(not tc.upfront_compliant for tc in target_comparisons)

        # Find dominant stage
        stage_map = {sr.stage: sr for sr in stage_results}
        a1a3 = stage_map.get("A1-A3")
        b6 = stage_map.get("B6")
        b1b5 = stage_map.get("B1-B5")

        # Recommendation 1: Material substitutions (if significant savings available)
        if substitutions:
            total_sub_savings = sum(_decimal(s.savings_kgCO2e) for s in substitutions[:5])
            if total_sub_savings > Decimal("0"):
                top_sub = substitutions[0]
                recs.append(DesignRecommendation(
                    priority=priority,
                    category="material_substitution",
                    recommendation=f"Substitute {top_sub.original_material_id} with {top_sub.alternative_material_id} "
                                   f"to save {top_sub.savings_kgCO2e} kgCO2e ({top_sub.savings_pct}% reduction). "
                                   f"{top_sub.notes}",
                    estimated_savings_pct=top_sub.savings_pct,
                    lifecycle_stage_affected="A1-A3",
                ))
                priority += 1

        # Recommendation 2: Concrete optimisation
        concrete_mats = [mr for mr in material_results if "concrete" in mr.material_id]
        if concrete_mats:
            concrete_ec = sum(_decimal(mr.embodied_carbon_A1A3_kgCO2e) for mr in concrete_mats)
            total_ec = sum(_decimal(mr.embodied_carbon_A1A3_kgCO2e) for mr in material_results)
            concrete_pct = _safe_pct(concrete_ec, total_ec) if total_ec > Decimal("0") else Decimal("0")
            if concrete_pct > Decimal("30"):
                recs.append(DesignRecommendation(
                    priority=priority,
                    category="structural_optimisation",
                    recommendation=f"Concrete contributes {_round1(float(concrete_pct))}% of embodied carbon. "
                                   "Consider: (a) optimising structural design to reduce concrete volume, "
                                   "(b) specifying lower strength classes where possible, "
                                   "(c) maximising GGBS/PFA replacement (target 50%+).",
                    estimated_savings_pct=_round1(float(concrete_pct * Decimal("0.3"))),
                    lifecycle_stage_affected="A1-A3",
                ))
                priority += 1

        # Recommendation 3: Operational energy
        if b6 and b6.pct_of_whole_life > 30:
            recs.append(DesignRecommendation(
                priority=priority,
                category="operational_energy",
                recommendation=f"Operational energy (B6) represents {b6.pct_of_whole_life}% of whole life carbon. "
                               "Reduce through: improved fabric U-values, high-efficiency heat pump, "
                               "on-site renewables, LED lighting with controls.",
                estimated_savings_pct=_round1(b6.pct_of_whole_life * 0.3),
                lifecycle_stage_affected="B6",
            ))
            priority += 1

        # Recommendation 4: Replacement frequency
        if b1b5 and b1b5.pct_of_whole_life > 15:
            recs.append(DesignRecommendation(
                priority=priority,
                category="durability",
                recommendation=f"Replacement/maintenance (B1-B5) is {b1b5.pct_of_whole_life}% of WLC. "
                               "Specify durable materials with longer lifespans, design for disassembly, "
                               "and select components with EPD-verified longevity.",
                estimated_savings_pct=_round1(b1b5.pct_of_whole_life * 0.25),
                lifecycle_stage_affected="B1-B5",
            ))
            priority += 1

        # Recommendation 5: Transport
        a4 = stage_map.get("A4")
        if a4 and a4.pct_of_whole_life > 5:
            recs.append(DesignRecommendation(
                priority=priority,
                category="transport",
                recommendation=f"Transport to site (A4) is {a4.pct_of_whole_life}% of WLC. "
                               "Source materials locally (within 50km where possible), "
                               "use rail for heavy/bulk materials, consolidate deliveries.",
                estimated_savings_pct=_round1(a4.pct_of_whole_life * 0.4),
                lifecycle_stage_affected="A4",
            ))
            priority += 1

        # Recommendation 6: Timber as carbon store
        timber_used = any("timber" in mr.material_id for mr in material_results)
        if not timber_used:
            recs.append(DesignRecommendation(
                priority=priority,
                category="biogenic_carbon",
                recommendation="No timber products identified. Consider timber structure (CLT/glulam) "
                               "or timber cladding to store biogenic carbon (-1.63 kgCO2e/kg sequestered).",
                estimated_savings_pct=5.0,
                lifecycle_stage_affected="A1-A3",
            ))
            priority += 1

        # Recommendation 7: Design for disassembly
        recs.append(DesignRecommendation(
            priority=priority,
            category="circular_economy",
            recommendation="Design for disassembly and reuse: use bolted/mechanical connections, "
                           "avoid composite materials, create a materials passport for Module D credits.",
            estimated_savings_pct=3.0,
            lifecycle_stage_affected="C1-C4 / D",
        ))
        priority += 1

        # Recommendation 8: If over budget
        if over_budget:
            recs.insert(0, DesignRecommendation(
                priority=0,
                category="target_compliance",
                recommendation="CRITICAL: Current design exceeds carbon budget targets. "
                               "Implement all high-feasibility substitutions and structural optimisation "
                               "recommendations as a priority.",
                estimated_savings_pct=15.0,
                lifecycle_stage_affected="A1-A5",
            ))

        return recs

    # ------------------------------------------------------------------ #
    # analyze  (main entry point)
    # ------------------------------------------------------------------ #

    def analyze(self, inp: WholeLifeCarbonInput) -> WholeLifeCarbonResult:
        """Execute full whole life carbon assessment.

        Main entry point.  Calculates all EN 15978 lifecycle stages,
        compares against carbon budgets, identifies top contributors.

        Args:
            inp: Validated assessment input.

        Returns:
            Complete whole life carbon result with provenance hash.
        """
        t0 = time.perf_counter()
        assessment_id = _new_uuid()

        gia = _decimal(inp.gross_internal_area_m2)
        study_period = inp.study_period_years

        # -- Step 1: Calculate embodied carbon per material --
        mat_results: List[MaterialEmbodiedResult] = []
        for m in inp.materials:
            mr = self.calculate_embodied_carbon(
                m, study_period, inp.include_biogenic or m.include_biogenic,
            )
            mat_results.append(mr)

        # -- Step 2: Aggregate A1-A3 --
        total_a1a3 = sum(_decimal(mr.embodied_carbon_A1A3_kgCO2e) for mr in mat_results)

        # -- Step 3: Transport A4 --
        total_a4 = sum(_decimal(mr.transport_A4_kgCO2e) for mr in mat_results)

        # -- Step 4: Construction A5 --
        total_a5 = self.calculate_construction_carbon(mat_results, inp.construction_waste_pct)

        # -- Step 5: Replacement B4 --
        total_b4 = self.calculate_replacement_carbon(mat_results)
        # B1-B3, B5 typically minor -- approximate as 10% of B4
        total_b1b3b5 = total_b4 * Decimal("0.1")
        total_b1b5 = total_b4 + total_b1b3b5

        # -- Step 6: Operational energy B6 --
        annual_energy = _decimal(inp.annual_energy_kwh_m2)
        total_b6 = Decimal("0")
        if annual_energy > Decimal("0"):
            total_b6 = self.calculate_operational_carbon(
                annual_energy, gia, inp.country_code, inp.start_year, study_period,
            )

        # -- Step 7: Operational water B7 --
        if inp.annual_water_m3_m2 is not None:
            annual_water = _decimal(inp.annual_water_m3_m2)
        else:
            annual_water = _decimal(WATER_USE_INTENSITY.get(inp.building_type, "0.7"))
        water_cf = _decimal(WATER_CARBON_FACTOR)
        total_b7 = annual_water * gia * water_cf * _decimal(study_period)

        # -- Step 8: End of life C1-C4 --
        total_c1c4 = self.calculate_end_of_life(mat_results)

        # -- Step 9: Module D --
        total_d = Decimal("0")
        if inp.include_module_d:
            total_d = self.calculate_module_d(mat_results)

        # -- Step 10: Biogenic carbon --
        total_biogenic = sum(_decimal(mr.biogenic_carbon_kgCO2e) for mr in mat_results)

        # -- Step 11: Whole life totals --
        total_upfront = total_a1a3 + total_a4 + total_a5
        wlc_ac, wlc_acd = self.calculate_whole_life(
            total_a1a3, total_a4, total_a5, total_b1b5, total_b6, total_b7, total_c1c4, total_d,
        )

        wlc_ac_per_m2 = _safe_divide(wlc_ac, gia)
        wlc_acd_per_m2 = _safe_divide(wlc_acd, gia)
        wlc_ac_per_m2_yr = _safe_divide(wlc_ac_per_m2, _decimal(study_period))

        # -- Step 12: Stage breakdown --
        stage_data = [
            ("A1-A3", "Product Stage (Raw Material + Transport + Manufacturing)", total_a1a3),
            ("A4", "Transport to Site", total_a4),
            ("A5", "Construction / Installation", total_a5),
            ("B1-B5", "Use Stage (Maintenance, Repair, Replacement, Refurbishment)", total_b1b5),
            ("B6", "Operational Energy Use", total_b6),
            ("B7", "Operational Water Use", total_b7),
            ("C1-C4", "End of Life (Deconstruction, Transport, Processing, Disposal)", total_c1c4),
            ("D", "Benefits Beyond System Boundary", total_d),
        ]

        stage_results: List[LifecycleStageResult] = []
        for stage_id, stage_name, stage_total in stage_data:
            pct = _safe_pct(abs(stage_total), wlc_ac) if wlc_ac > Decimal("0") else Decimal("0")
            stage_results.append(LifecycleStageResult(
                stage=stage_id,
                stage_name=stage_name,
                total_kgCO2e=_round2(float(stage_total)),
                per_m2_kgCO2e=_round2(float(_safe_divide(stage_total, gia))),
                pct_of_whole_life=_round1(float(pct)),
            ))

        # -- Step 13: Target comparisons --
        upfront_per_m2 = _safe_divide(total_upfront, gia)
        comparisons = self.compare_targets(
            upfront_per_m2, wlc_ac_per_m2, wlc_acd_per_m2,
            inp.building_type, inp.target_standards,
        )

        # -- Step 14: Top contributors --
        sorted_mats = sorted(mat_results, key=lambda x: x.embodied_carbon_A1A3_kgCO2e, reverse=True)
        top_n = min(10, len(sorted_mats))
        top_contributors: List[TopMaterialContributor] = []
        for mr in sorted_mats[:top_n]:
            pct = _safe_pct(_decimal(mr.embodied_carbon_A1A3_kgCO2e), total_a1a3) if total_a1a3 > Decimal("0") else Decimal("0")
            top_contributors.append(TopMaterialContributor(
                material_id=mr.material_id,
                material_category=mr.material_category,
                embodied_A1A3_kgCO2e=mr.embodied_carbon_A1A3_kgCO2e,
                pct_of_total=_round1(float(pct)),
            ))

        # -- Step 15: Substitution opportunities --
        substitutions = self.analyze_substitutions(mat_results)

        # -- Step 16: Sensitivity analysis --
        sensitivity = self.run_sensitivity_analysis(
            total_a1a3, total_a4, total_a5, total_b1b5,
            total_b6, total_b7, total_c1c4, total_d, gia,
        )

        # -- Step 17: Design recommendations --
        recommendations = self.generate_recommendations(
            stage_results, mat_results, comparisons, substitutions,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = WholeLifeCarbonResult(
            assessment_id=assessment_id,
            building_id=inp.building_id,
            building_type=inp.building_type,
            country_code=inp.country_code,
            gross_internal_area_m2=_round2(float(gia)),
            study_period_years=study_period,
            start_year=inp.start_year,
            material_results=mat_results,
            stage_results=stage_results,
            total_A1A3_kgCO2e=_round2(float(total_a1a3)),
            total_A4_kgCO2e=_round2(float(total_a4)),
            total_A5_kgCO2e=_round2(float(total_a5)),
            total_upfront_A1A5_kgCO2e=_round2(float(total_upfront)),
            total_B1B5_kgCO2e=_round2(float(total_b1b5)),
            total_B6_kgCO2e=_round2(float(total_b6)),
            total_B7_kgCO2e=_round2(float(total_b7)),
            total_C1C4_kgCO2e=_round2(float(total_c1c4)),
            total_D_kgCO2e=_round2(float(total_d)),
            total_biogenic_kgCO2e=_round2(float(total_biogenic)),
            whole_life_AC_kgCO2e=_round2(float(wlc_ac)),
            whole_life_ACD_kgCO2e=_round2(float(wlc_acd)),
            whole_life_AC_per_m2=_round2(float(wlc_ac_per_m2)),
            whole_life_ACD_per_m2=_round2(float(wlc_acd_per_m2)),
            whole_life_AC_per_m2_per_yr=_round3(float(wlc_ac_per_m2_yr)),
            target_comparisons=comparisons,
            top_contributors=top_contributors,
            substitution_opportunities=substitutions,
            sensitivity_scenarios=sensitivity,
            design_recommendations=recommendations,
            engine_version=_MODULE_VERSION,
            calculated_at=utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 2),
            provenance_hash="",
        )

        result.provenance_hash = _compute_hash(result)
        return result
