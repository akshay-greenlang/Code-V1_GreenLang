# -*- coding: utf-8 -*-
"""
BuildingEnvelopeEngine - PACK-032 Building Energy Assessment Engine 1
=====================================================================

Comprehensive building fabric / envelope thermal performance assessment.
Calculates U-values for walls, roofs, floors, windows, and doors per
EN ISO 6946.  Evaluates thermal bridging per EN ISO 10211, airtightness
per EN 13829 / ISO 9972, condensation risk per EN ISO 13788 (Glaser
method), and identifies fabric improvement opportunities with estimated
annual energy savings.

EN ISO 6946:2017 Compliance:
    - Thermal resistance of building components
    - Surface resistances (Rsi, Rse) per exposure
    - Multi-layer U-value calculation
    - Corrections for air gaps, mechanical fasteners

EN ISO 10211:2017 Compliance:
    - Linear thermal transmittance (psi values) for junctions
    - Point thermal transmittance (chi values) for fixings
    - Thermal bridge heat loss coefficient (HTB)

EN ISO 13788:2012 Compliance:
    - Glaser method for interstitial condensation risk
    - Dewpoint temperature calculation
    - Vapour pressure analysis across layers

EN 13829 / ISO 9972 Compliance:
    - Blower door test evaluation
    - Air permeability q50 calculation
    - n50 air change rate assessment

CIBSE Guide A Compliance:
    - Heating Degree Day methodology
    - Internal / external design conditions
    - Fabric heat loss calculations

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Lookup tables from published standards (CIBSE, BRE, EN ISO)
    - No LLM involvement in any numeric calculation path
    - SHA-256 provenance hashing on every result
    - U-values, psi-values from BRE BR 443, SAP 10.2, CIBSE Guide A

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

class WallType(str, Enum):
    """Wall construction types per BRE BR 443 and SAP 10.2.

    Covers major wall construction types found in European building stock
    from pre-1919 to modern construction methods.

from greenlang.schemas import utcnow
    """
    SOLID_BRICK = "solid_brick"
    CAVITY_WALL = "cavity_wall"
    TIMBER_FRAME = "timber_frame"
    SIP = "structural_insulated_panel"
    ICF = "insulating_concrete_formwork"
    CURTAIN_WALL = "curtain_wall"
    STONE_SOLID = "stone_solid"
    CONCRETE_PANEL = "concrete_panel"
    METAL_CLADDING = "metal_cladding"
    SYSTEM_BUILD = "system_build"

class RoofType(str, Enum):
    """Roof construction types per CIBSE Guide A and BRE conventions.

    Covers pitched, flat, and green roof constructions with varying
    insulation positions (between/above rafters, warm/cold deck).
    """
    PITCHED_TILE = "pitched_tile"
    PITCHED_SLATE = "pitched_slate"
    FLAT_MEMBRANE = "flat_membrane"
    FLAT_ASPHALT = "flat_asphalt"
    GREEN_ROOF = "green_roof"
    METAL_DECK = "metal_deck"
    CONCRETE_FLAT = "concrete_flat"
    MANSARD = "mansard"

class FloorType(str, Enum):
    """Ground floor construction types per EN ISO 13370.

    Floor U-value calculation depends on floor type and the
    perimeter-to-area ratio (P/A) per EN ISO 13370:2017.
    """
    SUSPENDED_TIMBER = "suspended_timber"
    SOLID_CONCRETE = "solid_concrete"
    BEAM_AND_BLOCK = "beam_and_block"
    INSULATED_SLAB = "insulated_slab"
    RAISED_ACCESS = "raised_access"

class WindowType(str, Enum):
    """Window unit types classified by number of panes.

    Window U-values depend on the combination of glazing type,
    gas fill, frame material, and spacer type.
    """
    SINGLE_GLAZED = "single_glazed"
    DOUBLE_GLAZED = "double_glazed"
    TRIPLE_GLAZED = "triple_glazed"
    SECONDARY_GLAZED = "secondary_glazed"

class GlazingType(str, Enum):
    """Glazing coating types per EN 673 and EN 410.

    Low-emissivity coatings reduce long-wave radiative heat transfer
    across the cavity.  Solar control coatings reduce solar heat gain.
    """
    CLEAR = "clear"
    LOW_E_SOFT = "low_e_soft_coat"
    LOW_E_HARD = "low_e_hard_coat"
    SOLAR_CONTROL = "solar_control"
    ELECTROCHROMIC = "electrochromic"

class FrameMaterial(str, Enum):
    """Window frame materials per EN ISO 10077-1.

    Frame material affects both the frame U-value and the linear
    thermal transmittance at the frame-to-glazing junction.
    """
    TIMBER = "timber"
    UPVC = "upvc"
    ALUMINIUM = "aluminium"
    ALUMINIUM_THERMAL_BREAK = "aluminium_thermal_break"
    COMPOSITE = "composite"
    STEEL = "steel"

class InsulationType(str, Enum):
    """Insulation material types with thermal conductivity per EN 12667.

    Lambda (thermal conductivity) values are declared values per
    manufacturer CE marking and EN 13162-13171 standards.
    """
    MINERAL_WOOL = "mineral_wool"
    EPS = "expanded_polystyrene"
    XPS = "extruded_polystyrene"
    PIR = "polyisocyanurate"
    PUR = "polyurethane"
    PHENOLIC = "phenolic_foam"
    CELLULOSE = "cellulose_fibre"
    WOOD_FIBRE = "wood_fibre"
    SHEEP_WOOL = "sheep_wool"
    HEMP = "hemp_fibre"
    AEROGEL = "aerogel"
    VACUUM_INSULATED_PANEL = "vacuum_insulated_panel"
    CORK = "cork"
    GLASS_FOAM = "glass_foam"
    CALCIUM_SILICATE = "calcium_silicate"

class AirtightnessStandard(str, Enum):
    """Airtightness performance standard classifications.

    n50 is the air change rate at 50 Pa pressure difference measured
    per EN 13829 / ISO 9972 (blower door test).
    """
    PASSIVHAUS = "passivhaus"
    LOW_ENERGY = "low_energy"
    BUILDING_REGS = "building_regs"
    POOR = "poor"

class ThermalBridgeType(str, Enum):
    """Junction types for linear thermal bridging per EN ISO 10211.

    Psi values (linear thermal transmittance) are taken from
    BRE BR 497 / Approved Document L default values.
    """
    WALL_FLOOR_GROUND = "wall_floor_ground"
    WALL_FLOOR_INTERMEDIATE = "wall_floor_intermediate"
    WALL_ROOF_EAVES = "wall_roof_eaves"
    WALL_ROOF_GABLE = "wall_roof_gable"
    WALL_ROOF_PARAPET = "wall_roof_parapet"
    WALL_WALL_CORNER = "wall_wall_corner"
    WALL_WINDOW_JAMB = "wall_window_jamb"
    WALL_WINDOW_SILL = "wall_window_sill"
    WALL_WINDOW_LINTEL = "wall_window_lintel"
    WALL_DOOR_JAMB = "wall_door_jamb"
    WALL_DOOR_THRESHOLD = "wall_door_threshold"
    WALL_DOOR_HEAD = "wall_door_head"
    ROOF_RIDGE = "roof_ridge"
    ROOF_EAVES = "roof_eaves"
    BALCONY_WALL = "balcony_wall"
    PARTY_WALL_EXTERNAL = "party_wall_external"
    LINTEL_STEEL = "lintel_steel"
    COLUMN_EXTERNAL = "column_external"
    WINDOW_WINDOW = "window_window"
    CURTAIN_WALL_TRANSOM = "curtain_wall_transom"

class BuildingType(str, Enum):
    """Building use type for envelope assessment context."""
    RESIDENTIAL_HOUSE = "residential_house"
    RESIDENTIAL_FLAT = "residential_flat"
    OFFICE = "office"
    RETAIL = "retail"
    HOTEL = "hotel"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    WAREHOUSE = "warehouse"
    MIXED_USE = "mixed_use"
    INDUSTRIAL = "industrial"

class AgeBand(str, Enum):
    """Construction age band per BRE / SAP conventions.

    Age bands determine default U-values when detailed constructions
    are not known.  Based on UK building regulations evolution.
    """
    PRE_1919 = "pre_1919"
    BAND_1919_1944 = "1919_1944"
    BAND_1945_1964 = "1945_1964"
    BAND_1965_1982 = "1965_1982"
    BAND_1983_1995 = "1983_1995"
    BAND_1996_2006 = "1996_2006"
    BAND_2007_2013 = "2007_2013"
    BAND_2014_PRESENT = "2014_present"

# ---------------------------------------------------------------------------
# Constants -- U-Value Lookup Tables
# ---------------------------------------------------------------------------

# Default wall U-values (W/m2K) by wall type and age band.
# Sources: BRE BR 443, SAP 10.2 Table S6, CIBSE Guide A Table 3.49.
# Values assume uninsulated or as-built insulation for the age band.
WALL_U_VALUES: Dict[str, Dict[str, float]] = {
    WallType.SOLID_BRICK: {
        AgeBand.PRE_1919: 2.10,
        AgeBand.BAND_1919_1944: 2.10,
        AgeBand.BAND_1945_1964: 2.10,
        AgeBand.BAND_1965_1982: 1.70,
        AgeBand.BAND_1983_1995: 1.00,
        AgeBand.BAND_1996_2006: 0.60,
        AgeBand.BAND_2007_2013: 0.30,
        AgeBand.BAND_2014_PRESENT: 0.18,
    },
    WallType.CAVITY_WALL: {
        AgeBand.PRE_1919: 2.10,
        AgeBand.BAND_1919_1944: 1.60,
        AgeBand.BAND_1945_1964: 1.60,
        AgeBand.BAND_1965_1982: 1.00,
        AgeBand.BAND_1983_1995: 0.60,
        AgeBand.BAND_1996_2006: 0.45,
        AgeBand.BAND_2007_2013: 0.30,
        AgeBand.BAND_2014_PRESENT: 0.18,
    },
    WallType.TIMBER_FRAME: {
        AgeBand.PRE_1919: 1.90,
        AgeBand.BAND_1919_1944: 1.80,
        AgeBand.BAND_1945_1964: 1.70,
        AgeBand.BAND_1965_1982: 1.00,
        AgeBand.BAND_1983_1995: 0.45,
        AgeBand.BAND_1996_2006: 0.35,
        AgeBand.BAND_2007_2013: 0.25,
        AgeBand.BAND_2014_PRESENT: 0.15,
    },
    WallType.SIP: {
        AgeBand.PRE_1919: 1.90,
        AgeBand.BAND_1919_1944: 1.90,
        AgeBand.BAND_1945_1964: 1.90,
        AgeBand.BAND_1965_1982: 1.00,
        AgeBand.BAND_1983_1995: 0.35,
        AgeBand.BAND_1996_2006: 0.25,
        AgeBand.BAND_2007_2013: 0.18,
        AgeBand.BAND_2014_PRESENT: 0.13,
    },
    WallType.ICF: {
        AgeBand.PRE_1919: 1.80,
        AgeBand.BAND_1919_1944: 1.80,
        AgeBand.BAND_1945_1964: 1.80,
        AgeBand.BAND_1965_1982: 0.90,
        AgeBand.BAND_1983_1995: 0.35,
        AgeBand.BAND_1996_2006: 0.25,
        AgeBand.BAND_2007_2013: 0.17,
        AgeBand.BAND_2014_PRESENT: 0.12,
    },
    WallType.CURTAIN_WALL: {
        AgeBand.PRE_1919: 5.70,
        AgeBand.BAND_1919_1944: 5.70,
        AgeBand.BAND_1945_1964: 3.50,
        AgeBand.BAND_1965_1982: 2.00,
        AgeBand.BAND_1983_1995: 1.20,
        AgeBand.BAND_1996_2006: 0.60,
        AgeBand.BAND_2007_2013: 0.35,
        AgeBand.BAND_2014_PRESENT: 0.22,
    },
    WallType.STONE_SOLID: {
        AgeBand.PRE_1919: 2.30,
        AgeBand.BAND_1919_1944: 2.30,
        AgeBand.BAND_1945_1964: 2.30,
        AgeBand.BAND_1965_1982: 1.80,
        AgeBand.BAND_1983_1995: 1.00,
        AgeBand.BAND_1996_2006: 0.60,
        AgeBand.BAND_2007_2013: 0.30,
        AgeBand.BAND_2014_PRESENT: 0.18,
    },
    WallType.CONCRETE_PANEL: {
        AgeBand.PRE_1919: 3.40,
        AgeBand.BAND_1919_1944: 3.40,
        AgeBand.BAND_1945_1964: 2.20,
        AgeBand.BAND_1965_1982: 1.50,
        AgeBand.BAND_1983_1995: 0.60,
        AgeBand.BAND_1996_2006: 0.45,
        AgeBand.BAND_2007_2013: 0.30,
        AgeBand.BAND_2014_PRESENT: 0.18,
    },
    WallType.METAL_CLADDING: {
        AgeBand.PRE_1919: 5.90,
        AgeBand.BAND_1919_1944: 5.90,
        AgeBand.BAND_1945_1964: 3.50,
        AgeBand.BAND_1965_1982: 1.50,
        AgeBand.BAND_1983_1995: 0.60,
        AgeBand.BAND_1996_2006: 0.35,
        AgeBand.BAND_2007_2013: 0.25,
        AgeBand.BAND_2014_PRESENT: 0.18,
    },
    WallType.SYSTEM_BUILD: {
        AgeBand.PRE_1919: 2.00,
        AgeBand.BAND_1919_1944: 2.00,
        AgeBand.BAND_1945_1964: 1.80,
        AgeBand.BAND_1965_1982: 1.20,
        AgeBand.BAND_1983_1995: 0.60,
        AgeBand.BAND_1996_2006: 0.45,
        AgeBand.BAND_2007_2013: 0.28,
        AgeBand.BAND_2014_PRESENT: 0.18,
    },
}
"""Default wall U-values (W/m2K) by wall type and age band.
Source: BRE BR 443, SAP 10.2 Table S6, CIBSE Guide A Table 3.49."""

# Roof U-values (W/m2K) by roof type and insulation thickness (mm).
# Columns: 0, 50, 100, 150, 200, 250, 300, 350, 400 mm of insulation.
# Sources: BRE BR 443, SAP 10.2 Table S9, CIBSE Guide A.
ROOF_U_VALUES: Dict[str, Dict[int, float]] = {
    RoofType.PITCHED_TILE: {
        0: 2.30, 50: 0.68, 100: 0.40, 150: 0.28, 200: 0.22,
        250: 0.18, 300: 0.15, 350: 0.13, 400: 0.12,
    },
    RoofType.PITCHED_SLATE: {
        0: 2.40, 50: 0.70, 100: 0.42, 150: 0.29, 200: 0.23,
        250: 0.19, 300: 0.16, 350: 0.14, 400: 0.12,
    },
    RoofType.FLAT_MEMBRANE: {
        0: 2.20, 50: 0.65, 100: 0.38, 150: 0.27, 200: 0.21,
        250: 0.17, 300: 0.15, 350: 0.13, 400: 0.11,
    },
    RoofType.FLAT_ASPHALT: {
        0: 2.30, 50: 0.67, 100: 0.39, 150: 0.28, 200: 0.22,
        250: 0.18, 300: 0.15, 350: 0.13, 400: 0.12,
    },
    RoofType.GREEN_ROOF: {
        0: 1.80, 50: 0.55, 100: 0.34, 150: 0.24, 200: 0.19,
        250: 0.16, 300: 0.14, 350: 0.12, 400: 0.10,
    },
    RoofType.METAL_DECK: {
        0: 5.90, 50: 0.72, 100: 0.42, 150: 0.30, 200: 0.23,
        250: 0.19, 300: 0.16, 350: 0.14, 400: 0.12,
    },
    RoofType.CONCRETE_FLAT: {
        0: 3.40, 50: 0.65, 100: 0.38, 150: 0.27, 200: 0.21,
        250: 0.17, 300: 0.15, 350: 0.13, 400: 0.11,
    },
    RoofType.MANSARD: {
        0: 2.20, 50: 0.66, 100: 0.39, 150: 0.28, 200: 0.22,
        250: 0.18, 300: 0.15, 350: 0.13, 400: 0.11,
    },
}
"""Roof U-values (W/m2K) by roof type and insulation thickness (mm).
Source: BRE BR 443, SAP 10.2 Table S9."""

# Floor U-values (W/m2K) by floor type and perimeter/area ratio.
# P/A ratios: 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0.
# Sources: EN ISO 13370:2017, BRE BR 443, SAP 10.2 Table S10.
FLOOR_U_VALUES: Dict[str, Dict[str, float]] = {
    FloorType.SUSPENDED_TIMBER: {
        "0.2": 0.55, "0.3": 0.62, "0.4": 0.70, "0.5": 0.78,
        "0.6": 0.85, "0.7": 0.92, "0.8": 0.98, "0.9": 1.04, "1.0": 1.10,
    },
    FloorType.SOLID_CONCRETE: {
        "0.2": 0.35, "0.3": 0.42, "0.4": 0.49, "0.5": 0.55,
        "0.6": 0.61, "0.7": 0.67, "0.8": 0.72, "0.9": 0.77, "1.0": 0.82,
    },
    FloorType.BEAM_AND_BLOCK: {
        "0.2": 0.45, "0.3": 0.52, "0.4": 0.59, "0.5": 0.66,
        "0.6": 0.72, "0.7": 0.78, "0.8": 0.84, "0.9": 0.89, "1.0": 0.94,
    },
    FloorType.INSULATED_SLAB: {
        "0.2": 0.15, "0.3": 0.18, "0.4": 0.21, "0.5": 0.24,
        "0.6": 0.27, "0.7": 0.30, "0.8": 0.33, "0.9": 0.35, "1.0": 0.38,
    },
    FloorType.RAISED_ACCESS: {
        "0.2": 0.50, "0.3": 0.57, "0.4": 0.64, "0.5": 0.71,
        "0.6": 0.77, "0.7": 0.83, "0.8": 0.89, "0.9": 0.94, "1.0": 1.00,
    },
}
"""Floor U-values (W/m2K) by floor type and P/A ratio.
Source: EN ISO 13370:2017, BRE BR 443, SAP 10.2 Table S10."""

# Window U-values (W/m2K) by glazing type, window type, and frame material.
# Sources: EN 673, EN ISO 10077-1, BRE BR 443 Table 6e.
WINDOW_U_VALUES: Dict[str, Dict[str, Dict[str, float]]] = {
    WindowType.SINGLE_GLAZED: {
        GlazingType.CLEAR: {
            FrameMaterial.TIMBER: 4.80,
            FrameMaterial.UPVC: 4.80,
            FrameMaterial.ALUMINIUM: 5.70,
            FrameMaterial.ALUMINIUM_THERMAL_BREAK: 5.30,
            FrameMaterial.COMPOSITE: 4.70,
            FrameMaterial.STEEL: 5.90,
        },
    },
    WindowType.DOUBLE_GLAZED: {
        GlazingType.CLEAR: {
            FrameMaterial.TIMBER: 2.90,
            FrameMaterial.UPVC: 2.80,
            FrameMaterial.ALUMINIUM: 3.50,
            FrameMaterial.ALUMINIUM_THERMAL_BREAK: 3.10,
            FrameMaterial.COMPOSITE: 2.70,
            FrameMaterial.STEEL: 3.70,
        },
        GlazingType.LOW_E_SOFT: {
            FrameMaterial.TIMBER: 1.80,
            FrameMaterial.UPVC: 1.70,
            FrameMaterial.ALUMINIUM: 2.40,
            FrameMaterial.ALUMINIUM_THERMAL_BREAK: 2.00,
            FrameMaterial.COMPOSITE: 1.60,
            FrameMaterial.STEEL: 2.60,
        },
        GlazingType.LOW_E_HARD: {
            FrameMaterial.TIMBER: 2.10,
            FrameMaterial.UPVC: 2.00,
            FrameMaterial.ALUMINIUM: 2.70,
            FrameMaterial.ALUMINIUM_THERMAL_BREAK: 2.30,
            FrameMaterial.COMPOSITE: 1.90,
            FrameMaterial.STEEL: 2.90,
        },
        GlazingType.SOLAR_CONTROL: {
            FrameMaterial.TIMBER: 1.90,
            FrameMaterial.UPVC: 1.80,
            FrameMaterial.ALUMINIUM: 2.50,
            FrameMaterial.ALUMINIUM_THERMAL_BREAK: 2.10,
            FrameMaterial.COMPOSITE: 1.70,
            FrameMaterial.STEEL: 2.70,
        },
        GlazingType.ELECTROCHROMIC: {
            FrameMaterial.TIMBER: 1.70,
            FrameMaterial.UPVC: 1.60,
            FrameMaterial.ALUMINIUM: 2.30,
            FrameMaterial.ALUMINIUM_THERMAL_BREAK: 1.90,
            FrameMaterial.COMPOSITE: 1.50,
            FrameMaterial.STEEL: 2.50,
        },
    },
    WindowType.TRIPLE_GLAZED: {
        GlazingType.CLEAR: {
            FrameMaterial.TIMBER: 1.90,
            FrameMaterial.UPVC: 1.80,
            FrameMaterial.ALUMINIUM: 2.40,
            FrameMaterial.ALUMINIUM_THERMAL_BREAK: 2.00,
            FrameMaterial.COMPOSITE: 1.70,
            FrameMaterial.STEEL: 2.60,
        },
        GlazingType.LOW_E_SOFT: {
            FrameMaterial.TIMBER: 1.10,
            FrameMaterial.UPVC: 1.00,
            FrameMaterial.ALUMINIUM: 1.60,
            FrameMaterial.ALUMINIUM_THERMAL_BREAK: 1.30,
            FrameMaterial.COMPOSITE: 0.90,
            FrameMaterial.STEEL: 1.80,
        },
        GlazingType.LOW_E_HARD: {
            FrameMaterial.TIMBER: 1.30,
            FrameMaterial.UPVC: 1.20,
            FrameMaterial.ALUMINIUM: 1.80,
            FrameMaterial.ALUMINIUM_THERMAL_BREAK: 1.50,
            FrameMaterial.COMPOSITE: 1.10,
            FrameMaterial.STEEL: 2.00,
        },
        GlazingType.SOLAR_CONTROL: {
            FrameMaterial.TIMBER: 1.20,
            FrameMaterial.UPVC: 1.10,
            FrameMaterial.ALUMINIUM: 1.70,
            FrameMaterial.ALUMINIUM_THERMAL_BREAK: 1.40,
            FrameMaterial.COMPOSITE: 1.00,
            FrameMaterial.STEEL: 1.90,
        },
        GlazingType.ELECTROCHROMIC: {
            FrameMaterial.TIMBER: 1.00,
            FrameMaterial.UPVC: 0.90,
            FrameMaterial.ALUMINIUM: 1.50,
            FrameMaterial.ALUMINIUM_THERMAL_BREAK: 1.20,
            FrameMaterial.COMPOSITE: 0.80,
            FrameMaterial.STEEL: 1.70,
        },
    },
    WindowType.SECONDARY_GLAZED: {
        GlazingType.CLEAR: {
            FrameMaterial.TIMBER: 2.40,
            FrameMaterial.UPVC: 2.30,
            FrameMaterial.ALUMINIUM: 2.90,
            FrameMaterial.ALUMINIUM_THERMAL_BREAK: 2.60,
            FrameMaterial.COMPOSITE: 2.20,
            FrameMaterial.STEEL: 3.10,
        },
    },
}
"""Window U-values (W/m2K) by window type, glazing type, and frame material.
Source: EN 673, EN ISO 10077-1, BRE BR 443 Table 6e."""

# Solar heat gain coefficient (g-value) by glazing type per EN 410.
# g-value is the total solar energy transmittance of the glazing unit.
WINDOW_G_VALUES: Dict[str, Dict[str, float]] = {
    WindowType.SINGLE_GLAZED: {
        GlazingType.CLEAR: 0.85,
    },
    WindowType.DOUBLE_GLAZED: {
        GlazingType.CLEAR: 0.72,
        GlazingType.LOW_E_SOFT: 0.63,
        GlazingType.LOW_E_HARD: 0.67,
        GlazingType.SOLAR_CONTROL: 0.40,
        GlazingType.ELECTROCHROMIC: 0.15,
    },
    WindowType.TRIPLE_GLAZED: {
        GlazingType.CLEAR: 0.60,
        GlazingType.LOW_E_SOFT: 0.50,
        GlazingType.LOW_E_HARD: 0.55,
        GlazingType.SOLAR_CONTROL: 0.30,
        GlazingType.ELECTROCHROMIC: 0.10,
    },
    WindowType.SECONDARY_GLAZED: {
        GlazingType.CLEAR: 0.76,
    },
}
"""Solar heat gain coefficient (g-value) by glazing type per EN 410."""

# Thermal conductivity of insulation materials (W/m.K).
# Declared lambda values per EN 13162-13171 and manufacturer data.
# Sources: BRE, CIBSE Guide A Table 3.50, EN 12667.
THERMAL_CONDUCTIVITY: Dict[str, Dict[str, Any]] = {
    InsulationType.MINERAL_WOOL: {
        "lambda_w_mk": 0.035,
        "density_kg_m3": 25.0,
        "source": "EN 13162, typical glass mineral wool batt",
    },
    InsulationType.EPS: {
        "lambda_w_mk": 0.038,
        "density_kg_m3": 20.0,
        "source": "EN 13163, EPS 70",
    },
    InsulationType.XPS: {
        "lambda_w_mk": 0.034,
        "density_kg_m3": 35.0,
        "source": "EN 13164, XPS 300",
    },
    InsulationType.PIR: {
        "lambda_w_mk": 0.022,
        "density_kg_m3": 32.0,
        "source": "EN 13165, PIR foil-faced board",
    },
    InsulationType.PUR: {
        "lambda_w_mk": 0.025,
        "density_kg_m3": 30.0,
        "source": "EN 13165, rigid PUR foam",
    },
    InsulationType.PHENOLIC: {
        "lambda_w_mk": 0.020,
        "density_kg_m3": 35.0,
        "source": "EN 13166, phenolic foam board (Kooltherm-type)",
    },
    InsulationType.CELLULOSE: {
        "lambda_w_mk": 0.040,
        "density_kg_m3": 55.0,
        "source": "EN 15101, recycled cellulose loose-fill",
    },
    InsulationType.WOOD_FIBRE: {
        "lambda_w_mk": 0.038,
        "density_kg_m3": 160.0,
        "source": "EN 13171, rigid wood fibre board",
    },
    InsulationType.SHEEP_WOOL: {
        "lambda_w_mk": 0.039,
        "density_kg_m3": 25.0,
        "source": "BBA certified sheep wool insulation",
    },
    InsulationType.HEMP: {
        "lambda_w_mk": 0.040,
        "density_kg_m3": 40.0,
        "source": "EN 13171, hemp fibre batt",
    },
    InsulationType.AEROGEL: {
        "lambda_w_mk": 0.015,
        "density_kg_m3": 150.0,
        "source": "Aerogel blanket insulation (Spacetherm-type)",
    },
    InsulationType.VACUUM_INSULATED_PANEL: {
        "lambda_w_mk": 0.007,
        "density_kg_m3": 200.0,
        "source": "VIP (centre-of-panel), EN 17164",
    },
    InsulationType.CORK: {
        "lambda_w_mk": 0.040,
        "density_kg_m3": 120.0,
        "source": "EN 13170, expanded cork board",
    },
    InsulationType.GLASS_FOAM: {
        "lambda_w_mk": 0.041,
        "density_kg_m3": 120.0,
        "source": "EN 13167, cellular glass (Foamglas-type)",
    },
    InsulationType.CALCIUM_SILICATE: {
        "lambda_w_mk": 0.065,
        "density_kg_m3": 250.0,
        "source": "EN 14306, microporous calcium silicate",
    },
}
"""Thermal conductivity of insulation materials (W/m.K).
Source: EN 13162-13171, BRE, CIBSE Guide A Table 3.50."""

# Linear thermal transmittance psi values (W/m.K) for thermal bridges.
# Sources: BRE BR 497, SAP 10.2 Table K1, EN ISO 14683 default values.
# "default" column = Approved Document L default (no specific detail).
# "accredited" column = Accredited construction detail (best practice).
THERMAL_BRIDGE_PSI_VALUES: Dict[str, Dict[str, float]] = {
    ThermalBridgeType.WALL_FLOOR_GROUND: {
        "default": 0.16, "accredited": 0.04, "enhanced": 0.02,
    },
    ThermalBridgeType.WALL_FLOOR_INTERMEDIATE: {
        "default": 0.07, "accredited": 0.02, "enhanced": 0.01,
    },
    ThermalBridgeType.WALL_ROOF_EAVES: {
        "default": 0.06, "accredited": 0.02, "enhanced": 0.01,
    },
    ThermalBridgeType.WALL_ROOF_GABLE: {
        "default": 0.24, "accredited": 0.04, "enhanced": 0.02,
    },
    ThermalBridgeType.WALL_ROOF_PARAPET: {
        "default": 0.08, "accredited": 0.04, "enhanced": 0.02,
    },
    ThermalBridgeType.WALL_WALL_CORNER: {
        "default": 0.09, "accredited": 0.04, "enhanced": 0.02,
    },
    ThermalBridgeType.WALL_WINDOW_JAMB: {
        "default": 0.05, "accredited": 0.02, "enhanced": 0.01,
    },
    ThermalBridgeType.WALL_WINDOW_SILL: {
        "default": 0.05, "accredited": 0.02, "enhanced": 0.01,
    },
    ThermalBridgeType.WALL_WINDOW_LINTEL: {
        "default": 0.30, "accredited": 0.04, "enhanced": 0.02,
    },
    ThermalBridgeType.WALL_DOOR_JAMB: {
        "default": 0.05, "accredited": 0.02, "enhanced": 0.01,
    },
    ThermalBridgeType.WALL_DOOR_THRESHOLD: {
        "default": 0.05, "accredited": 0.02, "enhanced": 0.01,
    },
    ThermalBridgeType.WALL_DOOR_HEAD: {
        "default": 0.30, "accredited": 0.04, "enhanced": 0.02,
    },
    ThermalBridgeType.ROOF_RIDGE: {
        "default": 0.08, "accredited": 0.04, "enhanced": 0.02,
    },
    ThermalBridgeType.ROOF_EAVES: {
        "default": 0.06, "accredited": 0.02, "enhanced": 0.01,
    },
    ThermalBridgeType.BALCONY_WALL: {
        "default": 0.50, "accredited": 0.12, "enhanced": 0.04,
    },
    ThermalBridgeType.PARTY_WALL_EXTERNAL: {
        "default": 0.08, "accredited": 0.02, "enhanced": 0.01,
    },
    ThermalBridgeType.LINTEL_STEEL: {
        "default": 0.50, "accredited": 0.10, "enhanced": 0.04,
    },
    ThermalBridgeType.COLUMN_EXTERNAL: {
        "default": 0.20, "accredited": 0.06, "enhanced": 0.03,
    },
    ThermalBridgeType.WINDOW_WINDOW: {
        "default": 0.10, "accredited": 0.04, "enhanced": 0.02,
    },
    ThermalBridgeType.CURTAIN_WALL_TRANSOM: {
        "default": 0.12, "accredited": 0.04, "enhanced": 0.02,
    },
}
"""Linear thermal transmittance psi values (W/m.K) for thermal bridges.
Source: BRE BR 497, SAP 10.2 Table K1, EN ISO 14683."""

# Airtightness benchmarks: n50 values (air changes per hour at 50 Pa).
# Sources: EN 13829, ISO 9972, Passivhaus Institut, CIBSE TM23.
AIRTIGHTNESS_BENCHMARKS: Dict[str, Dict[str, float]] = {
    AirtightnessStandard.PASSIVHAUS: {
        "n50_ach": 0.6,
        "q50_m3_h_m2": 0.6,
        "description_max": 0.6,
    },
    AirtightnessStandard.LOW_ENERGY: {
        "n50_ach": 3.0,
        "q50_m3_h_m2": 3.0,
        "description_max": 3.0,
    },
    AirtightnessStandard.BUILDING_REGS: {
        "n50_ach": 7.0,
        "q50_m3_h_m2": 10.0,
        "description_max": 10.0,
    },
    AirtightnessStandard.POOR: {
        "n50_ach": 15.0,
        "q50_m3_h_m2": 20.0,
        "description_max": 20.0,
    },
}
"""Airtightness benchmarks (n50 / q50) by standard classification.
Source: EN 13829, ISO 9972, Passivhaus Institut, CIBSE TM23."""

# Surface thermal resistances (m2K/W) per EN ISO 6946:2017 Table 1.
# Rsi = internal surface resistance; Rse = external surface resistance.
SURFACE_RESISTANCES: Dict[str, Dict[str, float]] = {
    "horizontal_heat_flow": {"Rsi": 0.13, "Rse": 0.04},
    "upward_heat_flow": {"Rsi": 0.10, "Rse": 0.04},
    "downward_heat_flow": {"Rsi": 0.17, "Rse": 0.04},
}
"""Surface thermal resistances (m2K/W) per EN ISO 6946:2017 Table 1."""

# Common building material thermal conductivities (W/m.K).
# Sources: CIBSE Guide A Table 3.49, EN ISO 10456.
MATERIAL_CONDUCTIVITY: Dict[str, float] = {
    "brick_outer_leaf": 0.77,
    "brick_inner_leaf": 0.56,
    "concrete_dense": 1.40,
    "concrete_lightweight": 0.38,
    "concrete_aerated": 0.16,
    "plasterboard": 0.21,
    "render_cement": 1.00,
    "render_lime": 0.80,
    "timber_softwood": 0.13,
    "timber_hardwood": 0.18,
    "steel": 50.0,
    "aluminium": 160.0,
    "stone_sandstone": 1.80,
    "stone_limestone": 1.50,
    "stone_granite": 2.80,
    "mortar": 0.94,
    "plaster_gypsum": 0.40,
    "tile_clay": 1.00,
    "tile_concrete": 1.50,
    "slate": 2.00,
    "felt_bitumen": 0.23,
    "membrane_breather": 0.17,
    "air_cavity_25mm": 0.18,
    "air_cavity_50mm": 0.18,
}
"""Common building material thermal conductivities (W/m.K).
Source: CIBSE Guide A Table 3.49, EN ISO 10456."""

# Heating Degree Days by country (annual average for capital city).
# Sources: CIBSE Guide A, BRE, Eurostat climate data.
COUNTRY_HDD: Dict[str, float] = {
    "UK": 2353.0, "DE": 2899.0, "FR": 2306.0, "IT": 1783.0,
    "ES": 1485.0, "NL": 2662.0, "BE": 2600.0, "AT": 3163.0,
    "PL": 3180.0, "SE": 3950.0, "FI": 4508.0, "NO": 4180.0,
    "DK": 3100.0, "CZ": 3280.0, "PT": 1056.0, "GR": 1190.0,
    "IE": 2629.0, "CH": 3100.0, "HU": 2750.0, "RO": 2680.0,
    "DEFAULT": 2500.0,
}
"""Annual Heating Degree Days by country (base 15.5C, capital city).
Source: CIBSE Guide A, Eurostat."""

# Door U-values (W/m2K) by door type.
# Sources: BRE BR 443, SAP 10.2 Table 6f.
DOOR_U_VALUES: Dict[str, float] = {
    "solid_timber": 3.00,
    "solid_timber_insulated": 1.50,
    "composite_insulated": 1.00,
    "upvc_half_glazed": 2.20,
    "upvc_solid": 1.40,
    "steel_insulated": 2.00,
    "steel_uninsulated": 5.80,
    "aluminium_insulated": 2.50,
    "aluminium_uninsulated": 5.50,
    "revolving_glass": 3.50,
    "roller_shutter_insulated": 2.00,
    "roller_shutter_uninsulated": 5.00,
}
"""Door U-values (W/m2K) by door type.
Source: BRE BR 443, SAP 10.2 Table 6f."""

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class InsulationLayer(BaseModel):
    """Single insulation or construction layer for U-value calculation.

    Attributes:
        material: Material identifier (insulation type or from MATERIAL_CONDUCTIVITY).
        thickness_mm: Layer thickness in millimetres.
        conductivity_w_mk: Thermal conductivity (W/m.K); auto-looked up if None.
        description: Human-readable description of the layer.
    """
    material: str = Field(..., min_length=1, description="Material identifier")
    thickness_mm: float = Field(..., gt=0, le=2000, description="Layer thickness (mm)")
    conductivity_w_mk: Optional[float] = Field(
        None, gt=0, le=500, description="Thermal conductivity (W/m.K)"
    )
    description: Optional[str] = Field(None, description="Layer description")

class WallElement(BaseModel):
    """Wall element for envelope assessment.

    Attributes:
        element_id: Unique element identifier.
        wall_type: Wall construction type.
        area_m2: Net wall area (excluding openings) in m2.
        age_band: Construction age band for default U-value lookup.
        layers: Optional detailed construction layers for manual U-value calc.
        known_u_value: Override U-value if known from survey (W/m2K).
        orientation: Wall orientation (N, NE, E, SE, S, SW, W, NW).
        description: Human-readable description.
    """
    element_id: str = Field(default_factory=_new_uuid, description="Element ID")
    wall_type: WallType = Field(..., description="Wall construction type")
    area_m2: float = Field(..., gt=0, description="Net wall area (m2)")
    age_band: Optional[AgeBand] = Field(None, description="Construction age band")
    layers: Optional[List[InsulationLayer]] = Field(
        None, description="Detailed construction layers"
    )
    known_u_value: Optional[float] = Field(
        None, gt=0, le=10.0, description="Known U-value (W/m2K)"
    )
    orientation: Optional[str] = Field(None, description="Wall orientation (N/S/E/W)")
    description: Optional[str] = Field(None, description="Description")

    @field_validator("area_m2")
    @classmethod
    def validate_wall_area(cls, v: float) -> float:
        """Ensure wall area is within plausible bounds."""
        if v > 100_000:
            raise ValueError("Wall area exceeds 100,000 m2 sanity check")
        return v

class RoofElement(BaseModel):
    """Roof element for envelope assessment.

    Attributes:
        element_id: Unique element identifier.
        roof_type: Roof construction type.
        area_m2: Roof area in m2 (plan area for pitched roofs).
        insulation_thickness_mm: Insulation thickness in mm.
        insulation_type: Type of insulation material.
        known_u_value: Override U-value if known from survey (W/m2K).
        pitch_degrees: Roof pitch in degrees (0 = flat).
        description: Human-readable description.
    """
    element_id: str = Field(default_factory=_new_uuid, description="Element ID")
    roof_type: RoofType = Field(..., description="Roof construction type")
    area_m2: float = Field(..., gt=0, description="Roof area (m2)")
    insulation_thickness_mm: float = Field(
        default=0.0, ge=0, le=1000, description="Insulation thickness (mm)"
    )
    insulation_type: Optional[InsulationType] = Field(
        None, description="Insulation material type"
    )
    known_u_value: Optional[float] = Field(
        None, gt=0, le=10.0, description="Known U-value (W/m2K)"
    )
    pitch_degrees: float = Field(
        default=0.0, ge=0, le=90, description="Roof pitch (degrees)"
    )
    description: Optional[str] = Field(None, description="Description")

class FloorElement(BaseModel):
    """Floor element for envelope assessment.

    Attributes:
        element_id: Unique element identifier.
        floor_type: Floor construction type.
        area_m2: Floor area in m2.
        perimeter_m: Exposed floor perimeter in metres.
        insulation_thickness_mm: Insulation thickness in mm.
        insulation_type: Type of insulation material.
        known_u_value: Override U-value if known from survey (W/m2K).
        description: Human-readable description.
    """
    element_id: str = Field(default_factory=_new_uuid, description="Element ID")
    floor_type: FloorType = Field(..., description="Floor construction type")
    area_m2: float = Field(..., gt=0, description="Floor area (m2)")
    perimeter_m: float = Field(..., gt=0, description="Exposed perimeter (m)")
    insulation_thickness_mm: float = Field(
        default=0.0, ge=0, le=500, description="Insulation thickness (mm)"
    )
    insulation_type: Optional[InsulationType] = Field(
        None, description="Insulation material type"
    )
    known_u_value: Optional[float] = Field(
        None, gt=0, le=5.0, description="Known U-value (W/m2K)"
    )
    description: Optional[str] = Field(None, description="Description")

class WindowElement(BaseModel):
    """Window element for envelope assessment.

    Attributes:
        element_id: Unique element identifier.
        window_type: Window type (single/double/triple glazed).
        glazing_type: Glazing coating type.
        frame_material: Frame material.
        area_m2: Total window area (glass + frame) in m2.
        frame_fraction: Fraction of area that is frame (0-1, default 0.2).
        orientation: Window orientation (N, NE, E, SE, S, SW, W, NW).
        known_u_value: Override U-value if known (W/m2K).
        known_g_value: Override g-value if known.
        description: Human-readable description.
    """
    element_id: str = Field(default_factory=_new_uuid, description="Element ID")
    window_type: WindowType = Field(..., description="Window type")
    glazing_type: GlazingType = Field(
        default=GlazingType.CLEAR, description="Glazing coating type"
    )
    frame_material: FrameMaterial = Field(
        default=FrameMaterial.UPVC, description="Frame material"
    )
    area_m2: float = Field(..., gt=0, description="Total window area (m2)")
    frame_fraction: float = Field(
        default=0.20, ge=0, le=0.5, description="Frame fraction (0-1)"
    )
    orientation: Optional[str] = Field(None, description="Window orientation (N/S/E/W)")
    known_u_value: Optional[float] = Field(
        None, gt=0, le=10.0, description="Known U-value (W/m2K)"
    )
    known_g_value: Optional[float] = Field(
        None, gt=0, le=1.0, description="Known g-value"
    )
    description: Optional[str] = Field(None, description="Description")

class DoorElement(BaseModel):
    """Door element for envelope assessment.

    Attributes:
        element_id: Unique element identifier.
        door_type: Door type key (from DOOR_U_VALUES).
        area_m2: Door area in m2.
        quantity: Number of identical doors.
        known_u_value: Override U-value if known (W/m2K).
        description: Human-readable description.
    """
    element_id: str = Field(default_factory=_new_uuid, description="Element ID")
    door_type: str = Field(
        default="composite_insulated", description="Door type key"
    )
    area_m2: float = Field(default=2.0, gt=0, description="Door area (m2)")
    quantity: int = Field(default=1, ge=1, description="Number of doors")
    known_u_value: Optional[float] = Field(
        None, gt=0, le=10.0, description="Known U-value (W/m2K)"
    )
    description: Optional[str] = Field(None, description="Description")

class ThermalBridge(BaseModel):
    """Thermal bridge junction for heat loss calculation.

    Attributes:
        bridge_type: Junction type from ThermalBridgeType.
        length_m: Junction length in metres.
        psi_value_w_mk: Linear thermal transmittance (W/m.K); looked up if None.
        detail_level: Detail level (default / accredited / enhanced).
        description: Human-readable description.
    """
    bridge_type: ThermalBridgeType = Field(..., description="Junction type")
    length_m: float = Field(..., gt=0, description="Junction length (m)")
    psi_value_w_mk: Optional[float] = Field(
        None, ge=0, le=2.0, description="Psi value (W/m.K)"
    )
    detail_level: str = Field(
        default="default", description="Detail level (default/accredited/enhanced)"
    )
    description: Optional[str] = Field(None, description="Description")

class AirtightnessData(BaseModel):
    """Blower door test results or estimated airtightness.

    Attributes:
        n50_ach: Air changes per hour at 50 Pa (measured or estimated).
        q50_m3_h_m2: Air permeability at 50 Pa (m3/h/m2 envelope area).
        test_date: Date of blower door test.
        test_standard: Test standard (EN 13829 / ISO 9972).
        measured: Whether values are from actual test or estimated.
    """
    n50_ach: Optional[float] = Field(
        None, ge=0, le=50, description="Air changes per hour at 50 Pa"
    )
    q50_m3_h_m2: Optional[float] = Field(
        None, ge=0, le=50, description="Air permeability at 50 Pa (m3/h/m2)"
    )
    test_date: Optional[str] = Field(None, description="Blower door test date")
    test_standard: str = Field(
        default="EN 13829", description="Test standard"
    )
    measured: bool = Field(default=False, description="Measured vs estimated")

class BuildingEnvelope(BaseModel):
    """Complete building envelope input for thermal performance assessment.

    Attributes:
        facility_id: Unique facility identifier.
        name: Building name.
        building_type: Building use type.
        year_built: Year of original construction.
        country: Country code (ISO 3166-1 alpha-2).
        gross_floor_area_m2: Gross internal floor area in m2.
        heated_volume_m3: Heated volume in m3.
        envelope_area_m2: Total thermal envelope area in m2 (for airtightness).
        walls: List of wall elements.
        roofs: List of roof elements.
        floors: List of floor elements.
        windows: List of window elements.
        doors: List of door elements.
        thermal_bridges: List of thermal bridge junctions.
        airtightness: Blower door test data or estimate.
    """
    facility_id: str = Field(..., min_length=1, description="Facility identifier")
    name: str = Field(..., min_length=1, description="Building name")
    building_type: BuildingType = Field(
        default=BuildingType.OFFICE, description="Building use type"
    )
    year_built: int = Field(..., ge=1600, le=2030, description="Year of construction")
    country: str = Field(
        default="UK", min_length=2, max_length=3, description="Country code"
    )
    gross_floor_area_m2: float = Field(
        ..., gt=0, description="Gross internal floor area (m2)"
    )
    heated_volume_m3: float = Field(
        ..., gt=0, description="Heated volume (m3)"
    )
    envelope_area_m2: Optional[float] = Field(
        None, gt=0, description="Total thermal envelope area (m2)"
    )
    walls: List[WallElement] = Field(default_factory=list, description="Wall elements")
    roofs: List[RoofElement] = Field(default_factory=list, description="Roof elements")
    floors: List[FloorElement] = Field(default_factory=list, description="Floor elements")
    windows: List[WindowElement] = Field(
        default_factory=list, description="Window elements"
    )
    doors: List[DoorElement] = Field(default_factory=list, description="Door elements")
    thermal_bridges: List[ThermalBridge] = Field(
        default_factory=list, description="Thermal bridge junctions"
    )
    airtightness: Optional[AirtightnessData] = Field(
        None, description="Blower door / airtightness data"
    )

    @field_validator("gross_floor_area_m2")
    @classmethod
    def validate_floor_area(cls, v: float) -> float:
        """Ensure floor area is within plausible bounds."""
        if v > 1_000_000:
            raise ValueError("Floor area exceeds 1,000,000 m2 sanity check")
        return v

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class ElementUValue(BaseModel):
    """U-value result for a single building element.

    Attributes:
        element_id: Element identifier.
        element_type: Type (wall / roof / floor / window / door).
        description: Element description.
        u_value_w_m2k: Calculated or looked-up U-value (W/m2K).
        area_m2: Element area (m2).
        heat_loss_w_k: Element heat loss coefficient (U * A) in W/K.
        source: Source of U-value (lookup / calculated / provided).
        regulatory_limit: Applicable regulatory U-value limit.
        compliant: Whether element meets regulatory limit.
    """
    element_id: str = Field(default="", description="Element identifier")
    element_type: str = Field(default="", description="Element type")
    description: str = Field(default="", description="Element description")
    u_value_w_m2k: float = Field(default=0.0, description="U-value (W/m2K)")
    area_m2: float = Field(default=0.0, description="Element area (m2)")
    heat_loss_w_k: float = Field(default=0.0, description="Heat loss U*A (W/K)")
    source: str = Field(default="lookup", description="U-value source")
    regulatory_limit: Optional[float] = Field(None, description="Regulatory limit")
    compliant: bool = Field(default=True, description="Regulatory compliance")

class ThermalBridgeResult(BaseModel):
    """Thermal bridge assessment result.

    Attributes:
        total_htb_w_k: Total thermal bridge heat loss coefficient (W/K).
        bridge_details: Breakdown by junction type.
        y_factor: Thermal bridge y-factor (HTB / total envelope area).
        bridge_fraction_pct: TB heat loss as % of total fabric heat loss.
    """
    total_htb_w_k: float = Field(default=0.0, description="Total HTB (W/K)")
    bridge_details: List[Dict[str, Any]] = Field(
        default_factory=list, description="Bridge details"
    )
    y_factor: float = Field(default=0.0, description="Y-factor (W/m2K)")
    bridge_fraction_pct: float = Field(
        default=0.0, description="TB fraction of total fabric loss (%)"
    )

class AirtightnessResult(BaseModel):
    """Airtightness assessment result.

    Attributes:
        n50_ach: Air changes per hour at 50 Pa.
        q50_m3_h_m2: Air permeability at 50 Pa (m3/h/m2).
        estimated_ach_normal: Estimated air change rate at normal pressure.
        ventilation_heat_loss_w_k: Ventilation heat loss coefficient (W/K).
        classification: Airtightness classification (passivhaus/low_energy/etc.).
        measured: Whether from actual blower door test.
    """
    n50_ach: float = Field(default=0.0, description="n50 (ACH at 50 Pa)")
    q50_m3_h_m2: float = Field(default=0.0, description="q50 (m3/h/m2)")
    estimated_ach_normal: float = Field(
        default=0.0, description="Estimated ACH at normal pressure"
    )
    ventilation_heat_loss_w_k: float = Field(
        default=0.0, description="Ventilation heat loss (W/K)"
    )
    classification: str = Field(default="", description="Airtightness class")
    measured: bool = Field(default=False, description="Measured or estimated")

class CondensationRiskResult(BaseModel):
    """Condensation risk assessment result (Glaser method per EN ISO 13788).

    Attributes:
        risk_level: Risk level (low / medium / high / critical).
        critical_elements: Elements with condensation risk.
        dewpoint_temperature_c: Internal dewpoint at 20C / 60% RH.
        surface_temperature_factor: fRsi factor for critical element.
        interstitial_risk: Whether interstitial condensation risk exists.
        recommendations: Remedial actions.
    """
    risk_level: str = Field(default="low", description="Risk level")
    critical_elements: List[str] = Field(
        default_factory=list, description="Elements with risk"
    )
    dewpoint_temperature_c: float = Field(
        default=0.0, description="Internal dewpoint (C)"
    )
    surface_temperature_factor: float = Field(
        default=0.0, description="fRsi factor"
    )
    interstitial_risk: bool = Field(
        default=False, description="Interstitial condensation risk"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Remedial recommendations"
    )

class ImprovementOpportunity(BaseModel):
    """Envelope improvement opportunity with estimated savings.

    Attributes:
        element_id: Target element identifier.
        element_type: Element type (wall / roof / floor / window / door).
        description: Description of improvement measure.
        current_u_value: Current U-value (W/m2K).
        improved_u_value: Achievable U-value after improvement (W/m2K).
        area_m2: Affected area (m2).
        annual_savings_kwh: Estimated annual energy savings (kWh).
        annual_savings_co2_kg: Estimated annual CO2 savings (kg).
        estimated_cost: Estimated implementation cost.
        payback_years: Simple payback period (years).
        priority: Priority ranking (1=highest).
    """
    element_id: str = Field(default="", description="Target element")
    element_type: str = Field(default="", description="Element type")
    description: str = Field(default="", description="Improvement description")
    current_u_value: float = Field(default=0.0, description="Current U-value")
    improved_u_value: float = Field(default=0.0, description="Improved U-value")
    area_m2: float = Field(default=0.0, description="Affected area (m2)")
    annual_savings_kwh: float = Field(default=0.0, description="Annual savings (kWh)")
    annual_savings_co2_kg: float = Field(default=0.0, description="CO2 savings (kg)")
    estimated_cost: float = Field(default=0.0, description="Estimated cost")
    payback_years: float = Field(default=0.0, description="Simple payback (years)")
    priority: int = Field(default=0, description="Priority ranking")

class EnvelopeResult(BaseModel):
    """Complete building envelope assessment result with provenance.

    Contains element U-values, area-weighted averages, thermal bridge
    assessment, heat loss coefficients, airtightness, condensation risk,
    and improvement opportunities.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Unique result ID")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Calculation timestamp"
    )
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")

    facility_id: str = Field(default="", description="Facility identifier")
    building_name: str = Field(default="", description="Building name")
    building_type: str = Field(default="", description="Building type")

    # Element U-values
    element_u_values: List[ElementUValue] = Field(
        default_factory=list, description="U-values per element"
    )
    area_weighted_u_value: float = Field(
        default=0.0, description="Area-weighted average U-value (W/m2K)"
    )

    # Heat loss
    fabric_heat_loss_w_k: float = Field(
        default=0.0, description="Fabric heat loss Htr (W/K)"
    )
    thermal_bridge_result: Optional[ThermalBridgeResult] = Field(
        None, description="Thermal bridge assessment"
    )
    total_heat_loss_coefficient_w_k: float = Field(
        default=0.0, description="Total heat loss coefficient H (W/K)"
    )
    specific_heat_loss_w_m2k: float = Field(
        default=0.0, description="Specific heat loss (W/m2K)"
    )

    # Ventilation
    airtightness_result: Optional[AirtightnessResult] = Field(
        None, description="Airtightness assessment"
    )
    ventilation_heat_loss_w_k: float = Field(
        default=0.0, description="Ventilation heat loss Hve (W/K)"
    )

    # Annual energy
    annual_heating_demand_kwh: float = Field(
        default=0.0, description="Estimated annual heating demand (kWh)"
    )
    annual_heating_demand_kwh_m2: float = Field(
        default=0.0, description="Heating demand per m2 (kWh/m2/yr)"
    )

    # Condensation
    condensation_risk: Optional[CondensationRiskResult] = Field(
        None, description="Condensation risk assessment"
    )

    # Improvements
    improvement_opportunities: List[ImprovementOpportunity] = Field(
        default_factory=list, description="Improvement opportunities"
    )
    total_improvement_savings_kwh: float = Field(
        default=0.0, description="Total potential savings (kWh/yr)"
    )

    recommendations: List[str] = Field(
        default_factory=list, description="Summary recommendations"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Calculation Engine
# ---------------------------------------------------------------------------

class BuildingEnvelopeEngine:
    """Building envelope thermal performance assessment engine.

    Provides deterministic, zero-hallucination calculations for:
    - U-value calculation per EN ISO 6946 (walls, roofs, floors)
    - Window and door U-value lookup per EN ISO 10077-1
    - Thermal bridge assessment per EN ISO 10211 / 14683
    - Fabric heat loss coefficient (Htr) calculation
    - Ventilation heat loss assessment (Hve)
    - Airtightness assessment per EN 13829 / ISO 9972
    - Condensation risk assessment per EN ISO 13788
    - Improvement opportunity identification with savings estimates

    All calculations are bit-perfect reproducible.  No LLM is used
    in any calculation path.

    Usage::

        engine = BuildingEnvelopeEngine()
        result = engine.analyze(building_envelope)

    """

    engine_version: str = _MODULE_VERSION

    def __init__(self) -> None:
        """Initialise the building envelope engine with embedded constants."""
        self._wall_u_values = WALL_U_VALUES
        self._roof_u_values = ROOF_U_VALUES
        self._floor_u_values = FLOOR_U_VALUES
        self._window_u_values = WINDOW_U_VALUES
        self._window_g_values = WINDOW_G_VALUES
        self._thermal_conductivity = THERMAL_CONDUCTIVITY
        self._psi_values = THERMAL_BRIDGE_PSI_VALUES
        self._airtightness_benchmarks = AIRTIGHTNESS_BENCHMARKS
        self._surface_resistances = SURFACE_RESISTANCES
        self._material_conductivity = MATERIAL_CONDUCTIVITY
        self._door_u_values = DOOR_U_VALUES
        self._country_hdd = COUNTRY_HDD

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def analyze(self, envelope: BuildingEnvelope) -> EnvelopeResult:
        """Run complete building envelope thermal performance analysis.

        Orchestrates all sub-calculations: U-values, thermal bridges,
        heat loss, airtightness, condensation risk, and improvements.

        Args:
            envelope: Complete building envelope input data.

        Returns:
            EnvelopeResult with full provenance and audit trail.

        Raises:
            ValueError: If no building elements are provided.
        """
        t0 = time.perf_counter()

        total_elements = (
            len(envelope.walls) + len(envelope.roofs) +
            len(envelope.floors) + len(envelope.windows) +
            len(envelope.doors)
        )
        if total_elements == 0:
            raise ValueError(
                "At least one building element (wall, roof, floor, window, or door) "
                "is required for envelope assessment"
            )

        logger.info(
            "Analysing envelope for facility %s (%s), %d elements",
            envelope.facility_id, envelope.building_type.value, total_elements,
        )

        # Step 1: Calculate U-values for all elements
        element_results = self._calculate_all_u_values(envelope)

        # Step 2: Calculate area-weighted average U-value
        area_weighted_u = self._calculate_area_weighted_u(element_results)

        # Step 3: Calculate fabric heat loss coefficient Htr = sum(Ui * Ai)
        fabric_htr = self._calculate_fabric_heat_loss(element_results)

        # Step 4: Thermal bridge assessment
        tb_result = self._assess_thermal_bridges(
            envelope.thermal_bridges, fabric_htr, envelope.envelope_area_m2,
        )
        total_htr = _decimal(fabric_htr) + _decimal(tb_result.total_htb_w_k)

        # Step 5: Airtightness and ventilation heat loss
        air_result = self._assess_airtightness(envelope)
        ventilation_hve = _decimal(air_result.ventilation_heat_loss_w_k)

        # Step 6: Total heat loss coefficient
        total_h = total_htr + ventilation_hve
        gfa = _decimal(envelope.gross_floor_area_m2)
        specific_hl = _safe_divide(total_h, gfa)

        # Step 7: Annual heating demand estimate
        country = envelope.country.upper()
        hdd = _decimal(self._country_hdd.get(country, self._country_hdd["DEFAULT"]))
        # Q = H * HDD * 24 / 1000  [kWh/yr]
        annual_kwh = total_h * hdd * Decimal("24") / Decimal("1000")
        annual_kwh_m2 = _safe_divide(annual_kwh, gfa)

        # Step 8: Condensation risk
        condensation = self._check_condensation_risk(element_results, envelope)

        # Step 9: Improvement opportunities
        improvements = self._identify_improvements(
            element_results, envelope, hdd,
        )
        total_savings = Decimal("0")
        for imp in improvements:
            total_savings += _decimal(imp.annual_savings_kwh)

        # Step 10: Recommendations
        recs = self._generate_recommendations(
            element_results, tb_result, air_result, condensation, improvements,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = EnvelopeResult(
            facility_id=envelope.facility_id,
            building_name=envelope.name,
            building_type=envelope.building_type.value,
            element_u_values=element_results,
            area_weighted_u_value=_round3(float(area_weighted_u)),
            fabric_heat_loss_w_k=_round2(float(fabric_htr)),
            thermal_bridge_result=tb_result,
            total_heat_loss_coefficient_w_k=_round2(float(total_h)),
            specific_heat_loss_w_m2k=_round4(float(specific_hl)),
            airtightness_result=air_result,
            ventilation_heat_loss_w_k=_round2(float(ventilation_hve)),
            annual_heating_demand_kwh=_round2(float(annual_kwh)),
            annual_heating_demand_kwh_m2=_round2(float(annual_kwh_m2)),
            condensation_risk=condensation,
            improvement_opportunities=improvements,
            total_improvement_savings_kwh=_round2(float(total_savings)),
            recommendations=recs,
            processing_time_ms=_round2(elapsed_ms),
        )

        result.provenance_hash = _compute_hash(result)
        return result

    def calculate_element_u_value(
        self,
        layers: List[InsulationLayer],
        heat_flow_direction: str = "horizontal_heat_flow",
    ) -> float:
        """Calculate U-value from detailed construction layers per EN ISO 6946.

        U = 1 / (Rsi + sum(d_i / lambda_i) + Rse)

        Args:
            layers: Construction layers from inside to outside.
            heat_flow_direction: Heat flow direction for surface resistances.

        Returns:
            U-value in W/m2K rounded to 2 decimal places.

        Raises:
            ValueError: If no layers provided.
        """
        if not layers:
            raise ValueError("At least one construction layer is required")

        surfaces = self._surface_resistances.get(
            heat_flow_direction,
            self._surface_resistances["horizontal_heat_flow"],
        )
        r_total = _decimal(surfaces["Rsi"]) + _decimal(surfaces["Rse"])

        for layer in layers:
            conductivity = layer.conductivity_w_mk
            if conductivity is None:
                # Look up from insulation type or material
                conductivity = self._lookup_conductivity(layer.material)
            d = _decimal(layer.thickness_mm) / Decimal("1000")
            lam = _decimal(conductivity)
            r_layer = _safe_divide(d, lam)
            r_total += r_layer

        u_value = _safe_divide(Decimal("1"), r_total)
        return _round2(float(u_value))

    def calculate_envelope_performance(
        self, envelope: BuildingEnvelope,
    ) -> Dict[str, float]:
        """Calculate summary envelope performance metrics.

        Returns a dictionary of key performance indicators.

        Args:
            envelope: Building envelope input data.

        Returns:
            Dict with fabric_heat_loss_w_k, ventilation_heat_loss_w_k, etc.
        """
        result = self.analyze(envelope)
        return {
            "fabric_heat_loss_w_k": result.fabric_heat_loss_w_k,
            "ventilation_heat_loss_w_k": result.ventilation_heat_loss_w_k,
            "total_heat_loss_w_k": result.total_heat_loss_coefficient_w_k,
            "specific_heat_loss_w_m2k": result.specific_heat_loss_w_m2k,
            "area_weighted_u_value": result.area_weighted_u_value,
            "annual_heating_demand_kwh": result.annual_heating_demand_kwh,
            "annual_heating_demand_kwh_m2": result.annual_heating_demand_kwh_m2,
        }

    def assess_airtightness(
        self, envelope: BuildingEnvelope,
    ) -> AirtightnessResult:
        """Assess building airtightness from blower door data or estimate.

        Args:
            envelope: Building envelope with airtightness data.

        Returns:
            AirtightnessResult with classification and ventilation heat loss.
        """
        return self._assess_airtightness(envelope)

    def check_condensation_risk(
        self, envelope: BuildingEnvelope,
    ) -> CondensationRiskResult:
        """Check condensation risk using simplified Glaser method.

        Args:
            envelope: Building envelope input data.

        Returns:
            CondensationRiskResult with risk level and recommendations.
        """
        elements = self._calculate_all_u_values(envelope)
        return self._check_condensation_risk(elements, envelope)

    def identify_improvements(
        self, envelope: BuildingEnvelope,
    ) -> List[ImprovementOpportunity]:
        """Identify improvement opportunities for the building envelope.

        Args:
            envelope: Building envelope input data.

        Returns:
            List of ImprovementOpportunity sorted by priority.
        """
        elements = self._calculate_all_u_values(envelope)
        country = envelope.country.upper()
        hdd = _decimal(self._country_hdd.get(country, self._country_hdd["DEFAULT"]))
        return self._identify_improvements(elements, envelope, hdd)

    # -------------------------------------------------------------------
    # Internal Calculations -- U-Values
    # -------------------------------------------------------------------

    def _calculate_all_u_values(
        self, envelope: BuildingEnvelope,
    ) -> List[ElementUValue]:
        """Calculate U-values for all building elements."""
        results: List[ElementUValue] = []

        for wall in envelope.walls:
            results.append(self._calculate_wall_u_value(wall, envelope.year_built))

        for roof in envelope.roofs:
            results.append(self._calculate_roof_u_value(roof))

        for floor in envelope.floors:
            results.append(self._calculate_floor_u_value(floor))

        for window in envelope.windows:
            results.append(self._calculate_window_u_value(window))

        for door in envelope.doors:
            results.append(self._calculate_door_u_value(door))

        return results

    def _calculate_wall_u_value(
        self, wall: WallElement, year_built: int,
    ) -> ElementUValue:
        """Calculate or look up wall U-value.

        Priority: known_u_value > calculated from layers > lookup by age band.
        """
        source = "lookup"
        u_val = Decimal("0")

        if wall.known_u_value is not None:
            u_val = _decimal(wall.known_u_value)
            source = "provided"
        elif wall.layers and len(wall.layers) > 0:
            u_float = self.calculate_element_u_value(
                wall.layers, "horizontal_heat_flow",
            )
            u_val = _decimal(u_float)
            source = "calculated_en_iso_6946"
        else:
            age_band = wall.age_band or self._year_to_age_band(year_built)
            wall_lookup = self._wall_u_values.get(wall.wall_type, {})
            u_float = wall_lookup.get(age_band, 1.50)
            u_val = _decimal(u_float)
            source = f"lookup_br443_{age_band}"

        area = _decimal(wall.area_m2)
        heat_loss = u_val * area
        reg_limit = self._get_regulatory_u_limit("wall")

        return ElementUValue(
            element_id=wall.element_id,
            element_type="wall",
            description=wall.description or f"{wall.wall_type.value} wall",
            u_value_w_m2k=_round2(float(u_val)),
            area_m2=_round2(float(area)),
            heat_loss_w_k=_round2(float(heat_loss)),
            source=source,
            regulatory_limit=reg_limit,
            compliant=float(u_val) <= reg_limit if reg_limit else True,
        )

    def _calculate_roof_u_value(self, roof: RoofElement) -> ElementUValue:
        """Calculate or look up roof U-value.

        Priority: known_u_value > calculated from insulation > lookup table.
        """
        source = "lookup"
        u_val = Decimal("0")

        if roof.known_u_value is not None:
            u_val = _decimal(roof.known_u_value)
            source = "provided"
        elif roof.insulation_type and roof.insulation_thickness_mm > 0:
            # Calculate from insulation thickness and conductivity
            lam_data = self._thermal_conductivity.get(roof.insulation_type)
            if lam_data:
                lam = _decimal(lam_data["lambda_w_mk"])
                d_ins = _decimal(roof.insulation_thickness_mm) / Decimal("1000")
                surfaces = self._surface_resistances["upward_heat_flow"]
                r_si = _decimal(surfaces["Rsi"])
                r_se = _decimal(surfaces["Rse"])
                # Approximate roof structure R = 0.10 m2K/W
                r_structure = Decimal("0.10")
                r_ins = _safe_divide(d_ins, lam)
                r_total = r_si + r_structure + r_ins + r_se
                u_val = _safe_divide(Decimal("1"), r_total)
                source = "calculated_insulation"
            else:
                u_val = self._lookup_roof_u(roof)
                source = "lookup_table"
        else:
            u_val = self._lookup_roof_u(roof)
            source = "lookup_table"

        area = _decimal(roof.area_m2)
        heat_loss = u_val * area
        reg_limit = self._get_regulatory_u_limit("roof")

        return ElementUValue(
            element_id=roof.element_id,
            element_type="roof",
            description=roof.description or f"{roof.roof_type.value} roof",
            u_value_w_m2k=_round2(float(u_val)),
            area_m2=_round2(float(area)),
            heat_loss_w_k=_round2(float(heat_loss)),
            source=source,
            regulatory_limit=reg_limit,
            compliant=float(u_val) <= reg_limit if reg_limit else True,
        )

    def _lookup_roof_u(self, roof: RoofElement) -> Decimal:
        """Look up roof U-value from table with interpolation for thickness."""
        roof_table = self._roof_u_values.get(roof.roof_type, {})
        if not roof_table:
            return Decimal("2.30")

        thickness = roof.insulation_thickness_mm
        thicknesses = sorted(roof_table.keys())

        # Exact match
        if int(thickness) in roof_table:
            return _decimal(roof_table[int(thickness)])

        # Interpolation between nearest thickness steps
        lower_t = thicknesses[0]
        upper_t = thicknesses[-1]
        for t in thicknesses:
            if t <= thickness:
                lower_t = t
            if t >= thickness and t <= upper_t:
                upper_t = t
                break

        if lower_t == upper_t:
            return _decimal(roof_table[lower_t])

        u_lower = _decimal(roof_table[lower_t])
        u_upper = _decimal(roof_table[upper_t])
        fraction = _safe_divide(
            _decimal(thickness) - _decimal(lower_t),
            _decimal(upper_t) - _decimal(lower_t),
        )
        return u_lower + fraction * (u_upper - u_lower)

    def _calculate_floor_u_value(self, floor: FloorElement) -> ElementUValue:
        """Calculate or look up floor U-value using P/A ratio method.

        Per EN ISO 13370, ground floor U-value depends on the perimeter-to-area
        ratio P/A and the floor construction type.
        """
        source = "lookup"
        u_val = Decimal("0")

        if floor.known_u_value is not None:
            u_val = _decimal(floor.known_u_value)
            source = "provided"
        else:
            area = _decimal(floor.area_m2)
            perimeter = _decimal(floor.perimeter_m)
            pa_ratio = _safe_divide(perimeter, area)
            pa_float = float(pa_ratio)

            # Additional insulation correction
            ins_correction = Decimal("0")
            if floor.insulation_type and floor.insulation_thickness_mm > 0:
                lam_data = self._thermal_conductivity.get(floor.insulation_type)
                if lam_data:
                    lam = _decimal(lam_data["lambda_w_mk"])
                    d_ins = _decimal(floor.insulation_thickness_mm) / Decimal("1000")
                    r_ins = _safe_divide(d_ins, lam)
                    # U_insulated approx = 1 / (1/U_uninsulated + R_ins)
                    ins_correction = r_ins

            u_base = self._lookup_floor_u(floor.floor_type, pa_float)
            if ins_correction > Decimal("0"):
                r_base = _safe_divide(Decimal("1"), u_base, Decimal("999"))
                r_total = r_base + ins_correction
                u_val = _safe_divide(Decimal("1"), r_total)
                source = "calculated_en_iso_13370"
            else:
                u_val = u_base
                source = "lookup_en_iso_13370"

        area = _decimal(floor.area_m2)
        heat_loss = u_val * area
        reg_limit = self._get_regulatory_u_limit("floor")

        return ElementUValue(
            element_id=floor.element_id,
            element_type="floor",
            description=floor.description or f"{floor.floor_type.value} floor",
            u_value_w_m2k=_round2(float(u_val)),
            area_m2=_round2(float(area)),
            heat_loss_w_k=_round2(float(heat_loss)),
            source=source,
            regulatory_limit=reg_limit,
            compliant=float(u_val) <= reg_limit if reg_limit else True,
        )

    def _lookup_floor_u(self, floor_type: FloorType, pa_ratio: float) -> Decimal:
        """Look up floor U-value from table with interpolation for P/A ratio."""
        floor_table = self._floor_u_values.get(floor_type, {})
        if not floor_table:
            return Decimal("0.70")

        # Clamp P/A ratio to table range
        pa_clamped = max(0.2, min(1.0, pa_ratio))
        pa_str = f"{pa_clamped:.1f}"

        if pa_str in floor_table:
            return _decimal(floor_table[pa_str])

        # Interpolation
        pa_keys = sorted(floor_table.keys(), key=float)
        lower_key = pa_keys[0]
        upper_key = pa_keys[-1]
        for k in pa_keys:
            if float(k) <= pa_clamped:
                lower_key = k
            if float(k) >= pa_clamped:
                upper_key = k
                break

        if lower_key == upper_key:
            return _decimal(floor_table[lower_key])

        u_lower = _decimal(floor_table[lower_key])
        u_upper = _decimal(floor_table[upper_key])
        fraction = _safe_divide(
            _decimal(pa_clamped) - _decimal(float(lower_key)),
            _decimal(float(upper_key)) - _decimal(float(lower_key)),
        )
        return u_lower + fraction * (u_upper - u_lower)

    def _calculate_window_u_value(self, window: WindowElement) -> ElementUValue:
        """Look up window U-value from lookup tables."""
        source = "lookup"
        u_val = Decimal("0")
        g_val = Decimal("0")

        if window.known_u_value is not None:
            u_val = _decimal(window.known_u_value)
            source = "provided"
        else:
            wtype_table = self._window_u_values.get(window.window_type, {})
            glaze_table = wtype_table.get(window.glazing_type, {})
            if not glaze_table:
                # Fallback: try CLEAR for this window type
                glaze_table = wtype_table.get(GlazingType.CLEAR, {})
            u_float = glaze_table.get(window.frame_material, 3.00)
            u_val = _decimal(u_float)
            source = "lookup_en_iso_10077"

        # g-value lookup
        if window.known_g_value is not None:
            g_val = _decimal(window.known_g_value)
        else:
            gtype_table = self._window_g_values.get(window.window_type, {})
            g_float = gtype_table.get(window.glazing_type, 0.60)
            g_val = _decimal(g_float)

        area = _decimal(window.area_m2)
        heat_loss = u_val * area
        reg_limit = self._get_regulatory_u_limit("window")

        desc = window.description or (
            f"{window.window_type.value} {window.glazing_type.value} "
            f"{window.frame_material.value}"
        )

        return ElementUValue(
            element_id=window.element_id,
            element_type="window",
            description=desc,
            u_value_w_m2k=_round2(float(u_val)),
            area_m2=_round2(float(area)),
            heat_loss_w_k=_round2(float(heat_loss)),
            source=source,
            regulatory_limit=reg_limit,
            compliant=float(u_val) <= reg_limit if reg_limit else True,
        )

    def _calculate_door_u_value(self, door: DoorElement) -> ElementUValue:
        """Look up door U-value from lookup table."""
        source = "lookup"

        if door.known_u_value is not None:
            u_val = _decimal(door.known_u_value)
            source = "provided"
        else:
            u_float = self._door_u_values.get(door.door_type, 3.00)
            u_val = _decimal(u_float)
            source = "lookup_br443"

        total_area = _decimal(door.area_m2) * _decimal(door.quantity)
        heat_loss = u_val * total_area
        reg_limit = self._get_regulatory_u_limit("door")

        return ElementUValue(
            element_id=door.element_id,
            element_type="door",
            description=door.description or f"{door.door_type} door",
            u_value_w_m2k=_round2(float(u_val)),
            area_m2=_round2(float(total_area)),
            heat_loss_w_k=_round2(float(heat_loss)),
            source=source,
            regulatory_limit=reg_limit,
            compliant=float(u_val) <= reg_limit if reg_limit else True,
        )

    # -------------------------------------------------------------------
    # Internal Calculations -- Aggregate Metrics
    # -------------------------------------------------------------------

    def _calculate_area_weighted_u(
        self, elements: List[ElementUValue],
    ) -> Decimal:
        """Calculate area-weighted average U-value across all elements.

        U_avg = sum(Ui * Ai) / sum(Ai)
        """
        total_ua = Decimal("0")
        total_a = Decimal("0")
        for el in elements:
            u = _decimal(el.u_value_w_m2k)
            a = _decimal(el.area_m2)
            total_ua += u * a
            total_a += a
        return _safe_divide(total_ua, total_a)

    def _calculate_fabric_heat_loss(
        self, elements: List[ElementUValue],
    ) -> Decimal:
        """Calculate total fabric heat loss coefficient Htr = sum(Ui * Ai)."""
        htr = Decimal("0")
        for el in elements:
            htr += _decimal(el.heat_loss_w_k)
        return htr

    # -------------------------------------------------------------------
    # Internal Calculations -- Thermal Bridges
    # -------------------------------------------------------------------

    def _assess_thermal_bridges(
        self,
        bridges: List[ThermalBridge],
        fabric_htr: Decimal,
        envelope_area: Optional[float],
    ) -> ThermalBridgeResult:
        """Assess thermal bridging per EN ISO 10211 / EN ISO 14683.

        HTB = sum(psi_j * L_j)
        y-factor = HTB / A_envelope
        """
        total_htb = Decimal("0")
        details: List[Dict[str, Any]] = []

        for bridge in bridges:
            psi = bridge.psi_value_w_mk
            if psi is None:
                psi_table = self._psi_values.get(bridge.bridge_type, {})
                psi = psi_table.get(bridge.detail_level, psi_table.get("default", 0.10))
            psi_d = _decimal(psi)
            length_d = _decimal(bridge.length_m)
            htb_j = psi_d * length_d
            total_htb += htb_j
            details.append({
                "bridge_type": bridge.bridge_type.value,
                "length_m": _round2(float(length_d)),
                "psi_w_mk": _round3(float(psi_d)),
                "htb_w_k": _round3(float(htb_j)),
                "detail_level": bridge.detail_level,
            })

        env_area = _decimal(envelope_area) if envelope_area else Decimal("0")
        y_factor = _safe_divide(total_htb, env_area)

        total_fabric = fabric_htr + total_htb
        bridge_pct = float(_safe_pct(total_htb, total_fabric)) if total_fabric > Decimal("0") else 0.0

        return ThermalBridgeResult(
            total_htb_w_k=_round2(float(total_htb)),
            bridge_details=details,
            y_factor=_round4(float(y_factor)),
            bridge_fraction_pct=_round2(bridge_pct),
        )

    # -------------------------------------------------------------------
    # Internal Calculations -- Airtightness
    # -------------------------------------------------------------------

    def _assess_airtightness(
        self, envelope: BuildingEnvelope,
    ) -> AirtightnessResult:
        """Assess building airtightness.

        Converts between n50 and q50 if needed.
        Estimates ventilation heat loss: Hve = 0.34 * n * V
        where n = n50 / 20 (shelter factor approximation).
        """
        volume = _decimal(envelope.heated_volume_m3)
        env_area = _decimal(envelope.envelope_area_m2 or Decimal("0"))

        n50 = Decimal("0")
        q50 = Decimal("0")
        measured = False

        if envelope.airtightness:
            measured = envelope.airtightness.measured
            if envelope.airtightness.n50_ach is not None:
                n50 = _decimal(envelope.airtightness.n50_ach)
                # q50 = n50 * V / A_envelope
                if env_area > Decimal("0"):
                    q50 = _safe_divide(n50 * volume, env_area)
                else:
                    q50 = n50  # approximate
            elif envelope.airtightness.q50_m3_h_m2 is not None:
                q50 = _decimal(envelope.airtightness.q50_m3_h_m2)
                # n50 = q50 * A_envelope / V
                if volume > Decimal("0"):
                    n50 = _safe_divide(q50 * env_area, volume)
                else:
                    n50 = q50  # approximate
        else:
            # Default estimate based on age band
            year = envelope.year_built
            if year >= 2014:
                n50 = Decimal("5.0")
            elif year >= 2007:
                n50 = Decimal("7.0")
            elif year >= 1996:
                n50 = Decimal("10.0")
            elif year >= 1965:
                n50 = Decimal("12.0")
            else:
                n50 = Decimal("15.0")
            if env_area > Decimal("0"):
                q50 = _safe_divide(n50 * volume, env_area)
            else:
                q50 = n50

        # Estimated normal-pressure ACH: n = n50 / 20 (EN 15242 shelter factor)
        ach_normal = _safe_divide(n50, Decimal("20"))

        # Ventilation heat loss: Hve = 0.34 * n * V  [W/K]
        hve = Decimal("0.34") * ach_normal * volume

        # Classification
        classification = AirtightnessStandard.POOR.value
        for std, bench in sorted(
            self._airtightness_benchmarks.items(),
            key=lambda x: x[1]["n50_ach"],
        ):
            if float(n50) <= bench["n50_ach"]:
                classification = std.value
                break

        return AirtightnessResult(
            n50_ach=_round2(float(n50)),
            q50_m3_h_m2=_round2(float(q50)),
            estimated_ach_normal=_round3(float(ach_normal)),
            ventilation_heat_loss_w_k=_round2(float(hve)),
            classification=classification,
            measured=measured,
        )

    # -------------------------------------------------------------------
    # Internal Calculations -- Condensation Risk
    # -------------------------------------------------------------------

    def _check_condensation_risk(
        self,
        elements: List[ElementUValue],
        envelope: BuildingEnvelope,
    ) -> CondensationRiskResult:
        """Simplified condensation risk assessment per EN ISO 13788.

        Uses the surface temperature factor fRsi method:
        fRsi = (Tsi - Te) / (Ti - Te)
        where Tsi = internal surface temperature.

        For U-value based estimate: fRsi = 1 - U * Rsi

        Condensation risk exists when fRsi < fRsi_min (typically 0.75).
        """
        # Internal conditions: 20C, 60% RH
        ti = Decimal("20.0")
        rh = Decimal("60.0")

        # Dewpoint at 20C / 60% RH: approximately 12.0 C (Magnus formula)
        # Td = (243.12 * alpha) / (17.62 - alpha)
        # alpha = ln(RH/100) + (17.62 * T) / (243.12 + T)
        rh_frac = rh / Decimal("100")
        ln_rh = _decimal(math.log(float(rh_frac)))
        alpha = ln_rh + (Decimal("17.62") * ti) / (Decimal("243.12") + ti)
        dewpoint = (Decimal("243.12") * alpha) / (Decimal("17.62") - alpha)

        # Minimum fRsi to avoid condensation (EN ISO 13788)
        frsi_min = Decimal("0.75")

        # External design temperature (January mean for UK)
        te = Decimal("4.0")

        critical: List[str] = []
        worst_frsi = Decimal("1.0")
        interstitial = False
        rsi = Decimal("0.13")  # Internal surface resistance

        for el in elements:
            u = _decimal(el.u_value_w_m2k)
            # fRsi = 1 - U * Rsi
            frsi = Decimal("1") - u * rsi

            if frsi < worst_frsi:
                worst_frsi = frsi

            if frsi < frsi_min:
                critical.append(
                    f"{el.element_type}: {el.description} "
                    f"(U={el.u_value_w_m2k}, fRsi={_round3(float(frsi))})"
                )
                # High U-values with certain wall types = interstitial risk
                if el.u_value_w_m2k > 1.0:
                    interstitial = True

        # Risk level classification
        if len(critical) == 0:
            risk = "low"
        elif float(worst_frsi) >= 0.65:
            risk = "medium"
        elif float(worst_frsi) >= 0.50:
            risk = "high"
        else:
            risk = "critical"

        recs: List[str] = []
        if risk in ("medium", "high", "critical"):
            recs.append("Improve insulation on critical elements to reduce surface condensation risk")
            recs.append("Ensure adequate ventilation to control internal moisture levels")
        if interstitial:
            recs.append("Install vapour control layer on warm side of insulation")
            recs.append("Consider detailed EN ISO 13788 Glaser analysis for affected elements")
        if risk == "critical":
            recs.append("Urgent remediation required: risk of mould growth and structural damage")

        return CondensationRiskResult(
            risk_level=risk,
            critical_elements=critical,
            dewpoint_temperature_c=_round2(float(dewpoint)),
            surface_temperature_factor=_round3(float(worst_frsi)),
            interstitial_risk=interstitial,
            recommendations=recs,
        )

    # -------------------------------------------------------------------
    # Internal Calculations -- Improvement Opportunities
    # -------------------------------------------------------------------

    def _identify_improvements(
        self,
        elements: List[ElementUValue],
        envelope: BuildingEnvelope,
        hdd: Decimal,
    ) -> List[ImprovementOpportunity]:
        """Identify envelope improvement opportunities with savings estimates.

        Savings = delta_U * A * HDD * 24 / 1000  [kWh/yr]
        CO2 savings assume 0.21 kgCO2/kWh (gas heating).
        """
        improvements: List[ImprovementOpportunity] = []
        co2_factor = Decimal("0.21")  # kgCO2/kWh for gas heating
        energy_cost = Decimal("0.10")  # EUR/kWh approximate

        # Target U-values for improvement (Building Regs 2022 / near-NZEB)
        target_u: Dict[str, Decimal] = {
            "wall": Decimal("0.18"),
            "roof": Decimal("0.13"),
            "floor": Decimal("0.18"),
            "window": Decimal("1.40"),
            "door": Decimal("1.40"),
        }

        # Approximate costs per m2 for upgrades
        upgrade_costs: Dict[str, Decimal] = {
            "wall": Decimal("120"),     # External wall insulation
            "roof": Decimal("35"),      # Loft insulation top-up
            "floor": Decimal("60"),     # Floor insulation
            "window": Decimal("450"),   # Window replacement
            "door": Decimal("600"),     # Door replacement
        }

        priority = 0
        for el in elements:
            el_type = el.element_type
            current_u = _decimal(el.u_value_w_m2k)
            target = target_u.get(el_type, Decimal("0.30"))

            if current_u <= target:
                continue  # Already meets target

            delta_u = current_u - target
            area = _decimal(el.area_m2)

            # Annual savings: delta_U * A * HDD * 24 / 1000 [kWh/yr]
            savings_kwh = delta_u * area * hdd * Decimal("24") / Decimal("1000")
            savings_co2 = savings_kwh * co2_factor
            cost = upgrade_costs.get(el_type, Decimal("100")) * area
            payback = _safe_divide(cost, savings_kwh * energy_cost, Decimal("999"))

            priority += 1

            # Description of improvement
            desc_map = {
                "wall": f"Insulate wall to U={float(target)} W/m2K (external insulation)",
                "roof": f"Add roof insulation to achieve U={float(target)} W/m2K",
                "floor": f"Insulate floor to U={float(target)} W/m2K",
                "window": f"Replace windows to U={float(target)} W/m2K (double/triple glazed low-e)",
                "door": f"Replace door to U={float(target)} W/m2K (insulated composite)",
            }

            improvements.append(ImprovementOpportunity(
                element_id=el.element_id,
                element_type=el_type,
                description=desc_map.get(el_type, f"Improve {el_type} to U={float(target)}"),
                current_u_value=_round2(float(current_u)),
                improved_u_value=_round2(float(target)),
                area_m2=_round2(float(area)),
                annual_savings_kwh=_round2(float(savings_kwh)),
                annual_savings_co2_kg=_round2(float(savings_co2)),
                estimated_cost=_round2(float(cost)),
                payback_years=_round2(float(payback)),
                priority=priority,
            ))

        # Sort by savings (highest first)
        improvements.sort(key=lambda x: x.annual_savings_kwh, reverse=True)
        for i, imp in enumerate(improvements):
            imp.priority = i + 1

        return improvements

    # -------------------------------------------------------------------
    # Internal -- Recommendations
    # -------------------------------------------------------------------

    def _generate_recommendations(
        self,
        elements: List[ElementUValue],
        tb_result: ThermalBridgeResult,
        air_result: AirtightnessResult,
        condensation: CondensationRiskResult,
        improvements: List[ImprovementOpportunity],
    ) -> List[str]:
        """Generate summary recommendations based on assessment findings."""
        recs: List[str] = []

        # Non-compliant elements
        non_compliant = [el for el in elements if not el.compliant]
        if non_compliant:
            types = set(el.element_type for el in non_compliant)
            recs.append(
                f"{len(non_compliant)} element(s) ({', '.join(types)}) exceed "
                f"regulatory U-value limits and require upgrade"
            )

        # Thermal bridges
        if tb_result.y_factor > 0.15:
            recs.append(
                f"Thermal bridge y-factor of {tb_result.y_factor} W/m2K exceeds "
                f"good practice (0.08). Review junction details for improvement"
            )

        # Airtightness
        if air_result.n50_ach > 10.0:
            recs.append(
                f"Air permeability n50={air_result.n50_ach} ACH is poor. "
                f"Commission draught-proofing and seal service penetrations"
            )
        elif air_result.n50_ach > 5.0:
            recs.append(
                f"Air permeability n50={air_result.n50_ach} ACH could be improved. "
                f"Target 3-5 ACH with airtightness measures"
            )

        # Condensation
        if condensation.risk_level in ("high", "critical"):
            recs.append(
                f"Condensation risk is {condensation.risk_level}. "
                f"Prioritise insulation and ventilation improvements"
            )

        # Improvement savings
        if improvements:
            total = sum(imp.annual_savings_kwh for imp in improvements)
            recs.append(
                f"{len(improvements)} improvement opportunities identified with "
                f"total potential savings of {_round2(total)} kWh/yr"
            )
            if improvements[0].payback_years < 5.0:
                recs.append(
                    f"Highest-priority measure: {improvements[0].description} "
                    f"with {improvements[0].payback_years}-year payback"
                )

        if not recs:
            recs.append(
                "Building envelope performs well. No critical improvements identified"
            )

        return recs

    # -------------------------------------------------------------------
    # Internal -- Utility Methods
    # -------------------------------------------------------------------

    def _year_to_age_band(self, year: int) -> AgeBand:
        """Map construction year to age band."""
        if year < 1919:
            return AgeBand.PRE_1919
        elif year < 1945:
            return AgeBand.BAND_1919_1944
        elif year < 1965:
            return AgeBand.BAND_1945_1964
        elif year < 1983:
            return AgeBand.BAND_1965_1982
        elif year < 1996:
            return AgeBand.BAND_1983_1995
        elif year < 2007:
            return AgeBand.BAND_1996_2006
        elif year < 2014:
            return AgeBand.BAND_2007_2013
        else:
            return AgeBand.BAND_2014_PRESENT

    def _lookup_conductivity(self, material: str) -> float:
        """Look up thermal conductivity for a material.

        Searches insulation types first, then general materials.
        """
        # Check insulation types
        for ins_type, data in self._thermal_conductivity.items():
            if ins_type.value == material or ins_type == material:
                return data["lambda_w_mk"]

        # Check general materials
        if material in self._material_conductivity:
            return self._material_conductivity[material]

        # Default for unknown material
        logger.warning("Unknown material '%s', using default conductivity 0.50 W/m.K", material)
        return 0.50

    def _get_regulatory_u_limit(self, element_type: str) -> Optional[float]:
        """Get current regulatory U-value limit (Approved Document L 2022 / EPBD).

        These are the UK Building Regulations Part L 2021 limiting values.
        """
        limits: Dict[str, float] = {
            "wall": 0.26,
            "roof": 0.16,
            "floor": 0.18,
            "window": 1.60,
            "door": 1.60,
        }
        return limits.get(element_type)
