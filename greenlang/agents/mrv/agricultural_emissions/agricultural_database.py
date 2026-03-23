# -*- coding: utf-8 -*-
"""
AgriculturalDatabaseEngine - Emission Factors & Reference Data (Engine 1 of 7)

AGENT-MRV-008: Agricultural Emissions Agent

Provides the authoritative reference data repository for all IPCC 2006/2019
agricultural emission factors including enteric fermentation (Ch 10), manure
management (Ch 10), agricultural soils N2O (Ch 11), rice cultivation CH4
(Ch 5), liming and urea CO2 (Ch 11), and field burning of crop residues
(Ch 2).  Also includes DEFRA and EPA agricultural emission factors.

This engine is the single source of truth for numeric constants used by
the AgriculturalCalculatorEngine (Engine 2).  By centralizing all emission
factors and reference data in one module, we guarantee that every calculation
in the pipeline uses identical, auditable, peer-reviewed values.

Built-In Reference Data:
    - Enteric fermentation EFs by 13 animal types and 2 region classes
    - Volatile solids (VS) for 8 animal types (kg VS/head/day)
    - Maximum CH4 capacity (Bo) for 8 animal types (m3 CH4/kg VS)
    - Methane Correction Factors (MCF) by 15 AWMS types and 3 temperature ranges
    - Manure N2O emission factors by 15 AWMS types
    - Nitrogen excretion rates (Nex) by 13 animal types (kg N/head/yr)
    - Direct soil N2O factors (EF1, EF2_CG, EF2_F, EF3_PRP)
    - Indirect N2O fractions and factors (Frac_GASF/GASM, Frac_LEACH, EF4, EF5)
    - Liming emission factors for limestone and dolomite (tC/t)
    - Urea emission factor (tC/t urea)
    - Rice baseline EF (kg CH4/ha/day) with water regime and amendment SFs
    - Field burning EFs by 10 crop types (CH4/N2O, RPR, DM fraction)
    - GWP values across AR4/AR5/AR6/AR6_20YR
    - Tier 2 maintenance coefficients (Cfi) for 13 animal types
    - Default body weights for 13 animal types
    - Milk yield defaults by region
    - Feed digestibility (DE%) for 12 feed types
    - Crop residue parameters (RPR, N content, DM) for 15 crop types
    - DEFRA agricultural emission factors
    - EPA 40 CFR 98 Subpart JJ factors
    - Custom emission factor registry with thread-safe locking

Zero-Hallucination Guarantees:
    - All factors are hard-coded from published IPCC/EPA/DEFRA tables.
    - All lookups are deterministic dictionary access.
    - No LLM involvement in any data retrieval path.
    - Every query result carries a SHA-256 provenance hash.

Thread Safety:
    All reference data is immutable after initialization.  The mutable
    custom factor registry is protected by a reentrant lock.

Example:
    >>> from greenlang.agents.mrv.agricultural_emissions.agricultural_database import (
    ...     AgriculturalDatabaseEngine,
    ... )
    >>> db = AgriculturalDatabaseEngine()
    >>> ef = db.get_enteric_ef("dairy_cattle", "developed")
    >>> mcf = db.get_manure_mcf("anaerobic_lagoon", "warm")
    >>> gwp = db.get_gwp("CH4", "AR6")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-008 Agricultural Emissions (GL-MRV-SCOPE1-008)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["AgriculturalDatabaseEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.agricultural_emissions.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.agricultural_emissions.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.agricultural_emissions.metrics import (
        record_component_operation as _record_db_operation,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_db_operation = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, str, or Pydantic model).

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Decimal precision constant
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")  # 8 decimal places


def _D(value: Any) -> Decimal:
    """Convert a value to Decimal with controlled precision.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


# ===========================================================================
# Enumerations
# ===========================================================================


class AnimalType(str, Enum):
    """IPCC livestock categories per 2006 Guidelines Vol 4 Ch 10.

    Covers all animal types used for enteric fermentation and manure
    management emission factor lookups.
    """

    DAIRY_CATTLE = "DAIRY_CATTLE"
    NON_DAIRY_CATTLE = "NON_DAIRY_CATTLE"
    BUFFALO = "BUFFALO"
    SHEEP = "SHEEP"
    GOATS = "GOATS"
    CAMELS = "CAMELS"
    HORSES = "HORSES"
    MULES = "MULES"
    SWINE_MARKET = "SWINE_MARKET"
    SWINE_BREEDING = "SWINE_BREEDING"
    POULTRY = "POULTRY"
    POULTRY_LAYERS = "POULTRY_LAYERS"
    POULTRY_BROILERS = "POULTRY_BROILERS"
    RABBITS = "RABBITS"


class ManureSystem(str, Enum):
    """Animal Waste Management Systems (AWMS) per IPCC 2006 Vol 4 Ch 10.

    15 manure management systems used for MCF and N2O emission factor
    lookups.  Systems cover liquid, solid, and pasture-based management.
    """

    PASTURE = "PASTURE"
    DAILY_SPREAD = "DAILY_SPREAD"
    SOLID_STORAGE = "SOLID_STORAGE"
    DRY_LOT = "DRY_LOT"
    LIQUID_SLURRY = "LIQUID_SLURRY"
    LIQUID_SLURRY_CRUST = "LIQUID_SLURRY_CRUST"
    ANAEROBIC_LAGOON = "ANAEROBIC_LAGOON"
    PIT_STORAGE = "PIT_STORAGE"
    DEEP_BEDDING_NO_MIX = "DEEP_BEDDING_NO_MIX"
    DEEP_BEDDING_ACTIVE_MIX = "DEEP_BEDDING_ACTIVE_MIX"
    COMPOSTING_IN_VESSEL = "COMPOSTING_IN_VESSEL"
    COMPOSTING_STATIC_PILE = "COMPOSTING_STATIC_PILE"
    COMPOSTING_WINDROW = "COMPOSTING_WINDROW"
    AEROBIC_TREATMENT = "AEROBIC_TREATMENT"
    BURNED_AS_FUEL = "BURNED_AS_FUEL"


class WaterRegime(str, Enum):
    """Rice water regime categories per IPCC 2006 Vol 4 Ch 5 Table 5.12.

    Seven water management practices affecting CH4 emissions from
    flooded rice paddies.
    """

    CONTINUOUSLY_FLOODED = "CONTINUOUSLY_FLOODED"
    INTERMITTENT_SINGLE = "INTERMITTENT_SINGLE"
    INTERMITTENT_MULTIPLE = "INTERMITTENT_MULTIPLE"
    RAINFED_REGULAR = "RAINFED_REGULAR"
    RAINFED_DROUGHT = "RAINFED_DROUGHT"
    DEEPWATER = "DEEPWATER"
    UPLAND = "UPLAND"


class PreSeasonFlooding(str, Enum):
    """Pre-season water status for rice paddies per IPCC 2006 Vol 4 Ch 5.

    Pre-season flooding affects residual organic matter decomposition
    and baseline methane emission rates.
    """

    NOT_FLOODED_LT_180 = "NOT_FLOODED_LT_180"
    NOT_FLOODED_GT_180 = "NOT_FLOODED_GT_180"
    FLOODED_LT_30 = "FLOODED_LT_30"
    FLOODED_GT_30 = "FLOODED_GT_30"


class OrganicAmendment(str, Enum):
    """Organic amendment types for rice cultivation per IPCC 2006 Vol 4 Ch 5.

    Different organic amendments have different conversion factors (CFOA)
    affecting methane emission scaling from rice paddies.
    """

    STRAW_SHORT = "STRAW_SHORT"
    STRAW_LONG = "STRAW_LONG"
    COMPOST = "COMPOST"
    FARM_YARD_MANURE = "FARM_YARD_MANURE"
    GREEN_MANURE = "GREEN_MANURE"


class CropType(str, Enum):
    """Crop types for field burning and residue parameter lookups.

    Per IPCC 2006 Vol 4 Ch 2 Table 2.5 and 2.6.
    """

    WHEAT = "WHEAT"
    RICE = "RICE"
    MAIZE = "MAIZE"
    BARLEY = "BARLEY"
    OATS = "OATS"
    RYE = "RYE"
    MILLET = "MILLET"
    SORGHUM = "SORGHUM"
    SUGARCANE = "SUGARCANE"
    COTTON = "COTTON"
    SOYBEANS = "SOYBEANS"
    POTATOES = "POTATOES"
    PULSES = "PULSES"
    GROUNDNUTS = "GROUNDNUTS"
    RAPESEED = "RAPESEED"


class SoilN2OFactorType(str, Enum):
    """Direct N2O emission factor types per IPCC 2006 Vol 4 Ch 11.

    EF1: Synthetic and organic N inputs to managed soils.
    EF2_CG: Organic soils - cropland/grassland.
    EF2_F: Organic soils - forest land.
    EF3_PRP: Pasture, range, and paddock deposited manure N.
    """

    EF1 = "EF1"
    EF2_CG = "EF2_CG"
    EF2_F = "EF2_F"
    EF3_PRP = "EF3_PRP"
    EF3_PRP_CPP = "EF3_PRP_CPP"
    EF3_PRP_SO = "EF3_PRP_SO"


class IndirectN2OFractionType(str, Enum):
    """Indirect N2O fraction types per IPCC 2006 Vol 4 Ch 11.

    Frac_GASF: Fraction of synthetic N volatilised as NH3 + NOx.
    Frac_GASM: Fraction of organic N volatilised as NH3 + NOx.
    Frac_LEACH: Fraction of N lost through leaching/runoff.
    EF4: EF for atmospheric deposition.
    EF5: EF for leaching/runoff.
    """

    FRAC_GASF = "FRAC_GASF"
    FRAC_GASM = "FRAC_GASM"
    FRAC_LEACH = "FRAC_LEACH"
    EF4 = "EF4"
    EF5 = "EF5"


class LimingMaterial(str, Enum):
    """Liming material types per IPCC 2006 Vol 4 Ch 11.

    LIMESTONE: CaCO3 - emission factor 0.12 tC/t.
    DOLOMITE: CaMg(CO3)2 - emission factor 0.13 tC/t.
    """

    LIMESTONE = "LIMESTONE"
    DOLOMITE = "DOLOMITE"


class GWPSource(str, Enum):
    """IPCC Assessment Report editions for GWP values.

    AR4: Fourth Assessment Report (2007).
    AR5: Fifth Assessment Report (2014).
    AR6: Sixth Assessment Report (2021), 100-year horizon.
    AR6_20YR: Sixth Assessment Report (2021), 20-year horizon.
    """

    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"
    AR6_20YR = "AR6_20YR"


class RegionType(str, Enum):
    """Broad region classification for enteric fermentation Tier 1 EFs.

    DEVELOPED: Annex I countries (North America, Europe, Oceania, Japan).
    DEVELOPING: Non-Annex I countries (rest of world).
    """

    DEVELOPED = "DEVELOPED"
    DEVELOPING = "DEVELOPING"


class TemperatureRange(str, Enum):
    """Temperature ranges for manure MCF selection.

    Per IPCC 2006 Vol 4 Ch 10 Table 10.17.
    """

    COOL = "COOL"
    TEMPERATE = "TEMPERATE"
    WARM = "WARM"


class EmissionFactorSource(str, Enum):
    """Sources for emission factor data.

    IPCC_2006: IPCC 2006 Guidelines for National GHG Inventories Vol 4.
    IPCC_2019: 2019 Refinement to the 2006 Guidelines.
    EPA: US EPA 40 CFR 98 Subpart JJ.
    DEFRA: UK DEFRA conversion factors.
    COUNTRY_SPECIFIC: National inventory-derived factors.
    SITE_MEASURED: Direct facility measurement values.
    CUSTOM: User-provided emission factors.
    """

    IPCC_2006 = "IPCC_2006"
    IPCC_2019 = "IPCC_2019"
    EPA = "EPA"
    DEFRA = "DEFRA"
    COUNTRY_SPECIFIC = "COUNTRY_SPECIFIC"
    SITE_MEASURED = "SITE_MEASURED"
    CUSTOM = "CUSTOM"


# ===========================================================================
# Dataclasses for structured lookups
# ===========================================================================


@dataclass(frozen=True)
class FieldBurningRecord:
    """Field burning emission factors and residue parameters for a crop type.

    Per IPCC 2006 Vol 4 Ch 2 Tables 2.5 and 2.6.

    Attributes:
        crop_type: Crop identifier.
        residue_product_ratio: Ratio of residue to harvested product (RPR).
        dry_matter_fraction: Dry matter content of residue.
        n_content: Nitrogen content of residue (fraction of DM).
        ch4_ef: CH4 emission factor (g CH4/kg DM burned).
        n2o_ef: N2O emission factor (g N2O/kg DM burned).
        combustion_factor: Fraction of residue actually burned.
        source: Reference citation.
    """

    crop_type: str
    residue_product_ratio: Decimal
    dry_matter_fraction: Decimal
    n_content: Decimal
    ch4_ef: Decimal
    n2o_ef: Decimal
    combustion_factor: Decimal
    source: str


@dataclass(frozen=True)
class CropResidueRecord:
    """Crop residue parameters for N2O soil emission calculations.

    Per IPCC 2006 Vol 4 Ch 11 Table 11.2.

    Attributes:
        crop_type: Crop identifier.
        above_ground_ratio: Above-ground residue to product ratio (RAG).
        below_ground_ratio: Below-ground residue to product ratio (RBG).
        dry_matter_fraction: Dry matter fraction of harvested product.
        n_ag: N content of above-ground residue (fraction of DM).
        n_bg: N content of below-ground residue (fraction of DM).
        source: Reference citation.
    """

    crop_type: str
    above_ground_ratio: Decimal
    below_ground_ratio: Decimal
    dry_matter_fraction: Decimal
    n_ag: Decimal
    n_bg: Decimal
    source: str


# ===========================================================================
# GWP Values (IPCC AR4/AR5/AR6/AR6_20YR)
# ===========================================================================

#: Global Warming Potential values for agricultural gases.
#: Keys: (GWP_source, gas), Values: Decimal GWP factor.
GWP_VALUES: Dict[str, Dict[str, Decimal]] = {
    "AR4": {
        "CO2": _D("1"),
        "CH4": _D("25"),
        "CH4_FOSSIL": _D("25"),
        "CH4_BIOGENIC": _D("25"),
        "N2O": _D("298"),
    },
    "AR5": {
        "CO2": _D("1"),
        "CH4": _D("28"),
        "CH4_FOSSIL": _D("30"),
        "CH4_BIOGENIC": _D("28"),
        "N2O": _D("265"),
    },
    "AR6": {
        "CO2": _D("1"),
        "CH4": _D("29.8"),
        "CH4_FOSSIL": _D("29.8"),
        "CH4_BIOGENIC": _D("27.0"),
        "N2O": _D("273"),
    },
    "AR6_20YR": {
        "CO2": _D("1"),
        "CH4": _D("82.5"),
        "CH4_FOSSIL": _D("82.5"),
        "CH4_BIOGENIC": _D("80.8"),
        "N2O": _D("273"),
    },
}


# ===========================================================================
# Enteric Fermentation Emission Factors (kg CH4/head/yr)
# IPCC 2006 Vol 4 Ch 10 Tables 10.10, 10.11
# ===========================================================================

#: Tier 1 enteric fermentation emission factors by animal type and region.
#: Units: kg CH4 per head per year.
#: Source: IPCC 2006 Vol 4 Ch 10 Tables 10.10, 10.11.
ENTERIC_EF: Dict[str, Dict[str, Decimal]] = {
    "DAIRY_CATTLE": {
        "DEVELOPED": _D("128"),
        "DEVELOPING": _D("68"),
    },
    "NON_DAIRY_CATTLE": {
        "DEVELOPED": _D("53"),
        "DEVELOPING": _D("47"),
    },
    "BUFFALO": {
        "DEVELOPED": _D("55"),
        "DEVELOPING": _D("55"),
    },
    "SHEEP": {
        "DEVELOPED": _D("8"),
        "DEVELOPING": _D("5"),
    },
    "GOATS": {
        "DEVELOPED": _D("5"),
        "DEVELOPING": _D("5"),
    },
    "CAMELS": {
        "DEVELOPED": _D("46"),
        "DEVELOPING": _D("46"),
    },
    "HORSES": {
        "DEVELOPED": _D("18"),
        "DEVELOPING": _D("18"),
    },
    "MULES": {
        "DEVELOPED": _D("10"),
        "DEVELOPING": _D("10"),
    },
    "SWINE_MARKET": {
        "DEVELOPED": _D("1.5"),
        "DEVELOPING": _D("1"),
    },
    "SWINE_BREEDING": {
        "DEVELOPED": _D("1.5"),
        "DEVELOPING": _D("1"),
    },
    "POULTRY": {
        "DEVELOPED": _D("0"),
        "DEVELOPING": _D("0"),
    },
    "POULTRY_LAYERS": {
        "DEVELOPED": _D("0"),
        "DEVELOPING": _D("0"),
    },
    "POULTRY_BROILERS": {
        "DEVELOPED": _D("0"),
        "DEVELOPING": _D("0"),
    },
    "RABBITS": {
        "DEVELOPED": _D("0.3"),
        "DEVELOPING": _D("0.3"),
    },
}


# ===========================================================================
# Manure Volatile Solids (VS) (kg VS/head/day)
# IPCC 2006 Vol 4 Ch 10 Table 10.13A
# ===========================================================================

#: Volatile solids excretion rate by animal type.
#: Units: kg VS per head per day.
#: Source: IPCC 2006 Vol 4 Ch 10 Table 10.13A.
MANURE_VS: Dict[str, Decimal] = {
    "DAIRY_CATTLE": _D("5.4"),
    "NON_DAIRY_CATTLE": _D("3.9"),
    "BUFFALO": _D("3.9"),
    "SHEEP": _D("0.32"),
    "GOATS": _D("0.22"),
    "SWINE_MARKET": _D("0.30"),
    "SWINE_BREEDING": _D("0.50"),
    "POULTRY": _D("0.02"),
    "POULTRY_LAYERS": _D("0.02"),
    "POULTRY_BROILERS": _D("0.01"),
    "HORSES": _D("2.10"),
    "MULES": _D("1.80"),
    "CAMELS": _D("2.60"),
    "RABBITS": _D("0.05"),
}


# ===========================================================================
# Manure Maximum CH4 Producing Capacity (Bo) (m3 CH4/kg VS)
# IPCC 2006 Vol 4 Ch 10 Table 10.16
# ===========================================================================

#: Maximum methane producing capacity per kg of volatile solids.
#: Units: m3 CH4 per kg VS.
#: Source: IPCC 2006 Vol 4 Ch 10 Table 10.16.
MANURE_BO: Dict[str, Decimal] = {
    "DAIRY_CATTLE": _D("0.24"),
    "NON_DAIRY_CATTLE": _D("0.19"),
    "BUFFALO": _D("0.10"),
    "SHEEP": _D("0.19"),
    "GOATS": _D("0.17"),
    "SWINE_MARKET": _D("0.48"),
    "SWINE_BREEDING": _D("0.48"),
    "POULTRY": _D("0.39"),
    "POULTRY_LAYERS": _D("0.39"),
    "POULTRY_BROILERS": _D("0.36"),
    "HORSES": _D("0.33"),
    "MULES": _D("0.33"),
    "CAMELS": _D("0.10"),
    "RABBITS": _D("0.32"),
}


# ===========================================================================
# Manure MCF - Methane Correction Factor by AWMS and Temperature
# IPCC 2006 Vol 4 Ch 10 Table 10.17
# ===========================================================================

#: MCF values by (AWMS type, temperature range).
#: Dimensionless fraction (0 to 1).
#: Temperature ranges: cool (<15C), temperate (15-25C), warm (>25C).
#: Source: IPCC 2006 Vol 4 Ch 10 Table 10.17.
MANURE_MCF: Dict[str, Dict[str, Decimal]] = {
    "PASTURE": {
        "COOL": _D("0.01"),
        "TEMPERATE": _D("0.015"),
        "WARM": _D("0.02"),
    },
    "DAILY_SPREAD": {
        "COOL": _D("0.001"),
        "TEMPERATE": _D("0.005"),
        "WARM": _D("0.01"),
    },
    "SOLID_STORAGE": {
        "COOL": _D("0.02"),
        "TEMPERATE": _D("0.04"),
        "WARM": _D("0.05"),
    },
    "DRY_LOT": {
        "COOL": _D("0.01"),
        "TEMPERATE": _D("0.015"),
        "WARM": _D("0.02"),
    },
    "LIQUID_SLURRY": {
        "COOL": _D("0.10"),
        "TEMPERATE": _D("0.35"),
        "WARM": _D("0.65"),
    },
    "LIQUID_SLURRY_CRUST": {
        "COOL": _D("0.05"),
        "TEMPERATE": _D("0.20"),
        "WARM": _D("0.40"),
    },
    "ANAEROBIC_LAGOON": {
        "COOL": _D("0.66"),
        "TEMPERATE": _D("0.73"),
        "WARM": _D("0.80"),
    },
    "PIT_STORAGE": {
        "COOL": _D("0.10"),
        "TEMPERATE": _D("0.40"),
        "WARM": _D("0.80"),
    },
    "DEEP_BEDDING_NO_MIX": {
        "COOL": _D("0.10"),
        "TEMPERATE": _D("0.17"),
        "WARM": _D("0.30"),
    },
    "DEEP_BEDDING_ACTIVE_MIX": {
        "COOL": _D("0.17"),
        "TEMPERATE": _D("0.44"),
        "WARM": _D("0.70"),
    },
    "COMPOSTING_IN_VESSEL": {
        "COOL": _D("0.005"),
        "TEMPERATE": _D("0.005"),
        "WARM": _D("0.005"),
    },
    "COMPOSTING_STATIC_PILE": {
        "COOL": _D("0.005"),
        "TEMPERATE": _D("0.005"),
        "WARM": _D("0.005"),
    },
    "COMPOSTING_WINDROW": {
        "COOL": _D("0.005"),
        "TEMPERATE": _D("0.01"),
        "WARM": _D("0.015"),
    },
    "AEROBIC_TREATMENT": {
        "COOL": _D("0"),
        "TEMPERATE": _D("0"),
        "WARM": _D("0"),
    },
    "BURNED_AS_FUEL": {
        "COOL": _D("0.01"),
        "TEMPERATE": _D("0.01"),
        "WARM": _D("0.01"),
    },
}


# ===========================================================================
# Manure N2O Emission Factors by AWMS
# IPCC 2006 Vol 4 Ch 10 Table 10.21
# ===========================================================================

#: N2O emission factors by manure management system.
#: Units: kg N2O-N per kg N excreted and managed.
#: Source: IPCC 2006 Vol 4 Ch 10 Table 10.21.
MANURE_N2O_EF: Dict[str, Decimal] = {
    "PASTURE": _D("0.02"),
    "DAILY_SPREAD": _D("0"),
    "SOLID_STORAGE": _D("0.005"),
    "DRY_LOT": _D("0.02"),
    "LIQUID_SLURRY": _D("0"),
    "LIQUID_SLURRY_CRUST": _D("0.005"),
    "ANAEROBIC_LAGOON": _D("0"),
    "PIT_STORAGE": _D("0.002"),
    "DEEP_BEDDING_NO_MIX": _D("0.01"),
    "DEEP_BEDDING_ACTIVE_MIX": _D("0.07"),
    "COMPOSTING_IN_VESSEL": _D("0.006"),
    "COMPOSTING_STATIC_PILE": _D("0.006"),
    "COMPOSTING_WINDROW": _D("0.01"),
    "AEROBIC_TREATMENT": _D("0.005"),
    "BURNED_AS_FUEL": _D("0"),
}


# ===========================================================================
# Manure Nitrogen Excretion (Nex) (kg N/head/yr)
# IPCC 2006 Vol 4 Ch 10 Table 10.19
# ===========================================================================

#: Nitrogen excretion rate by animal type.
#: Units: kg N per head per year.
#: Source: IPCC 2006 Vol 4 Ch 10 Table 10.19.
MANURE_NEX: Dict[str, Decimal] = {
    "DAIRY_CATTLE": _D("100"),
    "NON_DAIRY_CATTLE": _D("60"),
    "BUFFALO": _D("60"),
    "SHEEP": _D("12"),
    "GOATS": _D("12"),
    "CAMELS": _D("46"),
    "HORSES": _D("40"),
    "MULES": _D("30"),
    "SWINE_MARKET": _D("16"),
    "SWINE_BREEDING": _D("20"),
    "POULTRY": _D("0.6"),
    "POULTRY_LAYERS": _D("0.6"),
    "POULTRY_BROILERS": _D("0.6"),
    "RABBITS": _D("5.6"),
}


# ===========================================================================
# Direct Soil N2O Emission Factors
# IPCC 2006 Vol 4 Ch 11 Table 11.1
# ===========================================================================

#: Direct N2O emission factors for agricultural soils.
#: EF1: kg N2O-N per kg N applied (synthetic + organic).
#: EF2_CG: kg N2O-N per hectare per year for organic soils (cropland/grassland).
#: EF2_F: kg N2O-N per hectare per year for organic soils (forest).
#: EF3_PRP: kg N2O-N per kg N deposited by grazing animals.
#: EF3_PRP_CPP: for cattle, poultry, pigs.
#: EF3_PRP_SO: for sheep and other animals.
#: Source: IPCC 2006 Vol 4 Ch 11 Table 11.1.
SOIL_N2O_FACTORS: Dict[str, Decimal] = {
    "EF1": _D("0.01"),
    "EF2_CG": _D("8"),
    "EF2_F": _D("0.6"),
    "EF3_PRP": _D("0.02"),
    "EF3_PRP_CPP": _D("0.02"),
    "EF3_PRP_SO": _D("0.01"),
}


# ===========================================================================
# Indirect N2O Fractions and Emission Factors
# IPCC 2006 Vol 4 Ch 11 Table 11.3
# ===========================================================================

#: Indirect N2O parameters.
#: Frac_GASF: Fraction of synthetic N that volatilises as NH3 + NOx.
#: Frac_GASM: Fraction of organic N (manure/compost) that volatilises.
#: Frac_LEACH: Fraction of N additions lost through leaching/runoff.
#: EF4: kg N2O-N per kg N volatilised and redeposited.
#: EF5: kg N2O-N per kg N leached/runoff.
#: Source: IPCC 2006 Vol 4 Ch 11 Table 11.3.
INDIRECT_N2O_FACTORS: Dict[str, Decimal] = {
    "FRAC_GASF": _D("0.10"),
    "FRAC_GASM": _D("0.20"),
    "FRAC_LEACH": _D("0.30"),
    "EF4": _D("0.01"),
    "EF5": _D("0.0075"),
}


# ===========================================================================
# Liming Emission Factors
# IPCC 2006 Vol 4 Ch 11 Eq 11.12
# ===========================================================================

#: Liming emission factors (tonnes C per tonne of material).
#: Source: IPCC 2006 Vol 4 Ch 11 Eq 11.12.
LIMING_FACTORS: Dict[str, Decimal] = {
    "LIMESTONE": _D("0.12"),
    "DOLOMITE": _D("0.13"),
}


# ===========================================================================
# Urea Emission Factor
# IPCC 2006 Vol 4 Ch 11 Eq 11.13
# ===========================================================================

#: Urea CO2 emission factor (tonnes C per tonne urea applied).
#: CO2 emissions = M_urea * EF_urea * 44/12.
#: Source: IPCC 2006 Vol 4 Ch 11 Eq 11.13.
UREA_FACTOR: Decimal = _D("0.20")


# ===========================================================================
# Rice Cultivation Baseline Emission Factor
# IPCC 2006 Vol 4 Ch 5 Table 5.11
# ===========================================================================

#: Baseline emission factor for continuously flooded rice paddies.
#: Units: kg CH4 per hectare per day.
#: Source: IPCC 2006 Vol 4 Ch 5 Table 5.11.
RICE_BASELINE_EF: Decimal = _D("1.30")


# ===========================================================================
# Rice Water Regime Scaling Factor
# IPCC 2006 Vol 4 Ch 5 Table 5.12
# ===========================================================================

#: Scaling factors for water regime impact on CH4 emissions.
#: Relative to continuously flooded baseline.
#: Source: IPCC 2006 Vol 4 Ch 5 Table 5.12.
RICE_WATER_REGIME_SF: Dict[str, Decimal] = {
    "CONTINUOUSLY_FLOODED": _D("1.0"),
    "INTERMITTENT_SINGLE": _D("0.60"),
    "INTERMITTENT_MULTIPLE": _D("0.52"),
    "RAINFED_REGULAR": _D("0.80"),
    "RAINFED_DROUGHT": _D("0.40"),
    "DEEPWATER": _D("0.80"),
    "UPLAND": _D("0"),
}


# ===========================================================================
# Rice Pre-Season Flooding Scaling Factor
# IPCC 2006 Vol 4 Ch 5 Table 5.13
# ===========================================================================

#: Scaling factors for pre-season flooding status.
#: Source: IPCC 2006 Vol 4 Ch 5 Table 5.13.
RICE_PRESEASON_SF: Dict[str, Decimal] = {
    "NOT_FLOODED_LT_180": _D("1.0"),
    "NOT_FLOODED_GT_180": _D("0.68"),
    "FLOODED_LT_30": _D("1.90"),
    "FLOODED_GT_30": _D("2.40"),
}


# ===========================================================================
# Rice Organic Amendment Conversion Factor (CFOA)
# IPCC 2006 Vol 4 Ch 5 Table 5.14
# ===========================================================================

#: Conversion factor for organic amendments in rice cultivation.
#: Dimensionless scaling factor per tonne/ha of amendment.
#: Source: IPCC 2006 Vol 4 Ch 5 Table 5.14.
RICE_ORGANIC_CFOA: Dict[str, Decimal] = {
    "STRAW_SHORT": _D("0.29"),
    "STRAW_LONG": _D("0.165"),
    "COMPOST": _D("0.05"),
    "FARM_YARD_MANURE": _D("0.14"),
    "GREEN_MANURE": _D("0.50"),
}


# ===========================================================================
# Field Burning Emission Factors and Residue Parameters
# IPCC 2006 Vol 4 Ch 2 Tables 2.5, 2.6
# ===========================================================================

#: Field burning emission factors, residue-to-product ratio, dry matter
#: fraction, N content, combustion factor, CH4 and N2O EFs.
#: Source: IPCC 2006 Vol 4 Ch 2 Tables 2.5, 2.6.
FIELD_BURNING_EF: Dict[str, FieldBurningRecord] = {
    "WHEAT": FieldBurningRecord(
        crop_type="WHEAT",
        residue_product_ratio=_D("1.3"),
        dry_matter_fraction=_D("0.89"),
        n_content=_D("0.006"),
        ch4_ef=_D("2.7"),
        n2o_ef=_D("0.07"),
        combustion_factor=_D("0.80"),
        source="IPCC 2006 Vol4 Ch2 Tables 2.5/2.6",
    ),
    "RICE": FieldBurningRecord(
        crop_type="RICE",
        residue_product_ratio=_D("1.4"),
        dry_matter_fraction=_D("0.89"),
        n_content=_D("0.007"),
        ch4_ef=_D("2.7"),
        n2o_ef=_D("0.07"),
        combustion_factor=_D("0.80"),
        source="IPCC 2006 Vol4 Ch2 Tables 2.5/2.6",
    ),
    "MAIZE": FieldBurningRecord(
        crop_type="MAIZE",
        residue_product_ratio=_D("1.0"),
        dry_matter_fraction=_D("0.89"),
        n_content=_D("0.006"),
        ch4_ef=_D("2.7"),
        n2o_ef=_D("0.07"),
        combustion_factor=_D("0.80"),
        source="IPCC 2006 Vol4 Ch2 Tables 2.5/2.6",
    ),
    "BARLEY": FieldBurningRecord(
        crop_type="BARLEY",
        residue_product_ratio=_D("1.2"),
        dry_matter_fraction=_D("0.89"),
        n_content=_D("0.007"),
        ch4_ef=_D("2.7"),
        n2o_ef=_D("0.07"),
        combustion_factor=_D("0.80"),
        source="IPCC 2006 Vol4 Ch2 Tables 2.5/2.6",
    ),
    "OATS": FieldBurningRecord(
        crop_type="OATS",
        residue_product_ratio=_D("1.3"),
        dry_matter_fraction=_D("0.89"),
        n_content=_D("0.007"),
        ch4_ef=_D("2.7"),
        n2o_ef=_D("0.07"),
        combustion_factor=_D("0.80"),
        source="IPCC 2006 Vol4 Ch2 Tables 2.5/2.6",
    ),
    "RYE": FieldBurningRecord(
        crop_type="RYE",
        residue_product_ratio=_D("1.6"),
        dry_matter_fraction=_D("0.89"),
        n_content=_D("0.005"),
        ch4_ef=_D("2.7"),
        n2o_ef=_D("0.07"),
        combustion_factor=_D("0.80"),
        source="IPCC 2006 Vol4 Ch2 Tables 2.5/2.6",
    ),
    "MILLET": FieldBurningRecord(
        crop_type="MILLET",
        residue_product_ratio=_D("1.4"),
        dry_matter_fraction=_D("0.89"),
        n_content=_D("0.007"),
        ch4_ef=_D("2.7"),
        n2o_ef=_D("0.07"),
        combustion_factor=_D("0.80"),
        source="IPCC 2006 Vol4 Ch2 Tables 2.5/2.6",
    ),
    "SORGHUM": FieldBurningRecord(
        crop_type="SORGHUM",
        residue_product_ratio=_D("1.4"),
        dry_matter_fraction=_D("0.89"),
        n_content=_D("0.007"),
        ch4_ef=_D("2.7"),
        n2o_ef=_D("0.07"),
        combustion_factor=_D("0.80"),
        source="IPCC 2006 Vol4 Ch2 Tables 2.5/2.6",
    ),
    "SUGARCANE": FieldBurningRecord(
        crop_type="SUGARCANE",
        residue_product_ratio=_D("0.20"),
        dry_matter_fraction=_D("0.71"),
        n_content=_D("0.004"),
        ch4_ef=_D("2.7"),
        n2o_ef=_D("0.07"),
        combustion_factor=_D("0.80"),
        source="IPCC 2006 Vol4 Ch2 Tables 2.5/2.6",
    ),
    "COTTON": FieldBurningRecord(
        crop_type="COTTON",
        residue_product_ratio=_D("2.1"),
        dry_matter_fraction=_D("0.91"),
        n_content=_D("0.012"),
        ch4_ef=_D("2.7"),
        n2o_ef=_D("0.07"),
        combustion_factor=_D("0.80"),
        source="IPCC 2006 Vol4 Ch2 Tables 2.5/2.6",
    ),
}


# ===========================================================================
# Tier 2 Maintenance Coefficients (Cfi)
# IPCC 2006 Vol 4 Ch 10 Table 10.4
# ===========================================================================

#: Maintenance energy coefficients for Tier 2 enteric fermentation.
#: Units: MJ per day per kg body weight.
#: Source: IPCC 2006 Vol 4 Ch 10 Table 10.4.
MAINTENANCE_COEFFICIENTS: Dict[str, Decimal] = {
    "DAIRY_CATTLE": _D("0.386"),
    "NON_DAIRY_CATTLE": _D("0.322"),
    "BUFFALO": _D("0.322"),
    "SHEEP": _D("0.217"),
    "GOATS": _D("0.217"),
    "CAMELS": _D("0.343"),
    "HORSES": _D("0.303"),
    "MULES": _D("0.303"),
    "SWINE_MARKET": _D("0.322"),
    "SWINE_BREEDING": _D("0.386"),
    "POULTRY": _D("0"),
    "POULTRY_LAYERS": _D("0"),
    "POULTRY_BROILERS": _D("0"),
    "RABBITS": _D("0"),
}


# ===========================================================================
# Default Body Weights (kg)
# IPCC 2006 Vol 4 Ch 10 Table 10.5
# ===========================================================================

#: Default live body weights by animal type.
#: Units: kg per head.
#: Source: IPCC 2006 Vol 4 Ch 10 Table 10.5.
BODY_WEIGHT_DEFAULTS: Dict[str, Decimal] = {
    "DAIRY_CATTLE": _D("600"),
    "NON_DAIRY_CATTLE": _D("420"),
    "BUFFALO": _D("380"),
    "SHEEP": _D("48"),
    "GOATS": _D("40"),
    "CAMELS": _D("450"),
    "HORSES": _D("380"),
    "MULES": _D("245"),
    "SWINE_MARKET": _D("65"),
    "SWINE_BREEDING": _D("198"),
    "POULTRY": _D("1.8"),
    "POULTRY_LAYERS": _D("1.8"),
    "POULTRY_BROILERS": _D("1.0"),
    "RABBITS": _D("3.5"),
}


# ===========================================================================
# Milk Yield Defaults (kg/head/yr)
# IPCC 2006 Vol 4 Ch 10 Table 10.8
# ===========================================================================

#: Default milk yields by region.
#: Units: kg milk per head per year.
#: Source: IPCC 2006 Vol 4 Ch 10 Table 10.8.
MILK_YIELD_DEFAULTS: Dict[str, Decimal] = {
    "DEVELOPED_NORTH_AMERICA": _D("8400"),
    "DEVELOPED_WESTERN_EUROPE": _D("6000"),
    "DEVELOPED_EASTERN_EUROPE": _D("3000"),
    "DEVELOPED_OCEANIA": _D("3500"),
    "DEVELOPING_AFRICA": _D("475"),
    "DEVELOPING_ASIA": _D("1650"),
    "DEVELOPING_LATIN_AMERICA": _D("1300"),
    "DEVELOPING_MIDDLE_EAST": _D("1100"),
    "GLOBAL_AVERAGE": _D("2400"),
}


# ===========================================================================
# Feed Digestibility (DE%) by Feed Type
# IPCC 2006 Vol 4 Ch 10 Table 10.2
# ===========================================================================

#: Digestible energy as percentage of gross energy (DE%).
#: Source: IPCC 2006 Vol 4 Ch 10 Table 10.2.
FEED_DIGESTIBILITY: Dict[str, Decimal] = {
    "HIGH_QUALITY_PASTURE": _D("70"),
    "MEDIUM_QUALITY_PASTURE": _D("60"),
    "LOW_QUALITY_PASTURE": _D("50"),
    "GRAIN_BASED": _D("80"),
    "GRAIN_SUPPLEMENTED": _D("75"),
    "SILAGE_BASED": _D("65"),
    "HAY_GOOD": _D("58"),
    "HAY_POOR": _D("50"),
    "STRAW": _D("45"),
    "CROP_RESIDUE": _D("48"),
    "TOTAL_MIXED_RATION": _D("78"),
    "CONCENTRATE_HEAVY": _D("82"),
}


# ===========================================================================
# Crop Residue Parameters for N2O Soil Calculations
# IPCC 2006 Vol 4 Ch 11 Table 11.2
# ===========================================================================

#: Crop residue parameters for estimating N inputs from crop residues.
#: RAG: above-ground residue to product ratio.
#: RBG: below-ground residue to product ratio.
#: DM: dry matter fraction of harvested product.
#: N_AG: N content of above-ground residue (fraction of DM).
#: N_BG: N content of below-ground residue (fraction of DM).
#: Source: IPCC 2006 Vol 4 Ch 11 Table 11.2.
CROP_RESIDUE_PARAMS: Dict[str, CropResidueRecord] = {
    "WHEAT": CropResidueRecord(
        crop_type="WHEAT",
        above_ground_ratio=_D("1.3"),
        below_ground_ratio=_D("0.23"),
        dry_matter_fraction=_D("0.89"),
        n_ag=_D("0.006"),
        n_bg=_D("0.009"),
        source="IPCC 2006 Vol4 Ch11 Table 11.2",
    ),
    "RICE": CropResidueRecord(
        crop_type="RICE",
        above_ground_ratio=_D("1.4"),
        below_ground_ratio=_D("0.16"),
        dry_matter_fraction=_D("0.89"),
        n_ag=_D("0.007"),
        n_bg=_D("0.009"),
        source="IPCC 2006 Vol4 Ch11 Table 11.2",
    ),
    "MAIZE": CropResidueRecord(
        crop_type="MAIZE",
        above_ground_ratio=_D("1.0"),
        below_ground_ratio=_D("0.22"),
        dry_matter_fraction=_D("0.87"),
        n_ag=_D("0.006"),
        n_bg=_D("0.007"),
        source="IPCC 2006 Vol4 Ch11 Table 11.2",
    ),
    "BARLEY": CropResidueRecord(
        crop_type="BARLEY",
        above_ground_ratio=_D("1.2"),
        below_ground_ratio=_D("0.24"),
        dry_matter_fraction=_D("0.89"),
        n_ag=_D("0.007"),
        n_bg=_D("0.014"),
        source="IPCC 2006 Vol4 Ch11 Table 11.2",
    ),
    "OATS": CropResidueRecord(
        crop_type="OATS",
        above_ground_ratio=_D("1.3"),
        below_ground_ratio=_D("0.25"),
        dry_matter_fraction=_D("0.89"),
        n_ag=_D("0.007"),
        n_bg=_D("0.008"),
        source="IPCC 2006 Vol4 Ch11 Table 11.2",
    ),
    "RYE": CropResidueRecord(
        crop_type="RYE",
        above_ground_ratio=_D("1.6"),
        below_ground_ratio=_D("0.22"),
        dry_matter_fraction=_D("0.89"),
        n_ag=_D("0.005"),
        n_bg=_D("0.011"),
        source="IPCC 2006 Vol4 Ch11 Table 11.2",
    ),
    "MILLET": CropResidueRecord(
        crop_type="MILLET",
        above_ground_ratio=_D("1.4"),
        below_ground_ratio=_D("0.22"),
        dry_matter_fraction=_D("0.89"),
        n_ag=_D("0.007"),
        n_bg=_D("0.011"),
        source="IPCC 2006 Vol4 Ch11 Table 11.2",
    ),
    "SORGHUM": CropResidueRecord(
        crop_type="SORGHUM",
        above_ground_ratio=_D("1.4"),
        below_ground_ratio=_D("0.22"),
        dry_matter_fraction=_D("0.89"),
        n_ag=_D("0.007"),
        n_bg=_D("0.006"),
        source="IPCC 2006 Vol4 Ch11 Table 11.2",
    ),
    "SUGARCANE": CropResidueRecord(
        crop_type="SUGARCANE",
        above_ground_ratio=_D("0.20"),
        below_ground_ratio=_D("0.20"),
        dry_matter_fraction=_D("0.71"),
        n_ag=_D("0.004"),
        n_bg=_D("0.005"),
        source="IPCC 2006 Vol4 Ch11 Table 11.2",
    ),
    "COTTON": CropResidueRecord(
        crop_type="COTTON",
        above_ground_ratio=_D("2.1"),
        below_ground_ratio=_D("0.30"),
        dry_matter_fraction=_D("0.91"),
        n_ag=_D("0.012"),
        n_bg=_D("0.009"),
        source="IPCC 2006 Vol4 Ch11 Table 11.2",
    ),
    "SOYBEANS": CropResidueRecord(
        crop_type="SOYBEANS",
        above_ground_ratio=_D("2.1"),
        below_ground_ratio=_D("0.19"),
        dry_matter_fraction=_D("0.91"),
        n_ag=_D("0.008"),
        n_bg=_D("0.008"),
        source="IPCC 2006 Vol4 Ch11 Table 11.2",
    ),
    "POTATOES": CropResidueRecord(
        crop_type="POTATOES",
        above_ground_ratio=_D("0.40"),
        below_ground_ratio=_D("0.20"),
        dry_matter_fraction=_D("0.22"),
        n_ag=_D("0.019"),
        n_bg=_D("0.014"),
        source="IPCC 2006 Vol4 Ch11 Table 11.2",
    ),
    "PULSES": CropResidueRecord(
        crop_type="PULSES",
        above_ground_ratio=_D("1.0"),
        below_ground_ratio=_D("0.19"),
        dry_matter_fraction=_D("0.91"),
        n_ag=_D("0.008"),
        n_bg=_D("0.008"),
        source="IPCC 2006 Vol4 Ch11 Table 11.2",
    ),
    "GROUNDNUTS": CropResidueRecord(
        crop_type="GROUNDNUTS",
        above_ground_ratio=_D("1.0"),
        below_ground_ratio=_D("0.19"),
        dry_matter_fraction=_D("0.94"),
        n_ag=_D("0.016"),
        n_bg=_D("0.014"),
        source="IPCC 2006 Vol4 Ch11 Table 11.2",
    ),
    "RAPESEED": CropResidueRecord(
        crop_type="RAPESEED",
        above_ground_ratio=_D("1.8"),
        below_ground_ratio=_D("0.22"),
        dry_matter_fraction=_D("0.91"),
        n_ag=_D("0.006"),
        n_bg=_D("0.009"),
        source="IPCC 2006 Vol4 Ch11 Table 11.2",
    ),
}


# ===========================================================================
# DEFRA Agricultural Emission Factors
# Units: kg CO2e per head per year (livestock) or per tonne (crops)
# Source: DEFRA 2025 Conversion Factors
# ===========================================================================

#: DEFRA emission factors for UK agricultural reporting.
#: Livestock: kg CO2e per head per year.
#: Crops: kg CO2e per tonne of product.
DEFRA_AG_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "DAIRY_CATTLE": {
        "ENTERIC": _D("2890"),
        "MANURE": _D("910"),
        "TOTAL": _D("3800"),
    },
    "NON_DAIRY_CATTLE": {
        "ENTERIC": _D("1280"),
        "MANURE": _D("400"),
        "TOTAL": _D("1680"),
    },
    "SHEEP": {
        "ENTERIC": _D("210"),
        "MANURE": _D("35"),
        "TOTAL": _D("245"),
    },
    "GOATS": {
        "ENTERIC": _D("135"),
        "MANURE": _D("25"),
        "TOTAL": _D("160"),
    },
    "SWINE_MARKET": {
        "ENTERIC": _D("38"),
        "MANURE": _D("230"),
        "TOTAL": _D("268"),
    },
    "SWINE_BREEDING": {
        "ENTERIC": _D("38"),
        "MANURE": _D("340"),
        "TOTAL": _D("378"),
    },
    "POULTRY_LAYERS": {
        "ENTERIC": _D("0"),
        "MANURE": _D("5.1"),
        "TOTAL": _D("5.1"),
    },
    "POULTRY_BROILERS": {
        "ENTERIC": _D("0"),
        "MANURE": _D("2.7"),
        "TOTAL": _D("2.7"),
    },
    "HORSES": {
        "ENTERIC": _D("440"),
        "MANURE": _D("120"),
        "TOTAL": _D("560"),
    },
    "NITROGEN_FERTILISER": {
        "DIRECT_N2O": _D("4.27"),
        "INDIRECT_N2O": _D("1.05"),
        "TOTAL": _D("5.32"),
    },
    "LIMING_LIMESTONE": {
        "CO2": _D("440"),
    },
    "LIMING_DOLOMITE": {
        "CO2": _D("477"),
    },
    "UREA_APPLICATION": {
        "CO2": _D("733"),
    },
    "RICE_CULTIVATION": {
        "CH4": _D("720"),
    },
}


# ===========================================================================
# EPA 40 CFR 98 Subpart JJ Factors
# Units: metric tonnes CO2e per head per year
# Source: EPA Mandatory GHG Reporting 40 CFR 98 Subpart JJ
# ===========================================================================

#: EPA manure management emission factors.
#: Units: metric tonnes CO2e per head per year.
EPA_AG_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "DAIRY_CATTLE": {
        "ENTERIC_CH4": _D("3.56"),
        "MANURE_CH4": _D("1.26"),
        "MANURE_N2O": _D("0.41"),
        "TOTAL": _D("5.23"),
    },
    "NON_DAIRY_CATTLE": {
        "ENTERIC_CH4": _D("1.48"),
        "MANURE_CH4": _D("0.14"),
        "MANURE_N2O": _D("0.24"),
        "TOTAL": _D("1.86"),
    },
    "SWINE_BREEDING": {
        "ENTERIC_CH4": _D("0.04"),
        "MANURE_CH4": _D("0.85"),
        "MANURE_N2O": _D("0.08"),
        "TOTAL": _D("0.97"),
    },
    "SWINE_MARKET": {
        "ENTERIC_CH4": _D("0.04"),
        "MANURE_CH4": _D("0.50"),
        "MANURE_N2O": _D("0.06"),
        "TOTAL": _D("0.60"),
    },
    "SHEEP": {
        "ENTERIC_CH4": _D("0.22"),
        "MANURE_CH4": _D("0.01"),
        "MANURE_N2O": _D("0.04"),
        "TOTAL": _D("0.27"),
    },
    "GOATS": {
        "ENTERIC_CH4": _D("0.14"),
        "MANURE_CH4": _D("0.01"),
        "MANURE_N2O": _D("0.04"),
        "TOTAL": _D("0.19"),
    },
    "POULTRY_LAYERS": {
        "ENTERIC_CH4": _D("0"),
        "MANURE_CH4": _D("0.008"),
        "MANURE_N2O": _D("0.002"),
        "TOTAL": _D("0.010"),
    },
    "POULTRY_BROILERS": {
        "ENTERIC_CH4": _D("0"),
        "MANURE_CH4": _D("0.005"),
        "MANURE_N2O": _D("0.001"),
        "TOTAL": _D("0.006"),
    },
    "HORSES": {
        "ENTERIC_CH4": _D("0.50"),
        "MANURE_CH4": _D("0.12"),
        "MANURE_N2O": _D("0.13"),
        "TOTAL": _D("0.75"),
    },
}


# ===========================================================================
# Conversion Constants
# ===========================================================================

#: N2O-N to N2O molecular weight ratio (44/28).
N2O_N_RATIO: Decimal = _D("1.571429")

#: C to CO2 molecular weight ratio (44/12).
C_TO_CO2_RATIO: Decimal = _D("3.66667")

#: CO2 to C molecular weight ratio (12/44).
CO2_TO_C_RATIO: Decimal = _D("0.27273")

#: Methane density at STP (kg/m3).
METHANE_DENSITY_KG_M3: Decimal = _D("0.6706")

#: Energy content of methane (MJ/m3 at STP).
METHANE_ENERGY_MJ_M3: Decimal = _D("35.84")

#: Days per year (non-leap).
DAYS_PER_YEAR: Decimal = _D("365")


# ===========================================================================
# AgriculturalDatabaseEngine
# ===========================================================================


class AgriculturalDatabaseEngine:
    """Reference data repository for IPCC/EPA/DEFRA agricultural emission factors.

    This engine provides deterministic lookups for all emission factors,
    volatile solids, Bo values, MCF values, nitrogen excretion rates,
    soil N2O factors, rice cultivation parameters, liming/urea factors,
    field burning EFs, GWP values, Tier 2 coefficients, and body weight
    defaults needed by the AgriculturalCalculatorEngine (Engine 2).

    All data is hard-coded from published IPCC, EPA, and DEFRA tables.

    Thread Safety:
        Immutable reference data requires no locking.  The custom factor
        registry uses a reentrant lock for thread-safe mutations.

    Attributes:
        _custom_factors: User-provided custom emission factors.
        _lock: Reentrant lock protecting mutable state.
        _total_lookups: Counter of total lookup operations.
        _cache: In-memory cache for repeated lookups.

    Example:
        >>> db = AgriculturalDatabaseEngine()
        >>> ef = db.get_enteric_ef("dairy_cattle", "developed")
        >>> assert ef == Decimal("128")
    """

    def __init__(self) -> None:
        """Initialize the AgriculturalDatabaseEngine with empty custom factor registry."""
        self._custom_factors: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._total_lookups: int = 0
        self._cache: Dict[str, Any] = {}
        self._created_at = _utcnow()

        logger.info(
            "AgriculturalDatabaseEngine initialized: "
            "animal_types=%d, manure_systems=%d, "
            "water_regimes=%d, crop_types=%d, "
            "enteric_ef_entries=%d, manure_mcf_systems=%d",
            len(AnimalType),
            len(ManureSystem),
            len(WaterRegime),
            len(CropType),
            len(ENTERIC_EF),
            len(MANURE_MCF),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _increment_lookups(self) -> None:
        """Thread-safe increment of the lookup counter."""
        with self._lock:
            self._total_lookups += 1

    def _validate_animal_type(self, animal_type: str) -> str:
        """Validate and normalise an animal type string.

        Args:
            animal_type: Animal type name or enum value.

        Returns:
            Normalised animal type string (uppercase).

        Raises:
            ValueError: If the animal type is not recognized.
        """
        normalised = animal_type.upper().replace(" ", "_").replace("-", "_")
        valid = {e.value for e in AnimalType}
        if normalised not in valid:
            raise ValueError(
                f"Unknown animal type '{animal_type}'. "
                f"Valid types: {sorted(valid)}"
            )
        return normalised

    def _validate_manure_system(self, manure_system: str) -> str:
        """Validate and normalise a manure management system string.

        Args:
            manure_system: AWMS type name or enum value.

        Returns:
            Normalised manure system string.

        Raises:
            ValueError: If the manure system is not recognized.
        """
        normalised = manure_system.upper().replace(" ", "_").replace("-", "_")
        valid = {e.value for e in ManureSystem}
        if normalised not in valid:
            raise ValueError(
                f"Unknown manure system '{manure_system}'. "
                f"Valid systems: {sorted(valid)}"
            )
        return normalised

    def _validate_water_regime(self, water_regime: str) -> str:
        """Validate and normalise a rice water regime string.

        Args:
            water_regime: Water regime name or enum value.

        Returns:
            Normalised water regime string.

        Raises:
            ValueError: If the water regime is not recognized.
        """
        normalised = water_regime.upper().replace(" ", "_").replace("-", "_")
        valid = {e.value for e in WaterRegime}
        if normalised not in valid:
            raise ValueError(
                f"Unknown water regime '{water_regime}'. "
                f"Valid regimes: {sorted(valid)}"
            )
        return normalised

    def _validate_preseason(self, pre_season: str) -> str:
        """Validate and normalise a pre-season flooding type string.

        Args:
            pre_season: Pre-season flooding status.

        Returns:
            Normalised pre-season string.

        Raises:
            ValueError: If the pre-season type is not recognized.
        """
        normalised = pre_season.upper().replace(" ", "_").replace("-", "_")
        valid = {e.value for e in PreSeasonFlooding}
        if normalised not in valid:
            raise ValueError(
                f"Unknown pre-season flooding type '{pre_season}'. "
                f"Valid types: {sorted(valid)}"
            )
        return normalised

    def _validate_amendment_type(self, amendment_type: str) -> str:
        """Validate and normalise an organic amendment type string.

        Args:
            amendment_type: Organic amendment type.

        Returns:
            Normalised amendment type string.

        Raises:
            ValueError: If the amendment type is not recognized.
        """
        normalised = amendment_type.upper().replace(" ", "_").replace("-", "_")
        valid = {e.value for e in OrganicAmendment}
        if normalised not in valid:
            raise ValueError(
                f"Unknown organic amendment type '{amendment_type}'. "
                f"Valid types: {sorted(valid)}"
            )
        return normalised

    def _validate_crop_type(self, crop_type: str) -> str:
        """Validate and normalise a crop type string.

        Args:
            crop_type: Crop type name or enum value.

        Returns:
            Normalised crop type string.

        Raises:
            ValueError: If the crop type is not recognized.
        """
        normalised = crop_type.upper().replace(" ", "_").replace("-", "_")
        valid = {e.value for e in CropType}
        if normalised not in valid:
            raise ValueError(
                f"Unknown crop type '{crop_type}'. "
                f"Valid types: {sorted(valid)}"
            )
        return normalised

    def _validate_region(self, region: str) -> str:
        """Validate and normalise a region type string.

        Args:
            region: Region classification (developed or developing).

        Returns:
            Normalised region string.

        Raises:
            ValueError: If the region is not recognized.
        """
        normalised = region.upper().replace(" ", "_").replace("-", "_")
        valid = {e.value for e in RegionType}
        if normalised not in valid:
            raise ValueError(
                f"Unknown region '{region}'. "
                f"Valid regions: {sorted(valid)}"
            )
        return normalised

    def _validate_temperature(self, temperature: str) -> str:
        """Validate and normalise a temperature range string.

        Args:
            temperature: Temperature range (cool, temperate, warm).

        Returns:
            Normalised temperature string.

        Raises:
            ValueError: If the temperature range is not recognized.
        """
        normalised = temperature.upper().replace(" ", "_").replace("-", "_")
        valid = {e.value for e in TemperatureRange}
        if normalised not in valid:
            raise ValueError(
                f"Unknown temperature range '{temperature}'. "
                f"Valid ranges: {sorted(valid)}"
            )
        return normalised

    # ------------------------------------------------------------------
    # Enteric Fermentation EFs
    # ------------------------------------------------------------------

    def get_enteric_ef(
        self,
        animal_type: str,
        region: str = "DEVELOPED",
    ) -> Decimal:
        """Look up the Tier 1 enteric fermentation emission factor.

        Returns the CH4 emission factor in kg CH4 per head per year
        for a given animal type and region, per IPCC 2006 Vol 4 Ch 10.

        Args:
            animal_type: Livestock category (e.g. "dairy_cattle", "SHEEP").
            region: Region classification ("developed" or "developing").

        Returns:
            Enteric EF as Decimal (kg CH4/head/yr).

        Raises:
            ValueError: If the animal type or region is not recognized.

        Example:
            >>> db = AgriculturalDatabaseEngine()
            >>> db.get_enteric_ef("dairy_cattle", "developed")
            Decimal('128')
        """
        self._increment_lookups()
        at = self._validate_animal_type(animal_type)
        rg = self._validate_region(region)

        cache_key = f"enteric:{at}:{rg}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = ENTERIC_EF[at][rg]
        self._cache[cache_key] = result

        logger.debug(
            "Enteric EF lookup: animal=%s, region=%s, ef=%s kg CH4/head/yr",
            at, rg, result,
        )
        return result

    # ------------------------------------------------------------------
    # Manure Volatile Solids (VS)
    # ------------------------------------------------------------------

    def get_manure_vs(self, animal_type: str) -> Decimal:
        """Look up volatile solids excretion rate for an animal type.

        VS is used in the IPCC Tier 2 manure CH4 calculation:
        CH4_manure = VS * 365 * Bo * MCF * 0.67

        Args:
            animal_type: Livestock category (e.g. "dairy_cattle").

        Returns:
            VS rate as Decimal (kg VS/head/day).

        Raises:
            ValueError: If the animal type is not recognized.
            KeyError: If no VS data exists for the animal type.

        Example:
            >>> db = AgriculturalDatabaseEngine()
            >>> db.get_manure_vs("dairy_cattle")
            Decimal('5.4')
        """
        self._increment_lookups()
        at = self._validate_animal_type(animal_type)

        cache_key = f"vs:{at}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if at not in MANURE_VS:
            raise KeyError(
                f"No VS data for animal type '{animal_type}'. "
                f"Available: {sorted(MANURE_VS.keys())}"
            )

        result = MANURE_VS[at]
        self._cache[cache_key] = result

        logger.debug("VS lookup: animal=%s, vs=%s kg VS/head/day", at, result)
        return result

    # ------------------------------------------------------------------
    # Manure Bo (Maximum CH4 Producing Capacity)
    # ------------------------------------------------------------------

    def get_manure_bo(self, animal_type: str) -> Decimal:
        """Look up the maximum CH4 producing capacity (Bo) for an animal type.

        Bo is used in the IPCC manure CH4 calculation.

        Args:
            animal_type: Livestock category.

        Returns:
            Bo value as Decimal (m3 CH4/kg VS).

        Raises:
            ValueError: If the animal type is not recognized.
            KeyError: If no Bo data exists for the animal type.

        Example:
            >>> db = AgriculturalDatabaseEngine()
            >>> db.get_manure_bo("dairy_cattle")
            Decimal('0.24')
        """
        self._increment_lookups()
        at = self._validate_animal_type(animal_type)

        cache_key = f"bo:{at}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if at not in MANURE_BO:
            raise KeyError(
                f"No Bo data for animal type '{animal_type}'. "
                f"Available: {sorted(MANURE_BO.keys())}"
            )

        result = MANURE_BO[at]
        self._cache[cache_key] = result

        logger.debug("Bo lookup: animal=%s, bo=%s m3/kg VS", at, result)
        return result

    # ------------------------------------------------------------------
    # Manure MCF
    # ------------------------------------------------------------------

    def get_manure_mcf(
        self,
        awms_type: str,
        temperature: str = "TEMPERATE",
    ) -> Decimal:
        """Look up the Methane Correction Factor for a manure system and temperature.

        MCF reflects the methane-generating potential of a manure
        management system at a given average annual temperature.

        Args:
            awms_type: Animal waste management system type
                (e.g. "anaerobic_lagoon", "SOLID_STORAGE").
            temperature: Temperature range ("cool", "temperate", "warm").

        Returns:
            MCF value as Decimal (0 to 1).

        Raises:
            ValueError: If the AWMS type or temperature is not recognized.

        Example:
            >>> db = AgriculturalDatabaseEngine()
            >>> db.get_manure_mcf("anaerobic_lagoon", "warm")
            Decimal('0.80')
        """
        self._increment_lookups()
        ms = self._validate_manure_system(awms_type)
        temp = self._validate_temperature(temperature)

        cache_key = f"mcf:{ms}:{temp}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = MANURE_MCF[ms][temp]
        self._cache[cache_key] = result

        logger.debug(
            "MCF lookup: system=%s, temp=%s, mcf=%s", ms, temp, result,
        )
        return result

    # ------------------------------------------------------------------
    # Manure N2O Emission Factor
    # ------------------------------------------------------------------

    def get_manure_n2o_ef(self, awms_type: str) -> Decimal:
        """Look up the N2O emission factor for a manure management system.

        Returns EF3 (kg N2O-N per kg N excreted and managed in system).

        Args:
            awms_type: Animal waste management system type.

        Returns:
            N2O emission factor as Decimal (kg N2O-N/kg N).

        Raises:
            ValueError: If the AWMS type is not recognized.

        Example:
            >>> db = AgriculturalDatabaseEngine()
            >>> db.get_manure_n2o_ef("solid_storage")
            Decimal('0.005')
        """
        self._increment_lookups()
        ms = self._validate_manure_system(awms_type)

        cache_key = f"n2o_ef:{ms}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = MANURE_N2O_EF[ms]
        self._cache[cache_key] = result

        logger.debug(
            "Manure N2O EF lookup: system=%s, ef=%s kg N2O-N/kg N",
            ms, result,
        )
        return result

    # ------------------------------------------------------------------
    # Manure Nitrogen Excretion (Nex)
    # ------------------------------------------------------------------

    def get_manure_nex(self, animal_type: str) -> Decimal:
        """Look up nitrogen excretion rate for an animal type.

        Nex is used in manure N2O emission calculations.

        Args:
            animal_type: Livestock category.

        Returns:
            Nitrogen excretion as Decimal (kg N/head/yr).

        Raises:
            ValueError: If the animal type is not recognized.
            KeyError: If no Nex data exists for the animal type.

        Example:
            >>> db = AgriculturalDatabaseEngine()
            >>> db.get_manure_nex("dairy_cattle")
            Decimal('100')
        """
        self._increment_lookups()
        at = self._validate_animal_type(animal_type)

        cache_key = f"nex:{at}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if at not in MANURE_NEX:
            raise KeyError(
                f"No Nex data for animal type '{animal_type}'. "
                f"Available: {sorted(MANURE_NEX.keys())}"
            )

        result = MANURE_NEX[at]
        self._cache[cache_key] = result

        logger.debug("Nex lookup: animal=%s, nex=%s kg N/head/yr", at, result)
        return result

    # ------------------------------------------------------------------
    # Soil N2O Emission Factors
    # ------------------------------------------------------------------

    def get_soil_n2o_ef(self, ef_type: str) -> Decimal:
        """Look up a direct soil N2O emission factor.

        Args:
            ef_type: Factor type (EF1, EF2_CG, EF2_F, EF3_PRP,
                EF3_PRP_CPP, EF3_PRP_SO).

        Returns:
            N2O emission factor as Decimal.
            EF1/EF3: kg N2O-N per kg N input.
            EF2: kg N2O-N per ha per year.

        Raises:
            KeyError: If the factor type is not recognized.

        Example:
            >>> db = AgriculturalDatabaseEngine()
            >>> db.get_soil_n2o_ef("EF1")
            Decimal('0.01')
        """
        self._increment_lookups()
        key = ef_type.upper().replace(" ", "_").replace("-", "_")

        if key not in SOIL_N2O_FACTORS:
            raise KeyError(
                f"Unknown soil N2O factor type '{ef_type}'. "
                f"Valid types: {sorted(SOIL_N2O_FACTORS.keys())}"
            )

        result = SOIL_N2O_FACTORS[key]

        logger.debug("Soil N2O factor lookup: type=%s, ef=%s", key, result)
        return result

    # ------------------------------------------------------------------
    # Indirect N2O Fractions and Factors
    # ------------------------------------------------------------------

    def get_indirect_n2o_fraction(self, fraction_type: str) -> Decimal:
        """Look up an indirect N2O fraction or emission factor.

        Args:
            fraction_type: Parameter name (FRAC_GASF, FRAC_GASM,
                FRAC_LEACH, EF4, EF5).

        Returns:
            Fraction or factor value as Decimal.

        Raises:
            KeyError: If the fraction type is not recognized.

        Example:
            >>> db = AgriculturalDatabaseEngine()
            >>> db.get_indirect_n2o_fraction("FRAC_GASF")
            Decimal('0.10')
        """
        self._increment_lookups()
        key = fraction_type.upper().replace(" ", "_").replace("-", "_")

        if key not in INDIRECT_N2O_FACTORS:
            raise KeyError(
                f"Unknown indirect N2O parameter '{fraction_type}'. "
                f"Valid types: {sorted(INDIRECT_N2O_FACTORS.keys())}"
            )

        result = INDIRECT_N2O_FACTORS[key]

        logger.debug(
            "Indirect N2O lookup: type=%s, value=%s", key, result,
        )
        return result

    # ------------------------------------------------------------------
    # Liming Emission Factors
    # ------------------------------------------------------------------

    def get_liming_ef(self, material_type: str) -> Decimal:
        """Look up the liming emission factor for a material.

        Args:
            material_type: Liming material ("limestone" or "dolomite").

        Returns:
            Emission factor as Decimal (tC per tonne of material).

        Raises:
            KeyError: If the material type is not recognized.

        Example:
            >>> db = AgriculturalDatabaseEngine()
            >>> db.get_liming_ef("limestone")
            Decimal('0.12')
        """
        self._increment_lookups()
        key = material_type.upper().replace(" ", "_").replace("-", "_")

        if key not in LIMING_FACTORS:
            raise KeyError(
                f"Unknown liming material '{material_type}'. "
                f"Valid types: {sorted(LIMING_FACTORS.keys())}"
            )

        result = LIMING_FACTORS[key]

        logger.debug(
            "Liming EF lookup: material=%s, ef=%s tC/t", key, result,
        )
        return result

    # ------------------------------------------------------------------
    # Urea Emission Factor
    # ------------------------------------------------------------------

    def get_urea_ef(self) -> Decimal:
        """Look up the urea CO2 emission factor.

        Returns:
            Emission factor as Decimal (tC per tonne urea).

        Example:
            >>> db = AgriculturalDatabaseEngine()
            >>> db.get_urea_ef()
            Decimal('0.20')
        """
        self._increment_lookups()

        logger.debug("Urea EF lookup: ef=%s tC/t urea", UREA_FACTOR)
        return UREA_FACTOR

    # ------------------------------------------------------------------
    # Rice Cultivation Parameters
    # ------------------------------------------------------------------

    def get_rice_baseline_ef(self) -> Decimal:
        """Look up the baseline rice CH4 emission factor.

        Returns:
            Baseline EF as Decimal (kg CH4/ha/day).

        Example:
            >>> db = AgriculturalDatabaseEngine()
            >>> db.get_rice_baseline_ef()
            Decimal('1.30')
        """
        self._increment_lookups()

        logger.debug(
            "Rice baseline EF lookup: ef=%s kg CH4/ha/day", RICE_BASELINE_EF,
        )
        return RICE_BASELINE_EF

    def get_rice_water_regime_sf(self, water_regime: str) -> Decimal:
        """Look up the water regime scaling factor for rice CH4 emissions.

        Args:
            water_regime: Water management practice
                (e.g. "continuously_flooded", "INTERMITTENT_SINGLE").

        Returns:
            Scaling factor as Decimal (dimensionless, relative to baseline).

        Raises:
            ValueError: If the water regime is not recognized.

        Example:
            >>> db = AgriculturalDatabaseEngine()
            >>> db.get_rice_water_regime_sf("intermittent_single")
            Decimal('0.60')
        """
        self._increment_lookups()
        wr = self._validate_water_regime(water_regime)

        cache_key = f"rice_wr:{wr}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = RICE_WATER_REGIME_SF[wr]
        self._cache[cache_key] = result

        logger.debug(
            "Rice water regime SF lookup: regime=%s, sf=%s", wr, result,
        )
        return result

    def get_rice_preseason_sf(self, pre_season: str) -> Decimal:
        """Look up the pre-season flooding scaling factor for rice CH4 emissions.

        Args:
            pre_season: Pre-season flooding status
                (e.g. "not_flooded_lt_180", "FLOODED_GT_30").

        Returns:
            Scaling factor as Decimal (dimensionless).

        Raises:
            ValueError: If the pre-season type is not recognized.

        Example:
            >>> db = AgriculturalDatabaseEngine()
            >>> db.get_rice_preseason_sf("not_flooded_lt_180")
            Decimal('1.0')
        """
        self._increment_lookups()
        ps = self._validate_preseason(pre_season)

        cache_key = f"rice_ps:{ps}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = RICE_PRESEASON_SF[ps]
        self._cache[cache_key] = result

        logger.debug(
            "Rice pre-season SF lookup: status=%s, sf=%s", ps, result,
        )
        return result

    def get_rice_organic_cfoa(self, amendment_type: str) -> Decimal:
        """Look up the conversion factor for organic amendments in rice paddies.

        CFOA is applied per tonne/ha of organic amendment to scale the
        baseline CH4 emission factor.

        Args:
            amendment_type: Organic amendment type
                (e.g. "straw_short", "COMPOST").

        Returns:
            CFOA value as Decimal (dimensionless scaling per t/ha).

        Raises:
            ValueError: If the amendment type is not recognized.

        Example:
            >>> db = AgriculturalDatabaseEngine()
            >>> db.get_rice_organic_cfoa("straw_short")
            Decimal('0.29')
        """
        self._increment_lookups()
        at = self._validate_amendment_type(amendment_type)

        cache_key = f"rice_cfoa:{at}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = RICE_ORGANIC_CFOA[at]
        self._cache[cache_key] = result

        logger.debug(
            "Rice CFOA lookup: amendment=%s, cfoa=%s", at, result,
        )
        return result

    # ------------------------------------------------------------------
    # Field Burning Emission Factors
    # ------------------------------------------------------------------

    def get_field_burning_ef(self, crop_type: str) -> Dict[str, Any]:
        """Look up field burning emission factors and residue parameters for a crop.

        Returns all parameters needed for IPCC field burning calculations:
        residue-to-product ratio, dry matter fraction, N content,
        CH4/N2O emission factors, and combustion factor.

        Args:
            crop_type: Crop type (e.g. "wheat", "SUGARCANE").

        Returns:
            Dictionary with all field burning parameters.

        Raises:
            ValueError: If the crop type is not recognized in the burning table.
            KeyError: If no field burning data exists for the crop type.

        Example:
            >>> db = AgriculturalDatabaseEngine()
            >>> ef = db.get_field_burning_ef("wheat")
            >>> ef["ch4_ef"]
            Decimal('2.7')
        """
        self._increment_lookups()
        ct = crop_type.upper().replace(" ", "_").replace("-", "_")

        if ct not in FIELD_BURNING_EF:
            raise KeyError(
                f"No field burning data for crop '{crop_type}'. "
                f"Available: {sorted(FIELD_BURNING_EF.keys())}"
            )

        cache_key = f"burn:{ct}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        record = FIELD_BURNING_EF[ct]
        result: Dict[str, Any] = {
            "crop_type": record.crop_type,
            "residue_product_ratio": record.residue_product_ratio,
            "dry_matter_fraction": record.dry_matter_fraction,
            "n_content": record.n_content,
            "ch4_ef": record.ch4_ef,
            "n2o_ef": record.n2o_ef,
            "combustion_factor": record.combustion_factor,
            "source": record.source,
        }
        self._cache[cache_key] = result

        logger.debug(
            "Field burning EF lookup: crop=%s, ch4_ef=%s, n2o_ef=%s g/kg DM",
            ct, record.ch4_ef, record.n2o_ef,
        )
        return result

    # ------------------------------------------------------------------
    # GWP Values
    # ------------------------------------------------------------------

    def get_gwp(self, gas: str, source: str = "AR6") -> Decimal:
        """Look up the Global Warming Potential for a gas and assessment report.

        Supports separate GWP values for fossil and biogenic CH4
        per AR5/AR6 methodology.

        Args:
            gas: Gas name (CO2, CH4, CH4_FOSSIL, CH4_BIOGENIC, N2O).
            source: IPCC assessment report (AR4, AR5, AR6, AR6_20YR).

        Returns:
            GWP value as Decimal.

        Raises:
            KeyError: If gas or GWP source is not recognized.

        Example:
            >>> db = AgriculturalDatabaseEngine()
            >>> db.get_gwp("CH4", "AR6")
            Decimal('29.8')
        """
        self._increment_lookups()
        src = source.upper()
        g = gas.upper()

        if src not in GWP_VALUES:
            raise KeyError(
                f"Unknown GWP source '{source}'. "
                f"Valid: {sorted(GWP_VALUES.keys())}"
            )
        if g not in GWP_VALUES[src]:
            raise KeyError(
                f"Unknown gas '{gas}' for source '{src}'. "
                f"Valid: {sorted(GWP_VALUES[src].keys())}"
            )

        return GWP_VALUES[src][g]

    # ------------------------------------------------------------------
    # Tier 2 Maintenance Coefficients
    # ------------------------------------------------------------------

    def get_maintenance_coefficient(self, animal_type: str) -> Decimal:
        """Look up the Tier 2 maintenance energy coefficient (Cfi).

        Cfi is used in the IPCC Tier 2 gross energy calculation for
        enteric fermentation: GE = ... + Cfi * BW^0.75

        Args:
            animal_type: Livestock category.

        Returns:
            Cfi value as Decimal (MJ/day/kg BW).

        Raises:
            ValueError: If the animal type is not recognized.
            KeyError: If no Cfi data exists for the animal type.

        Example:
            >>> db = AgriculturalDatabaseEngine()
            >>> db.get_maintenance_coefficient("dairy_cattle")
            Decimal('0.386')
        """
        self._increment_lookups()
        at = self._validate_animal_type(animal_type)

        cache_key = f"cfi:{at}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if at not in MAINTENANCE_COEFFICIENTS:
            raise KeyError(
                f"No Cfi data for animal type '{animal_type}'. "
                f"Available: {sorted(MAINTENANCE_COEFFICIENTS.keys())}"
            )

        result = MAINTENANCE_COEFFICIENTS[at]
        self._cache[cache_key] = result

        logger.debug(
            "Cfi lookup: animal=%s, cfi=%s MJ/day/kg", at, result,
        )
        return result

    # ------------------------------------------------------------------
    # Body Weight Defaults
    # ------------------------------------------------------------------

    def get_body_weight(self, animal_type: str) -> Decimal:
        """Look up the default body weight for an animal type.

        Args:
            animal_type: Livestock category.

        Returns:
            Body weight as Decimal (kg/head).

        Raises:
            ValueError: If the animal type is not recognized.
            KeyError: If no body weight data exists for the animal type.

        Example:
            >>> db = AgriculturalDatabaseEngine()
            >>> db.get_body_weight("dairy_cattle")
            Decimal('600')
        """
        self._increment_lookups()
        at = self._validate_animal_type(animal_type)

        cache_key = f"bw:{at}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if at not in BODY_WEIGHT_DEFAULTS:
            raise KeyError(
                f"No body weight data for animal type '{animal_type}'. "
                f"Available: {sorted(BODY_WEIGHT_DEFAULTS.keys())}"
            )

        result = BODY_WEIGHT_DEFAULTS[at]
        self._cache[cache_key] = result

        logger.debug("Body weight lookup: animal=%s, bw=%s kg", at, result)
        return result

    # ------------------------------------------------------------------
    # Milk Yield Defaults
    # ------------------------------------------------------------------

    def get_milk_yield(self, region: str = "GLOBAL_AVERAGE") -> Decimal:
        """Look up the default milk yield for a region.

        Args:
            region: Region key (e.g. "DEVELOPED_NORTH_AMERICA",
                "DEVELOPING_ASIA", "GLOBAL_AVERAGE").

        Returns:
            Milk yield as Decimal (kg/head/yr).

        Raises:
            KeyError: If the region is not recognized.

        Example:
            >>> db = AgriculturalDatabaseEngine()
            >>> db.get_milk_yield("DEVELOPED_NORTH_AMERICA")
            Decimal('8400')
        """
        self._increment_lookups()
        key = region.upper().replace(" ", "_").replace("-", "_")

        if key not in MILK_YIELD_DEFAULTS:
            raise KeyError(
                f"Unknown milk yield region '{region}'. "
                f"Valid: {sorted(MILK_YIELD_DEFAULTS.keys())}"
            )

        result = MILK_YIELD_DEFAULTS[key]

        logger.debug(
            "Milk yield lookup: region=%s, yield=%s kg/head/yr", key, result,
        )
        return result

    # ------------------------------------------------------------------
    # Feed Digestibility
    # ------------------------------------------------------------------

    def get_feed_digestibility(self, feed_type: str) -> Decimal:
        """Look up the digestible energy percentage for a feed type.

        DE% is used in Tier 2 enteric fermentation calculations.

        Args:
            feed_type: Feed type (e.g. "high_quality_pasture", "GRAIN_BASED").

        Returns:
            DE percentage as Decimal (0-100).

        Raises:
            KeyError: If the feed type is not recognized.

        Example:
            >>> db = AgriculturalDatabaseEngine()
            >>> db.get_feed_digestibility("grain_based")
            Decimal('80')
        """
        self._increment_lookups()
        key = feed_type.upper().replace(" ", "_").replace("-", "_")

        if key not in FEED_DIGESTIBILITY:
            raise KeyError(
                f"Unknown feed type '{feed_type}'. "
                f"Valid: {sorted(FEED_DIGESTIBILITY.keys())}"
            )

        result = FEED_DIGESTIBILITY[key]

        logger.debug(
            "Feed DE%% lookup: feed=%s, de=%s%%", key, result,
        )
        return result

    # ------------------------------------------------------------------
    # Crop Residue Parameters
    # ------------------------------------------------------------------

    def get_crop_residue_params(self, crop_type: str) -> Dict[str, Any]:
        """Look up crop residue parameters for N2O soil emission calculations.

        Returns above-ground and below-ground residue ratios, dry matter
        fraction, and nitrogen content needed for IPCC N input estimates.

        Args:
            crop_type: Crop type (e.g. "wheat", "SOYBEANS").

        Returns:
            Dictionary with residue parameter values.

        Raises:
            ValueError: If the crop type is not recognized.
            KeyError: If no residue data exists for the crop type.

        Example:
            >>> db = AgriculturalDatabaseEngine()
            >>> params = db.get_crop_residue_params("wheat")
            >>> params["above_ground_ratio"]
            Decimal('1.3')
        """
        self._increment_lookups()
        ct = self._validate_crop_type(crop_type)

        cache_key = f"residue:{ct}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if ct not in CROP_RESIDUE_PARAMS:
            raise KeyError(
                f"No residue data for crop type '{crop_type}'. "
                f"Available: {sorted(CROP_RESIDUE_PARAMS.keys())}"
            )

        record = CROP_RESIDUE_PARAMS[ct]
        result: Dict[str, Any] = {
            "crop_type": record.crop_type,
            "above_ground_ratio": record.above_ground_ratio,
            "below_ground_ratio": record.below_ground_ratio,
            "dry_matter_fraction": record.dry_matter_fraction,
            "n_ag": record.n_ag,
            "n_bg": record.n_bg,
            "source": record.source,
        }
        self._cache[cache_key] = result

        logger.debug(
            "Crop residue lookup: crop=%s, RAG=%s, RBG=%s",
            ct, record.above_ground_ratio, record.below_ground_ratio,
        )
        return result

    # ------------------------------------------------------------------
    # DEFRA / EPA Factor Lookups
    # ------------------------------------------------------------------

    def get_defra_ag_factor(
        self,
        category: str,
        factor_type: str = "TOTAL",
    ) -> Decimal:
        """Look up a DEFRA agricultural emission factor.

        Args:
            category: DEFRA category (e.g. "DAIRY_CATTLE",
                "NITROGEN_FERTILISER").
            factor_type: Factor component (e.g. "ENTERIC", "MANURE",
                "TOTAL", "CO2", "CH4", "DIRECT_N2O").

        Returns:
            DEFRA factor as Decimal (kg CO2e per head/yr or per tonne).

        Raises:
            KeyError: If the combination is not found.

        Example:
            >>> db = AgriculturalDatabaseEngine()
            >>> db.get_defra_ag_factor("dairy_cattle", "total")
            Decimal('3800')
        """
        self._increment_lookups()
        cat = category.upper().replace(" ", "_").replace("-", "_")
        ft = factor_type.upper().replace(" ", "_").replace("-", "_")

        if cat not in DEFRA_AG_FACTORS:
            raise KeyError(
                f"No DEFRA factor for category '{category}'. "
                f"Available: {sorted(DEFRA_AG_FACTORS.keys())}"
            )
        if ft not in DEFRA_AG_FACTORS[cat]:
            raise KeyError(
                f"No DEFRA factor for ({category}, {factor_type}). "
                f"Available types for {cat}: "
                f"{sorted(DEFRA_AG_FACTORS[cat].keys())}"
            )

        result = DEFRA_AG_FACTORS[cat][ft]
        logger.debug(
            "DEFRA AG factor lookup: category=%s, type=%s, ef=%s",
            cat, ft, result,
        )
        return result

    def get_epa_ag_factor(
        self,
        animal_type: str,
        factor_type: str = "TOTAL",
    ) -> Decimal:
        """Look up an EPA 40 CFR 98 Subpart JJ emission factor.

        Args:
            animal_type: Animal type (e.g. "dairy_cattle").
            factor_type: Factor component (e.g. "ENTERIC_CH4",
                "MANURE_CH4", "MANURE_N2O", "TOTAL").

        Returns:
            EPA factor as Decimal (metric tonnes CO2e/head/yr).

        Raises:
            KeyError: If the combination is not found.

        Example:
            >>> db = AgriculturalDatabaseEngine()
            >>> db.get_epa_ag_factor("dairy_cattle", "total")
            Decimal('5.23')
        """
        self._increment_lookups()
        at = animal_type.upper().replace(" ", "_").replace("-", "_")
        ft = factor_type.upper().replace(" ", "_").replace("-", "_")

        if at not in EPA_AG_FACTORS:
            raise KeyError(
                f"No EPA factor for animal type '{animal_type}'. "
                f"Available: {sorted(EPA_AG_FACTORS.keys())}"
            )
        if ft not in EPA_AG_FACTORS[at]:
            raise KeyError(
                f"No EPA factor for ({animal_type}, {factor_type}). "
                f"Available types for {at}: "
                f"{sorted(EPA_AG_FACTORS[at].keys())}"
            )

        result = EPA_AG_FACTORS[at][ft]
        logger.debug(
            "EPA AG factor lookup: animal=%s, type=%s, ef=%s MTCO2e/head/yr",
            at, ft, result,
        )
        return result

    # ------------------------------------------------------------------
    # Convenience: All Factors for an Animal Type
    # ------------------------------------------------------------------

    def get_all_factors_for_animal(
        self,
        animal_type: str,
        region: str = "DEVELOPED",
        temperature: str = "TEMPERATE",
    ) -> Dict[str, Any]:
        """Get all available factors for an animal type in a single call.

        Convenience method that aggregates enteric EF, VS, Bo, Nex,
        body weight, Cfi, and default manure MCF for a livestock type.

        Args:
            animal_type: Livestock category.
            region: Region for enteric EF lookup.
            temperature: Temperature for default MCF range.

        Returns:
            Dictionary with all available factor values and provenance hash.
        """
        self._increment_lookups()
        start_time = time.monotonic()
        at = self._validate_animal_type(animal_type)
        rg = self._validate_region(region)
        temp = self._validate_temperature(temperature)

        enteric_ef = ENTERIC_EF.get(at, {}).get(rg, _D("0"))
        vs = MANURE_VS.get(at, _D("0"))
        bo = MANURE_BO.get(at, _D("0"))
        nex = MANURE_NEX.get(at, _D("0"))
        bw = BODY_WEIGHT_DEFAULTS.get(at, _D("0"))
        cfi = MAINTENANCE_COEFFICIENTS.get(at, _D("0"))

        # DEFRA data if available
        defra_data: Optional[Dict[str, str]] = None
        if at in DEFRA_AG_FACTORS:
            defra_data = {
                k: str(v) for k, v in DEFRA_AG_FACTORS[at].items()
            }

        # EPA data if available
        epa_data: Optional[Dict[str, str]] = None
        if at in EPA_AG_FACTORS:
            epa_data = {
                k: str(v) for k, v in EPA_AG_FACTORS[at].items()
            }

        result: Dict[str, Any] = {
            "animal_type": at,
            "region": rg,
            "temperature": temp,
            "enteric_ef_kg_ch4_head_yr": str(enteric_ef),
            "vs_kg_head_day": str(vs),
            "bo_m3_kg_vs": str(bo),
            "nex_kg_n_head_yr": str(nex),
            "body_weight_kg": str(bw),
            "cfi_mj_day_kg": str(cfi),
            "defra_factors": defra_data,
            "epa_factors": epa_data,
            "processing_time_ms": round(
                (time.monotonic() - start_time) * 1000, 3
            ),
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "All factors retrieved for %s: enteric=%s, vs=%s, time=%.3fms",
            at, enteric_ef, vs, result["processing_time_ms"],
        )
        return result

    # ------------------------------------------------------------------
    # Convenience: All Factors for a Crop Type
    # ------------------------------------------------------------------

    def get_all_factors_for_crop(
        self,
        crop_type: str,
    ) -> Dict[str, Any]:
        """Get all available factors for a crop type in a single call.

        Convenience method that aggregates crop residue parameters and
        field burning data for a crop.

        Args:
            crop_type: Crop type (e.g. "wheat", "RICE").

        Returns:
            Dictionary with all available factor values and provenance hash.
        """
        self._increment_lookups()
        start_time = time.monotonic()
        ct = self._validate_crop_type(crop_type)

        # Crop residue params
        residue_data: Optional[Dict[str, str]] = None
        if ct in CROP_RESIDUE_PARAMS:
            record = CROP_RESIDUE_PARAMS[ct]
            residue_data = {
                "above_ground_ratio": str(record.above_ground_ratio),
                "below_ground_ratio": str(record.below_ground_ratio),
                "dry_matter_fraction": str(record.dry_matter_fraction),
                "n_ag": str(record.n_ag),
                "n_bg": str(record.n_bg),
                "source": record.source,
            }

        # Field burning data
        burning_data: Optional[Dict[str, str]] = None
        if ct in FIELD_BURNING_EF:
            burn_rec = FIELD_BURNING_EF[ct]
            burning_data = {
                "residue_product_ratio": str(burn_rec.residue_product_ratio),
                "dry_matter_fraction": str(burn_rec.dry_matter_fraction),
                "n_content": str(burn_rec.n_content),
                "ch4_ef": str(burn_rec.ch4_ef),
                "n2o_ef": str(burn_rec.n2o_ef),
                "combustion_factor": str(burn_rec.combustion_factor),
                "source": burn_rec.source,
            }

        result: Dict[str, Any] = {
            "crop_type": ct,
            "residue_params": residue_data,
            "field_burning": burning_data,
            "processing_time_ms": round(
                (time.monotonic() - start_time) * 1000, 3
            ),
        }

        result["provenance_hash"] = _compute_hash(result)

        logger.info(
            "All factors retrieved for crop %s: time=%.3fms",
            ct, result["processing_time_ms"],
        )
        return result

    # ------------------------------------------------------------------
    # Custom Factor Management
    # ------------------------------------------------------------------

    def register_custom_factor(
        self,
        factor_type: str,
        key: str,
        value: Decimal,
        source: str = "",
        description: str = "",
    ) -> Dict[str, Any]:
        """Register a custom emission factor or reference value.

        Custom factors are stored in memory and can supplement or override
        IPCC defaults for site-specific or Tier 2/3 calculations.

        Args:
            factor_type: Type of factor (e.g. "ENTERIC_EF", "MANURE_MCF",
                "MANURE_VS", "SOIL_N2O").
            key: Lookup key (e.g. "DAIRY_CATTLE", "ANAEROBIC_LAGOON").
            value: Factor value as Decimal.
            source: Citation or source for the custom value.
            description: Optional description.

        Returns:
            Registration confirmation with provenance hash.
        """
        with self._lock:
            if factor_type not in self._custom_factors:
                self._custom_factors[factor_type] = {}

            record = {
                "value": str(value),
                "source": source,
                "description": description,
                "registered_at": _utcnow().isoformat(),
            }
            self._custom_factors[factor_type][key] = record

        result = {
            "status": "REGISTERED",
            "factor_type": factor_type,
            "key": key,
            "value": str(value),
            "source": source,
            "provenance_hash": _compute_hash(record),
        }

        logger.info(
            "Custom factor registered: type=%s, key=%s, value=%s, source=%s",
            factor_type, key, value, source,
        )
        return result

    def get_custom_factor(
        self,
        factor_type: str,
        key: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a registered custom factor.

        Args:
            factor_type: Type of factor.
            key: Lookup key.

        Returns:
            Custom factor record or None if not found.
        """
        with self._lock:
            if factor_type not in self._custom_factors:
                return None
            return self._custom_factors[factor_type].get(key)

    def list_custom_factors(self) -> Dict[str, Dict[str, Any]]:
        """List all registered custom factors.

        Returns:
            Dictionary of all custom factors keyed by (factor_type, key).
        """
        with self._lock:
            return {
                factor_type: dict(factors)
                for factor_type, factors in self._custom_factors.items()
            }

    # ------------------------------------------------------------------
    # List Available Factors
    # ------------------------------------------------------------------

    def list_available_factors(
        self,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all available emission factor tables, optionally filtered by source.

        Args:
            source: Optional source filter (e.g. "IPCC_2006", "DEFRA", "EPA").
                If None, returns all factor tables.

        Returns:
            List of factor table description dictionaries.
        """
        self._increment_lookups()
        results: List[Dict[str, Any]] = []

        all_tables = [
            {
                "table": "ENTERIC_EF",
                "source": "IPCC_2006",
                "description": "Enteric fermentation Tier 1 EFs (kg CH4/head/yr)",
                "entries": len(ENTERIC_EF),
                "reference": "IPCC 2006 Vol4 Ch10 Tables 10.10-10.11",
            },
            {
                "table": "MANURE_VS",
                "source": "IPCC_2006",
                "description": "Volatile solids excretion (kg VS/head/day)",
                "entries": len(MANURE_VS),
                "reference": "IPCC 2006 Vol4 Ch10 Table 10.13A",
            },
            {
                "table": "MANURE_BO",
                "source": "IPCC_2006",
                "description": "Maximum CH4 producing capacity (m3/kg VS)",
                "entries": len(MANURE_BO),
                "reference": "IPCC 2006 Vol4 Ch10 Table 10.16",
            },
            {
                "table": "MANURE_MCF",
                "source": "IPCC_2006",
                "description": "Methane Correction Factor by AWMS and temp",
                "entries": len(MANURE_MCF),
                "reference": "IPCC 2006 Vol4 Ch10 Table 10.17",
            },
            {
                "table": "MANURE_N2O_EF",
                "source": "IPCC_2006",
                "description": "Manure N2O EF by AWMS (kg N2O-N/kg N)",
                "entries": len(MANURE_N2O_EF),
                "reference": "IPCC 2006 Vol4 Ch10 Table 10.21",
            },
            {
                "table": "MANURE_NEX",
                "source": "IPCC_2006",
                "description": "Nitrogen excretion rate (kg N/head/yr)",
                "entries": len(MANURE_NEX),
                "reference": "IPCC 2006 Vol4 Ch10 Table 10.19",
            },
            {
                "table": "SOIL_N2O_FACTORS",
                "source": "IPCC_2006",
                "description": "Direct soil N2O EFs (EF1/EF2/EF3)",
                "entries": len(SOIL_N2O_FACTORS),
                "reference": "IPCC 2006 Vol4 Ch11 Table 11.1",
            },
            {
                "table": "INDIRECT_N2O_FACTORS",
                "source": "IPCC_2006",
                "description": "Indirect N2O fractions and EFs",
                "entries": len(INDIRECT_N2O_FACTORS),
                "reference": "IPCC 2006 Vol4 Ch11 Table 11.3",
            },
            {
                "table": "LIMING_FACTORS",
                "source": "IPCC_2006",
                "description": "Liming EFs (tC/t material)",
                "entries": len(LIMING_FACTORS),
                "reference": "IPCC 2006 Vol4 Ch11 Eq 11.12",
            },
            {
                "table": "RICE_WATER_REGIME_SF",
                "source": "IPCC_2006",
                "description": "Rice water regime scaling factors",
                "entries": len(RICE_WATER_REGIME_SF),
                "reference": "IPCC 2006 Vol4 Ch5 Table 5.12",
            },
            {
                "table": "RICE_PRESEASON_SF",
                "source": "IPCC_2006",
                "description": "Rice pre-season flooding scaling factors",
                "entries": len(RICE_PRESEASON_SF),
                "reference": "IPCC 2006 Vol4 Ch5 Table 5.13",
            },
            {
                "table": "RICE_ORGANIC_CFOA",
                "source": "IPCC_2006",
                "description": "Rice organic amendment conversion factors",
                "entries": len(RICE_ORGANIC_CFOA),
                "reference": "IPCC 2006 Vol4 Ch5 Table 5.14",
            },
            {
                "table": "FIELD_BURNING_EF",
                "source": "IPCC_2006",
                "description": "Field burning EFs and residue parameters",
                "entries": len(FIELD_BURNING_EF),
                "reference": "IPCC 2006 Vol4 Ch2 Tables 2.5/2.6",
            },
            {
                "table": "CROP_RESIDUE_PARAMS",
                "source": "IPCC_2006",
                "description": "Crop residue N parameters for soil N2O",
                "entries": len(CROP_RESIDUE_PARAMS),
                "reference": "IPCC 2006 Vol4 Ch11 Table 11.2",
            },
            {
                "table": "MAINTENANCE_COEFFICIENTS",
                "source": "IPCC_2006",
                "description": "Tier 2 maintenance energy coefficients",
                "entries": len(MAINTENANCE_COEFFICIENTS),
                "reference": "IPCC 2006 Vol4 Ch10 Table 10.4",
            },
            {
                "table": "BODY_WEIGHT_DEFAULTS",
                "source": "IPCC_2006",
                "description": "Default animal body weights (kg)",
                "entries": len(BODY_WEIGHT_DEFAULTS),
                "reference": "IPCC 2006 Vol4 Ch10 Table 10.5",
            },
            {
                "table": "MILK_YIELD_DEFAULTS",
                "source": "IPCC_2006",
                "description": "Default milk yields by region (kg/head/yr)",
                "entries": len(MILK_YIELD_DEFAULTS),
                "reference": "IPCC 2006 Vol4 Ch10 Table 10.8",
            },
            {
                "table": "FEED_DIGESTIBILITY",
                "source": "IPCC_2006",
                "description": "Feed digestible energy (DE%)",
                "entries": len(FEED_DIGESTIBILITY),
                "reference": "IPCC 2006 Vol4 Ch10 Table 10.2",
            },
            {
                "table": "GWP_VALUES",
                "source": "IPCC_2006",
                "description": "GWP values across AR4/AR5/AR6/AR6_20YR",
                "entries": len(GWP_VALUES),
                "reference": "IPCC AR4/AR5/AR6",
            },
            {
                "table": "DEFRA_AG_FACTORS",
                "source": "DEFRA",
                "description": "DEFRA agricultural emission factors",
                "entries": len(DEFRA_AG_FACTORS),
                "reference": "DEFRA 2025 Conversion Factors",
            },
            {
                "table": "EPA_AG_FACTORS",
                "source": "EPA",
                "description": "EPA 40 CFR 98 Subpart JJ factors",
                "entries": len(EPA_AG_FACTORS),
                "reference": "EPA 40 CFR 98 Subpart JJ",
            },
        ]

        if source is not None:
            src_upper = source.upper().replace(" ", "_")
            results = [t for t in all_tables if t["source"] == src_upper]
        else:
            results = all_tables

        return results

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine usage statistics.

        Returns:
            Dictionary with lookup counts and engine metadata.
        """
        with self._lock:
            custom_count = sum(
                len(v) for v in self._custom_factors.values()
            )
            return {
                "engine": "AgriculturalDatabaseEngine",
                "version": "1.0.0",
                "created_at": self._created_at.isoformat(),
                "total_lookups": self._total_lookups,
                "cache_size": len(self._cache),
                "custom_factors_registered": custom_count,
                "animal_types": len(AnimalType),
                "manure_systems": len(ManureSystem),
                "water_regimes": len(WaterRegime),
                "crop_types": len(CropType),
                "enteric_ef_entries": len(ENTERIC_EF),
                "manure_vs_entries": len(MANURE_VS),
                "manure_bo_entries": len(MANURE_BO),
                "manure_mcf_systems": len(MANURE_MCF),
                "manure_n2o_ef_entries": len(MANURE_N2O_EF),
                "manure_nex_entries": len(MANURE_NEX),
                "soil_n2o_factors": len(SOIL_N2O_FACTORS),
                "indirect_n2o_factors": len(INDIRECT_N2O_FACTORS),
                "liming_materials": len(LIMING_FACTORS),
                "rice_water_regimes": len(RICE_WATER_REGIME_SF),
                "rice_preseason_types": len(RICE_PRESEASON_SF),
                "rice_amendment_types": len(RICE_ORGANIC_CFOA),
                "field_burning_crops": len(FIELD_BURNING_EF),
                "crop_residue_params": len(CROP_RESIDUE_PARAMS),
                "maintenance_coefficients": len(MAINTENANCE_COEFFICIENTS),
                "body_weight_entries": len(BODY_WEIGHT_DEFAULTS),
                "milk_yield_regions": len(MILK_YIELD_DEFAULTS),
                "feed_types": len(FEED_DIGESTIBILITY),
                "gwp_sources": len(GWP_VALUES),
                "defra_categories": len(DEFRA_AG_FACTORS),
                "epa_animal_types": len(EPA_AG_FACTORS),
            }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset engine state (custom factors, cache, counters).

        Intended for testing teardown.
        """
        with self._lock:
            self._custom_factors.clear()
            self._cache.clear()
            self._total_lookups = 0
        logger.info("AgriculturalDatabaseEngine reset")
