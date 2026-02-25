# -*- coding: utf-8 -*-
"""
Recycling, Composting & Anaerobic Digestion Engine - AGENT-MRV-018 Engine 4

GHG Protocol Scope 3 Category 5 recycling/composting/AD emissions calculator.

This engine covers three distinct waste treatment pathways:

1. **Recycling** (GHG Protocol Cut-Off Approach):
   - Only transport-to-facility and MRF sorting emissions counted in Category 5
   - Avoided emissions from displaced virgin production reported as a MEMO item
   - Open-loop vs closed-loop distinction with quality factor for downcycling
   - Emissions = transport_emissions + mrf_sorting_emissions

2. **Composting** (IPCC 2006 Vol 5 Ch 4):
   - CH4 = M x EF_CH4 (default: 4 g/kg wet, 10 g/kg dry for industrial)
   - N2O = M x EF_N2O (default: 0.3 g/kg wet, 0.6 g/kg dry for industrial)
   - Home composting uses higher factors (CH4: 10/20, N2O: 0.6/1.2 g/kg)
   - Ranges: CH4 0.08-20.0 g/kg, N2O 0.06-0.6 g/kg

3. **Anaerobic Digestion** (IPCC 2019 Refinement):
   - CH4 leakage from digester (2-7% by plant type)
   - Fugitive emissions from digestate storage (gastight vs open)
   - Combustion emissions from biogas CHP
   - Biogas yield varies by waste category (60-200+ m3/tonne)

All calculations use Decimal arithmetic for regulatory precision.
Thread-safe singleton pattern for concurrent pipeline use.

References:
    - GHG Protocol Technical Guidance for Calculating Scope 3 Emissions, Category 5
    - IPCC 2006 Guidelines for National Greenhouse Gas Inventories, Vol 5 Ch 4
    - IPCC 2019 Refinement to the 2006 Guidelines
    - EPA WARM v16 (Waste Reduction Model)
    - DEFRA/DESNZ GHG Reporting Conversion Factors

Example:
    >>> engine = get_recycling_composting_engine()
    >>> result = engine.calculate_recycling(RecyclingInput(
    ...     mass_tonnes=Decimal("50.0"),
    ...     waste_category=WasteCategory.PAPER_CARDBOARD,
    ...     recycling_type=RecyclingType.CLOSED_LOOP,
    ... ))
    >>> result.treatment_emissions_co2e
    Decimal('1.050000')

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-005
"""

import hashlib
import logging
import threading
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple, Union

from greenlang.waste_generated.models import (
    AGENT_ID,
    VERSION,
    TABLE_PREFIX,
    # Enumerations
    WasteCategory,
    WasteTreatmentMethod,
    RecyclingType,
    EFSource,
    GWPVersion,
    DataQualityTier,
    EmissionGas,
    # Constant tables
    GWP_VALUES,
    COMPOSTING_EF,
    AD_LEAKAGE_RATES,
    EPA_WARM_FACTORS,
    DEFRA_WASTE_FACTORS,
    # Input models
    RecyclingInput,
    CompostingInput,
    AnaerobicDigestionInput,
    # Output models
    RecyclingCompostingResult,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

ENGINE_ID: str = "recycling_composting_engine"
ENGINE_VERSION: str = "1.0.0"

# Decimal precision for rounding (6 decimal places for tonne-level precision)
PRECISION: int = 6
ROUNDING: str = ROUND_HALF_UP

# Conversion factors
KG_PER_TONNE: Decimal = Decimal("1000")
G_PER_KG: Decimal = Decimal("1000")
TONNES_PER_KG: Decimal = Decimal("0.001")
TONNES_PER_G: Decimal = Decimal("0.000001")

# EPA WARM conversion: MTCO2e/short_ton to kgCO2e/tonne
# 1 MTCO2e = 1000 kgCO2e; 1 short ton = 0.90718 tonnes
# So multiply MTCO2e/short_ton by 1000/0.90718 = 1102.31
EPA_WARM_TO_KG_PER_TONNE: Decimal = Decimal("1102.31")

# Default transport emission factors (kgCO2e per tonne-km)
# GHG Protocol / DEFRA road freight defaults
TRANSPORT_EF_ROAD: Decimal = Decimal("0.10694")  # kgCO2e per tonne-km (rigid HGV)
TRANSPORT_EF_RAIL: Decimal = Decimal("0.02726")  # kgCO2e per tonne-km (rail freight)
TRANSPORT_EF_SHIP: Decimal = Decimal("0.01616")  # kgCO2e per tonne-km (bulk carrier)

# Default MRF (Materials Recovery Facility) sorting emission factors
# kgCO2e per tonne of material processed through MRF
MRF_EF_DEFAULT: Decimal = Decimal("21.0")  # DEFRA default for MRF sorting

# ==============================================================================
# DATA TABLE: RECYCLING PROCESS EMISSION FACTORS
# ==============================================================================

# Recycling process EFs: transport + sorting only (kgCO2e per tonne)
# These represent the ACTUAL emissions in Category 5 under cut-off approach.
# Source: DEFRA/DESNZ 2025, EPA WARM v16 (processing-only component)
RECYCLING_PROCESS_EF: Dict[WasteCategory, Dict[str, Decimal]] = {
    WasteCategory.PAPER_CARDBOARD: {
        "transport_and_sorting": Decimal("21.0"),
        "mrf_sorting": Decimal("18.0"),
        "reprocessing": Decimal("3.0"),
        "source": "defra_beis",
        "source_year": Decimal("2025"),
    },
    WasteCategory.PLASTICS_HDPE: {
        "transport_and_sorting": Decimal("21.0"),
        "mrf_sorting": Decimal("19.0"),
        "reprocessing": Decimal("2.0"),
        "source": "defra_beis",
        "source_year": Decimal("2025"),
    },
    WasteCategory.PLASTICS_LDPE: {
        "transport_and_sorting": Decimal("21.0"),
        "mrf_sorting": Decimal("19.0"),
        "reprocessing": Decimal("2.0"),
        "source": "defra_beis",
        "source_year": Decimal("2025"),
    },
    WasteCategory.PLASTICS_PET: {
        "transport_and_sorting": Decimal("21.0"),
        "mrf_sorting": Decimal("19.0"),
        "reprocessing": Decimal("2.0"),
        "source": "defra_beis",
        "source_year": Decimal("2025"),
    },
    WasteCategory.PLASTICS_PP: {
        "transport_and_sorting": Decimal("21.0"),
        "mrf_sorting": Decimal("19.0"),
        "reprocessing": Decimal("2.0"),
        "source": "defra_beis",
        "source_year": Decimal("2025"),
    },
    WasteCategory.PLASTICS_MIXED: {
        "transport_and_sorting": Decimal("21.0"),
        "mrf_sorting": Decimal("19.0"),
        "reprocessing": Decimal("2.0"),
        "source": "defra_beis",
        "source_year": Decimal("2025"),
    },
    WasteCategory.GLASS: {
        "transport_and_sorting": Decimal("21.0"),
        "mrf_sorting": Decimal("15.0"),
        "reprocessing": Decimal("6.0"),
        "source": "defra_beis",
        "source_year": Decimal("2025"),
    },
    WasteCategory.METALS_ALUMINUM: {
        "transport_and_sorting": Decimal("21.0"),
        "mrf_sorting": Decimal("16.0"),
        "reprocessing": Decimal("5.0"),
        "source": "defra_beis",
        "source_year": Decimal("2025"),
    },
    WasteCategory.METALS_STEEL: {
        "transport_and_sorting": Decimal("21.0"),
        "mrf_sorting": Decimal("16.0"),
        "reprocessing": Decimal("5.0"),
        "source": "defra_beis",
        "source_year": Decimal("2025"),
    },
    WasteCategory.METALS_MIXED: {
        "transport_and_sorting": Decimal("21.0"),
        "mrf_sorting": Decimal("16.0"),
        "reprocessing": Decimal("5.0"),
        "source": "defra_beis",
        "source_year": Decimal("2025"),
    },
    WasteCategory.WOOD: {
        "transport_and_sorting": Decimal("21.0"),
        "mrf_sorting": Decimal("14.0"),
        "reprocessing": Decimal("7.0"),
        "source": "defra_beis",
        "source_year": Decimal("2025"),
    },
    WasteCategory.TEXTILES: {
        "transport_and_sorting": Decimal("21.0"),
        "mrf_sorting": Decimal("17.0"),
        "reprocessing": Decimal("4.0"),
        "source": "defra_beis",
        "source_year": Decimal("2025"),
    },
    WasteCategory.ELECTRONICS: {
        "transport_and_sorting": Decimal("25.0"),
        "mrf_sorting": Decimal("20.0"),
        "reprocessing": Decimal("5.0"),
        "source": "defra_beis",
        "source_year": Decimal("2025"),
    },
    WasteCategory.CONSTRUCTION_DEMOLITION: {
        "transport_and_sorting": Decimal("21.0"),
        "mrf_sorting": Decimal("13.0"),
        "reprocessing": Decimal("8.0"),
        "source": "defra_beis",
        "source_year": Decimal("2025"),
    },
    WasteCategory.RUBBER_LEATHER: {
        "transport_and_sorting": Decimal("21.0"),
        "mrf_sorting": Decimal("18.0"),
        "reprocessing": Decimal("3.0"),
        "source": "defra_beis",
        "source_year": Decimal("2025"),
    },
    WasteCategory.MIXED_MSW: {
        "transport_and_sorting": Decimal("21.0"),
        "mrf_sorting": Decimal("18.0"),
        "reprocessing": Decimal("3.0"),
        "source": "defra_beis",
        "source_year": Decimal("2025"),
    },
    WasteCategory.OTHER: {
        "transport_and_sorting": Decimal("21.0"),
        "mrf_sorting": Decimal("18.0"),
        "reprocessing": Decimal("3.0"),
        "source": "defra_beis",
        "source_year": Decimal("2025"),
    },
}


# ==============================================================================
# DATA TABLE: VIRGIN PRODUCTION EMISSION FACTORS
# ==============================================================================

# Virgin production EFs by material (kgCO2e per tonne of virgin material)
# Used to calculate AVOIDED emissions (memo item only, never deducted).
# Source: EPA WARM v16 (net GHG benefit from recycling), ecoinvent 3.9
VIRGIN_PRODUCTION_EF: Dict[WasteCategory, Dict[str, Decimal]] = {
    WasteCategory.PAPER_CARDBOARD: {
        "virgin_production_ef": Decimal("3428"),  # kgCO2e/tonne virgin paper
        "recycled_production_ef": Decimal("1710"),  # kgCO2e/tonne recycled paper
        "avoided_ef": Decimal("1718"),  # Difference = avoided
        "source": "epa_warm_v16",
    },
    WasteCategory.PLASTICS_HDPE: {
        "virgin_production_ef": Decimal("1520"),
        "recycled_production_ef": Decimal("660"),
        "avoided_ef": Decimal("860"),
        "source": "epa_warm_v16",
    },
    WasteCategory.PLASTICS_LDPE: {
        "virgin_production_ef": Decimal("1710"),
        "recycled_production_ef": Decimal("730"),
        "avoided_ef": Decimal("980"),
        "source": "epa_warm_v16",
    },
    WasteCategory.PLASTICS_PET: {
        "virgin_production_ef": Decimal("2730"),
        "recycled_production_ef": Decimal("1020"),
        "avoided_ef": Decimal("1710"),
        "source": "epa_warm_v16",
    },
    WasteCategory.PLASTICS_PP: {
        "virgin_production_ef": Decimal("1490"),
        "recycled_production_ef": Decimal("620"),
        "avoided_ef": Decimal("870"),
        "source": "epa_warm_v16",
    },
    WasteCategory.PLASTICS_MIXED: {
        "virgin_production_ef": Decimal("1860"),
        "recycled_production_ef": Decimal("780"),
        "avoided_ef": Decimal("1080"),
        "source": "epa_warm_v16",
    },
    WasteCategory.GLASS: {
        "virgin_production_ef": Decimal("843"),
        "recycled_production_ef": Decimal("535"),
        "avoided_ef": Decimal("308"),
        "source": "epa_warm_v16",
    },
    WasteCategory.METALS_ALUMINUM: {
        "virgin_production_ef": Decimal("11890"),
        "recycled_production_ef": Decimal("830"),
        "avoided_ef": Decimal("11060"),
        "source": "epa_warm_v16",
    },
    WasteCategory.METALS_STEEL: {
        "virgin_production_ef": Decimal("2890"),
        "recycled_production_ef": Decimal("870"),
        "avoided_ef": Decimal("2020"),
        "source": "epa_warm_v16",
    },
    WasteCategory.METALS_MIXED: {
        "virgin_production_ef": Decimal("7390"),
        "recycled_production_ef": Decimal("850"),
        "avoided_ef": Decimal("6540"),
        "source": "epa_warm_v16",
    },
    WasteCategory.WOOD: {
        "virgin_production_ef": Decimal("423"),
        "recycled_production_ef": Decimal("150"),
        "avoided_ef": Decimal("273"),
        "source": "epa_warm_v16",
    },
    WasteCategory.TEXTILES: {
        "virgin_production_ef": Decimal("5300"),
        "recycled_production_ef": Decimal("2780"),
        "avoided_ef": Decimal("2520"),
        "source": "epa_warm_v16",
    },
    WasteCategory.ELECTRONICS: {
        "virgin_production_ef": Decimal("8500"),
        "recycled_production_ef": Decimal("3200"),
        "avoided_ef": Decimal("5300"),
        "source": "epa_warm_v16",
    },
    WasteCategory.CONSTRUCTION_DEMOLITION: {
        "virgin_production_ef": Decimal("265"),
        "recycled_production_ef": Decimal("78"),
        "avoided_ef": Decimal("187"),
        "source": "epa_warm_v16",
    },
    WasteCategory.RUBBER_LEATHER: {
        "virgin_production_ef": Decimal("3100"),
        "recycled_production_ef": Decimal("1500"),
        "avoided_ef": Decimal("1600"),
        "source": "epa_warm_v16",
    },
}


# ==============================================================================
# DATA TABLE: COMPOSTING EMISSION FACTORS (IPCC 2006 Vol 5 Table 4.1)
# ==============================================================================

# Extended composting EF table with ranges (g gas per kg waste)
COMPOSTING_EF_EXTENDED: Dict[str, Dict[str, Decimal]] = {
    "industrial_wet": {
        "ch4_default": Decimal("4.0"),
        "ch4_min": Decimal("0.08"),
        "ch4_max": Decimal("20.0"),
        "n2o_default": Decimal("0.3"),
        "n2o_min": Decimal("0.06"),
        "n2o_max": Decimal("0.6"),
    },
    "industrial_dry": {
        "ch4_default": Decimal("10.0"),
        "ch4_min": Decimal("0.20"),
        "ch4_max": Decimal("50.0"),
        "n2o_default": Decimal("0.6"),
        "n2o_min": Decimal("0.15"),
        "n2o_max": Decimal("1.5"),
    },
    "home_wet": {
        "ch4_default": Decimal("10.0"),
        "ch4_min": Decimal("0.20"),
        "ch4_max": Decimal("50.0"),
        "n2o_default": Decimal("0.6"),
        "n2o_min": Decimal("0.12"),
        "n2o_max": Decimal("1.2"),
    },
    "home_dry": {
        "ch4_default": Decimal("20.0"),
        "ch4_min": Decimal("0.40"),
        "ch4_max": Decimal("100.0"),
        "n2o_default": Decimal("1.2"),
        "n2o_min": Decimal("0.24"),
        "n2o_max": Decimal("2.4"),
    },
}

# Composting waste-specific mass reduction factors
# Fraction of input mass remaining after composting (rest is water/CO2 loss)
COMPOSTING_MASS_REDUCTION: Dict[WasteCategory, Decimal] = {
    WasteCategory.FOOD_WASTE: Decimal("0.35"),  # 65% mass loss
    WasteCategory.GARDEN_WASTE: Decimal("0.45"),  # 55% mass loss
    WasteCategory.PAPER_CARDBOARD: Decimal("0.40"),  # 60% mass loss
    WasteCategory.WOOD: Decimal("0.50"),  # 50% mass loss
    WasteCategory.MIXED_MSW: Decimal("0.40"),  # 60% mass loss (organic fraction)
    WasteCategory.OTHER: Decimal("0.40"),  # 60% default
}

# Compatible waste categories for composting
COMPOSTABLE_CATEGORIES: set = {
    WasteCategory.FOOD_WASTE,
    WasteCategory.GARDEN_WASTE,
    WasteCategory.PAPER_CARDBOARD,
    WasteCategory.WOOD,
    WasteCategory.MIXED_MSW,
    WasteCategory.OTHER,
}


# ==============================================================================
# DATA TABLE: ANAEROBIC DIGESTION PARAMETERS
# ==============================================================================

# Biogas yield by waste type (m3 biogas per tonne wet waste)
# Source: IPCC 2019 Refinement, IEA Bioenergy Task 37
BIOGAS_YIELD: Dict[WasteCategory, Dict[str, Decimal]] = {
    WasteCategory.FOOD_WASTE: {
        "yield_m3_per_tonne": Decimal("150"),  # Typical range 100-200
        "yield_min": Decimal("100"),
        "yield_max": Decimal("200"),
        "vs_fraction": Decimal("0.87"),  # Volatile solids as fraction of TS
        "ts_fraction": Decimal("0.25"),  # Total solids (dry matter) fraction
    },
    WasteCategory.GARDEN_WASTE: {
        "yield_m3_per_tonne": Decimal("60"),
        "yield_min": Decimal("40"),
        "yield_max": Decimal("100"),
        "vs_fraction": Decimal("0.85"),
        "ts_fraction": Decimal("0.35"),
    },
    WasteCategory.PAPER_CARDBOARD: {
        "yield_m3_per_tonne": Decimal("90"),
        "yield_min": Decimal("50"),
        "yield_max": Decimal("130"),
        "vs_fraction": Decimal("0.90"),
        "ts_fraction": Decimal("0.85"),
    },
    WasteCategory.MIXED_MSW: {
        "yield_m3_per_tonne": Decimal("100"),
        "yield_min": Decimal("60"),
        "yield_max": Decimal("160"),
        "vs_fraction": Decimal("0.70"),
        "ts_fraction": Decimal("0.40"),
    },
    WasteCategory.WOOD: {
        "yield_m3_per_tonne": Decimal("40"),
        "yield_min": Decimal("20"),
        "yield_max": Decimal("80"),
        "vs_fraction": Decimal("0.95"),
        "ts_fraction": Decimal("0.80"),
    },
    WasteCategory.OTHER: {
        "yield_m3_per_tonne": Decimal("80"),
        "yield_min": Decimal("40"),
        "yield_max": Decimal("120"),
        "vs_fraction": Decimal("0.80"),
        "ts_fraction": Decimal("0.30"),
    },
}

# AD leakage rates by plant type (fraction of total CH4 produced)
# Source: IPCC 2019 Refinement to the 2006 IPCC Guidelines
AD_LEAKAGE_RATES_EXTENDED: Dict[str, Dict[str, Decimal]] = {
    "biowaste": {
        "default": Decimal("0.028"),  # 2.8% leakage
        "min": Decimal("0.01"),
        "max": Decimal("0.05"),
        "description": "Modern enclosed biowaste AD plant",
    },
    "wastewater": {
        "default": Decimal("0.07"),  # 7.0% leakage
        "min": Decimal("0.03"),
        "max": Decimal("0.10"),
        "description": "Wastewater sludge AD plant",
    },
    "manure": {
        "default": Decimal("0.037"),  # 3.7% leakage
        "min": Decimal("0.02"),
        "max": Decimal("0.06"),
        "description": "Agricultural manure AD plant",
    },
    "energy_crop": {
        "default": Decimal("0.019"),  # 1.9% leakage
        "min": Decimal("0.01"),
        "max": Decimal("0.03"),
        "description": "Energy crop silage AD plant",
    },
    "dry_fermentation": {
        "default": Decimal("0.035"),  # 3.5% leakage
        "min": Decimal("0.02"),
        "max": Decimal("0.05"),
        "description": "Dry fermentation AD plant (batch/plug-flow)",
    },
    "thermophilic": {
        "default": Decimal("0.022"),  # 2.2% leakage
        "min": Decimal("0.01"),
        "max": Decimal("0.04"),
        "description": "Thermophilic (high temperature) AD plant",
    },
}

# Biogas CH4 content ranges by waste type
BIOGAS_CH4_CONTENT: Dict[str, Dict[str, Decimal]] = {
    "food_waste": {
        "default": Decimal("0.60"),
        "min": Decimal("0.55"),
        "max": Decimal("0.65"),
    },
    "garden_waste": {
        "default": Decimal("0.55"),
        "min": Decimal("0.50"),
        "max": Decimal("0.60"),
    },
    "mixed_waste": {
        "default": Decimal("0.58"),
        "min": Decimal("0.52"),
        "max": Decimal("0.65"),
    },
    "default": {
        "default": Decimal("0.60"),
        "min": Decimal("0.55"),
        "max": Decimal("0.65"),
    },
}

# Digestate emission factors (open storage vs gastight)
# kgCO2e per tonne of digestate
DIGESTATE_EF: Dict[str, Dict[str, Decimal]] = {
    "gastight": {
        "ch4_ef_kg_per_tonne": Decimal("0.5"),  # Minimal fugitive from sealed tank
        "n2o_ef_kg_per_tonne": Decimal("0.02"),
        "total_co2e_per_tonne": Decimal("15.5"),  # Using AR5 GWPs
    },
    "open_storage": {
        "ch4_ef_kg_per_tonne": Decimal("8.0"),  # Significant fugitive from open lagoon
        "n2o_ef_kg_per_tonne": Decimal("0.15"),
        "total_co2e_per_tonne": Decimal("263.8"),  # Using AR5 GWPs
    },
    "covered_lagoon": {
        "ch4_ef_kg_per_tonne": Decimal("3.0"),  # Partial capture
        "n2o_ef_kg_per_tonne": Decimal("0.08"),
        "total_co2e_per_tonne": Decimal("105.2"),  # Using AR5 GWPs
    },
}

# Biogas combustion emission factors (CHP)
# Source: IPCC 2006 Vol 2 (Stationary Combustion)
BIOGAS_COMBUSTION_EF: Dict[str, Decimal] = {
    "co2_kg_per_m3_biogas": Decimal("0.0"),  # Biogenic CO2, not counted
    "ch4_kg_per_m3_biogas": Decimal("0.000054"),  # Incomplete combustion
    "n2o_kg_per_m3_biogas": Decimal("0.0000011"),  # Trace N2O from CHP
    "biogenic_co2_kg_per_m3": Decimal("1.145"),  # Biogenic CO2 (memo item)
}

# CH4 density at STP (kg/m3)
CH4_DENSITY_KG_PER_M3: Decimal = Decimal("0.717")

# Biogas energy content (kWh per m3 biogas at ~60% CH4)
BIOGAS_ENERGY_CONTENT: Dict[str, Decimal] = {
    "kwh_per_m3_raw": Decimal("6.0"),  # Raw biogas (~60% CH4)
    "kwh_per_m3_55pct": Decimal("5.5"),
    "kwh_per_m3_60pct": Decimal("6.0"),
    "kwh_per_m3_65pct": Decimal("6.5"),
    "chp_electrical_efficiency": Decimal("0.35"),  # Typical CHP electrical efficiency
    "chp_thermal_efficiency": Decimal("0.50"),  # Typical CHP thermal efficiency
    "chp_total_efficiency": Decimal("0.85"),
}

# Compatible waste categories for AD
AD_COMPATIBLE_CATEGORIES: set = {
    WasteCategory.FOOD_WASTE,
    WasteCategory.GARDEN_WASTE,
    WasteCategory.PAPER_CARDBOARD,
    WasteCategory.MIXED_MSW,
    WasteCategory.WOOD,
    WasteCategory.OTHER,
}

# Recyclable waste categories
RECYCLABLE_CATEGORIES: set = {
    WasteCategory.PAPER_CARDBOARD,
    WasteCategory.PLASTICS_HDPE,
    WasteCategory.PLASTICS_LDPE,
    WasteCategory.PLASTICS_PET,
    WasteCategory.PLASTICS_PP,
    WasteCategory.PLASTICS_MIXED,
    WasteCategory.GLASS,
    WasteCategory.METALS_ALUMINUM,
    WasteCategory.METALS_STEEL,
    WasteCategory.METALS_MIXED,
    WasteCategory.WOOD,
    WasteCategory.TEXTILES,
    WasteCategory.ELECTRONICS,
    WasteCategory.CONSTRUCTION_DEMOLITION,
    WasteCategory.RUBBER_LEATHER,
}

# Transport mode emission factors
TRANSPORT_MODE_EF: Dict[str, Decimal] = {
    "road": TRANSPORT_EF_ROAD,
    "rail": TRANSPORT_EF_RAIL,
    "ship": TRANSPORT_EF_SHIP,
    "road_articulated": Decimal("0.08976"),  # Articulated HGV
    "road_rigid_small": Decimal("0.24682"),  # Rigid HGV <7.5t
    "road_rigid_large": Decimal("0.10694"),  # Rigid HGV >17t
    "road_van": Decimal("0.60459"),  # Light commercial vehicle
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def _round_decimal(value: Decimal, precision: int = PRECISION) -> Decimal:
    """
    Round a Decimal to the specified number of decimal places.

    Args:
        value: Decimal value to round.
        precision: Number of decimal places.

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * precision
    return value.quantize(Decimal(quantize_str), rounding=ROUNDING)


def _compute_hash(data: str) -> str:
    """
    Compute SHA-256 hash for provenance tracking.

    Args:
        data: String data to hash.

    Returns:
        SHA-256 hex digest.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _generate_id(prefix: str = "rc") -> str:
    """
    Generate a unique identifier with prefix.

    Args:
        prefix: Identifier prefix.

    Returns:
        Unique ID string.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ==============================================================================
# ENGINE CLASS
# ==============================================================================


class RecyclingCompostingEngine:
    """
    Engine 4: Recycling, Composting and Anaerobic Digestion emissions calculator.

    Implements GHG Protocol Scope 3 Category 5 calculations for three
    waste treatment pathways: recycling (open-loop / closed-loop),
    composting (industrial / home), and anaerobic digestion.

    Thread-safe singleton. All numeric calculations are deterministic
    (zero-hallucination) using Decimal arithmetic.

    Attributes:
        _gwp_version: GWP assessment report version for CO2e conversion.
        _gwp_ch4: GWP value for CH4.
        _gwp_n2o: GWP value for N2O.
        _ef_source: Default emission factor source.
        _initialized: Whether engine is fully initialized.

    Example:
        >>> engine = RecyclingCompostingEngine()
        >>> result = engine.calculate_recycling(RecyclingInput(
        ...     mass_tonnes=Decimal("100"),
        ...     waste_category=WasteCategory.METALS_ALUMINUM,
        ...     recycling_type=RecyclingType.CLOSED_LOOP,
        ... ))
        >>> assert result.treatment_emissions_co2e > Decimal("0")
        >>> assert result.avoided_emissions_memo_co2e is not None
    """

    _instance: Optional["RecyclingCompostingEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "RecyclingCompostingEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        gwp_version: GWPVersion = GWPVersion.AR5,
        ef_source: EFSource = EFSource.DEFRA_BEIS,
    ) -> None:
        """
        Initialize the RecyclingCompostingEngine.

        Args:
            gwp_version: IPCC GWP assessment report version for CO2e conversion.
            ef_source: Default emission factor source preference.
        """
        if self._initialized:
            return

        self._gwp_version: GWPVersion = gwp_version
        self._ef_source: EFSource = ef_source

        # Resolve GWP values
        gwp_table = GWP_VALUES.get(gwp_version, GWP_VALUES[GWPVersion.AR5])
        self._gwp_ch4: Decimal = gwp_table["ch4"]
        self._gwp_n2o: Decimal = gwp_table["n2o"]

        self._initialized = True
        logger.info(
            "RecyclingCompostingEngine initialized: gwp=%s, ef_source=%s, "
            "GWP_CH4=%s, GWP_N2O=%s",
            gwp_version.value,
            ef_source.value,
            self._gwp_ch4,
            self._gwp_n2o,
        )

    # ==========================================================================
    # RESET (for testing)
    # ==========================================================================

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance. Used in testing only.

        This clears the cached singleton so that the next call to
        ``RecyclingCompostingEngine()`` creates a fresh instance.
        """
        with cls._lock:
            cls._instance = None

    # ==========================================================================
    # DISPATCHER
    # ==========================================================================

    def calculate(
        self,
        treatment_method: str,
        input_data: Union[RecyclingInput, CompostingInput, AnaerobicDigestionInput],
    ) -> RecyclingCompostingResult:
        """
        Dispatch to the appropriate treatment calculation method.

        This is the primary entry point for the engine. It routes to the
        correct calculation method based on treatment_method.

        Args:
            treatment_method: One of 'recycling', 'composting', 'anaerobic_digestion'.
            input_data: Treatment-specific input model.

        Returns:
            RecyclingCompostingResult with emissions and avoided-emissions memo.

        Raises:
            ValueError: If treatment_method is not recognized or input type mismatches.
        """
        start_time = datetime.now(timezone.utc)
        logger.info(
            "RecyclingCompostingEngine.calculate: method=%s", treatment_method
        )

        method_map = {
            "recycling": self._dispatch_recycling,
            "composting": self._dispatch_composting,
            "anaerobic_digestion": self._dispatch_ad,
        }

        handler = method_map.get(treatment_method)
        if handler is None:
            valid = ", ".join(sorted(method_map.keys()))
            raise ValueError(
                f"Unknown treatment_method '{treatment_method}'. "
                f"Valid options: {valid}"
            )

        result = handler(input_data)

        elapsed_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000
        logger.info(
            "RecyclingCompostingEngine.calculate completed: method=%s, "
            "treatment_co2e=%s, elapsed_ms=%.2f",
            treatment_method,
            result.treatment_emissions_co2e,
            elapsed_ms,
        )
        return result

    def _dispatch_recycling(
        self,
        input_data: Union[RecyclingInput, CompostingInput, AnaerobicDigestionInput],
    ) -> RecyclingCompostingResult:
        """Dispatch recycling calculation with type check."""
        if not isinstance(input_data, RecyclingInput):
            raise ValueError(
                f"Expected RecyclingInput for 'recycling', got {type(input_data).__name__}"
            )
        return self.calculate_recycling(input_data)

    def _dispatch_composting(
        self,
        input_data: Union[RecyclingInput, CompostingInput, AnaerobicDigestionInput],
    ) -> RecyclingCompostingResult:
        """Dispatch composting calculation with type check."""
        if not isinstance(input_data, CompostingInput):
            raise ValueError(
                f"Expected CompostingInput for 'composting', got {type(input_data).__name__}"
            )
        return self.calculate_composting(input_data)

    def _dispatch_ad(
        self,
        input_data: Union[RecyclingInput, CompostingInput, AnaerobicDigestionInput],
    ) -> RecyclingCompostingResult:
        """Dispatch anaerobic digestion calculation with type check."""
        if not isinstance(input_data, AnaerobicDigestionInput):
            raise ValueError(
                f"Expected AnaerobicDigestionInput for 'anaerobic_digestion', "
                f"got {type(input_data).__name__}"
            )
        return self.calculate_anaerobic_digestion(input_data)

    # ==========================================================================
    # BATCH PROCESSING
    # ==========================================================================

    def calculate_batch(
        self,
        inputs: List[
            Dict[str, Union[str, RecyclingInput, CompostingInput, AnaerobicDigestionInput]]
        ],
    ) -> List[RecyclingCompostingResult]:
        """
        Process a batch of recycling/composting/AD calculations.

        Each item in the list is a dict with:
          - "treatment_method": str ("recycling", "composting", "anaerobic_digestion")
          - "input": RecyclingInput | CompostingInput | AnaerobicDigestionInput

        Args:
            inputs: List of dicts containing treatment_method and input data.

        Returns:
            List of RecyclingCompostingResult (one per input).

        Raises:
            ValueError: If any individual calculation fails and batch_mode
                        does not suppress errors.
        """
        if not inputs:
            logger.warning("calculate_batch called with empty inputs list")
            return []

        start_time = datetime.now(timezone.utc)
        results: List[RecyclingCompostingResult] = []
        errors: List[str] = []

        for idx, item in enumerate(inputs):
            treatment_method = item.get("treatment_method")
            input_data = item.get("input")

            if treatment_method is None or input_data is None:
                msg = (
                    f"Batch item {idx}: missing 'treatment_method' or 'input'"
                )
                logger.error(msg)
                errors.append(msg)
                continue

            try:
                result = self.calculate(str(treatment_method), input_data)  # type: ignore[arg-type]
                results.append(result)
            except (ValueError, TypeError, InvalidOperation) as exc:
                msg = f"Batch item {idx} ({treatment_method}): {exc}"
                logger.error(msg)
                errors.append(msg)

        elapsed_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        logger.info(
            "calculate_batch completed: total=%d, success=%d, errors=%d, "
            "elapsed_ms=%.2f",
            len(inputs),
            len(results),
            len(errors),
            elapsed_ms,
        )

        if errors:
            logger.warning(
                "Batch had %d errors: %s", len(errors), "; ".join(errors)
            )

        return results

    # ==========================================================================
    # INPUT VALIDATION
    # ==========================================================================

    def validate_input(
        self,
        input_data: Union[RecyclingInput, CompostingInput, AnaerobicDigestionInput],
    ) -> List[str]:
        """
        Validate input data and return list of error messages.

        An empty list means the input is valid. This provides detailed
        validation beyond the Pydantic model constraints.

        Args:
            input_data: Any of the three supported input types.

        Returns:
            List of error message strings. Empty if valid.
        """
        errors: List[str] = []

        if isinstance(input_data, RecyclingInput):
            errors.extend(self._validate_recycling_input(input_data))
        elif isinstance(input_data, CompostingInput):
            errors.extend(self._validate_composting_input(input_data))
        elif isinstance(input_data, AnaerobicDigestionInput):
            errors.extend(self._validate_ad_input(input_data))
        else:
            errors.append(
                f"Unsupported input type: {type(input_data).__name__}. "
                "Expected RecyclingInput, CompostingInput, or AnaerobicDigestionInput."
            )

        if errors:
            logger.warning(
                "Validation errors for %s: %s",
                type(input_data).__name__,
                "; ".join(errors),
            )

        return errors

    def _validate_recycling_input(self, input_data: RecyclingInput) -> List[str]:
        """Validate recycling-specific input constraints."""
        errors: List[str] = []

        if input_data.mass_tonnes <= Decimal("0"):
            errors.append("mass_tonnes must be > 0")

        if input_data.waste_category not in RECYCLABLE_CATEGORIES:
            errors.append(
                f"Waste category '{input_data.waste_category.value}' is not "
                f"recyclable. Recyclable categories: "
                f"{sorted(c.value for c in RECYCLABLE_CATEGORIES)}"
            )

        if input_data.quality_factor < Decimal("0") or input_data.quality_factor > Decimal("1"):
            errors.append(
                f"quality_factor must be 0-1, got {input_data.quality_factor}"
            )

        if (
            input_data.recycling_type == RecyclingType.CLOSED_LOOP
            and input_data.quality_factor < Decimal("1.0")
        ):
            errors.append(
                "Closed-loop recycling should have quality_factor=1.0 "
                "(no quality degradation). Got "
                f"{input_data.quality_factor}."
            )

        return errors

    def _validate_composting_input(self, input_data: CompostingInput) -> List[str]:
        """Validate composting-specific input constraints."""
        errors: List[str] = []

        if input_data.mass_tonnes <= Decimal("0"):
            errors.append("mass_tonnes must be > 0")

        if input_data.waste_category not in COMPOSTABLE_CATEGORIES:
            errors.append(
                f"Waste category '{input_data.waste_category.value}' is not "
                f"compostable. Compostable categories: "
                f"{sorted(c.value for c in COMPOSTABLE_CATEGORIES)}"
            )

        if input_data.ch4_ef_override is not None:
            if input_data.ch4_ef_override <= Decimal("0"):
                errors.append("ch4_ef_override must be > 0")
            elif input_data.ch4_ef_override > Decimal("100"):
                errors.append(
                    f"ch4_ef_override={input_data.ch4_ef_override} g/kg exceeds "
                    "plausible range (max ~100 g/kg). Check units."
                )

        if input_data.n2o_ef_override is not None:
            if input_data.n2o_ef_override <= Decimal("0"):
                errors.append("n2o_ef_override must be > 0")
            elif input_data.n2o_ef_override > Decimal("10"):
                errors.append(
                    f"n2o_ef_override={input_data.n2o_ef_override} g/kg exceeds "
                    "plausible range (max ~10 g/kg). Check units."
                )

        return errors

    def _validate_ad_input(self, input_data: AnaerobicDigestionInput) -> List[str]:
        """Validate anaerobic digestion input constraints."""
        errors: List[str] = []

        if input_data.mass_tonnes <= Decimal("0"):
            errors.append("mass_tonnes must be > 0")

        if input_data.waste_category not in AD_COMPATIBLE_CATEGORIES:
            errors.append(
                f"Waste category '{input_data.waste_category.value}' is not "
                f"compatible with anaerobic digestion. Compatible: "
                f"{sorted(c.value for c in AD_COMPATIBLE_CATEGORIES)}"
            )

        valid_plant_types = set(AD_LEAKAGE_RATES_EXTENDED.keys())
        if input_data.plant_type not in valid_plant_types:
            errors.append(
                f"Unknown plant_type '{input_data.plant_type}'. "
                f"Valid: {sorted(valid_plant_types)}"
            )

        if (
            input_data.biogas_ch4_content < Decimal("0.30")
            or input_data.biogas_ch4_content > Decimal("0.80")
        ):
            errors.append(
                f"biogas_ch4_content={input_data.biogas_ch4_content} is outside "
                "plausible range (0.30-0.80). Typical biogas is 55-65% CH4."
            )

        if input_data.leakage_rate_override is not None:
            if input_data.leakage_rate_override > Decimal("0.20"):
                errors.append(
                    f"leakage_rate_override={input_data.leakage_rate_override} "
                    "exceeds 20%. This is unusually high for any AD plant."
                )

        return errors

    # ==========================================================================
    # RECYCLING CALCULATIONS
    # ==========================================================================

    def calculate_recycling(
        self, input_data: RecyclingInput
    ) -> RecyclingCompostingResult:
        """
        Calculate emissions from recycling under GHG Protocol cut-off approach.

        Under the cut-off approach:
        - Category 5 emissions = transport to facility + MRF sorting
        - Avoided emissions from displaced virgin production are a MEMO item
        - Avoided emissions are NEVER deducted from Category 5 total

        For open-loop recycling, the quality_factor adjusts the avoided
        emissions downward to reflect material quality loss (downcycling).

        Args:
            input_data: RecyclingInput with mass, waste_category, recycling_type.

        Returns:
            RecyclingCompostingResult with treatment and avoided emissions.

        Raises:
            ValueError: If input validation fails.
        """
        start_time = datetime.now(timezone.utc)
        logger.info(
            "calculate_recycling: mass=%.4f t, category=%s, type=%s",
            input_data.mass_tonnes,
            input_data.waste_category.value,
            input_data.recycling_type.value,
        )

        # Validate
        errors = self._validate_recycling_input(input_data)
        if errors:
            raise ValueError(
                f"Recycling input validation failed: {'; '.join(errors)}"
            )

        # Route to open-loop or closed-loop
        if input_data.recycling_type == RecyclingType.OPEN_LOOP:
            result = self.calculate_open_loop(
                mass=input_data.mass_tonnes,
                waste_category=input_data.waste_category,
                quality_factor=input_data.quality_factor,
            )
        else:
            result = self.calculate_closed_loop(
                mass=input_data.mass_tonnes,
                waste_category=input_data.waste_category,
            )

        # Optionally suppress avoided emissions from result
        if not input_data.calculate_avoided_emissions:
            result = RecyclingCompostingResult(
                treatment_emissions_co2e=result.treatment_emissions_co2e,
                avoided_emissions_memo_co2e=None,
                net_emissions_co2e=result.net_emissions_co2e,
                ch4_tonnes=result.ch4_tonnes,
                n2o_tonnes=result.n2o_tonnes,
                method_detail=result.method_detail,
            )

        elapsed_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000
        logger.info(
            "calculate_recycling completed: treatment_co2e=%s, "
            "avoided_co2e=%s, elapsed_ms=%.2f",
            result.treatment_emissions_co2e,
            result.avoided_emissions_memo_co2e,
            elapsed_ms,
        )
        return result

    def calculate_open_loop(
        self,
        mass: Decimal,
        waste_category: WasteCategory,
        quality_factor: Decimal,
    ) -> RecyclingCompostingResult:
        """
        Calculate open-loop recycling emissions.

        Open-loop recycling means material is recycled into a different or
        lower-grade product (e.g., PET bottles to polyester fiber). The
        quality_factor (0-1) scales the avoided emissions to reflect
        quality degradation.

        Args:
            mass: Waste mass in tonnes.
            waste_category: Material type.
            quality_factor: Downcycling quality factor (0-1, 1=no loss).

        Returns:
            RecyclingCompostingResult with treatment and quality-adjusted
            avoided emissions.
        """
        logger.debug(
            "calculate_open_loop: mass=%s, category=%s, quality_factor=%s",
            mass, waste_category.value, quality_factor,
        )

        # Treatment emissions (transport + sorting)
        recycling_ef = self.get_recycling_ef(waste_category, self._ef_source)
        treatment_co2e = self.calculate_mrf_emissions(mass, recycling_ef)

        # Avoided emissions (memo only) scaled by quality factor
        avoided_co2e = self.calculate_avoided_emissions(
            mass, waste_category, quality_factor
        )

        treatment_co2e_rounded = _round_decimal(treatment_co2e)
        avoided_co2e_rounded = _round_decimal(avoided_co2e)

        return RecyclingCompostingResult(
            treatment_emissions_co2e=treatment_co2e_rounded,
            avoided_emissions_memo_co2e=avoided_co2e_rounded,
            net_emissions_co2e=treatment_co2e_rounded,
            ch4_tonnes=Decimal("0"),
            n2o_tonnes=Decimal("0"),
            method_detail="recycling_open_loop",
        )

    def calculate_closed_loop(
        self,
        mass: Decimal,
        waste_category: WasteCategory,
    ) -> RecyclingCompostingResult:
        """
        Calculate closed-loop recycling emissions.

        Closed-loop recycling means material is recycled into the same
        product type with no quality loss (e.g., aluminum can to aluminum can).
        Quality factor is implicitly 1.0.

        Args:
            mass: Waste mass in tonnes.
            waste_category: Material type.

        Returns:
            RecyclingCompostingResult with treatment and full avoided emissions.
        """
        logger.debug(
            "calculate_closed_loop: mass=%s, category=%s",
            mass, waste_category.value,
        )

        # Treatment emissions (transport + sorting)
        recycling_ef = self.get_recycling_ef(waste_category, self._ef_source)
        treatment_co2e = self.calculate_mrf_emissions(mass, recycling_ef)

        # Avoided emissions (memo only) with quality_factor = 1.0 (no loss)
        avoided_co2e = self.calculate_avoided_emissions(
            mass, waste_category, Decimal("1.0")
        )

        treatment_co2e_rounded = _round_decimal(treatment_co2e)
        avoided_co2e_rounded = _round_decimal(avoided_co2e)

        return RecyclingCompostingResult(
            treatment_emissions_co2e=treatment_co2e_rounded,
            avoided_emissions_memo_co2e=avoided_co2e_rounded,
            net_emissions_co2e=treatment_co2e_rounded,
            ch4_tonnes=Decimal("0"),
            n2o_tonnes=Decimal("0"),
            method_detail="recycling_closed_loop",
        )

    def calculate_mrf_emissions(
        self, mass: Decimal, mrf_ef: Decimal
    ) -> Decimal:
        """
        Calculate MRF (Materials Recovery Facility) sorting emissions.

        Formula:
            MRF_emissions (tCO2e) = mass (t) x mrf_ef (kgCO2e/t) / 1000

        Args:
            mass: Waste mass in tonnes.
            mrf_ef: MRF sorting emission factor in kgCO2e per tonne.

        Returns:
            MRF sorting emissions in tonnes CO2e.

        Raises:
            ValueError: If mass or mrf_ef is negative.
        """
        if mass < Decimal("0"):
            raise ValueError(f"mass must be >= 0, got {mass}")
        if mrf_ef < Decimal("0"):
            raise ValueError(f"mrf_ef must be >= 0, got {mrf_ef}")

        # kgCO2e / 1000 = tCO2e
        emissions_tco2e = mass * mrf_ef / KG_PER_TONNE

        logger.debug(
            "calculate_mrf_emissions: mass=%s t, ef=%s kgCO2e/t, "
            "emissions=%s tCO2e",
            mass, mrf_ef, emissions_tco2e,
        )
        return emissions_tco2e

    def calculate_transport_to_facility(
        self,
        mass: Decimal,
        distance_km: Decimal,
        mode: str = "road",
    ) -> Decimal:
        """
        Calculate transport emissions from operations to treatment facility.

        Formula:
            Transport_emissions (tCO2e) = mass (t) x distance (km) x EF (kgCO2e/t-km) / 1000

        Args:
            mass: Waste mass in tonnes.
            distance_km: Distance to treatment facility in km.
            mode: Transport mode ('road', 'rail', 'ship', etc.).

        Returns:
            Transport emissions in tonnes CO2e.

        Raises:
            ValueError: If mode is not recognized or values are negative.
        """
        if mass < Decimal("0"):
            raise ValueError(f"mass must be >= 0, got {mass}")
        if distance_km < Decimal("0"):
            raise ValueError(f"distance_km must be >= 0, got {distance_km}")

        transport_ef = TRANSPORT_MODE_EF.get(mode)
        if transport_ef is None:
            valid_modes = sorted(TRANSPORT_MODE_EF.keys())
            raise ValueError(
                f"Unknown transport mode '{mode}'. Valid: {valid_modes}"
            )

        emissions_tco2e = mass * distance_km * transport_ef / KG_PER_TONNE

        logger.debug(
            "calculate_transport_to_facility: mass=%s t, dist=%s km, "
            "mode=%s, ef=%s, emissions=%s tCO2e",
            mass, distance_km, mode, transport_ef, emissions_tco2e,
        )
        return _round_decimal(emissions_tco2e)

    def calculate_avoided_emissions(
        self,
        mass: Decimal,
        waste_category: WasteCategory,
        quality_factor: Decimal,
    ) -> Decimal:
        """
        Calculate avoided emissions from recycling (MEMO ITEM ONLY).

        Avoided emissions represent the GHG reduction from using recycled
        material instead of virgin material. Under GHG Protocol's cut-off
        approach, these are reported SEPARATELY and NEVER deducted from
        Category 5 totals.

        Formula:
            Avoided_CO2e = mass x virgin_ef x quality_factor / 1000

        The quality_factor (0-1) adjusts for downcycling (open-loop).
        For closed-loop (same quality), quality_factor = 1.0.

        Args:
            mass: Recycled mass in tonnes.
            waste_category: Material type.
            quality_factor: Quality adjustment for downcycling (0-1).

        Returns:
            Avoided emissions in tonnes CO2e (always positive, memo item).
        """
        if quality_factor < Decimal("0") or quality_factor > Decimal("1"):
            raise ValueError(
                f"quality_factor must be 0-1, got {quality_factor}"
            )

        virgin_ef = self.get_virgin_production_ef(waste_category)

        # avoided_ef in kgCO2e per tonne, convert to tCO2e
        avoided_tco2e = mass * virgin_ef * quality_factor / KG_PER_TONNE

        logger.debug(
            "calculate_avoided_emissions: mass=%s, category=%s, "
            "quality_factor=%s, virgin_ef=%s kgCO2e/t, avoided=%s tCO2e",
            mass, waste_category.value, quality_factor,
            virgin_ef, avoided_tco2e,
        )
        return avoided_tco2e

    def get_recycling_ef(
        self,
        waste_category: WasteCategory,
        source: EFSource = EFSource.DEFRA_BEIS,
    ) -> Decimal:
        """
        Look up recycling process emission factor for a waste category.

        Returns the transport+sorting EF (kgCO2e per tonne) for the given
        waste category. Falls back to the default MRF EF if category is
        not found.

        Args:
            waste_category: Material type.
            source: Emission factor source preference.

        Returns:
            Emission factor in kgCO2e per tonne.
        """
        # Try primary table
        category_ef = RECYCLING_PROCESS_EF.get(waste_category)
        if category_ef is not None:
            ef = category_ef.get("transport_and_sorting", MRF_EF_DEFAULT)
            logger.debug(
                "get_recycling_ef: category=%s, source=%s, ef=%s kgCO2e/t",
                waste_category.value, source.value, ef,
            )
            return ef

        # Try DEFRA table
        if source in (EFSource.DEFRA_BEIS, EFSource.CUSTOM):
            defra_entry = DEFRA_WASTE_FACTORS.get(waste_category)
            if defra_entry is not None and "recycling" in defra_entry:
                ef = defra_entry["recycling"]
                logger.debug(
                    "get_recycling_ef (DEFRA fallback): category=%s, ef=%s",
                    waste_category.value, ef,
                )
                return ef

        # Default fallback
        logger.warning(
            "get_recycling_ef: no EF found for %s, using default %s",
            waste_category.value, MRF_EF_DEFAULT,
        )
        return MRF_EF_DEFAULT

    def get_virgin_production_ef(
        self, waste_category: WasteCategory
    ) -> Decimal:
        """
        Look up virgin production emission factor for avoided emissions.

        Returns the EF representing CO2e savings per tonne of material
        recycled vs. produced from virgin feedstock.

        Args:
            waste_category: Material type.

        Returns:
            Virgin production avoided EF in kgCO2e per tonne. Returns 0 if
            the waste category has no virgin production EF.
        """
        entry = VIRGIN_PRODUCTION_EF.get(waste_category)
        if entry is not None:
            ef = entry.get("avoided_ef", Decimal("0"))
            logger.debug(
                "get_virgin_production_ef: category=%s, ef=%s kgCO2e/t",
                waste_category.value, ef,
            )
            return ef

        # Fallback: try EPA WARM negative recycling factor
        warm_entry = EPA_WARM_FACTORS.get(waste_category)
        if warm_entry is not None and "recycling" in warm_entry:
            # EPA WARM negative value = avoided; take absolute, convert
            warm_ef = abs(warm_entry["recycling"]) * EPA_WARM_TO_KG_PER_TONNE
            logger.debug(
                "get_virgin_production_ef (WARM fallback): category=%s, "
                "ef=%s kgCO2e/t",
                waste_category.value, warm_ef,
            )
            return _round_decimal(warm_ef)

        logger.warning(
            "get_virgin_production_ef: no EF for %s, returning 0",
            waste_category.value,
        )
        return Decimal("0")

    def get_recycling_ef_by_source(
        self,
        waste_category: WasteCategory,
        source: EFSource,
    ) -> Dict[str, Decimal]:
        """
        Get recycling emission factor with full metadata by source.

        Returns a dict with 'transport_and_sorting', 'mrf_sorting',
        'reprocessing', and 'source' keys.

        Args:
            waste_category: Material type.
            source: Emission factor source.

        Returns:
            Dict with EF components and metadata.
        """
        if source == EFSource.DEFRA_BEIS:
            entry = RECYCLING_PROCESS_EF.get(waste_category)
            if entry is not None:
                return {
                    "transport_and_sorting": entry["transport_and_sorting"],
                    "mrf_sorting": entry["mrf_sorting"],
                    "reprocessing": entry["reprocessing"],
                    "source": "defra_beis",
                    "source_year": entry.get("source_year", Decimal("2025")),
                }

        if source == EFSource.EPA_WARM:
            warm_entry = EPA_WARM_FACTORS.get(waste_category)
            if warm_entry is not None and "recycling" in warm_entry:
                warm_ef_kg = abs(warm_entry["recycling"]) * EPA_WARM_TO_KG_PER_TONNE
                return {
                    "transport_and_sorting": _round_decimal(warm_ef_kg),
                    "mrf_sorting": _round_decimal(warm_ef_kg * Decimal("0.85")),
                    "reprocessing": _round_decimal(warm_ef_kg * Decimal("0.15")),
                    "source": "epa_warm_v16",
                    "source_year": Decimal("2024"),
                }

        # Default
        return {
            "transport_and_sorting": MRF_EF_DEFAULT,
            "mrf_sorting": Decimal("18.0"),
            "reprocessing": Decimal("3.0"),
            "source": "default",
            "source_year": Decimal("2025"),
        }

    # ==========================================================================
    # COMPOSTING CALCULATIONS
    # ==========================================================================

    def calculate_composting(
        self, input_data: CompostingInput
    ) -> RecyclingCompostingResult:
        """
        Calculate composting emissions using IPCC 2006 Vol 5 Ch 4 methodology.

        Composting produces CH4 (from anaerobic pockets) and N2O (from
        nitrification/denitrification). These are converted to CO2e using
        the configured GWP values.

        Formulas (IPCC 2006 Vol 5 Equation 4.1):
            CH4_emissions = M x EF_CH4 (g CH4 / kg waste)
            N2O_emissions = M x EF_N2O (g N2O / kg waste)
            Total_CO2e = CH4_emissions x GWP_CH4 + N2O_emissions x GWP_N2O

        Args:
            input_data: CompostingInput with mass, category, composting type.

        Returns:
            RecyclingCompostingResult with CH4/N2O breakdown and CO2e total.

        Raises:
            ValueError: If input validation fails.
        """
        start_time = datetime.now(timezone.utc)
        logger.info(
            "calculate_composting: mass=%.4f t, category=%s, "
            "home=%s, dry_basis=%s",
            input_data.mass_tonnes,
            input_data.waste_category.value,
            input_data.is_home_composting,
            input_data.dry_weight_basis,
        )

        # Validate
        errors = self._validate_composting_input(input_data)
        if errors:
            raise ValueError(
                f"Composting input validation failed: {'; '.join(errors)}"
            )

        mass = input_data.mass_tonnes
        dry_basis = input_data.dry_weight_basis

        # Calculate individual gas emissions (returns tonnes)
        ch4_tonnes = self.calculate_composting_ch4(
            mass=mass,
            dry_weight_basis=dry_basis,
            is_home=input_data.is_home_composting,
            ch4_ef_override=input_data.ch4_ef_override,
        )

        n2o_tonnes = self.calculate_composting_n2o(
            mass=mass,
            dry_weight_basis=dry_basis,
            is_home=input_data.is_home_composting,
            n2o_ef_override=input_data.n2o_ef_override,
        )

        # Convert to CO2e
        ch4_co2e = ch4_tonnes * self._gwp_ch4
        n2o_co2e = n2o_tonnes * self._gwp_n2o
        total_co2e = ch4_co2e + n2o_co2e

        # Round
        ch4_tonnes_r = _round_decimal(ch4_tonnes)
        n2o_tonnes_r = _round_decimal(n2o_tonnes)
        total_co2e_r = _round_decimal(total_co2e)

        elapsed_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000
        logger.info(
            "calculate_composting completed: CH4=%s t, N2O=%s t, "
            "CO2e=%s t, elapsed_ms=%.2f",
            ch4_tonnes_r, n2o_tonnes_r, total_co2e_r, elapsed_ms,
        )

        return RecyclingCompostingResult(
            treatment_emissions_co2e=total_co2e_r,
            avoided_emissions_memo_co2e=None,
            net_emissions_co2e=total_co2e_r,
            ch4_tonnes=ch4_tonnes_r,
            n2o_tonnes=n2o_tonnes_r,
            method_detail="composting",
        )

    def calculate_composting_ch4(
        self,
        mass: Decimal,
        dry_weight_basis: bool = False,
        is_home: bool = False,
        ch4_ef_override: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Calculate CH4 emissions from composting.

        IPCC 2006 Vol 5 Table 4.1 defaults:
            - Industrial (wet):  4 g CH4/kg waste  (range 0.08-20.0)
            - Industrial (dry): 10 g CH4/kg waste
            - Home (wet):       10 g CH4/kg waste
            - Home (dry):       20 g CH4/kg waste

        Formula:
            CH4 (tonnes) = mass (t) x 1000 (kg/t) x EF (g/kg) / 1e6 (g/t)

        Args:
            mass: Waste mass in tonnes.
            dry_weight_basis: Whether mass is on dry weight basis.
            is_home: Whether this is home composting (higher EFs).
            ch4_ef_override: Optional override EF in g CH4/kg waste.

        Returns:
            CH4 emissions in tonnes.
        """
        if ch4_ef_override is not None:
            ef_g_per_kg = ch4_ef_override
        else:
            ef_data = self.get_composting_ef(dry_weight_basis, not is_home)
            ef_g_per_kg = ef_data["ch4_ef"]

        # mass (t) x 1000 (kg/t) x EF (g/kg) / 1,000,000 (g/t) = tonnes CH4
        ch4_tonnes = mass * KG_PER_TONNE * ef_g_per_kg / (KG_PER_TONNE * G_PER_KG)

        logger.debug(
            "calculate_composting_ch4: mass=%s t, ef=%s g/kg, "
            "ch4=%s t",
            mass, ef_g_per_kg, ch4_tonnes,
        )
        return ch4_tonnes

    def calculate_composting_n2o(
        self,
        mass: Decimal,
        dry_weight_basis: bool = False,
        is_home: bool = False,
        n2o_ef_override: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Calculate N2O emissions from composting.

        IPCC 2006 Vol 5 Table 4.1 defaults:
            - Industrial (wet):  0.3 g N2O/kg waste (range 0.06-0.6)
            - Industrial (dry):  0.6 g N2O/kg waste
            - Home (wet):        0.6 g N2O/kg waste
            - Home (dry):        1.2 g N2O/kg waste

        Formula:
            N2O (tonnes) = mass (t) x 1000 (kg/t) x EF (g/kg) / 1e6 (g/t)

        Args:
            mass: Waste mass in tonnes.
            dry_weight_basis: Whether mass is on dry weight basis.
            is_home: Whether this is home composting.
            n2o_ef_override: Optional override EF in g N2O/kg waste.

        Returns:
            N2O emissions in tonnes.
        """
        if n2o_ef_override is not None:
            ef_g_per_kg = n2o_ef_override
        else:
            ef_data = self.get_composting_ef(dry_weight_basis, not is_home)
            ef_g_per_kg = ef_data["n2o_ef"]

        # mass (t) x 1000 (kg/t) x EF (g/kg) / 1,000,000 (g/t) = tonnes N2O
        n2o_tonnes = mass * KG_PER_TONNE * ef_g_per_kg / (KG_PER_TONNE * G_PER_KG)

        logger.debug(
            "calculate_composting_n2o: mass=%s t, ef=%s g/kg, "
            "n2o=%s t",
            mass, ef_g_per_kg, n2o_tonnes,
        )
        return n2o_tonnes

    def estimate_compost_output(
        self,
        mass: Decimal,
        waste_category: WasteCategory = WasteCategory.FOOD_WASTE,
        reduction_factor: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Estimate compost output mass after composting process.

        During composting, organic matter decomposes releasing CO2 and water,
        resulting in mass loss. The reduction factor represents the fraction
        of input mass remaining as finished compost.

        Args:
            mass: Input waste mass in tonnes.
            waste_category: Waste material type (affects mass loss).
            reduction_factor: Optional override for mass reduction factor (0-1).

        Returns:
            Estimated compost output in tonnes.
        """
        if reduction_factor is not None:
            factor = reduction_factor
        else:
            factor = COMPOSTING_MASS_REDUCTION.get(
                waste_category, Decimal("0.40")
            )

        output_mass = mass * factor

        logger.debug(
            "estimate_compost_output: input=%s t, factor=%s, output=%s t",
            mass, factor, output_mass,
        )
        return _round_decimal(output_mass)

    def get_composting_ef(
        self,
        dry_weight_basis: bool = False,
        industrial: bool = True,
    ) -> Dict[str, Decimal]:
        """
        Get composting emission factors from IPCC 2006 Vol 5 Table 4.1.

        Returns CH4 and N2O emission factors in g gas per kg waste with
        the associated min/max ranges.

        Args:
            dry_weight_basis: Whether to return dry-weight EFs.
            industrial: Industrial composting (True) or home composting (False).

        Returns:
            Dict with keys: ch4_ef, ch4_min, ch4_max, n2o_ef, n2o_min, n2o_max.
        """
        # Build key
        composting_type = "industrial" if industrial else "home"
        weight_basis = "dry" if dry_weight_basis else "wet"
        key = f"{composting_type}_{weight_basis}"

        ef_data = COMPOSTING_EF_EXTENDED.get(key)
        if ef_data is None:
            logger.warning(
                "get_composting_ef: key '%s' not found, using industrial_wet",
                key,
            )
            ef_data = COMPOSTING_EF_EXTENDED["industrial_wet"]

        result = {
            "ch4_ef": ef_data["ch4_default"],
            "ch4_min": ef_data["ch4_min"],
            "ch4_max": ef_data["ch4_max"],
            "n2o_ef": ef_data["n2o_default"],
            "n2o_min": ef_data["n2o_min"],
            "n2o_max": ef_data["n2o_max"],
            "composting_type": composting_type,
            "weight_basis": weight_basis,
        }

        logger.debug(
            "get_composting_ef: type=%s, basis=%s, ch4=%s g/kg, n2o=%s g/kg",
            composting_type, weight_basis,
            result["ch4_ef"], result["n2o_ef"],
        )
        return result

    def get_composting_ef_all(self) -> Dict[str, Dict[str, Decimal]]:
        """
        Get all composting emission factor variants.

        Returns the complete IPCC 2006 Vol 5 Table 4.1 emission factor
        table as a nested dict.

        Returns:
            Nested dict keyed by composting type (industrial_wet, etc).
        """
        result: Dict[str, Dict[str, Decimal]] = {}
        for key, ef_data in COMPOSTING_EF_EXTENDED.items():
            result[key] = {
                "ch4_ef": ef_data["ch4_default"],
                "ch4_min": ef_data["ch4_min"],
                "ch4_max": ef_data["ch4_max"],
                "n2o_ef": ef_data["n2o_default"],
                "n2o_min": ef_data["n2o_min"],
                "n2o_max": ef_data["n2o_max"],
            }
        return result

    # ==========================================================================
    # ANAEROBIC DIGESTION CALCULATIONS
    # ==========================================================================

    def calculate_anaerobic_digestion(
        self, input_data: AnaerobicDigestionInput
    ) -> RecyclingCompostingResult:
        """
        Calculate anaerobic digestion emissions.

        Emission sources for AD:
        1. CH4 leakage from digester (2-7% of CH4 produced, by plant type)
        2. Fugitive CH4 from digestate storage (gastight vs open)
        3. CH4 and N2O from biogas combustion (CHP engine)

        Formula:
            Total_CO2e = leakage_CO2e + digestate_CO2e + combustion_CO2e

        Args:
            input_data: AnaerobicDigestionInput with mass, category, plant type.

        Returns:
            RecyclingCompostingResult with emissions breakdown.

        Raises:
            ValueError: If input validation fails.
        """
        start_time = datetime.now(timezone.utc)
        logger.info(
            "calculate_anaerobic_digestion: mass=%.4f t, category=%s, "
            "plant_type=%s, gastight=%s",
            input_data.mass_tonnes,
            input_data.waste_category.value,
            input_data.plant_type,
            input_data.gastight_storage,
        )

        # Validate
        errors = self._validate_ad_input(input_data)
        if errors:
            raise ValueError(
                f"AD input validation failed: {'; '.join(errors)}"
            )

        mass = input_data.mass_tonnes
        waste_cat = input_data.waste_category

        # Step 1: Calculate biogas production
        biogas_volume = self.calculate_biogas_production(mass, waste_cat)

        # Step 2: Calculate total CH4 in biogas
        ch4_volume = biogas_volume * input_data.biogas_ch4_content  # m3 CH4
        ch4_mass_kg = ch4_volume * CH4_DENSITY_KG_PER_M3  # kg CH4

        # Step 3: CH4 leakage from digester
        leakage_rate = self.get_leakage_rate(input_data.plant_type)
        if input_data.leakage_rate_override is not None:
            leakage_rate = input_data.leakage_rate_override

        ch4_leaked_kg = self.calculate_ch4_leakage(ch4_mass_kg, leakage_rate)

        # Step 4: Digestate storage emissions
        digestate_co2e_kg = self.calculate_digestate_emissions(
            mass, input_data.gastight_storage
        )

        # Step 5: Biogas combustion emissions (CHP)
        # Only combust the non-leaked portion
        biogas_combusted = biogas_volume * (Decimal("1") - leakage_rate)
        combustion_co2e_kg = self.calculate_biogas_combustion_emissions(
            biogas_combusted
        )

        # Step 6: Sum all emission sources
        leakage_co2e_kg = ch4_leaked_kg * self._gwp_ch4
        total_co2e_kg = leakage_co2e_kg + digestate_co2e_kg + combustion_co2e_kg

        # Convert kg to tonnes
        total_co2e_t = total_co2e_kg / KG_PER_TONNE
        ch4_total_t = (
            ch4_leaked_kg
            + self._get_digestate_ch4_kg(mass, input_data.gastight_storage)
            + self._get_combustion_ch4_kg(biogas_combusted)
        ) / KG_PER_TONNE
        n2o_total_t = (
            self._get_digestate_n2o_kg(mass, input_data.gastight_storage)
            + self._get_combustion_n2o_kg(biogas_combusted)
        ) / KG_PER_TONNE

        total_co2e_r = _round_decimal(total_co2e_t)
        ch4_r = _round_decimal(ch4_total_t)
        n2o_r = _round_decimal(n2o_total_t)

        elapsed_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000
        logger.info(
            "calculate_anaerobic_digestion completed: biogas=%s m3, "
            "CH4=%s t, N2O=%s t, CO2e=%s t, elapsed_ms=%.2f",
            _round_decimal(biogas_volume), ch4_r, n2o_r,
            total_co2e_r, elapsed_ms,
        )

        return RecyclingCompostingResult(
            treatment_emissions_co2e=total_co2e_r,
            avoided_emissions_memo_co2e=None,
            net_emissions_co2e=total_co2e_r,
            ch4_tonnes=ch4_r,
            n2o_tonnes=n2o_r,
            method_detail="anaerobic_digestion",
        )

    def calculate_biogas_production(
        self, mass: Decimal, waste_category: WasteCategory
    ) -> Decimal:
        """
        Calculate biogas production from anaerobic digestion.

        Uses waste-category-specific biogas yield factors (m3 per tonne).

        Args:
            mass: Wet waste mass in tonnes.
            waste_category: Waste material type.

        Returns:
            Biogas volume in m3.
        """
        yield_data = BIOGAS_YIELD.get(waste_category)
        if yield_data is not None:
            yield_m3 = yield_data["yield_m3_per_tonne"]
        else:
            # Fallback to generic default
            yield_m3 = BIOGAS_YIELD[WasteCategory.OTHER]["yield_m3_per_tonne"]
            logger.warning(
                "calculate_biogas_production: no yield for %s, "
                "using default %s m3/t",
                waste_category.value, yield_m3,
            )

        biogas_volume = mass * yield_m3

        logger.debug(
            "calculate_biogas_production: mass=%s t, yield=%s m3/t, "
            "biogas=%s m3",
            mass, yield_m3, biogas_volume,
        )
        return biogas_volume

    def calculate_ch4_leakage(
        self, ch4_produced_kg: Decimal, leakage_rate: Decimal
    ) -> Decimal:
        """
        Calculate CH4 leakage from anaerobic digester.

        Formula:
            CH4_leaked (kg) = CH4_produced (kg) x leakage_rate

        Args:
            ch4_produced_kg: Total CH4 produced by the digester in kg.
            leakage_rate: Fraction of CH4 that leaks (0-1).

        Returns:
            Leaked CH4 in kg.

        Raises:
            ValueError: If leakage_rate is outside 0-1.
        """
        if leakage_rate < Decimal("0") or leakage_rate > Decimal("1"):
            raise ValueError(
                f"leakage_rate must be 0-1, got {leakage_rate}"
            )

        ch4_leaked = ch4_produced_kg * leakage_rate

        logger.debug(
            "calculate_ch4_leakage: produced=%s kg, rate=%s, leaked=%s kg",
            ch4_produced_kg, leakage_rate, ch4_leaked,
        )
        return ch4_leaked

    def calculate_digestate_emissions(
        self, mass: Decimal, gastight_storage: bool = True
    ) -> Decimal:
        """
        Calculate fugitive emissions from digestate storage.

        Gastight storage has minimal emissions (sealed tank), while open
        storage allows significant CH4 and N2O fugitive release.

        Args:
            mass: Original feedstock mass in tonnes (digestate ~proportional).
            gastight_storage: Whether storage is gastight (sealed).

        Returns:
            Digestate storage emissions in kgCO2e.
        """
        storage_type = "gastight" if gastight_storage else "open_storage"
        ef_data = DIGESTATE_EF.get(storage_type, DIGESTATE_EF["open_storage"])

        # Total CO2e from digestate storage
        co2e_kg = mass * ef_data["total_co2e_per_tonne"]

        logger.debug(
            "calculate_digestate_emissions: mass=%s t, type=%s, "
            "co2e=%s kgCO2e",
            mass, storage_type, co2e_kg,
        )
        return co2e_kg

    def _get_digestate_ch4_kg(
        self, mass: Decimal, gastight_storage: bool
    ) -> Decimal:
        """
        Get CH4 component of digestate storage emissions in kg.

        Args:
            mass: Original feedstock mass in tonnes.
            gastight_storage: Whether storage is gastight.

        Returns:
            CH4 emissions from digestate storage in kg.
        """
        storage_type = "gastight" if gastight_storage else "open_storage"
        ef_data = DIGESTATE_EF.get(storage_type, DIGESTATE_EF["open_storage"])
        return mass * ef_data["ch4_ef_kg_per_tonne"]

    def _get_digestate_n2o_kg(
        self, mass: Decimal, gastight_storage: bool
    ) -> Decimal:
        """
        Get N2O component of digestate storage emissions in kg.

        Args:
            mass: Original feedstock mass in tonnes.
            gastight_storage: Whether storage is gastight.

        Returns:
            N2O emissions from digestate storage in kg.
        """
        storage_type = "gastight" if gastight_storage else "open_storage"
        ef_data = DIGESTATE_EF.get(storage_type, DIGESTATE_EF["open_storage"])
        return mass * ef_data["n2o_ef_kg_per_tonne"]

    def calculate_biogas_combustion_emissions(
        self, biogas_volume: Decimal
    ) -> Decimal:
        """
        Calculate emissions from biogas combustion in CHP engine.

        Biogas combustion produces trace CH4 (incomplete combustion) and
        N2O. The CO2 from biogas combustion is biogenic and not counted.

        Formula:
            CH4_combustion = biogas_volume x CH4_EF
            N2O_combustion = biogas_volume x N2O_EF
            CO2e = CH4_combustion x GWP_CH4 + N2O_combustion x GWP_N2O

        Args:
            biogas_volume: Volume of biogas combusted in m3.

        Returns:
            Combustion emissions in kgCO2e.
        """
        ch4_kg = self._get_combustion_ch4_kg(biogas_volume)
        n2o_kg = self._get_combustion_n2o_kg(biogas_volume)

        co2e_kg = (ch4_kg * self._gwp_ch4) + (n2o_kg * self._gwp_n2o)

        logger.debug(
            "calculate_biogas_combustion_emissions: volume=%s m3, "
            "ch4=%s kg, n2o=%s kg, co2e=%s kgCO2e",
            biogas_volume, ch4_kg, n2o_kg, co2e_kg,
        )
        return co2e_kg

    def _get_combustion_ch4_kg(self, biogas_volume: Decimal) -> Decimal:
        """
        Get CH4 from biogas combustion in kg.

        Args:
            biogas_volume: Volume of biogas combusted in m3.

        Returns:
            CH4 emissions from combustion in kg.
        """
        return biogas_volume * BIOGAS_COMBUSTION_EF["ch4_kg_per_m3_biogas"]

    def _get_combustion_n2o_kg(self, biogas_volume: Decimal) -> Decimal:
        """
        Get N2O from biogas combustion in kg.

        Args:
            biogas_volume: Volume of biogas combusted in m3.

        Returns:
            N2O emissions from combustion in kg.
        """
        return biogas_volume * BIOGAS_COMBUSTION_EF["n2o_kg_per_m3_biogas"]

    def get_leakage_rate(self, plant_type: str) -> Decimal:
        """
        Look up CH4 leakage rate by AD plant type.

        Args:
            plant_type: AD plant type (biowaste, wastewater, manure, etc.).

        Returns:
            CH4 leakage rate as a fraction (0-1).

        Raises:
            ValueError: If plant_type is not recognized.
        """
        entry = AD_LEAKAGE_RATES_EXTENDED.get(plant_type)
        if entry is not None:
            rate = entry["default"]
            logger.debug(
                "get_leakage_rate: plant_type=%s, rate=%s (%s%%)",
                plant_type, rate, rate * Decimal("100"),
            )
            return rate

        # Fallback to models.py AD_LEAKAGE_RATES
        rate = AD_LEAKAGE_RATES.get(plant_type)
        if rate is not None:
            logger.debug(
                "get_leakage_rate (fallback): plant_type=%s, rate=%s",
                plant_type, rate,
            )
            return rate

        raise ValueError(
            f"Unknown AD plant_type '{plant_type}'. "
            f"Valid: {sorted(AD_LEAKAGE_RATES_EXTENDED.keys())}"
        )

    def get_leakage_rate_range(self, plant_type: str) -> Dict[str, Decimal]:
        """
        Get leakage rate with min/max range for a plant type.

        Args:
            plant_type: AD plant type.

        Returns:
            Dict with default, min, max leakage rates.
        """
        entry = AD_LEAKAGE_RATES_EXTENDED.get(plant_type)
        if entry is not None:
            return {
                "default": entry["default"],
                "min": entry["min"],
                "max": entry["max"],
            }

        # Fallback
        rate = AD_LEAKAGE_RATES.get(plant_type, Decimal("0.03"))
        return {
            "default": rate,
            "min": rate * Decimal("0.5"),
            "max": rate * Decimal("1.5"),
        }

    def estimate_energy_from_biogas(
        self, biogas_volume: Decimal
    ) -> Decimal:
        """
        Estimate electrical energy output from biogas combustion in CHP.

        Formula:
            Energy (kWh) = biogas_volume (m3) x energy_content (kWh/m3)
                           x electrical_efficiency

        Args:
            biogas_volume: Biogas volume in m3.

        Returns:
            Estimated electrical energy output in kWh.
        """
        energy_content = BIOGAS_ENERGY_CONTENT["kwh_per_m3_raw"]
        elec_eff = BIOGAS_ENERGY_CONTENT["chp_electrical_efficiency"]

        energy_kwh = biogas_volume * energy_content * elec_eff

        logger.debug(
            "estimate_energy_from_biogas: volume=%s m3, "
            "content=%s kWh/m3, efficiency=%s, energy=%s kWh",
            biogas_volume, energy_content, elec_eff, energy_kwh,
        )
        return _round_decimal(energy_kwh)

    def estimate_thermal_energy_from_biogas(
        self, biogas_volume: Decimal
    ) -> Decimal:
        """
        Estimate thermal energy output from biogas combustion in CHP.

        Args:
            biogas_volume: Biogas volume in m3.

        Returns:
            Estimated thermal energy output in kWh.
        """
        energy_content = BIOGAS_ENERGY_CONTENT["kwh_per_m3_raw"]
        thermal_eff = BIOGAS_ENERGY_CONTENT["chp_thermal_efficiency"]

        thermal_kwh = biogas_volume * energy_content * thermal_eff

        logger.debug(
            "estimate_thermal_energy_from_biogas: volume=%s m3, "
            "thermal=%s kWh",
            biogas_volume, thermal_kwh,
        )
        return _round_decimal(thermal_kwh)

    def get_biogas_yield(
        self, waste_category: WasteCategory
    ) -> Dict[str, Decimal]:
        """
        Get biogas yield data for a waste category.

        Args:
            waste_category: Waste material type.

        Returns:
            Dict with yield_m3_per_tonne, yield_min, yield_max, etc.
        """
        entry = BIOGAS_YIELD.get(waste_category)
        if entry is not None:
            return dict(entry)

        default = BIOGAS_YIELD[WasteCategory.OTHER]
        logger.warning(
            "get_biogas_yield: no data for %s, using OTHER defaults",
            waste_category.value,
        )
        return dict(default)

    def get_biogas_ch4_content(
        self, waste_category: WasteCategory
    ) -> Dict[str, Decimal]:
        """
        Get typical biogas CH4 content for a waste category.

        Args:
            waste_category: Waste material type.

        Returns:
            Dict with default, min, max CH4 content fractions.
        """
        # Map waste categories to biogas CH4 content keys
        key_map: Dict[WasteCategory, str] = {
            WasteCategory.FOOD_WASTE: "food_waste",
            WasteCategory.GARDEN_WASTE: "garden_waste",
            WasteCategory.MIXED_MSW: "mixed_waste",
        }

        key = key_map.get(waste_category, "default")
        entry = BIOGAS_CH4_CONTENT.get(key, BIOGAS_CH4_CONTENT["default"])
        return dict(entry)

    # ==========================================================================
    # PROVENANCE AND HASHING
    # ==========================================================================

    def compute_provenance_hash(
        self,
        input_data: Union[RecyclingInput, CompostingInput, AnaerobicDigestionInput],
        result: RecyclingCompostingResult,
    ) -> str:
        """
        Compute SHA-256 provenance hash for a calculation.

        Hashes the input parameters together with the output to create
        a tamper-evident audit record.

        Args:
            input_data: Calculation input.
            result: Calculation result.

        Returns:
            SHA-256 hex digest string.
        """
        input_str = input_data.model_dump_json(indent=None)
        output_str = result.model_dump_json(indent=None)
        combined = f"{AGENT_ID}|{ENGINE_ID}|{input_str}|{output_str}"
        prov_hash = _compute_hash(combined)

        logger.debug(
            "compute_provenance_hash: hash=%s (first 16 chars)",
            prov_hash[:16],
        )
        return prov_hash

    def compute_input_hash(
        self,
        input_data: Union[RecyclingInput, CompostingInput, AnaerobicDigestionInput],
    ) -> str:
        """
        Compute SHA-256 hash of input data only.

        Args:
            input_data: Calculation input.

        Returns:
            SHA-256 hex digest of the input.
        """
        return _compute_hash(input_data.model_dump_json(indent=None))

    def compute_output_hash(self, result: RecyclingCompostingResult) -> str:
        """
        Compute SHA-256 hash of output data only.

        Args:
            result: Calculation result.

        Returns:
            SHA-256 hex digest of the output.
        """
        return _compute_hash(result.model_dump_json(indent=None))

    # ==========================================================================
    # EMISSION FACTOR QUERIES
    # ==========================================================================

    def get_all_recycling_efs(self) -> Dict[str, Dict[str, Decimal]]:
        """
        Get all recycling process emission factors.

        Returns:
            Dict keyed by waste category value with EF data.
        """
        result: Dict[str, Dict[str, Decimal]] = {}
        for cat, ef_data in RECYCLING_PROCESS_EF.items():
            result[cat.value] = {
                "transport_and_sorting": ef_data["transport_and_sorting"],
                "mrf_sorting": ef_data["mrf_sorting"],
                "reprocessing": ef_data["reprocessing"],
            }
        return result

    def get_all_virgin_production_efs(self) -> Dict[str, Dict[str, Decimal]]:
        """
        Get all virgin production emission factors for avoided emissions.

        Returns:
            Dict keyed by waste category value with virgin/recycled/avoided EFs.
        """
        result: Dict[str, Dict[str, Decimal]] = {}
        for cat, ef_data in VIRGIN_PRODUCTION_EF.items():
            result[cat.value] = {
                "virgin_production_ef": ef_data["virgin_production_ef"],
                "recycled_production_ef": ef_data["recycled_production_ef"],
                "avoided_ef": ef_data["avoided_ef"],
            }
        return result

    def get_all_biogas_yields(self) -> Dict[str, Dict[str, Decimal]]:
        """
        Get all biogas yield data by waste category.

        Returns:
            Dict keyed by waste category value with yield data.
        """
        result: Dict[str, Dict[str, Decimal]] = {}
        for cat, yield_data in BIOGAS_YIELD.items():
            result[cat.value] = {
                "yield_m3_per_tonne": yield_data["yield_m3_per_tonne"],
                "yield_min": yield_data["yield_min"],
                "yield_max": yield_data["yield_max"],
            }
        return result

    def get_all_leakage_rates(self) -> Dict[str, Dict[str, Decimal]]:
        """
        Get all AD leakage rates by plant type.

        Returns:
            Dict keyed by plant type with default/min/max rates.
        """
        result: Dict[str, Dict[str, Decimal]] = {}
        for plant_type, rate_data in AD_LEAKAGE_RATES_EXTENDED.items():
            result[plant_type] = {
                "default": rate_data["default"],
                "min": rate_data["min"],
                "max": rate_data["max"],
            }
        return result

    def get_all_digestate_efs(self) -> Dict[str, Dict[str, Decimal]]:
        """
        Get all digestate storage emission factors.

        Returns:
            Dict keyed by storage type with CH4/N2O/total EFs.
        """
        result: Dict[str, Dict[str, Decimal]] = {}
        for storage_type, ef_data in DIGESTATE_EF.items():
            result[storage_type] = {
                "ch4_ef_kg_per_tonne": ef_data["ch4_ef_kg_per_tonne"],
                "n2o_ef_kg_per_tonne": ef_data["n2o_ef_kg_per_tonne"],
                "total_co2e_per_tonne": ef_data["total_co2e_per_tonne"],
            }
        return result

    # ==========================================================================
    # TRANSPORT HELPERS
    # ==========================================================================

    def get_transport_ef(self, mode: str) -> Decimal:
        """
        Get transport emission factor for a given mode.

        Args:
            mode: Transport mode (road, rail, ship, etc.).

        Returns:
            Transport EF in kgCO2e per tonne-km.

        Raises:
            ValueError: If mode is unknown.
        """
        ef = TRANSPORT_MODE_EF.get(mode)
        if ef is None:
            raise ValueError(
                f"Unknown transport mode '{mode}'. "
                f"Valid: {sorted(TRANSPORT_MODE_EF.keys())}"
            )
        return ef

    def get_all_transport_efs(self) -> Dict[str, Decimal]:
        """
        Get all transport mode emission factors.

        Returns:
            Dict keyed by transport mode with EF in kgCO2e/tonne-km.
        """
        return dict(TRANSPORT_MODE_EF)

    # ==========================================================================
    # UTILITY: CATEGORY LOOKUPS
    # ==========================================================================

    def get_recyclable_categories(self) -> List[str]:
        """
        Get list of waste categories eligible for recycling.

        Returns:
            Sorted list of recyclable waste category values.
        """
        return sorted(c.value for c in RECYCLABLE_CATEGORIES)

    def get_compostable_categories(self) -> List[str]:
        """
        Get list of waste categories eligible for composting.

        Returns:
            Sorted list of compostable waste category values.
        """
        return sorted(c.value for c in COMPOSTABLE_CATEGORIES)

    def get_ad_compatible_categories(self) -> List[str]:
        """
        Get list of waste categories eligible for anaerobic digestion.

        Returns:
            Sorted list of AD-compatible waste category values.
        """
        return sorted(c.value for c in AD_COMPATIBLE_CATEGORIES)

    def is_recyclable(self, waste_category: WasteCategory) -> bool:
        """
        Check if a waste category is recyclable.

        Args:
            waste_category: Waste material type.

        Returns:
            True if the category can be recycled.
        """
        return waste_category in RECYCLABLE_CATEGORIES

    def is_compostable(self, waste_category: WasteCategory) -> bool:
        """
        Check if a waste category is compostable.

        Args:
            waste_category: Waste material type.

        Returns:
            True if the category can be composted.
        """
        return waste_category in COMPOSTABLE_CATEGORIES

    def is_ad_compatible(self, waste_category: WasteCategory) -> bool:
        """
        Check if a waste category is compatible with anaerobic digestion.

        Args:
            waste_category: Waste material type.

        Returns:
            True if the category can be processed via AD.
        """
        return waste_category in AD_COMPATIBLE_CATEGORIES

    # ==========================================================================
    # GWP CONFIGURATION
    # ==========================================================================

    def get_gwp_version(self) -> GWPVersion:
        """
        Get the configured GWP version.

        Returns:
            Currently configured GWP assessment report version.
        """
        return self._gwp_version

    def get_gwp_values(self) -> Dict[str, Decimal]:
        """
        Get the currently configured GWP conversion factors.

        Returns:
            Dict with ch4 and n2o GWP values.
        """
        return {
            "ch4": self._gwp_ch4,
            "n2o": self._gwp_n2o,
            "gwp_version": Decimal(
                str(hash(self._gwp_version.value) % 10000)
            ),
        }

    def set_gwp_version(self, version: GWPVersion) -> None:
        """
        Update the GWP version for subsequent calculations.

        This changes the GWP conversion factors used for CH4 and N2O.
        It is thread-safe but affects all subsequent calculations.

        Args:
            version: New GWP assessment report version.
        """
        gwp_table = GWP_VALUES.get(version)
        if gwp_table is None:
            raise ValueError(
                f"Unknown GWP version '{version}'. "
                f"Valid: {[v.value for v in GWPVersion]}"
            )

        self._gwp_version = version
        self._gwp_ch4 = gwp_table["ch4"]
        self._gwp_n2o = gwp_table["n2o"]

        logger.info(
            "GWP version updated: %s (CH4=%s, N2O=%s)",
            version.value, self._gwp_ch4, self._gwp_n2o,
        )

    # ==========================================================================
    # SUMMARY / COMPARISON METHODS
    # ==========================================================================

    def compare_treatment_emissions(
        self,
        mass: Decimal,
        waste_category: WasteCategory,
    ) -> Dict[str, Dict[str, Decimal]]:
        """
        Compare emissions across available treatment methods for a waste category.

        Returns treatment emissions for each compatible method, useful for
        waste hierarchy analysis and decision support.

        Args:
            mass: Waste mass in tonnes.
            waste_category: Waste material type.

        Returns:
            Dict keyed by method name with treatment_co2e and avoided_co2e_memo.
        """
        results: Dict[str, Dict[str, Decimal]] = {}

        # Recycling (if compatible)
        if waste_category in RECYCLABLE_CATEGORIES:
            try:
                closed = self.calculate_closed_loop(mass, waste_category)
                results["recycling_closed_loop"] = {
                    "treatment_co2e": closed.treatment_emissions_co2e,
                    "avoided_co2e_memo": closed.avoided_emissions_memo_co2e or Decimal("0"),
                }
            except (ValueError, KeyError) as exc:
                logger.warning("Could not calculate closed-loop recycling: %s", exc)

            try:
                open_loop = self.calculate_open_loop(
                    mass, waste_category, Decimal("0.8")
                )
                results["recycling_open_loop_q80"] = {
                    "treatment_co2e": open_loop.treatment_emissions_co2e,
                    "avoided_co2e_memo": open_loop.avoided_emissions_memo_co2e or Decimal("0"),
                }
            except (ValueError, KeyError) as exc:
                logger.warning("Could not calculate open-loop recycling: %s", exc)

        # Composting (if compatible)
        if waste_category in COMPOSTABLE_CATEGORIES:
            try:
                comp_input = CompostingInput(
                    mass_tonnes=mass,
                    waste_category=waste_category,
                )
                comp_result = self.calculate_composting(comp_input)
                results["composting_industrial"] = {
                    "treatment_co2e": comp_result.treatment_emissions_co2e,
                    "avoided_co2e_memo": Decimal("0"),
                }
            except (ValueError, KeyError) as exc:
                logger.warning("Could not calculate composting: %s", exc)

        # Anaerobic digestion (if compatible)
        if waste_category in AD_COMPATIBLE_CATEGORIES:
            try:
                ad_input = AnaerobicDigestionInput(
                    mass_tonnes=mass,
                    waste_category=waste_category,
                )
                ad_result = self.calculate_anaerobic_digestion(ad_input)
                results["anaerobic_digestion"] = {
                    "treatment_co2e": ad_result.treatment_emissions_co2e,
                    "avoided_co2e_memo": Decimal("0"),
                }
            except (ValueError, KeyError) as exc:
                logger.warning("Could not calculate AD: %s", exc)

        logger.info(
            "compare_treatment_emissions: category=%s, methods_compared=%d",
            waste_category.value, len(results),
        )
        return results

    def calculate_diversion_benefit(
        self,
        mass: Decimal,
        waste_category: WasteCategory,
        treatment_method: str,
    ) -> Dict[str, Decimal]:
        """
        Calculate the emissions benefit of diverting waste from landfill.

        Compares the given treatment method emissions to a hypothetical
        landfill scenario to quantify diversion benefit.

        Args:
            mass: Waste mass in tonnes.
            waste_category: Waste material type.
            treatment_method: One of 'recycling', 'composting',
                              'anaerobic_digestion'.

        Returns:
            Dict with landfill_co2e, treatment_co2e, benefit_co2e, and
            benefit_pct keys.
        """
        # Get DEFRA landfill EF as baseline
        defra_entry = DEFRA_WASTE_FACTORS.get(waste_category, {})
        landfill_ef = defra_entry.get("landfill", Decimal("578"))  # MSW default
        landfill_co2e = mass * landfill_ef / KG_PER_TONNE

        # Calculate treatment emissions
        treatment_co2e = Decimal("0")
        if treatment_method == "recycling" and waste_category in RECYCLABLE_CATEGORIES:
            result = self.calculate_closed_loop(mass, waste_category)
            treatment_co2e = result.treatment_emissions_co2e
        elif treatment_method == "composting" and waste_category in COMPOSTABLE_CATEGORIES:
            comp_input = CompostingInput(
                mass_tonnes=mass, waste_category=waste_category
            )
            result = self.calculate_composting(comp_input)
            treatment_co2e = result.treatment_emissions_co2e
        elif treatment_method == "anaerobic_digestion" and waste_category in AD_COMPATIBLE_CATEGORIES:
            ad_input = AnaerobicDigestionInput(
                mass_tonnes=mass, waste_category=waste_category
            )
            result = self.calculate_anaerobic_digestion(ad_input)
            treatment_co2e = result.treatment_emissions_co2e

        benefit_co2e = landfill_co2e - treatment_co2e
        benefit_pct = Decimal("0")
        if landfill_co2e > Decimal("0"):
            benefit_pct = _round_decimal(
                (benefit_co2e / landfill_co2e) * Decimal("100"), 2
            )

        return {
            "landfill_co2e": _round_decimal(landfill_co2e),
            "treatment_co2e": _round_decimal(treatment_co2e),
            "benefit_co2e": _round_decimal(benefit_co2e),
            "benefit_pct": benefit_pct,
        }

    # ==========================================================================
    # ENGINE METADATA
    # ==========================================================================

    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get engine metadata and configuration summary.

        Returns:
            Dict with engine_id, version, gwp, supported methods, and counts.
        """
        return {
            "engine_id": ENGINE_ID,
            "engine_version": ENGINE_VERSION,
            "agent_id": AGENT_ID,
            "agent_version": VERSION,
            "gwp_version": self._gwp_version.value,
            "gwp_ch4": str(self._gwp_ch4),
            "gwp_n2o": str(self._gwp_n2o),
            "ef_source": self._ef_source.value,
            "supported_methods": [
                "recycling",
                "composting",
                "anaerobic_digestion",
            ],
            "recyclable_categories_count": len(RECYCLABLE_CATEGORIES),
            "compostable_categories_count": len(COMPOSTABLE_CATEGORIES),
            "ad_compatible_categories_count": len(AD_COMPATIBLE_CATEGORIES),
            "recycling_ef_count": len(RECYCLING_PROCESS_EF),
            "virgin_production_ef_count": len(VIRGIN_PRODUCTION_EF),
            "biogas_yield_count": len(BIOGAS_YIELD),
            "leakage_rate_count": len(AD_LEAKAGE_RATES_EXTENDED),
            "transport_mode_count": len(TRANSPORT_MODE_EF),
            "composting_ef_variants": len(COMPOSTING_EF_EXTENDED),
        }

    def get_supported_waste_categories(self) -> Dict[str, List[str]]:
        """
        Get supported waste categories grouped by treatment method.

        Returns:
            Dict with recycling, composting, and anaerobic_digestion lists.
        """
        return {
            "recycling": self.get_recyclable_categories(),
            "composting": self.get_compostable_categories(),
            "anaerobic_digestion": self.get_ad_compatible_categories(),
        }


# ==============================================================================
# SINGLETON ACCESSOR
# ==============================================================================

_engine_instance: Optional[RecyclingCompostingEngine] = None
_engine_lock: threading.Lock = threading.Lock()


def get_recycling_composting_engine(
    gwp_version: GWPVersion = GWPVersion.AR5,
    ef_source: EFSource = EFSource.DEFRA_BEIS,
) -> RecyclingCompostingEngine:
    """
    Get the singleton RecyclingCompostingEngine instance.

    Thread-safe factory function that returns a shared engine instance.
    The first call initializes the engine with the provided parameters;
    subsequent calls return the same instance.

    Args:
        gwp_version: GWP assessment report version (default AR5).
        ef_source: Default emission factor source (default DEFRA/BEIS).

    Returns:
        Shared RecyclingCompostingEngine instance.

    Example:
        >>> engine = get_recycling_composting_engine()
        >>> engine.get_engine_info()["engine_id"]
        'recycling_composting_engine'
    """
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = RecyclingCompostingEngine(
                    gwp_version=gwp_version,
                    ef_source=ef_source,
                )
                logger.info(
                    "RecyclingCompostingEngine singleton created via "
                    "get_recycling_composting_engine()"
                )
    return _engine_instance


def reset_recycling_composting_engine() -> None:
    """
    Reset the module-level singleton. Used in testing only.

    Clears both the module-level reference and the class-level singleton
    so the next call to ``get_recycling_composting_engine()`` creates a
    fresh instance.
    """
    global _engine_instance
    with _engine_lock:
        _engine_instance = None
    RecyclingCompostingEngine.reset()
    logger.info("RecyclingCompostingEngine singleton reset")


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "RecyclingCompostingEngine",
    "get_recycling_composting_engine",
    "reset_recycling_composting_engine",
    "RECYCLING_PROCESS_EF",
    "VIRGIN_PRODUCTION_EF",
    "BIOGAS_YIELD",
    "DIGESTATE_EF",
    "BIOGAS_COMBUSTION_EF",
    "TRANSPORT_EF_ROAD",
    "TRANSPORT_EF_RAIL",
    "TRANSPORT_EF_SHIP",
]
