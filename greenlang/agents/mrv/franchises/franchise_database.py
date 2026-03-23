# -*- coding: utf-8 -*-
"""
FranchiseDatabaseEngine - Reference data engine for franchise emission factors.

This module implements the FranchiseDatabaseEngine for AGENT-MRV-027
(Franchises, GHG Protocol Scope 3 Category 14). It provides thread-safe
singleton access to franchise EUI benchmarks, revenue intensity factors,
cooking fuel profiles, refrigerant data, grid/fuel emission factors,
EEIO factors, hotel energy benchmarks, vehicle EFs, double-counting rules,
compliance framework rules, DQI scoring, uncertainty ranges, and country
climate zone mappings.

Features:
- 15 reference data tables with in-memory storage and DB persistence
- 10 franchise types x 5 climate zones EUI benchmarks (kWh/m2/yr)
- Revenue intensity factors by franchise type (kgCO2e/$ revenue)
- Cooking fuel profiles by restaurant type (gas/propane/electric split)
- Refrigeration leakage rates for 4 equipment types
- Grid emission factors for 12 countries + 26 eGRID subregions
- Fuel emission factors for 8 fuel types
- EEIO spend factors for 10 NAICS codes
- Refrigerant GWPs for 10 common types
- Hotel energy benchmarks by class and climate zone
- Vehicle emission factors for delivery fleet
- 8 double-counting prevention rules (DC-FRN-001 to DC-FRN-008)
- 7 compliance frameworks with detailed requirements
- 5-dimension x 3-tier DQI scoring matrix
- Uncertainty ranges per calculation method and tier
- 30+ country climate zone mappings
- Thread-safe singleton pattern with __new__
- Zero-hallucination factor retrieval from frozen constant tables
- Provenance tracking via SHA-256 hashes
- Prometheus metrics recording for all lookups

Example:
    >>> engine = FranchiseDatabaseEngine()
    >>> eui = engine.get_eui_benchmark("qsr", "temperate")
    >>> eui
    Decimal('620')
    >>> ri = engine.get_revenue_intensity("hotel")
    >>> ri
    Decimal('0.35')

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-014
"""

import logging
import threading
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

AGENT_ID = "GL-MRV-S3-014"
AGENT_COMPONENT = "AGENT-MRV-027"
VERSION = "1.0.0"
TABLE_PREFIX = "gl_frn_"

# Quantization constant: 8 decimal places
_QUANT_8DP = Decimal("0.00000001")

# Quantization constant: 2 decimal places
_QUANT_2DP = Decimal("0.01")

# Quantization constant: 5 decimal places
_QUANT_5DP = Decimal("0.00001")


# ============================================================================
# METRICS COLLECTOR (graceful import)
# ============================================================================

_metrics_collector: Optional[Any] = None
_metrics_lock = threading.Lock()


def get_metrics_collector() -> Any:
    """
    Get the singleton metrics collector instance.

    Returns a no-op stub if the franchise metrics module is not available,
    ensuring the database engine can operate independently.

    Returns:
        Metrics collector instance or no-op stub.
    """
    global _metrics_collector
    if _metrics_collector is None:
        with _metrics_lock:
            if _metrics_collector is None:
                try:
                    from greenlang.agents.mrv.franchises.metrics import get_metrics
                    _metrics_collector = get_metrics()
                except ImportError:
                    logger.warning(
                        "Franchise metrics module not available; "
                        "using no-op metrics stub"
                    )
                    _metrics_collector = _NoOpMetrics()
    return _metrics_collector


class _NoOpMetrics:
    """No-op metrics stub when franchise metrics module is unavailable."""

    def record_factor_selection(self, **kwargs: Any) -> None:
        """No-op."""

    def record_calculation(self, **kwargs: Any) -> None:
        """No-op."""

    def record_batch(self, **kwargs: Any) -> None:
        """No-op."""


# ============================================================================
# PROVENANCE MANAGER (graceful import)
# ============================================================================

_provenance_manager: Optional[Any] = None
_provenance_lock = threading.Lock()


def get_provenance_manager() -> Any:
    """
    Get the singleton provenance manager instance.

    Returns a no-op stub if the franchise provenance module is not available.

    Returns:
        Provenance manager instance or no-op stub.
    """
    global _provenance_manager
    if _provenance_manager is None:
        with _provenance_lock:
            if _provenance_manager is None:
                try:
                    from greenlang.agents.mrv.franchises.provenance import (
                        get_provenance_tracker,
                    )
                    _provenance_manager = get_provenance_tracker()
                except ImportError:
                    logger.warning(
                        "Franchise provenance module not available; "
                        "using no-op provenance stub"
                    )
                    _provenance_manager = _NoOpProvenance()
    return _provenance_manager


class _NoOpProvenance:
    """No-op provenance stub when franchise provenance module is unavailable."""

    def start_chain(self, **kwargs: Any) -> str:
        """No-op chain start."""
        return "no-op-chain"

    def record_stage(self, *args: Any, **kwargs: Any) -> None:
        """No-op stage recording."""

    def seal_chain(self, *args: Any, **kwargs: Any) -> str:
        """No-op chain sealing."""
        return "no-op-hash"


# ============================================================================
# TABLE 1: FRANCHISE EUI BENCHMARKS
# 10 franchise types x 5 climate zones (kWh/m2/yr)
# ============================================================================

# Franchise type identifiers
FRANCHISE_TYPES: List[str] = [
    "qsr",                  # Quick-service restaurant
    "hotel",                # Hotel / lodging
    "convenience_store",    # Convenience store / mini-market
    "retail",               # General retail
    "fitness",              # Fitness center / gym
    "automotive",           # Automotive service (quick lube, tire, wash)
    "casual_dining",        # Casual dining restaurant
    "coffee_shop",          # Coffee shop / cafe
    "gas_station",          # Gas station with convenience
    "pharmacy",             # Pharmacy / drugstore
]

# Climate zone identifiers (ASHRAE simplified)
CLIMATE_ZONES: List[str] = [
    "tropical",       # ASHRAE zones 0A-1A (hot-humid)
    "arid",           # ASHRAE zones 2B-3B (hot-dry)
    "temperate",      # ASHRAE zones 3A-4A (mixed-humid)
    "continental",    # ASHRAE zones 5A-6A (cold)
    "polar",          # ASHRAE zones 7-8 (very cold / subarctic)
]

EUI_BENCHMARKS: Dict[str, Dict[str, Decimal]] = {
    # ---------------------------------------------------------------
    # QSR (Quick-Service Restaurant)
    # High EUI driven by cooking equipment (fryers, grills, ovens),
    # walk-in coolers/freezers, and HVAC in open-kitchen layouts.
    # Tropical: high cooling load; Polar: extended heating season.
    # ---------------------------------------------------------------
    "qsr": {
        "tropical": Decimal("750"),
        "arid": Decimal("680"),
        "temperate": Decimal("620"),
        "continental": Decimal("700"),
        "polar": Decimal("780"),
    },
    # ---------------------------------------------------------------
    # Hotel / Lodging
    # Moderate EUI; large HVAC zones, domestic hot water (DHW), laundry,
    # pool heating (where applicable). Polar climate inflates heating.
    # ---------------------------------------------------------------
    "hotel": {
        "tropical": Decimal("350"),
        "arid": Decimal("320"),
        "temperate": Decimal("280"),
        "continental": Decimal("340"),
        "polar": Decimal("400"),
    },
    # ---------------------------------------------------------------
    # Convenience Store
    # Open refrigeration cases run 24/7, contributing 40-60 % of total
    # EUI. Lighting is high-intensity for merchandising.
    # ---------------------------------------------------------------
    "convenience_store": {
        "tropical": Decimal("550"),
        "arid": Decimal("500"),
        "temperate": Decimal("450"),
        "continental": Decimal("520"),
        "polar": Decimal("600"),
    },
    # ---------------------------------------------------------------
    # Retail (General Merchandise)
    # Lower EUI than food-service; primarily HVAC and lighting. Large
    # floor plates dilute per-m2 energy when averaged.
    # ---------------------------------------------------------------
    "retail": {
        "tropical": Decimal("300"),
        "arid": Decimal("270"),
        "temperate": Decimal("240"),
        "continental": Decimal("280"),
        "polar": Decimal("330"),
    },
    # ---------------------------------------------------------------
    # Fitness Center / Gym
    # Elevated ventilation rates, pool/spa heating (if present),
    # hot-water demand for showers, and heavy HVAC due to occupant
    # metabolic heat gain.
    # ---------------------------------------------------------------
    "fitness": {
        "tropical": Decimal("400"),
        "arid": Decimal("360"),
        "temperate": Decimal("320"),
        "continental": Decimal("380"),
        "polar": Decimal("440"),
    },
    # ---------------------------------------------------------------
    # Automotive Service (Quick-lube, Tire, Car Wash)
    # Equipment loads (lifts, compressors), water heating for car wash,
    # partially conditioned bays.
    # ---------------------------------------------------------------
    "automotive": {
        "tropical": Decimal("380"),
        "arid": Decimal("340"),
        "temperate": Decimal("300"),
        "continental": Decimal("360"),
        "polar": Decimal("420"),
    },
    # ---------------------------------------------------------------
    # Casual Dining Restaurant
    # Similar to QSR but lower throughput per m2. Full-service kitchen
    # with exhaust hoods; dining-area HVAC is a bigger share.
    # ---------------------------------------------------------------
    "casual_dining": {
        "tropical": Decimal("650"),
        "arid": Decimal("590"),
        "temperate": Decimal("530"),
        "continental": Decimal("610"),
        "polar": Decimal("700"),
    },
    # ---------------------------------------------------------------
    # Coffee Shop / Cafe
    # Espresso machines, refrigeration, baking ovens (some locations),
    # high customer turnover with frequent door openings.
    # ---------------------------------------------------------------
    "coffee_shop": {
        "tropical": Decimal("480"),
        "arid": Decimal("430"),
        "temperate": Decimal("390"),
        "continental": Decimal("450"),
        "polar": Decimal("520"),
    },
    # ---------------------------------------------------------------
    # Gas Station with Convenience
    # Combination of convenience-store refrigeration plus canopy
    # lighting, fuel-pump electronics, and forecourt HVAC.
    # ---------------------------------------------------------------
    "gas_station": {
        "tropical": Decimal("500"),
        "arid": Decimal("460"),
        "temperate": Decimal("410"),
        "continental": Decimal("480"),
        "polar": Decimal("550"),
    },
    # ---------------------------------------------------------------
    # Pharmacy / Drugstore
    # Controlled-substance storage refrigeration, pharmacy lighting,
    # moderate HVAC loads.
    # ---------------------------------------------------------------
    "pharmacy": {
        "tropical": Decimal("340"),
        "arid": Decimal("310"),
        "temperate": Decimal("270"),
        "continental": Decimal("320"),
        "polar": Decimal("370"),
    },
}


# ============================================================================
# TABLE 2: REVENUE INTENSITY FACTORS
# 10 franchise types (kgCO2e per $ revenue)
# Source: EPA EEIO / GHG Protocol Scope 3 Technical Guidance (2024)
# ============================================================================

REVENUE_INTENSITY_FACTORS: Dict[str, Decimal] = {
    "qsr": Decimal("0.45"),
    "hotel": Decimal("0.35"),
    "convenience_store": Decimal("0.38"),
    "retail": Decimal("0.22"),
    "fitness": Decimal("0.28"),
    "automotive": Decimal("0.32"),
    "casual_dining": Decimal("0.42"),
    "coffee_shop": Decimal("0.36"),
    "gas_station": Decimal("0.40"),
    "pharmacy": Decimal("0.24"),
}


# ============================================================================
# TABLE 3: COOKING FUEL PROFILES
# By restaurant/food-service type (gas / propane / electric split %)
# Percentages represent share of total cooking energy by fuel type.
# Source: CBECS 2018, RECS 2020, industry surveys
# ============================================================================

COOKING_FUEL_PROFILES: Dict[str, Dict[str, Decimal]] = {
    # QSR: Gas-dominant (high-BTU fryers, charbroilers)
    "qsr": {
        "natural_gas_pct": Decimal("65"),
        "propane_pct": Decimal("15"),
        "electric_pct": Decimal("20"),
    },
    # Casual Dining: Similar to QSR but more electric ovens
    "casual_dining": {
        "natural_gas_pct": Decimal("55"),
        "propane_pct": Decimal("10"),
        "electric_pct": Decimal("35"),
    },
    # Coffee Shop: Mostly electric (espresso machines, convection ovens)
    "coffee_shop": {
        "natural_gas_pct": Decimal("20"),
        "propane_pct": Decimal("5"),
        "electric_pct": Decimal("75"),
    },
    # Convenience Store: Minimal cooking, mostly electric roller grills
    "convenience_store": {
        "natural_gas_pct": Decimal("10"),
        "propane_pct": Decimal("5"),
        "electric_pct": Decimal("85"),
    },
    # Hotel (kitchen): Gas ranges and ovens in full-service hotels
    "hotel": {
        "natural_gas_pct": Decimal("50"),
        "propane_pct": Decimal("10"),
        "electric_pct": Decimal("40"),
    },
    # Gas Station (food prep): Mostly electric
    "gas_station": {
        "natural_gas_pct": Decimal("15"),
        "propane_pct": Decimal("10"),
        "electric_pct": Decimal("75"),
    },
    # Default profile for non-food-service franchise types
    "default": {
        "natural_gas_pct": Decimal("30"),
        "propane_pct": Decimal("10"),
        "electric_pct": Decimal("60"),
    },
}


# ============================================================================
# TABLE 4: REFRIGERATION LEAKAGE RATES
# Equipment type -> annual leakage rate as fraction of charge (0-1)
# Source: EPA GreenChill, ASHRAE 15-2019, ARB 2020
# ============================================================================

REFRIGERATION_LEAKAGE_RATES: Dict[str, Decimal] = {
    "walk_in_cooler": Decimal("0.15"),    # 15% annual leakage
    "walk_in_freezer": Decimal("0.15"),   # 15% annual leakage
    "reach_in_cooler": Decimal("0.08"),   # 8% annual leakage
    "reach_in_freezer": Decimal("0.08"),  # 8% annual leakage
    "display_case": Decimal("0.12"),      # 12% annual leakage
    "ice_machine": Decimal("0.10"),       # 10% annual leakage
    "vending_machine": Decimal("0.06"),   # 6% annual leakage
    "hvac_split": Decimal("0.04"),        # 4% annual leakage
    "hvac_chiller": Decimal("0.02"),      # 2% annual leakage
    "hvac_rooftop": Decimal("0.05"),      # 5% annual leakage
    "transport_refrigeration": Decimal("0.20"),  # 20% (vibration, wear)
    "self_contained": Decimal("0.03"),    # 3% hermetic sealed
}


# ============================================================================
# TABLE 5: GRID EMISSION FACTORS
# 12 countries (kgCO2e/kWh) + 26 eGRID subregions (US)
# Source: IEA 2024, eGRID 2022, EEA 2023
# ============================================================================

GRID_EMISSION_FACTORS: Dict[str, Decimal] = {
    # ----- Country-level (ISO 3166-1 alpha-2) -----
    "US": Decimal("0.3937"),    # US national average (eGRID 2022)
    "GB": Decimal("0.2070"),    # UK DEFRA 2024
    "DE": Decimal("0.3380"),    # Germany UBA 2024
    "FR": Decimal("0.0520"),    # France - nuclear dominant
    "JP": Decimal("0.4570"),    # Japan MOE 2024
    "AU": Decimal("0.6800"),    # Australia CER 2024
    "CA": Decimal("0.1200"),    # Canada NIR 2024
    "IN": Decimal("0.7080"),    # India CEA 2024
    "BR": Decimal("0.0740"),    # Brazil MCTI 2024
    "CN": Decimal("0.5810"),    # China MEE 2024
    "KR": Decimal("0.4590"),    # South Korea KEEI 2024
    "MX": Decimal("0.4310"),    # Mexico SENER 2024
    # ----- US eGRID subregions -----
    "AKGD": Decimal("0.4420"),   # ASCC Alaska Grid
    "AKMS": Decimal("0.2810"),   # ASCC Miscellaneous
    "AZNM": Decimal("0.4180"),   # WECC Southwest
    "CAMX": Decimal("0.2280"),   # WECC California
    "ERCT": Decimal("0.3880"),   # ERCOT All
    "FRCC": Decimal("0.3910"),   # FRCC All
    "HIMS": Decimal("0.5430"),   # HICC Miscellaneous
    "HIOA": Decimal("0.6470"),   # HICC Oahu
    "MROE": Decimal("0.5070"),   # MRO East
    "MROW": Decimal("0.4310"),   # MRO West
    "NEWE": Decimal("0.2180"),   # NPCC New England
    "NWPP": Decimal("0.2860"),   # WECC Northwest
    "NYCW": Decimal("0.2520"),   # NPCC NYC/Westchester
    "NYLI": Decimal("0.4870"),   # NPCC Long Island
    "NYUP": Decimal("0.1260"),   # NPCC Upstate NY
    "PRMS": Decimal("0.5320"),   # Puerto Rico
    "RFCE": Decimal("0.2960"),   # RFC East
    "RFCM": Decimal("0.5350"),   # RFC Michigan
    "RFCW": Decimal("0.5090"),   # RFC West
    "RMPA": Decimal("0.5540"),   # WECC Rockies
    "SPNO": Decimal("0.4660"),   # SPP North
    "SPSO": Decimal("0.4350"),   # SPP South
    "SRMV": Decimal("0.3610"),   # SERC Mississippi Valley
    "SRMW": Decimal("0.6270"),   # SERC Midwest
    "SRSO": Decimal("0.3930"),   # SERC South
    "SRTV": Decimal("0.4360"),   # SERC Tennessee Valley
}


# ============================================================================
# TABLE 6: FUEL EMISSION FACTORS
# 8 fuel types with CO2, CH4, N2O factors
# Units: kgCO2/unit as listed, plus kgCO2e/unit with AR5 GWPs
# Source: DEFRA 2024, EPA AP-42, IPCC 2006 GL
# ============================================================================

FUEL_EMISSION_FACTORS: Dict[str, Dict[str, Any]] = {
    # Natural gas: kgCO2e per cubic metre (m3)
    "natural_gas": {
        "co2_per_unit": Decimal("2.02"),
        "ch4_per_unit": Decimal("0.000037"),
        "n2o_per_unit": Decimal("0.000035"),
        "co2e_per_unit": Decimal("2.03"),   # with AR5 GWPs
        "unit": "m3",
        "source": "DEFRA 2024",
        "description": "Natural gas (pipeline quality)",
    },
    # Diesel: kgCO2e per litre
    "diesel": {
        "co2_per_unit": Decimal("2.68"),
        "ch4_per_unit": Decimal("0.000120"),
        "n2o_per_unit": Decimal("0.000122"),
        "co2e_per_unit": Decimal("2.72"),
        "unit": "litre",
        "source": "DEFRA 2024",
        "description": "Diesel / gas oil",
    },
    # Petrol / Gasoline: kgCO2e per litre
    "petrol": {
        "co2_per_unit": Decimal("2.31"),
        "ch4_per_unit": Decimal("0.000190"),
        "n2o_per_unit": Decimal("0.000600"),
        "co2e_per_unit": Decimal("2.34"),
        "unit": "litre",
        "source": "DEFRA 2024",
        "description": "Petrol / motor gasoline",
    },
    # Propane / LPG: kgCO2e per litre
    "propane": {
        "co2_per_unit": Decimal("1.51"),
        "ch4_per_unit": Decimal("0.000058"),
        "n2o_per_unit": Decimal("0.000097"),
        "co2e_per_unit": Decimal("1.53"),
        "unit": "litre",
        "source": "DEFRA 2024",
        "description": "Propane / LPG",
    },
    # Heating oil (No. 2): kgCO2e per litre
    "heating_oil": {
        "co2_per_unit": Decimal("2.54"),
        "ch4_per_unit": Decimal("0.000120"),
        "n2o_per_unit": Decimal("0.000110"),
        "co2e_per_unit": Decimal("2.58"),
        "unit": "litre",
        "source": "DEFRA 2024",
        "description": "Heating oil / kerosene",
    },
    # Biodiesel (B100): kgCO2e per litre (biogenic excluded)
    "biodiesel": {
        "co2_per_unit": Decimal("0.00"),
        "ch4_per_unit": Decimal("0.000120"),
        "n2o_per_unit": Decimal("0.000122"),
        "co2e_per_unit": Decimal("0.04"),
        "unit": "litre",
        "source": "DEFRA 2024",
        "description": "Biodiesel B100 (biogenic CO2 excluded)",
    },
    # Wood pellets: kgCO2e per kg
    "wood_pellets": {
        "co2_per_unit": Decimal("0.01"),
        "ch4_per_unit": Decimal("0.000300"),
        "n2o_per_unit": Decimal("0.000040"),
        "co2e_per_unit": Decimal("0.02"),
        "unit": "kg",
        "source": "DEFRA 2024",
        "description": "Wood pellets (biogenic CO2 excluded)",
    },
    # Coal: kgCO2e per kg
    "coal": {
        "co2_per_unit": Decimal("2.42"),
        "ch4_per_unit": Decimal("0.000240"),
        "n2o_per_unit": Decimal("0.000050"),
        "co2e_per_unit": Decimal("2.45"),
        "unit": "kg",
        "source": "DEFRA 2024",
        "description": "Bituminous coal",
    },
}


# ============================================================================
# TABLE 7: EEIO SPEND FACTORS
# 10 NAICS codes relevant to franchise sectors (kgCO2e per $ of revenue)
# Source: EPA USEEIO v2.0.1-411, EXIOBASE 3.8
# ============================================================================

EEIO_SPEND_FACTORS: Dict[str, Dict[str, Any]] = {
    "722511": {
        "factor": Decimal("0.42"),
        "description": "Full-service restaurants (QSR mapped here)",
        "sector": "Food services - limited-service",
        "source": "USEEIO v2.0.1",
    },
    "721110": {
        "factor": Decimal("0.33"),
        "description": "Hotels (except casino hotels) and motels",
        "sector": "Accommodation",
        "source": "USEEIO v2.0.1",
    },
    "445120": {
        "factor": Decimal("0.36"),
        "description": "Convenience stores",
        "sector": "Food and beverage stores",
        "source": "USEEIO v2.0.1",
    },
    "452210": {
        "factor": Decimal("0.19"),
        "description": "Department stores (general retail mapped)",
        "sector": "General merchandise stores",
        "source": "USEEIO v2.0.1",
    },
    "713940": {
        "factor": Decimal("0.25"),
        "description": "Fitness and recreational sports centres",
        "sector": "Amusement and recreation",
        "source": "USEEIO v2.0.1",
    },
    "811111": {
        "factor": Decimal("0.30"),
        "description": "General automotive repair",
        "sector": "Repair and maintenance",
        "source": "USEEIO v2.0.1",
    },
    "722511_cd": {
        "factor": Decimal("0.39"),
        "description": "Full-service restaurants (casual dining)",
        "sector": "Food services - full-service",
        "source": "USEEIO v2.0.1",
    },
    "722515": {
        "factor": Decimal("0.34"),
        "description": "Snack and nonalcoholic beverage bars (coffee)",
        "sector": "Food services - snack bars",
        "source": "USEEIO v2.0.1",
    },
    "447110": {
        "factor": Decimal("0.38"),
        "description": "Gasoline stations with convenience stores",
        "sector": "Gasoline stations",
        "source": "USEEIO v2.0.1",
    },
    "446110": {
        "factor": Decimal("0.21"),
        "description": "Pharmacies and drug stores",
        "sector": "Health and personal care stores",
        "source": "USEEIO v2.0.1",
    },
}


# ============================================================================
# TABLE 8: REFRIGERANT GWPs
# 10 common refrigerant types, AR5 100-year GWP
# Source: IPCC AR5 WG1, EPA SNAP, EU F-Gas Regulation 517/2014
# ============================================================================

REFRIGERANT_GWPS: Dict[str, Dict[str, Any]] = {
    "R-134a": {
        "gwp": Decimal("1430"),
        "chemical_name": "1,1,1,2-Tetrafluoroethane",
        "class": "HFC",
        "ozone_depleting": False,
        "phase_down": True,
        "common_use": "Automotive AC, medium-temp commercial",
    },
    "R-410A": {
        "gwp": Decimal("2088"),
        "chemical_name": "R-32/R-125 (50/50 wt%)",
        "class": "HFC blend",
        "ozone_depleting": False,
        "phase_down": True,
        "common_use": "Residential/commercial HVAC",
    },
    "R-404A": {
        "gwp": Decimal("3922"),
        "chemical_name": "R-125/R-143a/R-134a (44/52/4 wt%)",
        "class": "HFC blend",
        "ozone_depleting": False,
        "phase_down": True,
        "common_use": "Commercial refrigeration, supermarkets",
    },
    "R-32": {
        "gwp": Decimal("675"),
        "chemical_name": "Difluoromethane",
        "class": "HFC",
        "ozone_depleting": False,
        "phase_down": True,
        "common_use": "Next-gen HVAC (lower GWP replacement)",
    },
    "R-407C": {
        "gwp": Decimal("1774"),
        "chemical_name": "R-32/R-125/R-134a (23/25/52 wt%)",
        "class": "HFC blend",
        "ozone_depleting": False,
        "phase_down": True,
        "common_use": "Rooftop units, chillers (R-22 replacement)",
    },
    "R-507A": {
        "gwp": Decimal("3985"),
        "chemical_name": "R-125/R-143a (50/50 wt%)",
        "class": "HFC blend",
        "ozone_depleting": False,
        "phase_down": True,
        "common_use": "Low-temp commercial refrigeration",
    },
    "R-22": {
        "gwp": Decimal("1810"),
        "chemical_name": "Chlorodifluoromethane",
        "class": "HCFC",
        "ozone_depleting": True,
        "phase_down": True,
        "common_use": "Legacy HVAC (phase-out in progress)",
    },
    "R-290": {
        "gwp": Decimal("3"),
        "chemical_name": "Propane",
        "class": "HC",
        "ozone_depleting": False,
        "phase_down": False,
        "common_use": "Small commercial, vending machines",
    },
    "R-744": {
        "gwp": Decimal("1"),
        "chemical_name": "Carbon dioxide (CO2)",
        "class": "Natural",
        "ozone_depleting": False,
        "phase_down": False,
        "common_use": "Transcritical CO2 supermarket systems",
    },
    "R-1234yf": {
        "gwp": Decimal("4"),
        "chemical_name": "2,3,3,3-Tetrafluoroprop-1-ene",
        "class": "HFO",
        "ozone_depleting": False,
        "phase_down": False,
        "common_use": "Automotive MAC (R-134a replacement)",
    },
}


# ============================================================================
# TABLE 9: HOTEL ENERGY BENCHMARKS
# 4 hotel classes x 5 climate zones (kWh/room-night)
# Source: ENERGY STAR Portfolio Manager, CBECS 2018, IEA Hotel Benchmarks
# ============================================================================

HOTEL_ENERGY_BENCHMARKS: Dict[str, Dict[str, Decimal]] = {
    "economy": {
        "tropical": Decimal("38"),
        "arid": Decimal("35"),
        "temperate": Decimal("30"),
        "continental": Decimal("36"),
        "polar": Decimal("44"),
    },
    "midscale": {
        "tropical": Decimal("55"),
        "arid": Decimal("50"),
        "temperate": Decimal("45"),
        "continental": Decimal("52"),
        "polar": Decimal("62"),
    },
    "upscale": {
        "tropical": Decimal("80"),
        "arid": Decimal("72"),
        "temperate": Decimal("65"),
        "continental": Decimal("75"),
        "polar": Decimal("90"),
    },
    "luxury": {
        "tropical": Decimal("120"),
        "arid": Decimal("108"),
        "temperate": Decimal("95"),
        "continental": Decimal("110"),
        "polar": Decimal("135"),
    },
}


# ============================================================================
# TABLE 10: VEHICLE EMISSION FACTORS
# Delivery vehicles (kgCO2e/km) -- TTW + WTT combined
# Source: DEFRA 2024, EPA SmartWay, GLEC Framework v3
# ============================================================================

VEHICLE_EMISSION_FACTORS: Dict[str, Dict[str, Any]] = {
    "light_van": {
        "ef_per_km": Decimal("0.21"),
        "description": "Light commercial vehicle < 3.5 t (diesel)",
        "source": "DEFRA 2024",
        "fuel_type": "diesel",
        "payload_capacity_kg": 800,
    },
    "medium_truck": {
        "ef_per_km": Decimal("0.47"),
        "description": "Medium rigid truck 3.5-7.5 t (diesel)",
        "source": "DEFRA 2024",
        "fuel_type": "diesel",
        "payload_capacity_kg": 3500,
    },
    "heavy_truck": {
        "ef_per_km": Decimal("0.89"),
        "description": "Heavy articulated truck > 33 t (diesel)",
        "source": "DEFRA 2024",
        "fuel_type": "diesel",
        "payload_capacity_kg": 25000,
    },
    "motorcycle": {
        "ef_per_km": Decimal("0.11"),
        "description": "Motorcycle / moped for last-mile delivery",
        "source": "DEFRA 2024",
        "fuel_type": "petrol",
        "payload_capacity_kg": 20,
    },
    "electric_van": {
        "ef_per_km": Decimal("0.05"),
        "description": "Battery-electric light van (grid average)",
        "source": "DEFRA 2024",
        "fuel_type": "electricity",
        "payload_capacity_kg": 600,
    },
    "car_small": {
        "ef_per_km": Decimal("0.15"),
        "description": "Small car (petrol, < 1.4L)",
        "source": "DEFRA 2024",
        "fuel_type": "petrol",
        "payload_capacity_kg": 200,
    },
    "car_medium": {
        "ef_per_km": Decimal("0.19"),
        "description": "Medium car (petrol, 1.4-2.0L)",
        "source": "DEFRA 2024",
        "fuel_type": "petrol",
        "payload_capacity_kg": 300,
    },
    "car_large": {
        "ef_per_km": Decimal("0.28"),
        "description": "Large car (petrol, > 2.0L) / SUV",
        "source": "DEFRA 2024",
        "fuel_type": "petrol",
        "payload_capacity_kg": 400,
    },
}


# ============================================================================
# TABLE 11: DOUBLE-COUNTING PREVENTION RULES
# 8 rules: DC-FRN-001 through DC-FRN-008
# Ensures emissions are allocated to the correct Scope/Category without
# double-counting between franchisor Scope 1/2 and franchisee reporting.
# ============================================================================

DC_RULES: List[Dict[str, Any]] = [
    {
        "rule_id": "DC-FRN-001",
        "severity": "CRITICAL",
        "title": "Exclude company-owned units",
        "description": (
            "Franchise emissions (Scope 3 Cat 14) must ONLY include franchisee-"
            "operated units. Company-owned and company-operated (COCO) units "
            "must be reported under the franchisor's Scope 1 and Scope 2. "
            "If ownership_type == 'company_owned', the unit MUST be excluded "
            "from Cat 14 and a ValueError raised."
        ),
        "check_field": "ownership_type",
        "excluded_values": ["company_owned", "coco"],
        "action": "REJECT",
        "ghg_reference": "GHG Protocol Scope 3, Chapter 14, Table 14.1",
    },
    {
        "rule_id": "DC-FRN-002",
        "severity": "HIGH",
        "title": "No overlap with Scope 2 purchased electricity",
        "description": (
            "Franchisee purchased electricity must not also appear in the "
            "franchisor's Scope 2 market-based or location-based inventories. "
            "Verify franchise unit is not listed in Scope 2 electricity "
            "purchase records."
        ),
        "check_field": "scope2_exclusion_verified",
        "action": "WARN_IF_MISSING",
        "ghg_reference": "GHG Protocol Scope 2 Guidance, Chapter 7",
    },
    {
        "rule_id": "DC-FRN-003",
        "severity": "HIGH",
        "title": "No overlap with Scope 1 stationary combustion",
        "description": (
            "Franchisee stationary combustion (gas cooking, heating) must "
            "not also appear in the franchisor's Scope 1 inventory. Verify "
            "franchise unit fuel records are not duplicated in Scope 1 "
            "combustion records."
        ),
        "check_field": "scope1_exclusion_verified",
        "action": "WARN_IF_MISSING",
        "ghg_reference": "GHG Protocol Corporate Standard, Chapter 4",
    },
    {
        "rule_id": "DC-FRN-004",
        "severity": "MEDIUM",
        "title": "No overlap with Cat 1 purchased goods and services",
        "description": (
            "Franchise-specific energy and refrigerant data must not also "
            "be counted as purchased goods/services in Cat 1. Ensure no "
            "material overlap in spend categorization."
        ),
        "check_field": "cat1_exclusion_verified",
        "action": "WARN_IF_MISSING",
        "ghg_reference": "GHG Protocol Scope 3, Chapter 1",
    },
    {
        "rule_id": "DC-FRN-005",
        "severity": "MEDIUM",
        "title": "No overlap with Cat 8 upstream leased assets",
        "description": (
            "If franchise units are treated as leased assets, emissions "
            "must be reported under EITHER Cat 8 or Cat 14, not both. "
            "Ownership classification determines the correct category."
        ),
        "check_field": "cat8_exclusion_verified",
        "action": "WARN_IF_MISSING",
        "ghg_reference": "GHG Protocol Scope 3, Chapters 8 and 14",
    },
    {
        "rule_id": "DC-FRN-006",
        "severity": "LOW",
        "title": "Avoid double-counting delivery vehicle emissions",
        "description": (
            "Franchisee delivery fleet mobile combustion must not also "
            "appear in Cat 4 (upstream transportation) or Cat 9 (downstream "
            "transportation). Verify transportation boundary."
        ),
        "check_field": "transport_boundary_verified",
        "action": "WARN_IF_MISSING",
        "ghg_reference": "GHG Protocol Scope 3, Chapters 4 and 9",
    },
    {
        "rule_id": "DC-FRN-007",
        "severity": "LOW",
        "title": "Avoid double-counting refrigerant emissions",
        "description": (
            "Franchisee refrigerant leakage must not also appear in the "
            "franchisor's Scope 1 fugitive emissions. Verify refrigerant "
            "equipment is franchisee-owned/operated."
        ),
        "check_field": "refrigerant_boundary_verified",
        "action": "WARN_IF_MISSING",
        "ghg_reference": "GHG Protocol Corporate Standard, Chapter 4",
    },
    {
        "rule_id": "DC-FRN-008",
        "severity": "MEDIUM",
        "title": "Temporal boundary alignment",
        "description": (
            "Franchise unit emissions must align with the franchisor's "
            "reporting period. Pro-rata adjustments required for units that "
            "open/close mid-period or change ownership type mid-period."
        ),
        "check_field": "reporting_period_aligned",
        "action": "WARN_IF_MISSING",
        "ghg_reference": "GHG Protocol Corporate Standard, Chapter 5",
    },
]


# ============================================================================
# TABLE 12: COMPLIANCE FRAMEWORK RULES
# 7 regulatory/voluntary frameworks with specific requirements
# ============================================================================

COMPLIANCE_FRAMEWORK_RULES: Dict[str, Dict[str, Any]] = {
    "ghg_protocol": {
        "framework_id": "GHG_PROTOCOL",
        "full_name": "GHG Protocol Corporate Value Chain (Scope 3) Standard",
        "version": "2011 (amended 2013)",
        "scope": "Scope 3, Category 14: Franchises",
        "mandatory_fields": [
            "franchise_type",
            "ownership_type",
            "reporting_period",
            "calculation_method",
            "emission_factor_source",
        ],
        "required_disclosures": [
            "Total Cat 14 emissions (tCO2e)",
            "Calculation method used",
            "Emission factor sources",
            "Number of franchise units included",
            "Percentage of units with primary data",
            "Data quality assessment",
            "Exclusions and justification",
        ],
        "data_quality_requirements": {
            "minimum_dqi_score": Decimal("2.0"),
            "primary_data_preferred": True,
            "uncertainty_disclosure_required": True,
        },
        "double_counting_rules": ["DC-FRN-001", "DC-FRN-002", "DC-FRN-003"],
        "reference_url": (
            "https://ghgprotocol.org/sites/default/files/"
            "standards/Scope3_Calculation_Guidance_0.pdf"
        ),
    },
    "iso_14064": {
        "framework_id": "ISO_14064",
        "full_name": "ISO 14064-1:2018 Quantification and reporting of GHG emissions",
        "version": "2018",
        "scope": "Category 4 - Indirect GHG emissions from products",
        "mandatory_fields": [
            "franchise_type",
            "ownership_type",
            "reporting_period",
            "calculation_method",
            "uncertainty_assessment",
        ],
        "required_disclosures": [
            "Indirect GHG emissions by category",
            "Quantification methodology",
            "Emission factors and data sources",
            "Uncertainty analysis",
            "Base year emissions",
            "Organizational boundary",
        ],
        "data_quality_requirements": {
            "minimum_dqi_score": Decimal("2.5"),
            "primary_data_preferred": True,
            "uncertainty_disclosure_required": True,
        },
        "double_counting_rules": ["DC-FRN-001", "DC-FRN-005"],
        "reference_url": "https://www.iso.org/standard/66453.html",
    },
    "csrd": {
        "framework_id": "CSRD",
        "full_name": "Corporate Sustainability Reporting Directive (EU) 2022/2464",
        "version": "ESRS E1 Climate Change (2023)",
        "scope": "ESRS E1-6 Scope 3 GHG emissions",
        "mandatory_fields": [
            "franchise_type",
            "ownership_type",
            "reporting_period",
            "calculation_method",
            "emission_factor_source",
            "data_quality_assessment",
        ],
        "required_disclosures": [
            "Scope 3 Cat 14 gross GHG emissions (tCO2e)",
            "Calculation approach and data sources",
            "Significant estimation uncertainties",
            "Proportion estimated vs measured",
            "Year-on-year change and explanation",
            "SBTi alignment status (if applicable)",
        ],
        "data_quality_requirements": {
            "minimum_dqi_score": Decimal("2.0"),
            "primary_data_preferred": True,
            "uncertainty_disclosure_required": True,
        },
        "double_counting_rules": [
            "DC-FRN-001", "DC-FRN-002", "DC-FRN-003", "DC-FRN-004",
        ],
        "reference_url": (
            "https://eur-lex.europa.eu/legal-content/"
            "EN/TXT/?uri=CELEX:32022L2464"
        ),
    },
    "cdp": {
        "framework_id": "CDP",
        "full_name": "CDP Climate Change Questionnaire",
        "version": "2024",
        "scope": "C6.5 - Scope 3 emissions by category",
        "mandatory_fields": [
            "franchise_type",
            "ownership_type",
            "reporting_period",
            "calculation_method",
        ],
        "required_disclosures": [
            "Cat 14 metric tonnes CO2e",
            "Evaluation status (relevant, not relevant, not evaluated)",
            "Calculation methodology",
            "Percentage calculated using primary data",
            "Explanation of changes year-on-year",
        ],
        "data_quality_requirements": {
            "minimum_dqi_score": Decimal("1.5"),
            "primary_data_preferred": True,
            "uncertainty_disclosure_required": False,
        },
        "double_counting_rules": ["DC-FRN-001"],
        "reference_url": "https://www.cdp.net/en/guidance/guidance-for-companies",
    },
    "sbti": {
        "framework_id": "SBTI",
        "full_name": "Science Based Targets initiative (SBTi)",
        "version": "Corporate Net-Zero Standard v1.1 (2024)",
        "scope": "Scope 3 target boundary",
        "mandatory_fields": [
            "franchise_type",
            "ownership_type",
            "reporting_period",
            "calculation_method",
            "base_year_emissions",
        ],
        "required_disclosures": [
            "Cat 14 emissions in base year and reporting year",
            "SBTi target progress",
            "Engagement strategy with franchisees",
            "Proportion with primary vs secondary data",
        ],
        "data_quality_requirements": {
            "minimum_dqi_score": Decimal("2.0"),
            "primary_data_preferred": True,
            "uncertainty_disclosure_required": True,
        },
        "double_counting_rules": ["DC-FRN-001", "DC-FRN-005"],
        "reference_url": "https://sciencebasedtargets.org/net-zero",
    },
    "gri": {
        "framework_id": "GRI",
        "full_name": "Global Reporting Initiative GRI 305: Emissions 2016",
        "version": "2016 (GRI 305-3)",
        "scope": "GRI 305-3 Other indirect (Scope 3) GHG emissions",
        "mandatory_fields": [
            "franchise_type",
            "ownership_type",
            "reporting_period",
            "calculation_method",
        ],
        "required_disclosures": [
            "Gross other indirect GHG emissions (tCO2e)",
            "Gases included (CO2, CH4, N2O, HFCs)",
            "Biogenic CO2 emissions (if any)",
            "Source of emission factors",
            "Standards and methodologies used",
        ],
        "data_quality_requirements": {
            "minimum_dqi_score": Decimal("1.5"),
            "primary_data_preferred": False,
            "uncertainty_disclosure_required": False,
        },
        "double_counting_rules": ["DC-FRN-001"],
        "reference_url": (
            "https://www.globalreporting.org/standards/"
            "media/1012/gri-305-emissions-2016.pdf"
        ),
    },
    "sec_climate": {
        "framework_id": "SEC_CLIMATE",
        "full_name": "SEC Climate-Related Disclosure Rule (S7-10-22)",
        "version": "2024 (Final Rule)",
        "scope": "Scope 3 if material / part of target",
        "mandatory_fields": [
            "franchise_type",
            "ownership_type",
            "reporting_period",
            "calculation_method",
            "materiality_assessment",
        ],
        "required_disclosures": [
            "Cat 14 emissions if material (tCO2e)",
            "Calculation methodology and assumptions",
            "Safe harbor disclosures",
            "Material changes in methodology",
        ],
        "data_quality_requirements": {
            "minimum_dqi_score": Decimal("2.0"),
            "primary_data_preferred": True,
            "uncertainty_disclosure_required": True,
        },
        "double_counting_rules": ["DC-FRN-001", "DC-FRN-004"],
        "reference_url": "https://www.sec.gov/rules/final/2024/33-11275.pdf",
    },
}


# ============================================================================
# TABLE 13: DQI SCORING MATRIX
# 5 dimensions x 3 tiers (Tier 1: primary, Tier 2: average, Tier 3: spend)
# Score range: 1 (best) to 5 (worst), per GHG Protocol DQI guidance
# ============================================================================

DQI_SCORING_MATRIX: Dict[str, Dict[str, Decimal]] = {
    # Data Source Quality
    "data_source": {
        "tier_1": Decimal("1.0"),   # Primary metered data from franchise unit
        "tier_2": Decimal("3.0"),   # Industry benchmarks / averages
        "tier_3": Decimal("4.0"),   # Spend-based EEIO estimates
    },
    # Temporal Representativeness
    "temporal": {
        "tier_1": Decimal("1.0"),   # Same reporting year
        "tier_2": Decimal("2.0"),   # Within 3 years
        "tier_3": Decimal("3.0"),   # Older than 3 years
    },
    # Geographical Representativeness
    "geographical": {
        "tier_1": Decimal("1.0"),   # Same country/region
        "tier_2": Decimal("2.0"),   # Same continent
        "tier_3": Decimal("4.0"),   # Global average
    },
    # Technological Representativeness
    "technological": {
        "tier_1": Decimal("1.0"),   # Same franchise type and equipment
        "tier_2": Decimal("2.0"),   # Same sector average
        "tier_3": Decimal("3.0"),   # Generic sector average
    },
    # Completeness
    "completeness": {
        "tier_1": Decimal("1.0"),   # All emission sources covered
        "tier_2": Decimal("2.0"),   # Major sources covered (> 80%)
        "tier_3": Decimal("3.0"),   # Partial coverage (< 80%)
    },
}


# ============================================================================
# TABLE 14: UNCERTAINTY RANGES
# By calculation method and tier (% uncertainty at 95% confidence)
# Source: IPCC 2006 Guidelines Vol 1 Ch 3, GHG Protocol Scope 3 Guidance
# ============================================================================

UNCERTAINTY_RANGES: Dict[str, Dict[str, Dict[str, Decimal]]] = {
    "franchise_specific": {
        "tier_1": {
            "lower_pct": Decimal("-10"),
            "upper_pct": Decimal("10"),
            "confidence": Decimal("95"),
            "description": "Primary metered data with full coverage",
        },
        "tier_2": {
            "lower_pct": Decimal("-15"),
            "upper_pct": Decimal("15"),
            "confidence": Decimal("95"),
            "description": "Primary data with some estimation/gaps",
        },
        "tier_3": {
            "lower_pct": Decimal("-25"),
            "upper_pct": Decimal("25"),
            "confidence": Decimal("95"),
            "description": "Primary data for subset, extrapolated",
        },
    },
    "average_data": {
        "tier_1": {
            "lower_pct": Decimal("-20"),
            "upper_pct": Decimal("20"),
            "confidence": Decimal("95"),
            "description": "Type-specific EUI with local climate zone",
        },
        "tier_2": {
            "lower_pct": Decimal("-30"),
            "upper_pct": Decimal("30"),
            "confidence": Decimal("95"),
            "description": "Type-specific EUI with default climate zone",
        },
        "tier_3": {
            "lower_pct": Decimal("-40"),
            "upper_pct": Decimal("40"),
            "confidence": Decimal("95"),
            "description": "Generic EUI average across types",
        },
    },
    "spend_based": {
        "tier_1": {
            "lower_pct": Decimal("-30"),
            "upper_pct": Decimal("30"),
            "confidence": Decimal("95"),
            "description": "Sector-specific EEIO with deflated spend",
        },
        "tier_2": {
            "lower_pct": Decimal("-40"),
            "upper_pct": Decimal("40"),
            "confidence": Decimal("95"),
            "description": "Broad-sector EEIO with nominal spend",
        },
        "tier_3": {
            "lower_pct": Decimal("-50"),
            "upper_pct": Decimal("50"),
            "confidence": Decimal("95"),
            "description": "Economy-wide average EEIO",
        },
    },
    "hybrid": {
        "tier_1": {
            "lower_pct": Decimal("-15"),
            "upper_pct": Decimal("15"),
            "confidence": Decimal("95"),
            "description": "Weighted hybrid with majority primary data",
        },
        "tier_2": {
            "lower_pct": Decimal("-25"),
            "upper_pct": Decimal("25"),
            "confidence": Decimal("95"),
            "description": "Weighted hybrid with mixed data sources",
        },
        "tier_3": {
            "lower_pct": Decimal("-35"),
            "upper_pct": Decimal("35"),
            "confidence": Decimal("95"),
            "description": "Weighted hybrid with majority estimates",
        },
    },
}


# ============================================================================
# TABLE 15: COUNTRY CLIMATE ZONE MAPPINGS
# 30+ countries mapped to simplified ASHRAE climate zone categories
# Source: ASHRAE 90.1-2019, Koppen-Geiger classification
# ============================================================================

COUNTRY_CLIMATE_ZONES: Dict[str, str] = {
    # ---- Tropical ----
    "SG": "tropical",     # Singapore
    "TH": "tropical",     # Thailand
    "MY": "tropical",     # Malaysia
    "ID": "tropical",     # Indonesia
    "PH": "tropical",     # Philippines
    "CO": "tropical",     # Colombia
    "NG": "tropical",     # Nigeria
    "KE": "tropical",     # Kenya
    "VN": "tropical",     # Vietnam
    "BR": "tropical",     # Brazil (dominant zone)
    # ---- Arid ----
    "SA": "arid",         # Saudi Arabia
    "AE": "arid",         # United Arab Emirates
    "EG": "arid",         # Egypt
    "AU": "arid",         # Australia (dominant zone)
    "MX": "arid",         # Mexico (dominant zone)
    "IN": "arid",         # India (dominant zone)
    "PK": "arid",         # Pakistan
    # ---- Temperate ----
    "US": "temperate",    # United States (dominant zone)
    "GB": "temperate",    # United Kingdom
    "FR": "temperate",    # France
    "ES": "temperate",    # Spain
    "IT": "temperate",    # Italy
    "PT": "temperate",    # Portugal
    "JP": "temperate",    # Japan
    "KR": "temperate",    # South Korea
    "CN": "temperate",    # China (dominant zone)
    "AR": "temperate",    # Argentina
    "CL": "temperate",    # Chile
    "NZ": "temperate",    # New Zealand
    "ZA": "temperate",    # South Africa
    "TR": "temperate",    # Turkey
    # ---- Continental ----
    "DE": "continental",  # Germany
    "PL": "continental",  # Poland
    "CZ": "continental",  # Czech Republic
    "AT": "continental",  # Austria
    "HU": "continental",  # Hungary
    "RO": "continental",  # Romania
    "UA": "continental",  # Ukraine
    "RU": "continental",  # Russia (dominant zone)
    "CA": "continental",  # Canada (dominant zone)
    # ---- Polar ----
    "IS": "polar",        # Iceland
    "NO": "polar",        # Norway
    "SE": "polar",        # Sweden
    "FI": "polar",        # Finland
    "DK": "polar",        # Denmark (borderline continental)
    "GL": "polar",        # Greenland
}


# ============================================================================
# VALID EF SOURCES
# Recognized emission factor data sources for provenance validation
# ============================================================================

VALID_EF_SOURCES: List[str] = [
    "DEFRA",
    "EPA",
    "IEA",
    "IPCC",
    "USEEIO",
    "EXIOBASE",
    "eGRID",
    "CBECS",
    "ENERGY_STAR",
    "GHG_PROTOCOL",
    "ECOINVENT",
    "ARB",
    "SUPPLIER_SPECIFIC",
    "CUSTOM",
]


# ============================================================================
# ENGINE CLASS
# ============================================================================


class FranchiseDatabaseEngine:
    """
    Thread-safe singleton engine for franchise emission factor lookups.

    Provides deterministic, zero-hallucination factor retrieval for all 15
    reference data tables covering franchise EUI benchmarks, revenue intensity,
    cooking profiles, refrigerant data, grid/fuel emission factors, EEIO
    factors, hotel benchmarks, vehicle EFs, double-counting rules, compliance
    frameworks, DQI scoring, uncertainty ranges, and climate zone mappings.

    This engine does NOT perform any LLM calls. All factors are retrieved
    from validated, frozen constant tables defined in this module.

    Thread Safety:
        Uses the __new__ singleton pattern with threading.Lock to ensure
        only one instance is created across all threads.

    Attributes:
        _lookup_count: Total number of factor lookups performed
        _lookup_lock: Lock protecting the lookup counter

    Example:
        >>> engine = FranchiseDatabaseEngine()
        >>> eui = engine.get_eui_benchmark("qsr", "temperate")
        >>> eui
        Decimal('620')
        >>> ef = engine.get_grid_ef("US")
        >>> ef
        Decimal('0.39370000')
    """

    _instance: Optional["FranchiseDatabaseEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "FranchiseDatabaseEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the database engine (only once due to singleton)."""
        if hasattr(self, "_initialized"):
            return

        self._initialized: bool = True
        self._lookup_count: int = 0
        self._lookup_lock: threading.RLock = threading.RLock()

        logger.info(
            "FranchiseDatabaseEngine initialized: "
            "franchise_types=%d, climate_zones=%d, "
            "grid_regions=%d, fuel_types=%d, "
            "eeio_codes=%d, refrigerants=%d, "
            "hotel_classes=%d, vehicle_types=%d, "
            "dc_rules=%d, frameworks=%d, "
            "dqi_dimensions=%d, countries=%d",
            len(FRANCHISE_TYPES),
            len(CLIMATE_ZONES),
            len(GRID_EMISSION_FACTORS),
            len(FUEL_EMISSION_FACTORS),
            len(EEIO_SPEND_FACTORS),
            len(REFRIGERANT_GWPS),
            len(HOTEL_ENERGY_BENCHMARKS),
            len(VEHICLE_EMISSION_FACTORS),
            len(DC_RULES),
            len(COMPLIANCE_FRAMEWORK_RULES),
            len(DQI_SCORING_MATRIX),
            len(COUNTRY_CLIMATE_ZONES),
        )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _increment_lookup(self) -> None:
        """Increment the lookup counter in a thread-safe manner."""
        with self._lookup_lock:
            self._lookup_count += 1

    def _record_factor_selection(self, source: str, category: str) -> None:
        """
        Record a factor selection in Prometheus metrics.

        Args:
            source: EF source identifier (e.g., "defra", "epa", "eeio")
            category: Factor category (e.g., "eui", "grid", "fuel", "eeio")
        """
        try:
            metrics = get_metrics_collector()
            metrics.record_factor_selection(source=source, category=category)
        except Exception as exc:
            logger.warning(
                "Failed to record factor selection metric: %s", exc
            )

    def _quantize(self, value: Decimal, precision: Decimal = _QUANT_8DP) -> Decimal:
        """
        Quantize a Decimal value with ROUND_HALF_UP.

        Args:
            value: Decimal value to quantize.
            precision: Quantization precision (default: 8 decimal places).

        Returns:
            Quantized Decimal value.
        """
        return value.quantize(precision, rounding=ROUND_HALF_UP)

    def _validate_franchise_type(self, franchise_type: str) -> str:
        """
        Validate and normalize franchise type identifier.

        Args:
            franchise_type: Franchise type string.

        Returns:
            Normalized lowercase franchise type.

        Raises:
            ValueError: If franchise_type is not recognized.
        """
        normalized = franchise_type.strip().lower()
        if normalized not in FRANCHISE_TYPES:
            raise ValueError(
                f"Unknown franchise type '{franchise_type}'. "
                f"Valid types: {FRANCHISE_TYPES}"
            )
        return normalized

    def _validate_climate_zone(self, climate_zone: str) -> str:
        """
        Validate and normalize climate zone identifier.

        Args:
            climate_zone: Climate zone string.

        Returns:
            Normalized lowercase climate zone.

        Raises:
            ValueError: If climate_zone is not recognized.
        """
        normalized = climate_zone.strip().lower()
        if normalized not in CLIMATE_ZONES:
            raise ValueError(
                f"Unknown climate zone '{climate_zone}'. "
                f"Valid zones: {CLIMATE_ZONES}"
            )
        return normalized

    # =========================================================================
    # TABLE 1: EUI BENCHMARK LOOKUPS
    # =========================================================================

    def get_eui_benchmark(
        self,
        franchise_type: str,
        climate_zone: str,
    ) -> Decimal:
        """
        Get Energy Use Intensity (EUI) benchmark for a franchise type and zone.

        Returns the annual energy consumption per unit floor area (kWh/m2/yr)
        from Table 1 of the franchise reference database.

        Args:
            franchise_type: One of 10 franchise types (e.g., "qsr", "hotel").
            climate_zone: One of 5 climate zones (e.g., "temperate", "arid").

        Returns:
            EUI benchmark in kWh/m2/yr as Decimal.

        Raises:
            ValueError: If franchise_type or climate_zone is not recognized.

        Example:
            >>> engine = FranchiseDatabaseEngine()
            >>> engine.get_eui_benchmark("qsr", "temperate")
            Decimal('620')
        """
        self._increment_lookup()
        ft = self._validate_franchise_type(franchise_type)
        cz = self._validate_climate_zone(climate_zone)

        eui_value = EUI_BENCHMARKS[ft][cz]
        self._record_factor_selection("CBECS", "eui")

        logger.debug(
            "EUI benchmark lookup: type=%s, zone=%s, eui=%s kWh/m2/yr",
            ft, cz, eui_value,
        )
        return eui_value

    def get_eui_benchmarks_for_type(
        self, franchise_type: str
    ) -> Dict[str, Decimal]:
        """
        Get all EUI benchmarks for a franchise type across all climate zones.

        Args:
            franchise_type: One of 10 franchise types.

        Returns:
            Dict mapping climate zone to EUI benchmark (kWh/m2/yr).

        Raises:
            ValueError: If franchise_type is not recognized.

        Example:
            >>> engine.get_eui_benchmarks_for_type("hotel")
            {'tropical': Decimal('350'), 'arid': Decimal('320'), ...}
        """
        self._increment_lookup()
        ft = self._validate_franchise_type(franchise_type)
        return dict(EUI_BENCHMARKS[ft])

    def get_all_eui_benchmarks(self) -> Dict[str, Dict[str, Decimal]]:
        """
        Get the entire EUI benchmark table (10 types x 5 zones).

        Returns:
            Nested dict: franchise_type -> climate_zone -> EUI (kWh/m2/yr).
        """
        self._increment_lookup()
        return {ft: dict(zones) for ft, zones in EUI_BENCHMARKS.items()}

    # =========================================================================
    # TABLE 2: REVENUE INTENSITY LOOKUPS
    # =========================================================================

    def get_revenue_intensity(self, franchise_type: str) -> Decimal:
        """
        Get revenue-based emission intensity factor for a franchise type.

        Returns the carbon intensity per dollar of revenue (kgCO2e/$) from
        Table 2 of the franchise reference database.

        Args:
            franchise_type: One of 10 franchise types (e.g., "qsr", "hotel").

        Returns:
            Revenue intensity factor in kgCO2e/$ as Decimal.

        Raises:
            ValueError: If franchise_type is not recognized.

        Example:
            >>> engine.get_revenue_intensity("hotel")
            Decimal('0.35')
        """
        self._increment_lookup()
        ft = self._validate_franchise_type(franchise_type)

        factor = REVENUE_INTENSITY_FACTORS[ft]
        self._record_factor_selection("USEEIO", "revenue_intensity")

        logger.debug(
            "Revenue intensity lookup: type=%s, factor=%s kgCO2e/$",
            ft, factor,
        )
        return factor

    def get_all_revenue_intensities(self) -> Dict[str, Decimal]:
        """
        Get all revenue intensity factors for all franchise types.

        Returns:
            Dict mapping franchise type to kgCO2e/$ factor.
        """
        self._increment_lookup()
        return dict(REVENUE_INTENSITY_FACTORS)

    # =========================================================================
    # TABLE 3: COOKING FUEL PROFILE LOOKUPS
    # =========================================================================

    def get_cooking_profile(self, franchise_type: str) -> Dict[str, Decimal]:
        """
        Get cooking fuel split profile for a franchise type.

        Returns the percentage split of cooking energy by fuel type
        (natural_gas_pct, propane_pct, electric_pct) from Table 3.

        Args:
            franchise_type: Franchise type (e.g., "qsr", "casual_dining").
                If the type has no specific profile, the "default" profile
                is returned.

        Returns:
            Dict with keys natural_gas_pct, propane_pct, electric_pct,
            each as Decimal percentage (0-100).

        Example:
            >>> engine.get_cooking_profile("qsr")
            {'natural_gas_pct': Decimal('65'), 'propane_pct': Decimal('15'),
             'electric_pct': Decimal('20')}
        """
        self._increment_lookup()
        ft = franchise_type.strip().lower()

        profile = COOKING_FUEL_PROFILES.get(ft)
        if profile is None:
            profile = COOKING_FUEL_PROFILES["default"]
            logger.debug(
                "No specific cooking profile for '%s'; using default", ft
            )

        self._record_factor_selection("CBECS", "cooking_profile")

        logger.debug(
            "Cooking profile lookup: type=%s, gas=%s%%, propane=%s%%, "
            "electric=%s%%",
            ft,
            profile["natural_gas_pct"],
            profile["propane_pct"],
            profile["electric_pct"],
        )
        return dict(profile)

    # =========================================================================
    # TABLE 4: REFRIGERATION LEAKAGE RATE LOOKUPS
    # =========================================================================

    def get_leakage_rate(self, equipment_type: str) -> Decimal:
        """
        Get annual refrigerant leakage rate for an equipment type.

        Returns the annual leakage rate as a fraction (0-1) of the total
        refrigerant charge from Table 4.

        Args:
            equipment_type: Equipment type (e.g., "walk_in_cooler",
                "display_case", "hvac_rooftop").

        Returns:
            Annual leakage rate as Decimal fraction (e.g., 0.15 = 15%).

        Raises:
            ValueError: If equipment_type is not recognized.

        Example:
            >>> engine.get_leakage_rate("walk_in_cooler")
            Decimal('0.15')
        """
        self._increment_lookup()
        et = equipment_type.strip().lower()

        rate = REFRIGERATION_LEAKAGE_RATES.get(et)
        if rate is None:
            raise ValueError(
                f"Unknown equipment type '{equipment_type}'. "
                f"Valid types: {list(REFRIGERATION_LEAKAGE_RATES.keys())}"
            )

        self._record_factor_selection("EPA_GREENCHILL", "leakage_rate")

        logger.debug(
            "Leakage rate lookup: equipment=%s, rate=%s (%.0f%%)",
            et, rate, rate * 100,
        )
        return rate

    def get_all_leakage_rates(self) -> Dict[str, Decimal]:
        """
        Get all refrigeration leakage rates.

        Returns:
            Dict mapping equipment type to annual leakage rate fraction.
        """
        self._increment_lookup()
        return dict(REFRIGERATION_LEAKAGE_RATES)

    # =========================================================================
    # TABLE 5: GRID EMISSION FACTOR LOOKUPS
    # =========================================================================

    def get_grid_ef(
        self,
        country: str,
        region: Optional[str] = None,
    ) -> Decimal:
        """
        Get grid electricity emission factor for a country or eGRID region.

        Looks up the emission factor (kgCO2e/kWh) from Table 5. For US
        locations, the eGRID subregion is preferred over the national average
        when a region code is provided.

        Args:
            country: ISO 3166-1 alpha-2 country code (e.g., "US", "GB").
            region: Optional eGRID subregion code (e.g., "CAMX", "ERCT").

        Returns:
            Grid emission factor in kgCO2e/kWh as quantized Decimal.

        Raises:
            ValueError: If country and region are both not found.

        Example:
            >>> engine.get_grid_ef("US", "CAMX")
            Decimal('0.22800000')
            >>> engine.get_grid_ef("GB")
            Decimal('0.20700000')
        """
        self._increment_lookup()

        # Prefer eGRID subregion if provided
        if region is not None:
            region_upper = region.strip().upper()
            ef = GRID_EMISSION_FACTORS.get(region_upper)
            if ef is not None:
                result = self._quantize(ef)
                self._record_factor_selection("eGRID", "grid_ef")
                logger.debug(
                    "Grid EF lookup: region=%s, ef=%s kgCO2e/kWh",
                    region_upper, result,
                )
                return result

        # Fall back to country-level
        country_upper = country.strip().upper()
        ef = GRID_EMISSION_FACTORS.get(country_upper)
        if ef is not None:
            result = self._quantize(ef)
            self._record_factor_selection("IEA", "grid_ef")
            logger.debug(
                "Grid EF lookup: country=%s, ef=%s kgCO2e/kWh",
                country_upper, result,
            )
            return result

        raise ValueError(
            f"Grid emission factor not found for country='{country}', "
            f"region='{region}'. Available keys: "
            f"{sorted(GRID_EMISSION_FACTORS.keys())}"
        )

    def get_all_grid_efs(self) -> Dict[str, Decimal]:
        """
        Get all grid emission factors.

        Returns:
            Dict mapping country/region code to kgCO2e/kWh.
        """
        self._increment_lookup()
        return dict(GRID_EMISSION_FACTORS)

    # =========================================================================
    # TABLE 6: FUEL EMISSION FACTOR LOOKUPS
    # =========================================================================

    def get_fuel_ef(self, fuel_type: str) -> Decimal:
        """
        Get fuel combustion emission factor for a fuel type.

        Returns the total CO2-equivalent emission factor (kgCO2e per unit)
        from Table 6. The unit depends on the fuel type (m3 for natural gas,
        litres for liquids, kg for solids).

        Args:
            fuel_type: Fuel type identifier (e.g., "natural_gas", "diesel").

        Returns:
            Emission factor in kgCO2e per native unit as quantized Decimal.

        Raises:
            ValueError: If fuel_type is not recognized.

        Example:
            >>> engine.get_fuel_ef("natural_gas")
            Decimal('2.03000000')
        """
        self._increment_lookup()
        ft = fuel_type.strip().lower()

        fuel_data = FUEL_EMISSION_FACTORS.get(ft)
        if fuel_data is None:
            raise ValueError(
                f"Unknown fuel type '{fuel_type}'. "
                f"Valid types: {list(FUEL_EMISSION_FACTORS.keys())}"
            )

        result = self._quantize(fuel_data["co2e_per_unit"])
        self._record_factor_selection("DEFRA", "fuel_ef")

        logger.debug(
            "Fuel EF lookup: fuel=%s, ef=%s kgCO2e/%s",
            ft, result, fuel_data["unit"],
        )
        return result

    def get_fuel_ef_detail(self, fuel_type: str) -> Dict[str, Any]:
        """
        Get detailed fuel emission factor data including CO2, CH4, N2O.

        Args:
            fuel_type: Fuel type identifier.

        Returns:
            Dict with co2_per_unit, ch4_per_unit, n2o_per_unit, co2e_per_unit,
            unit, source, description.

        Raises:
            ValueError: If fuel_type is not recognized.

        Example:
            >>> detail = engine.get_fuel_ef_detail("diesel")
            >>> detail["unit"]
            'litre'
        """
        self._increment_lookup()
        ft = fuel_type.strip().lower()

        fuel_data = FUEL_EMISSION_FACTORS.get(ft)
        if fuel_data is None:
            raise ValueError(
                f"Unknown fuel type '{fuel_type}'. "
                f"Valid types: {list(FUEL_EMISSION_FACTORS.keys())}"
            )

        return dict(fuel_data)

    def get_all_fuel_efs(self) -> Dict[str, Decimal]:
        """
        Get all fuel emission factors (kgCO2e/unit).

        Returns:
            Dict mapping fuel type to kgCO2e per unit.
        """
        self._increment_lookup()
        return {
            ft: data["co2e_per_unit"]
            for ft, data in FUEL_EMISSION_FACTORS.items()
        }

    # =========================================================================
    # TABLE 7: EEIO SPEND FACTOR LOOKUPS
    # =========================================================================

    def get_eeio_factor(self, naics_code: str) -> Decimal:
        """
        Get EEIO spend-based emission factor for a NAICS code.

        Returns the carbon intensity per dollar of spend (kgCO2e/$) from
        Table 7 using US EEIO model factors.

        Args:
            naics_code: NAICS code (e.g., "722511", "721110").

        Returns:
            EEIO emission factor in kgCO2e/$ as Decimal.

        Raises:
            ValueError: If naics_code is not found in the EEIO table.

        Example:
            >>> engine.get_eeio_factor("722511")
            Decimal('0.42')
        """
        self._increment_lookup()
        code = naics_code.strip()

        eeio_data = EEIO_SPEND_FACTORS.get(code)
        if eeio_data is None:
            raise ValueError(
                f"EEIO factor not found for NAICS code '{naics_code}'. "
                f"Available codes: {sorted(EEIO_SPEND_FACTORS.keys())}"
            )

        factor = eeio_data["factor"]
        self._record_factor_selection("USEEIO", "eeio")

        logger.debug(
            "EEIO factor lookup: naics=%s, factor=%s kgCO2e/$, desc='%s'",
            code, factor, eeio_data["description"],
        )
        return factor

    def get_eeio_factor_detail(self, naics_code: str) -> Dict[str, Any]:
        """
        Get detailed EEIO factor data including description and source.

        Args:
            naics_code: NAICS code.

        Returns:
            Dict with factor, description, sector, source.

        Raises:
            ValueError: If naics_code is not found.
        """
        self._increment_lookup()
        code = naics_code.strip()

        eeio_data = EEIO_SPEND_FACTORS.get(code)
        if eeio_data is None:
            raise ValueError(
                f"EEIO factor not found for NAICS code '{naics_code}'. "
                f"Available codes: {sorted(EEIO_SPEND_FACTORS.keys())}"
            )

        return dict(eeio_data)

    def get_all_eeio_factors(self) -> Dict[str, Decimal]:
        """
        Get all EEIO spend factors.

        Returns:
            Dict mapping NAICS code to kgCO2e/$ factor.
        """
        self._increment_lookup()
        return {code: data["factor"] for code, data in EEIO_SPEND_FACTORS.items()}

    # =========================================================================
    # TABLE 8: REFRIGERANT GWP LOOKUPS
    # =========================================================================

    def get_refrigerant_gwp(self, refrigerant_type: str) -> Decimal:
        """
        Get Global Warming Potential (GWP) for a refrigerant type.

        Returns the AR5 100-year GWP from Table 8.

        Args:
            refrigerant_type: Refrigerant designation (e.g., "R-410A", "R-32").

        Returns:
            GWP value as Decimal (dimensionless, relative to CO2 = 1).

        Raises:
            ValueError: If refrigerant_type is not recognized.

        Example:
            >>> engine.get_refrigerant_gwp("R-410A")
            Decimal('2088')
        """
        self._increment_lookup()
        rt = refrigerant_type.strip().upper()

        # Normalize to canonical form (e.g., "r410a" -> "R-410A")
        ref_data = REFRIGERANT_GWPS.get(rt)
        if ref_data is None:
            # Try case-insensitive search
            for key, data in REFRIGERANT_GWPS.items():
                if key.upper() == rt or key.replace("-", "").upper() == rt.replace("-", ""):
                    ref_data = data
                    break

        if ref_data is None:
            raise ValueError(
                f"Unknown refrigerant type '{refrigerant_type}'. "
                f"Valid types: {list(REFRIGERANT_GWPS.keys())}"
            )

        gwp = ref_data["gwp"]
        self._record_factor_selection("IPCC_AR5", "refrigerant_gwp")

        logger.debug(
            "Refrigerant GWP lookup: type=%s, gwp=%s, class=%s",
            refrigerant_type, gwp, ref_data["class"],
        )
        return gwp

    def get_refrigerant_detail(self, refrigerant_type: str) -> Dict[str, Any]:
        """
        Get detailed refrigerant data including GWP, class, and use.

        Args:
            refrigerant_type: Refrigerant designation.

        Returns:
            Dict with gwp, chemical_name, class, ozone_depleting,
            phase_down, common_use.

        Raises:
            ValueError: If refrigerant_type is not recognized.
        """
        self._increment_lookup()
        rt = refrigerant_type.strip().upper()

        ref_data = REFRIGERANT_GWPS.get(rt)
        if ref_data is None:
            for key, data in REFRIGERANT_GWPS.items():
                if key.upper() == rt or key.replace("-", "").upper() == rt.replace("-", ""):
                    ref_data = data
                    break

        if ref_data is None:
            raise ValueError(
                f"Unknown refrigerant type '{refrigerant_type}'. "
                f"Valid types: {list(REFRIGERANT_GWPS.keys())}"
            )

        return dict(ref_data)

    def get_all_refrigerant_gwps(self) -> Dict[str, Decimal]:
        """
        Get all refrigerant GWPs.

        Returns:
            Dict mapping refrigerant type to GWP value.
        """
        self._increment_lookup()
        return {rt: data["gwp"] for rt, data in REFRIGERANT_GWPS.items()}

    # =========================================================================
    # TABLE 9: HOTEL ENERGY BENCHMARK LOOKUPS
    # =========================================================================

    def get_hotel_benchmark(
        self,
        class_type: str,
        climate_zone: str,
    ) -> Decimal:
        """
        Get hotel energy benchmark by class and climate zone.

        Returns energy consumption per room-night (kWh/room-night)
        from Table 9.

        Args:
            class_type: Hotel class ("economy", "midscale", "upscale", "luxury").
            climate_zone: One of 5 climate zones.

        Returns:
            Energy benchmark in kWh/room-night as Decimal.

        Raises:
            ValueError: If class_type or climate_zone is not recognized.

        Example:
            >>> engine.get_hotel_benchmark("luxury", "tropical")
            Decimal('120')
        """
        self._increment_lookup()
        ct = class_type.strip().lower()
        cz = self._validate_climate_zone(climate_zone)

        class_data = HOTEL_ENERGY_BENCHMARKS.get(ct)
        if class_data is None:
            raise ValueError(
                f"Unknown hotel class '{class_type}'. "
                f"Valid classes: {list(HOTEL_ENERGY_BENCHMARKS.keys())}"
            )

        benchmark = class_data[cz]
        self._record_factor_selection("ENERGY_STAR", "hotel_benchmark")

        logger.debug(
            "Hotel benchmark lookup: class=%s, zone=%s, "
            "benchmark=%s kWh/room-night",
            ct, cz, benchmark,
        )
        return benchmark

    def get_all_hotel_benchmarks(self) -> Dict[str, Dict[str, Decimal]]:
        """
        Get the entire hotel energy benchmark table (4 classes x 5 zones).

        Returns:
            Nested dict: hotel_class -> climate_zone -> kWh/room-night.
        """
        self._increment_lookup()
        return {ct: dict(zones) for ct, zones in HOTEL_ENERGY_BENCHMARKS.items()}

    # =========================================================================
    # TABLE 10: VEHICLE EMISSION FACTOR LOOKUPS
    # =========================================================================

    def get_vehicle_ef(self, vehicle_type: str) -> Decimal:
        """
        Get vehicle emission factor for a delivery vehicle type.

        Returns the emission factor (kgCO2e/km) from Table 10, combining
        tank-to-wheel (TTW) and well-to-tank (WTT) emissions.

        Args:
            vehicle_type: Vehicle type (e.g., "light_van", "medium_truck").

        Returns:
            Emission factor in kgCO2e/km as Decimal.

        Raises:
            ValueError: If vehicle_type is not recognized.

        Example:
            >>> engine.get_vehicle_ef("light_van")
            Decimal('0.21')
        """
        self._increment_lookup()
        vt = vehicle_type.strip().lower()

        vehicle_data = VEHICLE_EMISSION_FACTORS.get(vt)
        if vehicle_data is None:
            raise ValueError(
                f"Unknown vehicle type '{vehicle_type}'. "
                f"Valid types: {list(VEHICLE_EMISSION_FACTORS.keys())}"
            )

        ef = vehicle_data["ef_per_km"]
        self._record_factor_selection("DEFRA", "vehicle_ef")

        logger.debug(
            "Vehicle EF lookup: type=%s, ef=%s kgCO2e/km, desc='%s'",
            vt, ef, vehicle_data["description"],
        )
        return ef

    def get_vehicle_ef_detail(self, vehicle_type: str) -> Dict[str, Any]:
        """
        Get detailed vehicle emission factor data.

        Args:
            vehicle_type: Vehicle type identifier.

        Returns:
            Dict with ef_per_km, description, source, fuel_type,
            payload_capacity_kg.

        Raises:
            ValueError: If vehicle_type is not recognized.
        """
        self._increment_lookup()
        vt = vehicle_type.strip().lower()

        vehicle_data = VEHICLE_EMISSION_FACTORS.get(vt)
        if vehicle_data is None:
            raise ValueError(
                f"Unknown vehicle type '{vehicle_type}'. "
                f"Valid types: {list(VEHICLE_EMISSION_FACTORS.keys())}"
            )

        return dict(vehicle_data)

    def get_all_vehicle_efs(self) -> Dict[str, Decimal]:
        """
        Get all vehicle emission factors.

        Returns:
            Dict mapping vehicle type to kgCO2e/km.
        """
        self._increment_lookup()
        return {
            vt: data["ef_per_km"]
            for vt, data in VEHICLE_EMISSION_FACTORS.items()
        }

    # =========================================================================
    # TABLE 11: DOUBLE-COUNTING RULES
    # =========================================================================

    def get_dc_rules(self) -> List[Dict[str, Any]]:
        """
        Get all double-counting prevention rules.

        Returns the 8 DC-FRN rules that define boundaries between
        Category 14 and other Scope/Category inventories.

        Returns:
            List of 8 rule dictionaries, each with rule_id, severity,
            title, description, check_field, action, ghg_reference.

        Example:
            >>> rules = engine.get_dc_rules()
            >>> rules[0]["rule_id"]
            'DC-FRN-001'
        """
        self._increment_lookup()
        # Return deep copy to prevent mutation
        return [dict(rule) for rule in DC_RULES]

    def get_dc_rule(self, rule_id: str) -> Dict[str, Any]:
        """
        Get a specific double-counting prevention rule by ID.

        Args:
            rule_id: Rule identifier (e.g., "DC-FRN-001").

        Returns:
            Rule dictionary.

        Raises:
            ValueError: If rule_id is not found.

        Example:
            >>> rule = engine.get_dc_rule("DC-FRN-001")
            >>> rule["severity"]
            'CRITICAL'
        """
        self._increment_lookup()
        for rule in DC_RULES:
            if rule["rule_id"] == rule_id:
                return dict(rule)

        raise ValueError(
            f"DC rule '{rule_id}' not found. "
            f"Available: {[r['rule_id'] for r in DC_RULES]}"
        )

    def get_dc_rules_by_severity(self, severity: str) -> List[Dict[str, Any]]:
        """
        Get double-counting rules filtered by severity level.

        Args:
            severity: Severity level ("CRITICAL", "HIGH", "MEDIUM", "LOW").

        Returns:
            List of matching rule dictionaries.
        """
        self._increment_lookup()
        severity_upper = severity.strip().upper()
        return [
            dict(rule) for rule in DC_RULES
            if rule["severity"] == severity_upper
        ]

    # =========================================================================
    # TABLE 12: COMPLIANCE FRAMEWORK RULES
    # =========================================================================

    def get_framework_rules(self, framework: str) -> Dict[str, Any]:
        """
        Get compliance framework rules and requirements.

        Args:
            framework: Framework identifier (e.g., "ghg_protocol", "csrd",
                "cdp", "sbti", "gri", "iso_14064", "sec_climate").

        Returns:
            Dict with framework_id, full_name, version, scope,
            mandatory_fields, required_disclosures,
            data_quality_requirements, double_counting_rules,
            reference_url.

        Raises:
            ValueError: If framework is not recognized.

        Example:
            >>> rules = engine.get_framework_rules("ghg_protocol")
            >>> rules["mandatory_fields"]
            ['franchise_type', 'ownership_type', ...]
        """
        self._increment_lookup()
        fw = framework.strip().lower()

        framework_data = COMPLIANCE_FRAMEWORK_RULES.get(fw)
        if framework_data is None:
            raise ValueError(
                f"Unknown framework '{framework}'. "
                f"Valid frameworks: {list(COMPLIANCE_FRAMEWORK_RULES.keys())}"
            )

        self._record_factor_selection("GHG_PROTOCOL", "compliance")

        logger.debug(
            "Framework rules lookup: framework=%s, disclosures=%d",
            fw, len(framework_data["required_disclosures"]),
        )
        return dict(framework_data)

    def get_all_framework_ids(self) -> List[str]:
        """
        Get all supported compliance framework identifiers.

        Returns:
            List of framework ID strings.
        """
        self._increment_lookup()
        return list(COMPLIANCE_FRAMEWORK_RULES.keys())

    def get_mandatory_fields(self, framework: str) -> List[str]:
        """
        Get mandatory input fields for a specific framework.

        Args:
            framework: Framework identifier.

        Returns:
            List of mandatory field names.

        Raises:
            ValueError: If framework is not recognized.
        """
        rules = self.get_framework_rules(framework)
        return rules["mandatory_fields"]

    # =========================================================================
    # TABLE 13: DQI SCORING
    # =========================================================================

    def get_dqi_score(self, dimension: str, tier: str) -> Decimal:
        """
        Get DQI (Data Quality Indicator) score for a dimension and tier.

        Args:
            dimension: DQI dimension ("data_source", "temporal",
                "geographical", "technological", "completeness").
            tier: Calculation tier ("tier_1", "tier_2", "tier_3").

        Returns:
            DQI score as Decimal (1.0 = best, 5.0 = worst).

        Raises:
            ValueError: If dimension or tier is not recognized.

        Example:
            >>> engine.get_dqi_score("data_source", "tier_1")
            Decimal('1.0')
        """
        self._increment_lookup()
        dim = dimension.strip().lower()
        t = tier.strip().lower()

        dim_data = DQI_SCORING_MATRIX.get(dim)
        if dim_data is None:
            raise ValueError(
                f"Unknown DQI dimension '{dimension}'. "
                f"Valid dimensions: {list(DQI_SCORING_MATRIX.keys())}"
            )

        score = dim_data.get(t)
        if score is None:
            raise ValueError(
                f"Unknown tier '{tier}' for dimension '{dimension}'. "
                f"Valid tiers: {list(dim_data.keys())}"
            )

        logger.debug(
            "DQI score lookup: dimension=%s, tier=%s, score=%s",
            dim, t, score,
        )
        return score

    def get_composite_dqi(self, tier: str) -> Decimal:
        """
        Calculate composite DQI score for a given tier.

        Composite = arithmetic mean of all 5 dimension scores for the tier.

        Args:
            tier: Calculation tier ("tier_1", "tier_2", "tier_3").

        Returns:
            Composite DQI score as Decimal (rounded to 2 decimal places).

        Example:
            >>> engine.get_composite_dqi("tier_1")
            Decimal('1.00')
        """
        self._increment_lookup()
        t = tier.strip().lower()

        total = Decimal("0")
        count = 0
        for dim_data in DQI_SCORING_MATRIX.values():
            score = dim_data.get(t)
            if score is not None:
                total += score
                count += 1

        if count == 0:
            raise ValueError(
                f"Unknown tier '{tier}'. Valid tiers: tier_1, tier_2, tier_3"
            )

        composite = total / Decimal(str(count))
        return self._quantize(composite, _QUANT_2DP)

    def get_dqi_matrix(self) -> Dict[str, Dict[str, Decimal]]:
        """
        Get the full DQI scoring matrix.

        Returns:
            Nested dict: dimension -> tier -> score.
        """
        self._increment_lookup()
        return {dim: dict(tiers) for dim, tiers in DQI_SCORING_MATRIX.items()}

    # =========================================================================
    # TABLE 14: UNCERTAINTY RANGES
    # =========================================================================

    def get_uncertainty_range(
        self, method: str, tier: str
    ) -> Dict[str, Decimal]:
        """
        Get uncertainty range for a calculation method and tier.

        Returns the percentage uncertainty bounds at 95% confidence from
        Table 14.

        Args:
            method: Calculation method ("franchise_specific", "average_data",
                "spend_based", "hybrid").
            tier: Data tier ("tier_1", "tier_2", "tier_3").

        Returns:
            Dict with lower_pct, upper_pct, confidence, description.

        Raises:
            ValueError: If method or tier is not recognized.

        Example:
            >>> rng = engine.get_uncertainty_range("franchise_specific", "tier_1")
            >>> rng["lower_pct"]
            Decimal('-10')
        """
        self._increment_lookup()
        m = method.strip().lower()
        t = tier.strip().lower()

        method_data = UNCERTAINTY_RANGES.get(m)
        if method_data is None:
            raise ValueError(
                f"Unknown calculation method '{method}'. "
                f"Valid methods: {list(UNCERTAINTY_RANGES.keys())}"
            )

        tier_data = method_data.get(t)
        if tier_data is None:
            raise ValueError(
                f"Unknown tier '{tier}' for method '{method}'. "
                f"Valid tiers: {list(method_data.keys())}"
            )

        logger.debug(
            "Uncertainty range lookup: method=%s, tier=%s, "
            "range=[%s%%, %s%%] at %s%% confidence",
            m, t,
            tier_data["lower_pct"], tier_data["upper_pct"],
            tier_data["confidence"],
        )
        return dict(tier_data)

    def get_all_uncertainty_ranges(
        self,
    ) -> Dict[str, Dict[str, Dict[str, Decimal]]]:
        """
        Get all uncertainty ranges.

        Returns:
            Nested dict: method -> tier -> range parameters.
        """
        self._increment_lookup()
        return {
            method: {tier: dict(data) for tier, data in tiers.items()}
            for method, tiers in UNCERTAINTY_RANGES.items()
        }

    # =========================================================================
    # TABLE 15: CLIMATE ZONE LOOKUPS
    # =========================================================================

    def get_climate_zone(self, country: str) -> str:
        """
        Get simplified climate zone for a country.

        Args:
            country: ISO 3166-1 alpha-2 country code (e.g., "US", "GB").

        Returns:
            Climate zone string ("tropical", "arid", "temperate",
            "continental", "polar").

        Raises:
            ValueError: If country is not found in the mapping.

        Example:
            >>> engine.get_climate_zone("US")
            'temperate'
        """
        self._increment_lookup()
        cc = country.strip().upper()

        zone = COUNTRY_CLIMATE_ZONES.get(cc)
        if zone is None:
            raise ValueError(
                f"Climate zone not found for country '{country}'. "
                f"Available countries: {sorted(COUNTRY_CLIMATE_ZONES.keys())}"
            )

        logger.debug(
            "Climate zone lookup: country=%s, zone=%s",
            cc, zone,
        )
        return zone

    def get_all_climate_zones(self) -> Dict[str, str]:
        """
        Get all country-to-climate-zone mappings.

        Returns:
            Dict mapping ISO alpha-2 country code to climate zone.
        """
        self._increment_lookup()
        return dict(COUNTRY_CLIMATE_ZONES)

    # =========================================================================
    # EF SOURCE VALIDATION
    # =========================================================================

    def validate_ef_source(self, source: str) -> bool:
        """
        Validate whether an emission factor source is recognized.

        Args:
            source: Emission factor source identifier.

        Returns:
            True if the source is in the recognized list, False otherwise.

        Example:
            >>> engine.validate_ef_source("DEFRA")
            True
            >>> engine.validate_ef_source("UNKNOWN_SOURCE")
            False
        """
        self._increment_lookup()
        return source.strip().upper() in VALID_EF_SOURCES

    def get_valid_ef_sources(self) -> List[str]:
        """
        Get list of all recognized emission factor sources.

        Returns:
            List of valid EF source identifiers.
        """
        return list(VALID_EF_SOURCES)

    # =========================================================================
    # CROSS-TABLE SEARCH
    # =========================================================================

    def search_factors(self, query: str) -> List[Dict[str, Any]]:
        """
        Search across all factor tables for matching entries.

        Performs a case-insensitive substring search across table keys,
        descriptions, and identifiers.

        Args:
            query: Search string (case-insensitive).

        Returns:
            List of matching factor entries with table source metadata.

        Example:
            >>> results = engine.search_factors("diesel")
            >>> len(results) > 0
            True
            >>> results[0]["table"]
            'fuel_emission_factors'
        """
        self._increment_lookup()
        q = query.strip().lower()
        results: List[Dict[str, Any]] = []

        if not q:
            return results

        # Search EUI benchmarks
        for ft in FRANCHISE_TYPES:
            if q in ft:
                for cz, val in EUI_BENCHMARKS[ft].items():
                    results.append({
                        "table": "eui_benchmarks",
                        "key": f"{ft}/{cz}",
                        "value": str(val),
                        "unit": "kWh/m2/yr",
                        "description": f"EUI for {ft} in {cz} zone",
                    })

        # Search revenue intensity
        for ft, val in REVENUE_INTENSITY_FACTORS.items():
            if q in ft:
                results.append({
                    "table": "revenue_intensity",
                    "key": ft,
                    "value": str(val),
                    "unit": "kgCO2e/$",
                    "description": f"Revenue intensity for {ft}",
                })

        # Search fuel emission factors
        for ft, data in FUEL_EMISSION_FACTORS.items():
            if q in ft or q in data.get("description", "").lower():
                results.append({
                    "table": "fuel_emission_factors",
                    "key": ft,
                    "value": str(data["co2e_per_unit"]),
                    "unit": f"kgCO2e/{data['unit']}",
                    "description": data["description"],
                })

        # Search grid emission factors
        for region, val in GRID_EMISSION_FACTORS.items():
            if q in region.lower():
                results.append({
                    "table": "grid_emission_factors",
                    "key": region,
                    "value": str(val),
                    "unit": "kgCO2e/kWh",
                    "description": f"Grid EF for {region}",
                })

        # Search EEIO factors
        for code, data in EEIO_SPEND_FACTORS.items():
            if q in code or q in data.get("description", "").lower():
                results.append({
                    "table": "eeio_spend_factors",
                    "key": code,
                    "value": str(data["factor"]),
                    "unit": "kgCO2e/$",
                    "description": data["description"],
                })

        # Search refrigerant GWPs
        for rt, data in REFRIGERANT_GWPS.items():
            if (
                q in rt.lower()
                or q in data.get("chemical_name", "").lower()
                or q in data.get("common_use", "").lower()
            ):
                results.append({
                    "table": "refrigerant_gwps",
                    "key": rt,
                    "value": str(data["gwp"]),
                    "unit": "GWP (AR5)",
                    "description": f"{data['chemical_name']} - {data['common_use']}",
                })

        # Search vehicle emission factors
        for vt, data in VEHICLE_EMISSION_FACTORS.items():
            if q in vt or q in data.get("description", "").lower():
                results.append({
                    "table": "vehicle_emission_factors",
                    "key": vt,
                    "value": str(data["ef_per_km"]),
                    "unit": "kgCO2e/km",
                    "description": data["description"],
                })

        # Search country climate zones
        for cc, zone in COUNTRY_CLIMATE_ZONES.items():
            if q in cc.lower() or q in zone:
                results.append({
                    "table": "country_climate_zones",
                    "key": cc,
                    "value": zone,
                    "unit": "zone",
                    "description": f"{cc} -> {zone}",
                })

        logger.debug(
            "Factor search: query='%s', results=%d",
            query, len(results),
        )
        return results

    # =========================================================================
    # DIAGNOSTICS AND STATISTICS
    # =========================================================================

    def get_lookup_count(self) -> int:
        """
        Get the total number of factor lookups performed.

        Returns:
            Total lookup count.
        """
        with self._lookup_lock:
            return self._lookup_count

    def get_table_statistics(self) -> Dict[str, int]:
        """
        Get record counts for all 15 reference data tables.

        Returns:
            Dict mapping table name to number of records.
        """
        return {
            "eui_benchmarks": len(FRANCHISE_TYPES) * len(CLIMATE_ZONES),
            "revenue_intensity_factors": len(REVENUE_INTENSITY_FACTORS),
            "cooking_fuel_profiles": len(COOKING_FUEL_PROFILES),
            "refrigeration_leakage_rates": len(REFRIGERATION_LEAKAGE_RATES),
            "grid_emission_factors": len(GRID_EMISSION_FACTORS),
            "fuel_emission_factors": len(FUEL_EMISSION_FACTORS),
            "eeio_spend_factors": len(EEIO_SPEND_FACTORS),
            "refrigerant_gwps": len(REFRIGERANT_GWPS),
            "hotel_energy_benchmarks": (
                len(HOTEL_ENERGY_BENCHMARKS) * len(CLIMATE_ZONES)
            ),
            "vehicle_emission_factors": len(VEHICLE_EMISSION_FACTORS),
            "dc_rules": len(DC_RULES),
            "compliance_framework_rules": len(COMPLIANCE_FRAMEWORK_RULES),
            "dqi_scoring_matrix": len(DQI_SCORING_MATRIX) * 3,
            "uncertainty_ranges": len(UNCERTAINTY_RANGES) * 3,
            "country_climate_zones": len(COUNTRY_CLIMATE_ZONES),
        }

    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get engine metadata and diagnostics.

        Returns:
            Dict with agent info, table counts, lookup statistics.
        """
        return {
            "agent_id": AGENT_ID,
            "agent_component": AGENT_COMPONENT,
            "version": VERSION,
            "table_prefix": TABLE_PREFIX,
            "scope": "Scope 3 Category 14",
            "description": "Franchises emission factor database",
            "franchise_types": FRANCHISE_TYPES,
            "climate_zones": CLIMATE_ZONES,
            "table_statistics": self.get_table_statistics(),
            "total_lookups": self.get_lookup_count(),
            "valid_ef_sources": VALID_EF_SOURCES,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # =========================================================================
    # SINGLETON LIFECYCLE
    # =========================================================================

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance for testing purposes.

        WARNING: This is NOT safe for concurrent use. It should only
        be called in test teardown when no other threads are accessing
        the engine instance.
        """
        with cls._lock:
            if cls._instance is not None:
                if hasattr(cls._instance, "_initialized"):
                    del cls._instance._initialized
                cls._instance = None

                global _database_engine_instance
                _database_engine_instance = None

                logger.info("FranchiseDatabaseEngine singleton reset")


# ============================================================================
# MODULE-LEVEL SINGLETON ACCESSOR
# ============================================================================

_database_engine_instance: Optional[FranchiseDatabaseEngine] = None
_database_engine_lock: threading.Lock = threading.Lock()


def get_database_engine() -> FranchiseDatabaseEngine:
    """
    Get the singleton FranchiseDatabaseEngine instance.

    Thread-safe accessor for the global database engine instance. Prefer
    this function over direct instantiation for consistency across the
    franchise agent codebase.

    Returns:
        FranchiseDatabaseEngine singleton instance.

    Example:
        >>> from greenlang.agents.mrv.franchises.franchise_database import get_database_engine
        >>> engine = get_database_engine()
        >>> eui = engine.get_eui_benchmark("qsr", "temperate")
    """
    global _database_engine_instance

    if _database_engine_instance is None:
        with _database_engine_lock:
            if _database_engine_instance is None:
                _database_engine_instance = FranchiseDatabaseEngine()

    return _database_engine_instance


def reset_database_engine() -> None:
    """
    Reset the singleton database engine instance for testing purposes.

    Convenience function that delegates to FranchiseDatabaseEngine.reset().
    Should only be called in test teardown.
    """
    FranchiseDatabaseEngine.reset()


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Constants
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    # Franchise types and climate zones
    "FRANCHISE_TYPES",
    "CLIMATE_ZONES",
    # Reference data tables (15)
    "EUI_BENCHMARKS",
    "REVENUE_INTENSITY_FACTORS",
    "COOKING_FUEL_PROFILES",
    "REFRIGERATION_LEAKAGE_RATES",
    "GRID_EMISSION_FACTORS",
    "FUEL_EMISSION_FACTORS",
    "EEIO_SPEND_FACTORS",
    "REFRIGERANT_GWPS",
    "HOTEL_ENERGY_BENCHMARKS",
    "VEHICLE_EMISSION_FACTORS",
    "DC_RULES",
    "COMPLIANCE_FRAMEWORK_RULES",
    "DQI_SCORING_MATRIX",
    "UNCERTAINTY_RANGES",
    "COUNTRY_CLIMATE_ZONES",
    "VALID_EF_SOURCES",
    # Engine class
    "FranchiseDatabaseEngine",
    # Singleton accessors
    "get_database_engine",
    "reset_database_engine",
    # Helpers
    "get_metrics_collector",
    "get_provenance_manager",
]
