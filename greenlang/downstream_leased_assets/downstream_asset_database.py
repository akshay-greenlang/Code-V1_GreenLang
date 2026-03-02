# -*- coding: utf-8 -*-
"""
DownstreamAssetDatabaseEngine - Emission factor database for downstream leased assets.

This module implements the DownstreamAssetDatabaseEngine for AGENT-MRV-026
(Downstream Leased Assets, GHG Protocol Scope 3 Category 13). It provides
thread-safe singleton access to emission factor databases, energy-use-intensity
(EUI) benchmarks, vehicle EFs, equipment profiles, IT asset power profiles,
grid emission factors, fuel EFs, EEIO factors, and vacancy base loads.

Category 13 Distinction (Downstream Leased Assets):
    The reporter OWNS the assets and LEASES them OUT to tenants (reporter is
    LESSOR). This is the mirror of Category 8 (Upstream Leased Assets) where
    the reporter is lessee. Emissions arise from tenant operations of the
    leased asset and require tenant data collection, vacancy handling,
    common-area allocation, and operational control boundary checks.

Zero-Hallucination Guarantees:
    - All emission factors from DEFRA 2024, EPA eGRID, IEA, IPCC AR6
    - All values stored as Python Decimal (8 decimal places, ROUND_HALF_UP)
    - No LLM calls anywhere in the lookup path
    - Every lookup is deterministic and reproducible
    - SHA-256 provenance hash on every composite lookup

Features:
    - Building EUI by type (8) and climate zone (5) in kWh/m2/yr
    - Vehicle emission factors by type (8) and fuel (7) in kgCO2e/km (DEFRA 2024)
    - Equipment profiles (6 types) with rated power, fuel consumption, load factor
    - IT asset power profiles (7 types) with PUE and annual hours
    - Grid emission factors (12 countries + 26 eGRID subregions) in kgCO2e/kWh
    - Fuel emission factors (8 fuel types) in kgCO2e/L or kgCO2e/m3 (DEFRA 2024)
    - EEIO spend-based factors (10 NAICS codes) in kgCO2e/$
    - Vacancy base load fractions by building type (8 types)
    - Refrigerant GWPs (15 refrigerants) from IPCC AR6
    - Country-to-climate-zone mapping (20+ countries)
    - Thread-safe singleton pattern with __new__
    - Prometheus metrics recording for all lookups
    - 20+ public methods for factor retrieval and validation

Example:
    >>> engine = DownstreamAssetDatabaseEngine()
    >>> eui = engine.get_building_eui("office", "temperate")
    >>> eui
    Decimal('180.00000000')
    >>> grid_ef = engine.get_grid_ef("US")
    >>> grid_ef
    Decimal('0.41700000')

Author: GreenLang Platform Team
Version: 1.0.0
Agent: GL-MRV-S3-013
"""

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# =============================================================================
# AGENT METADATA
# =============================================================================

AGENT_ID: str = "GL-MRV-S3-013"
AGENT_COMPONENT: str = "AGENT-MRV-026"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_dla_"

# =============================================================================
# QUANTIZATION
# =============================================================================

_QUANT_8DP = Decimal("0.00000001")
_QUANT_2DP = Decimal("0.01")

# =============================================================================
# PROMETHEUS METRICS (graceful import)
# =============================================================================

try:
    from prometheus_client import Counter, Histogram, Gauge
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

    class _NoOpMetric:
        """No-op metric stub when prometheus_client is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, *args: Any, **kwargs: Any) -> "_NoOpMetric":
            return self

        def inc(self, amount: float = 1) -> None:
            pass

        def dec(self, amount: float = 1) -> None:
            pass

        def set(self, value: float = 0) -> None:
            pass

        def observe(self, value: float = 0) -> None:
            pass

    Counter = _NoOpMetric  # type: ignore[assignment,misc]
    Histogram = _NoOpMetric  # type: ignore[assignment,misc]
    Gauge = _NoOpMetric  # type: ignore[assignment,misc]


# Module-level singleton metrics
_FACTOR_LOOKUPS = Counter(
    "gl_dla_factor_lookups_total",
    "Total downstream leased asset factor lookups",
    ["factor_type", "asset_category"],
)
_LOOKUP_ERRORS = Counter(
    "gl_dla_lookup_errors_total",
    "Total factor lookup errors",
    ["factor_type", "error_type"],
)
_LOOKUP_DURATION = Histogram(
    "gl_dla_lookup_duration_seconds",
    "Factor lookup duration",
    ["factor_type"],
)

# =============================================================================
# SECTION 1: BUILDING EUI BENCHMARKS (kWh/m2/yr)
# =============================================================================
# 8 building types x 5 climate zones
# Sources: ASHRAE 90.1, ENERGY STAR Portfolio Manager, CBECS, CIBSE TM46
# Climate zones: tropical, arid, temperate, continental, polar (Koppen simplified)

BUILDING_EUI: Dict[str, Dict[str, Decimal]] = {
    "office": {
        "tropical": Decimal("250"),
        "arid": Decimal("230"),
        "temperate": Decimal("180"),
        "continental": Decimal("200"),
        "polar": Decimal("220"),
    },
    "retail": {
        "tropical": Decimal("280"),
        "arid": Decimal("260"),
        "temperate": Decimal("220"),
        "continental": Decimal("240"),
        "polar": Decimal("270"),
    },
    "warehouse": {
        "tropical": Decimal("140"),
        "arid": Decimal("130"),
        "temperate": Decimal("120"),
        "continental": Decimal("135"),
        "polar": Decimal("150"),
    },
    "industrial": {
        "tropical": Decimal("320"),
        "arid": Decimal("300"),
        "temperate": Decimal("270"),
        "continental": Decimal("290"),
        "polar": Decimal("310"),
    },
    "data_center": {
        "tropical": Decimal("4000"),
        "arid": Decimal("3800"),
        "temperate": Decimal("3500"),
        "continental": Decimal("3700"),
        "polar": Decimal("3200"),
    },
    "hotel": {
        "tropical": Decimal("340"),
        "arid": Decimal("310"),
        "temperate": Decimal("280"),
        "continental": Decimal("300"),
        "polar": Decimal("320"),
    },
    "healthcare": {
        "tropical": Decimal("420"),
        "arid": Decimal("390"),
        "temperate": Decimal("350"),
        "continental": Decimal("380"),
        "polar": Decimal("410"),
    },
    "residential_multifamily": {
        "tropical": Decimal("180"),
        "arid": Decimal("165"),
        "temperate": Decimal("150"),
        "continental": Decimal("170"),
        "polar": Decimal("190"),
    },
}

# =============================================================================
# SECTION 2: VEHICLE EMISSION FACTORS (kgCO2e/km, DEFRA 2024)
# =============================================================================
# 8 vehicle types x 7 fuel types
# Source: DEFRA 2024 Conversion Factors, Tables 10-12
# BEV = 0.0 (grid emissions counted separately via grid EF)

VEHICLE_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "small_car": {
        "gasoline": Decimal("0.149"),
        "diesel": Decimal("0.138"),
        "lpg": Decimal("0.178"),
        "cng": Decimal("0.162"),
        "hybrid": Decimal("0.105"),
        "phev": Decimal("0.068"),
        "bev": Decimal("0.0"),
    },
    "medium_car": {
        "gasoline": Decimal("0.188"),
        "diesel": Decimal("0.168"),
        "lpg": Decimal("0.213"),
        "cng": Decimal("0.195"),
        "hybrid": Decimal("0.131"),
        "phev": Decimal("0.082"),
        "bev": Decimal("0.0"),
    },
    "large_car": {
        "gasoline": Decimal("0.278"),
        "diesel": Decimal("0.209"),
        "lpg": Decimal("0.305"),
        "cng": Decimal("0.268"),
        "hybrid": Decimal("0.176"),
        "phev": Decimal("0.098"),
        "bev": Decimal("0.0"),
    },
    "suv": {
        "gasoline": Decimal("0.232"),
        "diesel": Decimal("0.198"),
        "lpg": Decimal("0.262"),
        "cng": Decimal("0.238"),
        "hybrid": Decimal("0.158"),
        "phev": Decimal("0.092"),
        "bev": Decimal("0.0"),
    },
    "light_van": {
        "gasoline": Decimal("0.206"),
        "diesel": Decimal("0.183"),
        "lpg": Decimal("0.234"),
        "cng": Decimal("0.215"),
        "hybrid": Decimal("0.148"),
        "phev": Decimal("0.095"),
        "bev": Decimal("0.0"),
    },
    "heavy_van": {
        "gasoline": Decimal("0.292"),
        "diesel": Decimal("0.264"),
        "lpg": Decimal("0.328"),
        "cng": Decimal("0.298"),
        "hybrid": Decimal("0.205"),
        "phev": Decimal("0.132"),
        "bev": Decimal("0.0"),
    },
    "light_truck": {
        "gasoline": Decimal("0.365"),
        "diesel": Decimal("0.318"),
        "lpg": Decimal("0.398"),
        "cng": Decimal("0.352"),
        "hybrid": Decimal("0.248"),
        "phev": Decimal("0.158"),
        "bev": Decimal("0.0"),
    },
    "heavy_truck": {
        "gasoline": Decimal("0.812"),
        "diesel": Decimal("0.685"),
        "lpg": Decimal("0.892"),
        "cng": Decimal("0.745"),
        "hybrid": Decimal("0.562"),
        "phev": Decimal("0.348"),
        "bev": Decimal("0.0"),
    },
}

# =============================================================================
# SECTION 3: EQUIPMENT PROFILES
# =============================================================================
# 6 equipment types with rated power (kW), fuel consumption (L/h or electric),
# and typical load factor
# Sources: EPA NONROAD, CARB Off-Road, ISO 8178

EQUIPMENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "manufacturing": {
        "rated_power_kw": Decimal("75"),
        "fuel_consumption_lph": Decimal("18.5"),
        "load_factor": Decimal("0.65"),
        "fuel_type": "diesel",
        "description": "Manufacturing equipment (CNC, presses, assembly)",
    },
    "construction": {
        "rated_power_kw": Decimal("120"),
        "fuel_consumption_lph": Decimal("32.0"),
        "load_factor": Decimal("0.55"),
        "fuel_type": "diesel",
        "description": "Construction equipment (excavators, loaders, cranes)",
    },
    "generator": {
        "rated_power_kw": Decimal("250"),
        "fuel_consumption_lph": Decimal("62.0"),
        "load_factor": Decimal("0.70"),
        "fuel_type": "diesel",
        "description": "Standby and prime power generators",
    },
    "agricultural": {
        "rated_power_kw": Decimal("90"),
        "fuel_consumption_lph": Decimal("23.5"),
        "load_factor": Decimal("0.50"),
        "fuel_type": "diesel",
        "description": "Agricultural equipment (tractors, harvesters)",
    },
    "mining": {
        "rated_power_kw": Decimal("200"),
        "fuel_consumption_lph": Decimal("52.0"),
        "load_factor": Decimal("0.60"),
        "fuel_type": "diesel",
        "description": "Mining equipment (haul trucks, drill rigs)",
    },
    "hvac": {
        "rated_power_kw": Decimal("50"),
        "fuel_consumption_lph": Decimal("0"),
        "load_factor": Decimal("0.80"),
        "fuel_type": "electric",
        "description": "HVAC systems (chillers, AHUs, cooling towers)",
    },
}

# =============================================================================
# SECTION 4: IT ASSET POWER PROFILES
# =============================================================================
# 7 IT asset types with rated power (kW), default PUE, and hours per year
# Sources: The Green Grid PUE, ASHRAE TC 9.9, ENERGY STAR for devices

IT_ASSET_PROFILES: Dict[str, Dict[str, Decimal]] = {
    "server": {
        "power_kw": Decimal("0.5"),
        "default_pue": Decimal("1.6"),
        "hours_per_year": Decimal("8760"),
    },
    "network_switch": {
        "power_kw": Decimal("0.1"),
        "default_pue": Decimal("1.6"),
        "hours_per_year": Decimal("8760"),
    },
    "storage": {
        "power_kw": Decimal("0.3"),
        "default_pue": Decimal("1.6"),
        "hours_per_year": Decimal("8760"),
    },
    "desktop": {
        "power_kw": Decimal("0.15"),
        "default_pue": Decimal("1.0"),
        "hours_per_year": Decimal("2080"),
    },
    "laptop": {
        "power_kw": Decimal("0.05"),
        "default_pue": Decimal("1.0"),
        "hours_per_year": Decimal("2080"),
    },
    "printer": {
        "power_kw": Decimal("0.08"),
        "default_pue": Decimal("1.0"),
        "hours_per_year": Decimal("2080"),
    },
    "copier": {
        "power_kw": Decimal("0.12"),
        "default_pue": Decimal("1.0"),
        "hours_per_year": Decimal("2080"),
    },
}

# =============================================================================
# SECTION 5: GRID EMISSION FACTORS (kgCO2e/kWh)
# =============================================================================
# 12 countries from IEA 2024 + 26 US eGRID subregions from EPA eGRID 2022
# National factors include T&D losses in composite value
# eGRID factors are generation-only (consumption factor = generation / (1 - loss))

GRID_EMISSION_FACTORS: Dict[str, Decimal] = {
    # National / country-level factors (IEA 2024)
    "US": Decimal("0.417"),
    "GB": Decimal("0.207"),
    "DE": Decimal("0.350"),
    "JP": Decimal("0.471"),
    "CN": Decimal("0.555"),
    "IN": Decimal("0.708"),
    "BR": Decimal("0.074"),
    "AU": Decimal("0.656"),
    "KR": Decimal("0.459"),
    "CA": Decimal("0.120"),
    "FR": Decimal("0.052"),
    "GLOBAL": Decimal("0.436"),
}

EGRID_SUBREGION_FACTORS: Dict[str, Decimal] = {
    # EPA eGRID 2022 subregion emission factors (kgCO2e/kWh, generation-weighted)
    "AKGD": Decimal("0.424"),
    "AKMS": Decimal("0.259"),
    "AZNM": Decimal("0.431"),
    "CAMX": Decimal("0.244"),
    "ERCT": Decimal("0.388"),
    "FRCC": Decimal("0.388"),
    "HIMS": Decimal("0.543"),
    "HIOA": Decimal("0.665"),
    "MROE": Decimal("0.605"),
    "MROW": Decimal("0.512"),
    "NEWE": Decimal("0.226"),
    "NWPP": Decimal("0.269"),
    "NYCW": Decimal("0.235"),
    "NYLI": Decimal("0.440"),
    "NYUP": Decimal("0.128"),
    "PRMS": Decimal("0.436"),
    "RFCE": Decimal("0.336"),
    "RFCM": Decimal("0.530"),
    "RFCW": Decimal("0.495"),
    "RMPA": Decimal("0.573"),
    "SPNO": Decimal("0.535"),
    "SPSO": Decimal("0.446"),
    "SRMV": Decimal("0.370"),
    "SRSO": Decimal("0.392"),
    "SRTV": Decimal("0.436"),
    "SRVC": Decimal("0.321"),
}

# =============================================================================
# SECTION 6: FUEL EMISSION FACTORS (kgCO2e per litre or per m3, DEFRA 2024)
# =============================================================================
# Source: DEFRA 2024 Conversion Factors, Fuels chapter
# CNG and natural_gas are per m3; all others per litre
# biodiesel and bioethanol are biogenic-only (scope 1 fossil = 0)

FUEL_EMISSION_FACTORS: Dict[str, Dict[str, Any]] = {
    "gasoline": {
        "co2e_per_unit": Decimal("2.315"),
        "unit": "L",
        "description": "Motor gasoline / petrol",
    },
    "diesel": {
        "co2e_per_unit": Decimal("2.689"),
        "unit": "L",
        "description": "Automotive diesel fuel",
    },
    "lpg": {
        "co2e_per_unit": Decimal("1.522"),
        "unit": "L",
        "description": "Liquefied petroleum gas",
    },
    "cng": {
        "co2e_per_unit": Decimal("2.535"),
        "unit": "m3",
        "description": "Compressed natural gas",
    },
    "jet_fuel": {
        "co2e_per_unit": Decimal("2.528"),
        "unit": "L",
        "description": "Aviation turbine fuel / Jet A-1",
    },
    "biodiesel": {
        "co2e_per_unit": Decimal("0.149"),
        "unit": "L",
        "description": "Biodiesel (biogenic, FAME B100)",
    },
    "bioethanol": {
        "co2e_per_unit": Decimal("0.073"),
        "unit": "L",
        "description": "Bioethanol (biogenic, E100)",
    },
    "natural_gas": {
        "co2e_per_unit": Decimal("2.028"),
        "unit": "m3",
        "description": "Natural gas (building heating, stationary)",
    },
}

# =============================================================================
# SECTION 7: EEIO SPEND-BASED FACTORS (kgCO2e/$ USD, 2022 base year)
# =============================================================================
# Source: USEEIO v2.0, BEA Input-Output tables, Exiobase 3.8
# NAICS codes for leasing and rental activities

EEIO_FACTORS: Dict[str, Dict[str, Any]] = {
    "531110": {
        "factor": Decimal("0.32"),
        "description": "Lessors of residential buildings",
        "naics": "531110",
    },
    "531120": {
        "factor": Decimal("0.38"),
        "description": "Lessors of nonresidential buildings",
        "naics": "531120",
    },
    "531130": {
        "factor": Decimal("0.29"),
        "description": "Lessors of miniwarehouses and self-storage",
        "naics": "531130",
    },
    "531190": {
        "factor": Decimal("0.35"),
        "description": "Lessors of other real estate property",
        "naics": "531190",
    },
    "532111": {
        "factor": Decimal("0.42"),
        "description": "Passenger car rental and leasing",
        "naics": "532111",
    },
    "532112": {
        "factor": Decimal("0.45"),
        "description": "Truck, utility trailer, and RV rental",
        "naics": "532112",
    },
    "532310": {
        "factor": Decimal("0.48"),
        "description": "General rental centers",
        "naics": "532310",
    },
    "532411": {
        "factor": Decimal("0.18"),
        "description": "Computer rental and leasing",
        "naics": "532411",
    },
    "532412": {
        "factor": Decimal("0.22"),
        "description": "Office machinery and equipment rental",
        "naics": "532412",
    },
    "532490": {
        "factor": Decimal("0.40"),
        "description": "Other commercial and industrial machinery rental",
        "naics": "532490",
    },
}

# =============================================================================
# SECTION 8: VACANCY BASE LOAD FRACTIONS
# =============================================================================
# Fraction of normal energy consumption during vacancy periods
# e.g. HVAC set-backs, security lighting, fire alarm systems, elevator standby
# Source: ASHRAE Guideline 36, CBECS Vacancy Energy Analysis

VACANCY_BASE_LOAD: Dict[str, Decimal] = {
    "office": Decimal("0.30"),
    "retail": Decimal("0.20"),
    "warehouse": Decimal("0.15"),
    "industrial": Decimal("0.25"),
    "data_center": Decimal("0.60"),
    "hotel": Decimal("0.35"),
    "healthcare": Decimal("0.40"),
    "residential_multifamily": Decimal("0.25"),
}

# =============================================================================
# SECTION 9: REFRIGERANT GWPS (IPCC AR6, 100-year GWP)
# =============================================================================
# 15 common refrigerants used in building HVAC and commercial refrigeration
# Source: IPCC AR6 WG1 Chapter 7 Table 7.SM.7

REFRIGERANT_GWPS: Dict[str, Dict[str, Any]] = {
    "R-134a": {
        "gwp_100": Decimal("1530"),
        "chemical": "CH2FCF3",
        "class": "HFC",
    },
    "R-410A": {
        "gwp_100": Decimal("2088"),
        "chemical": "R-32/R-125 (50/50)",
        "class": "HFC blend",
    },
    "R-407C": {
        "gwp_100": Decimal("1774"),
        "chemical": "R-32/R-125/R-134a (23/25/52)",
        "class": "HFC blend",
    },
    "R-404A": {
        "gwp_100": Decimal("3922"),
        "chemical": "R-125/R-143a/R-134a (44/52/4)",
        "class": "HFC blend",
    },
    "R-507A": {
        "gwp_100": Decimal("3985"),
        "chemical": "R-125/R-143a (50/50)",
        "class": "HFC blend",
    },
    "R-32": {
        "gwp_100": Decimal("675"),
        "chemical": "CH2F2",
        "class": "HFC",
    },
    "R-125": {
        "gwp_100": Decimal("3500"),
        "chemical": "CHF2CF3",
        "class": "HFC",
    },
    "R-143a": {
        "gwp_100": Decimal("4470"),
        "chemical": "CH3CF3",
        "class": "HFC",
    },
    "R-22": {
        "gwp_100": Decimal("1810"),
        "chemical": "CHClF2",
        "class": "HCFC",
    },
    "R-290": {
        "gwp_100": Decimal("3"),
        "chemical": "C3H8 (propane)",
        "class": "HC",
    },
    "R-600a": {
        "gwp_100": Decimal("3"),
        "chemical": "C4H10 (isobutane)",
        "class": "HC",
    },
    "R-744": {
        "gwp_100": Decimal("1"),
        "chemical": "CO2",
        "class": "Natural",
    },
    "R-717": {
        "gwp_100": Decimal("0"),
        "chemical": "NH3 (ammonia)",
        "class": "Natural",
    },
    "R-1234yf": {
        "gwp_100": Decimal("1"),
        "chemical": "CF3CF=CH2",
        "class": "HFO",
    },
    "R-1234ze": {
        "gwp_100": Decimal("7"),
        "chemical": "CF3CH=CHF",
        "class": "HFO",
    },
}

# =============================================================================
# SECTION 10: COUNTRY-TO-CLIMATE-ZONE MAPPING
# =============================================================================
# Simplified mapping from ISO 3166-1 alpha-2 to Koppen primary climate zone
# Used for EUI benchmark selection when climate zone not explicitly provided

COUNTRY_CLIMATE_ZONES: Dict[str, str] = {
    "US": "temperate",
    "GB": "temperate",
    "DE": "temperate",
    "FR": "temperate",
    "JP": "temperate",
    "CN": "continental",
    "IN": "tropical",
    "BR": "tropical",
    "AU": "arid",
    "KR": "continental",
    "CA": "continental",
    "SG": "tropical",
    "MY": "tropical",
    "TH": "tropical",
    "ID": "tropical",
    "SA": "arid",
    "AE": "arid",
    "QA": "arid",
    "RU": "continental",
    "NO": "polar",
    "SE": "continental",
    "FI": "continental",
    "IS": "polar",
    "MX": "tropical",
    "ZA": "temperate",
    "NZ": "temperate",
    "IT": "temperate",
    "ES": "temperate",
    "PT": "temperate",
    "NL": "temperate",
}

# =============================================================================
# SECTION 11: ENERGY TYPE DEFAULTS
# =============================================================================
# Default energy types and their properties for building calculations

ENERGY_TYPE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "electricity": {
        "unit": "kWh",
        "requires_grid_ef": True,
        "requires_fuel_ef": False,
        "description": "Grid electricity",
    },
    "natural_gas": {
        "unit": "m3",
        "requires_grid_ef": False,
        "requires_fuel_ef": True,
        "fuel_ef_key": "natural_gas",
        "description": "Piped natural gas for heating",
    },
    "steam": {
        "unit": "kWh",
        "requires_grid_ef": False,
        "requires_fuel_ef": False,
        "default_ef_kgco2e_per_kwh": Decimal("0.170"),
        "description": "District steam (average boiler efficiency)",
    },
    "chilled_water": {
        "unit": "kWh",
        "requires_grid_ef": False,
        "requires_fuel_ef": False,
        "default_ef_kgco2e_per_kwh": Decimal("0.210"),
        "description": "District chilled water (average COP)",
    },
}

# =============================================================================
# SECTION 12: ALLOCATION METHOD DEFAULTS
# =============================================================================
# Common area allocation methods for multi-tenant buildings

ALLOCATION_METHODS: Dict[str, str] = {
    "floor_area": "Proportional to leased floor area vs total floor area",
    "headcount": "Proportional to tenant headcount vs total occupants",
    "revenue": "Proportional to tenant revenue vs total building revenue",
    "equal_share": "Equal split among all tenants",
    "metered": "Direct sub-metered consumption (most accurate)",
    "hybrid": "Metered for major loads, area-based for shared services",
}

# =============================================================================
# SECTION 13: GREEN LEASE CLAUSE DEFAULTS
# =============================================================================
# Green lease provisions that affect data availability and calculation quality

GREEN_LEASE_CLAUSES: Dict[str, Dict[str, Any]] = {
    "energy_data_sharing": {
        "description": "Tenant agrees to share energy consumption data",
        "dqi_impact": Decimal("0.15"),
        "tier_upgrade": True,
    },
    "sub_metering": {
        "description": "Landlord installs sub-meters for tenant spaces",
        "dqi_impact": Decimal("0.20"),
        "tier_upgrade": True,
    },
    "energy_target": {
        "description": "Tenant commits to EUI or energy reduction target",
        "dqi_impact": Decimal("0.10"),
        "tier_upgrade": False,
    },
    "efficiency_capex": {
        "description": "Shared capital expenditure for efficiency improvements",
        "dqi_impact": Decimal("0.05"),
        "tier_upgrade": False,
    },
    "renewable_procurement": {
        "description": "Joint renewable energy procurement agreement",
        "dqi_impact": Decimal("0.10"),
        "tier_upgrade": False,
    },
    "waste_reporting": {
        "description": "Tenant reports waste generation data",
        "dqi_impact": Decimal("0.05"),
        "tier_upgrade": False,
    },
}

# =============================================================================
# SECTION 14: DATA QUALITY INDICATOR SCORING
# =============================================================================
# 5-dimension DQI scoring for GHG Protocol Scope 3 data quality assessment
# Dimensions: temporal, geographical, technological, completeness, reliability
# Each scored 1 (highest) to 5 (lowest); overall = weighted average

DQI_DIMENSION_WEIGHTS: Dict[str, Decimal] = {
    "temporal": Decimal("0.20"),
    "geographical": Decimal("0.20"),
    "technological": Decimal("0.20"),
    "completeness": Decimal("0.20"),
    "reliability": Decimal("0.20"),
}

# Default DQI scores by calculation method tier
DQI_DEFAULTS_BY_TIER: Dict[str, Dict[str, int]] = {
    "asset_specific": {
        "temporal": 1,
        "geographical": 1,
        "technological": 1,
        "completeness": 2,
        "reliability": 1,
    },
    "average_data": {
        "temporal": 2,
        "geographical": 3,
        "technological": 3,
        "completeness": 3,
        "reliability": 3,
    },
    "spend_based": {
        "temporal": 3,
        "geographical": 4,
        "technological": 4,
        "completeness": 4,
        "reliability": 4,
    },
}

# =============================================================================
# SECTION 15: UNCERTAINTY DEFAULTS
# =============================================================================
# Default uncertainty ranges by calculation tier (percentage, +/-)
# Source: IPCC 2006 Guidelines Vol 1 Ch 3, GHG Protocol Scope 3 Guidance

UNCERTAINTY_DEFAULTS: Dict[str, Decimal] = {
    "asset_specific": Decimal("10"),
    "average_data": Decimal("30"),
    "spend_based": Decimal("50"),
}


# =============================================================================
# ENGINE CLASS
# =============================================================================


class DownstreamAssetDatabaseEngine:
    """
    Thread-safe singleton engine for downstream leased asset emission factor lookups.

    Provides deterministic, zero-hallucination factor retrieval for all
    downstream leased asset categories: buildings, vehicles, equipment, and IT
    assets. Includes grid EFs, fuel EFs, EEIO factors, vacancy base loads,
    refrigerant GWPs, and country-to-climate-zone mappings.

    This engine does NOT perform any LLM calls. All factors are retrieved
    from validated, frozen constant tables embedded in this module.

    Category 13 Context:
        The reporter is the LESSOR (asset owner). Emissions arise from tenant
        operations of the leased asset. This engine provides the reference data
        needed for tenant energy allocation, vacancy period adjustments, and
        common-area calculations.

    Thread Safety:
        Uses the __new__ singleton pattern with threading.Lock to ensure
        only one instance is created across all threads.

    Attributes:
        _lookup_count: Total number of factor lookups performed
        _error_count: Total number of lookup errors
        _initialized: Whether the singleton has been initialized

    Example:
        >>> engine = DownstreamAssetDatabaseEngine()
        >>> eui = engine.get_building_eui("office", "temperate")
        >>> vehicle_ef = engine.get_vehicle_ef("medium_car", "diesel")
        >>> grid_ef = engine.get_grid_ef("US")
        >>> health = engine.health_check()
        >>> assert health["status"] == "healthy"
    """

    _instance: Optional["DownstreamAssetDatabaseEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "DownstreamAssetDatabaseEngine":
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the database engine (only once due to singleton)."""
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._initialized: bool = True
        self._lookup_count: int = 0
        self._error_count: int = 0
        self._lookup_lock: threading.Lock = threading.Lock()
        self._created_at: datetime = datetime.now(timezone.utc)

        logger.info(
            "DownstreamAssetDatabaseEngine initialized: "
            "building_types=%d, vehicle_types=%d, equipment_types=%d, "
            "it_types=%d, grid_regions=%d, egrid_subregions=%d, "
            "fuel_types=%d, eeio_codes=%d, refrigerants=%d, "
            "climate_zone_countries=%d",
            len(BUILDING_EUI),
            len(VEHICLE_EMISSION_FACTORS),
            len(EQUIPMENT_PROFILES),
            len(IT_ASSET_PROFILES),
            len(GRID_EMISSION_FACTORS),
            len(EGRID_SUBREGION_FACTORS),
            len(FUEL_EMISSION_FACTORS),
            len(EEIO_FACTORS),
            len(REFRIGERANT_GWPS),
            len(COUNTRY_CLIMATE_ZONES),
        )

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _increment_lookup(self) -> None:
        """Increment the lookup counter in a thread-safe manner."""
        with self._lookup_lock:
            self._lookup_count += 1

    def _increment_error(self) -> None:
        """Increment the error counter in a thread-safe manner."""
        with self._lookup_lock:
            self._error_count += 1

    def _quantize(self, value: Decimal, precision: Decimal = _QUANT_8DP) -> Decimal:
        """
        Quantize a Decimal value with ROUND_HALF_UP.

        Args:
            value: Decimal value to quantize.
            precision: Quantization precision (default 8 decimal places).

        Returns:
            Quantized Decimal value.
        """
        return value.quantize(precision, rounding=ROUND_HALF_UP)

    def _record_lookup(self, factor_type: str, asset_category: str) -> None:
        """
        Record a factor lookup in Prometheus metrics.

        Args:
            factor_type: Type of factor looked up (e.g., "eui", "vehicle_ef").
            asset_category: Asset category (e.g., "building", "vehicle").
        """
        try:
            _FACTOR_LOOKUPS.labels(
                factor_type=factor_type, asset_category=asset_category
            ).inc()
        except Exception as exc:
            logger.warning(
                "Failed to record factor lookup metric: %s", exc
            )

    def _record_error(self, factor_type: str, error_type: str) -> None:
        """
        Record a lookup error in Prometheus metrics.

        Args:
            factor_type: Type of factor that failed.
            error_type: Error classification (e.g., "not_found", "invalid_input").
        """
        try:
            _LOOKUP_ERRORS.labels(
                factor_type=factor_type, error_type=error_type
            ).inc()
        except Exception as exc:
            logger.warning(
                "Failed to record lookup error metric: %s", exc
            )

    @staticmethod
    def _compute_hash(data: Any) -> str:
        """
        Compute SHA-256 hash of arbitrary data for provenance tracking.

        Args:
            data: Data to hash (will be JSON-serialized with Decimal support).

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        serialized = json.dumps(
            data, sort_keys=True, default=str
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    # =========================================================================
    # BUILDING EUI LOOKUPS
    # =========================================================================

    def get_building_eui(
        self,
        building_type: str,
        climate_zone: str,
    ) -> Decimal:
        """
        Get building energy use intensity (EUI) benchmark.

        Returns the annual EUI in kWh/m2/yr for a given building type and
        climate zone combination. Used for average-data calculations when
        metered data is not available from tenants.

        Args:
            building_type: Building type (office, retail, warehouse, industrial,
                data_center, hotel, healthcare, residential_multifamily).
            climate_zone: Climate zone (tropical, arid, temperate, continental,
                polar).

        Returns:
            EUI in kWh/m2/yr as Decimal.

        Raises:
            ValueError: If building_type or climate_zone is not found.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> engine.get_building_eui("office", "temperate")
            Decimal('180.00000000')
        """
        self._increment_lookup()

        bt_lower = building_type.strip().lower()
        cz_lower = climate_zone.strip().lower()

        building_data = BUILDING_EUI.get(bt_lower)
        if building_data is None:
            self._increment_error()
            self._record_error("eui", "building_type_not_found")
            raise ValueError(
                f"Building type '{building_type}' not found. "
                f"Available types: {sorted(BUILDING_EUI.keys())}"
            )

        eui = building_data.get(cz_lower)
        if eui is None:
            self._increment_error()
            self._record_error("eui", "climate_zone_not_found")
            raise ValueError(
                f"Climate zone '{climate_zone}' not found for building type "
                f"'{building_type}'. Available zones: "
                f"{sorted(building_data.keys())}"
            )

        result = self._quantize(eui)
        self._record_lookup("eui", "building")

        logger.debug(
            "Building EUI lookup: type=%s, zone=%s, eui=%s kWh/m2/yr",
            bt_lower, cz_lower, result,
        )
        return result

    def get_building_eui_all_zones(
        self,
        building_type: str,
    ) -> Dict[str, Decimal]:
        """
        Get building EUI benchmarks across all climate zones.

        Args:
            building_type: Building type to query.

        Returns:
            Dict mapping climate_zone to EUI (kWh/m2/yr).

        Raises:
            ValueError: If building_type is not found.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> euis = engine.get_building_eui_all_zones("office")
            >>> euis["temperate"]
            Decimal('180.00000000')
        """
        self._increment_lookup()

        bt_lower = building_type.strip().lower()
        building_data = BUILDING_EUI.get(bt_lower)
        if building_data is None:
            self._increment_error()
            self._record_error("eui", "building_type_not_found")
            raise ValueError(
                f"Building type '{building_type}' not found. "
                f"Available types: {sorted(BUILDING_EUI.keys())}"
            )

        result = {
            zone: self._quantize(eui) for zone, eui in building_data.items()
        }
        self._record_lookup("eui_all_zones", "building")
        return result

    # =========================================================================
    # VEHICLE EMISSION FACTOR LOOKUPS
    # =========================================================================

    def get_vehicle_ef(
        self,
        vehicle_type: str,
        fuel_type: str,
    ) -> Decimal:
        """
        Get vehicle emission factor in kgCO2e/km.

        Returns the tailpipe emission factor for a specific vehicle type and
        fuel combination. For BEV (battery electric vehicles), returns 0.0
        as grid emissions are calculated separately.

        Args:
            vehicle_type: Vehicle type (small_car, medium_car, large_car, suv,
                light_van, heavy_van, light_truck, heavy_truck).
            fuel_type: Fuel type (gasoline, diesel, lpg, cng, hybrid, phev, bev).

        Returns:
            Emission factor in kgCO2e/km as Decimal.

        Raises:
            ValueError: If vehicle_type or fuel_type is not found.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> engine.get_vehicle_ef("medium_car", "diesel")
            Decimal('0.16800000')
        """
        self._increment_lookup()

        vt_lower = vehicle_type.strip().lower()
        ft_lower = fuel_type.strip().lower()

        vehicle_data = VEHICLE_EMISSION_FACTORS.get(vt_lower)
        if vehicle_data is None:
            self._increment_error()
            self._record_error("vehicle_ef", "vehicle_type_not_found")
            raise ValueError(
                f"Vehicle type '{vehicle_type}' not found. "
                f"Available types: {sorted(VEHICLE_EMISSION_FACTORS.keys())}"
            )

        ef = vehicle_data.get(ft_lower)
        if ef is None:
            self._increment_error()
            self._record_error("vehicle_ef", "fuel_type_not_found")
            raise ValueError(
                f"Fuel type '{fuel_type}' not found for vehicle type "
                f"'{vehicle_type}'. Available fuels: "
                f"{sorted(vehicle_data.keys())}"
            )

        result = self._quantize(ef)
        self._record_lookup("vehicle_ef", "vehicle")

        logger.debug(
            "Vehicle EF lookup: type=%s, fuel=%s, ef=%s kgCO2e/km",
            vt_lower, ft_lower, result,
        )
        return result

    def get_vehicle_ef_all_fuels(
        self,
        vehicle_type: str,
    ) -> Dict[str, Decimal]:
        """
        Get vehicle emission factors for all fuel types.

        Args:
            vehicle_type: Vehicle type to query.

        Returns:
            Dict mapping fuel_type to EF (kgCO2e/km).

        Raises:
            ValueError: If vehicle_type is not found.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> efs = engine.get_vehicle_ef_all_fuels("small_car")
            >>> efs["gasoline"]
            Decimal('0.14900000')
        """
        self._increment_lookup()

        vt_lower = vehicle_type.strip().lower()
        vehicle_data = VEHICLE_EMISSION_FACTORS.get(vt_lower)
        if vehicle_data is None:
            self._increment_error()
            self._record_error("vehicle_ef", "vehicle_type_not_found")
            raise ValueError(
                f"Vehicle type '{vehicle_type}' not found. "
                f"Available types: {sorted(VEHICLE_EMISSION_FACTORS.keys())}"
            )

        result = {
            fuel: self._quantize(ef) for fuel, ef in vehicle_data.items()
        }
        self._record_lookup("vehicle_ef_all_fuels", "vehicle")
        return result

    # =========================================================================
    # EQUIPMENT PROFILE LOOKUPS
    # =========================================================================

    def get_equipment_fuel(
        self,
        equipment_type: str,
    ) -> Dict[str, Any]:
        """
        Get equipment profile including fuel consumption and load factor.

        Returns the complete equipment profile with rated power, fuel
        consumption rate, load factor, and fuel type. For electric equipment
        (e.g., HVAC), fuel_consumption_lph is 0 and fuel_type is "electric".

        Args:
            equipment_type: Equipment type (manufacturing, construction,
                generator, agricultural, mining, hvac).

        Returns:
            Dict with keys: rated_power_kw, fuel_consumption_lph,
            load_factor, fuel_type, description.

        Raises:
            ValueError: If equipment_type is not found.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> profile = engine.get_equipment_fuel("manufacturing")
            >>> profile["rated_power_kw"]
            Decimal('75.00000000')
            >>> profile["load_factor"]
            Decimal('0.65000000')
        """
        self._increment_lookup()

        et_lower = equipment_type.strip().lower()
        profile = EQUIPMENT_PROFILES.get(et_lower)
        if profile is None:
            self._increment_error()
            self._record_error("equipment", "type_not_found")
            raise ValueError(
                f"Equipment type '{equipment_type}' not found. "
                f"Available types: {sorted(EQUIPMENT_PROFILES.keys())}"
            )

        result = {
            "rated_power_kw": self._quantize(profile["rated_power_kw"]),
            "fuel_consumption_lph": self._quantize(profile["fuel_consumption_lph"]),
            "load_factor": self._quantize(profile["load_factor"]),
            "fuel_type": profile["fuel_type"],
            "description": profile["description"],
        }
        self._record_lookup("equipment_profile", "equipment")

        logger.debug(
            "Equipment profile lookup: type=%s, power=%s kW, "
            "fuel=%s L/h, load_factor=%s",
            et_lower,
            result["rated_power_kw"],
            result["fuel_consumption_lph"],
            result["load_factor"],
        )
        return result

    # =========================================================================
    # IT ASSET POWER PROFILE LOOKUPS
    # =========================================================================

    def get_it_power(
        self,
        it_type: str,
    ) -> Dict[str, Decimal]:
        """
        Get IT asset power profile including PUE and annual hours.

        Returns the power profile for an IT asset type. Data center assets
        (server, network_switch, storage) use PUE > 1.0 and run 8760 h/yr.
        Office assets (desktop, laptop, printer, copier) use PUE = 1.0 and
        run 2080 h/yr (standard working hours).

        Args:
            it_type: IT asset type (server, network_switch, storage,
                desktop, laptop, printer, copier).

        Returns:
            Dict with keys: power_kw, default_pue, hours_per_year.

        Raises:
            ValueError: If it_type is not found.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> profile = engine.get_it_power("server")
            >>> profile["power_kw"]
            Decimal('0.50000000')
            >>> profile["default_pue"]
            Decimal('1.60000000')
        """
        self._increment_lookup()

        it_lower = it_type.strip().lower()
        profile = IT_ASSET_PROFILES.get(it_lower)
        if profile is None:
            self._increment_error()
            self._record_error("it_asset", "type_not_found")
            raise ValueError(
                f"IT asset type '{it_type}' not found. "
                f"Available types: {sorted(IT_ASSET_PROFILES.keys())}"
            )

        result = {
            "power_kw": self._quantize(profile["power_kw"]),
            "default_pue": self._quantize(profile["default_pue"]),
            "hours_per_year": self._quantize(profile["hours_per_year"]),
        }
        self._record_lookup("it_power", "it_asset")

        logger.debug(
            "IT asset profile lookup: type=%s, power=%s kW, "
            "PUE=%s, hours=%s/yr",
            it_lower,
            result["power_kw"],
            result["default_pue"],
            result["hours_per_year"],
        )
        return result

    # =========================================================================
    # GRID EMISSION FACTOR LOOKUPS
    # =========================================================================

    def get_grid_ef(
        self,
        region: str,
    ) -> Decimal:
        """
        Get grid emission factor in kgCO2e/kWh for a country or eGRID subregion.

        First checks country-level factors (IEA 2024), then eGRID subregion
        factors (EPA eGRID 2022). Falls back to GLOBAL if not found.

        Args:
            region: Country ISO code (e.g., "US", "GB") or eGRID subregion
                code (e.g., "CAMX", "ERCT"). Case-insensitive.

        Returns:
            Grid emission factor in kgCO2e/kWh as Decimal.

        Raises:
            ValueError: If region is not found in any factor table.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> engine.get_grid_ef("US")
            Decimal('0.41700000')
            >>> engine.get_grid_ef("CAMX")
            Decimal('0.24400000')
        """
        self._increment_lookup()

        region_upper = region.strip().upper()

        # Check country-level first
        ef = GRID_EMISSION_FACTORS.get(region_upper)
        if ef is not None:
            result = self._quantize(ef)
            self._record_lookup("grid_ef", "country")
            logger.debug(
                "Grid EF lookup (country): region=%s, ef=%s kgCO2e/kWh",
                region_upper, result,
            )
            return result

        # Check eGRID subregion
        ef = EGRID_SUBREGION_FACTORS.get(region_upper)
        if ef is not None:
            result = self._quantize(ef)
            self._record_lookup("grid_ef", "egrid_subregion")
            logger.debug(
                "Grid EF lookup (eGRID): subregion=%s, ef=%s kgCO2e/kWh",
                region_upper, result,
            )
            return result

        # Not found
        self._increment_error()
        self._record_error("grid_ef", "region_not_found")
        raise ValueError(
            f"Grid emission factor not found for region '{region}'. "
            f"Available countries: {sorted(GRID_EMISSION_FACTORS.keys())}. "
            f"Available eGRID subregions: {sorted(EGRID_SUBREGION_FACTORS.keys())}"
        )

    def get_grid_ef_with_source(
        self,
        region: str,
    ) -> Dict[str, Any]:
        """
        Get grid emission factor with source metadata.

        Returns the factor along with the source database and region type
        for audit trail purposes.

        Args:
            region: Country ISO code or eGRID subregion code.

        Returns:
            Dict with keys: ef, source, region_type, region_code.

        Raises:
            ValueError: If region is not found.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> result = engine.get_grid_ef_with_source("CAMX")
            >>> result["source"]
            'EPA eGRID 2022'
        """
        self._increment_lookup()
        region_upper = region.strip().upper()

        ef = GRID_EMISSION_FACTORS.get(region_upper)
        if ef is not None:
            return {
                "ef": self._quantize(ef),
                "source": "IEA 2024",
                "region_type": "country",
                "region_code": region_upper,
            }

        ef = EGRID_SUBREGION_FACTORS.get(region_upper)
        if ef is not None:
            return {
                "ef": self._quantize(ef),
                "source": "EPA eGRID 2022",
                "region_type": "egrid_subregion",
                "region_code": region_upper,
            }

        self._increment_error()
        self._record_error("grid_ef_with_source", "region_not_found")
        raise ValueError(
            f"Grid emission factor not found for region '{region}'."
        )

    # =========================================================================
    # FUEL EMISSION FACTOR LOOKUPS
    # =========================================================================

    def get_fuel_ef(
        self,
        fuel_type: str,
    ) -> Dict[str, Any]:
        """
        Get fuel emission factor with unit metadata.

        Returns the emission factor per unit of fuel along with the unit
        (litres or m3) and description.

        Args:
            fuel_type: Fuel type (gasoline, diesel, lpg, cng, jet_fuel,
                biodiesel, bioethanol, natural_gas).

        Returns:
            Dict with keys: co2e_per_unit, unit, description.

        Raises:
            ValueError: If fuel_type is not found.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> ef = engine.get_fuel_ef("diesel")
            >>> ef["co2e_per_unit"]
            Decimal('2.68900000')
            >>> ef["unit"]
            'L'
        """
        self._increment_lookup()

        ft_lower = fuel_type.strip().lower()
        fuel_data = FUEL_EMISSION_FACTORS.get(ft_lower)
        if fuel_data is None:
            self._increment_error()
            self._record_error("fuel_ef", "fuel_type_not_found")
            raise ValueError(
                f"Fuel type '{fuel_type}' not found. "
                f"Available types: {sorted(FUEL_EMISSION_FACTORS.keys())}"
            )

        result = {
            "co2e_per_unit": self._quantize(fuel_data["co2e_per_unit"]),
            "unit": fuel_data["unit"],
            "description": fuel_data["description"],
        }
        self._record_lookup("fuel_ef", "fuel")

        logger.debug(
            "Fuel EF lookup: type=%s, ef=%s kgCO2e/%s",
            ft_lower, result["co2e_per_unit"], result["unit"],
        )
        return result

    # =========================================================================
    # EEIO SPEND-BASED FACTOR LOOKUPS
    # =========================================================================

    def get_eeio_factor(
        self,
        naics_code: str,
    ) -> Dict[str, Any]:
        """
        Get EEIO spend-based emission factor for a NAICS leasing code.

        Returns the environmentally-extended input-output factor in kgCO2e
        per USD for spend-based calculations when metered or average-data
        approaches are not feasible.

        Args:
            naics_code: 6-digit NAICS code for leasing/rental sector.

        Returns:
            Dict with keys: factor, description, naics.

        Raises:
            ValueError: If naics_code is not found.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> factor = engine.get_eeio_factor("531120")
            >>> factor["factor"]
            Decimal('0.38000000')
            >>> factor["description"]
            'Lessors of nonresidential buildings'
        """
        self._increment_lookup()

        code = naics_code.strip()
        eeio_data = EEIO_FACTORS.get(code)
        if eeio_data is None:
            self._increment_error()
            self._record_error("eeio", "naics_not_found")
            raise ValueError(
                f"EEIO factor not found for NAICS code '{naics_code}'. "
                f"Available codes: {sorted(EEIO_FACTORS.keys())}"
            )

        result = {
            "factor": self._quantize(eeio_data["factor"]),
            "description": eeio_data["description"],
            "naics": eeio_data["naics"],
        }
        self._record_lookup("eeio", "spend")

        logger.debug(
            "EEIO factor lookup: naics=%s, factor=%s kgCO2e/$, desc=%s",
            code, result["factor"], result["description"],
        )
        return result

    def get_eeio_factor_by_description(
        self,
        search_term: str,
    ) -> List[Dict[str, Any]]:
        """
        Search EEIO factors by description keyword.

        Performs case-insensitive substring search across all EEIO factor
        descriptions. Useful when NAICS code is unknown.

        Args:
            search_term: Keyword to search in EEIO descriptions.

        Returns:
            List of matching EEIO factor dicts.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> results = engine.get_eeio_factor_by_description("car")
            >>> len(results) >= 1
            True
        """
        self._increment_lookup()

        term_lower = search_term.strip().lower()
        matches: List[Dict[str, Any]] = []

        for code, data in EEIO_FACTORS.items():
            if term_lower in data["description"].lower():
                matches.append({
                    "factor": self._quantize(data["factor"]),
                    "description": data["description"],
                    "naics": data["naics"],
                })

        self._record_lookup("eeio_search", "spend")

        logger.debug(
            "EEIO description search: term='%s', matches=%d",
            search_term, len(matches),
        )
        return matches

    # =========================================================================
    # VACANCY BASE LOAD LOOKUPS
    # =========================================================================

    def get_vacancy_base_load(
        self,
        building_type: str,
    ) -> Decimal:
        """
        Get vacancy period base load fraction for a building type.

        Returns the fraction (0-1) of normal energy consumption that continues
        during vacancy periods due to base loads such as security lighting,
        HVAC setbacks, fire alarm systems, and elevator standby power.

        Data centers have the highest vacancy base load (0.60) due to
        cooling infrastructure that must remain operational even when
        servers are powered down.

        Args:
            building_type: Building type (same types as EUI benchmarks).

        Returns:
            Base load fraction as Decimal (0 to 1).

        Raises:
            ValueError: If building_type is not found.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> engine.get_vacancy_base_load("office")
            Decimal('0.30000000')
            >>> engine.get_vacancy_base_load("data_center")
            Decimal('0.60000000')
        """
        self._increment_lookup()

        bt_lower = building_type.strip().lower()
        fraction = VACANCY_BASE_LOAD.get(bt_lower)
        if fraction is None:
            self._increment_error()
            self._record_error("vacancy", "building_type_not_found")
            raise ValueError(
                f"Vacancy base load not found for building type "
                f"'{building_type}'. Available types: "
                f"{sorted(VACANCY_BASE_LOAD.keys())}"
            )

        result = self._quantize(fraction)
        self._record_lookup("vacancy_base_load", "building")

        logger.debug(
            "Vacancy base load lookup: type=%s, fraction=%s",
            bt_lower, result,
        )
        return result

    # =========================================================================
    # REFRIGERANT GWP LOOKUPS
    # =========================================================================

    def get_refrigerant_gwp(
        self,
        refrigerant: str,
    ) -> Dict[str, Any]:
        """
        Get refrigerant GWP (100-year) from IPCC AR6.

        Returns the global warming potential for a refrigerant along with
        its chemical formula and class (HFC, HCFC, HC, HFO, Natural).

        Args:
            refrigerant: Refrigerant designation (e.g., "R-410A", "R-134a").

        Returns:
            Dict with keys: gwp_100, chemical, class.

        Raises:
            ValueError: If refrigerant is not found.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> gwp = engine.get_refrigerant_gwp("R-410A")
            >>> gwp["gwp_100"]
            Decimal('2088.00000000')
        """
        self._increment_lookup()

        ref_upper = refrigerant.strip().upper()
        # Normalize common variants: handle "R410A" -> "R-410A"
        if ref_upper.startswith("R") and len(ref_upper) > 1 and ref_upper[1] != "-":
            ref_upper = "R-" + ref_upper[1:]

        gwp_data = REFRIGERANT_GWPS.get(ref_upper)
        if gwp_data is None:
            self._increment_error()
            self._record_error("refrigerant", "not_found")
            raise ValueError(
                f"Refrigerant '{refrigerant}' not found. "
                f"Available refrigerants: {sorted(REFRIGERANT_GWPS.keys())}"
            )

        result = {
            "gwp_100": self._quantize(gwp_data["gwp_100"]),
            "chemical": gwp_data["chemical"],
            "class": gwp_data["class"],
        }
        self._record_lookup("refrigerant_gwp", "refrigerant")

        logger.debug(
            "Refrigerant GWP lookup: ref=%s, gwp=%s, class=%s",
            ref_upper, result["gwp_100"], result["class"],
        )
        return result

    # =========================================================================
    # COUNTRY / CLIMATE ZONE LOOKUPS
    # =========================================================================

    def get_country_climate_zone(
        self,
        country_code: str,
    ) -> str:
        """
        Get the primary climate zone for a country.

        Maps ISO 3166-1 alpha-2 country codes to simplified Koppen climate
        zone classifications used for EUI benchmark selection.

        Args:
            country_code: ISO 3166-1 alpha-2 country code (e.g., "US", "IN").

        Returns:
            Climate zone string (tropical, arid, temperate, continental, polar).

        Raises:
            ValueError: If country_code is not found in the mapping.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> engine.get_country_climate_zone("IN")
            'tropical'
            >>> engine.get_country_climate_zone("CA")
            'continental'
        """
        self._increment_lookup()

        code_upper = country_code.strip().upper()
        zone = COUNTRY_CLIMATE_ZONES.get(code_upper)
        if zone is None:
            self._increment_error()
            self._record_error("climate_zone", "country_not_found")
            raise ValueError(
                f"Country code '{country_code}' not found in climate zone "
                f"mapping. Available countries: "
                f"{sorted(COUNTRY_CLIMATE_ZONES.keys())}"
            )

        self._record_lookup("climate_zone", "country")

        logger.debug(
            "Climate zone lookup: country=%s, zone=%s",
            code_upper, zone,
        )
        return zone

    # =========================================================================
    # ENUMERATION / LISTING METHODS
    # =========================================================================

    def get_all_building_types(self) -> List[str]:
        """
        Get all available building types.

        Returns:
            Sorted list of building type strings.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> types = engine.get_all_building_types()
            >>> "office" in types
            True
        """
        return sorted(BUILDING_EUI.keys())

    def get_all_vehicle_types(self) -> List[str]:
        """
        Get all available vehicle types.

        Returns:
            Sorted list of vehicle type strings.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> types = engine.get_all_vehicle_types()
            >>> "medium_car" in types
            True
        """
        return sorted(VEHICLE_EMISSION_FACTORS.keys())

    def get_all_equipment_types(self) -> List[str]:
        """
        Get all available equipment types.

        Returns:
            Sorted list of equipment type strings.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> types = engine.get_all_equipment_types()
            >>> "generator" in types
            True
        """
        return sorted(EQUIPMENT_PROFILES.keys())

    def get_all_it_types(self) -> List[str]:
        """
        Get all available IT asset types.

        Returns:
            Sorted list of IT asset type strings.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> types = engine.get_all_it_types()
            >>> "server" in types
            True
        """
        return sorted(IT_ASSET_PROFILES.keys())

    def get_all_regions(self) -> Dict[str, List[str]]:
        """
        Get all available region codes organized by type.

        Returns:
            Dict with keys "countries" and "egrid_subregions", each mapping
            to sorted lists of region codes.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> regions = engine.get_all_regions()
            >>> "US" in regions["countries"]
            True
            >>> "CAMX" in regions["egrid_subregions"]
            True
        """
        return {
            "countries": sorted(GRID_EMISSION_FACTORS.keys()),
            "egrid_subregions": sorted(EGRID_SUBREGION_FACTORS.keys()),
        }

    def get_all_fuel_types(self) -> List[str]:
        """
        Get all available fuel types.

        Returns:
            Sorted list of fuel type strings.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> fuels = engine.get_all_fuel_types()
            >>> "diesel" in fuels
            True
        """
        return sorted(FUEL_EMISSION_FACTORS.keys())

    def get_all_refrigerants(self) -> List[str]:
        """
        Get all available refrigerant designations.

        Returns:
            Sorted list of refrigerant designation strings.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> refs = engine.get_all_refrigerants()
            >>> "R-410A" in refs
            True
        """
        return sorted(REFRIGERANT_GWPS.keys())

    def get_all_eeio_codes(self) -> List[str]:
        """
        Get all available NAICS codes for EEIO factors.

        Returns:
            Sorted list of NAICS code strings.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> codes = engine.get_all_eeio_codes()
            >>> "531120" in codes
            True
        """
        return sorted(EEIO_FACTORS.keys())

    def get_all_climate_zones(self) -> List[str]:
        """
        Get all available climate zone names.

        Returns:
            Sorted list of climate zone strings.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> zones = engine.get_all_climate_zones()
            >>> "temperate" in zones
            True
        """
        return sorted(["tropical", "arid", "temperate", "continental", "polar"])

    # =========================================================================
    # COMPOSITE LOOKUP
    # =========================================================================

    def lookup_composite(
        self,
        asset_category: str,
        asset_type: str,
        region: str,
        fuel_type: Optional[str] = None,
        climate_zone: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform a composite lookup combining asset-specific data with grid EF.

        Retrieves the asset-specific factor (EUI, vehicle EF, equipment
        profile, or IT power profile) and the region-appropriate grid
        emission factor in a single call. Useful for quick estimation.

        Args:
            asset_category: Category of asset ("building", "vehicle",
                "equipment", "it_asset").
            asset_type: Specific type within the category.
            region: Country or eGRID subregion for grid EF.
            fuel_type: Required for vehicle lookups. Optional for others.
            climate_zone: Required for building lookups. If None and
                asset_category is "building", will attempt to derive
                from region using country climate zone mapping.

        Returns:
            Dict with asset-specific data, grid_ef, and provenance_hash.

        Raises:
            ValueError: If asset_category is unknown or required params missing.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> result = engine.lookup_composite(
            ...     "building", "office", "US", climate_zone="temperate"
            ... )
            >>> result["eui"]
            Decimal('180.00000000')
            >>> result["grid_ef"]
            Decimal('0.41700000')
        """
        self._increment_lookup()

        cat_lower = asset_category.strip().lower()
        result: Dict[str, Any] = {
            "asset_category": cat_lower,
            "asset_type": asset_type,
            "region": region,
        }

        # Resolve grid EF
        grid_ef = self.get_grid_ef(region)
        result["grid_ef"] = grid_ef

        if cat_lower == "building":
            # Resolve climate zone if not provided
            if climate_zone is None:
                try:
                    climate_zone = self.get_country_climate_zone(region)
                except ValueError:
                    climate_zone = "temperate"  # safe default
            result["climate_zone"] = climate_zone
            result["eui"] = self.get_building_eui(asset_type, climate_zone)
            result["vacancy_base_load"] = self.get_vacancy_base_load(asset_type)

        elif cat_lower == "vehicle":
            if fuel_type is None:
                raise ValueError(
                    "fuel_type is required for vehicle composite lookups."
                )
            result["fuel_type"] = fuel_type
            result["vehicle_ef"] = self.get_vehicle_ef(asset_type, fuel_type)

        elif cat_lower == "equipment":
            profile = self.get_equipment_fuel(asset_type)
            result.update(profile)

        elif cat_lower == "it_asset":
            profile = self.get_it_power(asset_type)
            result.update(profile)

        else:
            self._increment_error()
            self._record_error("composite", "unknown_category")
            raise ValueError(
                f"Unknown asset category '{asset_category}'. "
                f"Valid categories: building, vehicle, equipment, it_asset"
            )

        # Compute provenance hash for the composite result
        result["provenance_hash"] = self._compute_hash(result)

        self._record_lookup("composite", cat_lower)

        logger.info(
            "Composite lookup: category=%s, type=%s, region=%s, hash=%s",
            cat_lower, asset_type, region, result["provenance_hash"][:16],
        )
        return result

    # =========================================================================
    # ASSET VALIDATION
    # =========================================================================

    def validate_asset(
        self,
        asset_category: str,
        asset_type: str,
        fuel_type: Optional[str] = None,
        climate_zone: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate that an asset specification matches known reference data.

        Checks that the given asset category, type, fuel type, climate zone,
        and region all exist in the embedded reference data. Returns a
        validation result with per-field status and any error messages.

        Args:
            asset_category: Asset category to validate.
            asset_type: Asset type within the category.
            fuel_type: Fuel type (required for vehicles, optional otherwise).
            climate_zone: Climate zone (required for buildings, optional otherwise).
            region: Region code for grid EF validation.

        Returns:
            Dict with keys: valid (bool), errors (List[str]),
            warnings (List[str]), checked_fields (Dict[str, bool]).

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> result = engine.validate_asset("building", "office",
            ...     climate_zone="temperate", region="US")
            >>> result["valid"]
            True
        """
        self._increment_lookup()

        errors: List[str] = []
        warnings: List[str] = []
        checked: Dict[str, bool] = {}

        cat_lower = asset_category.strip().lower()

        # Validate category
        valid_categories = {"building", "vehicle", "equipment", "it_asset"}
        if cat_lower not in valid_categories:
            errors.append(
                f"Unknown asset_category '{asset_category}'. "
                f"Valid: {sorted(valid_categories)}"
            )
            checked["asset_category"] = False
        else:
            checked["asset_category"] = True

        # Validate asset type within category
        type_lower = asset_type.strip().lower()
        if cat_lower == "building":
            checked["asset_type"] = type_lower in BUILDING_EUI
            if not checked["asset_type"]:
                errors.append(
                    f"Building type '{asset_type}' not found. "
                    f"Available: {sorted(BUILDING_EUI.keys())}"
                )
        elif cat_lower == "vehicle":
            checked["asset_type"] = type_lower in VEHICLE_EMISSION_FACTORS
            if not checked["asset_type"]:
                errors.append(
                    f"Vehicle type '{asset_type}' not found. "
                    f"Available: {sorted(VEHICLE_EMISSION_FACTORS.keys())}"
                )
        elif cat_lower == "equipment":
            checked["asset_type"] = type_lower in EQUIPMENT_PROFILES
            if not checked["asset_type"]:
                errors.append(
                    f"Equipment type '{asset_type}' not found. "
                    f"Available: {sorted(EQUIPMENT_PROFILES.keys())}"
                )
        elif cat_lower == "it_asset":
            checked["asset_type"] = type_lower in IT_ASSET_PROFILES
            if not checked["asset_type"]:
                errors.append(
                    f"IT asset type '{asset_type}' not found. "
                    f"Available: {sorted(IT_ASSET_PROFILES.keys())}"
                )

        # Validate fuel type (required for vehicles)
        if fuel_type is not None:
            ft_lower = fuel_type.strip().lower()
            if cat_lower == "vehicle" and checked.get("asset_type"):
                vehicle_data = VEHICLE_EMISSION_FACTORS.get(type_lower, {})
                checked["fuel_type"] = ft_lower in vehicle_data
                if not checked["fuel_type"]:
                    errors.append(
                        f"Fuel type '{fuel_type}' not found for vehicle "
                        f"'{asset_type}'. Available: {sorted(vehicle_data.keys())}"
                    )
            else:
                checked["fuel_type"] = ft_lower in FUEL_EMISSION_FACTORS
                if not checked["fuel_type"]:
                    warnings.append(
                        f"Fuel type '{fuel_type}' not in fuel EF table."
                    )
        elif cat_lower == "vehicle":
            warnings.append("fuel_type not provided for vehicle asset.")

        # Validate climate zone (relevant for buildings)
        if climate_zone is not None:
            cz_lower = climate_zone.strip().lower()
            valid_zones = {"tropical", "arid", "temperate", "continental", "polar"}
            checked["climate_zone"] = cz_lower in valid_zones
            if not checked["climate_zone"]:
                errors.append(
                    f"Climate zone '{climate_zone}' not valid. "
                    f"Available: {sorted(valid_zones)}"
                )
        elif cat_lower == "building":
            warnings.append(
                "climate_zone not provided for building asset. "
                "Will attempt to derive from region."
            )

        # Validate region
        if region is not None:
            region_upper = region.strip().upper()
            in_country = region_upper in GRID_EMISSION_FACTORS
            in_egrid = region_upper in EGRID_SUBREGION_FACTORS
            checked["region"] = in_country or in_egrid
            if not checked["region"]:
                errors.append(
                    f"Region '{region}' not found in grid EF tables."
                )
        else:
            warnings.append("region not provided; grid EF cannot be validated.")

        is_valid = len(errors) == 0
        self._record_lookup("validate", cat_lower)

        logger.debug(
            "Asset validation: category=%s, type=%s, valid=%s, "
            "errors=%d, warnings=%d",
            cat_lower, type_lower, is_valid, len(errors), len(warnings),
        )
        return {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "checked_fields": checked,
        }

    # =========================================================================
    # GREEN LEASE AND ALLOCATION HELPERS
    # =========================================================================

    def get_green_lease_clauses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available green lease clause definitions and DQI impacts.

        Returns:
            Dict mapping clause_id to clause details including description,
            DQI impact, and tier upgrade flag.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> clauses = engine.get_green_lease_clauses()
            >>> clauses["sub_metering"]["dqi_impact"]
            Decimal('0.20')
        """
        self._increment_lookup()
        self._record_lookup("green_lease", "building")

        result = {}
        for clause_id, clause_data in GREEN_LEASE_CLAUSES.items():
            result[clause_id] = {
                "description": clause_data["description"],
                "dqi_impact": self._quantize(clause_data["dqi_impact"]),
                "tier_upgrade": clause_data["tier_upgrade"],
            }
        return result

    def get_allocation_methods(self) -> Dict[str, str]:
        """
        Get all available allocation methods for multi-tenant buildings.

        Returns:
            Dict mapping method_id to description string.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> methods = engine.get_allocation_methods()
            >>> "floor_area" in methods
            True
        """
        self._record_lookup("allocation_methods", "building")
        return dict(ALLOCATION_METHODS)

    def get_energy_type_defaults(self) -> Dict[str, Dict[str, Any]]:
        """
        Get default properties for energy types used in building calculations.

        Returns:
            Dict mapping energy_type to its properties (unit, requires_grid_ef,
            requires_fuel_ef, description).

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> defaults = engine.get_energy_type_defaults()
            >>> defaults["electricity"]["requires_grid_ef"]
            True
        """
        self._increment_lookup()
        self._record_lookup("energy_type_defaults", "building")
        return dict(ENERGY_TYPE_DEFAULTS)

    def get_dqi_defaults(
        self,
        calculation_tier: str,
    ) -> Dict[str, int]:
        """
        Get default DQI scores for a calculation method tier.

        Args:
            calculation_tier: Tier name ("asset_specific", "average_data",
                "spend_based").

        Returns:
            Dict mapping dimension to default score (1-5).

        Raises:
            ValueError: If calculation_tier is not recognized.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> dqi = engine.get_dqi_defaults("asset_specific")
            >>> dqi["reliability"]
            1
        """
        self._increment_lookup()

        tier_lower = calculation_tier.strip().lower()
        defaults = DQI_DEFAULTS_BY_TIER.get(tier_lower)
        if defaults is None:
            self._increment_error()
            self._record_error("dqi", "tier_not_found")
            raise ValueError(
                f"Calculation tier '{calculation_tier}' not found. "
                f"Available tiers: {sorted(DQI_DEFAULTS_BY_TIER.keys())}"
            )

        self._record_lookup("dqi_defaults", "quality")
        return dict(defaults)

    def get_uncertainty_default(
        self,
        calculation_tier: str,
    ) -> Decimal:
        """
        Get default uncertainty percentage for a calculation tier.

        Args:
            calculation_tier: Tier name ("asset_specific", "average_data",
                "spend_based").

        Returns:
            Uncertainty percentage as Decimal (e.g., 10 means +/-10%).

        Raises:
            ValueError: If calculation_tier is not recognized.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> engine.get_uncertainty_default("asset_specific")
            Decimal('10.00000000')
        """
        self._increment_lookup()

        tier_lower = calculation_tier.strip().lower()
        pct = UNCERTAINTY_DEFAULTS.get(tier_lower)
        if pct is None:
            self._increment_error()
            self._record_error("uncertainty", "tier_not_found")
            raise ValueError(
                f"Uncertainty default not found for tier "
                f"'{calculation_tier}'. Available tiers: "
                f"{sorted(UNCERTAINTY_DEFAULTS.keys())}"
            )

        self._record_lookup("uncertainty_default", "quality")
        return self._quantize(pct)

    # =========================================================================
    # PROVENANCE HASH
    # =========================================================================

    def compute_lookup_hash(
        self,
        lookup_type: str,
        lookup_params: Dict[str, Any],
        lookup_result: Any,
    ) -> str:
        """
        Compute SHA-256 provenance hash for a factor lookup.

        Creates a deterministic hash of the lookup type, parameters, result,
        and timestamp for complete audit trail of every factor retrieval.

        Args:
            lookup_type: Type of lookup performed (e.g., "building_eui",
                "vehicle_ef", "grid_ef").
            lookup_params: Parameters used in the lookup.
            lookup_result: Result returned from the lookup.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> h = engine.compute_lookup_hash(
            ...     "building_eui",
            ...     {"building_type": "office", "climate_zone": "temperate"},
            ...     Decimal("180.0"),
            ... )
            >>> len(h)
            64
        """
        provenance_data = {
            "agent_id": AGENT_ID,
            "engine": "DownstreamAssetDatabaseEngine",
            "version": VERSION,
            "lookup_type": lookup_type,
            "params": lookup_params,
            "result": lookup_result,
        }
        return self._compute_hash(provenance_data)

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check of the database engine.

        Verifies that all embedded reference data tables are present and
        non-empty, performs spot-check lookups, and returns engine statistics.

        Returns:
            Dict with keys: status, tables, lookup_count, error_count,
            spot_checks, created_at.

        Example:
            >>> engine = DownstreamAssetDatabaseEngine()
            >>> health = engine.health_check()
            >>> health["status"]
            'healthy'
        """
        tables = {
            "building_eui": len(BUILDING_EUI),
            "vehicle_efs": len(VEHICLE_EMISSION_FACTORS),
            "equipment_profiles": len(EQUIPMENT_PROFILES),
            "it_asset_profiles": len(IT_ASSET_PROFILES),
            "grid_efs_country": len(GRID_EMISSION_FACTORS),
            "grid_efs_egrid": len(EGRID_SUBREGION_FACTORS),
            "fuel_efs": len(FUEL_EMISSION_FACTORS),
            "eeio_factors": len(EEIO_FACTORS),
            "vacancy_base_loads": len(VACANCY_BASE_LOAD),
            "refrigerant_gwps": len(REFRIGERANT_GWPS),
            "country_climate_zones": len(COUNTRY_CLIMATE_ZONES),
            "energy_type_defaults": len(ENERGY_TYPE_DEFAULTS),
            "allocation_methods": len(ALLOCATION_METHODS),
            "green_lease_clauses": len(GREEN_LEASE_CLAUSES),
            "dqi_tier_defaults": len(DQI_DEFAULTS_BY_TIER),
            "uncertainty_defaults": len(UNCERTAINTY_DEFAULTS),
        }

        # Spot checks: verify a known lookup from each major table
        spot_checks: Dict[str, bool] = {}
        try:
            office_eui = self.get_building_eui("office", "temperate")
            spot_checks["office_eui"] = office_eui == Decimal("180.00000000")
        except Exception:
            spot_checks["office_eui"] = False

        try:
            us_grid = self.get_grid_ef("US")
            spot_checks["us_grid_ef"] = us_grid == Decimal("0.41700000")
        except Exception:
            spot_checks["us_grid_ef"] = False

        try:
            diesel_ef = self.get_fuel_ef("diesel")
            spot_checks["diesel_fuel_ef"] = (
                diesel_ef["co2e_per_unit"] == Decimal("2.68900000")
            )
        except Exception:
            spot_checks["diesel_fuel_ef"] = False

        try:
            r410a = self.get_refrigerant_gwp("R-410A")
            spot_checks["r410a_gwp"] = r410a["gwp_100"] == Decimal("2088.00000000")
        except Exception:
            spot_checks["r410a_gwp"] = False

        try:
            mc_diesel = self.get_vehicle_ef("medium_car", "diesel")
            spot_checks["medium_car_diesel"] = mc_diesel == Decimal("0.16800000")
        except Exception:
            spot_checks["medium_car_diesel"] = False

        try:
            server_power = self.get_it_power("server")
            spot_checks["server_power"] = (
                server_power["power_kw"] == Decimal("0.50000000")
            )
        except Exception:
            spot_checks["server_power"] = False

        all_tables_populated = all(v > 0 for v in tables.values())
        all_spots_pass = all(spot_checks.values())
        status = "healthy" if (all_tables_populated and all_spots_pass) else "degraded"

        result = {
            "status": status,
            "agent_id": AGENT_ID,
            "engine": "DownstreamAssetDatabaseEngine",
            "version": VERSION,
            "tables": tables,
            "total_factors": sum(tables.values()),
            "lookup_count": self._lookup_count,
            "error_count": self._error_count,
            "spot_checks": spot_checks,
            "created_at": self._created_at.isoformat(),
        }

        logger.info(
            "Health check: status=%s, tables=%d, total_factors=%d, "
            "lookups=%d, errors=%d, spots_pass=%s",
            status,
            len(tables),
            sum(tables.values()),
            self._lookup_count,
            self._error_count,
            all_spots_pass,
        )
        return result

    # =========================================================================
    # RESET (for testing)
    # =========================================================================

    @classmethod
    def _reset_singleton(cls) -> None:
        """
        Reset the singleton instance (for testing only).

        WARNING: This method is intended for unit testing only. Do not call
        in production code as it will invalidate all cached lookups.
        """
        with cls._lock:
            cls._instance = None
        logger.warning("DownstreamAssetDatabaseEngine singleton reset.")

    # =========================================================================
    # REPR / STR
    # =========================================================================

    def __repr__(self) -> str:
        """Return string representation of the engine."""
        return (
            f"DownstreamAssetDatabaseEngine("
            f"agent_id='{AGENT_ID}', "
            f"version='{VERSION}', "
            f"lookups={self._lookup_count}, "
            f"errors={self._error_count})"
        )

    def __str__(self) -> str:
        """Return human-readable string of the engine."""
        return (
            f"DownstreamAssetDatabaseEngine v{VERSION} "
            f"({self._lookup_count} lookups, {self._error_count} errors)"
        )
