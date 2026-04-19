"""
PACK-035 Energy Benchmark Pack - Configuration Manager

This module implements the EnergyBenchmarkConfig and PackConfig classes
that load, merge, and validate all configuration for the Energy Benchmark
Pack. It provides comprehensive Pydantic v2 models for every aspect of
building and facility energy benchmarking: EUI calculation, weather
normalisation, peer comparison, portfolio benchmarking, gap analysis,
performance ratings, trend monitoring, and reporting.

Building Types:
    - OFFICE: Commercial offices (single/multi-tenant, open plan, cellular)
    - RETAIL: Retail stores, shopping centres, showrooms
    - WAREHOUSE: Warehouses, distribution centres, cold storage
    - MANUFACTURING: Manufacturing facilities, factories, workshops
    - HEALTHCARE: Hospitals, clinics, care homes, laboratories
    - EDUCATION: Schools, universities, research facilities
    - DATA_CENTER: Data centres, server farms, colocation facilities
    - HOTEL: Hotels, serviced apartments, hostels
    - RESTAURANT: Restaurants, cafeterias, commercial kitchens
    - MIXED_USE: Mixed-use developments (office + retail + residential)
    - RESIDENTIAL_MULTIFAMILY: Multi-family residential buildings, apartments
    - LABORATORY: Research laboratories, cleanrooms, testing facilities
    - LIBRARY: Public/academic libraries, archives
    - WORSHIP: Places of worship (churches, mosques, synagogues, temples)
    - ENTERTAINMENT: Cinemas, theatres, concert halls, museums
    - SPORTS: Sports facilities, leisure centres, swimming pools, gyms
    - PARKING: Multi-storey car parks, underground parking
    - SME: Small commercial premises (simplified benchmarking)

Climate Zones:
    - ASHRAE 1A through 8: Per ASHRAE 169-2021 climate zone definitions
    - Koppen-Geiger: Af, Am, Aw, BWh, BWk, BSh, BSk, Cfa, Cfb, Cfc,
      Csa, Csb, Csc, Cwa, Cwb, Cwc, Dfa, Dfb, Dfc, Dfd, Dsa, Dsb,
      Dsc, Dsd, Dwa, Dwb, Dwc, Dwd, ET

Benchmark Sources:
    - ENERGY_STAR: US EPA ENERGY STAR Portfolio Manager methodology
    - CIBSE_TM46: UK CIBSE Technical Memorandum 46 building energy benchmarks
    - DIN_V_18599: German energy performance calculation standard
    - RT_2020: French RE 2020 regulatory thermal method
    - BPIE: Buildings Performance Institute Europe cross-country benchmarks
    - NATIONAL_AGENCY: Country-specific national energy agency benchmarks
    - CUSTOM: User-defined benchmark datasets

Rating Systems:
    - ENERGY_STAR: US EPA 1-100 score (normalised regression model)
    - EPC_EU: EU Energy Performance Certificate A-G classification
    - DEC_UK: UK Display Energy Certificate operational rating
    - NABERS_AU: Australian NABERS 1-6 star rating
    - CRREM: Carbon Risk Real Estate Monitor decarbonisation pathway
    - CUSTOM: User-defined rating thresholds

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (commercial_office / industrial_manufacturing / retail_store /
       warehouse_logistics / healthcare_facility / educational_campus /
       data_center / multi_site_portfolio)
    3. Environment overrides (ENERGY_BENCHMARK_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - EPBD: Directive 2024/1275 (Energy Performance of Buildings Directive recast)
    - EED: Directive (EU) 2023/1791 (Energy Efficiency Directive recast)
    - ISO 50001:2018 (Energy management systems)
    - ISO 50006:2014 (Energy baselines and EnPIs)
    - ASHRAE Standard 100-2018 (Energy efficiency in existing buildings)
    - ENERGY STAR Portfolio Manager Technical Reference
    - CIBSE TM46:2008 (Energy benchmarks)
    - EN 15603:2008 (Primary energy factors)
    - MEES: Minimum Energy Efficiency Standards (UK)
    - LL97: Local Law 97 (NYC carbon emissions limits)
    - NABERS: National Australian Built Environment Rating System

Example:
    >>> config = PackConfig.from_preset("commercial_office")
    >>> print(config.pack.building_type)
    BuildingType.OFFICE
    >>> print(config.pack.eui.floor_area_type)
    FloorAreaType.GIA
    >>> print(config.pack.peer_comparison.benchmark_sources)
    [BenchmarkSource.ENERGY_STAR, BenchmarkSource.CIBSE_TM46]
"""

import hashlib
import json
import logging
import os
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from greenlang.schemas.enums import ReportFormat

logger = logging.getLogger(__name__)

# Base directory for all pack configuration files
PACK_BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent


# =============================================================================
# Enums - Energy benchmark enumeration types
# =============================================================================


class BuildingType(str, Enum):
    """Building type classification for benchmark scoping.

    Each type maps to a distinct set of reference EUI ranges, end-use
    profiles, and applicable benchmark sources. The type drives which
    comparison cohort is selected for peer benchmarking.
    """

    OFFICE = "OFFICE"
    RETAIL = "RETAIL"
    WAREHOUSE = "WAREHOUSE"
    MANUFACTURING = "MANUFACTURING"
    HEALTHCARE = "HEALTHCARE"
    EDUCATION = "EDUCATION"
    DATA_CENTER = "DATA_CENTER"
    HOTEL = "HOTEL"
    RESTAURANT = "RESTAURANT"
    MIXED_USE = "MIXED_USE"
    RESIDENTIAL_MULTIFAMILY = "RESIDENTIAL_MULTIFAMILY"
    LABORATORY = "LABORATORY"
    LIBRARY = "LIBRARY"
    WORSHIP = "WORSHIP"
    ENTERTAINMENT = "ENTERTAINMENT"
    SPORTS = "SPORTS"
    PARKING = "PARKING"
    SME = "SME"


class ClimateZone(str, Enum):
    """Climate zone classification for weather-normalised benchmarking.

    ASHRAE zones per ASHRAE 169-2021. Koppen-Geiger zones for
    international coverage where ASHRAE zones are not mapped.
    """

    # ASHRAE 169-2021 zones
    ASHRAE_1A = "ASHRAE_1A"
    ASHRAE_1B = "ASHRAE_1B"
    ASHRAE_2A = "ASHRAE_2A"
    ASHRAE_2B = "ASHRAE_2B"
    ASHRAE_3A = "ASHRAE_3A"
    ASHRAE_3B = "ASHRAE_3B"
    ASHRAE_3C = "ASHRAE_3C"
    ASHRAE_4A = "ASHRAE_4A"
    ASHRAE_4B = "ASHRAE_4B"
    ASHRAE_4C = "ASHRAE_4C"
    ASHRAE_5A = "ASHRAE_5A"
    ASHRAE_5B = "ASHRAE_5B"
    ASHRAE_5C = "ASHRAE_5C"
    ASHRAE_6A = "ASHRAE_6A"
    ASHRAE_6B = "ASHRAE_6B"
    ASHRAE_7 = "ASHRAE_7"
    ASHRAE_8 = "ASHRAE_8"

    # Koppen-Geiger climate classifications
    KOPPEN_AF = "KOPPEN_AF"
    KOPPEN_AM = "KOPPEN_AM"
    KOPPEN_AW = "KOPPEN_AW"
    KOPPEN_BWH = "KOPPEN_BWH"
    KOPPEN_BWK = "KOPPEN_BWK"
    KOPPEN_BSH = "KOPPEN_BSH"
    KOPPEN_BSK = "KOPPEN_BSK"
    KOPPEN_CFA = "KOPPEN_CFA"
    KOPPEN_CFB = "KOPPEN_CFB"
    KOPPEN_CFC = "KOPPEN_CFC"
    KOPPEN_CSA = "KOPPEN_CSA"
    KOPPEN_CSB = "KOPPEN_CSB"
    KOPPEN_CSC = "KOPPEN_CSC"
    KOPPEN_CWA = "KOPPEN_CWA"
    KOPPEN_CWB = "KOPPEN_CWB"
    KOPPEN_CWC = "KOPPEN_CWC"
    KOPPEN_DFA = "KOPPEN_DFA"
    KOPPEN_DFB = "KOPPEN_DFB"
    KOPPEN_DFC = "KOPPEN_DFC"
    KOPPEN_DFD = "KOPPEN_DFD"
    KOPPEN_DSA = "KOPPEN_DSA"
    KOPPEN_DSB = "KOPPEN_DSB"
    KOPPEN_DSC = "KOPPEN_DSC"
    KOPPEN_DSD = "KOPPEN_DSD"
    KOPPEN_DWA = "KOPPEN_DWA"
    KOPPEN_DWB = "KOPPEN_DWB"
    KOPPEN_DWC = "KOPPEN_DWC"
    KOPPEN_DWD = "KOPPEN_DWD"
    KOPPEN_ET = "KOPPEN_ET"


class FloorAreaType(str, Enum):
    """Floor area measurement type for EUI denominator.

    The choice of floor area type can change the EUI by 15-30%.
    Consistent use within a peer group is critical for valid comparison.
    """

    GIA = "GIA"  # Gross Internal Area (RICS definition)
    NIA = "NIA"  # Net Internal Area (RICS definition, excludes walls/shafts)
    GLA = "GLA"  # Gross Lettable Area (retail/commercial leasing metric)
    TFA = "TFA"  # Treated Floor Area (EPBD/EPC definition, heated/cooled only)


class EnergyCarrier(str, Enum):
    """Energy carrier types tracked for benchmark calculations.

    Each carrier has associated source energy and primary energy
    conversion factors used in site-to-source and primary energy
    calculations.
    """

    ELECTRICITY = "ELECTRICITY"
    NATURAL_GAS = "NATURAL_GAS"
    FUEL_OIL = "FUEL_OIL"
    LPG = "LPG"
    DISTRICT_HEATING = "DISTRICT_HEATING"
    DISTRICT_COOLING = "DISTRICT_COOLING"
    BIOMASS = "BIOMASS"
    SOLAR_THERMAL = "SOLAR_THERMAL"
    OTHER = "OTHER"


class BenchmarkSource(str, Enum):
    """Benchmark data source for peer comparison.

    Each source provides reference EUI distributions for building types
    within specific geographies and climate zones.
    """

    ENERGY_STAR = "ENERGY_STAR"
    CIBSE_TM46 = "CIBSE_TM46"
    DIN_V_18599 = "DIN_V_18599"
    RT_2020 = "RT_2020"
    BPIE = "BPIE"
    NATIONAL_AGENCY = "NATIONAL_AGENCY"
    CUSTOM = "CUSTOM"


class NormalisationMethod(str, Enum):
    """Weather normalisation method for EUI adjustment.

    SIMPLE_RATIO: Degree-day ratio method (TMY / actual HDD or CDD).
    REGRESSION: Change-point regression (2P-5P models per ASHRAE RP-1050).
    CHANGE_POINT: Automated change-point model selection (best-fit among
    2P heating, 2P cooling, 3P heating, 3P cooling, 4P, 5P).
    """

    SIMPLE_RATIO = "SIMPLE_RATIO"
    REGRESSION = "REGRESSION"
    CHANGE_POINT = "CHANGE_POINT"


class RatingSystem(str, Enum):
    """Performance rating system for benchmark score assignment.

    Each system produces a score or grade that communicates performance
    relative to the applicable population or regulatory threshold.
    """

    ENERGY_STAR = "ENERGY_STAR"
    EPC_EU = "EPC_EU"
    DEC_UK = "DEC_UK"
    NABERS_AU = "NABERS_AU"
    CRREM = "CRREM"
    CUSTOM = "CUSTOM"


class AggregationMethod(str, Enum):
    """Portfolio-level aggregation method.

    AREA_WEIGHTED: Sum(EUI_i * Area_i) / Sum(Area_i) -- standard method.
    SIMPLE_AVERAGE: Mean(EUI_i) -- treats all buildings equally.
    CONSUMPTION_WEIGHTED: Sum(Consumption_i) / Sum(Area_i) -- same as
    area-weighted when EUI = Consumption / Area, used for mixed metrics.
    MEDIAN: Median(EUI_i) -- robust to outliers.
    """

    AREA_WEIGHTED = "AREA_WEIGHTED"
    SIMPLE_AVERAGE = "SIMPLE_AVERAGE"
    CONSUMPTION_WEIGHTED = "CONSUMPTION_WEIGHTED"
    MEDIAN = "MEDIAN"


class AlertChannel(str, Enum):
    """Alert notification channel for continuous monitoring."""

    EMAIL = "EMAIL"
    SLACK = "SLACK"
    WEBHOOK = "WEBHOOK"
    SMS = "SMS"


class DataFrequency(str, Enum):
    """Metered data collection frequency.

    Finer frequencies (HALF_HOURLY, FIFTEEN_MINUTE) enable load-duration
    curve analysis and peak demand attribution but require more storage
    and processing resources.
    """

    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    ANNUAL = "ANNUAL"
    HALF_HOURLY = "HALF_HOURLY"
    FIFTEEN_MINUTE = "FIFTEEN_MINUTE"


# =============================================================================
# Reference Data Constants
# =============================================================================


# Building type display names, typical EUI ranges (kWh/m2/yr site energy),
# dominant end uses, and applicable benchmark sources.
BUILDING_TYPE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "OFFICE": {
        "name": "Commercial Office",
        "typical_eui_range_kwh_m2": (100, 350),
        "good_practice_eui_kwh_m2": 150,
        "typical_practice_eui_kwh_m2": 250,
        "dominant_end_uses": ["HVAC", "Lighting", "Plug Loads", "DHW"],
        "applicable_sources": ["ENERGY_STAR", "CIBSE_TM46", "DIN_V_18599", "BPIE"],
        "cibse_tm46_category": "General Office",
        "energy_star_property_type": "Office",
        "heating_benchmark_applicable": True,
        "cooling_benchmark_applicable": True,
    },
    "RETAIL": {
        "name": "Retail Store",
        "typical_eui_range_kwh_m2": (200, 500),
        "good_practice_eui_kwh_m2": 220,
        "typical_practice_eui_kwh_m2": 370,
        "dominant_end_uses": ["Lighting", "Refrigeration", "HVAC", "Plug Loads"],
        "applicable_sources": ["ENERGY_STAR", "CIBSE_TM46", "BPIE"],
        "cibse_tm46_category": "General Retail",
        "energy_star_property_type": "Retail Store",
        "heating_benchmark_applicable": True,
        "cooling_benchmark_applicable": True,
    },
    "WAREHOUSE": {
        "name": "Warehouse / Distribution Centre",
        "typical_eui_range_kwh_m2": (50, 200),
        "good_practice_eui_kwh_m2": 70,
        "typical_practice_eui_kwh_m2": 130,
        "dominant_end_uses": ["Heating", "Lighting", "MHE Charging", "Dock Doors"],
        "applicable_sources": ["ENERGY_STAR", "CIBSE_TM46", "BPIE"],
        "cibse_tm46_category": "Distribution Warehouse",
        "energy_star_property_type": "Warehouse",
        "heating_benchmark_applicable": True,
        "cooling_benchmark_applicable": False,
    },
    "MANUFACTURING": {
        "name": "Manufacturing Facility",
        "typical_eui_range_kwh_m2": (200, 1000),
        "good_practice_eui_kwh_m2": 300,
        "typical_practice_eui_kwh_m2": 600,
        "dominant_end_uses": ["Process Energy", "Compressed Air", "HVAC", "Lighting"],
        "applicable_sources": ["NATIONAL_AGENCY", "BPIE"],
        "cibse_tm46_category": "Factory",
        "energy_star_property_type": "Manufacturing/Industrial Plant",
        "heating_benchmark_applicable": True,
        "cooling_benchmark_applicable": True,
    },
    "HEALTHCARE": {
        "name": "Healthcare Facility",
        "typical_eui_range_kwh_m2": (300, 700),
        "good_practice_eui_kwh_m2": 350,
        "typical_practice_eui_kwh_m2": 520,
        "dominant_end_uses": ["HVAC/Ventilation", "Lighting", "Medical Equipment", "Steam/DHW"],
        "applicable_sources": ["ENERGY_STAR", "CIBSE_TM46", "DIN_V_18599"],
        "cibse_tm46_category": "General Acute Hospital",
        "energy_star_property_type": "Hospital (General Medical & Surgical)",
        "heating_benchmark_applicable": True,
        "cooling_benchmark_applicable": True,
    },
    "EDUCATION": {
        "name": "Educational Facility",
        "typical_eui_range_kwh_m2": (100, 350),
        "good_practice_eui_kwh_m2": 120,
        "typical_practice_eui_kwh_m2": 230,
        "dominant_end_uses": ["Heating", "Lighting", "Ventilation", "IT Equipment"],
        "applicable_sources": ["ENERGY_STAR", "CIBSE_TM46", "DIN_V_18599", "BPIE"],
        "cibse_tm46_category": "Schools and Seasonal Public Buildings",
        "energy_star_property_type": "K-12 School",
        "heating_benchmark_applicable": True,
        "cooling_benchmark_applicable": True,
    },
    "DATA_CENTER": {
        "name": "Data Centre",
        "typical_eui_range_kwh_m2": (1000, 5000),
        "good_practice_eui_kwh_m2": 1200,
        "typical_practice_eui_kwh_m2": 2500,
        "dominant_end_uses": ["IT Load", "Cooling", "UPS Losses", "Lighting"],
        "applicable_sources": ["ENERGY_STAR", "BPIE"],
        "cibse_tm46_category": None,
        "energy_star_property_type": "Data Center",
        "heating_benchmark_applicable": False,
        "cooling_benchmark_applicable": True,
    },
    "HOTEL": {
        "name": "Hotel / Hospitality",
        "typical_eui_range_kwh_m2": (200, 500),
        "good_practice_eui_kwh_m2": 230,
        "typical_practice_eui_kwh_m2": 380,
        "dominant_end_uses": ["HVAC", "DHW", "Lighting", "Laundry", "Kitchen"],
        "applicable_sources": ["ENERGY_STAR", "CIBSE_TM46", "BPIE"],
        "cibse_tm46_category": "Hotels",
        "energy_star_property_type": "Hotel",
        "heating_benchmark_applicable": True,
        "cooling_benchmark_applicable": True,
    },
    "RESTAURANT": {
        "name": "Restaurant / Food Service",
        "typical_eui_range_kwh_m2": (500, 1200),
        "good_practice_eui_kwh_m2": 550,
        "typical_practice_eui_kwh_m2": 850,
        "dominant_end_uses": ["Kitchen Equipment", "HVAC", "Refrigeration", "DHW", "Lighting"],
        "applicable_sources": ["ENERGY_STAR", "CIBSE_TM46"],
        "cibse_tm46_category": "Restaurants",
        "energy_star_property_type": "Restaurant",
        "heating_benchmark_applicable": True,
        "cooling_benchmark_applicable": True,
    },
    "MIXED_USE": {
        "name": "Mixed Use Development",
        "typical_eui_range_kwh_m2": (150, 400),
        "good_practice_eui_kwh_m2": 180,
        "typical_practice_eui_kwh_m2": 300,
        "dominant_end_uses": ["HVAC", "Lighting", "Plug Loads", "Common Areas"],
        "applicable_sources": ["BPIE", "NATIONAL_AGENCY"],
        "cibse_tm46_category": "Mixed Use",
        "energy_star_property_type": "Mixed Use Property",
        "heating_benchmark_applicable": True,
        "cooling_benchmark_applicable": True,
    },
    "RESIDENTIAL_MULTIFAMILY": {
        "name": "Multi-Family Residential",
        "typical_eui_range_kwh_m2": (80, 250),
        "good_practice_eui_kwh_m2": 100,
        "typical_practice_eui_kwh_m2": 180,
        "dominant_end_uses": ["Heating", "DHW", "Lighting", "Appliances"],
        "applicable_sources": ["ENERGY_STAR", "BPIE", "NATIONAL_AGENCY"],
        "cibse_tm46_category": "Residential Spaces",
        "energy_star_property_type": "Multifamily Housing",
        "heating_benchmark_applicable": True,
        "cooling_benchmark_applicable": True,
    },
    "LABORATORY": {
        "name": "Laboratory / Research",
        "typical_eui_range_kwh_m2": (400, 1000),
        "good_practice_eui_kwh_m2": 450,
        "typical_practice_eui_kwh_m2": 700,
        "dominant_end_uses": ["Ventilation/Fume Hoods", "HVAC", "Process Equipment", "Lighting"],
        "applicable_sources": ["ENERGY_STAR", "CIBSE_TM46"],
        "cibse_tm46_category": "Laboratory or Operating Theatre",
        "energy_star_property_type": "Laboratory",
        "heating_benchmark_applicable": True,
        "cooling_benchmark_applicable": True,
    },
    "LIBRARY": {
        "name": "Library / Archive",
        "typical_eui_range_kwh_m2": (100, 300),
        "good_practice_eui_kwh_m2": 120,
        "typical_practice_eui_kwh_m2": 210,
        "dominant_end_uses": ["HVAC", "Lighting", "IT Systems", "Preservation HVAC"],
        "applicable_sources": ["CIBSE_TM46", "BPIE"],
        "cibse_tm46_category": "Libraries",
        "energy_star_property_type": "Library",
        "heating_benchmark_applicable": True,
        "cooling_benchmark_applicable": True,
    },
    "WORSHIP": {
        "name": "Place of Worship",
        "typical_eui_range_kwh_m2": (80, 250),
        "good_practice_eui_kwh_m2": 100,
        "typical_practice_eui_kwh_m2": 180,
        "dominant_end_uses": ["Heating", "Lighting", "Ventilation"],
        "applicable_sources": ["CIBSE_TM46", "NATIONAL_AGENCY"],
        "cibse_tm46_category": "Religious Buildings",
        "energy_star_property_type": "Worship Facility",
        "heating_benchmark_applicable": True,
        "cooling_benchmark_applicable": False,
    },
    "ENTERTAINMENT": {
        "name": "Entertainment Venue",
        "typical_eui_range_kwh_m2": (200, 500),
        "good_practice_eui_kwh_m2": 220,
        "typical_practice_eui_kwh_m2": 380,
        "dominant_end_uses": ["HVAC", "Lighting", "AV Equipment", "Stage/Display"],
        "applicable_sources": ["CIBSE_TM46", "NATIONAL_AGENCY"],
        "cibse_tm46_category": "Entertainment Halls",
        "energy_star_property_type": "Performing Arts",
        "heating_benchmark_applicable": True,
        "cooling_benchmark_applicable": True,
    },
    "SPORTS": {
        "name": "Sports / Leisure Facility",
        "typical_eui_range_kwh_m2": (300, 800),
        "good_practice_eui_kwh_m2": 350,
        "typical_practice_eui_kwh_m2": 600,
        "dominant_end_uses": ["Pool Heating", "HVAC", "Lighting", "DHW", "Ventilation"],
        "applicable_sources": ["CIBSE_TM46", "BPIE"],
        "cibse_tm46_category": "Fitness and Health Centre/Sports Centre",
        "energy_star_property_type": "Fitness Center/Health Club/Gym",
        "heating_benchmark_applicable": True,
        "cooling_benchmark_applicable": True,
    },
    "PARKING": {
        "name": "Multi-Storey Car Park",
        "typical_eui_range_kwh_m2": (15, 60),
        "good_practice_eui_kwh_m2": 20,
        "typical_practice_eui_kwh_m2": 40,
        "dominant_end_uses": ["Lighting", "Ventilation", "EV Charging", "Lifts"],
        "applicable_sources": ["CIBSE_TM46", "NATIONAL_AGENCY"],
        "cibse_tm46_category": "Car Park",
        "energy_star_property_type": "Parking",
        "heating_benchmark_applicable": False,
        "cooling_benchmark_applicable": False,
    },
    "SME": {
        "name": "Small Commercial Premises",
        "typical_eui_range_kwh_m2": (100, 350),
        "good_practice_eui_kwh_m2": 130,
        "typical_practice_eui_kwh_m2": 250,
        "dominant_end_uses": ["Heating", "Lighting", "Plug Loads"],
        "applicable_sources": ["NATIONAL_AGENCY", "BPIE"],
        "cibse_tm46_category": "General Office",
        "energy_star_property_type": "Office",
        "heating_benchmark_applicable": True,
        "cooling_benchmark_applicable": True,
    },
}


# Climate zone adjustments for heating and cooling intensity.
# Multipliers applied to baseline EUI to account for climate severity.
# Reference: ASHRAE 169-2021, degree-day normals from NOAA ISD/Meteonorm.
CLIMATE_ZONE_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    "ASHRAE_1A": {"heating_factor": 0.10, "cooling_factor": 1.80, "description": "Very Hot-Humid (Miami)"},
    "ASHRAE_1B": {"heating_factor": 0.08, "cooling_factor": 1.70, "description": "Very Hot-Dry"},
    "ASHRAE_2A": {"heating_factor": 0.25, "cooling_factor": 1.50, "description": "Hot-Humid (Houston)"},
    "ASHRAE_2B": {"heating_factor": 0.20, "cooling_factor": 1.60, "description": "Hot-Dry (Phoenix)"},
    "ASHRAE_3A": {"heating_factor": 0.45, "cooling_factor": 1.20, "description": "Warm-Humid (Atlanta)"},
    "ASHRAE_3B": {"heating_factor": 0.35, "cooling_factor": 1.30, "description": "Warm-Dry (Los Angeles)"},
    "ASHRAE_3C": {"heating_factor": 0.40, "cooling_factor": 0.60, "description": "Warm-Marine (San Francisco)"},
    "ASHRAE_4A": {"heating_factor": 0.70, "cooling_factor": 1.00, "description": "Mixed-Humid (New York)"},
    "ASHRAE_4B": {"heating_factor": 0.60, "cooling_factor": 1.10, "description": "Mixed-Dry (Albuquerque)"},
    "ASHRAE_4C": {"heating_factor": 0.65, "cooling_factor": 0.50, "description": "Mixed-Marine (Seattle)"},
    "ASHRAE_5A": {"heating_factor": 1.00, "cooling_factor": 0.70, "description": "Cool-Humid (Chicago)"},
    "ASHRAE_5B": {"heating_factor": 0.90, "cooling_factor": 0.60, "description": "Cool-Dry (Denver)"},
    "ASHRAE_5C": {"heating_factor": 0.85, "cooling_factor": 0.40, "description": "Cool-Marine (Vancouver)"},
    "ASHRAE_6A": {"heating_factor": 1.20, "cooling_factor": 0.50, "description": "Cold-Humid (Minneapolis)"},
    "ASHRAE_6B": {"heating_factor": 1.15, "cooling_factor": 0.45, "description": "Cold-Dry (Helena)"},
    "ASHRAE_7": {"heating_factor": 1.50, "cooling_factor": 0.25, "description": "Very Cold (Duluth)"},
    "ASHRAE_8": {"heating_factor": 1.80, "cooling_factor": 0.10, "description": "Subarctic (Fairbanks)"},
    # Koppen zones -- representative factors
    "KOPPEN_AF": {"heating_factor": 0.05, "cooling_factor": 1.90, "description": "Tropical Rainforest"},
    "KOPPEN_AM": {"heating_factor": 0.08, "cooling_factor": 1.80, "description": "Tropical Monsoon"},
    "KOPPEN_AW": {"heating_factor": 0.10, "cooling_factor": 1.70, "description": "Tropical Savanna"},
    "KOPPEN_BWH": {"heating_factor": 0.15, "cooling_factor": 1.85, "description": "Hot Desert"},
    "KOPPEN_BWK": {"heating_factor": 0.50, "cooling_factor": 1.20, "description": "Cold Desert"},
    "KOPPEN_BSH": {"heating_factor": 0.25, "cooling_factor": 1.50, "description": "Hot Semi-Arid"},
    "KOPPEN_BSK": {"heating_factor": 0.60, "cooling_factor": 0.90, "description": "Cold Semi-Arid"},
    "KOPPEN_CFA": {"heating_factor": 0.55, "cooling_factor": 1.10, "description": "Humid Subtropical"},
    "KOPPEN_CFB": {"heating_factor": 0.80, "cooling_factor": 0.50, "description": "Oceanic (London/Paris)"},
    "KOPPEN_CFC": {"heating_factor": 0.95, "cooling_factor": 0.30, "description": "Subpolar Oceanic"},
    "KOPPEN_CSA": {"heating_factor": 0.45, "cooling_factor": 1.30, "description": "Hot-Summer Mediterranean"},
    "KOPPEN_CSB": {"heating_factor": 0.55, "cooling_factor": 0.70, "description": "Warm-Summer Mediterranean"},
    "KOPPEN_CSC": {"heating_factor": 0.70, "cooling_factor": 0.40, "description": "Cold-Summer Mediterranean"},
    "KOPPEN_CWA": {"heating_factor": 0.40, "cooling_factor": 1.20, "description": "Monsoon-Subtropical"},
    "KOPPEN_CWB": {"heating_factor": 0.55, "cooling_factor": 0.80, "description": "Subtropical Highland"},
    "KOPPEN_CWC": {"heating_factor": 0.70, "cooling_factor": 0.50, "description": "Cold Subtropical Highland"},
    "KOPPEN_DFA": {"heating_factor": 1.00, "cooling_factor": 0.80, "description": "Hot-Summer Humid Continental"},
    "KOPPEN_DFB": {"heating_factor": 1.15, "cooling_factor": 0.55, "description": "Warm-Summer Humid Continental"},
    "KOPPEN_DFC": {"heating_factor": 1.40, "cooling_factor": 0.30, "description": "Subarctic"},
    "KOPPEN_DFD": {"heating_factor": 1.70, "cooling_factor": 0.20, "description": "Extremely Cold Subarctic"},
    "KOPPEN_DSA": {"heating_factor": 0.90, "cooling_factor": 1.00, "description": "Mediterranean Continental (Hot)"},
    "KOPPEN_DSB": {"heating_factor": 1.00, "cooling_factor": 0.70, "description": "Mediterranean Continental (Warm)"},
    "KOPPEN_DSC": {"heating_factor": 1.20, "cooling_factor": 0.40, "description": "Mediterranean Subarctic"},
    "KOPPEN_DSD": {"heating_factor": 1.50, "cooling_factor": 0.25, "description": "Mediterranean Extremely Cold"},
    "KOPPEN_DWA": {"heating_factor": 1.05, "cooling_factor": 0.85, "description": "Monsoon Continental (Hot)"},
    "KOPPEN_DWB": {"heating_factor": 1.20, "cooling_factor": 0.55, "description": "Monsoon Continental (Warm)"},
    "KOPPEN_DWC": {"heating_factor": 1.45, "cooling_factor": 0.30, "description": "Monsoon Subarctic"},
    "KOPPEN_DWD": {"heating_factor": 1.75, "cooling_factor": 0.15, "description": "Monsoon Extremely Cold"},
    "KOPPEN_ET": {"heating_factor": 1.90, "cooling_factor": 0.05, "description": "Tundra"},
}


# Source energy factors by energy carrier.
# Source: ENERGY STAR Technical Reference - Source Energy (2024).
# Site-to-source conversion for US grid average; adjustable per locale.
SOURCE_ENERGY_FACTORS: Dict[str, float] = {
    "ELECTRICITY": 2.80,
    "NATURAL_GAS": 1.05,
    "FUEL_OIL": 1.01,
    "LPG": 1.01,
    "DISTRICT_HEATING": 1.20,
    "DISTRICT_COOLING": 1.04,
    "BIOMASS": 1.00,
    "SOLAR_THERMAL": 1.00,
    "OTHER": 1.00,
}


# Primary energy factors by energy carrier.
# Source: EN 15603:2008 / prEN 15603:2016 national annex defaults.
# Non-renewable primary energy factor (f_p,nren).
PRIMARY_ENERGY_FACTORS: Dict[str, float] = {
    "ELECTRICITY": 2.50,
    "NATURAL_GAS": 1.10,
    "FUEL_OIL": 1.10,
    "LPG": 1.10,
    "DISTRICT_HEATING": 0.80,
    "DISTRICT_COOLING": 0.70,
    "BIOMASS": 0.20,
    "SOLAR_THERMAL": 0.00,
    "OTHER": 1.00,
}


# EPC grade thresholds (kWh/m2/yr primary energy) -- EU generic scale.
# National variations apply; these are illustrative reference values.
EPC_GRADE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "OFFICE": {"A": 50, "B": 100, "C": 150, "D": 200, "E": 275, "F": 350, "G": 999},
    "RETAIL": {"A": 70, "B": 140, "C": 210, "D": 310, "E": 400, "F": 500, "G": 999},
    "WAREHOUSE": {"A": 30, "B": 60, "C": 90, "D": 130, "E": 170, "F": 220, "G": 999},
    "HEALTHCARE": {"A": 100, "B": 200, "C": 300, "D": 400, "E": 530, "F": 660, "G": 999},
    "EDUCATION": {"A": 50, "B": 95, "C": 145, "D": 200, "E": 270, "F": 340, "G": 999},
    "DATA_CENTER": {"A": 500, "B": 1000, "C": 1500, "D": 2200, "E": 3000, "F": 4000, "G": 999},
    "HOTEL": {"A": 70, "B": 130, "C": 200, "D": 280, "E": 370, "F": 460, "G": 999},
    "RESTAURANT": {"A": 150, "B": 300, "C": 450, "D": 600, "E": 800, "F": 1000, "G": 999},
    "DEFAULT": {"A": 50, "B": 100, "C": 150, "D": 200, "E": 275, "F": 350, "G": 999},
}


# ENERGY STAR score lookup boundaries by property type (simplified).
# Full model uses multivariate regression; these are illustrative medians.
ENERGY_STAR_MEDIAN_EUI: Dict[str, float] = {
    "OFFICE": 210,
    "RETAIL": 370,
    "WAREHOUSE": 100,
    "HEALTHCARE": 500,
    "EDUCATION": 200,
    "DATA_CENTER": 2000,
    "HOTEL": 350,
    "RESTAURANT": 800,
    "RESIDENTIAL_MULTIFAMILY": 165,
    "LABORATORY": 650,
    "WORSHIP": 160,
}


# Available presets for the Energy Benchmark Pack
AVAILABLE_PRESETS: Dict[str, str] = {
    "commercial_office": "Commercial office buildings (single/multi-tenant), EUI 150-300, HVAC+lighting dominant",
    "industrial_manufacturing": "Manufacturing and industrial facilities, EUI 200-1000, process energy dominant",
    "retail_store": "Retail stores and shopping centres, EUI 200-500, lighting+refrigeration dominant",
    "warehouse_logistics": "Warehouses and distribution centres, EUI 50-200, heating+lighting dominant",
    "healthcare_facility": "Hospitals and healthcare facilities, EUI 300-700, 24/7 ventilation critical",
    "educational_campus": "Schools and universities, EUI 100-350, seasonal occupancy patterns",
    "data_center": "Data centres and server farms, PUE-based, cooling dominant, no heating benchmark",
    "multi_site_portfolio": "Portfolio mode for mixed building types, area-weighted aggregation, facility ranking",
}


# CIBSE TM46 benchmark categories with fossil-thermal and electricity splits
CIBSE_TM46_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "General Office": {"fossil_thermal_kwh_m2": 120, "electricity_kwh_m2": 95},
    "General Retail": {"fossil_thermal_kwh_m2": 120, "electricity_kwh_m2": 165},
    "Distribution Warehouse": {"fossil_thermal_kwh_m2": 55, "electricity_kwh_m2": 30},
    "Factory": {"fossil_thermal_kwh_m2": 200, "electricity_kwh_m2": 120},
    "General Acute Hospital": {"fossil_thermal_kwh_m2": 350, "electricity_kwh_m2": 80},
    "Schools and Seasonal Public Buildings": {"fossil_thermal_kwh_m2": 150, "electricity_kwh_m2": 32},
    "Hotels": {"fossil_thermal_kwh_m2": 200, "electricity_kwh_m2": 105},
    "Restaurants": {"fossil_thermal_kwh_m2": 370, "electricity_kwh_m2": 135},
    "Libraries": {"fossil_thermal_kwh_m2": 120, "electricity_kwh_m2": 54},
    "Religious Buildings": {"fossil_thermal_kwh_m2": 100, "electricity_kwh_m2": 15},
    "Entertainment Halls": {"fossil_thermal_kwh_m2": 150, "electricity_kwh_m2": 100},
    "Fitness and Health Centre/Sports Centre": {"fossil_thermal_kwh_m2": 300, "electricity_kwh_m2": 105},
    "Car Park": {"fossil_thermal_kwh_m2": 0, "electricity_kwh_m2": 20},
    "Laboratory or Operating Theatre": {"fossil_thermal_kwh_m2": 200, "electricity_kwh_m2": 160},
    "Residential Spaces": {"fossil_thermal_kwh_m2": 120, "electricity_kwh_m2": 40},
}


# =============================================================================
# Pydantic Sub-Config Models
# =============================================================================


class EUIConfig(BaseModel):
    """Configuration for Energy Use Intensity (EUI) calculation.

    EUI is the primary benchmark metric: total energy consumption divided
    by floor area (kWh/m2/yr). Configuration controls the accounting
    boundary (site vs. source vs. primary energy), floor area definition,
    renewable energy handling, and rolling period.
    """

    accounting_boundary: str = Field(
        "site",
        description="Energy accounting boundary: site (metered), source (upstream losses), "
        "primary (non-renewable primary energy per EN 15603)",
    )
    floor_area_type: FloorAreaType = Field(
        FloorAreaType.GIA,
        description="Floor area measurement type used as EUI denominator",
    )
    include_renewables: bool = Field(
        True,
        description="Include on-site renewable generation in net energy calculation",
    )
    renewable_offset_method: str = Field(
        "net_metered",
        description="How on-site renewables reduce consumption: net_metered, gross_generation, "
        "exported_only",
    )
    rolling_months: int = Field(
        12,
        ge=3,
        le=36,
        description="Rolling window in months for EUI calculation (12 = annual EUI)",
    )
    include_plug_loads: bool = Field(
        True,
        description="Include tenant plug loads in whole-building EUI",
    )
    include_process_loads: bool = Field(
        False,
        description="Include process/manufacturing loads (separate from building systems)",
    )
    include_exterior_loads: bool = Field(
        True,
        description="Include exterior lighting, signage, and site loads",
    )
    data_frequency: DataFrequency = Field(
        DataFrequency.MONTHLY,
        description="Input data frequency for EUI calculation",
    )
    minimum_data_coverage_pct: float = Field(
        90.0,
        ge=50.0,
        le=100.0,
        description="Minimum percentage of period with valid metered data",
    )
    energy_carriers: List[EnergyCarrier] = Field(
        default_factory=lambda: [
            EnergyCarrier.ELECTRICITY,
            EnergyCarrier.NATURAL_GAS,
        ],
        description="Energy carriers included in EUI calculation",
    )

    @field_validator("accounting_boundary")
    @classmethod
    def validate_accounting_boundary(cls, v: str) -> str:
        """Validate accounting boundary is a recognised type."""
        valid = {"site", "source", "primary"}
        if v.lower() not in valid:
            raise ValueError(
                f"Invalid accounting_boundary: {v}. Must be one of {sorted(valid)}."
            )
        return v.lower()

    @field_validator("renewable_offset_method")
    @classmethod
    def validate_renewable_offset_method(cls, v: str) -> str:
        """Validate renewable offset method."""
        valid = {"net_metered", "gross_generation", "exported_only"}
        if v.lower() not in valid:
            raise ValueError(
                f"Invalid renewable_offset_method: {v}. Must be one of {sorted(valid)}."
            )
        return v.lower()


class WeatherConfig(BaseModel):
    """Configuration for weather normalisation of energy consumption.

    Weather normalisation removes the effect of annual weather variation
    so that year-on-year comparisons and peer comparisons are meaningful.
    Uses degree-day regression models (2P-5P) per ASHRAE RP-1050 methodology.
    """

    enabled: bool = Field(
        True,
        description="Enable weather normalisation for EUI calculation",
    )
    normalisation_method: NormalisationMethod = Field(
        NormalisationMethod.CHANGE_POINT,
        description="Weather normalisation method",
    )
    base_temp_heating_c: float = Field(
        15.5,
        ge=5.0,
        le=25.0,
        description="Base temperature for heating degree days (deg C)",
    )
    base_temp_cooling_c: float = Field(
        18.0,
        ge=10.0,
        le=30.0,
        description="Base temperature for cooling degree days (deg C)",
    )
    min_r_squared: float = Field(
        0.70,
        ge=0.0,
        le=1.0,
        description="Minimum R-squared for regression model acceptance",
    )
    max_cv_rmse: float = Field(
        0.25,
        ge=0.0,
        le=1.0,
        description="Maximum CV(RMSE) for regression model acceptance",
    )
    weather_source: str = Field(
        "auto",
        description="Weather data source: auto (nearest station), noaa_isd, meteonorm, "
        "era5, custom_file",
    )
    weather_station_id: Optional[str] = Field(
        None,
        description="Specific weather station ID (overrides auto-selection)",
    )
    tmy_source: str = Field(
        "auto",
        description="Typical Meteorological Year source for normalisation baseline: "
        "auto, tmy3_noaa, meteonorm_tmy, era5_tmy",
    )
    degree_day_method: str = Field(
        "mean_temperature",
        description="Degree-day calculation: mean_temperature, integration (hourly)",
    )
    min_months_regression: int = Field(
        12,
        ge=6,
        le=36,
        description="Minimum months of paired energy-weather data for regression",
    )

    @field_validator("weather_source")
    @classmethod
    def validate_weather_source(cls, v: str) -> str:
        """Validate weather data source."""
        valid = {"auto", "noaa_isd", "meteonorm", "era5", "custom_file"}
        if v.lower() not in valid:
            raise ValueError(
                f"Invalid weather_source: {v}. Must be one of {sorted(valid)}."
            )
        return v.lower()

    @field_validator("degree_day_method")
    @classmethod
    def validate_degree_day_method(cls, v: str) -> str:
        """Validate degree-day calculation method."""
        valid = {"mean_temperature", "integration"}
        if v.lower() not in valid:
            raise ValueError(
                f"Invalid degree_day_method: {v}. Must be one of {sorted(valid)}."
            )
        return v.lower()


class PeerComparisonConfig(BaseModel):
    """Configuration for peer comparison benchmarking.

    Compares facility EUI against reference benchmark datasets filtered
    by building type, size, climate zone, and operating hours.
    """

    enabled: bool = Field(
        True,
        description="Enable peer comparison benchmarking",
    )
    benchmark_sources: List[BenchmarkSource] = Field(
        default_factory=lambda: [
            BenchmarkSource.ENERGY_STAR,
            BenchmarkSource.CIBSE_TM46,
        ],
        description="Benchmark data sources for peer comparison",
    )
    peer_group_criteria: List[str] = Field(
        default_factory=lambda: [
            "building_type",
            "climate_zone",
            "floor_area_band",
        ],
        description="Criteria used to form peer comparison group",
    )
    min_peer_count: int = Field(
        10,
        ge=3,
        le=100,
        description="Minimum number of peers for statistically valid comparison",
    )
    include_energy_star_score: bool = Field(
        True,
        description="Calculate ENERGY STAR 1-100 score (US properties or proxy)",
    )
    percentile_display: List[int] = Field(
        default_factory=lambda: [10, 25, 50, 75, 90],
        description="Percentile levels to display in peer distribution chart",
    )
    include_quartile_analysis: bool = Field(
        True,
        description="Show quartile breakdown (Q1 best-in-class to Q4 poor)",
    )
    include_best_practice_target: bool = Field(
        True,
        description="Show best-practice / top-decile target line",
    )
    normalise_for_hours: bool = Field(
        True,
        description="Normalise EUI for operating hours where peers have different schedules",
    )
    custom_benchmark_file: Optional[str] = Field(
        None,
        description="Path to custom benchmark CSV file (building_type, eui_kwh_m2 columns)",
    )


class PortfolioConfig(BaseModel):
    """Configuration for multi-facility portfolio benchmarking.

    Enables comparison across an entire real estate portfolio with
    mixed building types, sizes, and geographies.
    """

    enabled: bool = Field(
        False,
        description="Enable portfolio-level benchmarking (multi-facility mode)",
    )
    aggregation_method: AggregationMethod = Field(
        AggregationMethod.AREA_WEIGHTED,
        description="Method for aggregating portfolio-level EUI",
    )
    ranking_criteria: List[str] = Field(
        default_factory=lambda: [
            "eui_site",
            "eui_weather_normalised",
            "energy_star_score",
        ],
        description="Criteria used for facility ranking tables",
    )
    entity_levels: List[str] = Field(
        default_factory=lambda: [
            "portfolio",
            "region",
            "country",
            "building_type",
        ],
        description="Hierarchy levels for aggregation roll-up",
    )
    include_yoy_trend: bool = Field(
        True,
        description="Include year-on-year trend analysis at portfolio level",
    )
    include_league_table: bool = Field(
        True,
        description="Generate facility league table (worst to best)",
    )
    league_table_top_n: int = Field(
        10,
        ge=5,
        le=50,
        description="Number of top/bottom performers to highlight",
    )
    include_bubble_chart: bool = Field(
        True,
        description="Generate bubble chart (EUI vs. floor area, coloured by type)",
    )
    target_reduction_pct_pa: Optional[float] = Field(
        None,
        ge=0.0,
        le=20.0,
        description="Annual EUI reduction target (%) for portfolio trajectory",
    )
    sbti_pathway_enabled: bool = Field(
        False,
        description="Overlay SBTi 1.5C pathway on portfolio trajectory",
    )
    crrem_pathway_enabled: bool = Field(
        False,
        description="Overlay CRREM decarbonisation pathway on portfolio trajectory",
    )


class GapAnalysisConfig(BaseModel):
    """Configuration for energy performance gap analysis.

    Quantifies the gap between current performance and target/benchmark
    at the whole-building and end-use levels.
    """

    enabled: bool = Field(
        True,
        description="Enable performance gap analysis",
    )
    disaggregation_method: str = Field(
        "end_use_split",
        description="Gap disaggregation: end_use_split (sub-metered or estimated), "
        "carrier_split (by fuel), both",
    )
    end_use_categories: List[str] = Field(
        default_factory=lambda: [
            "heating",
            "cooling",
            "ventilation",
            "lighting",
            "dhw",
            "plug_loads",
            "process",
            "other",
        ],
        description="End-use categories for disaggregated gap analysis",
    )
    min_gap_threshold_pct: float = Field(
        5.0,
        ge=1.0,
        le=50.0,
        description="Minimum gap (%) to flag as significant improvement opportunity",
    )
    link_to_measures: bool = Field(
        True,
        description="Link identified gaps to recommended Energy Conservation Measures (ECMs)",
    )
    include_savings_potential: bool = Field(
        True,
        description="Estimate kWh and cost savings potential for each gap",
    )
    target_source: str = Field(
        "peer_median",
        description="Gap target reference: peer_median, peer_top_quartile, "
        "peer_top_decile, regulatory, custom",
    )
    include_capital_cost_estimate: bool = Field(
        False,
        description="Include indicative capital cost estimates for gap closure measures",
    )
    include_payback_period: bool = Field(
        True,
        description="Calculate simple payback period for gap closure measures",
    )

    @field_validator("disaggregation_method")
    @classmethod
    def validate_disaggregation_method(cls, v: str) -> str:
        """Validate gap disaggregation method."""
        valid = {"end_use_split", "carrier_split", "both"}
        if v.lower() not in valid:
            raise ValueError(
                f"Invalid disaggregation_method: {v}. Must be one of {sorted(valid)}."
            )
        return v.lower()

    @field_validator("target_source")
    @classmethod
    def validate_target_source(cls, v: str) -> str:
        """Validate gap target source."""
        valid = {"peer_median", "peer_top_quartile", "peer_top_decile", "regulatory", "custom"}
        if v.lower() not in valid:
            raise ValueError(
                f"Invalid target_source: {v}. Must be one of {sorted(valid)}."
            )
        return v.lower()


class RatingConfig(BaseModel):
    """Configuration for performance rating system assignments.

    Maps calculated EUI to one or more rating systems (EPC, ENERGY STAR,
    DEC, NABERS, CRREM) and generates the corresponding score or grade.
    """

    enabled: bool = Field(
        True,
        description="Enable performance rating calculations",
    )
    rating_systems: List[RatingSystem] = Field(
        default_factory=lambda: [
            RatingSystem.EPC_EU,
            RatingSystem.ENERGY_STAR,
        ],
        description="Rating systems to calculate",
    )
    epc_methodology: str = Field(
        "operational",
        description="EPC calculation basis: operational (metered), asset (modelled/calculated)",
    )
    epc_country_annex: str = Field(
        "DE",
        description="National annex for EPC thresholds (ISO 3166-1 alpha-2)",
    )
    crrem_pathway_year: int = Field(
        2050,
        ge=2025,
        le=2100,
        description="CRREM pathway target year for stranding risk assessment",
    )
    crrem_scenario: str = Field(
        "1.5C",
        description="CRREM climate scenario: 1.5C, 2.0C, NDC",
    )
    nabers_state: Optional[str] = Field(
        None,
        description="Australian state for NABERS rating (NSW, VIC, QLD, etc.)",
    )
    include_regulatory_compliance: bool = Field(
        True,
        description="Check compliance with regulatory minimum standards (MEES, LL97, MEPS)",
    )
    mees_threshold_grade: str = Field(
        "E",
        description="MEES minimum EPC grade (UK regulation: E from April 2023, C from 2028?)",
    )
    custom_rating_thresholds: Optional[Dict[str, float]] = Field(
        None,
        description="Custom grade thresholds {grade: max_eui_kwh_m2} for CUSTOM rating system",
    )

    @field_validator("epc_methodology")
    @classmethod
    def validate_epc_methodology(cls, v: str) -> str:
        """Validate EPC calculation methodology."""
        valid = {"operational", "asset"}
        if v.lower() not in valid:
            raise ValueError(
                f"Invalid epc_methodology: {v}. Must be one of {sorted(valid)}."
            )
        return v.lower()

    @field_validator("crrem_scenario")
    @classmethod
    def validate_crrem_scenario(cls, v: str) -> str:
        """Validate CRREM climate scenario."""
        valid = {"1.5C", "2.0C", "NDC"}
        if v not in valid:
            raise ValueError(
                f"Invalid crrem_scenario: {v}. Must be one of {sorted(valid)}."
            )
        return v


class TrendConfig(BaseModel):
    """Configuration for trend analysis and continuous monitoring.

    Provides rolling performance tracking, CUSUM (Cumulative Sum) charts
    for detecting drift, Statistical Process Control (SPC) for
    maintaining performance, and automated alerting.
    """

    enabled: bool = Field(
        True,
        description="Enable trend analysis and continuous monitoring",
    )
    rolling_window_months: int = Field(
        12,
        ge=3,
        le=36,
        description="Rolling window for trend calculation (months)",
    )
    cusum_enabled: bool = Field(
        True,
        description="Enable CUSUM (Cumulative Sum) chart for performance drift detection",
    )
    cusum_threshold_kwh_m2: float = Field(
        10.0,
        ge=1.0,
        le=100.0,
        description="CUSUM alarm threshold (kWh/m2 cumulative deviation from target)",
    )
    spc_enabled: bool = Field(
        True,
        description="Enable Statistical Process Control (SPC) monitoring",
    )
    spc_sigma: float = Field(
        2.0,
        ge=1.0,
        le=4.0,
        description="SPC control limit in standard deviations (2 = warning, 3 = action)",
    )
    yoy_comparison: bool = Field(
        True,
        description="Include year-on-year same-month comparison",
    )
    seasonality_detection: bool = Field(
        True,
        description="Auto-detect and adjust for seasonal energy patterns",
    )
    alert_enabled: bool = Field(
        True,
        description="Enable automated alerts for performance deviations",
    )
    alert_channels: List[AlertChannel] = Field(
        default_factory=lambda: [AlertChannel.EMAIL],
        description="Notification channels for performance alerts",
    )
    alert_threshold_pct: float = Field(
        10.0,
        ge=1.0,
        le=50.0,
        description="Alert trigger: EUI deviation above target by this percentage",
    )
    alert_recipients: List[str] = Field(
        default_factory=list,
        description="Email addresses or webhook URLs for alert delivery",
    )
    forecast_enabled: bool = Field(
        False,
        description="Enable 12-month forward EUI forecast using historical trends",
    )
    forecast_method: str = Field(
        "linear_trend",
        description="Forecast method: linear_trend, exponential_smoothing, arima",
    )

    @field_validator("forecast_method")
    @classmethod
    def validate_forecast_method(cls, v: str) -> str:
        """Validate forecast method."""
        valid = {"linear_trend", "exponential_smoothing", "arima"}
        if v.lower() not in valid:
            raise ValueError(
                f"Invalid forecast_method: {v}. Must be one of {sorted(valid)}."
            )
        return v.lower()


class ReportConfig(BaseModel):
    """Configuration for benchmark report generation."""

    formats: List[ReportFormat] = Field(
        default_factory=lambda: [
            ReportFormat.MARKDOWN,
            ReportFormat.JSON,
        ],
        description="Output formats for benchmark reports",
    )
    include_methodology: bool = Field(
        True,
        description="Include methodology appendix describing calculation approach",
    )
    include_provenance: bool = Field(
        True,
        description="Include data provenance section with SHA-256 hashes",
    )
    include_data_quality_notes: bool = Field(
        True,
        description="Include data quality assessment and caveats",
    )
    include_recommendations: bool = Field(
        True,
        description="Include energy efficiency recommendations based on gaps",
    )
    include_executive_summary: bool = Field(
        True,
        description="Include executive summary with key findings",
    )
    include_charts: bool = Field(
        True,
        description="Include charts and visualisations in HTML/Markdown reports",
    )
    language: str = Field(
        "en",
        description="Report language (ISO 639-1 code)",
    )
    watermark_draft: bool = Field(
        True,
        description="Apply DRAFT watermark to unapproved reports",
    )
    output_directory: Optional[str] = Field(
        None,
        description="Output directory for generated reports (uses temp dir if None)",
    )
    template_override: Optional[str] = Field(
        None,
        description="Path to custom Jinja2 report template",
    )


class PerformanceConfig(BaseModel):
    """Performance and resource limits for pack execution.

    Governs caching, parallelism, memory, and timeout parameters
    to ensure the pack runs within infrastructure constraints.
    """

    cache_ttl_seconds: int = Field(
        3600,
        ge=60,
        le=86400,
        description="Cache TTL for benchmark reference data and weather data (seconds)",
    )
    max_facilities: int = Field(
        100,
        ge=1,
        le=5000,
        description="Maximum number of facilities per benchmark run",
    )
    parallel_processing: bool = Field(
        True,
        description="Enable parallel processing for multi-facility benchmarking",
    )
    max_parallel_workers: int = Field(
        4,
        ge=1,
        le=32,
        description="Maximum number of parallel worker threads/processes",
    )
    memory_ceiling_mb: int = Field(
        2048,
        ge=256,
        le=16384,
        description="Maximum memory ceiling in MB for benchmark calculations",
    )
    batch_size: int = Field(
        500,
        ge=50,
        le=10000,
        description="Batch size for bulk facility processing",
    )
    calculation_timeout_seconds: int = Field(
        120,
        ge=10,
        le=3600,
        description="Timeout for individual facility benchmark calculation (seconds)",
    )
    weather_data_cache_enabled: bool = Field(
        True,
        description="Cache downloaded weather data to avoid repeated API calls",
    )
    lazy_loading: bool = Field(
        True,
        description="Lazy-load benchmark reference datasets (load on first use)",
    )


class SecurityConfig(BaseModel):
    """Security and access control configuration."""

    roles: List[str] = Field(
        default_factory=lambda: [
            "energy_manager",
            "facility_manager",
            "portfolio_analyst",
            "sustainability_lead",
            "viewer",
            "admin",
        ],
        description="Available RBAC roles for the pack",
    )
    data_classification: str = Field(
        "INTERNAL",
        description="Default data classification: PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED",
    )
    audit_logging: bool = Field(
        True,
        description="Enable security audit logging for all data access",
    )
    pii_redaction: bool = Field(
        True,
        description="Enable PII redaction in exported reports",
    )
    encryption_at_rest: bool = Field(
        True,
        description="Require encryption at rest for stored benchmark data",
    )


class AuditTrailConfig(BaseModel):
    """Configuration for calculation audit trail and provenance."""

    enabled: bool = Field(
        True,
        description="Enable audit trail for all benchmark calculations",
    )
    sha256_provenance: bool = Field(
        True,
        description="Generate SHA-256 provenance hashes for all outputs",
    )
    calculation_logging: bool = Field(
        True,
        description="Log all intermediate calculation steps",
    )
    assumption_tracking: bool = Field(
        True,
        description="Track all assumptions and default values used in calculations",
    )
    data_lineage_enabled: bool = Field(
        True,
        description="Track full data lineage from source meter data to benchmark output",
    )
    retention_years: int = Field(
        7,
        ge=1,
        le=15,
        description="Audit trail retention period in years",
    )
    external_audit_export: bool = Field(
        True,
        description="Enable export format for third-party energy auditors",
    )


# =============================================================================
# Main Configuration Model
# =============================================================================


class EnergyBenchmarkConfig(BaseModel):
    """Main configuration for PACK-035 Energy Benchmark Pack.

    This is the root configuration model that contains all sub-configurations
    for energy benchmarking. The building_type and climate_zone fields drive
    which benchmark datasets, rating thresholds, and weather adjustments are
    applied.
    """

    # Facility identification
    facility_id: str = Field(
        "",
        description="Unique facility identifier (internal reference)",
    )
    facility_name: str = Field(
        "",
        description="Facility name or site identifier",
    )
    building_type: BuildingType = Field(
        BuildingType.OFFICE,
        description="Primary building type classification",
    )
    climate_zone: ClimateZone = Field(
        ClimateZone.ASHRAE_4A,
        description="Climate zone of the facility location",
    )
    country_code: str = Field(
        "DE",
        description="Facility country (ISO 3166-1 alpha-2)",
    )
    currency: str = Field(
        "EUR",
        description="Currency for cost calculations (ISO 4217)",
    )

    # Building characteristics
    floor_area_m2: Optional[float] = Field(
        None,
        ge=1.0,
        description="Total floor area in square metres",
    )
    floor_area_type: FloorAreaType = Field(
        FloorAreaType.GIA,
        description="Floor area measurement type",
    )
    year_built: Optional[int] = Field(
        None,
        ge=1800,
        le=2030,
        description="Year the building was constructed",
    )
    year_last_refurbished: Optional[int] = Field(
        None,
        ge=1950,
        le=2030,
        description="Year of last major refurbishment",
    )
    number_of_floors: Optional[int] = Field(
        None,
        ge=1,
        le=200,
        description="Number of above-ground floors",
    )
    operating_hours_per_week: Optional[float] = Field(
        None,
        ge=1.0,
        le=168.0,
        description="Weekly operating hours (168 = 24/7)",
    )
    occupancy_pct: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Average occupancy as percentage of design capacity",
    )
    number_of_occupants: Optional[int] = Field(
        None,
        ge=0,
        description="Typical number of building occupants",
    )

    # Location
    latitude: Optional[float] = Field(
        None,
        ge=-90.0,
        le=90.0,
        description="Facility latitude (decimal degrees, for weather station lookup)",
    )
    longitude: Optional[float] = Field(
        None,
        ge=-180.0,
        le=180.0,
        description="Facility longitude (decimal degrees)",
    )

    # Reporting
    reporting_year: int = Field(
        2025,
        ge=2015,
        le=2035,
        description="Reporting / assessment year",
    )

    # Energy cost
    electricity_cost_per_kwh: Optional[float] = Field(
        None,
        ge=0.0,
        description="Blended electricity cost per kWh (for savings calculations)",
    )
    gas_cost_per_kwh: Optional[float] = Field(
        None,
        ge=0.0,
        description="Blended natural gas cost per kWh (for savings calculations)",
    )

    # Sub-configurations for each engine
    eui: EUIConfig = Field(
        default_factory=EUIConfig,
        description="EUI calculation configuration",
    )
    weather: WeatherConfig = Field(
        default_factory=WeatherConfig,
        description="Weather normalisation configuration",
    )
    peer_comparison: PeerComparisonConfig = Field(
        default_factory=PeerComparisonConfig,
        description="Peer comparison benchmarking configuration",
    )
    portfolio: PortfolioConfig = Field(
        default_factory=PortfolioConfig,
        description="Portfolio-level benchmarking configuration",
    )
    gap_analysis: GapAnalysisConfig = Field(
        default_factory=GapAnalysisConfig,
        description="Performance gap analysis configuration",
    )
    rating: RatingConfig = Field(
        default_factory=RatingConfig,
        description="Performance rating configuration",
    )
    trend: TrendConfig = Field(
        default_factory=TrendConfig,
        description="Trend analysis and monitoring configuration",
    )
    report: ReportConfig = Field(
        default_factory=ReportConfig,
        description="Report generation configuration",
    )

    # Supporting configurations
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance and resource limits",
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security and access control",
    )
    audit_trail: AuditTrailConfig = Field(
        default_factory=AuditTrailConfig,
        description="Audit trail and provenance",
    )

    @model_validator(mode="after")
    def validate_data_center_disables_heating(self) -> "EnergyBenchmarkConfig":
        """Data centres do not need heating benchmarks or weather heating normalisation."""
        if self.building_type == BuildingType.DATA_CENTER:
            # Data centres are cooling-dominated; heating benchmark is irrelevant
            building_info = BUILDING_TYPE_DEFAULTS.get("DATA_CENTER", {})
            if not building_info.get("heating_benchmark_applicable", False):
                logger.info(
                    "Data centre building type: heating benchmarks not applicable. "
                    "Weather normalisation will focus on cooling degree days only."
                )
        return self

    @model_validator(mode="after")
    def validate_portfolio_requires_multi_facility(self) -> "EnergyBenchmarkConfig":
        """Portfolio mode requires multi-facility support in performance config."""
        if self.portfolio.enabled and self.performance.max_facilities < 2:
            logger.warning(
                "Portfolio mode enabled but max_facilities is less than 2. "
                "Setting max_facilities to 100 for portfolio analysis."
            )
            object.__setattr__(self.performance, "max_facilities", 100)
        return self

    @model_validator(mode="after")
    def validate_crrem_requires_rating(self) -> "EnergyBenchmarkConfig":
        """CRREM pathway in portfolio requires CRREM in rating systems."""
        if self.portfolio.crrem_pathway_enabled:
            if RatingSystem.CRREM not in self.rating.rating_systems:
                logger.info(
                    "CRREM pathway enabled in portfolio; adding CRREM to rating systems."
                )
                self.rating.rating_systems.append(RatingSystem.CRREM)
        return self

    @model_validator(mode="after")
    def validate_energy_star_score_requires_source(self) -> "EnergyBenchmarkConfig":
        """ENERGY STAR score requires ENERGY STAR in benchmark sources."""
        if self.peer_comparison.include_energy_star_score:
            if BenchmarkSource.ENERGY_STAR not in self.peer_comparison.benchmark_sources:
                logger.info(
                    "ENERGY STAR score enabled; adding ENERGY_STAR to benchmark sources."
                )
                self.peer_comparison.benchmark_sources.append(BenchmarkSource.ENERGY_STAR)
        return self

    @model_validator(mode="after")
    def validate_parking_minimal_benchmarks(self) -> "EnergyBenchmarkConfig":
        """Parking structures have minimal HVAC and no heating/cooling benchmarks."""
        if self.building_type == BuildingType.PARKING:
            if self.weather.enabled:
                logger.info(
                    "Parking building type: weather normalisation has limited value. "
                    "Consider disabling for simplicity."
                )
        return self


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """Top-level pack configuration wrapper.

    Handles preset loading, environment variable overrides, and
    configuration merging. Follows the standard GreenLang pack config
    pattern with from_preset(), from_yaml(), and merge() support.
    """

    config: EnergyBenchmarkConfig = Field(
        default_factory=EnergyBenchmarkConfig,
        description="Main Energy Benchmark configuration",
    )
    preset_name: Optional[str] = Field(
        None,
        description="Name of the loaded preset",
    )
    config_version: str = Field(
        "1.0.0",
        description="Configuration schema version",
    )
    pack_id: str = Field(
        "PACK-035-energy-benchmark",
        description="Pack identifier",
    )

    @classmethod
    def from_preset(
        cls, preset_name: str, overrides: Optional[Dict[str, Any]] = None
    ) -> "PackConfig":
        """Load configuration from a named preset.

        Args:
            preset_name: Name of the preset (commercial_office, data_center, etc.)
            overrides: Optional dictionary of configuration overrides.

        Returns:
            PackConfig instance with preset values applied.

        Raises:
            FileNotFoundError: If preset YAML file does not exist.
            ValueError: If preset_name is not in AVAILABLE_PRESETS.
        """
        if preset_name not in AVAILABLE_PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. "
                f"Available presets: {sorted(AVAILABLE_PRESETS.keys())}"
            )

        preset_path = CONFIG_DIR / "presets" / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(
                f"Preset file not found: {preset_path}. "
                f"Run setup wizard to generate presets."
            )

        with open(preset_path, "r", encoding="utf-8") as f:
            preset_data = yaml.safe_load(f) or {}

        # Apply environment variable overrides
        env_overrides = cls._load_env_overrides()
        if env_overrides:
            preset_data = cls._deep_merge(preset_data, env_overrides)

        # Apply explicit overrides
        if overrides:
            preset_data = cls._deep_merge(preset_data, overrides)

        benchmark_config = EnergyBenchmarkConfig(**preset_data)
        return cls(config=benchmark_config, preset_name=preset_name)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "PackConfig":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            PackConfig instance with YAML values applied.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        benchmark_config = EnergyBenchmarkConfig(**config_data)
        return cls(config=benchmark_config)

    @classmethod
    def merge(
        cls,
        base: "PackConfig",
        overrides: Dict[str, Any],
    ) -> "PackConfig":
        """Create a new PackConfig by merging overrides into an existing config.

        Args:
            base: Base PackConfig instance.
            overrides: Dictionary of configuration overrides.

        Returns:
            New PackConfig with merged values.
        """
        base_dict = base.config.model_dump()
        merged = cls._deep_merge(base_dict, overrides)
        benchmark_config = EnergyBenchmarkConfig(**merged)
        return cls(
            config=benchmark_config,
            preset_name=base.preset_name,
            config_version=base.config_version,
        )

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """Load configuration overrides from environment variables.

        Environment variables prefixed with ENERGY_BENCHMARK_PACK_ are loaded
        and mapped to configuration keys. Nested keys use double underscore.

        Example: ENERGY_BENCHMARK_PACK_WEATHER__ENABLED=true
                 ENERGY_BENCHMARK_PACK_EUI__ROLLING_MONTHS=24
        """
        overrides: Dict[str, Any] = {}
        prefix = "ENERGY_BENCHMARK_PACK_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                parts = config_key.split("__")
                current = overrides
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                # Parse value type
                if value.lower() in ("true", "yes", "1"):
                    current[parts[-1]] = True
                elif value.lower() in ("false", "no", "0"):
                    current[parts[-1]] = False
                else:
                    try:
                        current[parts[-1]] = int(value)
                    except ValueError:
                        try:
                            current[parts[-1]] = float(value)
                        except ValueError:
                            current[parts[-1]] = value
        return overrides

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence.

        Args:
            base: Base dictionary.
            override: Override dictionary (values take precedence).

        Returns:
            Merged dictionary.
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = PackConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get_config_hash(self) -> str:
        """Generate SHA-256 hash of the current configuration for provenance.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        config_json = self.model_dump_json(indent=None)
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()

    def validate_completeness(self) -> List[str]:
        """Validate configuration completeness and return warnings.

        Returns:
            List of warning messages (empty if fully valid).
        """
        return validate_config(self.config)


# =============================================================================
# Utility Functions
# =============================================================================


def load_preset(
    preset_name: str, overrides: Optional[Dict[str, Any]] = None
) -> PackConfig:
    """Load a named preset configuration.

    Convenience wrapper around PackConfig.from_preset().

    Args:
        preset_name: Name of the preset to load.
        overrides: Optional configuration overrides.

    Returns:
        PackConfig instance with preset applied.
    """
    return PackConfig.from_preset(preset_name, overrides)


def validate_config(config: EnergyBenchmarkConfig) -> List[str]:
    """Validate an energy benchmark configuration and return any warnings.

    Args:
        config: EnergyBenchmarkConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check facility identification
    if not config.facility_name:
        warnings.append(
            "No facility_name configured. Add a facility name for report identification."
        )

    # Check floor area
    if config.floor_area_m2 is None:
        warnings.append(
            "No floor_area_m2 configured. EUI calculation requires floor area."
        )

    # Check building type has applicable benchmark sources
    building_info = BUILDING_TYPE_DEFAULTS.get(config.building_type.value, {})
    applicable = building_info.get("applicable_sources", [])
    configured_sources = [s.value for s in config.peer_comparison.benchmark_sources]
    mismatched = [s for s in configured_sources if s not in applicable and s != "CUSTOM"]
    if mismatched:
        warnings.append(
            f"Benchmark sources {mismatched} may not have data for building type "
            f"{config.building_type.value}. Applicable sources: {applicable}."
        )

    # Check data center specific
    if config.building_type == BuildingType.DATA_CENTER:
        if config.weather.enabled and config.weather.base_temp_heating_c > 12.0:
            warnings.append(
                "Data centres are cooling-dominated. Consider reducing heating "
                "base temperature or disabling heating degree-day normalisation."
            )

    # Check portfolio mode
    if config.portfolio.enabled and not config.facility_id:
        warnings.append(
            "Portfolio mode enabled but no facility_id set. Each facility in a "
            "portfolio must have a unique facility_id."
        )

    # Check weather normalisation data
    if config.weather.enabled and config.latitude is None:
        warnings.append(
            "Weather normalisation enabled but no latitude/longitude configured. "
            "Provide coordinates or set a weather_station_id for accurate degree days."
        )

    # Check EUI carriers match available carriers
    eui_carriers = set(c.value for c in config.eui.energy_carriers)
    if len(eui_carriers) == 0:
        warnings.append(
            "No energy carriers configured in EUI section. "
            "At least ELECTRICITY should be included."
        )

    # Check rating system applicability
    for rs in config.rating.rating_systems:
        if rs == RatingSystem.ENERGY_STAR and config.country_code not in ("US", "CA"):
            warnings.append(
                f"ENERGY STAR rating is US/Canada-specific. Country {config.country_code} "
                "will use a proxy score based on the ENERGY STAR regression model."
            )
        if rs == RatingSystem.NABERS_AU and config.country_code != "AU":
            warnings.append(
                f"NABERS rating is Australia-specific. Country {config.country_code} "
                "is not supported for official NABERS rating."
            )
        if rs == RatingSystem.DEC_UK and config.country_code not in ("GB", "UK"):
            warnings.append(
                f"DEC rating is UK-specific. Country {config.country_code} "
                "is not supported for official DEC rating."
            )

    # Check CRREM pathway
    if config.portfolio.crrem_pathway_enabled:
        if RatingSystem.CRREM not in config.rating.rating_systems:
            warnings.append(
                "CRREM pathway enabled in portfolio but CRREM not in rating systems. "
                "CRREM will be auto-added."
            )

    # Check trend monitoring makes sense
    if config.trend.enabled and config.eui.data_frequency == DataFrequency.ANNUAL:
        warnings.append(
            "Trend monitoring is enabled but data frequency is ANNUAL. "
            "Monthly or finer data is recommended for meaningful trend analysis."
        )

    # Check alert channels have recipients
    if config.trend.alert_enabled and len(config.trend.alert_recipients) == 0:
        warnings.append(
            "Alerts are enabled but no alert_recipients configured. "
            "Add email addresses or webhook URLs."
        )

    # Check MEES compliance relevance
    if config.rating.include_regulatory_compliance:
        if config.country_code == "GB" and RatingSystem.EPC_EU not in config.rating.rating_systems:
            warnings.append(
                "MEES compliance requires EPC rating for UK properties. "
                "Add EPC_EU to rating systems."
            )

    return warnings


def get_building_type_info(
    building_type: Union[str, BuildingType],
) -> Dict[str, Any]:
    """Get detailed information about a building type.

    Args:
        building_type: Building type enum or string value.

    Returns:
        Dictionary with name, EUI ranges, end uses, and applicable sources.
    """
    key = building_type.value if isinstance(building_type, BuildingType) else building_type
    return BUILDING_TYPE_DEFAULTS.get(
        key,
        {
            "name": key,
            "typical_eui_range_kwh_m2": (100, 500),
            "good_practice_eui_kwh_m2": 150,
            "typical_practice_eui_kwh_m2": 350,
            "dominant_end_uses": ["HVAC", "Lighting"],
            "applicable_sources": ["BPIE"],
            "heating_benchmark_applicable": True,
            "cooling_benchmark_applicable": True,
        },
    )


def get_climate_zone_adjustment(
    climate_zone: Union[str, ClimateZone],
) -> Dict[str, float]:
    """Get heating and cooling adjustment factors for a climate zone.

    Args:
        climate_zone: Climate zone enum or string value.

    Returns:
        Dictionary with heating_factor, cooling_factor, and description.
    """
    key = climate_zone.value if isinstance(climate_zone, ClimateZone) else climate_zone
    return CLIMATE_ZONE_ADJUSTMENTS.get(
        key,
        {"heating_factor": 1.00, "cooling_factor": 1.00, "description": "Unknown zone"},
    )


def get_source_energy_factor(carrier: Union[str, EnergyCarrier]) -> float:
    """Get source energy conversion factor for an energy carrier.

    Args:
        carrier: Energy carrier enum or string value.

    Returns:
        Site-to-source energy factor (dimensionless).
    """
    key = carrier.value if isinstance(carrier, EnergyCarrier) else carrier
    return SOURCE_ENERGY_FACTORS.get(key, 1.00)


def get_primary_energy_factor(carrier: Union[str, EnergyCarrier]) -> float:
    """Get primary energy factor for an energy carrier per EN 15603.

    Args:
        carrier: Energy carrier enum or string value.

    Returns:
        Non-renewable primary energy factor (dimensionless).
    """
    key = carrier.value if isinstance(carrier, EnergyCarrier) else carrier
    return PRIMARY_ENERGY_FACTORS.get(key, 1.00)


def get_epc_grade(
    primary_energy_kwh_m2: float,
    building_type: Union[str, BuildingType] = "DEFAULT",
) -> str:
    """Determine EPC grade from primary energy intensity.

    Args:
        primary_energy_kwh_m2: Primary energy in kWh/m2/yr.
        building_type: Building type for grade thresholds.

    Returns:
        EPC grade letter (A through G).
    """
    key = building_type.value if isinstance(building_type, BuildingType) else building_type
    thresholds = EPC_GRADE_THRESHOLDS.get(key, EPC_GRADE_THRESHOLDS["DEFAULT"])
    for grade in ["A", "B", "C", "D", "E", "F", "G"]:
        if primary_energy_kwh_m2 <= thresholds[grade]:
            return grade
    return "G"


def get_cibse_tm46_benchmark(
    category: str,
) -> Optional[Dict[str, float]]:
    """Get CIBSE TM46 benchmark for a building category.

    Args:
        category: CIBSE TM46 building category name.

    Returns:
        Dictionary with fossil_thermal_kwh_m2 and electricity_kwh_m2,
        or None if category not found.
    """
    return CIBSE_TM46_BENCHMARKS.get(category)


def get_default_config(
    building_type: BuildingType = BuildingType.OFFICE,
) -> EnergyBenchmarkConfig:
    """Get default configuration for a given building type.

    Args:
        building_type: Building type to configure for.

    Returns:
        EnergyBenchmarkConfig instance with building-type-appropriate defaults.
    """
    return EnergyBenchmarkConfig(building_type=building_type)


def list_available_presets() -> Dict[str, str]:
    """List all available configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return AVAILABLE_PRESETS.copy()
