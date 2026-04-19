"""
PACK-032 Building Energy Assessment Pack - Configuration Manager

This module implements the BuildingEnergyAssessmentConfig and PackConfig classes
that load, merge, and validate all configuration for the Building Energy Assessment
Pack. It provides comprehensive Pydantic v2 models for every aspect of building
energy assessment: envelope thermal performance, HVAC systems, lighting, domestic
hot water, renewables, indoor environment quality, retrofit planning, benchmarking,
whole-life carbon, and regulatory compliance across European frameworks.

Building Types:
    - OFFICE: Commercial office buildings (cellular, open-plan, serviced)
    - RETAIL: Retail premises (high street, shopping centre, supermarket)
    - HOTEL: Hotels and serviced accommodation (24h operation)
    - HOSPITAL: Healthcare facilities (acute, outpatient, care homes)
    - EDUCATION_PRIMARY: Primary schools
    - EDUCATION_SECONDARY: Secondary schools
    - EDUCATION_UNIVERSITY: Higher education and university buildings
    - WAREHOUSE: Storage, logistics, and distribution centres
    - DATA_CENTER: Co-located data centres within commercial buildings
    - RESTAURANT: Restaurants, cafes, and commercial kitchens
    - LEISURE_CENTRE: Sports and leisure facilities (pools, gyms)
    - LIBRARY: Libraries and community buildings
    - RESIDENTIAL_APARTMENT: Multi-family residential apartment blocks
    - RESIDENTIAL_HOUSE: Individual dwellings and terraced housing
    - MIXED_USE: Mixed-use developments (retail ground, office/resi upper)
    - PUBLIC_SECTOR: Government, municipal, and civic buildings

Assessment Levels:
    - WALK_THROUGH: Rapid visual survey and quick energy check
    - STANDARD: Standard assessment with meter data and basic modelling
    - DETAILED: Detailed assessment with sub-metering, monitoring, and modelling
    - INVESTMENT_GRADE: Full ASHRAE Level III / RICS TDD-grade assessment

Climate Zones:
    - NORTHERN_EUROPE: Scandinavia, Baltics (HDD-dominated, >4000 HDD18)
    - CENTRAL_MARITIME: UK, Ireland, Benelux, NW France (moderate, 2500-4000 HDD18)
    - CENTRAL_CONTINENTAL: Germany, Poland, Czech, Austria (cold winters, 3000-4500 HDD18)
    - SOUTHERN_MARITIME: Atlantic coast Spain/Portugal (mild winters, 1500-2500 HDD18)
    - SOUTHERN_CONTINENTAL: Northern Italy, Hungary, Romania (mixed, 2000-3500 HDD18)
    - MEDITERRANEAN: Southern Spain, S. Italy, Greece, Cyprus (<1500 HDD18, cooling-dominated)

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (commercial_office / retail_building / hotel_hospitality /
       healthcare_facility / education_building / residential_multifamily /
       mixed_use_development / public_sector_building)
    3. Environment overrides (BUILDING_ENERGY_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - EPBD: Energy Performance of Buildings Directive 2024/1275 (recast)
    - EED: EU Energy Efficiency Directive 2023/1791 (Article 6 public bodies)
    - EN 15603 / ISO 52000: Building energy performance calculation
    - EN ISO 13790 / EN ISO 52016: Energy needs for heating and cooling
    - EN 15459: Economic evaluation of energy systems in buildings
    - EN 16798-1: Indoor environmental parameters for building design
    - SAP/RdSAP: UK Standard Assessment Procedure (domestic)
    - SBEM: UK Simplified Building Energy Model (non-domestic)
    - LEED v4.1: Leadership in Energy and Environmental Design
    - BREEAM: Building Research Establishment Environmental Assessment Method
    - ENERGY STAR: US EPA commercial building benchmark

Example:
    >>> config = PackConfig.from_preset("commercial_office")
    >>> print(config.pack.building_type)
    BuildingType.OFFICE
    >>> print(config.pack.envelope.wall_u_target)
    0.26
    >>> print(config.pack.hvac.target_heating_cop)
    3.5
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

logger = logging.getLogger(__name__)

# Base directory for all pack configuration files
PACK_BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent


# =============================================================================
# Enums - Building energy assessment enumeration types
# =============================================================================


class BuildingType(str, Enum):
    """Building type classification for assessment scoping and benchmarking."""

    OFFICE = "OFFICE"
    RETAIL = "RETAIL"
    HOTEL = "HOTEL"
    HOSPITAL = "HOSPITAL"
    EDUCATION_PRIMARY = "EDUCATION_PRIMARY"
    EDUCATION_SECONDARY = "EDUCATION_SECONDARY"
    EDUCATION_UNIVERSITY = "EDUCATION_UNIVERSITY"
    WAREHOUSE = "WAREHOUSE"
    DATA_CENTER = "DATA_CENTER"
    RESTAURANT = "RESTAURANT"
    LEISURE_CENTRE = "LEISURE_CENTRE"
    LIBRARY = "LIBRARY"
    RESIDENTIAL_APARTMENT = "RESIDENTIAL_APARTMENT"
    RESIDENTIAL_HOUSE = "RESIDENTIAL_HOUSE"
    MIXED_USE = "MIXED_USE"
    PUBLIC_SECTOR = "PUBLIC_SECTOR"


class BuildingAge(str, Enum):
    """Building age band classification per EPBD and national standards.

    Age bands determine default U-values, construction quality assumptions,
    and typical retrofit scope. Aligned with UK/DE/FR national building
    stock categorisation.
    """

    PRE_1919 = "PRE_1919"
    BAND_1919_1944 = "1919_1944"
    BAND_1945_1964 = "1945_1964"
    BAND_1965_1982 = "1965_1982"
    BAND_1983_1995 = "1983_1995"
    BAND_1996_2006 = "1996_2006"
    BAND_2007_2013 = "2007_2013"
    BAND_2014_PRESENT = "2014_PRESENT"


class ClimateZone(str, Enum):
    """European climate zone classification for heating/cooling design loads."""

    NORTHERN_EUROPE = "NORTHERN_EUROPE"
    CENTRAL_MARITIME = "CENTRAL_MARITIME"
    CENTRAL_CONTINENTAL = "CENTRAL_CONTINENTAL"
    SOUTHERN_MARITIME = "SOUTHERN_MARITIME"
    SOUTHERN_CONTINENTAL = "SOUTHERN_CONTINENTAL"
    MEDITERRANEAN = "MEDITERRANEAN"


class AssessmentLevel(str, Enum):
    """Building energy assessment depth level."""

    WALK_THROUGH = "WALK_THROUGH"
    STANDARD = "STANDARD"
    DETAILED = "DETAILED"
    INVESTMENT_GRADE = "INVESTMENT_GRADE"


class CertificationTarget(str, Enum):
    """Green building certification target."""

    NONE = "NONE"
    LEED_CERTIFIED = "LEED_CERTIFIED"
    LEED_SILVER = "LEED_SILVER"
    LEED_GOLD = "LEED_GOLD"
    LEED_PLATINUM = "LEED_PLATINUM"
    BREEAM_PASS = "BREEAM_PASS"
    BREEAM_GOOD = "BREEAM_GOOD"
    BREEAM_VERY_GOOD = "BREEAM_VERY_GOOD"
    BREEAM_EXCELLENT = "BREEAM_EXCELLENT"
    BREEAM_OUTSTANDING = "BREEAM_OUTSTANDING"
    ENERGY_STAR = "ENERGY_STAR"


class OccupancyPattern(str, Enum):
    """Building occupancy schedule pattern."""

    SINGLE_SHIFT = "SINGLE_SHIFT"
    DOUBLE_SHIFT = "DOUBLE_SHIFT"
    CONTINUOUS = "CONTINUOUS"
    SCHOOL_HOURS = "SCHOOL_HOURS"
    RETAIL_HOURS = "RETAIL_HOURS"
    HOTEL_24H = "HOTEL_24H"


class HeatingFuel(str, Enum):
    """Primary heating fuel type."""

    NATURAL_GAS = "NATURAL_GAS"
    ELECTRICITY = "ELECTRICITY"
    OIL = "OIL"
    LPG = "LPG"
    BIOMASS = "BIOMASS"
    DISTRICT_HEATING = "DISTRICT_HEATING"
    HEAT_PUMP = "HEAT_PUMP"


class OwnershipType(str, Enum):
    """Building ownership and tenure classification."""

    OWNER_OCCUPIED = "OWNER_OCCUPIED"
    SINGLE_TENANT = "SINGLE_TENANT"
    MULTI_TENANT = "MULTI_TENANT"
    PUBLIC_SECTOR = "PUBLIC_SECTOR"


class EPCMethodology(str, Enum):
    """Energy Performance Certificate calculation methodology by country."""

    SAP = "SAP"          # UK domestic (Standard Assessment Procedure)
    RDSAP = "RDSAP"      # UK domestic reduced data (existing dwellings)
    GEG = "GEG"          # Germany (Gebaeudeenergiegesetz)
    DPE = "DPE"          # France (Diagnostic de Performance Energetique)
    APE = "APE"          # Italy (Attestato di Prestazione Energetica)
    SBEM = "SBEM"        # UK non-domestic (Simplified Building Energy Model)
    EPBD_GENERIC = "EPBD_GENERIC"  # Generic EPBD-compliant methodology


class RetrofitAmbition(str, Enum):
    """Retrofit depth and ambition level for energy improvement."""

    MINIMUM_COMPLIANCE = "MINIMUM_COMPLIANCE"
    COST_OPTIMAL = "COST_OPTIMAL"
    LOW_ENERGY = "LOW_ENERGY"
    NZEB = "NZEB"
    NET_ZERO = "NET_ZERO"


class OutputFormat(str, Enum):
    """Output format for assessment reports."""

    PDF = "PDF"
    XLSX = "XLSX"
    JSON = "JSON"
    HTML = "HTML"


class ThermalBridgeMethod(str, Enum):
    """Thermal bridge calculation method per EN ISO 10211 / EN ISO 14683."""

    DEFAULT_UPLIFT = "DEFAULT_UPLIFT"      # Flat percentage uplift (e.g. +15%)
    TABULATED_PSI = "TABULATED_PSI"        # Tabulated psi-values per detail type
    CALCULATED_PSI = "CALCULATED_PSI"      # 2D/3D numerical calculation
    ACCREDITED_DETAILS = "ACCREDITED_DETAILS"  # Accredited construction details


class VentilationType(str, Enum):
    """Building ventilation strategy type."""

    NATURAL = "NATURAL"
    MECHANICAL_EXTRACT = "MECHANICAL_EXTRACT"
    MECHANICAL_SUPPLY_EXTRACT = "MECHANICAL_SUPPLY_EXTRACT"
    MVHR = "MVHR"  # Mechanical Ventilation with Heat Recovery
    MIXED_MODE = "MIXED_MODE"


class LightingControlType(str, Enum):
    """Lighting control strategy."""

    MANUAL = "MANUAL"
    OCCUPANCY_SENSING = "OCCUPANCY_SENSING"
    DAYLIGHT_DIMMING = "DAYLIGHT_DIMMING"
    COMBINED_OCCUPANCY_DAYLIGHT = "COMBINED_OCCUPANCY_DAYLIGHT"
    DALI_ADDRESSABLE = "DALI_ADDRESSABLE"
    SMART_SCHEDULING = "SMART_SCHEDULING"


class ThermalComfortMethod(str, Enum):
    """Thermal comfort assessment methodology."""

    PMV_PPD = "PMV_PPD"          # EN ISO 7730 / EN 16798-1 Category
    ADAPTIVE = "ADAPTIVE"        # EN 16798-1 Adaptive comfort model
    CIBSE_GUIDE_A = "CIBSE_GUIDE_A"  # CIBSE overheating criteria
    ASHRAE_55 = "ASHRAE_55"      # ASHRAE Standard 55


class IEQCategory(str, Enum):
    """Indoor Environmental Quality category per EN 16798-1."""

    CATEGORY_I = "CATEGORY_I"    # High expectations (sensitive/vulnerable)
    CATEGORY_II = "CATEGORY_II"  # Normal expectations (new/renovated)
    CATEGORY_III = "CATEGORY_III"  # Acceptable moderate expectations
    CATEGORY_IV = "CATEGORY_IV"  # Relaxed (only limited periods)


class BMSProtocol(str, Enum):
    """Building Management System communication protocol."""

    BACNET = "BACNET"
    MODBUS = "MODBUS"
    LONWORKS = "LONWORKS"
    KNX = "KNX"
    OPC_UA = "OPC_UA"
    MQTT = "MQTT"


# =============================================================================
# Reference Data Constants
# =============================================================================


# Building type display names, typical characteristics, and benchmarks
BUILDING_TYPE_INFO: Dict[str, Dict[str, Any]] = {
    "OFFICE": {
        "name": "Commercial Office",
        "typical_gia_m2": "1000-50000",
        "typical_eui_kwh_m2_yr": "150-300",
        "good_practice_eui_kwh_m2_yr": 120,
        "best_practice_eui_kwh_m2_yr": 85,
        "key_systems": ["HVAC", "Lighting", "IT Equipment", "DHW", "Lifts"],
        "typical_energy_split": "Electricity 55-70%, Gas 25-40%, Other 5-10%",
        "dec_benchmark": "CIBSE TM46 Type 1",
    },
    "RETAIL": {
        "name": "Retail Building",
        "typical_gia_m2": "200-20000",
        "typical_eui_kwh_m2_yr": "200-400",
        "good_practice_eui_kwh_m2_yr": 165,
        "best_practice_eui_kwh_m2_yr": 120,
        "key_systems": ["Lighting", "HVAC", "Refrigeration", "Display", "DHW"],
        "typical_energy_split": "Electricity 65-85%, Gas 10-25%, Other 5-10%",
        "dec_benchmark": "CIBSE TM46 Type 2",
    },
    "HOTEL": {
        "name": "Hotel / Hospitality",
        "typical_gia_m2": "1000-30000",
        "typical_eui_kwh_m2_yr": "250-450",
        "good_practice_eui_kwh_m2_yr": 200,
        "best_practice_eui_kwh_m2_yr": 150,
        "key_systems": ["HVAC", "DHW", "Lighting", "Kitchen", "Laundry", "Lifts"],
        "typical_energy_split": "Gas 40-55%, Electricity 35-50%, Other 5-10%",
        "dec_benchmark": "CIBSE TM46 Type 5",
    },
    "HOSPITAL": {
        "name": "Healthcare Facility",
        "typical_gia_m2": "5000-100000",
        "typical_eui_kwh_m2_yr": "300-600",
        "good_practice_eui_kwh_m2_yr": 280,
        "best_practice_eui_kwh_m2_yr": 200,
        "key_systems": ["HVAC", "Ventilation", "Medical Equipment", "DHW", "Lighting", "Sterilisation"],
        "typical_energy_split": "Gas 45-60%, Electricity 35-50%, Steam 5-10%",
        "dec_benchmark": "CIBSE TM46 Type 6",
    },
    "EDUCATION_PRIMARY": {
        "name": "Primary School",
        "typical_gia_m2": "500-5000",
        "typical_eui_kwh_m2_yr": "100-200",
        "good_practice_eui_kwh_m2_yr": 95,
        "best_practice_eui_kwh_m2_yr": 65,
        "key_systems": ["Heating", "Lighting", "Ventilation", "DHW", "Kitchen"],
        "typical_energy_split": "Gas 55-70%, Electricity 25-40%, Other 5%",
        "dec_benchmark": "CIBSE TM46 Type 14",
    },
    "EDUCATION_SECONDARY": {
        "name": "Secondary School",
        "typical_gia_m2": "2000-15000",
        "typical_eui_kwh_m2_yr": "100-250",
        "good_practice_eui_kwh_m2_yr": 105,
        "best_practice_eui_kwh_m2_yr": 70,
        "key_systems": ["Heating", "Lighting", "Ventilation", "DHW", "Kitchen", "IT"],
        "typical_energy_split": "Gas 50-65%, Electricity 30-45%, Other 5%",
        "dec_benchmark": "CIBSE TM46 Type 14",
    },
    "EDUCATION_UNIVERSITY": {
        "name": "University Building",
        "typical_gia_m2": "2000-50000",
        "typical_eui_kwh_m2_yr": "150-350",
        "good_practice_eui_kwh_m2_yr": 140,
        "best_practice_eui_kwh_m2_yr": 95,
        "key_systems": ["HVAC", "Lighting", "IT/Labs", "Ventilation", "DHW"],
        "typical_energy_split": "Gas 40-55%, Electricity 40-55%, Other 5-10%",
        "dec_benchmark": "CIBSE TM46 Type 15",
    },
    "WAREHOUSE": {
        "name": "Warehouse / Distribution Centre",
        "typical_gia_m2": "1000-50000",
        "typical_eui_kwh_m2_yr": "30-120",
        "good_practice_eui_kwh_m2_yr": 40,
        "best_practice_eui_kwh_m2_yr": 25,
        "key_systems": ["Lighting", "Heating", "Dock Doors", "MHE Charging", "Refrigeration"],
        "typical_energy_split": "Electricity 50-70%, Gas 20-40%, Other 5-10%",
        "dec_benchmark": "CIBSE TM46 Type 10",
    },
    "DATA_CENTER": {
        "name": "Data Centre (Commercial Building)",
        "typical_gia_m2": "500-10000",
        "typical_eui_kwh_m2_yr": "500-2000",
        "good_practice_eui_kwh_m2_yr": 500,
        "best_practice_eui_kwh_m2_yr": 350,
        "key_systems": ["IT Load", "Cooling", "UPS", "Power Distribution", "Lighting"],
        "typical_energy_split": "Electricity 95-100%",
        "dec_benchmark": "N/A (PUE metric)",
    },
    "RESTAURANT": {
        "name": "Restaurant / Commercial Kitchen",
        "typical_gia_m2": "100-2000",
        "typical_eui_kwh_m2_yr": "300-700",
        "good_practice_eui_kwh_m2_yr": 290,
        "best_practice_eui_kwh_m2_yr": 200,
        "key_systems": ["Kitchen Equipment", "HVAC", "DHW", "Refrigeration", "Lighting"],
        "typical_energy_split": "Gas 40-60%, Electricity 35-55%, Other 5%",
        "dec_benchmark": "CIBSE TM46 Type 24",
    },
    "LEISURE_CENTRE": {
        "name": "Leisure Centre / Sports Facility",
        "typical_gia_m2": "1000-15000",
        "typical_eui_kwh_m2_yr": "200-500",
        "good_practice_eui_kwh_m2_yr": 210,
        "best_practice_eui_kwh_m2_yr": 150,
        "key_systems": ["Pool Heating", "HVAC", "Lighting", "DHW", "Ventilation"],
        "typical_energy_split": "Gas 50-70%, Electricity 25-45%, Other 5%",
        "dec_benchmark": "CIBSE TM46 Type 9",
    },
    "LIBRARY": {
        "name": "Library / Community Building",
        "typical_gia_m2": "300-5000",
        "typical_eui_kwh_m2_yr": "100-250",
        "good_practice_eui_kwh_m2_yr": 100,
        "best_practice_eui_kwh_m2_yr": 70,
        "key_systems": ["Heating", "Lighting", "Ventilation", "IT", "DHW"],
        "typical_energy_split": "Gas 45-60%, Electricity 35-50%, Other 5%",
        "dec_benchmark": "CIBSE TM46 Type 17",
    },
    "RESIDENTIAL_APARTMENT": {
        "name": "Residential Apartment Block",
        "typical_gia_m2": "500-10000",
        "typical_eui_kwh_m2_yr": "80-200",
        "good_practice_eui_kwh_m2_yr": 60,
        "best_practice_eui_kwh_m2_yr": 35,
        "key_systems": ["Heating", "DHW", "Lighting", "Ventilation", "Common Areas"],
        "typical_energy_split": "Gas 55-75%, Electricity 20-35%, Other 5-10%",
        "dec_benchmark": "SAP/RdSAP",
    },
    "RESIDENTIAL_HOUSE": {
        "name": "Residential Dwelling",
        "typical_gia_m2": "50-300",
        "typical_eui_kwh_m2_yr": "80-300",
        "good_practice_eui_kwh_m2_yr": 65,
        "best_practice_eui_kwh_m2_yr": 30,
        "key_systems": ["Heating", "DHW", "Lighting", "Ventilation"],
        "typical_energy_split": "Gas 60-80%, Electricity 15-30%, Other 5%",
        "dec_benchmark": "SAP/RdSAP",
    },
    "MIXED_USE": {
        "name": "Mixed-Use Development",
        "typical_gia_m2": "2000-50000",
        "typical_eui_kwh_m2_yr": "150-350",
        "good_practice_eui_kwh_m2_yr": 130,
        "best_practice_eui_kwh_m2_yr": 90,
        "key_systems": ["HVAC", "Lighting", "DHW", "Lifts", "Common Areas", "Retail Systems"],
        "typical_energy_split": "Gas 35-50%, Electricity 40-55%, Other 5-10%",
        "dec_benchmark": "Multiple TM46 types",
    },
    "PUBLIC_SECTOR": {
        "name": "Public Sector / Government Building",
        "typical_gia_m2": "500-20000",
        "typical_eui_kwh_m2_yr": "150-350",
        "good_practice_eui_kwh_m2_yr": 130,
        "best_practice_eui_kwh_m2_yr": 90,
        "key_systems": ["HVAC", "Lighting", "IT Equipment", "DHW", "Lifts"],
        "typical_energy_split": "Gas 40-55%, Electricity 35-50%, Other 5-10%",
        "dec_benchmark": "CIBSE TM46 Type 1",
    },
}

# Available presets
AVAILABLE_PRESETS: Dict[str, str] = {
    "commercial_office": "Commercial office buildings (cellular, open-plan, serviced)",
    "retail_building": "Retail premises (high street, shopping centre, supermarket)",
    "hotel_hospitality": "Hotels and serviced accommodation (24h operation)",
    "healthcare_facility": "Healthcare facilities (hospitals, outpatient, care homes)",
    "education_building": "Schools and university buildings (term-time occupancy)",
    "residential_multifamily": "Multi-family residential apartment blocks",
    "mixed_use_development": "Mixed-use developments (retail, office, residential)",
    "public_sector_building": "Government and municipal buildings (DEC required)",
}

# U-value targets by climate zone (W/m2K) per EPBD cost-optimal / NZEB levels
U_VALUE_TARGETS: Dict[str, Dict[str, float]] = {
    "NORTHERN_EUROPE": {
        "wall": 0.18,
        "roof": 0.13,
        "floor": 0.15,
        "window": 1.0,
    },
    "CENTRAL_MARITIME": {
        "wall": 0.26,
        "roof": 0.18,
        "floor": 0.22,
        "window": 1.4,
    },
    "CENTRAL_CONTINENTAL": {
        "wall": 0.22,
        "roof": 0.15,
        "floor": 0.20,
        "window": 1.2,
    },
    "SOUTHERN_MARITIME": {
        "wall": 0.35,
        "roof": 0.30,
        "floor": 0.35,
        "window": 2.0,
    },
    "SOUTHERN_CONTINENTAL": {
        "wall": 0.30,
        "roof": 0.25,
        "floor": 0.30,
        "window": 1.8,
    },
    "MEDITERRANEAN": {
        "wall": 0.45,
        "roof": 0.35,
        "floor": 0.45,
        "window": 2.5,
    },
}

# EPC rating bands (kWh/m2/yr thresholds for non-domestic, indicative)
EPC_RATING_THRESHOLDS: Dict[str, float] = {
    "A+": 0,
    "A": 25,
    "B": 50,
    "C": 100,
    "D": 150,
    "E": 200,
    "F": 250,
    "G": 300,
}

# Lighting Power Density standards (W/m2) per EN 15193/12464 for building types
LPD_BUILDING_STANDARDS: Dict[str, float] = {
    "office_cellular": 8.0,
    "office_open_plan": 10.0,
    "retail_general": 15.0,
    "retail_supermarket": 12.0,
    "hotel_corridor": 5.0,
    "hotel_room": 7.0,
    "hotel_lobby": 10.0,
    "hospital_ward": 6.0,
    "hospital_theatre": 20.0,
    "education_classroom": 8.0,
    "education_laboratory": 12.0,
    "warehouse_general": 6.0,
    "warehouse_high_bay": 8.0,
    "restaurant": 10.0,
    "leisure_sports_hall": 12.0,
    "library_reading": 8.0,
    "residential_living": 6.0,
    "circulation_corridor": 4.0,
    "circulation_stairwell": 3.0,
}

# BREEAM credit thresholds for energy (Ene 01)
BREEAM_ENERGY_CREDITS: Dict[str, Dict[str, Any]] = {
    "PASS": {"epr_ratio_max": 1.0, "credits": 0},
    "GOOD": {"epr_ratio_max": 0.85, "credits": 4},
    "VERY_GOOD": {"epr_ratio_max": 0.65, "credits": 8},
    "EXCELLENT": {"epr_ratio_max": 0.45, "credits": 12},
    "OUTSTANDING": {"epr_ratio_max": 0.25, "credits": 15},
}

# LEED EAc2 energy performance thresholds (% improvement over ASHRAE 90.1)
LEED_ENERGY_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "CERTIFIED": {"improvement_pct_min": 5, "points": 1},
    "SILVER": {"improvement_pct_min": 10, "points": 4},
    "GOLD": {"improvement_pct_min": 25, "points": 10},
    "PLATINUM": {"improvement_pct_min": 50, "points": 18},
}

# Airtightness benchmarks (m3/h/m2 @ 50Pa) per building type
AIRTIGHTNESS_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "new_build_residential": {"best_practice": 1.0, "regulation": 5.0},
    "new_build_non_domestic": {"best_practice": 2.0, "regulation": 5.0},
    "passivhaus": {"best_practice": 0.6, "regulation": 0.6},
    "existing_pre_1980": {"typical": 15.0, "good_retrofit": 5.0},
    "existing_post_2000": {"typical": 7.0, "good_retrofit": 3.0},
}

# Country to climate zone mapping (simplified)
COUNTRY_CLIMATE_ZONE: Dict[str, str] = {
    "SE": "NORTHERN_EUROPE",
    "NO": "NORTHERN_EUROPE",
    "FI": "NORTHERN_EUROPE",
    "DK": "NORTHERN_EUROPE",
    "EE": "NORTHERN_EUROPE",
    "LV": "NORTHERN_EUROPE",
    "LT": "NORTHERN_EUROPE",
    "GB": "CENTRAL_MARITIME",
    "IE": "CENTRAL_MARITIME",
    "NL": "CENTRAL_MARITIME",
    "BE": "CENTRAL_MARITIME",
    "LU": "CENTRAL_MARITIME",
    "FR": "CENTRAL_MARITIME",
    "DE": "CENTRAL_CONTINENTAL",
    "AT": "CENTRAL_CONTINENTAL",
    "PL": "CENTRAL_CONTINENTAL",
    "CZ": "CENTRAL_CONTINENTAL",
    "SK": "CENTRAL_CONTINENTAL",
    "CH": "CENTRAL_CONTINENTAL",
    "PT": "SOUTHERN_MARITIME",
    "ES": "SOUTHERN_MARITIME",
    "IT": "SOUTHERN_CONTINENTAL",
    "HU": "SOUTHERN_CONTINENTAL",
    "RO": "SOUTHERN_CONTINENTAL",
    "SI": "SOUTHERN_CONTINENTAL",
    "HR": "SOUTHERN_CONTINENTAL",
    "GR": "MEDITERRANEAN",
    "CY": "MEDITERRANEAN",
    "MT": "MEDITERRANEAN",
    "BG": "SOUTHERN_CONTINENTAL",
}


# =============================================================================
# Pydantic Sub-Config Models
# =============================================================================


class EnvelopeConfig(BaseModel):
    """Configuration for building envelope thermal performance.

    Covers walls, roof, floor, windows, airtightness, and thermal bridging.
    Targets are aligned with EPBD cost-optimal / NZEB levels and national
    building regulations. U-values in W/m2K.
    """

    wall_u_target: float = Field(
        0.26,
        ge=0.10,
        le=2.0,
        description="Target wall U-value (W/m2K)",
    )
    roof_u_target: float = Field(
        0.18,
        ge=0.08,
        le=1.5,
        description="Target roof U-value (W/m2K)",
    )
    floor_u_target: float = Field(
        0.22,
        ge=0.10,
        le=1.5,
        description="Target ground floor U-value (W/m2K)",
    )
    window_u_target: float = Field(
        1.4,
        ge=0.5,
        le=5.0,
        description="Target window U-value (W/m2K), including frame",
    )
    window_g_value_target: float = Field(
        0.50,
        ge=0.10,
        le=0.85,
        description="Target window solar heat gain coefficient (g-value)",
    )
    airtightness_target_m3_h_m2: float = Field(
        5.0,
        ge=0.3,
        le=20.0,
        description="Target airtightness (m3/h/m2 at 50Pa)",
    )
    thermal_bridge_method: ThermalBridgeMethod = Field(
        ThermalBridgeMethod.DEFAULT_UPLIFT,
        description="Thermal bridge calculation method",
    )
    thermal_bridge_uplift_pct: float = Field(
        15.0,
        ge=0.0,
        le=50.0,
        description="Default thermal bridge U-value uplift percentage (if DEFAULT_UPLIFT method)",
    )
    condensation_risk_check: bool = Field(
        True,
        description="Perform interstitial and surface condensation risk check",
    )
    overheating_assessment: bool = Field(
        True,
        description="Perform overheating risk assessment (TM59/CIBSE Guide A)",
    )
    wall_area_m2: Optional[float] = Field(
        None,
        ge=0,
        description="Total external wall area (m2), auto-calculated if None",
    )
    roof_area_m2: Optional[float] = Field(
        None,
        ge=0,
        description="Total roof area (m2), auto-calculated if None",
    )
    window_to_wall_ratio: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Window-to-wall ratio (0.0-1.0), auto-detected if None",
    )


class HVACConfig(BaseModel):
    """Configuration for HVAC system analysis.

    Covers heating, cooling, ventilation, and air handling systems including
    system type identification, efficiency targets, and control assessment.
    """

    heating_system_type: str = Field(
        "gas_boiler",
        description="Primary heating system type (gas_boiler, heat_pump, district_heating, "
        "electric_boiler, biomass_boiler, oil_boiler)",
    )
    cooling_system_type: Optional[str] = Field(
        "split_system",
        description="Primary cooling system type (split_system, chiller, vrf, none)",
    )
    ventilation_type: VentilationType = Field(
        VentilationType.MECHANICAL_SUPPLY_EXTRACT,
        description="Ventilation strategy type",
    )
    heat_recovery_enabled: bool = Field(
        True,
        description="Heat recovery installed or targeted on ventilation system",
    )
    heat_recovery_efficiency_pct: float = Field(
        70.0,
        ge=0.0,
        le=95.0,
        description="Target heat recovery efficiency (%)",
    )
    target_heating_cop: float = Field(
        3.5,
        ge=0.8,
        le=6.0,
        description="Target heating COP (for heat pumps) or efficiency ratio",
    )
    target_cooling_seer: float = Field(
        4.0,
        ge=1.5,
        le=10.0,
        description="Target cooling SEER (Seasonal Energy Efficiency Ratio)",
    )
    boiler_efficiency_target_pct: float = Field(
        92.0,
        ge=70.0,
        le=99.0,
        description="Target boiler seasonal efficiency (% gross)",
    )
    ahu_optimization: bool = Field(
        True,
        description="Assess Air Handling Unit scheduling and setpoint optimization",
    )
    free_cooling_assessment: bool = Field(
        True,
        description="Assess free cooling / economizer hours potential",
    )
    bms_optimization: bool = Field(
        True,
        description="Assess Building Management System optimization opportunities",
    )
    zone_control_assessment: bool = Field(
        True,
        description="Assess zone-level temperature and ventilation control adequacy",
    )
    demand_controlled_ventilation: bool = Field(
        False,
        description="Demand-controlled ventilation (CO2/occupancy) enabled or targeted",
    )
    vsd_on_pumps_fans: bool = Field(
        True,
        description="Variable speed drives on circulation pumps and AHU fans",
    )
    pipe_insulation_assessment: bool = Field(
        True,
        description="Assess heating/cooling pipe insulation condition",
    )

    @field_validator("heating_system_type")
    @classmethod
    def validate_heating_system(cls, v: str) -> str:
        """Validate heating system type."""
        valid = {
            "gas_boiler", "heat_pump", "district_heating", "electric_boiler",
            "biomass_boiler", "oil_boiler", "lpg_boiler", "chp",
        }
        if v.lower() not in valid:
            raise ValueError(f"Invalid heating system type: {v}. Must be one of {sorted(valid)}.")
        return v.lower()


class LightingConfig(BaseModel):
    """Configuration for lighting system analysis.

    Covers installed lighting power density, controls, daylighting, and
    LED retrofit assessment per EN 15193-1 and EN 12464-1.
    """

    lpd_target_w_m2: float = Field(
        8.0,
        ge=1.0,
        le=30.0,
        description="Target Lighting Power Density (W/m2)",
    )
    control_type: LightingControlType = Field(
        LightingControlType.COMBINED_OCCUPANCY_DAYLIGHT,
        description="Target lighting control strategy",
    )
    daylight_factor_target: float = Field(
        2.0,
        ge=0.5,
        le=10.0,
        description="Target average daylight factor (%)",
    )
    led_retrofit_analysis: bool = Field(
        True,
        description="Analyse LED retrofit potential for non-LED luminaires",
    )
    emergency_lighting_audit: bool = Field(
        False,
        description="Include emergency lighting in energy audit",
    )
    outdoor_lighting: bool = Field(
        True,
        description="Include external and car park lighting in analysis",
    )
    lux_level_measurement: bool = Field(
        True,
        description="Perform lux level measurements to verify illumination adequacy",
    )
    parasitic_lighting_load: bool = Field(
        True,
        description="Assess parasitic/standby loads from lighting controls and drivers",
    )
    display_lighting: bool = Field(
        False,
        description="Include display and accent lighting assessment (retail/hospitality)",
    )


class DHWConfig(BaseModel):
    """Configuration for Domestic Hot Water system analysis.

    Covers hot water generation, distribution, storage, and solar thermal
    integration. Legionella compliance per HSG274/BS 8558.
    """

    system_type: str = Field(
        "gas_boiler",
        description="DHW generation type (gas_boiler, electric_immersion, heat_pump, "
        "district_heating, solar_thermal_preheat, point_of_use)",
    )
    solar_thermal_enabled: bool = Field(
        False,
        description="Solar thermal preheat installed or targeted for DHW",
    )
    solar_thermal_contribution_pct: float = Field(
        0.0,
        ge=0.0,
        le=80.0,
        description="Target solar thermal contribution to annual DHW demand (%)",
    )
    legionella_temp_c: float = Field(
        60.0,
        ge=55.0,
        le=70.0,
        description="Minimum storage temperature for Legionella control (deg C)",
    )
    distribution_loss_assessment: bool = Field(
        True,
        description="Assess DHW distribution pipe losses and dead-legs",
    )
    secondary_return_enabled: bool = Field(
        False,
        description="Secondary circulation return loop present",
    )
    storage_volume_litres: Optional[float] = Field(
        None,
        ge=0,
        description="Total DHW storage volume (litres), auto-detected if None",
    )
    daily_demand_litres: Optional[float] = Field(
        None,
        ge=0,
        description="Estimated daily DHW demand (litres), auto-calculated if None",
    )

    @field_validator("system_type")
    @classmethod
    def validate_dhw_system(cls, v: str) -> str:
        """Validate DHW system type."""
        valid = {
            "gas_boiler", "electric_immersion", "heat_pump", "district_heating",
            "solar_thermal_preheat", "point_of_use", "combi_boiler",
        }
        if v.lower() not in valid:
            raise ValueError(f"Invalid DHW system type: {v}. Must be one of {sorted(valid)}.")
        return v.lower()


class RenewableConfig(BaseModel):
    """Configuration for on-site renewable energy systems.

    Covers solar PV, solar thermal, heat pumps, and other on-site generation.
    Sizing aligned with available roof area and building energy demand.
    """

    solar_pv_enabled: bool = Field(
        False,
        description="Solar PV system installed or targeted",
    )
    pv_capacity_kwp: float = Field(
        0.0,
        ge=0.0,
        le=5000.0,
        description="Installed or target PV capacity (kWp)",
    )
    pv_orientation_deg: float = Field(
        180.0,
        ge=0.0,
        le=360.0,
        description="PV array azimuth orientation (degrees, 180=south)",
    )
    pv_tilt_deg: float = Field(
        35.0,
        ge=0.0,
        le=90.0,
        description="PV array tilt angle (degrees from horizontal)",
    )
    pv_yield_kwh_per_kwp: float = Field(
        900.0,
        ge=400.0,
        le=1800.0,
        description="Expected annual PV yield (kWh per kWp installed)",
    )
    battery_storage_enabled: bool = Field(
        False,
        description="Battery energy storage system installed or targeted",
    )
    battery_capacity_kwh: float = Field(
        0.0,
        ge=0.0,
        le=10000.0,
        description="Battery storage capacity (kWh)",
    )
    solar_thermal_enabled: bool = Field(
        False,
        description="Solar thermal collectors installed or targeted",
    )
    collector_area_m2: float = Field(
        0.0,
        ge=0.0,
        le=500.0,
        description="Solar thermal collector area (m2)",
    )
    heat_pump_enabled: bool = Field(
        False,
        description="Heat pump installed or targeted as primary heating source",
    )
    heat_pump_type: str = Field(
        "air_source",
        description="Heat pump type (air_source, ground_source, water_source)",
    )
    wind_turbine_enabled: bool = Field(
        False,
        description="Small wind turbine installed or targeted",
    )

    @field_validator("heat_pump_type")
    @classmethod
    def validate_heat_pump_type(cls, v: str) -> str:
        """Validate heat pump type."""
        valid = {"air_source", "ground_source", "water_source", "exhaust_air"}
        if v.lower() not in valid:
            raise ValueError(f"Invalid heat pump type: {v}. Must be one of {sorted(valid)}.")
        return v.lower()


class BenchmarkConfig(BaseModel):
    """Configuration for building energy performance benchmarking.

    Benchmarks building against DEC/CIBSE TM46, ENERGY STAR, national
    median/best-practice, and peer group comparisons.
    """

    enabled: bool = Field(
        True,
        description="Enable energy benchmarking",
    )
    benchmark_standard: str = Field(
        "CIBSE_TM46",
        description="Primary benchmark standard: CIBSE_TM46, ENERGY_STAR, "
        "DIN_18599, EN_15603, NATIONAL",
    )
    reference_eui_kwh_m2: Optional[float] = Field(
        None,
        ge=0,
        description="Custom reference EUI (kWh/m2/yr); uses standard default if None",
    )
    peer_group_size: int = Field(
        10,
        ge=3,
        le=100,
        description="Minimum peer group size for meaningful benchmark",
    )
    dec_rating_enabled: bool = Field(
        True,
        description="Calculate Display Energy Certificate (DEC) operational rating",
    )
    asset_rating_enabled: bool = Field(
        True,
        description="Calculate asset (design) energy rating",
    )
    percentile_tracking: bool = Field(
        True,
        description="Track percentile ranking within peer group",
    )
    gap_to_best_practice: bool = Field(
        True,
        description="Calculate gap to sector best practice EUI",
    )
    trend_analysis_years: int = Field(
        3,
        ge=1,
        le=10,
        description="Number of years for energy use trend analysis",
    )
    kpi_set: List[str] = Field(
        default_factory=lambda: [
            "eui_kwh_m2_yr",
            "heating_kwh_m2_yr",
            "cooling_kwh_m2_yr",
            "electricity_kwh_m2_yr",
            "gas_kwh_m2_yr",
            "carbon_kgco2_m2_yr",
            "epc_rating",
        ],
        description="KPI set for benchmarking",
    )


class RetrofitConfig(BaseModel):
    """Configuration for energy retrofit planning and financial analysis.

    Covers measure identification, financial evaluation per EN 15459,
    and retrofit sequencing for fabric-first and deep retrofit strategies.
    """

    budget_eur: float = Field(
        500000.0,
        ge=0,
        description="Total retrofit budget (EUR)",
    )
    retrofit_ambition: RetrofitAmbition = Field(
        RetrofitAmbition.COST_OPTIMAL,
        description="Retrofit depth and ambition level",
    )
    discount_rate_pct: float = Field(
        3.5,
        ge=0.0,
        le=15.0,
        description="Real discount rate for NPV calculation (%)",
    )
    energy_price_escalation_pct: float = Field(
        2.5,
        ge=0.0,
        le=10.0,
        description="Annual energy price escalation rate (%)",
    )
    study_period_years: int = Field(
        30,
        ge=5,
        le=60,
        description="Economic study period for lifecycle cost analysis (years)",
    )
    carbon_price_eur_per_tonne: float = Field(
        85.0,
        ge=0.0,
        le=500.0,
        description="Shadow carbon price for monetised carbon savings (EUR/tCO2)",
    )
    fabric_first_approach: bool = Field(
        True,
        description="Prioritise building fabric improvements before systems upgrades",
    )
    include_grants_subsidies: bool = Field(
        True,
        description="Include available grants and subsidies in financial analysis",
    )
    measure_interaction_modelling: bool = Field(
        True,
        description="Model interactions between retrofit measures (avoid double-counting)",
    )
    max_payback_years: float = Field(
        15.0,
        ge=1.0,
        le=40.0,
        description="Maximum acceptable simple payback period (years)",
    )
    include_maintenance_costs: bool = Field(
        True,
        description="Include maintenance cost changes in lifecycle cost analysis",
    )
    enable_staged_retrofit: bool = Field(
        True,
        description="Allow phased/staged retrofit planning over multiple years",
    )


class IndoorEnvironmentConfig(BaseModel):
    """Configuration for indoor environmental quality assessment.

    Covers thermal comfort, air quality, and occupant satisfaction per
    EN 16798-1 and CIBSE Guide A.
    """

    ieq_category: IEQCategory = Field(
        IEQCategory.CATEGORY_II,
        description="Target IEQ category per EN 16798-1",
    )
    thermal_comfort_method: ThermalComfortMethod = Field(
        ThermalComfortMethod.PMV_PPD,
        description="Thermal comfort assessment methodology",
    )
    target_co2_ppm: int = Field(
        800,
        ge=400,
        le=2000,
        description="Target indoor CO2 concentration (ppm above outdoor)",
    )
    target_temperature_heating_c: float = Field(
        21.0,
        ge=16.0,
        le=24.0,
        description="Target indoor temperature during heating season (deg C)",
    )
    target_temperature_cooling_c: float = Field(
        25.0,
        ge=22.0,
        le=30.0,
        description="Target indoor temperature during cooling season (deg C)",
    )
    relative_humidity_range_pct: Tuple[float, float] = Field(
        default=(40.0, 60.0),
        description="Acceptable relative humidity range (min%, max%)",
    )
    daylighting_assessment: bool = Field(
        True,
        description="Include daylighting adequacy assessment",
    )
    acoustic_assessment: bool = Field(
        False,
        description="Include acoustic comfort assessment (noise levels)",
    )
    post_occupancy_evaluation: bool = Field(
        False,
        description="Include post-occupancy evaluation survey",
    )
    overheating_hours_threshold: int = Field(
        35,
        ge=0,
        le=200,
        description="Maximum acceptable overheating hours per year (CIBSE TM59 criterion)",
    )


class CarbonConfig(BaseModel):
    """Configuration for operational and whole-life carbon assessment.

    Covers operational carbon (Scope 1+2), embodied carbon for retrofit
    measures, and whole-life carbon aligned with EN 15978 / RICS methodology.
    """

    grid_emission_factor_kg_per_kwh: float = Field(
        0.233,
        ge=0.0,
        le=1.5,
        description="Grid electricity emission factor (kgCO2e/kWh)",
    )
    gas_emission_factor_kg_per_kwh: float = Field(
        0.204,
        ge=0.0,
        le=0.5,
        description="Natural gas emission factor (kgCO2e/kWh)",
    )
    study_period_years: int = Field(
        60,
        ge=10,
        le=120,
        description="Whole-life carbon study period (years)",
    )
    include_embodied: bool = Field(
        False,
        description="Include embodied carbon of retrofit materials (EN 15978 modules A-C)",
    )
    carbon_target_kgco2_m2_yr: Optional[float] = Field(
        None,
        ge=0.0,
        description="Target operational carbon intensity (kgCO2e/m2/yr)",
    )
    grid_decarbonisation_enabled: bool = Field(
        True,
        description="Apply projected grid decarbonisation to future carbon calculations",
    )
    grid_decarbonisation_rate_pct_yr: float = Field(
        3.0,
        ge=0.0,
        le=15.0,
        description="Annual grid decarbonisation rate (%)",
    )
    scope_1_included: bool = Field(
        True,
        description="Include Scope 1 (direct) emissions from on-site combustion",
    )
    scope_2_included: bool = Field(
        True,
        description="Include Scope 2 (indirect) emissions from purchased electricity",
    )
    refrigerant_leakage_included: bool = Field(
        False,
        description="Include refrigerant leakage in Scope 1 emissions",
    )


class ComplianceConfig(BaseModel):
    """Configuration for regulatory compliance tracking.

    Covers EPBD, national EPC regulations, MEES (Minimum Energy Efficiency
    Standards), BPS (Building Performance Standards), and public sector
    requirements under the Energy Efficiency Directive.
    """

    epbd_country: str = Field(
        "GB",
        description="Country for EPBD transposition requirements (ISO 3166-1 alpha-2)",
    )
    epc_methodology: EPCMethodology = Field(
        EPCMethodology.SBEM,
        description="EPC calculation methodology",
    )
    mees_enabled: bool = Field(
        True,
        description="Track Minimum Energy Efficiency Standards compliance",
    )
    mees_minimum_rating: str = Field(
        "E",
        description="Minimum EPC rating for MEES compliance (e.g. E for England & Wales)",
    )
    bps_enabled: bool = Field(
        False,
        description="Track Building Performance Standards compliance (e.g. NABERS UK)",
    )
    bps_target_rating: Optional[str] = Field(
        None,
        description="BPS target rating (e.g. '3_star' for NABERS)",
    )
    dec_required: bool = Field(
        False,
        description="Display Energy Certificate required (public buildings >250m2 in UK)",
    )
    eed_public_body: bool = Field(
        False,
        description="Subject to EED Article 6 public body renovation requirements",
    )
    epbd_renovation_passport: bool = Field(
        False,
        description="Generate EPBD Building Renovation Passport",
    )
    national_nzeb_definition: str = Field(
        "EPBD_2024",
        description="NZEB definition reference (EPBD_2024, national_specific)",
    )
    zero_emission_building_target: bool = Field(
        False,
        description="Target Zero-Emission Building (ZEB) per EPBD 2024 recast Article 7",
    )

    @field_validator("mees_minimum_rating")
    @classmethod
    def validate_mees_rating(cls, v: str) -> str:
        """Validate MEES minimum EPC rating."""
        valid = {"A+", "A", "B", "C", "D", "E", "F", "G"}
        if v.upper() not in valid:
            raise ValueError(f"Invalid MEES minimum rating: {v}. Must be one of {sorted(valid)}.")
        return v.upper()


class ReportConfig(BaseModel):
    """Configuration for assessment report generation."""

    language: str = Field(
        "en",
        description="Report language (ISO 639-1)",
    )
    currency: str = Field(
        "GBP",
        description="Currency for financial analysis (ISO 4217)",
    )
    include_executive_summary: bool = Field(
        True,
        description="Include executive summary with key findings and quick wins",
    )
    include_financial_analysis: bool = Field(
        True,
        description="Include lifecycle cost and financial appraisal section",
    )
    include_thermal_model: bool = Field(
        True,
        description="Include building thermal model results and assumptions",
    )
    include_retrofit_roadmap: bool = Field(
        True,
        description="Include phased retrofit roadmap with Gantt chart",
    )
    include_epc_simulation: bool = Field(
        True,
        description="Include simulated EPC rating before and after retrofit",
    )
    include_carbon_trajectory: bool = Field(
        True,
        description="Include carbon trajectory towards net-zero target",
    )
    output_formats: List[OutputFormat] = Field(
        default_factory=lambda: [OutputFormat.PDF, OutputFormat.XLSX],
        description="Output formats for reports",
    )
    watermark_draft: bool = Field(
        True,
        description="Apply DRAFT watermark to unapproved reports",
    )
    client_logo_enabled: bool = Field(
        False,
        description="Include client logo on report cover page",
    )


class IntegrationConfig(BaseModel):
    """Configuration for external system integrations.

    Covers Building Management Systems, weather data, certification
    platforms, and utility data connections.
    """

    bms_enabled: bool = Field(
        False,
        description="Enable Building Management System data integration",
    )
    bms_protocol: BMSProtocol = Field(
        BMSProtocol.BACNET,
        description="BMS communication protocol",
    )
    weather_source: str = Field(
        "met_office",
        description="Weather data source (met_office, dwd, meteonorm, epw_file)",
    )
    degree_day_base_heating_c: float = Field(
        15.5,
        ge=10.0,
        le=20.0,
        description="Heating degree day base temperature (deg C)",
    )
    degree_day_base_cooling_c: float = Field(
        18.0,
        ge=15.0,
        le=28.0,
        description="Cooling degree day base temperature (deg C)",
    )
    certification_api_enabled: bool = Field(
        False,
        description="Enable API connection to LEED/BREEAM certification platforms",
    )
    utility_data_api: bool = Field(
        False,
        description="Enable automated utility bill data import (API/EDI)",
    )
    epc_register_lookup: bool = Field(
        True,
        description="Enable EPC register lookup for existing building certificates",
    )
    cad_import_enabled: bool = Field(
        False,
        description="Enable CAD/BIM model import for geometry extraction",
    )


class PerformanceConfig(BaseModel):
    """Performance and resource limits for pack execution."""

    max_buildings: int = Field(
        50,
        ge=1,
        le=500,
        description="Maximum number of buildings per assessment run",
    )
    max_zones: int = Field(
        200,
        ge=10,
        le=5000,
        description="Maximum thermal zones per building model",
    )
    max_meters: int = Field(
        500,
        ge=10,
        le=10000,
        description="Maximum sub-meters per building",
    )
    cache_ttl_seconds: int = Field(
        3600,
        ge=60,
        le=86400,
        description="Cache TTL for weather data and reference data (seconds)",
    )
    batch_size: int = Field(
        1000,
        ge=100,
        le=10000,
        description="Batch size for bulk data processing",
    )
    calculation_timeout_seconds: int = Field(
        300,
        ge=30,
        le=3600,
        description="Timeout for individual engine calculations (seconds)",
    )
    parallel_engines: int = Field(
        4,
        ge=1,
        le=16,
        description="Maximum number of engines running in parallel",
    )


class SecurityConfig(BaseModel):
    """Security and access control configuration."""

    roles: List[str] = Field(
        default_factory=lambda: [
            "energy_assessor",
            "building_manager",
            "sustainability_officer",
            "property_owner",
            "tenant",
            "viewer",
            "admin",
        ],
        description="Available RBAC roles for the pack",
    )
    data_classification: str = Field(
        "CONFIDENTIAL",
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
        description="Require encryption at rest for stored data",
    )
    tenant_data_isolation: bool = Field(
        True,
        description="Enforce tenant data isolation in multi-tenant buildings",
    )


class AuditTrailConfig(BaseModel):
    """Configuration for calculation audit trail and provenance."""

    enabled: bool = Field(
        True,
        description="Enable audit trail for all calculations",
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
        description="Track all assumptions and default values used",
    )
    data_lineage_enabled: bool = Field(
        True,
        description="Track full data lineage from source to output",
    )
    retention_years: int = Field(
        10,
        ge=1,
        le=25,
        description="Audit trail retention period in years",
    )
    external_audit_export: bool = Field(
        True,
        description="Enable export format for third-party energy assessors",
    )
    measure_tracking: bool = Field(
        True,
        description="Track retrofit measure implementation status",
    )
    epc_lodgement_record: bool = Field(
        True,
        description="Record EPC lodgement reference for regulatory compliance",
    )


# =============================================================================
# Main Configuration Model
# =============================================================================


class BuildingEnergyAssessmentConfig(BaseModel):
    """Main configuration for PACK-032 Building Energy Assessment Pack.

    This is the root configuration model that contains all sub-configurations
    for building energy assessment. The building_type and climate_zone fields
    drive which benchmarks are used and which systems are prioritized.
    """

    # Building identification
    building_name: str = Field(
        "",
        description="Building name or site identifier",
    )
    building_address: str = Field(
        "",
        description="Full postal address of the building",
    )
    client_name: str = Field(
        "",
        description="Client or building owner name",
    )
    building_type: BuildingType = Field(
        BuildingType.OFFICE,
        description="Primary building type",
    )
    building_age: BuildingAge = Field(
        BuildingAge.BAND_1996_2006,
        description="Building age band (year of construction)",
    )
    climate_zone: ClimateZone = Field(
        ClimateZone.CENTRAL_MARITIME,
        description="European climate zone for design conditions",
    )
    country: str = Field(
        "GB",
        description="Building country (ISO 3166-1 alpha-2)",
    )
    assessment_level: AssessmentLevel = Field(
        AssessmentLevel.STANDARD,
        description="Assessment depth level",
    )
    certification_target: CertificationTarget = Field(
        CertificationTarget.NONE,
        description="Green building certification target (if any)",
    )
    reporting_year: int = Field(
        2025,
        ge=2020,
        le=2035,
        description="Reporting year for the assessment",
    )

    # Building characteristics
    gross_internal_area_m2: Optional[float] = Field(
        None,
        ge=0,
        description="Gross Internal Area in square meters (GIA)",
    )
    net_lettable_area_m2: Optional[float] = Field(
        None,
        ge=0,
        description="Net Lettable Area in square meters (NLA)",
    )
    number_of_floors: int = Field(
        1,
        ge=1,
        le=100,
        description="Number of above-ground storeys",
    )
    number_of_basements: int = Field(
        0,
        ge=0,
        le=10,
        description="Number of below-ground storeys",
    )
    floor_to_ceiling_height_m: float = Field(
        2.7,
        ge=2.2,
        le=6.0,
        description="Typical floor-to-ceiling height (m)",
    )
    occupancy_pattern: OccupancyPattern = Field(
        OccupancyPattern.SINGLE_SHIFT,
        description="Building occupancy schedule pattern",
    )
    typical_occupancy_persons: Optional[int] = Field(
        None,
        ge=0,
        description="Typical building occupancy (persons)",
    )
    operating_hours_per_year: int = Field(
        2500,
        ge=500,
        le=8760,
        description="Annual building operating hours",
    )
    ownership_type: OwnershipType = Field(
        OwnershipType.OWNER_OCCUPIED,
        description="Building ownership and tenure type",
    )
    primary_heating_fuel: HeatingFuel = Field(
        HeatingFuel.NATURAL_GAS,
        description="Primary heating fuel type",
    )

    # Current energy performance (if known)
    current_epc_rating: Optional[str] = Field(
        None,
        description="Current EPC rating (A+ to G) if available",
    )
    current_eui_kwh_m2: Optional[float] = Field(
        None,
        ge=0,
        description="Current measured Energy Use Intensity (kWh/m2/yr) if available",
    )
    target_epc_rating: Optional[str] = Field(
        None,
        description="Target EPC rating after retrofit (A+ to G)",
    )
    target_eui_kwh_m2: Optional[float] = Field(
        None,
        ge=0,
        description="Target EUI after retrofit (kWh/m2/yr)",
    )

    # Sub-configurations for each engine
    envelope: EnvelopeConfig = Field(
        default_factory=EnvelopeConfig,
        description="Building envelope thermal performance configuration",
    )
    hvac: HVACConfig = Field(
        default_factory=HVACConfig,
        description="HVAC system analysis configuration",
    )
    lighting: LightingConfig = Field(
        default_factory=LightingConfig,
        description="Lighting system analysis configuration",
    )
    dhw: DHWConfig = Field(
        default_factory=DHWConfig,
        description="Domestic hot water system configuration",
    )
    renewable: RenewableConfig = Field(
        default_factory=RenewableConfig,
        description="On-site renewable energy configuration",
    )
    benchmark: BenchmarkConfig = Field(
        default_factory=BenchmarkConfig,
        description="Energy benchmarking configuration",
    )
    retrofit: RetrofitConfig = Field(
        default_factory=RetrofitConfig,
        description="Retrofit planning and financial analysis configuration",
    )
    indoor_environment: IndoorEnvironmentConfig = Field(
        default_factory=IndoorEnvironmentConfig,
        description="Indoor environmental quality configuration",
    )
    carbon: CarbonConfig = Field(
        default_factory=CarbonConfig,
        description="Operational and whole-life carbon configuration",
    )
    compliance: ComplianceConfig = Field(
        default_factory=ComplianceConfig,
        description="Regulatory compliance tracking configuration",
    )
    report: ReportConfig = Field(
        default_factory=ReportConfig,
        description="Report generation configuration",
    )
    integration: IntegrationConfig = Field(
        default_factory=IntegrationConfig,
        description="External system integration configuration",
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
    def validate_cooling_for_heating_dominant(self) -> "BuildingEnergyAssessmentConfig":
        """Auto-disable cooling assessment for heating-dominant climates with no cooling."""
        heating_dominant_zones = {
            ClimateZone.NORTHERN_EUROPE,
        }
        heating_only_types = {
            BuildingType.RESIDENTIAL_HOUSE,
            BuildingType.RESIDENTIAL_APARTMENT,
            BuildingType.WAREHOUSE,
        }
        if (
            self.climate_zone in heating_dominant_zones
            and self.building_type in heating_only_types
            and self.hvac.cooling_system_type is not None
            and self.hvac.cooling_system_type != "none"
        ):
            logger.info(
                f"Heating-dominant zone ({self.climate_zone.value}) with "
                f"residential/warehouse building: setting cooling to none."
            )
            object.__setattr__(self.hvac, "cooling_system_type", "none")
        return self

    @model_validator(mode="after")
    def validate_heat_pump_for_electrification(self) -> "BuildingEnergyAssessmentConfig":
        """Auto-enable heat pump in renewable config when retrofit ambition is net-zero."""
        electrification_ambitions = {
            RetrofitAmbition.NZEB,
            RetrofitAmbition.NET_ZERO,
        }
        if (
            self.retrofit.retrofit_ambition in electrification_ambitions
            and not self.renewable.heat_pump_enabled
        ):
            logger.info(
                f"Retrofit ambition {self.retrofit.retrofit_ambition.value} requires "
                "electrification. Enabling heat pump in renewable configuration."
            )
            object.__setattr__(self.renewable, "heat_pump_enabled", True)
        return self

    @model_validator(mode="after")
    def validate_climate_zone_vs_country(self) -> "BuildingEnergyAssessmentConfig":
        """Warn if climate zone does not match the country."""
        expected_zone = COUNTRY_CLIMATE_ZONE.get(self.country)
        if expected_zone and expected_zone != self.climate_zone.value:
            logger.warning(
                f"Climate zone {self.climate_zone.value} may not match country "
                f"{self.country} (expected {expected_zone}). Verify climate zone is correct."
            )
        return self

    @model_validator(mode="after")
    def validate_certification_vs_building_type(self) -> "BuildingEnergyAssessmentConfig":
        """Validate certification target is appropriate for building type."""
        residential_types = {
            BuildingType.RESIDENTIAL_APARTMENT,
            BuildingType.RESIDENTIAL_HOUSE,
        }
        commercial_certifications = {
            CertificationTarget.ENERGY_STAR,
        }
        if (
            self.building_type in residential_types
            and self.certification_target in commercial_certifications
        ):
            logger.warning(
                f"Certification target {self.certification_target.value} is not "
                f"applicable to residential building type {self.building_type.value}."
            )
        return self

    @model_validator(mode="after")
    def validate_dec_for_public_buildings(self) -> "BuildingEnergyAssessmentConfig":
        """Auto-enable DEC compliance for public sector buildings."""
        if (
            self.building_type == BuildingType.PUBLIC_SECTOR
            and not self.compliance.dec_required
        ):
            logger.info(
                "Public sector building: enabling Display Energy Certificate requirement."
            )
            object.__setattr__(self.compliance, "dec_required", True)
        return self

    @model_validator(mode="after")
    def validate_epc_methodology_vs_country(self) -> "BuildingEnergyAssessmentConfig":
        """Set EPC methodology based on country if using generic."""
        country_methodology = {
            "GB": EPCMethodology.SBEM,
            "DE": EPCMethodology.GEG,
            "FR": EPCMethodology.DPE,
            "IT": EPCMethodology.APE,
        }
        residential_types = {
            BuildingType.RESIDENTIAL_APARTMENT,
            BuildingType.RESIDENTIAL_HOUSE,
        }
        if self.compliance.epc_methodology == EPCMethodology.EPBD_GENERIC:
            method = country_methodology.get(self.country)
            if method:
                logger.info(
                    f"Setting EPC methodology to {method.value} for country {self.country}."
                )
                object.__setattr__(self.compliance, "epc_methodology", method)
        # UK residential should use SAP/RdSAP not SBEM
        if (
            self.country == "GB"
            and self.building_type in residential_types
            and self.compliance.epc_methodology == EPCMethodology.SBEM
        ):
            logger.info(
                "UK residential building: switching EPC methodology from SBEM to SAP."
            )
            object.__setattr__(self.compliance, "epc_methodology", EPCMethodology.SAP)
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

    pack: BuildingEnergyAssessmentConfig = Field(
        default_factory=BuildingEnergyAssessmentConfig,
        description="Main Building Energy Assessment configuration",
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
        "PACK-032-building-energy-assessment",
        description="Pack identifier",
    )

    @classmethod
    def from_preset(
        cls, preset_name: str, overrides: Optional[Dict[str, Any]] = None
    ) -> "PackConfig":
        """Load configuration from a named preset.

        Args:
            preset_name: Name of the preset (commercial_office, retail_building, etc.)
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

        pack_config = BuildingEnergyAssessmentConfig(**preset_data)
        return cls(pack=pack_config, preset_name=preset_name)

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

        pack_config = BuildingEnergyAssessmentConfig(**config_data)
        return cls(pack=pack_config)

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
        base_dict = base.pack.model_dump()
        merged = cls._deep_merge(base_dict, overrides)
        pack_config = BuildingEnergyAssessmentConfig(**merged)
        return cls(
            pack=pack_config,
            preset_name=base.preset_name,
            config_version=base.config_version,
        )

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """Load configuration overrides from environment variables.

        Environment variables prefixed with BUILDING_ENERGY_PACK_ are loaded
        and mapped to configuration keys. Nested keys use double underscore.

        Example: BUILDING_ENERGY_PACK_ENVELOPE__WALL_U_TARGET=0.20
                 BUILDING_ENERGY_PACK_HVAC__TARGET_HEATING_COP=4.0
        """
        overrides: Dict[str, Any] = {}
        prefix = "BUILDING_ENERGY_PACK_"
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
        return validate_config(self.pack)


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


def validate_config(config: BuildingEnergyAssessmentConfig) -> List[str]:
    """Validate a building energy assessment configuration and return any warnings.

    Args:
        config: BuildingEnergyAssessmentConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check building identification
    if not config.building_name:
        warnings.append(
            "No building_name configured. Add a building name for report identification."
        )
    if not config.building_address:
        warnings.append(
            "No building_address configured. Address is required for EPC lodgement."
        )

    # Check floor area
    if config.gross_internal_area_m2 is None:
        warnings.append(
            "No gross_internal_area_m2 configured. Floor area is required for EUI "
            "calculation and benchmarking."
        )

    # Check envelope targets vs climate zone
    expected_u = U_VALUE_TARGETS.get(config.climate_zone.value, {})
    if expected_u:
        if config.envelope.wall_u_target > expected_u.get("wall", 999) * 1.5:
            warnings.append(
                f"Wall U-value target ({config.envelope.wall_u_target} W/m2K) is significantly "
                f"above the NZEB level ({expected_u.get('wall')}) for climate zone "
                f"{config.climate_zone.value}."
            )

    # Check residential EPC methodology
    residential_types = {
        BuildingType.RESIDENTIAL_APARTMENT,
        BuildingType.RESIDENTIAL_HOUSE,
    }
    if config.building_type in residential_types:
        if config.country == "GB" and config.compliance.epc_methodology not in (
            EPCMethodology.SAP, EPCMethodology.RDSAP
        ):
            warnings.append(
                "UK residential buildings should use SAP or RdSAP methodology."
            )

    # Check DEC requirement for public buildings
    if config.building_type == BuildingType.PUBLIC_SECTOR:
        if not config.compliance.dec_required:
            warnings.append(
                "Public sector buildings typically require Display Energy Certificates."
            )

    # Check MEES compliance
    if config.compliance.mees_enabled and config.current_epc_rating:
        mees_min = config.compliance.mees_minimum_rating
        rating_order = ["A+", "A", "B", "C", "D", "E", "F", "G"]
        if config.current_epc_rating in rating_order and mees_min in rating_order:
            current_idx = rating_order.index(config.current_epc_rating)
            mees_idx = rating_order.index(mees_min)
            if current_idx > mees_idx:
                warnings.append(
                    f"Current EPC rating ({config.current_epc_rating}) is below "
                    f"MEES minimum ({mees_min}). Building may be non-compliant for lettings."
                )

    # Check cooling configuration for Mediterranean climate
    if config.climate_zone == ClimateZone.MEDITERRANEAN:
        if config.hvac.cooling_system_type == "none" or config.hvac.cooling_system_type is None:
            warnings.append(
                "Mediterranean climate zone typically requires active cooling. "
                "Consider specifying a cooling system type."
            )

    # Check renewable potential
    if config.retrofit.retrofit_ambition in (RetrofitAmbition.NZEB, RetrofitAmbition.NET_ZERO):
        if not config.renewable.solar_pv_enabled:
            warnings.append(
                f"Retrofit ambition {config.retrofit.retrofit_ambition.value} typically "
                "requires on-site renewable generation. Consider enabling solar PV."
            )

    # Check occupancy data
    if config.typical_occupancy_persons is None:
        warnings.append(
            "No typical_occupancy_persons configured. Occupancy data improves "
            "ventilation and DHW demand calculations."
        )

    # Check certification prerequisites
    if config.certification_target != CertificationTarget.NONE:
        if config.gross_internal_area_m2 is None:
            warnings.append(
                f"Certification target {config.certification_target.value} requires "
                "accurate floor area measurement."
            )

    return warnings


def get_default_config(
    building_type: BuildingType = BuildingType.OFFICE,
) -> BuildingEnergyAssessmentConfig:
    """Get default configuration for a given building type.

    Args:
        building_type: Building type to configure for.

    Returns:
        BuildingEnergyAssessmentConfig instance with building-type-appropriate defaults.
    """
    return BuildingEnergyAssessmentConfig(building_type=building_type)


def get_building_type_info(building_type: Union[str, BuildingType]) -> Dict[str, Any]:
    """Get detailed information about a building type.

    Args:
        building_type: Building type enum or string value.

    Returns:
        Dictionary with name, typical area, EUI benchmarks, and key systems.
    """
    key = building_type.value if isinstance(building_type, BuildingType) else building_type
    return BUILDING_TYPE_INFO.get(
        key,
        {
            "name": key,
            "typical_gia_m2": "Varies",
            "typical_eui_kwh_m2_yr": "Varies",
            "good_practice_eui_kwh_m2_yr": None,
            "best_practice_eui_kwh_m2_yr": None,
            "key_systems": ["HVAC", "Lighting", "DHW"],
            "typical_energy_split": "Varies",
            "dec_benchmark": "N/A",
        },
    )


def get_u_value_targets(climate_zone: Union[str, ClimateZone]) -> Dict[str, float]:
    """Get U-value targets for a climate zone.

    Args:
        climate_zone: Climate zone enum or string value.

    Returns:
        Dictionary with wall, roof, floor, and window U-value targets (W/m2K).
    """
    key = climate_zone.value if isinstance(climate_zone, ClimateZone) else climate_zone
    return U_VALUE_TARGETS.get(
        key,
        {"wall": 0.30, "roof": 0.20, "floor": 0.25, "window": 1.6},
    )


def get_lpd_standard(area_type: str = "office_open_plan") -> float:
    """Get Lighting Power Density standard for a building area type.

    Args:
        area_type: Area type (office_cellular, retail_general, hotel_room, etc.).

    Returns:
        LPD target in W/m2 per EN 15193-1 / EN 12464-1.
    """
    return LPD_BUILDING_STANDARDS.get(area_type, LPD_BUILDING_STANDARDS["office_open_plan"])


def get_airtightness_benchmark(building_category: str = "new_build_non_domestic") -> Dict[str, float]:
    """Get airtightness benchmark values for a building category.

    Args:
        building_category: Building category key (new_build_residential, existing_pre_1980, etc.).

    Returns:
        Dictionary with benchmark values (m3/h/m2 @ 50Pa).
    """
    return AIRTIGHTNESS_BENCHMARKS.get(
        building_category,
        AIRTIGHTNESS_BENCHMARKS["new_build_non_domestic"],
    )


def get_climate_zone_for_country(country: str) -> Optional[str]:
    """Get the expected climate zone for a given country code.

    Args:
        country: ISO 3166-1 alpha-2 country code.

    Returns:
        Climate zone string, or None if country not mapped.
    """
    return COUNTRY_CLIMATE_ZONE.get(country)


def list_available_presets() -> Dict[str, str]:
    """List all available configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return AVAILABLE_PRESETS.copy()
