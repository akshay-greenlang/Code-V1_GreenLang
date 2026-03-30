"""
PACK-027 Enterprise Net Zero Pack - Configuration Manager

This module implements the EnterpriseNetZeroConfig and PackConfig classes that
load, merge, and validate all configuration for the Enterprise Net Zero Pack.
It provides comprehensive Pydantic v2 models designed for large enterprises
with complex organizational structures, financial-grade data quality requirements,
multi-entity consolidation, and full SBTi Corporate Standard compliance.

Enterprise Size Definition:
    - >250 employees AND >$50M / EUR 50M annual revenue
    - Complex corporate structures: 10-500+ subsidiaries, JVs, associates
    - Multi-country operations across 30+ jurisdictions
    - 50,000-500,000+ employees modeled

Enterprise Sectors (20+):
    - MANUFACTURING, ENERGY_UTILITIES, FINANCIAL_SERVICES, TECHNOLOGY
    - CONSUMER_GOODS, TRANSPORT_LOGISTICS, REAL_ESTATE, HEALTHCARE_PHARMA
    - MINING_METALS, CHEMICALS, TELECOMMUNICATIONS, MEDIA_ENTERTAINMENT
    - AEROSPACE_DEFENSE, AUTOMOTIVE, FOOD_BEVERAGE, AGRICULTURE
    - CONSTRUCTION, HOSPITALITY_LEISURE, EDUCATION, PUBLIC_SECTOR, OTHER

Consolidation Approaches (GHG Protocol Chapter 3):
    - FINANCIAL_CONTROL: 100% of entities with financial control
    - OPERATIONAL_CONTROL: 100% of entities with operational control
    - EQUITY_SHARE: Proportional to equity ownership percentage

SBTi Pathways:
    - ACA: Absolute Contraction Approach (4.2%/yr for 1.5C, 2.5%/yr for WB2C)
    - SDA: Sectoral Decarbonization Approach (12 sectors, intensity convergence)
    - FLAG: Forest, Land and Agriculture targets (3.03%/yr if >20% land use)
    - MIXED: ACA for general + SDA for specific divisions + FLAG if applicable

Data Quality Tiers (GHG Protocol 5-level hierarchy):
    - LEVEL_1: Supplier-specific, verified (CDP, PACT) -- +/-3% accuracy
    - LEVEL_2: Supplier-specific, unverified (questionnaire) -- +/-5-10%
    - LEVEL_3: Average data, physical units -- +/-10-20%
    - LEVEL_4: Spend-based EEIO -- +/-20-40%
    - LEVEL_5: Proxy/extrapolation -- +/-40-60%

Assurance Levels (ISAE 3410 / ISO 14064-3):
    - LIMITED: Negative assurance ("nothing has come to our attention...")
    - REASONABLE: Positive assurance ("in our opinion, the GHG statement...")

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (manufacturing / financial_services / technology / etc.)
    3. Environment overrides (ENTERPRISE_NET_ZERO_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - GHG Protocol Corporate Standard (2004, amended 2015)
    - GHG Protocol Scope 2 Guidance (2015)
    - GHG Protocol Corporate Value Chain (Scope 3) Standard (2011)
    - SBTi Corporate Manual V5.3 (2024) -- 28 near-term criteria (C1-C28)
    - SBTi Corporate Net-Zero Standard V1.3 (2024) -- 14 net-zero criteria
    - SBTi FLAG Guidance V1.1 (2022)
    - SBTi SDA Tool V3.0 (2024) -- 12 sector pathways
    - IPCC AR6 GWP100 values
    - SEC Climate Disclosure Rule S7-10-22 (2024)
    - CSRD / ESRS E1 (Directive 2022/2464)
    - California SB 253 / SB 261 (2023)
    - ISSB S2 (IFRS S2, 2023)
    - CDP Climate Change Questionnaire (2024/2025)
    - TCFD Recommendations (2017, final 2023)
    - ISO 14064-1:2018, ISO 14064-3:2019
    - ISAE 3410 (2012), ISAE 3000 (Revised, 2013)
    - PCAF Global GHG Accounting Standard (2022)
    - WBCSD Avoided Emissions Guidance (2023)
    - VCMI Claims Code (2023), Oxford Principles (2020)
    - IEA Net Zero Roadmap (2023), TPI Carbon Performance (2024)
    - EU Taxonomy Climate Delegated Act (2021/2139)

Example:
    >>> config = PackConfig.from_preset("manufacturing")
    >>> print(config.pack.organization.sector)
    EnterpriseSector.MANUFACTURING
    >>> print(config.pack.consolidation.approach)
    ConsolidationApproach.FINANCIAL_CONTROL
    >>> print(config.pack.target.sbti_pathway)
    SBTiPathway.MIXED
"""

import copy
import hashlib
import logging
import os
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from greenlang.schemas.enums import ReportFormat

logger = logging.getLogger(__name__)

# Base directory for all pack configuration files
PACK_BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent

# =============================================================================
# Constants
# =============================================================================

DEFAULT_BASE_YEAR: int = 2024
DEFAULT_TARGET_YEAR_NEAR: int = 2030
DEFAULT_TARGET_YEAR_LONG: int = 2050
DEFAULT_REPORTING_YEAR: int = 2025

# All 15 Scope 3 categories for enterprise
ALL_SCOPE3_CATEGORIES: List[int] = list(range(1, 16))

# Enterprise scale thresholds
ENTERPRISE_THRESHOLDS: Dict[str, Any] = {
    "min_employees": 250,
    "min_revenue_eur": 50_000_000,
    "max_entities": 500,
    "max_suppliers": 100_000,
    "max_facilities": 5_000,
    "max_employees_modeled": 500_000,
    "description": "Large enterprise (>250 employees, >EUR 50M revenue)",
}

SUPPORTED_PRESETS: Dict[str, str] = {
    "manufacturing": "Heavy industry, process emissions, SDA pathways, ETS/CBAM exposure",
    "financial_services": "Low direct emissions, PCAF financed emissions (Cat 15), FINZ targets",
    "technology": "Data center Scope 2, hardware Scope 3 Cat 1, RE100, avoided emissions",
    "energy_utilities": "Very high Scope 1, SDA mandatory, stranded assets, renewable transition",
    "retail_consumer": "Supply chain dominant (Cat 1), use-phase (Cat 11), FLAG for agri inputs",
    "healthcare": "Mixed S1 (labs) + high S3 (Cat 1 procurement), cold chain, anesthetic gases",
    "transport_logistics": "SDA transport, fleet electrification, alternative fuels, modal shift",
    "agriculture_food": "FLAG mandatory, land use + enteric + manure, farm-to-fork supply chain",
}

# =============================================================================
# Enums - Enterprise-specific enumeration types (22 enums)
# =============================================================================


class EnterpriseSector(str, Enum):
    """Enterprise sector classification for configuration preset selection.

    Based on NACE/GICS sector taxonomy aligned with SBTi sector coverage
    and SDA pathway availability.
    """

    MANUFACTURING = "MANUFACTURING"
    ENERGY_UTILITIES = "ENERGY_UTILITIES"
    FINANCIAL_SERVICES = "FINANCIAL_SERVICES"
    TECHNOLOGY = "TECHNOLOGY"
    CONSUMER_GOODS = "CONSUMER_GOODS"
    TRANSPORT_LOGISTICS = "TRANSPORT_LOGISTICS"
    REAL_ESTATE = "REAL_ESTATE"
    HEALTHCARE_PHARMA = "HEALTHCARE_PHARMA"
    MINING_METALS = "MINING_METALS"
    CHEMICALS = "CHEMICALS"
    TELECOMMUNICATIONS = "TELECOMMUNICATIONS"
    MEDIA_ENTERTAINMENT = "MEDIA_ENTERTAINMENT"
    AEROSPACE_DEFENSE = "AEROSPACE_DEFENSE"
    AUTOMOTIVE = "AUTOMOTIVE"
    FOOD_BEVERAGE = "FOOD_BEVERAGE"
    AGRICULTURE = "AGRICULTURE"
    CONSTRUCTION = "CONSTRUCTION"
    HOSPITALITY_LEISURE = "HOSPITALITY_LEISURE"
    EDUCATION = "EDUCATION"
    PUBLIC_SECTOR = "PUBLIC_SECTOR"
    PROFESSIONAL_SERVICES = "PROFESSIONAL_SERVICES"
    OTHER = "OTHER"


class ConsolidationApproach(str, Enum):
    """GHG Protocol organizational boundary consolidation approach.

    Per GHG Protocol Corporate Standard Chapter 3. Choice of approach
    materially affects total reported emissions (20-40% variance typical).
    """

    FINANCIAL_CONTROL = "FINANCIAL_CONTROL"
    OPERATIONAL_CONTROL = "OPERATIONAL_CONTROL"
    EQUITY_SHARE = "EQUITY_SHARE"


class SBTiPathway(str, Enum):
    """SBTi target-setting pathway type.

    ACA: Absolute Contraction Approach -- 4.2%/yr (1.5C) or 2.5%/yr (WB2C)
    SDA: Sectoral Decarbonization Approach -- intensity convergence (12 sectors)
    FLAG: Forest, Land and Agriculture -- 3.03%/yr for land-use-intensive orgs
    MIXED: Combination of ACA, SDA, and/or FLAG for diversified enterprises
    """

    ACA_15C = "ACA_15C"
    ACA_WB2C = "ACA_WB2C"
    SDA = "SDA"
    FLAG = "FLAG"
    MIXED = "MIXED"


class AssuranceLevel(str, Enum):
    """External assurance level per ISAE 3410 / ISO 14064-3.

    LIMITED: Negative assurance -- lower evidence threshold, inquiry + analytics
    REASONABLE: Positive assurance -- higher evidence threshold, substantive testing
    """

    LIMITED = "LIMITED"
    REASONABLE = "REASONABLE"


class DataQualityTier(str, Enum):
    """GHG Protocol 5-level data quality hierarchy for Scope 3 calculations.

    LEVEL_1: Supplier-specific, verified (CDP, PACT exchange) -- +/-3%
    LEVEL_2: Supplier-specific, unverified (questionnaire) -- +/-5-10%
    LEVEL_3: Average data, physical units (industry factors) -- +/-10-20%
    LEVEL_4: Spend-based EEIO (economic input-output) -- +/-20-40%
    LEVEL_5: Proxy / extrapolation (revenue, headcount) -- +/-40-60%
    """

    LEVEL_1 = "LEVEL_1"
    LEVEL_2 = "LEVEL_2"
    LEVEL_3 = "LEVEL_3"
    LEVEL_4 = "LEVEL_4"
    LEVEL_5 = "LEVEL_5"


class Scope3Method(str, Enum):
    """Scope 3 calculation methodology for enterprise use."""

    SUPPLIER_SPECIFIC = "SUPPLIER_SPECIFIC"
    HYBRID = "HYBRID"
    AVERAGE_DATA = "AVERAGE_DATA"
    SPEND_BASED = "SPEND_BASED"
    ACTIVITY_BASED = "ACTIVITY_BASED"


class CarbonPricingApproach(str, Enum):
    """Internal carbon pricing methodology."""

    SHADOW_PRICE = "SHADOW_PRICE"
    INTERNAL_FEE = "INTERNAL_FEE"
    IMPLICIT = "IMPLICIT"
    REGULATORY = "REGULATORY"
    HYBRID = "HYBRID"


class ERPSystem(str, Enum):
    """Enterprise ERP system integration options."""

    SAP_S4HANA = "SAP_S4HANA"
    SAP_ECC = "SAP_ECC"
    ORACLE_ERP_CLOUD = "ORACLE_ERP_CLOUD"
    ORACLE_EBUSINESS = "ORACLE_EBUSINESS"
    WORKDAY = "WORKDAY"
    DYNAMICS_365 = "DYNAMICS_365"
    INFOR = "INFOR"
    SAGE_X3 = "SAGE_X3"
    CUSTOM = "CUSTOM"
    NONE = "NONE"


class AmbitionLevel(str, Enum):
    """SBTi target ambition level."""

    CELSIUS_1_5 = "CELSIUS_1_5"
    WELL_BELOW_2C = "WELL_BELOW_2C"


class SupplierEngagementTier(str, Enum):
    """Supplier engagement tier for Scope 3 reduction."""

    COLLABORATE = "COLLABORATE"
    REQUIRE = "REQUIRE"
    ENGAGE = "ENGAGE"
    INFORM = "INFORM"


class EmissionsProfileType(str, Enum):
    """Typical emissions profile types for enterprise sectors."""

    SCOPE1_DOMINANT = "SCOPE1_DOMINANT"
    SCOPE2_DOMINANT = "SCOPE2_DOMINANT"
    SCOPE3_DOMINANT = "SCOPE3_DOMINANT"
    BALANCED = "BALANCED"
    INVESTMENTS_DOMINANT = "INVESTMENTS_DOMINANT"


class ScenarioType(str, Enum):
    """Climate scenario types for enterprise modeling."""

    AGGRESSIVE_1_5C = "AGGRESSIVE_1_5C"
    MODERATE_2C = "MODERATE_2C"
    CONSERVATIVE_BAU = "CONSERVATIVE_BAU"
    CUSTOM = "CUSTOM"


class SDASector(str, Enum):
    """SBTi Sectoral Decarbonization Approach available sectors."""

    POWER_GENERATION = "POWER_GENERATION"
    CEMENT = "CEMENT"
    IRON_STEEL = "IRON_STEEL"
    ALUMINIUM = "ALUMINIUM"
    PULP_PAPER = "PULP_PAPER"
    CHEMICALS = "CHEMICALS"
    AVIATION = "AVIATION"
    MARITIME_SHIPPING = "MARITIME_SHIPPING"
    ROAD_TRANSPORT = "ROAD_TRANSPORT"
    COMMERCIAL_BUILDINGS = "COMMERCIAL_BUILDINGS"
    RESIDENTIAL_BUILDINGS = "RESIDENTIAL_BUILDINGS"
    FOOD_BEVERAGE = "FOOD_BEVERAGE"


class MaturityAssessment(str, Enum):
    """Enterprise net-zero maturity assessment depth."""

    COMPREHENSIVE = "COMPREHENSIVE"
    STANDARD = "STANDARD"
    SKIP = "SKIP"


class RegulatoryFramework(str, Enum):
    """Regulatory frameworks for enterprise compliance."""

    GHG_PROTOCOL = "GHG_PROTOCOL"
    SBTI = "SBTI"
    CDP = "CDP"
    TCFD_ISSB = "TCFD_ISSB"
    SEC_CLIMATE = "SEC_CLIMATE"
    CSRD_ESRS = "CSRD_ESRS"
    ISO_14064 = "ISO_14064"
    CA_SB253 = "CA_SB253"
    CA_SB261 = "CA_SB261"
    PCAF = "PCAF"
    EU_TAXONOMY = "EU_TAXONOMY"


class PCAFAssetClass(str, Enum):
    """PCAF asset classes for financed emissions (financial services)."""

    LISTED_EQUITY = "LISTED_EQUITY"
    CORPORATE_BONDS = "CORPORATE_BONDS"
    BUSINESS_LOANS = "BUSINESS_LOANS"
    MORTGAGES = "MORTGAGES"
    COMMERCIAL_REAL_ESTATE = "COMMERCIAL_REAL_ESTATE"
    PROJECT_FINANCE = "PROJECT_FINANCE"
    SOVEREIGN_BONDS = "SOVEREIGN_BONDS"
    MOTOR_VEHICLE_LOANS = "MOTOR_VEHICLE_LOANS"


class FLAGCommodity(str, Enum):
    """FLAG commodity categories for land use targets."""

    PALM_OIL = "PALM_OIL"
    SOY = "SOY"
    COCOA = "COCOA"
    COFFEE = "COFFEE"
    COTTON = "COTTON"
    RUBBER = "RUBBER"
    TIMBER = "TIMBER"
    CATTLE = "CATTLE"
    DAIRY = "DAIRY"
    RICE = "RICE"
    WHEAT = "WHEAT"


# =============================================================================
# Reference Data Constants
# =============================================================================

# Sector-specific default emissions profile split (S1%, S2%, S3%)
SECTOR_EMISSIONS_PROFILE: Dict[str, Dict[str, Any]] = {
    "MANUFACTURING": {
        "scope1_pct": 35.0,
        "scope2_pct": 25.0,
        "scope3_pct": 40.0,
        "profile_type": "SCOPE1_DOMINANT",
        "primary_scope3_categories": [1, 2, 3, 4, 11, 12],
        "sda_applicable": True,
        "sda_sectors": ["CEMENT", "IRON_STEEL", "ALUMINIUM", "CHEMICALS"],
    },
    "ENERGY_UTILITIES": {
        "scope1_pct": 55.0,
        "scope2_pct": 5.0,
        "scope3_pct": 40.0,
        "profile_type": "SCOPE1_DOMINANT",
        "primary_scope3_categories": [1, 3, 9, 10, 11],
        "sda_applicable": True,
        "sda_sectors": ["POWER_GENERATION"],
    },
    "FINANCIAL_SERVICES": {
        "scope1_pct": 2.0,
        "scope2_pct": 3.0,
        "scope3_pct": 95.0,
        "profile_type": "INVESTMENTS_DOMINANT",
        "primary_scope3_categories": [15, 1, 6, 7],
        "sda_applicable": False,
        "sda_sectors": [],
    },
    "TECHNOLOGY": {
        "scope1_pct": 3.0,
        "scope2_pct": 22.0,
        "scope3_pct": 75.0,
        "profile_type": "SCOPE3_DOMINANT",
        "primary_scope3_categories": [1, 2, 3, 11, 12],
        "sda_applicable": False,
        "sda_sectors": [],
    },
    "CONSUMER_GOODS": {
        "scope1_pct": 8.0,
        "scope2_pct": 7.0,
        "scope3_pct": 85.0,
        "profile_type": "SCOPE3_DOMINANT",
        "primary_scope3_categories": [1, 4, 9, 11, 12],
        "sda_applicable": False,
        "sda_sectors": [],
    },
    "TRANSPORT_LOGISTICS": {
        "scope1_pct": 50.0,
        "scope2_pct": 5.0,
        "scope3_pct": 45.0,
        "profile_type": "SCOPE1_DOMINANT",
        "primary_scope3_categories": [3, 1, 4, 9],
        "sda_applicable": True,
        "sda_sectors": ["AVIATION", "MARITIME_SHIPPING", "ROAD_TRANSPORT"],
    },
    "REAL_ESTATE": {
        "scope1_pct": 10.0,
        "scope2_pct": 15.0,
        "scope3_pct": 75.0,
        "profile_type": "SCOPE3_DOMINANT",
        "primary_scope3_categories": [13, 1, 2, 11],
        "sda_applicable": True,
        "sda_sectors": ["COMMERCIAL_BUILDINGS", "RESIDENTIAL_BUILDINGS"],
    },
    "HEALTHCARE_PHARMA": {
        "scope1_pct": 15.0,
        "scope2_pct": 15.0,
        "scope3_pct": 70.0,
        "profile_type": "SCOPE3_DOMINANT",
        "primary_scope3_categories": [1, 2, 4, 6, 12],
        "sda_applicable": False,
        "sda_sectors": [],
    },
    "MINING_METALS": {
        "scope1_pct": 40.0,
        "scope2_pct": 25.0,
        "scope3_pct": 35.0,
        "profile_type": "SCOPE1_DOMINANT",
        "primary_scope3_categories": [1, 3, 4, 10],
        "sda_applicable": True,
        "sda_sectors": ["IRON_STEEL", "ALUMINIUM"],
    },
    "CHEMICALS": {
        "scope1_pct": 35.0,
        "scope2_pct": 20.0,
        "scope3_pct": 45.0,
        "profile_type": "SCOPE1_DOMINANT",
        "primary_scope3_categories": [1, 3, 4, 10, 11],
        "sda_applicable": True,
        "sda_sectors": ["CHEMICALS"],
    },
    "TELECOMMUNICATIONS": {
        "scope1_pct": 5.0,
        "scope2_pct": 25.0,
        "scope3_pct": 70.0,
        "profile_type": "SCOPE3_DOMINANT",
        "primary_scope3_categories": [1, 2, 11, 13],
        "sda_applicable": False,
        "sda_sectors": [],
    },
    "AUTOMOTIVE": {
        "scope1_pct": 20.0,
        "scope2_pct": 15.0,
        "scope3_pct": 65.0,
        "profile_type": "SCOPE3_DOMINANT",
        "primary_scope3_categories": [1, 4, 11, 12],
        "sda_applicable": True,
        "sda_sectors": ["ROAD_TRANSPORT"],
    },
    "FOOD_BEVERAGE": {
        "scope1_pct": 15.0,
        "scope2_pct": 10.0,
        "scope3_pct": 75.0,
        "profile_type": "SCOPE3_DOMINANT",
        "primary_scope3_categories": [1, 4, 9, 12],
        "sda_applicable": True,
        "sda_sectors": ["FOOD_BEVERAGE"],
    },
    "AGRICULTURE": {
        "scope1_pct": 45.0,
        "scope2_pct": 5.0,
        "scope3_pct": 50.0,
        "profile_type": "SCOPE1_DOMINANT",
        "primary_scope3_categories": [1, 3, 4, 10],
        "sda_applicable": False,
        "sda_sectors": [],
    },
    "CONSTRUCTION": {
        "scope1_pct": 25.0,
        "scope2_pct": 10.0,
        "scope3_pct": 65.0,
        "profile_type": "SCOPE3_DOMINANT",
        "primary_scope3_categories": [1, 2, 4, 5],
        "sda_applicable": True,
        "sda_sectors": ["CEMENT"],
    },
    "AEROSPACE_DEFENSE": {
        "scope1_pct": 20.0,
        "scope2_pct": 15.0,
        "scope3_pct": 65.0,
        "profile_type": "SCOPE3_DOMINANT",
        "primary_scope3_categories": [1, 2, 4, 11],
        "sda_applicable": True,
        "sda_sectors": ["AVIATION"],
    },
    "HOSPITALITY_LEISURE": {
        "scope1_pct": 20.0,
        "scope2_pct": 25.0,
        "scope3_pct": 55.0,
        "profile_type": "BALANCED",
        "primary_scope3_categories": [1, 3, 5, 14],
        "sda_applicable": True,
        "sda_sectors": ["COMMERCIAL_BUILDINGS"],
    },
    "PROFESSIONAL_SERVICES": {
        "scope1_pct": 5.0,
        "scope2_pct": 10.0,
        "scope3_pct": 85.0,
        "profile_type": "SCOPE3_DOMINANT",
        "primary_scope3_categories": [1, 6, 7, 8],
        "sda_applicable": False,
        "sda_sectors": [],
    },
    "OTHER": {
        "scope1_pct": 20.0,
        "scope2_pct": 15.0,
        "scope3_pct": 65.0,
        "profile_type": "SCOPE3_DOMINANT",
        "primary_scope3_categories": [1, 2, 3, 4, 6, 7],
        "sda_applicable": False,
        "sda_sectors": [],
    },
}

# IPCC AR6 GWP100 values for all seven Kyoto gases + common HFCs
IPCC_AR6_GWP100: Dict[str, int] = {
    "CO2": 1,
    "CH4": 27,
    "N2O": 273,
    "HFC_23": 14600,
    "HFC_32": 771,
    "HFC_125": 3740,
    "HFC_134A": 1430,
    "HFC_143A": 5810,
    "HFC_152A": 164,
    "HFC_227EA": 3600,
    "HFC_236FA": 8690,
    "HFC_245FA": 962,
    "HFC_365MFC": 804,
    "HFC_4310MEE": 1650,
    "R404A": 3922,
    "R407C": 1774,
    "R410A": 2088,
    "R507A": 3985,
    "PFC_14": 7380,
    "PFC_116": 12400,
    "PFC_218": 9290,
    "PFC_318": 10200,
    "SF6": 25200,
    "NF3": 17400,
}

# SBTi Corporate Standard parameters
SBTI_CORPORATE_PARAMETERS: Dict[str, Any] = {
    "aca_15c_annual_reduction_pct": 4.2,
    "aca_wb2c_annual_reduction_pct": 2.5,
    "scope1_2_coverage_pct": 95.0,
    "scope3_near_term_coverage_pct": 67.0,
    "scope3_long_term_coverage_pct": 90.0,
    "near_term_horizon_min_years": 5,
    "near_term_horizon_max_years": 10,
    "net_zero_year_max": 2050,
    "long_term_reduction_min_pct": 90.0,
    "residual_emissions_max_pct": 10.0,
    "base_year_recalculation_threshold_pct": 5.0,
    "flag_threshold_pct": 20.0,
    "flag_annual_reduction_pct": 3.03,
    "flag_no_deforestation_by": 2025,
    "near_term_criteria_count": 28,
    "net_zero_criteria_count": 14,
    "total_criteria_count": 42,
    "five_year_review_required": True,
}

# SDA sector benchmarks (2030 and 2050 targets)
SDA_SECTOR_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "POWER_GENERATION": {
        "intensity_metric": "tCO2/MWh",
        "target_2030": 0.14,
        "target_2050": 0.00,
        "source": "IEA NZE",
    },
    "CEMENT": {
        "intensity_metric": "tCO2/t cement",
        "target_2030": 0.42,
        "target_2050": 0.07,
        "source": "SBTi SDA Tool",
    },
    "IRON_STEEL": {
        "intensity_metric": "tCO2/t crude steel",
        "target_2030": 1.06,
        "target_2050": 0.05,
        "source": "SBTi SDA Tool",
    },
    "ALUMINIUM": {
        "intensity_metric": "tCO2/t aluminium",
        "target_2030": 3.10,
        "target_2050": 0.20,
        "source": "SBTi SDA Tool",
    },
    "PULP_PAPER": {
        "intensity_metric": "tCO2/t product",
        "target_2030": 0.22,
        "target_2050": 0.04,
        "source": "SBTi SDA Tool",
    },
    "AVIATION": {
        "intensity_metric": "gCO2/pkm",
        "target_2030": 62.0,
        "target_2050": 8.0,
        "source": "SBTi SDA Tool",
    },
    "MARITIME_SHIPPING": {
        "intensity_metric": "gCO2/tkm",
        "target_2030": 5.8,
        "target_2050": 0.8,
        "source": "SBTi SDA Tool",
    },
    "ROAD_TRANSPORT": {
        "intensity_metric": "gCO2/vkm",
        "target_2030": 85.0,
        "target_2050": 0.0,
        "source": "SBTi SDA Tool",
    },
    "COMMERCIAL_BUILDINGS": {
        "intensity_metric": "kgCO2/sqm",
        "target_2030": 25.0,
        "target_2050": 2.0,
        "source": "SBTi SDA Tool",
    },
    "RESIDENTIAL_BUILDINGS": {
        "intensity_metric": "kgCO2/sqm",
        "target_2030": 12.0,
        "target_2050": 1.0,
        "source": "SBTi SDA Tool",
    },
    "FOOD_BEVERAGE": {
        "intensity_metric": "tCO2/t product",
        "target_2030": None,
        "target_2050": None,
        "source": "SBTi SDA Tool (varies by sub-sector)",
    },
    "CHEMICALS": {
        "intensity_metric": "tCO2/t product",
        "target_2030": None,
        "target_2050": None,
        "source": "SBTi SDA Tool (varies by chemical type)",
    },
}

# Carbon price trajectory scenarios (USD/tCO2e)
CARBON_PRICE_SCENARIOS: Dict[str, Dict[int, float]] = {
    "LOW": {2025: 30, 2030: 50, 2035: 75, 2040: 100, 2050: 150},
    "MEDIUM": {2025: 60, 2030: 100, 2035: 150, 2040: 200, 2050: 300},
    "HIGH": {2025: 100, 2030: 200, 2035: 300, 2040: 400, 2050: 500},
}

# Enterprise report templates (10 templates)
ENTERPRISE_REPORT_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "ghg_inventory_report": {
        "name": "GHG Protocol Inventory Report",
        "description": "Full GHG Protocol Corporate Standard report with S1/S2/S3 by entity",
        "pages": "20-40",
        "formats": ["PDF", "EXCEL", "HTML", "JSON", "MARKDOWN"],
    },
    "sbti_target_submission": {
        "name": "SBTi Target Submission Package",
        "description": "Complete SBTi submission with 42 criteria validation",
        "pages": "15-25",
        "formats": ["PDF", "HTML", "JSON", "MARKDOWN"],
    },
    "cdp_climate_response": {
        "name": "CDP Climate Change Response",
        "description": "Full CDP questionnaire response (C0-C15 modules)",
        "pages": "40-80",
        "formats": ["HTML", "JSON", "MARKDOWN"],
    },
    "tcfd_report": {
        "name": "TCFD / ISSB S2 Disclosure",
        "description": "Four-pillar TCFD report with scenario analysis",
        "pages": "15-30",
        "formats": ["PDF", "HTML", "JSON", "MARKDOWN"],
    },
    "executive_dashboard": {
        "name": "Executive Climate Dashboard",
        "description": "Board-level 15-20 KPI dashboard with traffic-light status",
        "pages": "1-3",
        "formats": ["HTML", "JSON", "MARKDOWN"],
    },
    "supply_chain_heatmap": {
        "name": "Supply Chain Emissions Heatmap",
        "description": "Tier 1/2/3 supplier heatmap with scorecards for top 50",
        "pages": "10-20",
        "formats": ["HTML", "JSON", "MARKDOWN"],
    },
    "scenario_comparison": {
        "name": "Scenario Comparison Report",
        "description": "1.5C vs. 2C vs. BAU with fan charts and sensitivity analysis",
        "pages": "10-15",
        "formats": ["HTML", "JSON", "MARKDOWN"],
    },
    "assurance_statement": {
        "name": "Assurance Statement Template",
        "description": "ISO 14064-3 assurance statement (limited/reasonable)",
        "pages": "5-10",
        "formats": ["PDF", "HTML", "JSON", "MARKDOWN"],
    },
    "board_climate_report": {
        "name": "Quarterly Board Climate Report",
        "description": "Board paper covering performance, initiatives, compliance, risk",
        "pages": "5-10",
        "formats": ["PDF", "HTML", "JSON", "MARKDOWN"],
    },
    "regulatory_filings": {
        "name": "Multi-Framework Regulatory Filings",
        "description": "SEC, CSRD ESRS E1, CA SB 253, ISO 14064-1, CDP extract",
        "pages": "30-60",
        "formats": ["PDF", "EXCEL", "HTML", "JSON", "MARKDOWN"],
    },
}

# Sector display names and enterprise-specific guidance
ENTERPRISE_SECTOR_INFO: Dict[str, Dict[str, Any]] = {
    "MANUFACTURING": {
        "name": "Manufacturing & Industrial",
        "typical_scope_split": "S1: 35%, S2: 25%, S3: 40%",
        "profile_type": "SCOPE1_DOMINANT",
        "key_levers": [
            "process electrification",
            "hydrogen fuel switching",
            "carbon capture (CCS/CCUS)",
            "energy efficiency at scale",
            "renewable PPAs",
            "supplier engagement (Cat 1/2)",
        ],
        "sbti_pathway": "MIXED (ACA + SDA for applicable divisions)",
        "target_organizations": [
            "Siemens", "BASF", "3M", "Honeywell", "ABB", "Schneider Electric",
        ],
    },
    "ENERGY_UTILITIES": {
        "name": "Energy & Utilities",
        "typical_scope_split": "S1: 55%, S2: 5%, S3: 40%",
        "profile_type": "SCOPE1_DOMINANT",
        "key_levers": [
            "renewable energy transition",
            "gas-to-zero strategy",
            "grid decarbonization",
            "stranded asset management",
            "battery storage deployment",
            "green hydrogen production",
        ],
        "sbti_pathway": "SDA (mandatory for power generation)",
        "target_organizations": [
            "Shell", "TotalEnergies", "Enel", "Iberdrola", "NextEra Energy",
        ],
    },
    "FINANCIAL_SERVICES": {
        "name": "Financial Services",
        "typical_scope_split": "S1: 2%, S2: 3%, S3: 95% (Cat 15 dominant)",
        "profile_type": "INVESTMENTS_DOMINANT",
        "key_levers": [
            "portfolio decarbonization (PCAF)",
            "green bond/loan underwriting",
            "borrower climate engagement",
            "portfolio temperature scoring",
            "Taxonomy-aligned lending",
            "stranded asset divestment",
        ],
        "sbti_pathway": "ACA (direct) + FINZ (portfolio)",
        "target_organizations": [
            "JPMorgan", "HSBC", "Allianz", "BlackRock", "BNP Paribas",
        ],
    },
    "TECHNOLOGY": {
        "name": "Technology & Software",
        "typical_scope_split": "S1: 3%, S2: 22%, S3: 75%",
        "profile_type": "SCOPE3_DOMINANT",
        "key_levers": [
            "renewable energy (RE100)",
            "data center PUE optimization",
            "hardware supply chain engagement",
            "cloud carbon efficiency",
            "product energy efficiency (Cat 11)",
            "avoided emissions (SaaS vs. on-premise)",
        ],
        "sbti_pathway": "ACA (1.5C)",
        "target_organizations": [
            "Microsoft", "Google", "Apple", "SAP", "Salesforce",
        ],
    },
    "CONSUMER_GOODS": {
        "name": "Consumer Goods / FMCG",
        "typical_scope_split": "S1: 8%, S2: 7%, S3: 85%",
        "profile_type": "SCOPE3_DOMINANT",
        "key_levers": [
            "supply chain decarbonization (Cat 1)",
            "product reformulation/lightweighting",
            "packaging circularity",
            "FLAG commodity sourcing",
            "consumer use-phase reduction",
            "end-of-life design for recycling",
        ],
        "sbti_pathway": "MIXED (ACA + FLAG for agri inputs)",
        "target_organizations": [
            "Unilever", "P&G", "Nestle", "L'Oreal", "Nike",
        ],
    },
    "TRANSPORT_LOGISTICS": {
        "name": "Transport & Logistics",
        "typical_scope_split": "S1: 50%, S2: 5%, S3: 45%",
        "profile_type": "SCOPE1_DOMINANT",
        "key_levers": [
            "fleet electrification (road)",
            "sustainable aviation fuel (SAF)",
            "alternative marine fuels (ammonia, methanol)",
            "modal shift (road to rail)",
            "route optimization",
            "last-mile EV delivery",
        ],
        "sbti_pathway": "SDA (mandatory for transport sectors)",
        "target_organizations": [
            "Maersk", "DHL", "FedEx", "Ryanair", "Delta Air Lines",
        ],
    },
    "REAL_ESTATE": {
        "name": "Real Estate & Property",
        "typical_scope_split": "S1: 10%, S2: 15%, S3: 75% (Cat 13 dominant)",
        "profile_type": "SCOPE3_DOMINANT",
        "key_levers": [
            "building retrofit (CRREM alignment)",
            "green building certification (BREEAM/LEED)",
            "on-site renewable energy",
            "smart building management systems",
            "embodied carbon (new construction)",
            "green lease clauses",
        ],
        "sbti_pathway": "SDA (buildings sector)",
        "target_organizations": [
            "Prologis", "Vonovia", "British Land", "Brookfield",
        ],
    },
    "HEALTHCARE_PHARMA": {
        "name": "Healthcare & Pharmaceuticals",
        "typical_scope_split": "S1: 15%, S2: 15%, S3: 70%",
        "profile_type": "SCOPE3_DOMINANT",
        "key_levers": [
            "anesthetic gas reduction (desflurane phase-out)",
            "API supplier engagement",
            "cold chain optimization",
            "green chemistry in R&D",
            "clinical trial emission reduction",
            "sustainable packaging (pharma)",
        ],
        "sbti_pathway": "ACA (1.5C)",
        "target_organizations": [
            "Novartis", "Roche", "Johnson & Johnson", "Pfizer",
        ],
    },
}


# =============================================================================
# Pydantic Sub-Config Models (18 models)
# =============================================================================


class EnterpriseOrganizationConfig(BaseModel):
    """Configuration for the enterprise organization profile.

    Defines the company identity, sector, scale, and operational
    characteristics that drive engine configuration.
    """

    name: str = Field(
        "",
        description="Legal entity name of the parent organization",
    )
    sector: EnterpriseSector = Field(
        EnterpriseSector.OTHER,
        description="Primary business sector (drives preset selection)",
    )
    region: str = Field(
        "GLOBAL",
        description="Primary operating region (GLOBAL, EU, NA, APAC, etc.)",
    )
    headquarters_country: str = Field(
        "US",
        description="Headquarters country (ISO 3166-1 alpha-2)",
    )
    operating_countries: List[str] = Field(
        default_factory=list,
        description="List of countries with operations (ISO 3166-1 alpha-2)",
    )
    revenue_eur: Optional[float] = Field(
        None,
        ge=0,
        description="Annual consolidated revenue in EUR",
    )
    employee_count: Optional[int] = Field(
        None,
        ge=1,
        description="Total headcount across all entities",
    )
    entity_count: int = Field(
        1,
        ge=1,
        le=500,
        description="Number of legal entities in consolidation scope",
    )
    facility_count: int = Field(
        1,
        ge=1,
        le=5000,
        description="Number of physical facilities/sites globally",
    )
    supplier_count: Optional[int] = Field(
        None,
        ge=0,
        le=500_000,
        description="Approximate number of Tier 1 suppliers",
    )
    fiscal_year_end: str = Field(
        "12-31",
        description="Fiscal year end date in MM-DD format",
    )
    nace_code: Optional[str] = Field(
        None,
        description="NACE Rev.2 industry classification code",
    )
    gics_code: Optional[str] = Field(
        None,
        description="GICS (Global Industry Classification Standard) code",
    )
    lei_code: Optional[str] = Field(
        None,
        description="Legal Entity Identifier (LEI) for regulatory filings",
    )
    erp_system: ERPSystem = Field(
        ERPSystem.NONE,
        description="Primary ERP system for data integration",
    )
    erp_secondary: Optional[ERPSystem] = Field(
        None,
        description="Secondary ERP system (if multi-ERP landscape)",
    )
    stock_exchange: Optional[str] = Field(
        None,
        description="Primary stock exchange listing (e.g., NYSE, LSE, XETRA)",
    )
    is_large_accelerated_filer: bool = Field(
        False,
        description="SEC Large Accelerated Filer status (affects SEC Climate Rule timing)",
    )
    csrd_in_scope: bool = Field(
        True,
        description="Whether entity is in scope of EU CSRD",
    )

    @field_validator("fiscal_year_end")
    @classmethod
    def validate_fiscal_year_end(cls, v: str) -> str:
        """Validate fiscal year end is in MM-DD format."""
        parts = v.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"fiscal_year_end must be in MM-DD format, got: {v}"
            )
        month, day = int(parts[0]), int(parts[1])
        if month < 1 or month > 12:
            raise ValueError(f"Invalid month: {month}. Must be 1-12.")
        if day < 1 or day > 31:
            raise ValueError(f"Invalid day: {day}. Must be 1-31.")
        return v


class ConsolidationConfig(BaseModel):
    """Configuration for multi-entity GHG consolidation.

    Per GHG Protocol Corporate Standard Chapter 3, the choice of
    consolidation approach materially affects total reported emissions.
    """

    approach: ConsolidationApproach = Field(
        ConsolidationApproach.FINANCIAL_CONTROL,
        description="GHG Protocol consolidation approach",
    )
    entity_count: int = Field(
        1,
        ge=1,
        le=500,
        description="Number of entities in consolidation scope",
    )
    include_joint_ventures: bool = Field(
        True,
        description="Include joint ventures in consolidation",
    )
    include_associates: bool = Field(
        False,
        description="Include associates (20-49% ownership) -- equity share only",
    )
    intercompany_elimination: bool = Field(
        True,
        description="Perform intercompany emission elimination to avoid double-counting",
    )
    base_year_recalculation_threshold_pct: float = Field(
        5.0,
        ge=0.0,
        le=100.0,
        description="Significance threshold (%) for base year recalculation per GHG Protocol",
    )
    mid_year_acquisition_prorata: bool = Field(
        True,
        description="Pro-rata allocation for mid-year acquisitions/divestitures",
    )
    reporting_currency: str = Field(
        "EUR",
        description="Reporting currency for financial metrics (ISO 4217)",
    )
    alignment_with_financial_reporting: bool = Field(
        True,
        description="Align GHG boundary with financial reporting boundary",
    )

    @field_validator("reporting_currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Validate reporting currency is a 3-letter ISO code."""
        if len(v) != 3 or not v.isalpha():
            raise ValueError(
                f"reporting_currency must be a 3-letter ISO 4217 code, got: {v}"
            )
        return v.upper()


class DataQualityConfig(BaseModel):
    """Configuration for enterprise data quality management.

    Enterprise target is +/-3% accuracy (financial-grade), using the
    GHG Protocol 5-level data quality hierarchy.
    """

    target_tier: DataQualityTier = Field(
        DataQualityTier.LEVEL_1,
        description="Target data quality tier for top suppliers/sources",
    )
    minimum_tier: DataQualityTier = Field(
        DataQualityTier.LEVEL_4,
        description="Minimum acceptable data quality tier",
    )
    accuracy_target_pct: float = Field(
        3.0,
        ge=1.0,
        le=40.0,
        description="Target accuracy (+/- %) for GHG calculations (enterprise: 3%)",
    )
    data_completeness_target_pct: float = Field(
        95.0,
        ge=50.0,
        le=100.0,
        description="Target data completeness percentage",
    )
    emission_factor_sources: List[str] = Field(
        default_factory=lambda: [
            "DEFRA_2024",
            "EPA_2024",
            "IPCC_AR6",
            "ecoinvent_3.10",
            "ADEME_2024",
        ],
        description="Prioritized emission factor databases",
    )
    require_supplier_specific_data: bool = Field(
        True,
        description="Require supplier-specific data for top suppliers",
    )
    supplier_specific_threshold_count: int = Field(
        50,
        ge=10,
        le=500,
        description="Number of top suppliers requiring specific data (by Scope 3 contribution)",
    )
    allow_spend_based_fallback: bool = Field(
        True,
        description="Allow spend-based EEIO as fallback for remaining suppliers",
    )
    data_improvement_plan_enabled: bool = Field(
        True,
        description="Generate data quality improvement plan per category per entity",
    )
    dq_monitoring_frequency: str = Field(
        "MONTHLY",
        description="Frequency of data quality monitoring (DAILY/WEEKLY/MONTHLY/QUARTERLY)",
    )


class ScopeConfig(BaseModel):
    """Configuration for Scope 1, 2, and 3 emissions calculation.

    Enterprises calculate all scopes with all 15 Scope 3 categories
    using activity-based methodology where available.
    """

    include_scope1: bool = Field(
        True,
        description="Include Scope 1 direct emissions (always true for enterprise)",
    )
    include_scope2: bool = Field(
        True,
        description="Include Scope 2 indirect emissions (always true for enterprise)",
    )
    include_scope3: bool = Field(
        True,
        description="Include Scope 3 value chain emissions (always true for enterprise)",
    )
    scope1_agents: List[str] = Field(
        default_factory=lambda: [
            "MRV-001", "MRV-002", "MRV-003", "MRV-004",
            "MRV-005", "MRV-006", "MRV-007", "MRV-008",
        ],
        description="Scope 1 MRV agents to activate",
    )
    scope2_methods: List[str] = Field(
        default_factory=lambda: ["location_based", "market_based"],
        description="Scope 2 calculation methods (dual reporting mandatory)",
    )
    scope2_agents: List[str] = Field(
        default_factory=lambda: [
            "MRV-009", "MRV-010", "MRV-011", "MRV-012", "MRV-013",
        ],
        description="Scope 2 MRV agents to activate",
    )
    scope3_categories: List[int] = Field(
        default_factory=lambda: ALL_SCOPE3_CATEGORIES.copy(),
        description="Scope 3 categories (1-15) to calculate (enterprise: all 15)",
    )
    scope3_priority_categories: List[int] = Field(
        default_factory=lambda: [1, 2, 3, 4, 6, 7],
        description="Priority Scope 3 categories requiring highest data quality",
    )
    scope3_default_method: Scope3Method = Field(
        Scope3Method.HYBRID,
        description="Default Scope 3 calculation methodology",
    )
    scope3_category_methods: Dict[str, str] = Field(
        default_factory=lambda: {
            "cat_1": "supplier_specific",
            "cat_2": "average_data",
            "cat_3": "activity_based",
            "cat_4": "activity_based",
            "cat_5": "activity_based",
            "cat_6": "activity_based",
            "cat_7": "hybrid",
            "cat_8": "average_data",
            "cat_9": "activity_based",
            "cat_10": "average_data",
            "cat_11": "activity_based",
            "cat_12": "average_data",
            "cat_13": "average_data",
            "cat_14": "activity_based",
            "cat_15": "pcaf",
            "default": "hybrid",
        },
        description="Calculation methodology per Scope 3 category",
    )
    scope3_materiality_threshold_pct: float = Field(
        1.0,
        ge=0.0,
        le=10.0,
        description="Threshold (%) below which a Scope 3 category may use simplified method",
    )
    scope3_total_exclusion_max_pct: float = Field(
        5.0,
        ge=0.0,
        le=10.0,
        description="Maximum total Scope 3 exclusions allowed (%)",
    )
    gases_included: List[str] = Field(
        default_factory=lambda: ["CO2", "CH4", "N2O", "HFCs", "PFCs", "SF6", "NF3"],
        description="Greenhouse gases in scope (all seven Kyoto gases for enterprise)",
    )

    @field_validator("scope3_categories")
    @classmethod
    def validate_scope3_categories(cls, v: List[int]) -> List[int]:
        """Validate Scope 3 category numbers are 1-15."""
        invalid = [c for c in v if c < 1 or c > 15]
        if invalid:
            raise ValueError(
                f"Invalid Scope 3 categories: {invalid}. Must be 1-15."
            )
        return sorted(set(v))


class SBTiTargetConfig(BaseModel):
    """Configuration for SBTi Corporate Standard target setting.

    Supports full SBTi Corporate Manual V5.3 (28 near-term criteria,
    14 net-zero criteria) plus SDA and FLAG pathways.
    """

    sbti_pathway: SBTiPathway = Field(
        SBTiPathway.ACA_15C,
        description="SBTi target-setting pathway",
    )
    ambition_level: AmbitionLevel = Field(
        AmbitionLevel.CELSIUS_1_5,
        description="SBTi target ambition level",
    )
    near_term_target_year: int = Field(
        DEFAULT_TARGET_YEAR_NEAR,
        ge=2025,
        le=2040,
        description="Near-term target year (5-10 years from submission)",
    )
    long_term_target_year: int = Field(
        DEFAULT_TARGET_YEAR_LONG,
        ge=2040,
        le=2055,
        description="Long-term / net-zero target year",
    )
    near_term_scope1_2_reduction_pct: float = Field(
        42.0,
        ge=20.0,
        le=80.0,
        description="Near-term Scope 1+2 absolute reduction target (%)",
    )
    near_term_scope3_reduction_pct: float = Field(
        25.0,
        ge=10.0,
        le=70.0,
        description="Near-term Scope 3 reduction target (%)",
    )
    long_term_reduction_pct: float = Field(
        90.0,
        ge=80.0,
        le=100.0,
        description="Long-term total reduction target (% from base year, min 90%)",
    )
    scope1_2_coverage_pct: float = Field(
        95.0,
        ge=80.0,
        le=100.0,
        description="Scope 1+2 coverage (% of total, SBTi requires >= 95%)",
    )
    scope3_near_term_coverage_pct: float = Field(
        67.0,
        ge=50.0,
        le=100.0,
        description="Scope 3 near-term coverage (%, SBTi requires >= 67%)",
    )
    scope3_long_term_coverage_pct: float = Field(
        90.0,
        ge=67.0,
        le=100.0,
        description="Scope 3 long-term coverage (%, SBTi requires >= 90%)",
    )
    sda_sectors: List[str] = Field(
        default_factory=list,
        description="SDA sectors applicable to this enterprise (if SDA or MIXED pathway)",
    )
    flag_enabled: bool = Field(
        False,
        description="Enable FLAG targets (required if >20% land use emissions)",
    )
    flag_commodities: List[str] = Field(
        default_factory=list,
        description="FLAG commodity categories (palm_oil, soy, cocoa, etc.)",
    )
    sbti_submission_planned: bool = Field(
        True,
        description="Whether formal SBTi target submission is planned",
    )
    sbti_committed: bool = Field(
        False,
        description="Whether SBTi commitment letter has been submitted",
    )
    sbti_target_validated: bool = Field(
        False,
        description="Whether SBTi targets have been validated",
    )
    five_year_review_schedule: bool = Field(
        True,
        description="Enable five-year review and revalidation schedule",
    )

    @model_validator(mode="after")
    def validate_target_years(self) -> "SBTiTargetConfig":
        """Ensure long-term target year is after near-term target year."""
        if self.long_term_target_year <= self.near_term_target_year:
            raise ValueError(
                f"long_term_target_year ({self.long_term_target_year}) must be "
                f"after near_term_target_year ({self.near_term_target_year})"
            )
        return self

    @model_validator(mode="after")
    def validate_sda_if_needed(self) -> "SBTiTargetConfig":
        """Warn if SDA pathway selected but no sectors specified."""
        if self.sbti_pathway in (SBTiPathway.SDA, SBTiPathway.MIXED):
            if not self.sda_sectors:
                logger.warning(
                    "SDA or MIXED pathway selected but no sda_sectors specified. "
                    "SDA sectors are required for intensity convergence validation."
                )
        return self


class ScenarioModelingConfig(BaseModel):
    """Configuration for Monte Carlo scenario modeling.

    Enterprise-grade scenario analysis with 10,000+ simulation runs
    across 1.5C, 2C, and BAU pathways.
    """

    enabled: bool = Field(
        True,
        description="Enable scenario modeling engine",
    )
    scenarios: List[str] = Field(
        default_factory=lambda: ["AGGRESSIVE_1_5C", "MODERATE_2C", "CONSERVATIVE_BAU"],
        description="Scenarios to model",
    )
    monte_carlo_runs: int = Field(
        10_000,
        ge=1_000,
        le=100_000,
        description="Number of Monte Carlo simulation runs per scenario",
    )
    confidence_intervals: List[int] = Field(
        default_factory=lambda: [10, 25, 50, 75, 90],
        description="Percentile confidence intervals to calculate",
    )
    sensitivity_method: str = Field(
        "sobol",
        description="Sensitivity analysis method (sobol, tornado, morris)",
    )
    technology_focus: List[str] = Field(
        default_factory=lambda: [
            "electrification",
            "renewable_energy",
            "energy_efficiency",
        ],
        description="Technology adoption areas to model",
    )
    stranded_asset_analysis: bool = Field(
        False,
        description="Include stranded asset risk analysis by asset class",
    )
    custom_scenario_enabled: bool = Field(
        True,
        description="Allow custom scenario definition",
    )


class CarbonPricingConfig(BaseModel):
    """Configuration for internal carbon pricing.

    Supports shadow pricing ($50-$200/tCO2e), internal fees,
    CBAM exposure, and carbon-adjusted financial metrics.
    """

    enabled: bool = Field(
        True,
        description="Enable internal carbon pricing engine",
    )
    approach: CarbonPricingApproach = Field(
        CarbonPricingApproach.SHADOW_PRICE,
        description="Carbon pricing approach",
    )
    price_usd_per_tco2e: float = Field(
        100.0,
        ge=0.0,
        le=1000.0,
        description="Internal carbon price (USD/tCO2e)",
    )
    price_escalation_pct_per_year: float = Field(
        5.0,
        ge=0.0,
        le=20.0,
        description="Annual carbon price escalation (%)",
    )
    price_scenario: str = Field(
        "MEDIUM",
        description="Carbon price scenario trajectory (LOW, MEDIUM, HIGH)",
    )
    apply_to_capex: bool = Field(
        True,
        description="Apply carbon price to CapEx investment appraisals",
    )
    apply_to_bu_allocation: bool = Field(
        True,
        description="Allocate carbon costs to business units",
    )
    cbam_enabled: bool = Field(
        False,
        description="Enable EU CBAM exposure calculation",
    )
    ets_enabled: bool = Field(
        False,
        description="Enable EU/UK ETS compliance cost modeling",
    )
    carbon_adjusted_financials: bool = Field(
        True,
        description="Generate carbon-adjusted P&L and balance sheet",
    )


class Scope4Config(BaseModel):
    """Configuration for Scope 4 avoided emissions quantification.

    Per WBCSD Avoided Emissions Guidance, reported separately
    from Scope 1/2/3 and never netted against footprint.
    """

    enabled: bool = Field(
        False,
        description="Enable Scope 4 avoided emissions calculation",
    )
    methodology: str = Field(
        "WBCSD",
        description="Avoided emissions methodology (WBCSD, custom)",
    )
    product_categories: List[str] = Field(
        default_factory=list,
        description="Product/service categories for avoided emissions quantification",
    )
    baseline_scenario: str = Field(
        "market_average",
        description="Baseline scenario type (market_average, regulatory_minimum)",
    )
    include_rebound_effects: bool = Field(
        True,
        description="Quantify and deduct rebound effects",
    )
    attribution_methodology: str = Field(
        "proportional",
        description="Attribution method for enabling effects (proportional, marginal)",
    )


class SupplyChainConfig(BaseModel):
    """Configuration for multi-tier supply chain mapping and engagement.

    Supports Tier 1-5 mapping, supplier tiering, and engagement
    program design for 100,000+ suppliers.
    """

    enabled: bool = Field(
        True,
        description="Enable supply chain mapping and engagement engine",
    )
    tier_depth: int = Field(
        3,
        ge=1,
        le=5,
        description="Supply chain tier depth for mapping (1-5)",
    )
    tier1_critical_count: int = Field(
        50,
        ge=10,
        le=200,
        description="Number of Tier 1 (critical) suppliers for direct engagement",
    )
    tier2_strategic_count: int = Field(
        200,
        ge=50,
        le=1000,
        description="Number of Tier 2 (strategic) suppliers",
    )
    tier3_managed_count: int = Field(
        1000,
        ge=100,
        le=5000,
        description="Number of Tier 3 (managed) suppliers",
    )
    cdp_supply_chain_integration: bool = Field(
        True,
        description="Integrate with CDP Supply Chain program",
    )
    ecovadis_integration: bool = Field(
        False,
        description="Integrate with EcoVadis supplier ratings",
    )
    wbcsd_pact_integration: bool = Field(
        False,
        description="Integrate with WBCSD PACT data exchange",
    )
    engagement_program_enabled: bool = Field(
        True,
        description="Enable structured supplier engagement program",
    )
    sbti_cascade_target: bool = Field(
        True,
        description="Set target for supplier SBTi adoption cascade",
    )
    commodity_tracking: List[str] = Field(
        default_factory=list,
        description="Specific commodities to track (palm_oil, soy, cocoa, etc.)",
    )


class AssuranceConfig(BaseModel):
    """Configuration for external assurance readiness.

    Supports ISO 14064-3, ISAE 3410/3000 for limited and reasonable
    assurance engagements.
    """

    level: AssuranceLevel = Field(
        AssuranceLevel.LIMITED,
        description="Target assurance level (limited -> reasonable by 2028)",
    )
    standard: str = Field(
        "ISAE_3410",
        description="Assurance standard (ISAE_3410, ISAE_3000, ISO_14064_3, AA1000AS)",
    )
    scope_of_assurance: List[str] = Field(
        default_factory=lambda: ["scope_1", "scope_2", "scope_3"],
        description="Which scopes are covered by external assurance",
    )
    provider_preference: Optional[str] = Field(
        None,
        description="Preferred assurance provider (Deloitte, EY, KPMG, PwC, SGS, DNV, etc.)",
    )
    workpaper_generation: bool = Field(
        True,
        description="Auto-generate audit workpapers per Big 4 format",
    )
    sample_selection_automated: bool = Field(
        True,
        description="Automated sample selection for substantive testing",
    )
    management_assertion_template: bool = Field(
        True,
        description="Generate management assertion letter template",
    )
    target_auditor_hours: int = Field(
        80,
        ge=20,
        le=400,
        description="Target auditor hours (vs. 200-400 manual)",
    )
    reasonable_assurance_timeline_year: int = Field(
        2028,
        ge=2025,
        le=2035,
        description="Year to transition from limited to reasonable assurance",
    )


class FinancialIntegrationConfig(BaseModel):
    """Configuration for carbon-financial integration.

    Integrates carbon data into P&L, balance sheet, CapEx decisions,
    and ESRS E1-8/E1-9 disclosures.
    """

    enabled: bool = Field(
        True,
        description="Enable financial integration engine",
    )
    carbon_adjusted_pnl: bool = Field(
        True,
        description="Generate carbon-adjusted P&L",
    )
    carbon_balance_sheet: bool = Field(
        True,
        description="Generate carbon balance sheet items",
    )
    ebitda_carbon_intensity: bool = Field(
        True,
        description="Calculate EBITDA carbon intensity",
    )
    product_level_footprint: bool = Field(
        False,
        description="Calculate product-level carbon footprint for costing",
    )
    eu_taxonomy_alignment: bool = Field(
        True,
        description="Calculate EU Taxonomy CapEx alignment",
    )
    esrs_e1_8_disclosure: bool = Field(
        True,
        description="Generate ESRS E1-8 internal carbon pricing disclosure",
    )
    esrs_e1_9_disclosure: bool = Field(
        True,
        description="Generate ESRS E1-9 anticipated financial effects disclosure",
    )
    green_bond_screening: bool = Field(
        False,
        description="Green bond eligibility screening against Taxonomy criteria",
    )
    stranded_asset_assessment: bool = Field(
        False,
        description="Asset-level stranded asset risk assessment",
    )


class ReportingConfig(BaseModel):
    """Configuration for enterprise reporting (full template suite).

    Supports 10 enterprise report templates across PDF, Excel, HTML,
    JSON, and Markdown formats.
    """

    formats: List[ReportFormat] = Field(
        default_factory=lambda: [
            ReportFormat.PDF, ReportFormat.EXCEL, ReportFormat.HTML,
            ReportFormat.JSON, ReportFormat.MARKDOWN,
        ],
        description="Output formats for generated reports",
    )
    templates: List[str] = Field(
        default_factory=lambda: [
            "ghg_inventory_report",
            "sbti_target_submission",
            "cdp_climate_response",
            "tcfd_report",
            "executive_dashboard",
            "supply_chain_heatmap",
            "scenario_comparison",
            "assurance_statement",
            "board_climate_report",
            "regulatory_filings",
        ],
        description="Report templates to generate (all 10 enterprise templates)",
    )
    regulatory_frameworks: List[str] = Field(
        default_factory=lambda: [
            "GHG_PROTOCOL", "SBTI", "CDP", "TCFD_ISSB",
            "SEC_CLIMATE", "CSRD_ESRS", "ISO_14064",
        ],
        description="Regulatory frameworks for automated crosswalk",
    )
    language: str = Field(
        "en",
        description="Primary report language (ISO 639-1)",
    )
    secondary_languages: List[str] = Field(
        default_factory=list,
        description="Secondary report languages for multi-language enterprises",
    )
    review_workflow_enabled: bool = Field(
        True,
        description="Enable multi-step review and approval workflow",
    )
    watermark_draft: bool = Field(
        True,
        description="Apply DRAFT watermark to unapproved documents",
    )
    board_reporting_frequency: str = Field(
        "QUARTERLY",
        description="Board climate report frequency (QUARTERLY, SEMI_ANNUAL, ANNUAL)",
    )


class PerformanceConfig(BaseModel):
    """Configuration for runtime performance tuning (enterprise-scale).

    Optimized for enterprise workloads: 8-64GB memory, parallel processing
    across 100+ entities, Monte Carlo simulation throughput.
    """

    cache_enabled: bool = Field(
        True,
        description="Enable caching for emission factors and intermediate results",
    )
    cache_ttl_seconds: int = Field(
        7200,
        ge=300,
        le=86400,
        description="Cache time-to-live in seconds (enterprise: 2 hours default)",
    )
    max_concurrent_calcs: int = Field(
        16,
        ge=1,
        le=64,
        description="Maximum concurrent calculation threads",
    )
    timeout_seconds: int = Field(
        300,
        ge=30,
        le=3600,
        description="Maximum timeout for a single engine calculation (seconds)",
    )
    batch_size: int = Field(
        5000,
        ge=500,
        le=50000,
        description="Batch size for bulk data processing",
    )
    memory_limit_mb: int = Field(
        16384,
        ge=4096,
        le=65536,
        description="Memory limit in MB (enterprise: 16GB default, up to 64GB)",
    )
    monte_carlo_parallelism: int = Field(
        8,
        ge=1,
        le=32,
        description="Parallel threads for Monte Carlo simulation runs",
    )
    entity_parallel_processing: bool = Field(
        True,
        description="Enable parallel processing across entities",
    )
    lightweight_mode: bool = Field(
        False,
        description="Lightweight mode (reduced precision, faster execution) -- disabled for enterprise",
    )


class AuditTrailConfig(BaseModel):
    """Configuration for audit trail and provenance (enterprise: 7-year retention).

    Full audit trail with SHA-256 provenance hashing, data lineage,
    and external audit export for Big 4 assurance engagements.
    """

    enabled: bool = Field(
        True,
        description="Enable full audit trail for all calculations",
    )
    sha256_provenance: bool = Field(
        True,
        description="Generate SHA-256 provenance hashes for all outputs",
    )
    calculation_logging: bool = Field(
        True,
        description="Log all calculation steps with inputs and outputs",
    )
    assumption_tracking: bool = Field(
        True,
        description="Track all assumptions used in calculations",
    )
    data_lineage_enabled: bool = Field(
        True,
        description="Track full data lineage from source to reported figure",
    )
    retention_years: int = Field(
        7,
        ge=3,
        le=10,
        description="Audit trail retention period in years (enterprise: 7 years)",
    )
    external_audit_export: bool = Field(
        True,
        description="Enable export format for external auditors (Big 4 compatible)",
    )
    immutable_log: bool = Field(
        True,
        description="Immutable audit log (append-only, no deletion)",
    )


class ScorecardConfig(BaseModel):
    """Configuration for enterprise net-zero maturity scorecard."""

    enabled: bool = Field(
        True,
        description="Enable net-zero maturity scorecard",
    )
    assessment_mode: MaturityAssessment = Field(
        MaturityAssessment.COMPREHENSIVE,
        description="Assessment depth: COMPREHENSIVE (12 dimensions) or STANDARD (8)",
    )
    dimensions: List[str] = Field(
        default_factory=lambda: [
            "governance_oversight",
            "baseline_completeness",
            "target_ambition",
            "scope3_coverage",
            "data_quality",
            "reduction_progress",
            "supply_chain_engagement",
            "scenario_resilience",
            "assurance_readiness",
            "regulatory_compliance",
            "financial_integration",
            "stakeholder_communication",
        ],
        description="Scorecard dimensions for enterprise maturity assessment",
    )
    benchmark_enabled: bool = Field(
        True,
        description="Enable peer benchmarking against sector peers",
    )
    peer_group: str = Field(
        "SECTOR_AND_SCALE",
        description="Peer group for benchmarking (SECTOR, SCALE, SECTOR_AND_SCALE, GEOGRAPHY)",
    )
    kpi_set: List[str] = Field(
        default_factory=lambda: [
            "absolute_emissions_trajectory",
            "emission_intensity_revenue",
            "scope3_data_quality_score",
            "yoy_reduction_rate",
            "sbti_criteria_compliance",
            "supplier_engagement_rate",
            "renewable_electricity_pct",
            "assurance_coverage_pct",
            "regulatory_framework_coverage",
            "carbon_price_coverage",
        ],
        description="Enterprise KPI set for scorecard",
    )
    target_years: List[int] = Field(
        default_factory=lambda: [2025, 2027, 2030, 2035, 2040, 2045, 2050],
        description="Milestone years for trajectory tracking",
    )
    tpi_alignment_check: bool = Field(
        True,
        description="Check alignment with TPI Carbon Performance benchmarks",
    )


# =============================================================================
# Main Configuration Models
# =============================================================================


class EnterpriseNetZeroConfig(BaseModel):
    """Main configuration for PACK-027 Enterprise Net Zero Pack.

    This is the root configuration model that contains all sub-configurations
    for enterprise net-zero management. The organization.sector field drives
    preset selection, SBTi pathway, and engine configuration.
    """

    # Temporal settings
    reporting_year: int = Field(
        DEFAULT_REPORTING_YEAR,
        ge=2020,
        le=2040,
        description="Current reporting year for GHG inventory",
    )
    base_year: int = Field(
        DEFAULT_BASE_YEAR,
        ge=2015,
        le=2030,
        description="Base year for emissions baseline and target tracking",
    )
    pack_version: str = Field(
        "1.0.0",
        description="Pack configuration version",
    )

    # Sub-configurations (18 sections)
    organization: EnterpriseOrganizationConfig = Field(
        default_factory=EnterpriseOrganizationConfig,
        description="Enterprise organization profile configuration",
    )
    consolidation: ConsolidationConfig = Field(
        default_factory=ConsolidationConfig,
        description="Multi-entity consolidation configuration",
    )
    data_quality: DataQualityConfig = Field(
        default_factory=DataQualityConfig,
        description="Data quality management configuration",
    )
    scope: ScopeConfig = Field(
        default_factory=ScopeConfig,
        description="Scope 1/2/3 emissions calculation configuration",
    )
    target: SBTiTargetConfig = Field(
        default_factory=SBTiTargetConfig,
        description="SBTi target setting configuration (Corporate Standard)",
    )
    scenarios: ScenarioModelingConfig = Field(
        default_factory=ScenarioModelingConfig,
        description="Monte Carlo scenario modeling configuration",
    )
    carbon_pricing: CarbonPricingConfig = Field(
        default_factory=CarbonPricingConfig,
        description="Internal carbon pricing configuration",
    )
    scope4: Scope4Config = Field(
        default_factory=Scope4Config,
        description="Scope 4 avoided emissions configuration",
    )
    supply_chain: SupplyChainConfig = Field(
        default_factory=SupplyChainConfig,
        description="Supply chain mapping and engagement configuration",
    )
    assurance: AssuranceConfig = Field(
        default_factory=AssuranceConfig,
        description="External assurance readiness configuration",
    )
    financial_integration: FinancialIntegrationConfig = Field(
        default_factory=FinancialIntegrationConfig,
        description="Carbon-financial integration configuration",
    )
    reporting: ReportingConfig = Field(
        default_factory=ReportingConfig,
        description="Reporting and template configuration",
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Runtime performance tuning configuration",
    )
    audit_trail: AuditTrailConfig = Field(
        default_factory=AuditTrailConfig,
        description="Audit trail and provenance configuration",
    )
    scorecard: ScorecardConfig = Field(
        default_factory=ScorecardConfig,
        description="Net-zero maturity scorecard configuration",
    )

    @model_validator(mode="after")
    def validate_base_year_before_reporting(self) -> "EnterpriseNetZeroConfig":
        """Ensure base year is not after reporting year."""
        if self.base_year > self.reporting_year:
            raise ValueError(
                f"base_year ({self.base_year}) must not be after "
                f"reporting_year ({self.reporting_year})"
            )
        return self

    @model_validator(mode="after")
    def validate_scope3_coverage(self) -> "EnterpriseNetZeroConfig":
        """Warn if Scope 3 coverage is below SBTi threshold."""
        if len(self.scope.scope3_categories) < 15:
            missing = set(ALL_SCOPE3_CATEGORIES) - set(self.scope.scope3_categories)
            logger.warning(
                "Enterprise pack should calculate all 15 Scope 3 categories. "
                "Missing categories: %s. Ensure exclusions are documented and "
                "do not exceed %s%% of total Scope 3.",
                sorted(missing),
                self.scope.scope3_total_exclusion_max_pct,
            )
        return self

    @model_validator(mode="after")
    def validate_dual_scope2(self) -> "EnterpriseNetZeroConfig":
        """Ensure dual Scope 2 reporting is enabled."""
        if len(self.scope.scope2_methods) < 2:
            logger.warning(
                "Enterprise GHG reporting requires dual Scope 2 reporting "
                "(both location-based and market-based). Only %s configured.",
                self.scope.scope2_methods,
            )
        return self

    def get_enabled_engines(self) -> List[str]:
        """Return list of engine names that should be enabled based on config.

        Returns:
            List of engine identifier strings.
        """
        engines = [
            "enterprise_baseline_engine",
            "sbti_target_engine",
            "multi_entity_consolidation_engine",
        ]

        if self.scenarios.enabled:
            engines.append("scenario_modeling_engine")

        if self.carbon_pricing.enabled:
            engines.append("carbon_pricing_engine")

        if self.scope4.enabled:
            engines.append("scope4_avoided_emissions_engine")

        if self.supply_chain.enabled:
            engines.append("supply_chain_mapping_engine")

        if self.financial_integration.enabled:
            engines.append("financial_integration_engine")

        return sorted(set(engines))

    def get_sector_info(self) -> Dict[str, Any]:
        """Get enterprise sector-specific guidance information.

        Returns:
            Dictionary with sector name, typical scope split, key levers,
            and SBTi pathway recommendation.
        """
        return ENTERPRISE_SECTOR_INFO.get(
            self.organization.sector.value,
            ENTERPRISE_SECTOR_INFO.get("OTHER", {}),
        )

    def get_emissions_profile(self) -> Dict[str, Any]:
        """Get typical emissions profile for the configured sector.

        Returns:
            Dictionary with scope percentages, profile type, and
            priority Scope 3 categories.
        """
        return SECTOR_EMISSIONS_PROFILE.get(
            self.organization.sector.value,
            SECTOR_EMISSIONS_PROFILE["OTHER"],
        )

    def get_sda_benchmarks(self) -> List[Dict[str, Any]]:
        """Get SDA sector benchmarks for applicable sectors.

        Returns:
            List of SDA benchmark dictionaries for each applicable sector.
        """
        benchmarks = []
        for sector_key in self.target.sda_sectors:
            sector_upper = sector_key.upper()
            if sector_upper in SDA_SECTOR_BENCHMARKS:
                benchmarks.append({
                    "sector": sector_upper,
                    **SDA_SECTOR_BENCHMARKS[sector_upper],
                })
        return benchmarks


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """Top-level pack configuration wrapper for PACK-027 Enterprise Net Zero Pack.

    Handles preset loading, environment variable overrides, and
    configuration merging.

    Example:
        >>> config = PackConfig.from_preset("manufacturing")
        >>> print(config.pack.organization.sector)
        EnterpriseSector.MANUFACTURING
        >>> config = PackConfig.from_preset("financial_services", overrides={"reporting_year": 2026})
    """

    pack: EnterpriseNetZeroConfig = Field(
        default_factory=EnterpriseNetZeroConfig,
        description="Main Enterprise Net Zero configuration",
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
        "PACK-027-enterprise-net-zero",
        description="Pack identifier",
    )

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "PackConfig":
        """Load configuration from a named sector preset.

        Args:
            preset_name: Name of the preset (manufacturing, financial_services,
                technology, energy_utilities, retail_consumer, healthcare,
                transport_logistics, agriculture_food).
            overrides: Optional dictionary of configuration overrides.

        Returns:
            PackConfig instance with preset values applied.

        Raises:
            FileNotFoundError: If preset YAML file does not exist.
            ValueError: If preset_name is not in SUPPORTED_PRESETS.
        """
        if preset_name not in SUPPORTED_PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. "
                f"Available presets: {sorted(SUPPORTED_PRESETS.keys())}"
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
        env_overrides = _get_env_overrides("ENTERPRISE_NET_ZERO_")
        if env_overrides:
            preset_data = _merge_config(preset_data, env_overrides)

        # Apply explicit overrides
        if overrides:
            preset_data = _merge_config(preset_data, overrides)

        pack_config = EnterpriseNetZeroConfig(**preset_data)
        return cls(pack=pack_config, preset_name=preset_name)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "PackConfig":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            PackConfig instance with YAML values applied.

        Raises:
            FileNotFoundError: If YAML file does not exist.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        pack_config = EnterpriseNetZeroConfig(**config_data)
        return cls(pack=pack_config)

    @classmethod
    def from_sector(
        cls,
        sector: EnterpriseSector,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "PackConfig":
        """Auto-select preset based on enterprise sector.

        Args:
            sector: Enterprise sector classification.
            overrides: Optional configuration overrides.

        Returns:
            PackConfig instance with appropriate sector preset.
        """
        sector_preset_map: Dict[str, str] = {
            "MANUFACTURING": "manufacturing",
            "MINING_METALS": "manufacturing",
            "CHEMICALS": "manufacturing",
            "CONSTRUCTION": "manufacturing",
            "AUTOMOTIVE": "manufacturing",
            "AEROSPACE_DEFENSE": "manufacturing",
            "ENERGY_UTILITIES": "energy_utilities",
            "FINANCIAL_SERVICES": "financial_services",
            "TECHNOLOGY": "technology",
            "TELECOMMUNICATIONS": "technology",
            "MEDIA_ENTERTAINMENT": "technology",
            "CONSUMER_GOODS": "retail_consumer",
            "FOOD_BEVERAGE": "agriculture_food",
            "AGRICULTURE": "agriculture_food",
            "TRANSPORT_LOGISTICS": "transport_logistics",
            "REAL_ESTATE": "retail_consumer",
            "HEALTHCARE_PHARMA": "healthcare",
            "HOSPITALITY_LEISURE": "retail_consumer",
            "PROFESSIONAL_SERVICES": "technology",
            "EDUCATION": "technology",
            "PUBLIC_SECTOR": "manufacturing",
            "OTHER": "manufacturing",
        }

        preset_name = sector_preset_map.get(sector.value, "manufacturing")
        return cls.from_preset(preset_name, overrides=overrides)

    def get_config_hash(self) -> str:
        """Generate SHA-256 hash of the current configuration for provenance.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        config_json = self.model_dump_json(indent=None)
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()

    def validate_config(self) -> List[str]:
        """Cross-field validation returning warnings.

        Returns:
            List of warning messages (empty if fully valid).
        """
        return validate_config(self.pack)

    def get_regulatory_frameworks(self) -> List[str]:
        """Get applicable regulatory frameworks based on configuration.

        Returns:
            List of applicable regulatory framework identifiers.
        """
        frameworks = ["GHG_PROTOCOL", "SBTI"]

        if self.pack.reporting.regulatory_frameworks:
            return self.pack.reporting.regulatory_frameworks

        if self.pack.organization.csrd_in_scope:
            frameworks.extend(["CSRD_ESRS", "EU_TAXONOMY"])
        if self.pack.organization.is_large_accelerated_filer:
            frameworks.append("SEC_CLIMATE")
        if self.pack.organization.headquarters_country == "US":
            frameworks.extend(["CA_SB253", "CA_SB261"])
        frameworks.extend(["CDP", "TCFD_ISSB", "ISO_14064"])

        return sorted(set(frameworks))


# =============================================================================
# Utility Functions
# =============================================================================


def load_preset(
    preset_name: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> "PackConfig":
    """Load a named preset configuration.

    Convenience wrapper around PackConfig.from_preset().

    Args:
        preset_name: Name of the preset to load.
        overrides: Optional configuration overrides.

    Returns:
        PackConfig instance with preset applied.
    """
    return PackConfig.from_preset(preset_name, overrides)


def _merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base configuration dictionary.
        override: Override dictionary (values take precedence).

    Returns:
        Merged dictionary.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Public deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base configuration dictionary.
        override: Override dictionary (values take precedence).

    Returns:
        Merged dictionary.
    """
    return _merge_config(base, override)


def _get_env_overrides(prefix: str) -> Dict[str, Any]:
    """Load configuration overrides from environment variables.

    Environment variables prefixed with the given prefix are loaded and
    mapped to configuration keys. Nested keys use double underscore.

    Example:
        ENTERPRISE_NET_ZERO_REPORTING_YEAR=2026
        ENTERPRISE_NET_ZERO_SCOPE__SCOPE3_DEFAULT_METHOD=supplier_specific
        ENTERPRISE_NET_ZERO_TARGET__NEAR_TERM_SCOPE1_2_REDUCTION_PCT=50

    Args:
        prefix: Environment variable prefix to search for.

    Returns:
        Dictionary of parsed overrides.
    """
    overrides: Dict[str, Any] = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower()
            parts = config_key.split("__")
            current = overrides
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            # Parse value types
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


def get_env_overrides(prefix: str) -> Dict[str, Any]:
    """Public wrapper for loading environment variable overrides.

    Args:
        prefix: Environment variable prefix to search for.

    Returns:
        Dictionary of parsed overrides.
    """
    return _get_env_overrides(prefix)


def validate_config(config: EnterpriseNetZeroConfig) -> List[str]:
    """Validate an enterprise net-zero configuration and return any warnings.

    Performs cross-field validation beyond what Pydantic validators cover.
    Returns advisory warnings, not hard errors.

    Args:
        config: EnterpriseNetZeroConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check organization name is set
    if not config.organization.name:
        warnings.append(
            "Organization name is empty. Set organization.name for meaningful reports."
        )

    # Check all 15 Scope 3 categories
    if len(config.scope.scope3_categories) < 15:
        missing = set(ALL_SCOPE3_CATEGORIES) - set(config.scope.scope3_categories)
        warnings.append(
            f"Not all 15 Scope 3 categories are included. Missing: {sorted(missing)}. "
            f"Enterprise GHG Protocol compliance requires all material categories."
        )

    # Check dual Scope 2 reporting
    if "location_based" not in config.scope.scope2_methods:
        warnings.append(
            "Location-based Scope 2 method is not configured. "
            "Dual Scope 2 reporting is mandatory per GHG Protocol Scope 2 Guidance."
        )
    if "market_based" not in config.scope.scope2_methods:
        warnings.append(
            "Market-based Scope 2 method is not configured. "
            "Dual Scope 2 reporting is mandatory per GHG Protocol Scope 2 Guidance."
        )

    # Check SBTi coverage requirements
    if config.target.scope1_2_coverage_pct < 95.0:
        warnings.append(
            f"Scope 1+2 coverage ({config.target.scope1_2_coverage_pct}%) is below "
            f"SBTi minimum of 95%. Increase coverage for target validation."
        )
    if config.target.scope3_near_term_coverage_pct < 67.0:
        warnings.append(
            f"Scope 3 near-term coverage ({config.target.scope3_near_term_coverage_pct}%) "
            f"is below SBTi minimum of 67%."
        )

    # Check data quality target for enterprise
    if config.data_quality.accuracy_target_pct > 10.0:
        warnings.append(
            f"Accuracy target (+/-{config.data_quality.accuracy_target_pct}%) is too loose "
            f"for enterprise financial-grade reporting. Target should be +/-3% or better."
        )

    # Check assurance configuration
    if config.assurance.level == AssuranceLevel.LIMITED:
        if config.reporting_year >= 2028:
            warnings.append(
                "Limited assurance is configured but reporting year is 2028+. "
                "Consider transitioning to reasonable assurance per CSRD timeline."
            )

    # Check consolidation for multi-entity
    if config.organization.entity_count > 1:
        if config.consolidation.entity_count != config.organization.entity_count:
            warnings.append(
                f"Organization entity count ({config.organization.entity_count}) does not "
                f"match consolidation entity count ({config.consolidation.entity_count}). "
                f"These should be consistent."
            )

    # Check FLAG threshold
    profile = SECTOR_EMISSIONS_PROFILE.get(config.organization.sector.value, {})
    if profile.get("scope1_pct", 0) > 0:
        if config.organization.sector.value in ("AGRICULTURE", "FOOD_BEVERAGE"):
            if not config.target.flag_enabled:
                warnings.append(
                    f"Sector {config.organization.sector.value} likely has >20% land use "
                    f"emissions. FLAG targets should be enabled per SBTi guidance."
                )

    # Check carbon pricing range
    if config.carbon_pricing.enabled:
        if config.carbon_pricing.price_usd_per_tco2e < 50.0:
            warnings.append(
                f"Internal carbon price (${config.carbon_pricing.price_usd_per_tco2e}/tCO2e) "
                f"is below recommended minimum of $50/tCO2e for enterprise decision-making."
            )

    # Check performance for Monte Carlo
    if config.scenarios.enabled:
        if config.scenarios.monte_carlo_runs >= 10000:
            if config.performance.monte_carlo_parallelism < 4:
                warnings.append(
                    f"Monte Carlo parallelism ({config.performance.monte_carlo_parallelism}) "
                    f"is low for {config.scenarios.monte_carlo_runs} runs. "
                    f"Consider increasing to 8+ for reasonable execution time."
                )

    # Check audit trail for assurance
    if not config.audit_trail.enabled:
        warnings.append(
            "Audit trail is disabled. This is required for external assurance "
            "readiness and regulatory compliance."
        )

    return warnings


def get_sector_info(sector: Union[str, EnterpriseSector]) -> Dict[str, Any]:
    """Get detailed information about an enterprise sector.

    Args:
        sector: Sector enum or string value.

    Returns:
        Dictionary with name, typical scope split, key levers,
        SBTi pathway, and target organizations.
    """
    key = sector.value if isinstance(sector, EnterpriseSector) else sector
    return ENTERPRISE_SECTOR_INFO.get(key, {})


def get_emissions_profile(sector: Union[str, EnterpriseSector]) -> Dict[str, Any]:
    """Get typical emissions profile split for an enterprise sector.

    Args:
        sector: Sector enum or string value.

    Returns:
        Dictionary with scope percentages, profile type, primary Scope 3
        categories, and SDA applicability.
    """
    key = sector.value if isinstance(sector, EnterpriseSector) else sector
    return SECTOR_EMISSIONS_PROFILE.get(key, SECTOR_EMISSIONS_PROFILE["OTHER"])


def get_sda_benchmark(sector: Union[str, SDASector]) -> Dict[str, Any]:
    """Get SDA sector benchmark for a given sector.

    Args:
        sector: SDA sector enum or string value.

    Returns:
        Dictionary with intensity metric, 2030 and 2050 targets, and source.
    """
    key = sector.value if isinstance(sector, SDASector) else sector
    return SDA_SECTOR_BENCHMARKS.get(key, {})


def get_carbon_price_scenario(
    scenario: str = "MEDIUM",
) -> Dict[int, float]:
    """Get carbon price trajectory for a given scenario.

    Args:
        scenario: Scenario name (LOW, MEDIUM, HIGH).

    Returns:
        Dictionary mapping year to USD/tCO2e price.
    """
    return CARBON_PRICE_SCENARIOS.get(scenario.upper(), CARBON_PRICE_SCENARIOS["MEDIUM"])


def get_sbti_parameters() -> Dict[str, Any]:
    """Get SBTi Corporate Standard parameters.

    Returns:
        Dictionary with all SBTi criteria and threshold parameters.
    """
    return SBTI_CORPORATE_PARAMETERS.copy()


def get_gwp100(gas: str) -> int:
    """Get IPCC AR6 GWP100 value for a greenhouse gas.

    Args:
        gas: Greenhouse gas identifier (CO2, CH4, N2O, HFC_134A, SF6, etc.).

    Returns:
        GWP100 value (dimensionless, relative to CO2). Returns 0 if not found.
    """
    return IPCC_AR6_GWP100.get(gas.upper(), 0)


def get_default_config(
    sector: EnterpriseSector = EnterpriseSector.OTHER,
) -> EnterpriseNetZeroConfig:
    """Get default configuration for a given enterprise sector.

    Args:
        sector: Enterprise sector classification.

    Returns:
        EnterpriseNetZeroConfig instance with sector-appropriate defaults.
    """
    profile = SECTOR_EMISSIONS_PROFILE.get(
        sector.value, SECTOR_EMISSIONS_PROFILE["OTHER"]
    )
    scope3_priorities = profile.get("primary_scope3_categories", [1, 2, 3, 4, 6, 7])

    # Determine SBTi pathway from sector profile
    sda_applicable = profile.get("sda_applicable", False)
    sda_sectors = profile.get("sda_sectors", [])

    if sda_applicable and sda_sectors:
        pathway = SBTiPathway.MIXED
    else:
        pathway = SBTiPathway.ACA_15C

    return EnterpriseNetZeroConfig(
        organization=EnterpriseOrganizationConfig(
            sector=sector,
        ),
        scope=ScopeConfig(
            scope3_priority_categories=scope3_priorities,
        ),
        target=SBTiTargetConfig(
            sbti_pathway=pathway,
            sda_sectors=sda_sectors,
        ),
    )


def list_available_presets() -> Dict[str, str]:
    """List all available configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return SUPPORTED_PRESETS.copy()


def get_report_templates() -> Dict[str, Dict[str, Any]]:
    """Get information about all enterprise report templates.

    Returns:
        Dictionary mapping template names to template info.
    """
    return ENTERPRISE_REPORT_TEMPLATES.copy()
