"""
PACK-046 Intensity Metrics Pack - Configuration Manager

Pydantic v2 configuration for GHG emissions intensity metric management
including denominator selection, multi-scope intensity computation,
LMDI decomposition analysis, sector benchmarking, SBTi SDA target
pathways, trend analysis, scenario modelling, uncertainty quantification,
multi-framework disclosure mapping, and automated reporting.

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (sector-specific defaults)
    3. Environment overrides (INTENSITY_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    EU CSRD / ESRS E1-6 - Climate change intensity disclosures
    SBTi Sectoral Decarbonisation Approach (SDA) v2.0
    CDP Climate Change Questionnaire C6.10 (2026) - Emissions intensities
    US SEC Climate Disclosure Rules (2024) - Intensity metrics
    ISO 14064-1:2018 Clause 5 - Quantification per unit of output
    TCFD Recommendations - Metrics and Targets (cross-industry intensity)
    GRI 305-4 (2016) - GHG emissions intensity
    IFRS S2 (2023) - Climate-related disclosures intensity metrics

Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
"""

import hashlib
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from greenlang.schemas.enums import NotificationChannel

logger = logging.getLogger(__name__)

PACK_BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent


# =============================================================================
# Enums
# =============================================================================


class DenominatorCategory(str, Enum):
    """Category of activity denominator used for intensity normalisation."""
    FINANCIAL = "FINANCIAL"
    PHYSICAL = "PHYSICAL"
    HEADCOUNT = "HEADCOUNT"
    AREA = "AREA"
    ENERGY = "ENERGY"
    CUSTOM = "CUSTOM"


class ScopeInclusion(str, Enum):
    """Which GHG scopes to include in the intensity numerator."""
    SCOPE_1_ONLY = "SCOPE_1_ONLY"
    SCOPE_2_LOCATION = "SCOPE_2_LOCATION"
    SCOPE_2_MARKET = "SCOPE_2_MARKET"
    SCOPE_1_2_LOCATION = "SCOPE_1_2_LOCATION"
    SCOPE_1_2_MARKET = "SCOPE_1_2_MARKET"
    SCOPE_1_2_3 = "SCOPE_1_2_3"
    SCOPE_3_SPECIFIC = "SCOPE_3_SPECIFIC"
    CUSTOM = "CUSTOM"


class DecompositionMethod(str, Enum):
    """Logarithmic Mean Divisia Index (LMDI) decomposition variant."""
    LMDI_I_ADDITIVE = "LMDI_I_ADDITIVE"
    LMDI_I_MULTIPLICATIVE = "LMDI_I_MULTIPLICATIVE"
    LMDI_II_ADDITIVE = "LMDI_II_ADDITIVE"
    LMDI_II_MULTIPLICATIVE = "LMDI_II_MULTIPLICATIVE"


class BenchmarkSource(str, Enum):
    """External data source for sector benchmarking."""
    CDP = "CDP"
    TPI = "TPI"
    GRESB = "GRESB"
    CRREM = "CRREM"
    CUSTOM = "CUSTOM"


class TargetPathway(str, Enum):
    """Science-based temperature alignment pathway."""
    WELL_BELOW_2C = "WELL_BELOW_2C"
    ONE_POINT_FIVE_C = "ONE_POINT_FIVE_C"
    NET_ZERO = "NET_ZERO"


class ScenarioType(str, Enum):
    """Type of intensity scenario for what-if modelling."""
    EFFICIENCY = "EFFICIENCY"
    GROWTH = "GROWTH"
    STRUCTURAL = "STRUCTURAL"
    METHODOLOGY = "METHODOLOGY"
    COMBINED = "COMBINED"


class DisclosureFramework(str, Enum):
    """Supported regulatory and voluntary reporting frameworks for intensity."""
    ESRS_E1 = "ESRS_E1"
    CDP = "CDP"
    SEC = "SEC"
    SBTI = "SBTI"
    ISO_14064 = "ISO_14064"
    TCFD = "TCFD"
    GRI = "GRI"
    IFRS_S2 = "IFRS_S2"


class DataQualityLevel(int, Enum):
    """Data quality tier for uncertainty quantification (1 = best, 5 = worst)."""
    AUDITED = 1
    CALCULATED = 2
    ESTIMATED = 3
    PROXY = 4
    DEFAULT = 5


class ConsolidationApproach(str, Enum):
    """Organisational boundary consolidation approach per GHG Protocol Ch 3."""
    EQUITY_SHARE = "EQUITY_SHARE"
    OPERATIONAL_CONTROL = "OPERATIONAL_CONTROL"
    FINANCIAL_CONTROL = "FINANCIAL_CONTROL"


class IntensitySector(str, Enum):
    """Industry sector classification for intensity preset defaults."""
    MULTI_SECTOR = "MULTI_SECTOR"
    MANUFACTURING = "MANUFACTURING"
    REAL_ESTATE = "REAL_ESTATE"
    POWER_GENERATION = "POWER_GENERATION"
    TRANSPORT_LOGISTICS = "TRANSPORT_LOGISTICS"
    FINANCIAL_SERVICES = "FINANCIAL_SERVICES"
    FOOD_AGRICULTURE = "FOOD_AGRICULTURE"
    HEALTHCARE = "HEALTHCARE"
    EDUCATION = "EDUCATION"
    DATA_CENTER = "DATA_CENTER"
    RETAIL = "RETAIL"
    SME = "SME"


class OutputFormat(str, Enum):
    """Supported report output formats."""
    MARKDOWN = "MARKDOWN"
    HTML = "HTML"
    JSON = "JSON"
    CSV = "CSV"
    PDF = "PDF"
    XLSX = "XLSX"


class RegressionModel(str, Enum):
    """Statistical model for trend regression."""
    OLS = "OLS"
    WEIGHTED_LS = "WEIGHTED_LS"
    ROBUST = "ROBUST"
    LOESS = "LOESS"


class NullHandling(str, Enum):
    """Strategy for handling null denominator or numerator values."""
    EXCLUDE = "EXCLUDE"
    ZERO_FILL = "ZERO_FILL"
    INTERPOLATE = "INTERPOLATE"
    PREVIOUS_YEAR = "PREVIOUS_YEAR"
    RAISE_ERROR = "RAISE_ERROR"


class WeightedAverageMethod(str, Enum):
    """Method for calculating weighted average intensity across entities."""
    REVENUE_WEIGHTED = "REVENUE_WEIGHTED"
    ACTIVITY_WEIGHTED = "ACTIVITY_WEIGHTED"
    EQUAL_WEIGHTED = "EQUAL_WEIGHTED"
    EMISSION_WEIGHTED = "EMISSION_WEIGHTED"


class PropagationMethod(str, Enum):
    """Uncertainty propagation method."""
    MONTE_CARLO = "MONTE_CARLO"
    ANALYTICAL_GUM = "ANALYTICAL_GUM"
    BOOTSTRAP = "BOOTSTRAP"


# =============================================================================
# Reference Data Constants
# =============================================================================


STANDARD_DENOMINATORS: Dict[str, Dict[str, Any]] = {
    # --- Financial Denominators ---
    "revenue_meur": {
        "id": "revenue_meur",
        "name": "Revenue (MEUR)",
        "unit": "MEUR",
        "category": "FINANCIAL",
        "sectors": ["MULTI_SECTOR", "MANUFACTURING", "RETAIL", "HEALTHCARE", "EDUCATION", "SME"],
        "frameworks": ["ESRS_E1", "CDP", "SEC", "TCFD", "GRI", "IFRS_S2"],
        "description": "Annual revenue in millions of euros",
    },
    "revenue_musd": {
        "id": "revenue_musd",
        "name": "Revenue (MUSD)",
        "unit": "MUSD",
        "category": "FINANCIAL",
        "sectors": ["MULTI_SECTOR", "MANUFACTURING", "RETAIL", "HEALTHCARE"],
        "frameworks": ["SEC", "CDP", "TCFD", "IFRS_S2"],
        "description": "Annual revenue in millions of US dollars",
    },
    "ebitda_meur": {
        "id": "ebitda_meur",
        "name": "EBITDA (MEUR)",
        "unit": "MEUR",
        "category": "FINANCIAL",
        "sectors": ["MULTI_SECTOR", "FINANCIAL_SERVICES"],
        "frameworks": ["TCFD", "IFRS_S2"],
        "description": "Earnings before interest, taxes, depreciation, and amortisation",
    },
    "gross_profit_meur": {
        "id": "gross_profit_meur",
        "name": "Gross Profit (MEUR)",
        "unit": "MEUR",
        "category": "FINANCIAL",
        "sectors": ["MULTI_SECTOR", "RETAIL"],
        "frameworks": ["ESRS_E1", "TCFD"],
        "description": "Gross profit in millions of euros",
    },
    "assets_under_management_meur": {
        "id": "assets_under_management_meur",
        "name": "AUM (MEUR)",
        "unit": "MEUR",
        "category": "FINANCIAL",
        "sectors": ["FINANCIAL_SERVICES"],
        "frameworks": ["CDP", "TCFD", "IFRS_S2", "SBTI"],
        "description": "Total assets under management in millions of euros (PCAF)",
    },
    "total_lending_meur": {
        "id": "total_lending_meur",
        "name": "Total Lending (MEUR)",
        "unit": "MEUR",
        "category": "FINANCIAL",
        "sectors": ["FINANCIAL_SERVICES"],
        "frameworks": ["CDP", "TCFD", "SBTI"],
        "description": "Total outstanding lending portfolio in millions of euros",
    },
    # --- Physical Output Denominators ---
    "tonnes_output": {
        "id": "tonnes_output",
        "name": "Physical Output (tonnes)",
        "unit": "tonne",
        "category": "PHYSICAL",
        "sectors": ["MANUFACTURING", "FOOD_AGRICULTURE"],
        "frameworks": ["ESRS_E1", "CDP", "GRI", "SBTI", "ISO_14064"],
        "description": "Total physical output in metric tonnes",
    },
    "units_produced": {
        "id": "units_produced",
        "name": "Units Produced",
        "unit": "unit",
        "category": "PHYSICAL",
        "sectors": ["MANUFACTURING"],
        "frameworks": ["GRI", "CDP"],
        "description": "Total number of discrete units produced",
    },
    "mwh_generated": {
        "id": "mwh_generated",
        "name": "Electricity Generated (MWh)",
        "unit": "MWh",
        "category": "ENERGY",
        "sectors": ["POWER_GENERATION"],
        "frameworks": ["ESRS_E1", "CDP", "SBTI", "TCFD", "ISO_14064"],
        "description": "Total net electricity generated in megawatt-hours",
    },
    "gwh_generated": {
        "id": "gwh_generated",
        "name": "Electricity Generated (GWh)",
        "unit": "GWh",
        "category": "ENERGY",
        "sectors": ["POWER_GENERATION"],
        "frameworks": ["CDP", "SBTI"],
        "description": "Total net electricity generated in gigawatt-hours",
    },
    "tkm": {
        "id": "tkm",
        "name": "Tonne-Kilometres",
        "unit": "tkm",
        "category": "PHYSICAL",
        "sectors": ["TRANSPORT_LOGISTICS"],
        "frameworks": ["ESRS_E1", "CDP", "SBTI", "GRI", "ISO_14064"],
        "description": "Total freight transport activity in tonne-kilometres (GLEC Framework)",
    },
    "pkm": {
        "id": "pkm",
        "name": "Passenger-Kilometres",
        "unit": "pkm",
        "category": "PHYSICAL",
        "sectors": ["TRANSPORT_LOGISTICS"],
        "frameworks": ["CDP", "SBTI", "GRI"],
        "description": "Total passenger transport activity in passenger-kilometres",
    },
    "vehicle_km": {
        "id": "vehicle_km",
        "name": "Vehicle-Kilometres",
        "unit": "vkm",
        "category": "PHYSICAL",
        "sectors": ["TRANSPORT_LOGISTICS"],
        "frameworks": ["GRI"],
        "description": "Total distance travelled by vehicles in kilometres",
    },
    "hectare_crop": {
        "id": "hectare_crop",
        "name": "Crop Area (hectares)",
        "unit": "ha",
        "category": "AREA",
        "sectors": ["FOOD_AGRICULTURE"],
        "frameworks": ["SBTI", "CDP"],
        "description": "Total crop area harvested in hectares (SBTi FLAG)",
    },
    "litres_milk": {
        "id": "litres_milk",
        "name": "Milk Produced (litres)",
        "unit": "litre",
        "category": "PHYSICAL",
        "sectors": ["FOOD_AGRICULTURE"],
        "frameworks": ["SBTI", "CDP"],
        "description": "Total milk production in litres (dairy sector)",
    },
    "tonnes_protein": {
        "id": "tonnes_protein",
        "name": "Protein Produced (tonnes)",
        "unit": "tonne",
        "category": "PHYSICAL",
        "sectors": ["FOOD_AGRICULTURE"],
        "frameworks": ["SBTI"],
        "description": "Total protein output in metric tonnes (SBTi FLAG)",
    },
    # --- Area Denominators ---
    "sqm_floor_area": {
        "id": "sqm_floor_area",
        "name": "Floor Area (m2)",
        "unit": "m2",
        "category": "AREA",
        "sectors": ["REAL_ESTATE", "RETAIL", "HEALTHCARE", "EDUCATION", "DATA_CENTER"],
        "frameworks": ["ESRS_E1", "CDP", "GRESB", "CRREM", "TCFD", "GRI"],
        "description": "Total gross internal floor area in square metres",
    },
    "sqm_lettable_area": {
        "id": "sqm_lettable_area",
        "name": "Lettable Area (m2)",
        "unit": "m2",
        "category": "AREA",
        "sectors": ["REAL_ESTATE"],
        "frameworks": ["GRESB", "CRREM"],
        "description": "Total net lettable area in square metres",
    },
    "sqft_floor_area": {
        "id": "sqft_floor_area",
        "name": "Floor Area (ft2)",
        "unit": "ft2",
        "category": "AREA",
        "sectors": ["REAL_ESTATE", "RETAIL"],
        "frameworks": ["SEC"],
        "description": "Total gross internal floor area in square feet",
    },
    # --- Headcount Denominators ---
    "fte": {
        "id": "fte",
        "name": "Full-Time Equivalents",
        "unit": "FTE",
        "category": "HEADCOUNT",
        "sectors": ["MULTI_SECTOR", "HEALTHCARE", "EDUCATION", "FINANCIAL_SERVICES", "SME"],
        "frameworks": ["ESRS_E1", "CDP", "GRI", "TCFD"],
        "description": "Full-time equivalent employees",
    },
    "headcount": {
        "id": "headcount",
        "name": "Total Headcount",
        "unit": "person",
        "category": "HEADCOUNT",
        "sectors": ["MULTI_SECTOR", "SME"],
        "frameworks": ["GRI"],
        "description": "Total number of employees (head count)",
    },
    "beds": {
        "id": "beds",
        "name": "Hospital Beds",
        "unit": "bed",
        "category": "PHYSICAL",
        "sectors": ["HEALTHCARE"],
        "frameworks": ["CDP", "GRI"],
        "description": "Total number of staffed hospital beds",
    },
    "students_fte": {
        "id": "students_fte",
        "name": "Student FTE",
        "unit": "student_FTE",
        "category": "HEADCOUNT",
        "sectors": ["EDUCATION"],
        "frameworks": ["CDP", "GRI"],
        "description": "Full-time equivalent students enrolled",
    },
    # --- Energy Denominators ---
    "mwh_consumed": {
        "id": "mwh_consumed",
        "name": "Energy Consumed (MWh)",
        "unit": "MWh",
        "category": "ENERGY",
        "sectors": ["DATA_CENTER", "MANUFACTURING"],
        "frameworks": ["ESRS_E1", "CDP", "GRI"],
        "description": "Total energy consumed in megawatt-hours",
    },
    "mwh_it_load": {
        "id": "mwh_it_load",
        "name": "IT Load (MWh)",
        "unit": "MWh",
        "category": "ENERGY",
        "sectors": ["DATA_CENTER"],
        "frameworks": ["CDP"],
        "description": "IT equipment energy load in megawatt-hours (for PUE-adjusted intensity)",
    },
    "rack_count": {
        "id": "rack_count",
        "name": "Rack Count",
        "unit": "rack",
        "category": "PHYSICAL",
        "sectors": ["DATA_CENTER"],
        "frameworks": ["CDP"],
        "description": "Total number of server racks deployed",
    },
}

SBTI_SECTOR_PATHWAYS: Dict[str, Dict[str, Any]] = {
    "power_generation": {
        "sector": "Electricity Generation",
        "metric": "tCO2e/MWh",
        "base_year": 2020,
        "well_below_2c": {
            "2025": 0.34,
            "2030": 0.14,
            "2035": 0.05,
            "2040": 0.02,
            "2050": 0.00,
        },
        "one_point_five_c": {
            "2025": 0.28,
            "2030": 0.07,
            "2035": 0.01,
            "2040": 0.00,
            "2050": 0.00,
        },
        "source": "SBTi SDA Power Sector v2.0",
    },
    "cement": {
        "sector": "Cement",
        "metric": "tCO2e/tonne_cite",
        "base_year": 2020,
        "well_below_2c": {
            "2025": 0.57,
            "2030": 0.47,
            "2035": 0.38,
            "2040": 0.30,
            "2050": 0.15,
        },
        "one_point_five_c": {
            "2025": 0.52,
            "2030": 0.40,
            "2035": 0.29,
            "2040": 0.20,
            "2050": 0.06,
        },
        "source": "SBTi SDA Cement Sector v2.0",
    },
    "steel": {
        "sector": "Iron and Steel",
        "metric": "tCO2e/tonne_steel",
        "base_year": 2020,
        "well_below_2c": {
            "2025": 1.65,
            "2030": 1.35,
            "2035": 1.10,
            "2040": 0.85,
            "2050": 0.42,
        },
        "one_point_five_c": {
            "2025": 1.50,
            "2030": 1.10,
            "2035": 0.80,
            "2040": 0.55,
            "2050": 0.18,
        },
        "source": "SBTi SDA Iron & Steel Sector v2.0",
    },
    "aluminium": {
        "sector": "Aluminium",
        "metric": "tCO2e/tonne_aluminium",
        "base_year": 2020,
        "well_below_2c": {
            "2025": 7.80,
            "2030": 5.80,
            "2035": 4.20,
            "2040": 3.00,
            "2050": 1.20,
        },
        "one_point_five_c": {
            "2025": 7.00,
            "2030": 4.80,
            "2035": 3.10,
            "2040": 1.80,
            "2050": 0.50,
        },
        "source": "SBTi SDA Aluminium Sector v2.0",
    },
    "transport_road_freight": {
        "sector": "Road Freight Transport",
        "metric": "gCO2e/tkm",
        "base_year": 2020,
        "well_below_2c": {
            "2025": 78.0,
            "2030": 60.0,
            "2035": 45.0,
            "2040": 32.0,
            "2050": 10.0,
        },
        "one_point_five_c": {
            "2025": 72.0,
            "2030": 50.0,
            "2035": 33.0,
            "2040": 20.0,
            "2050": 3.0,
        },
        "source": "SBTi SDA Transport Sector v2.0",
    },
    "commercial_buildings": {
        "sector": "Commercial Buildings (Services)",
        "metric": "kgCO2e/m2",
        "base_year": 2020,
        "well_below_2c": {
            "2025": 42.0,
            "2030": 30.0,
            "2035": 20.0,
            "2040": 12.0,
            "2050": 2.0,
        },
        "one_point_five_c": {
            "2025": 38.0,
            "2030": 24.0,
            "2035": 14.0,
            "2040": 7.0,
            "2050": 0.5,
        },
        "source": "SBTi SDA Buildings Sector v2.0 / CRREM 1.5C",
    },
    "flag_agriculture": {
        "sector": "Forest, Land and Agriculture (FLAG)",
        "metric": "tCO2e/tonne_product",
        "base_year": 2020,
        "well_below_2c": {
            "2025": 0.90,
            "2030": 0.72,
            "2035": 0.58,
            "2040": 0.47,
            "2050": 0.30,
        },
        "one_point_five_c": {
            "2025": 0.82,
            "2030": 0.60,
            "2035": 0.42,
            "2040": 0.30,
            "2050": 0.15,
        },
        "source": "SBTi FLAG Guidance v1.1",
    },
}

DATA_QUALITY_UNCERTAINTY: Dict[int, Dict[str, Any]] = {
    1: {
        "level": "AUDITED",
        "description": "Third-party verified, metered, or audited data",
        "typical_uncertainty_pct": 2.0,
        "min_uncertainty_pct": 1.0,
        "max_uncertainty_pct": 5.0,
    },
    2: {
        "level": "CALCULATED",
        "description": "Calculated from activity data with verified emission factors",
        "typical_uncertainty_pct": 8.0,
        "min_uncertainty_pct": 3.0,
        "max_uncertainty_pct": 15.0,
    },
    3: {
        "level": "ESTIMATED",
        "description": "Estimated from partial data or regional averages",
        "typical_uncertainty_pct": 20.0,
        "min_uncertainty_pct": 10.0,
        "max_uncertainty_pct": 35.0,
    },
    4: {
        "level": "PROXY",
        "description": "Proxy data from similar facilities or industry averages",
        "typical_uncertainty_pct": 40.0,
        "min_uncertainty_pct": 25.0,
        "max_uncertainty_pct": 60.0,
    },
    5: {
        "level": "DEFAULT",
        "description": "Default values from literature or regulatory tables",
        "typical_uncertainty_pct": 60.0,
        "min_uncertainty_pct": 40.0,
        "max_uncertainty_pct": 100.0,
    },
}

SECTOR_INFO: Dict[str, Dict[str, Any]] = {
    "MULTI_SECTOR": {
        "name": "Multi-Sector Conglomerate",
        "primary_denominators": ["revenue_meur", "fte"],
        "secondary_denominators": ["ebitda_meur", "gross_profit_meur"],
        "typical_frameworks": ["ESRS_E1", "CDP", "GRI", "TCFD"],
        "benchmark_sources": ["CDP", "TPI"],
        "decomposition_levels": ["division", "business_unit", "region"],
        "typical_scope_inclusion": "SCOPE_1_2_MARKET",
        "notes": "Revenue-weighted intensity with decomposition by division",
    },
    "MANUFACTURING": {
        "name": "Manufacturing",
        "primary_denominators": ["tonnes_output", "units_produced"],
        "secondary_denominators": ["revenue_meur", "mwh_consumed"],
        "typical_frameworks": ["ESRS_E1", "CDP", "SBTI", "GRI", "ISO_14064"],
        "benchmark_sources": ["CDP", "TPI"],
        "decomposition_levels": ["product_line", "facility", "process"],
        "typical_scope_inclusion": "SCOPE_1_2_LOCATION",
        "notes": "Physical output intensity with SBTi SDA pathway alignment",
    },
    "REAL_ESTATE": {
        "name": "Real Estate Portfolio",
        "primary_denominators": ["sqm_floor_area", "sqm_lettable_area"],
        "secondary_denominators": ["revenue_meur"],
        "typical_frameworks": ["ESRS_E1", "CDP", "GRESB", "CRREM", "TCFD"],
        "benchmark_sources": ["GRESB", "CRREM", "CDP"],
        "decomposition_levels": ["asset_class", "geography", "vintage"],
        "typical_scope_inclusion": "SCOPE_1_2_LOCATION",
        "notes": "kgCO2e/m2 with CRREM pathway and GRESB benchmarking",
    },
    "POWER_GENERATION": {
        "name": "Power Generation",
        "primary_denominators": ["mwh_generated", "gwh_generated"],
        "secondary_denominators": ["revenue_meur"],
        "typical_frameworks": ["ESRS_E1", "CDP", "SBTI", "TCFD", "ISO_14064"],
        "benchmark_sources": ["CDP", "TPI"],
        "decomposition_levels": ["fuel_type", "plant", "technology"],
        "typical_scope_inclusion": "SCOPE_1_ONLY",
        "notes": "gCO2/kWh or tCO2e/MWh with SBTi power sector SDA pathway",
    },
    "TRANSPORT_LOGISTICS": {
        "name": "Transport & Logistics",
        "primary_denominators": ["tkm", "pkm"],
        "secondary_denominators": ["vehicle_km", "revenue_meur"],
        "typical_frameworks": ["ESRS_E1", "CDP", "SBTI", "GRI", "ISO_14064"],
        "benchmark_sources": ["CDP", "TPI"],
        "decomposition_levels": ["mode", "fleet_segment", "route"],
        "typical_scope_inclusion": "SCOPE_1_2_LOCATION",
        "notes": "gCO2e/tkm with GLEC Framework and SBTi transport pathway",
    },
    "FINANCIAL_SERVICES": {
        "name": "Financial Services",
        "primary_denominators": ["assets_under_management_meur", "total_lending_meur"],
        "secondary_denominators": ["revenue_meur", "fte"],
        "typical_frameworks": ["CDP", "TCFD", "SBTI", "IFRS_S2"],
        "benchmark_sources": ["CDP", "TPI"],
        "decomposition_levels": ["asset_class", "sector_exposure", "geography"],
        "typical_scope_inclusion": "SCOPE_1_2_3",
        "notes": "PCAF-aligned tCO2e/MEUR with financed emissions intensity",
    },
    "FOOD_AGRICULTURE": {
        "name": "Food & Agriculture",
        "primary_denominators": ["tonnes_output", "hectare_crop"],
        "secondary_denominators": ["tonnes_protein", "litres_milk", "revenue_meur"],
        "typical_frameworks": ["ESRS_E1", "CDP", "SBTI", "GRI"],
        "benchmark_sources": ["CDP"],
        "decomposition_levels": ["commodity", "farm_type", "region"],
        "typical_scope_inclusion": "SCOPE_1_2_3",
        "notes": "tCO2e/tonne product with SBTi FLAG pathway alignment",
    },
    "HEALTHCARE": {
        "name": "Healthcare",
        "primary_denominators": ["sqm_floor_area", "beds"],
        "secondary_denominators": ["fte", "revenue_meur"],
        "typical_frameworks": ["ESRS_E1", "CDP", "GRI"],
        "benchmark_sources": ["CDP"],
        "decomposition_levels": ["facility_type", "department", "region"],
        "typical_scope_inclusion": "SCOPE_1_2_LOCATION",
        "notes": "kgCO2e/m2 or tCO2e/bed with NHS Net Zero-aligned targets",
    },
    "EDUCATION": {
        "name": "Education",
        "primary_denominators": ["sqm_floor_area", "students_fte"],
        "secondary_denominators": ["fte", "revenue_meur"],
        "typical_frameworks": ["CDP", "GRI"],
        "benchmark_sources": ["CDP"],
        "decomposition_levels": ["campus", "building_type"],
        "typical_scope_inclusion": "SCOPE_1_2_LOCATION",
        "notes": "kgCO2e/m2 or tCO2e/student FTE",
    },
    "DATA_CENTER": {
        "name": "Data Centre",
        "primary_denominators": ["mwh_it_load", "rack_count"],
        "secondary_denominators": ["sqm_floor_area", "mwh_consumed"],
        "typical_frameworks": ["ESRS_E1", "CDP", "TCFD"],
        "benchmark_sources": ["CDP"],
        "decomposition_levels": ["facility", "tier_level"],
        "typical_scope_inclusion": "SCOPE_1_2_MARKET",
        "notes": "CUE (Carbon Usage Effectiveness) = tCO2e / MWh IT load",
    },
    "RETAIL": {
        "name": "Retail",
        "primary_denominators": ["sqm_floor_area", "revenue_meur"],
        "secondary_denominators": ["fte"],
        "typical_frameworks": ["ESRS_E1", "CDP", "GRI", "TCFD"],
        "benchmark_sources": ["CDP"],
        "decomposition_levels": ["store_format", "region"],
        "typical_scope_inclusion": "SCOPE_1_2_LOCATION",
        "notes": "kgCO2e/m2 sales floor or tCO2e/MEUR revenue",
    },
    "SME": {
        "name": "Small-Medium Enterprise",
        "primary_denominators": ["revenue_meur", "fte"],
        "secondary_denominators": [],
        "typical_frameworks": ["GRI", "CDP"],
        "benchmark_sources": [],
        "decomposition_levels": [],
        "typical_scope_inclusion": "SCOPE_1_2_LOCATION",
        "notes": "Simplified intensity using revenue or FTE; 5 engines only",
    },
}

AVAILABLE_PRESETS: Dict[str, str] = {
    "corporate_multi_sector": "Multi-sector conglomerate with revenue-weighted intensity and division decomposition",
    "manufacturing": "Manufacturing with physical output intensity and SBTi SDA pathway alignment",
    "real_estate": "Real estate portfolio with kgCO2e/m2, CRREM pathway, and GRESB benchmarking",
    "power_generation": "Power generation with gCO2/kWh, SBTi power sector pathway",
    "transport_logistics": "Transport and logistics with gCO2e/tkm, GLEC Framework, SBTi transport pathway",
    "financial_services": "Financial services with PCAF-aligned tCO2e/MEUR invested, financed emissions",
    "food_agriculture": "Food and agriculture with tCO2e/tonne product, SBTi FLAG pathway",
    "sme_simplified": "Simplified intensity metrics for SMEs using revenue and FTE denominators only",
}


# =============================================================================
# Sub-Config Models
# =============================================================================


class DenominatorConfig(BaseModel):
    """Configuration for denominator selection, validation, and unit management."""
    selected_denominators: List[str] = Field(
        default_factory=lambda: ["revenue_meur"],
        description="List of denominator IDs from STANDARD_DENOMINATORS",
    )
    primary_denominator: str = Field(
        "revenue_meur",
        description="Primary denominator for headline intensity metric",
    )
    custom_denominators: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Custom denominator definitions [{id, name, unit, category, description}]",
    )
    validation_rules: Dict[str, Any] = Field(
        default_factory=lambda: {
            "require_non_zero": True,
            "require_positive": True,
            "max_year_over_year_change_pct": 50.0,
            "require_same_period": True,
        },
        description="Validation rules for denominator values",
    )
    unit_preferences: Dict[str, str] = Field(
        default_factory=lambda: {
            "currency": "EUR",
            "area": "m2",
            "mass": "tonne",
            "distance": "km",
        },
        description="Preferred units by category for normalisation",
    )

    @field_validator("selected_denominators")
    @classmethod
    def validate_selected_denominators(cls, v: List[str]) -> List[str]:
        """Validate at least one denominator is selected."""
        if not v:
            raise ValueError("At least one denominator must be selected")
        return v

    @model_validator(mode="after")
    def validate_primary_in_selected(self) -> "DenominatorConfig":
        """Ensure primary denominator is within the selected list."""
        all_ids = list(self.selected_denominators)
        for custom in self.custom_denominators:
            if "id" in custom:
                all_ids.append(custom["id"])
        if self.primary_denominator not in all_ids:
            raise ValueError(
                f"primary_denominator '{self.primary_denominator}' must be in "
                f"selected_denominators or custom_denominators"
            )
        return self


class IntensityCalculationConfig(BaseModel):
    """Configuration for multi-scope intensity computation."""
    scope_inclusion: ScopeInclusion = Field(
        ScopeInclusion.SCOPE_1_2_MARKET,
        description="Which GHG scopes to include in the numerator",
    )
    scope_3_categories: List[int] = Field(
        default_factory=list,
        description="Scope 3 categories to include when scope_inclusion is SCOPE_1_2_3 or SCOPE_3_SPECIFIC",
    )
    decimal_places: int = Field(
        4, ge=0, le=10,
        description="Decimal places for intensity result rounding",
    )
    weighted_average_method: WeightedAverageMethod = Field(
        WeightedAverageMethod.ACTIVITY_WEIGHTED,
        description="Method for calculating weighted average intensity across entities",
    )
    null_handling: NullHandling = Field(
        NullHandling.EXCLUDE,
        description="Strategy for handling null numerator or denominator values",
    )
    exclude_negative_denominators: bool = Field(
        True,
        description="Exclude records where denominator is negative (e.g., negative revenue)",
    )
    include_biogenic: bool = Field(
        False,
        description="Include biogenic CO2 in the intensity numerator",
    )
    separate_location_market: bool = Field(
        True,
        description="Calculate separate intensities for Scope 2 location-based and market-based",
    )

    @field_validator("scope_3_categories")
    @classmethod
    def validate_scope_3_categories(cls, v: List[int]) -> List[int]:
        """Validate Scope 3 category numbers are within 1-15."""
        for cat in v:
            if cat < 1 or cat > 15:
                raise ValueError(f"Scope 3 category must be 1-15, got {cat}")
        return sorted(set(v))


class DecompositionConfig(BaseModel):
    """Configuration for LMDI decomposition analysis."""
    method: DecompositionMethod = Field(
        DecompositionMethod.LMDI_I_ADDITIVE,
        description="LMDI decomposition variant to use",
    )
    include_activity_effect: bool = Field(
        True, description="Decompose the activity (scale) effect",
    )
    include_structure_effect: bool = Field(
        True, description="Decompose the structural (mix) effect",
    )
    include_intensity_effect: bool = Field(
        True, description="Decompose the intensity (efficiency) effect",
    )
    multi_level: bool = Field(
        False, description="Enable multi-level decomposition (e.g., sector -> sub-sector -> facility)",
    )
    decomposition_levels: List[str] = Field(
        default_factory=lambda: ["division"],
        description="Hierarchy levels for decomposition (e.g., ['division', 'facility'])",
    )
    zero_handling: str = Field(
        "SMALL_VALUE",
        description="Strategy for handling zero values in LMDI (SMALL_VALUE, EXCLUDE, LIMIT)",
    )
    small_value_epsilon: float = Field(
        1e-10, gt=0,
        description="Small value substituted for zeros when zero_handling is SMALL_VALUE",
    )

    @field_validator("zero_handling")
    @classmethod
    def validate_zero_handling(cls, v: str) -> str:
        """Validate zero handling strategy."""
        allowed = {"SMALL_VALUE", "EXCLUDE", "LIMIT"}
        if v.upper() not in allowed:
            raise ValueError(f"zero_handling must be one of {allowed}, got '{v}'")
        return v.upper()


class BenchmarkConfig(BaseModel):
    """Configuration for sector and peer group benchmarking."""
    sources: List[BenchmarkSource] = Field(
        default_factory=lambda: [BenchmarkSource.CDP],
        description="External benchmark data sources to use",
    )
    peer_group_criteria: Dict[str, Any] = Field(
        default_factory=lambda: {
            "same_sector": True,
            "same_region": False,
            "revenue_range_pct": 50.0,
            "min_peers": 5,
        },
        description="Criteria for constructing the peer group",
    )
    normalisation_settings: Dict[str, Any] = Field(
        default_factory=lambda: {
            "purchasing_power_parity": False,
            "climate_zone_adjust": False,
            "occupancy_adjust": False,
        },
        description="Normalisation adjustments for fair comparison",
    )
    update_frequency: str = Field(
        "ANNUAL",
        description="How often benchmark data is refreshed (QUARTERLY, SEMI_ANNUAL, ANNUAL)",
    )
    percentile_thresholds: List[int] = Field(
        default_factory=lambda: [10, 25, 50, 75, 90],
        description="Percentile thresholds for benchmark positioning",
    )

    @field_validator("update_frequency")
    @classmethod
    def validate_update_frequency(cls, v: str) -> str:
        """Validate update frequency value."""
        allowed = {"QUARTERLY", "SEMI_ANNUAL", "ANNUAL"}
        if v.upper() not in allowed:
            raise ValueError(f"update_frequency must be one of {allowed}, got '{v}'")
        return v.upper()


class TargetConfig(BaseModel):
    """Configuration for SBTi SDA target pathway generation and tracking."""
    pathway: TargetPathway = Field(
        TargetPathway.ONE_POINT_FIVE_C,
        description="Temperature alignment pathway",
    )
    base_year: int = Field(
        2020, ge=2015, le=2030,
        description="Target pathway base year",
    )
    target_years: List[int] = Field(
        default_factory=lambda: [2025, 2030, 2035, 2040, 2050],
        description="Milestone years for target pathway",
    )
    sector: str = Field(
        "commercial_buildings",
        description="SBTi SDA sector key from SBTI_SECTOR_PATHWAYS",
    )
    custom_targets: Dict[str, float] = Field(
        default_factory=dict,
        description="Custom intensity targets by year (overrides SDA pathway)",
    )
    annual_progress_tracking: bool = Field(
        True, description="Calculate annual target vs actual progress",
    )
    convergence_approach: str = Field(
        "SDA",
        description="Target convergence approach (SDA, ACA, or CUSTOM)",
    )
    include_scope_3_target: bool = Field(
        False, description="Include separate Scope 3 intensity target",
    )

    @field_validator("convergence_approach")
    @classmethod
    def validate_convergence_approach(cls, v: str) -> str:
        """Validate convergence approach."""
        allowed = {"SDA", "ACA", "CUSTOM"}
        if v.upper() not in allowed:
            raise ValueError(f"convergence_approach must be one of {allowed}, got '{v}'")
        return v.upper()

    @model_validator(mode="after")
    def validate_target_years_sorted(self) -> "TargetConfig":
        """Ensure target years are sorted and after base year."""
        for year in self.target_years:
            if year <= self.base_year:
                raise ValueError(
                    f"Target year {year} must be after base_year {self.base_year}"
                )
        self.target_years = sorted(set(self.target_years))
        return self


class TrendConfig(BaseModel):
    """Configuration for time-series trend analysis and projection."""
    rolling_window: int = Field(
        5, ge=3, le=15,
        description="Rolling window size in years for trend calculations",
    )
    projection_years: int = Field(
        5, ge=1, le=30,
        description="Number of years to project forward",
    )
    regression_model: RegressionModel = Field(
        RegressionModel.OLS,
        description="Statistical model for trend regression",
    )
    significance_level: float = Field(
        0.05, gt=0.0, lt=1.0,
        description="Statistical significance level (alpha) for trend tests",
    )
    min_data_points: int = Field(
        3, ge=2, le=10,
        description="Minimum number of data points required for trend analysis",
    )
    detect_structural_breaks: bool = Field(
        True, description="Detect structural breaks in the intensity time series",
    )
    seasonal_adjustment: bool = Field(
        False, description="Apply seasonal adjustment for sub-annual data",
    )


class ScenarioConfig(BaseModel):
    """Configuration for what-if scenario modelling and sensitivity analysis."""
    scenario_types: List[ScenarioType] = Field(
        default_factory=lambda: [ScenarioType.EFFICIENCY, ScenarioType.GROWTH],
        description="Types of scenarios to model",
    )
    monte_carlo_iterations: int = Field(
        10000, ge=1000, le=1000000,
        description="Number of Monte Carlo iterations for probabilistic scenarios",
    )
    confidence_levels: List[float] = Field(
        default_factory=lambda: [0.90, 0.95, 0.99],
        description="Confidence levels for scenario output ranges",
    )
    sensitivity_parameters: List[str] = Field(
        default_factory=lambda: [
            "emission_factor", "activity_data", "denominator_value",
        ],
        description="Parameters to include in sensitivity analysis",
    )
    sensitivity_range_pct: float = Field(
        20.0, ge=5.0, le=100.0,
        description="Percentage range (+/-) for one-at-a-time sensitivity analysis",
    )
    growth_rate_assumptions: Dict[str, float] = Field(
        default_factory=lambda: {
            "revenue_annual_pct": 5.0,
            "production_annual_pct": 3.0,
            "headcount_annual_pct": 2.0,
        },
        description="Default annual growth rate assumptions by denominator type",
    )
    efficiency_improvement_pct: float = Field(
        2.0, ge=0.0, le=20.0,
        description="Assumed annual efficiency improvement percentage",
    )


class UncertaintyConfig(BaseModel):
    """Configuration for uncertainty quantification and propagation."""
    propagation_method: PropagationMethod = Field(
        PropagationMethod.MONTE_CARLO,
        description="Method for propagating uncertainty through calculations",
    )
    confidence_interval: float = Field(
        0.95, gt=0.0, lt=1.0,
        description="Confidence interval for uncertainty bounds",
    )
    data_quality_defaults: Dict[int, float] = Field(
        default_factory=lambda: {
            1: 2.0,
            2: 8.0,
            3: 20.0,
            4: 40.0,
            5: 60.0,
        },
        description="Default uncertainty % by data quality level (1=best, 5=worst)",
    )
    monte_carlo_iterations: int = Field(
        10000, ge=1000, le=1000000,
        description="Number of Monte Carlo iterations for uncertainty propagation",
    )
    include_denominator_uncertainty: bool = Field(
        True,
        description="Include uncertainty in the denominator as well as the numerator",
    )
    correlation_assumptions: str = Field(
        "INDEPENDENT",
        description="Correlation assumption between input uncertainties (INDEPENDENT, CORRELATED, MIXED)",
    )

    @field_validator("correlation_assumptions")
    @classmethod
    def validate_correlation_assumptions(cls, v: str) -> str:
        """Validate correlation assumption value."""
        allowed = {"INDEPENDENT", "CORRELATED", "MIXED"}
        if v.upper() not in allowed:
            raise ValueError(f"correlation_assumptions must be one of {allowed}, got '{v}'")
        return v.upper()


class DisclosureConfig(BaseModel):
    """Configuration for multi-framework intensity disclosure mapping."""
    frameworks: List[DisclosureFramework] = Field(
        default_factory=lambda: [
            DisclosureFramework.ESRS_E1,
            DisclosureFramework.CDP,
            DisclosureFramework.GRI,
        ],
        description="Target disclosure frameworks",
    )
    mandatory_only: bool = Field(
        False,
        description="Only map mandatory (required) disclosure fields",
    )
    xbrl_taxonomy: str = Field(
        "ESRS_2024",
        description="XBRL taxonomy version for machine-readable tagging",
    )
    include_methodology_notes: bool = Field(
        True,
        description="Include methodology descriptions in disclosure output",
    )
    include_data_quality_notes: bool = Field(
        True,
        description="Include data quality information in disclosure output",
    )
    field_mapping_overrides: Dict[str, str] = Field(
        default_factory=dict,
        description="Manual overrides for framework field mappings",
    )


class ReportingConfig(BaseModel):
    """Configuration for intensity report generation."""
    formats: List[OutputFormat] = Field(
        default_factory=lambda: [OutputFormat.HTML, OutputFormat.JSON],
        description="Output format(s) for generated reports",
    )
    sections: List[str] = Field(
        default_factory=lambda: [
            "executive_summary",
            "intensity_results",
            "trend_analysis",
            "decomposition",
            "benchmarking",
            "target_progress",
            "uncertainty",
            "methodology",
        ],
        description="Report sections to include",
    )
    branding: Dict[str, str] = Field(
        default_factory=lambda: {
            "logo_url": "",
            "primary_colour": "#1B5E20",
            "company_name": "",
        },
        description="Report branding configuration",
    )
    language: str = Field("en", description="Report language (ISO 639-1)")
    include_charts: bool = Field(True, description="Include charts and visualisations")
    include_data_tables: bool = Field(True, description="Include detailed data tables")
    include_appendices: bool = Field(True, description="Include technical appendices")
    decimal_places_display: int = Field(
        2, ge=0, le=6,
        description="Decimal places for display in reports (may differ from calculation precision)",
    )


class PerformanceConfig(BaseModel):
    """Configuration for computational performance tuning."""
    max_calculation_time_seconds: int = Field(
        300, ge=30, le=3600,
        description="Maximum allowed calculation time in seconds",
    )
    cache_intensity_results: bool = Field(
        True, description="Cache calculated intensity results for repeated access",
    )
    parallel_denominator_processing: bool = Field(
        True, description="Process multiple denominators in parallel",
    )
    batch_size: int = Field(
        500, ge=50, le=5000,
        description="Batch size for bulk entity processing",
    )
    cache_ttl_seconds: int = Field(
        3600, ge=60, le=86400,
        description="Cache TTL in seconds",
    )
    lazy_load_benchmarks: bool = Field(
        True, description="Lazy-load benchmark data only when needed",
    )


class SecurityConfig(BaseModel):
    """Configuration for access control and data protection."""
    rbac_enabled: bool = Field(True, description="Enable role-based access control")
    audit_trail_enabled: bool = Field(True, description="Enable audit trail for all operations")
    encryption_at_rest: bool = Field(True, description="Encrypt intensity data at rest (AES-256)")
    roles: List[str] = Field(
        default_factory=lambda: [
            "intensity_analyst", "intensity_manager", "reviewer",
            "approver", "benchmark_admin", "viewer", "admin",
        ],
        description="Available RBAC roles for intensity metrics management",
    )


class NotificationConfig(BaseModel):
    """Configuration for notification delivery on intensity metric events."""
    channels: List[NotificationChannel] = Field(
        default_factory=lambda: [NotificationChannel.EMAIL],
        description="Notification delivery channels",
    )
    notify_on_threshold_breach: bool = Field(
        True, description="Notify when intensity exceeds a target or benchmark threshold",
    )
    notify_on_decomposition_complete: bool = Field(
        False, description="Notify when decomposition analysis is complete",
    )
    notify_on_benchmark_update: bool = Field(
        True, description="Notify when new benchmark data is available",
    )
    notify_on_report_generated: bool = Field(
        True, description="Notify when an intensity report is generated",
    )


class IntegrationConfig(BaseModel):
    """Configuration for integration with other GreenLang packs."""
    pack041_enabled: bool = Field(True, description="Integrate with PACK-041 Scope 1-2 Complete")
    pack042_enabled: bool = Field(True, description="Integrate with PACK-042 Scope 3 Starter")
    pack043_enabled: bool = Field(False, description="Integrate with PACK-043 Scope 3 Complete (Enterprise)")
    pack044_enabled: bool = Field(True, description="Integrate with PACK-044 Inventory Management")
    pack045_enabled: bool = Field(True, description="Integrate with PACK-045 Base Year Management")
    mrv_bridge_enabled: bool = Field(True, description="Bridge to MRV agent layer for emission data")
    erp_connector_enabled: bool = Field(False, description="Connect to ERP for denominator data")
    benchmark_api_enabled: bool = Field(False, description="Connect to external benchmark APIs")


class AuditConfig(BaseModel):
    """Configuration for audit trail and provenance tracking."""
    sha256_provenance: bool = Field(
        True, description="Enable SHA-256 provenance hashing for all calculations",
    )
    track_all_changes: bool = Field(
        True, description="Log every change to intensity data with full lineage",
    )
    evidence_retention_years: int = Field(
        7, ge=3, le=15, description="Evidence retention period in years",
    )
    require_digital_signature: bool = Field(
        False, description="Require digital signature on intensity report approvals",
    )
    log_calculation_inputs: bool = Field(
        True, description="Log all calculation inputs for reproducibility",
    )


# =============================================================================
# Main Configuration Model
# =============================================================================


class IntensityMetricsConfig(BaseModel):
    """
    Top-level configuration for PACK-046 Intensity Metrics.

    Combines all sub-configurations required for denominator management,
    intensity calculation, LMDI decomposition, benchmarking, SBTi SDA
    target tracking, trend analysis, scenario modelling, uncertainty
    quantification, disclosure mapping, and reporting.
    """
    company_name: str = Field("", description="Reporting company legal name")
    sector: IntensitySector = Field(
        IntensitySector.MULTI_SECTOR, description="Sector classification",
    )
    consolidation_approach: ConsolidationApproach = Field(
        ConsolidationApproach.OPERATIONAL_CONTROL,
        description="Organisational boundary approach",
    )
    country: str = Field("DE", description="Primary country (ISO 3166-1 alpha-2)")
    reporting_year: int = Field(2026, ge=2020, le=2035, description="Current reporting year")
    base_year: int = Field(2020, ge=2015, le=2030, description="Base year for intensity trends")
    revenue_meur: Optional[float] = Field(None, ge=0, description="Annual revenue in MEUR")
    employees_fte: Optional[int] = Field(None, ge=0, description="Full-time equivalent employees")

    denominator: DenominatorConfig = Field(default_factory=DenominatorConfig)
    intensity_calculation: IntensityCalculationConfig = Field(default_factory=IntensityCalculationConfig)
    decomposition: DecompositionConfig = Field(default_factory=DecompositionConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    trend: TrendConfig = Field(default_factory=TrendConfig)
    scenario: ScenarioConfig = Field(default_factory=ScenarioConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    disclosure: DisclosureConfig = Field(default_factory=DisclosureConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    notification: NotificationConfig = Field(default_factory=NotificationConfig)
    integration: IntegrationConfig = Field(default_factory=IntegrationConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)

    @model_validator(mode="after")
    def validate_sme_simplified(self) -> "IntensityMetricsConfig":
        """Apply SME-specific simplifications to reduce configuration burden."""
        if self.sector == IntensitySector.SME:
            # Limit to revenue and FTE only
            allowed = {"revenue_meur", "fte"}
            self.denominator.selected_denominators = [
                d for d in self.denominator.selected_denominators if d in allowed
            ]
            if not self.denominator.selected_denominators:
                self.denominator.selected_denominators = ["revenue_meur", "fte"]
            if self.denominator.primary_denominator not in allowed:
                self.denominator.primary_denominator = "revenue_meur"
            # Disable advanced features
            self.decomposition.multi_level = False
            self.decomposition.decomposition_levels = []
            self.benchmark.sources = []
            self.scenario.monte_carlo_iterations = 1000
            self.uncertainty.monte_carlo_iterations = 1000
            if self.intensity_calculation.scope_inclusion == ScopeInclusion.SCOPE_1_2_3:
                logger.warning("SME preset: Scope 3 disabled for simplified management.")
                self.intensity_calculation.scope_inclusion = ScopeInclusion.SCOPE_1_2_LOCATION
        return self

    @model_validator(mode="after")
    def validate_scope_3_consistency(self) -> "IntensityMetricsConfig":
        """Ensure Scope 3 categories are provided when SCOPE_1_2_3 or SCOPE_3_SPECIFIC."""
        si = self.intensity_calculation.scope_inclusion
        if si in (ScopeInclusion.SCOPE_1_2_3, ScopeInclusion.SCOPE_3_SPECIFIC):
            if not self.intensity_calculation.scope_3_categories:
                logger.warning(
                    "Scope 3 included in numerator but no categories specified; "
                    "defaulting to categories 1-3."
                )
                self.intensity_calculation.scope_3_categories = [1, 2, 3]
        return self

    @model_validator(mode="after")
    def validate_base_year_consistency(self) -> "IntensityMetricsConfig":
        """Ensure base year is before reporting year."""
        if self.base_year > self.reporting_year:
            raise ValueError(
                f"base_year ({self.base_year}) cannot be after "
                f"reporting_year ({self.reporting_year})"
            )
        return self

    @model_validator(mode="after")
    def validate_financial_denominator_currency(self) -> "IntensityMetricsConfig":
        """Warn if financial denominators are selected without revenue data."""
        financial_denoms = {"revenue_meur", "revenue_musd", "ebitda_meur",
                           "gross_profit_meur", "assets_under_management_meur",
                           "total_lending_meur"}
        selected_financial = set(self.denominator.selected_denominators) & financial_denoms
        if selected_financial and self.revenue_meur is None:
            logger.warning(
                f"Financial denominators selected ({selected_financial}) but "
                "revenue_meur is not set. Ensure denominator values are provided at runtime."
            )
        return self


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """
    Top-level wrapper for PACK-046 configuration.

    Provides factory methods for loading from presets, YAML files,
    environment overrides, and runtime merges. Includes SHA-256
    config hashing for provenance tracking.
    """
    pack: IntensityMetricsConfig = Field(default_factory=IntensityMetricsConfig)
    preset_name: Optional[str] = Field(None, description="Name of the loaded preset")
    config_version: str = Field("1.0.0", description="Configuration schema version")
    pack_id: str = Field("PACK-046-intensity-metrics", description="Unique pack identifier")

    @classmethod
    def from_preset(cls, preset_name: str, overrides: Optional[Dict[str, Any]] = None) -> "PackConfig":
        """
        Load configuration from a named sector preset.

        Args:
            preset_name: Key from AVAILABLE_PRESETS (e.g., 'manufacturing').
            overrides: Optional dict of overrides applied after preset load.

        Returns:
            Fully initialised PackConfig.

        Raises:
            ValueError: If preset_name is not recognised.
            FileNotFoundError: If preset YAML file is missing.
        """
        if preset_name not in AVAILABLE_PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. Available: {sorted(AVAILABLE_PRESETS.keys())}"
            )
        preset_path = CONFIG_DIR / "presets" / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Preset file not found: {preset_path}")
        with open(preset_path, "r", encoding="utf-8") as f:
            preset_data = yaml.safe_load(f) or {}
        env_overrides = cls._load_env_overrides()
        if env_overrides:
            preset_data = cls._deep_merge(preset_data, env_overrides)
        if overrides:
            preset_data = cls._deep_merge(preset_data, overrides)
        pack_config = IntensityMetricsConfig(**preset_data)
        return cls(pack=pack_config, preset_name=preset_name)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "PackConfig":
        """
        Load configuration from an arbitrary YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            Fully initialised PackConfig.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        pack_config = IntensityMetricsConfig(**config_data)
        return cls(pack=pack_config)

    @classmethod
    def merge(cls, base: "PackConfig", overrides: Dict[str, Any]) -> "PackConfig":
        """
        Create a new PackConfig by merging overrides into a base config.

        Args:
            base: Existing PackConfig to use as the base.
            overrides: Dict of overrides (supports nested keys).

        Returns:
            New PackConfig with merged values.
        """
        base_dict = base.pack.model_dump()
        merged = cls._deep_merge(base_dict, overrides)
        pack_config = IntensityMetricsConfig(**merged)
        return cls(pack=pack_config, preset_name=base.preset_name, config_version=base.config_version)

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """
        Load configuration overrides from environment variables.

        Environment variables prefixed with INTENSITY_PACK_ are parsed.
        Double underscores denote nested keys.
        Example: INTENSITY_PACK_DECOMPOSITION__METHOD=LMDI_I_MULTIPLICATIVE
        """
        overrides: Dict[str, Any] = {}
        prefix = "INTENSITY_PACK_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                parts = config_key.split("__")
                current = overrides
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
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
        """Recursively merge override dict into base dict."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = PackConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get_config_hash(self) -> str:
        """
        Compute SHA-256 hash of the full configuration.

        Returns:
            Hex-encoded SHA-256 hash string for provenance tracking.
        """
        config_json = self.model_dump_json(indent=None)
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()

    def validate_completeness(self) -> List[str]:
        """
        Run domain-specific validation checks on the configuration.

        Returns:
            List of warning messages (empty list means no issues).
        """
        return validate_config(self.pack)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise the full configuration to a plain dictionary.

        Returns:
            Dict representation of the entire PackConfig.
        """
        return self.model_dump()


# =============================================================================
# Utility Functions
# =============================================================================


def load_preset(preset_name: str, overrides: Optional[Dict[str, Any]] = None) -> PackConfig:
    """
    Convenience function to load a preset configuration.

    Args:
        preset_name: Key from AVAILABLE_PRESETS.
        overrides: Optional dict of overrides.

    Returns:
        Initialised PackConfig from the named preset.
    """
    return PackConfig.from_preset(preset_name, overrides)


def validate_config(config: IntensityMetricsConfig) -> List[str]:
    """
    Validate configuration for domain-specific consistency.

    Args:
        config: The intensity metrics configuration to validate.

    Returns:
        List of warning strings. Empty list indicates no issues found.
    """
    warnings: List[str] = []

    # Company name check
    if not config.company_name:
        warnings.append("No company_name configured.")

    # Denominator existence check
    for denom_id in config.denominator.selected_denominators:
        custom_ids = [c.get("id") for c in config.denominator.custom_denominators]
        if denom_id not in STANDARD_DENOMINATORS and denom_id not in custom_ids:
            warnings.append(
                f"Denominator '{denom_id}' is not in STANDARD_DENOMINATORS and "
                "not defined as a custom denominator."
            )

    # Sector-denominator alignment check
    sector_key = config.sector.value
    if sector_key in SECTOR_INFO:
        sector_data = SECTOR_INFO[sector_key]
        recommended = set(sector_data.get("primary_denominators", []))
        selected = set(config.denominator.selected_denominators)
        if recommended and not (selected & recommended):
            warnings.append(
                f"No recommended denominators for sector {sector_key} are selected. "
                f"Recommended: {sorted(recommended)}. Selected: {sorted(selected)}."
            )

    # Scope 3 integration check
    si = config.intensity_calculation.scope_inclusion
    if si in (ScopeInclusion.SCOPE_1_2_3, ScopeInclusion.SCOPE_3_SPECIFIC):
        if not config.integration.pack042_enabled and not config.integration.pack043_enabled:
            warnings.append(
                "Scope 3 is included in intensity numerator but neither "
                "PACK-042 nor PACK-043 integration is enabled."
            )

    # SBTi pathway sector check
    if config.target.sector not in SBTI_SECTOR_PATHWAYS:
        if config.target.convergence_approach == "SDA":
            warnings.append(
                f"SBTi SDA pathway sector '{config.target.sector}' is not in "
                f"SBTI_SECTOR_PATHWAYS. Available: {sorted(SBTI_SECTOR_PATHWAYS.keys())}. "
                "Use convergence_approach=CUSTOM if no SDA pathway exists."
            )

    # Base year / target base year alignment
    if config.target.base_year != config.base_year:
        warnings.append(
            f"Pack base_year ({config.base_year}) differs from target pathway "
            f"base_year ({config.target.base_year}). Ensure alignment for consistent tracking."
        )

    # Decomposition levels check
    if config.decomposition.multi_level and not config.decomposition.decomposition_levels:
        warnings.append(
            "Multi-level decomposition is enabled but no decomposition_levels are defined."
        )

    # Uncertainty - denominator uncertainty requires separate propagation
    if (
        config.uncertainty.include_denominator_uncertainty
        and config.uncertainty.propagation_method != PropagationMethod.MONTE_CARLO
    ):
        warnings.append(
            "Denominator uncertainty propagation is most accurate with MONTE_CARLO method. "
            f"Current method: {config.uncertainty.propagation_method.value}."
        )

    # Benchmark source and framework alignment
    if BenchmarkSource.GRESB in config.benchmark.sources:
        if config.sector not in (IntensitySector.REAL_ESTATE,):
            warnings.append(
                "GRESB benchmark source is selected but sector is not REAL_ESTATE. "
                "GRESB scores are only available for real estate portfolios."
            )

    if BenchmarkSource.CRREM in config.benchmark.sources:
        if config.sector not in (IntensitySector.REAL_ESTATE,):
            warnings.append(
                "CRREM benchmark source is selected but sector is not REAL_ESTATE."
            )

    # Reporting and audit consistency
    if config.audit.require_digital_signature and not config.security.rbac_enabled:
        warnings.append(
            "Digital signatures require RBAC to be enabled for signer identity verification."
        )

    # SME high-complexity warning
    if config.sector == IntensitySector.SME:
        if config.decomposition.multi_level:
            warnings.append(
                "Multi-level decomposition is unusual for SME sector. "
                "Consider disabling for simplicity."
            )
        if len(config.disclosure.frameworks) > 3:
            warnings.append(
                f"SME sector with {len(config.disclosure.frameworks)} disclosure frameworks "
                "may create unnecessary reporting burden."
            )

    return warnings


def get_default_config(
    sector: IntensitySector = IntensitySector.MULTI_SECTOR,
) -> IntensityMetricsConfig:
    """
    Create a default configuration for the given sector type.

    Args:
        sector: Industry sector classification.

    Returns:
        Default IntensityMetricsConfig for the sector.
    """
    return IntensityMetricsConfig(sector=sector)


def list_available_presets() -> Dict[str, str]:
    """
    Return a copy of all available preset names and descriptions.

    Returns:
        Dict mapping preset name to human-readable description.
    """
    return AVAILABLE_PRESETS.copy()


def get_sector_info(sector: Union[str, IntensitySector]) -> Dict[str, Any]:
    """
    Return sector reference data for a given sector type.

    Args:
        sector: Sector enum value or string key.

    Returns:
        Dict of sector metadata including recommended denominators and frameworks.
    """
    key = sector.value if isinstance(sector, IntensitySector) else sector
    return SECTOR_INFO.get(key, {"name": key, "primary_denominators": ["revenue_meur"]})


def get_denominator_info(denominator_id: str) -> Optional[Dict[str, Any]]:
    """
    Return reference data for a standard denominator.

    Args:
        denominator_id: Denominator ID from STANDARD_DENOMINATORS.

    Returns:
        Dict of denominator metadata, or None if not found.
    """
    return STANDARD_DENOMINATORS.get(denominator_id)


def get_sbti_pathway(
    sector_key: str, pathway: TargetPathway = TargetPathway.ONE_POINT_FIVE_C,
) -> Optional[Dict[str, float]]:
    """
    Return SBTi SDA pathway targets for a given sector and temperature scenario.

    Args:
        sector_key: Key from SBTI_SECTOR_PATHWAYS.
        pathway: Target temperature pathway.

    Returns:
        Dict mapping year string to target intensity value, or None if sector not found.
    """
    sector_data = SBTI_SECTOR_PATHWAYS.get(sector_key)
    if sector_data is None:
        return None
    pathway_key = pathway.value.lower()
    return sector_data.get(pathway_key)
