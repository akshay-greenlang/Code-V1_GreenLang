"""
PACK-028 Sector Pathway Pack - Configuration Manager

This module implements the SectorPathwayConfig and PackConfig classes that load,
merge, and validate all configuration for the Sector Pathway Pack. It provides
comprehensive Pydantic v2 models for sector-specific decarbonization pathway
analysis: SBTi SDA pathway design, IEA NZE 2050 scenario alignment, IPCC AR6
pathway integration, sector intensity convergence modeling, technology transition
roadmaps, abatement waterfall analysis, sector benchmarking, and multi-scenario
pathway comparison.

Primary Sectors (15+):
    - POWER: Power generation, grid electricity (gCO2/kWh)
    - STEEL: Iron and steel production (tCO2e/tonne crude steel)
    - CEMENT: Cement and clinker production (tCO2e/tonne cement)
    - ALUMINUM: Primary and secondary aluminium (tCO2e/tonne aluminium)
    - CHEMICALS: Chemical manufacturing (tCO2e/tonne product)
    - AVIATION: Commercial and freight aviation (gCO2/pkm)
    - SHIPPING: Maritime shipping (gCO2/tkm)
    - ROAD_TRANSPORT: Road freight and passenger (gCO2/vkm)
    - RAIL: Passenger and freight rail (gCO2/pkm)
    - BUILDINGS_RESIDENTIAL: Residential buildings (kgCO2/m2/year)
    - BUILDINGS_COMMERCIAL: Commercial buildings (kgCO2/m2/year)
    - AGRICULTURE: Agriculture and land use (tCO2e/tonne food)
    - FOOD_BEVERAGE: Food and beverage processing (tCO2e/tonne product)
    - PULP_PAPER: Pulp and paper manufacturing (tCO2e/tonne product)
    - OIL_GAS: Oil and gas upstream/downstream (gCO2/MJ energy)
    - MIXED: Multi-sector conglomerates (sector-weighted composite)

SBTi SDA Sectors (12):
    POWER, CEMENT, STEEL, ALUMINIUM, PULP_PAPER, CHEMICALS,
    TRANSPORT_ROAD, TRANSPORT_RAIL, AVIATION, SHIPPING,
    BUILDINGS_RESIDENTIAL, BUILDINGS_COMMERCIAL

IEA NZE 2050 Scenarios:
    - NZE: Net Zero Emissions by 2050 (1.5C, 50% probability)
    - WB2C: Well-Below 2C (<2C, 66% probability)
    - 2C: 2 Degrees Celsius (+2C, 50% probability)
    - APS: Announced Pledges Scenario (+1.7C)
    - STEPS: Stated Policies Scenario (+2.4C)

IPCC AR6 Pathways:
    C1-C8 illustrative pathways with varying overshoot levels

Convergence Models:
    - LINEAR: Linear convergence (constant annual reduction)
    - EXPONENTIAL: Exponential decay (constant percentage reduction)
    - S_CURVE: Sigmoidal S-curve (technology adoption pattern)
    - STEPPED: Stepped convergence (policy milestone driven)

Technology Readiness Levels:
    TRL 1 (basic research) through TRL 9 (commercial deployment)

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (6 sector presets)
    3. Environment overrides (SECTOR_PATHWAY_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - SBTi Corporate Standard v2.0 (2024)
    - SBTi Sectoral Decarbonization Approach (2015, updated 2024)
    - SBTi FLAG Guidance v1.1 (2024)
    - IEA Net Zero by 2050 Roadmap (2023 update)
    - IEA Energy Technology Perspectives 2023
    - IPCC AR6 WG3 Mitigation (2022)
    - GHG Protocol Corporate Standard (revised)
    - EU ETS Phase 4 Benchmarks
    - IMO GHG Strategy (2023 revision)
    - ISO 14064-1:2018
    - Paris Agreement Article 4

Example:
    >>> config = PackConfig.from_preset("heavy_industry")
    >>> print(config.pack.primary_sector)
    PrimarySector.STEEL
    >>> print(config.pack.sbti_sda.sbti_pathway)
    SBTiPathway.CELSIUS_1_5
    >>> config = PackConfig.from_preset("power_generation", overrides={"iea_region": "US"})
"""

import hashlib
import logging
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Base directory for all pack configuration files
PACK_BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(__file__).resolve().parent


# =============================================================================
# Constants
# =============================================================================

DEFAULT_BASE_YEAR: int = 2019
DEFAULT_TARGET_YEAR_NEAR: int = 2030
DEFAULT_TARGET_YEAR_LONG: int = 2050
DEFAULT_CARBON_PRICE_FLOOR: float = 50.0
DEFAULT_CARBON_PRICE_CEILING: float = 200.0
DEFAULT_CONVERGENCE_RATE: float = 0.05
DEFAULT_CATCH_UP_PERIOD: int = 10
DEFAULT_TRL_THRESHOLD: int = 7
DEFAULT_ABATEMENT_TARGET: float = 0.80
DEFAULT_LEADER_PERCENTILE: int = 10

SUPPORTED_PRESETS: Dict[str, str] = {
    "power_generation": "Power sector with gCO2/kWh intensity, coal phase-out, renewable targets",
    "heavy_industry": "Steel, Cement, Aluminum with process emissions, CCUS, hydrogen",
    "chemicals": "Chemical manufacturing with steam cracking, green hydrogen, bio-feedstocks",
    "transport": "Aviation, Shipping, Road, Rail with SAF, e-fuels, electrification",
    "buildings": "Commercial/residential with heat pumps, insulation, renewable heat",
    "mixed_sectors": "Conglomerates with multiple SDA-eligible sectors",
}


# =============================================================================
# SBTi SDA Sector Reference Data
# =============================================================================

# SBTi SDA supported sectors (12 sectors)
SDA_SECTORS: Dict[str, str] = {
    "POWER": "Power Generation",
    "CEMENT": "Cement",
    "STEEL": "Iron and Steel",
    "ALUMINIUM": "Aluminium",
    "PULP_PAPER": "Pulp and Paper",
    "CHEMICALS": "Chemicals",
    "TRANSPORT_ROAD": "Road Transport",
    "TRANSPORT_RAIL": "Rail Transport",
    "AVIATION": "Aviation",
    "SHIPPING": "Shipping",
    "BUILDINGS_RESIDENTIAL": "Residential Buildings",
    "BUILDINGS_COMMERCIAL": "Commercial Buildings",
}

# SBTi SDA sector intensity metrics
SDA_INTENSITY_METRICS: Dict[str, str] = {
    "POWER": "gCO2/kWh",
    "CEMENT": "tCO2e/tonne cement",
    "STEEL": "tCO2e/tonne crude steel",
    "ALUMINIUM": "tCO2e/tonne aluminium",
    "PULP_PAPER": "tCO2e/tonne product",
    "CHEMICALS": "tCO2e/tonne product",
    "TRANSPORT_ROAD": "gCO2/vkm",
    "TRANSPORT_RAIL": "gCO2/pkm",
    "AVIATION": "gCO2/pkm",
    "SHIPPING": "gCO2/tkm",
    "BUILDINGS_RESIDENTIAL": "kgCO2/m2/year",
    "BUILDINGS_COMMERCIAL": "kgCO2/m2/year",
}

# Extended sector intensity metrics (non-SDA sectors)
EXTENDED_INTENSITY_METRICS: Dict[str, str] = {
    "AGRICULTURE": "tCO2e/tonne food",
    "FOOD_BEVERAGE": "tCO2e/tonne product",
    "OIL_GAS": "gCO2/MJ energy produced",
    "MIXED": "tCO2e/M EUR revenue",
}

# All supported intensity metrics (SDA + extended)
ALL_INTENSITY_METRICS: Dict[str, str] = {**SDA_INTENSITY_METRICS, **EXTENDED_INTENSITY_METRICS}

# SBTi SDA 2050 convergence targets by sector (tCO2e per unit, NZE 1.5C)
SDA_2050_TARGETS: Dict[str, float] = {
    "POWER": 0.0,             # gCO2/kWh (net zero grid)
    "CEMENT": 0.12,           # tCO2e/tonne cement
    "STEEL": 0.06,            # tCO2e/tonne crude steel
    "ALUMINIUM": 0.50,        # tCO2e/tonne aluminium
    "PULP_PAPER": 0.05,       # tCO2e/tonne product
    "CHEMICALS": 0.10,        # tCO2e/tonne product
    "TRANSPORT_ROAD": 0.0,    # gCO2/vkm (full electrification)
    "TRANSPORT_RAIL": 0.0,    # gCO2/pkm (full electrification)
    "AVIATION": 15.0,         # gCO2/pkm (SAF + efficiency)
    "SHIPPING": 2.0,          # gCO2/tkm (green fuels)
    "BUILDINGS_RESIDENTIAL": 2.0,  # kgCO2/m2/year
    "BUILDINGS_COMMERCIAL": 3.0,   # kgCO2/m2/year
}

# IEA NZE scenario parameters
IEA_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "NZE": {
        "name": "Net Zero Emissions by 2050",
        "temperature": 1.5,
        "probability_pct": 50,
        "description": "IEA NZE 2050 (2023 update) aligned with 1.5C",
    },
    "WB2C": {
        "name": "Well-Below 2C",
        "temperature": 1.8,
        "probability_pct": 66,
        "description": "Well-below 2C with limited overshoot",
    },
    "2C": {
        "name": "2 Degrees Celsius",
        "temperature": 2.0,
        "probability_pct": 50,
        "description": "Paris Agreement minimum ambition",
    },
    "APS": {
        "name": "Announced Pledges Scenario",
        "temperature": 1.7,
        "probability_pct": None,
        "description": "Based on announced government pledges and NDCs",
    },
    "STEPS": {
        "name": "Stated Policies Scenario",
        "temperature": 2.4,
        "probability_pct": None,
        "description": "Current policies and measures in place",
    },
}

# IPCC AR6 pathway categories
IPCC_PATHWAYS: Dict[str, Dict[str, Any]] = {
    "C1": {"temperature": 1.5, "overshoot": "no", "description": "1.5C with no or limited overshoot"},
    "C2": {"temperature": 1.5, "overshoot": "limited", "description": "1.5C with high overshoot"},
    "C3": {"temperature": 2.0, "overshoot": "no", "description": "Likely below 2C"},
    "C4": {"temperature": 2.0, "overshoot": "limited", "description": "Below 2C"},
    "C5": {"temperature": 2.5, "overshoot": "no", "description": "Below 2.5C"},
    "C6": {"temperature": 3.0, "overshoot": "no", "description": "Below 3C"},
    "C7": {"temperature": 4.0, "overshoot": "no", "description": "Below 4C"},
    "C8": {"temperature": 4.0, "overshoot": "high", "description": "Above 4C"},
}

# IPCC AR6 GWP100 values for common GHGs
IPCC_AR6_GWP100: Dict[str, int] = {
    "CO2": 1,
    "CH4": 27,
    "N2O": 273,
    "HFC_134A": 1430,
    "HFC_32": 675,
    "R404A": 3922,
    "R410A": 2088,
    "SF6": 25200,
    "NF3": 17400,
    "R290": 3,
    "R744": 1,
}

# SBTi minimum annual reduction rates by ambition level (% per year)
SBTI_REDUCTION_RATES: Dict[str, Dict[str, float]] = {
    "CELSIUS_1_5": {
        "scope_1_2_linear_annual": 4.2,
        "scope_3_linear_annual": 2.5,
        "long_term_reduction_pct": 90.0,
    },
    "WELL_BELOW_2": {
        "scope_1_2_linear_annual": 2.5,
        "scope_3_linear_annual": 1.8,
        "long_term_reduction_pct": 90.0,
    },
    "CELSIUS_2": {
        "scope_1_2_linear_annual": 1.5,
        "scope_3_linear_annual": 1.2,
        "long_term_reduction_pct": 80.0,
    },
}

# SBTi coverage thresholds
SBTI_COVERAGE_THRESHOLDS: Dict[str, float] = {
    "scope_1_near_term_pct": 95.0,
    "scope_2_near_term_pct": 95.0,
    "scope_3_near_term_pct": 67.0,
    "scope_1_long_term_pct": 95.0,
    "scope_2_long_term_pct": 95.0,
    "scope_3_long_term_pct": 90.0,
}

# Sector-specific key decarbonization levers
SECTOR_DECARBONIZATION_LEVERS: Dict[str, List[str]] = {
    "POWER": [
        "coal_phase_out", "renewable_capacity", "grid_storage", "nuclear",
        "ccs_power", "hydrogen_turbines", "demand_response", "grid_modernization",
    ],
    "STEEL": [
        "eaf_conversion", "green_hydrogen_dri", "ccs_blast_furnace",
        "scrap_recycling", "process_optimization", "renewable_electricity",
        "biomass_injection", "electrochemical_reduction",
    ],
    "CEMENT": [
        "clinker_substitution", "ccs_cement", "alternative_fuels",
        "waste_heat_recovery", "novel_cements", "calcined_clay",
        "carbon_curing", "electrification_kilns",
    ],
    "ALUMINIUM": [
        "inert_anodes", "renewable_smelting", "scrap_recycling",
        "energy_efficiency", "ccs_alumina", "digital_process_control",
        "mechanical_vapour_recompression", "demand_reduction",
    ],
    "CHEMICALS": [
        "green_hydrogen", "electrification_crackers", "bio_feedstocks",
        "ccs_chemicals", "catalyst_optimization", "waste_heat_integration",
        "circular_chemistry", "methanol_to_olefins",
    ],
    "AVIATION": [
        "saf_blending", "airframe_efficiency", "engine_improvement",
        "fleet_renewal", "atm_optimization", "electric_short_haul",
        "hydrogen_aircraft", "demand_management",
    ],
    "SHIPPING": [
        "green_methanol", "green_ammonia", "lng_transition",
        "wind_assist", "hull_optimization", "slow_steaming",
        "shore_power", "electric_short_sea",
    ],
    "ROAD_TRANSPORT": [
        "ev_fleet_transition", "hydrogen_heavy_duty", "biofuels",
        "route_optimization", "eco_driving", "modal_shift",
        "autonomous_efficiency", "charging_infrastructure",
    ],
    "RAIL": [
        "electrification", "hydrogen_trains", "regenerative_braking",
        "lightweight_rolling_stock", "timetable_optimization",
        "renewable_traction", "battery_electric", "modal_shift",
    ],
    "BUILDINGS_RESIDENTIAL": [
        "heat_pumps", "deep_retrofit", "insulation_upgrade",
        "solar_thermal", "district_heating", "smart_controls",
        "renewable_heat", "behavior_change",
    ],
    "BUILDINGS_COMMERCIAL": [
        "heat_pumps", "building_automation", "led_lighting",
        "hvac_optimization", "envelope_upgrade", "on_site_solar",
        "green_leases", "occupancy_optimization",
    ],
    "AGRICULTURE": [
        "precision_agriculture", "nitrogen_management", "livestock_feed",
        "agroforestry", "soil_carbon", "methane_capture",
        "renewable_farm_energy", "cover_crops",
    ],
    "FOOD_BEVERAGE": [
        "process_electrification", "heat_recovery", "cold_chain_optimization",
        "packaging_reduction", "ingredient_sourcing", "water_efficiency",
        "biogas_from_waste", "logistics_optimization",
    ],
    "PULP_PAPER": [
        "biomass_boilers", "black_liquor_gasification", "waste_heat_recovery",
        "fiber_recycling", "process_electrification", "energy_efficiency",
        "sustainable_forestry", "ccs_pulp",
    ],
    "OIL_GAS": [
        "methane_leak_repair", "flaring_reduction", "electrification",
        "renewable_power", "ccs_upstream", "energy_efficiency",
        "hydrogen_production", "portfolio_diversification",
    ],
}

# Sector display names and pathway recommendations
SECTOR_INFO: Dict[str, Dict[str, Any]] = {
    "POWER": {
        "name": "Power Generation",
        "sda_eligible": True,
        "recommended_convergence": "exponential",
        "iea_chapter": "Chapter 3: Electricity",
        "typical_scope_split": "Scope 1: 60-85%, Scope 2: 2-5%, Scope 3: 10-30%",
    },
    "STEEL": {
        "name": "Iron and Steel",
        "sda_eligible": True,
        "recommended_convergence": "s_curve",
        "iea_chapter": "Chapter 5: Industry (Steel)",
        "typical_scope_split": "Scope 1: 50-75%, Scope 2: 10-20%, Scope 3: 15-35%",
    },
    "CEMENT": {
        "name": "Cement and Clinker",
        "sda_eligible": True,
        "recommended_convergence": "s_curve",
        "iea_chapter": "Chapter 5: Industry (Cement)",
        "typical_scope_split": "Scope 1: 60-80%, Scope 2: 10-15%, Scope 3: 10-25%",
    },
    "ALUMINIUM": {
        "name": "Aluminium Smelting",
        "sda_eligible": True,
        "recommended_convergence": "s_curve",
        "iea_chapter": "Chapter 5: Industry (Aluminium)",
        "typical_scope_split": "Scope 1: 30-50%, Scope 2: 30-50%, Scope 3: 15-25%",
    },
    "CHEMICALS": {
        "name": "Chemical Manufacturing",
        "sda_eligible": True,
        "recommended_convergence": "s_curve",
        "iea_chapter": "Chapter 5: Industry (Chemicals)",
        "typical_scope_split": "Scope 1: 40-60%, Scope 2: 15-25%, Scope 3: 20-40%",
    },
    "AVIATION": {
        "name": "Aviation",
        "sda_eligible": True,
        "recommended_convergence": "s_curve",
        "iea_chapter": "Chapter 4: Transport (Aviation)",
        "typical_scope_split": "Scope 1: 85-95%, Scope 2: 1-3%, Scope 3: 5-12%",
    },
    "SHIPPING": {
        "name": "Maritime Shipping",
        "sda_eligible": True,
        "recommended_convergence": "stepped",
        "iea_chapter": "Chapter 4: Transport (Shipping)",
        "typical_scope_split": "Scope 1: 80-95%, Scope 2: 1-3%, Scope 3: 5-15%",
    },
    "ROAD_TRANSPORT": {
        "name": "Road Transport",
        "sda_eligible": True,
        "recommended_convergence": "exponential",
        "iea_chapter": "Chapter 4: Transport (Road)",
        "typical_scope_split": "Scope 1: 70-90%, Scope 2: 2-10%, Scope 3: 5-20%",
    },
    "RAIL": {
        "name": "Rail Transport",
        "sda_eligible": True,
        "recommended_convergence": "linear",
        "iea_chapter": "Chapter 4: Transport (Rail)",
        "typical_scope_split": "Scope 1: 40-70%, Scope 2: 20-40%, Scope 3: 10-20%",
    },
    "BUILDINGS_RESIDENTIAL": {
        "name": "Residential Buildings",
        "sda_eligible": True,
        "recommended_convergence": "linear",
        "iea_chapter": "Chapter 2: Buildings (Residential)",
        "typical_scope_split": "Scope 1: 30-50%, Scope 2: 30-50%, Scope 3: 10-30%",
    },
    "BUILDINGS_COMMERCIAL": {
        "name": "Commercial Buildings",
        "sda_eligible": True,
        "recommended_convergence": "linear",
        "iea_chapter": "Chapter 2: Buildings (Commercial)",
        "typical_scope_split": "Scope 1: 20-40%, Scope 2: 35-55%, Scope 3: 15-30%",
    },
    "AGRICULTURE": {
        "name": "Agriculture and Land Use",
        "sda_eligible": False,
        "recommended_convergence": "linear",
        "iea_chapter": "Chapter 6: Agriculture",
        "typical_scope_split": "Scope 1: 40-70%, Scope 2: 5-15%, Scope 3: 20-45%",
    },
    "FOOD_BEVERAGE": {
        "name": "Food and Beverage Processing",
        "sda_eligible": False,
        "recommended_convergence": "linear",
        "iea_chapter": "Chapter 5: Industry (Food)",
        "typical_scope_split": "Scope 1: 25-45%, Scope 2: 15-25%, Scope 3: 35-55%",
    },
    "PULP_PAPER": {
        "name": "Pulp and Paper",
        "sda_eligible": True,
        "recommended_convergence": "exponential",
        "iea_chapter": "Chapter 5: Industry (Pulp)",
        "typical_scope_split": "Scope 1: 35-55%, Scope 2: 15-25%, Scope 3: 25-40%",
    },
    "OIL_GAS": {
        "name": "Oil and Gas",
        "sda_eligible": False,
        "recommended_convergence": "stepped",
        "iea_chapter": "Chapter 1: Energy Supply",
        "typical_scope_split": "Scope 1: 15-30%, Scope 2: 5-10%, Scope 3: 65-80%",
    },
    "MIXED": {
        "name": "Multi-Sector Conglomerate",
        "sda_eligible": False,
        "recommended_convergence": "linear",
        "iea_chapter": "Multiple chapters",
        "typical_scope_split": "Varies by sector composition",
    },
}


# =============================================================================
# Helper Functions
# =============================================================================


def _utcnow() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hash of a string.

    Args:
        data: Input string to hash.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# Enums (8 enums)
# =============================================================================


class PrimarySector(str, Enum):
    """Primary sector classification for the organization.

    Maps to SBTi SDA sectors, IEA NZE chapters, and sector-specific
    intensity metrics. The MIXED value supports multi-sector conglomerates.
    """

    POWER = "POWER"
    STEEL = "STEEL"
    CEMENT = "CEMENT"
    ALUMINUM = "ALUMINUM"
    CHEMICALS = "CHEMICALS"
    AVIATION = "AVIATION"
    SHIPPING = "SHIPPING"
    ROAD_TRANSPORT = "ROAD_TRANSPORT"
    RAIL = "RAIL"
    BUILDINGS_RESIDENTIAL = "BUILDINGS_RESIDENTIAL"
    BUILDINGS_COMMERCIAL = "BUILDINGS_COMMERCIAL"
    AGRICULTURE = "AGRICULTURE"
    FOOD_BEVERAGE = "FOOD_BEVERAGE"
    PULP_PAPER = "PULP_PAPER"
    OIL_GAS = "OIL_GAS"
    MIXED = "MIXED"


class SBTiPathway(str, Enum):
    """SBTi temperature alignment pathway level."""

    CELSIUS_1_5 = "CELSIUS_1_5"
    WELL_BELOW_2 = "WELL_BELOW_2"
    CELSIUS_2 = "CELSIUS_2"


class IEAScenario(str, Enum):
    """IEA World Energy Outlook scenario selection."""

    NZE = "NZE"
    WB2C = "WB2C"
    CELSIUS_2 = "2C"
    APS = "APS"
    STEPS = "STEPS"


class IPCCOvershoot(str, Enum):
    """IPCC AR6 pathway overshoot classification."""

    NO = "no"
    LIMITED = "limited"
    HIGH = "high"


class ConvergenceModel(str, Enum):
    """Intensity convergence mathematical model.

    Determines the shape of the intensity reduction curve from
    baseline year to target year.
    """

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    S_CURVE = "s_curve"
    STEPPED = "stepped"


class CapExPhasing(str, Enum):
    """Capital expenditure phasing strategy for technology transitions."""

    FRONT_LOADED = "front_loaded"
    LINEAR = "linear"
    BACK_LOADED = "back_loaded"


class IntensityBoundary(str, Enum):
    """Emission scope boundary for intensity metric calculation."""

    SCOPE_1 = "scope_1"
    SCOPE_1_2 = "scope_1_2"
    SCOPE_1_2_3 = "scope_1_2_3"


class AssuranceLevel(str, Enum):
    """External assurance engagement level for reported data."""

    NONE = "none"
    LIMITED = "limited"
    REASONABLE = "reasonable"


class PeerSelection(str, Enum):
    """Peer group selection method for benchmarking."""

    AUTO = "auto"
    MANUAL = "manual"


class ReportingFrequency(str, Enum):
    """Reporting and monitoring frequency."""

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"


# =============================================================================
# Pydantic Sub-Config Models (8 models)
# =============================================================================


class SBTiSDAConfig(BaseModel):
    """Configuration for SBTi Sectoral Decarbonization Approach.

    Defines SDA-specific parameters including sector categories, pathway
    ambition level, target years, and coverage thresholds aligned with
    SBTi Corporate Standard v2.0.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    sbti_sda_enabled: bool = Field(
        True,
        description="Enable SBTi SDA pathway analysis",
    )
    sbti_sector_category: str = Field(
        "STEEL",
        description="SBTi SDA sector category (one of 12 SDA sectors)",
    )
    sbti_pathway: SBTiPathway = Field(
        SBTiPathway.CELSIUS_1_5,
        description="SBTi temperature alignment pathway",
    )
    sbti_target_year: int = Field(
        DEFAULT_TARGET_YEAR_NEAR,
        ge=2025,
        le=2035,
        description="Near-term SBTi target year (5-10 years from submission)",
    )
    sbti_net_zero_year: int = Field(
        DEFAULT_TARGET_YEAR_LONG,
        ge=2040,
        le=2055,
        description="Long-term net-zero target year (no later than 2050 per SBTi)",
    )
    coverage_scope1_pct: float = Field(
        95.0,
        ge=0.0,
        le=100.0,
        description="Percentage of Scope 1 emissions covered by SBTi target",
    )
    coverage_scope2_pct: float = Field(
        95.0,
        ge=0.0,
        le=100.0,
        description="Percentage of Scope 2 emissions covered by SBTi target",
    )
    coverage_scope3_pct: float = Field(
        67.0,
        ge=0.0,
        le=100.0,
        description="Percentage of Scope 3 emissions covered by near-term target",
    )
    sbti_submission_planned: bool = Field(
        True,
        description="Whether SBTi target submission is planned",
    )
    flag_pathway_enabled: bool = Field(
        False,
        description="Enable FLAG pathway for land-use related emissions (agriculture)",
    )

    @field_validator("sbti_sector_category")
    @classmethod
    def validate_sda_sector(cls, v: str) -> str:
        """Validate SDA sector category is recognized."""
        # Map common aliases
        aliases = {
            "STEEL": "STEEL", "IRON_STEEL": "STEEL",
            "CEMENT": "CEMENT", "CLINKER": "CEMENT",
            "ALUMINIUM": "ALUMINIUM", "ALUMINUM": "ALUMINIUM",
            "POWER": "POWER", "ELECTRICITY": "POWER",
            "PULP_PAPER": "PULP_PAPER", "PAPER": "PULP_PAPER",
            "CHEMICALS": "CHEMICALS", "CHEMICAL": "CHEMICALS",
            "TRANSPORT_ROAD": "TRANSPORT_ROAD", "ROAD": "TRANSPORT_ROAD",
            "TRANSPORT_RAIL": "TRANSPORT_RAIL", "RAIL": "TRANSPORT_RAIL",
            "AVIATION": "AVIATION",
            "SHIPPING": "SHIPPING",
            "BUILDINGS_RESIDENTIAL": "BUILDINGS_RESIDENTIAL",
            "BUILDINGS_COMMERCIAL": "BUILDINGS_COMMERCIAL",
        }
        normalized = aliases.get(v.upper(), v.upper())
        if normalized not in SDA_SECTORS:
            logger.warning(
                "SDA sector '%s' is not in the 12 recognized SBTi SDA sectors. "
                "Valid sectors: %s", v, sorted(SDA_SECTORS.keys())
            )
        return normalized

    @model_validator(mode="after")
    def validate_target_years(self) -> "SBTiSDAConfig":
        """Ensure net-zero year is after near-term target year."""
        if self.sbti_net_zero_year <= self.sbti_target_year:
            raise ValueError(
                f"sbti_net_zero_year ({self.sbti_net_zero_year}) must be after "
                f"sbti_target_year ({self.sbti_target_year})"
            )
        return self


class IEANZEConfig(BaseModel):
    """Configuration for IEA Net Zero Emissions 2050 scenario alignment.

    Defines IEA scenario selection, regional parameters, and milestone
    tracking for sector-specific pathway alignment.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    iea_nze_enabled: bool = Field(
        True,
        description="Enable IEA NZE 2050 scenario alignment",
    )
    iea_scenario: IEAScenario = Field(
        IEAScenario.NZE,
        description="Primary IEA scenario for pathway alignment",
    )
    iea_region: str = Field(
        "Global",
        description="IEA region for sector pathway data (Global, EU, US, China, India, etc.)",
    )
    iea_milestone_tracking: bool = Field(
        True,
        description="Track IEA sector-specific milestones (400+ milestones)",
    )
    iea_technology_outlook: bool = Field(
        True,
        description="Integrate IEA Energy Technology Perspectives data",
    )

    @field_validator("iea_region")
    @classmethod
    def validate_region(cls, v: str) -> str:
        """Validate IEA region is recognized."""
        valid_regions = {
            "Global", "EU", "US", "China", "India", "Japan",
            "Korea", "Southeast_Asia", "Africa", "Middle_East",
            "Latin_America", "OECD", "Non_OECD",
        }
        if v not in valid_regions:
            logger.warning(
                "IEA region '%s' may not have complete sector data. "
                "Recommended regions: %s", v, sorted(valid_regions)
            )
        return v


class IPCCConfig(BaseModel):
    """Configuration for IPCC AR6 pathway alignment.

    Defines IPCC illustrative pathway selection and overshoot
    tolerance for carbon budget calculations.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    ipcc_pathway: str = Field(
        "C1",
        description="IPCC AR6 illustrative pathway category (C1-C8)",
    )
    ipcc_overshoot: IPCCOvershoot = Field(
        IPCCOvershoot.NO,
        description="IPCC pathway overshoot classification",
    )
    ipcc_gwp_metric: str = Field(
        "GWP100",
        description="Global Warming Potential metric (GWP100 or GWP20)",
    )
    ipcc_carbon_budget_gt: Optional[float] = Field(
        None,
        ge=0.0,
        description="Remaining global carbon budget in GtCO2 from IPCC AR6",
    )

    @field_validator("ipcc_pathway")
    @classmethod
    def validate_ipcc_pathway(cls, v: str) -> str:
        """Validate IPCC pathway is C1-C8."""
        if v not in IPCC_PATHWAYS:
            raise ValueError(
                f"Invalid IPCC pathway: {v}. Must be one of: {sorted(IPCC_PATHWAYS.keys())}"
            )
        return v


class IntensityConfig(BaseModel):
    """Configuration for sector-specific intensity metric calculation.

    Defines the baseline intensity value, metric unit, boundary scope,
    and base year for intensity convergence tracking.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    baseline_intensity: float = Field(
        0.0,
        ge=0.0,
        description="Base year sector intensity value (in sector-specific units)",
    )
    baseline_year: int = Field(
        DEFAULT_BASE_YEAR,
        ge=2010,
        le=2030,
        description="Base year for intensity baseline measurement",
    )
    intensity_metric: str = Field(
        "tCO2e/tonne product",
        description="Sector-specific intensity metric unit",
    )
    intensity_boundary: IntensityBoundary = Field(
        IntensityBoundary.SCOPE_1_2,
        description="Emission scope boundary for intensity calculation",
    )
    production_unit: str = Field(
        "tonne",
        description="Production unit denominator for intensity metric",
    )
    production_forecast_enabled: bool = Field(
        True,
        description="Enable production volume forecasting for absolute emissions",
    )
    production_growth_rate_pct: float = Field(
        0.0,
        ge=-20.0,
        le=20.0,
        description="Annual production growth rate (%) for pathway modeling",
    )


class ConvergenceConfig(BaseModel):
    """Configuration for intensity convergence modeling.

    Defines the mathematical convergence model, annual improvement rate,
    and catch-up period for sector pathway alignment.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    convergence_model: ConvergenceModel = Field(
        ConvergenceModel.LINEAR,
        description="Mathematical model for intensity convergence curve",
    )
    convergence_rate: float = Field(
        DEFAULT_CONVERGENCE_RATE,
        ge=0.01,
        le=0.30,
        description="Annual improvement rate (as fraction, e.g. 0.05 = 5% per year)",
    )
    catch_up_period: int = Field(
        DEFAULT_CATCH_UP_PERIOD,
        ge=3,
        le=30,
        description="Years to reach sector leader intensity level",
    )
    inflection_year: Optional[int] = Field(
        None,
        ge=2025,
        le=2050,
        description="S-curve inflection point year (for s_curve model only)",
    )
    step_years: Optional[List[int]] = Field(
        None,
        description="Step years for stepped convergence model (e.g., [2030, 2035, 2040, 2045])",
    )
    step_reductions_pct: Optional[List[float]] = Field(
        None,
        description="Step reduction percentages aligned with step_years",
    )

    @model_validator(mode="after")
    def validate_s_curve_inflection(self) -> "ConvergenceConfig":
        """Ensure S-curve model has an inflection year."""
        if self.convergence_model == ConvergenceModel.S_CURVE and self.inflection_year is None:
            logger.warning(
                "S-curve convergence model selected but no inflection_year set. "
                "Defaulting to 2035."
            )
            object.__setattr__(self, "inflection_year", 2035)
        return self

    @model_validator(mode="after")
    def validate_stepped_has_steps(self) -> "ConvergenceConfig":
        """Ensure stepped model has step years and reductions."""
        if self.convergence_model == ConvergenceModel.STEPPED:
            if not self.step_years or not self.step_reductions_pct:
                logger.warning(
                    "Stepped convergence model selected but step_years or "
                    "step_reductions_pct not set. Defaulting to 5-year steps."
                )
                if not self.step_years:
                    object.__setattr__(self, "step_years", [2030, 2035, 2040, 2045, 2050])
                if not self.step_reductions_pct:
                    object.__setattr__(self, "step_reductions_pct", [20.0, 40.0, 60.0, 80.0, 95.0])
            elif len(self.step_years) != len(self.step_reductions_pct):
                raise ValueError(
                    f"step_years ({len(self.step_years)}) and step_reductions_pct "
                    f"({len(self.step_reductions_pct)}) must have the same length"
                )
        return self


class TechnologyRoadmapConfig(BaseModel):
    """Configuration for sector technology transition roadmap planning.

    Defines TRL thresholds, CapEx phasing, critical materials risk
    assessment, and IEA milestone integration.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    technology_roadmap_enabled: bool = Field(
        True,
        description="Enable technology transition roadmap engine",
    )
    trl_threshold: int = Field(
        DEFAULT_TRL_THRESHOLD,
        ge=1,
        le=9,
        description="Minimum Technology Readiness Level for deployment planning",
    )
    capex_phasing: CapExPhasing = Field(
        CapExPhasing.LINEAR,
        description="CapEx phasing strategy for technology investments",
    )
    capex_budget_eur: Optional[float] = Field(
        None,
        ge=0.0,
        description="Total CapEx budget (EUR) for technology transitions",
    )
    planning_horizon_years: int = Field(
        15,
        ge=5,
        le=30,
        description="Technology planning horizon in years",
    )
    critical_materials_risk: bool = Field(
        True,
        description="Enable critical materials supply risk assessment",
    )
    dependency_analysis: bool = Field(
        True,
        description="Enable technology interdependency analysis",
    )
    iea_milestone_mapping: bool = Field(
        True,
        description="Map technology transitions to IEA NZE milestones",
    )
    max_parallel_transitions: int = Field(
        5,
        ge=1,
        le=20,
        description="Maximum parallel technology transitions",
    )
    s_curve_adoption: bool = Field(
        True,
        description="Model technology adoption using S-curve diffusion",
    )


class MACCConfig(BaseModel):
    """Configuration for Marginal Abatement Cost Curve (MACC) waterfall.

    Defines carbon price ranges, abatement potential targets, lever
    interaction modeling, and implementation sequencing.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    macc_enabled: bool = Field(
        True,
        description="Enable MACC waterfall analysis engine",
    )
    carbon_price_floor: float = Field(
        DEFAULT_CARBON_PRICE_FLOOR,
        ge=0.0,
        le=500.0,
        description="Carbon price floor (USD/tCO2e) for MACC valuation",
    )
    carbon_price_ceiling: float = Field(
        DEFAULT_CARBON_PRICE_CEILING,
        ge=10.0,
        le=1000.0,
        description="Carbon price ceiling (USD/tCO2e) for sensitivity analysis",
    )
    abatement_potential_target: float = Field(
        DEFAULT_ABATEMENT_TARGET,
        ge=0.10,
        le=1.00,
        description="Target abatement potential as fraction of baseline (e.g., 0.80 = 80%)",
    )
    lever_interaction_modeling: bool = Field(
        True,
        description="Model interactions and dependencies between abatement levers",
    )
    implementation_sequencing: bool = Field(
        True,
        description="Optimize lever implementation sequence by cost-effectiveness",
    )
    discount_rate_pct: float = Field(
        8.0,
        ge=0.0,
        le=30.0,
        description="Discount rate (%) for NPV calculation of abatement actions",
    )
    max_levers: int = Field(
        50,
        ge=5,
        le=200,
        description="Maximum number of abatement levers to evaluate",
    )
    include_negative_cost: bool = Field(
        True,
        description="Include negative-cost (cost-saving) abatement levers",
    )

    @model_validator(mode="after")
    def validate_price_range(self) -> "MACCConfig":
        """Ensure carbon price floor is below ceiling."""
        if self.carbon_price_floor >= self.carbon_price_ceiling:
            raise ValueError(
                f"carbon_price_floor ({self.carbon_price_floor}) must be below "
                f"carbon_price_ceiling ({self.carbon_price_ceiling})"
            )
        return self


class BenchmarkConfig(BaseModel):
    """Configuration for sector benchmarking analysis.

    Defines peer group selection, leader percentile, IEA benchmark
    integration, and regulatory benchmark sources.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    benchmark_peer_enabled: bool = Field(
        True,
        description="Enable peer group benchmarking",
    )
    benchmark_leader_enabled: bool = Field(
        True,
        description="Enable sector leader benchmarking",
    )
    benchmark_iea_enabled: bool = Field(
        True,
        description="Enable IEA pathway benchmarking",
    )
    peer_selection: PeerSelection = Field(
        PeerSelection.AUTO,
        description="Peer group selection method",
    )
    leader_percentile: int = Field(
        DEFAULT_LEADER_PERCENTILE,
        ge=1,
        le=25,
        description="Top percentile for sector leader definition",
    )
    peer_group_size: int = Field(
        20,
        ge=5,
        le=100,
        description="Target peer group size for benchmarking",
    )
    sbti_validated_only: bool = Field(
        False,
        description="Restrict benchmarking to SBTi-validated companies only",
    )
    regulatory_benchmarks: List[str] = Field(
        default_factory=lambda: ["EU_ETS", "SBTi_SDA"],
        description="Regulatory benchmark sources to include",
    )
    benchmark_dimensions: List[str] = Field(
        default_factory=lambda: [
            "intensity_rank",
            "reduction_rate",
            "technology_mix",
            "target_ambition",
            "pathway_gap",
        ],
        description="Benchmarking dimensions for multi-dimensional comparison",
    )


class ScenarioAnalysisConfig(BaseModel):
    """Configuration for multi-scenario pathway comparison.

    Defines which scenarios to model, sensitivity analysis parameters,
    and risk-return assessment settings.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    scenarios_enabled: List[str] = Field(
        default_factory=lambda: ["NZE", "WB2C", "2C"],
        description="IEA scenarios to model for pathway comparison",
    )
    scenario_sensitivity: bool = Field(
        True,
        description="Enable scenario sensitivity analysis",
    )
    carbon_price_sensitivity: bool = Field(
        True,
        description="Enable carbon price sensitivity across scenarios",
    )
    technology_sensitivity: bool = Field(
        True,
        description="Enable technology adoption rate sensitivity",
    )
    monte_carlo_runs: int = Field(
        1000,
        ge=100,
        le=10000,
        description="Monte Carlo simulation runs for uncertainty quantification",
    )
    confidence_interval_pct: float = Field(
        90.0,
        ge=80.0,
        le=99.0,
        description="Confidence interval for scenario projections",
    )
    risk_return_analysis: bool = Field(
        True,
        description="Enable transition risk-return analysis across scenarios",
    )
    optimal_pathway_recommendation: bool = Field(
        True,
        description="Generate optimal pathway recommendation based on constraints",
    )

    @field_validator("scenarios_enabled")
    @classmethod
    def validate_scenarios(cls, v: List[str]) -> List[str]:
        """Validate scenario identifiers."""
        valid = {"NZE", "WB2C", "2C", "APS", "STEPS"}
        invalid = [s for s in v if s not in valid]
        if invalid:
            raise ValueError(
                f"Invalid scenarios: {invalid}. Valid options: {sorted(valid)}"
            )
        if len(v) < 2:
            logger.warning(
                "Fewer than 2 scenarios selected. Multi-scenario comparison "
                "requires at least 2 scenarios for meaningful analysis."
            )
        return v


class ReportingConfig(BaseModel):
    """Configuration for multi-framework reporting.

    Defines output frameworks, reporting frequency, and assurance
    level for sector pathway disclosures.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    reporting_frameworks: List[str] = Field(
        default_factory=lambda: ["SBTi", "CDP", "TCFD", "GRI"],
        description="Reporting frameworks for pathway disclosure mapping",
    )
    reporting_frequency: ReportingFrequency = Field(
        ReportingFrequency.QUARTERLY,
        description="Reporting and progress monitoring frequency",
    )
    assurance_level: AssuranceLevel = Field(
        AssuranceLevel.LIMITED,
        description="External assurance engagement level",
    )
    sha256_provenance: bool = Field(
        True,
        description="Generate SHA-256 provenance hashes for all outputs",
    )
    calculation_trace: bool = Field(
        True,
        description="Generate step-by-step calculation trace for auditability",
    )
    assumption_register: bool = Field(
        True,
        description="Maintain assumption register for all pathway calculations",
    )
    data_lineage_enabled: bool = Field(
        True,
        description="Track full data lineage from source to pathway output",
    )
    retention_years: int = Field(
        7,
        ge=3,
        le=15,
        description="Report and audit trail retention period in years",
    )

    @field_validator("reporting_frameworks")
    @classmethod
    def validate_frameworks(cls, v: List[str]) -> List[str]:
        """Validate reporting framework identifiers."""
        valid = {"SBTi", "CDP", "TCFD", "GRI", "ESRS", "ISO14064", "SASB"}
        invalid = [f for f in v if f not in valid]
        if invalid:
            logger.warning(
                "Unrecognized reporting frameworks: %s. "
                "Recognized: %s", invalid, sorted(valid)
            )
        return v


class PerformanceConfig(BaseModel):
    """Configuration for runtime performance tuning.

    Defines caching, concurrency, and timeout settings for the
    sector pathway calculation pipeline.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    cache_enabled: bool = Field(
        True,
        description="Enable Redis-based caching for pathway calculations",
    )
    cache_ttl_seconds: int = Field(
        3600,
        ge=60,
        le=86400,
        description="Cache time-to-live in seconds",
    )
    max_concurrent_calcs: int = Field(
        4,
        ge=1,
        le=32,
        description="Maximum concurrent pathway calculation threads",
    )
    timeout_seconds: int = Field(
        300,
        ge=30,
        le=3600,
        description="Maximum timeout for a single engine calculation",
    )
    batch_size: int = Field(
        1000,
        ge=100,
        le=10000,
        description="Batch size for bulk data processing",
    )
    memory_limit_mb: int = Field(
        4096,
        ge=512,
        le=32768,
        description="Memory limit in MB for the calculation pipeline",
    )


# =============================================================================
# Main Configuration Model
# =============================================================================


class SectorPathwayConfig(BaseModel):
    """Main configuration model for PACK-028 Sector Pathway Pack.

    This is the root Pydantic v2 configuration model containing all parameters
    for sector-specific decarbonization pathway analysis. The primary_sector
    field drives SDA sector selection, intensity metric, convergence model
    recommendation, and technology roadmap priorities.

    The model supports 15+ sectors with SBTi SDA pathway alignment for 12
    eligible sectors, IEA NZE 2050 scenario comparison, IPCC AR6 pathway
    integration, and 4 convergence models for intensity trajectory design.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        json_schema_extra={
            "title": "PACK-028 Sector Pathway Configuration",
            "description": "Configuration for sector-specific decarbonization pathway analysis",
        },
    )

    # --- Organization Identity ---
    organization_name: str = Field(
        "",
        description="Legal entity name of the organization",
    )
    primary_sector: PrimarySector = Field(
        PrimarySector.STEEL,
        description="Primary sector classification (drives pathway and engine configuration)",
    )
    secondary_sectors: List[str] = Field(
        default_factory=list,
        description="Secondary sector codes for multi-sector organizations",
    )
    nace_code: Optional[str] = Field(
        None,
        description="NACE Rev.2 industry classification code (e.g., C24.10 for steel)",
    )
    gics_code: Optional[str] = Field(
        None,
        description="GICS sector classification code for benchmarking",
    )
    isic_code: Optional[str] = Field(
        None,
        description="ISIC Rev.4 classification code for IEA sector mapping",
    )
    region: str = Field(
        "EU",
        description="Primary operating region (ISO 3166 or continent code)",
    )
    country: str = Field(
        "DE",
        description="Headquarters country (ISO 3166-1 alpha-2)",
    )

    # --- Temporal Settings ---
    reporting_year: int = Field(
        2025,
        ge=2020,
        le=2035,
        description="Current reporting year for pathway progress tracking",
    )
    base_year: int = Field(
        DEFAULT_BASE_YEAR,
        ge=2010,
        le=2030,
        description="Base year for sector intensity baseline measurement",
    )
    pack_version: str = Field(
        "1.0.0",
        description="Pack configuration version",
    )

    # --- Sub-Configurations ---
    sbti_sda: SBTiSDAConfig = Field(
        default_factory=SBTiSDAConfig,
        description="SBTi Sectoral Decarbonization Approach configuration",
    )
    iea_nze: IEANZEConfig = Field(
        default_factory=IEANZEConfig,
        description="IEA NZE 2050 scenario alignment configuration",
    )
    ipcc: IPCCConfig = Field(
        default_factory=IPCCConfig,
        description="IPCC AR6 pathway alignment configuration",
    )
    intensity: IntensityConfig = Field(
        default_factory=IntensityConfig,
        description="Sector intensity metric configuration",
    )
    convergence: ConvergenceConfig = Field(
        default_factory=ConvergenceConfig,
        description="Intensity convergence model configuration",
    )
    technology: TechnologyRoadmapConfig = Field(
        default_factory=TechnologyRoadmapConfig,
        description="Technology transition roadmap configuration",
    )
    macc: MACCConfig = Field(
        default_factory=MACCConfig,
        description="MACC waterfall analysis configuration",
    )
    benchmark: BenchmarkConfig = Field(
        default_factory=BenchmarkConfig,
        description="Sector benchmarking configuration",
    )
    scenarios: ScenarioAnalysisConfig = Field(
        default_factory=ScenarioAnalysisConfig,
        description="Multi-scenario pathway comparison configuration",
    )
    reporting: ReportingConfig = Field(
        default_factory=ReportingConfig,
        description="Multi-framework reporting configuration",
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Runtime performance tuning configuration",
    )

    # --- Cross-Cutting Validators ---

    @model_validator(mode="after")
    def validate_base_year_before_reporting(self) -> "SectorPathwayConfig":
        """Ensure base year is not after reporting year."""
        if self.base_year > self.reporting_year:
            raise ValueError(
                f"base_year ({self.base_year}) must not be after "
                f"reporting_year ({self.reporting_year})"
            )
        return self

    @model_validator(mode="after")
    def validate_sda_sector_alignment(self) -> "SectorPathwayConfig":
        """Warn if SDA is enabled for a non-SDA-eligible sector."""
        if self.sbti_sda.sbti_sda_enabled:
            sector_info = SECTOR_INFO.get(self.primary_sector.value, {})
            if not sector_info.get("sda_eligible", False):
                logger.warning(
                    "SDA pathway enabled for %s sector which is not SDA-eligible. "
                    "Consider disabling sbti_sda.sbti_sda_enabled or using ACA approach.",
                    self.primary_sector.value,
                )
        return self

    @model_validator(mode="after")
    def validate_agriculture_flag(self) -> "SectorPathwayConfig":
        """Auto-enable FLAG pathway for agriculture sector."""
        if self.primary_sector == PrimarySector.AGRICULTURE:
            if not self.sbti_sda.flag_pathway_enabled:
                logger.warning(
                    "FLAG pathway is recommended for agriculture sector. "
                    "Auto-enabling flag_pathway_enabled."
                )
                object.__setattr__(self.sbti_sda, "flag_pathway_enabled", True)
        return self

    @model_validator(mode="after")
    def validate_intensity_metric_matches_sector(self) -> "SectorPathwayConfig":
        """Warn if intensity metric does not match sector default."""
        expected_metric = ALL_INTENSITY_METRICS.get(self.primary_sector.value)
        if expected_metric and self.intensity.intensity_metric != expected_metric:
            logger.warning(
                "Intensity metric '%s' does not match expected sector metric '%s' "
                "for %s. Verify this is intentional.",
                self.intensity.intensity_metric,
                expected_metric,
                self.primary_sector.value,
            )
        return self

    def get_enabled_engines(self) -> List[str]:
        """Return list of engine identifiers that should be enabled.

        Returns:
            Sorted list of enabled engine identifier strings.
        """
        engines = [
            "sector_classification",
            "intensity_calculator",
            "pathway_generator",
            "convergence_analyzer",
        ]

        if self.technology.technology_roadmap_enabled:
            engines.append("technology_roadmap")

        if self.macc.macc_enabled:
            engines.append("abatement_waterfall")

        if (self.benchmark.benchmark_peer_enabled
                or self.benchmark.benchmark_leader_enabled
                or self.benchmark.benchmark_iea_enabled):
            engines.append("sector_benchmark")

        if len(self.scenarios.scenarios_enabled) >= 2:
            engines.append("scenario_comparison")

        return sorted(set(engines))

    def get_sector_levers(self) -> List[str]:
        """Return the decarbonization levers for the primary sector.

        Returns:
            List of lever identifiers appropriate for the sector.
        """
        return SECTOR_DECARBONIZATION_LEVERS.get(
            self.primary_sector.value,
            SECTOR_DECARBONIZATION_LEVERS.get("MIXED", []),
        )

    def get_sda_intensity_metric(self) -> str:
        """Return the SDA intensity metric for the primary sector.

        Returns:
            Intensity metric string (e.g., 'tCO2e/tonne crude steel').
        """
        return ALL_INTENSITY_METRICS.get(
            self.primary_sector.value,
            "tCO2e/unit",
        )

    def get_sda_2050_target(self) -> Optional[float]:
        """Return the SDA 2050 convergence target for the primary sector.

        Returns:
            Target intensity value for 2050, or None if not available.
        """
        # Map primary sector to SDA sector key
        sector_map = {
            "ALUMINUM": "ALUMINIUM",
            "ROAD_TRANSPORT": "TRANSPORT_ROAD",
            "RAIL": "TRANSPORT_RAIL",
        }
        sda_key = sector_map.get(self.primary_sector.value, self.primary_sector.value)
        return SDA_2050_TARGETS.get(sda_key)


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """Top-level pack configuration wrapper for PACK-028.

    Handles preset loading, environment variable overrides, and
    configuration merging. Provides SHA-256 config hashing for
    provenance tracking and JSON Schema export for API documentation.

    Example:
        >>> config = PackConfig.from_preset("heavy_industry")
        >>> print(config.pack.primary_sector)
        PrimarySector.STEEL
        >>> config = PackConfig.from_preset("power_generation", overrides={"iea_region": "US"})
        >>> print(config.pack.iea_nze.iea_region)
        US
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    pack: SectorPathwayConfig = Field(
        default_factory=SectorPathwayConfig,
        description="Main Sector Pathway configuration",
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
        "PACK-028-sector-pathway",
        description="Pack identifier",
    )

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "PackConfig":
        """Load configuration from a named preset.

        Loads the sector-specific YAML preset file, applies environment
        variable overrides (SECTOR_PATHWAY_* prefix), then applies any
        explicit runtime overrides.

        Args:
            preset_name: Name of the preset (power_generation, heavy_industry,
                chemicals, transport, buildings, mixed_sectors).
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
        env_overrides = _get_env_overrides("SECTOR_PATHWAY_")
        if env_overrides:
            preset_data = _merge_config(preset_data, env_overrides)

        # Apply explicit overrides
        if overrides:
            preset_data = _merge_config(preset_data, overrides)

        pack_config = SectorPathwayConfig(**preset_data)
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

        pack_config = SectorPathwayConfig(**config_data)
        return cls(pack=pack_config)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PackConfig":
        """Load configuration from a dictionary.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            PackConfig instance.
        """
        pack_config = SectorPathwayConfig(**config_dict)
        return cls(pack=pack_config)

    def get_config_hash(self) -> str:
        """Generate SHA-256 hash of the current configuration for provenance.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        config_json = self.model_dump_json(indent=None)
        return _compute_hash(config_json)

    def validate_config(self) -> List[str]:
        """Cross-field validation returning warnings.

        Performs advisory validation beyond Pydantic's built-in validation.
        Returns warnings, not hard errors.

        Returns:
            List of warning messages (empty if fully valid).
        """
        return validate_config(self.pack)

    def export_json_schema(self) -> Dict[str, Any]:
        """Export the configuration JSON Schema for API documentation.

        Returns:
            JSON Schema dictionary for the SectorPathwayConfig model.
        """
        return SectorPathwayConfig.model_json_schema()


# =============================================================================
# Utility Functions
# =============================================================================


def load_config(yaml_path: Union[str, Path]) -> PackConfig:
    """Load configuration from a YAML file.

    Convenience wrapper around PackConfig.from_yaml().

    Args:
        yaml_path: Path to YAML configuration file.

    Returns:
        PackConfig instance.
    """
    return PackConfig.from_yaml(yaml_path)


def load_preset(
    preset_name: str,
    overrides: Optional[Dict[str, Any]] = None,
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


def get_sector_defaults(
    sector: Union[str, PrimarySector],
) -> SectorPathwayConfig:
    """Get default configuration for a given sector.

    Creates a SectorPathwayConfig with sector-appropriate defaults
    for SDA category, intensity metric, convergence model, and
    technology roadmap settings.

    Args:
        sector: Primary sector enum or string value.

    Returns:
        SectorPathwayConfig with sector defaults applied.
    """
    if isinstance(sector, str):
        sector = PrimarySector(sector)

    sector_info = SECTOR_INFO.get(sector.value, {})
    sda_eligible = sector_info.get("sda_eligible", False)
    recommended_convergence = sector_info.get("recommended_convergence", "linear")
    intensity_metric = ALL_INTENSITY_METRICS.get(sector.value, "tCO2e/unit")

    # Map PrimarySector to SDA sector category
    sda_sector_map = {
        "ALUMINUM": "ALUMINIUM",
        "ROAD_TRANSPORT": "TRANSPORT_ROAD",
        "RAIL": "TRANSPORT_RAIL",
    }
    sda_category = sda_sector_map.get(sector.value, sector.value)

    return SectorPathwayConfig(
        primary_sector=sector,
        sbti_sda=SBTiSDAConfig(
            sbti_sda_enabled=sda_eligible,
            sbti_sector_category=sda_category if sda_eligible else "STEEL",
            flag_pathway_enabled=(sector == PrimarySector.AGRICULTURE),
        ),
        intensity=IntensityConfig(
            intensity_metric=intensity_metric,
        ),
        convergence=ConvergenceConfig(
            convergence_model=ConvergenceModel(recommended_convergence),
        ),
    )


def _merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base configuration dictionary.
        override: Override dictionary (values take precedence).

    Returns:
        Merged dictionary.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
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
        SECTOR_PATHWAY_REPORTING_YEAR=2026
        SECTOR_PATHWAY_SBTI_SDA__SBTI_PATHWAY=CELSIUS_1_5
        SECTOR_PATHWAY_CONVERGENCE__CONVERGENCE_MODEL=exponential

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


def validate_config(config: SectorPathwayConfig) -> List[str]:
    """Validate a sector pathway configuration and return any warnings.

    Performs cross-field validation beyond what Pydantic validators cover.
    Returns advisory warnings, not hard errors.

    Args:
        config: SectorPathwayConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check organization name is set
    if not config.organization_name:
        warnings.append(
            "Organization name is empty. Set organization_name for meaningful reports."
        )

    # Check base year is reasonable
    if config.base_year > config.reporting_year:
        warnings.append(
            f"Base year ({config.base_year}) is after reporting year "
            f"({config.reporting_year}). Base year should precede reporting year."
        )

    # Check SDA eligibility
    sector_info = SECTOR_INFO.get(config.primary_sector.value, {})
    if config.sbti_sda.sbti_sda_enabled and not sector_info.get("sda_eligible", False):
        warnings.append(
            f"SDA pathway enabled for {config.primary_sector.value} which is not "
            f"SDA-eligible. Consider ACA approach or disable sbti_sda.sbti_sda_enabled."
        )

    # Check intensity baseline is set
    if config.intensity.baseline_intensity <= 0.0:
        warnings.append(
            "Baseline intensity is 0 or not set. Set intensity.baseline_intensity "
            "for convergence analysis."
        )

    # Check convergence model matches sector recommendation
    recommended = sector_info.get("recommended_convergence", "linear")
    if config.convergence.convergence_model.value != recommended:
        warnings.append(
            f"Convergence model '{config.convergence.convergence_model.value}' differs from "
            f"recommended '{recommended}' for {config.primary_sector.value} sector. "
            f"Verify this is intentional."
        )

    # Check net-zero target year
    if config.sbti_sda.sbti_net_zero_year > 2050:
        warnings.append(
            "SBTi Net-Zero Standard requires net-zero by no later than 2050. "
            f"Current net-zero year: {config.sbti_sda.sbti_net_zero_year}."
        )

    # Check SBTi coverage thresholds
    if config.sbti_sda.sbti_submission_planned:
        if config.sbti_sda.coverage_scope1_pct < 95.0:
            warnings.append(
                f"SBTi requires at least 95% Scope 1 coverage. "
                f"Current: {config.sbti_sda.coverage_scope1_pct}%."
            )
        if config.sbti_sda.coverage_scope2_pct < 95.0:
            warnings.append(
                f"SBTi requires at least 95% Scope 2 coverage. "
                f"Current: {config.sbti_sda.coverage_scope2_pct}%."
            )
        if config.sbti_sda.coverage_scope3_pct < 67.0:
            warnings.append(
                f"SBTi requires at least 67% Scope 3 coverage for near-term. "
                f"Current: {config.sbti_sda.coverage_scope3_pct}%."
            )

    # Check MACC carbon price range
    if config.macc.macc_enabled:
        if config.macc.carbon_price_floor < 30.0:
            warnings.append(
                f"Carbon price floor ({config.macc.carbon_price_floor} USD/tCO2e) "
                f"is below EU ETS current levels. Consider raising."
            )

    # Check technology TRL threshold
    if config.technology.technology_roadmap_enabled:
        if config.technology.trl_threshold < 5:
            warnings.append(
                f"TRL threshold ({config.technology.trl_threshold}) is very low. "
                f"Technologies below TRL 5 may not be commercially viable within "
                f"the planning horizon."
            )

    # Check multi-sector consistency
    if config.primary_sector == PrimarySector.MIXED and not config.secondary_sectors:
        warnings.append(
            "MIXED sector selected but no secondary_sectors specified. "
            "Add secondary sector codes for sector-weighted composite analysis."
        )

    # Check agriculture FLAG pathway
    if config.primary_sector == PrimarySector.AGRICULTURE:
        if not config.sbti_sda.flag_pathway_enabled:
            warnings.append(
                "FLAG pathway is strongly recommended for agriculture sector. "
                "Enable sbti_sda.flag_pathway_enabled for SBTi compliance."
            )

    # Check scenario count
    if len(config.scenarios.scenarios_enabled) < 2:
        warnings.append(
            "Fewer than 2 scenarios selected. Multi-scenario pathway comparison "
            "requires at least 2 scenarios for meaningful analysis."
        )

    # Check reporting frameworks
    if not config.reporting.reporting_frameworks:
        warnings.append(
            "No reporting frameworks configured. Add at least one framework "
            "(SBTi, CDP, TCFD, GRI) for disclosure readiness."
        )

    return warnings


def get_sector_info(sector: Union[str, PrimarySector]) -> Dict[str, Any]:
    """Get detailed information about a sector.

    Args:
        sector: Primary sector enum or string value.

    Returns:
        Dictionary with name, SDA eligibility, recommended convergence,
        IEA chapter, and typical scope split.
    """
    key = sector.value if isinstance(sector, PrimarySector) else sector
    return SECTOR_INFO.get(key, {
        "name": key,
        "sda_eligible": False,
        "recommended_convergence": "linear",
        "iea_chapter": "N/A",
        "typical_scope_split": "Varies",
    })


def get_sda_intensity_metric(sector: str) -> str:
    """Get the SDA intensity metric for a given sector.

    Args:
        sector: SDA sector code (e.g., POWER, CEMENT, STEEL).

    Returns:
        Intensity metric string (e.g., tCO2e/tonne crude steel).
    """
    return ALL_INTENSITY_METRICS.get(sector, "tCO2e/unit")


def get_sda_2050_target(sector: str) -> Optional[float]:
    """Get the SDA 2050 convergence target for a sector.

    Args:
        sector: SDA sector code.

    Returns:
        Target intensity value, or None if not available.
    """
    return SDA_2050_TARGETS.get(sector)


def get_iea_scenario_info(scenario: Union[str, IEAScenario]) -> Dict[str, Any]:
    """Get IEA scenario parameters.

    Args:
        scenario: IEA scenario enum or string value.

    Returns:
        Dictionary with name, temperature, probability, and description.
    """
    key = scenario.value if isinstance(scenario, IEAScenario) else scenario
    return IEA_SCENARIOS.get(key, IEA_SCENARIOS["NZE"])


def get_ipcc_pathway_info(pathway: str) -> Dict[str, Any]:
    """Get IPCC AR6 pathway parameters.

    Args:
        pathway: IPCC pathway category (C1-C8).

    Returns:
        Dictionary with temperature, overshoot, and description.
    """
    return IPCC_PATHWAYS.get(pathway, IPCC_PATHWAYS["C1"])


def get_sector_levers(sector: Union[str, PrimarySector]) -> List[str]:
    """Get decarbonization levers for a sector.

    Args:
        sector: Primary sector enum or string value.

    Returns:
        List of lever identifiers.
    """
    key = sector.value if isinstance(sector, PrimarySector) else sector
    return SECTOR_DECARBONIZATION_LEVERS.get(key, [])


def get_sbti_reduction_rate(ambition: Union[str, SBTiPathway]) -> Dict[str, float]:
    """Get SBTi minimum annual reduction rates for an ambition level.

    Args:
        ambition: SBTi pathway enum or string value.

    Returns:
        Dictionary with scope_1_2_linear_annual, scope_3_linear_annual,
        and long_term_reduction_pct.
    """
    key = ambition.value if isinstance(ambition, SBTiPathway) else ambition
    return SBTI_REDUCTION_RATES.get(key, SBTI_REDUCTION_RATES["CELSIUS_1_5"])


def get_gwp100(gas: str) -> int:
    """Get IPCC AR6 GWP100 value for a greenhouse gas.

    Args:
        gas: Greenhouse gas identifier (CO2, CH4, N2O, etc.).

    Returns:
        GWP100 value (dimensionless, relative to CO2).
    """
    return IPCC_AR6_GWP100.get(gas.upper(), 0)


def list_available_presets() -> Dict[str, str]:
    """List all available configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return SUPPORTED_PRESETS.copy()


def list_sda_sectors() -> Dict[str, str]:
    """List all supported SBTi SDA sectors.

    Returns:
        Dictionary mapping sector codes to display names.
    """
    return SDA_SECTORS.copy()


def list_iea_scenarios() -> Dict[str, str]:
    """List all supported IEA scenarios.

    Returns:
        Dictionary mapping scenario codes to names.
    """
    return {k: v["name"] for k, v in IEA_SCENARIOS.items()}


def list_primary_sectors() -> Dict[str, str]:
    """List all supported primary sectors.

    Returns:
        Dictionary mapping sector codes to display names.
    """
    return {k: v["name"] for k, v in SECTOR_INFO.items()}
