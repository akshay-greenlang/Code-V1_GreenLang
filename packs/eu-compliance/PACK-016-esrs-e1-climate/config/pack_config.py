"""
PACK-016 ESRS E1 Climate Change Pack - Configuration Manager

This module implements the E1ClimateConfig and PackConfig classes that load,
merge, and validate all configuration for the ESRS E1 Climate Change Pack.
It provides comprehensive Pydantic v2 models for every aspect of the E1
disclosure process: GHG inventory compilation (Scope 1, 2, 3), energy
consumption and mix, GHG reduction target tracking, transition plan,
climate risk and opportunity assessment, carbon credit management, internal
carbon pricing, and full E1 report generation.

ESRS E1 Disclosure Requirements:
    - E1-1: Transition plan for climate change mitigation (para 14-16)
    - E1-2: Policies related to climate change (para 22-24)
    - E1-3: Actions and resources related to climate change (para 26-28)
    - E1-4: Targets related to climate change (para 30-33)
    - E1-5: Energy consumption and mix (para 35-39)
    - E1-6: Gross Scopes 1, 2, 3 and total GHG emissions (para 44-55)
    - E1-7: GHG removals and carbon credits (para 56-58)
    - E1-8: Internal carbon pricing (para 59-61)
    - E1-9: Anticipated financial effects from climate risks (para 64-68)

GHG Protocol Methodology:
    - Scope 1: Direct emissions from owned/controlled sources
    - Scope 2 Location-Based: Grid-average emission factors
    - Scope 2 Market-Based: Supplier-specific, residual mix, PPA
    - Scope 3: All 15 upstream and downstream categories

Target Alignment:
    - SBTi 1.5C pathway: ~4.2% annual linear reduction
    - SBTi Well-Below 2C: ~2.5% annual linear reduction
    - SBTi Net-Zero: 90%+ reduction by 2050, neutralize residual

IPCC AR6 GWP-100 Values (key gases):
    - CO2: 1
    - CH4 (fossil): 29.8 (with climate-carbon feedback)
    - CH4 (non-fossil): 27.2
    - N2O: 273
    - SF6: 25,200
    - NF3: 17,400
    - HFCs: vary by compound (e.g., HFC-134a = 1,530)
    - PFCs: vary by compound (e.g., CF4 = 7,380)

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (power_generation / manufacturing / transport /
       financial_services / real_estate / multi_sector)
    3. Environment overrides (E1_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - ESRS E1: Climate Change (Delegated Regulation (EU) 2023/2772)
    - GHG Protocol Corporate Standard (2004, revised 2015)
    - GHG Protocol Scope 2 Guidance (2015)
    - GHG Protocol Value Chain (Scope 3) Standard (2011)
    - SBTi Corporate Net-Zero Standard v1.2 (2024)
    - IPCC AR6 GWP values (2021)
    - EU Taxonomy Regulation 2020/852 (climate objectives)
    - Paris Agreement (2015) - 1.5C and well-below-2C pathways

Example:
    >>> config = PackConfig.from_preset("manufacturing")
    >>> print(config.pack.ghg.consolidation_approach)
    OPERATIONAL_CONTROL
    >>> print(config.pack.targets.sbti_commitment_level)
    SBTi_1_5C
    >>> print(config.pack.energy.include_renewables)
    True
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
# Enums - E1 Climate-specific enumeration types (18 enums)
# =============================================================================


class GHGScope(str, Enum):
    """GHG Protocol emission scope classification."""

    SCOPE_1 = "SCOPE_1"                  # Direct emissions
    SCOPE_2_LOCATION = "SCOPE_2_LOCATION"  # Indirect - location-based
    SCOPE_2_MARKET = "SCOPE_2_MARKET"      # Indirect - market-based
    SCOPE_3 = "SCOPE_3"                    # Other indirect (value chain)


class EmissionGas(str, Enum):
    """Seven Kyoto Protocol greenhouse gases per ESRS E1-6 para 48."""

    CO2 = "CO2"        # Carbon dioxide (GWP-100 = 1)
    CH4 = "CH4"        # Methane (GWP-100 = 29.8 fossil, 27.2 biogenic)
    N2O = "N2O"        # Nitrous oxide (GWP-100 = 273)
    HFCS = "HFCS"      # Hydrofluorocarbons (GWP varies by compound)
    PFCS = "PFCS"      # Perfluorocarbons (GWP varies by compound)
    SF6 = "SF6"        # Sulfur hexafluoride (GWP-100 = 25,200)
    NF3 = "NF3"        # Nitrogen trifluoride (GWP-100 = 17,400)


class FuelType(str, Enum):
    """Fuel types for Scope 1 stationary and mobile combustion."""

    NATURAL_GAS = "NATURAL_GAS"
    DIESEL = "DIESEL"
    GASOLINE = "GASOLINE"
    COAL = "COAL"
    LPG = "LPG"
    FUEL_OIL = "FUEL_OIL"
    JET_FUEL = "JET_FUEL"
    BIOMASS = "BIOMASS"
    BIOGAS = "BIOGAS"
    HYDROGEN = "HYDROGEN"


class EnergySource(str, Enum):
    """Energy source types for E1-5 energy consumption and mix disclosure."""

    GRID_ELECTRICITY = "GRID_ELECTRICITY"
    NATURAL_GAS = "NATURAL_GAS"
    DIESEL = "DIESEL"
    SOLAR_PV = "SOLAR_PV"
    WIND_ONSHORE = "WIND_ONSHORE"
    WIND_OFFSHORE = "WIND_OFFSHORE"
    HYDROPOWER = "HYDROPOWER"
    NUCLEAR = "NUCLEAR"
    BIOMASS = "BIOMASS"
    GEOTHERMAL = "GEOTHERMAL"
    DISTRICT_HEATING = "DISTRICT_HEATING"
    DISTRICT_COOLING = "DISTRICT_COOLING"


class RenewableCategory(str, Enum):
    """Renewable energy categories for E1-5 renewable share calculation."""

    SOLAR = "SOLAR"
    WIND = "WIND"
    HYDRO = "HYDRO"
    GEOTHERMAL = "GEOTHERMAL"
    BIOMASS = "BIOMASS"
    OTHER = "OTHER"


class TargetType(str, Enum):
    """GHG reduction target type per E1-4."""

    ABSOLUTE = "ABSOLUTE"        # Absolute reduction (tCO2e)
    INTENSITY = "INTENSITY"      # Intensity reduction (tCO2e per unit)
    NET_ZERO = "NET_ZERO"        # Net-zero (90%+ reduction + neutralization)


class TargetPathway(str, Enum):
    """Science-based target pathway alignment per SBTi."""

    SBTi_1_5C = "SBTi_1_5C"                   # 1.5C aligned (~4.2% annual)
    SBTi_WELL_BELOW_2C = "SBTi_WELL_BELOW_2C"  # Well-below 2C (~2.5% annual)
    SBTi_NET_ZERO = "SBTi_NET_ZERO"            # Net-zero by 2050
    CUSTOM = "CUSTOM"                           # Custom non-SBTi pathway


class BaseYearApproach(str, Enum):
    """Base year selection approach per GHG Protocol."""

    FIXED = "FIXED"                 # Single fixed base year
    ROLLING = "ROLLING"             # Rolling base year (e.g., 3-year average)
    RECALCULATED = "RECALCULATED"   # Recalculated when structural changes occur


class ConsolidationApproach(str, Enum):
    """GHG Protocol organizational boundary consolidation approach."""

    EQUITY_SHARE = "EQUITY_SHARE"
    FINANCIAL_CONTROL = "FINANCIAL_CONTROL"
    OPERATIONAL_CONTROL = "OPERATIONAL_CONTROL"


class Scope3Method(str, Enum):
    """Scope 3 calculation methodology options per GHG Protocol."""

    SPEND_BASED = "SPEND_BASED"
    AVERAGE_DATA = "AVERAGE_DATA"
    SUPPLIER_SPECIFIC = "SUPPLIER_SPECIFIC"
    HYBRID = "HYBRID"


class CarbonCreditStandard(str, Enum):
    """Carbon credit certification standards for E1-7."""

    VERRA_VCS = "VERRA_VCS"          # Verified Carbon Standard
    GOLD_STANDARD = "GOLD_STANDARD"  # Gold Standard for the Global Goals
    ACR = "ACR"                      # American Carbon Registry
    CAR = "CAR"                      # Climate Action Reserve
    CDM = "CDM"                      # Clean Development Mechanism
    CORSIA = "CORSIA"                # Carbon Offsetting and Reduction Scheme
    REDD_PLUS = "REDD_PLUS"          # Reducing Emissions from Deforestation
    CUSTOM = "CUSTOM"                # Other or emerging standards


class CarbonCreditType(str, Enum):
    """Carbon credit type classification per SBTi guidance."""

    AVOIDANCE = "AVOIDANCE"    # Emission avoidance/reduction credits
    REMOVAL = "REMOVAL"        # Carbon dioxide removal credits
    BOTH = "BOTH"              # Mixed portfolio


class CarbonPricingMethod(str, Enum):
    """Internal carbon pricing methodology for E1-8."""

    SHADOW_PRICE = "SHADOW_PRICE"      # Used in investment appraisals
    INTERNAL_FEE = "INTERNAL_FEE"      # Actual charge to business units
    IMPLICIT_PRICE = "IMPLICIT_PRICE"  # Derived from actual investments
    REGULATORY = "REGULATORY"          # External regulatory price (ETS)


class PhysicalRiskType(str, Enum):
    """Physical climate risk types per TCFD and E1-9."""

    ACUTE = "ACUTE"      # Extreme weather events (floods, storms, wildfires)
    CHRONIC = "CHRONIC"  # Long-term shifts (temperature, sea level, water stress)


class TransitionRiskType(str, Enum):
    """Transition climate risk types per TCFD and E1-9."""

    POLICY = "POLICY"            # Carbon pricing, emissions regulations
    TECHNOLOGY = "TECHNOLOGY"    # Low-carbon technology disruption
    MARKET = "MARKET"            # Demand shifts, commodity price changes
    REPUTATION = "REPUTATION"    # Stakeholder pressure, consumer preferences
    LEGAL = "LEGAL"              # Climate litigation, liability


class ClimateScenario(str, Enum):
    """Climate scenario pathways for E1-9 scenario analysis."""

    RCP_2_6 = "RCP_2_6"        # ~1.5C by 2100 (strong mitigation)
    RCP_4_5 = "RCP_4_5"        # ~2.4C by 2100 (intermediate)
    RCP_6_0 = "RCP_6_0"        # ~2.8C by 2100 (higher baseline)
    RCP_8_5 = "RCP_8_5"        # ~4.3C by 2100 (no mitigation)
    SSP1_2_6 = "SSP1_2_6"      # Sustainability pathway
    SSP2_4_5 = "SSP2_4_5"      # Middle of the road
    SSP3_7_0 = "SSP3_7_0"      # Regional rivalry
    SSP5_8_5 = "SSP5_8_5"      # Fossil-fueled development


class TimeHorizon(str, Enum):
    """Time horizon classification per ESRS 1 sect. 77 and E1-9."""

    SHORT_TERM = "SHORT_TERM"        # 0-1 year
    MEDIUM_TERM = "MEDIUM_TERM"      # 1-5 years
    LONG_TERM = "LONG_TERM"          # 5+ years (up to 2050 for net-zero)


class DisclosureStatus(str, Enum):
    """Disclosure document lifecycle status."""

    DRAFT = "DRAFT"
    REVIEW = "REVIEW"
    APPROVED = "APPROVED"
    PUBLISHED = "PUBLISHED"


class DisclosureFormat(str, Enum):
    """Output format for E1 disclosure documents."""

    XBRL = "XBRL"
    PDF = "PDF"
    HTML = "HTML"
    JSON = "JSON"
    CSV = "CSV"


# =============================================================================
# Reference Data Constants
# =============================================================================

# IPCC AR6 GWP-100 values for key greenhouse gases
GWP_AR6_VALUES: Dict[str, float] = {
    "CO2": 1.0,
    "CH4_FOSSIL": 29.8,
    "CH4_BIOGENIC": 27.2,
    "N2O": 273.0,
    "SF6": 25200.0,
    "NF3": 17400.0,
    # Common HFCs
    "HFC_23": 14600.0,
    "HFC_32": 771.0,
    "HFC_125": 3740.0,
    "HFC_134A": 1530.0,
    "HFC_143A": 5810.0,
    "HFC_152A": 164.0,
    "HFC_227EA": 3600.0,
    "HFC_236FA": 8690.0,
    "HFC_245FA": 962.0,
    "HFC_365MFC": 914.0,
    "HFC_4310MEE": 1560.0,
    # Common PFCs
    "CF4": 7380.0,
    "C2F6": 12400.0,
    "C3F8": 9290.0,
    "C4F10": 10000.0,
    "C5F12": 9220.0,
    "C6F14": 7910.0,
}

# Energy source classification (fossil / nuclear / renewable)
ENERGY_SOURCE_CLASSIFICATION: Dict[str, str] = {
    "GRID_ELECTRICITY": "MIXED",
    "NATURAL_GAS": "FOSSIL",
    "DIESEL": "FOSSIL",
    "COAL": "FOSSIL",
    "LPG": "FOSSIL",
    "FUEL_OIL": "FOSSIL",
    "JET_FUEL": "FOSSIL",
    "GASOLINE": "FOSSIL",
    "SOLAR_PV": "RENEWABLE",
    "WIND_ONSHORE": "RENEWABLE",
    "WIND_OFFSHORE": "RENEWABLE",
    "HYDROPOWER": "RENEWABLE",
    "NUCLEAR": "NUCLEAR",
    "BIOMASS": "RENEWABLE",
    "BIOGAS": "RENEWABLE",
    "GEOTHERMAL": "RENEWABLE",
    "HYDROGEN_GREEN": "RENEWABLE",
    "HYDROGEN_GREY": "FOSSIL",
    "HYDROGEN_BLUE": "FOSSIL",
    "DISTRICT_HEATING": "MIXED",
    "DISTRICT_COOLING": "MIXED",
}

# Scope 3 category names per GHG Protocol
SCOPE_3_CATEGORIES: Dict[int, str] = {
    1: "Purchased Goods and Services",
    2: "Capital Goods",
    3: "Fuel- and Energy-Related Activities",
    4: "Upstream Transportation and Distribution",
    5: "Waste Generated in Operations",
    6: "Business Travel",
    7: "Employee Commuting",
    8: "Upstream Leased Assets",
    9: "Downstream Transportation and Distribution",
    10: "Processing of Sold Products",
    11: "Use of Sold Products",
    12: "End-of-Life Treatment of Sold Products",
    13: "Downstream Leased Assets",
    14: "Franchises",
    15: "Investments",
}

# SBTi annual linear reduction rates by pathway
SBTI_REDUCTION_RATES: Dict[str, float] = {
    "SBTi_1_5C": 0.042,              # 4.2% annual linear reduction
    "SBTi_WELL_BELOW_2C": 0.025,     # 2.5% annual linear reduction
    "SBTi_NET_ZERO": 0.042,          # Near-term same as 1.5C, long-term 90%+
}

# E1 disclosure requirement reference information
E1_DISCLOSURE_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "E1-1": {
        "name": "Transition Plan for Climate Change Mitigation",
        "paragraphs": "14-16",
        "application_requirements": "AR E1-1 through AR E1-12",
        "mandatory": False,
        "quantitative": False,
    },
    "E1-2": {
        "name": "Policies Related to Climate Change",
        "paragraphs": "22-24",
        "application_requirements": "AR E1-13 through AR E1-17",
        "mandatory": False,
        "quantitative": False,
    },
    "E1-3": {
        "name": "Actions and Resources",
        "paragraphs": "26-28",
        "application_requirements": "AR E1-18 through AR E1-23",
        "mandatory": False,
        "quantitative": True,
    },
    "E1-4": {
        "name": "Targets Related to Climate Change",
        "paragraphs": "30-33",
        "application_requirements": "AR E1-24 through AR E1-37",
        "mandatory": False,
        "quantitative": True,
    },
    "E1-5": {
        "name": "Energy Consumption and Mix",
        "paragraphs": "35-39",
        "application_requirements": "AR E1-38 through AR E1-45",
        "mandatory": False,
        "quantitative": True,
    },
    "E1-6": {
        "name": "Gross Scopes 1, 2, 3 and Total GHG Emissions",
        "paragraphs": "44-55",
        "application_requirements": "AR E1-46 through AR E1-62",
        "mandatory": False,
        "quantitative": True,
    },
    "E1-7": {
        "name": "GHG Removals and Carbon Credits",
        "paragraphs": "56-58",
        "application_requirements": "AR E1-63 through AR E1-68",
        "mandatory": False,
        "quantitative": True,
    },
    "E1-8": {
        "name": "Internal Carbon Pricing",
        "paragraphs": "59-61",
        "application_requirements": "AR E1-69 through AR E1-73",
        "mandatory": False,
        "quantitative": True,
    },
    "E1-9": {
        "name": "Anticipated Financial Effects",
        "paragraphs": "64-68",
        "application_requirements": "AR E1-74 through AR E1-81",
        "mandatory": False,
        "quantitative": True,
    },
}

# Available presets
AVAILABLE_PRESETS: Dict[str, str] = {
    "power_generation": "Power generation with coal/gas/renewable transition focus",
    "manufacturing": "Industrial manufacturing with process emissions and energy intensity",
    "transport": "Transport and logistics with fleet mobile combustion emphasis",
    "financial_services": "Financial services with financed emissions (Scope 3 Cat 15) focus",
    "real_estate": "Real estate with building energy and embodied carbon emphasis",
    "multi_sector": "Multi-sector conglomerate with comprehensive coverage",
}

# Default emission factor sources by region
EMISSION_FACTOR_SOURCES: Dict[str, str] = {
    "EU": "European Environment Agency (EEA)",
    "US": "EPA Emission Factors Hub",
    "UK": "DEFRA/BEIS Conversion Factors",
    "GLOBAL": "IEA Emission Factors",
    "IPCC": "IPCC Emission Factor Database (EFDB)",
}


# =============================================================================
# Pydantic Sub-Config Models (8 sub-config models)
# =============================================================================


class GHGConfig(BaseModel):
    """Configuration for GHG inventory compilation engine.

    Implements GHG Protocol Corporate Standard methodology for
    organizational and operational boundaries, consolidation approach,
    emission factor sources, and GWP values per ESRS E1-6.
    """

    enabled: bool = Field(
        True,
        description="Enable GHG inventory compilation",
    )
    base_year: int = Field(
        2020,
        ge=2015,
        le=2030,
        description="GHG inventory base year for target tracking",
    )
    reporting_year: int = Field(
        2025,
        ge=2020,
        le=2035,
        description="Current reporting year",
    )
    consolidation_approach: ConsolidationApproach = Field(
        ConsolidationApproach.OPERATIONAL_CONTROL,
        description="GHG Protocol organizational boundary approach",
    )
    gwp_source: str = Field(
        "IPCC_AR6",
        description="GWP value source: IPCC_AR6 (required by ESRS E1 para 48)",
    )
    gwp_time_horizon: int = Field(
        100,
        description="GWP time horizon in years (GWP-100 per ESRS E1)",
    )
    emission_factors_source: str = Field(
        "DEFRA_2025",
        description="Primary emission factor database: DEFRA_2025, EPA_2025, IEA_2024, IPCC_EFDB",
    )
    emission_factors_region: str = Field(
        "EU",
        description="Region for default emission factors: EU, US, UK, GLOBAL",
    )
    scopes_enabled: List[GHGScope] = Field(
        default_factory=lambda: [
            GHGScope.SCOPE_1,
            GHGScope.SCOPE_2_LOCATION,
            GHGScope.SCOPE_2_MARKET,
            GHGScope.SCOPE_3,
        ],
        description="GHG scopes to calculate",
    )
    scope_3_categories: List[int] = Field(
        default_factory=lambda: list(range(1, 16)),
        description="Scope 3 categories to include (1-15 per GHG Protocol)",
    )
    scope_3_significance_threshold_pct: float = Field(
        1.0,
        ge=0.0,
        le=10.0,
        description="Threshold (% of total Scope 3) below which a category is not significant",
    )
    scope_3_default_method: Scope3Method = Field(
        Scope3Method.HYBRID,
        description="Default Scope 3 calculation methodology",
    )
    kyoto_gas_disaggregation: bool = Field(
        True,
        description="Disaggregate emissions by Kyoto gas (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3)",
    )
    biogenic_co2_separate: bool = Field(
        True,
        description="Report biogenic CO2 emissions separately (required by ESRS E1-6)",
    )
    ghg_intensity_denominators: List[str] = Field(
        default_factory=lambda: ["net_revenue_eur_million"],
        description="Denominators for GHG intensity ratios (net_revenue_eur_million, production_units, fte)",
    )
    base_year_recalculation_threshold_pct: float = Field(
        5.0,
        ge=1.0,
        le=20.0,
        description="Threshold for triggering base year recalculation (% change from structural event)",
    )
    data_quality_scoring: bool = Field(
        True,
        description="Score data quality for each emission source (1-5 scale per PCAF)",
    )

    @field_validator("scope_3_categories")
    @classmethod
    def validate_scope_3_categories(cls, v: List[int]) -> List[int]:
        """Validate Scope 3 categories are in range 1-15."""
        invalid = [c for c in v if c < 1 or c > 15]
        if invalid:
            raise ValueError(
                f"Invalid Scope 3 categories: {invalid}. Must be 1-15."
            )
        return sorted(set(v))


class EnergyConfig(BaseModel):
    """Configuration for energy consumption and mix engine.

    Implements ESRS E1-5 energy disclosure requirements including total
    energy consumption, source breakdown, and renewable share.
    """

    enabled: bool = Field(
        True,
        description="Enable energy consumption and mix calculation",
    )
    reporting_unit: str = Field(
        "MWh",
        description="Energy reporting unit: MWh (required by ESRS E1-5)",
    )
    include_renewables: bool = Field(
        True,
        description="Calculate renewable energy share",
    )
    renewable_categories: List[RenewableCategory] = Field(
        default_factory=lambda: [
            RenewableCategory.SOLAR,
            RenewableCategory.WIND,
            RenewableCategory.HYDRO,
            RenewableCategory.GEOTHERMAL,
            RenewableCategory.BIOMASS,
        ],
        description="Renewable energy categories to track",
    )
    ppa_tracking: bool = Field(
        True,
        description="Track Power Purchase Agreement (PPA) renewable energy",
    )
    rec_tracking: bool = Field(
        True,
        description="Track Renewable Energy Certificate (REC/GO) purchases",
    )
    self_generation_tracking: bool = Field(
        True,
        description="Track self-generated renewable energy (on-site solar, wind)",
    )
    energy_intensity_denominators: List[str] = Field(
        default_factory=lambda: ["net_revenue_eur_million"],
        description="Denominators for energy intensity ratios",
    )
    fossil_fuel_types: List[FuelType] = Field(
        default_factory=lambda: [
            FuelType.NATURAL_GAS,
            FuelType.DIESEL,
            FuelType.GASOLINE,
            FuelType.COAL,
            FuelType.LPG,
            FuelType.FUEL_OIL,
            FuelType.JET_FUEL,
        ],
        description="Fossil fuel types to track in energy mix",
    )
    nuclear_included: bool = Field(
        True,
        description="Include nuclear energy in energy mix (separate from fossil and renewable)",
    )
    district_energy_included: bool = Field(
        True,
        description="Include district heating and cooling in energy mix",
    )
    cross_validate_ghg: bool = Field(
        True,
        description="Cross-validate energy data against GHG inventory for consistency",
    )
    epc_rating_integration: bool = Field(
        False,
        description="Integrate Energy Performance Certificate (EPC) ratings for buildings",
    )

    @field_validator("reporting_unit")
    @classmethod
    def validate_reporting_unit(cls, v: str) -> str:
        """Validate energy reporting unit is ESRS-compliant."""
        valid_units = {"MWh", "GJ", "TJ"}
        if v not in valid_units:
            raise ValueError(
                f"Invalid energy unit: {v}. Valid: {sorted(valid_units)}. "
                f"ESRS E1-5 requires MWh."
            )
        return v


class TransitionPlanConfig(BaseModel):
    """Configuration for transition plan compilation engine.

    Implements ESRS E1-1 transition plan disclosure requirements including
    decarbonization levers, locked-in emissions, CapEx alignment, and
    Paris Agreement pathway compatibility.
    """

    enabled: bool = Field(
        True,
        description="Enable transition plan compilation",
    )
    target_year: int = Field(
        2050,
        ge=2030,
        le=2070,
        description="Long-term target year for transition plan (typically 2050 for net-zero)",
    )
    interim_targets_enabled: bool = Field(
        True,
        description="Include interim milestone targets (e.g., 2030, 2035, 2040)",
    )
    interim_target_years: List[int] = Field(
        default_factory=lambda: [2030, 2035, 2040, 2045],
        description="Years for interim target milestones",
    )
    locked_in_emissions_calc: bool = Field(
        True,
        description="Calculate locked-in GHG emissions from existing assets",
    )
    capex_alignment: bool = Field(
        True,
        description="Track CapEx alignment with transition plan (climate-aligned share)",
    )
    opex_alignment: bool = Field(
        False,
        description="Track OpEx alignment with transition plan",
    )
    scenario_alignment: str = Field(
        "1.5C",
        description="Paris Agreement pathway alignment: 1.5C, WELL_BELOW_2C",
    )
    decarbonization_levers: List[str] = Field(
        default_factory=lambda: [
            "energy_efficiency",
            "electrification",
            "renewable_energy",
            "fuel_switching",
            "process_changes",
            "carbon_capture",
        ],
        description="Decarbonization lever categories to track",
    )
    taxonomy_alignment: bool = Field(
        True,
        description="Assess EU Taxonomy alignment of transition investments",
    )
    has_transition_plan: bool = Field(
        True,
        description="Whether the undertaking has adopted a transition plan",
    )

    @field_validator("interim_target_years")
    @classmethod
    def validate_interim_years_ascending(cls, v: List[int]) -> List[int]:
        """Validate interim target years are ascending."""
        sorted_years = sorted(set(v))
        return sorted_years


class TargetConfig(BaseModel):
    """Configuration for GHG reduction target tracking engine.

    Implements ESRS E1-4 target disclosure requirements including SBTi
    alignment, base year management, and progress tracking.
    """

    enabled: bool = Field(
        True,
        description="Enable GHG reduction target tracking",
    )
    sbti_commitment_level: TargetPathway = Field(
        TargetPathway.SBTi_1_5C,
        description="SBTi commitment level for science-based targets",
    )
    sbti_validated: bool = Field(
        False,
        description="Whether SBTi targets are formally validated by SBTi",
    )
    sbti_submission_date: Optional[str] = Field(
        None,
        description="Date of SBTi target submission (YYYY-MM-DD)",
    )
    sbti_validation_date: Optional[str] = Field(
        None,
        description="Date of SBTi target validation (YYYY-MM-DD)",
    )
    base_year: int = Field(
        2020,
        ge=2015,
        le=2030,
        description="Base year for target calculation",
    )
    base_year_approach: BaseYearApproach = Field(
        BaseYearApproach.FIXED,
        description="Base year selection approach per GHG Protocol",
    )
    target_year: int = Field(
        2030,
        ge=2025,
        le=2060,
        description="Near-term target year",
    )
    long_term_target_year: int = Field(
        2050,
        ge=2040,
        le=2070,
        description="Long-term (net-zero) target year",
    )
    target_types: List[TargetType] = Field(
        default_factory=lambda: [TargetType.ABSOLUTE, TargetType.INTENSITY],
        description="Types of GHG reduction targets",
    )
    target_scopes: List[GHGScope] = Field(
        default_factory=lambda: [
            GHGScope.SCOPE_1,
            GHGScope.SCOPE_2_MARKET,
            GHGScope.SCOPE_3,
        ],
        description="GHG scopes covered by targets",
    )
    reduction_path: str = Field(
        "linear",
        description="Reduction pathway: linear, front_loaded, back_loaded",
    )
    intensity_denominator: str = Field(
        "net_revenue_eur_million",
        description="Denominator for intensity targets",
    )
    progress_tracking_frequency: str = Field(
        "annual",
        description="Frequency of target progress tracking: annual, quarterly",
    )
    variance_analysis: bool = Field(
        True,
        description="Enable variance analysis (actual vs. required reduction)",
    )
    base_year_recalculation_policy: bool = Field(
        True,
        description="Enable base year recalculation per GHG Protocol policy",
    )

    @model_validator(mode="after")
    def validate_target_year_after_base(self) -> "TargetConfig":
        """Validate target year is after base year."""
        if self.target_year <= self.base_year:
            raise ValueError(
                f"Target year ({self.target_year}) must be after "
                f"base year ({self.base_year})."
            )
        return self


class CarbonCreditConfig(BaseModel):
    """Configuration for carbon credit and removals engine.

    Implements ESRS E1-7 disclosure requirements for GHG removals
    and carbon credit management.
    """

    enabled: bool = Field(
        True,
        description="Enable carbon credit and removal tracking",
    )
    allowed_standards: List[CarbonCreditStandard] = Field(
        default_factory=lambda: [
            CarbonCreditStandard.VERRA_VCS,
            CarbonCreditStandard.GOLD_STANDARD,
            CarbonCreditStandard.ACR,
            CarbonCreditStandard.CDM,
        ],
        description="Allowed carbon credit certification standards",
    )
    credit_types: List[CarbonCreditType] = Field(
        default_factory=lambda: [CarbonCreditType.AVOIDANCE, CarbonCreditType.REMOVAL],
        description="Allowed carbon credit types (avoidance, removal)",
    )
    vintage_requirements: int = Field(
        5,
        ge=1,
        le=15,
        description="Maximum vintage age in years for acceptable credits",
    )
    retirement_policy: str = Field(
        "retire_on_claim",
        description="Credit retirement policy: retire_on_claim, retire_annually, retire_on_purchase",
    )
    quality_assessment: bool = Field(
        True,
        description="Enable credit quality assessment (additionality, permanence, co-benefits)",
    )
    sbti_offset_guidance: bool = Field(
        True,
        description="Apply SBTi guidance on offset use (credits do not count toward near-term targets)",
    )
    removal_tracking: bool = Field(
        True,
        description="Track own-operations GHG removals (afforestation, DACCS, BECCS)",
    )
    separate_from_gross_emissions: bool = Field(
        True,
        description="Ensure credits are reported separately from gross emissions (ESRS requirement)",
    )

    @field_validator("vintage_requirements")
    @classmethod
    def validate_vintage(cls, v: int) -> int:
        """Validate vintage requirement is reasonable."""
        if v > 10:
            logger.warning(
                "Vintage requirement of %d years is lenient. "
                "Best practice is 5 years or less.",
                v,
            )
        return v


class CarbonPricingConfig(BaseModel):
    """Configuration for internal carbon pricing engine.

    Implements ESRS E1-8 disclosure requirements for internal
    carbon pricing mechanisms.
    """

    enabled: bool = Field(
        True,
        description="Enable internal carbon pricing disclosure",
    )
    has_carbon_pricing: bool = Field(
        False,
        description="Whether the undertaking uses internal carbon pricing",
    )
    pricing_methods: List[CarbonPricingMethod] = Field(
        default_factory=lambda: [CarbonPricingMethod.SHADOW_PRICE],
        description="Internal carbon pricing methodologies used",
    )
    price_per_tco2e: str = Field(
        "100.00",
        description="Current internal carbon price per tCO2e (EUR, as Decimal-safe string)",
    )
    price_currency: str = Field(
        "EUR",
        description="Currency for carbon price",
    )
    coverage_scope: List[GHGScope] = Field(
        default_factory=lambda: [GHGScope.SCOPE_1, GHGScope.SCOPE_2_MARKET],
        description="GHG scopes covered by the internal carbon price",
    )
    coverage_share_pct: str = Field(
        "80.0",
        description="Percentage of total emissions covered by carbon price (Decimal-safe string)",
    )
    shadow_price_scenarios: List[Dict[str, str]] = Field(
        default_factory=lambda: [
            {"scenario": "current", "price_per_tco2e": "100.00", "year": "2025"},
            {"scenario": "medium", "price_per_tco2e": "150.00", "year": "2030"},
            {"scenario": "high", "price_per_tco2e": "250.00", "year": "2035"},
        ],
        description="Shadow price scenarios for investment appraisals",
    )
    influences_investment_decisions: bool = Field(
        True,
        description="Whether the carbon price influences investment decisions",
    )
    influences_procurement: bool = Field(
        False,
        description="Whether the carbon price influences procurement decisions",
    )


class ClimateRiskConfig(BaseModel):
    """Configuration for climate risk and opportunity assessment engine.

    Implements ESRS E1-9 disclosure requirements for anticipated financial
    effects from physical and transition risks and climate-related
    opportunities, aligned with TCFD recommendations.
    """

    enabled: bool = Field(
        True,
        description="Enable climate risk and opportunity assessment",
    )
    physical_risk_types: List[PhysicalRiskType] = Field(
        default_factory=lambda: [PhysicalRiskType.ACUTE, PhysicalRiskType.CHRONIC],
        description="Physical climate risk types to assess",
    )
    transition_risk_types: List[TransitionRiskType] = Field(
        default_factory=lambda: [
            TransitionRiskType.POLICY,
            TransitionRiskType.TECHNOLOGY,
            TransitionRiskType.MARKET,
            TransitionRiskType.REPUTATION,
            TransitionRiskType.LEGAL,
        ],
        description="Transition risk types to assess",
    )
    scenarios: List[ClimateScenario] = Field(
        default_factory=lambda: [
            ClimateScenario.SSP1_2_6,
            ClimateScenario.SSP2_4_5,
            ClimateScenario.SSP5_8_5,
        ],
        description="Climate scenarios for risk assessment",
    )
    time_horizons: List[TimeHorizon] = Field(
        default_factory=lambda: [
            TimeHorizon.SHORT_TERM,
            TimeHorizon.MEDIUM_TERM,
            TimeHorizon.LONG_TERM,
        ],
        description="Time horizons for financial effect estimation",
    )
    short_term_years: int = Field(
        1,
        ge=1,
        le=2,
        description="Short-term horizon definition in years",
    )
    medium_term_years: int = Field(
        5,
        ge=2,
        le=10,
        description="Medium-term horizon definition in years",
    )
    long_term_years: int = Field(
        10,
        ge=5,
        le=30,
        description="Long-term horizon definition in years",
    )
    quantify_financial_effects: bool = Field(
        True,
        description="Quantify anticipated financial effects in EUR",
    )
    financial_effect_currency: str = Field(
        "EUR",
        description="Currency for financial effect quantification",
    )
    opportunity_assessment: bool = Field(
        True,
        description="Include climate-related opportunity assessment",
    )
    site_level_exposure: bool = Field(
        False,
        description="Assess physical risk exposure at site/asset level",
    )
    portfolio_risk: bool = Field(
        False,
        description="Assess portfolio-level risk (for financial institutions)",
    )
    tcfd_alignment: bool = Field(
        True,
        description="Align risk assessment with TCFD recommendation pillars",
    )
    use_climate_hazard_data: bool = Field(
        True,
        description="Use AGENT-DATA-020 Climate Hazard Connector for physical risk data",
    )


class ReportingConfig(BaseModel):
    """Configuration for E1 disclosure report generation."""

    enabled: bool = Field(
        True,
        description="Enable E1 report generation",
    )
    disclosure_requirements: List[str] = Field(
        default_factory=lambda: [
            "E1-1", "E1-2", "E1-3", "E1-4", "E1-5",
            "E1-6", "E1-7", "E1-8", "E1-9",
        ],
        description="E1 disclosure requirements to include in report",
    )
    output_formats: List[DisclosureFormat] = Field(
        default_factory=lambda: [DisclosureFormat.PDF, DisclosureFormat.XBRL],
        description="Output formats for E1 reports",
    )
    xbrl_tagging: bool = Field(
        True,
        description="Tag all quantitative datapoints with EFRAG XBRL taxonomy identifiers",
    )
    sha256_provenance: bool = Field(
        True,
        description="Generate SHA-256 provenance hashes for all calculated values",
    )
    multi_language: bool = Field(
        False,
        description="Enable multi-language report generation",
    )
    supported_languages: List[str] = Field(
        default_factory=lambda: ["en"],
        description="Supported languages for E1 reports",
    )
    review_workflow: bool = Field(
        True,
        description="Enable review and approval workflow for E1 reports",
    )
    watermark_draft: bool = Field(
        True,
        description="Apply DRAFT watermark to unapproved E1 documents",
    )
    audit_trail_report: bool = Field(
        True,
        description="Generate separate audit trail report for assurance",
    )
    executive_summary: bool = Field(
        True,
        description="Generate executive summary with key climate metrics",
    )
    year_over_year_comparison: bool = Field(
        True,
        description="Include year-over-year comparison in reports",
    )
    data_quality_disclosure: bool = Field(
        True,
        description="Disclose data quality scores for GHG inventory",
    )
    assumption_disclosure: bool = Field(
        True,
        description="Disclose key assumptions in E1 calculations",
    )
    retention_years: int = Field(
        10,
        ge=1,
        le=15,
        description="Report and audit trail retention period in years",
    )

    @field_validator("disclosure_requirements")
    @classmethod
    def validate_disclosure_requirements(cls, v: List[str]) -> List[str]:
        """Validate disclosure requirement identifiers."""
        valid = set(E1_DISCLOSURE_REQUIREMENTS.keys())
        invalid = [dr for dr in v if dr not in valid]
        if invalid:
            raise ValueError(
                f"Invalid disclosure requirements: {invalid}. "
                f"Valid: {sorted(valid)}"
            )
        return v


# =============================================================================
# Main Configuration Model
# =============================================================================


class E1ClimateConfig(BaseModel):
    """Main configuration for PACK-016 ESRS E1 Climate Change Pack.

    This is the root configuration model that contains all sub-configurations
    for the complete E1 disclosure process. The sector and operational profile
    drive emission factor selection, energy mix defaults, target pathway
    calibration, and climate risk scenario selection.
    """

    # Company identification
    company_name: str = Field(
        "",
        description="Legal entity name of the undertaking",
    )
    reporting_year: int = Field(
        2025,
        ge=2024,
        le=2035,
        description="Reporting year for CSRD E1 disclosure",
    )
    sector: str = Field(
        "GENERAL",
        description="Primary NACE sector: ENERGY, MANUFACTURING, TRANSPORT, FINANCIAL_SERVICES, REAL_ESTATE, GENERAL",
    )
    fiscal_year_end: str = Field(
        "12-31",
        description="Fiscal year end date (MM-DD)",
    )
    currency: str = Field(
        "EUR",
        description="Reporting currency for financial values",
    )
    reporting_boundary: str = Field(
        "group",
        description="Reporting boundary: group, parent_only, specific_entities",
    )

    # Engine sub-configurations
    ghg: GHGConfig = Field(
        default_factory=GHGConfig,
        description="GHG inventory compilation configuration (E1-6)",
    )
    energy: EnergyConfig = Field(
        default_factory=EnergyConfig,
        description="Energy consumption and mix configuration (E1-5)",
    )
    transition_plan: TransitionPlanConfig = Field(
        default_factory=TransitionPlanConfig,
        description="Transition plan compilation configuration (E1-1)",
    )
    targets: TargetConfig = Field(
        default_factory=TargetConfig,
        description="GHG reduction target tracking configuration (E1-4)",
    )
    carbon_credits: CarbonCreditConfig = Field(
        default_factory=CarbonCreditConfig,
        description="Carbon credit and removals configuration (E1-7)",
    )
    carbon_pricing: CarbonPricingConfig = Field(
        default_factory=CarbonPricingConfig,
        description="Internal carbon pricing configuration (E1-8)",
    )
    climate_risk: ClimateRiskConfig = Field(
        default_factory=ClimateRiskConfig,
        description="Climate risk and opportunity configuration (E1-9)",
    )
    reporting: ReportingConfig = Field(
        default_factory=ReportingConfig,
        description="E1 report generation configuration",
    )

    @model_validator(mode="after")
    def validate_base_years_consistent(self) -> "E1ClimateConfig":
        """Ensure GHG base year and target base year are consistent."""
        if self.ghg.base_year != self.targets.base_year:
            logger.warning(
                "GHG inventory base year (%d) differs from target base year (%d). "
                "Consider using the same base year for consistency.",
                self.ghg.base_year,
                self.targets.base_year,
            )
        return self

    @model_validator(mode="after")
    def validate_reporting_years_consistent(self) -> "E1ClimateConfig":
        """Ensure reporting year is consistent across configs."""
        if self.reporting_year != self.ghg.reporting_year:
            logger.warning(
                "Pack reporting year (%d) differs from GHG reporting year (%d). "
                "Synchronizing to pack reporting year.",
                self.reporting_year,
                self.ghg.reporting_year,
            )
            object.__setattr__(self.ghg, "reporting_year", self.reporting_year)
        return self

    @model_validator(mode="after")
    def validate_energy_ghg_cross_validation(self) -> "E1ClimateConfig":
        """Warn if energy and GHG cross-validation is disabled."""
        if self.energy.enabled and self.ghg.enabled and not self.energy.cross_validate_ghg:
            logger.warning(
                "Energy-GHG cross-validation is disabled. Energy data should "
                "be reconcilable with Scope 1 and Scope 2 emissions."
            )
        return self

    @model_validator(mode="after")
    def validate_financial_services_profile(self) -> "E1ClimateConfig":
        """Apply financial services specific validations."""
        if self.sector == "FINANCIAL_SERVICES":
            if 15 not in self.ghg.scope_3_categories:
                logger.warning(
                    "Financial services sector: Scope 3 Category 15 (Investments) "
                    "is typically the dominant emission source. Consider including it."
                )
            if not self.climate_risk.portfolio_risk:
                logger.warning(
                    "Financial services sector: portfolio-level climate risk "
                    "assessment is recommended."
                )
        return self


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """Top-level pack configuration wrapper.

    Handles preset loading, environment variable overrides, and
    configuration merging.
    """

    pack: E1ClimateConfig = Field(
        default_factory=E1ClimateConfig,
        description="Main E1 Climate configuration",
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
        "PACK-016-esrs-e1-climate",
        description="Pack identifier",
    )

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "PackConfig":
        """Load configuration from a named preset.

        Args:
            preset_name: Name of the preset (power_generation, manufacturing, etc.)
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

        pack_config = E1ClimateConfig(**preset_data)
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

        pack_config = E1ClimateConfig(**config_data)
        return cls(pack=pack_config)

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """Load configuration overrides from environment variables.

        Environment variables prefixed with E1_PACK_ are loaded and mapped
        to configuration keys. Nested keys use double underscore.

        Example: E1_PACK_GHG__BASE_YEAR=2019
        """
        overrides: Dict[str, Any] = {}
        prefix = "E1_PACK_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                parts = config_key.split("__")
                current = overrides
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                # Parse value
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
    def _deep_merge(
        base: Dict[str, Any],
        override: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence."""
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = PackConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get_config_hash(self) -> str:
        """Generate SHA-256 hash of the current configuration for provenance."""
        config_json = self.model_dump_json(indent=None)
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()


# =============================================================================
# Utility Functions
# =============================================================================


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


def validate_config(config: E1ClimateConfig) -> List[str]:
    """Validate an E1 configuration and return any warnings.

    Args:
        config: E1ClimateConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check GHG scopes coverage
    required_scopes = {GHGScope.SCOPE_1, GHGScope.SCOPE_2_LOCATION, GHGScope.SCOPE_2_MARKET}
    configured_scopes = set(config.ghg.scopes_enabled)
    missing_scopes = required_scopes - configured_scopes
    if missing_scopes:
        scope_names = ", ".join(s.value for s in sorted(missing_scopes, key=lambda s: s.value))
        warnings.append(
            f"Required GHG scopes not enabled: {scope_names}. "
            f"ESRS E1-6 requires Scope 1 and Scope 2 (both methods)."
        )

    # Check Scope 3 coverage
    if GHGScope.SCOPE_3 in configured_scopes:
        if not config.ghg.scope_3_categories:
            warnings.append(
                "Scope 3 is enabled but no categories are configured. "
                "Include at least the significant categories per GHG Protocol."
            )
        elif len(config.ghg.scope_3_categories) < 3:
            warnings.append(
                f"Only {len(config.ghg.scope_3_categories)} Scope 3 categories "
                f"configured. A comprehensive E1-6 typically covers 5+ categories."
            )

    # Check base year is reasonable
    if config.ghg.base_year > config.reporting_year:
        warnings.append(
            f"GHG base year ({config.ghg.base_year}) is after reporting year "
            f"({config.reporting_year}). Base year must precede reporting year."
        )

    # Check target alignment
    if config.targets.enabled:
        if config.targets.target_year <= config.reporting_year:
            warnings.append(
                f"Target year ({config.targets.target_year}) is not after "
                f"reporting year ({config.reporting_year}). "
                f"Near-term targets should be 5-10 years ahead."
            )
        if not config.targets.sbti_validated:
            warnings.append(
                "SBTi targets are not validated. Consider submitting for "
                "SBTi validation for ESRS E1-4 credibility."
            )

    # Check energy configuration
    if config.energy.enabled and not config.energy.include_renewables:
        warnings.append(
            "Renewable energy tracking is disabled. ESRS E1-5 requires "
            "disclosure of renewable energy share."
        )

    # Check biogenic CO2
    if config.ghg.enabled and not config.ghg.biogenic_co2_separate:
        warnings.append(
            "Biogenic CO2 separate reporting is disabled. ESRS E1-6 "
            "requires biogenic CO2 to be reported separately."
        )

    # Check Kyoto gas disaggregation
    if config.ghg.enabled and not config.ghg.kyoto_gas_disaggregation:
        warnings.append(
            "Kyoto gas disaggregation is disabled. ESRS E1-6 para 48 "
            "requires disaggregation where individual gases are significant."
        )

    # Check carbon credit separation
    if config.carbon_credits.enabled and not config.carbon_credits.separate_from_gross_emissions:
        warnings.append(
            "Carbon credits are not configured to be reported separately "
            "from gross emissions. ESRS E1-7 requires separate reporting."
        )

    # Check provenance tracking
    if not config.reporting.sha256_provenance:
        warnings.append(
            "SHA-256 provenance tracking is disabled. Consider enabling "
            "for audit trail integrity."
        )

    return warnings


def get_default_config(sector: str = "GENERAL") -> E1ClimateConfig:
    """Get default E1 configuration for a given sector.

    Args:
        sector: Sector identifier.

    Returns:
        E1ClimateConfig instance with sector-appropriate defaults.
    """
    return E1ClimateConfig(sector=sector)


def get_gwp_value(gas: str) -> float:
    """Get IPCC AR6 GWP-100 value for a greenhouse gas.

    Args:
        gas: Gas identifier (e.g., "CO2", "CH4_FOSSIL", "SF6").

    Returns:
        GWP-100 value as float.
    """
    return GWP_AR6_VALUES.get(gas, 0.0)


def get_scope_3_category_name(category: int) -> str:
    """Get Scope 3 category name by number.

    Args:
        category: Category number (1-15).

    Returns:
        Category name string.
    """
    return SCOPE_3_CATEGORIES.get(category, f"Unknown Category {category}")


def get_e1_disclosure_info(dr_id: str) -> Dict[str, Any]:
    """Get detailed information about an E1 disclosure requirement.

    Args:
        dr_id: Disclosure requirement identifier (e.g., "E1-6").

    Returns:
        Dictionary with DR name, paragraphs, and application requirements.
    """
    return E1_DISCLOSURE_REQUIREMENTS.get(dr_id, {
        "name": dr_id,
        "paragraphs": "Unknown",
        "application_requirements": "Unknown",
        "mandatory": False,
        "quantitative": False,
    })


def get_sbti_reduction_rate(pathway: Union[str, TargetPathway]) -> float:
    """Get SBTi annual linear reduction rate for a pathway.

    Args:
        pathway: Target pathway enum or string value.

    Returns:
        Annual linear reduction rate as float (e.g., 0.042 = 4.2%).
    """
    key = pathway.value if isinstance(pathway, TargetPathway) else pathway
    return SBTI_REDUCTION_RATES.get(key, 0.0)


def list_available_presets() -> Dict[str, str]:
    """List all available E1 configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return AVAILABLE_PRESETS.copy()
