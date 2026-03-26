"""
PACK-041 Scope 1-2 Complete Pack - Configuration Manager

This module implements the Scope12CompleteConfig and PackConfig classes that
load, merge, and validate all configuration for the Scope 1-2 Complete Pack.
It provides comprehensive Pydantic v2 models for a production-ready,
verification-grade GHG inventory covering all Scope 1 and Scope 2 emission
sources per the GHG Protocol Corporate Standard.

Consolidation Approaches:
    - EQUITY_SHARE: Proportional allocation by ownership percentage
    - OPERATIONAL_CONTROL: 100% of operations under management authority
    - FINANCIAL_CONTROL: 100% of operations under financial direction

GWP Sources:
    - AR4: IPCC Fourth Assessment Report (2007)
    - AR5: IPCC Fifth Assessment Report (2014)
    - AR6: IPCC Sixth Assessment Report (2021)

Methodology Tiers:
    - TIER_1: Default emission factors (spend-based or average data)
    - TIER_2: Country/technology-specific factors
    - TIER_3: Facility-specific measurement or modelling

Emission Factor Sources:
    - IPCC, DEFRA, EPA, UBA, ADEME, ISPRA, IEA, SUPPLIER, FACILITY

Sector Presets:
    corporate_office / manufacturing / energy_utility / transport_logistics /
    food_agriculture / real_estate / healthcare / sme_simplified

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (sector-specific defaults)
    3. Environment overrides (SCOPE12_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - GHG Protocol Corporate Standard (Revised Edition, 2015)
    - GHG Protocol Scope 2 Guidance (2015)
    - ISO 14064-1:2018
    - EU CSRD / ESRS E1
    - CDP Climate Change 2026
    - SBTi Corporate Net-Zero Standard v1.1
    - US SEC Climate Disclosure Rules
    - California SB 253

Example:
    >>> config = PackConfig.from_preset("corporate_office")
    >>> print(config.pack.sector_type)
    SectorType.OFFICE
    >>> print(config.pack.boundary.consolidation_approach)
    ConsolidationApproach.OPERATIONAL_CONTROL
    >>> print(config.pack.scope2.dual_reporting)
    True
"""

import hashlib
import json
import logging
import os
from datetime import date, datetime
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
# Enums - GHG inventory enumeration types
# =============================================================================


class ConsolidationApproach(str, Enum):
    """Organisational boundary consolidation approach per GHG Protocol Ch. 3."""

    EQUITY_SHARE = "EQUITY_SHARE"
    OPERATIONAL_CONTROL = "OPERATIONAL_CONTROL"
    FINANCIAL_CONTROL = "FINANCIAL_CONTROL"


class GWPSource(str, Enum):
    """IPCC Assessment Report source for GWP values."""

    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"


class MethodologyTier(str, Enum):
    """IPCC methodology tier for emission quantification."""

    TIER_1 = "TIER_1"
    TIER_2 = "TIER_2"
    TIER_3 = "TIER_3"


class EmissionFactorSource(str, Enum):
    """Source database for emission factors."""

    IPCC = "IPCC"
    DEFRA = "DEFRA"
    EPA = "EPA"
    UBA = "UBA"
    ADEME = "ADEME"
    ISPRA = "ISPRA"
    IEA = "IEA"
    SUPPLIER = "SUPPLIER"
    FACILITY = "FACILITY"


class FrameworkType(str, Enum):
    """Regulatory and reporting framework identifiers."""

    GHG_PROTOCOL = "GHG_PROTOCOL"
    ESRS_E1 = "ESRS_E1"
    CDP = "CDP"
    ISO_14064 = "ISO_14064"
    SBTI = "SBTI"
    SEC = "SEC"
    SB_253 = "SB_253"


class InventoryScope(str, Enum):
    """GHG inventory scope classification."""

    SCOPE_1 = "SCOPE_1"
    SCOPE_2_LOCATION = "SCOPE_2_LOCATION"
    SCOPE_2_MARKET = "SCOPE_2_MARKET"


class GasType(str, Enum):
    """Kyoto basket greenhouse gas types."""

    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"
    HFCS = "HFCS"
    PFCS = "PFCS"
    SF6 = "SF6"
    NF3 = "NF3"


class SectorType(str, Enum):
    """Sector classification for preset selection."""

    OFFICE = "OFFICE"
    MANUFACTURING = "MANUFACTURING"
    ENERGY_UTILITY = "ENERGY_UTILITY"
    TRANSPORT_LOGISTICS = "TRANSPORT_LOGISTICS"
    FOOD_AGRICULTURE = "FOOD_AGRICULTURE"
    REAL_ESTATE = "REAL_ESTATE"
    HEALTHCARE = "HEALTHCARE"
    SME = "SME"


class OutputFormat(str, Enum):
    """Output format for inventory reports."""

    MARKDOWN = "MARKDOWN"
    HTML = "HTML"
    JSON = "JSON"
    CSV = "CSV"
    XBRL = "XBRL"


class ReportingFrequency(str, Enum):
    """Reporting and monitoring frequency."""

    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    ANNUAL = "ANNUAL"


# =============================================================================
# Reference Data Constants
# =============================================================================


# Sector information for preset guidance
SECTOR_INFO: Dict[str, Dict[str, Any]] = {
    "OFFICE": {
        "name": "Corporate Office",
        "typical_scope1_sources": [
            "Natural gas heating (boilers)",
            "Backup diesel generators",
            "Refrigerants in HVAC systems",
        ],
        "typical_scope2_sources": [
            "Purchased electricity",
            "District heating (where applicable)",
        ],
        "dominant_scope": "Scope 2 (typically 60-85% of Scope 1+2)",
        "key_gases": ["CO2", "HFCs"],
        "intensity_metrics": ["tCO2e/FTE", "tCO2e/m2", "tCO2e/MEUR revenue"],
        "typical_total_tco2e_per_fte": "1.0-3.0",
    },
    "MANUFACTURING": {
        "name": "Manufacturing Facility",
        "typical_scope1_sources": [
            "Process heat (natural gas, fuel oil)",
            "Process emissions (chemical reactions)",
            "Fugitive emissions (equipment leaks)",
            "Company vehicles",
            "Refrigerants in cooling systems",
        ],
        "typical_scope2_sources": [
            "Purchased electricity for motors and processes",
            "Purchased steam (CHP or district)",
        ],
        "dominant_scope": "Scope 1 (typically 50-80% of Scope 1+2)",
        "key_gases": ["CO2", "CH4", "N2O", "HFCs"],
        "intensity_metrics": ["tCO2e/unit produced", "tCO2e/MEUR revenue", "tCO2e/tonne output"],
        "typical_total_tco2e_per_meur": "50-500",
    },
    "ENERGY_UTILITY": {
        "name": "Energy Utility",
        "typical_scope1_sources": [
            "Fossil fuel combustion (coal, gas, oil)",
            "SF6 from electrical switchgear",
            "Process emissions (desulphurisation)",
            "Fugitive emissions (gas distribution)",
        ],
        "typical_scope2_sources": [
            "Purchased electricity for plant auxiliaries",
            "Grid losses (for transmission operators)",
        ],
        "dominant_scope": "Scope 1 (typically 85-99% of Scope 1+2)",
        "key_gases": ["CO2", "CH4", "N2O", "SF6"],
        "intensity_metrics": ["tCO2e/MWh generated", "tCO2e/MEUR revenue"],
        "typical_total_tco2e_per_mwh": "0.3-1.0",
    },
    "TRANSPORT_LOGISTICS": {
        "name": "Transport & Logistics",
        "typical_scope1_sources": [
            "Road freight vehicles (diesel, LNG)",
            "Company cars and vans",
            "Rail locomotives (diesel)",
            "Marine vessels",
            "Aviation (owned aircraft)",
        ],
        "typical_scope2_sources": [
            "Purchased electricity for depots and offices",
            "Electric vehicle charging (fleet EVs)",
        ],
        "dominant_scope": "Scope 1 (typically 70-95% of Scope 1+2)",
        "key_gases": ["CO2", "CH4", "N2O"],
        "intensity_metrics": ["tCO2e/tonne-km", "tCO2e/vehicle-km", "tCO2e/MEUR revenue"],
        "typical_total_tco2e_per_m_tkm": "0.05-0.15",
    },
    "FOOD_AGRICULTURE": {
        "name": "Food & Agriculture",
        "typical_scope1_sources": [
            "Livestock enteric fermentation (CH4)",
            "Manure management (CH4, N2O)",
            "Agricultural soils (N2O from fertiliser)",
            "Rice cultivation (CH4)",
            "Process heat for food processing",
            "Company vehicles and farm equipment",
            "Land use change",
        ],
        "typical_scope2_sources": [
            "Purchased electricity for irrigation, cold storage, processing",
        ],
        "dominant_scope": "Scope 1 (typically 70-90% of Scope 1+2)",
        "key_gases": ["CO2", "CH4", "N2O"],
        "intensity_metrics": ["tCO2e/tonne product", "tCO2e/hectare", "tCO2e/MEUR revenue"],
        "typical_total_tco2e_per_hectare": "1.0-10.0",
    },
    "REAL_ESTATE": {
        "name": "Real Estate Portfolio",
        "typical_scope1_sources": [
            "Central heating plant (natural gas)",
            "Backup generators",
            "Refrigerants in HVAC systems",
        ],
        "typical_scope2_sources": [
            "Purchased electricity for common areas and tenants (if operational control)",
            "District heating and cooling",
        ],
        "dominant_scope": "Scope 2 (typically 55-80% of Scope 1+2)",
        "key_gases": ["CO2", "HFCs"],
        "intensity_metrics": ["kgCO2e/m2 GLA", "tCO2e/MEUR asset value"],
        "typical_total_kgco2e_per_m2": "20-80",
    },
    "HEALTHCARE": {
        "name": "Healthcare System",
        "typical_scope1_sources": [
            "Central heating plant (natural gas, fuel oil)",
            "Backup generators (24/7 critical systems)",
            "Medical gas (N2O for anaesthesia)",
            "Refrigerants in extensive HVAC and lab cooling",
            "Fleet vehicles (ambulances, service vehicles)",
        ],
        "typical_scope2_sources": [
            "Purchased electricity (24/7 operation, high intensity)",
            "District heating (large campus facilities)",
        ],
        "dominant_scope": "Mixed (Scope 1 40-60%, Scope 2 40-60%)",
        "key_gases": ["CO2", "N2O", "HFCs"],
        "intensity_metrics": ["tCO2e/bed", "tCO2e/m2", "tCO2e/MEUR budget"],
        "typical_total_tco2e_per_bed": "5.0-15.0",
    },
    "SME": {
        "name": "Small-Medium Enterprise",
        "typical_scope1_sources": [
            "Natural gas heating",
            "Company vehicles",
        ],
        "typical_scope2_sources": [
            "Purchased electricity",
        ],
        "dominant_scope": "Scope 2 (typically 50-75% of Scope 1+2)",
        "key_gases": ["CO2"],
        "intensity_metrics": ["tCO2e/FTE", "tCO2e/MEUR revenue"],
        "typical_total_tco2e_per_fte": "0.5-2.0",
    },
}

# GWP values by assessment report (100-year horizon)
GWP_VALUES: Dict[str, Dict[str, float]] = {
    "AR4": {
        "CO2": 1.0,
        "CH4": 25.0,
        "N2O": 298.0,
        "SF6": 22800.0,
        "NF3": 17200.0,
        "HFC-23": 14800.0,
        "HFC-32": 675.0,
        "HFC-125": 3500.0,
        "HFC-134a": 1430.0,
        "HFC-143a": 4470.0,
        "HFC-152a": 124.0,
        "HFC-227ea": 3220.0,
        "HFC-245fa": 1030.0,
        "HFC-365mfc": 794.0,
        "HFC-4310mee": 1640.0,
        "PFC-14": 7390.0,
        "PFC-116": 12200.0,
        "PFC-218": 8830.0,
        "PFC-31-10": 8860.0,
        "R-410A": 2088.0,
        "R-404A": 3922.0,
        "R-407C": 1774.0,
        "R-507A": 3985.0,
    },
    "AR5": {
        "CO2": 1.0,
        "CH4": 28.0,
        "N2O": 265.0,
        "SF6": 23500.0,
        "NF3": 16100.0,
        "HFC-23": 12400.0,
        "HFC-32": 677.0,
        "HFC-125": 3170.0,
        "HFC-134a": 1300.0,
        "HFC-143a": 4800.0,
        "HFC-152a": 138.0,
        "HFC-227ea": 3350.0,
        "HFC-245fa": 858.0,
        "HFC-365mfc": 804.0,
        "HFC-4310mee": 1650.0,
        "PFC-14": 6630.0,
        "PFC-116": 11100.0,
        "PFC-218": 8900.0,
        "PFC-31-10": 9200.0,
        "R-410A": 1924.0,
        "R-404A": 3943.0,
        "R-407C": 1624.0,
        "R-507A": 3985.0,
    },
    "AR6": {
        "CO2": 1.0,
        "CH4": 27.9,
        "N2O": 273.0,
        "SF6": 25200.0,
        "NF3": 17400.0,
        "HFC-23": 14600.0,
        "HFC-32": 771.0,
        "HFC-125": 3740.0,
        "HFC-134a": 1530.0,
        "HFC-143a": 5810.0,
        "HFC-152a": 164.0,
        "HFC-227ea": 3600.0,
        "HFC-245fa": 962.0,
        "HFC-365mfc": 914.0,
        "HFC-4310mee": 1600.0,
        "PFC-14": 7380.0,
        "PFC-116": 12400.0,
        "PFC-218": 9290.0,
        "PFC-31-10": 10000.0,
        "R-410A": 2256.0,
        "R-404A": 4728.0,
        "R-407C": 1907.0,
        "R-507A": 4772.0,
    },
}

# Available presets
AVAILABLE_PRESETS: Dict[str, str] = {
    "corporate_office": "Office-based organisations (financial services, technology, consulting)",
    "manufacturing": "Industrial manufacturing with process emissions and stationary combustion",
    "energy_utility": "Power generation and energy distribution utilities",
    "transport_logistics": "Fleet operators and logistics companies (mobile combustion dominant)",
    "food_agriculture": "Agricultural operations and food processing",
    "real_estate": "Property portfolios and REIT companies",
    "healthcare": "Hospitals and healthcare systems",
    "sme_simplified": "Simplified inventory for small-medium enterprises",
}


# =============================================================================
# Pydantic Sub-Config Models
# =============================================================================


class BoundaryConfig(BaseModel):
    """Configuration for organisational and operational boundary setting.

    Defines the consolidation approach and significance thresholds for
    the GHG inventory boundary per GHG Protocol Chapter 3 and Chapter 4.
    """

    consolidation_approach: ConsolidationApproach = Field(
        ConsolidationApproach.OPERATIONAL_CONTROL,
        description="Consolidation approach: EQUITY_SHARE, OPERATIONAL_CONTROL, or FINANCIAL_CONTROL",
    )
    significance_threshold_pct: float = Field(
        1.0,
        ge=0.0,
        le=10.0,
        description="Significance threshold (%) below which sources may be excluded",
    )
    de_minimis_threshold_pct: float = Field(
        0.5,
        ge=0.0,
        le=5.0,
        description="De minimis threshold (%) below which sources are automatically excluded",
    )
    include_biogenic_co2: bool = Field(
        True,
        description="Report biogenic CO2 separately per ESRS E1 requirements",
    )
    include_ets_breakdown: bool = Field(
        False,
        description="Report ETS-covered vs non-ETS emissions separately per ESRS E1",
    )


class SourceCompletenessConfig(BaseModel):
    """Configuration for source completeness assessment.

    Controls which Scope 1 and Scope 2 source categories are enabled
    and at what methodology tier they should be quantified.
    """

    require_all_scope1_categories: bool = Field(
        False,
        description="Require all 8 Scope 1 categories to be assessed (even if zero)",
    )
    require_dual_reporting: bool = Field(
        True,
        description="Require both location-based and market-based Scope 2 per GHG Protocol",
    )
    minimum_source_coverage_pct: float = Field(
        95.0,
        ge=80.0,
        le=100.0,
        description="Minimum percentage of total emissions that must be from quantified sources",
    )
    document_exclusions: bool = Field(
        True,
        description="Require documented rationale for every excluded source",
    )


class EmissionFactorConfig(BaseModel):
    """Configuration for emission factor management.

    Defines preferred emission factor sources, GWP assessment report,
    and factor update policies.
    """

    preferred_sources: List[EmissionFactorSource] = Field(
        default_factory=lambda: [
            EmissionFactorSource.DEFRA,
            EmissionFactorSource.IPCC,
            EmissionFactorSource.IEA,
        ],
        description="Emission factor source preference hierarchy (first = highest priority)",
    )
    gwp_source: GWPSource = Field(
        GWPSource.AR5,
        description="IPCC Assessment Report for GWP values (AR4, AR5, or AR6)",
    )
    allow_supplier_factors: bool = Field(
        True,
        description="Allow supplier-provided emission factors (Tier 2/3)",
    )
    allow_facility_factors: bool = Field(
        True,
        description="Allow facility-specific measured factors (Tier 3)",
    )
    factor_vintage_max_years: int = Field(
        5,
        ge=1,
        le=10,
        description="Maximum age (years) for emission factors before requiring update",
    )
    auto_update_factors: bool = Field(
        True,
        description="Automatically check for and apply emission factor updates",
    )

    @field_validator("preferred_sources")
    @classmethod
    def validate_sources_not_empty(
        cls, v: List[EmissionFactorSource]
    ) -> List[EmissionFactorSource]:
        """At least one emission factor source must be configured."""
        if not v:
            raise ValueError("At least one emission factor source must be configured.")
        return v


class Scope1Config(BaseModel):
    """Configuration for Scope 1 direct emission categories.

    Controls which of the 8 Scope 1 categories are enabled and at
    what methodology tier they should be quantified.
    """

    stationary_combustion_enabled: bool = Field(
        True,
        description="Enable MRV-001 stationary combustion (boilers, furnaces, generators)",
    )
    stationary_combustion_tier: MethodologyTier = Field(
        MethodologyTier.TIER_2,
        description="Methodology tier for stationary combustion",
    )
    mobile_combustion_enabled: bool = Field(
        True,
        description="Enable MRV-002 mobile combustion (company vehicles, off-road)",
    )
    mobile_combustion_tier: MethodologyTier = Field(
        MethodologyTier.TIER_2,
        description="Methodology tier for mobile combustion",
    )
    process_emissions_enabled: bool = Field(
        False,
        description="Enable MRV-003 process emissions (chemical/physical transformations)",
    )
    process_emissions_tier: MethodologyTier = Field(
        MethodologyTier.TIER_1,
        description="Methodology tier for process emissions",
    )
    fugitive_emissions_enabled: bool = Field(
        False,
        description="Enable MRV-004 fugitive emissions (equipment leaks, venting)",
    )
    fugitive_emissions_tier: MethodologyTier = Field(
        MethodologyTier.TIER_1,
        description="Methodology tier for fugitive emissions",
    )
    refrigerant_fgas_enabled: bool = Field(
        True,
        description="Enable MRV-005 refrigerant and F-gas losses (HFCs, PFCs, SF6)",
    )
    refrigerant_fgas_tier: MethodologyTier = Field(
        MethodologyTier.TIER_1,
        description="Methodology tier for refrigerant and F-gas",
    )
    land_use_enabled: bool = Field(
        False,
        description="Enable MRV-006 land use emissions",
    )
    land_use_tier: MethodologyTier = Field(
        MethodologyTier.TIER_1,
        description="Methodology tier for land use emissions",
    )
    waste_treatment_enabled: bool = Field(
        False,
        description="Enable MRV-007 on-site waste treatment emissions",
    )
    waste_treatment_tier: MethodologyTier = Field(
        MethodologyTier.TIER_1,
        description="Methodology tier for waste treatment emissions",
    )
    agricultural_enabled: bool = Field(
        False,
        description="Enable MRV-008 agricultural emissions (livestock, manure, crops)",
    )
    agricultural_tier: MethodologyTier = Field(
        MethodologyTier.TIER_1,
        description="Methodology tier for agricultural emissions",
    )


class Scope2Config(BaseModel):
    """Configuration for Scope 2 indirect energy emission categories.

    Controls Scope 2 accounting methods, dual reporting, and
    contractual instrument allocation.
    """

    dual_reporting: bool = Field(
        True,
        description="Report both location-based and market-based Scope 2 (GHG Protocol required)",
    )
    location_based_enabled: bool = Field(
        True,
        description="Enable MRV-009 location-based Scope 2 (grid-average factors)",
    )
    market_based_enabled: bool = Field(
        True,
        description="Enable MRV-010 market-based Scope 2 (contractual instruments)",
    )
    steam_heat_enabled: bool = Field(
        False,
        description="Enable MRV-011 steam and heat purchase",
    )
    cooling_enabled: bool = Field(
        False,
        description="Enable MRV-012 cooling purchase",
    )
    instrument_allocation: bool = Field(
        True,
        description="Enable contractual instrument allocation (PPAs, RECs, GOs)",
    )
    grid_ef_source: EmissionFactorSource = Field(
        EmissionFactorSource.IEA,
        description="Primary source for grid emission factors",
    )
    residual_mix_enabled: bool = Field(
        True,
        description="Use residual mix factors for uncontracted electricity (EU requirement)",
    )
    validate_instrument_quality: bool = Field(
        True,
        description="Validate instrument quality criteria per GHG Protocol Scope 2 Guidance",
    )


class UncertaintyConfig(BaseModel):
    """Configuration for uncertainty quantification.

    Controls Monte Carlo simulation parameters and uncertainty
    reporting requirements per ISO 14064-1 Clause 6.3.
    """

    enabled: bool = Field(
        True,
        description="Enable uncertainty analysis",
    )
    monte_carlo_iterations: int = Field(
        10000,
        ge=1000,
        le=100000,
        description="Number of Monte Carlo iterations",
    )
    confidence_level: float = Field(
        0.95,
        ge=0.80,
        le=0.99,
        description="Confidence level for uncertainty intervals (0.95 = 95%)",
    )
    include_activity_data_uncertainty: bool = Field(
        True,
        description="Include activity data uncertainty in analysis",
    )
    include_emission_factor_uncertainty: bool = Field(
        True,
        description="Include emission factor uncertainty in analysis",
    )
    sensitivity_analysis: bool = Field(
        True,
        description="Perform sensitivity analysis (tornado diagrams)",
    )
    contribution_analysis: bool = Field(
        True,
        description="Calculate source contribution to total uncertainty",
    )


class BaseYearConfig(BaseModel):
    """Configuration for base year management.

    Defines the base year, recalculation triggers, and threshold rules
    per GHG Protocol Chapter 5.
    """

    year: int = Field(
        2024,
        ge=2015,
        le=2030,
        description="Base year for the GHG inventory",
    )
    recalculation_threshold_pct: float = Field(
        5.0,
        ge=1.0,
        le=20.0,
        description="Significance threshold (%) for triggering base year recalculation",
    )
    auto_detect_triggers: bool = Field(
        True,
        description="Automatically detect recalculation triggers (acquisitions, method changes)",
    )
    rolling_base_year: bool = Field(
        False,
        description="Use rolling base year (recalculate to most recent verified year)",
    )
    fixed_base_year: bool = Field(
        True,
        description="Use fixed base year (default, recommended by GHG Protocol)",
    )


class TrendAnalysisConfig(BaseModel):
    """Configuration for multi-year trend analysis and projections."""

    enabled: bool = Field(
        True,
        description="Enable multi-year trend analysis",
    )
    intensity_metrics: List[str] = Field(
        default_factory=lambda: [
            "tCO2e/MEUR_revenue",
            "tCO2e/FTE",
        ],
        description="Intensity denominators to track (revenue, FTE, m2, unit)",
    )
    lmdi_decomposition: bool = Field(
        False,
        description="Enable LMDI decomposition analysis (activity, structure, intensity effects)",
    )
    sbti_pathway_comparison: bool = Field(
        False,
        description="Compare trend against SBTi 1.5C linear reduction pathway",
    )
    sbti_annual_reduction_rate: float = Field(
        4.2,
        ge=0.0,
        le=15.0,
        description="SBTi required annual linear reduction rate (% per annum)",
    )
    projection_years: int = Field(
        5,
        ge=1,
        le=30,
        description="Number of years to project under business-as-usual scenario",
    )

    @field_validator("intensity_metrics")
    @classmethod
    def validate_metrics_not_empty(cls, v: List[str]) -> List[str]:
        """At least one intensity metric must be configured."""
        if not v:
            raise ValueError("At least one intensity metric must be configured.")
        return v


class ComplianceConfig(BaseModel):
    """Configuration for regulatory compliance framework mapping."""

    frameworks: List[FrameworkType] = Field(
        default_factory=lambda: [
            FrameworkType.GHG_PROTOCOL,
            FrameworkType.ESRS_E1,
            FrameworkType.CDP,
        ],
        description="Regulatory frameworks to map inventory output to",
    )
    esrs_biogenic_co2: bool = Field(
        True,
        description="Report biogenic CO2 separately as required by ESRS E1",
    )
    esrs_ets_percentage: bool = Field(
        False,
        description="Report ETS-covered emissions percentage for ESRS E1",
    )
    cdp_country_disaggregation: bool = Field(
        True,
        description="Disaggregate Scope 1 by country for CDP C6.2",
    )
    cdp_gas_disaggregation: bool = Field(
        True,
        description="Disaggregate Scope 1 by GHG type for CDP C6.3",
    )
    sec_materiality_threshold: bool = Field(
        False,
        description="Apply SEC materiality threshold for Scope 1+2 disclosure",
    )
    sbti_target_tracking: bool = Field(
        False,
        description="Track progress against SBTi-validated targets",
    )
    sb253_reporting: bool = Field(
        False,
        description="Generate SB 253 annual report data",
    )

    @field_validator("frameworks")
    @classmethod
    def validate_frameworks_not_empty(cls, v: List[FrameworkType]) -> List[FrameworkType]:
        """At least one compliance framework must be configured."""
        if not v:
            raise ValueError("At least one compliance framework must be selected.")
        return v


class ReportingConfig(BaseModel):
    """Configuration for GHG inventory report generation."""

    frequency: ReportingFrequency = Field(
        ReportingFrequency.ANNUAL,
        description="Reporting frequency for inventory updates",
    )
    formats: List[OutputFormat] = Field(
        default_factory=lambda: [OutputFormat.HTML, OutputFormat.JSON],
        description="Output formats for inventory reports",
    )
    executive_summary: bool = Field(
        True,
        description="Generate executive summary with KPI dashboard",
    )
    detailed_inventory: bool = Field(
        True,
        description="Generate detailed source-level inventory report",
    )
    verification_package: bool = Field(
        True,
        description="Generate verification-ready data package (ISO 14064-3)",
    )
    output_language: str = Field(
        "en",
        description="Report language (ISO 639-1)",
    )
    include_methodology_notes: bool = Field(
        True,
        description="Include calculation methodology notes in reports",
    )
    include_emission_factor_register: bool = Field(
        True,
        description="Include emission factor register with provenance in reports",
    )
    watermark_draft: bool = Field(
        True,
        description="Apply DRAFT watermark to unapproved inventory reports",
    )


class SecurityConfig(BaseModel):
    """Security and access control configuration."""

    roles: List[str] = Field(
        default_factory=lambda: [
            "ghg_manager",
            "sustainability_officer",
            "facility_manager",
            "data_analyst",
            "verifier",
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
        description="Require encryption at rest for stored inventory data",
    )


class PerformanceConfig(BaseModel):
    """Performance and resource limits for pack execution."""

    max_facilities: int = Field(
        1000,
        ge=1,
        le=10000,
        description="Maximum number of facilities per inventory run",
    )
    max_entities: int = Field(
        500,
        ge=1,
        le=5000,
        description="Maximum number of entities for consolidation",
    )
    cache_ttl_seconds: int = Field(
        3600,
        ge=60,
        le=86400,
        description="Cache TTL for emission factors and grid factors (seconds)",
    )
    batch_size: int = Field(
        500,
        ge=50,
        le=5000,
        description="Batch size for bulk facility processing",
    )
    calculation_timeout_seconds: int = Field(
        300,
        ge=30,
        le=1800,
        description="Timeout for individual MRV agent calculations (seconds)",
    )
    parallel_agents: int = Field(
        4,
        ge=1,
        le=13,
        description="Maximum number of MRV agents running in parallel",
    )


class AuditTrailConfig(BaseModel):
    """Configuration for calculation audit trail and provenance."""

    enabled: bool = Field(
        True,
        description="Enable audit trail for all emission calculations",
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
        description="Track full data lineage from source to inventory total",
    )
    retention_years: int = Field(
        7,
        ge=1,
        le=15,
        description="Audit trail retention period in years",
    )
    emission_factor_citation: bool = Field(
        True,
        description="Cite emission factor source and vintage for every calculation",
    )


# =============================================================================
# Main Configuration Model
# =============================================================================


class Scope12CompleteConfig(BaseModel):
    """Main configuration for PACK-041 Scope 1-2 Complete Pack.

    This is the root configuration model that contains all sub-configurations
    for the complete Scope 1 and Scope 2 GHG inventory. The sector_type
    field drives which emission sources are prioritised and which presets
    are available.
    """

    # Organisation identification
    company_name: str = Field(
        "",
        description="Legal entity name of the reporting company",
    )
    facility_name: str = Field(
        "",
        description="Primary facility or site identifier",
    )
    sector_type: SectorType = Field(
        SectorType.OFFICE,
        description="Primary sector classification for preset selection",
    )
    country: str = Field(
        "DE",
        description="Primary country of operations (ISO 3166-1 alpha-2)",
    )
    reporting_year: int = Field(
        2026,
        ge=2020,
        le=2035,
        description="Reporting year for the GHG inventory",
    )

    # Organisation characteristics
    revenue_meur: Optional[float] = Field(
        None,
        ge=0,
        description="Annual revenue in million EUR for intensity metrics",
    )
    employees_fte: Optional[int] = Field(
        None,
        ge=0,
        description="Full-time equivalent employees for intensity metrics",
    )
    floor_area_m2: Optional[float] = Field(
        None,
        ge=0,
        description="Total floor area in m2 for intensity metrics",
    )

    # Sub-configurations
    boundary: BoundaryConfig = Field(
        default_factory=BoundaryConfig,
        description="Organisational and operational boundary configuration",
    )
    source_completeness: SourceCompletenessConfig = Field(
        default_factory=SourceCompletenessConfig,
        description="Source completeness assessment configuration",
    )
    emission_factors: EmissionFactorConfig = Field(
        default_factory=EmissionFactorConfig,
        description="Emission factor management configuration",
    )
    scope1: Scope1Config = Field(
        default_factory=Scope1Config,
        description="Scope 1 direct emission categories configuration",
    )
    scope2: Scope2Config = Field(
        default_factory=Scope2Config,
        description="Scope 2 indirect energy emission categories configuration",
    )
    uncertainty: UncertaintyConfig = Field(
        default_factory=UncertaintyConfig,
        description="Uncertainty quantification configuration",
    )
    base_year: BaseYearConfig = Field(
        default_factory=BaseYearConfig,
        description="Base year management configuration",
    )
    trend_analysis: TrendAnalysisConfig = Field(
        default_factory=TrendAnalysisConfig,
        description="Trend analysis and projection configuration",
    )
    compliance: ComplianceConfig = Field(
        default_factory=ComplianceConfig,
        description="Regulatory compliance framework configuration",
    )
    reporting: ReportingConfig = Field(
        default_factory=ReportingConfig,
        description="Report generation configuration",
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security and access control",
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance and resource limits",
    )
    audit_trail: AuditTrailConfig = Field(
        default_factory=AuditTrailConfig,
        description="Audit trail and provenance configuration",
    )

    @model_validator(mode="after")
    def validate_dual_reporting_consistency(self) -> "Scope12CompleteConfig":
        """Dual reporting requires both location-based and market-based enabled."""
        if self.scope2.dual_reporting:
            if not self.scope2.location_based_enabled:
                logger.warning(
                    "Dual reporting requires location-based Scope 2. Enabling automatically."
                )
                self.scope2.location_based_enabled = True
            if not self.scope2.market_based_enabled:
                logger.warning(
                    "Dual reporting requires market-based Scope 2. Enabling automatically."
                )
                self.scope2.market_based_enabled = True
        return self

    @model_validator(mode="after")
    def validate_sme_uses_simplified(self) -> "Scope12CompleteConfig":
        """SME organisations default to simplified configuration."""
        if self.sector_type == SectorType.SME:
            if self.uncertainty.monte_carlo_iterations > 5000:
                logger.info(
                    "SME sector: reducing Monte Carlo iterations to 5000 for efficiency."
                )
                self.uncertainty.monte_carlo_iterations = 5000
        return self

    @model_validator(mode="after")
    def validate_agriculture_categories(self) -> "Scope12CompleteConfig":
        """Agriculture sector should enable agricultural and land use categories."""
        if self.sector_type == SectorType.FOOD_AGRICULTURE:
            if not self.scope1.agricultural_enabled:
                logger.info(
                    "Food/agriculture sector: enabling agricultural emissions category."
                )
                self.scope1.agricultural_enabled = True
            if not self.scope1.land_use_enabled:
                logger.info(
                    "Food/agriculture sector: enabling land use emissions category."
                )
                self.scope1.land_use_enabled = True
        return self

    @model_validator(mode="after")
    def validate_energy_utility_categories(self) -> "Scope12CompleteConfig":
        """Energy utilities should enable process and fugitive emissions."""
        if self.sector_type == SectorType.ENERGY_UTILITY:
            if not self.scope1.process_emissions_enabled:
                logger.info(
                    "Energy utility sector: enabling process emissions category."
                )
                self.scope1.process_emissions_enabled = True
            if not self.scope1.fugitive_emissions_enabled:
                logger.info(
                    "Energy utility sector: enabling fugitive emissions category."
                )
                self.scope1.fugitive_emissions_enabled = True
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

    pack: Scope12CompleteConfig = Field(
        default_factory=Scope12CompleteConfig,
        description="Main Scope 1-2 Complete inventory configuration",
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
        "PACK-041-scope-1-2-complete",
        description="Pack identifier",
    )

    @classmethod
    def from_preset(
        cls, preset_name: str, overrides: Optional[Dict[str, Any]] = None
    ) -> "PackConfig":
        """Load configuration from a named preset.

        Args:
            preset_name: Name of the preset (corporate_office, manufacturing, etc.)
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

        pack_config = Scope12CompleteConfig(**preset_data)
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

        pack_config = Scope12CompleteConfig(**config_data)
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
        pack_config = Scope12CompleteConfig(**merged)
        return cls(
            pack=pack_config,
            preset_name=base.preset_name,
            config_version=base.config_version,
        )

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """Load configuration overrides from environment variables.

        Environment variables prefixed with SCOPE12_PACK_ are loaded
        and mapped to configuration keys. Nested keys use double underscore.

        Example: SCOPE12_PACK_BOUNDARY__CONSOLIDATION_APPROACH=EQUITY_SHARE
                 SCOPE12_PACK_EMISSION_FACTORS__GWP_SOURCE=AR6
        """
        overrides: Dict[str, Any] = {}
        prefix = "SCOPE12_PACK_"
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


def validate_config(config: Scope12CompleteConfig) -> List[str]:
    """Validate a Scope 1-2 Complete configuration and return any warnings.

    Args:
        config: Scope12CompleteConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check company identification
    if not config.company_name:
        warnings.append(
            "No company_name configured. Add a company name for report identification."
        )

    # Check intensity metric data
    if config.trend_analysis.enabled:
        metrics = config.trend_analysis.intensity_metrics
        if any("revenue" in m.lower() for m in metrics) and config.revenue_meur is None:
            warnings.append(
                "Revenue-based intensity metric configured but revenue_meur not provided."
            )
        if any("fte" in m.lower() for m in metrics) and config.employees_fte is None:
            warnings.append(
                "FTE-based intensity metric configured but employees_fte not provided."
            )
        if any("m2" in m.lower() for m in metrics) and config.floor_area_m2 is None:
            warnings.append(
                "Floor area intensity metric configured but floor_area_m2 not provided."
            )

    # Check Scope 1 source enablement
    scope1_enabled = any([
        config.scope1.stationary_combustion_enabled,
        config.scope1.mobile_combustion_enabled,
        config.scope1.process_emissions_enabled,
        config.scope1.fugitive_emissions_enabled,
        config.scope1.refrigerant_fgas_enabled,
        config.scope1.land_use_enabled,
        config.scope1.waste_treatment_enabled,
        config.scope1.agricultural_enabled,
    ])
    if not scope1_enabled:
        warnings.append(
            "No Scope 1 categories enabled. At least one Scope 1 source should be configured."
        )

    # Check Scope 2 dual reporting
    if config.scope2.dual_reporting:
        if not config.scope2.location_based_enabled:
            warnings.append(
                "Dual reporting enabled but location-based Scope 2 is disabled."
            )
        if not config.scope2.market_based_enabled:
            warnings.append(
                "Dual reporting enabled but market-based Scope 2 is disabled."
            )

    # Check compliance framework consistency
    if FrameworkType.ESRS_E1 in config.compliance.frameworks:
        if not config.boundary.include_biogenic_co2:
            warnings.append(
                "ESRS E1 compliance requires biogenic CO2 reporting. "
                "Set boundary.include_biogenic_co2 to true."
            )

    if FrameworkType.CDP in config.compliance.frameworks:
        if not config.compliance.cdp_country_disaggregation:
            warnings.append(
                "CDP compliance requires country disaggregation for C6.2."
            )

    if FrameworkType.SBTI in config.compliance.frameworks:
        if not config.trend_analysis.sbti_pathway_comparison:
            warnings.append(
                "SBTi compliance selected but sbti_pathway_comparison is disabled."
            )

    # Check base year
    if config.base_year.year >= config.reporting_year:
        warnings.append(
            f"Base year ({config.base_year.year}) should be earlier than "
            f"reporting year ({config.reporting_year})."
        )

    # Check emission factor source
    if not config.emission_factors.preferred_sources:
        warnings.append(
            "No emission factor sources configured."
        )

    # Check agriculture-specific
    if config.sector_type == SectorType.FOOD_AGRICULTURE:
        if not config.scope1.agricultural_enabled:
            warnings.append(
                "Food/agriculture sector should enable agricultural emissions (MRV-008)."
            )

    # Check energy utility-specific
    if config.sector_type == SectorType.ENERGY_UTILITY:
        if not config.scope1.process_emissions_enabled:
            warnings.append(
                "Energy utility sector should enable process emissions (MRV-003)."
            )
        if not config.scope1.fugitive_emissions_enabled:
            warnings.append(
                "Energy utility sector should enable fugitive emissions (MRV-004)."
            )

    # Check transport-specific
    if config.sector_type == SectorType.TRANSPORT_LOGISTICS:
        if not config.scope1.mobile_combustion_enabled:
            warnings.append(
                "Transport/logistics sector should enable mobile combustion (MRV-002)."
            )

    return warnings


def get_default_config(
    sector_type: SectorType = SectorType.OFFICE,
) -> Scope12CompleteConfig:
    """Get default configuration for a given sector type.

    Args:
        sector_type: Sector type to configure for.

    Returns:
        Scope12CompleteConfig instance with sector-appropriate defaults.
    """
    return Scope12CompleteConfig(sector_type=sector_type)


def get_sector_info(sector_type: Union[str, SectorType]) -> Dict[str, Any]:
    """Get detailed information about a sector type.

    Args:
        sector_type: Sector type enum or string value.

    Returns:
        Dictionary with name, typical sources, dominant scope, and intensity metrics.
    """
    key = sector_type.value if isinstance(sector_type, SectorType) else sector_type
    return SECTOR_INFO.get(
        key,
        {
            "name": key,
            "typical_scope1_sources": ["Varies by operation"],
            "typical_scope2_sources": ["Purchased electricity"],
            "dominant_scope": "Varies",
            "key_gases": ["CO2"],
            "intensity_metrics": ["tCO2e/MEUR revenue"],
        },
    )


def list_available_presets() -> Dict[str, str]:
    """List all available configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return AVAILABLE_PRESETS.copy()
