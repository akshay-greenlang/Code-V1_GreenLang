"""
PACK-013 CSRD Manufacturing Pack - Configuration Manager

This module implements the CSRDManufacturingConfig and PackConfig classes that
load, merge, and validate all configuration for the CSRD Manufacturing Pack.
It provides comprehensive Pydantic v2 models for every aspect of manufacturing
sector CSRD compliance: process emissions, energy intensity, product carbon
footprint (PCF), circular economy metrics, water and pollution management,
BAT/BREF compliance, supply chain emissions, and sector benchmarking.

Manufacturing Sub-Sectors:
    - CEMENT: Portland cement, clinker, specialty cements
    - STEEL: Integrated steelworks (BF-BOF), electric arc furnace (EAF), DRI
    - ALUMINUM: Primary smelting (Hall-Heroult), secondary recycling
    - CHEMICALS: Basic chemicals, specialty chemicals, petrochemicals
    - GLASS: Float glass, container glass, fibreglass
    - CERAMICS: Bricks, tiles, refractory products, technical ceramics
    - PULP_PAPER: Pulp mills, paper mills, board manufacturing
    - FOOD_BEVERAGE: Food processing, beverage production, dairy
    - TEXTILES: Fibre production, weaving, dyeing, finishing
    - ELECTRONICS: Semiconductor fabrication, PCB assembly, consumer devices
    - AUTOMOTIVE: Vehicle assembly, powertrain, body-in-white
    - MACHINERY: Industrial equipment, turbines, compressors
    - PHARMACEUTICALS: API synthesis, formulation, packaging
    - PLASTICS_RUBBER: Polymer production, moulding, extrusion
    - FURNITURE: Wood furniture, upholstered, metal furniture
    - PACKAGING: Paper/board packaging, plastic packaging, metal cans

Manufacturing Tiers:
    - HEAVY_INDUSTRY: Cement, steel, aluminium, glass, ceramics (EU ETS covered)
    - DISCRETE: Automotive, electronics, machinery (product-centric)
    - PROCESS: Chemicals, food & beverage, textiles, pulp & paper (continuous)
    - LIGHT: Pharmaceuticals, plastics/rubber, furniture, packaging (consumer)

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (heavy_industry / discrete_manufacturing / process_manufacturing
       / light_manufacturing / multi_site / sme_manufacturer)
    3. Environment overrides (CSRD_MFG_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - CSRD: Directive (EU) 2022/2464
    - ESRS: Delegated Regulation (EU) 2023/2772 (Set 1)
    - EU ETS: Directive 2003/87/EC (as amended by 2023/959)
    - IED: Directive 2010/75/EU (BAT/BREF)
    - CBAM: Regulation (EU) 2023/956
    - ESPR: Regulation (EU) 2024/1781 (Ecodesign / DPP)
    - EU Taxonomy: Regulation (EU) 2020/852
    - ISO 14067:2018 (Product Carbon Footprint)
    - ISO 50001:2018 (Energy Management Systems)
    - REACH: Regulation (EC) No 1907/2006
    - SBTi: Science Based Targets initiative

Example:
    >>> config = PackConfig.from_preset("heavy_industry")
    >>> print(config.pack.manufacturing_tier)
    ManufacturingTier.HEAVY_INDUSTRY
    >>> print(config.pack.process_emissions.cbam_affected)
    True
    >>> print(config.pack.bat_compliance.compliance_level)
    BATComplianceLevel.COMPLIANT
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
# Enums - Manufacturing-specific enumeration types
# =============================================================================


class ManufacturingSubSector(str, Enum):
    """Manufacturing sub-sector classification."""

    CEMENT = "CEMENT"
    STEEL = "STEEL"
    ALUMINUM = "ALUMINUM"
    CHEMICALS = "CHEMICALS"
    GLASS = "GLASS"
    CERAMICS = "CERAMICS"
    PULP_PAPER = "PULP_PAPER"
    FOOD_BEVERAGE = "FOOD_BEVERAGE"
    TEXTILES = "TEXTILES"
    ELECTRONICS = "ELECTRONICS"
    AUTOMOTIVE = "AUTOMOTIVE"
    MACHINERY = "MACHINERY"
    PHARMACEUTICALS = "PHARMACEUTICALS"
    PLASTICS_RUBBER = "PLASTICS_RUBBER"
    FURNITURE = "FURNITURE"
    PACKAGING = "PACKAGING"


class ManufacturingTier(str, Enum):
    """Manufacturing tier classification by energy intensity and process type."""

    HEAVY_INDUSTRY = "HEAVY_INDUSTRY"
    DISCRETE = "DISCRETE"
    PROCESS = "PROCESS"
    LIGHT = "LIGHT"


class EUETSPhase(str, Enum):
    """EU Emissions Trading System phase."""

    PHASE_3 = "PHASE_3"  # 2013-2020
    PHASE_4 = "PHASE_4"  # 2021-2030


class CBAMStatus(str, Enum):
    """CBAM applicability status for the manufacturing entity."""

    NOT_AFFECTED = "NOT_AFFECTED"
    TRANSITIONAL = "TRANSITIONAL"
    FULL_COMPLIANCE = "FULL_COMPLIANCE"


class BATComplianceLevel(str, Enum):
    """BAT/BREF compliance assessment level."""

    COMPLIANT = "COMPLIANT"
    WITHIN_RANGE = "WITHIN_RANGE"
    NON_COMPLIANT = "NON_COMPLIANT"
    DEROGATION = "DEROGATION"


class LifecycleScope(str, Enum):
    """Product carbon footprint lifecycle scope."""

    CRADLE_TO_GATE = "CRADLE_TO_GATE"
    CRADLE_TO_GRAVE = "CRADLE_TO_GRAVE"


class AllocationMethod(str, Enum):
    """Emission allocation method for multi-product lines."""

    MASS = "MASS"
    ECONOMIC = "ECONOMIC"
    PHYSICAL_CAUSALITY = "PHYSICAL_CAUSALITY"
    ENERGY = "ENERGY"


class WaterSourceType(str, Enum):
    """Water source classification for water tracking."""

    MUNICIPAL = "MUNICIPAL"
    GROUNDWATER = "GROUNDWATER"
    SURFACE_WATER = "SURFACE_WATER"
    RAINWATER = "RAINWATER"
    RECYCLED = "RECYCLED"
    SEAWATER = "SEAWATER"


class PollutantCategory(str, Enum):
    """Industrial pollutant category classification."""

    AIR_EMISSIONS = "AIR_EMISSIONS"
    WATER_DISCHARGES = "WATER_DISCHARGES"
    SOIL_RELEASES = "SOIL_RELEASES"
    WASTE_TRANSFERS = "WASTE_TRANSFERS"


class EnergySource(str, Enum):
    """Energy source type for energy intensity tracking."""

    ELECTRICITY = "ELECTRICITY"
    NATURAL_GAS = "NATURAL_GAS"
    COAL = "COAL"
    HEAVY_FUEL_OIL = "HEAVY_FUEL_OIL"
    DIESEL = "DIESEL"
    BIOMASS = "BIOMASS"
    HYDROGEN = "HYDROGEN"
    WASTE_HEAT = "WASTE_HEAT"
    SOLAR_THERMAL = "SOLAR_THERMAL"
    BIOGAS = "BIOGAS"
    LPG = "LPG"
    STEAM_PURCHASED = "STEAM_PURCHASED"


class ESRSTopic(str, Enum):
    """ESRS topical standards for materiality assessment."""

    E1 = "E1"  # Climate change
    E2 = "E2"  # Pollution
    E3 = "E3"  # Water and marine resources
    E4 = "E4"  # Biodiversity and ecosystems
    E5 = "E5"  # Resource use and circular economy
    S1 = "S1"  # Own workforce
    S2 = "S2"  # Workers in the value chain
    S3 = "S3"  # Affected communities
    S4 = "S4"  # Consumers and end-users
    G1 = "G1"  # Business conduct


class ReportingFrequency(str, Enum):
    """Reporting and disclosure frequency."""

    ANNUAL = "ANNUAL"
    SEMI_ANNUAL = "SEMI_ANNUAL"
    QUARTERLY = "QUARTERLY"


class ComplianceStatus(str, Enum):
    """Overall compliance status."""

    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"
    NOT_ASSESSED = "NOT_ASSESSED"


class DisclosureFormat(str, Enum):
    """Output format for disclosure documents."""

    PDF = "PDF"
    XLSX = "XLSX"
    HTML = "HTML"
    JSON = "JSON"
    XML = "XML"
    XBRL = "XBRL"


class WasteStreamType(str, Enum):
    """Waste stream classification."""

    HAZARDOUS = "HAZARDOUS"
    NON_HAZARDOUS = "NON_HAZARDOUS"
    BY_PRODUCT = "BY_PRODUCT"
    RECYCLABLE = "RECYCLABLE"
    ORGANIC = "ORGANIC"
    CONSTRUCTION = "CONSTRUCTION"


class SBTiPathway(str, Enum):
    """SBTi decarbonization pathway alignment."""

    WELL_BELOW_2C = "WELL_BELOW_2C"
    ONE_POINT_FIVE = "ONE_POINT_FIVE"
    NET_ZERO = "NET_ZERO"


class EPRScheme(str, Enum):
    """Extended Producer Responsibility scheme types."""

    PACKAGING = "PACKAGING"
    WEEE = "WEEE"
    BATTERIES = "BATTERIES"
    END_OF_LIFE_VEHICLES = "END_OF_LIFE_VEHICLES"
    TEXTILES = "TEXTILES"


# =============================================================================
# Reference Data Constants
# =============================================================================

# Manufacturing sub-sector display names and NACE codes
SUBSECTOR_INFO: Dict[str, Dict[str, str]] = {
    "CEMENT": {
        "name": "Cement & Lime",
        "nace": "C23.51",
        "bref": "CLM BREF",
        "eu_ets": "Yes",
        "cbam": "Yes",
    },
    "STEEL": {
        "name": "Iron & Steel",
        "nace": "C24.1",
        "bref": "IS BREF",
        "eu_ets": "Yes",
        "cbam": "Yes",
    },
    "ALUMINUM": {
        "name": "Aluminium",
        "nace": "C24.42",
        "bref": "NFM BREF",
        "eu_ets": "Yes",
        "cbam": "Yes",
    },
    "CHEMICALS": {
        "name": "Chemicals",
        "nace": "C20",
        "bref": "LVOC/LVIC/SIC BREF",
        "eu_ets": "Yes",
        "cbam": "Partial",
    },
    "GLASS": {
        "name": "Glass",
        "nace": "C23.1",
        "bref": "GLS BREF",
        "eu_ets": "Yes",
        "cbam": "No",
    },
    "CERAMICS": {
        "name": "Ceramics",
        "nace": "C23.3",
        "bref": "CER BREF",
        "eu_ets": "Yes",
        "cbam": "No",
    },
    "PULP_PAPER": {
        "name": "Pulp & Paper",
        "nace": "C17",
        "bref": "PP BREF",
        "eu_ets": "Yes",
        "cbam": "No",
    },
    "FOOD_BEVERAGE": {
        "name": "Food & Beverage",
        "nace": "C10-C11",
        "bref": "FDM BREF",
        "eu_ets": "Partial",
        "cbam": "No",
    },
    "TEXTILES": {
        "name": "Textiles",
        "nace": "C13-C14",
        "bref": "TXT BREF",
        "eu_ets": "No",
        "cbam": "No",
    },
    "ELECTRONICS": {
        "name": "Electronics",
        "nace": "C26",
        "bref": "N/A",
        "eu_ets": "No",
        "cbam": "No",
    },
    "AUTOMOTIVE": {
        "name": "Automotive",
        "nace": "C29",
        "bref": "STM BREF (surface treatment)",
        "eu_ets": "No",
        "cbam": "No",
    },
    "MACHINERY": {
        "name": "Machinery & Equipment",
        "nace": "C28",
        "bref": "STM BREF",
        "eu_ets": "No",
        "cbam": "No",
    },
    "PHARMACEUTICALS": {
        "name": "Pharmaceuticals",
        "nace": "C21",
        "bref": "OFC BREF",
        "eu_ets": "No",
        "cbam": "No",
    },
    "PLASTICS_RUBBER": {
        "name": "Plastics & Rubber",
        "nace": "C22",
        "bref": "POL BREF",
        "eu_ets": "Partial",
        "cbam": "No",
    },
    "FURNITURE": {
        "name": "Furniture",
        "nace": "C31",
        "bref": "N/A",
        "eu_ets": "No",
        "cbam": "No",
    },
    "PACKAGING": {
        "name": "Packaging",
        "nace": "C17/C22",
        "bref": "N/A",
        "eu_ets": "No",
        "cbam": "No",
    },
}

# Manufacturing tier descriptions
TIER_DESCRIPTIONS: Dict[str, str] = {
    "HEAVY_INDUSTRY": "Energy-intensive manufacturing with dominant process emissions (EU ETS covered)",
    "DISCRETE": "Product-centric manufacturing with supply chain-dominant emissions",
    "PROCESS": "Continuous process manufacturing with process + water/pollution focus",
    "LIGHT": "Consumer-oriented manufacturing with circular economy focus",
}

# Sub-sectors by tier
TIER_SUBSECTORS: Dict[str, List[str]] = {
    "HEAVY_INDUSTRY": ["CEMENT", "STEEL", "ALUMINUM", "GLASS", "CERAMICS"],
    "DISCRETE": ["AUTOMOTIVE", "ELECTRONICS", "MACHINERY"],
    "PROCESS": ["CHEMICALS", "FOOD_BEVERAGE", "TEXTILES", "PULP_PAPER"],
    "LIGHT": ["PHARMACEUTICALS", "PLASTICS_RUBBER", "FURNITURE", "PACKAGING"],
}

# Available presets
AVAILABLE_PRESETS: Dict[str, str] = {
    "heavy_industry": "Energy-intensive manufacturing (cement, steel, aluminium) with EU ETS and CBAM",
    "discrete_manufacturing": "Discrete product manufacturing (automotive, electronics) with PCF focus",
    "process_manufacturing": "Continuous process manufacturing (chemicals, food) with water/pollution",
    "light_manufacturing": "Light manufacturing (packaging, furniture) with circular economy focus",
    "multi_site": "Multi-facility manufacturing groups with consolidation",
    "sme_manufacturer": "Simplified SME manufacturer with Omnibus threshold assessment",
}

# Priority Scope 3 categories by manufacturing tier
PRIORITY_SCOPE3_BY_TIER: Dict[str, List[int]] = {
    "HEAVY_INDUSTRY": [1, 3, 4, 5, 9],
    "DISCRETE": [1, 2, 4, 9, 11, 12],
    "PROCESS": [1, 3, 4, 5, 10, 12],
    "LIGHT": [1, 4, 5, 9, 12],
}


# =============================================================================
# Pydantic Sub-Config Models
# =============================================================================


class FacilityConfig(BaseModel):
    """Configuration for a single manufacturing facility."""

    facility_id: str = Field(
        "",
        description="Unique identifier for the facility",
    )
    facility_name: str = Field(
        "",
        description="Human-readable facility name",
    )
    country: str = Field(
        "DE",
        description="ISO 3166-1 alpha-2 country code",
    )
    sub_sector: ManufacturingSubSector = Field(
        ManufacturingSubSector.STEEL,
        description="Manufacturing sub-sector of this facility",
    )
    eu_ets_installation_id: Optional[str] = Field(
        None,
        description="EU ETS installation identifier (if applicable)",
    )
    ied_permit_number: Optional[str] = Field(
        None,
        description="IED installation permit number (if applicable)",
    )
    production_capacity_tonnes: Optional[float] = Field(
        None,
        ge=0,
        description="Annual production capacity in tonnes",
    )
    employees: Optional[int] = Field(
        None,
        ge=0,
        description="Number of employees at the facility",
    )


class ProcessEmissionsConfig(BaseModel):
    """Configuration for process emissions engine.

    Handles direct process emissions from manufacturing operations per
    IED MRV methodology and EU ETS monitoring rules.
    """

    enabled: bool = Field(
        True,
        description="Enable process emissions calculation",
    )
    sub_sector: ManufacturingSubSector = Field(
        ManufacturingSubSector.STEEL,
        description="Primary manufacturing sub-sector",
    )
    process_lines: List[str] = Field(
        default_factory=list,
        description="List of process line identifiers to monitor",
    )
    cbam_affected: bool = Field(
        False,
        description="Whether the facility produces CBAM-affected goods",
    )
    ets_installation_id: Optional[str] = Field(
        None,
        description="EU ETS installation identifier for emissions reporting",
    )
    ets_phase: EUETSPhase = Field(
        EUETSPhase.PHASE_4,
        description="Current EU ETS phase for benchmark reference",
    )
    cbam_status: CBAMStatus = Field(
        CBAMStatus.NOT_AFFECTED,
        description="CBAM applicability status",
    )
    emission_sources: List[str] = Field(
        default_factory=lambda: [
            "combustion",
            "process",
            "fugitive",
        ],
        description="Types of emission sources to track",
    )
    mass_balance_enabled: bool = Field(
        True,
        description="Enable mass balance cross-check for process emissions",
    )
    product_allocation_enabled: bool = Field(
        True,
        description="Enable allocation of shared emissions to products",
    )
    allocation_method: AllocationMethod = Field(
        AllocationMethod.MASS,
        description="Default allocation method for shared process emissions",
    )
    emission_factor_source: str = Field(
        "EU_MRV",
        description="Primary emission factor source: EU_MRV, IPCC, NATIONAL",
    )
    year_over_year_tracking: bool = Field(
        True,
        description="Enable year-over-year process emissions trend analysis",
    )

    @field_validator("process_lines")
    @classmethod
    def validate_process_lines(cls, v: List[str]) -> List[str]:
        """Deduplicate process line identifiers."""
        return list(dict.fromkeys(v))


class EnergyIntensityConfig(BaseModel):
    """Configuration for energy intensity benchmarking engine.

    Calculates energy intensity per production unit and benchmarks against
    BAT-AEL and EU ETS product benchmark values.
    """

    enabled: bool = Field(
        True,
        description="Enable energy intensity calculation",
    )
    production_unit: str = Field(
        "tonne",
        description="Production unit for intensity metrics (tonne, unit, m2, m3)",
    )
    energy_sources: List[EnergySource] = Field(
        default_factory=lambda: [
            EnergySource.ELECTRICITY,
            EnergySource.NATURAL_GAS,
        ],
        description="Energy sources consumed at the facility",
    )
    benchmark_reference: str = Field(
        "BAT_AEL",
        description="Benchmark reference: BAT_AEL, EU_ETS_BENCHMARK, PEER_AVERAGE",
    )
    iso50001_certified: bool = Field(
        False,
        description="Whether the facility holds ISO 50001 certification",
    )
    enpi_tracking: bool = Field(
        True,
        description="Enable Energy Performance Indicator (EnPI) tracking",
    )
    significant_energy_use_threshold_pct: float = Field(
        5.0,
        ge=0.0,
        le=100.0,
        description="Threshold (%) to classify a Significant Energy Use (SEU)",
    )
    renewable_energy_tracking: bool = Field(
        True,
        description="Track renewable energy share and Guarantee of Origin",
    )
    energy_cost_tracking: bool = Field(
        True,
        description="Track energy costs alongside consumption",
    )
    baseline_year: Optional[int] = Field(
        None,
        ge=2015,
        le=2030,
        description="Energy baseline year for improvement tracking",
    )


class ProductPCFConfig(BaseModel):
    """Configuration for Product Carbon Footprint engine.

    Calculates product-level carbon footprints per ISO 14067, GHG Protocol
    Product Standard, and EU PEF methodology.
    """

    enabled: bool = Field(
        True,
        description="Enable product carbon footprint calculation",
    )
    lifecycle_scope: LifecycleScope = Field(
        LifecycleScope.CRADLE_TO_GATE,
        description="Lifecycle scope: CRADLE_TO_GATE or CRADLE_TO_GRAVE",
    )
    allocation_method: AllocationMethod = Field(
        AllocationMethod.MASS,
        description="Default allocation method for shared process emissions",
    )
    dpp_enabled: bool = Field(
        False,
        description="Enable Digital Product Passport (DPP) data generation per ESPR",
    )
    pef_methodology: bool = Field(
        True,
        description="Use EU Product Environmental Footprint (PEF) methodology",
    )
    iso14067_compliant: bool = Field(
        True,
        description="Ensure ISO 14067 compliance in PCF calculations",
    )
    emission_factor_database: str = Field(
        "ECOINVENT",
        description="LCA emission factor database: ECOINVENT, GABI, ELCD",
    )
    co_product_allocation: bool = Field(
        True,
        description="Enable co-product allocation for multi-output processes",
    )
    recycled_content_crediting: bool = Field(
        True,
        description="Apply recycled content credits in PCF calculations",
    )
    functional_unit: str = Field(
        "1 kg product",
        description="Default functional unit for PCF declaration",
    )
    epd_output: bool = Field(
        False,
        description="Generate Environmental Product Declaration (EPD) compatible output",
    )
    product_families: List[str] = Field(
        default_factory=list,
        description="Product families to calculate PCF for",
    )
    uncertainty_analysis: bool = Field(
        True,
        description="Include uncertainty analysis in PCF results",
    )


class CircularEconomyConfig(BaseModel):
    """Configuration for circular economy metrics engine.

    Tracks circular economy metrics per ESRS E5 and EU Taxonomy criteria.
    """

    enabled: bool = Field(
        True,
        description="Enable circular economy metrics tracking",
    )
    waste_streams: List[WasteStreamType] = Field(
        default_factory=lambda: [
            WasteStreamType.HAZARDOUS,
            WasteStreamType.NON_HAZARDOUS,
            WasteStreamType.RECYCLABLE,
        ],
        description="Waste stream types to track",
    )
    recycled_content_tracking: bool = Field(
        True,
        description="Track recycled content ratios by product",
    )
    epr_schemes: List[EPRScheme] = Field(
        default_factory=list,
        description="Extended Producer Responsibility schemes applicable",
    )
    material_circularity_indicator: bool = Field(
        True,
        description="Calculate Material Circularity Indicator (MCI)",
    )
    waste_diversion_target_pct: float = Field(
        80.0,
        ge=0.0,
        le=100.0,
        description="Target waste diversion rate (%)",
    )
    industrial_symbiosis_tracking: bool = Field(
        False,
        description="Track industrial symbiosis (by-product exchanges)",
    )
    espr_recycled_content_mandates: bool = Field(
        False,
        description="Track compliance with ESPR recycled content mandates",
    )
    closed_loop_tracking: bool = Field(
        False,
        description="Track closed-loop material flows",
    )
    packaging_waste_directive: bool = Field(
        False,
        description="Enable EU Packaging and Packaging Waste Directive tracking",
    )


class WaterPollutionConfig(BaseModel):
    """Configuration for water and pollution tracking engine.

    Covers ESRS E2 (pollution) and ESRS E3 (water) requirements.
    """

    enabled: bool = Field(
        True,
        description="Enable water and pollution tracking",
    )
    water_sources: List[WaterSourceType] = Field(
        default_factory=lambda: [
            WaterSourceType.MUNICIPAL,
            WaterSourceType.GROUNDWATER,
        ],
        description="Water sources used at the facility",
    )
    pollutant_types: List[PollutantCategory] = Field(
        default_factory=lambda: [
            PollutantCategory.AIR_EMISSIONS,
            PollutantCategory.WATER_DISCHARGES,
        ],
        description="Pollutant categories to track",
    )
    water_stress_assessment: bool = Field(
        True,
        description="Enable WRI Aqueduct water stress assessment for facility locations",
    )
    reach_svhc_tracking: bool = Field(
        False,
        description="Track REACH Substances of Very High Concern (SVHCs)",
    )
    eprtr_reporting: bool = Field(
        False,
        description="Enable E-PRTR pollutant release reporting",
    )
    water_framework_directive: bool = Field(
        False,
        description="Assess compliance with EU Water Framework Directive",
    )
    wastewater_treatment_monitoring: bool = Field(
        True,
        description="Monitor wastewater treatment efficiency",
    )
    water_recycling_rate_tracking: bool = Field(
        True,
        description="Track water recycling and reuse rates",
    )
    clp_hazard_classification: bool = Field(
        False,
        description="Track CLP hazard classification for chemical substances",
    )
    water_intensity_metric: str = Field(
        "m3_per_tonne",
        description="Water intensity metric: m3_per_tonne, m3_per_unit, m3_per_revenue",
    )


class BATComplianceConfig(BaseModel):
    """Configuration for BAT/BREF compliance assessment engine.

    Assesses compliance against Best Available Techniques Reference
    Documents (BREFs) per Industrial Emissions Directive 2010/75/EU.
    """

    enabled: bool = Field(
        True,
        description="Enable BAT/BREF compliance assessment",
    )
    applicable_brefs: List[str] = Field(
        default_factory=list,
        description="List of applicable BREF document identifiers (e.g., IS_BREF, CLM_BREF)",
    )
    compliance_level: BATComplianceLevel = Field(
        BATComplianceLevel.COMPLIANT,
        description="Current BAT compliance level assessment",
    )
    compliance_deadline: Optional[date] = Field(
        None,
        description="Deadline for achieving full BAT compliance",
    )
    transformation_plan: bool = Field(
        False,
        description="Generate transformation plan for non-compliant parameters",
    )
    bat_ael_monitoring: bool = Field(
        True,
        description="Enable continuous monitoring against BAT-AEL values",
    )
    derogation_tracking: bool = Field(
        False,
        description="Track permit derogations and their expiry dates",
    )
    permit_conditions_monitoring: bool = Field(
        True,
        description="Monitor IED permit conditions and reporting deadlines",
    )
    ied_inspection_readiness: bool = Field(
        True,
        description="Maintain readiness for IED regulatory inspections",
    )

    @field_validator("applicable_brefs")
    @classmethod
    def validate_brefs(cls, v: List[str]) -> List[str]:
        """Normalize BREF identifiers to uppercase."""
        return [bref.upper().strip() for bref in v]


class SupplyChainConfig(BaseModel):
    """Configuration for supply chain emissions engine.

    Maps and calculates Scope 3 supply chain emissions.
    """

    enabled: bool = Field(
        True,
        description="Enable supply chain emissions calculation",
    )
    tier_depth: int = Field(
        2,
        ge=1,
        le=5,
        description="Depth of supply chain tiers to assess (1=Tier 1 only, up to 5)",
    )
    priority_categories: List[int] = Field(
        default_factory=lambda: [1, 2, 3, 4, 5, 9, 10, 11, 12],
        description="Priority Scope 3 categories to calculate (1-15)",
    )
    supplier_engagement_platform: str = Field(
        "CDP_SUPPLY_CHAIN",
        description="Supplier engagement platform: CDP_SUPPLY_CHAIN, ECOVADIS, CUSTOM",
    )
    spend_based_screening: bool = Field(
        True,
        description="Use spend-based screening for uncovered categories",
    )
    supplier_specific_enabled: bool = Field(
        True,
        description="Use supplier-specific emission factors where available",
    )
    hybrid_method: bool = Field(
        True,
        description="Use hybrid method (supplier-specific + spend-based + average)",
    )
    hotspot_analysis: bool = Field(
        True,
        description="Identify emission hotspots by category, supplier, and material",
    )
    engagement_scoring: bool = Field(
        True,
        description="Score suppliers on emissions data quality and reduction targets",
    )
    cbam_import_integration: bool = Field(
        False,
        description="Integrate CBAM embedded emissions for imported raw materials",
    )
    transport_mode_tracking: bool = Field(
        True,
        description="Track transport mode split for Cat 4 and Cat 9",
    )

    @field_validator("priority_categories")
    @classmethod
    def validate_categories(cls, v: List[int]) -> List[int]:
        """Validate Scope 3 category numbers."""
        invalid = [c for c in v if c < 1 or c > 15]
        if invalid:
            raise ValueError(
                f"Invalid Scope 3 categories: {invalid}. Must be 1-15."
            )
        return sorted(set(v))


class BenchmarkConfig(BaseModel):
    """Configuration for manufacturing benchmark engine.

    Benchmarks against EU BAT-AEL, EU ETS benchmarks, peers, and SBTi.
    """

    enabled: bool = Field(
        True,
        description="Enable manufacturing benchmarking",
    )
    peer_group: str = Field(
        "NACE_SECTOR",
        description="Peer group definition: NACE_SECTOR, PRODUCTION_VOLUME, GEOGRAPHY",
    )
    kpi_set: List[str] = Field(
        default_factory=lambda: [
            "carbon_intensity",
            "energy_intensity",
            "water_intensity",
            "waste_intensity",
            "circularity_rate",
        ],
        description="KPI set for benchmarking",
    )
    sbti_pathway: SBTiPathway = Field(
        SBTiPathway.WELL_BELOW_2C,
        description="SBTi decarbonization pathway for alignment tracking",
    )
    abatement_cost_curve: bool = Field(
        True,
        description="Generate marginal abatement cost curve (MACC)",
    )
    eu_ets_benchmark_comparison: bool = Field(
        True,
        description="Compare against EU ETS product benchmark values",
    )
    bat_ael_benchmark_comparison: bool = Field(
        True,
        description="Compare against BAT-AEL emission levels",
    )
    decarbonization_technologies: List[str] = Field(
        default_factory=lambda: [
            "electrification",
            "hydrogen",
            "ccus",
            "biomass",
            "energy_efficiency",
            "waste_heat_recovery",
        ],
        description="Decarbonization technologies to screen in MACC",
    )
    target_years: List[int] = Field(
        default_factory=lambda: [2025, 2030, 2035, 2040, 2050],
        description="Milestone years for decarbonization roadmap",
    )
    gap_analysis_enabled: bool = Field(
        True,
        description="Generate performance gap analysis vs. benchmarks",
    )


class AuditTrailConfig(BaseModel):
    """Configuration for audit trail and provenance tracking."""

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
        description="Track all assumptions used in calculations",
    )
    data_lineage_enabled: bool = Field(
        True,
        description="Track full data lineage from source to output",
    )
    retention_years: int = Field(
        10,
        ge=1,
        le=15,
        description="Audit trail retention period in years",
    )
    external_audit_export: bool = Field(
        True,
        description="Enable export format for external auditors",
    )


class DisclosureConfig(BaseModel):
    """Configuration for disclosure document generation."""

    esrs_chapter_enabled: bool = Field(
        True,
        description="Generate manufacturing ESRS chapter for management report",
    )
    process_emissions_report_enabled: bool = Field(
        True,
        description="Generate process emissions report",
    )
    pcf_label_enabled: bool = Field(
        True,
        description="Generate product carbon footprint labels",
    )
    energy_report_enabled: bool = Field(
        True,
        description="Generate energy performance report",
    )
    circular_economy_report_enabled: bool = Field(
        True,
        description="Generate circular economy report",
    )
    bat_report_enabled: bool = Field(
        True,
        description="Generate BAT compliance report",
    )
    scorecard_enabled: bool = Field(
        True,
        description="Generate manufacturing sustainability scorecard",
    )
    decarbonization_roadmap_enabled: bool = Field(
        True,
        description="Generate decarbonization roadmap",
    )
    output_formats: List[DisclosureFormat] = Field(
        default_factory=lambda: [DisclosureFormat.PDF, DisclosureFormat.XLSX],
        description="Output formats for disclosure documents",
    )
    multi_language_support: bool = Field(
        False,
        description="Enable multi-language disclosure generation",
    )
    supported_languages: List[str] = Field(
        default_factory=lambda: ["en"],
        description="Supported languages for disclosures",
    )
    review_workflow_enabled: bool = Field(
        True,
        description="Enable review and approval workflow",
    )
    watermark_draft: bool = Field(
        True,
        description="Apply DRAFT watermark to unapproved documents",
    )


class OmnibusConfig(BaseModel):
    """Configuration for CSRD Omnibus Directive threshold assessment."""

    enabled: bool = Field(
        True,
        description="Enable Omnibus Directive threshold assessment",
    )
    total_assets_eur: Optional[float] = Field(
        None,
        ge=0,
        description="Total assets in EUR for threshold assessment",
    )
    net_turnover_eur: Optional[float] = Field(
        None,
        ge=0,
        description="Net turnover in EUR for threshold assessment",
    )
    average_employees: Optional[int] = Field(
        None,
        ge=0,
        description="Average number of employees for threshold assessment",
    )
    listed_entity: bool = Field(
        False,
        description="Whether the entity is listed on a regulated market",
    )


# =============================================================================
# Main Configuration Model
# =============================================================================


class CSRDManufacturingConfig(BaseModel):
    """Main configuration for PACK-013 CSRD Manufacturing Pack.

    This is the root configuration model that contains all sub-configurations
    for manufacturing sector CSRD compliance. The manufacturing_tier field
    drives which engines are prioritized and which regulatory requirements
    are most critical.
    """

    # Company identification
    company_name: str = Field(
        "",
        description="Legal entity name of the manufacturing company",
    )
    reporting_year: int = Field(
        2025,
        ge=2024,
        le=2035,
        description="Reporting year for CSRD disclosure",
    )
    manufacturing_tier: ManufacturingTier = Field(
        ManufacturingTier.HEAVY_INDUSTRY,
        description="Manufacturing tier (drives engine prioritization)",
    )
    sub_sectors: List[ManufacturingSubSector] = Field(
        default_factory=lambda: [ManufacturingSubSector.STEEL],
        description="Manufacturing sub-sectors of the company",
    )

    # Facilities
    facilities: List[FacilityConfig] = Field(
        default_factory=list,
        description="List of manufacturing facility configurations",
    )

    # Omnibus threshold
    omnibus_threshold: OmnibusConfig = Field(
        default_factory=OmnibusConfig,
        description="CSRD Omnibus Directive threshold assessment",
    )

    # Sub-configurations for each engine
    process_emissions: ProcessEmissionsConfig = Field(
        default_factory=ProcessEmissionsConfig,
        description="Process emissions engine configuration",
    )
    energy_intensity: EnergyIntensityConfig = Field(
        default_factory=EnergyIntensityConfig,
        description="Energy intensity engine configuration",
    )
    product_pcf: ProductPCFConfig = Field(
        default_factory=ProductPCFConfig,
        description="Product carbon footprint engine configuration",
    )
    circular_economy: CircularEconomyConfig = Field(
        default_factory=CircularEconomyConfig,
        description="Circular economy engine configuration",
    )
    water_pollution: WaterPollutionConfig = Field(
        default_factory=WaterPollutionConfig,
        description="Water and pollution engine configuration",
    )
    bat_compliance: BATComplianceConfig = Field(
        default_factory=BATComplianceConfig,
        description="BAT/BREF compliance engine configuration",
    )
    supply_chain: SupplyChainConfig = Field(
        default_factory=SupplyChainConfig,
        description="Supply chain emissions engine configuration",
    )
    benchmark: BenchmarkConfig = Field(
        default_factory=BenchmarkConfig,
        description="Manufacturing benchmark engine configuration",
    )

    # Supporting configurations
    audit_trail: AuditTrailConfig = Field(
        default_factory=AuditTrailConfig,
        description="Audit trail and provenance configuration",
    )
    disclosure: DisclosureConfig = Field(
        default_factory=DisclosureConfig,
        description="Disclosure document generation configuration",
    )

    @model_validator(mode="after")
    def validate_heavy_industry_requires_ets(self) -> "CSRDManufacturingConfig":
        """Ensure heavy industry tier has EU ETS and BAT compliance enabled."""
        if self.manufacturing_tier == ManufacturingTier.HEAVY_INDUSTRY:
            if not self.process_emissions.enabled:
                logger.warning(
                    "Process emissions calculation is critical for heavy industry. "
                    "Enabling process_emissions."
                )
                object.__setattr__(self.process_emissions, "enabled", True)
            if not self.bat_compliance.enabled:
                logger.warning(
                    "BAT/BREF compliance is mandatory for IED-permitted heavy "
                    "industry installations. Enabling bat_compliance."
                )
                object.__setattr__(self.bat_compliance, "enabled", True)
        return self

    @model_validator(mode="after")
    def validate_cbam_consistency(self) -> "CSRDManufacturingConfig":
        """Ensure CBAM status consistency with sub-sector."""
        cbam_sectors = {"CEMENT", "STEEL", "ALUMINUM", "CHEMICALS"}
        has_cbam_sector = any(
            s.value in cbam_sectors for s in self.sub_sectors
        )
        if has_cbam_sector and self.process_emissions.cbam_status == CBAMStatus.NOT_AFFECTED:
            logger.info(
                "Sub-sectors include CBAM-affected goods. Consider setting "
                "process_emissions.cbam_status to TRANSITIONAL or FULL_COMPLIANCE."
            )
        return self

    @model_validator(mode="after")
    def validate_water_pollution_for_process(self) -> "CSRDManufacturingConfig":
        """Ensure water/pollution tracking for process manufacturing."""
        if self.manufacturing_tier == ManufacturingTier.PROCESS:
            if not self.water_pollution.enabled:
                logger.warning(
                    "Water and pollution tracking is critical for process "
                    "manufacturing (ESRS E2/E3). Enabling water_pollution."
                )
                object.__setattr__(self.water_pollution, "enabled", True)
        return self

    @model_validator(mode="after")
    def validate_circular_economy_for_light(self) -> "CSRDManufacturingConfig":
        """Ensure circular economy is prioritized for light manufacturing."""
        if self.manufacturing_tier == ManufacturingTier.LIGHT:
            if not self.circular_economy.enabled:
                logger.warning(
                    "Circular economy metrics are the primary focus for light "
                    "manufacturing. Enabling circular_economy."
                )
                object.__setattr__(self.circular_economy, "enabled", True)
        return self


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """Top-level pack configuration wrapper.

    Handles preset loading, environment variable overrides, and
    configuration merging.
    """

    pack: CSRDManufacturingConfig = Field(
        default_factory=CSRDManufacturingConfig,
        description="Main CSRD Manufacturing configuration",
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
        "PACK-013-csrd-manufacturing",
        description="Pack identifier",
    )

    @classmethod
    def from_preset(cls, preset_name: str, overrides: Optional[Dict[str, Any]] = None) -> "PackConfig":
        """Load configuration from a named preset.

        Args:
            preset_name: Name of the preset (heavy_industry, discrete_manufacturing, etc.)
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

        pack_config = CSRDManufacturingConfig(**preset_data)
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

        pack_config = CSRDManufacturingConfig(**config_data)
        return cls(pack=pack_config)

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """Load configuration overrides from environment variables.

        Environment variables prefixed with CSRD_MFG_PACK_ are loaded and
        mapped to configuration keys. Nested keys use double underscore.

        Example: CSRD_MFG_PACK_PROCESS_EMISSIONS__CBAM_AFFECTED=true
        """
        overrides: Dict[str, Any] = {}
        prefix = "CSRD_MFG_PACK_"
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
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
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


def load_preset(preset_name: str, overrides: Optional[Dict[str, Any]] = None) -> PackConfig:
    """Load a named preset configuration.

    Convenience wrapper around PackConfig.from_preset().

    Args:
        preset_name: Name of the preset to load.
        overrides: Optional configuration overrides.

    Returns:
        PackConfig instance with preset applied.
    """
    return PackConfig.from_preset(preset_name, overrides)


def validate_config(config: CSRDManufacturingConfig) -> List[str]:
    """Validate a manufacturing configuration and return any warnings.

    Args:
        config: CSRDManufacturingConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check for missing facility data
    if not config.facilities:
        warnings.append(
            "No facilities configured. Add at least one facility for meaningful results."
        )

    # Check sub-sector alignment with tier
    expected_subsectors = TIER_SUBSECTORS.get(config.manufacturing_tier.value, [])
    for sub in config.sub_sectors:
        if sub.value not in expected_subsectors:
            warnings.append(
                f"Sub-sector {sub.value} is not typical for tier "
                f"{config.manufacturing_tier.value}. Consider adjusting."
            )

    # Check BAT compliance for IED-applicable sectors
    ied_sectors = {"CEMENT", "STEEL", "ALUMINUM", "CHEMICALS", "GLASS",
                   "CERAMICS", "PULP_PAPER", "FOOD_BEVERAGE", "TEXTILES"}
    has_ied_sector = any(s.value in ied_sectors for s in config.sub_sectors)
    if has_ied_sector and not config.bat_compliance.enabled:
        warnings.append(
            "BAT/BREF compliance assessment is recommended for IED-applicable "
            "manufacturing sub-sectors."
        )

    # Check EU ETS relevance
    ets_sectors = {"CEMENT", "STEEL", "ALUMINUM", "CHEMICALS", "GLASS",
                   "CERAMICS", "PULP_PAPER"}
    has_ets_sector = any(s.value in ets_sectors for s in config.sub_sectors)
    if has_ets_sector and not config.process_emissions.ets_installation_id:
        warnings.append(
            "EU ETS installation ID not set for an ETS-covered sub-sector. "
            "Set process_emissions.ets_installation_id for EU ETS reporting."
        )

    # Check Scope 3 coverage
    if config.supply_chain.enabled and len(config.supply_chain.priority_categories) < 3:
        warnings.append(
            "Fewer than 3 Scope 3 categories configured. Manufacturing companies "
            "typically need at least categories 1, 3, 4, 5, and 9."
        )

    return warnings


def get_default_config(tier: ManufacturingTier = ManufacturingTier.HEAVY_INDUSTRY) -> CSRDManufacturingConfig:
    """Get default configuration for a given manufacturing tier.

    Args:
        tier: Manufacturing tier to configure for.

    Returns:
        CSRDManufacturingConfig instance with tier-appropriate defaults.
    """
    default_subsectors = {
        ManufacturingTier.HEAVY_INDUSTRY: [ManufacturingSubSector.STEEL],
        ManufacturingTier.DISCRETE: [ManufacturingSubSector.AUTOMOTIVE],
        ManufacturingTier.PROCESS: [ManufacturingSubSector.CHEMICALS],
        ManufacturingTier.LIGHT: [ManufacturingSubSector.PACKAGING],
    }

    return CSRDManufacturingConfig(
        manufacturing_tier=tier,
        sub_sectors=default_subsectors.get(tier, [ManufacturingSubSector.STEEL]),
    )


def get_subsector_info(sub_sector: Union[str, ManufacturingSubSector]) -> Dict[str, str]:
    """Get detailed information about a manufacturing sub-sector.

    Args:
        sub_sector: Sub-sector enum or string value.

    Returns:
        Dictionary with name, NACE code, BREF, EU ETS, and CBAM status.
    """
    key = sub_sector.value if isinstance(sub_sector, ManufacturingSubSector) else sub_sector
    return SUBSECTOR_INFO.get(key, {"name": key, "nace": "N/A", "bref": "N/A", "eu_ets": "N/A", "cbam": "N/A"})


def get_tier_description(tier: Union[str, ManufacturingTier]) -> str:
    """Get description for a manufacturing tier.

    Args:
        tier: Manufacturing tier enum or string value.

    Returns:
        Human-readable description of the manufacturing tier.
    """
    key = tier.value if isinstance(tier, ManufacturingTier) else tier
    return TIER_DESCRIPTIONS.get(key, f"Unknown tier ({key})")


def list_available_presets() -> Dict[str, str]:
    """List all available configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return AVAILABLE_PRESETS.copy()
