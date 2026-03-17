"""
PACK-012 CSRD Financial Service Pack - Configuration Manager

This module implements the CSRDFinancialServiceConfig and PackConfig classes that
load, merge, and validate all configuration for the CSRD Financial Service Pack.
It provides comprehensive Pydantic v2 models for every aspect of financial
institution CSRD compliance: financed emissions (PCAF), Green Asset Ratio (GAR),
Banking Book Taxonomy Alignment Ratio (BTAR), insurance underwriting emissions,
climate risk scoring, financial sector double materiality assessment, transition
plan generation, and EBA Pillar 3 ESG disclosures.

Financial Institution Types:
    - BANK: Credit institutions subject to CRR/CRD VI (GAR, BTAR, Pillar 3)
    - INSURANCE: Insurance/reinsurance undertakings (Solvency II, underwriting)
    - ASSET_MANAGER: UCITS management companies and AIFM (SFDR, WACI)
    - INVESTMENT_FIRM: MiFID II investment firms (product governance, ESG prefs)
    - PENSION_FUND: IORP II pension schemes (stewardship, long-horizon)
    - CONGLOMERATE: Financial conglomerates (multi-entity, cross-sector)

PCAF Asset Classes (6):
    - Listed equity and corporate bonds
    - Business loans and unlisted equity
    - Project finance
    - Commercial real estate
    - Mortgages
    - Motor vehicle loans

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (bank / insurance / asset_manager / investment_firm /
       pension_fund / conglomerate)
    3. Environment overrides (CSRD_FS_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - CSRD: Directive (EU) 2022/2464
    - ESRS: Delegated Regulation (EU) 2023/2772
    - EU Taxonomy: Regulation (EU) 2020/852
    - CRR/CRD VI: Article 449a (Pillar 3 ESG)
    - SFDR: Regulation (EU) 2019/2088
    - Solvency II: Directive 2009/138/EC
    - PCAF: Global GHG Accounting Standard v3
    - SBTi FI: Financial Institutions Framework v1.1
    - EBA ITS: Pillar 3 ESG Disclosures (2022/01, 2024 update)

Example:
    >>> config = PackConfig.from_preset("bank")
    >>> print(config.pack.institution_type)
    FinancialInstitutionType.BANK
    >>> print(config.pack.pcaf.enabled)
    True
    >>> print(config.pack.gar_btar.gar_enabled)
    True
    >>> print(config.pack.pillar3.enabled)
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
# Enums - Financial services-specific enumeration types
# =============================================================================


class FinancialInstitutionType(str, Enum):
    """Financial institution type classification."""

    BANK = "BANK"  # Credit institution subject to CRR/CRD VI
    INSURANCE = "INSURANCE"  # Insurance or reinsurance undertaking (Solvency II)
    ASSET_MANAGER = "ASSET_MANAGER"  # UCITS management company or AIFM
    INVESTMENT_FIRM = "INVESTMENT_FIRM"  # MiFID II investment firm
    PENSION_FUND = "PENSION_FUND"  # IORP II pension scheme
    CONGLOMERATE = "CONGLOMERATE"  # Financial conglomerate (multi-entity)


class PCAFAssetClass(str, Enum):
    """PCAF asset class categories per Global Standard v3."""

    LISTED_EQUITY_CORPORATE_BONDS = "LISTED_EQUITY_CORPORATE_BONDS"
    BUSINESS_LOANS_UNLISTED_EQUITY = "BUSINESS_LOANS_UNLISTED_EQUITY"
    PROJECT_FINANCE = "PROJECT_FINANCE"
    COMMERCIAL_REAL_ESTATE = "COMMERCIAL_REAL_ESTATE"
    MORTGAGES = "MORTGAGES"
    MOTOR_VEHICLE_LOANS = "MOTOR_VEHICLE_LOANS"
    SOVEREIGN_BONDS = "SOVEREIGN_BONDS"  # Extension
    PRIVATE_EQUITY = "PRIVATE_EQUITY"  # Extension
    GREEN_BONDS = "GREEN_BONDS"  # Extension for taxonomy alignment tracking
    SECURITIZED_PRODUCTS = "SECURITIZED_PRODUCTS"  # Extension


class PCAFDataQuality(int, Enum):
    """PCAF data quality score (1 = highest, 5 = lowest)."""

    SCORE_1 = 1  # Reported, verified emissions from counterparty
    SCORE_2 = 2  # Reported, unverified emissions from counterparty
    SCORE_3 = 3  # Estimated using physical activity data
    SCORE_4 = 4  # Estimated using economic activity data (revenue-based)
    SCORE_5 = 5  # Estimated using sector-average or asset-class proxies


class GARScope(str, Enum):
    """Green Asset Ratio KPI scope per EBA ITS."""

    TURNOVER = "TURNOVER"  # Turnover-based GAR
    CAPEX = "CAPEX"  # CapEx-based GAR
    OPEX = "OPEX"  # OpEx-based GAR


class NGFSScenario(str, Enum):
    """NGFS climate scenario types."""

    NET_ZERO_2050 = "NET_ZERO_2050"  # Orderly: 1.5C-aligned
    BELOW_2C = "BELOW_2C"  # Orderly: <2C with moderate action
    DELAYED_TRANSITION = "DELAYED_TRANSITION"  # Disorderly: late, sudden action
    NDCS = "NDCS"  # Hot house: current NDC pledges only
    DIVERGENT_NET_ZERO = "DIVERGENT_NET_ZERO"  # Disorderly: divergent policies
    CURRENT_POLICIES = "CURRENT_POLICIES"  # Hot house: no additional policy


class Pillar3Template(str, Enum):
    """EBA Pillar 3 ESG disclosure templates per ITS."""

    TEMPLATE_1 = "TEMPLATE_1"  # Qualitative information on ESG risks
    TEMPLATE_2 = "TEMPLATE_2"  # Scope 3 financed emissions by NACE sector
    TEMPLATE_3 = "TEMPLATE_3"  # Exposures to top carbon-intensive companies
    TEMPLATE_4 = "TEMPLATE_4"  # Exposures by physical risk (acute/chronic)
    TEMPLATE_5 = "TEMPLATE_5"  # Mitigating actions (green lending, ESG bonds)
    TEMPLATE_7 = "TEMPLATE_7"  # GAR: overview of KPIs
    TEMPLATE_8 = "TEMPLATE_8"  # GAR: by environmental objective
    TEMPLATE_9 = "TEMPLATE_9"  # GAR: by counterparty type
    TEMPLATE_10 = "TEMPLATE_10"  # GAR: flow vs. stock + BTAR


class ClimateRiskType(str, Enum):
    """Climate risk dimension categories."""

    PHYSICAL = "PHYSICAL"  # Physical climate risk (acute + chronic)
    TRANSITION = "TRANSITION"  # Transition risk (policy, tech, market, rep)


class PhysicalHazardType(str, Enum):
    """Physical climate hazard types for risk assessment."""

    FLOOD = "FLOOD"  # River and coastal flooding
    WILDFIRE = "WILDFIRE"  # Forest and bush fire risk
    STORM = "STORM"  # Windstorm, cyclone, hurricane
    HEATWAVE = "HEATWAVE"  # Extreme heat events
    SEA_LEVEL_RISE = "SEA_LEVEL_RISE"  # Chronic sea level rise
    DROUGHT = "DROUGHT"  # Chronic water stress and drought


class TransitionRiskChannel(str, Enum):
    """Transition risk transmission channels."""

    POLICY_LEGAL = "POLICY_LEGAL"  # Carbon pricing, regulation, litigation
    TECHNOLOGY = "TECHNOLOGY"  # Technology disruption, substitution
    MARKET = "MARKET"  # Consumer preference shifts, repricing
    REPUTATION = "REPUTATION"  # Stigmatization, stakeholder pressure
    STRANDED_ASSETS = "STRANDED_ASSETS"  # Asset write-downs, devaluation


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

    ANNUAL = "ANNUAL"  # Annual reporting
    SEMI_ANNUAL = "SEMI_ANNUAL"  # Semi-annual (Pillar 3)
    QUARTERLY = "QUARTERLY"  # Quarterly updates


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
    XBRL = "XBRL"  # EBA Pillar 3 XBRL format


class MaterialityLevel(str, Enum):
    """Double materiality assessment result levels."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NOT_MATERIAL = "NOT_MATERIAL"


class SBTiFIMethod(str, Enum):
    """SBTi Financial Institutions target-setting methods."""

    SDA = "SDA"  # Sectoral Decarbonization Approach
    TEMPERATURE_RATING = "TEMPERATURE_RATING"  # Temperature Rating
    PORTFOLIO_COVERAGE = "PORTFOLIO_COVERAGE"  # Portfolio Coverage


class InsuranceLineType(str, Enum):
    """Insurance business line types for underwriting emissions."""

    COMMERCIAL_PROPERTY = "COMMERCIAL_PROPERTY"
    COMMERCIAL_CASUALTY = "COMMERCIAL_CASUALTY"
    PERSONAL_PROPERTY = "PERSONAL_PROPERTY"
    PERSONAL_AUTO = "PERSONAL_AUTO"
    SPECIALTY = "SPECIALTY"
    LIFE_HEALTH = "LIFE_HEALTH"
    REINSURANCE = "REINSURANCE"


class GARExposureCategory(str, Enum):
    """GAR exposure categories per EBA ITS."""

    FINANCIAL_CORPORATES = "FINANCIAL_CORPORATES"
    NON_FINANCIAL_CORPORATES = "NON_FINANCIAL_CORPORATES"
    HOUSEHOLDS_MORTGAGES = "HOUSEHOLDS_MORTGAGES"
    HOUSEHOLDS_MOTOR_VEHICLE = "HOUSEHOLDS_MOTOR_VEHICLE"
    LOCAL_GOVERNMENTS = "LOCAL_GOVERNMENTS"
    COLLATERAL_OBTAINED = "COLLATERAL_OBTAINED"


# =============================================================================
# Reference Data Constants
# =============================================================================

# PCAF asset class display names and descriptions
PCAF_ASSET_CLASS_INFO: Dict[str, Dict[str, str]] = {
    "LISTED_EQUITY_CORPORATE_BONDS": {
        "name": "Listed Equity & Corporate Bonds",
        "attribution": "Enterprise Value Including Cash (EVIC)",
        "description": "Publicly traded equity and corporate bonds",
    },
    "BUSINESS_LOANS_UNLISTED_EQUITY": {
        "name": "Business Loans & Unlisted Equity",
        "attribution": "Total equity + debt, or Total assets for unlisted",
        "description": "Private lending and unlisted equity holdings",
    },
    "PROJECT_FINANCE": {
        "name": "Project Finance",
        "attribution": "Project equity + debt at commitment",
        "description": "Dedicated project financing arrangements",
    },
    "COMMERCIAL_REAL_ESTATE": {
        "name": "Commercial Real Estate",
        "attribution": "Property value at origination",
        "description": "Commercial property loans and direct ownership",
    },
    "MORTGAGES": {
        "name": "Mortgages",
        "attribution": "Property value at origination",
        "description": "Residential mortgage lending",
    },
    "MOTOR_VEHICLE_LOANS": {
        "name": "Motor Vehicle Loans",
        "attribution": "Vehicle value at origination",
        "description": "Auto loans and vehicle leasing",
    },
}

# Financial institution type display names
INSTITUTION_DISPLAY_NAMES: Dict[str, str] = {
    "BANK": "Credit Institution (CRR-regulated Bank)",
    "INSURANCE": "Insurance / Reinsurance Undertaking (Solvency II)",
    "ASSET_MANAGER": "Asset Management Company (UCITS/AIFM)",
    "INVESTMENT_FIRM": "Investment Firm (MiFID II)",
    "PENSION_FUND": "Pension Fund (IORP II)",
    "CONGLOMERATE": "Financial Conglomerate (Multi-sector)",
}

# NGFS scenario descriptions
NGFS_SCENARIO_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "NET_ZERO_2050": {
        "name": "Net Zero 2050",
        "category": "Orderly",
        "temperature": "1.5C",
        "description": "Immediate, ambitious action to limit warming to 1.5C",
    },
    "BELOW_2C": {
        "name": "Below 2C",
        "category": "Orderly",
        "temperature": "<2C",
        "description": "Gradual tightening of policies to keep below 2C",
    },
    "DELAYED_TRANSITION": {
        "name": "Delayed Transition",
        "category": "Disorderly",
        "temperature": "<2C",
        "description": "Action delayed until 2030, then rapid, disruptive",
    },
    "NDCS": {
        "name": "Nationally Determined Contributions",
        "category": "Hot House",
        "temperature": ">2.5C",
        "description": "Only current NDC pledges implemented",
    },
    "DIVERGENT_NET_ZERO": {
        "name": "Divergent Net Zero",
        "category": "Disorderly",
        "temperature": "1.5C",
        "description": "Divergent policies across sectors and regions",
    },
    "CURRENT_POLICIES": {
        "name": "Current Policies",
        "category": "Hot House",
        "temperature": ">3C",
        "description": "No additional climate policy beyond current",
    },
}

# Required disclosures by institution type
REQUIRED_DISCLOSURES: Dict[str, List[str]] = {
    "BANK": [
        "ESRS_E1_FINANCED_EMISSIONS",
        "ESRS_E1_TRANSITION_PLAN",
        "PILLAR3_TEMPLATE_1",
        "PILLAR3_TEMPLATE_2",
        "PILLAR3_TEMPLATE_3",
        "PILLAR3_TEMPLATE_4",
        "PILLAR3_TEMPLATE_5",
        "PILLAR3_TEMPLATE_7",
        "PILLAR3_TEMPLATE_8",
        "PILLAR3_TEMPLATE_9",
        "PILLAR3_TEMPLATE_10",
        "GAR_TURNOVER",
        "GAR_CAPEX",
        "GAR_OPEX",
        "BTAR",
    ],
    "INSURANCE": [
        "ESRS_E1_FINANCED_EMISSIONS",
        "ESRS_E1_UNDERWRITING_EMISSIONS",
        "ESRS_E1_TRANSITION_PLAN",
        "SOLVENCY_II_ORSA_CLIMATE",
        "SOLVENCY_II_PRUDENT_PERSON",
    ],
    "ASSET_MANAGER": [
        "ESRS_E1_FINANCED_EMISSIONS",
        "ESRS_E1_TRANSITION_PLAN",
        "SFDR_ENTITY_LEVEL_PAI",
        "WACI",
    ],
    "INVESTMENT_FIRM": [
        "ESRS_E1_FINANCED_EMISSIONS",
        "ESRS_E1_TRANSITION_PLAN",
        "MIFID_ESG_PREFERENCES",
    ],
    "PENSION_FUND": [
        "ESRS_E1_FINANCED_EMISSIONS",
        "ESRS_E1_TRANSITION_PLAN",
        "IORP_STEWARDSHIP",
        "TCFD_SCENARIO_ANALYSIS",
    ],
    "CONGLOMERATE": [
        "ESRS_E1_FINANCED_EMISSIONS",
        "ESRS_E1_UNDERWRITING_EMISSIONS",
        "ESRS_E1_TRANSITION_PLAN",
        "PILLAR3_TEMPLATE_1",
        "PILLAR3_TEMPLATE_2",
        "PILLAR3_TEMPLATE_4",
        "PILLAR3_TEMPLATE_5",
        "PILLAR3_TEMPLATE_7",
        "GAR_TURNOVER",
        "GAR_CAPEX",
        "GAR_OPEX",
        "SOLVENCY_II_ORSA_CLIMATE",
        "CONSOLIDATED_GROUP_REPORT",
    ],
}

# Available presets
AVAILABLE_PRESETS: Dict[str, str] = {
    "bank": "CRR-regulated credit institutions with GAR, BTAR, Pillar 3, and PCAF",
    "insurance": "Solvency II insurance undertakings with underwriting emissions",
    "asset_manager": "UCITS/AIFM asset managers with SFDR bridge and WACI",
    "investment_firm": "MiFID II investment firms with ESG product governance",
    "pension_fund": "IORP II pension funds with long-horizon climate risk",
    "conglomerate": "Multi-entity financial conglomerates with cross-sector consolidation",
}


# =============================================================================
# Pydantic Sub-Config Models
# =============================================================================


class PCAFConfig(BaseModel):
    """Configuration for PCAF financed emissions calculation.

    Implements the Partnership for Carbon Accounting Financials Global
    Standard v3 for measuring and disclosing GHG emissions associated
    with financial portfolios.
    """

    enabled: bool = Field(
        True,
        description="Enable PCAF financed emissions calculation",
    )
    enabled_asset_classes: List[PCAFAssetClass] = Field(
        default_factory=lambda: [
            PCAFAssetClass.LISTED_EQUITY_CORPORATE_BONDS,
            PCAFAssetClass.BUSINESS_LOANS_UNLISTED_EQUITY,
            PCAFAssetClass.PROJECT_FINANCE,
            PCAFAssetClass.COMMERCIAL_REAL_ESTATE,
            PCAFAssetClass.MORTGAGES,
            PCAFAssetClass.MOTOR_VEHICLE_LOANS,
        ],
        description="PCAF asset classes to include in financed emissions calculation",
    )
    include_scope_1: bool = Field(
        True,
        description="Include Scope 1 emissions in financed emissions",
    )
    include_scope_2: bool = Field(
        True,
        description="Include Scope 2 emissions in financed emissions",
    )
    include_scope_3: bool = Field(
        True,
        description="Include Scope 3 emissions in financed emissions (recommended)",
    )
    min_data_quality_score: PCAFDataQuality = Field(
        PCAFDataQuality.SCORE_5,
        description="Maximum acceptable data quality score (1=best, 5=worst)",
    )
    target_data_quality_score: PCAFDataQuality = Field(
        PCAFDataQuality.SCORE_3,
        description="Target data quality score for improvement roadmap",
    )
    estimation_enabled: bool = Field(
        True,
        description="Allow PCAF-compliant estimation when reported data unavailable",
    )
    estimation_methodology: str = Field(
        "SECTOR_AVERAGE",
        description="Estimation methodology: SECTOR_AVERAGE, REVENUE_BASED, ASSET_BASED",
    )
    emission_factor_source: str = Field(
        "PCAF_DATABASE",
        description="Primary emission factor source: PCAF_DATABASE, IEA, DEFRA, EXIOBASE",
    )
    emission_factor_vintage_year: int = Field(
        2024,
        ge=2020,
        le=2030,
        description="Vintage year for emission factors",
    )
    attribution_method: str = Field(
        "OUTSTANDING_DIVIDED_BY_EVIC",
        description="PCAF attribution method for financed emissions",
    )
    sector_aggregation_code: str = Field(
        "NACE_REV2",
        description="Sector classification for aggregation: NACE_REV2, GICS, ICB",
    )
    year_over_year_tracking: bool = Field(
        True,
        description="Enable year-over-year financed emissions trend analysis",
    )
    portfolio_carbon_footprint: bool = Field(
        True,
        description="Calculate portfolio-level carbon footprint intensity (tCO2e/EUR M)",
    )
    waci_enabled: bool = Field(
        True,
        description="Calculate Weighted Average Carbon Intensity (WACI)",
    )
    data_quality_improvement_plan: bool = Field(
        True,
        description="Generate data quality improvement roadmap per asset class",
    )

    @field_validator("emission_factor_vintage_year")
    @classmethod
    def validate_vintage(cls, v: int) -> int:
        """Warn if emission factors are too old."""
        if v < 2022:
            logger.warning(
                "PCAF emission_factor_vintage_year %d is outdated. "
                "Recommend using 2023 or later for current reporting.",
                v,
            )
        return v


class GARBTARConfig(BaseModel):
    """Configuration for Green Asset Ratio and BTAR calculation.

    Implements EBA ITS templates for GAR (mandatory for CRR institutions)
    and BTAR (voluntary complementary KPI).
    """

    gar_enabled: bool = Field(
        True,
        description="Enable Green Asset Ratio calculation (mandatory for banks)",
    )
    btar_enabled: bool = Field(
        True,
        description="Enable Banking Book Taxonomy Alignment Ratio (voluntary)",
    )
    gar_scopes: List[GARScope] = Field(
        default_factory=lambda: [GARScope.TURNOVER, GARScope.CAPEX, GARScope.OPEX],
        description="GAR KPI scopes to calculate",
    )
    exposure_categories: List[GARExposureCategory] = Field(
        default_factory=lambda: [
            GARExposureCategory.FINANCIAL_CORPORATES,
            GARExposureCategory.NON_FINANCIAL_CORPORATES,
            GARExposureCategory.HOUSEHOLDS_MORTGAGES,
            GARExposureCategory.HOUSEHOLDS_MOTOR_VEHICLE,
            GARExposureCategory.LOCAL_GOVERNMENTS,
        ],
        description="Exposure categories to include in GAR denominator",
    )
    exclude_sovereign_central_bank: bool = Field(
        True,
        description="Exclude sovereign and central bank exposures from GAR",
    )
    exclude_trading_book: bool = Field(
        True,
        description="Exclude trading book from GAR (banking book only)",
    )
    flow_gar_enabled: bool = Field(
        True,
        description="Calculate flow GAR (new originations in reporting period)",
    )
    stock_gar_enabled: bool = Field(
        True,
        description="Calculate stock GAR (total outstanding balance sheet)",
    )
    btar_estimation_method: str = Field(
        "SECTOR_PROXY",
        description="BTAR estimation method: SECTOR_PROXY, COUNTERPARTY_LEVEL, MIXED",
    )
    btar_confidence_interval: bool = Field(
        True,
        description="Generate confidence intervals for BTAR estimates",
    )
    taxonomy_environmental_objectives: List[str] = Field(
        default_factory=lambda: [
            "CLIMATE_MITIGATION",
            "CLIMATE_ADAPTATION",
            "WATER",
            "CIRCULAR_ECONOMY",
            "POLLUTION_PREVENTION",
            "BIODIVERSITY",
        ],
        description="EU Taxonomy environmental objectives for GAR breakdown",
    )
    nace_sector_mapping_enabled: bool = Field(
        True,
        description="Enable NACE Rev.2 sector mapping for GAR reporting",
    )
    pillar3_template_output: bool = Field(
        True,
        description="Generate Pillar 3 Templates 7-10 from GAR/BTAR data",
    )


class InsuranceConfig(BaseModel):
    """Configuration for insurance-specific emissions and reporting.

    Implements PCAF Insurance-Associated Emissions Standard and
    Solvency II sustainability integration requirements.
    """

    enabled: bool = Field(
        False,
        description="Enable insurance underwriting emissions calculation",
    )
    attribution_method: str = Field(
        "PREMIUM_BASED",
        description="Attribution method: PREMIUM_BASED, CLAIMS_BASED, HYBRID",
    )
    enabled_lines: List[InsuranceLineType] = Field(
        default_factory=lambda: [
            InsuranceLineType.COMMERCIAL_PROPERTY,
            InsuranceLineType.COMMERCIAL_CASUALTY,
            InsuranceLineType.PERSONAL_PROPERTY,
            InsuranceLineType.PERSONAL_AUTO,
        ],
        description="Insurance business lines to include",
    )
    reinsurance_adjustment: bool = Field(
        True,
        description="Apply reinsurance ceding adjustments to net emissions",
    )
    gross_net_reporting: bool = Field(
        True,
        description="Report both gross and net insurance-associated emissions",
    )
    orsa_climate_integration: bool = Field(
        False,
        description="Integrate climate scenarios into ORSA (Solvency II)",
    )
    solvency_ii_reporting: bool = Field(
        False,
        description="Generate Solvency II prudential sustainability disclosures",
    )
    physical_risk_insured_assets: bool = Field(
        True,
        description="Assess physical climate risk for insured asset portfolio",
    )
    investment_portfolio_alignment: bool = Field(
        True,
        description="Calculate taxonomy alignment for investment portfolio",
    )


class ClimateRiskConfig(BaseModel):
    """Configuration for climate risk scoring engine.

    Implements NGFS scenario analysis for physical and transition risk
    assessment, compatible with ECB/SSM and EBA stress test frameworks.
    """

    enabled: bool = Field(
        True,
        description="Enable climate risk scoring",
    )
    enabled_scenarios: List[NGFSScenario] = Field(
        default_factory=lambda: [
            NGFSScenario.NET_ZERO_2050,
            NGFSScenario.BELOW_2C,
            NGFSScenario.DELAYED_TRANSITION,
            NGFSScenario.NDCS,
            NGFSScenario.DIVERGENT_NET_ZERO,
            NGFSScenario.CURRENT_POLICIES,
        ],
        description="NGFS scenarios to include in climate risk analysis",
    )
    physical_risk_enabled: bool = Field(
        True,
        description="Enable physical climate risk assessment",
    )
    transition_risk_enabled: bool = Field(
        True,
        description="Enable transition climate risk assessment",
    )
    physical_hazards: List[PhysicalHazardType] = Field(
        default_factory=lambda: [
            PhysicalHazardType.FLOOD,
            PhysicalHazardType.WILDFIRE,
            PhysicalHazardType.STORM,
            PhysicalHazardType.HEATWAVE,
            PhysicalHazardType.SEA_LEVEL_RISE,
            PhysicalHazardType.DROUGHT,
        ],
        description="Physical hazard types to assess",
    )
    transition_channels: List[TransitionRiskChannel] = Field(
        default_factory=lambda: [
            TransitionRiskChannel.POLICY_LEGAL,
            TransitionRiskChannel.TECHNOLOGY,
            TransitionRiskChannel.MARKET,
            TransitionRiskChannel.REPUTATION,
            TransitionRiskChannel.STRANDED_ASSETS,
        ],
        description="Transition risk channels to assess",
    )
    time_horizons_years: List[int] = Field(
        default_factory=lambda: [5, 15, 30],
        description="Time horizons for risk assessment (years from now)",
    )
    ecb_ssm_compatible: bool = Field(
        True,
        description="Generate ECB/SSM climate stress test compatible output",
    )
    eba_exercise_compatible: bool = Field(
        True,
        description="Generate EBA one-off climate risk exercise compatible output",
    )
    expected_credit_loss_impact: bool = Field(
        True,
        description="Calculate expected credit loss (ECL) adjustments under scenarios",
    )
    sector_heatmap_enabled: bool = Field(
        True,
        description="Generate sector-level risk heatmap by NACE code",
    )
    geography_heatmap_enabled: bool = Field(
        True,
        description="Generate geography-level risk heatmap by country/region",
    )
    pillar3_template_4_output: bool = Field(
        True,
        description="Generate Pillar 3 Template 4 (physical risk exposures)",
    )


class Pillar3Config(BaseModel):
    """Configuration for EBA Pillar 3 ESG disclosure generation.

    Implements CRR Article 449a and EBA ITS on ESG risk disclosures.
    """

    enabled: bool = Field(
        True,
        description="Enable Pillar 3 ESG disclosure generation",
    )
    enabled_templates: List[Pillar3Template] = Field(
        default_factory=lambda: [
            Pillar3Template.TEMPLATE_1,
            Pillar3Template.TEMPLATE_2,
            Pillar3Template.TEMPLATE_3,
            Pillar3Template.TEMPLATE_4,
            Pillar3Template.TEMPLATE_5,
            Pillar3Template.TEMPLATE_7,
            Pillar3Template.TEMPLATE_8,
            Pillar3Template.TEMPLATE_9,
            Pillar3Template.TEMPLATE_10,
        ],
        description="Pillar 3 ESG templates to generate",
    )
    large_institution: bool = Field(
        False,
        description="Large institution flag (total assets > EUR 750bn for extended T2/T3)",
    )
    total_assets_eur_bn: Optional[float] = Field(
        None,
        description="Total assets in EUR billions for template applicability",
    )
    reporting_frequency: ReportingFrequency = Field(
        ReportingFrequency.SEMI_ANNUAL,
        description="Pillar 3 ESG reporting frequency",
    )
    xbrl_output: bool = Field(
        True,
        description="Generate XBRL-tagged output for EBA submission",
    )
    corep_finrep_cross_validation: bool = Field(
        True,
        description="Cross-validate Pillar 3 data against COREP/FINREP",
    )
    qualitative_narrative_enabled: bool = Field(
        True,
        description="Generate qualitative narrative for Template 1 (LLM-assisted)",
    )

    @model_validator(mode="after")
    def validate_large_institution(self) -> "Pillar3Config":
        """Auto-set large institution flag based on total assets."""
        if self.total_assets_eur_bn is not None and self.total_assets_eur_bn >= 750.0:
            if not self.large_institution:
                logger.info(
                    "Auto-setting large_institution=True based on total_assets_eur_bn=%.1f",
                    self.total_assets_eur_bn,
                )
                object.__setattr__(self, "large_institution", True)
        return self


class FSMaterialityConfig(BaseModel):
    """Configuration for financial sector double materiality assessment.

    Implements ESRS 1 (IRO-1) with financial sector adaptations for
    indirect impacts through lending, investing, and underwriting.
    """

    enabled: bool = Field(
        True,
        description="Enable financial sector double materiality assessment",
    )
    esrs_topics: List[ESRSTopic] = Field(
        default_factory=lambda: [
            ESRSTopic.E1, ESRSTopic.E2, ESRSTopic.E3, ESRSTopic.E4, ESRSTopic.E5,
            ESRSTopic.S1, ESRSTopic.S2, ESRSTopic.S3, ESRSTopic.S4,
            ESRSTopic.G1,
        ],
        description="ESRS topics to assess for materiality",
    )
    impact_channels: List[str] = Field(
        default_factory=lambda: [
            "LENDING",
            "INVESTING",
            "UNDERWRITING",
            "ADVISORY",
        ],
        description="Financial sector impact channels for materiality assessment",
    )
    financial_materiality_enabled: bool = Field(
        True,
        description="Assess financial materiality (risks and opportunities)",
    )
    impact_materiality_enabled: bool = Field(
        True,
        description="Assess impact materiality (impacts on people/environment)",
    )
    stakeholder_engagement_required: bool = Field(
        True,
        description="Require stakeholder engagement input for materiality",
    )
    materiality_threshold_financial: MaterialityLevel = Field(
        MaterialityLevel.MEDIUM,
        description="Minimum level to classify as financially material",
    )
    materiality_threshold_impact: MaterialityLevel = Field(
        MaterialityLevel.MEDIUM,
        description="Minimum level to classify as impact material",
    )
    weight_by_exposure_size: bool = Field(
        True,
        description="Weight materiality assessment by portfolio exposure size",
    )
    generate_iro1_documentation: bool = Field(
        True,
        description="Generate ESRS IRO-1 assessment documentation",
    )


class FSTransitionPlanConfig(BaseModel):
    """Configuration for financial institution transition plan generation.

    Implements ESRS E1 (E1-1 Transition Plan) with SBTi FI alignment.
    """

    enabled: bool = Field(
        True,
        description="Enable transition plan generation",
    )
    sbti_fi_aligned: bool = Field(
        True,
        description="Align transition plan with SBTi FI framework",
    )
    sbti_method: SBTiFIMethod = Field(
        SBTiFIMethod.SDA,
        description="SBTi FI target-setting method",
    )
    target_years: List[int] = Field(
        default_factory=lambda: [2025, 2030, 2035, 2040, 2050],
        description="Milestone years for transition plan targets",
    )
    sector_targets_enabled: bool = Field(
        True,
        description="Set sector-specific decarbonization targets (SDA)",
    )
    priority_sectors: List[str] = Field(
        default_factory=lambda: [
            "POWER_GENERATION",
            "OIL_GAS",
            "TRANSPORT",
            "STEEL",
            "CEMENT",
            "REAL_ESTATE",
            "AGRICULTURE",
        ],
        description="Priority NACE sectors for sector-specific targets",
    )
    operational_emissions_target: bool = Field(
        True,
        description="Include operational emissions reduction targets",
    )
    financed_emissions_target: bool = Field(
        True,
        description="Include financed emissions reduction targets",
    )
    client_engagement_strategy: bool = Field(
        True,
        description="Include client engagement and transition support strategy",
    )
    fossil_fuel_phasedown: bool = Field(
        True,
        description="Include fossil fuel exposure phase-down schedule",
    )
    capital_allocation_alignment: bool = Field(
        True,
        description="Include capital allocation alignment with Paris goals",
    )
    governance_integration: bool = Field(
        True,
        description="Include board-level governance integration of transition",
    )
    temperature_alignment_target: float = Field(
        1.5,
        ge=1.0,
        le=3.0,
        description="Target temperature alignment (degrees Celsius)",
    )


class SBTiFIConfig(BaseModel):
    """Configuration for SBTi Financial Institutions integration."""

    enabled: bool = Field(
        True,
        description="Enable SBTi FI target tracking",
    )
    commitment_status: str = Field(
        "COMMITTED",
        description="SBTi commitment status: COMMITTED, TARGET_SET, VALIDATED",
    )
    target_type: str = Field(
        "NEAR_TERM",
        description="Target type: NEAR_TERM (2030), LONG_TERM (2050)",
    )
    portfolio_coverage_target_pct: float = Field(
        67.0,
        ge=0.0,
        le=100.0,
        description="Portfolio coverage target (% of financed emissions covered)",
    )
    engagement_threshold_pct: float = Field(
        50.0,
        ge=0.0,
        le=100.0,
        description="Engagement threshold (% of counterparties with SBTs)",
    )
    temperature_rating_method: bool = Field(
        True,
        description="Enable temperature rating methodology",
    )
    sda_sectors_enabled: bool = Field(
        True,
        description="Enable Sectoral Decarbonization Approach for key sectors",
    )


class DataQualityConfig(BaseModel):
    """Configuration for data quality management across all engines."""

    pcaf_score_tracking: bool = Field(
        True,
        description="Track PCAF data quality scores (1-5) per asset class",
    )
    min_coverage_pct: float = Field(
        70.0,
        ge=0.0,
        le=100.0,
        description="Minimum data coverage percentage for financed emissions",
    )
    allow_sector_proxies: bool = Field(
        True,
        description="Allow sector-average proxies when counterparty data unavailable",
    )
    proxy_usage_cap_pct: float = Field(
        30.0,
        ge=0.0,
        le=100.0,
        description="Maximum percentage of portfolio using proxy/estimated data",
    )
    require_audited_emissions: bool = Field(
        False,
        description="Require audited/verified emissions data where available",
    )
    data_quality_improvement_roadmap: bool = Field(
        True,
        description="Generate data quality improvement roadmap",
    )
    stale_data_threshold_days: int = Field(
        365,
        ge=30,
        le=730,
        description="Maximum age of emissions data before flagging as stale",
    )


class DisclosureConfig(BaseModel):
    """Configuration for disclosure document generation."""

    esrs_chapter_enabled: bool = Field(
        True,
        description="Generate financial services ESRS chapter",
    )
    pillar3_package_enabled: bool = Field(
        True,
        description="Generate Pillar 3 ESG disclosure package",
    )
    pcaf_report_enabled: bool = Field(
        True,
        description="Generate PCAF financed emissions report",
    )
    gar_btar_report_enabled: bool = Field(
        True,
        description="Generate GAR/BTAR disclosure report",
    )
    climate_risk_report_enabled: bool = Field(
        True,
        description="Generate climate risk assessment report",
    )
    sbti_fi_report_enabled: bool = Field(
        True,
        description="Generate SBTi FI progress report",
    )
    dashboard_enabled: bool = Field(
        True,
        description="Generate interactive financed emissions dashboard",
    )
    output_formats: List[DisclosureFormat] = Field(
        default_factory=lambda: [DisclosureFormat.PDF, DisclosureFormat.XLSX],
        description="Output formats for disclosure documents",
    )
    xbrl_tagging: bool = Field(
        True,
        description="Include XBRL taxonomy tagging for Pillar 3 templates",
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
        description="Enable review and approval workflow for disclosures",
    )
    watermark_draft: bool = Field(
        True,
        description="Apply DRAFT watermark to unapproved documents",
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
        7,
        ge=1,
        le=15,
        description="Audit trail retention period in years",
    )
    external_audit_export: bool = Field(
        True,
        description="Enable export format for external auditors",
    )


# =============================================================================
# Main Configuration Model
# =============================================================================


class CSRDFinancialServiceConfig(BaseModel):
    """Main configuration for PACK-012 CSRD Financial Service Pack.

    This is the root configuration model that contains all sub-configurations
    for financial institution CSRD compliance. The institution_type field
    drives which engines and disclosures are required.
    """

    # Institution identification
    institution_type: FinancialInstitutionType = Field(
        FinancialInstitutionType.BANK,
        description="Financial institution type (drives required engines)",
    )
    institution_name: str = Field(
        "",
        description="Legal entity name of the financial institution",
    )
    lei_code: str = Field(
        "",
        description="Legal Entity Identifier (LEI) code",
    )
    reporting_currency: str = Field(
        "EUR",
        description="Reporting currency (ISO 4217 code)",
    )
    reporting_period_start: Optional[date] = Field(
        None,
        description="Start date of the reporting period",
    )
    reporting_period_end: Optional[date] = Field(
        None,
        description="End date of the reporting period",
    )
    consolidation_scope: str = Field(
        "GROUP",
        description="Consolidation scope: GROUP, SOLO, SUB_CONSOLIDATED",
    )

    # Sub-configurations
    pcaf: PCAFConfig = Field(
        default_factory=PCAFConfig,
        description="PCAF financed emissions configuration",
    )
    gar_btar: GARBTARConfig = Field(
        default_factory=GARBTARConfig,
        description="GAR/BTAR configuration",
    )
    insurance: InsuranceConfig = Field(
        default_factory=InsuranceConfig,
        description="Insurance-specific configuration",
    )
    climate_risk: ClimateRiskConfig = Field(
        default_factory=ClimateRiskConfig,
        description="Climate risk scoring configuration",
    )
    pillar3: Pillar3Config = Field(
        default_factory=Pillar3Config,
        description="Pillar 3 ESG disclosure configuration",
    )
    materiality: FSMaterialityConfig = Field(
        default_factory=FSMaterialityConfig,
        description="Financial sector double materiality configuration",
    )
    transition_plan: FSTransitionPlanConfig = Field(
        default_factory=FSTransitionPlanConfig,
        description="Transition plan configuration",
    )
    sbti_fi: SBTiFIConfig = Field(
        default_factory=SBTiFIConfig,
        description="SBTi Financial Institutions configuration",
    )
    data_quality: DataQualityConfig = Field(
        default_factory=DataQualityConfig,
        description="Data quality management configuration",
    )
    disclosure: DisclosureConfig = Field(
        default_factory=DisclosureConfig,
        description="Disclosure document generation configuration",
    )
    audit_trail: AuditTrailConfig = Field(
        default_factory=AuditTrailConfig,
        description="Audit trail and provenance configuration",
    )

    @field_validator("institution_type")
    @classmethod
    def validate_institution_type(cls, v: FinancialInstitutionType) -> FinancialInstitutionType:
        """Validate institution type is a supported value."""
        valid_types = {e.value for e in FinancialInstitutionType}
        if v.value not in valid_types:
            raise ValueError(
                f"Invalid institution_type: {v.value}. "
                f"Must be one of: {sorted(valid_types)}"
            )
        return v

    @model_validator(mode="after")
    def validate_pcaf_required_for_banks(self) -> "CSRDFinancialServiceConfig":
        """Ensure PCAF is enabled for bank institution types."""
        if self.institution_type == FinancialInstitutionType.BANK:
            if not self.pcaf.enabled:
                logger.warning(
                    "PCAF financed emissions are strongly recommended for credit "
                    "institutions. Enabling PCAF for bank institution type."
                )
                object.__setattr__(self.pcaf, "enabled", True)
        return self

    @model_validator(mode="after")
    def validate_insurance_config_for_insurers(self) -> "CSRDFinancialServiceConfig":
        """Ensure insurance config is enabled for insurance institution types."""
        if self.institution_type == FinancialInstitutionType.INSURANCE:
            if not self.insurance.enabled:
                logger.warning(
                    "Insurance underwriting emissions calculation is required for "
                    "insurance undertakings. Enabling insurance config."
                )
                object.__setattr__(self.insurance, "enabled", True)
        return self

    @model_validator(mode="after")
    def validate_pillar3_for_crr_banks(self) -> "CSRDFinancialServiceConfig":
        """Ensure Pillar 3 is enabled for CRR-regulated banks."""
        if self.institution_type == FinancialInstitutionType.BANK:
            if not self.pillar3.enabled:
                logger.warning(
                    "Pillar 3 ESG disclosures are mandatory for CRR-regulated "
                    "credit institutions (Article 449a). Enabling Pillar 3."
                )
                object.__setattr__(self.pillar3, "enabled", True)
        return self

    @model_validator(mode="after")
    def validate_conglomerate_all_engines(self) -> "CSRDFinancialServiceConfig":
        """Ensure conglomerate has all relevant engines enabled."""
        if self.institution_type == FinancialInstitutionType.CONGLOMERATE:
            if not self.insurance.enabled:
                logger.info(
                    "Conglomerate: enabling insurance config for cross-sector coverage."
                )
                object.__setattr__(self.insurance, "enabled", True)
            if not self.pillar3.enabled:
                logger.info(
                    "Conglomerate: enabling Pillar 3 for banking subsidiaries."
                )
                object.__setattr__(self.pillar3, "enabled", True)
        return self


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """Top-level pack configuration wrapper.

    Handles preset loading, environment variable overrides, and
    configuration merging.
    """

    pack: CSRDFinancialServiceConfig = Field(
        default_factory=CSRDFinancialServiceConfig,
        description="Main CSRD Financial Service configuration",
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
        "PACK-012-csrd-financial-service",
        description="Pack identifier",
    )

    @classmethod
    def from_preset(cls, preset_name: str, overrides: Optional[Dict[str, Any]] = None) -> "PackConfig":
        """Load configuration from a named preset.

        Args:
            preset_name: Name of the preset (bank, insurance, asset_manager, etc.)
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

        pack_config = CSRDFinancialServiceConfig(**preset_data)
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

        pack_config = CSRDFinancialServiceConfig(**config_data)
        return cls(pack=pack_config)

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """Load configuration overrides from environment variables.

        Environment variables prefixed with CSRD_FS_PACK_ are loaded and
        mapped to configuration keys. Nested keys use double underscore.

        Example: CSRD_FS_PACK_PCAF__INCLUDE_SCOPE_3=true
        """
        overrides: Dict[str, Any] = {}
        prefix = "CSRD_FS_PACK_"
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


def get_institution_display_name(institution_type: Union[str, FinancialInstitutionType]) -> str:
    """Get human-readable display name for an institution type.

    Args:
        institution_type: Institution type enum or string value.

    Returns:
        Display name string for the institution type.
    """
    key = institution_type.value if isinstance(institution_type, FinancialInstitutionType) else institution_type
    return INSTITUTION_DISPLAY_NAMES.get(key, f"Unknown ({key})")


def get_required_disclosures(institution_type: Union[str, FinancialInstitutionType]) -> List[str]:
    """Get list of required disclosures for an institution type.

    Args:
        institution_type: Institution type enum or string value.

    Returns:
        List of required disclosure identifiers.
    """
    key = institution_type.value if isinstance(institution_type, FinancialInstitutionType) else institution_type
    return REQUIRED_DISCLOSURES.get(key, [])


def get_pcaf_asset_class_info(asset_class: Union[str, PCAFAssetClass]) -> Dict[str, str]:
    """Get detailed information about a PCAF asset class.

    Args:
        asset_class: PCAF asset class enum or string value.

    Returns:
        Dictionary with name, attribution method, and description.
    """
    key = asset_class.value if isinstance(asset_class, PCAFAssetClass) else asset_class
    return PCAF_ASSET_CLASS_INFO.get(key, {"name": key, "attribution": "N/A", "description": "N/A"})


def get_ngfs_scenario_info(scenario: Union[str, NGFSScenario]) -> Dict[str, str]:
    """Get detailed information about an NGFS scenario.

    Args:
        scenario: NGFS scenario enum or string value.

    Returns:
        Dictionary with name, category, temperature, and description.
    """
    key = scenario.value if isinstance(scenario, NGFSScenario) else scenario
    return NGFS_SCENARIO_DESCRIPTIONS.get(
        key, {"name": key, "category": "Unknown", "temperature": "N/A", "description": "N/A"}
    )


def list_available_presets() -> Dict[str, str]:
    """List all available configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return AVAILABLE_PRESETS.copy()