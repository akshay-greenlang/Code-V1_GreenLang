"""
GL-TCFD-APP v1.0 -- TCFD Climate Disclosure Platform Configuration

Enumerations, constants, scenario parameter libraries, and application settings for
implementing the TCFD four-pillar framework with ISSB/IFRS S2 cross-walk.

The TCFD framework comprises four thematic areas: Governance, Strategy, Risk
Management, and Metrics & Targets.  Each area has recommended disclosures that
organizations should publish.  This platform automates the collection,
calculation, scenario analysis, and reporting for all 11 recommended disclosures
plus the ISSB/IFRS S2 cross-walk.

All settings use the TCFD_APP_ prefix for environment variable overrides.

Reference:
    - TCFD Final Report (June 2017)
    - TCFD Guidance on Scenario Analysis (October 2020)
    - TCFD Annex: Implementing the Recommendations (June 2017)
    - IFRS S2 Climate-related Disclosures (June 2023)
    - NGFS Climate Scenarios (September 2022)
    - IEA World Energy Outlook (2023)
    - IPCC AR6 WG1 Table 7.15 (GWP values)

Example:
    >>> config = TCFDAppConfig()
    >>> config.app_name
    'GL-TCFD-APP'
    >>> config.default_scenario_type
    <ScenarioType.IEA_NZE: 'iea_nze'>
    >>> SCENARIO_CARBON_PRICES[ScenarioType.IEA_NZE][2030]
    130
"""

from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Tuple

from pydantic import Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TCFDPillar(str, Enum):
    """TCFD four-pillar thematic areas for climate-related disclosures."""

    GOVERNANCE = "governance"
    STRATEGY = "strategy"
    RISK_MANAGEMENT = "risk_management"
    METRICS_TARGETS = "metrics_targets"


class DisclosureCode(str, Enum):
    """
    TCFD 11 recommended disclosures.

    Governance (a-b), Strategy (a-c), Risk Management (a-c),
    Metrics & Targets (a-c).
    """

    GOV_A = "gov_a"
    GOV_B = "gov_b"
    STRAT_A = "strat_a"
    STRAT_B = "strat_b"
    STRAT_C = "strat_c"
    RM_A = "rm_a"
    RM_B = "rm_b"
    RM_C = "rm_c"
    MT_A = "mt_a"
    MT_B = "mt_b"
    MT_C = "mt_c"


class RiskType(str, Enum):
    """Climate risk types per TCFD classification."""

    PHYSICAL_ACUTE = "physical_acute"
    PHYSICAL_CHRONIC = "physical_chronic"
    TRANSITION_POLICY = "transition_policy"
    TRANSITION_TECHNOLOGY = "transition_technology"
    TRANSITION_MARKET = "transition_market"
    TRANSITION_REPUTATION = "transition_reputation"


class PhysicalHazard(str, Enum):
    """Physical climate hazard types per TCFD/IPCC classification."""

    CYCLONE = "cyclone"
    FLOOD = "flood"
    WILDFIRE = "wildfire"
    HEATWAVE = "heatwave"
    DROUGHT = "drought"
    SEA_LEVEL_RISE = "sea_level_rise"
    TEMPERATURE_RISE = "temperature_rise"
    WATER_STRESS = "water_stress"
    PRECIPITATION_CHANGE = "precipitation_change"
    ECOSYSTEM_DEGRADATION = "ecosystem_degradation"


class TransitionRiskSubType(str, Enum):
    """Transition risk sub-category drivers per TCFD framework."""

    CARBON_PRICING = "carbon_pricing"
    REGULATION = "regulation"
    LITIGATION = "litigation"
    TECHNOLOGY_SUBSTITUTION = "technology_substitution"
    TECHNOLOGY_DISRUPTION = "technology_disruption"
    DEMAND_SHIFT = "demand_shift"
    SUPPLY_CHAIN = "supply_chain"
    COMMODITY_PRICE = "commodity_price"
    STAKEHOLDER_SENTIMENT = "stakeholder_sentiment"
    BRAND_VALUE = "brand_value"


class OpportunityCategory(str, Enum):
    """TCFD climate-related opportunity categories."""

    RESOURCE_EFFICIENCY = "resource_efficiency"
    ENERGY_SOURCE = "energy_source"
    PRODUCTS_SERVICES = "products_services"
    MARKETS = "markets"
    RESILIENCE = "resilience"


class ScenarioType(str, Enum):
    """
    Pre-built climate scenario archetypes.

    Covers IEA (International Energy Agency) and NGFS (Network for Greening
    the Financial System) pathways plus custom user-defined scenarios.
    """

    IEA_NZE = "iea_nze"
    IEA_APS = "iea_aps"
    IEA_STEPS = "iea_steps"
    NGFS_CURRENT_POLICIES = "ngfs_current_policies"
    NGFS_DELAYED_TRANSITION = "ngfs_delayed_transition"
    NGFS_BELOW_2C = "ngfs_below_2c"
    NGFS_DIVERGENT_NET_ZERO = "ngfs_divergent_nz"
    CUSTOM = "custom"


class TimeHorizon(str, Enum):
    """
    TCFD time horizons for climate risk assessment.

    Short-term: 0-3 years (aligned with financial planning cycles)
    Medium-term: 3-10 years (aligned with capital planning)
    Long-term: 10-30+ years (aligned with climate scenarios)
    """

    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


class FinancialStatementType(str, Enum):
    """Areas of financial statements impacted by climate risks."""

    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"


class FinancialImpactCategory(str, Enum):
    """Granular financial impact categories for climate risk quantification."""

    REVENUE = "revenue"
    OPERATING_COST = "operating_cost"
    CAPITAL_EXPENDITURE = "capital_expenditure"
    ASSET_IMPAIRMENT = "asset_impairment"
    INSURANCE_COST = "insurance_cost"
    CARBON_COST = "carbon_cost"
    COMPLIANCE_COST = "compliance_cost"
    ADAPTATION_COST = "adaptation_cost"
    OPPORTUNITY_REVENUE = "opportunity_revenue"
    COST_SAVINGS = "cost_savings"


class AssetType(str, Enum):
    """Types of physical assets subject to climate risk assessment."""

    BUILDING = "building"
    PLANT = "plant"
    EQUIPMENT = "equipment"
    VEHICLE_FLEET = "vehicle_fleet"
    LAND = "land"
    INFRASTRUCTURE = "infrastructure"
    INVENTORY = "inventory"
    INTELLECTUAL_PROPERTY = "intellectual_property"


class GovernanceMaturityLevel(str, Enum):
    """
    Organizational maturity level for TCFD governance assessment.

    Scored 1 (Initial) through 5 (Optimized) based on the CMMI
    maturity model adapted for climate governance.
    """

    INITIAL = "initial"
    DEVELOPING = "developing"
    DEFINED = "defined"
    MANAGED = "managed"
    OPTIMIZED = "optimized"


class RiskLikelihood(str, Enum):
    """Likelihood scale for climate risk assessment (1-5)."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class RiskImpact(str, Enum):
    """Impact scale for climate risk assessment (1-5)."""

    NEGLIGIBLE = "negligible"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CATASTROPHIC = "catastrophic"


class RiskResponse(str, Enum):
    """Climate risk response strategies."""

    MITIGATE = "mitigate"
    ADAPT = "adapt"
    TRANSFER = "transfer"
    ACCEPT = "accept"
    AVOID = "avoid"


class TargetType(str, Enum):
    """Types of climate-related targets."""

    ABSOLUTE_REDUCTION = "absolute_reduction"
    INTENSITY_REDUCTION = "intensity_reduction"
    NET_ZERO = "net_zero"
    RENEWABLE_ENERGY = "renewable_energy"
    SCIENCE_BASED = "science_based"
    CUSTOM = "custom"


class SBTiAlignment(str, Enum):
    """Science Based Targets initiative alignment classification."""

    WELL_BELOW_2C = "well_below_2c"
    ONE_POINT_FIVE_C = "1_5c"
    NOT_ALIGNED = "not_aligned"
    PENDING_VALIDATION = "pending_validation"


class MetricCategory(str, Enum):
    """Classification of climate-related metrics."""

    CROSS_INDUSTRY = "cross_industry"
    INDUSTRY_SPECIFIC = "industry_specific"
    CUSTOM = "custom"


class ISSBS2Paragraph(str, Enum):
    """
    ISSB/IFRS S2 paragraph ranges corresponding to TCFD pillars.

    Each value maps a TCFD pillar to its IFRS S2 paragraph reference.
    """

    GOVERNANCE_5_9 = "5-9"
    STRATEGY_10_22 = "10-22"
    RISK_MANAGEMENT_23_24 = "23-24"
    METRICS_25_33 = "25-33"


class DisclosureStatus(str, Enum):
    """Lifecycle status of a TCFD disclosure document."""

    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class DataQualityTier(str, Enum):
    """
    Data quality tiers for climate data inputs.

    Tier 1 (Measured) is highest quality; Tier 5 (Default) is lowest.
    """

    MEASURED = "measured"
    CALCULATED = "calculated"
    ESTIMATED = "estimated"
    PROXY = "proxy"
    DEFAULT = "default"


class TemperatureOutcome(str, Enum):
    """Temperature pathway outcomes for scenario analysis."""

    BELOW_1_5C = "below_1_5c"
    AROUND_2C = "around_2c"
    ABOVE_2_5C = "above_2_5c"
    ABOVE_3C = "above_3c"


class SectorType(str, Enum):
    """TCFD sector classification covering all 11 sector supplemental guides."""

    ENERGY = "energy"
    TRANSPORTATION = "transportation"
    MATERIALS_BUILDINGS = "materials_buildings"
    AGRICULTURE_FOOD_FOREST = "agriculture_food_forest"
    BANKING = "banking"
    INSURANCE = "insurance"
    ASSET_OWNERS = "asset_owners"
    ASSET_MANAGERS = "asset_managers"
    CONSUMER_GOODS = "consumer_goods"
    TECHNOLOGY_MEDIA = "technology_media"
    HEALTHCARE = "healthcare"


class ReportFormat(str, Enum):
    """Supported report output formats."""

    PDF = "pdf"
    EXCEL = "excel"
    JSON = "json"
    XML = "xml"


class ISSBMetricType(str, Enum):
    """
    ISSB/IFRS S2 cross-industry climate-related metrics.

    Seven metrics required for all entities reporting under IFRS S2.
    """

    GHG_EMISSIONS = "ghg_emissions"
    TRANSITION_RISK_ASSETS = "transition_risk_assets"
    PHYSICAL_RISK_ASSETS = "physical_risk_assets"
    OPPORTUNITY_REVENUE = "opportunity_revenue"
    CAPITAL_DEPLOYMENT = "capital_deployment"
    INTERNAL_CARBON_PRICE = "internal_carbon_price"
    REMUNERATION_LINKED = "remuneration_linked"


# ---------------------------------------------------------------------------
# Time Horizon Year Ranges
# ---------------------------------------------------------------------------

TIME_HORIZON_YEARS: Dict[TimeHorizon, Dict[str, int]] = {
    TimeHorizon.SHORT_TERM: {"min_years": 0, "max_years": 3},
    TimeHorizon.MEDIUM_TERM: {"min_years": 3, "max_years": 10},
    TimeHorizon.LONG_TERM: {"min_years": 10, "max_years": 30},
}


# ---------------------------------------------------------------------------
# Risk Likelihood / Impact Numeric Scores
# ---------------------------------------------------------------------------

LIKELIHOOD_SCORES: Dict[RiskLikelihood, int] = {
    RiskLikelihood.VERY_LOW: 1,
    RiskLikelihood.LOW: 2,
    RiskLikelihood.MEDIUM: 3,
    RiskLikelihood.HIGH: 4,
    RiskLikelihood.VERY_HIGH: 5,
}

IMPACT_SCORES: Dict[RiskImpact, int] = {
    RiskImpact.NEGLIGIBLE: 1,
    RiskImpact.MINOR: 2,
    RiskImpact.MODERATE: 3,
    RiskImpact.MAJOR: 4,
    RiskImpact.CATASTROPHIC: 5,
}


# ---------------------------------------------------------------------------
# Governance Maturity Numeric Mapping
# ---------------------------------------------------------------------------

MATURITY_SCORES: Dict[GovernanceMaturityLevel, int] = {
    GovernanceMaturityLevel.INITIAL: 1,
    GovernanceMaturityLevel.DEVELOPING: 2,
    GovernanceMaturityLevel.DEFINED: 3,
    GovernanceMaturityLevel.MANAGED: 4,
    GovernanceMaturityLevel.OPTIMIZED: 5,
}


# ---------------------------------------------------------------------------
# Data Quality Tier Numeric Mapping (1 = best, 5 = worst)
# ---------------------------------------------------------------------------

DATA_QUALITY_SCORES: Dict[DataQualityTier, int] = {
    DataQualityTier.MEASURED: 1,
    DataQualityTier.CALCULATED: 2,
    DataQualityTier.ESTIMATED: 3,
    DataQualityTier.PROXY: 4,
    DataQualityTier.DEFAULT: 5,
}


# ---------------------------------------------------------------------------
# TCFD 11 Recommended Disclosures
# ---------------------------------------------------------------------------

TCFD_DISCLOSURES: Dict[str, Dict[str, str]] = {
    "gov_a": {
        "pillar": "governance",
        "ref": "Governance (a)",
        "title": "Board Oversight",
        "description": (
            "Describe the board's oversight of climate-related risks "
            "and opportunities."
        ),
    },
    "gov_b": {
        "pillar": "governance",
        "ref": "Governance (b)",
        "title": "Management Role",
        "description": (
            "Describe management's role in assessing and managing "
            "climate-related risks and opportunities."
        ),
    },
    "strat_a": {
        "pillar": "strategy",
        "ref": "Strategy (a)",
        "title": "Risks and Opportunities",
        "description": (
            "Describe the climate-related risks and opportunities the "
            "organization has identified over the short, medium, and "
            "long term."
        ),
    },
    "strat_b": {
        "pillar": "strategy",
        "ref": "Strategy (b)",
        "title": "Business Impact",
        "description": (
            "Describe the impact of climate-related risks and opportunities "
            "on the organization's businesses, strategy, and financial planning."
        ),
    },
    "strat_c": {
        "pillar": "strategy",
        "ref": "Strategy (c)",
        "title": "Scenario Analysis",
        "description": (
            "Describe the resilience of the organization's strategy, taking "
            "into consideration different climate-related scenarios, including "
            "a 2 degrees C or lower scenario."
        ),
    },
    "rm_a": {
        "pillar": "risk_management",
        "ref": "Risk Management (a)",
        "title": "Risk Identification",
        "description": (
            "Describe the organization's processes for identifying and "
            "assessing climate-related risks."
        ),
    },
    "rm_b": {
        "pillar": "risk_management",
        "ref": "Risk Management (b)",
        "title": "Risk Management Process",
        "description": (
            "Describe the organization's processes for managing "
            "climate-related risks."
        ),
    },
    "rm_c": {
        "pillar": "risk_management",
        "ref": "Risk Management (c)",
        "title": "ERM Integration",
        "description": (
            "Describe how processes for identifying, assessing, and managing "
            "climate-related risks are integrated into the organization's "
            "overall risk management."
        ),
    },
    "mt_a": {
        "pillar": "metrics_targets",
        "ref": "Metrics & Targets (a)",
        "title": "Climate Metrics",
        "description": (
            "Disclose the metrics used by the organization to assess "
            "climate-related risks and opportunities in line with its "
            "strategy and risk management process."
        ),
    },
    "mt_b": {
        "pillar": "metrics_targets",
        "ref": "Metrics & Targets (b)",
        "title": "GHG Emissions",
        "description": (
            "Disclose Scope 1, Scope 2, and, if appropriate, Scope 3 "
            "greenhouse gas (GHG) emissions and the related risks."
        ),
    },
    "mt_c": {
        "pillar": "metrics_targets",
        "ref": "Metrics & Targets (c)",
        "title": "Targets",
        "description": (
            "Describe the targets used by the organization to manage "
            "climate-related risks and opportunities and performance "
            "against targets."
        ),
    },
}


# ---------------------------------------------------------------------------
# Pillar Display Names
# ---------------------------------------------------------------------------

PILLAR_NAMES: Dict[TCFDPillar, str] = {
    TCFDPillar.GOVERNANCE: "Governance",
    TCFDPillar.STRATEGY: "Strategy",
    TCFDPillar.RISK_MANAGEMENT: "Risk Management",
    TCFDPillar.METRICS_TARGETS: "Metrics and Targets",
}


# ---------------------------------------------------------------------------
# Scenario Carbon Prices (USD/tCO2e by scenario and year)
# ---------------------------------------------------------------------------

SCENARIO_CARBON_PRICES: Dict[ScenarioType, Dict[int, int]] = {
    ScenarioType.IEA_NZE: {
        2025: 75, 2030: 130, 2040: 205, 2050: 250,
    },
    ScenarioType.IEA_APS: {
        2025: 50, 2030: 90, 2040: 155, 2050: 200,
    },
    ScenarioType.IEA_STEPS: {
        2025: 25, 2030: 40, 2040: 55, 2050: 65,
    },
    ScenarioType.NGFS_CURRENT_POLICIES: {
        2025: 10, 2030: 15, 2040: 20, 2050: 25,
    },
    ScenarioType.NGFS_DELAYED_TRANSITION: {
        2025: 12, 2030: 25, 2040: 220, 2050: 350,
    },
    ScenarioType.NGFS_BELOW_2C: {
        2025: 55, 2030: 100, 2040: 160, 2050: 200,
    },
    ScenarioType.NGFS_DIVERGENT_NET_ZERO: {
        2025: 65, 2030: 120, 2040: 195, 2050: 230,
    },
    ScenarioType.CUSTOM: {
        2025: 0, 2030: 0, 2040: 0, 2050: 0,
    },
}


# ---------------------------------------------------------------------------
# Scenario Energy Mix (Renewable share % by scenario and year)
# ---------------------------------------------------------------------------

SCENARIO_ENERGY_MIX: Dict[ScenarioType, Dict[int, int]] = {
    ScenarioType.IEA_NZE: {
        2025: 30, 2030: 50, 2040: 75, 2050: 90,
    },
    ScenarioType.IEA_APS: {
        2025: 28, 2030: 40, 2040: 58, 2050: 70,
    },
    ScenarioType.IEA_STEPS: {
        2025: 25, 2030: 32, 2040: 40, 2050: 50,
    },
    ScenarioType.NGFS_CURRENT_POLICIES: {
        2025: 22, 2030: 25, 2040: 30, 2050: 35,
    },
    ScenarioType.NGFS_DELAYED_TRANSITION: {
        2025: 23, 2030: 28, 2040: 55, 2050: 72,
    },
    ScenarioType.NGFS_BELOW_2C: {
        2025: 28, 2030: 42, 2040: 60, 2050: 75,
    },
    ScenarioType.NGFS_DIVERGENT_NET_ZERO: {
        2025: 28, 2030: 45, 2040: 65, 2050: 82,
    },
    ScenarioType.CUSTOM: {
        2025: 0, 2030: 0, 2040: 0, 2050: 0,
    },
}


# ---------------------------------------------------------------------------
# Scenario Temperature Outcomes (degrees C by 2100)
# ---------------------------------------------------------------------------

SCENARIO_TEMPERATURE: Dict[ScenarioType, float] = {
    ScenarioType.IEA_NZE: 1.4,
    ScenarioType.IEA_APS: 1.9,
    ScenarioType.IEA_STEPS: 2.5,
    ScenarioType.NGFS_CURRENT_POLICIES: 3.2,
    ScenarioType.NGFS_DELAYED_TRANSITION: 1.9,
    ScenarioType.NGFS_BELOW_2C: 1.8,
    ScenarioType.NGFS_DIVERGENT_NET_ZERO: 1.5,
    ScenarioType.CUSTOM: 2.0,
}


# ---------------------------------------------------------------------------
# GWP AR6 Values (100-year time horizon, IPCC AR6 WG1 Table 7.15)
# ---------------------------------------------------------------------------

GWP_AR6: Dict[str, float] = {
    "CO2": 1.0,
    "CH4": 27.9,
    "N2O": 273.0,
    "HFC-134a": 1530.0,
    "HFC-23": 14600.0,
    "HFC-32": 771.0,
    "HFC-125": 3740.0,
    "HFC-143a": 5810.0,
    "HFC-152a": 164.0,
    "HFC-227ea": 3600.0,
    "HFC-245fa": 962.0,
    "HFC-365mfc": 914.0,
    "HFC-43-10mee": 1600.0,
    "CF4": 7380.0,
    "C2F6": 12400.0,
    "C3F8": 9290.0,
    "c-C4F8": 10200.0,
    "SF6": 25200.0,
    "NF3": 17400.0,
}


# ---------------------------------------------------------------------------
# TCFD Sector Supplemental Guides (11 sectors)
# ---------------------------------------------------------------------------

TCFD_SECTOR_GUIDES: List[Dict[str, str]] = [
    {
        "sector": "Energy",
        "code": SectorType.ENERGY.value,
        "description": "Oil and gas, coal, electric utilities",
        "key_metrics": "Scope 1/2/3 emissions, energy production mix, reserves",
    },
    {
        "sector": "Transportation",
        "code": SectorType.TRANSPORTATION.value,
        "description": "Air freight, airlines, auto, marine, rail, trucking",
        "key_metrics": "Fleet efficiency, alternative fuel adoption, emissions intensity",
    },
    {
        "sector": "Materials and Buildings",
        "code": SectorType.MATERIALS_BUILDINGS.value,
        "description": "Metals & mining, chemicals, construction, real estate",
        "key_metrics": "Process emissions, energy intensity, green building certifications",
    },
    {
        "sector": "Agriculture, Food, and Forest Products",
        "code": SectorType.AGRICULTURE_FOOD_FOREST.value,
        "description": "Agricultural products, food/beverage, paper/forestry",
        "key_metrics": "Land use change, deforestation, water usage, fertilizer emissions",
    },
    {
        "sector": "Banking",
        "code": SectorType.BANKING.value,
        "description": "Commercial and investment banks",
        "key_metrics": "Financed emissions, green bond issuance, loan book alignment",
    },
    {
        "sector": "Insurance",
        "code": SectorType.INSURANCE.value,
        "description": "Property, casualty, life, and health insurers",
        "key_metrics": "Insured losses, physical risk exposure, underwriting criteria",
    },
    {
        "sector": "Asset Owners",
        "code": SectorType.ASSET_OWNERS.value,
        "description": "Pension funds, sovereign wealth funds, endowments",
        "key_metrics": "Portfolio carbon footprint, WACI, alignment metrics",
    },
    {
        "sector": "Asset Managers",
        "code": SectorType.ASSET_MANAGERS.value,
        "description": "Investment management firms",
        "key_metrics": "AUM alignment, engagement outcomes, stewardship reporting",
    },
    {
        "sector": "Consumer Goods",
        "code": SectorType.CONSUMER_GOODS.value,
        "description": "Consumer durables, apparel, household products",
        "key_metrics": "Supply chain emissions, product lifecycle, circularity",
    },
    {
        "sector": "Technology and Media",
        "code": SectorType.TECHNOLOGY_MEDIA.value,
        "description": "Software, hardware, telecom, media",
        "key_metrics": "Data center energy, Scope 2 intensity, e-waste",
    },
    {
        "sector": "Healthcare",
        "code": SectorType.HEALTHCARE.value,
        "description": "Pharma, biotech, medical devices, providers",
        "key_metrics": "Facility emissions, anesthetic gas emissions, supply chain",
    },
]


# ---------------------------------------------------------------------------
# ISSB Cross-Industry Metrics (IFRS S2 para 29)
# ---------------------------------------------------------------------------

ISSB_CROSS_INDUSTRY_METRICS: List[Dict[str, str]] = [
    {
        "id": "ISSB-CI-01",
        "name": "GHG Emissions (Scope 1, 2, 3)",
        "ifrs_s2_paragraph": "29(a)",
        "unit": "tCO2e",
        "description": "Absolute Scope 1, 2, and 3 GHG emissions.",
    },
    {
        "id": "ISSB-CI-02",
        "name": "Transition Risk Exposed Assets (%)",
        "ifrs_s2_paragraph": "29(b)",
        "unit": "percent",
        "description": "Percentage of assets vulnerable to transition risks.",
    },
    {
        "id": "ISSB-CI-03",
        "name": "Physical Risk Exposed Assets (%)",
        "ifrs_s2_paragraph": "29(c)",
        "unit": "percent",
        "description": "Percentage of assets vulnerable to physical risks.",
    },
    {
        "id": "ISSB-CI-04",
        "name": "Climate Opportunity Revenue (%)",
        "ifrs_s2_paragraph": "29(d)",
        "unit": "percent",
        "description": "Percentage of revenue from climate-related opportunities.",
    },
    {
        "id": "ISSB-CI-05",
        "name": "Climate-related Capital Expenditure",
        "ifrs_s2_paragraph": "29(e)",
        "unit": "currency",
        "description": "Capital expenditure deployed towards climate risks/opportunities.",
    },
    {
        "id": "ISSB-CI-06",
        "name": "Internal Carbon Price",
        "ifrs_s2_paragraph": "29(f)",
        "unit": "currency_per_tCO2e",
        "description": "Price per tCO2e used for internal decision-making.",
    },
    {
        "id": "ISSB-CI-07",
        "name": "Remuneration Linked to Climate (%)",
        "ifrs_s2_paragraph": "29(g)",
        "unit": "percent",
        "description": "Percentage of management remuneration linked to climate.",
    },
]


# ---------------------------------------------------------------------------
# Default 5x5 Risk Matrix (likelihood x impact -> score)
# ---------------------------------------------------------------------------

DEFAULT_RISK_MATRIX: List[List[int]] = [
    # Impact:  Negligible  Minor  Moderate  Major  Catastrophic
    [1,  2,  3,  4,  5],   # Very Low likelihood
    [2,  4,  6,  8,  10],  # Low likelihood
    [3,  6,  9,  12, 15],  # Medium likelihood
    [4,  8,  12, 16, 20],  # High likelihood
    [5,  10, 15, 20, 25],  # Very High likelihood
]


# ---------------------------------------------------------------------------
# Risk Matrix Thresholds (score range -> risk band)
# ---------------------------------------------------------------------------

RISK_MATRIX_THRESHOLDS: Dict[str, Dict[str, int]] = {
    "low": {"min": 1, "max": 5},
    "medium": {"min": 6, "max": 12},
    "high": {"min": 13, "max": 19},
    "critical": {"min": 20, "max": 25},
}


# ---------------------------------------------------------------------------
# TCFD-to-IFRS-S2 Disclosure Mapping
# ---------------------------------------------------------------------------

TCFD_TO_IFRS_S2_MAPPING: Dict[str, Dict[str, str]] = {
    "gov_a": {
        "ifrs_s2_paragraph": "5-6",
        "ifrs_s2_topic": "Governance",
        "mapping_status": "fully_mapped",
        "notes": "IFRS S2 paragraphs 5-6 align with TCFD Gov(a).",
    },
    "gov_b": {
        "ifrs_s2_paragraph": "5-6",
        "ifrs_s2_topic": "Governance",
        "mapping_status": "fully_mapped",
        "notes": "IFRS S2 paragraphs 5-6 align with TCFD Gov(b).",
    },
    "strat_a": {
        "ifrs_s2_paragraph": "10-12",
        "ifrs_s2_topic": "Strategy",
        "mapping_status": "fully_mapped",
        "notes": "IFRS S2 requires quantitative assessment of risks.",
    },
    "strat_b": {
        "ifrs_s2_paragraph": "13-14",
        "ifrs_s2_topic": "Strategy",
        "mapping_status": "fully_mapped",
        "notes": "IFRS S2 extends to financial statement connectivity.",
    },
    "strat_c": {
        "ifrs_s2_paragraph": "22",
        "ifrs_s2_topic": "Climate Resilience",
        "mapping_status": "enhanced",
        "notes": (
            "IFRS S2 para 22 requires scenario analysis and climate "
            "resilience assessment regardless of size."
        ),
    },
    "rm_a": {
        "ifrs_s2_paragraph": "25",
        "ifrs_s2_topic": "Risk Management",
        "mapping_status": "fully_mapped",
        "notes": "IFRS S2 para 25 aligns with TCFD RM(a).",
    },
    "rm_b": {
        "ifrs_s2_paragraph": "25",
        "ifrs_s2_topic": "Risk Management",
        "mapping_status": "fully_mapped",
        "notes": "IFRS S2 para 25 aligns with TCFD RM(b).",
    },
    "rm_c": {
        "ifrs_s2_paragraph": "25",
        "ifrs_s2_topic": "Risk Management",
        "mapping_status": "fully_mapped",
        "notes": "IFRS S2 para 25 aligns with TCFD RM(c).",
    },
    "mt_a": {
        "ifrs_s2_paragraph": "29",
        "ifrs_s2_topic": "Metrics and Targets",
        "mapping_status": "enhanced",
        "notes": (
            "IFRS S2 para 29 specifies 7 cross-industry metrics beyond TCFD. "
            "Industry-specific metrics also required."
        ),
    },
    "mt_b": {
        "ifrs_s2_paragraph": "29(a)",
        "ifrs_s2_topic": "GHG Emissions",
        "mapping_status": "enhanced",
        "notes": (
            "IFRS S2 requires Scope 3 for all entities (not just if appropriate). "
            "GHG Protocol alignment is mandatory."
        ),
    },
    "mt_c": {
        "ifrs_s2_paragraph": "33-36",
        "ifrs_s2_topic": "Targets",
        "mapping_status": "enhanced",
        "notes": (
            "IFRS S2 paras 33-36 require detailed target disclosure including "
            "SBTi alignment and progress tracking."
        ),
    },
}


# ---------------------------------------------------------------------------
# Pre-Built Scenario Parameter Library (Full Detail)
# ---------------------------------------------------------------------------

SCENARIO_LIBRARY: Dict[ScenarioType, Dict[str, Any]] = {
    ScenarioType.IEA_NZE: {
        "name": "IEA Net Zero Emissions by 2050",
        "temperature_outcome": TemperatureOutcome.BELOW_1_5C,
        "description": (
            "The IEA NZE scenario charts a narrow but achievable pathway for the "
            "global energy sector to reach net zero CO2 emissions by 2050. "
            "Requires immediate and massive deployment of clean energy technologies."
        ),
        "carbon_price_trajectory": {
            2025: Decimal("75"), 2030: Decimal("130"), 2035: Decimal("170"),
            2040: Decimal("205"), 2045: Decimal("230"), 2050: Decimal("250"),
        },
        "energy_mix_trajectory": {
            2025: {"renewable_pct": 30, "fossil_pct": 65, "nuclear_pct": 5},
            2030: {"renewable_pct": 50, "fossil_pct": 40, "nuclear_pct": 10},
            2040: {"renewable_pct": 75, "fossil_pct": 12, "nuclear_pct": 13},
            2050: {"renewable_pct": 90, "fossil_pct": 2, "nuclear_pct": 8},
        },
        "temperature_projection": {
            2030: Decimal("1.3"), 2040: Decimal("1.4"),
            2050: Decimal("1.5"), 2100: Decimal("1.4"),
        },
    },
    ScenarioType.IEA_APS: {
        "name": "IEA Announced Pledges Scenario",
        "temperature_outcome": TemperatureOutcome.AROUND_2C,
        "description": (
            "The APS assumes that all governments fully implement their "
            "announced climate pledges and NDCs on time and in full."
        ),
        "carbon_price_trajectory": {
            2025: Decimal("50"), 2030: Decimal("90"), 2035: Decimal("125"),
            2040: Decimal("155"), 2045: Decimal("180"), 2050: Decimal("200"),
        },
        "energy_mix_trajectory": {
            2025: {"renewable_pct": 28, "fossil_pct": 67, "nuclear_pct": 5},
            2030: {"renewable_pct": 40, "fossil_pct": 52, "nuclear_pct": 8},
            2040: {"renewable_pct": 58, "fossil_pct": 32, "nuclear_pct": 10},
            2050: {"renewable_pct": 70, "fossil_pct": 18, "nuclear_pct": 12},
        },
        "temperature_projection": {
            2030: Decimal("1.3"), 2040: Decimal("1.5"),
            2050: Decimal("1.7"), 2100: Decimal("1.9"),
        },
    },
    ScenarioType.IEA_STEPS: {
        "name": "IEA Stated Policies Scenario",
        "temperature_outcome": TemperatureOutcome.ABOVE_2_5C,
        "description": (
            "STEPS reflects only existing policies and measures as of mid-2023. "
            "It does not assume that governments will meet their announced pledges."
        ),
        "carbon_price_trajectory": {
            2025: Decimal("25"), 2030: Decimal("40"), 2035: Decimal("48"),
            2040: Decimal("55"), 2045: Decimal("60"), 2050: Decimal("65"),
        },
        "energy_mix_trajectory": {
            2025: {"renewable_pct": 25, "fossil_pct": 70, "nuclear_pct": 5},
            2030: {"renewable_pct": 32, "fossil_pct": 62, "nuclear_pct": 6},
            2040: {"renewable_pct": 40, "fossil_pct": 52, "nuclear_pct": 8},
            2050: {"renewable_pct": 50, "fossil_pct": 42, "nuclear_pct": 8},
        },
        "temperature_projection": {
            2030: Decimal("1.4"), 2040: Decimal("1.8"),
            2050: Decimal("2.1"), 2100: Decimal("2.5"),
        },
    },
    ScenarioType.NGFS_CURRENT_POLICIES: {
        "name": "NGFS Current Policies",
        "temperature_outcome": TemperatureOutcome.ABOVE_3C,
        "description": (
            "Only currently implemented policies are preserved. No new climate "
            "policies are assumed. Results in high physical risks."
        ),
        "carbon_price_trajectory": {
            2025: Decimal("10"), 2030: Decimal("15"), 2035: Decimal("18"),
            2040: Decimal("20"), 2045: Decimal("22"), 2050: Decimal("25"),
        },
        "energy_mix_trajectory": {
            2025: {"renewable_pct": 22, "fossil_pct": 73, "nuclear_pct": 5},
            2030: {"renewable_pct": 25, "fossil_pct": 70, "nuclear_pct": 5},
            2040: {"renewable_pct": 30, "fossil_pct": 64, "nuclear_pct": 6},
            2050: {"renewable_pct": 35, "fossil_pct": 59, "nuclear_pct": 6},
        },
        "temperature_projection": {
            2030: Decimal("1.4"), 2040: Decimal("1.9"),
            2050: Decimal("2.5"), 2100: Decimal("3.2"),
        },
    },
    ScenarioType.NGFS_DELAYED_TRANSITION: {
        "name": "NGFS Delayed Transition",
        "temperature_outcome": TemperatureOutcome.AROUND_2C,
        "description": (
            "Assumes global annual emissions do not decrease until 2030. "
            "Strong policies are needed after 2030, resulting in a disorderly "
            "transition with higher transition risks."
        ),
        "carbon_price_trajectory": {
            2025: Decimal("12"), 2030: Decimal("25"), 2035: Decimal("120"),
            2040: Decimal("220"), 2045: Decimal("300"), 2050: Decimal("350"),
        },
        "energy_mix_trajectory": {
            2025: {"renewable_pct": 23, "fossil_pct": 72, "nuclear_pct": 5},
            2030: {"renewable_pct": 28, "fossil_pct": 66, "nuclear_pct": 6},
            2040: {"renewable_pct": 55, "fossil_pct": 35, "nuclear_pct": 10},
            2050: {"renewable_pct": 72, "fossil_pct": 16, "nuclear_pct": 12},
        },
        "temperature_projection": {
            2030: Decimal("1.5"), 2040: Decimal("1.8"),
            2050: Decimal("2.0"), 2100: Decimal("1.9"),
        },
    },
    ScenarioType.NGFS_BELOW_2C: {
        "name": "NGFS Below 2 Degrees C",
        "temperature_outcome": TemperatureOutcome.AROUND_2C,
        "description": (
            "Gradual and immediate climate policy strengthening starting now. "
            "An orderly transition that limits warming to below 2 degrees C."
        ),
        "carbon_price_trajectory": {
            2025: Decimal("55"), 2030: Decimal("100"), 2035: Decimal("135"),
            2040: Decimal("160"), 2045: Decimal("185"), 2050: Decimal("200"),
        },
        "energy_mix_trajectory": {
            2025: {"renewable_pct": 28, "fossil_pct": 67, "nuclear_pct": 5},
            2030: {"renewable_pct": 42, "fossil_pct": 50, "nuclear_pct": 8},
            2040: {"renewable_pct": 60, "fossil_pct": 28, "nuclear_pct": 12},
            2050: {"renewable_pct": 75, "fossil_pct": 13, "nuclear_pct": 12},
        },
        "temperature_projection": {
            2030: Decimal("1.3"), 2040: Decimal("1.5"),
            2050: Decimal("1.7"), 2100: Decimal("1.8"),
        },
    },
    ScenarioType.NGFS_DIVERGENT_NET_ZERO: {
        "name": "NGFS Divergent Net Zero",
        "temperature_outcome": TemperatureOutcome.BELOW_1_5C,
        "description": (
            "Reaches net zero by 2050 but with higher costs due to divergent "
            "policies across sectors. Some sectors decarbonize rapidly while "
            "others lag, creating uneven transition risks."
        ),
        "carbon_price_trajectory": {
            2025: Decimal("65"), 2030: Decimal("120"), 2035: Decimal("160"),
            2040: Decimal("195"), 2045: Decimal("215"), 2050: Decimal("230"),
        },
        "energy_mix_trajectory": {
            2025: {"renewable_pct": 28, "fossil_pct": 67, "nuclear_pct": 5},
            2030: {"renewable_pct": 45, "fossil_pct": 46, "nuclear_pct": 9},
            2040: {"renewable_pct": 65, "fossil_pct": 22, "nuclear_pct": 13},
            2050: {"renewable_pct": 82, "fossil_pct": 6, "nuclear_pct": 12},
        },
        "temperature_projection": {
            2030: Decimal("1.3"), 2040: Decimal("1.4"),
            2050: Decimal("1.5"), 2100: Decimal("1.5"),
        },
    },
    ScenarioType.CUSTOM: {
        "name": "Custom Scenario",
        "temperature_outcome": TemperatureOutcome.AROUND_2C,
        "description": (
            "User-defined scenario with custom parameters. "
            "All trajectories and assumptions are specified by the organization."
        ),
        "carbon_price_trajectory": {},
        "energy_mix_trajectory": {},
        "temperature_projection": {},
    },
}


# ---------------------------------------------------------------------------
# Sector Transition Risk Profiles
# ---------------------------------------------------------------------------

SECTOR_TRANSITION_PROFILES: Dict[SectorType, Dict[str, str]] = {
    SectorType.ENERGY: {
        "transition_exposure": "very_high",
        "stranding_risk": "high",
        "key_drivers": "carbon_pricing, regulation, technology_substitution",
        "decarbonization_pathway": "renewables, CCS, hydrogen",
    },
    SectorType.TRANSPORTATION: {
        "transition_exposure": "high",
        "stranding_risk": "medium",
        "key_drivers": "regulation, technology_substitution, demand_shift",
        "decarbonization_pathway": "electrification, sustainable fuels, modal shift",
    },
    SectorType.MATERIALS_BUILDINGS: {
        "transition_exposure": "high",
        "stranding_risk": "medium",
        "key_drivers": "carbon_pricing, regulation, technology_substitution",
        "decarbonization_pathway": "electrification, hydrogen, CCUS, circularity",
    },
    SectorType.AGRICULTURE_FOOD_FOREST: {
        "transition_exposure": "medium",
        "stranding_risk": "low",
        "key_drivers": "regulation, demand_shift, supply_chain",
        "decarbonization_pathway": "precision farming, methane reduction, soil carbon",
    },
    SectorType.BANKING: {
        "transition_exposure": "medium",
        "stranding_risk": "low",
        "key_drivers": "regulation, stakeholder_sentiment, litigation",
        "decarbonization_pathway": "portfolio alignment, green finance, exclusion",
    },
    SectorType.INSURANCE: {
        "transition_exposure": "medium",
        "stranding_risk": "low",
        "key_drivers": "regulation, stakeholder_sentiment, physical_risk_pricing",
        "decarbonization_pathway": "underwriting criteria, climate risk models",
    },
    SectorType.ASSET_OWNERS: {
        "transition_exposure": "medium",
        "stranding_risk": "low",
        "key_drivers": "regulation, stakeholder_sentiment, fiduciary_duty",
        "decarbonization_pathway": "portfolio decarbonization, engagement, divestment",
    },
    SectorType.ASSET_MANAGERS: {
        "transition_exposure": "medium",
        "stranding_risk": "low",
        "key_drivers": "regulation, stakeholder_sentiment, demand_shift",
        "decarbonization_pathway": "ESG integration, engagement, stewardship",
    },
    SectorType.CONSUMER_GOODS: {
        "transition_exposure": "medium",
        "stranding_risk": "low",
        "key_drivers": "demand_shift, supply_chain, stakeholder_sentiment",
        "decarbonization_pathway": "sustainable sourcing, packaging, circularity",
    },
    SectorType.TECHNOLOGY_MEDIA: {
        "transition_exposure": "low",
        "stranding_risk": "low",
        "key_drivers": "regulation, stakeholder_sentiment",
        "decarbonization_pathway": "renewable energy, efficiency, circular design",
    },
    SectorType.HEALTHCARE: {
        "transition_exposure": "low",
        "stranding_risk": "low",
        "key_drivers": "regulation, stakeholder_sentiment",
        "decarbonization_pathway": "energy efficiency, supply chain, waste reduction",
    },
}


# ---------------------------------------------------------------------------
# Hazard Exposure Matrices by RCP/SSP Pathway
# ---------------------------------------------------------------------------

HAZARD_EXPOSURE_MATRICES: Dict[str, Dict[PhysicalHazard, Dict[str, int]]] = {
    "ssp1_26": {
        PhysicalHazard.CYCLONE: {"baseline": 2, "2030": 2, "2050": 3},
        PhysicalHazard.FLOOD: {"baseline": 2, "2030": 2, "2050": 3},
        PhysicalHazard.WILDFIRE: {"baseline": 2, "2030": 2, "2050": 3},
        PhysicalHazard.HEATWAVE: {"baseline": 2, "2030": 3, "2050": 3},
        PhysicalHazard.DROUGHT: {"baseline": 2, "2030": 2, "2050": 3},
        PhysicalHazard.SEA_LEVEL_RISE: {"baseline": 1, "2030": 2, "2050": 2},
        PhysicalHazard.TEMPERATURE_RISE: {"baseline": 2, "2030": 2, "2050": 3},
        PhysicalHazard.WATER_STRESS: {"baseline": 2, "2030": 2, "2050": 3},
        PhysicalHazard.PRECIPITATION_CHANGE: {"baseline": 2, "2030": 2, "2050": 3},
        PhysicalHazard.ECOSYSTEM_DEGRADATION: {"baseline": 2, "2030": 2, "2050": 3},
    },
    "ssp2_45": {
        PhysicalHazard.CYCLONE: {"baseline": 2, "2030": 3, "2050": 4},
        PhysicalHazard.FLOOD: {"baseline": 2, "2030": 3, "2050": 4},
        PhysicalHazard.WILDFIRE: {"baseline": 2, "2030": 3, "2050": 4},
        PhysicalHazard.HEATWAVE: {"baseline": 2, "2030": 3, "2050": 4},
        PhysicalHazard.DROUGHT: {"baseline": 2, "2030": 3, "2050": 4},
        PhysicalHazard.SEA_LEVEL_RISE: {"baseline": 1, "2030": 2, "2050": 3},
        PhysicalHazard.TEMPERATURE_RISE: {"baseline": 2, "2030": 3, "2050": 4},
        PhysicalHazard.WATER_STRESS: {"baseline": 2, "2030": 3, "2050": 4},
        PhysicalHazard.PRECIPITATION_CHANGE: {"baseline": 2, "2030": 3, "2050": 3},
        PhysicalHazard.ECOSYSTEM_DEGRADATION: {"baseline": 2, "2030": 3, "2050": 4},
    },
    "ssp5_85": {
        PhysicalHazard.CYCLONE: {"baseline": 2, "2030": 3, "2050": 5},
        PhysicalHazard.FLOOD: {"baseline": 2, "2030": 4, "2050": 5},
        PhysicalHazard.WILDFIRE: {"baseline": 2, "2030": 4, "2050": 5},
        PhysicalHazard.HEATWAVE: {"baseline": 2, "2030": 4, "2050": 5},
        PhysicalHazard.DROUGHT: {"baseline": 2, "2030": 4, "2050": 5},
        PhysicalHazard.SEA_LEVEL_RISE: {"baseline": 1, "2030": 3, "2050": 5},
        PhysicalHazard.TEMPERATURE_RISE: {"baseline": 2, "2030": 4, "2050": 5},
        PhysicalHazard.WATER_STRESS: {"baseline": 2, "2030": 4, "2050": 5},
        PhysicalHazard.PRECIPITATION_CHANGE: {"baseline": 2, "2030": 3, "2050": 4},
        PhysicalHazard.ECOSYSTEM_DEGRADATION: {"baseline": 2, "2030": 4, "2050": 5},
    },
}


# ---------------------------------------------------------------------------
# Regulatory Template Jurisdictions
# ---------------------------------------------------------------------------

REGULATORY_JURISDICTIONS: List[str] = [
    "UK", "EU", "US", "JP", "SG", "HK", "AU", "NZ", "CA", "CH",
]


# ---------------------------------------------------------------------------
# MRV Agent to TCFD Scope Mapping
# ---------------------------------------------------------------------------

MRV_AGENT_TO_TCFD_SCOPE: Dict[str, Dict[str, str]] = {
    "MRV-001": {"scope": "scope_1", "name": "Stationary Combustion"},
    "MRV-002": {"scope": "scope_1", "name": "Refrigerants & F-Gas"},
    "MRV-003": {"scope": "scope_1", "name": "Mobile Combustion"},
    "MRV-004": {"scope": "scope_1", "name": "Process Emissions"},
    "MRV-005": {"scope": "scope_1", "name": "Fugitive Emissions"},
    "MRV-006": {"scope": "scope_1", "name": "Land Use Emissions"},
    "MRV-007": {"scope": "scope_1", "name": "Waste Treatment Emissions"},
    "MRV-008": {"scope": "scope_1", "name": "Agricultural Emissions"},
    "MRV-009": {"scope": "scope_2", "name": "Scope 2 Location-Based"},
    "MRV-010": {"scope": "scope_2", "name": "Scope 2 Market-Based"},
    "MRV-011": {"scope": "scope_2", "name": "Steam/Heat Purchase"},
    "MRV-012": {"scope": "scope_2", "name": "Cooling Purchase"},
    "MRV-013": {"scope": "scope_2", "name": "Dual Reporting Reconciliation"},
    "MRV-014": {"scope": "scope_3", "name": "Purchased Goods & Services"},
    "MRV-015": {"scope": "scope_3", "name": "Capital Goods"},
    "MRV-016": {"scope": "scope_3", "name": "Fuel & Energy Activities"},
    "MRV-017": {"scope": "scope_3", "name": "Upstream Transportation"},
    "MRV-018": {"scope": "scope_3", "name": "Waste Generated"},
    "MRV-019": {"scope": "scope_3", "name": "Business Travel"},
    "MRV-020": {"scope": "scope_3", "name": "Employee Commuting"},
    "MRV-021": {"scope": "scope_3", "name": "Upstream Leased Assets"},
    "MRV-022": {"scope": "scope_3", "name": "Downstream Transportation"},
    "MRV-023": {"scope": "scope_3", "name": "Processing of Sold Products"},
    "MRV-024": {"scope": "scope_3", "name": "Use of Sold Products"},
    "MRV-025": {"scope": "scope_3", "name": "End-of-Life Treatment"},
    "MRV-026": {"scope": "scope_3", "name": "Downstream Leased Assets"},
    "MRV-027": {"scope": "scope_3", "name": "Franchises"},
    "MRV-028": {"scope": "scope_3", "name": "Investments"},
    "MRV-029": {"scope": "cross_cutting", "name": "Scope 3 Category Mapper"},
    "MRV-030": {"scope": "cross_cutting", "name": "Audit Trail & Lineage"},
}


# ---------------------------------------------------------------------------
# Main Configuration Class
# ---------------------------------------------------------------------------

class TCFDAppConfig(BaseSettings):
    """
    GL-TCFD-APP v1.0 platform configuration.

    All settings can be overridden via environment variables prefixed
    with ``TCFD_APP_``.  For example ``TCFD_APP_DEFAULT_SCENARIO_TYPE``
    maps to ``default_scenario_type``.

    Example:
        >>> config = TCFDAppConfig()
        >>> config.app_name
        'GL-TCFD-APP'
        >>> config.enable_monte_carlo
        True
        >>> config.monte_carlo_iterations
        10000
    """

    model_config = {"env_prefix": "TCFD_APP_"}

    # -- Application Metadata -----------------------------------------------
    app_name: str = Field(
        default="GL-TCFD-APP",
        description="Application display name",
    )
    version: str = Field(
        default="1.0.0",
        description="Semantic version of the application",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode with verbose logging",
    )

    # -- Scenario Defaults --------------------------------------------------
    default_scenario: ScenarioType = Field(
        default=ScenarioType.IEA_NZE,
        description="Default scenario type for new analyses",
    )
    scenario_time_horizons: List[TimeHorizon] = Field(
        default=[
            TimeHorizon.SHORT_TERM,
            TimeHorizon.MEDIUM_TERM,
            TimeHorizon.LONG_TERM,
        ],
        description="Default time horizons for scenario analysis",
    )
    carbon_price_escalation_rate: Decimal = Field(
        default=Decimal("0.05"),
        ge=Decimal("0.0"),
        le=Decimal("0.50"),
        description="Annual escalation rate for carbon prices between defined years",
    )

    # -- Physical Risk Configuration ----------------------------------------
    physical_risk_data_source: str = Field(
        default="ssp2_45",
        description="Default RCP/SSP scenario for physical risk assessment",
    )

    # -- Monte Carlo Simulation ---------------------------------------------
    enable_monte_carlo: bool = Field(
        default=True,
        description="Enable Monte Carlo simulation for scenario analysis",
    )
    monte_carlo_iterations: int = Field(
        default=10_000,
        ge=1_000,
        le=1_000_000,
        description="Number of Monte Carlo iterations for uncertainty quantification",
    )

    # -- Financial Parameters -----------------------------------------------
    default_discount_rate: Decimal = Field(
        default=Decimal("0.08"),
        ge=Decimal("0.01"),
        le=Decimal("0.30"),
        description="Default discount rate for NPV calculations",
    )
    currency: str = Field(
        default="USD",
        description="Default currency for all monetary values",
    )
    financial_projection_years: int = Field(
        default=30,
        ge=5,
        le=100,
        description="Number of years for financial projections",
    )

    # -- SBTi Configuration -------------------------------------------------
    sbti_validation_enabled: bool = Field(
        default=True,
        description="Enable SBTi target alignment validation",
    )

    # -- MRV Agent Integration ----------------------------------------------
    mrv_agent_base_url: str = Field(
        default="http://localhost:8000/api/v1/mrv",
        description="Base URL for MRV agent API endpoints",
    )
    mrv_agent_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Timeout for individual MRV agent calls (seconds)",
    )
    mrv_agent_retry_count: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of retries for failed MRV agent calls",
    )

    # -- Reporting Year -----------------------------------------------------
    reporting_year: int = Field(
        default=2025,
        ge=1990,
        le=2100,
        description="Current reporting year",
    )

    # -- Supported Currencies -----------------------------------------------
    supported_currencies: List[str] = Field(
        default=[
            "USD", "EUR", "GBP", "JPY", "SGD",
            "HKD", "AUD", "NZD", "CAD", "CHF",
        ],
        description="Supported currencies for financial impact analysis",
    )

    # -- Governance Maturity Weights ----------------------------------------
    maturity_weights: Dict[str, Decimal] = Field(
        default={
            "board_oversight": Decimal("0.15"),
            "management_roles": Decimal("0.12"),
            "climate_competency": Decimal("0.12"),
            "meeting_frequency": Decimal("0.08"),
            "reporting_structure": Decimal("0.12"),
            "incentive_alignment": Decimal("0.13"),
            "risk_integration": Decimal("0.15"),
            "strategy_integration": Decimal("0.13"),
        },
        description="Weights for governance maturity dimension scoring (must sum to 1.0)",
    )

    # -- Report Generation --------------------------------------------------
    default_report_format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Default report export format",
    )
    report_storage_path: str = Field(
        default="reports/tcfd/",
        description="Path prefix for generated reports",
    )

    # -- Logging ------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
