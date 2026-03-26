"""
PACK-043 Scope 3 Complete Pack - Configuration Manager

This module implements the Scope3CompleteConfig and PackConfig classes that
load, merge, and validate all configuration for the Scope 3 Complete Pack.
It provides comprehensive Pydantic v2 models for an enterprise-grade Scope 3
value chain emissions management solution with lifecycle assessment, SBTi
pathway alignment, scenario analysis, multi-entity consolidation, supplier
programme management, climate risk quantification, and assurance readiness.

PACK-043 extends PACK-042 (Scope 3 Starter) with enterprise capabilities:
    - Maturity progression: Level 1 (Screening) through Level 5 (Verified)
    - LCA integration: ecoinvent, GaBi, ELCD, custom databases
    - SBTi targets: Absolute contraction, SDA, economic intensity
    - Scenario analysis: MACC, what-if, technology pathway, Paris alignment
    - Multi-entity: Equity share, operational control, financial control
    - Supplier programme: Target reduction, incentive models, monitoring
    - Climate risk: Transition, physical, opportunity with carbon pricing
    - Base-year recalculation: Significance thresholds and automated triggers
    - PCAF: Six asset classes with PCAF data quality scoring
    - Assurance: ISAE 3410 limited and reasonable assurance readiness
    - Enterprise dashboard: Investor-grade with real-time monitoring

Maturity Levels:
    - LEVEL_1_SCREENING: Spend-based EEIO only, top 5 categories
    - LEVEL_2_STARTER: Mixed spend/average-data, 10+ categories (PACK-042)
    - LEVEL_3_INTERMEDIATE: Average-data dominant, supplier engagement begun
    - LEVEL_4_ADVANCED: Supplier-specific dominant, LCA, scenario analysis
    - LEVEL_5_VERIFIED: External verification ready, ISAE 3410

Tier Upgrade Strategies:
    - ROI_OPTIMIZED: Upgrade highest-emission categories first
    - MATERIALITY_FIRST: Upgrade most material categories first
    - BUDGET_CONSTRAINED: Upgrade within fixed annual budget
    - SECTOR_SPECIFIC: Follow sector decarbonisation pathway

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (sector-specific defaults)
    3. PACK-042 inherited configuration (if upgrading)
    4. Environment overrides (SCOPE3_COMPLETE_* environment variables)
    5. Explicit runtime overrides

Prerequisites:
    - PACK-042 (Scope 3 Starter) is REQUIRED as a dependency
    - PACK-041 (Scope 1-2 Complete) is RECOMMENDED for Cat 3 and ratios

Regulatory Context:
    - GHG Protocol Corporate Value Chain (Scope 3) Standard (2011)
    - GHG Protocol Technical Guidance for Calculating Scope 3 (2013)
    - ISO 14064-1:2018 (Categories 3-6)
    - EU CSRD / ESRS E1 (Scope 3 phase-in)
    - CDP Climate Change 2026
    - SBTi Corporate Net-Zero Standard v1.1
    - SBTi FLAG Guidance
    - TCFD / ISSB IFRS S2
    - US SEC Climate Disclosure Rules
    - California SB 253
    - PCAF Global GHG Accounting Standard v3
    - ISAE 3410

Presets:
    enterprise_manufacturing / financial_institution / retail_chain /
    technology_enterprise / energy_company / food_beverage /
    multi_entity_group / advanced_reporter

Example:
    >>> config = PackConfig.from_preset("enterprise_manufacturing")
    >>> print(config.pack.maturity.target_level)
    MaturityLevel.LEVEL_4_ADVANCED
    >>> print(config.pack.sbti.target_type)
    SBTiTargetType.SDA
    >>> print(config.pack.consolidation.approach)
    ConsolidationApproach.OPERATIONAL_CONTROL
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
# Enums - Scope 3 Complete enumeration types (12+ enums)
# =============================================================================


class MaturityLevel(str, Enum):
    """Scope 3 maturity level for progressive capability unlocking."""

    LEVEL_1_SCREENING = "LEVEL_1_SCREENING"
    LEVEL_2_STARTER = "LEVEL_2_STARTER"
    LEVEL_3_INTERMEDIATE = "LEVEL_3_INTERMEDIATE"
    LEVEL_4_ADVANCED = "LEVEL_4_ADVANCED"
    LEVEL_5_VERIFIED = "LEVEL_5_VERIFIED"


class TierUpgradeStrategy(str, Enum):
    """Strategy for upgrading methodology tiers across categories."""

    ROI_OPTIMIZED = "ROI_OPTIMIZED"
    MATERIALITY_FIRST = "MATERIALITY_FIRST"
    BUDGET_CONSTRAINED = "BUDGET_CONSTRAINED"
    SECTOR_SPECIFIC = "SECTOR_SPECIFIC"


class LCADatabase(str, Enum):
    """Lifecycle assessment database for product-level emission factors."""

    ECOINVENT = "ECOINVENT"
    GABI = "GABI"
    ELCD = "ELCD"
    CUSTOM = "CUSTOM"


class LifecycleStage(str, Enum):
    """Product lifecycle stage for LCA boundary definition."""

    RAW_MATERIAL = "RAW_MATERIAL"
    MANUFACTURING = "MANUFACTURING"
    DISTRIBUTION = "DISTRIBUTION"
    USE_PHASE = "USE_PHASE"
    END_OF_LIFE = "END_OF_LIFE"


class ConsolidationApproach(str, Enum):
    """GHG Protocol organisational boundary consolidation approach."""

    EQUITY_SHARE = "EQUITY_SHARE"
    OPERATIONAL_CONTROL = "OPERATIONAL_CONTROL"
    FINANCIAL_CONTROL = "FINANCIAL_CONTROL"


class ScenarioType(str, Enum):
    """Scenario analysis type for Scope 3 reduction pathway modelling."""

    MACC = "MACC"
    WHAT_IF = "WHAT_IF"
    TECHNOLOGY_PATHWAY = "TECHNOLOGY_PATHWAY"
    SUPPLIER_PROGRAMME = "SUPPLIER_PROGRAMME"
    PARIS_ALIGNMENT = "PARIS_ALIGNMENT"


class SBTiTargetType(str, Enum):
    """SBTi target-setting approach for Scope 3."""

    ABSOLUTE_CONTRACTION = "ABSOLUTE_CONTRACTION"
    SDA = "SDA"
    ECONOMIC_INTENSITY = "ECONOMIC_INTENSITY"


class SBTiTimeframe(str, Enum):
    """SBTi target timeframe classification."""

    NEAR_TERM = "NEAR_TERM"
    LONG_TERM = "LONG_TERM"


class RiskType(str, Enum):
    """Climate risk type per TCFD/ISSB classification."""

    TRANSITION = "TRANSITION"
    PHYSICAL = "PHYSICAL"
    OPPORTUNITY = "OPPORTUNITY"


class PCAFAssetClass(str, Enum):
    """PCAF asset class for Category 15 financed emissions."""

    LISTED_EQUITY = "LISTED_EQUITY"
    CORPORATE_BONDS = "CORPORATE_BONDS"
    UNLISTED_EQUITY = "UNLISTED_EQUITY"
    PROJECT_FINANCE = "PROJECT_FINANCE"
    COMMERCIAL_RE = "COMMERCIAL_RE"
    MORTGAGES = "MORTGAGES"


class SectorFocus(str, Enum):
    """Sector focus for enterprise preset selection."""

    FINANCIAL_SERVICES = "FINANCIAL_SERVICES"
    RETAIL = "RETAIL"
    MANUFACTURING = "MANUFACTURING"
    TECHNOLOGY = "TECHNOLOGY"
    ENERGY = "ENERGY"
    FOOD_BEVERAGE = "FOOD_BEVERAGE"
    MULTI_ENTITY = "MULTI_ENTITY"
    GENERAL = "GENERAL"


class AssuranceLevel(str, Enum):
    """External assurance level per ISAE 3410."""

    LIMITED = "LIMITED"
    REASONABLE = "REASONABLE"


# Scope 3 category enum (inherited from PACK-042 for consistency)
class Scope3Category(str, Enum):
    """GHG Protocol Scope 3 category classification (15 categories)."""

    CAT_1 = "CAT_1"
    CAT_2 = "CAT_2"
    CAT_3 = "CAT_3"
    CAT_4 = "CAT_4"
    CAT_5 = "CAT_5"
    CAT_6 = "CAT_6"
    CAT_7 = "CAT_7"
    CAT_8 = "CAT_8"
    CAT_9 = "CAT_9"
    CAT_10 = "CAT_10"
    CAT_11 = "CAT_11"
    CAT_12 = "CAT_12"
    CAT_13 = "CAT_13"
    CAT_14 = "CAT_14"
    CAT_15 = "CAT_15"


class MethodologyTier(str, Enum):
    """Scope 3 methodology tier for emission quantification."""

    SPEND_BASED = "SPEND_BASED"
    AVERAGE_DATA = "AVERAGE_DATA"
    SUPPLIER_SPECIFIC = "SUPPLIER_SPECIFIC"
    HYBRID = "HYBRID"


class OutputFormat(str, Enum):
    """Output format for Scope 3 reports."""

    MARKDOWN = "MARKDOWN"
    HTML = "HTML"
    JSON = "JSON"
    CSV = "CSV"
    XLSX = "XLSX"
    PDF = "PDF"


class FrameworkType(str, Enum):
    """Regulatory and reporting framework identifiers for Scope 3."""

    GHG_PROTOCOL = "GHG_PROTOCOL"
    ESRS_E1 = "ESRS_E1"
    CDP = "CDP"
    SBTI = "SBTI"
    SEC = "SEC"
    SB_253 = "SB_253"
    ISO_14064 = "ISO_14064"
    PCAF = "PCAF"
    TCFD_ISSB = "TCFD_ISSB"
    ISAE_3410 = "ISAE_3410"


# =============================================================================
# Reference Data Constants
# =============================================================================

MATURITY_LEVEL_INFO: Dict[str, Dict[str, Any]] = {
    "LEVEL_1_SCREENING": {
        "name": "Level 1 - Screening",
        "description": "Spend-based EEIO screening only, top 5 categories",
        "typical_pack": "PACK-042 basic",
        "methodology": "Spend-based only",
        "data_quality_target": 5,
        "assurance": "None",
    },
    "LEVEL_2_STARTER": {
        "name": "Level 2 - Starter",
        "description": "Mixed spend and average-data, 10+ categories",
        "typical_pack": "PACK-042 full",
        "methodology": "Spend-based + Average-data",
        "data_quality_target": 4,
        "assurance": "None or limited",
    },
    "LEVEL_3_INTERMEDIATE": {
        "name": "Level 3 - Intermediate",
        "description": "Average-data dominant, supplier engagement begun, LCA for Cat 1",
        "typical_pack": "PACK-043 entry",
        "methodology": "Average-data + some supplier-specific",
        "data_quality_target": 3,
        "assurance": "Limited",
    },
    "LEVEL_4_ADVANCED": {
        "name": "Level 4 - Advanced",
        "description": "Supplier-specific dominant, LCA integrated, scenario analysis",
        "typical_pack": "PACK-043 full",
        "methodology": "Supplier-specific + LCA + hybrid",
        "data_quality_target": 2,
        "assurance": "Limited or reasonable",
    },
    "LEVEL_5_VERIFIED": {
        "name": "Level 5 - Verified",
        "description": "External verification ready, ISAE 3410, continuous monitoring",
        "typical_pack": "PACK-043 full + verification",
        "methodology": "Supplier-specific dominant + continuous",
        "data_quality_target": 1.5,
        "assurance": "Reasonable (ISAE 3410)",
    },
}


AVAILABLE_PRESETS: Dict[str, str] = {
    "enterprise_manufacturing": (
        "Large manufacturer with complex multi-tier supply chain, circular economy "
        "initiatives, LCA integration, and SBTi SDA pathway"
    ),
    "financial_institution": (
        "Bank or insurer with PCAF Category 15 financed emissions across six asset "
        "classes, investment portfolio decarbonisation, and TCFD scenario analysis"
    ),
    "retail_chain": (
        "Retail chain with last-mile logistics, packaging lifecycle, product returns, "
        "and supplier CDP integration"
    ),
    "technology_enterprise": (
        "Technology company with cloud infrastructure, hardware lifecycle, SaaS "
        "product use-phase, and Scope 3 intensity targets"
    ),
    "energy_company": (
        "Energy sector with Cat 3 and Cat 11 dominant, upstream fuel production, "
        "downstream sold energy use, and Paris alignment scenarios"
    ),
    "food_beverage": (
        "Food and beverage with agricultural supply chain, FLAG emissions, land use "
        "change, and deforestation-free commitments"
    ),
    "multi_entity_group": (
        "Corporate group with 50+ entities, joint ventures, franchises, equity share "
        "and operational control consolidation"
    ),
    "advanced_reporter": (
        "Generic advanced reporter with 2+ years of Scope 3 experience upgrading "
        "from PACK-042 to enterprise-grade capabilities"
    ),
}


# =============================================================================
# Pydantic Sub-Config Models (14+ sub-config models)
# =============================================================================


class MaturityConfig(BaseModel):
    """Configuration for Scope 3 maturity level progression.

    Defines the target maturity level, budget, timeline, and ROI threshold
    for upgrading methodology tiers across Scope 3 categories. Maturity
    progression drives which PACK-043 capabilities are unlocked.
    """

    target_level: MaturityLevel = Field(
        MaturityLevel.LEVEL_4_ADVANCED,
        description="Target maturity level (LEVEL_1 through LEVEL_5)",
    )
    current_level: MaturityLevel = Field(
        MaturityLevel.LEVEL_2_STARTER,
        description="Current maturity level baseline",
    )
    upgrade_strategy: TierUpgradeStrategy = Field(
        TierUpgradeStrategy.ROI_OPTIMIZED,
        description="Strategy for upgrading methodology tiers",
    )
    budget_usd: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Annual budget (USD) for Scope 3 data improvement",
    )
    timeline_months: int = Field(
        24,
        ge=6,
        le=60,
        description="Timeline (months) to reach target maturity level",
    )
    roi_threshold_pct: float = Field(
        15.0,
        ge=0.0,
        le=100.0,
        description="Minimum ROI threshold (%) for methodology tier upgrades",
    )
    milestone_tracking: bool = Field(
        True,
        description="Enable milestone tracking for maturity progression",
    )
    auto_upgrade_enabled: bool = Field(
        False,
        description="Automatically upgrade category tiers when data quality improves",
    )


class LCAConfig(BaseModel):
    """Configuration for lifecycle assessment integration.

    Controls LCA database selection, product type coverage, lifecycle
    stages, and sensitivity analysis parameters for product-level
    emission factors in Categories 1, 2, 10, 11, and 12.
    """

    enabled: bool = Field(
        True,
        description="Enable LCA database integration for product-level factors",
    )
    database: LCADatabase = Field(
        LCADatabase.ECOINVENT,
        description="Primary LCA database (ECOINVENT, GABI, ELCD, CUSTOM)",
    )
    database_version: str = Field(
        "3.10",
        description="LCA database version identifier",
    )
    secondary_database: Optional[LCADatabase] = Field(
        None,
        description="Secondary LCA database for cross-referencing",
    )
    product_types: List[str] = Field(
        default_factory=lambda: [
            "raw_materials",
            "components",
            "packaging",
            "fuels",
            "chemicals",
        ],
        description="Product types covered by LCA integration",
    )
    lifecycle_stages: List[LifecycleStage] = Field(
        default_factory=lambda: [
            LifecycleStage.RAW_MATERIAL,
            LifecycleStage.MANUFACTURING,
            LifecycleStage.DISTRIBUTION,
            LifecycleStage.USE_PHASE,
            LifecycleStage.END_OF_LIFE,
        ],
        description="Lifecycle stages included in LCA boundary",
    )
    sensitivity_params: Dict[str, float] = Field(
        default_factory=lambda: {
            "emission_factor_variation_pct": 20.0,
            "transport_distance_variation_pct": 15.0,
            "energy_mix_variation_pct": 10.0,
            "allocation_method_variation_pct": 25.0,
        },
        description="Sensitivity analysis parameters for LCA uncertainties",
    )
    cutoff_threshold_pct: float = Field(
        1.0,
        ge=0.1,
        le=5.0,
        description="LCA cutoff threshold (%) below which flows are excluded",
    )
    epd_integration: bool = Field(
        True,
        description="Integrate Environmental Product Declarations (EPDs) when available",
    )


class BoundaryConfig(BaseModel):
    """Configuration for multi-entity organisational boundary consolidation.

    Implements GHG Protocol organisational boundary approaches for
    corporate groups with subsidiaries, joint ventures, and franchises.
    """

    approach: ConsolidationApproach = Field(
        ConsolidationApproach.OPERATIONAL_CONTROL,
        description="Consolidation approach per GHG Protocol",
    )
    entities: List[str] = Field(
        default_factory=list,
        description="List of legal entity identifiers in the group",
    )
    total_entities: int = Field(
        1,
        ge=1,
        le=10000,
        description="Total number of entities in the consolidation boundary",
    )
    jv_equity_threshold_pct: float = Field(
        20.0,
        ge=0.0,
        le=100.0,
        description="Minimum equity share (%) for JV inclusion in equity share approach",
    )
    jv_inclusion_method: str = Field(
        "proportional",
        description="JV inclusion method: proportional (equity %) or full (100%)",
    )
    franchise_inclusion: bool = Field(
        False,
        description="Include franchise operations in consolidation boundary",
    )
    franchise_threshold_pct: float = Field(
        50.0,
        ge=0.0,
        le=100.0,
        description="Minimum control/ownership (%) for franchise inclusion",
    )
    inter_company_elimination: bool = Field(
        True,
        description="Eliminate inter-company emissions to prevent double-counting",
    )
    entity_level_reporting: bool = Field(
        True,
        description="Generate Scope 3 reports at individual entity level",
    )

    @field_validator("jv_equity_threshold_pct")
    @classmethod
    def validate_jv_threshold(cls, v: float) -> float:
        """JV threshold should be meaningful for equity share approach."""
        if v < 1.0:
            logger.warning(
                "JV equity threshold of %.1f%% is very low. "
                "Consider raising to at least 20%%.",
                v,
            )
        return v


class ScenarioConfig(BaseModel):
    """Configuration for Scope 3 scenario analysis and reduction modelling.

    Supports MACC (Marginal Abatement Cost Curve), what-if analysis,
    technology pathway modelling, supplier programme scenarios, and
    Paris alignment pathway assessment.
    """

    enabled: bool = Field(
        True,
        description="Enable scenario analysis capabilities",
    )
    scenario_types: List[ScenarioType] = Field(
        default_factory=lambda: [
            ScenarioType.MACC,
            ScenarioType.WHAT_IF,
            ScenarioType.PARIS_ALIGNMENT,
        ],
        description="Enabled scenario analysis types",
    )
    macc_max_interventions: int = Field(
        50,
        ge=5,
        le=200,
        description="Maximum number of abatement interventions in MACC analysis",
    )
    macc_currency: str = Field(
        "USD",
        description="Currency for MACC cost figures (ISO 4217)",
    )
    what_if_max_scenarios: int = Field(
        10,
        ge=2,
        le=50,
        description="Maximum number of what-if scenarios to model",
    )
    paris_target_celsius: float = Field(
        1.5,
        ge=1.0,
        le=3.0,
        description="Paris Agreement target temperature (1.5 or 2.0 degrees Celsius)",
    )
    paris_pathway_source: str = Field(
        "iea_nze_2050",
        description="Paris alignment pathway data source (e.g., IEA NZE 2050, NGFS)",
    )
    budget_constraint_usd: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Total budget constraint (USD) for scenario analysis",
    )
    time_horizon_years: int = Field(
        10,
        ge=3,
        le=30,
        description="Scenario analysis time horizon in years",
    )
    discount_rate_pct: float = Field(
        5.0,
        ge=0.0,
        le=20.0,
        description="Discount rate for NPV calculations in MACC analysis",
    )


class SBTiConfig(BaseModel):
    """Configuration for Science Based Targets initiative (SBTi) alignment.

    Supports near-term and long-term SBTi target setting for Scope 3,
    including absolute contraction, Sectoral Decarbonisation Approach (SDA),
    and economic intensity methods. Includes FLAG (Forest, Land and
    Agriculture) guidance support.
    """

    enabled: bool = Field(
        True,
        description="Enable SBTi alignment tracking and reporting",
    )
    base_year: int = Field(
        2022,
        ge=2015,
        le=2025,
        description="SBTi base year for Scope 3 target",
    )
    near_term_target_year: int = Field(
        2030,
        ge=2025,
        le=2035,
        description="SBTi near-term target year",
    )
    long_term_target_year: Optional[int] = Field(
        2050,
        ge=2035,
        le=2055,
        description="SBTi long-term (net-zero) target year",
    )
    target_type: SBTiTargetType = Field(
        SBTiTargetType.ABSOLUTE_CONTRACTION,
        description="SBTi target-setting method",
    )
    near_term_reduction_pct: float = Field(
        42.0,
        ge=0.0,
        le=100.0,
        description="Near-term Scope 3 reduction target (% from base year)",
    )
    long_term_reduction_pct: float = Field(
        90.0,
        ge=0.0,
        le=100.0,
        description="Long-term Scope 3 reduction target (% from base year)",
    )
    coverage_pct: float = Field(
        67.0,
        ge=0.0,
        le=100.0,
        description="Scope 3 coverage target (% of total Scope 3, SBTi minimum 67%)",
    )
    flag_enabled: bool = Field(
        False,
        description="Enable SBTi FLAG (Forest, Land and Agriculture) pathway",
    )
    flag_commodity_groups: List[str] = Field(
        default_factory=list,
        description="FLAG commodity groups (e.g., 'beef', 'palm_oil', 'soy', 'timber')",
    )
    supplier_engagement_target_pct: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description=(
            "Percentage of suppliers (by emissions) with own SBTi targets "
            "(SBTi supplier engagement pathway)"
        ),
    )
    intensity_metric: Optional[str] = Field(
        None,
        description="Intensity metric for economic intensity target (e.g., 'tCO2e/MEUR_revenue')",
    )

    @model_validator(mode="after")
    def validate_sbti_coherence(self) -> "SBTiConfig":
        """Validate SBTi configuration coherence."""
        if self.near_term_target_year >= (self.long_term_target_year or 2055):
            raise ValueError(
                f"Near-term target year ({self.near_term_target_year}) must be before "
                f"long-term target year ({self.long_term_target_year})."
            )
        if self.near_term_reduction_pct > self.long_term_reduction_pct:
            raise ValueError(
                f"Near-term reduction ({self.near_term_reduction_pct}%) cannot exceed "
                f"long-term reduction ({self.long_term_reduction_pct}%)."
            )
        if self.coverage_pct < 67.0:
            logger.warning(
                "SBTi requires minimum 67%% Scope 3 coverage for near-term targets. "
                "Current coverage target is %.1f%%.",
                self.coverage_pct,
            )
        return self


class SupplierProgrammeConfig(BaseModel):
    """Configuration for supplier programme management.

    Extends PACK-042 supplier engagement with structured programme
    management, target-setting for suppliers, incentive models, and
    progress monitoring against SBTi supplier engagement pathway.
    """

    enabled: bool = Field(
        True,
        description="Enable supplier programme management",
    )
    top_supplier_pct: float = Field(
        80.0,
        ge=10.0,
        le=100.0,
        description="Target percentage of Scope 3 emissions covered by programme",
    )
    top_supplier_count: int = Field(
        100,
        ge=10,
        le=1000,
        description="Number of top suppliers enrolled in programme",
    )
    target_reduction_pct: float = Field(
        25.0,
        ge=0.0,
        le=100.0,
        description="Target emission reduction (%) from engaged suppliers",
    )
    timeline_years: int = Field(
        5,
        ge=1,
        le=15,
        description="Programme timeline in years",
    )
    incentive_model: str = Field(
        "preferred_status",
        description=(
            "Incentive model: preferred_status, contract_extension, "
            "volume_increase, co_investment, penalty"
        ),
    )
    cdp_supply_chain: bool = Field(
        True,
        description="Integrate with CDP Supply Chain programme",
    )
    sbti_cascade_enabled: bool = Field(
        True,
        description="Track supplier SBTi target adoption per cascade methodology",
    )
    quarterly_reviews: bool = Field(
        True,
        description="Enable quarterly supplier progress reviews",
    )
    data_quality_improvement_target: float = Field(
        1.0,
        ge=0.0,
        le=3.0,
        description="Target DQR improvement per year from supplier data",
    )


class ClimateRiskConfig(BaseModel):
    """Configuration for climate risk quantification per TCFD/ISSB.

    Supports transition risk (policy, technology, market, reputation),
    physical risk (acute and chronic), and opportunity identification
    with carbon price impact modelling.
    """

    enabled: bool = Field(
        True,
        description="Enable climate risk quantification",
    )
    risk_types: List[RiskType] = Field(
        default_factory=lambda: [
            RiskType.TRANSITION,
            RiskType.PHYSICAL,
            RiskType.OPPORTUNITY,
        ],
        description="Climate risk types to assess",
    )
    carbon_price_usd_per_tonne: float = Field(
        75.0,
        ge=0.0,
        le=500.0,
        description="Carbon price assumption (USD/tCO2e) for transition risk",
    )
    carbon_price_escalation_pct: float = Field(
        5.0,
        ge=0.0,
        le=20.0,
        description="Annual carbon price escalation rate (%)",
    )
    time_horizons_years: List[int] = Field(
        default_factory=lambda: [5, 10, 30],
        description="Time horizons for climate risk assessment (years)",
    )
    scenarios: List[str] = Field(
        default_factory=lambda: [
            "iea_nze_2050",
            "iea_aps",
            "iea_steps",
            "ngfs_orderly",
            "ngfs_disorderly",
        ],
        description="Climate scenarios for risk modelling",
    )
    physical_risk_regions: List[str] = Field(
        default_factory=list,
        description="Geographic regions for physical risk assessment (ISO 3166)",
    )
    stranded_asset_analysis: bool = Field(
        False,
        description="Enable stranded asset analysis for investment portfolios",
    )
    value_at_risk_enabled: bool = Field(
        True,
        description="Calculate climate value-at-risk for Scope 3 exposure",
    )


class BaseYearConfig(BaseModel):
    """Configuration for base-year recalculation per GHG Protocol.

    Implements GHG Protocol base-year recalculation policy with
    significance thresholds and automated triggers for structural
    changes, methodology improvements, and error corrections.
    """

    base_year: int = Field(
        2022,
        ge=2015,
        le=2025,
        description="Base year for Scope 3 inventory",
    )
    significance_threshold_pct: float = Field(
        5.0,
        ge=1.0,
        le=20.0,
        description="Significance threshold (%) triggering base-year recalculation",
    )
    recalc_triggers_enabled: bool = Field(
        True,
        description="Enable automated base-year recalculation triggers",
    )
    recalc_triggers: List[str] = Field(
        default_factory=lambda: [
            "structural_change",
            "methodology_change",
            "error_correction",
            "boundary_change",
            "outsourcing_insourcing",
        ],
        description="Events that trigger base-year recalculation assessment",
    )
    rolling_base_year: bool = Field(
        False,
        description="Use rolling base year (latest n-year average) instead of fixed",
    )
    rolling_window_years: int = Field(
        3,
        ge=2,
        le=5,
        description="Rolling base year window size in years",
    )
    documentation_required: bool = Field(
        True,
        description="Require documentation for every base-year recalculation decision",
    )


class SectorConfig(BaseModel):
    """Configuration for sector-specific Scope 3 features.

    Enables sector-specific capabilities: PCAF for financial services,
    FLAG for food and agriculture, circular economy for manufacturing,
    last-mile logistics for retail, cloud footprint for technology.
    """

    sector_focus: SectorFocus = Field(
        SectorFocus.GENERAL,
        description="Primary sector focus for sector-specific features",
    )
    pcaf_asset_classes: List[PCAFAssetClass] = Field(
        default_factory=list,
        description="PCAF asset classes for financial services Category 15",
    )
    pcaf_data_quality_target: float = Field(
        3.0,
        ge=1.0,
        le=5.0,
        description="Target PCAF data quality score (1=best, 5=worst)",
    )
    retail_last_mile_enabled: bool = Field(
        False,
        description="Enable last-mile delivery emission modelling for retail",
    )
    retail_returns_enabled: bool = Field(
        False,
        description="Enable product returns emission modelling for retail",
    )
    retail_packaging_lifecycle: bool = Field(
        False,
        description="Enable packaging lifecycle analysis for retail",
    )
    manufacturing_circular_economy: bool = Field(
        False,
        description="Enable circular economy metrics (recycled content, recyclability)",
    )
    manufacturing_epd_integration: bool = Field(
        False,
        description="Integrate Environmental Product Declarations for manufacturing",
    )
    technology_cloud_footprint: bool = Field(
        False,
        description="Enable cloud infrastructure carbon footprint (AWS, Azure, GCP)",
    )
    technology_saas_use_phase: bool = Field(
        False,
        description="Enable SaaS product use-phase emission modelling",
    )
    energy_upstream_fuel: bool = Field(
        False,
        description="Enable upstream fuel production lifecycle modelling for energy",
    )
    energy_sold_product_use: bool = Field(
        False,
        description="Enable sold energy use-phase (Cat 11) modelling for energy",
    )
    food_flag_enabled: bool = Field(
        False,
        description="Enable FLAG (Forest, Land and Agriculture) pathway for food sector",
    )
    food_deforestation_free: bool = Field(
        False,
        description="Enable deforestation-free supply chain tracking",
    )


class AssuranceConfig(BaseModel):
    """Configuration for external assurance readiness per ISAE 3410.

    Defines target assurance level, evidence requirements, verifier
    integration, and data quality thresholds for assurance.
    """

    target_level: AssuranceLevel = Field(
        AssuranceLevel.LIMITED,
        description="Target assurance level (LIMITED or REASONABLE)",
    )
    isae_3410_compliant: bool = Field(
        True,
        description="Structure outputs for ISAE 3410 compliance",
    )
    evidence_requirements: List[str] = Field(
        default_factory=lambda: [
            "methodology_documentation",
            "emission_factor_sources",
            "data_quality_assessment",
            "uncertainty_analysis",
            "provenance_hashes",
            "assumption_registry",
            "double_counting_log",
            "calculation_audit_trail",
        ],
        description="Evidence artefacts required for assurance engagement",
    )
    verifier_portal_enabled: bool = Field(
        False,
        description="Enable verifier data room portal for evidence review",
    )
    materiality_threshold_pct: float = Field(
        5.0,
        ge=1.0,
        le=10.0,
        description="Materiality threshold (%) for assurance scope",
    )
    min_dqr_for_reasonable: float = Field(
        2.5,
        ge=1.0,
        le=4.0,
        description="Minimum average DQR required for reasonable assurance",
    )
    continuous_monitoring: bool = Field(
        False,
        description="Enable continuous data monitoring between annual assurance cycles",
    )


class IntegrationConfig(BaseModel):
    """Configuration for integration with other GreenLang packs and external systems."""

    pack042_required: bool = Field(
        True,
        description="PACK-042 (Scope 3 Starter) is a required dependency",
    )
    pack042_pack_id: str = Field(
        "PACK-042-scope-3-starter",
        description="PACK-042 identifier for dependency resolution",
    )
    pack041_enabled: bool = Field(
        True,
        description="Enable PACK-041 (Scope 1-2 Complete) integration",
    )
    pack041_pack_id: str = Field(
        "PACK-041-scope-1-2-complete",
        description="PACK-041 identifier for cross-pack data exchange",
    )
    lca_database_bridge: bool = Field(
        True,
        description="Enable LCA database bridge (ecoinvent, GaBi, ELCD)",
    )
    sbti_api_enabled: bool = Field(
        False,
        description="Enable SBTi API integration for target validation",
    )
    sbti_api_url: Optional[str] = Field(
        None,
        description="SBTi API endpoint URL",
    )
    tcfd_issb_bridge: bool = Field(
        True,
        description="Enable TCFD/ISSB scenario analysis bridge",
    )
    erp_type: Optional[str] = Field(
        None,
        description="ERP system type (SAP, Oracle, Dynamics) for spend data",
    )
    supplier_portal_enabled: bool = Field(
        True,
        description="Enable supplier programme management portal",
    )
    cdp_supply_chain: bool = Field(
        True,
        description="Enable CDP Supply Chain programme integration",
    )
    pcaf_database_bridge: bool = Field(
        False,
        description="Enable PCAF database bridge for financed emissions",
    )
    cloud_carbon_api: Optional[str] = Field(
        None,
        description="Cloud carbon footprint API (AWS, Azure, GCP)",
    )


class ReportingConfig(BaseModel):
    """Configuration for enterprise-grade Scope 3 reporting."""

    formats: List[OutputFormat] = Field(
        default_factory=lambda: [
            OutputFormat.HTML,
            OutputFormat.JSON,
            OutputFormat.XLSX,
            OutputFormat.PDF,
        ],
        description="Output formats for Scope 3 reports",
    )
    enterprise_dashboard: bool = Field(
        True,
        description="Enable investor-grade enterprise dashboard",
    )
    dashboard_refresh_minutes: int = Field(
        60,
        ge=5,
        le=1440,
        description="Dashboard data refresh interval (minutes)",
    )
    investor_grade: bool = Field(
        True,
        description="Generate investor-grade disclosure packages",
    )
    board_summary: bool = Field(
        True,
        description="Generate board-level executive summary",
    )
    scenario_reporting: bool = Field(
        True,
        description="Include scenario analysis results in reports",
    )
    benchmark_comparison: bool = Field(
        True,
        description="Include sector benchmark comparisons in reports",
    )
    multi_entity_rollup: bool = Field(
        True,
        description="Generate consolidated multi-entity reports",
    )
    verifier_data_room: bool = Field(
        False,
        description="Generate verifier data room package for assurance",
    )
    output_language: str = Field(
        "en",
        description="Report language (ISO 639-1)",
    )
    watermark_draft: bool = Field(
        True,
        description="Apply DRAFT watermark to unapproved reports",
    )


class SecurityConfig(BaseModel):
    """Security and access control configuration for enterprise Scope 3 data."""

    rbac_roles: List[str] = Field(
        default_factory=lambda: [
            "scope3_director",
            "scope3_manager",
            "sustainability_officer",
            "procurement_director",
            "supply_chain_analyst",
            "lca_specialist",
            "verifier",
            "viewer",
        ],
        description="Available RBAC roles for the Scope 3 Complete pack",
    )
    supplier_data_isolation: bool = Field(
        True,
        description="Isolate individual supplier data from other suppliers",
    )
    entity_level_access: bool = Field(
        True,
        description="Enforce entity-level access control in multi-entity setups",
    )
    audit_logging: bool = Field(
        True,
        description="Enable security audit logging for all data access",
    )
    audit_retention_years: int = Field(
        10,
        ge=1,
        le=15,
        description="Audit trail retention period in years",
    )
    pii_redaction: bool = Field(
        True,
        description="Enable PII redaction in exported reports",
    )
    encryption_at_rest: bool = Field(
        True,
        description="Require AES-256-GCM encryption at rest",
    )
    data_classification: str = Field(
        "CONFIDENTIAL",
        description="Default data classification: INTERNAL, CONFIDENTIAL, RESTRICTED",
    )


class PerformanceConfig(BaseModel):
    """Performance and resource limits for enterprise Scope 3 execution."""

    parallel_entities: int = Field(
        10,
        ge=1,
        le=100,
        description="Maximum number of entities to consolidate in parallel",
    )
    parallel_categories: int = Field(
        8,
        ge=1,
        le=15,
        description="Maximum number of categories to calculate in parallel",
    )
    batch_size: int = Field(
        5000,
        ge=100,
        le=50000,
        description="Batch size for spend line items and supplier records",
    )
    cache_ttl_seconds: int = Field(
        7200,
        ge=60,
        le=86400,
        description="Cache TTL for LCA factors and classification lookups (seconds)",
    )
    lca_query_timeout_seconds: int = Field(
        600,
        ge=60,
        le=3600,
        description="Timeout for LCA database queries (seconds)",
    )
    scenario_computation_timeout_seconds: int = Field(
        1800,
        ge=300,
        le=7200,
        description="Timeout for scenario analysis computation (seconds)",
    )
    max_suppliers: int = Field(
        10000,
        ge=100,
        le=100000,
        description="Maximum number of suppliers per entity",
    )
    max_line_items: int = Field(
        2000000,
        ge=10000,
        le=10000000,
        description="Maximum number of spend line items per calculation run",
    )
    memory_ceiling_mb: int = Field(
        16384,
        ge=2048,
        le=131072,
        description="Memory ceiling for Scope 3 calculation (MB)",
    )
    monte_carlo_iterations: int = Field(
        10000,
        ge=1000,
        le=100000,
        description="Monte Carlo iterations for uncertainty analysis",
    )


# =============================================================================
# Main Configuration Model
# =============================================================================


class Scope3CompleteConfig(BaseModel):
    """Main configuration for PACK-043 Scope 3 Complete Pack.

    This is the root configuration model that extends PACK-042 Scope 3 Starter
    with enterprise capabilities: maturity progression, LCA integration, SBTi
    alignment, scenario analysis, multi-entity consolidation, supplier programme
    management, climate risk quantification, base-year recalculation, and
    assurance readiness.
    """

    # Organisation identification
    company_name: str = Field(
        "",
        description="Legal entity name of the reporting company",
    )
    sector_focus: SectorFocus = Field(
        SectorFocus.GENERAL,
        description="Primary sector focus for enterprise configuration",
    )
    country: str = Field(
        "DE",
        description="Primary country of operations (ISO 3166-1 alpha-2)",
    )
    reporting_year: int = Field(
        2026,
        ge=2020,
        le=2035,
        description="Reporting year for the Scope 3 inventory",
    )

    # Organisation characteristics
    revenue_meur: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Annual revenue in million EUR for intensity metrics",
    )
    employees_fte: Optional[int] = Field(
        None,
        ge=0,
        description="Full-time equivalent employees",
    )
    total_procurement_spend_eur: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Total annual procurement spend (EUR)",
    )
    number_of_entities: int = Field(
        1,
        ge=1,
        le=10000,
        description="Number of legal entities in consolidation boundary",
    )
    years_of_scope3_reporting: int = Field(
        2,
        ge=0,
        le=20,
        description="Number of years of prior Scope 3 reporting experience",
    )

    # Enterprise sub-configurations
    maturity: MaturityConfig = Field(
        default_factory=MaturityConfig,
        description="Maturity level progression configuration",
    )
    lca: LCAConfig = Field(
        default_factory=LCAConfig,
        description="Lifecycle assessment integration configuration",
    )
    consolidation: BoundaryConfig = Field(
        default_factory=BoundaryConfig,
        description="Multi-entity consolidation boundary configuration",
    )
    scenarios: ScenarioConfig = Field(
        default_factory=ScenarioConfig,
        description="Scenario analysis and reduction modelling configuration",
    )
    sbti: SBTiConfig = Field(
        default_factory=SBTiConfig,
        description="SBTi alignment and target tracking configuration",
    )
    supplier_programme: SupplierProgrammeConfig = Field(
        default_factory=SupplierProgrammeConfig,
        description="Supplier programme management configuration",
    )
    climate_risk: ClimateRiskConfig = Field(
        default_factory=ClimateRiskConfig,
        description="Climate risk quantification per TCFD/ISSB",
    )
    base_year: BaseYearConfig = Field(
        default_factory=BaseYearConfig,
        description="Base-year recalculation configuration",
    )
    sector: SectorConfig = Field(
        default_factory=SectorConfig,
        description="Sector-specific features configuration",
    )
    assurance: AssuranceConfig = Field(
        default_factory=AssuranceConfig,
        description="External assurance readiness configuration",
    )
    integration: IntegrationConfig = Field(
        default_factory=IntegrationConfig,
        description="Integration with other packs and external systems",
    )
    reporting: ReportingConfig = Field(
        default_factory=ReportingConfig,
        description="Enterprise reporting configuration",
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security and access control",
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance and resource limits",
    )

    @model_validator(mode="after")
    def validate_pack042_dependency(self) -> "Scope3CompleteConfig":
        """PACK-042 is required as a dependency for PACK-043."""
        if not self.integration.pack042_required:
            logger.warning(
                "PACK-043 requires PACK-042 (Scope 3 Starter) as a dependency. "
                "Automatically re-enabling pack042_required."
            )
            self.integration.pack042_required = True
        return self

    @model_validator(mode="after")
    def validate_sbti_coherence(self) -> "Scope3CompleteConfig":
        """SBTi base year must align with base-year configuration."""
        if self.sbti.enabled and self.sbti.base_year != self.base_year.base_year:
            logger.info(
                "SBTi base year (%d) differs from inventory base year (%d). "
                "Ensure this is intentional.",
                self.sbti.base_year,
                self.base_year.base_year,
            )
        return self

    @model_validator(mode="after")
    def validate_financial_sector_pcaf(self) -> "Scope3CompleteConfig":
        """Financial services sector should enable PCAF asset classes."""
        if self.sector_focus == SectorFocus.FINANCIAL_SERVICES:
            if not self.sector.pcaf_asset_classes:
                logger.info(
                    "Financial services sector: enabling default PCAF asset classes."
                )
                self.sector.pcaf_asset_classes = [
                    PCAFAssetClass.LISTED_EQUITY,
                    PCAFAssetClass.CORPORATE_BONDS,
                    PCAFAssetClass.UNLISTED_EQUITY,
                    PCAFAssetClass.PROJECT_FINANCE,
                    PCAFAssetClass.COMMERCIAL_RE,
                    PCAFAssetClass.MORTGAGES,
                ]
            if not self.integration.pcaf_database_bridge:
                self.integration.pcaf_database_bridge = True
        return self

    @model_validator(mode="after")
    def validate_multi_entity_consolidation(self) -> "Scope3CompleteConfig":
        """Multi-entity groups need consolidation configuration."""
        if self.number_of_entities > 1:
            if self.consolidation.total_entities < self.number_of_entities:
                self.consolidation.total_entities = self.number_of_entities
        return self

    @model_validator(mode="after")
    def validate_assurance_dqr(self) -> "Scope3CompleteConfig":
        """Reasonable assurance requires minimum data quality."""
        if (
            self.assurance.target_level == AssuranceLevel.REASONABLE
            and self.maturity.target_level.value
            < MaturityLevel.LEVEL_4_ADVANCED.value
        ):
            logger.warning(
                "Reasonable assurance typically requires maturity Level 4+. "
                "Current target: %s.",
                self.maturity.target_level.value,
            )
        return self

    @model_validator(mode="after")
    def validate_food_flag(self) -> "Scope3CompleteConfig":
        """Food/beverage sector should enable FLAG pathway."""
        if self.sector_focus == SectorFocus.FOOD_BEVERAGE:
            if not self.sector.food_flag_enabled:
                logger.info(
                    "Food/beverage sector: enabling FLAG pathway."
                )
                self.sector.food_flag_enabled = True
            if not self.sbti.flag_enabled:
                self.sbti.flag_enabled = True
        return self


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """Top-level pack configuration wrapper for PACK-043.

    Handles preset loading, PACK-042 configuration inheritance, environment
    variable overrides, and configuration merging. Follows the standard
    GreenLang pack config pattern with from_preset(), from_yaml(), and
    merge() support.
    """

    pack: Scope3CompleteConfig = Field(
        default_factory=Scope3CompleteConfig,
        description="Main Scope 3 Complete Pack configuration",
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
        "PACK-043-scope-3-complete",
        description="Pack identifier",
    )

    @classmethod
    def from_preset(
        cls, preset_name: str, overrides: Optional[Dict[str, Any]] = None
    ) -> "PackConfig":
        """Load configuration from a named preset.

        Args:
            preset_name: Name of the preset (enterprise_manufacturing, etc.)
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

        pack_config = Scope3CompleteConfig(**preset_data)
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

        pack_config = Scope3CompleteConfig(**config_data)
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
        pack_config = Scope3CompleteConfig(**merged)
        return cls(
            pack=pack_config,
            preset_name=base.preset_name,
            config_version=base.config_version,
        )

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """Load configuration overrides from environment variables.

        Environment variables prefixed with SCOPE3_COMPLETE_ are loaded
        and mapped to configuration keys. Nested keys use double underscore.

        Example: SCOPE3_COMPLETE_MATURITY__TARGET_LEVEL=LEVEL_4_ADVANCED
                 SCOPE3_COMPLETE_SBTI__BASE_YEAR=2022
        """
        overrides: Dict[str, Any] = {}
        prefix = "SCOPE3_COMPLETE_"
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


def validate_config(config: Scope3CompleteConfig) -> List[str]:
    """Validate a Scope 3 Complete configuration and return any warnings.

    Args:
        config: Scope3CompleteConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check company identification
    if not config.company_name:
        warnings.append(
            "No company_name configured. Required for report identification."
        )

    # Validate PACK-042 dependency
    if not config.integration.pack042_required:
        warnings.append(
            "PACK-042 dependency is disabled. PACK-043 requires PACK-042."
        )

    # Validate maturity progression
    if config.maturity.current_level == config.maturity.target_level:
        warnings.append(
            f"Current and target maturity level are both "
            f"{config.maturity.target_level.value}. No progression configured."
        )

    # Validate SBTi configuration
    if config.sbti.enabled:
        if config.sbti.coverage_pct < 67.0:
            warnings.append(
                f"SBTi requires minimum 67% Scope 3 coverage. "
                f"Current target: {config.sbti.coverage_pct}%."
            )
        if (
            config.sbti.target_type == SBTiTargetType.SDA
            and config.sector_focus == SectorFocus.GENERAL
        ):
            warnings.append(
                "SDA target type requires sector-specific pathway data. "
                "Consider setting a specific sector_focus."
            )

    # Validate multi-entity configuration
    if config.number_of_entities > 1:
        if not config.consolidation.entities and config.consolidation.total_entities <= 1:
            warnings.append(
                f"Multi-entity group ({config.number_of_entities} entities) but "
                f"no entities configured in consolidation boundary."
            )
        if config.consolidation.inter_company_elimination is False:
            warnings.append(
                "Inter-company elimination is disabled for multi-entity group. "
                "This may result in double-counting."
            )

    # Validate financial sector PCAF
    if config.sector_focus == SectorFocus.FINANCIAL_SERVICES:
        if not config.sector.pcaf_asset_classes:
            warnings.append(
                "Financial services sector should configure PCAF asset classes."
            )
        if not config.integration.pcaf_database_bridge:
            warnings.append(
                "Financial services should enable PCAF database bridge."
            )

    # Validate assurance readiness
    if config.assurance.target_level == AssuranceLevel.REASONABLE:
        if config.maturity.target_level.value < MaturityLevel.LEVEL_4_ADVANCED.value:
            warnings.append(
                "Reasonable assurance typically requires maturity Level 4+."
            )

    # Validate scenario analysis
    if config.scenarios.enabled:
        if ScenarioType.PARIS_ALIGNMENT in config.scenarios.scenario_types:
            if not config.sbti.enabled:
                warnings.append(
                    "Paris alignment scenario analysis benefits from SBTi target. "
                    "Consider enabling SBTi configuration."
                )

    # Validate climate risk configuration
    if config.climate_risk.enabled:
        if config.climate_risk.carbon_price_usd_per_tonne <= 0:
            warnings.append(
                "Carbon price is zero or negative. Transition risk "
                "quantification requires a positive carbon price assumption."
            )

    # Validate base-year alignment
    if config.sbti.enabled:
        if config.sbti.base_year != config.base_year.base_year:
            warnings.append(
                f"SBTi base year ({config.sbti.base_year}) differs from inventory "
                f"base year ({config.base_year.base_year}). Ensure intentional."
            )

    # Validate LCA configuration
    if config.lca.enabled:
        if config.maturity.target_level.value < MaturityLevel.LEVEL_3_INTERMEDIATE.value:
            warnings.append(
                "LCA integration is enabled but maturity target is below Level 3. "
                "LCA is most beneficial at Level 3+."
            )

    # Validate food sector FLAG
    if config.sector_focus == SectorFocus.FOOD_BEVERAGE:
        if not config.sbti.flag_enabled:
            warnings.append(
                "Food/beverage sector should enable SBTi FLAG pathway."
            )
        if not config.sector.food_flag_enabled:
            warnings.append(
                "Food/beverage sector should enable FLAG features."
            )

    # Validate supplier programme
    if config.supplier_programme.enabled:
        if config.supplier_programme.top_supplier_count < 10:
            warnings.append(
                "Supplier programme with fewer than 10 suppliers may have "
                "insufficient coverage for meaningful emission reduction."
            )

    return warnings


def get_default_config(
    sector_focus: SectorFocus = SectorFocus.GENERAL,
) -> Scope3CompleteConfig:
    """Get default configuration for a given sector focus.

    Args:
        sector_focus: Sector focus to configure for.

    Returns:
        Scope3CompleteConfig instance with sector-appropriate defaults.
    """
    return Scope3CompleteConfig(sector_focus=sector_focus)


def get_maturity_info(level: Union[str, MaturityLevel]) -> Dict[str, Any]:
    """Get detailed information about a maturity level.

    Args:
        level: Maturity level enum or string value.

    Returns:
        Dictionary with name, description, methodology, and assurance details.
    """
    key = level.value if isinstance(level, MaturityLevel) else level
    return MATURITY_LEVEL_INFO.get(
        key,
        {
            "name": key,
            "description": "Unknown level",
            "typical_pack": "Unknown",
            "methodology": "Unknown",
            "data_quality_target": 5,
            "assurance": "Unknown",
        },
    )


def list_available_presets() -> Dict[str, str]:
    """List all available configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return AVAILABLE_PRESETS.copy()
