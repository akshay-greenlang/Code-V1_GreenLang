"""
PACK-024 Carbon Neutral Pack - Runtime Configuration Manager

This module implements comprehensive Pydantic v2 configuration management for
the Carbon Neutral Pack, covering the complete lifecycle from footprint
quantification through credit procurement, neutralization balance, claims
substantiation, and third-party verification.

Configuration Hierarchy:
    1. Base pack.yaml manifest
    2. Neutrality-type presets (corporate, sme, event, product, building,
       service, project, portfolio)
    3. Environment variable overrides (CARBON_NEUTRAL_* prefix)
    4. Runtime explicit overrides

Key Subsystems:
    - Organization Profile (type, size, sector, neutrality commitment)
    - Footprint Quantification (ISO 14064-1 scopes, GWP, data quality)
    - Carbon Management Plan (reduction-first, internal carbon price, MACC)
    - Credit Quality Assessment (ICVCM CCP 12-dimension scoring)
    - Portfolio Optimization (diversification, vintage, budget constraints)
    - Registry & Retirement (Verra, Gold Standard, ACR, CAR tracking)
    - Neutralization Balance (annual/event/product/project balance)
    - Claims Substantiation (ISO 14068-1, PAS 2060 compliance)
    - Verification Package (ISAE 3410 limited/reasonable assurance)
    - Annual Cycle Management (milestones, recertification, improvement)
    - Permanence Risk (buffer pools, reversal monitoring, insurance)
    - Multi-Framework Reporting (ISO 14068-1, PAS 2060, GHG Protocol)
    - Performance & Caching
    - Audit Trail & Provenance

Regulatory Framework:
    - ISO 14068-1:2023 - Carbon neutrality
    - PAS 2060:2014 - Carbon neutrality demonstration
    - ISO 14064-1:2018 - Organization-level GHG quantification
    - ISO 14064-2:2019 - Project-level GHG quantification
    - ISO 14067:2018 - Carbon footprint of products
    - GHG Protocol Corporate Standard (revised)
    - GHG Protocol Scope 2 Guidance (2015)
    - GHG Protocol Scope 3 Standard
    - IPCC AR6 WG1 (2021) - GWP values
    - ICVCM Core Carbon Principles (2023)
    - VCMI Claims Code of Practice (2023)
    - ICROA Code of Best Practice
    - Oxford Principles for Net Zero Aligned Carbon Offsetting (2020)

Example:
    >>> config = CarbonNeutralConfig.from_preset("corporate_neutrality")
    >>> print(config.pack.organization.neutrality_type)
    NeutralityType.CORPORATE
    >>> config = CarbonNeutralConfig.from_yaml("path/to/config.yaml")
    >>> hash_val = config.get_config_hash()
"""

import hashlib
import logging
import os
from datetime import date, datetime
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
DEFAULT_REPORTING_YEAR: int = 2025
DEFAULT_NEUTRALITY_TARGET_YEAR: int = 2025

SUPPORTED_PRESETS: Dict[str, str] = {
    "corporate_neutrality": "Full organizational Scope 1+2+3 carbon neutrality per ISO 14068-1",
    "sme_neutrality": "Simplified Scope 1+2 neutrality for small/medium enterprises",
    "event_neutrality": "Event-specific carbon neutrality per PAS 2060",
    "product_neutrality": "Product LCA-based carbon neutrality per ISO 14067",
    "building_neutrality": "Building operations carbon neutrality with CRREM alignment",
    "service_neutrality": "Service-based carbon neutrality for professional services",
    "project_neutrality": "Project-specific carbon neutrality per ISO 14064-2",
    "portfolio_neutrality": "Multi-entity portfolio carbon neutrality with entity-level tracking",
}

# ICVCM Core Carbon Principles - 12 Dimensions
ICVCM_CCP_DIMENSIONS: Dict[str, Dict[str, Any]] = {
    "additionality": {"name": "Additionality", "weight": 0.15, "description": "Project not viable without carbon revenue"},
    "permanence": {"name": "Permanence", "weight": 0.12, "description": "Long-term carbon storage durability"},
    "robust_quantification": {"name": "Robust Quantification", "weight": 0.12, "description": "Conservative baseline and monitoring"},
    "independent_validation": {"name": "Independent Validation", "weight": 0.10, "description": "Third-party audit and verification"},
    "double_counting": {"name": "No Double Counting", "weight": 0.10, "description": "Unique claim with corresponding adjustments"},
    "transition": {"name": "Transition to Net-Zero", "weight": 0.08, "description": "Supports global net-zero transition"},
    "sustainable_development": {"name": "Sustainable Development", "weight": 0.08, "description": "SDG co-benefits documented"},
    "no_net_harm": {"name": "No Net Harm", "weight": 0.07, "description": "No negative social/environmental impacts"},
    "host_country": {"name": "Host Country Approval", "weight": 0.05, "description": "Host country authorization obtained"},
    "registry": {"name": "Registry Operations", "weight": 0.05, "description": "Listed on recognized registry"},
    "governance": {"name": "Effective Governance", "weight": 0.04, "description": "Project governance and management"},
    "transparency": {"name": "Transparency", "weight": 0.04, "description": "Public disclosure of project information"},
}

# Credit Quality Rating Scale
CREDIT_QUALITY_RATINGS: Dict[str, Dict[str, Any]] = {
    "A_PLUS": {"min_score": 95, "label": "A+", "description": "Exceptional quality"},
    "A": {"min_score": 85, "label": "A", "description": "Very high quality"},
    "B_PLUS": {"min_score": 75, "label": "B+", "description": "High quality"},
    "B": {"min_score": 65, "label": "B", "description": "Good quality"},
    "C": {"min_score": 50, "label": "C", "description": "Acceptable quality"},
    "D": {"min_score": 35, "label": "D", "description": "Below standard"},
    "F": {"min_score": 0, "label": "F", "description": "Failing quality"},
}

# Carbon Credit Project Types
CREDIT_PROJECT_TYPES: Dict[str, Dict[str, Any]] = {
    "reforestation": {"category": "removal", "nature_based": True, "avg_permanence_years": 30},
    "afforestation": {"category": "removal", "nature_based": True, "avg_permanence_years": 40},
    "soil_carbon": {"category": "removal", "nature_based": True, "avg_permanence_years": 20},
    "blue_carbon": {"category": "removal", "nature_based": True, "avg_permanence_years": 50},
    "biochar": {"category": "removal", "nature_based": False, "avg_permanence_years": 100},
    "dac": {"category": "removal", "nature_based": False, "avg_permanence_years": 1000},
    "beccs": {"category": "removal", "nature_based": False, "avg_permanence_years": 1000},
    "mineralization": {"category": "removal", "nature_based": False, "avg_permanence_years": 10000},
    "renewable_energy": {"category": "avoidance", "nature_based": False, "avg_permanence_years": 0},
    "cookstoves": {"category": "avoidance", "nature_based": False, "avg_permanence_years": 0},
    "methane_capture": {"category": "avoidance", "nature_based": False, "avg_permanence_years": 0},
    "redd_plus": {"category": "avoidance", "nature_based": True, "avg_permanence_years": 25},
    "energy_efficiency": {"category": "avoidance", "nature_based": False, "avg_permanence_years": 0},
    "fuel_switching": {"category": "avoidance", "nature_based": False, "avg_permanence_years": 0},
}

# Supported Registries
SUPPORTED_REGISTRIES: Dict[str, Dict[str, str]] = {
    "verra": {"name": "Verra (VCS)", "url": "https://registry.verra.org"},
    "gold_standard": {"name": "Gold Standard", "url": "https://registry.goldstandard.org"},
    "american_carbon_registry": {"name": "American Carbon Registry (ACR)", "url": "https://acr2.apx.com"},
    "climate_action_reserve": {"name": "Climate Action Reserve (CAR)", "url": "https://thereserve2.apx.com"},
    "puro_earth": {"name": "Puro.earth", "url": "https://puro.earth"},
    "isometric": {"name": "Isometric", "url": "https://isometric.com"},
}

# Scope 3 Categories (ISO 14064-1 / GHG Protocol)
SCOPE3_CATEGORIES_INFO: Dict[int, Dict[str, Any]] = {
    1: {"name": "Purchased goods & services", "typical_importance": "HIGH"},
    2: {"name": "Capital goods", "typical_importance": "MEDIUM"},
    3: {"name": "Fuel & energy activities", "typical_importance": "HIGH"},
    4: {"name": "Upstream transportation", "typical_importance": "HIGH"},
    5: {"name": "Waste generated", "typical_importance": "MEDIUM"},
    6: {"name": "Business travel", "typical_importance": "MEDIUM"},
    7: {"name": "Employee commuting", "typical_importance": "MEDIUM"},
    8: {"name": "Upstream leased assets", "typical_importance": "LOW"},
    9: {"name": "Downstream transportation", "typical_importance": "MEDIUM"},
    10: {"name": "Processing of sold products", "typical_importance": "MEDIUM"},
    11: {"name": "Use of sold products", "typical_importance": "CRITICAL"},
    12: {"name": "End-of-life treatment", "typical_importance": "MEDIUM"},
    13: {"name": "Downstream leased assets", "typical_importance": "LOW"},
    14: {"name": "Franchises", "typical_importance": "LOW"},
    15: {"name": "Investments", "typical_importance": "MEDIUM"},
}


# =============================================================================
# Enums - Carbon Neutrality specific enumeration types (18 enums)
# =============================================================================


class NeutralityType(str, Enum):
    """Carbon neutrality boundary type."""
    CORPORATE = "CORPORATE"
    SME = "SME"
    EVENT = "EVENT"
    PRODUCT = "PRODUCT"
    BUILDING = "BUILDING"
    SERVICE = "SERVICE"
    PROJECT = "PROJECT"
    PORTFOLIO = "PORTFOLIO"


class OrganizationSize(str, Enum):
    """Organization size classification."""
    MICRO = "MICRO"
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"
    ENTERPRISE = "ENTERPRISE"


class ConsolidationApproach(str, Enum):
    """GHG Protocol consolidation approach."""
    OPERATIONAL_CONTROL = "OPERATIONAL_CONTROL"
    FINANCIAL_CONTROL = "FINANCIAL_CONTROL"
    EQUITY_SHARE = "EQUITY_SHARE"
    EVENT_BOUNDARY = "EVENT_BOUNDARY"
    PRODUCT_BOUNDARY = "PRODUCT_BOUNDARY"
    BUILDING_BOUNDARY = "BUILDING_BOUNDARY"
    PROJECT_BOUNDARY = "PROJECT_BOUNDARY"


class CreditCategory(str, Enum):
    """Carbon credit category."""
    AVOIDANCE = "AVOIDANCE"
    REMOVAL = "REMOVAL"
    HYBRID = "HYBRID"


class CreditStandard(str, Enum):
    """Carbon credit standard/registry."""
    VERRA_VCS = "VERRA_VCS"
    GOLD_STANDARD = "GOLD_STANDARD"
    ACR = "ACR"
    CAR = "CAR"
    PURO_EARTH = "PURO_EARTH"
    ISOMETRIC = "ISOMETRIC"
    OTHER = "OTHER"


class NeutralizationStandard(str, Enum):
    """Carbon neutrality standard."""
    ISO_14068_1 = "ISO_14068_1"
    PAS_2060 = "PAS_2060"
    BOTH = "BOTH"


class ClaimType(str, Enum):
    """Carbon neutrality claim type."""
    CARBON_NEUTRAL = "CARBON_NEUTRAL"
    CARBON_NEUTRAL_PRODUCT = "CARBON_NEUTRAL_PRODUCT"
    CARBON_NEUTRAL_EVENT = "CARBON_NEUTRAL_EVENT"
    CARBON_NEUTRAL_BUILDING = "CARBON_NEUTRAL_BUILDING"
    CARBON_NEUTRAL_SERVICE = "CARBON_NEUTRAL_SERVICE"
    CARBON_NEUTRAL_PROJECT = "CARBON_NEUTRAL_PROJECT"
    CARBON_NEUTRAL_PORTFOLIO = "CARBON_NEUTRAL_PORTFOLIO"
    CLIMATE_POSITIVE = "CLIMATE_POSITIVE"


class AssuranceLevel(str, Enum):
    """Verification assurance level."""
    LIMITED = "LIMITED"
    REASONABLE = "REASONABLE"


class BalanceMethod(str, Enum):
    """Neutralization balance calculation method."""
    ANNUAL = "ANNUAL"
    EVENT_TOTAL = "EVENT_TOTAL"
    PER_UNIT_PRODUCED = "PER_UNIT_PRODUCED"
    PROJECT_TOTAL = "PROJECT_TOTAL"
    ROLLING_12_MONTHS = "ROLLING_12_MONTHS"


class DataSourceType(str, Enum):
    """Data source type for emissions data."""
    MANUAL = "MANUAL"
    ERP_SAP = "ERP_SAP"
    ERP_ORACLE = "ERP_ORACLE"
    ERP_WORKDAY = "ERP_WORKDAY"
    ERP_DYNAMICS = "ERP_DYNAMICS"
    API = "API"
    FILE_UPLOAD = "FILE_UPLOAD"
    DATABASE = "DATABASE"
    SURVEY = "SURVEY"
    BMS = "BMS"


class Scope3Method(str, Enum):
    """Scope 3 calculation methodology."""
    SPEND_BASED = "SPEND_BASED"
    ACTIVITY_BASED = "ACTIVITY_BASED"
    HYBRID = "HYBRID"


class ERPType(str, Enum):
    """ERP system type."""
    SAP = "SAP"
    ORACLE = "ORACLE"
    WORKDAY = "WORKDAY"
    DYNAMICS_365 = "DYNAMICS_365"
    NETSUITE = "NETSUITE"
    NONE = "NONE"


class VCMIClaimTier(str, Enum):
    """VCMI offset claim tier."""
    SILVER = "SILVER"
    GOLD = "GOLD"
    PLATINUM = "PLATINUM"


class PermanenceCategory(str, Enum):
    """Carbon credit permanence category."""
    SHORT_TERM = "SHORT_TERM"       # < 25 years
    MEDIUM_TERM = "MEDIUM_TERM"     # 25-100 years
    LONG_TERM = "LONG_TERM"         # 100-1000 years
    GEOLOGICAL = "GEOLOGICAL"       # > 1000 years


class VerificationStatus(str, Enum):
    """Neutrality claim verification status."""
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    VERIFIED = "VERIFIED"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"


class MilestoneFrequency(str, Enum):
    """Progress milestone frequency."""
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    SEMI_ANNUAL = "SEMI_ANNUAL"
    ANNUAL = "ANNUAL"
    EVENT_BASED = "EVENT_BASED"
    PER_PHASE = "PER_PHASE"


class OrganizationSector(str, Enum):
    """Organization sector classification."""
    MANUFACTURING = "MANUFACTURING"
    TECHNOLOGY = "TECHNOLOGY"
    FINANCIAL_SERVICES = "FINANCIAL_SERVICES"
    PROFESSIONAL_SERVICES = "PROFESSIONAL_SERVICES"
    RETAIL = "RETAIL"
    REAL_ESTATE = "REAL_ESTATE"
    FOOD_BEVERAGE = "FOOD_BEVERAGE"
    HEALTHCARE = "HEALTHCARE"
    CONSTRUCTION = "CONSTRUCTION"
    TRANSPORT = "TRANSPORT"
    ENERGY = "ENERGY"
    OTHER = "OTHER"


# =============================================================================
# Pydantic Sub-Config Models
# =============================================================================


class OrganizationConfig(BaseModel):
    """Configuration for the organization profile."""
    name: str = Field("", description="Legal entity name")
    neutrality_type: NeutralityType = Field(
        NeutralityType.CORPORATE,
        description="Carbon neutrality boundary type"
    )
    sector: OrganizationSector = Field(
        OrganizationSector.MANUFACTURING,
        description="Primary business sector"
    )
    size: OrganizationSize = Field(
        OrganizationSize.LARGE,
        description="Organization size classification"
    )
    region: str = Field("EU", description="Primary operating region")
    country: str = Field("DE", description="Headquarters country (ISO 3166-1 alpha-2)")
    revenue_eur: Optional[float] = Field(None, ge=0, description="Annual revenue in EUR")
    employee_count: Optional[int] = Field(None, ge=0, description="Total employees")
    fiscal_year_end: str = Field("12-31", description="Fiscal year end (MM-DD format)")
    erp_system: ERPType = Field(ERPType.NONE, description="ERP system in use")

    @field_validator("fiscal_year_end")
    @classmethod
    def validate_fiscal_year_end(cls, v: str) -> str:
        """Validate fiscal year end is in MM-DD format."""
        parts = v.split("-")
        if len(parts) != 2:
            raise ValueError(f"fiscal_year_end must be in MM-DD format, got: {v}")
        month, day = int(parts[0]), int(parts[1])
        if month < 1 or month > 12 or day < 1 or day > 31:
            raise ValueError(f"Invalid date in fiscal_year_end: {v}")
        return v


class BoundaryConfig(BaseModel):
    """Configuration for GHG Protocol organizational boundary."""
    consolidation_approach: ConsolidationApproach = Field(
        ConsolidationApproach.OPERATIONAL_CONTROL,
        description="GHG Protocol consolidation approach"
    )
    entities_included: List[str] = Field(
        default_factory=list,
        description="Legal entities included in boundary"
    )
    reporting_currency: str = Field("EUR", description="Reporting currency (ISO 4217)")
    include_joint_ventures: bool = Field(False, description="Include JVs in boundary")
    include_scope3: bool = Field(True, description="Include Scope 3 in neutrality boundary")
    scope3_categories: List[int] = Field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12],
        description="Scope 3 categories included"
    )
    materiality_threshold_pct: float = Field(
        1.0, ge=0.0, le=100.0,
        description="Materiality threshold for emission source inclusion (%)"
    )
    de_minimis_pct: float = Field(
        5.0, ge=0.0, le=100.0,
        description="De minimis aggregate exclusion threshold (%)"
    )

    @field_validator("reporting_currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Validate currency is 3-letter ISO code."""
        if len(v) != 3 or not v.isalpha():
            raise ValueError(f"reporting_currency must be 3-letter ISO code, got: {v}")
        return v.upper()

    @field_validator("scope3_categories")
    @classmethod
    def validate_scope3_categories(cls, v: List[int]) -> List[int]:
        """Validate Scope 3 category numbers are 1-15."""
        invalid = [c for c in v if c < 1 or c > 15]
        if invalid:
            raise ValueError(f"Invalid Scope 3 categories: {invalid}. Must be 1-15.")
        return sorted(set(v))


class FootprintConfig(BaseModel):
    """Configuration for footprint quantification."""
    scope1_data_source: DataSourceType = Field(
        DataSourceType.MANUAL, description="Data source for Scope 1"
    )
    scope2_data_source: DataSourceType = Field(
        DataSourceType.MANUAL, description="Data source for Scope 2"
    )
    scope3_data_source: DataSourceType = Field(
        DataSourceType.MANUAL, description="Data source for Scope 3"
    )
    scope2_methods: List[str] = Field(
        default_factory=lambda: ["location_based", "market_based"],
        description="Scope 2 calculation methods"
    )
    emission_factors_source: str = Field(
        "IPCC_AR6_2022", description="Emission factors dataset"
    )
    include_biogenic_co2: bool = Field(False, description="Include biogenic CO2")
    gwp_source: str = Field("IPCC_AR6", description="GWP source")
    gwp_timeframe: int = Field(100, description="GWP timeframe (years)")
    uncertainty_assessment: bool = Field(True, description="Enable uncertainty quantification")
    data_quality_scoring: bool = Field(True, description="Enable data quality scoring")


class CarbonManagementPlanConfig(BaseModel):
    """Configuration for carbon management plan (reduction before offset)."""
    planning_horizon_years: int = Field(
        5, ge=1, le=30, description="Planning horizon"
    )
    reduction_first_strategy: bool = Field(
        True, description="Prioritize reduction over offsetting"
    )
    min_reduction_before_offset_pct: float = Field(
        50.0, ge=0.0, le=100.0,
        description="Minimum reduction effort before offsetting (%)"
    )
    internal_carbon_price_eur: float = Field(
        100.0, ge=0.0, description="Internal carbon price (EUR/tCO2e)"
    )
    max_offset_reliance_pct: float = Field(
        50.0, ge=0.0, le=100.0,
        description="Maximum reliance on offsets (%)"
    )
    annual_reduction_target_pct: float = Field(
        4.2, ge=0.0, le=50.0,
        description="Annual emission reduction target (%)"
    )


class CreditQualityConfig(BaseModel):
    """Configuration for carbon credit quality assessment."""
    min_quality_score: int = Field(
        65, ge=0, le=100, description="Minimum quality score"
    )
    icvcm_ccp_compliance: bool = Field(
        True, description="Require ICVCM CCP compliance"
    )
    preferred_standards: List[str] = Field(
        default_factory=lambda: ["verra_vcs", "gold_standard"],
        description="Preferred credit standards"
    )
    min_additionality_score: int = Field(
        7, ge=0, le=10, description="Minimum additionality score"
    )
    min_permanence_score: int = Field(
        7, ge=0, le=10, description="Minimum permanence score"
    )
    require_sdg_contribution: bool = Field(
        True, description="Require SDG co-benefits"
    )


class PortfolioOptimizationConfig(BaseModel):
    """Configuration for credit portfolio optimization."""
    max_nature_based_pct: float = Field(
        40.0, ge=0.0, le=100.0, description="Max nature-based credits (%)"
    )
    max_avoidance_pct: float = Field(
        50.0, ge=0.0, le=100.0, description="Max avoidance credits (%)"
    )
    min_removal_pct: float = Field(
        20.0, ge=0.0, le=100.0, description="Min removal credits (%)"
    )
    diversification_min_types: int = Field(
        3, ge=1, le=20, description="Minimum project types in portfolio"
    )
    vintage_max_age_years: int = Field(
        5, ge=1, le=10, description="Maximum credit vintage age"
    )
    geographic_diversification: bool = Field(
        True, description="Require geographic diversification"
    )
    budget_constraint_enabled: bool = Field(
        True, description="Enable budget-constrained optimization"
    )


class RegistryRetirementConfig(BaseModel):
    """Configuration for registry and credit retirement."""
    auto_retirement: bool = Field(
        False, description="Enable automatic retirement"
    )
    retirement_timing: str = Field(
        "within_reporting_period", description="Retirement timing requirement"
    )
    supported_registries: List[str] = Field(
        default_factory=lambda: ["verra", "gold_standard", "american_carbon_registry"],
        description="Supported credit registries"
    )
    serial_number_tracking: bool = Field(
        True, description="Track credit serial numbers"
    )
    double_counting_prevention: bool = Field(
        True, description="Prevent double counting"
    )


class NeutralizationBalanceConfig(BaseModel):
    """Configuration for neutralization balance calculations."""
    balance_method: BalanceMethod = Field(
        BalanceMethod.ANNUAL, description="Balance calculation method"
    )
    allow_forward_credits: bool = Field(
        False, description="Allow forward-dated credits"
    )
    buffer_pool_pct: float = Field(
        10.0, ge=0.0, le=50.0, description="Buffer pool contribution (%)"
    )
    shortfall_action: str = Field(
        "purchase_additional", description="Action on balance shortfall"
    )
    surplus_carryover: bool = Field(
        False, description="Allow surplus credit carryover"
    )
    verification_frequency: str = Field(
        "annual", description="Balance verification frequency"
    )


class ClaimsSubstantiationConfig(BaseModel):
    """Configuration for carbon neutrality claims."""
    claim_type: ClaimType = Field(
        ClaimType.CARBON_NEUTRAL, description="Claim type"
    )
    standard: NeutralizationStandard = Field(
        NeutralizationStandard.ISO_14068_1, description="Primary standard"
    )
    pas2060_compliance: bool = Field(True, description="PAS 2060 compliance")
    vcmi_claims_code: bool = Field(False, description="VCMI Claims Code compliance")
    public_disclosure_required: bool = Field(True, description="Public disclosure")
    third_party_verification: bool = Field(True, description="Third-party verification")
    claim_period: str = Field("annual", description="Claim period")


class VerificationPackageConfig(BaseModel):
    """Configuration for verification package."""
    assurance_level: AssuranceLevel = Field(
        AssuranceLevel.LIMITED, description="Assurance level"
    )
    standard: str = Field("ISAE_3410", description="Assurance standard")
    third_party_verifier: bool = Field(True, description="Use third-party verifier")
    evidence_completeness_target_pct: float = Field(
        95.0, ge=50.0, le=100.0, description="Evidence completeness target (%)"
    )
    documentation_format: str = Field("PDF", description="Documentation format")


class AnnualCycleConfig(BaseModel):
    """Configuration for annual cycle management."""
    cycle_start_month: int = Field(1, ge=1, le=12, description="Cycle start month")
    cycle_end_month: int = Field(12, ge=1, le=12, description="Cycle end month")
    milestone_frequency: MilestoneFrequency = Field(
        MilestoneFrequency.QUARTERLY, description="Milestone frequency"
    )
    progress_reporting: bool = Field(True, description="Enable progress reporting")
    recertification_reminder_days: int = Field(
        90, ge=14, le=365, description="Recertification reminder (days before)"
    )
    continuous_improvement_tracking: bool = Field(
        True, description="Track continuous improvement"
    )


class PermanenceRiskConfig(BaseModel):
    """Configuration for permanence risk assessment."""
    risk_assessment_frequency: str = Field(
        "annual", description="Assessment frequency"
    )
    buffer_contribution_pct: float = Field(
        15.0, ge=0.0, le=50.0, description="Buffer contribution (%)"
    )
    reversal_monitoring: bool = Field(True, description="Monitor reversals")
    replacement_trigger_threshold_pct: float = Field(
        10.0, ge=1.0, le=50.0, description="Replacement trigger threshold (%)"
    )
    insurance_coverage_required: bool = Field(
        False, description="Require insurance coverage"
    )


class ReportingConfig(BaseModel):
    """Configuration for multi-framework reporting."""
    formats: List[ReportFormat] = Field(
        default_factory=lambda: [ReportFormat.PDF, ReportFormat.HTML],
        description="Output formats"
    )
    include_iso14068_mapping: bool = Field(True, description="Map to ISO 14068-1")
    include_pas2060_mapping: bool = Field(True, description="Map to PAS 2060")
    include_ghg_protocol_mapping: bool = Field(True, description="Map to GHG Protocol")
    include_cdp_mapping: bool = Field(False, description="Map to CDP")
    language: str = Field("en", description="Primary report language")
    multi_language_support: bool = Field(False, description="Multi-language")
    supported_languages: List[str] = Field(
        default_factory=lambda: ["en"], description="Supported languages"
    )
    watermark_draft: bool = Field(True, description="Watermark drafts")


class PerformanceConfig(BaseModel):
    """Configuration for runtime performance tuning."""
    cache_enabled: bool = Field(True, description="Enable caching")
    cache_ttl_seconds: int = Field(3600, ge=60, le=86400, description="Cache TTL")
    max_concurrent_calcs: int = Field(4, ge=1, le=32, description="Max concurrency")
    timeout_seconds: int = Field(300, ge=30, le=3600, description="Timeout")
    batch_size: int = Field(1000, ge=100, le=10000, description="Batch size")
    memory_limit_mb: int = Field(4096, ge=512, le=32768, description="Memory limit (MB)")


class AuditTrailConfig(BaseModel):
    """Configuration for audit trail and provenance tracking."""
    enabled: bool = Field(True, description="Enable audit trail")
    sha256_provenance: bool = Field(True, description="SHA-256 provenance hashing")
    calculation_logging: bool = Field(True, description="Log calculation steps")
    assumption_tracking: bool = Field(True, description="Track assumptions")
    data_lineage_enabled: bool = Field(True, description="Track data lineage")
    retention_years: int = Field(7, ge=1, le=15, description="Retention period")
    external_audit_export: bool = Field(True, description="Export for external auditors")


# =============================================================================
# Main Configuration Model
# =============================================================================


class CarbonNeutralConfig(BaseModel):
    """Main configuration for PACK-024 Carbon Neutral Pack.

    Root configuration containing all sub-configurations for comprehensive
    carbon neutrality lifecycle management: footprint quantification, reduction
    planning, credit procurement, neutralization balance, claims substantiation,
    and third-party verification.
    """

    # Temporal settings
    reporting_year: int = Field(
        DEFAULT_REPORTING_YEAR, ge=2020, le=2035,
        description="Current reporting year"
    )
    base_year: int = Field(
        DEFAULT_BASE_YEAR, ge=2015, le=2030,
        description="Base year for emissions baseline"
    )
    pack_version: str = Field("1.0.0", description="Pack configuration version")
    config_created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )

    # Sub-configurations
    organization: OrganizationConfig = Field(
        default_factory=OrganizationConfig, description="Organization profile"
    )
    boundary: BoundaryConfig = Field(
        default_factory=BoundaryConfig, description="GHG boundary configuration"
    )
    footprint: FootprintConfig = Field(
        default_factory=FootprintConfig, description="Footprint quantification"
    )
    carbon_management_plan: CarbonManagementPlanConfig = Field(
        default_factory=CarbonManagementPlanConfig, description="Carbon management plan"
    )
    credit_quality: CreditQualityConfig = Field(
        default_factory=CreditQualityConfig, description="Credit quality assessment"
    )
    portfolio_optimization: PortfolioOptimizationConfig = Field(
        default_factory=PortfolioOptimizationConfig, description="Portfolio optimization"
    )
    registry_retirement: RegistryRetirementConfig = Field(
        default_factory=RegistryRetirementConfig, description="Registry retirement"
    )
    neutralization_balance: NeutralizationBalanceConfig = Field(
        default_factory=NeutralizationBalanceConfig, description="Neutralization balance"
    )
    claims_substantiation: ClaimsSubstantiationConfig = Field(
        default_factory=ClaimsSubstantiationConfig, description="Claims substantiation"
    )
    verification_package: VerificationPackageConfig = Field(
        default_factory=VerificationPackageConfig, description="Verification package"
    )
    annual_cycle: AnnualCycleConfig = Field(
        default_factory=AnnualCycleConfig, description="Annual cycle management"
    )
    permanence_risk: PermanenceRiskConfig = Field(
        default_factory=PermanenceRiskConfig, description="Permanence risk"
    )
    reporting: ReportingConfig = Field(
        default_factory=ReportingConfig, description="Multi-framework reporting"
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig, description="Performance tuning"
    )
    audit_trail: AuditTrailConfig = Field(
        default_factory=AuditTrailConfig, description="Audit trail"
    )

    @model_validator(mode="after")
    def validate_base_year_before_reporting(self) -> "CarbonNeutralConfig":
        """Ensure base year is not after reporting year."""
        if self.base_year > self.reporting_year:
            raise ValueError(
                f"base_year ({self.base_year}) must not be after "
                f"reporting_year ({self.reporting_year})"
            )
        return self

    def get_enabled_engines(self) -> List[str]:
        """Return list of engines that should be enabled.

        Returns:
            List of engine identifier strings.
        """
        engines = [
            "footprint_quantification",
            "carbon_management_plan",
            "credit_quality",
            "portfolio_optimization",
            "registry_retirement",
            "neutralization_balance",
            "claims_substantiation",
            "verification_package",
            "annual_cycle",
            "permanence_risk",
        ]
        return engines


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """Top-level pack configuration wrapper.

    Handles preset loading, environment variable overrides, and
    configuration merging.

    Example:
        >>> config = PackConfig.from_preset("corporate_neutrality")
        >>> config = PackConfig.from_yaml("path/to/config.yaml")
    """

    pack: CarbonNeutralConfig = Field(
        default_factory=CarbonNeutralConfig,
        description="Main Carbon Neutral configuration"
    )
    preset_name: Optional[str] = Field(None, description="Name of loaded preset")
    config_version: str = Field("1.0.0", description="Configuration schema version")
    pack_id: str = Field("PACK-024-carbon-neutral", description="Pack identifier")

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "PackConfig":
        """Load configuration from a named preset.

        Args:
            preset_name: Name of the preset.
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
        env_overrides = _get_env_overrides("CARBON_NEUTRAL_")
        if env_overrides:
            preset_data = _merge_config(preset_data, env_overrides)

        # Apply explicit overrides
        if overrides:
            preset_data = _merge_config(preset_data, overrides)

        pack_config = CarbonNeutralConfig(**preset_data)
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

        pack_config = CarbonNeutralConfig(**config_data)
        return cls(pack=pack_config)

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


# =============================================================================
# Utility Functions
# =============================================================================


def load_preset(
    preset_name: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> PackConfig:
    """Load a named preset configuration.

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
        CARBON_NEUTRAL_REPORTING_YEAR=2026
        CARBON_NEUTRAL_CREDIT_QUALITY__MIN_QUALITY_SCORE=70

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


def validate_config(config: CarbonNeutralConfig) -> List[str]:
    """Validate a Carbon Neutral configuration and return any warnings.

    Performs cross-field validation beyond what Pydantic validators cover.
    Returns advisory warnings, not hard errors.

    Args:
        config: CarbonNeutralConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check organization name is set
    if not config.organization.name:
        warnings.append(
            "Organization name is empty. Set organization.name for meaningful reports."
        )

    # Check reduction-first strategy
    if config.carbon_management_plan.max_offset_reliance_pct > 80.0:
        warnings.append(
            f"Offset reliance ({config.carbon_management_plan.max_offset_reliance_pct}%) "
            f"is very high. ISO 14068-1 and PAS 2060 require reduction-first approach."
        )

    # Check credit quality
    if config.credit_quality.min_quality_score < 50:
        warnings.append(
            f"Credit quality minimum ({config.credit_quality.min_quality_score}) is low. "
            f"ICVCM Core Carbon Principles recommend quality score >= 65."
        )

    # Check portfolio diversification
    if config.portfolio_optimization.min_removal_pct < 10.0:
        warnings.append(
            f"Removal credit minimum ({config.portfolio_optimization.min_removal_pct}%) "
            f"is low. Oxford Principles recommend increasing removal share over time."
        )

    if config.portfolio_optimization.vintage_max_age_years > 7:
        warnings.append(
            f"Maximum vintage age ({config.portfolio_optimization.vintage_max_age_years} years) "
            f"exceeds recommended 5 years. Consider fresher credits."
        )

    # Check claims standard
    if not config.claims_substantiation.public_disclosure_required:
        warnings.append(
            "Public disclosure is disabled. Both ISO 14068-1 and PAS 2060 "
            "require public disclosure of carbon neutrality claims."
        )

    # Check verification
    if not config.claims_substantiation.third_party_verification:
        warnings.append(
            "Third-party verification is disabled. PAS 2060 requires independent "
            "third-party verification for carbon neutrality claims."
        )

    # Check balance
    if config.neutralization_balance.allow_forward_credits:
        warnings.append(
            "Forward credits are enabled. ISO 14068-1 requires credits to be "
            "retired within or before the claiming period."
        )

    # Check reporting frameworks
    if not any([
        config.reporting.include_iso14068_mapping,
        config.reporting.include_pas2060_mapping,
    ]):
        warnings.append(
            "No neutrality standard mapping enabled. Enable ISO 14068-1 or "
            "PAS 2060 mapping for compliant neutrality claims."
        )

    return warnings


def list_available_presets() -> Dict[str, str]:
    """List all available configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return SUPPORTED_PRESETS.copy()
