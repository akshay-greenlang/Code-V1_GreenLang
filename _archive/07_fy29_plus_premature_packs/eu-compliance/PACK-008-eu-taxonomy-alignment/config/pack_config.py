"""
PACK-008 EU Taxonomy Alignment Pack - Configuration Manager

This module implements the TaxonomyAlignmentConfig and PackConfig classes that load,
merge, and validate all configuration for the EU Taxonomy Alignment Pack. It provides
comprehensive Pydantic v2 models for every aspect of EU Taxonomy Regulation compliance:
eligibility screening, substantial contribution assessment, DNSH evaluation, minimum
safeguards verification, KPI calculation, GAR computation, and disclosure generation.

Supported Organization Types:
    - Non-Financial Undertakings (Turnover/CapEx/OpEx KPIs)
    - Financial Institutions (GAR/BTAR, EBA Pillar 3)
    - Asset Managers (Fund-level taxonomy ratios, SFDR alignment)

Environmental Objectives:
    1. Climate Change Mitigation (CCM)
    2. Climate Change Adaptation (CCA)
    3. Sustainable Use and Protection of Water and Marine Resources (WTR)
    4. Transition to a Circular Economy (CE)
    5. Pollution Prevention and Control (PPC)
    6. Protection and Restoration of Biodiversity and Ecosystems (BIO)

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Size preset (non_financial_undertaking / financial_institution / etc.)
    3. Sector preset (energy / manufacturing / real_estate / etc.)
    4. Environment overrides (TAXONOMY_PACK_* environment variables)
    5. Explicit runtime overrides

Regulatory Context:
    - EU Taxonomy Regulation (EU) 2020/852
    - Climate Delegated Act (EU) 2021/2139
    - Environmental Delegated Act (EU) 2023/2486
    - Complementary Climate DA (EU) 2022/1214
    - Disclosures DA (EU) 2021/2178
    - Simplification DA 2025 (Omnibus)

Example:
    >>> config = PackConfig.load(
    ...     size_preset="non_financial_undertaking",
    ...     sector_preset="manufacturing",
    ... )
    >>> print(config.pack.metadata_display_name)
    'EU Taxonomy Alignment Pack'
    >>> print(config.pack.objectives_in_scope)
    [CCM, CCA, WTR, CE, PPC, BIO]
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
# Enums - EU Taxonomy-specific enumeration types
# =============================================================================


class EnvironmentalObjective(str, Enum):
    """Six environmental objectives defined in EU Taxonomy Regulation Article 9."""

    CCM = "CCM"  # Climate Change Mitigation
    CCA = "CCA"  # Climate Change Adaptation
    WTR = "WTR"  # Sustainable Use and Protection of Water and Marine Resources
    CE = "CE"  # Transition to a Circular Economy
    PPC = "PPC"  # Pollution Prevention and Control
    BIO = "BIO"  # Protection and Restoration of Biodiversity and Ecosystems


class AlignmentStatus(str, Enum):
    """Taxonomy alignment determination status."""

    NOT_SCREENED = "NOT_SCREENED"
    ELIGIBLE = "ELIGIBLE"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    ALIGNED = "ALIGNED"
    NOT_ALIGNED = "NOT_ALIGNED"
    PARTIALLY_ALIGNED = "PARTIALLY_ALIGNED"


class ActivityType(str, Enum):
    """Economic activity classification within the taxonomy framework."""

    ELIGIBLE = "ELIGIBLE"
    ALIGNED = "ALIGNED"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    NOT_ALIGNED = "NOT_ALIGNED"
    TRANSITIONAL = "TRANSITIONAL"
    ENABLING = "ENABLING"


class OrganizationType(str, Enum):
    """Organization type for taxonomy reporting obligations."""

    NON_FINANCIAL_UNDERTAKING = "NON_FINANCIAL_UNDERTAKING"
    FINANCIAL_INSTITUTION = "FINANCIAL_INSTITUTION"
    ASSET_MANAGER = "ASSET_MANAGER"


class KPIType(str, Enum):
    """Mandatory KPI types for non-financial undertakings per Article 8 DA."""

    TURNOVER = "TURNOVER"
    CAPEX = "CAPEX"
    OPEX = "OPEX"


class DelegatedActVersion(str, Enum):
    """Delegated Act versions for TSC criteria sourcing."""

    CLIMATE_DA_2021 = "CLIMATE_DA_2021"  # (EU) 2021/2139
    ENVIRONMENTAL_DA_2023 = "ENVIRONMENTAL_DA_2023"  # (EU) 2023/2486
    COMPLEMENTARY_DA_2022 = "COMPLEMENTARY_DA_2022"  # (EU) 2022/1214
    DISCLOSURES_DA_2021 = "DISCLOSURES_DA_2021"  # (EU) 2021/2178
    SIMPLIFICATION_DA_2025 = "SIMPLIFICATION_DA_2025"  # Omnibus package


class ExposureType(str, Enum):
    """Exposure types for GAR calculation (financial institutions)."""

    CORPORATE_LOANS = "CORPORATE_LOANS"
    DEBT_SECURITIES = "DEBT_SECURITIES"
    EQUITY_HOLDINGS = "EQUITY_HOLDINGS"
    RESIDENTIAL_MORTGAGES = "RESIDENTIAL_MORTGAGES"
    COMMERCIAL_MORTGAGES = "COMMERCIAL_MORTGAGES"
    PROJECT_FINANCE = "PROJECT_FINANCE"
    INTERBANK_LOANS = "INTERBANK_LOANS"
    SOVEREIGN_EXPOSURES = "SOVEREIGN_EXPOSURES"


class SCEvaluationMode(str, Enum):
    """Substantial contribution evaluation mode."""

    STRICT = "STRICT"  # All TSC must be met
    STANDARD = "STANDARD"  # Standard threshold evaluation
    SIMPLIFIED = "SIMPLIFIED"  # Simplified for SMEs


class ThresholdStrictness(str, Enum):
    """TSC threshold strictness levels."""

    STRICT = "STRICT"  # No tolerance, exact threshold compliance
    STANDARD = "STANDARD"  # Standard margin of error
    LENIENT = "LENIENT"  # Wider tolerance for transitional activities


class MinimumSafeguardsMode(str, Enum):
    """Minimum safeguards assessment mode."""

    FULL = "FULL"  # Full assessment with evidence
    PROCEDURAL = "PROCEDURAL"  # Procedural checks only
    DECLARATION = "DECLARATION"  # Self-declaration based


class ScreeningMode(str, Enum):
    """Eligibility screening mode."""

    NACE_BASED = "NACE_BASED"  # Screen by NACE code mapping
    ACTIVITY_BASED = "ACTIVITY_BASED"  # Screen by activity description
    HYBRID = "HYBRID"  # Combine NACE and activity-based screening


class NACEVersion(str, Enum):
    """NACE classification system version."""

    NACE_REV2 = "NACE_REV2"
    NACE_REV2_1 = "NACE_REV2_1"


class ReportingPeriod(str, Enum):
    """Disclosure reporting period frequency."""

    ANNUAL = "ANNUAL"
    SEMI_ANNUAL = "SEMI_ANNUAL"
    QUARTERLY = "QUARTERLY"


class DisclosureFormat(str, Enum):
    """Disclosure output format."""

    PDF = "PDF"
    XLSX = "XLSX"
    XBRL = "XBRL"
    IXBRL = "IXBRL"
    JSON = "JSON"
    HTML = "HTML"


class GARType(str, Enum):
    """Green Asset Ratio calculation type."""

    STOCK = "STOCK"  # On-balance-sheet assets
    FLOW = "FLOW"  # New originations


class EPCRating(str, Enum):
    """Energy Performance Certificate ratings for real estate exposures."""

    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


class DataQualityLevel(str, Enum):
    """Data quality classification for taxonomy assessments."""

    HIGH = "HIGH"  # Primary data, audited
    MEDIUM = "MEDIUM"  # Secondary data, verified
    LOW = "LOW"  # Estimated or proxy data
    UNKNOWN = "UNKNOWN"  # Quality not assessed


class CrossFramework(str, Enum):
    """Cross-framework regulation linkages."""

    CSRD_ESRS_E1 = "CSRD_ESRS_E1"  # Climate-related disclosures
    SFDR = "SFDR"  # Sustainable Finance Disclosure Regulation
    TCFD = "TCFD"  # Task Force on Climate-related Financial Disclosures
    CDP = "CDP"  # Carbon Disclosure Project
    EBA_PILLAR3 = "EBA_PILLAR3"  # EBA Pillar 3 ESG disclosures


class TaxonomySector(str, Enum):
    """Major taxonomy-relevant economic sectors."""

    ENERGY = "ENERGY"
    MANUFACTURING = "MANUFACTURING"
    REAL_ESTATE = "REAL_ESTATE"
    TRANSPORT = "TRANSPORT"
    FORESTRY_AGRICULTURE = "FORESTRY_AGRICULTURE"
    FINANCIAL_SERVICES = "FINANCIAL_SERVICES"
    WATER_SUPPLY = "WATER_SUPPLY"
    WASTE_MANAGEMENT = "WASTE_MANAGEMENT"
    ICT = "ICT"
    PROFESSIONAL_SERVICES = "PROFESSIONAL_SERVICES"


# =============================================================================
# Reference Data Constants - EU Taxonomy regulatory reference data
# =============================================================================

# All 6 environmental objectives
ALL_ENVIRONMENTAL_OBJECTIVES: List[EnvironmentalObjective] = list(EnvironmentalObjective)

# Climate-only objectives (first reporting wave)
CLIMATE_OBJECTIVES: List[EnvironmentalObjective] = [
    EnvironmentalObjective.CCM,
    EnvironmentalObjective.CCA,
]

# Objectives covered by Climate DA (EU) 2021/2139
CLIMATE_DA_OBJECTIVES: List[EnvironmentalObjective] = [
    EnvironmentalObjective.CCM,
    EnvironmentalObjective.CCA,
]

# Objectives covered by Environmental DA (EU) 2023/2486
ENVIRONMENTAL_DA_OBJECTIVES: List[EnvironmentalObjective] = [
    EnvironmentalObjective.WTR,
    EnvironmentalObjective.CE,
    EnvironmentalObjective.PPC,
    EnvironmentalObjective.BIO,
]

# Objective display names
OBJECTIVE_DISPLAY_NAMES: Dict[str, str] = {
    "CCM": "Climate Change Mitigation",
    "CCA": "Climate Change Adaptation",
    "WTR": "Sustainable Use and Protection of Water and Marine Resources",
    "CE": "Transition to a Circular Economy",
    "PPC": "Pollution Prevention and Control",
    "BIO": "Protection and Restoration of Biodiversity and Ecosystems",
}

# Delegated Act mapping to objectives
DELEGATED_ACT_OBJECTIVES: Dict[str, List[str]] = {
    "CLIMATE_DA_2021": ["CCM", "CCA"],
    "ENVIRONMENTAL_DA_2023": ["WTR", "CE", "PPC", "BIO"],
    "COMPLEMENTARY_DA_2022": ["CCM"],  # Nuclear and gas activities
}

# NACE sector to taxonomy activity mapping (top-level)
NACE_TAXONOMY_SECTORS: Dict[str, Dict[str, Any]] = {
    "A": {
        "name": "Agriculture, forestry and fishing",
        "taxonomy_activities": 14,
        "objectives": ["CCM", "CCA", "BIO"],
    },
    "B": {
        "name": "Mining and quarrying",
        "taxonomy_activities": 2,
        "objectives": ["CCM"],
    },
    "C": {
        "name": "Manufacturing",
        "taxonomy_activities": 28,
        "objectives": ["CCM", "CCA", "CE", "PPC"],
    },
    "D": {
        "name": "Electricity, gas, steam and air conditioning supply",
        "taxonomy_activities": 22,
        "objectives": ["CCM", "CCA"],
    },
    "E": {
        "name": "Water supply, sewerage, waste management",
        "taxonomy_activities": 12,
        "objectives": ["CCM", "CCA", "WTR", "CE", "PPC"],
    },
    "F": {
        "name": "Construction",
        "taxonomy_activities": 8,
        "objectives": ["CCM", "CCA", "CE"],
    },
    "G": {
        "name": "Wholesale and retail trade",
        "taxonomy_activities": 3,
        "objectives": ["CCM"],
    },
    "H": {
        "name": "Transportation and storage",
        "taxonomy_activities": 18,
        "objectives": ["CCM", "CCA", "PPC"],
    },
    "J": {
        "name": "Information and communication",
        "taxonomy_activities": 6,
        "objectives": ["CCM", "CCA"],
    },
    "K": {
        "name": "Financial and insurance activities",
        "taxonomy_activities": 4,
        "objectives": ["CCM", "CCA"],
    },
    "L": {
        "name": "Real estate activities",
        "taxonomy_activities": 3,
        "objectives": ["CCM", "CCA", "CE"],
    },
    "M": {
        "name": "Professional, scientific and technical activities",
        "taxonomy_activities": 5,
        "objectives": ["CCM", "CCA"],
    },
}

# Minimum safeguards reference frameworks
MINIMUM_SAFEGUARDS_FRAMEWORKS: Dict[str, Dict[str, str]] = {
    "human_rights": {
        "primary": "UN Guiding Principles on Business and Human Rights (UNGP)",
        "secondary": "OECD Guidelines for Multinational Enterprises",
        "article": "Article 18(1)",
    },
    "anti_corruption": {
        "primary": "UN Convention against Corruption",
        "secondary": "OECD Anti-Bribery Convention",
        "article": "Article 18(1)",
    },
    "taxation": {
        "primary": "EU Tax Good Governance principles",
        "secondary": "OECD BEPS recommendations",
        "article": "Article 18(1)",
    },
    "fair_competition": {
        "primary": "EU Competition Law",
        "secondary": "ILO Core Labour Standards",
        "article": "Article 18(1)",
    },
}

# Article 8 DA mandatory KPI tables
ARTICLE_8_MANDATORY_TABLES: List[Dict[str, str]] = [
    {
        "id": "turnover_table",
        "name": "Proportion of turnover from taxonomy-aligned activities",
        "kpi": "TURNOVER",
        "applies_to": "NON_FINANCIAL_UNDERTAKING",
    },
    {
        "id": "capex_table",
        "name": "Proportion of CapEx from taxonomy-aligned activities",
        "kpi": "CAPEX",
        "applies_to": "NON_FINANCIAL_UNDERTAKING",
    },
    {
        "id": "opex_table",
        "name": "Proportion of OpEx from taxonomy-aligned activities",
        "kpi": "OPEX",
        "applies_to": "NON_FINANCIAL_UNDERTAKING",
    },
    {
        "id": "gar_stock_table",
        "name": "Green Asset Ratio (stock)",
        "kpi": "GAR_STOCK",
        "applies_to": "FINANCIAL_INSTITUTION",
    },
    {
        "id": "gar_flow_table",
        "name": "Green Asset Ratio (flow)",
        "kpi": "GAR_FLOW",
        "applies_to": "FINANCIAL_INSTITUTION",
    },
    {
        "id": "btar_table",
        "name": "Banking Book Taxonomy Alignment Ratio",
        "kpi": "BTAR",
        "applies_to": "FINANCIAL_INSTITUTION",
    },
]

# EBA Pillar 3 ESG disclosure templates
EBA_PILLAR3_TEMPLATES: List[Dict[str, str]] = [
    {"id": "template_6", "name": "Summary of KPIs on taxonomy-aligned activities"},
    {"id": "template_7", "name": "Mitigating actions - Banking book"},
    {"id": "template_8", "name": "Exposures to taxonomy-aligned economic activities"},
    {"id": "template_9", "name": "Exposures to taxonomy-eligible economic activities"},
    {"id": "template_10", "name": "Other mitigating actions not covered by the taxonomy"},
]


# =============================================================================
# Pydantic Sub-Config Models
# =============================================================================


class EligibilityConfig(BaseModel):
    """Configuration for taxonomy eligibility screening engine.

    Controls how economic activities are screened against the EU Taxonomy's
    ~240 eligible activities using NACE code mapping and activity descriptions.
    """

    screening_mode: ScreeningMode = Field(
        ScreeningMode.HYBRID,
        description="Screening approach: NACE code, activity description, or hybrid",
    )
    nace_version: NACEVersion = Field(
        NACEVersion.NACE_REV2,
        description="NACE classification version used for mapping",
    )
    min_confidence: float = Field(
        0.85,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for eligibility classification",
    )
    batch_size: int = Field(
        100,
        ge=10,
        le=10000,
        description="Batch size for activity screening operations",
    )
    include_transitional: bool = Field(
        True,
        description="Include Article 10(2) transitional activities in screening",
    )
    include_enabling: bool = Field(
        True,
        description="Include Article 16 enabling activities in screening",
    )
    revenue_weighted: bool = Field(
        True,
        description="Calculate revenue-weighted eligibility ratios",
    )
    auto_classify_nace: bool = Field(
        True,
        description="Automatically classify activities based on NACE codes",
    )

    @field_validator("min_confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Validate confidence is within practical range."""
        if v < 0.5:
            logger.warning(
                "Eligibility min_confidence below 0.5 may produce unreliable results"
            )
        return v


class SCAssessmentConfig(BaseModel):
    """Configuration for substantial contribution assessment engine.

    Controls evaluation of Technical Screening Criteria (TSC) per environmental
    objective, including threshold checking and evidence requirements.
    """

    evaluation_mode: SCEvaluationMode = Field(
        SCEvaluationMode.STANDARD,
        description="SC evaluation strictness mode",
    )
    threshold_strictness: ThresholdStrictness = Field(
        ThresholdStrictness.STANDARD,
        description="TSC threshold compliance strictness",
    )
    evidence_required: bool = Field(
        True,
        description="Require documentary evidence for TSC compliance claims",
    )
    quantitative_tolerance_pct: float = Field(
        5.0,
        ge=0.0,
        le=20.0,
        description="Tolerance percentage for quantitative TSC thresholds",
    )
    track_enabling_activities: bool = Field(
        True,
        description="Track and flag enabling activities per Article 16",
    )
    track_transitional_activities: bool = Field(
        True,
        description="Track and flag transitional activities per Article 10(2)",
    )
    require_all_quantitative: bool = Field(
        True,
        description="Require all quantitative criteria to pass for SC determination",
    )
    gap_analysis_on_fail: bool = Field(
        True,
        description="Generate gap analysis when SC assessment fails",
    )


class DNSHConfig(BaseModel):
    """Configuration for Do No Significant Harm (DNSH) assessment engine.

    Controls the 6-objective DNSH matrix evaluation that ensures an activity
    making a substantial contribution to one objective does not significantly
    harm any of the other five objectives.
    """

    objectives_assessed: List[EnvironmentalObjective] = Field(
        default_factory=lambda: list(EnvironmentalObjective),
        description="Environmental objectives to assess for DNSH (typically all 6)",
    )
    require_all_pass: bool = Field(
        True,
        description="Require DNSH pass on all assessed objectives for alignment",
    )
    climate_risk_assessment_enabled: bool = Field(
        True,
        description="Enable climate risk and vulnerability assessment for CCA DNSH",
    )
    water_framework_directive_check: bool = Field(
        True,
        description="Enable Water Framework Directive compliance check for WTR DNSH",
    )
    circular_economy_waste_hierarchy: bool = Field(
        True,
        description="Enable circular economy waste hierarchy check for CE DNSH",
    )
    pollution_threshold_check: bool = Field(
        True,
        description="Enable pollution threshold checking for PPC DNSH",
    )
    biodiversity_impact_assessment: bool = Field(
        True,
        description="Enable biodiversity impact assessment for BIO DNSH",
    )
    evidence_required: bool = Field(
        True,
        description="Require documentary evidence for DNSH compliance claims",
    )

    @field_validator("objectives_assessed")
    @classmethod
    def validate_objectives(
        cls, v: List[EnvironmentalObjective],
    ) -> List[EnvironmentalObjective]:
        """Validate at least one objective is assessed."""
        if len(v) == 0:
            raise ValueError("At least one environmental objective must be assessed for DNSH")
        return v


class MinimumSafeguardsConfig(BaseModel):
    """Configuration for minimum safeguards verification engine.

    Implements Article 18 minimum safeguards checking across four topics:
    human rights, anti-corruption, taxation, and fair competition.
    """

    human_rights_check: bool = Field(
        True,
        description="Enable human rights due diligence check (UNGP/OECD)",
    )
    anti_corruption_check: bool = Field(
        True,
        description="Enable anti-corruption procedures assessment",
    )
    taxation_check: bool = Field(
        True,
        description="Enable taxation compliance verification",
    )
    fair_competition_check: bool = Field(
        True,
        description="Enable fair competition assessment",
    )
    assessment_mode: MinimumSafeguardsMode = Field(
        MinimumSafeguardsMode.FULL,
        description="Assessment depth: full evidence, procedural only, or declaration",
    )
    require_all_pass: bool = Field(
        True,
        description="Require all 4 topics to pass for MS compliance",
    )
    grievance_mechanism_required: bool = Field(
        False,
        description="Require operational grievance mechanism per UNGP Principle 31",
    )
    supply_chain_due_diligence: bool = Field(
        False,
        description="Extend MS checks to supply chain per CSDDD alignment",
    )

    @model_validator(mode="after")
    def validate_at_least_one_check(self) -> "MinimumSafeguardsConfig":
        """Ensure at least one safeguards check is enabled."""
        checks = [
            self.human_rights_check,
            self.anti_corruption_check,
            self.taxation_check,
            self.fair_competition_check,
        ]
        if not any(checks):
            raise ValueError(
                "At least one minimum safeguards check must be enabled"
            )
        return self


class KPIConfig(BaseModel):
    """Configuration for mandatory KPI calculation engine.

    Controls calculation of turnover, CapEx, and OpEx alignment ratios as
    required by the Article 8 Delegated Act (EU) 2021/2178.
    """

    calculate_turnover: bool = Field(
        True,
        description="Calculate turnover alignment ratio",
    )
    calculate_capex: bool = Field(
        True,
        description="Calculate CapEx alignment ratio",
    )
    calculate_opex: bool = Field(
        True,
        description="Calculate OpEx alignment ratio",
    )
    double_counting_prevention: bool = Field(
        True,
        description="Prevent double-counting of activities across objectives",
    )
    capex_plan_recognition: bool = Field(
        True,
        description="Recognize CapEx plans (up to 5 years) for alignment calculation",
    )
    capex_plan_max_years: int = Field(
        5,
        ge=1,
        le=10,
        description="Maximum years for CapEx plan recognition",
    )
    eligible_vs_aligned_breakdown: bool = Field(
        True,
        description="Report both eligible and aligned KPI breakdown",
    )
    activity_level_detail: bool = Field(
        True,
        description="Report activity-level financial data breakdown",
    )
    currency: str = Field(
        "EUR",
        description="Reporting currency (ISO 4217)",
    )
    rounding_precision: int = Field(
        2,
        ge=0,
        le=6,
        description="Decimal places for percentage KPIs",
    )

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Validate currency code format."""
        if len(v) != 3 or not v.isalpha() or not v.isupper():
            raise ValueError(f"Currency must be 3-letter uppercase ISO 4217 code: {v}")
        return v


class GARConfig(BaseModel):
    """Configuration for Green Asset Ratio calculation engine.

    Controls GAR (stock and flow), BTAR, and related calculations for
    financial institutions per EBA Pillar 3 requirements.
    """

    calculate_stock_gar: bool = Field(
        True,
        description="Calculate GAR stock (on-balance-sheet assets)",
    )
    calculate_flow_gar: bool = Field(
        True,
        description="Calculate GAR flow (new originations)",
    )
    calculate_btar: bool = Field(
        True,
        description="Calculate Banking Book Taxonomy Alignment Ratio",
    )
    exposure_types: List[ExposureType] = Field(
        default_factory=lambda: [
            ExposureType.CORPORATE_LOANS,
            ExposureType.DEBT_SECURITIES,
            ExposureType.EQUITY_HOLDINGS,
            ExposureType.RESIDENTIAL_MORTGAGES,
            ExposureType.PROJECT_FINANCE,
        ],
        description="Exposure types included in GAR calculation",
    )
    epc_integration: bool = Field(
        True,
        description="Integrate Energy Performance Certificate ratings for real estate",
    )
    epc_threshold_rating: EPCRating = Field(
        EPCRating.C,
        description="Minimum EPC rating for taxonomy-aligned real estate exposures",
    )
    de_minimis_threshold: float = Field(
        0.0,
        ge=0.0,
        le=10.0,
        description="De minimis threshold percentage (below which exposures excluded)",
    )
    counterparty_data_source: str = Field(
        "DIRECT",
        description="Counterparty taxonomy data source: DIRECT, ESTIMATED, PROXY",
    )
    sovereign_exclusion: bool = Field(
        True,
        description="Exclude sovereign exposures from GAR denominator",
    )
    interbank_exclusion: bool = Field(
        True,
        description="Exclude interbank exposures from GAR denominator",
    )

    @field_validator("exposure_types")
    @classmethod
    def validate_exposure_types(cls, v: List[ExposureType]) -> List[ExposureType]:
        """Validate at least one exposure type is configured."""
        if len(v) == 0:
            raise ValueError("At least one exposure type must be configured for GAR")
        return v


class ReportingConfig(BaseModel):
    """Configuration for taxonomy reporting and disclosure generation.

    Controls Article 8 disclosure template generation, EBA Pillar 3 templates,
    XBRL tagging, and cross-framework reporting.
    """

    article8_enabled: bool = Field(
        True,
        description="Enable Article 8 mandatory disclosure generation",
    )
    eba_pillar3_enabled: bool = Field(
        False,
        description="Enable EBA Pillar 3 ESG disclosure templates (for banks)",
    )
    xbrl_tagging: bool = Field(
        False,
        description="Enable XBRL/iXBRL tagging for digital filings",
    )
    nuclear_gas_supplementary: bool = Field(
        True,
        description="Include nuclear and gas supplementary disclosures per (EU) 2022/1214",
    )
    yoy_comparison: bool = Field(
        True,
        description="Include year-over-year comparison tables",
    )
    default_format: DisclosureFormat = Field(
        DisclosureFormat.PDF,
        description="Default disclosure output format",
    )
    include_methodology_note: bool = Field(
        True,
        description="Include methodology and assumptions note in disclosures",
    )
    include_audit_opinion: bool = Field(
        False,
        description="Include space for external auditor opinion",
    )
    language: str = Field(
        "en",
        description="Report language code (ISO 639-1)",
    )
    timezone: str = Field(
        "UTC",
        description="Timezone for timestamps in reports",
    )

    cross_framework_targets: List[CrossFramework] = Field(
        default_factory=list,
        description="Cross-framework regulation targets for consolidated reporting",
    )


class RegulatoryConfig(BaseModel):
    """Configuration for regulatory tracking and delegated act management.

    Controls DA version management, criteria change tracking, and automatic
    migration between regulatory framework versions.
    """

    delegated_act_version: DelegatedActVersion = Field(
        DelegatedActVersion.CLIMATE_DA_2021,
        description="Primary delegated act version for TSC sourcing",
    )
    active_delegated_acts: List[DelegatedActVersion] = Field(
        default_factory=lambda: [
            DelegatedActVersion.CLIMATE_DA_2021,
            DelegatedActVersion.ENVIRONMENTAL_DA_2023,
            DelegatedActVersion.DISCLOSURES_DA_2021,
        ],
        description="All active delegated acts for this assessment",
    )
    track_updates: bool = Field(
        True,
        description="Track delegated act updates and amendments",
    )
    auto_migration: bool = Field(
        False,
        description="Automatically migrate criteria when DA version changes",
    )
    update_check_interval_hours: int = Field(
        24,
        ge=1,
        le=168,
        description="Interval for checking regulatory updates (hours)",
    )
    include_complementary_da: bool = Field(
        True,
        description="Include Complementary DA (nuclear/gas) in assessments",
    )
    include_simplification_da: bool = Field(
        False,
        description="Apply Omnibus simplification DA 2025 rules where applicable",
    )


class TSCConfig(BaseModel):
    """Configuration for Technical Screening Criteria engine.

    Controls criteria lookup, threshold evaluation, and version management
    for the quantitative and qualitative TSC per Delegated Acts.
    """

    strict_threshold_compliance: bool = Field(
        True,
        description="Require strict compliance with quantitative thresholds",
    )
    tolerance_margin_pct: float = Field(
        0.0,
        ge=0.0,
        le=10.0,
        description="Tolerance margin for quantitative thresholds (0 = exact)",
    )
    track_criteria_changes: bool = Field(
        True,
        description="Track changes in TSC across DA versions",
    )
    gap_identification: bool = Field(
        True,
        description="Identify and report gaps for non-compliant criteria",
    )
    evidence_linking: bool = Field(
        True,
        description="Link evidence documents to individual TSC criteria",
    )
    qualitative_assessment_enabled: bool = Field(
        True,
        description="Enable qualitative criteria assessment (not just quantitative)",
    )


class TransitionActivityConfig(BaseModel):
    """Configuration for transition activity classification per Article 10(2)."""

    enabled: bool = Field(
        True,
        description="Enable transition activity identification and tracking",
    )
    best_available_technology_check: bool = Field(
        True,
        description="Verify use of best available technology",
    )
    lock_in_avoidance_check: bool = Field(
        True,
        description="Verify avoidance of carbon lock-in",
    )
    sunset_date_tracking: bool = Field(
        True,
        description="Track sunset dates for transitional status",
    )
    transition_pathway_documentation: bool = Field(
        True,
        description="Require documented transition pathway",
    )


class EnablingActivityConfig(BaseModel):
    """Configuration for enabling activity classification per Article 16."""

    enabled: bool = Field(
        True,
        description="Enable enabling activity identification and classification",
    )
    direct_enablement_verification: bool = Field(
        True,
        description="Verify direct enablement of other activities",
    )
    lifecycle_consideration: bool = Field(
        True,
        description="Consider full lifecycle impact of enabling activities",
    )
    technology_lock_in_check: bool = Field(
        True,
        description="Check for technology lock-in risk",
    )
    market_distortion_assessment: bool = Field(
        False,
        description="Assess potential market distortion from enabling classification",
    )


class DataQualityConfig(BaseModel):
    """Configuration for data quality management in taxonomy assessments."""

    min_quality_score: float = Field(
        0.80,
        ge=0.0,
        le=1.0,
        description="Minimum data quality score for assessment acceptance",
    )
    completeness_threshold: float = Field(
        0.90,
        ge=0.0,
        le=1.0,
        description="Minimum data completeness ratio",
    )
    require_primary_data: bool = Field(
        False,
        description="Require primary (audited) data for all assessments",
    )
    allow_estimates: bool = Field(
        True,
        description="Allow estimated/proxy data with quality flags",
    )
    validation_on_ingestion: bool = Field(
        True,
        description="Validate data quality on ingestion",
    )


class AuditTrailConfig(BaseModel):
    """Configuration for audit trail and provenance tracking."""

    retention_years: int = Field(
        5,
        ge=1,
        le=20,
        description="Audit trail retention period (years)",
    )
    hash_algorithm: str = Field(
        "SHA-256",
        description="Provenance hash algorithm",
    )
    export_formats: List[str] = Field(
        default_factory=lambda: ["JSON", "XML", "PDF"],
        description="Available export formats for audit trail",
    )
    immutable_log: bool = Field(
        True,
        description="Enforce immutable audit log (append-only)",
    )
    include_provenance_hash: bool = Field(
        True,
        description="Include SHA-256 provenance hash in all outputs",
    )


class DemoConfig(BaseModel):
    """Demo mode configuration for testing and training."""

    demo_mode_enabled: bool = Field(
        False,
        description="Enable demo mode with synthetic data",
    )
    use_synthetic_data: bool = Field(
        False,
        description="Use synthetic test data for all assessments",
    )
    mock_erp_responses: bool = Field(
        False,
        description="Mock ERP/finance system responses",
    )
    mock_mrv_data: bool = Field(
        False,
        description="Mock MRV agent emissions data",
    )
    tutorial_mode_enabled: bool = Field(
        False,
        description="Enable guided tutorial mode",
    )
    sample_activities_count: int = Field(
        10,
        ge=1,
        le=50,
        description="Number of sample activities in demo mode",
    )


# =============================================================================
# Main Configuration Class
# =============================================================================


class TaxonomyAlignmentConfig(BaseModel):
    """EU Taxonomy Alignment Pack main configuration.

    Central configuration class that combines all sub-configurations for
    the complete taxonomy alignment assessment pipeline. Supports all
    organization types (NFU, FI, AM) and all 6 environmental objectives.

    Attributes:
        pack_id: Unique pack identifier
        version: Pack version string
        tier: Pack tier classification
        organization_type: Target organization type (NFU, FI, AM)
        reporting_period: Disclosure reporting frequency
        objectives_in_scope: Environmental objectives included in assessment
        active_sectors: Economic sectors being assessed
        eligibility: Eligibility screening configuration
        sc_assessment: Substantial contribution assessment configuration
        dnsh: DNSH assessment configuration
        minimum_safeguards: Minimum safeguards verification configuration
        kpi: KPI calculation configuration
        gar: Green Asset Ratio configuration
        reporting: Disclosure and reporting configuration
        regulatory: Regulatory tracking configuration
        tsc: Technical screening criteria configuration
        transition_activity: Transition activity configuration
        enabling_activity: Enabling activity configuration
        data_quality: Data quality management configuration
        audit_trail: Audit trail and provenance configuration
        demo: Demo mode configuration

    Example:
        >>> config = TaxonomyAlignmentConfig(
        ...     organization_type=OrganizationType.NON_FINANCIAL_UNDERTAKING,
        ...     objectives_in_scope=[EnvironmentalObjective.CCM, EnvironmentalObjective.CCA],
        ... )
        >>> assert config.kpi.calculate_turnover is True
    """

    # Pack metadata
    pack_id: str = Field(
        "PACK-008-eu-taxonomy-alignment",
        description="Pack identifier",
    )
    version: str = Field(
        "1.0.0",
        description="Pack version",
    )
    tier: str = Field(
        "standalone",
        description="Pack tier classification",
    )

    # Organization context
    organization_type: OrganizationType = Field(
        OrganizationType.NON_FINANCIAL_UNDERTAKING,
        description="Organization type determining reporting obligations",
    )
    organization_name: str = Field(
        "",
        description="Legal name of the reporting organization",
    )
    reporting_period: ReportingPeriod = Field(
        ReportingPeriod.ANNUAL,
        description="Disclosure reporting frequency",
    )
    reporting_year: int = Field(
        2025,
        ge=2022,
        le=2030,
        description="Reporting fiscal year",
    )

    # Scope
    objectives_in_scope: List[EnvironmentalObjective] = Field(
        default_factory=lambda: list(EnvironmentalObjective),
        description="Environmental objectives included in assessment scope",
    )
    active_sectors: List[TaxonomySector] = Field(
        default_factory=list,
        description="Active economic sectors for assessment",
    )

    # Sub-configurations
    eligibility: EligibilityConfig = Field(default_factory=EligibilityConfig)
    sc_assessment: SCAssessmentConfig = Field(default_factory=SCAssessmentConfig)
    dnsh: DNSHConfig = Field(default_factory=DNSHConfig)
    minimum_safeguards: MinimumSafeguardsConfig = Field(
        default_factory=MinimumSafeguardsConfig,
    )
    kpi: KPIConfig = Field(default_factory=KPIConfig)
    gar: GARConfig = Field(default_factory=GARConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    regulatory: RegulatoryConfig = Field(default_factory=RegulatoryConfig)
    tsc: TSCConfig = Field(default_factory=TSCConfig)
    transition_activity: TransitionActivityConfig = Field(
        default_factory=TransitionActivityConfig,
    )
    enabling_activity: EnablingActivityConfig = Field(
        default_factory=EnablingActivityConfig,
    )
    data_quality: DataQualityConfig = Field(default_factory=DataQualityConfig)
    audit_trail: AuditTrailConfig = Field(default_factory=AuditTrailConfig)
    demo: DemoConfig = Field(default_factory=DemoConfig)

    @model_validator(mode="after")
    def validate_organization_config(self) -> "TaxonomyAlignmentConfig":
        """Validate configuration consistency based on organization type."""
        org_type = self.organization_type

        # Financial institutions require GAR configuration
        if org_type == OrganizationType.FINANCIAL_INSTITUTION:
            if not self.gar.calculate_stock_gar and not self.gar.calculate_flow_gar:
                raise ValueError(
                    "Financial institutions must enable at least one GAR calculation "
                    "(stock or flow)"
                )

        # Non-financial undertakings require at least one KPI
        if org_type == OrganizationType.NON_FINANCIAL_UNDERTAKING:
            kpis = [
                self.kpi.calculate_turnover,
                self.kpi.calculate_capex,
                self.kpi.calculate_opex,
            ]
            if not any(kpis):
                raise ValueError(
                    "Non-financial undertakings must enable at least one KPI "
                    "(turnover, CapEx, or OpEx)"
                )

        # Ensure objectives_in_scope is not empty
        if len(self.objectives_in_scope) == 0:
            raise ValueError("At least one environmental objective must be in scope")

        return self

    def get_active_agents(self) -> List[str]:
        """Get list of all active agents for this pack.

        Returns:
            List of agent identifiers (30 MRV + 10 data + 10 foundation + 1 app)
        """
        agents: List[str] = []

        # GL-Taxonomy-APP
        agents.append("GL-Taxonomy-APP")

        # All 30 MRV agents
        agents.extend(
            [f"AGENT-MRV-{str(i).zfill(3)}" for i in range(1, 31)]
        )

        # 10 data agents
        for agent_id in [1, 2, 3, 8, 9, 10, 11, 12, 18, 19]:
            agents.append(f"AGENT-DATA-{str(agent_id).zfill(3)}")

        # All 10 foundation agents
        agents.extend(
            [f"AGENT-FOUND-{str(i).zfill(3)}" for i in range(1, 11)]
        )

        return agents

    def get_required_delegated_acts(self) -> List[DelegatedActVersion]:
        """Determine required delegated acts based on objectives in scope.

        Returns:
            List of DelegatedActVersion enums needed for the configured objectives
        """
        required_das: Set[DelegatedActVersion] = set()

        for objective in self.objectives_in_scope:
            if objective in (EnvironmentalObjective.CCM, EnvironmentalObjective.CCA):
                required_das.add(DelegatedActVersion.CLIMATE_DA_2021)
            elif objective in (
                EnvironmentalObjective.WTR,
                EnvironmentalObjective.CE,
                EnvironmentalObjective.PPC,
                EnvironmentalObjective.BIO,
            ):
                required_das.add(DelegatedActVersion.ENVIRONMENTAL_DA_2023)

        # Always need the Disclosures DA
        required_das.add(DelegatedActVersion.DISCLOSURES_DA_2021)

        # Add Complementary DA if nuclear/gas disclosures enabled
        if self.reporting.nuclear_gas_supplementary:
            required_das.add(DelegatedActVersion.COMPLEMENTARY_DA_2022)

        return sorted(list(required_das), key=lambda x: x.value)

    def get_applicable_kpis(self) -> List[KPIType]:
        """Get applicable KPIs based on organization type and config.

        Returns:
            List of KPIType enums applicable to this configuration
        """
        kpis: List[KPIType] = []

        if self.organization_type == OrganizationType.NON_FINANCIAL_UNDERTAKING:
            if self.kpi.calculate_turnover:
                kpis.append(KPIType.TURNOVER)
            if self.kpi.calculate_capex:
                kpis.append(KPIType.CAPEX)
            if self.kpi.calculate_opex:
                kpis.append(KPIType.OPEX)

        return kpis

    def get_feature_summary(self) -> Dict[str, bool]:
        """Get summary of enabled features for this configuration.

        Returns:
            Dictionary mapping feature names to enabled status
        """
        return {
            "eligibility_screening": True,
            "substantial_contribution": True,
            "dnsh_assessment": self.dnsh.require_all_pass,
            "minimum_safeguards": any([
                self.minimum_safeguards.human_rights_check,
                self.minimum_safeguards.anti_corruption_check,
                self.minimum_safeguards.taxation_check,
                self.minimum_safeguards.fair_competition_check,
            ]),
            "kpi_turnover": self.kpi.calculate_turnover,
            "kpi_capex": self.kpi.calculate_capex,
            "kpi_opex": self.kpi.calculate_opex,
            "gar_stock": self.gar.calculate_stock_gar,
            "gar_flow": self.gar.calculate_flow_gar,
            "gar_btar": self.gar.calculate_btar,
            "article8_disclosure": self.reporting.article8_enabled,
            "eba_pillar3": self.reporting.eba_pillar3_enabled,
            "xbrl_tagging": self.reporting.xbrl_tagging,
            "nuclear_gas_supplementary": self.reporting.nuclear_gas_supplementary,
            "yoy_comparison": self.reporting.yoy_comparison,
            "capex_plan_recognition": self.kpi.capex_plan_recognition,
            "double_counting_prevention": self.kpi.double_counting_prevention,
            "transition_activities": self.transition_activity.enabled,
            "enabling_activities": self.enabling_activity.enabled,
            "regulatory_tracking": self.regulatory.track_updates,
            "cross_framework_reporting": len(self.reporting.cross_framework_targets) > 0,
            "audit_trail": self.audit_trail.include_provenance_hash,
        }

    def get_objectives_display(self) -> Dict[str, str]:
        """Get display names for in-scope objectives.

        Returns:
            Dictionary mapping objective codes to display names
        """
        return {
            obj.value: OBJECTIVE_DISPLAY_NAMES[obj.value]
            for obj in self.objectives_in_scope
        }


# =============================================================================
# PackConfig - Top-Level Configuration Loader
# =============================================================================


class PackConfig(BaseModel):
    """Top-level pack configuration loader with YAML support.

    Loads configuration from the pack manifest, applies preset overlays,
    and provides methods for export and hashing.

    Configuration Merge Order:
        1. Base defaults from TaxonomyAlignmentConfig
        2. Size preset YAML (if specified)
        3. Sector preset YAML (if specified)
        4. Environment variable overrides (TAXONOMY_PACK_*)
        5. Explicit runtime overrides

    Example:
        >>> config = PackConfig.load(
        ...     size_preset="non_financial_undertaking",
        ...     sector_preset="manufacturing",
        ... )
        >>> print(config.pack.organization_type)
        OrganizationType.NON_FINANCIAL_UNDERTAKING
        >>> print(config.get_config_hash()[:16])
        'a1b2c3d4e5f6g7h8'
    """

    pack: TaxonomyAlignmentConfig = Field(
        default_factory=TaxonomyAlignmentConfig,
    )
    loaded_from: List[str] = Field(
        default_factory=list,
        description="Configuration files loaded (in merge order)",
    )
    merge_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of configuration merge",
    )

    @classmethod
    def load(
        cls,
        size_preset: Optional[str] = None,
        sector_preset: Optional[str] = None,
        demo_mode: bool = False,
    ) -> "PackConfig":
        """Load configuration with preset and sector overlays.

        Args:
            size_preset: Size preset name (non_financial_undertaking,
                financial_institution, asset_manager, large_enterprise,
                sme_simplified)
            sector_preset: Sector preset name (energy, manufacturing,
                real_estate, transport, forestry_agriculture,
                financial_services)
            demo_mode: Enable demo mode with synthetic data

        Returns:
            Loaded PackConfig instance with merged configuration

        Raises:
            FileNotFoundError: If specified preset file does not exist
            ValueError: If configuration validation fails after merge
        """
        config = TaxonomyAlignmentConfig()
        loaded_files: List[str] = []

        # Load pack.yaml manifest for reference
        manifest_path = PACK_BASE_DIR / "pack.yaml"
        if manifest_path.exists():
            loaded_files.append(str(manifest_path))
            logger.info("Located pack manifest: %s", manifest_path)

        # Load size preset if specified
        if size_preset:
            preset_path = CONFIG_DIR / "presets" / f"{size_preset}.yaml"
            if preset_path.exists():
                with open(preset_path, "r", encoding="utf-8") as f:
                    preset_data = yaml.safe_load(f)
                if preset_data:
                    config = cls._merge_config(config, preset_data)
                    loaded_files.append(str(preset_path))
                    logger.info("Loaded size preset: %s", size_preset)
            else:
                logger.warning("Size preset not found: %s", preset_path)

        # Load sector preset if specified
        if sector_preset:
            sector_path = CONFIG_DIR / "sectors" / f"{sector_preset}.yaml"
            if sector_path.exists():
                with open(sector_path, "r", encoding="utf-8") as f:
                    sector_data = yaml.safe_load(f)
                if sector_data:
                    config = cls._merge_config(config, sector_data)
                    loaded_files.append(str(sector_path))
                    logger.info("Loaded sector preset: %s", sector_preset)
            else:
                logger.warning("Sector preset not found: %s", sector_path)

        # Apply environment variable overrides
        config = cls._apply_env_overrides(config)

        # Enable demo mode if requested
        if demo_mode:
            config.demo.demo_mode_enabled = True
            config.demo.use_synthetic_data = True
            config.demo.mock_erp_responses = True
            config.demo.mock_mrv_data = True
            logger.info("Demo mode enabled with synthetic data")

        return cls(pack=config, loaded_from=loaded_files)

    @staticmethod
    def _merge_config(
        base: TaxonomyAlignmentConfig,
        overlay: Dict[str, Any],
    ) -> TaxonomyAlignmentConfig:
        """Deep merge overlay configuration into base configuration.

        Args:
            base: Base TaxonomyAlignmentConfig instance
            overlay: Dictionary of override values to merge

        Returns:
            New TaxonomyAlignmentConfig with merged values
        """
        base_dict = base.model_dump()

        def deep_merge(d1: Dict, d2: Dict) -> Dict:
            """Recursively merge d2 into d1."""
            for key, value in d2.items():
                if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
                    d1[key] = deep_merge(d1[key], value)
                else:
                    d1[key] = value
            return d1

        merged = deep_merge(base_dict, overlay)
        return TaxonomyAlignmentConfig(**merged)

    @staticmethod
    def _apply_env_overrides(
        config: TaxonomyAlignmentConfig,
    ) -> TaxonomyAlignmentConfig:
        """Apply environment variable overrides to configuration.

        Looks for TAXONOMY_PACK_* environment variables and applies them
        as configuration overrides.

        Args:
            config: Current configuration instance

        Returns:
            Configuration with environment overrides applied
        """
        env_mapping: Dict[str, str] = {
            "TAXONOMY_PACK_ORG_TYPE": "organization_type",
            "TAXONOMY_PACK_ORG_NAME": "organization_name",
            "TAXONOMY_PACK_REPORTING_YEAR": "reporting_year",
            "TAXONOMY_PACK_DEMO_MODE": "demo.demo_mode_enabled",
        }

        config_dict = config.model_dump()

        for env_var, config_key in env_mapping.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                # Handle nested keys
                keys = config_key.split(".")
                target = config_dict
                for key in keys[:-1]:
                    target = target.setdefault(key, {})

                # Type coercion
                final_key = keys[-1]
                if isinstance(target.get(final_key), bool):
                    target[final_key] = env_value.lower() in ("true", "1", "yes")
                elif isinstance(target.get(final_key), int):
                    target[final_key] = int(env_value)
                elif isinstance(target.get(final_key), float):
                    target[final_key] = float(env_value)
                else:
                    target[final_key] = env_value

                logger.info(
                    "Applied env override: %s -> %s", env_var, config_key
                )

        return TaxonomyAlignmentConfig(**config_dict)

    def export_yaml(self, output_path: Path) -> None:
        """Export configuration to YAML file.

        Args:
            output_path: Path to write YAML output
        """
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.pack.model_dump(),
                f,
                default_flow_style=False,
                sort_keys=False,
            )
        logger.info("Exported configuration to %s", output_path)

    def export_json(self, output_path: Path) -> None:
        """Export configuration to JSON file.

        Args:
            output_path: Path to write JSON output
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.pack.model_dump(), f, indent=2, default=str)
        logger.info("Exported configuration to %s", output_path)

    def get_config_hash(self) -> str:
        """Get SHA-256 hash of configuration for change detection.

        Returns:
            Hex-encoded SHA-256 hash string of the serialized configuration
        """
        config_json = json.dumps(
            self.pack.model_dump(), sort_keys=True, default=str
        )
        return hashlib.sha256(config_json.encode()).hexdigest()

    @property
    def active_agents(self) -> List[str]:
        """Get all active agents for this configuration."""
        return self.pack.get_active_agents()

    @property
    def feature_summary(self) -> Dict[str, bool]:
        """Get feature summary for this configuration."""
        return self.pack.get_feature_summary()


# =============================================================================
# Utility Functions
# =============================================================================


def get_objective_display_name(objective: Union[str, EnvironmentalObjective]) -> str:
    """Get human-readable display name for an environmental objective.

    Args:
        objective: Objective code (string or enum)

    Returns:
        Full display name string
    """
    key = objective.value if isinstance(objective, EnvironmentalObjective) else objective
    return OBJECTIVE_DISPLAY_NAMES.get(key, f"Unknown Objective ({key})")


def get_delegated_act_for_objective(
    objective: Union[str, EnvironmentalObjective],
) -> Optional[DelegatedActVersion]:
    """Determine which delegated act contains TSC for a given objective.

    Args:
        objective: Environmental objective to look up

    Returns:
        DelegatedActVersion or None if not found
    """
    key = objective.value if isinstance(objective, EnvironmentalObjective) else objective
    if key in ("CCM", "CCA"):
        return DelegatedActVersion.CLIMATE_DA_2021
    elif key in ("WTR", "CE", "PPC", "BIO"):
        return DelegatedActVersion.ENVIRONMENTAL_DA_2023
    return None


def get_nace_sector_info(nace_section: str) -> Optional[Dict[str, Any]]:
    """Get taxonomy sector information for a NACE section code.

    Args:
        nace_section: Single-letter NACE section code (A-M)

    Returns:
        Dictionary with sector name, activity count, and objectives, or None
    """
    return NACE_TAXONOMY_SECTORS.get(nace_section.upper())


def get_mandatory_tables(
    org_type: Union[str, OrganizationType],
) -> List[Dict[str, str]]:
    """Get mandatory Article 8 DA tables for an organization type.

    Args:
        org_type: Organization type (string or enum)

    Returns:
        List of mandatory table definitions
    """
    key = org_type.value if isinstance(org_type, OrganizationType) else org_type
    return [
        table for table in ARTICLE_8_MANDATORY_TABLES
        if table["applies_to"] == key
    ]


def get_eba_pillar3_templates() -> List[Dict[str, str]]:
    """Get EBA Pillar 3 ESG disclosure template definitions.

    Returns:
        List of EBA Pillar 3 template definitions (Templates 6-10)
    """
    return EBA_PILLAR3_TEMPLATES


def validate_alignment_conditions(
    sc_pass: bool,
    dnsh_pass: bool,
    ms_pass: bool,
    tsc_pass: bool,
) -> Tuple[AlignmentStatus, str]:
    """Evaluate the four alignment conditions and determine status.

    An activity is taxonomy-aligned only if all four conditions are met:
    1. Substantially contributes to at least one objective
    2. Does no significant harm to any other objective
    3. Complies with minimum safeguards
    4. Meets technical screening criteria

    Args:
        sc_pass: Substantial contribution assessment result
        dnsh_pass: DNSH assessment result
        ms_pass: Minimum safeguards verification result
        tsc_pass: Technical screening criteria compliance result

    Returns:
        Tuple of (AlignmentStatus, explanation string)
    """
    conditions = {
        "Substantial Contribution": sc_pass,
        "Do No Significant Harm": dnsh_pass,
        "Minimum Safeguards": ms_pass,
        "Technical Screening Criteria": tsc_pass,
    }

    failed = [name for name, passed in conditions.items() if not passed]

    if len(failed) == 0:
        return (
            AlignmentStatus.ALIGNED,
            "All four alignment conditions met",
        )
    elif len(failed) < 4:
        return (
            AlignmentStatus.NOT_ALIGNED,
            f"Failed conditions: {', '.join(failed)}",
        )
    else:
        return (
            AlignmentStatus.NOT_ALIGNED,
            "All four alignment conditions failed",
        )
