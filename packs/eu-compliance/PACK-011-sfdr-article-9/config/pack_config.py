"""
PACK-011 SFDR Article 9 Pack - Configuration Manager

This module implements the SFDRArticle9Config and PackConfig classes that load,
merge, and validate all configuration for the SFDR Article 9 Pack. It provides
comprehensive Pydantic v2 models for every aspect of SFDR Article 9 financial
product compliance: sustainable investment objective verification, PAI mandatory
indicator calculation, taxonomy alignment ratio computation, SFDR-specific DNSH
assessment, good governance verification, benchmark alignment tracking, impact
measurement, carbon trajectory monitoring, downgrade risk detection, portfolio
carbon footprint calculation, and EET data management.

Article 9 Product Characteristics:
    - Sustainable investment as the product objective (100% SI target)
    - Mandatory PAI consideration (all 18 indicators required)
    - Higher DNSH and governance thresholds vs Article 8
    - Benchmark alignment tracking (CTB/PAB/Custom)
    - Impact measurement and reporting
    - Carbon reduction trajectory monitoring
    - Downgrade risk monitoring (Article 9 -> Article 8 reclassification)

Article 9 Sub-Types per SFDR:
    - Article 9(1): General sustainable investment objective
    - Article 9(2): Index-based sustainable investment (tracking CTB/PAB)
    - Article 9(3): Carbon emission reduction objective

PAI Indicator Categories:
    1. Climate & Environment (GHG emissions, carbon footprint, intensity)
    2. Environment (biodiversity, water, hazardous waste)
    3. Social & Employee (UNGC violations, gender pay gap, diversity)
    4. Sovereign (GHG intensity, social violations)
    5. Real Estate (energy efficiency)

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (asset_manager / insurance / bank / pension_fund / wealth_manager)
    3. Environment overrides (SFDR9_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - SFDR Level 1: Regulation (EU) 2019/2088, Articles 9 and 2(17)
    - SFDR RTS: Delegated Regulation (EU) 2022/1288
    - Taxonomy Disclosures DA: Delegated Regulation (EU) 2021/2178
    - ESMA Q&A: SFDR and Taxonomy-related supervisory guidance
    - EET Standard: European ESG Template v1.1 (FinDatEx)
    - EU Climate Benchmarks: Regulation (EU) 2019/2089 (CTB/PAB)

Example:
    >>> config = PackConfig.from_preset("asset_manager")
    >>> print(config.pack.sfdr_classification)
    SFDRClassification.ARTICLE_9
    >>> print(config.pack.sustainable_investment.minimum_proportion_pct)
    95.0
    >>> print(config.pack.pai.pai_mandatory)
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
# Enums - SFDR-specific enumeration types
# =============================================================================


class SFDRClassification(str, Enum):
    """SFDR product classification per Regulation (EU) 2019/2088."""

    ARTICLE_6 = "ARTICLE_6"  # No sustainability characteristics promoted
    ARTICLE_8 = "ARTICLE_8"  # Promotes E/S characteristics
    ARTICLE_8_PLUS = "ARTICLE_8_PLUS"  # Article 8 with sustainable investment proportion
    ARTICLE_9 = "ARTICLE_9"  # Sustainable investment as objective


class Article9SubType(str, Enum):
    """Article 9 sub-classification per SFDR Article 9(1), 9(2), 9(3)."""

    GENERAL_9_1 = "GENERAL_9_1"  # General sustainable investment objective
    INDEX_BASED_9_2 = "INDEX_BASED_9_2"  # Index-tracking sustainable investment
    CARBON_REDUCTION_9_3 = "CARBON_REDUCTION_9_3"  # Carbon emission reduction objective


class BenchmarkType(str, Enum):
    """EU Climate Benchmark types per Regulation (EU) 2019/2089."""

    CTB = "CTB"  # EU Climate Transition Benchmark
    PAB = "PAB"  # EU Paris-Aligned Benchmark
    CUSTOM = "CUSTOM"  # Custom benchmark with sustainability methodology


class PAICategory(str, Enum):
    """Principal Adverse Impact indicator categories per SFDR RTS Annex I."""

    CLIMATE = "CLIMATE"  # Climate and other environment-related indicators
    ENVIRONMENT = "ENVIRONMENT"  # Biodiversity, water, waste indicators
    SOCIAL = "SOCIAL"  # Social and employee, human rights, anti-corruption
    SOVEREIGN = "SOVEREIGN"  # Indicators for sovereigns and supranationals
    REAL_ESTATE = "REAL_ESTATE"  # Indicators for real estate assets


class PAIDataQuality(str, Enum):
    """Data quality classification for PAI indicator inputs."""

    REPORTED = "REPORTED"  # Directly reported by investee company
    ESTIMATED = "ESTIMATED"  # Estimated using approved methodology
    MODELED = "MODELED"  # Derived from statistical or sector models
    NOT_AVAILABLE = "NOT_AVAILABLE"  # Data not available, disclosed as gap


class DisclosureType(str, Enum):
    """SFDR disclosure document types per RTS annexes."""

    PRE_CONTRACTUAL = "PRE_CONTRACTUAL"  # Annex III (Article 9)
    PERIODIC = "PERIODIC"  # Annex V (Article 9)
    WEBSITE = "WEBSITE"  # Website disclosure per Articles 10/33/36


class ESCharacteristicType(str, Enum):
    """Environmental or Social characteristic type."""

    ENVIRONMENTAL = "ENVIRONMENTAL"  # Environmental characteristic
    SOCIAL = "SOCIAL"  # Social characteristic


class GovernanceCheckStatus(str, Enum):
    """Status of Article 2(17) good governance verification."""

    PASS = "PASS"  # Investee passes good governance check
    FAIL = "FAIL"  # Investee fails good governance check
    PARTIAL = "PARTIAL"  # Some governance dimensions pass, others fail
    NOT_ASSESSED = "NOT_ASSESSED"  # Governance not yet assessed


class SustainableInvestmentType(str, Enum):
    """Classification of sustainable investments per SFDR RTS."""

    TAXONOMY_ALIGNED = "TAXONOMY_ALIGNED"  # Aligned with EU Taxonomy
    OTHER_ENVIRONMENTAL = "OTHER_ENVIRONMENTAL"  # Environmental but not taxonomy-aligned
    SOCIAL = "SOCIAL"  # Social objective contribution


class ReportingFrequency(str, Enum):
    """Reporting and disclosure frequency."""

    ANNUAL = "ANNUAL"  # Annual periodic reporting
    SEMI_ANNUAL = "SEMI_ANNUAL"  # Semi-annual reporting
    QUARTERLY = "QUARTERLY"  # Quarterly reporting


class ScreeningType(str, Enum):
    """Investment screening methodology type."""

    NEGATIVE = "NEGATIVE"  # Exclusion-based screening
    POSITIVE = "POSITIVE"  # Best-in-class / inclusion-based screening
    NORMS_BASED = "NORMS_BASED"  # International norms-based screening


class ComplianceStatus(str, Enum):
    """Overall SFDR compliance status."""

    COMPLIANT = "COMPLIANT"  # Fully compliant with all requirements
    NON_COMPLIANT = "NON_COMPLIANT"  # Not compliant, remediation needed
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"  # Some requirements met
    NOT_ASSESSED = "NOT_ASSESSED"  # Compliance not yet assessed


class CarbonMetricType(str, Enum):
    """Portfolio-level carbon metric types."""

    WACI = "WACI"  # Weighted Average Carbon Intensity
    CARBON_FOOTPRINT = "CARBON_FOOTPRINT"  # tCO2e per EUR million invested
    TOTAL_EMISSIONS = "TOTAL_EMISSIONS"  # Total portfolio carbon emissions
    FINANCED_EMISSIONS = "FINANCED_EMISSIONS"  # PCAF-aligned financed emissions


class EETVersion(str, Enum):
    """European ESG Template version."""

    V1_0 = "V1_0"  # EET v1.0
    V1_1 = "V1_1"  # EET v1.1 (current)


class DNSHMethodology(str, Enum):
    """SFDR DNSH assessment methodology approach."""

    PAI_BASED = "PAI_BASED"  # Use PAI indicators as DNSH proxy
    THRESHOLD_BASED = "THRESHOLD_BASED"  # Apply quantitative thresholds
    POLICY_BASED = "POLICY_BASED"  # Evaluate policies and procedures
    COMBINED = "COMBINED"  # Combination of all approaches


class DisclosureFormat(str, Enum):
    """Output format for disclosure documents."""

    PDF = "PDF"
    XLSX = "XLSX"
    HTML = "HTML"
    JSON = "JSON"
    XML = "XML"


class ImpactObjective(str, Enum):
    """Sustainable impact measurement objective categories."""

    CLIMATE_MITIGATION = "CLIMATE_MITIGATION"  # Climate change mitigation
    CLIMATE_ADAPTATION = "CLIMATE_ADAPTATION"  # Climate change adaptation
    WATER = "WATER"  # Sustainable use of water and marine resources
    CIRCULAR_ECONOMY = "CIRCULAR_ECONOMY"  # Transition to circular economy
    POLLUTION = "POLLUTION"  # Pollution prevention and control
    BIODIVERSITY = "BIODIVERSITY"  # Protection of biodiversity and ecosystems
    SOCIAL_INCLUSION = "SOCIAL_INCLUSION"  # Social inclusion and equality
    HEALTH = "HEALTH"  # Health and well-being


class DowngradeRiskLevel(str, Enum):
    """Risk level for Article 9 to Article 8 downgrade."""

    LOW = "LOW"  # Low risk of downgrade
    MEDIUM = "MEDIUM"  # Medium risk, monitoring required
    HIGH = "HIGH"  # High risk, remediation required
    CRITICAL = "CRITICAL"  # Critical, immediate action required


# =============================================================================
# Reference Data Constants - SFDR regulatory reference data
# =============================================================================

# All 18 mandatory PAI indicators per SFDR RTS Annex I Table 1
MANDATORY_PAI_INDICATORS: List[Dict[str, Any]] = [
    # Climate and other environment-related indicators (14 mandatory)
    {"id": 1, "category": "CLIMATE", "name": "GHG Emissions",
     "metric": "Scope 1, 2, 3 and total GHG emissions", "unit": "tCO2e"},
    {"id": 2, "category": "CLIMATE", "name": "Carbon Footprint",
     "metric": "Carbon footprint", "unit": "tCO2e/EUR million invested"},
    {"id": 3, "category": "CLIMATE", "name": "GHG Intensity of Investee Companies",
     "metric": "GHG intensity of investee companies", "unit": "tCO2e/EUR million revenue"},
    {"id": 4, "category": "CLIMATE", "name": "Exposure to Fossil Fuel Sector",
     "metric": "Share of investments in companies active in the fossil fuel sector", "unit": "%"},
    {"id": 5, "category": "CLIMATE", "name": "Non-Renewable Energy Share",
     "metric": "Share of non-renewable energy consumption and production", "unit": "%"},
    {"id": 6, "category": "CLIMATE", "name": "Energy Consumption Intensity",
     "metric": "Energy consumption intensity per high impact climate sector", "unit": "GWh/EUR million revenue"},
    {"id": 7, "category": "ENVIRONMENT", "name": "Biodiversity-Sensitive Areas",
     "metric": "Activities negatively affecting biodiversity-sensitive areas", "unit": "%"},
    {"id": 8, "category": "ENVIRONMENT", "name": "Emissions to Water",
     "metric": "Emissions to water", "unit": "tonnes"},
    {"id": 9, "category": "ENVIRONMENT", "name": "Hazardous Waste Ratio",
     "metric": "Hazardous waste and radioactive waste ratio", "unit": "tonnes"},
    # Social and employee matters (5 mandatory)
    {"id": 10, "category": "SOCIAL", "name": "UNGC/OECD Violations",
     "metric": "Violations of UN Global Compact and OECD Guidelines", "unit": "%"},
    {"id": 11, "category": "SOCIAL", "name": "UNGC/OECD Compliance Processes",
     "metric": "Lack of processes and compliance mechanisms to monitor UNGC/OECD", "unit": "%"},
    {"id": 12, "category": "SOCIAL", "name": "Unadjusted Gender Pay Gap",
     "metric": "Average unadjusted gender pay gap", "unit": "%"},
    {"id": 13, "category": "SOCIAL", "name": "Board Gender Diversity",
     "metric": "Average ratio of female to male board members", "unit": "%"},
    {"id": 14, "category": "SOCIAL", "name": "Controversial Weapons Exposure",
     "metric": "Exposure to controversial weapons", "unit": "%"},
    # Sovereign indicators (2 mandatory)
    {"id": 15, "category": "SOVEREIGN", "name": "Sovereign GHG Intensity",
     "metric": "GHG intensity of investee countries", "unit": "tCO2e/EUR million GDP"},
    {"id": 16, "category": "SOVEREIGN", "name": "Investee Countries Social Violations",
     "metric": "Countries subject to social violations", "unit": "count"},
    # Real estate indicators (2 mandatory)
    {"id": 17, "category": "REAL_ESTATE", "name": "Fossil Fuel Exposure Real Estate",
     "metric": "Exposure to fossil fuels through real estate assets", "unit": "%"},
    {"id": 18, "category": "REAL_ESTATE", "name": "Energy Inefficient Real Estate",
     "metric": "Exposure to energy-inefficient real estate assets", "unit": "%"},
]

# PAI indicator IDs by category
PAI_CLIMATE_INDICATORS: List[int] = [1, 2, 3, 4, 5, 6]
PAI_ENVIRONMENT_INDICATORS: List[int] = [7, 8, 9]
PAI_SOCIAL_INDICATORS: List[int] = [10, 11, 12, 13, 14]
PAI_SOVEREIGN_INDICATORS: List[int] = [15, 16]
PAI_REAL_ESTATE_INDICATORS: List[int] = [17, 18]
ALL_MANDATORY_PAI_IDS: List[int] = list(range(1, 19))

# Good governance dimensions per Article 2(17)
GOOD_GOVERNANCE_DIMENSIONS: Dict[str, Dict[str, str]] = {
    "sound_management": {
        "name": "Sound Management Structures",
        "description": "Board composition, independence, oversight effectiveness",
        "article": "Article 2(17)",
    },
    "employee_relations": {
        "name": "Employee Relations",
        "description": "Labour rights, health and safety, collective bargaining",
        "article": "Article 2(17)",
    },
    "remuneration": {
        "name": "Remuneration of Staff",
        "description": "Fair remuneration policies, pay equity, incentive alignment",
        "article": "Article 2(17)",
    },
    "tax_compliance": {
        "name": "Tax Compliance",
        "description": "Tax transparency, aggressive tax planning avoidance",
        "article": "Article 2(17)",
    },
}

# SFDR Classification display names
CLASSIFICATION_DISPLAY_NAMES: Dict[str, str] = {
    "ARTICLE_6": "Article 6 - No sustainability characteristics",
    "ARTICLE_8": "Article 8 - Promotes E/S characteristics",
    "ARTICLE_8_PLUS": "Article 8+ - Article 8 with sustainable investments",
    "ARTICLE_9": "Article 9 - Sustainable investment objective",
}

# Article 9 sub-type display names
ARTICLE9_SUBTYPE_DISPLAY_NAMES: Dict[str, str] = {
    "GENERAL_9_1": "Article 9(1) - General sustainable investment objective",
    "INDEX_BASED_9_2": "Article 9(2) - Index-based sustainable investment",
    "CARBON_REDUCTION_9_3": "Article 9(3) - Carbon emission reduction objective",
}

# Disclosure type to Annex mapping
DISCLOSURE_ANNEX_MAPPING: Dict[str, Dict[str, str]] = {
    "PRE_CONTRACTUAL": {
        "ARTICLE_8": "Annex II",
        "ARTICLE_8_PLUS": "Annex II",
        "ARTICLE_9": "Annex III",
    },
    "PERIODIC": {
        "ARTICLE_8": "Annex IV",
        "ARTICLE_8_PLUS": "Annex IV",
        "ARTICLE_9": "Annex V",
    },
}

# Available presets
AVAILABLE_PRESETS: Dict[str, str] = {
    "asset_manager": "UCITS/AIF fund managers with Article 9 sustainable investment products",
    "insurance": "Insurance undertakings with Article 9 IBIPs and unit-linked funds",
    "bank": "Credit institutions with Article 9 green bonds and impact products",
    "pension_fund": "Occupational pension schemes with Article 9 sustainable options",
    "wealth_manager": "Discretionary portfolio managers with Article 9 impact mandates",
}


# =============================================================================
# Pydantic Sub-Config Models
# =============================================================================


class PAIConfig(BaseModel):
    """Configuration for Principal Adverse Impact indicator calculation.

    For Article 9 products, PAI consideration is mandatory per SFDR Article 7.
    All 18 mandatory indicators must be calculated and disclosed. Additional
    opt-in indicators are strongly recommended for Article 9 products.
    """

    enabled: bool = Field(
        True,
        description="Enable PAI indicator calculation (always True for Article 9)",
    )
    pai_mandatory: bool = Field(
        True,
        description="PAI consideration is mandatory for Article 9 products",
    )
    enabled_mandatory_indicators: List[int] = Field(
        default_factory=lambda: list(range(1, 19)),
        description="List of mandatory PAI indicator IDs to calculate (1-18, all required for Art.9)",
    )
    additional_climate_indicators: List[int] = Field(
        default_factory=lambda: [19, 20],
        description="Additional opt-in climate PAI indicator IDs (recommended for Art.9)",
    )
    additional_social_indicators: List[int] = Field(
        default_factory=lambda: [21, 22],
        description="Additional opt-in social PAI indicator IDs (recommended for Art.9)",
    )
    min_coverage_pct: float = Field(
        70.0,
        ge=0.0,
        le=100.0,
        description="Minimum data coverage percentage for PAI reporting (higher for Art.9)",
    )
    coverage_adjustment: bool = Field(
        True,
        description="Apply coverage adjustment to PAI values for partial data",
    )
    estimation_enabled: bool = Field(
        True,
        description="Allow estimation of PAI values when reported data unavailable",
    )
    estimation_methodology: str = Field(
        "SECTOR_AVERAGE",
        description="Estimation methodology: SECTOR_AVERAGE, PEER_PROXY, STATISTICAL_MODEL",
    )
    include_scope_3: bool = Field(
        True,
        description="Include Scope 3 GHG emissions in PAI indicators 1-3 (recommended for Art.9)",
    )
    entity_level_statement: bool = Field(
        True,
        description="Generate entity-level PAI statement (Article 4)",
    )
    product_level_pai: bool = Field(
        True,
        description="Calculate product-level PAI indicators",
    )
    prior_period_comparison: bool = Field(
        True,
        description="Include prior period comparison in PAI statement",
    )
    data_quality_tracking: bool = Field(
        True,
        description="Track and disclose data quality per indicator (reported vs. estimated)",
    )
    engagement_actions_tracking: bool = Field(
        True,
        description="Track and report engagement actions taken in response to PAI results",
    )

    @field_validator("enabled_mandatory_indicators")
    @classmethod
    def validate_pai_ids(cls, v: List[int]) -> List[int]:
        """Validate PAI indicator IDs are within valid range."""
        invalid = [i for i in v if i < 1 or i > 18]
        if invalid:
            raise ValueError(
                f"Invalid mandatory PAI indicator IDs: {invalid}. Must be 1-18."
            )
        return sorted(set(v))

    @field_validator("min_coverage_pct")
    @classmethod
    def validate_coverage(cls, v: float) -> float:
        """Warn if coverage threshold is below regulatory expectation."""
        if v < 70.0:
            logger.warning(
                "PAI min_coverage_pct below 70%% may not meet Article 9 expectations. "
                "Article 9 products require higher data quality standards."
            )
        return v


    @model_validator(mode="after")
    def validate_article9_pai_mandatory(self) -> "PAIConfig":
        """Ensure all 18 PAI indicators are enabled for Article 9."""
        if self.pai_mandatory:
            if len(self.enabled_mandatory_indicators) < 18:
                logger.warning(
                    "Article 9 products must consider all 18 mandatory PAI indicators. "
                    "Currently only %d are enabled.",
                    len(self.enabled_mandatory_indicators),
                )
        return self


class TaxonomyAlignmentConfig(BaseModel):
    """Configuration for EU Taxonomy alignment ratio calculation.

    For Article 9 products, taxonomy alignment disclosure is mandatory.
    The minimum commitment should reflect the product's sustainable
    investment objective. Article 9(2) products tracking an index must
    disclose the index's taxonomy alignment.
    """

    enabled: bool = Field(
        True,
        description="Enable taxonomy alignment ratio calculation (mandatory for Art.9)",
    )
    minimum_commitment_pct: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Minimum taxonomy alignment commitment in pre-contractual disclosure (%)",
    )
    include_sovereign_bonds: bool = Field(
        False,
        description="Include sovereign bonds in taxonomy alignment denominator",
    )
    look_through_enabled: bool = Field(
        True,
        description="Enable look-through for fund-of-funds structures",
    )
    look_through_max_levels: int = Field(
        3,
        ge=1,
        le=10,
        description="Maximum levels of look-through nesting",
    )
    enabling_activity_disclosure: bool = Field(
        True,
        description="Separately disclose enabling activity proportion",
    )
    transitional_activity_disclosure: bool = Field(
        True,
        description="Separately disclose transitional activity proportion",
    )
    alignment_methodology: str = Field(
        "REVENUE_BASED",
        description="Alignment ratio methodology: REVENUE_BASED, CAPEX_BASED, OPEX_BASED",
    )
    gas_nuclear_disclosure: bool = Field(
        True,
        description="Include separate gas and nuclear taxonomy alignment disclosure",
    )
    verification_frequency: ReportingFrequency = Field(
        ReportingFrequency.QUARTERLY,
        description="Frequency of alignment ratio verification against commitment",
    )

    @model_validator(mode="after")
    def validate_commitment(self) -> "TaxonomyAlignmentConfig":
        """Validate minimum commitment is reasonable."""
        if self.minimum_commitment_pct > 0 and not self.enabled:
            raise ValueError(
                "Cannot set minimum_commitment_pct > 0 when taxonomy alignment "
                "calculation is disabled"
            )
        return self


class DNSHConfig(BaseModel):
    """Configuration for SFDR-specific DNSH assessment.

    For Article 9 products, DNSH assessment is applied to ALL holdings (not
    just those classified as sustainable investments). The methodology must
    be robust and evidence-based, using the combined approach.
    """

    enabled: bool = Field(
        True,
        description="Enable SFDR DNSH assessment (mandatory for Art.9 on all holdings)",
    )
    methodology: DNSHMethodology = Field(
        DNSHMethodology.COMBINED,
        description="DNSH assessment methodology approach (COMBINED required for Art.9)",
    )
    pai_indicators_used: List[int] = Field(
        default_factory=lambda: list(range(1, 19)),
        description="PAI indicators used as DNSH proxy metrics (all 18 for Art.9)",
    )
    quantitative_thresholds_enabled: bool = Field(
        True,
        description="Apply quantitative thresholds for DNSH determination",
    )
    qualitative_assessment_enabled: bool = Field(
        True,
        description="Include qualitative policy-based DNSH assessment",
    )
    evidence_required: bool = Field(
        True,
        description="Require documentary evidence for DNSH pass determinations",
    )
    controversial_weapons_exclusion: bool = Field(
        True,
        description="Automatic DNSH fail for controversial weapons exposure (PAI 14)",
    )
    ungc_violations_exclusion: bool = Field(
        True,
        description="Automatic DNSH fail for UNGC/OECD violations (PAI 10)",
    )
    strict_mode: bool = Field(
        True,
        description="Enable strict DNSH mode for Article 9 (all 18 PAI indicators as DNSH proxy)",
    )
    custom_thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Custom per-indicator DNSH thresholds (indicator_id -> threshold)",
    )

    @field_validator("pai_indicators_used")
    @classmethod
    def validate_dnsh_pai_ids(cls, v: List[int]) -> List[int]:
        """Validate DNSH PAI indicator IDs are within valid range."""
        invalid = [i for i in v if i < 1 or i > 18]
        if invalid:
            raise ValueError(
                f"Invalid DNSH PAI indicator IDs: {invalid}. Must be 1-18."
            )
        return sorted(set(v))

    @model_validator(mode="after")
    def validate_article9_dnsh(self) -> "DNSHConfig":
        """Warn if DNSH configuration is not strict enough for Article 9."""
        if self.strict_mode and len(self.pai_indicators_used) < 18:
            logger.warning(
                "Article 9 strict DNSH mode requires all 18 PAI indicators. "
                "Currently only %d are configured.",
                len(self.pai_indicators_used),
            )
        return self


class GovernanceConfig(BaseModel):
    """Configuration for Article 2(17) good governance verification.

    For Article 9 products, all four governance dimensions must be assessed
    with stricter thresholds than Article 8. Good governance is a mandatory
    condition for sustainable investment classification.
    """

    enabled: bool = Field(
        True,
        description="Enable good governance verification (mandatory for Art.9)",
    )
    check_sound_management: bool = Field(
        True,
        description="Assess sound management structures",
    )
    check_employee_relations: bool = Field(
        True,
        description="Assess employee relations and labour rights",
    )
    check_remuneration: bool = Field(
        True,
        description="Assess remuneration of staff policies",
    )
    check_tax_compliance: bool = Field(
        True,
        description="Assess tax compliance and transparency",
    )
    minimum_governance_score: float = Field(
        70.0,
        ge=0.0,
        le=100.0,
        description="Minimum governance score for pass determination (stricter for Art.9)",
    )
    require_all_dimensions_pass: bool = Field(
        True,
        description="Require all four dimensions to pass for overall governance pass",
    )
    controversy_flag_threshold: int = Field(
        2,
        ge=1,
        le=10,
        description="Maximum number of active governance controversies before fail (lower for Art.9)",
    )
    monitoring_frequency: ReportingFrequency = Field(
        ReportingFrequency.QUARTERLY,
        description="Frequency of governance monitoring and reassessment",
    )
    data_sources: List[str] = Field(
        default_factory=lambda: [
            "ESG_DATA_PROVIDER",
            "ANNUAL_REPORT",
            "PROXY_STATEMENT",
            "ENGAGEMENT_RECORDS",
        ],
        description="Data sources for governance assessment (expanded for Art.9)",
    )

    @model_validator(mode="after")
    def validate_at_least_one_check(self) -> "GovernanceConfig":
        """Ensure at least one governance check is enabled."""
        if self.enabled:
            checks = [
                self.check_sound_management,
                self.check_employee_relations,
                self.check_remuneration,
                self.check_tax_compliance,
            ]
            if not any(checks):
                raise ValueError(
                    "At least one governance dimension must be enabled when "
                    "governance verification is active"
                )
        return self


class ESGCharacteristicsConfig(BaseModel):
    """Configuration for E/S characteristics tracking.

    For Article 9 products, the sustainable investment objective defines the
    product's characteristics. Unlike Article 8, Article 9 products do not
    merely promote characteristics but have sustainability as their objective.
    """

    environmental_characteristics: List[str] = Field(
        default_factory=lambda: [
            "climate_change_mitigation",
            "pollution_prevention",
            "circular_economy",
            "biodiversity_protection",
        ],
        description="Environmental objectives of the sustainable investment",
    )
    social_characteristics: List[str] = Field(
        default_factory=lambda: [
            "human_rights",
            "labour_standards",
            "community_development",
        ],
        description="Social objectives of the sustainable investment",
    )
    binding_elements: List[str] = Field(
        default_factory=lambda: [
            "sustainable_investment_minimum",
            "exclusion_criteria",
            "impact_contribution_threshold",
            "dnsh_compliance",
            "good_governance_compliance",
        ],
        description="Binding elements of the investment strategy",
    )
    sustainability_indicators: List[str] = Field(
        default_factory=lambda: [
            "carbon_intensity",
            "esg_score",
            "controversy_score",
            "green_revenue_share",
            "si_proportion",
            "impact_score",
        ],
        description="Sustainability indicators used to measure attainment",
    )
    measurement_methodology: str = Field(
        "QUANTITATIVE",
        description="Measurement approach: QUANTITATIVE, QUALITATIVE, MIXED",
    )
    benchmark_comparison: bool = Field(
        True,
        description="Compare characteristic performance against designated benchmark",
    )
    designated_benchmark_index: str = Field(
        "",
        description="Designated ESG benchmark index (e.g., EU Climate Transition Benchmark)",
    )
    track_attainment: bool = Field(
        True,
        description="Track and report characteristic attainment in periodic reports",
    )

    @model_validator(mode="after")
    def validate_characteristics(self) -> "ESGCharacteristicsConfig":
        """Validate at least one characteristic is defined."""
        total = len(self.environmental_characteristics) + len(self.social_characteristics)
        if total == 0:
            raise ValueError(
                "Article 9 products must have at least one environmental or "
                "social investment objective"
            )
        return self


class SustainableInvestmentConfig(BaseModel):
    """Configuration for sustainable investment classification and calculation.

    For Article 9 products, sustainable investment is the product OBJECTIVE.
    The minimum proportion is typically 95-100%, with the remainder only for
    hedging, liquidity, or cash management. All holdings (except cash/hedging)
    must qualify as sustainable investments under the three-pronged test.
    """

    enabled: bool = Field(
        True,
        description="Enable sustainable investment calculation (always True for Art.9)",
    )
    minimum_proportion_pct: float = Field(
        95.0,
        ge=0.0,
        le=100.0,
        description="Minimum proportion of sustainable investments (%); Article 9 typically 95-100%",
    )
    taxonomy_aligned_minimum_pct: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Minimum proportion that is taxonomy-aligned (%)",
    )
    other_environmental_minimum_pct: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Minimum proportion for other environmental objectives (%)",
    )
    social_minimum_pct: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Minimum proportion for social objectives (%)",
    )
    require_dnsh_pass: bool = Field(
        True,
        description="Require SFDR DNSH pass for sustainable investment classification",
    )
    require_governance_pass: bool = Field(
        True,
        description="Require good governance pass for sustainable investment classification",
    )
    double_counting_prevention: bool = Field(
        True,
        description="Prevent double-counting across environmental and social categories",
    )
    contribution_assessment_methodology: str = Field(
        "IMPACT_BASED",
        description="How contribution to E/S objective is assessed: THRESHOLD_BASED, REVENUE_ALIGNED, IMPACT_BASED",
    )

    cash_and_hedging_allowance_pct: float = Field(
        5.0,
        ge=0.0,
        le=20.0,
        description="Maximum allowance for cash, hedging, and liquidity instruments (%)",
    )
    continuous_monitoring: bool = Field(
        True,
        description="Continuously monitor SI proportion against minimum commitment",
    )
    breach_notification_enabled: bool = Field(
        True,
        description="Enable notification when SI proportion falls below minimum",
    )

    @model_validator(mode="after")
    def validate_proportions(self) -> "SustainableInvestmentConfig":
        """Validate sustainable investment proportion consistency for Article 9."""
        if self.enabled:
            sub_total = (
                self.taxonomy_aligned_minimum_pct
                + self.other_environmental_minimum_pct
                + self.social_minimum_pct
            )
            if sub_total > self.minimum_proportion_pct and self.minimum_proportion_pct > 0:
                raise ValueError(
                    f"Sub-category minimums ({sub_total}%) exceed overall minimum "
                    f"proportion ({self.minimum_proportion_pct}%)"
                )
            # Validate that minimum + cash allowance = 100%
            total = self.minimum_proportion_pct + self.cash_and_hedging_allowance_pct
            if total > 100.0:
                raise ValueError(
                    f"SI minimum ({self.minimum_proportion_pct}%) + cash allowance "
                    f"({self.cash_and_hedging_allowance_pct}%) exceeds 100%"
                )
        return self



class ImpactConfig(BaseModel):
    """Configuration for sustainable impact measurement and reporting.

    Article 9 products must demonstrate measurable impact towards their
    sustainable investment objective. This configuration controls impact
    KPIs, methodology, and reporting requirements.
    """

    enabled: bool = Field(
        True,
        description="Enable impact measurement and reporting",
    )
    primary_objectives: List[str] = Field(
        default_factory=lambda: [
            "CLIMATE_MITIGATION",
        ],
        description="Primary impact objectives (ImpactObjective enum values)",
    )
    secondary_objectives: List[str] = Field(
        default_factory=list,
        description="Secondary impact objectives",
    )
    impact_kpis: List[str] = Field(
        default_factory=lambda: [
            "tonnes_co2_avoided",
            "renewable_energy_generated_mwh",
            "green_revenue_share_pct",
        ],
        description="Key performance indicators for impact measurement",
    )
    impact_methodology: str = Field(
        "CONTRIBUTION_BASED",
        description="Impact methodology: CONTRIBUTION_BASED, ATTRIBUTION_BASED, COUNTERFACTUAL",
    )
    require_additionality: bool = Field(
        True,
        description="Require demonstration of additionality for impact claims",
    )
    theory_of_change_required: bool = Field(
        True,
        description="Require documented theory of change for the product",
    )
    impact_reporting_frequency: ReportingFrequency = Field(
        ReportingFrequency.ANNUAL,
        description="Frequency of impact reporting",
    )
    sdg_alignment_tracking: bool = Field(
        True,
        description="Track alignment with UN Sustainable Development Goals",
    )
    sdg_targets: List[int] = Field(
        default_factory=lambda: [7, 13],
        description="Targeted UN SDG numbers (1-17)",
    )
    third_party_verification: bool = Field(
        False,
        description="Require third-party verification of impact claims",
    )

    @field_validator("sdg_targets")
    @classmethod
    def validate_sdg_targets(cls, v: List[int]) -> List[int]:
        """Validate SDG target numbers are within valid range."""
        invalid = [i for i in v if i < 1 or i > 17]
        if invalid:
            raise ValueError(
                f"Invalid SDG target numbers: {invalid}. Must be 1-17."
            )
        return sorted(set(v))


class BenchmarkAlignmentConfig(BaseModel):
    """Configuration for EU Climate Benchmark alignment tracking.

    Article 9(2) products must track an EU Climate Transition Benchmark (CTB)
    or Paris-Aligned Benchmark (PAB). Article 9(3) products with carbon
    reduction objectives must explain how the benchmark methodology aligns
    with the carbon reduction target.
    """

    enabled: bool = Field(
        False,
        description="Enable benchmark alignment tracking",
    )
    benchmark_type: BenchmarkType = Field(
        BenchmarkType.PAB,
        description="EU Climate Benchmark type",
    )
    benchmark_name: str = Field(
        "",
        description="Designated benchmark index name",
    )
    benchmark_provider: str = Field(
        "",
        description="Benchmark index provider (e.g., MSCI, S&P, FTSE)",
    )
    tracking_error_threshold_pct: float = Field(
        2.0,
        ge=0.0,
        le=20.0,
        description="Maximum acceptable tracking error vs benchmark (%)",
    )
    explain_no_benchmark: bool = Field(
        False,
        description="Include explanation of how sustainable objective is achieved without benchmark",
    )
    pab_minimum_standards: bool = Field(
        True,
        description="Apply PAB minimum standards (50% GHG reduction vs broad market)",
    )
    ctb_minimum_standards: bool = Field(
        True,
        description="Apply CTB minimum standards (30% GHG reduction vs broad market)",
    )
    yoy_decarbonization_pct: float = Field(
        7.0,
        ge=0.0,
        le=20.0,
        description="Year-on-year self-decarbonization trajectory (% per annum, PAB=7%)",
    )
    benchmark_comparison_disclosure: bool = Field(
        True,
        description="Disclose comparison between product and benchmark performance",
    )

    @model_validator(mode="after")
    def validate_benchmark_config(self) -> "BenchmarkAlignmentConfig":
        """Validate benchmark configuration consistency."""
        if self.enabled and not self.benchmark_name and not self.explain_no_benchmark:
            logger.warning(
                "Benchmark alignment enabled but no benchmark_name specified. "
                "Either set a benchmark or enable explain_no_benchmark."
            )
        return self


class CarbonTrajectoryConfig(BaseModel):
    """Configuration for carbon reduction trajectory monitoring.

    Article 9(3) products with a carbon emission reduction objective must
    track portfolio carbon trajectory against targets. This includes
    alignment with Paris Agreement temperature goals.
    """

    enabled: bool = Field(
        False,
        description="Enable carbon trajectory monitoring (required for Art.9(3))",
    )
    target_year: int = Field(
        2030,
        ge=2025,
        le=2050,
        description="Target year for carbon reduction objective",
    )
    baseline_year: int = Field(
        2019,
        ge=2015,
        le=2025,
        description="Baseline year for carbon reduction calculation",
    )
    reduction_target_pct: float = Field(
        50.0,
        ge=0.0,
        le=100.0,
        description="Carbon reduction target vs baseline (%)",
    )
    interim_targets: Dict[str, float] = Field(
        default_factory=lambda: {
            "2025": 25.0,
            "2030": 50.0,
            "2040": 75.0,
            "2050": 90.0,
        },
        description="Interim carbon reduction targets by year (%)",
    )
    temperature_alignment: str = Field(
        "1.5C",
        description="Paris Agreement temperature alignment target: 1.5C, WELL_BELOW_2C, 2C",
    )
    trajectory_methodology: str = Field(
        "SBTi_SECTORAL",
        description="Trajectory methodology: SBTi_SECTORAL, ABSOLUTE_CONTRACTION, CONVERGENCE",
    )
    scope_coverage: List[str] = Field(
        default_factory=lambda: ["SCOPE_1", "SCOPE_2", "SCOPE_3"],
        description="Emission scopes included in trajectory: SCOPE_1, SCOPE_2, SCOPE_3",
    )
    monitoring_frequency: ReportingFrequency = Field(
        ReportingFrequency.QUARTERLY,
        description="Frequency of trajectory monitoring",
    )
    deviation_alert_threshold_pct: float = Field(
        5.0,
        ge=0.0,
        le=50.0,
        description="Alert threshold when trajectory deviates from target (%)",
    )


class DowngradeMonitorConfig(BaseModel):
    """Configuration for Article 9 to Article 8 downgrade risk monitoring.

    Following ESMA guidance, Article 9 products that cannot meet their
    sustainable investment commitment may need to be reclassified as
    Article 8. This configuration controls downgrade risk detection,
    alerting, and remediation procedures.
    """

    enabled: bool = Field(
        True,
        description="Enable downgrade risk monitoring",
    )
    si_proportion_warning_threshold_pct: float = Field(
        97.0,
        ge=50.0,
        le=100.0,
        description="SI proportion threshold triggering warning (%)",
    )
    si_proportion_critical_threshold_pct: float = Field(
        95.0,
        ge=50.0,
        le=100.0,
        description="SI proportion threshold triggering critical alert (%)",
    )
    dnsh_failure_tolerance_pct: float = Field(
        3.0,
        ge=0.0,
        le=10.0,
        description="Maximum DNSH failure rate before downgrade risk flagged (%)",
    )
    governance_failure_tolerance_pct: float = Field(
        3.0,
        ge=0.0,
        le=10.0,
        description="Maximum governance failure rate before downgrade risk flagged (%)",
    )
    monitoring_frequency: ReportingFrequency = Field(
        ReportingFrequency.QUARTERLY,
        description="Frequency of downgrade risk assessment",
    )
    alert_recipients: List[str] = Field(
        default_factory=list,
        description="Email recipients for downgrade risk alerts",
    )
    remediation_period_days: int = Field(
        30,
        ge=7,
        le=90,
        description="Days allowed for remediation before escalation",
    )
    auto_reclassification_enabled: bool = Field(
        False,
        description="Enable automatic reclassification to Article 8 on sustained breach",
    )
    escalation_chain: List[str] = Field(
        default_factory=lambda: [
            "portfolio_manager",
            "compliance_officer",
            "board",
        ],
        description="Escalation chain for downgrade risk events",
    )

    @model_validator(mode="after")
    def validate_thresholds(self) -> "DowngradeMonitorConfig":
        """Validate warning threshold is above critical threshold."""
        if self.si_proportion_warning_threshold_pct < self.si_proportion_critical_threshold_pct:
            raise ValueError(
                f"Warning threshold ({self.si_proportion_warning_threshold_pct}%) must be "
                f">= critical threshold ({self.si_proportion_critical_threshold_pct}%)"
            )
        return self


class CarbonFootprintConfig(BaseModel):
    """Configuration for portfolio carbon footprint calculation.

    Article 9 products require comprehensive carbon footprint disclosure
    including Scope 3, with higher coverage thresholds than Article 8.
    """

    enabled: bool = Field(
        True,
        description="Enable portfolio carbon footprint calculation",
    )
    calculate_waci: bool = Field(
        True,
        description="Calculate Weighted Average Carbon Intensity",
    )
    calculate_carbon_footprint: bool = Field(
        True,
        description="Calculate carbon footprint (tCO2e/EUR million invested)",
    )
    calculate_total_emissions: bool = Field(
        True,
        description="Calculate total portfolio carbon emissions",
    )
    calculate_financed_emissions: bool = Field(
        True,
        description="Calculate PCAF-aligned financed emissions (recommended for Art.9)",
    )
    scope_coverage: List[str] = Field(
        default_factory=lambda: ["SCOPE_1", "SCOPE_2", "SCOPE_3"],
        description="Emission scopes included (Scope 3 recommended for Art.9)",
    )
    pcaf_alignment: bool = Field(
        True,
        description="Align calculations with PCAF (recommended for Art.9)",
    )
    coverage_threshold_pct: float = Field(
        70.0,
        ge=0.0,
        le=100.0,
        description="Minimum data coverage for carbon metric reporting (higher for Art.9)",
    )
    benchmark_comparison: bool = Field(
        True,
        description="Compare portfolio carbon metrics against benchmark",
    )
    benchmark_index: str = Field(
        "",
        description="Benchmark index for carbon comparison (e.g., MSCI World)",
    )
    yoy_trend: bool = Field(
        True,
        description="Include year-over-year trend analysis",
    )
    attribution_analysis: bool = Field(
        True,
        description="Include carbon attribution analysis (allocation vs. selection effect)",
    )
    currency: str = Field(
        "EUR",
        description="Currency for financial normalization (ISO 4217)",
    )

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Validate currency code format."""
        if len(v) != 3 or not v.isalpha() or not v.isupper():
            raise ValueError(f"Currency must be 3-letter uppercase ISO 4217 code: {v}")
        return v

    @field_validator("scope_coverage")
    @classmethod
    def validate_scopes(cls, v: List[str]) -> List[str]:
        """Validate scope values."""
        valid_scopes = {"SCOPE_1", "SCOPE_2", "SCOPE_3"}
        invalid = [s for s in v if s not in valid_scopes]
        if invalid:
            raise ValueError(f"Invalid scope values: {invalid}. Must be {valid_scopes}")
        return v


class EETConfig(BaseModel):
    """Configuration for European ESG Template management.

    Controls EET data import/export, field mapping, and version management
    per the FinDatEx EET standard. Article 9 products require additional
    EET fields for sustainable investment proportion and impact data.
    """

    enabled: bool = Field(
        True,
        description="Enable EET data management",
    )
    eet_version: EETVersion = Field(
        EETVersion.V1_1,
        description="EET standard version",
    )
    import_enabled: bool = Field(
        True,
        description="Enable EET file import (manufacturer data)",
    )
    export_enabled: bool = Field(
        True,
        description="Enable EET file export (for distributors)",
    )
    export_format: str = Field(
        "XLSX",
        description="EET export file format: XLSX, CSV, XML",
    )
    validate_completeness: bool = Field(
        True,
        description="Validate EET field completeness on import",
    )
    validate_consistency: bool = Field(
        True,
        description="Validate EET data consistency with internal SFDR data",
    )
    auto_populate: bool = Field(
        True,
        description="Auto-populate EET fields from internal calculations",
    )
    sfdr_classification_field: bool = Field(
        True,
        description="Include SFDR product classification in EET (must be Article 9)",
    )
    taxonomy_alignment_fields: bool = Field(
        True,
        description="Include taxonomy alignment proportion fields in EET",
    )
    pai_consideration_fields: bool = Field(
        True,
        description="Include PAI consideration flags in EET",
    )
    sustainable_investment_fields: bool = Field(
        True,
        description="Include sustainable investment proportion fields in EET",
    )


class DisclosureConfig(BaseModel):
    """Configuration for SFDR disclosure generation.

    Article 9 products use Annex III (pre-contractual) and Annex V (periodic).
    These annexes have additional sections compared to Article 8 annexes,
    including sustainable investment objective description and impact reporting.
    """

    annex_iii_enabled: bool = Field(
        True,
        description="Enable Annex III pre-contractual disclosure (Article 9)",
    )
    annex_v_enabled: bool = Field(
        True,
        description="Enable Annex V periodic report (Article 9)",
    )
    website_disclosure_enabled: bool = Field(
        True,
        description="Enable website disclosure per Article 10",
    )
    default_format: DisclosureFormat = Field(
        DisclosureFormat.PDF,
        description="Default disclosure output format",
    )
    include_methodology_note: bool = Field(
        True,
        description="Include methodology and assumptions note in disclosures",
    )
    include_data_sources: bool = Field(
        True,
        description="Include data sources and limitations section",
    )
    include_asset_allocation_chart: bool = Field(
        True,
        description="Include visual asset allocation breakdown chart",
    )
    include_impact_section: bool = Field(
        True,
        description="Include impact measurement section in disclosures",
    )
    include_benchmark_comparison: bool = Field(
        True,
        description="Include benchmark comparison section (Art.9(2) requirement)",
    )
    multi_language_support: bool = Field(
        False,
        description="Generate disclosures in multiple languages",
    )
    languages: List[str] = Field(
        default_factory=lambda: ["en"],
        description="Languages for disclosure generation (ISO 639-1)",
    )
    review_workflow_enabled: bool = Field(
        True,
        description="Enable review and approval workflow before publication",
    )
    version_tracking: bool = Field(
        True,
        description="Track disclosure versions with change history",
    )
    greenwashing_check: bool = Field(
        True,
        description="Run automated greenwashing risk checks on disclosure content",
    )
    greenwashing_strict_mode: bool = Field(
        True,
        description="Enable strict greenwashing checks for Article 9 claims",
    )


class ScreeningConfig(BaseModel):
    """Configuration for portfolio screening (negative/positive/norms-based).

    Article 9 products require stricter screening than Article 8, with
    comprehensive exclusions and positive screening for sustainable investments.
    """

    negative_screening_enabled: bool = Field(
        True,
        description="Enable negative (exclusion) screening",
    )
    positive_screening_enabled: bool = Field(
        True,
        description="Enable positive (best-in-class) screening (expected for Art.9)",
    )
    norms_based_screening_enabled: bool = Field(
        True,
        description="Enable norms-based screening (UNGC, OECD, ILO)",
    )
    negative_exclusions: List[str] = Field(
        default_factory=lambda: [
            "controversial_weapons",
            "tobacco_production",
            "thermal_coal_extraction",
            "arctic_drilling",
            "tar_sands",
            "nuclear_weapons",
            "cluster_munitions",
            "anti_personnel_mines",
        ],
        description="Activities/sectors excluded via negative screening (broader for Art.9)",
    )
    revenue_threshold_pct: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Revenue threshold for activity-based exclusions (0% = zero tolerance for Art.9)",
    )
    norms_frameworks: List[str] = Field(
        default_factory=lambda: [
            "UN_GLOBAL_COMPACT",
            "OECD_GUIDELINES",
            "ILO_CONVENTIONS",
            "UNIVERSAL_DECLARATION_HUMAN_RIGHTS",
        ],
        description="International norms frameworks used for screening (expanded for Art.9)",
    )
    best_in_class_percentile: float = Field(
        75.0,
        ge=50.0,
        le=100.0,
        description="Percentile threshold for best-in-class selection (%)",
    )
    screening_frequency: ReportingFrequency = Field(
        ReportingFrequency.QUARTERLY,
        description="Frequency of screening re-evaluation",
    )
    breach_handling: str = Field(
        "IMMEDIATE_DIVEST",
        description="Action on screening breach: IMMEDIATE_DIVEST, DIVEST_90_DAYS, ENGAGE_FIRST",
    )
    monitoring_enabled: bool = Field(
        True,
        description="Enable continuous screening monitoring between formal re-evaluations",
    )


class ReportingConfig(BaseModel):
    """Configuration for SFDR reporting schedule and format.

    Controls reporting frequency, distribution, and formatting across
    all SFDR disclosure types. Article 9 products typically have more
    frequent and detailed reporting requirements.
    """

    reporting_frequency: ReportingFrequency = Field(
        ReportingFrequency.ANNUAL,
        description="Primary reporting frequency",
    )
    reporting_period_end: str = Field(
        "12-31",
        description="Reporting period end date (MM-DD format)",
    )
    reporting_deadline_days: int = Field(
        120,
        ge=30,
        le=365,
        description="Days after period end for reporting deadline",
    )
    distribution_channels: List[str] = Field(
        default_factory=lambda: ["email", "website", "in_app"],
        description="Channels for report distribution",
    )
    default_format: DisclosureFormat = Field(
        DisclosureFormat.PDF,
        description="Default report output format",
    )
    language: str = Field(
        "en",
        description="Report language code (ISO 639-1)",
    )
    timezone: str = Field(
        "UTC",
        description="Timezone for timestamps in reports",
    )
    include_board_summary: bool = Field(
        True,
        description="Generate board-level summary alongside detailed reports",
    )
    include_impact_report: bool = Field(
        True,
        description="Generate standalone impact report (Article 9 specific)",
    )
    recipients: List[str] = Field(
        default_factory=list,
        description="Default report recipient email addresses",
    )


class DataQualityConfig(BaseModel):
    """Configuration for ESG data quality management.

    Article 9 products require higher data quality standards than Article 8.
    Stricter coverage, estimation limits, and freshness requirements apply.
    """

    min_quality_score: float = Field(
        0.80,
        ge=0.0,
        le=1.0,
        description="Minimum data quality score for compliance acceptance (higher for Art.9)",
    )
    min_coverage_pct: float = Field(
        70.0,
        ge=0.0,
        le=100.0,
        description="Minimum data coverage across portfolio holdings (higher for Art.9)",
    )
    estimation_limit_pct: float = Field(
        30.0,
        ge=0.0,
        le=100.0,
        description="Maximum proportion of estimated data allowed (lower for Art.9)",
    )
    data_age_max_months: int = Field(
        12,
        ge=1,
        le=36,
        description="Maximum age of ESG data before flagging as stale (shorter for Art.9)",
    )
    require_audited_emissions: bool = Field(
        True,
        description="Require audited emissions data for PAI indicators 1-3 (stronger for Art.9)",
    )
    validation_on_ingestion: bool = Field(
        True,
        description="Validate data quality on ingestion",
    )
    flag_low_quality: bool = Field(
        True,
        description="Flag data points below quality threshold in outputs",
    )
    allow_sector_proxies: bool = Field(
        True,
        description="Allow sector-average proxy data when company data unavailable",
    )
    proxy_usage_cap_pct: float = Field(
        20.0,
        ge=0.0,
        le=100.0,
        description="Maximum proportion of portfolio using proxy data (Art.9 stricter)",
    )


class AuditTrailConfig(BaseModel):
    """Configuration for audit trail and provenance tracking."""

    retention_years: int = Field(
        7,
        ge=1,
        le=20,
        description="Audit trail retention period (years, longer for Art.9)",
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
        description="Use synthetic portfolio and ESG data",
    )
    mock_esg_provider: bool = Field(
        False,
        description="Mock ESG data provider API responses",
    )
    mock_portfolio_data: bool = Field(
        False,
        description="Mock portfolio management system data",
    )
    mock_mrv_data: bool = Field(
        False,
        description="Mock MRV agent emissions data",
    )
    mock_impact_data: bool = Field(
        False,
        description="Mock impact measurement data",
    )
    tutorial_mode_enabled: bool = Field(
        False,
        description="Enable guided tutorial mode for onboarding",
    )
    sample_holdings_count: int = Field(
        50,
        ge=10,
        le=500,
        description="Number of sample portfolio holdings in demo mode",
    )


# =============================================================================
# Main Configuration Class
# =============================================================================


class SFDRArticle9Config(BaseModel):
    """SFDR Article 9 Pack main configuration.

    Central configuration class that combines all sub-configurations for
    the complete SFDR Article 9 compliance pipeline. Article 9 products have
    sustainable investment as their OBJECTIVE, requiring 95-100% sustainable
    investment allocation, mandatory PAI on all 18 indicators, stricter DNSH
    and governance thresholds, impact measurement, benchmark alignment, carbon
    trajectory monitoring, and downgrade risk detection.

    Key Differences from Article 8 (PACK-010):
        - sfdr_classification defaults to ARTICLE_9
        - article9_sub_type specifies 9(1), 9(2), or 9(3)
        - sustainable_investment.minimum_proportion_pct defaults to 95%
        - pai.pai_mandatory = True (all 18 indicators required)
        - pai.include_scope_3 = True by default
        - dnsh.strict_mode = True (all holdings assessed)
        - governance.minimum_governance_score = 70.0 (vs 60.0 for Art.8)
        - New: impact configuration for impact measurement
        - New: benchmark_alignment for CTB/PAB tracking
        - New: carbon_trajectory for carbon reduction monitoring
        - New: downgrade_monitor for Article 9->8 reclassification risk
        - Disclosure uses Annex III / Annex V (vs Annex II / IV for Art.8)
        - Screening: broader exclusions, zero-tolerance revenue threshold

    Attributes:
        pack_id: Unique pack identifier
        version: Pack version string
        tier: Pack tier classification
        sfdr_classification: SFDR product classification (ARTICLE_9)
        article9_sub_type: Article 9 sub-classification (9(1), 9(2), 9(3))
        product_name: Financial product name
        product_isin: Financial product ISIN
        product_lei: Legal Entity Identifier
        reporting_frequency: Primary reporting frequency
        reporting_year: Active reporting fiscal year
        pai: PAI indicator calculation configuration
        taxonomy_alignment: Taxonomy alignment ratio configuration
        dnsh: SFDR DNSH assessment configuration
        governance: Good governance verification configuration
        esg_characteristics: E/S objectives tracking configuration
        sustainable_investment: Sustainable investment calculation configuration
        impact: Impact measurement configuration (Art.9 specific)
        benchmark_alignment: Benchmark alignment configuration (Art.9 specific)
        carbon_trajectory: Carbon trajectory monitoring (Art.9 specific)
        downgrade_monitor: Downgrade risk monitoring (Art.9 specific)
        carbon_footprint: Portfolio carbon footprint configuration
        eet: EET data management configuration
        disclosure: Disclosure generation configuration
        screening: Portfolio screening configuration
        reporting: Reporting schedule and format configuration
        data_quality: Data quality management configuration
        audit_trail: Audit trail and provenance configuration
        demo: Demo mode configuration

    Example:
        >>> config = SFDRArticle9Config(
        ...     sfdr_classification=SFDRClassification.ARTICLE_8,
        ...     product_name="Green Growth Fund",
        ... )
        >>> assert config.pai.enabled is True
        >>> assert len(config.pai.enabled_mandatory_indicators) == 18
    """

    # Pack metadata
    pack_id: str = Field(
        "PACK-011-sfdr-article-9",
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

    # Product context
    sfdr_classification: SFDRClassification = Field(
        SFDRClassification.ARTICLE_9,
        description="SFDR product classification (Article 9)",
    )
    article9_sub_type: Article9SubType = Field(
        Article9SubType.GENERAL_9_1,
        description="Article 9 sub-classification: 9(1) general, 9(2) index-based, 9(3) carbon reduction",
    )
    product_name: str = Field(
        "",
        description="Financial product name",
    )
    product_isin: str = Field(
        "",
        description="Financial product ISIN identifier",
    )
    product_lei: str = Field(
        "",
        description="Legal Entity Identifier of the product manufacturer",
    )
    reporting_frequency: ReportingFrequency = Field(
        ReportingFrequency.ANNUAL,
        description="Primary reporting frequency",
    )
    reporting_year: int = Field(
        2025,
        ge=2023,
        le=2030,
        description="Active reporting fiscal year",
    )

    # Sub-configurations
    pai: PAIConfig = Field(default_factory=PAIConfig)
    taxonomy_alignment: TaxonomyAlignmentConfig = Field(
        default_factory=TaxonomyAlignmentConfig,
    )
    dnsh: DNSHConfig = Field(default_factory=DNSHConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
    esg_characteristics: ESGCharacteristicsConfig = Field(
        default_factory=ESGCharacteristicsConfig,
    )
    sustainable_investment: SustainableInvestmentConfig = Field(
        default_factory=SustainableInvestmentConfig,
    )
    impact: ImpactConfig = Field(
        default_factory=ImpactConfig,
    )
    benchmark_alignment: BenchmarkAlignmentConfig = Field(
        default_factory=BenchmarkAlignmentConfig,
    )
    carbon_trajectory: CarbonTrajectoryConfig = Field(
        default_factory=CarbonTrajectoryConfig,
    )
    downgrade_monitor: DowngradeMonitorConfig = Field(
        default_factory=DowngradeMonitorConfig,
    )
    carbon_footprint: CarbonFootprintConfig = Field(
        default_factory=CarbonFootprintConfig,
    )
    eet: EETConfig = Field(default_factory=EETConfig)
    disclosure: DisclosureConfig = Field(default_factory=DisclosureConfig)
    screening: ScreeningConfig = Field(default_factory=ScreeningConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    data_quality: DataQualityConfig = Field(default_factory=DataQualityConfig)
    audit_trail: AuditTrailConfig = Field(default_factory=AuditTrailConfig)
    demo: DemoConfig = Field(default_factory=DemoConfig)

    @model_validator(mode="after")
    def validate_article9_config(self) -> "SFDRArticle9Config":
        """Validate configuration consistency for Article 9 products."""
        classification = self.sfdr_classification

        # Article 9 requires sustainable investment enabled
        if classification == SFDRClassification.ARTICLE_9:
            if not self.sustainable_investment.enabled:
                raise ValueError(
                    "Article 9 classification requires sustainable_investment.enabled=True. "
                    "Sustainable investment is the product's objective."
                )
            if self.sustainable_investment.minimum_proportion_pct < 80.0:
                logger.warning(
                    "Article 9 sustainable investment minimum proportion is %.1f%%, "
                    "which is below typical Article 9 threshold (95-100%%). "
                    "Consider increasing to avoid downgrade risk.",
                    self.sustainable_investment.minimum_proportion_pct,
                )

        # Article 9 must have PAI enabled and mandatory
        if classification == SFDRClassification.ARTICLE_9:
            if not self.pai.enabled:
                raise ValueError(
                    "Article 9 classification requires PAI calculation to be enabled. "
                    "PAI consideration is mandatory for Article 9 products."
                )

        # Article 9(2) requires benchmark alignment
        if self.article9_sub_type == Article9SubType.INDEX_BASED_9_2:
            if not self.benchmark_alignment.enabled:
                logger.warning(
                    "Article 9(2) products should have benchmark_alignment.enabled=True. "
                    "Index-based Article 9 products must track a CTB or PAB."
                )

        # Article 9(3) requires carbon trajectory
        if self.article9_sub_type == Article9SubType.CARBON_REDUCTION_9_3:
            if not self.carbon_trajectory.enabled:
                logger.warning(
                    "Article 9(3) products should have carbon_trajectory.enabled=True. "
                    "Carbon reduction objective requires trajectory monitoring."
                )

        # DNSH and governance must be enabled for Article 9
        if classification == SFDRClassification.ARTICLE_9:
            if not self.dnsh.enabled:
                raise ValueError(
                    "Article 9 classification requires DNSH assessment to be enabled."
                )
            if not self.governance.enabled:
                raise ValueError(
                    "Article 9 classification requires governance verification to be enabled."
                )

        return self

    def get_active_agents(self) -> List[str]:
        """Get list of all active agents for this pack.

        Returns:
            List of agent identifiers (30 MRV + 10 data + 10 foundation)
        """
        agents: List[str] = []

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

    def get_disclosure_annex(self) -> str:
        """Get the applicable SFDR RTS annex for this product classification.

        Article 9 products use Annex III (pre-contractual) and Annex V (periodic).

        Returns:
            Annex identifier string (e.g., 'Annex III' for Article 9)
        """
        classification_key = self.sfdr_classification.value
        mapping = DISCLOSURE_ANNEX_MAPPING.get("PRE_CONTRACTUAL", {})
        return mapping.get(classification_key, "Unknown")

    def get_enabled_pai_categories(self) -> List[PAICategory]:
        """Get PAI categories with at least one enabled indicator.

        Returns:
            List of PAICategory enums with active indicators.
        """
        categories: Set[PAICategory] = set()
        for indicator_id in self.pai.enabled_mandatory_indicators:
            for ind in MANDATORY_PAI_INDICATORS:
                if ind["id"] == indicator_id:
                    categories.add(PAICategory(ind["category"]))
                    break
        return sorted(list(categories), key=lambda x: x.value)

    def get_article9_subtype_display(self) -> str:
        """Get display name for the Article 9 sub-type.

        Returns:
            Human-readable sub-type string.
        """
        return ARTICLE9_SUBTYPE_DISPLAY_NAMES.get(
            self.article9_sub_type.value,
            f"Unknown ({self.article9_sub_type.value})",
        )

    def get_feature_summary(self) -> Dict[str, bool]:
        """Get summary of enabled features for this configuration.

        Returns:
            Dictionary mapping feature names to enabled status.
        """
        return {
            "pai_calculation": self.pai.enabled,
            "pai_mandatory": self.pai.pai_mandatory,
            "pai_entity_level": self.pai.entity_level_statement,
            "pai_product_level": self.pai.product_level_pai,
            "pai_scope_3": self.pai.include_scope_3,
            "pai_engagement_actions": self.pai.engagement_actions_tracking,
            "taxonomy_alignment": self.taxonomy_alignment.enabled,
            "taxonomy_look_through": self.taxonomy_alignment.look_through_enabled,
            "taxonomy_gas_nuclear": self.taxonomy_alignment.gas_nuclear_disclosure,
            "sfdr_dnsh": self.dnsh.enabled,
            "sfdr_dnsh_strict": self.dnsh.strict_mode,
            "good_governance": self.governance.enabled,
            "sustainable_investment": self.sustainable_investment.enabled,
            "sustainable_investment_monitoring": self.sustainable_investment.continuous_monitoring,
            "impact_measurement": self.impact.enabled,
            "impact_sdg_tracking": self.impact.sdg_alignment_tracking,
            "benchmark_alignment": self.benchmark_alignment.enabled,
            "carbon_trajectory": self.carbon_trajectory.enabled,
            "downgrade_monitoring": self.downgrade_monitor.enabled,
            "carbon_footprint": self.carbon_footprint.enabled,
            "waci": self.carbon_footprint.calculate_waci,
            "financed_emissions": self.carbon_footprint.calculate_financed_emissions,
            "pcaf_alignment": self.carbon_footprint.pcaf_alignment,
            "eet_management": self.eet.enabled,
            "eet_import": self.eet.import_enabled,
            "eet_export": self.eet.export_enabled,
            "annex_iii_disclosure": self.disclosure.annex_iii_enabled,
            "annex_v_disclosure": self.disclosure.annex_v_enabled,
            "website_disclosure": self.disclosure.website_disclosure_enabled,
            "negative_screening": self.screening.negative_screening_enabled,
            "positive_screening": self.screening.positive_screening_enabled,
            "norms_based_screening": self.screening.norms_based_screening_enabled,
            "greenwashing_check": self.disclosure.greenwashing_check,
            "greenwashing_strict": self.disclosure.greenwashing_strict_mode,
            "audit_trail": self.audit_trail.include_provenance_hash,
        }

    def get_classification_display(self) -> str:
        """Get display name for the current SFDR classification.

        Returns:
            Human-readable classification string.
        """
        return CLASSIFICATION_DISPLAY_NAMES.get(
            self.sfdr_classification.value,
            f"Unknown ({self.sfdr_classification.value})",
        )

    def get_downgrade_risk_level(
        self,
        current_si_pct: float,
        dnsh_failure_pct: float,
        governance_failure_pct: float,
    ) -> DowngradeRiskLevel:
        """Assess the current downgrade risk level.

        Args:
            current_si_pct: Current sustainable investment proportion (%).
            dnsh_failure_pct: Current DNSH failure rate (%).
            governance_failure_pct: Current governance failure rate (%).

        Returns:
            DowngradeRiskLevel enum value.
        """
        monitor = self.downgrade_monitor

        if current_si_pct < monitor.si_proportion_critical_threshold_pct:
            return DowngradeRiskLevel.CRITICAL
        if current_si_pct < monitor.si_proportion_warning_threshold_pct:
            return DowngradeRiskLevel.HIGH
        if dnsh_failure_pct > monitor.dnsh_failure_tolerance_pct:
            return DowngradeRiskLevel.HIGH
        if governance_failure_pct > monitor.governance_failure_tolerance_pct:
            return DowngradeRiskLevel.MEDIUM
        return DowngradeRiskLevel.LOW


# =============================================================================
# PackConfig - Top-Level Configuration Loader
# =============================================================================


class PackConfig(BaseModel):
    """Top-level pack configuration loader with YAML and preset support.

    Loads configuration from preset files, applies environment overrides,
    and provides methods for export, hashing, and introspection.

    Configuration Merge Order:
        1. Base defaults from SFDRArticle9Config
        2. Preset YAML (if specified)
        3. Environment variable overrides (SFDR9_PACK_*)
        4. Explicit runtime overrides

    Example:
        >>> config = PackConfig.from_preset("asset_manager")
        >>> print(config.pack.sfdr_classification)
        SFDRClassification.ARTICLE_9
        >>> print(config.get_config_hash()[:16])
        'a1b2c3d4e5f6g7h8'
    """

    pack: SFDRArticle9Config = Field(
        default_factory=SFDRArticle9Config,
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
    def from_yaml(
        cls,
        yaml_path: Union[str, Path],
    ) -> "PackConfig":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            PackConfig instance with loaded configuration.

        Raises:
            FileNotFoundError: If YAML file does not exist.
            ValueError: If configuration validation fails.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        config = SFDRArticle9Config(**data)
        config = cls._apply_env_overrides(config)

        loaded_files = [str(yaml_path)]
        logger.info("Loaded configuration from YAML: %s", yaml_path)

        return cls(pack=config, loaded_from=loaded_files)

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        demo_mode: bool = False,
    ) -> "PackConfig":
        """Load configuration from a named preset.

        Args:
            preset_name: Preset name (asset_manager, insurance, bank,
                pension_fund, wealth_manager).
            demo_mode: Enable demo mode with synthetic data.

        Returns:
            PackConfig instance with preset configuration applied.

        Raises:
            FileNotFoundError: If preset file does not exist.
            ValueError: If preset name is not recognized or validation fails.
        """
        if preset_name not in AVAILABLE_PRESETS:
            raise ValueError(
                f"Unknown preset: '{preset_name}'. "
                f"Available presets: {list(AVAILABLE_PRESETS.keys())}"
            )

        preset_path = CONFIG_DIR / "presets" / f"{preset_name}.yaml"
        loaded_files: List[str] = []

        config = SFDRArticle9Config()

        # Load pack manifest reference
        manifest_path = PACK_BASE_DIR / "pack.yaml"
        if manifest_path.exists():
            loaded_files.append(str(manifest_path))
            logger.info("Located pack manifest: %s", manifest_path)

        # Load preset YAML
        if preset_path.exists():
            with open(preset_path, "r", encoding="utf-8") as f:
                preset_data = yaml.safe_load(f)
            if preset_data:
                config = cls._merge_config(config, preset_data)
                loaded_files.append(str(preset_path))
                logger.info("Loaded preset: %s", preset_name)
        else:
            logger.warning("Preset file not found: %s", preset_path)

        # Apply environment variable overrides
        config = cls._apply_env_overrides(config)

        # Enable demo mode if requested
        if demo_mode:
            config.demo.demo_mode_enabled = True
            config.demo.use_synthetic_data = True
            config.demo.mock_esg_provider = True
            config.demo.mock_portfolio_data = True
            config.demo.mock_mrv_data = True
            config.demo.mock_impact_data = True
            logger.info("Demo mode enabled with synthetic data")

        return cls(pack=config, loaded_from=loaded_files)

    @classmethod
    def load(
        cls,
        preset: Optional[str] = None,
        demo_mode: bool = False,
    ) -> "PackConfig":
        """Load configuration with optional preset overlay.

        Args:
            preset: Optional preset name to apply.
            demo_mode: Enable demo mode with synthetic data.

        Returns:
            Loaded PackConfig instance with merged configuration.
        """
        if preset:
            return cls.from_preset(preset, demo_mode=demo_mode)

        config = SFDRArticle9Config()
        loaded_files: List[str] = []

        manifest_path = PACK_BASE_DIR / "pack.yaml"
        if manifest_path.exists():
            loaded_files.append(str(manifest_path))

        config = cls._apply_env_overrides(config)

        if demo_mode:
            config.demo.demo_mode_enabled = True
            config.demo.use_synthetic_data = True
            config.demo.mock_esg_provider = True
            config.demo.mock_portfolio_data = True
            config.demo.mock_mrv_data = True
            config.demo.mock_impact_data = True

        return cls(pack=config, loaded_from=loaded_files)

    @classmethod
    def available_presets(cls) -> Dict[str, str]:
        """Get dictionary of available presets and their descriptions.

        Returns:
            Dictionary mapping preset names to descriptions.
        """
        return dict(AVAILABLE_PRESETS)

    @staticmethod
    def _merge_config(
        base: SFDRArticle9Config,
        overlay: Dict[str, Any],
    ) -> SFDRArticle9Config:
        """Deep merge overlay configuration into base configuration.

        Args:
            base: Base SFDRArticle9Config instance.
            overlay: Dictionary of override values to merge.

        Returns:
            New SFDRArticle9Config with merged values.
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
        return SFDRArticle9Config(**merged)

    @staticmethod
    def _apply_env_overrides(
        config: SFDRArticle9Config,
    ) -> SFDRArticle9Config:
        """Apply environment variable overrides to configuration.

        Looks for SFDR9_PACK_* environment variables and applies them
        as configuration overrides.

        Args:
            config: Current configuration instance.

        Returns:
            Configuration with environment overrides applied.
        """
        env_mapping: Dict[str, str] = {
            "SFDR9_PACK_CLASSIFICATION": "sfdr_classification",
            "SFDR9_PACK_PRODUCT_NAME": "product_name",
            "SFDR9_PACK_PRODUCT_ISIN": "product_isin",
            "SFDR9_PACK_REPORTING_YEAR": "reporting_year",
            "SFDR9_PACK_DEMO_MODE": "demo.demo_mode_enabled",
            "SFDR9_PACK_PAI_COVERAGE": "pai.min_coverage_pct",
            "SFDR9_PACK_TAXONOMY_COMMITMENT": "taxonomy_alignment.minimum_commitment_pct",
            "SFDR9_PACK_SI_MINIMUM": "sustainable_investment.minimum_proportion_pct",
            "SFDR9_PACK_SUB_TYPE": "article9_sub_type",
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
                current_value = target.get(final_key)
                if isinstance(current_value, bool):
                    target[final_key] = env_value.lower() in ("true", "1", "yes")
                elif isinstance(current_value, int):
                    target[final_key] = int(env_value)
                elif isinstance(current_value, float):
                    target[final_key] = float(env_value)
                else:
                    target[final_key] = env_value

                logger.info(
                    "Applied env override: %s -> %s", env_var, config_key
                )

        return SFDRArticle9Config(**config_dict)

    def export_yaml(self, output_path: Union[str, Path]) -> None:
        """Export configuration to YAML file.

        Args:
            output_path: Path to write YAML output.
        """
        output_path = Path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.pack.model_dump(),
                f,
                default_flow_style=False,
                sort_keys=False,
            )
        logger.info("Exported configuration to %s", output_path)

    def export_json(self, output_path: Union[str, Path]) -> None:
        """Export configuration to JSON file.

        Args:
            output_path: Path to write JSON output.
        """
        output_path = Path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.pack.model_dump(), f, indent=2, default=str)
        logger.info("Exported configuration to %s", output_path)

    def get_config_hash(self) -> str:
        """Get SHA-256 hash of configuration for change detection.

        Returns:
            Hex-encoded SHA-256 hash string of the serialized configuration.
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

    @property
    def classification_display(self) -> str:
        """Get display name for SFDR classification."""
        return self.pack.get_classification_display()

    @property
    def disclosure_annex(self) -> str:
        """Get applicable RTS annex for this product."""
        return self.pack.get_disclosure_annex()

    @property
    def article9_subtype_display(self) -> str:
        """Get display name for Article 9 sub-type."""
        return self.pack.get_article9_subtype_display()


# =============================================================================
# Utility Functions
# =============================================================================


def get_pai_indicator_info(indicator_id: int) -> Optional[Dict[str, Any]]:
    """Get information about a mandatory PAI indicator.

    Args:
        indicator_id: PAI indicator ID (1-18).

    Returns:
        Dictionary with indicator details, or None if not found.
    """
    for indicator in MANDATORY_PAI_INDICATORS:
        if indicator["id"] == indicator_id:
            return dict(indicator)
    return None


def get_pai_indicators_by_category(
    category: Union[str, PAICategory],
) -> List[Dict[str, Any]]:
    """Get all PAI indicators for a given category.

    Args:
        category: PAI category (string or enum).

    Returns:
        List of PAI indicator dictionaries for the category.
    """
    key = category.value if isinstance(category, PAICategory) else category
    return [
        dict(ind) for ind in MANDATORY_PAI_INDICATORS
        if ind["category"] == key
    ]


def get_article9_subtype_display_name(
    sub_type: Union[str, Article9SubType],
) -> str:
    """Get human-readable display name for an Article 9 sub-type.

    Args:
        sub_type: Article 9 sub-type (string or enum).

    Returns:
        Full display name string.
    """
    key = (
        sub_type.value
        if isinstance(sub_type, Article9SubType)
        else sub_type
    )
    return ARTICLE9_SUBTYPE_DISPLAY_NAMES.get(key, f"Unknown ({key})")


def get_classification_display_name(
    classification: Union[str, SFDRClassification],
) -> str:
    """Get human-readable display name for an SFDR classification.

    Args:
        classification: SFDR classification (string or enum).

    Returns:
        Full display name string.
    """
    key = (
        classification.value
        if isinstance(classification, SFDRClassification)
        else classification
    )
    return CLASSIFICATION_DISPLAY_NAMES.get(key, f"Unknown ({key})")


def get_disclosure_annex_for_classification(
    classification: Union[str, SFDRClassification],
    disclosure_type: Union[str, DisclosureType],
) -> str:
    """Get the applicable RTS annex for a classification and disclosure type.

    Args:
        classification: SFDR product classification.
        disclosure_type: Type of disclosure (pre-contractual, periodic, website).

    Returns:
        Annex identifier string.
    """
    cls_key = (
        classification.value
        if isinstance(classification, SFDRClassification)
        else classification
    )
    disc_key = (
        disclosure_type.value
        if isinstance(disclosure_type, DisclosureType)
        else disclosure_type
    )
    mapping = DISCLOSURE_ANNEX_MAPPING.get(disc_key, {})
    return mapping.get(cls_key, "N/A")


def get_governance_dimension_info(dimension: str) -> Optional[Dict[str, str]]:
    """Get information about a good governance dimension.

    Args:
        dimension: Governance dimension key.

    Returns:
        Dictionary with dimension details, or None if not found.
    """
    return GOOD_GOVERNANCE_DIMENSIONS.get(dimension)


def validate_sustainable_investment(
    contributes_to_objective: bool,
    dnsh_pass: bool,
    governance_pass: bool,
) -> Tuple[bool, str]:
    """Evaluate the three-pronged sustainable investment test.

    A sustainable investment must:
    1. Contribute to an environmental or social objective
    2. Not significantly harm any other E/S objective (SFDR DNSH)
    3. Investee follows good governance practices

    For Article 9 products, ALL holdings (except cash/hedging) must pass
    this three-pronged test.

    Args:
        contributes_to_objective: Contribution assessment result.
        dnsh_pass: SFDR DNSH assessment result.
        governance_pass: Good governance verification result.

    Returns:
        Tuple of (is_sustainable, explanation string).
    """
    conditions = {
        "Contributes to E/S objective": contributes_to_objective,
        "Does not significantly harm": dnsh_pass,
        "Good governance": governance_pass,
    }

    failed = [name for name, passed in conditions.items() if not passed]

    if len(failed) == 0:
        return (
            True,
            "All three sustainable investment conditions met",
        )
    else:
        return (
            False,
            f"Failed conditions: {', '.join(failed)}",
        )


def assess_downgrade_risk(
    si_proportion_pct: float,
    dnsh_failure_pct: float,
    governance_failure_pct: float,
    config: Optional[DowngradeMonitorConfig] = None,
) -> Tuple[DowngradeRiskLevel, str]:
    """Assess Article 9 to Article 8 downgrade risk.

    Args:
        si_proportion_pct: Current sustainable investment proportion (%).
        dnsh_failure_pct: Current DNSH failure rate across holdings (%).
        governance_failure_pct: Current governance failure rate (%).
        config: Optional downgrade monitor config (uses defaults if None).

    Returns:
        Tuple of (DowngradeRiskLevel, explanation string).
    """
    if config is None:
        config = DowngradeMonitorConfig()

    reasons: List[str] = []

    if si_proportion_pct < config.si_proportion_critical_threshold_pct:
        reasons.append(
            f"SI proportion ({si_proportion_pct:.1f}%) below critical threshold "
            f"({config.si_proportion_critical_threshold_pct:.1f}%)"
        )
        return (DowngradeRiskLevel.CRITICAL, "; ".join(reasons))

    if si_proportion_pct < config.si_proportion_warning_threshold_pct:
        reasons.append(
            f"SI proportion ({si_proportion_pct:.1f}%) below warning threshold "
            f"({config.si_proportion_warning_threshold_pct:.1f}%)"
        )

    if dnsh_failure_pct > config.dnsh_failure_tolerance_pct:
        reasons.append(
            f"DNSH failure rate ({dnsh_failure_pct:.1f}%) exceeds tolerance "
            f"({config.dnsh_failure_tolerance_pct:.1f}%)"
        )

    if governance_failure_pct > config.governance_failure_tolerance_pct:
        reasons.append(
            f"Governance failure rate ({governance_failure_pct:.1f}%) exceeds tolerance "
            f"({config.governance_failure_tolerance_pct:.1f}%)"
        )

    if not reasons:
        return (DowngradeRiskLevel.LOW, "All metrics within acceptable thresholds")

    if len(reasons) >= 2:
        return (DowngradeRiskLevel.HIGH, "; ".join(reasons))

    return (DowngradeRiskLevel.MEDIUM, "; ".join(reasons))


def get_default_config() -> PackConfig:
    """Get default SFDR Article 9 configuration.

    Returns:
        PackConfig with default SFDRArticle9Config.

    Example:
        >>> config = get_default_config()
        >>> assert config.pack.sfdr_classification == SFDRClassification.ARTICLE_9
        >>> assert config.pack.pai.pai_mandatory is True
        >>> assert config.pack.sustainable_investment.minimum_proportion_pct == 95.0
    """
    return PackConfig(pack=SFDRArticle9Config())
