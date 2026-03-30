"""
PACK-009 EU Climate Compliance Bundle Pack - Configuration Manager

This module implements the BundleComplianceConfig and PackConfig classes that load,
merge, and validate all configuration for the EU Climate Compliance Bundle Pack.
It provides comprehensive Pydantic v2 models for cross-regulation compliance
management spanning CSRD, CBAM, EUDR, and EU Taxonomy.

The bundle pack does not extend any single regulation pack. Instead, it composes
four standalone packs and adds cross-regulation engines, workflows, templates,
and integrations for unified compliance management.

Constituent Packs:
    - PACK-001: CSRD Starter Pack (Corporate Sustainability Reporting)
    - PACK-004: CBAM Readiness Pack (Carbon Border Adjustment Mechanism)
    - PACK-006: EUDR Starter Pack (EU Deforestation Regulation)
    - PACK-008: EU Taxonomy Alignment Pack (EU Taxonomy Classification)

Bundle Presets:
    - enterprise_full: All 4 regulations, all engines, maximum capability
    - financial_institution: Taxonomy GAR focus + CSRD mandatory + portfolio CBAM/EUDR
    - eu_importer: CBAM + EUDR primary, CSRD + Taxonomy secondary
    - sme_essential: CSRD + Taxonomy basics only

Configuration Merge Order (later overrides earlier):
    1. Base BundleComplianceConfig defaults
    2. Bundle preset YAML (enterprise_full / financial_institution / etc.)
    3. Environment overrides (BUNDLE_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - CSRD: Directive (EU) 2022/2464
    - CBAM: Regulation (EU) 2023/956
    - EUDR: Regulation (EU) 2023/1115
    - EU Taxonomy: Regulation (EU) 2020/852

Example:
    >>> config = PackConfig.from_preset("enterprise_full")
    >>> print(config.pack.enabled_regulations)
    [CSRD, CBAM, EUDR, TAXONOMY]
    >>> print(config.pack.scoring.composite_method)
    'WEIGHTED_AVERAGE'
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
from greenlang.schemas.enums import NotificationChannel

logger = logging.getLogger(__name__)

# Base directory for all pack configuration files
PACK_BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent


# =============================================================================
# Enums - Cross-regulation enumeration types
# =============================================================================


class RegulationType(str, Enum):
    """EU regulation types covered by this bundle pack."""

    CSRD = "CSRD"  # Corporate Sustainability Reporting Directive (EU) 2022/2464
    CBAM = "CBAM"  # Carbon Border Adjustment Mechanism (EU) 2023/956
    EUDR = "EUDR"  # EU Deforestation Regulation (EU) 2023/1115
    TAXONOMY = "TAXONOMY"  # EU Taxonomy Regulation (EU) 2020/852


class ComplianceStatus(str, Enum):
    """Compliance status for a regulation or requirement."""

    NOT_STARTED = "NOT_STARTED"  # No compliance activities initiated
    IN_PROGRESS = "IN_PROGRESS"  # Compliance work underway
    COMPLIANT = "COMPLIANT"  # Fully compliant with all requirements
    NON_COMPLIANT = "NON_COMPLIANT"  # Not compliant, remediation needed
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"  # Some requirements met, gaps remain


class BundleTier(str, Enum):
    """Bundle deployment preset tiers."""

    ENTERPRISE_FULL = "ENTERPRISE_FULL"  # All regulations, maximum capability
    FINANCIAL_INSTITUTION = "FINANCIAL_INSTITUTION"  # GAR/BTAR focus + CSRD
    EU_IMPORTER = "EU_IMPORTER"  # CBAM + EUDR primary
    SME_ESSENTIAL = "SME_ESSENTIAL"  # CSRD + Taxonomy basics


class DataFieldCategory(str, Enum):
    """Categories for cross-regulation shared data fields."""

    GHG_EMISSIONS = "GHG_EMISSIONS"  # Scope 1/2/3 emissions data
    SUPPLY_CHAIN = "SUPPLY_CHAIN"  # Supply chain traceability
    ACTIVITY_CLASSIFICATION = "ACTIVITY_CLASSIFICATION"  # Economic activity codes
    FINANCIAL_DATA = "FINANCIAL_DATA"  # Turnover, CapEx, OpEx
    CLIMATE_RISK = "CLIMATE_RISK"  # Climate risk and vulnerability
    WATER_POLLUTION = "WATER_POLLUTION"  # Water and pollution metrics
    BIODIVERSITY = "BIODIVERSITY"  # Biodiversity and land use
    GOVERNANCE = "GOVERNANCE"  # Governance and due diligence
    SOCIAL = "SOCIAL"  # Social and human rights


class ConsistencyLevel(str, Enum):
    """Data consistency classification across regulation packs."""

    EXACT = "EXACT"  # Values match exactly across packs
    APPROXIMATE = "APPROXIMATE"  # Values within tolerance threshold
    CONFLICTING = "CONFLICTING"  # Values contradict across packs
    MISSING = "MISSING"  # Data present in one pack but absent in another


class GapSeverity(str, Enum):
    """Severity classification for cross-regulation compliance gaps."""

    CRITICAL = "CRITICAL"  # Must be resolved before next filing deadline
    HIGH = "HIGH"  # Should be resolved within current reporting period
    MEDIUM = "MEDIUM"  # Should be resolved within 6 months
    LOW = "LOW"  # Should be resolved within 12 months
    INFO = "INFO"  # Informational, no action required


class CalendarEventType(str, Enum):
    """Types of events in the unified regulatory calendar."""

    FILING_DEADLINE = "FILING_DEADLINE"  # Regulatory filing due date
    DATA_COLLECTION = "DATA_COLLECTION"  # Data collection window
    REVIEW_MILESTONE = "REVIEW_MILESTONE"  # Internal review checkpoint
    AUDIT_DATE = "AUDIT_DATE"  # External or internal audit date
    BOARD_REPORT = "BOARD_REPORT"  # Board reporting date


class EvidenceType(str, Enum):
    """Types of evidence in the cross-regulation evidence vault."""

    DOCUMENT = "DOCUMENT"  # PDF, Word, or other document
    CERTIFICATE = "CERTIFICATE"  # Third-party certification or attestation
    MEASUREMENT = "MEASUREMENT"  # Metered or measured data point
    CALCULATION = "CALCULATION"  # Derived calculation with provenance
    ATTESTATION = "ATTESTATION"  # Signed declaration or attestation
    REPORT = "REPORT"  # Audit report or verification statement


class ScoringMethod(str, Enum):
    """Methods for computing composite bundle compliance scores."""

    WEIGHTED_AVERAGE = "WEIGHTED_AVERAGE"  # Weighted by regulation priority
    MINIMUM_SCORE = "MINIMUM_SCORE"  # Overall score = lowest regulation score
    GEOMETRIC_MEAN = "GEOMETRIC_MEAN"  # Geometric mean of all regulation scores
    HARMONIC_MEAN = "HARMONIC_MEAN"  # Harmonic mean (penalizes low outliers)


class ReconciliationAction(str, Enum):
    """Actions for data consistency reconciliation."""

    AUTO_CORRECT = "AUTO_CORRECT"  # Automatically apply correction
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"  # Flag for manual review
    IGNORE = "IGNORE"  # Accept inconsistency
    ESCALATE = "ESCALATE"  # Escalate to data steward


# =============================================================================
# Reference Data Constants
# =============================================================================

# All four regulations
ALL_REGULATIONS: List[RegulationType] = list(RegulationType)

# Regulation display names
REGULATION_DISPLAY_NAMES: Dict[str, str] = {
    "CSRD": "Corporate Sustainability Reporting Directive",
    "CBAM": "Carbon Border Adjustment Mechanism",
    "EUDR": "EU Deforestation Regulation",
    "TAXONOMY": "EU Taxonomy Regulation",
}

# Regulation references
REGULATION_REFERENCES: Dict[str, str] = {
    "CSRD": "Directive (EU) 2022/2464",
    "CBAM": "Regulation (EU) 2023/956",
    "EUDR": "Regulation (EU) 2023/1115",
    "TAXONOMY": "Regulation (EU) 2020/852",
}

# Regulation effective dates
REGULATION_EFFECTIVE_DATES: Dict[str, str] = {
    "CSRD": "2024-01-01",
    "CBAM": "2023-10-01",
    "EUDR": "2024-12-30",
    "TAXONOMY": "2022-01-01",
}

# Cross-regulation data overlap matrix
CROSS_REGULATION_OVERLAPS: Dict[str, Dict[str, List[str]]] = {
    "CSRD-TAXONOMY": {
        "regulations": ["CSRD", "TAXONOMY"],
        "shared_fields": [
            "ghg_emissions_scope_1_2_3",
            "taxonomy_eligible_turnover",
            "taxonomy_aligned_capex",
            "environmental_objectives",
        ],
    },
    "CSRD-CBAM": {
        "regulations": ["CSRD", "CBAM"],
        "shared_fields": [
            "scope_1_emissions",
            "scope_3_upstream_emissions",
            "emission_factors",
            "energy_consumption",
        ],
    },
    "CSRD-EUDR": {
        "regulations": ["CSRD", "EUDR"],
        "shared_fields": [
            "supply_chain_mapping",
            "biodiversity_impact",
            "land_use_change",
            "deforestation_risk",
        ],
    },
    "TAXONOMY-EUDR": {
        "regulations": ["TAXONOMY", "EUDR"],
        "shared_fields": [
            "biodiversity_assessment",
            "land_use_classification",
            "forestry_activities",
            "sustainable_sourcing",
        ],
    },
    "TAXONOMY-CBAM": {
        "regulations": ["TAXONOMY", "CBAM"],
        "shared_fields": [
            "ghg_emissions_intensity",
            "energy_efficiency_metrics",
            "industrial_process_data",
            "emission_reduction_targets",
        ],
    },
    "CBAM-EUDR": {
        "regulations": ["CBAM", "EUDR"],
        "shared_fields": [
            "country_of_origin",
            "supplier_identification",
            "import_declarations",
        ],
    },
}

# Default regulation weights for composite scoring
DEFAULT_SCORING_WEIGHTS: Dict[str, float] = {
    "CSRD": 0.30,
    "CBAM": 0.25,
    "EUDR": 0.20,
    "TAXONOMY": 0.25,
}

# Available presets
AVAILABLE_PRESETS: Dict[str, str] = {
    "enterprise_full": "All 4 regulations enabled, maximum capability",
    "financial_institution": "Taxonomy GAR focus + CSRD mandatory",
    "eu_importer": "CBAM + EUDR primary, CSRD + Taxonomy secondary",
    "sme_essential": "CSRD + Taxonomy basics only",
}


# =============================================================================
# Pydantic Sub-Config Models
# =============================================================================


class RegulationConfig(BaseModel):
    """Configuration for a single regulation within the bundle.

    Each constituent regulation can be independently enabled/disabled with
    its own priority level and pack reference.
    """

    enabled: bool = Field(
        True,
        description="Whether this regulation is enabled in the bundle",
    )
    priority: int = Field(
        1,
        ge=1,
        le=10,
        description="Priority level (1=highest, 10=lowest) for scoring and scheduling",
    )
    pack_id: str = Field(
        ...,
        description="Pack identifier for this regulation (e.g., PACK-001-csrd-starter)",
    )
    display_name: str = Field(
        "",
        description="Human-readable name for this regulation",
    )
    regulation_reference: str = Field(
        "",
        description="Official regulation reference (e.g., Directive (EU) 2022/2464)",
    )
    scoring_weight: float = Field(
        0.25,
        ge=0.0,
        le=1.0,
        description="Weight in composite compliance score (0.0-1.0)",
    )
    data_quality_threshold: float = Field(
        0.80,
        ge=0.0,
        le=1.0,
        description="Minimum data quality score for this regulation",
    )
    reporting_frequency: str = Field(
        "ANNUAL",
        description="Reporting frequency: ANNUAL, QUARTERLY, PER_SHIPMENT",
    )

    @field_validator("pack_id")
    @classmethod
    def validate_pack_id(cls, v: str) -> str:
        """Validate pack ID follows naming convention."""
        if not v.startswith("PACK-"):
            raise ValueError(f"Pack ID must start with 'PACK-': {v}")
        return v


class CalendarConfig(BaseModel):
    """Configuration for the unified regulatory calendar engine.

    Controls deadline tracking, notification dispatch, and calendar
    synchronization across all four regulation packs.
    """

    enabled: bool = Field(
        True,
        description="Enable unified regulatory calendar",
    )
    lead_time_days: int = Field(
        30,
        ge=1,
        le=365,
        description="Default lead time in days before deadline for notifications",
    )
    critical_lead_time_days: int = Field(
        7,
        ge=1,
        le=90,
        description="Lead time for critical/urgent deadline notifications",
    )
    notification_channels: List[NotificationChannel] = Field(
        default_factory=lambda: [NotificationChannel.EMAIL, NotificationChannel.IN_APP],
        description="Active notification channels for calendar alerts",
    )
    conflict_detection: bool = Field(
        True,
        description="Detect overlapping or conflicting deadlines across regulations",
    )
    auto_schedule_reviews: bool = Field(
        True,
        description="Automatically schedule internal review milestones before deadlines",
    )
    review_lead_time_days: int = Field(
        14,
        ge=1,
        le=180,
        description="Days before filing deadline to schedule internal review",
    )
    include_csrd_deadlines: bool = Field(
        True,
        description="Include CSRD annual reporting deadlines",
    )
    include_cbam_deadlines: bool = Field(
        True,
        description="Include CBAM quarterly and annual deadlines",
    )
    include_eudr_deadlines: bool = Field(
        True,
        description="Include EUDR due diligence statement deadlines",
    )
    include_taxonomy_deadlines: bool = Field(
        True,
        description="Include EU Taxonomy annual disclosure deadlines",
    )
    fiscal_year_end_month: int = Field(
        12,
        ge=1,
        le=12,
        description="Fiscal year end month (1-12) for calculating reporting deadlines",
    )
    timezone: str = Field(
        "Europe/Brussels",
        description="Timezone for calendar events (EU-centric default)",
    )

    @field_validator("notification_channels")
    @classmethod
    def validate_channels(cls, v: List[NotificationChannel]) -> List[NotificationChannel]:
        """Validate at least one notification channel is configured."""
        if len(v) == 0:
            raise ValueError("At least one notification channel must be configured")
        return v


class DeduplicationConfig(BaseModel):
    """Configuration for the data deduplication engine.

    Controls how duplicate data entries are detected and eliminated across
    regulation-specific data stores.
    """

    enabled: bool = Field(
        True,
        description="Enable cross-regulation data deduplication",
    )
    fuzzy_match_threshold: float = Field(
        0.90,
        ge=0.50,
        le=1.0,
        description="Minimum similarity score for fuzzy match detection (0.5-1.0)",
    )
    hash_comparison: bool = Field(
        True,
        description="Use SHA-256 hash comparison for exact duplicate detection",
    )
    semantic_similarity: bool = Field(
        True,
        description="Use embedding-based semantic similarity for near-duplicate detection",
    )
    auto_merge: bool = Field(
        False,
        description="Automatically merge detected duplicates (false = flag for review)",
    )
    track_savings: bool = Field(
        True,
        description="Track and report deduplication savings metrics",
    )
    dedup_categories: List[DataFieldCategory] = Field(
        default_factory=lambda: list(DataFieldCategory),
        description="Data field categories to include in deduplication scanning",
    )
    batch_size: int = Field(
        500,
        ge=50,
        le=10000,
        description="Batch size for deduplication scanning operations",
    )
    preserve_provenance: bool = Field(
        True,
        description="Preserve provenance links to all original source records after merge",
    )

    @field_validator("fuzzy_match_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Warn if threshold is too low for production use."""
        if v < 0.75:
            logger.warning(
                "Deduplication fuzzy_match_threshold below 0.75 may produce false positives"
            )
        return v


class ConsistencyConfig(BaseModel):
    """Configuration for the multi-regulation consistency checking engine.

    Controls how data inconsistencies are detected and classified across
    regulation-specific submissions.
    """

    enabled: bool = Field(
        True,
        description="Enable cross-regulation consistency checking",
    )
    numeric_tolerance_pct: float = Field(
        5.0,
        ge=0.0,
        le=25.0,
        description="Percentage tolerance for numeric field comparisons (0=exact match)",
    )
    date_tolerance_days: int = Field(
        0,
        ge=0,
        le=30,
        description="Day tolerance for date field comparisons (0=exact match)",
    )
    auto_reconcile: bool = Field(
        False,
        description="Automatically reconcile detectable inconsistencies",
    )
    reconciliation_action: ReconciliationAction = Field(
        ReconciliationAction.FLAG_FOR_REVIEW,
        description="Default action for detected inconsistencies",
    )
    check_ghg_emissions: bool = Field(
        True,
        description="Check GHG emissions consistency between CSRD, CBAM, and Taxonomy",
    )
    check_financial_data: bool = Field(
        True,
        description="Check financial data consistency (turnover, CapEx, OpEx) across packs",
    )
    check_supply_chain: bool = Field(
        True,
        description="Check supply chain data consistency between EUDR and CSRD ESRS E4",
    )
    check_taxonomy_csrd: bool = Field(
        True,
        description="Check taxonomy KPIs match CSRD Article 8 disclosures",
    )
    check_cbam_taxonomy: bool = Field(
        True,
        description="Check CBAM emissions intensity aligns with taxonomy CCM data",
    )
    check_eudr_taxonomy: bool = Field(
        True,
        description="Check EUDR biodiversity data aligns with taxonomy BIO objective",
    )

    @model_validator(mode="after")
    def validate_at_least_one_check(self) -> "ConsistencyConfig":
        """Ensure at least one consistency check is enabled when engine is active."""
        if self.enabled:
            checks = [
                self.check_ghg_emissions,
                self.check_financial_data,
                self.check_supply_chain,
                self.check_taxonomy_csrd,
                self.check_cbam_taxonomy,
                self.check_eudr_taxonomy,
            ]
            if not any(checks):
                raise ValueError(
                    "At least one consistency check must be enabled when "
                    "consistency engine is active"
                )
        return self


class GapAnalysisConfig(BaseModel):
    """Configuration for the cross-regulation gap analysis engine.

    Controls how compliance gaps are identified, classified, and prioritized
    across all four EU regulations.
    """

    enabled: bool = Field(
        True,
        description="Enable cross-regulation gap analysis",
    )
    severity_levels: List[GapSeverity] = Field(
        default_factory=lambda: list(GapSeverity),
        description="Gap severity levels to include in analysis",
    )
    include_remediation_plans: bool = Field(
        True,
        description="Generate remediation plans for identified gaps",
    )
    include_effort_estimates: bool = Field(
        True,
        description="Include effort estimates (hours/days) for gap remediation",
    )
    include_cost_estimates: bool = Field(
        False,
        description="Include cost estimates for gap remediation",
    )
    cross_regulation_leverage: bool = Field(
        True,
        description="Identify where compliance in one regulation closes gaps in another",
    )
    prioritize_shared_gaps: bool = Field(
        True,
        description="Prioritize gaps that affect multiple regulations simultaneously",
    )
    gap_refresh_frequency: str = Field(
        "QUARTERLY",
        description="How often to refresh gap analysis: WEEKLY, MONTHLY, QUARTERLY",
    )
    max_gaps_per_report: int = Field(
        100,
        ge=10,
        le=1000,
        description="Maximum number of gaps to include in a single report",
    )
    include_dependency_chains: bool = Field(
        True,
        description="Show dependency chains between gaps that must be resolved in order",
    )


class EvidenceConfig(BaseModel):
    """Configuration for the cross-regulation evidence management engine.

    Controls the shared evidence vault that links documents, certificates,
    and measurements to requirements across all four regulations.
    """

    enabled: bool = Field(
        True,
        description="Enable cross-regulation evidence management",
    )
    evidence_types: List[EvidenceType] = Field(
        default_factory=lambda: list(EvidenceType),
        description="Evidence types accepted in the shared vault",
    )
    deduplication: bool = Field(
        True,
        description="Detect and link duplicate evidence across regulation packs",
    )
    hash_algorithm: str = Field(
        "SHA-256",
        description="Hash algorithm for evidence provenance tracking",
    )
    retention_years: int = Field(
        7,
        ge=1,
        le=25,
        description="Evidence retention period in years",
    )
    lifecycle_tracking: bool = Field(
        True,
        description="Track evidence lifecycle (collected, verified, approved, expired)",
    )
    auto_expire: bool = Field(
        True,
        description="Automatically flag evidence past retention or validity period",
    )
    require_verification: bool = Field(
        True,
        description="Require evidence verification before linking to compliance claims",
    )
    max_file_size_mb: int = Field(
        50,
        ge=1,
        le=500,
        description="Maximum file size for evidence uploads (MB)",
    )
    allowed_formats: List[str] = Field(
        default_factory=lambda: [
            "pdf", "xlsx", "xls", "csv", "docx", "doc",
            "png", "jpg", "jpeg", "tiff", "geojson", "json", "xml",
        ],
        description="Allowed file formats for evidence uploads",
    )
    cross_reference_tracking: bool = Field(
        True,
        description="Track which regulations each evidence document satisfies",
    )


class ReportingConfig(BaseModel):
    """Configuration for consolidated bundle reporting.

    Controls how reports are generated, formatted, and distributed
    across all four regulation contexts.
    """

    enabled: bool = Field(
        True,
        description="Enable consolidated bundle reporting",
    )
    default_format: str = Field(
        "PDF",
        description="Default report output format (PDF, XLSX, HTML, JSON)",
    )
    generate_executive_summary: bool = Field(
        True,
        description="Generate bundle-level executive summary report",
    )
    generate_dashboard: bool = Field(
        True,
        description="Generate consolidated compliance dashboard (HTML)",
    )
    generate_gap_report: bool = Field(
        True,
        description="Generate unified gap analysis report",
    )
    generate_calendar_report: bool = Field(
        True,
        description="Generate regulatory calendar report",
    )
    generate_consistency_report: bool = Field(
        True,
        description="Generate data consistency report",
    )
    generate_dedup_report: bool = Field(
        True,
        description="Generate deduplication savings report",
    )
    generate_audit_trail: bool = Field(
        True,
        description="Generate multi-regulation audit trail",
    )
    generate_data_map: bool = Field(
        True,
        description="Generate cross-regulation data map",
    )
    include_trend_analysis: bool = Field(
        True,
        description="Include year-over-year trend analysis in reports",
    )
    include_forecasting: bool = Field(
        False,
        description="Include compliance trajectory forecasting",
    )
    language: str = Field(
        "en",
        description="Report language code (ISO 639-1)",
    )
    timezone: str = Field(
        "UTC",
        description="Timezone for timestamps in reports",
    )
    xbrl_tagging: bool = Field(
        False,
        description="Enable XBRL/iXBRL tagging for digital filings",
    )
    distribution_channels: List[str] = Field(
        default_factory=lambda: ["email", "in_app"],
        description="Channels for report distribution",
    )

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Validate language code is reasonable."""
        supported = {"en", "de", "fr", "es", "it", "nl", "pt", "pl", "sv", "da", "fi"}
        if v not in supported:
            logger.warning(
                "Language '%s' is not in the standard supported set. "
                "Supported: %s",
                v,
                ", ".join(sorted(supported)),
            )
        return v


class ScoringConfig(BaseModel):
    """Configuration for the bundle compliance scoring engine.

    Controls how composite compliance scores are calculated across
    all four regulations.
    """

    enabled: bool = Field(
        True,
        description="Enable bundle compliance scoring",
    )
    composite_method: ScoringMethod = Field(
        ScoringMethod.WEIGHTED_AVERAGE,
        description="Method for computing composite score from regulation scores",
    )
    regulation_weights: Dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_SCORING_WEIGHTS),
        description="Scoring weights per regulation (must sum to 1.0)",
    )
    passing_score: float = Field(
        70.0,
        ge=0.0,
        le=100.0,
        description="Minimum composite score to achieve COMPLIANT status (%)",
    )
    letter_grade_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "A": 90.0,
            "B": 80.0,
            "C": 70.0,
            "D": 60.0,
            "F": 0.0,
        },
        description="Score thresholds for letter grade assignment",
    )
    include_trend: bool = Field(
        True,
        description="Include scoring trend analysis (improvement/decline)",
    )
    score_data_quality: bool = Field(
        True,
        description="Include data quality as a scoring factor",
    )
    score_evidence_completeness: bool = Field(
        True,
        description="Include evidence completeness as a scoring factor",
    )
    score_timeliness: bool = Field(
        True,
        description="Include submission timeliness as a scoring factor",
    )

    @model_validator(mode="after")
    def validate_weights(self) -> "ScoringConfig":
        """Validate regulation weights sum to approximately 1.0."""
        if self.enabled and self.composite_method == ScoringMethod.WEIGHTED_AVERAGE:
            total = sum(self.regulation_weights.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"Regulation weights must sum to 1.0, got {total:.4f}. "
                    f"Weights: {self.regulation_weights}"
                )
        return self

    @model_validator(mode="after")
    def validate_grade_thresholds(self) -> "ScoringConfig":
        """Validate letter grade thresholds are in descending order."""
        if self.enabled:
            grades = sorted(
                self.letter_grade_thresholds.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for i in range(len(grades) - 1):
                if grades[i][1] <= grades[i + 1][1]:
                    raise ValueError(
                        f"Letter grade thresholds must be in descending order: "
                        f"{grades[i][0]}={grades[i][1]} <= {grades[i+1][0]}={grades[i+1][1]}"
                    )
        return self


class DataMapperConfig(BaseModel):
    """Configuration for the cross-framework data mapper engine.

    Controls how shared data fields are mapped between regulation-specific
    data models.
    """

    enabled: bool = Field(
        True,
        description="Enable cross-framework data mapping",
    )
    active_categories: List[DataFieldCategory] = Field(
        default_factory=lambda: list(DataFieldCategory),
        description="Data field categories active for mapping",
    )
    unit_conversion: bool = Field(
        True,
        description="Enable automatic unit conversion between regulation formats",
    )
    currency_normalization: bool = Field(
        True,
        description="Normalize currencies across regulation packs",
    )
    base_currency: str = Field(
        "EUR",
        description="Base currency for normalization (ISO 4217)",
    )
    temporal_alignment: bool = Field(
        True,
        description="Align data across different reporting periods",
    )
    field_mapping_validation: bool = Field(
        True,
        description="Validate field mappings against data dictionary on load",
    )
    track_mapping_coverage: bool = Field(
        True,
        description="Track and report data mapping coverage statistics",
    )

    @field_validator("base_currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Validate currency code format."""
        if len(v) != 3 or not v.isalpha() or not v.isupper():
            raise ValueError(f"Currency must be 3-letter uppercase ISO 4217 code: {v}")
        return v


class HealthCheckConfig(BaseModel):
    """Configuration for the bundle health check system.

    Controls monitoring of constituent pack availability, agent status,
    and integration health.
    """

    enabled: bool = Field(
        True,
        description="Enable bundle health checking",
    )
    check_interval_minutes: int = Field(
        15,
        ge=1,
        le=1440,
        description="Interval between health checks (minutes)",
    )
    check_pack_availability: bool = Field(
        True,
        description="Verify all four constituent packs are available",
    )
    check_agent_status: bool = Field(
        True,
        description="Verify all inherited agents are operational",
    )
    check_database_connectivity: bool = Field(
        True,
        description="Verify database connections for all packs",
    )
    check_api_health: bool = Field(
        True,
        description="Verify API endpoints for all packs",
    )
    check_configuration_validity: bool = Field(
        True,
        description="Validate configuration consistency across packs",
    )
    check_integration_status: bool = Field(
        True,
        description="Verify all integration bridges are functional",
    )
    alert_on_degradation: bool = Field(
        True,
        description="Send alerts when any health check fails",
    )
    alert_channels: List[NotificationChannel] = Field(
        default_factory=lambda: [NotificationChannel.EMAIL, NotificationChannel.IN_APP],
        description="Channels for health check alerts",
    )


class AuditTrailConfig(BaseModel):
    """Configuration for cross-regulation audit trail management."""

    enabled: bool = Field(
        True,
        description="Enable cross-regulation audit trail",
    )
    retention_years: int = Field(
        7,
        ge=1,
        le=25,
        description="Audit trail retention period in years",
    )
    hash_algorithm: str = Field(
        "SHA-256",
        description="Provenance hash algorithm",
    )
    immutable_log: bool = Field(
        True,
        description="Enforce immutable audit log (append-only)",
    )
    include_provenance_hash: bool = Field(
        True,
        description="Include SHA-256 provenance hash in all outputs",
    )
    cross_pack_lineage: bool = Field(
        True,
        description="Track data lineage across all four regulation packs",
    )
    export_formats: List[str] = Field(
        default_factory=lambda: ["JSON", "XML", "PDF"],
        description="Available export formats for audit trail",
    )
    user_attribution: bool = Field(
        True,
        description="Include user attribution in all audit trail entries",
    )


class DemoConfig(BaseModel):
    """Demo mode configuration for testing and onboarding."""

    demo_mode_enabled: bool = Field(
        False,
        description="Enable demo mode with synthetic data",
    )
    use_synthetic_data: bool = Field(
        False,
        description="Use synthetic test data for all regulation packs",
    )
    mock_pack_responses: bool = Field(
        False,
        description="Mock constituent pack API responses",
    )
    mock_erp_data: bool = Field(
        False,
        description="Mock ERP/finance system data",
    )
    mock_supply_chain_data: bool = Field(
        False,
        description="Mock supply chain and supplier data",
    )
    tutorial_mode_enabled: bool = Field(
        False,
        description="Enable guided tutorial mode for onboarding",
    )
    sample_regulations: List[RegulationType] = Field(
        default_factory=lambda: [RegulationType.CSRD, RegulationType.TAXONOMY],
        description="Regulations to include in demo (subset for simplicity)",
    )
    sample_data_path: str = Field(
        "",
        description="Path to sample/demo data directory",
    )


# =============================================================================
# Main Configuration Class
# =============================================================================


class BundleComplianceConfig(BaseModel):
    """EU Climate Compliance Bundle Pack main configuration.

    Central configuration class that combines all sub-configurations for
    cross-regulation compliance management. Composes PACK-001 (CSRD),
    PACK-004 (CBAM), PACK-006 (EUDR), and PACK-008 (EU Taxonomy) with
    unified engines, workflows, templates, and integrations.

    Attributes:
        pack_id: Unique bundle pack identifier
        version: Pack version string
        tier: Bundle tier classification
        bundle_tier: Deployment preset tier
        organization_name: Legal name of the reporting organization
        reporting_year: Active reporting fiscal year
        regulation_configs: Per-regulation configuration dictionary
        calendar: Unified regulatory calendar configuration
        deduplication: Data deduplication engine configuration
        consistency: Multi-regulation consistency checking configuration
        gap_analysis: Cross-regulation gap analysis configuration
        evidence: Cross-regulation evidence management configuration
        reporting: Consolidated reporting configuration
        scoring: Bundle compliance scoring configuration
        data_mapper: Cross-framework data mapper configuration
        health_check: Bundle health check configuration
        audit_trail: Cross-regulation audit trail configuration
        demo: Demo mode configuration

    Example:
        >>> config = BundleComplianceConfig(
        ...     bundle_tier=BundleTier.ENTERPRISE_FULL,
        ...     organization_name="Acme Corp AG",
        ... )
        >>> assert len(config.enabled_regulations) == 4
        >>> assert config.scoring.composite_method == ScoringMethod.WEIGHTED_AVERAGE
    """

    # Pack metadata
    pack_id: str = Field(
        "PACK-009-eu-climate-compliance-bundle",
        description="Bundle pack identifier",
    )
    version: str = Field(
        "1.0.0",
        description="Pack version",
    )
    tier: str = Field(
        "bundle",
        description="Pack tier classification",
    )

    # Organization context
    bundle_tier: BundleTier = Field(
        BundleTier.ENTERPRISE_FULL,
        description="Deployment preset tier",
    )
    organization_name: str = Field(
        "",
        description="Legal name of the reporting organization",
    )
    reporting_year: int = Field(
        2025,
        ge=2022,
        le=2030,
        description="Active reporting fiscal year",
    )

    # Regulation-level configurations
    regulation_configs: Dict[str, RegulationConfig] = Field(
        default_factory=lambda: {
            "CSRD": RegulationConfig(
                enabled=True,
                priority=1,
                pack_id="PACK-001-csrd-starter",
                display_name="Corporate Sustainability Reporting Directive",
                regulation_reference="Directive (EU) 2022/2464",
                scoring_weight=0.30,
                reporting_frequency="ANNUAL",
            ),
            "CBAM": RegulationConfig(
                enabled=True,
                priority=2,
                pack_id="PACK-004-cbam-readiness",
                display_name="Carbon Border Adjustment Mechanism",
                regulation_reference="Regulation (EU) 2023/956",
                scoring_weight=0.25,
                reporting_frequency="QUARTERLY",
            ),
            "EUDR": RegulationConfig(
                enabled=True,
                priority=3,
                pack_id="PACK-006-eudr-starter",
                display_name="EU Deforestation Regulation",
                regulation_reference="Regulation (EU) 2023/1115",
                scoring_weight=0.20,
                reporting_frequency="PER_SHIPMENT",
            ),
            "TAXONOMY": RegulationConfig(
                enabled=True,
                priority=2,
                pack_id="PACK-008-eu-taxonomy-alignment",
                display_name="EU Taxonomy Regulation",
                regulation_reference="Regulation (EU) 2020/852",
                scoring_weight=0.25,
                reporting_frequency="ANNUAL",
            ),
        },
        description="Per-regulation configuration keyed by RegulationType value",
    )

    # Sub-configurations (bundle-specific engines)
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    consistency: ConsistencyConfig = Field(default_factory=ConsistencyConfig)
    gap_analysis: GapAnalysisConfig = Field(default_factory=GapAnalysisConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    data_mapper: DataMapperConfig = Field(default_factory=DataMapperConfig)
    health_check: HealthCheckConfig = Field(default_factory=HealthCheckConfig)
    audit_trail: AuditTrailConfig = Field(default_factory=AuditTrailConfig)
    demo: DemoConfig = Field(default_factory=DemoConfig)

    @property
    def enabled_regulations(self) -> List[RegulationType]:
        """Get list of enabled regulations.

        Returns:
            List of RegulationType enums for all enabled regulations,
            sorted by priority (lowest number = highest priority).
        """
        enabled = [
            RegulationType(key)
            for key, cfg in self.regulation_configs.items()
            if cfg.enabled
        ]
        enabled.sort(
            key=lambda r: self.regulation_configs[r.value].priority,
        )
        return enabled

    @model_validator(mode="after")
    def validate_bundle_config(self) -> "BundleComplianceConfig":
        """Validate overall bundle configuration consistency."""
        enabled = [
            key for key, cfg in self.regulation_configs.items()
            if cfg.enabled
        ]

        # At least one regulation must be enabled
        if len(enabled) == 0:
            raise ValueError(
                "At least one regulation must be enabled in the bundle"
            )

        # Validate scoring weights sum to 1.0 for enabled regulations
        if self.scoring.enabled and self.scoring.composite_method == ScoringMethod.WEIGHTED_AVERAGE:
            enabled_weights = {
                key: cfg.scoring_weight
                for key, cfg in self.regulation_configs.items()
                if cfg.enabled
            }
            total = sum(enabled_weights.values())
            if abs(total - 1.0) > 0.01:
                logger.warning(
                    "Enabled regulation scoring weights sum to %.4f, not 1.0. "
                    "Weights will be auto-normalized. Enabled: %s",
                    total,
                    enabled_weights,
                )

        # Validate calendar includes deadlines for enabled regulations
        if self.calendar.enabled:
            if "CSRD" in enabled and not self.calendar.include_csrd_deadlines:
                logger.warning("CSRD is enabled but CSRD calendar deadlines are disabled")
            if "CBAM" in enabled and not self.calendar.include_cbam_deadlines:
                logger.warning("CBAM is enabled but CBAM calendar deadlines are disabled")
            if "EUDR" in enabled and not self.calendar.include_eudr_deadlines:
                logger.warning("EUDR is enabled but EUDR calendar deadlines are disabled")
            if "TAXONOMY" in enabled and not self.calendar.include_taxonomy_deadlines:
                logger.warning(
                    "TAXONOMY is enabled but Taxonomy calendar deadlines are disabled"
                )

        return self

    def get_pack_ids(self) -> Dict[str, str]:
        """Get mapping of regulation type to constituent pack ID.

        Returns:
            Dictionary mapping regulation name to pack ID for enabled regulations.
        """
        return {
            key: cfg.pack_id
            for key, cfg in self.regulation_configs.items()
            if cfg.enabled
        }

    def get_inherited_agent_count(self) -> int:
        """Get total number of inherited agents from all enabled packs.

        Returns:
            Approximate total agent count across all enabled constituent packs.
        """
        agent_counts = {
            "CSRD": 51,
            "CBAM": 47,
            "EUDR": 59,
            "TAXONOMY": 51,
        }
        return sum(
            agent_counts.get(key, 0)
            for key, cfg in self.regulation_configs.items()
            if cfg.enabled
        )

    def get_scoring_weights_normalized(self) -> Dict[str, float]:
        """Get normalized scoring weights for enabled regulations only.

        Returns:
            Dictionary of regulation -> weight, normalized to sum to 1.0.
        """
        enabled_weights = {
            key: cfg.scoring_weight
            for key, cfg in self.regulation_configs.items()
            if cfg.enabled
        }
        total = sum(enabled_weights.values())
        if total == 0:
            return enabled_weights
        return {
            key: weight / total
            for key, weight in enabled_weights.items()
        }

    def get_active_overlaps(self) -> Dict[str, Dict[str, Any]]:
        """Get cross-regulation overlaps for enabled regulation pairs.

        Returns:
            Dictionary of overlap entries where both regulations are enabled.
        """
        enabled = set(
            key for key, cfg in self.regulation_configs.items()
            if cfg.enabled
        )
        active_overlaps: Dict[str, Dict[str, Any]] = {}
        for overlap_id, overlap_data in CROSS_REGULATION_OVERLAPS.items():
            regs = set(overlap_data["regulations"])
            if regs.issubset(enabled):
                active_overlaps[overlap_id] = overlap_data
        return active_overlaps

    def get_feature_summary(self) -> Dict[str, bool]:
        """Get summary of enabled features for this configuration.

        Returns:
            Dictionary mapping feature names to enabled status.
        """
        return {
            "csrd_enabled": self.regulation_configs.get(
                "CSRD", RegulationConfig(pack_id="PACK-001-csrd-starter")
            ).enabled,
            "cbam_enabled": self.regulation_configs.get(
                "CBAM", RegulationConfig(pack_id="PACK-004-cbam-readiness")
            ).enabled,
            "eudr_enabled": self.regulation_configs.get(
                "EUDR", RegulationConfig(pack_id="PACK-006-eudr-starter")
            ).enabled,
            "taxonomy_enabled": self.regulation_configs.get(
                "TAXONOMY", RegulationConfig(pack_id="PACK-008-eu-taxonomy-alignment")
            ).enabled,
            "calendar": self.calendar.enabled,
            "deduplication": self.deduplication.enabled,
            "consistency_checking": self.consistency.enabled,
            "gap_analysis": self.gap_analysis.enabled,
            "evidence_management": self.evidence.enabled,
            "consolidated_reporting": self.reporting.enabled,
            "compliance_scoring": self.scoring.enabled,
            "data_mapping": self.data_mapper.enabled,
            "health_check": self.health_check.enabled,
            "audit_trail": self.audit_trail.enabled,
            "xbrl_tagging": self.reporting.xbrl_tagging,
            "trend_analysis": self.reporting.include_trend_analysis,
            "forecasting": self.reporting.include_forecasting,
            "demo_mode": self.demo.demo_mode_enabled,
        }

    def get_regulation_display(self) -> Dict[str, str]:
        """Get display names for all enabled regulations.

        Returns:
            Dictionary mapping regulation codes to display names.
        """
        return {
            key: cfg.display_name or REGULATION_DISPLAY_NAMES.get(key, key)
            for key, cfg in self.regulation_configs.items()
            if cfg.enabled
        }


# =============================================================================
# PackConfig - Top-Level Configuration Loader
# =============================================================================


class PackConfig(BaseModel):
    """Top-level pack configuration loader with YAML and preset support.

    Loads configuration from preset files, applies environment overrides,
    and provides methods for export, hashing, and introspection.

    Configuration Merge Order:
        1. Base defaults from BundleComplianceConfig
        2. Bundle preset YAML (if specified)
        3. Environment variable overrides (BUNDLE_PACK_*)
        4. Explicit runtime overrides

    Example:
        >>> config = PackConfig.from_preset("enterprise_full")
        >>> print(config.pack.bundle_tier)
        BundleTier.ENTERPRISE_FULL
        >>> print(config.get_config_hash()[:16])
        'a1b2c3d4e5f6g7h8'
    """

    pack: BundleComplianceConfig = Field(
        default_factory=BundleComplianceConfig,
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

        config = cls._build_config_from_dict(data)
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
            preset_name: Preset name (enterprise_full, financial_institution,
                eu_importer, sme_essential).
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

        config = BundleComplianceConfig()

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
                logger.info("Loaded bundle preset: %s", preset_name)
        else:
            logger.warning("Preset file not found: %s", preset_path)

        # Apply environment variable overrides
        config = cls._apply_env_overrides(config)

        # Enable demo mode if requested
        if demo_mode:
            config.demo.demo_mode_enabled = True
            config.demo.use_synthetic_data = True
            config.demo.mock_pack_responses = True
            config.demo.mock_erp_data = True
            config.demo.mock_supply_chain_data = True
            logger.info("Demo mode enabled with synthetic data")

        return cls(pack=config, loaded_from=loaded_files)

    @classmethod
    def available_presets(cls) -> Dict[str, str]:
        """Get dictionary of available presets and their descriptions.

        Returns:
            Dictionary mapping preset names to descriptions.
        """
        return dict(AVAILABLE_PRESETS)

    @classmethod
    def available_regulations(cls) -> Dict[str, str]:
        """Get dictionary of available regulations and their references.

        Returns:
            Dictionary mapping regulation codes to official references.
        """
        return dict(REGULATION_REFERENCES)

    @staticmethod
    def _build_config_from_dict(
        data: Dict[str, Any],
    ) -> BundleComplianceConfig:
        """Build BundleComplianceConfig from a raw dictionary.

        Args:
            data: Dictionary of configuration values.

        Returns:
            BundleComplianceConfig instance.
        """
        return BundleComplianceConfig(**data)

    @staticmethod
    def _merge_config(
        base: BundleComplianceConfig,
        overlay: Dict[str, Any],
    ) -> BundleComplianceConfig:
        """Deep merge overlay configuration into base configuration.

        Args:
            base: Base BundleComplianceConfig instance.
            overlay: Dictionary of override values to merge.

        Returns:
            New BundleComplianceConfig with merged values.
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
        return BundleComplianceConfig(**merged)

    @staticmethod
    def _apply_env_overrides(
        config: BundleComplianceConfig,
    ) -> BundleComplianceConfig:
        """Apply environment variable overrides to configuration.

        Looks for BUNDLE_PACK_* environment variables and applies them
        as configuration overrides.

        Args:
            config: Current configuration instance.

        Returns:
            Configuration with environment overrides applied.
        """
        env_mapping: Dict[str, str] = {
            "BUNDLE_PACK_ORG_NAME": "organization_name",
            "BUNDLE_PACK_REPORTING_YEAR": "reporting_year",
            "BUNDLE_PACK_BUNDLE_TIER": "bundle_tier",
            "BUNDLE_PACK_DEMO_MODE": "demo.demo_mode_enabled",
            "BUNDLE_PACK_LANGUAGE": "reporting.language",
            "BUNDLE_PACK_TIMEZONE": "reporting.timezone",
            "BUNDLE_PACK_CSRD_ENABLED": "regulation_configs.CSRD.enabled",
            "BUNDLE_PACK_CBAM_ENABLED": "regulation_configs.CBAM.enabled",
            "BUNDLE_PACK_EUDR_ENABLED": "regulation_configs.EUDR.enabled",
            "BUNDLE_PACK_TAXONOMY_ENABLED": "regulation_configs.TAXONOMY.enabled",
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

        return BundleComplianceConfig(**config_dict)

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
    def active_regulations(self) -> List[RegulationType]:
        """Get all enabled regulations for this configuration."""
        return self.pack.enabled_regulations

    @property
    def feature_summary(self) -> Dict[str, bool]:
        """Get feature summary for this configuration."""
        return self.pack.get_feature_summary()

    @property
    def pack_ids(self) -> Dict[str, str]:
        """Get mapping of regulation to constituent pack ID."""
        return self.pack.get_pack_ids()

    @property
    def inherited_agent_count(self) -> int:
        """Get total inherited agent count."""
        return self.pack.get_inherited_agent_count()

    @property
    def scoring_weights(self) -> Dict[str, float]:
        """Get normalized scoring weights for enabled regulations."""
        return self.pack.get_scoring_weights_normalized()

    @property
    def active_overlaps(self) -> Dict[str, Dict[str, Any]]:
        """Get cross-regulation overlaps for enabled regulation pairs."""
        return self.pack.get_active_overlaps()


# =============================================================================
# Utility Functions
# =============================================================================


def get_default_config() -> PackConfig:
    """Get default bundle configuration with all regulations enabled.

    Returns:
        PackConfig with default BundleComplianceConfig (enterprise_full equivalent).

    Example:
        >>> config = get_default_config()
        >>> assert len(config.active_regulations) == 4
        >>> assert config.pack.scoring.enabled is True
    """
    return PackConfig(pack=BundleComplianceConfig())


def get_regulation_display_name(regulation: Union[str, RegulationType]) -> str:
    """Get human-readable display name for a regulation.

    Args:
        regulation: Regulation code (string or enum).

    Returns:
        Full display name string.
    """
    key = regulation.value if isinstance(regulation, RegulationType) else regulation
    return REGULATION_DISPLAY_NAMES.get(key, f"Unknown Regulation ({key})")


def get_regulation_reference(regulation: Union[str, RegulationType]) -> str:
    """Get official regulation reference for a regulation.

    Args:
        regulation: Regulation code (string or enum).

    Returns:
        Official EU regulation reference string.
    """
    key = regulation.value if isinstance(regulation, RegulationType) else regulation
    return REGULATION_REFERENCES.get(key, f"Unknown ({key})")


def get_regulation_effective_date(regulation: Union[str, RegulationType]) -> Optional[str]:
    """Get effective date for a regulation.

    Args:
        regulation: Regulation code (string or enum).

    Returns:
        Effective date string (YYYY-MM-DD) or None if not found.
    """
    key = regulation.value if isinstance(regulation, RegulationType) else regulation
    return REGULATION_EFFECTIVE_DATES.get(key)


def get_cross_regulation_overlaps(
    regulation_a: Union[str, RegulationType],
    regulation_b: Union[str, RegulationType],
) -> Optional[Dict[str, Any]]:
    """Get cross-regulation overlap data for a regulation pair.

    Args:
        regulation_a: First regulation code.
        regulation_b: Second regulation code.

    Returns:
        Dictionary with overlap data including shared fields, or None.
    """
    key_a = regulation_a.value if isinstance(regulation_a, RegulationType) else regulation_a
    key_b = regulation_b.value if isinstance(regulation_b, RegulationType) else regulation_b

    # Try both orderings
    overlap_key_1 = f"{key_a}-{key_b}"
    overlap_key_2 = f"{key_b}-{key_a}"

    return (
        CROSS_REGULATION_OVERLAPS.get(overlap_key_1)
        or CROSS_REGULATION_OVERLAPS.get(overlap_key_2)
    )


def validate_bundle_consistency(
    csrd_compliant: bool,
    cbam_compliant: bool,
    eudr_compliant: bool,
    taxonomy_compliant: bool,
) -> Tuple[ComplianceStatus, str]:
    """Evaluate overall bundle compliance status from per-regulation results.

    Args:
        csrd_compliant: CSRD compliance result.
        cbam_compliant: CBAM compliance result.
        eudr_compliant: EUDR compliance result.
        taxonomy_compliant: EU Taxonomy compliance result.

    Returns:
        Tuple of (ComplianceStatus, explanation string).
    """
    regulations = {
        "CSRD": csrd_compliant,
        "CBAM": cbam_compliant,
        "EUDR": eudr_compliant,
        "EU Taxonomy": taxonomy_compliant,
    }

    compliant = [name for name, status in regulations.items() if status]
    non_compliant = [name for name, status in regulations.items() if not status]

    if len(non_compliant) == 0:
        return (
            ComplianceStatus.COMPLIANT,
            "All four EU regulations are fully compliant",
        )
    elif len(compliant) == 0:
        return (
            ComplianceStatus.NON_COMPLIANT,
            "No EU regulations are compliant",
        )
    else:
        return (
            ComplianceStatus.PARTIALLY_COMPLIANT,
            f"Compliant: {', '.join(compliant)}. "
            f"Non-compliant: {', '.join(non_compliant)}",
        )


def compute_composite_score(
    regulation_scores: Dict[str, float],
    weights: Dict[str, float],
    method: ScoringMethod = ScoringMethod.WEIGHTED_AVERAGE,
) -> float:
    """Compute composite compliance score from per-regulation scores.

    Args:
        regulation_scores: Dictionary of regulation code to score (0-100).
        weights: Dictionary of regulation code to weight (0-1.0).
        method: Scoring aggregation method.

    Returns:
        Composite score (0-100).

    Raises:
        ValueError: If no scores provided or invalid method.
    """
    if not regulation_scores:
        raise ValueError("At least one regulation score is required")

    # Filter to regulations present in both scores and weights
    common_keys = set(regulation_scores.keys()) & set(weights.keys())
    if not common_keys:
        raise ValueError("No common regulations between scores and weights")

    scores = {k: regulation_scores[k] for k in common_keys}
    reg_weights = {k: weights[k] for k in common_keys}

    # Normalize weights to sum to 1.0
    total_weight = sum(reg_weights.values())
    if total_weight > 0:
        reg_weights = {k: v / total_weight for k, v in reg_weights.items()}

    if method == ScoringMethod.WEIGHTED_AVERAGE:
        return sum(
            scores[k] * reg_weights[k]
            for k in common_keys
        )

    elif method == ScoringMethod.MINIMUM_SCORE:
        return min(scores.values())

    elif method == ScoringMethod.GEOMETRIC_MEAN:
        import math
        product = 1.0
        count = 0
        for k in common_keys:
            if scores[k] > 0:
                product *= scores[k]
                count += 1
        if count == 0:
            return 0.0
        return product ** (1.0 / count)

    elif method == ScoringMethod.HARMONIC_MEAN:
        reciprocal_sum = 0.0
        count = 0
        for k in common_keys:
            if scores[k] > 0:
                reciprocal_sum += 1.0 / scores[k]
                count += 1
        if count == 0 or reciprocal_sum == 0:
            return 0.0
        return count / reciprocal_sum

    else:
        raise ValueError(f"Unknown scoring method: {method}")


def get_letter_grade(
    score: float,
    thresholds: Optional[Dict[str, float]] = None,
) -> str:
    """Convert a numeric score to a letter grade.

    Args:
        score: Numeric score (0-100).
        thresholds: Optional custom grade thresholds.

    Returns:
        Letter grade string (A, B, C, D, or F).
    """
    if thresholds is None:
        thresholds = {"A": 90.0, "B": 80.0, "C": 70.0, "D": 60.0, "F": 0.0}

    sorted_grades = sorted(thresholds.items(), key=lambda x: x[1], reverse=True)
    for grade, threshold in sorted_grades:
        if score >= threshold:
            return grade

    return "F"
