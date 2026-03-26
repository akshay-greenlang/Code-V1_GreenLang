"""
PACK-044 Inventory Management Pack - Configuration Manager

Pydantic v2 configuration for GHG inventory lifecycle management including
period management, data collection, quality assurance, change management,
review/approval, versioning, consolidation, gap analysis, documentation,
and benchmarking.

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (sector-specific defaults)
    3. Environment overrides (INVMGMT_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    GHG Protocol Corporate Standard (Revised Edition, 2015) Ch 1-8
    ISO 14064-1:2018
    ISO 14064-3:2019 (Verification)
    EU CSRD / ESRS E1
    CDP Climate Change 2026
    SBTi Corporate Net-Zero Standard v1.1

Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
"""

import hashlib
import logging
import os
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

PACK_BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent


# =============================================================================
# Enums
# =============================================================================


class InventoryPeriodStatus(str, Enum):
    PLANNING = "PLANNING"
    DATA_COLLECTION = "DATA_COLLECTION"
    CALCULATION = "CALCULATION"
    REVIEW = "REVIEW"
    APPROVED = "APPROVED"
    FINAL = "FINAL"
    AMENDED = "AMENDED"
    ARCHIVED = "ARCHIVED"


class DataCollectionStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    SUBMITTED = "SUBMITTED"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"


class QualityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class ChangeType(str, Enum):
    ORGANIZATIONAL = "ORGANIZATIONAL"
    METHODOLOGY = "METHODOLOGY"
    EMISSION_FACTOR = "EMISSION_FACTOR"
    ERROR_CORRECTION = "ERROR_CORRECTION"
    STRUCTURAL = "STRUCTURAL"


class ReviewStage(str, Enum):
    PREPARER = "PREPARER"
    REVIEWER = "REVIEWER"
    APPROVER = "APPROVER"
    VERIFIER = "VERIFIER"


class VersionStatus(str, Enum):
    DRAFT = "DRAFT"
    UNDER_REVIEW = "UNDER_REVIEW"
    FINAL = "FINAL"
    AMENDED = "AMENDED"
    SUPERSEDED = "SUPERSEDED"


class ConsolidationApproach(str, Enum):
    EQUITY_SHARE = "EQUITY_SHARE"
    OPERATIONAL_CONTROL = "OPERATIONAL_CONTROL"
    FINANCIAL_CONTROL = "FINANCIAL_CONTROL"


class GapPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class BenchmarkSource(str, Enum):
    SECTOR_AVERAGE = "SECTOR_AVERAGE"
    CDP_PEER = "CDP_PEER"
    INTERNAL_HISTORICAL = "INTERNAL_HISTORICAL"
    CUSTOM = "CUSTOM"


class SectorType(str, Enum):
    OFFICE = "OFFICE"
    MANUFACTURING = "MANUFACTURING"
    ENERGY_UTILITY = "ENERGY_UTILITY"
    TRANSPORT_LOGISTICS = "TRANSPORT_LOGISTICS"
    FOOD_AGRICULTURE = "FOOD_AGRICULTURE"
    REAL_ESTATE = "REAL_ESTATE"
    HEALTHCARE = "HEALTHCARE"
    SME = "SME"


class OutputFormat(str, Enum):
    MARKDOWN = "MARKDOWN"
    HTML = "HTML"
    JSON = "JSON"
    CSV = "CSV"
    XBRL = "XBRL"


class ReportingFrequency(str, Enum):
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    ANNUAL = "ANNUAL"


class NotificationChannel(str, Enum):
    EMAIL = "EMAIL"
    SLACK = "SLACK"
    TEAMS = "TEAMS"
    WEBHOOK = "WEBHOOK"
    IN_APP = "IN_APP"


class FrameworkType(str, Enum):
    GHG_PROTOCOL = "GHG_PROTOCOL"
    ESRS_E1 = "ESRS_E1"
    CDP = "CDP"
    ISO_14064 = "ISO_14064"
    SBTI = "SBTI"
    SEC = "SEC"
    SB_253 = "SB_253"


# =============================================================================
# Reference Data
# =============================================================================

SECTOR_INFO: Dict[str, Dict[str, Any]] = {
    "OFFICE": {
        "name": "Corporate Office",
        "typical_data_sources": ["Utility bills", "Fleet fuel cards", "Refrigerant service records"],
        "collection_frequency": "QUARTERLY",
        "review_levels": 2,
        "typical_inventory_complexity": "LOW",
    },
    "MANUFACTURING": {
        "name": "Manufacturing Facility",
        "typical_data_sources": ["ERP fuel records", "Process data", "Meter readings", "Stack monitoring"],
        "collection_frequency": "MONTHLY",
        "review_levels": 3,
        "typical_inventory_complexity": "HIGH",
    },
    "ENERGY_UTILITY": {
        "name": "Energy Utility",
        "typical_data_sources": ["CEMS", "Fuel receipts", "Generation logs", "SF6 tracking"],
        "collection_frequency": "MONTHLY",
        "review_levels": 3,
        "typical_inventory_complexity": "VERY_HIGH",
    },
    "TRANSPORT_LOGISTICS": {
        "name": "Transport & Logistics",
        "typical_data_sources": ["Fleet fuel cards", "Telematics", "Fuel receipts"],
        "collection_frequency": "MONTHLY",
        "review_levels": 2,
        "typical_inventory_complexity": "MEDIUM",
    },
    "FOOD_AGRICULTURE": {
        "name": "Food & Agriculture",
        "typical_data_sources": ["Farm records", "Fertiliser purchase", "Livestock data", "Energy bills"],
        "collection_frequency": "QUARTERLY",
        "review_levels": 2,
        "typical_inventory_complexity": "HIGH",
    },
    "REAL_ESTATE": {
        "name": "Real Estate Portfolio",
        "typical_data_sources": ["Utility bills", "Tenant data", "BMS exports"],
        "collection_frequency": "QUARTERLY",
        "review_levels": 2,
        "typical_inventory_complexity": "MEDIUM",
    },
    "HEALTHCARE": {
        "name": "Healthcare System",
        "typical_data_sources": ["Utility bills", "Medical gas records", "Fleet data", "BMS"],
        "collection_frequency": "QUARTERLY",
        "review_levels": 3,
        "typical_inventory_complexity": "HIGH",
    },
    "SME": {
        "name": "Small-Medium Enterprise",
        "typical_data_sources": ["Utility bills", "Fuel receipts"],
        "collection_frequency": "ANNUAL",
        "review_levels": 1,
        "typical_inventory_complexity": "LOW",
    },
}

AVAILABLE_PRESETS: Dict[str, str] = {
    "corporate_office": "Office-based organisations (financial services, technology, consulting)",
    "manufacturing": "Industrial manufacturing with process emissions and stationary combustion",
    "energy_utility": "Power generation and energy distribution utilities",
    "transport_logistics": "Fleet operators and logistics companies",
    "food_agriculture": "Agricultural operations and food processing",
    "real_estate": "Property portfolios and REIT companies",
    "healthcare": "Hospitals and healthcare systems",
    "sme_simplified": "Simplified management for small-medium enterprises",
}


# =============================================================================
# Sub-Config Models
# =============================================================================


class PeriodManagementConfig(BaseModel):
    auto_create_periods: bool = Field(True, description="Auto-create next period on finalization")
    lock_after_approval: bool = Field(True, description="Lock period data after final approval")
    amendment_requires_justification: bool = Field(True, description="Require justification for amendments")
    max_open_periods: int = Field(3, ge=1, le=10, description="Max concurrent open periods")
    retention_years: int = Field(7, ge=1, le=15, description="Period data retention in years")
    milestone_tracking: bool = Field(True, description="Track milestones and deadlines")
    auto_archive_after_months: int = Field(24, ge=6, le=120, description="Auto-archive finalized periods after N months")


class DataCollectionConfig(BaseModel):
    auto_scheduling: bool = Field(True, description="Auto-schedule data collection campaigns")
    reminder_frequency_days: int = Field(7, ge=1, le=30, description="Reminder frequency in days")
    escalation_after_days: int = Field(21, ge=7, le=90, description="Escalate overdue after N days")
    default_deadline_days: int = Field(30, ge=7, le=180, description="Default deadline for data requests")
    require_evidence_upload: bool = Field(True, description="Require evidence documents for submissions")
    min_data_quality_score: float = Field(3.0, ge=1.0, le=5.0, description="Minimum acceptable data quality score")
    collection_frequency: ReportingFrequency = Field(ReportingFrequency.QUARTERLY, description="Data collection frequency")
    parallel_collection: bool = Field(True, description="Allow parallel data collection across facilities")


class QualityManagementConfig(BaseModel):
    enabled: bool = Field(True, description="Enable QA/QC procedures")
    auto_qaqc: bool = Field(True, description="Run automated QA/QC on data submission")
    completeness_threshold_pct: float = Field(95.0, ge=80.0, le=100.0, description="Completeness threshold")
    consistency_threshold_pct: float = Field(20.0, ge=5.0, le=50.0, description="YoY change alert threshold")
    accuracy_threshold_pct: float = Field(10.0, ge=1.0, le=30.0, description="Accuracy tolerance threshold")
    yoy_change_alert_pct: float = Field(20.0, ge=5.0, le=50.0, description="Year-on-year change alert")
    review_levels: int = Field(2, ge=1, le=4, description="Number of review levels required")
    continuous_improvement: bool = Field(True, description="Track continuous improvement actions")


class ChangeManagementConfig(BaseModel):
    require_impact_assessment: bool = Field(True, description="Require impact assessment for all changes")
    significance_threshold_pct: float = Field(5.0, ge=1.0, le=20.0, description="Significance threshold for changes")
    auto_detect_triggers: bool = Field(True, description="Auto-detect recalculation triggers")
    require_approval_for_changes: bool = Field(True, description="Require approval for inventory changes")
    base_year_recalculation_threshold_pct: float = Field(5.0, ge=1.0, le=20.0, description="Base year recalculation trigger threshold")
    track_methodology_changes: bool = Field(True, description="Track methodology changes with versioning")


class ReviewApprovalConfig(BaseModel):
    review_levels: List[ReviewStage] = Field(
        default_factory=lambda: [ReviewStage.PREPARER, ReviewStage.REVIEWER, ReviewStage.APPROVER],
        description="Required review stages",
    )
    require_digital_signature: bool = Field(True, description="Require digital signature for approval")
    auto_remind_reviewers: bool = Field(True, description="Auto-remind reviewers of pending reviews")
    reminder_frequency_days: int = Field(3, ge=1, le=14, description="Reviewer reminder frequency")
    escalation_after_days: int = Field(14, ge=3, le=60, description="Escalate after N days")
    allow_delegation: bool = Field(True, description="Allow review delegation")


class VersioningConfig(BaseModel):
    auto_version_on_changes: bool = Field(True, description="Auto-create version on significant changes")
    max_draft_versions: int = Field(10, ge=1, le=50, description="Max draft versions before cleanup")
    require_change_description: bool = Field(True, description="Require description for version changes")
    allow_rollback: bool = Field(True, description="Allow rollback to previous versions")
    track_field_level_changes: bool = Field(True, description="Track field-level diffs between versions")
    immutable_after_finalization: bool = Field(True, description="Make finalized versions immutable")


class ConsolidationConfig(BaseModel):
    approach: ConsolidationApproach = Field(ConsolidationApproach.OPERATIONAL_CONTROL, description="Consolidation approach")
    auto_detect_subsidiaries: bool = Field(False, description="Auto-detect subsidiary entities from ERP")
    equity_threshold_pct: float = Field(20.0, ge=0.0, le=100.0, description="Min equity share for inclusion")
    require_subsidiary_approval: bool = Field(True, description="Require subsidiary sign-off on data")
    parallel_data_collection: bool = Field(True, description="Collect subsidiary data in parallel")
    eliminate_intragroup: bool = Field(True, description="Eliminate intra-group transfers")


class GapAnalysisConfig(BaseModel):
    enabled: bool = Field(True, description="Enable gap analysis")
    auto_detect_gaps: bool = Field(True, description="Auto-detect data quality gaps")
    methodology_tier_target: int = Field(2, ge=1, le=3, description="Target methodology tier (1-3)")
    data_quality_target: float = Field(4.0, ge=1.0, le=5.0, description="Target data quality score")
    improvement_horizon_years: int = Field(3, ge=1, le=10, description="Improvement planning horizon")
    include_cost_benefit: bool = Field(True, description="Include cost-benefit in recommendations")


class DocumentationConfig(BaseModel):
    require_methodology_docs: bool = Field(True, description="Require methodology documentation per source")
    require_assumptions_register: bool = Field(True, description="Maintain assumptions register")
    require_evidence_links: bool = Field(True, description="Link all data to source evidence")
    auto_generate_methodology_notes: bool = Field(True, description="Auto-generate methodology notes")
    documentation_review_required: bool = Field(True, description="Require documentation review")
    verification_readiness: bool = Field(True, description="Maintain verification-ready documentation")


class BenchmarkingConfig(BaseModel):
    enabled: bool = Field(True, description="Enable benchmarking")
    benchmark_sources: List[BenchmarkSource] = Field(
        default_factory=lambda: [BenchmarkSource.SECTOR_AVERAGE, BenchmarkSource.INTERNAL_HISTORICAL],
        description="Benchmark data sources",
    )
    peer_group_size: int = Field(10, ge=3, le=50, description="Peer group size for comparison")
    sector_comparison: bool = Field(True, description="Compare against sector averages")
    internal_facility_ranking: bool = Field(True, description="Rank internal facilities")
    intensity_metrics: List[str] = Field(
        default_factory=lambda: ["tCO2e/MEUR_revenue", "tCO2e/FTE"],
        description="Intensity metrics for benchmarking",
    )


class NotificationConfig(BaseModel):
    channels: List[NotificationChannel] = Field(
        default_factory=lambda: [NotificationChannel.EMAIL, NotificationChannel.IN_APP],
        description="Notification channels",
    )
    deadline_reminders: bool = Field(True, description="Send deadline reminders")
    quality_alerts: bool = Field(True, description="Send quality issue alerts")
    change_notifications: bool = Field(True, description="Notify on inventory changes")
    approval_notifications: bool = Field(True, description="Notify on approval actions")


class SecurityConfig(BaseModel):
    roles: List[str] = Field(
        default_factory=lambda: [
            "inventory_manager", "data_collector", "reviewer",
            "approver", "verifier", "viewer", "admin",
        ],
        description="Available RBAC roles",
    )
    data_classification: str = Field("CONFIDENTIAL", description="Default data classification")
    audit_logging: bool = Field(True, description="Enable audit logging")
    pii_redaction: bool = Field(True, description="Enable PII redaction")
    encryption_at_rest: bool = Field(True, description="Require encryption at rest")


class PerformanceConfig(BaseModel):
    max_inventory_periods: int = Field(20, ge=1, le=100, description="Max inventory periods")
    max_entities: int = Field(500, ge=1, le=5000, description="Max entities for consolidation")
    max_facilities: int = Field(1000, ge=1, le=10000, description="Max facilities")
    cache_ttl_seconds: int = Field(3600, ge=60, le=86400, description="Cache TTL seconds")
    batch_size: int = Field(500, ge=50, le=5000, description="Batch size for bulk operations")
    calculation_timeout_seconds: int = Field(300, ge=30, le=1800, description="Calculation timeout")


class AuditTrailConfig(BaseModel):
    enabled: bool = Field(True, description="Enable audit trail")
    sha256_provenance: bool = Field(True, description="SHA-256 provenance hashing")
    calculation_logging: bool = Field(True, description="Log calculations")
    assumption_tracking: bool = Field(True, description="Track assumptions")
    data_lineage_enabled: bool = Field(True, description="Track data lineage")
    retention_years: int = Field(7, ge=1, le=15, description="Retention years")


class ReportingConfig(BaseModel):
    frequency: ReportingFrequency = Field(ReportingFrequency.ANNUAL, description="Reporting frequency")
    formats: List[OutputFormat] = Field(
        default_factory=lambda: [OutputFormat.HTML, OutputFormat.JSON],
        description="Output formats",
    )
    executive_summary: bool = Field(True, description="Generate executive summary")
    detailed_inventory: bool = Field(True, description="Generate detailed inventory")
    verification_package: bool = Field(True, description="Generate verification package")
    output_language: str = Field("en", description="Report language (ISO 639-1)")


# =============================================================================
# Main Configuration Model
# =============================================================================


class InventoryManagementConfig(BaseModel):
    company_name: str = Field("", description="Reporting company legal name")
    sector_type: SectorType = Field(SectorType.OFFICE, description="Sector classification")
    country: str = Field("DE", description="Primary country (ISO 3166-1 alpha-2)")
    reporting_year: int = Field(2026, ge=2020, le=2035, description="Reporting year")
    revenue_meur: Optional[float] = Field(None, ge=0, description="Annual revenue MEUR")
    employees_fte: Optional[int] = Field(None, ge=0, description="FTE employees")
    floor_area_m2: Optional[float] = Field(None, ge=0, description="Floor area m2")

    period_management: PeriodManagementConfig = Field(default_factory=PeriodManagementConfig)
    data_collection: DataCollectionConfig = Field(default_factory=DataCollectionConfig)
    quality_management: QualityManagementConfig = Field(default_factory=QualityManagementConfig)
    change_management: ChangeManagementConfig = Field(default_factory=ChangeManagementConfig)
    review_approval: ReviewApprovalConfig = Field(default_factory=ReviewApprovalConfig)
    versioning: VersioningConfig = Field(default_factory=VersioningConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    gap_analysis: GapAnalysisConfig = Field(default_factory=GapAnalysisConfig)
    documentation: DocumentationConfig = Field(default_factory=DocumentationConfig)
    benchmarking: BenchmarkingConfig = Field(default_factory=BenchmarkingConfig)
    notification: NotificationConfig = Field(default_factory=NotificationConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    audit_trail: AuditTrailConfig = Field(default_factory=AuditTrailConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    @model_validator(mode="after")
    def validate_sme_simplified(self) -> "InventoryManagementConfig":
        if self.sector_type == SectorType.SME:
            if self.quality_management.review_levels > 2:
                self.quality_management.review_levels = 1
            if self.data_collection.collection_frequency != ReportingFrequency.ANNUAL:
                self.data_collection.collection_frequency = ReportingFrequency.ANNUAL
        return self


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    pack: InventoryManagementConfig = Field(default_factory=InventoryManagementConfig)
    preset_name: Optional[str] = Field(None)
    config_version: str = Field("1.0.0")
    pack_id: str = Field("PACK-044-inventory-management")

    @classmethod
    def from_preset(cls, preset_name: str, overrides: Optional[Dict[str, Any]] = None) -> "PackConfig":
        if preset_name not in AVAILABLE_PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {sorted(AVAILABLE_PRESETS.keys())}")
        preset_path = CONFIG_DIR / "presets" / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Preset file not found: {preset_path}")
        with open(preset_path, "r", encoding="utf-8") as f:
            preset_data = yaml.safe_load(f) or {}
        env_overrides = cls._load_env_overrides()
        if env_overrides:
            preset_data = cls._deep_merge(preset_data, env_overrides)
        if overrides:
            preset_data = cls._deep_merge(preset_data, overrides)
        pack_config = InventoryManagementConfig(**preset_data)
        return cls(pack=pack_config, preset_name=preset_name)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "PackConfig":
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        pack_config = InventoryManagementConfig(**config_data)
        return cls(pack=pack_config)

    @classmethod
    def merge(cls, base: "PackConfig", overrides: Dict[str, Any]) -> "PackConfig":
        base_dict = base.pack.model_dump()
        merged = cls._deep_merge(base_dict, overrides)
        pack_config = InventoryManagementConfig(**merged)
        return cls(pack=pack_config, preset_name=base.preset_name, config_version=base.config_version)

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        overrides: Dict[str, Any] = {}
        prefix = "INVMGMT_PACK_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                parts = config_key.split("__")
                current = overrides
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
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
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = PackConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get_config_hash(self) -> str:
        config_json = self.model_dump_json(indent=None)
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()

    def validate_completeness(self) -> List[str]:
        return validate_config(self.pack)


# =============================================================================
# Utility Functions
# =============================================================================


def load_preset(preset_name: str, overrides: Optional[Dict[str, Any]] = None) -> PackConfig:
    return PackConfig.from_preset(preset_name, overrides)


def validate_config(config: InventoryManagementConfig) -> List[str]:
    warnings: List[str] = []
    if not config.company_name:
        warnings.append("No company_name configured.")
    if config.quality_management.review_levels < 2 and config.sector_type != SectorType.SME:
        warnings.append("Non-SME organisations should have at least 2 review levels.")
    if config.documentation.verification_readiness and not config.documentation.require_evidence_links:
        warnings.append("Verification readiness requires evidence links.")
    if config.benchmarking.enabled and not config.benchmarking.intensity_metrics:
        warnings.append("Benchmarking enabled but no intensity metrics configured.")
    return warnings


def get_default_config(sector_type: SectorType = SectorType.OFFICE) -> InventoryManagementConfig:
    return InventoryManagementConfig(sector_type=sector_type)


def list_available_presets() -> Dict[str, str]:
    return AVAILABLE_PRESETS.copy()


def get_sector_info(sector_type: Union[str, SectorType]) -> Dict[str, Any]:
    key = sector_type.value if isinstance(sector_type, SectorType) else sector_type
    return SECTOR_INFO.get(key, {"name": key, "typical_inventory_complexity": "MEDIUM"})
