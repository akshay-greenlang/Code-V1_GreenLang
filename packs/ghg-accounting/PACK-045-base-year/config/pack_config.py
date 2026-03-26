"""
PACK-045 Base Year Management Pack - Configuration Manager

Pydantic v2 configuration for GHG base year establishment, recalculation
policy management, structural change detection, significance testing,
time series adjustment, target tracking, and audit-ready reporting.

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (sector-specific defaults)
    3. Environment overrides (BASEYEAR_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    GHG Protocol Corporate Standard (Revised Edition, 2015) Ch 5 - Tracking Emissions Over Time
    GHG Protocol Corporate Standard (Revised Edition, 2015) Ch 9 - Setting a GHG Target
    ISO 14064-1:2018 Clause 9 - Base Year
    SBTi Corporate Net-Zero Standard v1.1 - Target Setting Requirements
    EU CSRD / ESRS E1 - Climate Change Disclosure
    CDP Climate Change 2026 - Module C: Targets and Performance
    SEC Climate Disclosure Rule - Historical Emissions Restatement

Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
"""

import hashlib
import logging
import os
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


class BaseYearType(str, Enum):
    """Strategy for base year selection."""
    FIXED = "FIXED"
    ROLLING_3YR = "ROLLING_3YR"
    ROLLING_5YR = "ROLLING_5YR"


class RecalculationTriggerType(str, Enum):
    """Events that may trigger base year recalculation per GHG Protocol Ch 5."""
    ACQUISITION = "ACQUISITION"
    DIVESTITURE = "DIVESTITURE"
    MERGER = "MERGER"
    METHODOLOGY_CHANGE = "METHODOLOGY_CHANGE"
    ERROR_CORRECTION = "ERROR_CORRECTION"
    SOURCE_BOUNDARY_CHANGE = "SOURCE_BOUNDARY_CHANGE"
    OUTSOURCING_INSOURCING = "OUTSOURCING_INSOURCING"


class SignificanceMethod(str, Enum):
    """Method for determining whether a change is significant."""
    INDIVIDUAL = "INDIVIDUAL"
    CUMULATIVE = "CUMULATIVE"
    COMBINED = "COMBINED"


class AdjustmentApproach(str, Enum):
    """Approach for adjusting base year emissions after a trigger event."""
    PRO_RATA = "PRO_RATA"
    FULL_YEAR = "FULL_YEAR"
    WEIGHTED_AVERAGE = "WEIGHTED_AVERAGE"


class ConsolidationApproach(str, Enum):
    """Organizational boundary consolidation approach per GHG Protocol Ch 3."""
    EQUITY_SHARE = "EQUITY_SHARE"
    OPERATIONAL_CONTROL = "OPERATIONAL_CONTROL"
    FINANCIAL_CONTROL = "FINANCIAL_CONTROL"


class TargetType(str, Enum):
    """GHG reduction target type per SBTi and GHG Protocol."""
    ABSOLUTE = "ABSOLUTE"
    INTENSITY = "INTENSITY"
    BOTH = "BOTH"


class SBTiAmbitionLevel(str, Enum):
    """Science Based Targets initiative ambition levels."""
    WELL_BELOW_2C = "WELL_BELOW_2C"
    ONE_POINT_FIVE_C = "ONE_POINT_FIVE_C"
    NET_ZERO = "NET_ZERO"


class AuditLevel(str, Enum):
    """Third-party assurance engagement level per ISAE 3410."""
    INTERNAL = "INTERNAL"
    LIMITED_ASSURANCE = "LIMITED_ASSURANCE"
    REASONABLE_ASSURANCE = "REASONABLE_ASSURANCE"


class ReportingFramework(str, Enum):
    """Supported regulatory and voluntary reporting frameworks."""
    GHG_PROTOCOL = "GHG_PROTOCOL"
    ISO_14064 = "ISO_14064"
    ESRS_E1 = "ESRS_E1"
    CDP = "CDP"
    SBTI = "SBTI"
    SEC = "SEC"
    SB_253 = "SB_253"
    TCFD = "TCFD"


class GWPVersion(str, Enum):
    """IPCC Assessment Report version for Global Warming Potential values."""
    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"


class ScopeType(str, Enum):
    """GHG Protocol emission scope classifications."""
    SCOPE_1 = "SCOPE_1"
    SCOPE_2_LOCATION = "SCOPE_2_LOCATION"
    SCOPE_2_MARKET = "SCOPE_2_MARKET"
    SCOPE_3 = "SCOPE_3"


class OutputFormat(str, Enum):
    """Supported report output formats."""
    MARKDOWN = "MARKDOWN"
    HTML = "HTML"
    JSON = "JSON"
    CSV = "CSV"
    PDF = "PDF"


class NotificationChannel(str, Enum):
    """Available notification delivery channels."""
    EMAIL = "EMAIL"
    SLACK = "SLACK"
    TEAMS = "TEAMS"
    WEBHOOK = "WEBHOOK"


class SectorType(str, Enum):
    """Industry sector classification for preset defaults."""
    CORPORATE_OFFICE = "CORPORATE_OFFICE"
    MANUFACTURING = "MANUFACTURING"
    ENERGY_UTILITY = "ENERGY_UTILITY"
    TRANSPORT_LOGISTICS = "TRANSPORT_LOGISTICS"
    FOOD_AGRICULTURE = "FOOD_AGRICULTURE"
    REAL_ESTATE = "REAL_ESTATE"
    HEALTHCARE = "HEALTHCARE"
    SME = "SME"


# =============================================================================
# Reference Data
# =============================================================================

SECTOR_INFO: Dict[str, Dict[str, Any]] = {
    "CORPORATE_OFFICE": {
        "name": "Corporate Office",
        "typical_base_year_type": "FIXED",
        "typical_recalc_frequency": "LOW",
        "significance_threshold_pct": 5.0,
        "typical_scopes": ["SCOPE_1", "SCOPE_2_LOCATION", "SCOPE_2_MARKET"],
    },
    "MANUFACTURING": {
        "name": "Manufacturing Facility",
        "typical_base_year_type": "FIXED",
        "typical_recalc_frequency": "MEDIUM",
        "significance_threshold_pct": 5.0,
        "typical_scopes": ["SCOPE_1", "SCOPE_2_LOCATION", "SCOPE_2_MARKET", "SCOPE_3"],
    },
    "ENERGY_UTILITY": {
        "name": "Energy Utility",
        "typical_base_year_type": "FIXED",
        "typical_recalc_frequency": "HIGH",
        "significance_threshold_pct": 2.0,
        "typical_scopes": ["SCOPE_1", "SCOPE_2_LOCATION", "SCOPE_2_MARKET", "SCOPE_3"],
    },
    "TRANSPORT_LOGISTICS": {
        "name": "Transport & Logistics",
        "typical_base_year_type": "FIXED",
        "typical_recalc_frequency": "MEDIUM",
        "significance_threshold_pct": 5.0,
        "typical_scopes": ["SCOPE_1", "SCOPE_2_LOCATION", "SCOPE_2_MARKET"],
    },
    "FOOD_AGRICULTURE": {
        "name": "Food & Agriculture",
        "typical_base_year_type": "ROLLING_3YR",
        "typical_recalc_frequency": "MEDIUM",
        "significance_threshold_pct": 5.0,
        "typical_scopes": ["SCOPE_1", "SCOPE_2_LOCATION", "SCOPE_2_MARKET", "SCOPE_3"],
    },
    "REAL_ESTATE": {
        "name": "Real Estate Portfolio",
        "typical_base_year_type": "FIXED",
        "typical_recalc_frequency": "HIGH",
        "significance_threshold_pct": 5.0,
        "typical_scopes": ["SCOPE_1", "SCOPE_2_LOCATION", "SCOPE_2_MARKET"],
    },
    "HEALTHCARE": {
        "name": "Healthcare System",
        "typical_base_year_type": "FIXED",
        "typical_recalc_frequency": "MEDIUM",
        "significance_threshold_pct": 5.0,
        "typical_scopes": ["SCOPE_1", "SCOPE_2_LOCATION", "SCOPE_2_MARKET", "SCOPE_3"],
    },
    "SME": {
        "name": "Small-Medium Enterprise",
        "typical_base_year_type": "FIXED",
        "typical_recalc_frequency": "LOW",
        "significance_threshold_pct": 10.0,
        "typical_scopes": ["SCOPE_1", "SCOPE_2_LOCATION"],
    },
}

AVAILABLE_PRESETS: Dict[str, str] = {
    "corporate_office": "Office-based organisations with limited structural changes",
    "manufacturing": "Industrial manufacturing with M&A activity and process changes",
    "energy_utility": "Power generation utilities with frequent regulatory triggers",
    "transport_logistics": "Fleet operators with fleet composition changes",
    "food_agriculture": "Agricultural operations with seasonal variability (rolling base year)",
    "real_estate": "Property portfolios with acquisition/divestiture cycles",
    "healthcare": "Hospitals and healthcare systems with facility expansions",
    "sme_simplified": "Simplified base year management for small-medium enterprises",
}


# =============================================================================
# Sub-Config Models
# =============================================================================


class BaseYearSelectionConfig(BaseModel):
    """Configuration for base year selection criteria per GHG Protocol Ch 5."""
    type: BaseYearType = Field(BaseYearType.FIXED, description="Base year type strategy")
    base_year: int = Field(2022, ge=2015, le=2030, description="Selected base year")
    min_data_quality_score: float = Field(
        3.0, ge=1.0, le=5.0, description="Minimum data quality score (1-5) for base year eligibility"
    )
    min_completeness_pct: float = Field(
        90.0, ge=50.0, le=100.0, description="Minimum data completeness percentage for base year"
    )
    earliest_year: int = Field(2015, ge=2010, le=2025, description="Earliest permissible base year")
    latest_year: int = Field(2025, ge=2020, le=2030, description="Latest permissible base year")
    require_third_party_verified: bool = Field(
        False, description="Require third-party verification of base year data"
    )

    @model_validator(mode="after")
    def validate_year_range(self) -> "BaseYearSelectionConfig":
        """Ensure base year falls within the earliest/latest range."""
        if self.base_year < self.earliest_year:
            raise ValueError(
                f"base_year ({self.base_year}) cannot be earlier than earliest_year ({self.earliest_year})"
            )
        if self.base_year > self.latest_year:
            raise ValueError(
                f"base_year ({self.base_year}) cannot be later than latest_year ({self.latest_year})"
            )
        return self


class RecalculationPolicyConfig(BaseModel):
    """Recalculation policy thresholds per GHG Protocol Ch 5 and SBTi."""
    significance_threshold_pct: float = Field(
        5.0, ge=0.5, le=20.0,
        description="Individual change significance threshold (% of base year emissions)",
    )
    sbti_threshold_pct: float = Field(
        5.0, ge=1.0, le=10.0,
        description="SBTi mandatory recalculation threshold (typically 5%)",
    )
    cumulative_threshold_pct: float = Field(
        10.0, ge=1.0, le=25.0,
        description="Cumulative change threshold triggering recalculation",
    )
    auto_detect_triggers: bool = Field(True, description="Automatically detect recalculation triggers from data")
    require_approval: bool = Field(True, description="Require management approval before recalculation")
    require_documentation: bool = Field(True, description="Require written justification for recalculation decisions")


class TriggerConfig(BaseModel):
    """Configuration for structural change trigger detection."""
    enabled_triggers: List[RecalculationTriggerType] = Field(
        default_factory=lambda: [
            RecalculationTriggerType.ACQUISITION,
            RecalculationTriggerType.DIVESTITURE,
            RecalculationTriggerType.MERGER,
            RecalculationTriggerType.METHODOLOGY_CHANGE,
            RecalculationTriggerType.ERROR_CORRECTION,
            RecalculationTriggerType.SOURCE_BOUNDARY_CHANGE,
            RecalculationTriggerType.OUTSOURCING_INSOURCING,
        ],
        description="List of enabled recalculation triggers",
    )
    detection_frequency: str = Field(
        "QUARTERLY", description="How often to scan for trigger events (MONTHLY, QUARTERLY, ANNUAL)"
    )
    lookback_months: int = Field(
        12, ge=1, le=60, description="Number of months to look back for trigger events"
    )
    require_evidence: bool = Field(True, description="Require evidence documentation for trigger events")

    @field_validator("detection_frequency")
    @classmethod
    def validate_detection_frequency(cls, v: str) -> str:
        """Validate detection frequency value."""
        allowed = {"MONTHLY", "QUARTERLY", "ANNUAL"}
        if v.upper() not in allowed:
            raise ValueError(f"detection_frequency must be one of {allowed}, got '{v}'")
        return v.upper()


class SignificanceConfig(BaseModel):
    """Configuration for significance testing of structural changes."""
    method: SignificanceMethod = Field(
        SignificanceMethod.COMBINED, description="Method for significance determination"
    )
    individual_threshold_pct: float = Field(
        5.0, ge=0.5, le=20.0, description="Individual change significance threshold (%)"
    )
    cumulative_threshold_pct: float = Field(
        10.0, ge=1.0, le=25.0, description="Cumulative change significance threshold (%)"
    )
    sensitivity_range_pct: float = Field(
        2.0, ge=0.5, le=10.0, description="Sensitivity analysis range above/below threshold (%)"
    )
    require_quantitative_assessment: bool = Field(
        True, description="Require quantitative impact assessment for all triggers"
    )


class AdjustmentConfig(BaseModel):
    """Configuration for base year adjustment calculations."""
    approach: AdjustmentApproach = Field(
        AdjustmentApproach.PRO_RATA, description="Adjustment calculation approach"
    )
    pro_rata_method: str = Field(
        "CALENDAR_DAYS", description="Pro-rata allocation method (CALENDAR_DAYS, OPERATING_DAYS, REVENUE)"
    )
    propagate_to_subsidiaries: bool = Field(
        True, description="Propagate adjustments to subsidiary-level base years"
    )
    lock_after_approval: bool = Field(
        True, description="Lock adjusted base year values after management approval"
    )
    retain_original: bool = Field(
        True, description="Retain original (unadjusted) base year alongside adjusted values"
    )

    @field_validator("pro_rata_method")
    @classmethod
    def validate_pro_rata_method(cls, v: str) -> str:
        """Validate pro-rata method value."""
        allowed = {"CALENDAR_DAYS", "OPERATING_DAYS", "REVENUE"}
        if v.upper() not in allowed:
            raise ValueError(f"pro_rata_method must be one of {allowed}, got '{v}'")
        return v.upper()


class TimeSeriesConfig(BaseModel):
    """Configuration for time series analysis and trend tracking."""
    min_years: int = Field(3, ge=2, le=10, description="Minimum years of data required for trend analysis")
    max_gap_years: int = Field(1, ge=0, le=3, description="Maximum allowed gap years in time series")
    normalization_enabled: bool = Field(True, description="Normalize time series for structural changes")
    trend_window_years: int = Field(5, ge=3, le=15, description="Rolling window size for trend calculations")
    interpolation_method: str = Field(
        "LINEAR", description="Gap interpolation method (LINEAR, SPLINE, PREVIOUS_YEAR)"
    )

    @field_validator("interpolation_method")
    @classmethod
    def validate_interpolation_method(cls, v: str) -> str:
        """Validate interpolation method value."""
        allowed = {"LINEAR", "SPLINE", "PREVIOUS_YEAR"}
        if v.upper() not in allowed:
            raise ValueError(f"interpolation_method must be one of {allowed}, got '{v}'")
        return v.upper()


class TargetTrackingConfig(BaseModel):
    """Configuration for emission reduction target tracking per SBTi and GHG Protocol Ch 9."""
    target_type: TargetType = Field(TargetType.ABSOLUTE, description="Reduction target type")
    sbti_ambition: SBTiAmbitionLevel = Field(
        SBTiAmbitionLevel.ONE_POINT_FIVE_C, description="SBTi ambition level"
    )
    near_term_target_year: int = Field(2030, ge=2025, le=2040, description="Near-term target year")
    long_term_target_year: int = Field(2050, ge=2040, le=2060, description="Long-term target year")
    annual_reduction_rate_pct: float = Field(
        4.2, ge=0.5, le=15.0,
        description="Required annual linear reduction rate (%) - SBTi 1.5C requires ~4.2%",
    )
    intensity_metric: Optional[str] = Field(
        None, description="Intensity metric denominator (e.g., tCO2e/MEUR_revenue)"
    )
    track_progress_quarterly: bool = Field(True, description="Track target progress quarterly")

    @model_validator(mode="after")
    def validate_target_years(self) -> "TargetTrackingConfig":
        """Ensure near-term target year is before long-term."""
        if self.near_term_target_year >= self.long_term_target_year:
            raise ValueError(
                f"near_term_target_year ({self.near_term_target_year}) must be before "
                f"long_term_target_year ({self.long_term_target_year})"
            )
        return self


class AuditConfig(BaseModel):
    """Configuration for audit trail and third-party assurance per ISAE 3410."""
    audit_level: AuditLevel = Field(AuditLevel.LIMITED_ASSURANCE, description="Assurance engagement level")
    require_digital_signature: bool = Field(True, description="Require digital signature on approvals")
    evidence_retention_years: int = Field(7, ge=3, le=15, description="Evidence retention period in years")
    isae_3410_compliance: bool = Field(True, description="Comply with ISAE 3410 assurance standard")
    sha256_provenance: bool = Field(True, description="Enable SHA-256 provenance hashing for all calculations")
    track_all_changes: bool = Field(True, description="Log every change to base year data with full lineage")


class ReportingConfig(BaseModel):
    """Configuration for base year reporting outputs."""
    frameworks: List[ReportingFramework] = Field(
        default_factory=lambda: [ReportingFramework.GHG_PROTOCOL, ReportingFramework.CDP],
        description="Target reporting frameworks",
    )
    output_format: OutputFormat = Field(OutputFormat.HTML, description="Primary output format")
    include_methodology_notes: bool = Field(True, description="Include methodology notes in reports")
    include_adjustment_details: bool = Field(
        True, description="Include detailed adjustment history in reports"
    )
    include_significance_assessment: bool = Field(
        True, description="Include significance test results in reports"
    )
    output_language: str = Field("en", description="Report language (ISO 639-1)")


class GWPConfig(BaseModel):
    """Configuration for Global Warming Potential values."""
    version: GWPVersion = Field(GWPVersion.AR5, description="IPCC AR version for GWP values")
    include_seven_gases: bool = Field(
        True, description="Include all seven Kyoto Protocol gases (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3)"
    )
    custom_gwp_overrides: Dict[str, float] = Field(
        default_factory=dict,
        description="Custom GWP overrides by gas name (e.g., {'CH4': 28.0})",
    )

    @field_validator("custom_gwp_overrides")
    @classmethod
    def validate_gwp_overrides(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate custom GWP override values are positive."""
        for gas, gwp in v.items():
            if gwp <= 0:
                raise ValueError(f"GWP value for {gas} must be positive, got {gwp}")
        return v


class ScopeConfig(BaseModel):
    """Configuration for which GHG scopes are included in base year."""
    include_scope_1: bool = Field(True, description="Include Scope 1 direct emissions")
    include_scope_2: bool = Field(True, description="Include Scope 2 indirect emissions")
    include_scope_3: bool = Field(False, description="Include Scope 3 value chain emissions")
    scope_3_categories: List[int] = Field(
        default_factory=list,
        description="Scope 3 categories to include (1-15 per GHG Protocol)",
    )

    @field_validator("scope_3_categories")
    @classmethod
    def validate_scope_3_categories(cls, v: List[int]) -> List[int]:
        """Validate Scope 3 category numbers are within 1-15."""
        for cat in v:
            if cat < 1 or cat > 15:
                raise ValueError(f"Scope 3 category must be 1-15, got {cat}")
        return sorted(set(v))

    @model_validator(mode="after")
    def validate_scope_3_consistency(self) -> "ScopeConfig":
        """Ensure scope 3 categories are provided when scope 3 is included."""
        if self.include_scope_3 and not self.scope_3_categories:
            logger.warning("Scope 3 included but no categories specified; defaulting to categories 1-3.")
            self.scope_3_categories = [1, 2, 3]
        return self


class NotificationConfig(BaseModel):
    """Configuration for notification delivery on base year events."""
    channels: List[NotificationChannel] = Field(
        default_factory=lambda: [NotificationChannel.EMAIL],
        description="Notification delivery channels",
    )
    notify_on_trigger: bool = Field(True, description="Notify when a recalculation trigger is detected")
    notify_on_significance: bool = Field(
        True, description="Notify when a change exceeds the significance threshold"
    )
    notify_on_approval: bool = Field(True, description="Notify when a recalculation is approved/rejected")
    notify_on_completion: bool = Field(True, description="Notify when recalculation is complete")


class PerformanceConfig(BaseModel):
    """Configuration for computational performance tuning."""
    max_recalculation_time_seconds: int = Field(
        300, ge=30, le=3600, description="Maximum allowed recalculation time in seconds"
    )
    cache_base_year_data: bool = Field(True, description="Cache base year data in memory for repeated access")
    parallel_scope_processing: bool = Field(
        True, description="Process scopes in parallel during recalculation"
    )
    batch_size: int = Field(500, ge=50, le=5000, description="Batch size for bulk facility processing")
    cache_ttl_seconds: int = Field(3600, ge=60, le=86400, description="Cache TTL in seconds")


class SecurityConfig(BaseModel):
    """Configuration for access control and data protection."""
    rbac_enabled: bool = Field(True, description="Enable role-based access control")
    audit_trail_enabled: bool = Field(True, description="Enable audit trail for all operations")
    encryption_at_rest: bool = Field(True, description="Encrypt base year data at rest (AES-256)")
    roles: List[str] = Field(
        default_factory=lambda: [
            "base_year_manager", "recalculation_analyst", "reviewer",
            "approver", "verifier", "viewer", "admin",
        ],
        description="Available RBAC roles for base year management",
    )


class IntegrationConfig(BaseModel):
    """Configuration for integration with other GreenLang packs."""
    pack041_enabled: bool = Field(True, description="Integrate with PACK-041 Scope 1-2 Complete")
    pack042_enabled: bool = Field(True, description="Integrate with PACK-042 Scope 3 Starter")
    pack043_enabled: bool = Field(False, description="Integrate with PACK-043 Scope 3 Complete (Enterprise)")
    pack044_enabled: bool = Field(True, description="Integrate with PACK-044 Inventory Management")
    mrv_bridge_enabled: bool = Field(True, description="Bridge to MRV agent layer for emission factor lookup")
    erp_connector_enabled: bool = Field(False, description="Connect to ERP for structural change detection")


# =============================================================================
# Main Configuration Model
# =============================================================================


class BaseYearManagementConfig(BaseModel):
    """
    Top-level configuration for PACK-045 Base Year Management.

    Combines all sub-configurations required for base year establishment,
    recalculation policy enforcement, structural change detection,
    significance testing, adjustment calculation, target tracking,
    and audit-ready reporting.
    """
    company_name: str = Field("", description="Reporting company legal name")
    sector_type: SectorType = Field(SectorType.CORPORATE_OFFICE, description="Sector classification")
    consolidation_approach: ConsolidationApproach = Field(
        ConsolidationApproach.OPERATIONAL_CONTROL, description="Organizational boundary approach"
    )
    country: str = Field("DE", description="Primary country (ISO 3166-1 alpha-2)")
    reporting_year: int = Field(2026, ge=2020, le=2035, description="Current reporting year")
    revenue_meur: Optional[float] = Field(None, ge=0, description="Annual revenue in MEUR")
    employees_fte: Optional[int] = Field(None, ge=0, description="Full-time equivalent employees")

    base_year_selection: BaseYearSelectionConfig = Field(default_factory=BaseYearSelectionConfig)
    recalculation_policy: RecalculationPolicyConfig = Field(default_factory=RecalculationPolicyConfig)
    trigger: TriggerConfig = Field(default_factory=TriggerConfig)
    significance: SignificanceConfig = Field(default_factory=SignificanceConfig)
    adjustment: AdjustmentConfig = Field(default_factory=AdjustmentConfig)
    time_series: TimeSeriesConfig = Field(default_factory=TimeSeriesConfig)
    target_tracking: TargetTrackingConfig = Field(default_factory=TargetTrackingConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    gwp: GWPConfig = Field(default_factory=GWPConfig)
    scope: ScopeConfig = Field(default_factory=ScopeConfig)
    notification: NotificationConfig = Field(default_factory=NotificationConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    integration: IntegrationConfig = Field(default_factory=IntegrationConfig)

    @model_validator(mode="after")
    def validate_sme_simplified(self) -> "BaseYearManagementConfig":
        """Apply SME-specific simplifications to reduce configuration burden."""
        if self.sector_type == SectorType.SME:
            if self.significance.individual_threshold_pct < 10.0:
                self.significance.individual_threshold_pct = 10.0
            if self.scope.include_scope_3:
                logger.warning("SME preset: Scope 3 disabled for simplified management.")
                self.scope.include_scope_3 = False
                self.scope.scope_3_categories = []
            if self.audit.audit_level == AuditLevel.REASONABLE_ASSURANCE:
                self.audit.audit_level = AuditLevel.INTERNAL
        return self

    @model_validator(mode="after")
    def validate_threshold_consistency(self) -> "BaseYearManagementConfig":
        """Ensure significance thresholds are consistent with recalculation policy."""
        if self.significance.individual_threshold_pct > self.significance.cumulative_threshold_pct:
            raise ValueError(
                f"individual_threshold_pct ({self.significance.individual_threshold_pct}) "
                f"cannot exceed cumulative_threshold_pct ({self.significance.cumulative_threshold_pct})"
            )
        return self


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """
    Top-level wrapper for PACK-045 configuration.

    Provides factory methods for loading from presets, YAML files,
    environment overrides, and runtime merges. Includes SHA-256
    config hashing for provenance tracking.
    """
    pack: BaseYearManagementConfig = Field(default_factory=BaseYearManagementConfig)
    preset_name: Optional[str] = Field(None, description="Name of the loaded preset")
    config_version: str = Field("1.0.0", description="Configuration schema version")
    pack_id: str = Field("PACK-045-base-year", description="Unique pack identifier")

    @classmethod
    def from_preset(cls, preset_name: str, overrides: Optional[Dict[str, Any]] = None) -> "PackConfig":
        """
        Load configuration from a named sector preset.

        Args:
            preset_name: Key from AVAILABLE_PRESETS (e.g., 'manufacturing').
            overrides: Optional dict of overrides applied after preset load.

        Returns:
            Fully initialized PackConfig.

        Raises:
            ValueError: If preset_name is not recognized.
            FileNotFoundError: If preset YAML file is missing.
        """
        if preset_name not in AVAILABLE_PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. Available: {sorted(AVAILABLE_PRESETS.keys())}"
            )
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
        pack_config = BaseYearManagementConfig(**preset_data)
        return cls(pack=pack_config, preset_name=preset_name)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "PackConfig":
        """
        Load configuration from an arbitrary YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            Fully initialized PackConfig.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        pack_config = BaseYearManagementConfig(**config_data)
        return cls(pack=pack_config)

    @classmethod
    def merge(cls, base: "PackConfig", overrides: Dict[str, Any]) -> "PackConfig":
        """
        Create a new PackConfig by merging overrides into a base config.

        Args:
            base: Existing PackConfig to use as the base.
            overrides: Dict of overrides (supports nested keys).

        Returns:
            New PackConfig with merged values.
        """
        base_dict = base.pack.model_dump()
        merged = cls._deep_merge(base_dict, overrides)
        pack_config = BaseYearManagementConfig(**merged)
        return cls(pack=pack_config, preset_name=base.preset_name, config_version=base.config_version)

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """
        Load configuration overrides from environment variables.

        Environment variables prefixed with BASEYEAR_PACK_ are parsed.
        Double underscores denote nested keys.
        Example: BASEYEAR_PACK_SIGNIFICANCE__INDIVIDUAL_THRESHOLD_PCT=10.0
        """
        overrides: Dict[str, Any] = {}
        prefix = "BASEYEAR_PACK_"
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
        """Recursively merge override dict into base dict."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = PackConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get_config_hash(self) -> str:
        """
        Compute SHA-256 hash of the full configuration.

        Returns:
            Hex-encoded SHA-256 hash string for provenance tracking.
        """
        config_json = self.model_dump_json(indent=None)
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()

    def validate_completeness(self) -> List[str]:
        """
        Run domain-specific validation checks on the configuration.

        Returns:
            List of warning messages (empty list means no issues).
        """
        return validate_config(self.pack)


# =============================================================================
# Utility Functions
# =============================================================================


def load_preset(preset_name: str, overrides: Optional[Dict[str, Any]] = None) -> PackConfig:
    """
    Convenience function to load a preset configuration.

    Args:
        preset_name: Key from AVAILABLE_PRESETS.
        overrides: Optional dict of overrides.

    Returns:
        Initialized PackConfig from the named preset.
    """
    return PackConfig.from_preset(preset_name, overrides)


def validate_config(config: BaseYearManagementConfig) -> List[str]:
    """
    Validate configuration for domain-specific consistency.

    Args:
        config: The base year management configuration to validate.

    Returns:
        List of warning strings. Empty list indicates no issues found.
    """
    warnings: List[str] = []

    if not config.company_name:
        warnings.append("No company_name configured.")

    if config.base_year_selection.base_year > config.reporting_year:
        warnings.append(
            f"Base year ({config.base_year_selection.base_year}) is after "
            f"reporting year ({config.reporting_year})."
        )

    if config.scope.include_scope_3 and not config.integration.pack042_enabled:
        warnings.append(
            "Scope 3 is included but PACK-042 integration is disabled. "
            "Enable pack042_enabled for Scope 3 data."
        )

    if config.target_tracking.near_term_target_year <= config.base_year_selection.base_year:
        warnings.append(
            f"Near-term target year ({config.target_tracking.near_term_target_year}) "
            f"should be after base year ({config.base_year_selection.base_year})."
        )

    if (
        config.audit.audit_level != AuditLevel.INTERNAL
        and not config.audit.require_digital_signature
    ):
        warnings.append(
            "External assurance (limited or reasonable) should require digital signatures."
        )

    if (
        config.reporting.include_adjustment_details
        and not config.audit.track_all_changes
    ):
        warnings.append(
            "Adjustment details in reports require track_all_changes to be enabled."
        )

    if config.significance.method == SignificanceMethod.COMBINED:
        if (
            config.significance.individual_threshold_pct
            == config.significance.cumulative_threshold_pct
        ):
            warnings.append(
                "COMBINED significance method is redundant when individual and "
                "cumulative thresholds are equal."
            )

    if config.sector_type != SectorType.SME and config.significance.individual_threshold_pct > 10.0:
        warnings.append(
            f"Non-SME significance threshold ({config.significance.individual_threshold_pct}%) "
            "is unusually high. GHG Protocol recommends 5% or lower."
        )

    return warnings


def get_default_config(
    sector_type: SectorType = SectorType.CORPORATE_OFFICE,
) -> BaseYearManagementConfig:
    """
    Create a default configuration for the given sector type.

    Args:
        sector_type: Industry sector classification.

    Returns:
        Default BaseYearManagementConfig for the sector.
    """
    return BaseYearManagementConfig(sector_type=sector_type)


def list_available_presets() -> Dict[str, str]:
    """
    Return a copy of all available preset names and descriptions.

    Returns:
        Dict mapping preset name to human-readable description.
    """
    return AVAILABLE_PRESETS.copy()


def get_sector_info(sector_type: Union[str, SectorType]) -> Dict[str, Any]:
    """
    Return sector reference data for a given sector type.

    Args:
        sector_type: Sector enum value or string key.

    Returns:
        Dict of sector metadata including typical base year settings.
    """
    key = sector_type.value if isinstance(sector_type, SectorType) else sector_type
    return SECTOR_INFO.get(key, {"name": key, "typical_base_year_type": "FIXED"})
