"""
PACK-023 SBTi Alignment Pack - Configuration Module

Provides comprehensive configuration management for the SBTi Alignment Pack,
including runtime configuration models, preset loading, and validation.

Classes:
    SBTiAlignmentConfig: Main configuration container for the pack
    PackConfig: Top-level wrapper with preset/YAML loading support

Enums:
    OrganizationSector: Sector classification (POWER, HEAVY_INDUSTRY, etc.)
    BoundaryMethod: GHG Protocol boundary approach
    AmbitionLevel: SBTi target ambition level (1.5C, 2C, etc.)
    PathwayType: Target-setting pathway (ACA, SDA, FLAG)
    TemperatureRatingMethod: SBTi Temperature Rating v2.0 method
    VCMIClaimTier: VCMI offset portfolio tier
    SubmissionReadinessLevel: Submission readiness assessment level

Functions:
    load_preset(preset_name, overrides): Load a named preset configuration
    list_available_presets(): List all available presets
    validate_config(config): Validate configuration and return warnings

Example:
    >>> from packs.net_zero.PACK_023_sbti_alignment.config import (
    ...     SBTiAlignmentConfig, PackConfig, load_preset
    ... )
    >>> config = load_preset("manufacturing")
    >>> warnings = config.validate_config()
    >>> engines = config.pack.get_enabled_engines()
    >>> config_hash = config.get_config_hash()
"""

from .runtime_config import (
    SBTiAlignmentConfig,
    PackConfig,
    # Enums
    OrganizationSector,
    BoundaryMethod,
    AmbitionLevel,
    PathwayType,
    TargetTimeframe,
    DataSourceType,
    ReportFormat,
    Scope3Method,
    OffsetStrategy,
    MaturityAssessment,
    ERPType,
    OrganizationSize,
    ProgressStatus,
    RecalculationType,
    VCMIClaimTier,
    TemperatureRatingMethod,
    PCOFDataQuality,
    SubmissionReadinessLevel,
    # Sub-Config Models
    OrganizationConfig,
    BoundaryConfig,
    BaselineInventoryConfig,
    Scope3ScreeningConfig,
    TargetSettingConfig,
    SDAPathwayConfig,
    FLAGPathwayConfig,
    TemperatureRatingConfig,
    ProgressTrackingConfig,
    BaseYearRecalculationConfig,
    FINZPortfolioConfig,
    SubmissionReadinessConfig,
    OffsetPortfolioConfig,
    ReportingConfig,
    PerformanceConfig,
    AuditTrailConfig,
    # Utility Functions
    load_preset,
    list_available_presets,
    validate_config,
    merge_config,
    get_env_overrides,
    # Constants
    SUPPORTED_PRESETS,
    DEFAULT_BASE_YEAR,
    DEFAULT_REPORTING_YEAR,
    DEFAULT_NEAR_TERM_YEAR,
    DEFAULT_LONG_TERM_YEAR,
    DEFAULT_SCOPE3_CATEGORIES,
    SBTI_NEAR_TERM_CRITERIA,
    SBTI_NET_ZERO_CRITERIA,
    TEMPERATURE_RATING_METHODS,
    SDA_SECTORS,
    FLAG_COMMODITIES,
    FINZ_ASSET_CLASSES,
    SCOPE3_CATEGORIES_INFO,
)

__all__ = [
    # Main Config Classes
    "SBTiAlignmentConfig",
    "PackConfig",
    # Enums
    "OrganizationSector",
    "BoundaryMethod",
    "AmbitionLevel",
    "PathwayType",
    "TargetTimeframe",
    "DataSourceType",
    "ReportFormat",
    "Scope3Method",
    "OffsetStrategy",
    "MaturityAssessment",
    "ERPType",
    "OrganizationSize",
    "ProgressStatus",
    "RecalculationType",
    "VCMIClaimTier",
    "TemperatureRatingMethod",
    "PCOFDataQuality",
    "SubmissionReadinessLevel",
    # Sub-Config Models
    "OrganizationConfig",
    "BoundaryConfig",
    "BaselineInventoryConfig",
    "Scope3ScreeningConfig",
    "TargetSettingConfig",
    "SDAPathwayConfig",
    "FLAGPathwayConfig",
    "TemperatureRatingConfig",
    "ProgressTrackingConfig",
    "BaseYearRecalculationConfig",
    "FINZPortfolioConfig",
    "SubmissionReadinessConfig",
    "OffsetPortfolioConfig",
    "ReportingConfig",
    "PerformanceConfig",
    "AuditTrailConfig",
    # Functions
    "load_preset",
    "list_available_presets",
    "validate_config",
    "merge_config",
    "get_env_overrides",
    # Constants
    "SUPPORTED_PRESETS",
    "DEFAULT_BASE_YEAR",
    "DEFAULT_REPORTING_YEAR",
    "DEFAULT_NEAR_TERM_YEAR",
    "DEFAULT_LONG_TERM_YEAR",
    "DEFAULT_SCOPE3_CATEGORIES",
    "SBTI_NEAR_TERM_CRITERIA",
    "SBTI_NET_ZERO_CRITERIA",
    "TEMPERATURE_RATING_METHODS",
    "SDA_SECTORS",
    "FLAG_COMMODITIES",
    "FINZ_ASSET_CLASSES",
    "SCOPE3_CATEGORIES_INFO",
]

__version__ = "1.0.0"
__author__ = "GreenLang Platform Team"
