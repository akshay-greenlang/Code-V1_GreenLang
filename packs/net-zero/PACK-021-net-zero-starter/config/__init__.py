"""
PACK-021 Net Zero Starter Pack - Configuration Module

This module exports all configuration classes, enums, constants, and utility
functions for the Net Zero Starter Pack. Import from this module to access the
full configuration API.

Usage:
    >>> from packs.net_zero.PACK_021_net_zero_starter.config import (
    ...     PackConfig,
    ...     NetZeroStarterConfig,
    ...     OrganizationSector,
    ...     AmbitionLevel,
    ...     get_sector_info,
    ...     get_default_config,
    ... )
    >>> config = PackConfig.from_preset("manufacturing")
    >>> print(config.pack.organization.sector)
    OrganizationSector.MANUFACTURING
"""

from .pack_config import (
    # Enums
    AmbitionLevel,
    BoundaryMethod,
    DataSourceType,
    ERPType,
    MaturityAssessment,
    OffsetStrategy,
    OrganizationSector,
    OrganizationSize,
    PathwayType,
    ReportFormat,
    Scope3Method,
    TargetTimeframe,
    # Sub-config models
    AuditTrailConfig,
    BoundaryConfig,
    OffsetConfig,
    OrganizationConfig,
    PerformanceConfig,
    ReductionConfig,
    ReportingConfig,
    ScopeConfig,
    ScorecardConfig,
    TargetConfig,
    # Main config models
    NetZeroStarterConfig,
    PackConfig,
    # Constants
    DEFAULT_BASE_YEAR,
    DEFAULT_SCOPE3_CATEGORIES,
    DEFAULT_TARGET_YEAR_LONG,
    DEFAULT_TARGET_YEAR_NEAR,
    IPCC_AR6_GWP100,
    PRIORITY_SCOPE3_BY_SECTOR,
    SBTI_COVERAGE_THRESHOLDS,
    SBTI_REDUCTION_RATES,
    SECTOR_INFO,
    SECTOR_SCOPE3_PRIORITY,
    SUPPORTED_PRESETS,
    # Directories
    CONFIG_DIR,
    PACK_BASE_DIR,
    # Utility functions
    get_default_config,
    get_env_overrides,
    get_gwp100,
    get_sbti_reduction_rate,
    get_sector_info,
    list_available_presets,
    load_preset,
    merge_config,
    validate_config,
)

__all__ = [
    # Enums
    "AmbitionLevel",
    "BoundaryMethod",
    "DataSourceType",
    "ERPType",
    "MaturityAssessment",
    "OffsetStrategy",
    "OrganizationSector",
    "OrganizationSize",
    "PathwayType",
    "ReportFormat",
    "Scope3Method",
    "TargetTimeframe",
    # Sub-config models
    "AuditTrailConfig",
    "BoundaryConfig",
    "OffsetConfig",
    "OrganizationConfig",
    "PerformanceConfig",
    "ReductionConfig",
    "ReportingConfig",
    "ScopeConfig",
    "ScorecardConfig",
    "TargetConfig",
    # Main config models
    "NetZeroStarterConfig",
    "PackConfig",
    # Constants
    "DEFAULT_BASE_YEAR",
    "DEFAULT_SCOPE3_CATEGORIES",
    "DEFAULT_TARGET_YEAR_LONG",
    "DEFAULT_TARGET_YEAR_NEAR",
    "IPCC_AR6_GWP100",
    "PRIORITY_SCOPE3_BY_SECTOR",
    "SBTI_COVERAGE_THRESHOLDS",
    "SBTI_REDUCTION_RATES",
    "SECTOR_INFO",
    "SECTOR_SCOPE3_PRIORITY",
    "SUPPORTED_PRESETS",
    # Directories
    "CONFIG_DIR",
    "PACK_BASE_DIR",
    # Utility functions
    "get_default_config",
    "get_env_overrides",
    "get_gwp100",
    "get_sbti_reduction_rate",
    "get_sector_info",
    "list_available_presets",
    "load_preset",
    "merge_config",
    "validate_config",
]
