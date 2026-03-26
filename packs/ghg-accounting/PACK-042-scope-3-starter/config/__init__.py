"""
PACK-042 Scope 3 Starter Pack - Configuration Module

This module exports all configuration classes, enums, constants, and utility
functions for the Scope 3 Starter Pack. Import from this module to access
the full configuration API.

Usage:
    >>> from packs.ghg_accounting.PACK_042_scope_3_starter.config import (
    ...     PackConfig,
    ...     Scope3StarterConfig,
    ...     Scope3Category,
    ...     MethodologyTier,
    ...     get_sector_info,
    ...     get_default_config,
    ... )
    >>> config = PackConfig.from_preset("manufacturing")
    >>> print(config.pack.sector_type)
    SectorType.MANUFACTURING
"""

from .pack_config import (
    # Enums
    AllocationMethod,
    ClassificationCode,
    DataQualityLevel,
    EEIOModel,
    EngagementStatus,
    FrameworkType,
    MethodologyTier,
    OutputFormat,
    ReportingFrequency,
    Scope3Category,
    SectorType,
    UncertaintyMethod,
    # Sub-config models
    AuditTrailConfig,
    CategoryConfig,
    ComplianceConfig,
    DataQualityConfig,
    DoubleCountingConfig,
    HotspotConfig,
    IntegrationConfig,
    PerformanceConfig,
    ReportingConfig,
    ScreeningConfig,
    SecurityConfig,
    SingleCategoryConfig,
    SpendClassificationConfig,
    SupplierEngagementConfig,
    UncertaintyConfig,
    # Main config models
    Scope3StarterConfig,
    PackConfig,
    # Constants
    AVAILABLE_PRESETS,
    CATEGORY_INFO,
    DEFAULT_EEIO_FACTORS,
    SECTOR_INFO,
    # Directories
    CONFIG_DIR,
    PACK_BASE_DIR,
    # Utility functions
    get_category_info,
    get_default_config,
    get_sector_info,
    list_available_presets,
    load_preset,
    validate_config,
)

__all__ = [
    # Enums
    "AllocationMethod",
    "ClassificationCode",
    "DataQualityLevel",
    "EEIOModel",
    "EngagementStatus",
    "FrameworkType",
    "MethodologyTier",
    "OutputFormat",
    "ReportingFrequency",
    "Scope3Category",
    "SectorType",
    "UncertaintyMethod",
    # Sub-config models
    "AuditTrailConfig",
    "CategoryConfig",
    "ComplianceConfig",
    "DataQualityConfig",
    "DoubleCountingConfig",
    "HotspotConfig",
    "IntegrationConfig",
    "PerformanceConfig",
    "ReportingConfig",
    "ScreeningConfig",
    "SecurityConfig",
    "SingleCategoryConfig",
    "SpendClassificationConfig",
    "SupplierEngagementConfig",
    "UncertaintyConfig",
    # Main config models
    "Scope3StarterConfig",
    "PackConfig",
    # Constants
    "AVAILABLE_PRESETS",
    "CATEGORY_INFO",
    "DEFAULT_EEIO_FACTORS",
    "SECTOR_INFO",
    # Directories
    "CONFIG_DIR",
    "PACK_BASE_DIR",
    # Utility functions
    "get_category_info",
    "get_default_config",
    "get_sector_info",
    "list_available_presets",
    "load_preset",
    "validate_config",
]
