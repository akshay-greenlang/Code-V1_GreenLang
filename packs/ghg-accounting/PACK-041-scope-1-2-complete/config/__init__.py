"""
PACK-041 Scope 1-2 Complete Pack - Configuration Module

This module exports all configuration classes, enums, constants, and utility
functions for the Scope 1-2 Complete Pack. Import from this module to
access the full configuration API.

Usage:
    >>> from packs.ghg_accounting.PACK_041_scope_1_2_complete.config import (
    ...     PackConfig,
    ...     Scope12CompleteConfig,
    ...     ConsolidationApproach,
    ...     GWPSource,
    ...     get_sector_info,
    ...     get_default_config,
    ... )
    >>> config = PackConfig.from_preset("corporate_office")
    >>> print(config.pack.sector_type)
    SectorType.OFFICE
"""

from .pack_config import (
    # Enums
    ConsolidationApproach,
    EmissionFactorSource,
    FrameworkType,
    GasType,
    GWPSource,
    InventoryScope,
    MethodologyTier,
    OutputFormat,
    ReportingFrequency,
    SectorType,
    # Sub-config models
    AuditTrailConfig,
    BaseYearConfig,
    BoundaryConfig,
    ComplianceConfig,
    EmissionFactorConfig,
    PerformanceConfig,
    ReportingConfig,
    Scope1Config,
    Scope2Config,
    SecurityConfig,
    SourceCompletenessConfig,
    TrendAnalysisConfig,
    UncertaintyConfig,
    # Main config models
    Scope12CompleteConfig,
    PackConfig,
    # Constants
    AVAILABLE_PRESETS,
    GWP_VALUES,
    SECTOR_INFO,
    # Directories
    CONFIG_DIR,
    PACK_BASE_DIR,
    # Utility functions
    get_default_config,
    get_sector_info,
    list_available_presets,
    load_preset,
    validate_config,
)

__all__ = [
    # Enums
    "ConsolidationApproach",
    "EmissionFactorSource",
    "FrameworkType",
    "GasType",
    "GWPSource",
    "InventoryScope",
    "MethodologyTier",
    "OutputFormat",
    "ReportingFrequency",
    "SectorType",
    # Sub-config models
    "AuditTrailConfig",
    "BaseYearConfig",
    "BoundaryConfig",
    "ComplianceConfig",
    "EmissionFactorConfig",
    "PerformanceConfig",
    "ReportingConfig",
    "Scope1Config",
    "Scope2Config",
    "SecurityConfig",
    "SourceCompletenessConfig",
    "TrendAnalysisConfig",
    "UncertaintyConfig",
    # Main config models
    "Scope12CompleteConfig",
    "PackConfig",
    # Constants
    "AVAILABLE_PRESETS",
    "GWP_VALUES",
    "SECTOR_INFO",
    # Directories
    "CONFIG_DIR",
    "PACK_BASE_DIR",
    # Utility functions
    "get_default_config",
    "get_sector_info",
    "list_available_presets",
    "load_preset",
    "validate_config",
]
