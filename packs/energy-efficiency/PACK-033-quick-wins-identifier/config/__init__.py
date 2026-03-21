"""
PACK-033 Quick Wins Identifier Pack - Configuration Module

This module exports all configuration classes, enums, constants, and utility
functions for the Quick Wins Identifier Pack. Import from this module to
access the full configuration API.

Usage:
    >>> from packs.energy_efficiency.PACK_033_quick_wins_identifier.config import (
    ...     PackConfig,
    ...     QuickWinsConfig,
    ...     FacilityType,
    ...     ScanDepth,
    ...     get_facility_info,
    ...     get_default_config,
    ... )
    >>> config = PackConfig.from_preset("office_building")
    >>> print(config.pack.facility_type)
    FacilityType.OFFICE
"""

from .pack_config import (
    # Enums
    ActionCategory,
    FacilityType,
    ImplementationComplexity,
    MeasureStatus,
    OutputFormat,
    PriorityProfile,
    ReportingFrequency,
    ScanDepth,
    # Sub-config models
    AuditTrailConfig,
    BehavioralConfig,
    CarbonConfig,
    FinancialConfig,
    PerformanceConfig,
    PrioritizationConfig,
    RebateConfig,
    ReportingConfig,
    ScanConfig,
    SecurityConfig,
    # Main config models
    QuickWinsConfig,
    PackConfig,
    # Constants
    AVAILABLE_PRESETS,
    DEFAULT_FINANCIAL_PARAMS,
    FACILITY_TYPE_INFO,
    # Directories
    CONFIG_DIR,
    PACK_BASE_DIR,
    # Utility functions
    get_default_config,
    get_facility_info,
    list_available_presets,
    load_preset,
    validate_config,
)

__all__ = [
    # Enums
    "ActionCategory",
    "FacilityType",
    "ImplementationComplexity",
    "MeasureStatus",
    "OutputFormat",
    "PriorityProfile",
    "ReportingFrequency",
    "ScanDepth",
    # Sub-config models
    "AuditTrailConfig",
    "BehavioralConfig",
    "CarbonConfig",
    "FinancialConfig",
    "PerformanceConfig",
    "PrioritizationConfig",
    "RebateConfig",
    "ReportingConfig",
    "ScanConfig",
    "SecurityConfig",
    # Main config models
    "QuickWinsConfig",
    "PackConfig",
    # Constants
    "AVAILABLE_PRESETS",
    "DEFAULT_FINANCIAL_PARAMS",
    "FACILITY_TYPE_INFO",
    # Directories
    "CONFIG_DIR",
    "PACK_BASE_DIR",
    # Utility functions
    "get_default_config",
    "get_facility_info",
    "list_available_presets",
    "load_preset",
    "validate_config",
]
