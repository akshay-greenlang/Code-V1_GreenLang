"""
PACK-015 Double Materiality Assessment Pack - Configuration Module

This module exports all configuration classes, enums, constants, and utility
functions for the Double Materiality Assessment Pack. Import from this module
to access the full configuration API.

Usage:
    >>> from packs.eu_compliance.PACK_015_double_materiality.config import (
    ...     PackConfig,
    ...     DMAConfig,
    ...     SectorType,
    ...     CompanySize,
    ...     ESRSTopic,
    ...     get_sector_info,
    ...     get_default_config,
    ... )
    >>> config = PackConfig.from_preset("large_enterprise")
    >>> print(config.pack.sectors)
    [SectorType.GENERAL]
"""

from .pack_config import (
    # Enums
    CompanySize,
    DisclosureFormat,
    ESRSSubTopic,
    ESRSTopic,
    IRODirection,
    IROTemporality,
    IROType,
    MaterialityLevel,
    ReportingFrequency,
    ScoringMethodology,
    SectorType,
    StakeholderCategory,
    TimeHorizon,
    ValueChainPosition,
    # Sub-config models
    AuditTrailConfig,
    ESRSMappingConfig,
    FinancialMaterialityConfig,
    ImpactMaterialityConfig,
    IROConfig,
    MatrixConfig,
    ReportingConfig,
    StakeholderConfig,
    ThresholdConfig,
    # Main config models
    DMAConfig,
    PackConfig,
    # Constants
    AVAILABLE_PRESETS,
    ESRS_TOPIC_INFO,
    ESRS_SUBTOPIC_MAP,
    SECTOR_MATERIALITY_PROFILES,
    SEVERITY_DIMENSION_WEIGHTS,
    FINANCIAL_MAGNITUDE_THRESHOLDS,
    # Directories
    CONFIG_DIR,
    PACK_BASE_DIR,
    # Utility functions
    get_default_config,
    get_sector_info,
    get_esrs_topic_info,
    get_subtopics_for_topic,
    list_available_presets,
    load_preset,
    validate_config,
)

__all__ = [
    # Enums
    "CompanySize",
    "DisclosureFormat",
    "ESRSSubTopic",
    "ESRSTopic",
    "IRODirection",
    "IROTemporality",
    "IROType",
    "MaterialityLevel",
    "ReportingFrequency",
    "ScoringMethodology",
    "SectorType",
    "StakeholderCategory",
    "TimeHorizon",
    "ValueChainPosition",
    # Sub-config models
    "AuditTrailConfig",
    "ESRSMappingConfig",
    "FinancialMaterialityConfig",
    "ImpactMaterialityConfig",
    "IROConfig",
    "MatrixConfig",
    "ReportingConfig",
    "StakeholderConfig",
    "ThresholdConfig",
    # Main config models
    "DMAConfig",
    "PackConfig",
    # Constants
    "AVAILABLE_PRESETS",
    "ESRS_TOPIC_INFO",
    "ESRS_SUBTOPIC_MAP",
    "SECTOR_MATERIALITY_PROFILES",
    "SEVERITY_DIMENSION_WEIGHTS",
    "FINANCIAL_MAGNITUDE_THRESHOLDS",
    # Directories
    "CONFIG_DIR",
    "PACK_BASE_DIR",
    # Utility functions
    "get_default_config",
    "get_sector_info",
    "get_esrs_topic_info",
    "get_subtopics_for_topic",
    "list_available_presets",
    "load_preset",
    "validate_config",
]
