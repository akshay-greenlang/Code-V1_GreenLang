"""
PACK-013 CSRD Manufacturing Pack - Configuration Module

This module exports all configuration classes, enums, constants, and utility
functions for the CSRD Manufacturing Pack. Import from this module to
access the full configuration API.

Usage:
    >>> from packs.eu_compliance.PACK_013_csrd_manufacturing.config import (
    ...     PackConfig,
    ...     CSRDManufacturingConfig,
    ...     ManufacturingSubSector,
    ...     ManufacturingTier,
    ...     get_subsector_info,
    ...     get_default_config,
    ... )
    >>> config = PackConfig.from_preset("heavy_industry")
    >>> print(config.pack.manufacturing_tier)
    ManufacturingTier.HEAVY_INDUSTRY
"""

from .pack_config import (
    # Enums
    AllocationMethod,
    BATComplianceLevel,
    CBAMStatus,
    ComplianceStatus,
    DisclosureFormat,
    EnergySource,
    EPRScheme,
    ESRSTopic,
    EUETSPhase,
    LifecycleScope,
    ManufacturingSubSector,
    ManufacturingTier,
    PollutantCategory,
    ReportingFrequency,
    SBTiPathway,
    WasteStreamType,
    WaterSourceType,
    # Sub-config models
    AuditTrailConfig,
    BATComplianceConfig,
    BenchmarkConfig,
    CircularEconomyConfig,
    DisclosureConfig,
    EnergyIntensityConfig,
    FacilityConfig,
    OmnibusConfig,
    ProcessEmissionsConfig,
    ProductPCFConfig,
    SupplyChainConfig,
    WaterPollutionConfig,
    # Main config models
    CSRDManufacturingConfig,
    PackConfig,
    # Constants
    AVAILABLE_PRESETS,
    PRIORITY_SCOPE3_BY_TIER,
    SUBSECTOR_INFO,
    TIER_DESCRIPTIONS,
    TIER_SUBSECTORS,
    # Directories
    CONFIG_DIR,
    PACK_BASE_DIR,
    # Utility functions
    get_default_config,
    get_subsector_info,
    get_tier_description,
    list_available_presets,
    load_preset,
    validate_config,
)

__all__ = [
    # Enums
    "AllocationMethod",
    "BATComplianceLevel",
    "CBAMStatus",
    "ComplianceStatus",
    "DisclosureFormat",
    "EnergySource",
    "EPRScheme",
    "ESRSTopic",
    "EUETSPhase",
    "LifecycleScope",
    "ManufacturingSubSector",
    "ManufacturingTier",
    "PollutantCategory",
    "ReportingFrequency",
    "SBTiPathway",
    "WasteStreamType",
    "WaterSourceType",
    # Sub-config models
    "AuditTrailConfig",
    "BATComplianceConfig",
    "BenchmarkConfig",
    "CircularEconomyConfig",
    "DisclosureConfig",
    "EnergyIntensityConfig",
    "FacilityConfig",
    "OmnibusConfig",
    "ProcessEmissionsConfig",
    "ProductPCFConfig",
    "SupplyChainConfig",
    "WaterPollutionConfig",
    # Main config models
    "CSRDManufacturingConfig",
    "PackConfig",
    # Constants
    "AVAILABLE_PRESETS",
    "PRIORITY_SCOPE3_BY_TIER",
    "SUBSECTOR_INFO",
    "TIER_DESCRIPTIONS",
    "TIER_SUBSECTORS",
    # Directories
    "CONFIG_DIR",
    "PACK_BASE_DIR",
    # Utility functions
    "get_default_config",
    "get_subsector_info",
    "get_tier_description",
    "list_available_presets",
    "load_preset",
    "validate_config",
]
