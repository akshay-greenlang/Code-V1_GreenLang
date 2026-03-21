"""
PACK-022 Net Zero Acceleration Pack - Configuration Module

This module exports all configuration classes, enums, constants, and utility
functions for the Net Zero Acceleration Pack. Import from this module to access
the full configuration API.

Usage:
    >>> from packs.net_zero.PACK_022_net_zero_acceleration.config import (
    ...     PackConfig,
    ...     NetZeroAccelerationConfig,
    ...     SectorClassification,
    ...     PathwayMethodology,
    ...     get_sector_defaults,
    ...     load_preset,
    ... )
    >>> config = PackConfig.from_preset("heavy_industry")
    >>> print(config.pack.sector)
    SectorClassification.HEAVY_INDUSTRY
"""

from .pack_config import (
    # Enums
    AssuranceLevel,
    DecompositionMethod,
    EntityScope,
    FinanceInstrument,
    PathwayMethodology,
    ScenarioType,
    ScopeCategory,
    SectorClassification,
    SupplierTier,
    TemperatureTarget,
    VCMITier,
    # Sub-config models
    AssuranceConfig,
    DecompositionConfig,
    FinanceConfig,
    MultiEntityConfig,
    PathwayConfig,
    ScenarioConfig,
    Scope3Config,
    SupplierConfig,
    TemperatureConfig,
    VCMIConfig,
    # Main config models
    NetZeroAccelerationConfig,
    PackConfig,
    # Constants
    DEFAULT_BASE_YEAR,
    DEFAULT_LONG_TERM_YEAR,
    DEFAULT_MAX_ENTITIES,
    DEFAULT_MAX_SUPPLIERS,
    DEFAULT_MONTE_CARLO_RUNS,
    DEFAULT_NEAR_TERM_YEAR,
    DEFAULT_SCOPE3_CATEGORIES,
    IPCC_AR6_GWP100,
    PRIORITY_SCOPE3_BY_SECTOR,
    SBTI_COVERAGE_THRESHOLDS,
    SBTI_REDUCTION_RATES,
    SDA_INTENSITY_METRICS,
    SDA_SECTORS,
    SECTOR_INFO,
    SECTOR_SCOPE3_PRIORITY,
    SUPPORTED_PRESETS,
    TEMPERATURE_REGRESSION,
    VCMI_TIER_THRESHOLDS,
    # Directories
    CONFIG_DIR,
    PACK_BASE_DIR,
    # Utility functions
    get_env_overrides,
    get_gwp100,
    get_sbti_reduction_rate,
    get_sda_intensity_metric,
    get_sector_defaults,
    get_sector_info,
    get_temperature_regression,
    get_vcmi_tier_thresholds,
    list_available_presets,
    list_sda_sectors,
    load_config,
    load_preset,
    merge_config,
    validate_config,
)

__all__ = [
    # Enums
    "AssuranceLevel",
    "DecompositionMethod",
    "EntityScope",
    "FinanceInstrument",
    "PathwayMethodology",
    "ScenarioType",
    "ScopeCategory",
    "SectorClassification",
    "SupplierTier",
    "TemperatureTarget",
    "VCMITier",
    # Sub-config models
    "AssuranceConfig",
    "DecompositionConfig",
    "FinanceConfig",
    "MultiEntityConfig",
    "PathwayConfig",
    "ScenarioConfig",
    "Scope3Config",
    "SupplierConfig",
    "TemperatureConfig",
    "VCMIConfig",
    # Main config models
    "NetZeroAccelerationConfig",
    "PackConfig",
    # Constants
    "DEFAULT_BASE_YEAR",
    "DEFAULT_LONG_TERM_YEAR",
    "DEFAULT_MAX_ENTITIES",
    "DEFAULT_MAX_SUPPLIERS",
    "DEFAULT_MONTE_CARLO_RUNS",
    "DEFAULT_NEAR_TERM_YEAR",
    "DEFAULT_SCOPE3_CATEGORIES",
    "IPCC_AR6_GWP100",
    "PRIORITY_SCOPE3_BY_SECTOR",
    "SBTI_COVERAGE_THRESHOLDS",
    "SBTI_REDUCTION_RATES",
    "SDA_INTENSITY_METRICS",
    "SDA_SECTORS",
    "SECTOR_INFO",
    "SECTOR_SCOPE3_PRIORITY",
    "SUPPORTED_PRESETS",
    "TEMPERATURE_REGRESSION",
    "VCMI_TIER_THRESHOLDS",
    # Directories
    "CONFIG_DIR",
    "PACK_BASE_DIR",
    # Utility functions
    "get_env_overrides",
    "get_gwp100",
    "get_sbti_reduction_rate",
    "get_sda_intensity_metric",
    "get_sector_defaults",
    "get_sector_info",
    "get_temperature_regression",
    "get_vcmi_tier_thresholds",
    "list_available_presets",
    "list_sda_sectors",
    "load_config",
    "load_preset",
    "merge_config",
    "validate_config",
]
