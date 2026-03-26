"""
PACK-043 Scope 3 Complete Pack - Configuration Module

This module exports all configuration classes, enums, constants, and utility
functions for the Scope 3 Complete Pack. Import from this module to access
the full configuration API.

Usage:
    >>> from packs.ghg_accounting.PACK_043_scope_3_complete.config import (
    ...     PackConfig,
    ...     Scope3CompleteConfig,
    ...     MaturityLevel,
    ...     SBTiTargetType,
    ...     get_maturity_info,
    ...     get_default_config,
    ... )
    >>> config = PackConfig.from_preset("enterprise_manufacturing")
    >>> print(config.pack.maturity.target_level)
    MaturityLevel.LEVEL_4_ADVANCED
"""

from .pack_config import (
    # Enums
    AssuranceLevel,
    ConsolidationApproach,
    FrameworkType,
    LCADatabase,
    LifecycleStage,
    MaturityLevel,
    MethodologyTier,
    OutputFormat,
    PCAFAssetClass,
    RiskType,
    SBTiTargetType,
    SBTiTimeframe,
    ScenarioType,
    Scope3Category,
    SectorFocus,
    TierUpgradeStrategy,
    # Sub-config models
    AssuranceConfig,
    BaseYearConfig,
    BoundaryConfig,
    ClimateRiskConfig,
    IntegrationConfig,
    LCAConfig,
    MaturityConfig,
    PerformanceConfig,
    ReportingConfig,
    SBTiConfig,
    ScenarioConfig,
    SectorConfig,
    SecurityConfig,
    SupplierProgrammeConfig,
    # Main config models
    Scope3CompleteConfig,
    PackConfig,
    # Constants
    AVAILABLE_PRESETS,
    MATURITY_LEVEL_INFO,
    # Directories
    CONFIG_DIR,
    PACK_BASE_DIR,
    # Utility functions
    get_default_config,
    get_maturity_info,
    list_available_presets,
    load_preset,
    validate_config,
)

__all__ = [
    # Enums
    "AssuranceLevel",
    "ConsolidationApproach",
    "FrameworkType",
    "LCADatabase",
    "LifecycleStage",
    "MaturityLevel",
    "MethodologyTier",
    "OutputFormat",
    "PCAFAssetClass",
    "RiskType",
    "SBTiTargetType",
    "SBTiTimeframe",
    "ScenarioType",
    "Scope3Category",
    "SectorFocus",
    "TierUpgradeStrategy",
    # Sub-config models
    "AssuranceConfig",
    "BaseYearConfig",
    "BoundaryConfig",
    "ClimateRiskConfig",
    "IntegrationConfig",
    "LCAConfig",
    "MaturityConfig",
    "PerformanceConfig",
    "ReportingConfig",
    "SBTiConfig",
    "ScenarioConfig",
    "SectorConfig",
    "SecurityConfig",
    "SupplierProgrammeConfig",
    # Main config models
    "Scope3CompleteConfig",
    "PackConfig",
    # Constants
    "AVAILABLE_PRESETS",
    "MATURITY_LEVEL_INFO",
    # Directories
    "CONFIG_DIR",
    "PACK_BASE_DIR",
    # Utility functions
    "get_default_config",
    "get_maturity_info",
    "list_available_presets",
    "load_preset",
    "validate_config",
]
