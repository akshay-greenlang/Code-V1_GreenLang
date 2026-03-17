"""
PACK-016 ESRS E1 Climate Change Pack - Configuration Module

This module exports all configuration classes, enums, constants, and utility
functions for the ESRS E1 Climate Change Pack. Import from this module
to access the full configuration API.

Usage:
    >>> from packs.eu_compliance.PACK_016_esrs_e1_climate.config import (
    ...     PackConfig,
    ...     E1ClimateConfig,
    ...     GHGScope,
    ...     EmissionGas,
    ...     TargetPathway,
    ...     get_default_config,
    ...     get_gwp_value,
    ... )
    >>> config = PackConfig.from_preset("manufacturing")
    >>> print(config.pack.ghg.consolidation_approach)
    OPERATIONAL_CONTROL
"""

from .pack_config import (
    # Enums
    BaseYearApproach,
    CarbonCreditStandard,
    CarbonCreditType,
    CarbonPricingMethod,
    ClimateScenario,
    ConsolidationApproach,
    DisclosureFormat,
    DisclosureStatus,
    EmissionGas,
    EnergySource,
    FuelType,
    GHGScope,
    PhysicalRiskType,
    RenewableCategory,
    Scope3Method,
    TargetPathway,
    TargetType,
    TimeHorizon,
    TransitionRiskType,
    # Sub-config models
    CarbonCreditConfig,
    CarbonPricingConfig,
    ClimateRiskConfig,
    EnergyConfig,
    GHGConfig,
    ReportingConfig,
    TargetConfig,
    TransitionPlanConfig,
    # Main config models
    E1ClimateConfig,
    PackConfig,
    # Constants
    AVAILABLE_PRESETS,
    E1_DISCLOSURE_REQUIREMENTS,
    EMISSION_FACTOR_SOURCES,
    ENERGY_SOURCE_CLASSIFICATION,
    GWP_AR6_VALUES,
    SBTI_REDUCTION_RATES,
    SCOPE_3_CATEGORIES,
    # Directories
    CONFIG_DIR,
    PACK_BASE_DIR,
    # Utility functions
    get_default_config,
    get_e1_disclosure_info,
    get_gwp_value,
    get_sbti_reduction_rate,
    get_scope_3_category_name,
    list_available_presets,
    load_preset,
    validate_config,
)

__all__ = [
    # Enums
    "BaseYearApproach",
    "CarbonCreditStandard",
    "CarbonCreditType",
    "CarbonPricingMethod",
    "ClimateScenario",
    "ConsolidationApproach",
    "DisclosureFormat",
    "DisclosureStatus",
    "EmissionGas",
    "EnergySource",
    "FuelType",
    "GHGScope",
    "PhysicalRiskType",
    "RenewableCategory",
    "Scope3Method",
    "TargetPathway",
    "TargetType",
    "TimeHorizon",
    "TransitionRiskType",
    # Sub-config models
    "CarbonCreditConfig",
    "CarbonPricingConfig",
    "ClimateRiskConfig",
    "EnergyConfig",
    "GHGConfig",
    "ReportingConfig",
    "TargetConfig",
    "TransitionPlanConfig",
    # Main config models
    "E1ClimateConfig",
    "PackConfig",
    # Constants
    "AVAILABLE_PRESETS",
    "E1_DISCLOSURE_REQUIREMENTS",
    "EMISSION_FACTOR_SOURCES",
    "ENERGY_SOURCE_CLASSIFICATION",
    "GWP_AR6_VALUES",
    "SBTI_REDUCTION_RATES",
    "SCOPE_3_CATEGORIES",
    # Directories
    "CONFIG_DIR",
    "PACK_BASE_DIR",
    # Utility functions
    "get_default_config",
    "get_e1_disclosure_info",
    "get_gwp_value",
    "get_sbti_reduction_rate",
    "get_scope_3_category_name",
    "list_available_presets",
    "load_preset",
    "validate_config",
]
