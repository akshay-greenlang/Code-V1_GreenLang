"""
PACK-032 Building Energy Assessment Pack - Configuration Module

This module exports all configuration classes, enums, constants, and utility
functions for the Building Energy Assessment Pack. Import from this module to
access the full configuration API.

Usage:
    >>> from packs.energy_efficiency.PACK_032_building_energy_assessment.config import (
    ...     PackConfig,
    ...     BuildingEnergyAssessmentConfig,
    ...     BuildingType,
    ...     ClimateZone,
    ...     get_building_type_info,
    ...     get_default_config,
    ... )
    >>> config = PackConfig.from_preset("commercial_office")
    >>> print(config.pack.building_type)
    BuildingType.OFFICE
"""

from .pack_config import (
    # Enums
    AssessmentLevel,
    BMSProtocol,
    BuildingAge,
    BuildingType,
    CertificationTarget,
    ClimateZone,
    EPCMethodology,
    HeatingFuel,
    IEQCategory,
    LightingControlType,
    OccupancyPattern,
    OutputFormat,
    OwnershipType,
    RetrofitAmbition,
    ThermalBridgeMethod,
    ThermalComfortMethod,
    VentilationType,
    # Sub-config models
    AuditTrailConfig,
    BenchmarkConfig,
    CarbonConfig,
    ComplianceConfig,
    DHWConfig,
    EnvelopeConfig,
    HVACConfig,
    IndoorEnvironmentConfig,
    IntegrationConfig,
    LightingConfig,
    PerformanceConfig,
    RenewableConfig,
    ReportConfig,
    RetrofitConfig,
    SecurityConfig,
    # Main config models
    BuildingEnergyAssessmentConfig,
    PackConfig,
    # Constants
    AIRTIGHTNESS_BENCHMARKS,
    AVAILABLE_PRESETS,
    BREEAM_ENERGY_CREDITS,
    BUILDING_TYPE_INFO,
    COUNTRY_CLIMATE_ZONE,
    EPC_RATING_THRESHOLDS,
    LEED_ENERGY_THRESHOLDS,
    LPD_BUILDING_STANDARDS,
    U_VALUE_TARGETS,
    # Directories
    CONFIG_DIR,
    PACK_BASE_DIR,
    # Utility functions
    get_airtightness_benchmark,
    get_building_type_info,
    get_climate_zone_for_country,
    get_default_config,
    get_lpd_standard,
    get_u_value_targets,
    list_available_presets,
    load_preset,
    validate_config,
)

__all__ = [
    # Enums
    "AssessmentLevel",
    "BMSProtocol",
    "BuildingAge",
    "BuildingType",
    "CertificationTarget",
    "ClimateZone",
    "EPCMethodology",
    "HeatingFuel",
    "IEQCategory",
    "LightingControlType",
    "OccupancyPattern",
    "OutputFormat",
    "OwnershipType",
    "RetrofitAmbition",
    "ThermalBridgeMethod",
    "ThermalComfortMethod",
    "VentilationType",
    # Sub-config models
    "AuditTrailConfig",
    "BenchmarkConfig",
    "CarbonConfig",
    "ComplianceConfig",
    "DHWConfig",
    "EnvelopeConfig",
    "HVACConfig",
    "IndoorEnvironmentConfig",
    "IntegrationConfig",
    "LightingConfig",
    "PerformanceConfig",
    "RenewableConfig",
    "ReportConfig",
    "RetrofitConfig",
    "SecurityConfig",
    # Main config models
    "BuildingEnergyAssessmentConfig",
    "PackConfig",
    # Constants
    "AIRTIGHTNESS_BENCHMARKS",
    "AVAILABLE_PRESETS",
    "BREEAM_ENERGY_CREDITS",
    "BUILDING_TYPE_INFO",
    "COUNTRY_CLIMATE_ZONE",
    "EPC_RATING_THRESHOLDS",
    "LEED_ENERGY_THRESHOLDS",
    "LPD_BUILDING_STANDARDS",
    "U_VALUE_TARGETS",
    # Directories
    "CONFIG_DIR",
    "PACK_BASE_DIR",
    # Utility functions
    "get_airtightness_benchmark",
    "get_building_type_info",
    "get_climate_zone_for_country",
    "get_default_config",
    "get_lpd_standard",
    "get_u_value_targets",
    "list_available_presets",
    "load_preset",
    "validate_config",
]
