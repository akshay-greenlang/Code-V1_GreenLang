"""
PACK-035 Energy Benchmark Pack - Configuration Module

This module exports all configuration classes, enums, constants, and utility
functions for the Energy Benchmark Pack. Import from this module to access
the full configuration API.

Usage:
    >>> from packs.energy_efficiency.PACK_035_energy_benchmark.config import (
    ...     PackConfig,
    ...     EnergyBenchmarkConfig,
    ...     BuildingType,
    ...     ClimateZone,
    ...     get_building_type_info,
    ...     get_default_config,
    ... )
    >>> config = PackConfig.from_preset("commercial_office")
    >>> print(config.config.building_type)
    BuildingType.OFFICE
"""

from .pack_config import (
    # Enums
    AggregationMethod,
    AlertChannel,
    BenchmarkSource,
    BuildingType,
    ClimateZone,
    DataFrequency,
    EnergyCarrier,
    FloorAreaType,
    NormalisationMethod,
    RatingSystem,
    ReportFormat,
    # Sub-config models
    AuditTrailConfig,
    EUIConfig,
    GapAnalysisConfig,
    PerformanceConfig,
    PeerComparisonConfig,
    PortfolioConfig,
    RatingConfig,
    ReportConfig,
    SecurityConfig,
    TrendConfig,
    WeatherConfig,
    # Main config models
    EnergyBenchmarkConfig,
    PackConfig,
    # Constants
    AVAILABLE_PRESETS,
    BUILDING_TYPE_DEFAULTS,
    CIBSE_TM46_BENCHMARKS,
    CLIMATE_ZONE_ADJUSTMENTS,
    ENERGY_STAR_MEDIAN_EUI,
    EPC_GRADE_THRESHOLDS,
    PRIMARY_ENERGY_FACTORS,
    SOURCE_ENERGY_FACTORS,
    # Directories
    CONFIG_DIR,
    PACK_BASE_DIR,
    # Utility functions
    get_building_type_info,
    get_cibse_tm46_benchmark,
    get_climate_zone_adjustment,
    get_default_config,
    get_epc_grade,
    get_primary_energy_factor,
    get_source_energy_factor,
    list_available_presets,
    load_preset,
    validate_config,
)

__all__ = [
    # Enums
    "AggregationMethod",
    "AlertChannel",
    "BenchmarkSource",
    "BuildingType",
    "ClimateZone",
    "DataFrequency",
    "EnergyCarrier",
    "FloorAreaType",
    "NormalisationMethod",
    "RatingSystem",
    "ReportFormat",
    # Sub-config models
    "AuditTrailConfig",
    "EUIConfig",
    "GapAnalysisConfig",
    "PerformanceConfig",
    "PeerComparisonConfig",
    "PortfolioConfig",
    "RatingConfig",
    "ReportConfig",
    "SecurityConfig",
    "TrendConfig",
    "WeatherConfig",
    # Main config models
    "EnergyBenchmarkConfig",
    "PackConfig",
    # Constants
    "AVAILABLE_PRESETS",
    "BUILDING_TYPE_DEFAULTS",
    "CIBSE_TM46_BENCHMARKS",
    "CLIMATE_ZONE_ADJUSTMENTS",
    "ENERGY_STAR_MEDIAN_EUI",
    "EPC_GRADE_THRESHOLDS",
    "PRIMARY_ENERGY_FACTORS",
    "SOURCE_ENERGY_FACTORS",
    # Directories
    "CONFIG_DIR",
    "PACK_BASE_DIR",
    # Utility functions
    "get_building_type_info",
    "get_cibse_tm46_benchmark",
    "get_climate_zone_adjustment",
    "get_default_config",
    "get_epc_grade",
    "get_primary_energy_factor",
    "get_source_energy_factor",
    "list_available_presets",
    "load_preset",
    "validate_config",
]
