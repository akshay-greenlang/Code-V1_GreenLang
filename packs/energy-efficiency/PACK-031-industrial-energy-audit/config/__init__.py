"""
PACK-031 Industrial Energy Audit Pack - Configuration Module

This module exports all configuration classes, enums, constants, and utility
functions for the Industrial Energy Audit Pack. Import from this module to
access the full configuration API.

Usage:
    >>> from packs.energy_efficiency.PACK_031_industrial_energy_audit.config import (
    ...     PackConfig,
    ...     IndustrialEnergyAuditConfig,
    ...     IndustrySector,
    ...     FacilityTier,
    ...     get_sector_info,
    ...     get_default_config,
    ... )
    >>> config = PackConfig.from_preset("manufacturing_plant")
    >>> print(config.pack.industry_sector)
    IndustrySector.MANUFACTURING
"""

from .pack_config import (
    # Enums
    AuditLevel,
    ComplianceStatus,
    EnPIType,
    EnergyCarrier,
    FacilityTier,
    IndustrySector,
    MotorEfficiencyClass,
    NormalizationMethod,
    OutputFormat,
    ReportingFrequency,
    # Sub-config models
    AuditConfig,
    AuditTrailConfig,
    BaselineConfig,
    BenchmarkConfig,
    CompressedAirConfig,
    EEDConfig,
    EquipmentConfig,
    HVACConfig,
    ISO50001Config,
    LightingConfig,
    PerformanceConfig,
    ReportingConfig,
    SecurityConfig,
    SteamConfig,
    WasteHeatConfig,
    # Main config models
    IndustrialEnergyAuditConfig,
    PackConfig,
    # Constants
    AVAILABLE_PRESETS,
    COMPRESSED_AIR_BENCHMARKS,
    LPD_STANDARDS,
    MOTOR_EFFICIENCY_REQUIREMENTS,
    PUE_BENCHMARKS,
    SECTOR_INFO,
    STEAM_BENCHMARKS,
    # Directories
    CONFIG_DIR,
    PACK_BASE_DIR,
    # Utility functions
    get_compressed_air_benchmark,
    get_default_config,
    get_lpd_standard,
    get_pue_benchmark,
    get_sector_info,
    list_available_presets,
    load_preset,
    validate_config,
)

__all__ = [
    # Enums
    "AuditLevel",
    "ComplianceStatus",
    "EnPIType",
    "EnergyCarrier",
    "FacilityTier",
    "IndustrySector",
    "MotorEfficiencyClass",
    "NormalizationMethod",
    "OutputFormat",
    "ReportingFrequency",
    # Sub-config models
    "AuditConfig",
    "AuditTrailConfig",
    "BaselineConfig",
    "BenchmarkConfig",
    "CompressedAirConfig",
    "EEDConfig",
    "EquipmentConfig",
    "HVACConfig",
    "ISO50001Config",
    "LightingConfig",
    "PerformanceConfig",
    "ReportingConfig",
    "SecurityConfig",
    "SteamConfig",
    "WasteHeatConfig",
    # Main config models
    "IndustrialEnergyAuditConfig",
    "PackConfig",
    # Constants
    "AVAILABLE_PRESETS",
    "COMPRESSED_AIR_BENCHMARKS",
    "LPD_STANDARDS",
    "MOTOR_EFFICIENCY_REQUIREMENTS",
    "PUE_BENCHMARKS",
    "SECTOR_INFO",
    "STEAM_BENCHMARKS",
    # Directories
    "CONFIG_DIR",
    "PACK_BASE_DIR",
    # Utility functions
    "get_compressed_air_benchmark",
    "get_default_config",
    "get_lpd_standard",
    "get_pue_benchmark",
    "get_sector_info",
    "list_available_presets",
    "load_preset",
    "validate_config",
]
