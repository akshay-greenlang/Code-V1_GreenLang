"""
Thermodynamics Module for GL-012_SteamQual (Steam Quality Controller)

This module provides comprehensive thermodynamic calculations for steam quality
monitoring and control applications. It implements the IAPWS Industrial
Formulation 1997 (IAPWS-IF97) for water and steam properties.

Key Components:
---------------
1. iapws_wrapper: Core IAPWS-IF97 implementation with region detection
2. steam_properties: High-level property calculator and state detection
3. quality_estimation: Steam quality estimation using multiple methods
4. enthalpy_balance: Mass and energy balance calculations

Design Principles:
------------------
- DETERMINISTIC: All calculations produce identical outputs for identical inputs
- PROVENANCE: SHA-256 hashing for complete audit trails
- ZERO-HALLUCINATION: Physics-based calculations only (no LLM in calculation path)
- TYPE-SAFE: Full type hints with Pydantic-compatible dataclasses

Reference:
----------
IAPWS-IF97: International Association for the Properties of Water and Steam
            Industrial Formulation 1997 for the Thermodynamic Properties
            of Water and Steam

Author: GL-BackendDeveloper
Version: 1.0.0
"""

# =============================================================================
# IAPWS WRAPPER EXPORTS
# =============================================================================
from .iapws_wrapper import (
    # Constants
    IF97_CONSTANTS,
    REGION_BOUNDARIES,
    # Data classes
    SaturationData,
    # Unit conversions
    celsius_to_kelvin,
    kelvin_to_celsius,
    kpa_to_mpa,
    mpa_to_kpa,
    compute_density,
    # Saturation functions
    get_saturation_pressure as iapws_get_saturation_pressure,
    get_saturation_temperature as iapws_get_saturation_temperature,
    get_saturation_properties as iapws_get_saturation_properties,
    # Region 1 (Compressed Liquid)
    region1_specific_volume,
    region1_specific_enthalpy,
    region1_specific_entropy,
    region1_specific_isobaric_heat_capacity,
    # Region 2 (Superheated Vapor)
    region2_specific_volume,
    region2_specific_enthalpy,
    region2_specific_entropy,
    region2_specific_isobaric_heat_capacity,
    # Region 4 (Two-Phase)
    region4_mixture_enthalpy,
    region4_mixture_entropy,
    region4_mixture_specific_volume,
    # Region detection
    detect_region,
    # Provenance
    compute_provenance_hash,
)

# =============================================================================
# STEAM PROPERTIES EXPORTS
# =============================================================================
from .steam_properties import (
    # Enumerations
    SteamState,
    # Data classes
    SaturationPropertiesTuple,
    SteamStateResult,
    SuperheatMarginResult,
    SteamPropertiesResult,
    # Primary functions
    get_saturation_temperature,
    get_saturation_pressure,
    get_saturation_properties,
    get_saturation_properties_full,
    compute_superheat_margin,
    determine_steam_state,
    compute_steam_properties,
    # Quality calculations
    compute_quality_from_enthalpy,
    compute_quality_from_entropy,
    # Validation
    validate_pressure_temperature,
)

# =============================================================================
# QUALITY ESTIMATION EXPORTS
# =============================================================================
from .quality_estimation import (
    # Enumerations
    EstimationMethod,
    ConfidenceLevel,
    # Data classes
    QualityEstimate,
    FilterState,
    SoftSensorOutput,
    MeasurementInput,
    # Estimator classes
    PhysicsBasedEstimator,
    KalmanQualityEstimator,
    ExtendedKalmanEstimator,
    SoftSensorEstimator,
    # Confidence interval calculator
    ConfidenceIntervalCalculator,
    # Ensemble function
    estimate_quality_ensemble,
)

# =============================================================================
# ENTHALPY BALANCE EXPORTS
# =============================================================================
from .enthalpy_balance import (
    # Enumerations
    ComponentType,
    BalanceStatus,
    # Data classes
    StreamState,
    ComponentBalance,
    SystemBalance,
    QualityImpact,
    # Core functions
    compute_enthalpy_rate,
    compute_stream_enthalpy,
    estimate_specific_enthalpy,
    compute_enthalpy_change,
    # Component balance functions
    compute_component_balance,
    compute_system_balance,
    # Quality impact
    compute_quality_impact,
    analyze_quality_impacts,
    # Special component balances
    compute_flash_tank_balance,
    compute_prv_balance,
    compute_desuperheater_balance,
    compute_heat_exchanger_balance,
    # Utility functions
    create_stream_from_conditions,
)


# =============================================================================
# MODULE METADATA
# =============================================================================
__version__ = "1.0.0"
__author__ = "GL-BackendDeveloper"
__all__ = [
    # === IAPWS Wrapper ===
    "IF97_CONSTANTS",
    "REGION_BOUNDARIES",
    "SaturationData",
    "celsius_to_kelvin",
    "kelvin_to_celsius",
    "kpa_to_mpa",
    "mpa_to_kpa",
    "compute_density",
    "iapws_get_saturation_pressure",
    "iapws_get_saturation_temperature",
    "iapws_get_saturation_properties",
    "region1_specific_volume",
    "region1_specific_enthalpy",
    "region1_specific_entropy",
    "region1_specific_isobaric_heat_capacity",
    "region2_specific_volume",
    "region2_specific_enthalpy",
    "region2_specific_entropy",
    "region2_specific_isobaric_heat_capacity",
    "region4_mixture_enthalpy",
    "region4_mixture_entropy",
    "region4_mixture_specific_volume",
    "detect_region",
    "compute_provenance_hash",
    # === Steam Properties ===
    "SteamState",
    "SaturationPropertiesTuple",
    "SteamStateResult",
    "SuperheatMarginResult",
    "SteamPropertiesResult",
    "get_saturation_temperature",
    "get_saturation_pressure",
    "get_saturation_properties",
    "get_saturation_properties_full",
    "compute_superheat_margin",
    "determine_steam_state",
    "compute_steam_properties",
    "compute_quality_from_enthalpy",
    "compute_quality_from_entropy",
    "validate_pressure_temperature",
    # === Quality Estimation ===
    "EstimationMethod",
    "ConfidenceLevel",
    "QualityEstimate",
    "FilterState",
    "SoftSensorOutput",
    "MeasurementInput",
    "PhysicsBasedEstimator",
    "KalmanQualityEstimator",
    "ExtendedKalmanEstimator",
    "SoftSensorEstimator",
    "ConfidenceIntervalCalculator",
    "estimate_quality_ensemble",
    # === Enthalpy Balance ===
    "ComponentType",
    "BalanceStatus",
    "StreamState",
    "ComponentBalance",
    "SystemBalance",
    "QualityImpact",
    "compute_enthalpy_rate",
    "compute_stream_enthalpy",
    "estimate_specific_enthalpy",
    "compute_enthalpy_change",
    "compute_component_balance",
    "compute_system_balance",
    "compute_quality_impact",
    "analyze_quality_impacts",
    "compute_flash_tank_balance",
    "compute_prv_balance",
    "compute_desuperheater_balance",
    "compute_heat_exchanger_balance",
    "create_stream_from_conditions",
]
