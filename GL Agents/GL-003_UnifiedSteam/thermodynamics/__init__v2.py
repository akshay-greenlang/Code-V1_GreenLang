"""
IAPWS-IF97 Thermodynamics Engine for GL-003 UNIFIEDSTEAM SteamSystemOptimizer

This module implements the IAPWS Industrial Formulation 1997 for water and steam
properties. All calculations are deterministic with SHA-256 provenance hashing
to ensure zero-hallucination and complete audit trails.

UPGRADE v2.0.0: Full IAPWS-IF97 formulation accuracy including:
- Region 1: Full 34-term polynomial + backward equations T(p,h), T(p,s), v(p,h), v(p,s)
- Region 2: Full ideal + 43-term residual + metastable vapor extensions + backward equations
- Region 3: Complete 40-term Helmholtz energy formulation with v(p,T) iteration
- Region 4: Full saturation equations (forward and backward)
- Region 5: High-temperature steam equations (T > 800 C, P < 50 MPa)

Accuracy: <0.0001% error vs official IAPWS-IF97 verification tables (Tables 5, 7, 9, 15, 33)

Key Components:
- iapws_if97: Core IAPWS-IF97 implementation (backward compatible)
- iapws_if97_full: Full formulation with all regions and backward equations
- steam_properties: High-level property calculator
- enthalpy_balance: Mass and energy balance calculations
- steam_quality: Steam quality and dryness fraction calculations
- uncertainty: Uncertainty propagation for property calculations

Reference: IAPWS-IF97 (International Association for the Properties of Water and Steam)
           Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam

Author: GL-CalculatorEngineer
Version: 2.0.0 - Full Formulation
"""

# Import from full formulation module for upgraded accuracy
from .iapws_if97_full import (
    # Constants
    IF97_CONSTANTS,
    REGION_BOUNDARIES,
    IF97Constants,
    # Region detection
    detect_region,
    get_saturation_pressure,
    get_saturation_temperature,
    get_boundary_23_pressure,
    get_boundary_23_temperature,
    # Region 1 (Compressed Liquid) - Full 34-term
    region1_specific_volume,
    region1_specific_enthalpy,
    region1_specific_entropy,
    region1_specific_internal_energy,
    region1_specific_isobaric_heat_capacity,
    region1_specific_isochoric_heat_capacity,
    region1_speed_of_sound,
    # Region 1 Backward Equations
    region1_temperature_ph,
    region1_temperature_ps,
    region1_volume_ph,
    region1_volume_ps,
    # Region 2 (Superheated Vapor) - Full 43-term
    region2_specific_volume,
    region2_specific_enthalpy,
    region2_specific_entropy,
    region2_specific_internal_energy,
    region2_specific_isobaric_heat_capacity,
    region2_specific_isochoric_heat_capacity,
    region2_speed_of_sound,
    # Region 2 Backward Equations
    region2_temperature_ph,
    region2_temperature_ps,
    region2_volume_ph,
    region2_volume_ps,
    # Region 2 Metastable Vapor
    region2_metastable_specific_volume,
    region2_metastable_specific_enthalpy,
    # Region 3 (Supercritical) - Full 40-term Helmholtz
    region3_pressure,
    region3_specific_enthalpy,
    region3_specific_entropy,
    region3_specific_internal_energy,
    region3_specific_isobaric_heat_capacity,
    region3_specific_isochoric_heat_capacity,
    region3_speed_of_sound,
    region3_density_pt,
    region3_specific_volume_pt,
    # Region 4 (Two-Phase / Saturation)
    SaturationData,
    region4_saturation_properties,
    region4_mixture_enthalpy,
    region4_mixture_entropy,
    region4_mixture_specific_volume,
    region4_mixture_internal_energy,
    # Region 5 (High-Temperature Steam)
    region5_specific_volume,
    region5_specific_enthalpy,
    region5_specific_entropy,
    region5_specific_internal_energy,
    region5_specific_isobaric_heat_capacity,
    region5_speed_of_sound,
    # Derivatives
    compute_property_derivatives,
    # Provenance
    compute_calculation_provenance,
    # Utilities
    celsius_to_kelvin,
    kelvin_to_celsius,
    kpa_to_mpa,
    mpa_to_kpa,
    compute_density,
    # Cached versions
    cached_saturation_temperature,
    cached_saturation_pressure,
    cached_region1_properties,
    cached_region2_properties,
    clear_property_cache,
)

from .steam_properties import (
    # Data classes
    SteamProperties,
    SaturationProperties,
    SteamState,
    # Main functions
    compute_properties,
    get_saturation_properties,
    detect_steam_state,
    compute_superheat_degree,
    compute_dryness_fraction,
    validate_input_ranges,
)

from .enthalpy_balance import (
    # Data classes
    StreamData,
    MassBalanceResult,
    EnergyBalanceResult,
    LossEstimate,
    ReconciledState,
    # Functions
    compute_mass_balance,
    compute_energy_balance,
    compute_enthalpy_rate,
    estimate_distribution_losses,
    reconcile_measurements,
)

from .steam_quality import (
    # Data classes
    ValidationResult,
    InferredQuality,
    # Functions
    compute_wet_steam_enthalpy,
    validate_steam_state,
    infer_quality_from_process,
    compute_wet_steam_entropy,
    compute_wet_steam_specific_volume,
)

from .uncertainty import (
    # Data classes
    UncertaintyResult,
    MonteCarloResult,
    PropertyUncertainties,
    UncertaintyInput,
    # Functions
    propagate_uncertainty,
    monte_carlo_propagation,
    compute_property_uncertainty,
    compute_sensitivity_coefficients,
)

__version__ = "2.0.0"
__author__ = "GL-CalculatorEngineer"
__all__ = [
    # IAPWS-IF97 Core Constants
    "IF97_CONSTANTS",
    "REGION_BOUNDARIES",
    "IF97Constants",
    # Region Detection
    "detect_region",
    "get_saturation_pressure",
    "get_saturation_temperature",
    "get_boundary_23_pressure",
    "get_boundary_23_temperature",
    # Region 1 Forward Equations
    "region1_specific_volume",
    "region1_specific_enthalpy",
    "region1_specific_entropy",
    "region1_specific_internal_energy",
    "region1_specific_isobaric_heat_capacity",
    "region1_specific_isochoric_heat_capacity",
    "region1_speed_of_sound",
    # Region 1 Backward Equations
    "region1_temperature_ph",
    "region1_temperature_ps",
    "region1_volume_ph",
    "region1_volume_ps",
    # Region 2 Forward Equations
    "region2_specific_volume",
    "region2_specific_enthalpy",
    "region2_specific_entropy",
    "region2_specific_internal_energy",
    "region2_specific_isobaric_heat_capacity",
    "region2_specific_isochoric_heat_capacity",
    "region2_speed_of_sound",
    # Region 2 Backward Equations
    "region2_temperature_ph",
    "region2_temperature_ps",
    "region2_volume_ph",
    "region2_volume_ps",
    # Region 2 Metastable Vapor
    "region2_metastable_specific_volume",
    "region2_metastable_specific_enthalpy",
    # Region 3 Supercritical
    "region3_pressure",
    "region3_specific_enthalpy",
    "region3_specific_entropy",
    "region3_specific_internal_energy",
    "region3_specific_isobaric_heat_capacity",
    "region3_specific_isochoric_heat_capacity",
    "region3_speed_of_sound",
    "region3_density_pt",
    "region3_specific_volume_pt",
    # Region 4 Saturation
    "SaturationData",
    "region4_saturation_properties",
    "region4_mixture_enthalpy",
    "region4_mixture_entropy",
    "region4_mixture_specific_volume",
    "region4_mixture_internal_energy",
    # Region 5 High-Temperature
    "region5_specific_volume",
    "region5_specific_enthalpy",
    "region5_specific_entropy",
    "region5_specific_internal_energy",
    "region5_specific_isobaric_heat_capacity",
    "region5_speed_of_sound",
    # Derivatives and Provenance
    "compute_property_derivatives",
    "compute_calculation_provenance",
    # Utilities
    "celsius_to_kelvin",
    "kelvin_to_celsius",
    "kpa_to_mpa",
    "mpa_to_kpa",
    "compute_density",
    # Caching
    "cached_saturation_temperature",
    "cached_saturation_pressure",
    "cached_region1_properties",
    "cached_region2_properties",
    "clear_property_cache",
    # Steam Properties
    "SteamProperties",
    "SaturationProperties",
    "SteamState",
    "compute_properties",
    "get_saturation_properties",
    "detect_steam_state",
    "compute_superheat_degree",
    "compute_dryness_fraction",
    "validate_input_ranges",
    # Enthalpy Balance
    "StreamData",
    "MassBalanceResult",
    "EnergyBalanceResult",
    "LossEstimate",
    "ReconciledState",
    "compute_mass_balance",
    "compute_energy_balance",
    "compute_enthalpy_rate",
    "estimate_distribution_losses",
    "reconcile_measurements",
    # Steam Quality
    "ValidationResult",
    "InferredQuality",
    "compute_wet_steam_enthalpy",
    "validate_steam_state",
    "infer_quality_from_process",
    "compute_wet_steam_entropy",
    "compute_wet_steam_specific_volume",
    # Uncertainty
    "UncertaintyResult",
    "MonteCarloResult",
    "PropertyUncertainties",
    "UncertaintyInput",
    "propagate_uncertainty",
    "monte_carlo_propagation",
    "compute_property_uncertainty",
    "compute_sensitivity_coefficients",
]
