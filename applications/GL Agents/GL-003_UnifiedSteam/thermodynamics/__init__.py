"""
IAPWS-IF97 Thermodynamics Engine for GL-003 UNIFIEDSTEAM SteamSystemOptimizer

This module implements the IAPWS Industrial Formulation 1997 for water and steam
properties. All calculations are deterministic with SHA-256 provenance hashing
to ensure zero-hallucination and complete audit trails.

Key Components:
- iapws_if97: Core IAPWS-IF97 implementation with region detection
- steam_properties: High-level property calculator
- enthalpy_balance: Mass and energy balance calculations
- steam_quality: Steam quality and dryness fraction calculations
- uncertainty: Uncertainty propagation for property calculations

Reference: IAPWS-IF97 (International Association for the Properties of Water and Steam)
           Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from .iapws_if97 import (
    # Constants
    IF97_CONSTANTS,
    REGION_BOUNDARIES,
    # Region detection
    detect_region,
    get_saturation_pressure,
    get_saturation_temperature,
    get_boundary_23_pressure,
    get_boundary_23_temperature,
    # Region 1 (Compressed Liquid)
    region1_specific_volume,
    region1_specific_enthalpy,
    region1_specific_entropy,
    region1_specific_internal_energy,
    region1_specific_isobaric_heat_capacity,
    region1_speed_of_sound,
    # Region 2 (Superheated Vapor)
    region2_specific_volume,
    region2_specific_enthalpy,
    region2_specific_entropy,
    region2_specific_internal_energy,
    region2_specific_isobaric_heat_capacity,
    region2_speed_of_sound,
    # Region 4 (Two-Phase / Saturation)
    region4_saturation_properties,
    region4_mixture_enthalpy,
    region4_mixture_entropy,
    region4_mixture_specific_volume,
    # Derivatives
    compute_property_derivatives,
    # Provenance
    compute_calculation_provenance,
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

__version__ = "1.0.0"
__author__ = "GL-CalculatorEngineer"
__all__ = [
    # IAPWS-IF97 Core
    "IF97_CONSTANTS",
    "REGION_BOUNDARIES",
    "detect_region",
    "get_saturation_pressure",
    "get_saturation_temperature",
    "get_boundary_23_pressure",
    "get_boundary_23_temperature",
    "region1_specific_volume",
    "region1_specific_enthalpy",
    "region1_specific_entropy",
    "region1_specific_internal_energy",
    "region1_specific_isobaric_heat_capacity",
    "region1_speed_of_sound",
    "region2_specific_volume",
    "region2_specific_enthalpy",
    "region2_specific_entropy",
    "region2_specific_internal_energy",
    "region2_specific_isobaric_heat_capacity",
    "region2_speed_of_sound",
    "region4_saturation_properties",
    "region4_mixture_enthalpy",
    "region4_mixture_entropy",
    "region4_mixture_specific_volume",
    "compute_property_derivatives",
    "compute_calculation_provenance",
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
