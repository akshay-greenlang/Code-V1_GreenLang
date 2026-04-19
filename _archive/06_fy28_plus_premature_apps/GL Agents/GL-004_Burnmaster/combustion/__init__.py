"""
GL-004 BURNMASTER Combustion Module

Combustion calculations including stoichiometry, thermodynamics,
air-fuel ratios, fuel properties, and derived features for optimization.

This module provides zero-hallucination, deterministic calculations
with complete provenance tracking for regulatory compliance.

Author: GreenLang AI Agent Workforce
Version: 1.1.0

Modules:
- fuel_properties: Fuel database and property calculations
- stoichiometry: Air-fuel ratio and combustion stoichiometry
- thermodynamics: Heat capacity, enthalpy, efficiency calculations
- derived_variables: Feature engineering for ML models
- efficiency: Comprehensive combustion efficiency calculations
- flame_temperature: Adiabatic and actual flame temperature
- flue_gas_losses: Detailed flue gas heat loss breakdown
- excess_air: Excess air calculation and optimization
"""

# =============================================================================
# Fuel Properties Module
# =============================================================================
from .fuel_properties import (
    # Enums and Classes
    FuelType,
    FuelComposition,
    FuelProperties,
    FuelQuality,
    # Constants
    MOLECULAR_WEIGHTS as FUEL_MOLECULAR_WEIGHTS,
    HHV_NM3,
    LHV_NM3,
    STOICH_O2,
    CO2_PRODUCED,
    H2O_PRODUCED,
    ADIABATIC_FLAME_TEMP,
    LAMINAR_FLAME_SPEED,
    STANDARD_COMPOSITIONS,
    # Functions
    compute_molecular_weight,
    compute_heating_values,
    compute_specific_gravity,
    compute_wobbe_index,
    compute_stoichiometric_afr_from_composition,
    compute_co2_emission_factor,
    compute_fuel_properties,
    get_fuel_properties,
    estimate_fuel_quality_from_o2_co,
    validate_fuel_composition,
    validate_fuel_properties,
)

# =============================================================================
# Stoichiometry Module
# =============================================================================
from .stoichiometry import (
    # Classes
    StoichiometryResult,
    # Constants
    FUEL_O2_CONSTANTS,
    MAX_DRY_O2_PERCENT,
    # Core Functions
    compute_stoichiometric_air,
    compute_lambda,
    compute_excess_air_percent,
    compute_excess_o2,
    infer_lambda_from_o2,
    # Advanced Functions
    compute_stoichiometry_from_fuel_type,
    compute_air_flow_for_target_o2,
    compute_fuel_flow_for_target_duty,
    compute_flue_gas_flow,
    # Validation Functions
    validate_stoichiometry_inputs,
    check_stoichiometry_consistency,
)

# =============================================================================
# Thermodynamics Module
# =============================================================================
from .thermodynamics import (
    # Enums and Classes
    EfficiencyMethod,
    GasProperties,
    HeatBalanceResult,
    EfficiencyResult,
    # Constants
    SHOMATE_COEFFICIENTS,
    MOLECULAR_WEIGHTS as THERMO_MOLECULAR_WEIGHTS,
    HEATING_VALUES,
    R_UNIVERSAL,
    # Shomate Equation Functions
    compute_cp_shomate,
    compute_enthalpy_shomate,
    compute_entropy_shomate,
    # Flue Gas Functions
    compute_flue_gas_enthalpy,
    compute_flue_gas_cp,
    # Loss Calculations
    compute_stack_loss,
    compute_radiation_loss,
    compute_unburned_loss,
    # Efficiency Calculations
    compute_efficiency_indirect,
    compute_efficiency_direct,
    compute_heat_rate,
    compute_fuel_intensity,
    compute_heat_balance,
)

# =============================================================================
# Derived Variables Module
# =============================================================================
from .derived_variables import (
    # Classes
    StabilityFeatures,
    ConstraintMargins,
    # Load and Turndown
    compute_normalized_load,
    compute_turndown_ratio,
    compute_load_factor,
    # Constraint Margins
    compute_constraint_margins,
    # Stability Features
    compute_stability_features,
    # Additional Features
    compute_efficiency_deviation,
    compute_emission_intensity,
    compute_heat_rate_deviation,
    compute_operating_envelope_distance,
    compute_trend_features,
    compute_cross_correlation_lag,
    # Validation
    validate_operating_point,
    validate_signal_quality,
)

# =============================================================================
# Efficiency Module (Zero-Hallucination Calculator)
# =============================================================================
from .efficiency import (
    # Classes
    EfficiencyInput,
    EfficiencyResult as DetailedEfficiencyResult,
    LossBreakdownResult,
    CombustionEfficiencyCalculator,
    # Constants
    SIEGERT_COEFFICIENTS,
    FLUE_GAS_CP,
    LATENT_HEAT_WATER,
)

# =============================================================================
# Flame Temperature Module (Zero-Hallucination Calculator)
# =============================================================================
from .flame_temperature import (
    # Enums and Classes
    FlameType,
    FlameTemperatureInput,
    FlameTemperatureResult,
    FlameTemperatureCalculator,
    # Constants
    STOICHIOMETRIC_FLAME_TEMPS,
    HEATING_VALUES as FLAME_HEATING_VALUES,
    STOICH_AFR as FLAME_STOICH_AFR,
)

# =============================================================================
# Flue Gas Losses Module (Zero-Hallucination Calculator)
# =============================================================================
from .flue_gas_losses import (
    # Enums and Classes
    FlueGasComponent,
    FlueGasComposition,
    FlueGasLossResult,
    FlueGasLossCalculator,
    # Constants
    SPECIFIC_HEATS,
    MOLECULAR_WEIGHTS as FLUE_GAS_MOLECULAR_WEIGHTS,
    LATENT_HEAT_H2O,
    HEAT_OF_COMBUSTION_CO,
)

# =============================================================================
# Excess Air Module (Zero-Hallucination Calculator)
# =============================================================================
from .excess_air import (
    # Enums and Classes
    MeasurementBasis,
    ExcessAirInput,
    ExcessAirResult,
    ExcessAirCalculator,
    # Constants
    STOICHIOMETRIC_AFR,
    MAX_CO2,
    OPTIMAL_EXCESS_AIR,
)

# =============================================================================
# Aliases for Backward Compatibility
# =============================================================================
compute_stoichiometric_ratio = compute_stoichiometric_air
compute_excess_air = compute_excess_air_percent
compute_lambda_value = compute_lambda
compute_combustion_efficiency = compute_efficiency_direct
compute_flue_gas_losses = compute_stack_loss

# =============================================================================
# __all__ Export List
# =============================================================================
__all__ = [
    # Fuel Properties
    "FuelType",
    "FuelComposition",
    "FuelProperties",
    "FuelQuality",
    "FUEL_MOLECULAR_WEIGHTS",
    "HHV_NM3",
    "LHV_NM3",
    "STOICH_O2",
    "CO2_PRODUCED",
    "H2O_PRODUCED",
    "ADIABATIC_FLAME_TEMP",
    "LAMINAR_FLAME_SPEED",
    "STANDARD_COMPOSITIONS",
    "compute_molecular_weight",
    "compute_heating_values",
    "compute_specific_gravity",
    "compute_wobbe_index",
    "compute_stoichiometric_afr_from_composition",
    "compute_co2_emission_factor",
    "compute_fuel_properties",
    "get_fuel_properties",
    "estimate_fuel_quality_from_o2_co",
    "validate_fuel_composition",
    "validate_fuel_properties",
    # Stoichiometry
    "StoichiometryResult",
    "FUEL_O2_CONSTANTS",
    "MAX_DRY_O2_PERCENT",
    "compute_stoichiometric_air",
    "compute_stoichiometric_ratio",
    "compute_lambda",
    "compute_lambda_value",
    "compute_excess_air_percent",
    "compute_excess_air",
    "compute_excess_o2",
    "infer_lambda_from_o2",
    "compute_stoichiometry_from_fuel_type",
    "compute_air_flow_for_target_o2",
    "compute_fuel_flow_for_target_duty",
    "compute_flue_gas_flow",
    "validate_stoichiometry_inputs",
    "check_stoichiometry_consistency",
    # Thermodynamics
    "EfficiencyMethod",
    "GasProperties",
    "HeatBalanceResult",
    "EfficiencyResult",
    "SHOMATE_COEFFICIENTS",
    "THERMO_MOLECULAR_WEIGHTS",
    "HEATING_VALUES",
    "R_UNIVERSAL",
    "compute_cp_shomate",
    "compute_enthalpy_shomate",
    "compute_entropy_shomate",
    "compute_flue_gas_enthalpy",
    "compute_flue_gas_cp",
    "compute_stack_loss",
    "compute_flue_gas_losses",
    "compute_radiation_loss",
    "compute_unburned_loss",
    "compute_efficiency_indirect",
    "compute_efficiency_direct",
    "compute_combustion_efficiency",
    "compute_heat_rate",
    "compute_fuel_intensity",
    "compute_heat_balance",
    # Derived Variables
    "StabilityFeatures",
    "ConstraintMargins",
    "compute_normalized_load",
    "compute_turndown_ratio",
    "compute_load_factor",
    "compute_constraint_margins",
    "compute_stability_features",
    "compute_efficiency_deviation",
    "compute_emission_intensity",
    "compute_heat_rate_deviation",
    "compute_operating_envelope_distance",
    "compute_trend_features",
    "compute_cross_correlation_lag",
    "validate_operating_point",
    "validate_signal_quality",
    # Efficiency Module
    "EfficiencyInput",
    "DetailedEfficiencyResult",
    "LossBreakdownResult",
    "CombustionEfficiencyCalculator",
    "SIEGERT_COEFFICIENTS",
    "FLUE_GAS_CP",
    "LATENT_HEAT_WATER",
    # Flame Temperature Module
    "FlameType",
    "FlameTemperatureInput",
    "FlameTemperatureResult",
    "FlameTemperatureCalculator",
    "STOICHIOMETRIC_FLAME_TEMPS",
    "FLAME_HEATING_VALUES",
    "FLAME_STOICH_AFR",
    # Flue Gas Losses Module
    "FlueGasComponent",
    "FlueGasComposition",
    "FlueGasLossResult",
    "FlueGasLossCalculator",
    "SPECIFIC_HEATS",
    "FLUE_GAS_MOLECULAR_WEIGHTS",
    "LATENT_HEAT_H2O",
    "HEAT_OF_COMBUSTION_CO",
    # Excess Air Module
    "MeasurementBasis",
    "ExcessAirInput",
    "ExcessAirResult",
    "ExcessAirCalculator",
    "STOICHIOMETRIC_AFR",
    "MAX_CO2",
    "OPTIMAL_EXCESS_AIR",
]
