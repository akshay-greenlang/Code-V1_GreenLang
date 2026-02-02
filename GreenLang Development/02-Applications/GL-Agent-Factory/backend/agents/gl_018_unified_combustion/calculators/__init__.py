"""
Calculators for GL-018 UNIFIEDCOMBUSTION Agent

This module exports all calculator functions for unified combustion optimization.
All calculations are deterministic, following combustion engineering standards
with zero-hallucination principles.

Calculator Modules:
    - combustion: Core combustion calculations (efficiency, excess air, emissions)
    - nfpa_compliance: NFPA 85/86 compliance checking
    - optimization: O2 trim, CO optimization, excess air control

Zero-Hallucination Guarantee:
    All calculations in this package use deterministic physics-based formulas.
    NO ML/LLM is used in any calculation path, ensuring 100% reproducibility
    and audit trail compliance.

Example:
    >>> from calculators import calculate_combustion_efficiency, optimize_o2_setpoint
    >>> efficiency = calculate_combustion_efficiency(3.5, 350, "natural_gas")
    >>> optimization = optimize_o2_setpoint(3.5, 50, "natural_gas", "balanced")
"""

# Core combustion calculations
from .combustion import (
    # Fuel properties database
    FUEL_PROPERTIES,
    # Core calculations
    calculate_excess_air,
    calculate_lambda,
    calculate_air_fuel_ratio,
    calculate_combustion_efficiency,
    calculate_adiabatic_flame_temperature,
    calculate_heat_input,
    # Emissions calculations
    calculate_co2_emission_rate,
    calculate_nox_emission_rate,
    correct_emissions_to_reference_o2,
    calculate_emission_index,
    estimate_flue_gas_flow,
    # Fuel analysis
    calculate_fuel_composition_heating_value,
    calculate_stoichiometric_o2_requirement,
    # Utilities
    apply_decimal_precision,
)

# NFPA compliance checking
from .nfpa_compliance import (
    # Main compliance check
    check_nfpa_compliance,
    # Component checks
    check_required_interlocks,
    check_operating_limits,
    check_flame_detector_testing,
    check_purge_requirements,
    check_flame_response_time,
    check_interlock_certification,
    # Requirements databases
    NFPA_85_REQUIREMENTS,
    NFPA_86_REQUIREMENTS,
    REQUIRED_INTERLOCKS,
    # Utilities
    get_required_interlocks_for_equipment,
    generate_compliance_report,
)

# Optimization calculations
from .optimization import (
    # O2 optimization parameters
    O2_OPTIMIZATION_PARAMS,
    CO_CURVE_PARAMS,
    NOX_PARAMS,
    # O2 trim optimization
    optimize_o2_setpoint,
    assess_co_breakthrough_risk,
    calculate_efficiency_gain,
    generate_o2_optimization_reasoning,
    # Excess air optimization
    optimize_excess_air,
    calculate_fuel_savings,
    # CO optimization
    optimize_co_control,
    # Air-fuel ratio
    optimize_air_fuel_ratio,
    # Comprehensive optimization
    generate_optimization_recommendations,
    calculate_potential_savings,
)

__all__ = [
    # -------------------------------------------------------------------------
    # Combustion Module
    # -------------------------------------------------------------------------
    # Fuel properties
    "FUEL_PROPERTIES",
    # Core calculations
    "calculate_excess_air",
    "calculate_lambda",
    "calculate_air_fuel_ratio",
    "calculate_combustion_efficiency",
    "calculate_adiabatic_flame_temperature",
    "calculate_heat_input",
    # Emissions
    "calculate_co2_emission_rate",
    "calculate_nox_emission_rate",
    "correct_emissions_to_reference_o2",
    "calculate_emission_index",
    "estimate_flue_gas_flow",
    # Fuel analysis
    "calculate_fuel_composition_heating_value",
    "calculate_stoichiometric_o2_requirement",
    # Utilities
    "apply_decimal_precision",

    # -------------------------------------------------------------------------
    # NFPA Compliance Module
    # -------------------------------------------------------------------------
    # Main check
    "check_nfpa_compliance",
    # Component checks
    "check_required_interlocks",
    "check_operating_limits",
    "check_flame_detector_testing",
    "check_purge_requirements",
    "check_flame_response_time",
    "check_interlock_certification",
    # Requirements
    "NFPA_85_REQUIREMENTS",
    "NFPA_86_REQUIREMENTS",
    "REQUIRED_INTERLOCKS",
    # Utilities
    "get_required_interlocks_for_equipment",
    "generate_compliance_report",

    # -------------------------------------------------------------------------
    # Optimization Module
    # -------------------------------------------------------------------------
    # Parameters
    "O2_OPTIMIZATION_PARAMS",
    "CO_CURVE_PARAMS",
    "NOX_PARAMS",
    # O2 optimization
    "optimize_o2_setpoint",
    "assess_co_breakthrough_risk",
    "calculate_efficiency_gain",
    "generate_o2_optimization_reasoning",
    # Excess air
    "optimize_excess_air",
    "calculate_fuel_savings",
    # CO
    "optimize_co_control",
    # Air-fuel ratio
    "optimize_air_fuel_ratio",
    # Comprehensive
    "generate_optimization_recommendations",
    "calculate_potential_savings",
]
