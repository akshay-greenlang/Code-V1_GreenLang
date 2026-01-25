"""
GL-018 FLUEFLOW - Combustion Analysis Calculators

Zero-hallucination, deterministic calculations for flue gas analysis
and combustion optimization following ASME PTC 4.1 and EPA standards.

Available Calculators:
- FlueGasCompositionCalculator: Complete combustion stoichiometry, wet/dry conversions,
  dew points, molecular weights, JANAF thermochemical data
- CombustionEfficiencyCalculator: Siegert formula, heat loss method, ASME PTC 4.1
- CombustionAnalyzer: Flue gas analysis and excess air calculations
- EfficiencyCalculator: Basic combustion efficiency analysis
- AirFuelRatioCalculator: Theoretical air and lambda calculations
- EmissionsCalculator: NOx, CO, SO2 concentration conversions

Author: GL-CalculatorEngineer
Version: 1.1.0
"""

from .provenance import (
    ProvenanceTracker,
    ProvenanceRecord,
    CalculationStep,
    compute_input_fingerprint,
    compute_output_fingerprint,
    verify_provenance,
    format_provenance_report
)

# Flue Gas Composition Calculator (NEW)
from .flue_gas_composition_calculator import (
    FlueGasCompositionCalculator,
    FlueGasCompositionInput,
    FlueGasCompositionOutput,
    StoichiometryResult,
    FuelComposition,
    JANAFData,
    FUEL_COMPOSITIONS,
    # Standalone functions
    calculate_excess_air_from_o2,
    calculate_excess_air_from_co2,
    convert_wet_to_dry,
    convert_dry_to_wet,
    calculate_water_dew_point,
    calculate_molecular_weight,
    get_specific_heat,
)

# Combustion Efficiency Calculator (NEW)
from .combustion_efficiency_calculator import (
    CombustionEfficiencyCalculator,
    CombustionEfficiencyInput,
    CombustionEfficiencyOutput,
    HeatLossBreakdown,
    FuelProperties,
    SiegertConstants,
    FUEL_PROPERTIES_DB,
    # Standalone functions
    calculate_stack_loss_siegert,
    calculate_efficiency_from_losses,
    calculate_moisture_loss,
    calculate_co_loss,
    estimate_efficiency_quick,
    get_siegert_k_factor,
)

__all__ = [
    # Provenance tracking
    "ProvenanceTracker",
    "ProvenanceRecord",
    "CalculationStep",
    "compute_input_fingerprint",
    "compute_output_fingerprint",
    "verify_provenance",
    "format_provenance_report",

    # Flue Gas Composition Calculator
    "FlueGasCompositionCalculator",
    "FlueGasCompositionInput",
    "FlueGasCompositionOutput",
    "StoichiometryResult",
    "FuelComposition",
    "JANAFData",
    "FUEL_COMPOSITIONS",
    "calculate_excess_air_from_o2",
    "calculate_excess_air_from_co2",
    "convert_wet_to_dry",
    "convert_dry_to_wet",
    "calculate_water_dew_point",
    "calculate_molecular_weight",
    "get_specific_heat",

    # Combustion Efficiency Calculator
    "CombustionEfficiencyCalculator",
    "CombustionEfficiencyInput",
    "CombustionEfficiencyOutput",
    "HeatLossBreakdown",
    "FuelProperties",
    "SiegertConstants",
    "FUEL_PROPERTIES_DB",
    "calculate_stack_loss_siegert",
    "calculate_efficiency_from_losses",
    "calculate_moisture_loss",
    "calculate_co_loss",
    "estimate_efficiency_quick",
    "get_siegert_k_factor",
]

__version__ = "1.1.0"
