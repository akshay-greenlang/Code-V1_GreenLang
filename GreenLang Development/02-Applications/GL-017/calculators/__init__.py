"""
GL-017 CONDENSYNC - Calculators Module

Zero-hallucination, deterministic calculation engines for condenser
optimization in steam systems. All calculations are fully auditable
with SHA-256 provenance tracking.

This module provides comprehensive calculators for:
- Heat Transfer Analysis (U-value, LMTD, heat duty)
- Vacuum System Optimization (pressure, air in-leakage)
- Efficiency Analysis (CPI, thermal efficiency, cost-benefit)
- Fouling Assessment (fouling factor, cleaning optimization)
- Condenser Performance (HEI standards, TTD, CF trending)
- Fouling Prediction (LSI/RSI indices, biofouling, scale formation)

Standards Reference:
- HEI Standards for Steam Surface Condensers (11th Edition)
- ASME PTC 12.2 - Steam Surface Condensers Performance Test Code
- TEMA (Tubular Exchanger Manufacturers Association) Standards
- EPRI Heat Rate Improvement Guidelines
- Kern-Seaton Asymptotic Fouling Model

Guarantees:
- DETERMINISTIC: Same input always produces same output
- REPRODUCIBLE: SHA-256 verified calculation chain
- AUDITABLE: Complete step-by-step provenance trail
- ZERO HALLUCINATION: No LLM in calculation path

Example:
    >>> from calculators import HeatTransferCalculator, HeatTransferInput
    >>> calculator = HeatTransferCalculator()
    >>> inputs = HeatTransferInput(
    ...     steam_temp_c=40.0,
    ...     cw_inlet_temp_c=25.0,
    ...     cw_outlet_temp_c=35.0,
    ...     cw_flow_rate_kg_s=1000.0,
    ...     heat_transfer_area_m2=5000.0,
    ...     tube_od_mm=25.4,
    ...     tube_id_mm=22.9,
    ...     tube_length_m=10.0,
    ...     tube_material="admiralty_brass",
    ...     num_tubes=3000,
    ...     num_passes=2,
    ...     design_u_value_w_m2k=3500.0
    ... )
    >>> result, provenance = calculator.calculate(inputs)
    >>> print(f"Heat Duty: {result.heat_duty_mw:.2f} MW")

Author: GL-CalculatorEngineer
Version: 1.1.0
"""

__version__ = "1.1.0"
__author__ = "GL-CalculatorEngineer"
__standards__ = [
    "HEI Standards for Steam Surface Condensers",
    "ASME PTC 12.2",
    "TEMA Standards",
    "EPRI Guidelines"
]

# =============================================================================
# PROVENANCE MODULE EXPORTS
# =============================================================================

from .provenance import (
    # Classes
    ProvenanceTracker,
    ProvenanceRecord,
    CalculationStep,
    # Functions
    compute_input_fingerprint,
    compute_output_fingerprint,
    verify_provenance,
    get_utc_timestamp,
    format_provenance_report,
)

# =============================================================================
# HEAT TRANSFER CALCULATOR EXPORTS
# =============================================================================

from .heat_transfer_calculator import (
    # Main calculator class
    HeatTransferCalculator,
    # Input/Output dataclasses
    HeatTransferInput,
    HeatTransferOutput,
    # Enums
    UnitSystem,
    # Constants
    TUBE_THERMAL_CONDUCTIVITY,
    TUBE_WALL_THICKNESS,
    WATER_PROPERTIES,
    HEI_TUBE_COUNT_FACTOR,
    # Standalone functions
    calculate_lmtd,
    calculate_heat_duty,
    calculate_ntu,
)

# =============================================================================
# VACUUM CALCULATOR EXPORTS
# =============================================================================

from .vacuum_calculator import (
    # Main calculator class
    VacuumCalculator,
    # Input/Output dataclasses
    VacuumInput,
    VacuumOutput,
    # Enums
    VacuumUnit,
    # Constants
    ATMOSPHERIC_PRESSURE_MBAR,
    ATMOSPHERIC_PRESSURE_MMHG,
    ATMOSPHERIC_PRESSURE_KPA,
    ATMOSPHERIC_PRESSURE_PSIA,
    STEAM_SATURATION_PRESSURE,
    HEI_AIR_LEAKAGE_RATES,
    TURBINE_BACKPRESSURE_IMPACT,
    # Standalone functions
    convert_pressure_units,
    calculate_saturation_temperature,
    calculate_air_density_at_vacuum,
    calculate_sjae_steam_consumption,
    calculate_vacuum_pump_power,
)

# =============================================================================
# EFFICIENCY CALCULATOR EXPORTS
# =============================================================================

from .efficiency_calculator import (
    # Main calculator class
    EfficiencyCalculator,
    # Input/Output dataclasses
    EfficiencyInput,
    EfficiencyOutput,
    # Enums
    PerformanceRating,
    # Constants
    HEAT_RATE_IMPACT_KJ_KWH_PER_MMHG,
    CPI_REFERENCE_VALUES,
    DEFAULT_ELECTRICITY_COST_USD_MWH,
    CW_PUMP_POWER_KW_PER_1000M3HR,
    CARBON_EMISSION_FACTOR_KG_CO2_MWH,
    # Standalone functions
    calculate_cw_temperature_rise,
    calculate_optimal_cw_flow,
    calculate_cw_pumping_power,
    calculate_payback_period,
    calculate_npv,
)

# =============================================================================
# FOULING CALCULATOR EXPORTS
# =============================================================================

from .fouling_calculator import (
    # Main calculator class
    FoulingCalculator,
    # Input/Output dataclasses
    FoulingInput,
    FoulingOutput,
    # Enums
    WaterType,
    FoulingType,
    BiofoulingRisk,
    # Constants
    TEMA_FOULING_FACTORS,
    FOULING_RATE_CONSTANTS,
    BIOFOULING_RISK_THRESHOLDS,
    CLEANING_EFFECTIVENESS,
    # Standalone functions
    calculate_asymptotic_fouling,
    calculate_cleaning_benefit,
    estimate_cleaning_cost,
)

# =============================================================================
# CONDENSER PERFORMANCE CALCULATOR EXPORTS
# =============================================================================

from .condenser_performance_calculator import (
    # Main calculator class
    CondenserPerformanceCalculator,
    # Input/Output dataclasses
    CondenserPerformanceInput,
    CondenserPerformanceOutput,
    # Enums
    PerformanceStatus,
    VacuumUnit as CondVacuumUnit,
    # Constants
    HEI_BASE_U_VALUES,
    HEI_MATERIAL_FACTORS,
    HEI_TEMP_CORRECTION,
    HEI_VELOCITY_CORRECTION,
    HEI_CLEANLINESS_THRESHOLDS,
    HEAT_RATE_CORRECTION_PER_MMHG,
    STEAM_SATURATION,
    # Standalone functions
    calculate_hei_u_value,
    calculate_condenser_duty,
    calculate_vacuum_from_steam_temp,
    calculate_steam_temp_from_vacuum,
)

# =============================================================================
# FOULING PREDICTOR CALCULATOR EXPORTS
# =============================================================================

from .fouling_predictor import (
    # Main calculator class
    FoulingPredictor,
    # Input/Output dataclasses
    FoulingPredictorInput,
    FoulingPredictorOutput,
    # Enums
    FoulingMechanism,
    WaterQuality,
    CleaningUrgency,
    # Constants
    FOULING_MODEL_PARAMS,
    BIOFOULING_TEMP_FACTORS,
    BIOFOULING_SEASONAL_FACTORS,
    SCALE_FORMATION,
    CLEANING_COSTS,
    CLEANING_EFFECTIVENESS as PREDICTOR_CLEANING_EFFECTIVENESS,
    # Standalone functions
    calculate_langelier_index,
    calculate_ryznar_index,
    estimate_fouling_rate_from_cf_history,
    predict_cleaning_benefit,
)

# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

def get_all_calculators() -> dict:
    """
    Get dictionary of all available calculator classes.

    Returns:
        Dictionary mapping calculator names to classes
    """
    return {
        "heat_transfer": HeatTransferCalculator,
        "vacuum": VacuumCalculator,
        "efficiency": EfficiencyCalculator,
        "fouling": FoulingCalculator,
        "condenser_performance": CondenserPerformanceCalculator,
        "fouling_predictor": FoulingPredictor,
    }


def get_calculator(name: str):
    """
    Get calculator class by name.

    Args:
        name: Calculator name (heat_transfer, vacuum, efficiency, fouling)

    Returns:
        Calculator class

    Raises:
        ValueError: If calculator name not found
    """
    calculators = get_all_calculators()
    if name not in calculators:
        valid_names = list(calculators.keys())
        raise ValueError(
            f"Unknown calculator: {name}. Valid options: {valid_names}"
        )
    return calculators[name]


def validate_provenance(provenance: ProvenanceRecord) -> dict:
    """
    Validate a provenance record and return detailed results.

    Args:
        provenance: ProvenanceRecord to validate

    Returns:
        Dictionary with validation results
    """
    is_valid = verify_provenance(provenance)

    return {
        "is_valid": is_valid,
        "calculation_id": provenance.calculation_id,
        "calculator": provenance.calculator_name,
        "version": provenance.calculator_version,
        "timestamp": provenance.timestamp_utc,
        "input_hash": provenance.input_hash,
        "output_hash": provenance.output_hash,
        "provenance_hash": provenance.provenance_hash,
        "num_steps": len(provenance.calculation_steps),
    }


# =============================================================================
# PUBLIC API DEFINITION
# =============================================================================

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__standards__",

    # Provenance
    "ProvenanceTracker",
    "ProvenanceRecord",
    "CalculationStep",
    "compute_input_fingerprint",
    "compute_output_fingerprint",
    "verify_provenance",
    "get_utc_timestamp",
    "format_provenance_report",

    # Heat Transfer
    "HeatTransferCalculator",
    "HeatTransferInput",
    "HeatTransferOutput",
    "UnitSystem",
    "TUBE_THERMAL_CONDUCTIVITY",
    "TUBE_WALL_THICKNESS",
    "WATER_PROPERTIES",
    "HEI_TUBE_COUNT_FACTOR",
    "calculate_lmtd",
    "calculate_heat_duty",
    "calculate_ntu",

    # Vacuum
    "VacuumCalculator",
    "VacuumInput",
    "VacuumOutput",
    "VacuumUnit",
    "ATMOSPHERIC_PRESSURE_MBAR",
    "ATMOSPHERIC_PRESSURE_MMHG",
    "ATMOSPHERIC_PRESSURE_KPA",
    "ATMOSPHERIC_PRESSURE_PSIA",
    "STEAM_SATURATION_PRESSURE",
    "HEI_AIR_LEAKAGE_RATES",
    "TURBINE_BACKPRESSURE_IMPACT",
    "convert_pressure_units",
    "calculate_saturation_temperature",
    "calculate_air_density_at_vacuum",
    "calculate_sjae_steam_consumption",
    "calculate_vacuum_pump_power",

    # Efficiency
    "EfficiencyCalculator",
    "EfficiencyInput",
    "EfficiencyOutput",
    "PerformanceRating",
    "HEAT_RATE_IMPACT_KJ_KWH_PER_MMHG",
    "CPI_REFERENCE_VALUES",
    "DEFAULT_ELECTRICITY_COST_USD_MWH",
    "CW_PUMP_POWER_KW_PER_1000M3HR",
    "CARBON_EMISSION_FACTOR_KG_CO2_MWH",
    "calculate_cw_temperature_rise",
    "calculate_optimal_cw_flow",
    "calculate_cw_pumping_power",
    "calculate_payback_period",
    "calculate_npv",

    # Fouling
    "FoulingCalculator",
    "FoulingInput",
    "FoulingOutput",
    "WaterType",
    "FoulingType",
    "BiofoulingRisk",
    "TEMA_FOULING_FACTORS",
    "FOULING_RATE_CONSTANTS",
    "BIOFOULING_RISK_THRESHOLDS",
    "CLEANING_EFFECTIVENESS",
    "calculate_asymptotic_fouling",
    "calculate_cleaning_benefit",
    "estimate_cleaning_cost",

    # Condenser Performance (HEI Standards)
    "CondenserPerformanceCalculator",
    "CondenserPerformanceInput",
    "CondenserPerformanceOutput",
    "PerformanceStatus",
    "CondVacuumUnit",
    "HEI_BASE_U_VALUES",
    "HEI_MATERIAL_FACTORS",
    "HEI_TEMP_CORRECTION",
    "HEI_VELOCITY_CORRECTION",
    "HEI_CLEANLINESS_THRESHOLDS",
    "HEAT_RATE_CORRECTION_PER_MMHG",
    "STEAM_SATURATION",
    "calculate_hei_u_value",
    "calculate_condenser_duty",
    "calculate_vacuum_from_steam_temp",
    "calculate_steam_temp_from_vacuum",

    # Fouling Predictor (LSI/RSI, Biofouling, Scale)
    "FoulingPredictor",
    "FoulingPredictorInput",
    "FoulingPredictorOutput",
    "FoulingMechanism",
    "WaterQuality",
    "CleaningUrgency",
    "FOULING_MODEL_PARAMS",
    "BIOFOULING_TEMP_FACTORS",
    "BIOFOULING_SEASONAL_FACTORS",
    "SCALE_FORMATION",
    "CLEANING_COSTS",
    "PREDICTOR_CLEANING_EFFECTIVENESS",
    "calculate_langelier_index",
    "calculate_ryznar_index",
    "estimate_fouling_rate_from_cf_history",
    "predict_cleaning_benefit",

    # Convenience functions
    "get_all_calculators",
    "get_calculator",
    "validate_provenance",
]
