"""
Calculators for GL-016 WATERGUARD Agent (BoilerWaterTreatmentAgent)

This module exports all calculator functions for boiler water treatment
optimization. All calculations are deterministic, following ASME/ABMA
standards with zero-hallucination principles.

Calculator Modules:
    - chemistry: Water chemistry analysis and compliance checking
    - blowdown: Blowdown rate optimization and savings calculations
    - dosing: Chemical dosing calculations and recommendations

Standards Reference:
    - ASME Boiler and Pressure Vessel Code Section VII
    - ABMA Guidelines for Industrial Water Treatment
    - EPRI Water Chemistry Guidelines
"""

from .chemistry import (
    # Compliance and limits
    determine_pressure_class,
    get_asme_water_limits,
    check_parameter_compliance,
    check_chemistry_compliance,
    # Cycles of concentration
    calculate_cycles_of_concentration,
    calculate_max_cycles_by_silica,
    calculate_max_cycles_by_alkalinity,
    calculate_max_cycles_by_conductivity,
    calculate_optimal_cycles,
    # Utility calculations
    calculate_tds_from_conductivity,
    analyze_chemistry_trends,
    estimate_silica_solubility,
    calculate_caustic_concentration,
)

from .blowdown import (
    # Blowdown rate calculations
    calculate_blowdown_rate_from_cycles,
    calculate_cycles_from_blowdown,
    calculate_makeup_water_flow,
    # Heat and water loss
    calculate_blowdown_heat_loss,
    calculate_blowdown_water_loss,
    # Savings calculations
    calculate_blowdown_savings,
    determine_optimal_blowdown,
    # Heat recovery
    calculate_flash_tank_recovery,
    calculate_heat_exchanger_recovery,
    # Utility functions
    estimate_blowdown_temperature,
    generate_blowdown_recommendation,
)

from .dosing import (
    # Oxygen scavenger dosing
    calculate_oxygen_scavenger_dose,
    select_optimal_scavenger,
    # Phosphate dosing
    calculate_phosphate_dose,
    # pH control
    calculate_ph_adjustment_dose,
    # Comprehensive recommendations
    generate_dosing_recommendation,
    # Consumption tracking
    calculate_chemical_consumption,
    estimate_days_of_supply,
)

__all__ = [
    # Chemistry module
    "determine_pressure_class",
    "get_asme_water_limits",
    "check_parameter_compliance",
    "check_chemistry_compliance",
    "calculate_cycles_of_concentration",
    "calculate_max_cycles_by_silica",
    "calculate_max_cycles_by_alkalinity",
    "calculate_max_cycles_by_conductivity",
    "calculate_optimal_cycles",
    "calculate_tds_from_conductivity",
    "analyze_chemistry_trends",
    "estimate_silica_solubility",
    "calculate_caustic_concentration",
    # Blowdown module
    "calculate_blowdown_rate_from_cycles",
    "calculate_cycles_from_blowdown",
    "calculate_makeup_water_flow",
    "calculate_blowdown_heat_loss",
    "calculate_blowdown_water_loss",
    "calculate_blowdown_savings",
    "determine_optimal_blowdown",
    "calculate_flash_tank_recovery",
    "calculate_heat_exchanger_recovery",
    "estimate_blowdown_temperature",
    "generate_blowdown_recommendation",
    # Dosing module
    "calculate_oxygen_scavenger_dose",
    "select_optimal_scavenger",
    "calculate_phosphate_dose",
    "calculate_ph_adjustment_dose",
    "generate_dosing_recommendation",
    "calculate_chemical_consumption",
    "estimate_days_of_supply",
]
