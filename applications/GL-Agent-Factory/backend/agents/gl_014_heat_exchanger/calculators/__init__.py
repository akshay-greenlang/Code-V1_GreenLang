"""
Calculators for GL-014 EXCHANGERPRO Agent

This module exports all calculator functions for heat exchanger optimization.
All calculations are deterministic, following TEMA standards and heat transfer
engineering principles with zero-hallucination.

Calculator Modules:
    - epsilon_ntu: Epsilon-NTU method for effectiveness calculations
    - lmtd: Log Mean Temperature Difference method
    - fouling: Fouling analysis, prediction, and cleaning optimization
"""

from .epsilon_ntu import (
    # NTU calculations
    calculate_ntu,
    calculate_capacity_ratio,
    calculate_effectiveness,
    calculate_ntu_from_effectiveness,
    # Heat transfer calculations
    calculate_heat_transfer,
    calculate_outlet_temperatures,
    calculate_required_ua,
    # Analysis from measurements
    calculate_effectiveness_from_temperatures,
    calculate_ntu_from_temperatures,
    # Constants
    FLOW_ARRANGEMENTS,
)

from .lmtd import (
    # LMTD calculations
    calculate_lmtd,
    calculate_lmtd_correction_factor,
    calculate_corrected_lmtd,
    # Heat transfer calculations
    calculate_heat_transfer_area,
    calculate_ua_from_lmtd,
    calculate_heat_duty,
    # Overall coefficient
    calculate_overall_coefficient,
    # Design checks
    check_temperature_cross,
    # Rating calculations
    calculate_exchanger_duty_from_ua,
)

from .fouling import (
    # TEMA standards
    get_tema_fouling_resistance,
    TEMA_FOULING_RESISTANCES,
    # Fouling calculations
    calculate_fouling_resistance,
    calculate_fouling_rate,
    calculate_ua_degradation,
    # Prediction
    predict_fouling_over_time,
    # Cleaning optimization
    calculate_cleaning_benefit,
    optimize_cleaning_schedule,
    calculate_next_cleaning_date,
    # Analysis
    analyze_fouling_history,
    generate_fouling_report,
    # Constants
    FOULING_RATE_COEFFICIENTS,
)


__all__ = [
    # Epsilon-NTU calculators
    "calculate_ntu",
    "calculate_capacity_ratio",
    "calculate_effectiveness",
    "calculate_ntu_from_effectiveness",
    "calculate_heat_transfer",
    "calculate_outlet_temperatures",
    "calculate_required_ua",
    "calculate_effectiveness_from_temperatures",
    "calculate_ntu_from_temperatures",
    "FLOW_ARRANGEMENTS",
    # LMTD calculators
    "calculate_lmtd",
    "calculate_lmtd_correction_factor",
    "calculate_corrected_lmtd",
    "calculate_heat_transfer_area",
    "calculate_ua_from_lmtd",
    "calculate_heat_duty",
    "calculate_overall_coefficient",
    "check_temperature_cross",
    "calculate_exchanger_duty_from_ua",
    # Fouling calculators
    "get_tema_fouling_resistance",
    "TEMA_FOULING_RESISTANCES",
    "calculate_fouling_resistance",
    "calculate_fouling_rate",
    "calculate_ua_degradation",
    "predict_fouling_over_time",
    "calculate_cleaning_benefit",
    "optimize_cleaning_schedule",
    "calculate_next_cleaning_date",
    "analyze_fouling_history",
    "generate_fouling_report",
    "FOULING_RATE_COEFFICIENTS",
]
