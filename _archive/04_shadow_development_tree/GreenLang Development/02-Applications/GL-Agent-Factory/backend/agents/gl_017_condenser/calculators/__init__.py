"""
Calculators for GL-017 CONDENSYNC Agent

This module exports all calculator functions for condenser optimization.
All calculations are deterministic, following HEI Standards and heat transfer
engineering principles with zero-hallucination.

Calculator Modules:
    - heat_transfer: HEI-compliant heat transfer calculations
    - vacuum: Vacuum system optimization
    - fouling: Fouling analysis and prediction
    - cleanliness: HEI cleanliness factor calculations
"""

from .heat_transfer import (
    calculate_overall_heat_transfer_coefficient,
    calculate_hei_base_coefficient,
    calculate_temperature_correction_factor,
    calculate_lmtd,
    calculate_heat_duty,
    calculate_required_surface_area,
    calculate_tube_side_heat_transfer,
    calculate_shell_side_heat_transfer,
    calculate_tube_wall_resistance,
    calculate_overall_u_from_resistances,
    calculate_ttd,
    calculate_cooling_water_rise,
    calculate_cooling_water_flow,
    estimate_latent_heat,
    HEI_MATERIAL_FACTORS,
    TUBE_THERMAL_CONDUCTIVITY,
)

from .vacuum import (
    calculate_saturation_pressure,
    calculate_saturation_temperature,
    calculate_vacuum_inches_hg,
    calculate_absolute_pressure_from_vacuum,
    calculate_vacuum_efficiency,
    calculate_air_inleakage_rate,
    assess_air_leakage_severity,
    calculate_vacuum_pump_capacity,
    calculate_theoretical_vacuum,
    calculate_vacuum_deviation,
    calculate_power_loss_from_vacuum_degradation,
    calculate_optimal_vacuum,
    generate_vacuum_recommendations,
)

from .fouling import (
    calculate_fouling_resistance,
    calculate_fouling_rate,
    predict_fouling_linear,
    predict_fouling_asymptotic,
    estimate_fouling_parameters_from_history,
    calculate_hours_to_cleaning_threshold,
    recommend_cleaning_date,
    calculate_cleaning_benefit,
    assess_fouling_severity,
    get_standard_fouling_resistance,
    calculate_fouling_factor_components,
    generate_fouling_recommendations,
    STANDARD_FOULING_RESISTANCES,
    CLEANING_EFFECTIVENESS,
)

from .cleanliness import (
    calculate_cleanliness_factor,
    calculate_cleanliness_factor_from_temperatures,
    calculate_effective_u_from_cf,
    assess_cleanliness_status,
    calculate_cf_trend,
    calculate_performance_impact,
    calculate_cleaning_benefit_cf,
    get_design_cleanliness_factor,
    calculate_cf_from_fouling_resistance,
    generate_cf_recommendations,
    calculate_cf_score,
    CF_THRESHOLDS,
    DESIGN_CLEANLINESS_FACTORS,
)

__all__ = [
    # Heat Transfer
    "calculate_overall_heat_transfer_coefficient",
    "calculate_hei_base_coefficient",
    "calculate_temperature_correction_factor",
    "calculate_lmtd",
    "calculate_heat_duty",
    "calculate_required_surface_area",
    "calculate_tube_side_heat_transfer",
    "calculate_shell_side_heat_transfer",
    "calculate_tube_wall_resistance",
    "calculate_overall_u_from_resistances",
    "calculate_ttd",
    "calculate_cooling_water_rise",
    "calculate_cooling_water_flow",
    "estimate_latent_heat",
    "HEI_MATERIAL_FACTORS",
    "TUBE_THERMAL_CONDUCTIVITY",
    # Vacuum
    "calculate_saturation_pressure",
    "calculate_saturation_temperature",
    "calculate_vacuum_inches_hg",
    "calculate_absolute_pressure_from_vacuum",
    "calculate_vacuum_efficiency",
    "calculate_air_inleakage_rate",
    "assess_air_leakage_severity",
    "calculate_vacuum_pump_capacity",
    "calculate_theoretical_vacuum",
    "calculate_vacuum_deviation",
    "calculate_power_loss_from_vacuum_degradation",
    "calculate_optimal_vacuum",
    "generate_vacuum_recommendations",
    # Fouling
    "calculate_fouling_resistance",
    "calculate_fouling_rate",
    "predict_fouling_linear",
    "predict_fouling_asymptotic",
    "estimate_fouling_parameters_from_history",
    "calculate_hours_to_cleaning_threshold",
    "recommend_cleaning_date",
    "calculate_cleaning_benefit",
    "assess_fouling_severity",
    "get_standard_fouling_resistance",
    "calculate_fouling_factor_components",
    "generate_fouling_recommendations",
    "STANDARD_FOULING_RESISTANCES",
    "CLEANING_EFFECTIVENESS",
    # Cleanliness
    "calculate_cleanliness_factor",
    "calculate_cleanliness_factor_from_temperatures",
    "calculate_effective_u_from_cf",
    "assess_cleanliness_status",
    "calculate_cf_trend",
    "calculate_performance_impact",
    "calculate_cleaning_benefit_cf",
    "get_design_cleanliness_factor",
    "calculate_cf_from_fouling_resistance",
    "generate_cf_recommendations",
    "calculate_cf_score",
    "CF_THRESHOLDS",
    "DESIGN_CLEANLINESS_FACTORS",
]
