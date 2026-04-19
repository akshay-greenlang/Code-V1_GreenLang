"""
GL-047 Refractory Calculators

Physics-based calculators for refractory monitoring, heat flux analysis,
and remaining life estimation.
"""

from .heat_flux import (
    calculate_heat_flux,
    calculate_conduction_heat_flux,
    calculate_thermal_conductivity,
    calculate_temperature_gradient,
)
from .degradation import (
    calculate_wear_rate,
    calculate_remaining_life,
    calculate_degradation_factor,
    estimate_spalling_risk,
)
from .ir_analysis import (
    detect_hot_spots,
    calculate_surface_emissivity_correction,
    analyze_thermal_profile,
    calculate_anomaly_severity,
)

__all__ = [
    # Heat flux calculations
    "calculate_heat_flux",
    "calculate_conduction_heat_flux",
    "calculate_thermal_conductivity",
    "calculate_temperature_gradient",
    # Degradation calculations
    "calculate_wear_rate",
    "calculate_remaining_life",
    "calculate_degradation_factor",
    "estimate_spalling_risk",
    # IR analysis
    "detect_hot_spots",
    "calculate_surface_emissivity_correction",
    "analyze_thermal_profile",
    "calculate_anomaly_severity",
]
