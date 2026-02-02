"""
GL-048 Heat Loss Calculators

Physics-based calculators for surface heat loss, insulation analysis,
and pipe/tank heat transfer.
"""

from .surface import (
    calculate_convection_coefficient,
    calculate_convection_heat_loss,
    calculate_radiation_heat_loss,
    calculate_combined_surface_heat_loss,
)
from .insulation import (
    calculate_insulation_heat_loss,
    calculate_economic_thickness,
    calculate_insulation_effectiveness,
    calculate_surface_temperature_insulated,
)
from .cylindrical import (
    calculate_pipe_heat_loss,
    calculate_critical_radius,
    calculate_multilayer_pipe_heat_loss,
    calculate_tank_heat_loss,
)

__all__ = [
    # Surface calculations
    "calculate_convection_coefficient",
    "calculate_convection_heat_loss",
    "calculate_radiation_heat_loss",
    "calculate_combined_surface_heat_loss",
    # Insulation calculations
    "calculate_insulation_heat_loss",
    "calculate_economic_thickness",
    "calculate_insulation_effectiveness",
    "calculate_surface_temperature_insulated",
    # Cylindrical geometry
    "calculate_pipe_heat_loss",
    "calculate_critical_radius",
    "calculate_multilayer_pipe_heat_loss",
    "calculate_tank_heat_loss",
]
