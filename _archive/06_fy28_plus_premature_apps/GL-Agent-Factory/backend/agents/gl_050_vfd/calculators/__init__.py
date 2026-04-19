"""
GL-050 VFD Optimization Calculators

Physics-based calculators for Variable Frequency Drive optimization
using fan/pump affinity laws and motor efficiency calculations.
"""

from .affinity_laws import (
    calculate_flow_at_speed,
    calculate_head_at_speed,
    calculate_power_at_speed,
    calculate_speed_for_flow,
    calculate_speed_for_head,
)
from .motor_efficiency import (
    calculate_motor_efficiency,
    calculate_vfd_efficiency,
    calculate_system_efficiency,
    calculate_power_factor,
)
from .energy_savings import (
    calculate_vfd_energy_savings,
    calculate_payback_period,
    calculate_annual_operating_cost,
    compare_control_methods,
)

__all__ = [
    # Affinity laws
    "calculate_flow_at_speed",
    "calculate_head_at_speed",
    "calculate_power_at_speed",
    "calculate_speed_for_flow",
    "calculate_speed_for_head",
    # Motor efficiency
    "calculate_motor_efficiency",
    "calculate_vfd_efficiency",
    "calculate_system_efficiency",
    "calculate_power_factor",
    # Energy savings
    "calculate_vfd_energy_savings",
    "calculate_payback_period",
    "calculate_annual_operating_cost",
    "compare_control_methods",
]
