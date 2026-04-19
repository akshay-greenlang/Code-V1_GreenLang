"""
GL-049 Process Control Calculators

Physics-based calculators for control loop analysis, PID tuning,
and process optimization.
"""

from .pid_tuning import (
    calculate_pid_parameters,
    tune_ziegler_nichols,
    tune_cohen_coon,
    calculate_process_response,
)
from .loop_analysis import (
    analyze_control_loop,
    calculate_integral_absolute_error,
    calculate_settling_time,
    calculate_overshoot,
)
from .optimization import (
    calculate_setpoint_optimization,
    calculate_cascade_benefit,
    calculate_loop_interaction,
)

__all__ = [
    # PID tuning
    "calculate_pid_parameters",
    "tune_ziegler_nichols",
    "tune_cohen_coon",
    "calculate_process_response",
    # Loop analysis
    "analyze_control_loop",
    "calculate_integral_absolute_error",
    "calculate_settling_time",
    "calculate_overshoot",
    # Optimization
    "calculate_setpoint_optimization",
    "calculate_cascade_benefit",
    "calculate_loop_interaction",
]
