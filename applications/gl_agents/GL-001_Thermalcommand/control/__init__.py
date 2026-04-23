"""
GL-001 ThermalCommand Control Module

Cascade PID control hierarchy implementation for process heat systems.

Key Features:
    - Master-slave PID cascade architecture
    - Gain scheduling based on operating region
    - Anti-windup with back-calculation
    - Feedforward compensation
    - Bumpless transfer between modes
    - Auto/Manual/Cascade mode switching
    - Rate limiting and output clamping
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cascade_controller import (
        PIDController,
        PIDTuning,
        CascadeController,
        CascadeCoordinator,
        ControlMode,
        ControlAction,
        ControlOutput,
        CascadeOutput,
    )

__all__ = [
    "PIDController",
    "PIDTuning",
    "CascadeController",
    "CascadeCoordinator",
    "ControlMode",
    "ControlAction",
    "ControlOutput",
    "CascadeOutput",
]
