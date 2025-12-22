"""
GL-002 FLAMEGUARD BoilerEfficiencyOptimizer - Control Module

This module provides advanced combustion control, setpoint management,
and multi-boiler load optimization for industrial boiler systems.

Control Architecture:
    - Cascade PID for O2 trim and air-fuel ratio
    - Feedforward compensation for load changes
    - Load-based setpoint scheduling
    - Multi-boiler load sharing and sequencing

Reference Standards:
    - NFPA 85 Boiler and Combustion Systems Hazards Code
    - NFPA 86 Standard for Ovens and Furnaces
    - IEC 61511 Functional Safety

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from .combustion_controller import (
    CombustionController,
    CascadePIDController,
    AirFuelRatioController,
    FeedforwardCompensator,
    O2TrimController,
    CombustionOutput,
    CombustionState,
    CombustionConfig,
    ControlMode,
    ControlAction,
    PIDTuning,
    AirFuelRatio,
    O2TrimOutput,
)

from .setpoint_manager import (
    SetpointManager,
    O2SetpointOptimizer,
    LoadBasedScheduler,
    SetpointConstraint,
    SetpointAuditRecord,
    SetpointConfig,
    SetpointType,
    OptimizationMode,
    ConstraintType,
    SetpointOutput,
)

from .load_controller import (
    LoadController,
    BoilerLoadManager,
    LeadLagSequencer,
    LoadShedding,
    TurndownOptimizer,
    BoilerConfig,
    BoilerState,
    LoadAllocation,
    SequencingMode,
    SequencingStrategy,
    LoadSheddingConfig,
    LoadControlOutput,
)

__all__ = [
    "CombustionController",
    "CascadePIDController",
    "AirFuelRatioController",
    "FeedforwardCompensator",
    "O2TrimController",
    "CombustionOutput",
    "CombustionState",
    "CombustionConfig",
    "ControlMode",
    "ControlAction",
    "PIDTuning",
    "AirFuelRatio",
    "O2TrimOutput",
    "SetpointManager",
    "O2SetpointOptimizer",
    "LoadBasedScheduler",
    "SetpointConstraint",
    "SetpointAuditRecord",
    "SetpointConfig",
    "SetpointType",
    "OptimizationMode",
    "ConstraintType",
    "SetpointOutput",
    "LoadController",
    "BoilerLoadManager",
    "LeadLagSequencer",
    "LoadShedding",
    "TurndownOptimizer",
    "BoilerConfig",
    "BoilerState",
    "LoadAllocation",
    "SequencingMode",
    "SequencingStrategy",
    "LoadSheddingConfig",
    "LoadControlOutput",
]
