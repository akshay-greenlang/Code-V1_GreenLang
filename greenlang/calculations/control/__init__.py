"""
GreenLang Process Control Library

Zero-Hallucination Process Control Implementations

This module provides deterministic process control algorithms
for industrial automation applications.

Modules:
    - cascade: Cascade control structures
    - load_allocation: Multi-equipment load distribution (MILP)
    - feedforward: Feedforward control for disturbance rejection
    - ratio_control: Ratio control for proportional relationships
    - setpoint_optimizer: Setpoint optimization algorithms

All calculations provide:
    - Deterministic outputs (same input = same output)
    - Complete provenance tracking (SHA-256 hashes)
    - ISA standards compliance
    - Industrial-grade implementations

Author: GreenLang Engineering Team
License: MIT
"""

# Cascade Control
from .cascade import (
    CascadeController,
    PIDController,
    PIDParameters,
    CascadeLoopResult,
    ControllerState,
    ControlMode,
    ControlAction,
    create_temperature_cascade,
    create_level_cascade,
)

# Load Allocation
from .load_allocation import (
    LoadAllocator,
    AllocationResult,
    EquipmentUnit,
    EquipmentStatus,
    AllocationMethod,
    allocate_boiler_load,
    allocate_compressor_load,
)

# Feedforward Control
from .feedforward import (
    FeedforwardController,
    FeedforwardConfig,
    FeedforwardResult,
    FeedforwardType,
    MultiDisturbanceFeedforward,
    create_flow_feedforward,
    create_temperature_feedforward,
    calculate_feedforward_gain,
)

# Ratio Control
from .ratio_control import (
    RatioController,
    RatioConfig,
    RatioResult,
    RatioMode,
    AirFuelRatioController,
    create_ratio_controller,
    create_blend_ratio_controller,
)

# Setpoint Optimization
from .setpoint_optimizer import (
    SetpointOptimizer,
    OptimizationResult,
    OptimizationVariable,
    OptimizationConstraint,
    OptimizationObjective,
    ConstraintType,
    CombustionOptimizer,
    optimize_excess_air,
    optimize_approach_temperature,
)

# Tuning Parameter Management
from .tuning_manager import (
    TuningParameterManager,
    TuningParameters,
    FOPDTModel,
    TuningMethod,
    ControllerType,
    ProcessModel,
    tune_pid,
)

__version__ = "1.0.0"

__all__ = [
    # Cascade
    "CascadeController",
    "PIDController",
    "PIDParameters",
    "CascadeLoopResult",
    "ControllerState",
    "ControlMode",
    "ControlAction",
    "create_temperature_cascade",
    "create_level_cascade",
    # Load Allocation
    "LoadAllocator",
    "AllocationResult",
    "EquipmentUnit",
    "EquipmentStatus",
    "AllocationMethod",
    "allocate_boiler_load",
    "allocate_compressor_load",
    # Feedforward
    "FeedforwardController",
    "FeedforwardConfig",
    "FeedforwardResult",
    "FeedforwardType",
    "MultiDisturbanceFeedforward",
    "create_flow_feedforward",
    "create_temperature_feedforward",
    "calculate_feedforward_gain",
    # Ratio Control
    "RatioController",
    "RatioConfig",
    "RatioResult",
    "RatioMode",
    "AirFuelRatioController",
    "create_ratio_controller",
    "create_blend_ratio_controller",
    # Setpoint Optimization
    "SetpointOptimizer",
    "OptimizationResult",
    "OptimizationVariable",
    "OptimizationConstraint",
    "OptimizationObjective",
    "ConstraintType",
    "CombustionOptimizer",
    "optimize_excess_air",
    "optimize_approach_temperature",
    # Tuning Parameter Management
    "TuningParameterManager",
    "TuningParameters",
    "FOPDTModel",
    "TuningMethod",
    "ControllerType",
    "ProcessModel",
    "tune_pid",
]
