# -*- coding: utf-8 -*-
"""
GL-016 Waterguard Control System Package

This package provides the complete control system for boiler water treatment,
including blowdown control, chemical dosing, operating mode management,
setpoint calculation, command handling, and fallback operations.

Key Components:
    - BlowdownController: Continuous and intermittent blowdown control
    - DosingController: Chemical injection control with feed-forward/feedback
    - OperatingModeManager: Multi-mode operation (Recommend, Supervised, Autonomous, Fallback)
    - SetpointManager: Optimal setpoint calculation with constraints
    - CommandHandler: OPC-UA command execution with handshaking
    - FallbackManager: Degraded operation handling

Safety Integration:
    All control modules integrate with safety gates and support staged
    commissioning for safe deployment.

Compliance:
    - ISA-18.2 (Alarm Management)
    - ISA-95 (Enterprise-Control Integration)
    - IEC 62443 (Industrial Cybersecurity)

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from .blowdown_controller import (
    BlowdownController,
    BlowdownValveModel,
    RampRateLimiter,
    BlowdownMode,
    BlowdownConfig,
    BlowdownState,
    BlowdownOutput,
    BlowdownAlert,
    ConductivityReading,
    SilicaReading,
    ControlVariable,
    ValveCharacteristic,
    AlertPriority,
)

from .dosing_controller import (
    DosingController,
    DosingConfig,
    DosingOutput,
    ChemicalType,
    DosingMode,
    DosingReasonCode,
    PumpConstraints,
    TankLevelInterlock,
    ChemicalProgram,
)

from .operating_modes import (
    OperatingModeManager,
    OperatingMode,
    ModeTransitionRequest,
    ModeTransitionResult,
    SafetyGateStatus,
    ApprovalWorkflow,
    ModeConfiguration,
)

from .setpoint_manager import (
    SetpointManager,
    SetpointConfig,
    SetpointOutput,
    SetpointConstraint,
    SetpointHistory,
    OptimizationResult,
)

from .command_handler import (
    CommandHandler,
    CommandConfig,
    CommandRequest,
    CommandResponse,
    CommandStatus,
    HandshakeState,
    AuditLogEntry,
)

from .fallback_manager import (
    FallbackManager,
    FallbackConfig,
    FallbackTrigger,
    FallbackSchedule,
    FallbackState,
    RecoveryCondition,
    BaselineSchedule,
)


__all__ = [
    # Blowdown Controller
    "BlowdownController",
    "BlowdownValveModel",
    "RampRateLimiter",
    "BlowdownMode",
    "BlowdownConfig",
    "BlowdownState",
    "BlowdownOutput",
    "BlowdownAlert",
    "ConductivityReading",
    "SilicaReading",
    "ControlVariable",
    "ValveCharacteristic",
    "AlertPriority",
    "DosingController",
    "DosingConfig",
    "DosingOutput",
    "ChemicalType",
    "DosingMode",
    "DosingReasonCode",
    "PumpConstraints",
    "TankLevelInterlock",
    "ChemicalProgram",
    "OperatingModeManager",
    "OperatingMode",
    "ModeTransitionRequest",
    "ModeTransitionResult",
    "SafetyGateStatus",
    "ApprovalWorkflow",
    "ModeConfiguration",
    "SetpointManager",
    "SetpointConfig",
    "SetpointOutput",
    "SetpointConstraint",
    "SetpointHistory",
    "OptimizationResult",
    "CommandHandler",
    "CommandConfig",
    "CommandRequest",
    "CommandResponse",
    "CommandStatus",
    "HandshakeState",
    "AuditLogEntry",
    "FallbackManager",
    "FallbackConfig",
    "FallbackTrigger",
    "FallbackSchedule",
    "FallbackState",
    "RecoveryCondition",
    "BaselineSchedule",
]

__version__ = "1.0.0"
__author__ = "GreenLang Control Systems Team"
