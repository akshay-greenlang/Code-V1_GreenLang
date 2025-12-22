"""
GL-003 UNIFIEDSTEAM SteamSystemOptimizer - Safety Module

This module provides safety envelope management, constraint validation,
interlock management, and trip handling for the steam system optimizer.

Safety Architecture:
    - GL-003 respects ALL existing safety envelopes
    - NO control authority over Safety Instrumented Systems (SIS)
    - Permissive checking before any optimization action
    - Complete audit trail for all safety-related operations

Reference Standards:
    - IEC 61511 Functional Safety
    - ISA-84 Safety Instrumented Systems
    - ASME B31.1 Power Piping
    - ASME BPVC Section VIII

IMPORTANT SAFETY NOTICE:
GL-003 has MONITORING and ADVISORY capabilities only for safety functions.
It cannot modify SIF logic, bypass interlocks, or initiate trips.
All safety control authority remains with the plant's Safety Instrumented System.

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from .safety_envelope import (
    SafetyEnvelope,
    LimitType,
    EnvelopeStatus,
    AlarmSeverity,
    EquipmentType,
    AlarmMargins,
    PressureLimits,
    TemperatureLimits,
    QualityLimits,
    RateLimits,
    EnvelopeCheckResult,
    EquipmentRating,
)

from .constraint_validator import (
    ConstraintValidator,
    ValidationStatus,
    ConstraintCategory,
    ViolationType,
    Constraint,
    Recommendation,
    SetpointChange,
    Violation,
    ValidationResult,
    ActuatorCheck,
    RateCheck,
)

from .interlock_manager import (
    InterlockManager,
    InterlockState,
    PermissiveStatus,
    InterlockType,
    OperationType,
    ActionType,
    InterlockCondition,
    InterlockDefinition,
    ActiveInterlock,
    PermissiveResult,
    InterlockEvaluation,
    SystemState as InterlockSystemState,
)

from .trip_handler import (
    TripHandler,
    TripType,
    TripSeverity,
    TripState,
    SafeStateStatus,
    TripDefinition,
    TripCondition,
    TripEvent,
    SafeStateResult,
    TripReport,
    SystemState as TripSystemState,
)

__all__ = [
    # Safety Envelope
    "SafetyEnvelope",
    "LimitType",
    "EnvelopeStatus",
    "AlarmSeverity",
    "EquipmentType",
    "AlarmMargins",
    "PressureLimits",
    "TemperatureLimits",
    "QualityLimits",
    "RateLimits",
    "EnvelopeCheckResult",
    "EquipmentRating",
    # Constraint Validator
    "ConstraintValidator",
    "ValidationStatus",
    "ConstraintCategory",
    "ViolationType",
    "Constraint",
    "Recommendation",
    "SetpointChange",
    "Violation",
    "ValidationResult",
    "ActuatorCheck",
    "RateCheck",
    # Interlock Manager
    "InterlockManager",
    "InterlockState",
    "PermissiveStatus",
    "InterlockType",
    "OperationType",
    "ActionType",
    "InterlockCondition",
    "InterlockDefinition",
    "ActiveInterlock",
    "PermissiveResult",
    "InterlockEvaluation",
    "InterlockSystemState",
    # Trip Handler
    "TripHandler",
    "TripType",
    "TripSeverity",
    "TripState",
    "SafeStateStatus",
    "TripDefinition",
    "TripCondition",
    "TripEvent",
    "SafeStateResult",
    "TripReport",
    "TripSystemState",
]
