"""
GL-003 UNIFIEDSTEAM SteamSystemOptimizer - Control Module

This module provides advanced control capabilities for steam system optimization
including desuperheater control, steam quality management, PRV setpoint control,
and centralized setpoint management.

Control Architecture:
    - Advisory mode: Recommendations require operator confirmation
    - Closed-loop mode: Auto-implementation within safe bounds
    - Safety envelope enforcement at all times
    - Transport delay and sensor lag compensation

Reference Standards:
    - ISA-5.1 Instrumentation Symbols and Identification
    - ISA-18.2 Management of Alarm Systems
    - IEC 61511 Functional Safety
    - ASME B31.1 Power Piping

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from .desuperheater_controller import (
    DesuperheaterController,
    ControlMode,
    ProtectionStatusType,
    SetpointSource,
    PIDTuning,
    DesuperheaterState,
    DesuperheaterConstraints,
    Setpoint,
    ProtectionStatus,
    AdvisoryResult,
    ControlResult,
    PIDState,
)

from .steam_quality_controller import (
    SteamQualityController,
    QualityStatus,
    CorrectionType,
    ControlActionType,
    SteamQualityState,
    QualityTargets,
    Correction,
    CoordinatedResult,
    ControlAction,
)

from .prv_controller import (
    PRVController,
    PRVState,
    ValidationStatus as PRVValidationStatus,
    AdvisoryType,
    ConstraintType,
    PRVConfiguration,
    PRVOperatingState,
    PressureConstraints,
    DemandForecast,
    Setpoint as PRVSetpoint,
    ValidationResult as PRVValidationResult,
    PRVAdvisory,
)

from .setpoint_manager import (
    SetpointManager,
    SetpointCategory,
    SetpointSource as ManagedSetpointSource,
    AuthorizationLevel,
    ValidationStatus,
    ApplicationStatus,
    SetpointConstraint,
    SetpointDefinition,
    SetpointRegistration,
    ValidationResult,
    Authorization,
    ApplicationResult,
    RollbackResult,
    SetpointChange,
    AuditRecord,
)

__all__ = [
    # Desuperheater Controller
    "DesuperheaterController",
    "ControlMode",
    "ProtectionStatusType",
    "SetpointSource",
    "PIDTuning",
    "DesuperheaterState",
    "DesuperheaterConstraints",
    "Setpoint",
    "ProtectionStatus",
    "AdvisoryResult",
    "ControlResult",
    "PIDState",
    # Steam Quality Controller
    "SteamQualityController",
    "QualityStatus",
    "CorrectionType",
    "ControlActionType",
    "SteamQualityState",
    "QualityTargets",
    "Correction",
    "CoordinatedResult",
    "ControlAction",
    # PRV Controller
    "PRVController",
    "PRVState",
    "PRVValidationStatus",
    "AdvisoryType",
    "ConstraintType",
    "PRVConfiguration",
    "PRVOperatingState",
    "PressureConstraints",
    "DemandForecast",
    "PRVSetpoint",
    "PRVValidationResult",
    "PRVAdvisory",
    # Setpoint Manager
    "SetpointManager",
    "SetpointCategory",
    "ManagedSetpointSource",
    "AuthorizationLevel",
    "ValidationStatus",
    "ApplicationStatus",
    "SetpointConstraint",
    "SetpointDefinition",
    "SetpointRegistration",
    "ValidationResult",
    "Authorization",
    "ApplicationResult",
    "RollbackResult",
    "SetpointChange",
    "AuditRecord",
]
