"""
GL-016 Waterguard Safety System - IEC 61511 SIL-3 Compliant.

This package provides the complete safety system for WATERGUARD including:
    - Boundary Engine: Hard constraints and fail-safe modes
    - Safety Gates: Multiple gate types for defense-in-depth
    - SIS Integration: Read-only interface to Safety Instrumented System
    - Action Gate: Permission-based action control
    - Circuit Breaker: Fault isolation and graceful degradation
    - Emergency Shutdown: Coordinated shutdown sequences
    - Violation Handler: Escalation and CMMS integration

CRITICAL SAFETY PRINCIPLE:
    WATERGUARD is SUPERVISORY ONLY. It cannot override SIS, BMS, or
    safety valves. All safety trips are executed by the independent SIS.

Reference Standards:
    - IEC 61511-1:2016 Functional Safety
    - IEC 61508 Parts 1-7
    - NFPA 85 Boiler and Combustion Systems
    - ISA-18.2 Alarm Management

Author: GreenLang Safety Engineering Team
Version: 1.0.0
SIL Level: 3
"""

from .boundary_engine import (
    WaterguardBoundaryEngine,
    WaterguardConstraints,
    ConstraintViolation,
    FailSafeState,
    WatchdogTimer,
    HeartbeatStatus,
    ProposedAction,
    ConstraintType,
    ViolationSeverity as BoundaryViolationSeverity,
    FailSafeMode,
)

from .safety_gates import (
    # Base classes
    SafetyGate,
    SafetyGateConfig,
    SafetyGateManager,
    GateCheckResult,
    GateResult,
    GateStatus,
    TripAction,
    SafetyLevel,
    # Specialized gates
    AnalyzerHealthGate,
    CommunicationsGate,
    ManualOverrideGate,
    RateLimitGate,
    ConstraintGate,
    ChangeManagementGate,
    SafetyGateCoordinator,
    # Enums
    AnalyzerStatus,
    CommunicationStatus,
    OperatorMode,
    # Factory functions
    create_chemistry_gates,
    create_rate_limit_gates,
)

from .sis_integration import (
    SISInterface,
    SISStatus,
    ActiveTrip,
    SISHealthStatus,
    TripType,
    InterlockStatus,
    BoilerSafetyStatus,
)

from .action_gate import (
    ActionGate,
    PermissionResult,
    GatedAction,
    ActionType,
    PermissionStatus,
    CommissioningMode,
)

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerConfig,
    CircuitBreakerStatus,
    CircuitBreakerRegistry,
    ActuatorCircuitBreaker,
    FailureType,
    RecoveryStrategy,
    FailureRecord,
    StateTransition,
    CircuitOpenError,
)

from .emergency_shutdown import (
    EmergencyShutdownHandler,
    EmergencyEvent,
    EmergencySeverity,
    EmergencyType,
    SafeStateDefinition,
    ShutdownState,
    ShutdownRecord,
    ShutdownAction,
    EquipmentState,
)

from .violation_handler import (
    ViolationHandler,
    ViolationRecord,
    ViolationReport,
    ViolationSeverity,
    ViolationType,
    ViolationState,
    CMMSWorkOrder,
    EscalationLevel,
    EscalationConfig,
    WorkOrderPriority,
    WorkOrderStatus,
)

__version__ = "1.0.0"
__sil_level__ = 3

__all__ = [
    # Boundary Engine
    "WaterguardBoundaryEngine",
    "WaterguardConstraints",
    "ConstraintViolation",
    "FailSafeState",
    "WatchdogTimer",
    "HeartbeatStatus",
    "ProposedAction",
    "ConstraintType",
    "BoundaryViolationSeverity",
    "FailSafeMode",
    # Safety Gates - Base
    "SafetyGate",
    "SafetyGateConfig",
    "SafetyGateManager",
    "GateCheckResult",
    "GateResult",
    "GateStatus",
    "TripAction",
    "SafetyLevel",
    # Safety Gates - Specialized
    "AnalyzerHealthGate",
    "CommunicationsGate",
    "ManualOverrideGate",
    "RateLimitGate",
    "ConstraintGate",
    "ChangeManagementGate",
    "SafetyGateCoordinator",
    "AnalyzerStatus",
    "CommunicationStatus",
    "OperatorMode",
    "create_chemistry_gates",
    "create_rate_limit_gates",
    # SIS Integration
    "SISInterface",
    "SISStatus",
    "ActiveTrip",
    "SISHealthStatus",
    "TripType",
    "InterlockStatus",
    "BoilerSafetyStatus",
    # Action Gate
    "ActionGate",
    "PermissionResult",
    "GatedAction",
    "ActionType",
    "PermissionStatus",
    "CommissioningMode",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerState",
    "CircuitBreakerConfig",
    "CircuitBreakerStatus",
    "CircuitBreakerRegistry",
    "ActuatorCircuitBreaker",
    "FailureType",
    "RecoveryStrategy",
    "FailureRecord",
    "StateTransition",
    "CircuitOpenError",
    # Emergency Shutdown
    "EmergencyShutdownHandler",
    "EmergencyEvent",
    "EmergencySeverity",
    "EmergencyType",
    "SafeStateDefinition",
    "ShutdownState",
    "ShutdownRecord",
    "ShutdownAction",
    "EquipmentState",
    # Violation Handler
    "ViolationHandler",
    "ViolationRecord",
    "ViolationReport",
    "ViolationSeverity",
    "ViolationType",
    "ViolationState",
    "CMMSWorkOrder",
    "EscalationLevel",
    "EscalationConfig",
    "WorkOrderPriority",
    "WorkOrderStatus",
]
