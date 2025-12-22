"""
GL-004 BURNMASTER Control Module - Control Integration Layer

This module implements the control integration layer for advisory and closed-loop
operation of industrial burner management systems. It provides comprehensive
support for multiple operating modes with safety-first design.

OPERATING MODES:
    1. OBSERVE: Compute KPIs and recommendations, no writes to DCS
    2. ADVISORY: Present recommendations with explanations, manual acceptance
    3. CLOSED_LOOP: Write setpoints within bounded envelope, auto-fallback on anomalies
    4. FALLBACK: Safe state operation during anomalies or failures
    5. MAINTENANCE: System maintenance mode with restricted operations

Key Features:
    - Mode-based control with validated transitions
    - Bumpless transfer between operating modes
    - Advisory system with operator interaction
    - Closed-loop control with safety envelopes
    - Automatic fallback on anomaly detection
    - Actuator rate limiting and hunting prevention
    - Complete audit trail with SHA-256 provenance

Safety Principles:
    - All control actions respect BMS interlocks
    - Graceful degradation on component failures
    - Rollback capability for all setpoint changes
    - Before/after state auditing for all operations

Reference Standards:
    - IEC 61511 Functional Safety
    - ISA-84 Safety Instrumented Systems
    - NFPA 85/86 Boiler/Furnace Standards

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from gl_004_burnmaster.control.mode_manager import (
    OperatingMode,
    OperatingModeManager,
    ModeTransition,
    TransitionResult,
    ModeValidationResult,
    ModeState,
)

from gl_004_burnmaster.control.setpoint_manager import (
    SetpointManager,
    SetpointProposal,
    SetpointRecord,
    ApplyResult,
    ImpactAssessment,
    SetpointValidationResult,
)

from gl_004_burnmaster.control.bumpless_transfer import (
    BumplessTransferController,
    TransferPlan,
    WindupCorrection,
    AbortResult,
    TransferState,
)

from gl_004_burnmaster.control.advisory_controller import (
    AdvisoryController,
    Advisory,
    AdvisoryType,
    AdvisoryPriority,
    OperatorResponse,
    PresentationResult,
    EffectivenessMetrics,
)

from gl_004_burnmaster.control.closed_loop_controller import (
    ClosedLoopController,
    ControlCycleResult,
    WriteResult,
    VerificationResult,
    WriteFailure,
    RecoveryAction,
    BurnerState,
)

from gl_004_burnmaster.control.fallback_controller import (
    FallbackController,
    SafeState,
    FallbackTrigger,
    FallbackResult,
    FallbackEvent,
    FallbackIncident,
    RevertResult,
    NotificationResult,
)

from gl_004_burnmaster.control.rate_limiter import (
    ActuatorRateLimiter,
    ActuatorLimits,
    HuntingDetection,
    RateLimitResult,
)

__all__ = [
    "OperatingMode",
    "OperatingModeManager",
    "ModeTransition",
    "TransitionResult",
    "ModeValidationResult",
    "ModeState",
    "SetpointManager",
    "SetpointProposal",
    "SetpointRecord",
    "ApplyResult",
    "ImpactAssessment",
    "SetpointValidationResult",
    "BumplessTransferController",
    "TransferPlan",
    "WindupCorrection",
    "AbortResult",
    "TransferState",
    "AdvisoryController",
    "Advisory",
    "AdvisoryType",
    "AdvisoryPriority",
    "OperatorResponse",
    "PresentationResult",
    "EffectivenessMetrics",
    "ClosedLoopController",
    "ControlCycleResult",
    "WriteResult",
    "VerificationResult",
    "WriteFailure",
    "RecoveryAction",
    "BurnerState",
    "FallbackController",
    "SafeState",
    "FallbackTrigger",
    "FallbackResult",
    "FallbackEvent",
    "FallbackIncident",
    "RevertResult",
    "NotificationResult",
    "ActuatorRateLimiter",
    "ActuatorLimits",
    "HuntingDetection",
    "RateLimitResult",
]

__version__ = "1.0.0"
__author__ = "GreenLang Control Systems Team"
