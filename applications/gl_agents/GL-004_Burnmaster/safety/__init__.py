"""
GL-004 BURNMASTER Safety Module

This module provides comprehensive safety enforcement for the combustion optimizer.
CRITICAL: No action may bypass SIS/BMS; all writes must be within pre-approved envelopes.

The safety module ensures:
1. All setpoints remain within validated safety envelopes
2. BMS/SIS interlocks are respected (read-only, never bypassed)
3. Trip precursors are detected and handled proactively
4. Combustion hazards are detected and mitigated
5. Complete audit trail for regulatory compliance
6. Emergency response coordination (observe-only mode)

Example:
    >>> from safety import SafetyEnvelope, SafetyConstraintValidator
    >>> envelope = SafetyEnvelope(unit_id="BLR-001")
    >>> validation = envelope.validate_within_envelope(setpoint)
    >>> if not validation.is_valid:
    ...     # Block setpoint write, fallback to observe-only
    ...     pass
"""

from safety.safety_envelope import (
    SafetyEnvelope,
    EnvelopeValidation,
    EnvelopeLimits,
)
from safety.constraint_validator import (
    SafetyConstraintValidator,
    ValidationResult,
    ComprehensiveValidation,
    SafetyLimits,
    BurnerState,
)
from safety.interlock_manager import (
    InterlockManager,
    BMSStatus,
    SISStatus,
    PermissiveStatus,
    Interlock,
    InterlockEvent,
    BlockResult,
)
from safety.trip_handler import (
    TripHandler,
    TripPrecursor,
    TripEvent,
    TripAnalysis,
    PreventionRecommendation,
    ResponseResult,
)
from safety.hazard_detector import (
    CombustionHazardDetector,
    HazardLevel,
    Hazard,
    HazardAssessment,
)
from safety.audit_trail import (
    SafetyAuditTrail,
    AuditRecord,
    SafetyCheck,
    ConstraintViolation,
    EnvelopeChange,
    SafetyReport,
    DateRange,
)
from safety.emergency_response import (
    EmergencyResponseHandler,
    EmergencyScenario,
    EmergencyEvent,
    ActionPlan,
    ShutdownResult,
    NotificationResult,
)
from safety.velocity_limiter import (
    # Enums
    VelocityLimitStatus,
    SetpointDirection,
    SafetyMode,
    # Data classes
    VelocityLimit,
    # Models
    VelocityLimitConfig,
    VelocityLimitResult,
    VelocityAuditRecord,
    # Main class
    CombustionVelocityLimiter,
    # Constants
    DEFAULT_VELOCITY_LIMITS,
    # Functions
    create_velocity_limit,
)

__all__ = [
    # Safety Envelope
    "SafetyEnvelope",
    "EnvelopeValidation",
    "EnvelopeLimits",
    # Constraint Validator
    "SafetyConstraintValidator",
    "ValidationResult",
    "ComprehensiveValidation",
    "SafetyLimits",
    "BurnerState",
    # Interlock Manager
    "InterlockManager",
    "BMSStatus",
    "SISStatus",
    "PermissiveStatus",
    "Interlock",
    "InterlockEvent",
    "BlockResult",
    # Trip Handler
    "TripHandler",
    "TripPrecursor",
    "TripEvent",
    "TripAnalysis",
    "PreventionRecommendation",
    "ResponseResult",
    # Hazard Detector
    "CombustionHazardDetector",
    "HazardLevel",
    "Hazard",
    "HazardAssessment",
    # Audit Trail
    "SafetyAuditTrail",
    "AuditRecord",
    "SafetyCheck",
    "ConstraintViolation",
    "EnvelopeChange",
    "SafetyReport",
    "DateRange",
    # Emergency Response
    "EmergencyResponseHandler",
    "EmergencyScenario",
    "EmergencyEvent",
    "ActionPlan",
    "ShutdownResult",
    "NotificationResult",
    # Velocity Limiter
    "VelocityLimitStatus",
    "SetpointDirection",
    "SafetyMode",
    "VelocityLimit",
    "VelocityLimitConfig",
    "VelocityLimitResult",
    "VelocityAuditRecord",
    "CombustionVelocityLimiter",
    "DEFAULT_VELOCITY_LIMITS",
    "create_velocity_limit",
]

__version__ = "1.0.0"
__author__ = "GreenLang Safety Team"
