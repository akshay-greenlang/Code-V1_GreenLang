"""
GL-001 ThermalCommand Safety Boundary Policy Engine

This module provides comprehensive safety boundary enforcement for
GL-001 ThermalCommand operations. It ensures zero safety boundary
violations through deterministic policy enforcement.

Components:
- SafetyBoundaryEngine: Core boundary enforcement engine
- ThermalPolicyManager: Policy definition and management
- SISIndependenceValidator: Ensures GL-001 never interferes with SIS
- SafetyActionGate: Pre-actuation validation gate
- ViolationHandler: Violation handling and escalation

Key Principles:
1. ZERO SAFETY BOUNDARY VIOLATIONS - All actuations must pass safety gate
2. SIS INDEPENDENCE - GL-001 NEVER writes to SIS tags
3. IMMUTABLE AUDIT TRAIL - All violations logged with cryptographic hashes
4. DEFENSE IN DEPTH - Multiple layers of validation

Reference Standards:
    - IEC 61511 (SIS for Process Industries)
    - IEC 61508 (Functional Safety)
    - ISA-84 (Safety Instrumented Functions)

Example:
    >>> from safety import SafetyActionGate, ActionGateFactory
    >>> gate = ActionGateFactory.create_gate()
    >>> request = TagWriteRequest(tag_id="TIC-101", value=150.0)
    >>> result = gate.evaluate(request)
    >>> if result.is_allowed:
    ...     execute_write(result.final_value)
    ... else:
    ...     logger.warning(f"Write blocked: {result.violations}")

Safety Architecture:

    Optimization Engine
           |
           v
    +------------------+
    | Pre-Optimization |
    | Constraint Check |
    +------------------+
           |
           v
    +------------------+
    | SIS Independence |
    |    Validator     |
    +------------------+
           |
           v
    +------------------+
    | Safety Boundary  |
    |     Engine       |
    +------------------+
           |
           v
    +------------------+
    | Safety Action    |
    |      Gate        |
    +------------------+
           |
           v
       [ALLOW/BLOCK]
           |
           v (if blocked)
    +------------------+
    |   Violation      |
    |    Handler       |
    +------------------+
           |
           v
       Audit Record

"""

from typing import TYPE_CHECKING

# Import schemas
from .safety_schemas import (
    # Enums
    ViolationType,
    ViolationSeverity,
    SafetyLevel,
    GateDecision,
    PolicyType,
    ConditionOperator,
    # Models
    BoundaryPolicy,
    BoundaryViolation,
    TagWriteRequest,
    ActionGateResult,
    SafetyState,
    SafetyAuditRecord,
    SISState,
    InterlockState,
    AlarmState,
    TimeRestriction,
    Condition,
    RateLimitSpec,
)

# Import boundary policies
from .boundary_policies import (
    ThermalPolicyManager,
    get_policy_manager,
    reset_policy_manager,
    TEMPERATURE_POLICIES,
    PRESSURE_POLICIES,
    FLOW_POLICIES,
    LEVEL_POLICIES,
    VALVE_POLICIES,
    ALLOWED_WRITE_TAGS,
    BLACKLISTED_TAGS,
)

# Import boundary engine
from .boundary_engine import SafetyBoundaryEngine

# Import SIS validator
from .sis_validator import (
    SISIndependenceValidator,
    SISValidationResult,
    SISViolationType,
    SISMonitor,
    SIS_TAG_PATTERNS,
    SIS_CONFIG_PATTERNS,
)

# Import action gate
from .action_gate import (
    SafetyActionGate,
    GateConfiguration,
    VelocityLimit,
    ActionGateFactory,
)

# Import violation handler
from .violation_handler import (
    ViolationHandler,
    ViolationEvent,
    EscalationLevel,
    EscalationRule,
    NotificationType,
    NotificationTarget,
    ViolationHandlerFactory,
)


__all__ = [
    # Enums
    "ViolationType",
    "ViolationSeverity",
    "SafetyLevel",
    "GateDecision",
    "PolicyType",
    "ConditionOperator",
    "EscalationLevel",
    "NotificationType",
    "SISViolationType",
    # Core Models
    "BoundaryPolicy",
    "BoundaryViolation",
    "TagWriteRequest",
    "ActionGateResult",
    "SafetyState",
    "SafetyAuditRecord",
    "SISState",
    "InterlockState",
    "AlarmState",
    # Policy Models
    "TimeRestriction",
    "Condition",
    "RateLimitSpec",
    "GateConfiguration",
    "VelocityLimit",
    "EscalationRule",
    "NotificationTarget",
    "ViolationEvent",
    "SISValidationResult",
    # Engines and Validators
    "SafetyBoundaryEngine",
    "SISIndependenceValidator",
    "SafetyActionGate",
    "ViolationHandler",
    "SISMonitor",
    # Policy Manager
    "ThermalPolicyManager",
    "get_policy_manager",
    "reset_policy_manager",
    # Factories
    "ActionGateFactory",
    "ViolationHandlerFactory",
    # Policy Constants
    "TEMPERATURE_POLICIES",
    "PRESSURE_POLICIES",
    "FLOW_POLICIES",
    "LEVEL_POLICIES",
    "VALVE_POLICIES",
    "ALLOWED_WRITE_TAGS",
    "BLACKLISTED_TAGS",
    "SIS_TAG_PATTERNS",
    "SIS_CONFIG_PATTERNS",
]


def create_safety_system(
    violation_callback=None,
    tag_value_provider=None,
) -> tuple:
    """
    Create a complete safety system with all components wired together.

    This is a convenience function that creates and wires all safety
    components for typical use cases.

    Args:
        violation_callback: Optional callback for violations
        tag_value_provider: Optional function to get current tag values

    Returns:
        Tuple of (gate, boundary_engine, sis_validator, violation_handler)

    Example:
        >>> gate, engine, sis, handler = create_safety_system()
        >>> result = gate.evaluate(request)
    """
    # Create violation handler
    violation_handler = ViolationHandlerFactory.create_production_handler()

    # Create SIS validator
    sis_validator = SISIndependenceValidator(
        violation_callback=violation_callback or violation_handler.handle_violation,
    )

    # Create boundary engine
    boundary_engine = SafetyBoundaryEngine(
        violation_callback=violation_callback or violation_handler.handle_violation,
        tag_value_provider=tag_value_provider,
    )

    # Create action gate
    gate = SafetyActionGate(
        boundary_engine=boundary_engine,
        sis_validator=sis_validator,
        violation_handler=violation_callback or violation_handler.handle_violation,
    )

    return gate, boundary_engine, sis_validator, violation_handler


# Version
__version__ = "1.0.0"
