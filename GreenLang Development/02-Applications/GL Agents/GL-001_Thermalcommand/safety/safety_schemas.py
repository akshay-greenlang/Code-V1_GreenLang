"""
Safety Schemas for GL-001 ThermalCommand Safety Boundary Policy Engine

This module defines all Pydantic models for the safety boundary system including:
- BoundaryPolicy: Safety boundary policy definitions
- BoundaryViolation: Records of boundary violations
- ActionGateResult: Results from pre-actuation safety gates
- SafetyState: Overall system safety state

These schemas ensure type safety and validation for all safety-critical operations.
The safety system follows zero-hallucination principles with deterministic enforcement.

Example:
    >>> policy = BoundaryPolicy(
    ...     policy_id="TEMP_LIMIT_001",
    ...     tag_pattern="TI-*",
    ...     min_value=0.0,
    ...     max_value=150.0
    ... )
    >>> violation = BoundaryViolation(
    ...     policy_id="TEMP_LIMIT_001",
    ...     tag_id="TI-101",
    ...     requested_value=175.0,
    ...     boundary_value=150.0
    ... )
"""

from datetime import datetime, time
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
import hashlib
import uuid

from pydantic import BaseModel, Field, validator, root_validator


class ViolationType(str, Enum):
    """Types of boundary violations."""

    OVER_MAX = "over_max"
    UNDER_MIN = "under_min"
    RATE_EXCEEDED = "rate_exceeded"
    UNAUTHORIZED_TAG = "unauthorized_tag"
    TIME_RESTRICTED = "time_restricted"
    CONDITION_VIOLATED = "condition_violated"
    INTERLOCK_ACTIVE = "interlock_active"
    ALARM_ACTIVE = "alarm_active"
    SIS_VIOLATION = "sis_violation"


class ViolationSeverity(str, Enum):
    """Severity levels for violations."""

    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class SafetyLevel(str, Enum):
    """Safety integrity levels."""

    SIL_1 = "SIL_1"
    SIL_2 = "SIL_2"
    SIL_3 = "SIL_3"
    SIL_4 = "SIL_4"


class GateDecision(str, Enum):
    """Action gate decisions."""

    ALLOW = "allow"
    BLOCK = "block"
    CLAMP = "clamp"  # Allow but clamp to boundary


class PolicyType(str, Enum):
    """Types of boundary policies."""

    ABSOLUTE_LIMIT = "absolute_limit"
    RATE_LIMIT = "rate_limit"
    WHITELIST = "whitelist"
    CONDITIONAL = "conditional"
    TIME_BASED = "time_based"
    INTERLOCK = "interlock"


class ConditionOperator(str, Enum):
    """Operators for conditional policies."""

    EQUALS = "eq"
    NOT_EQUALS = "neq"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "gte"
    LESS_EQUAL = "lte"
    IN_RANGE = "in_range"
    NOT_IN_RANGE = "not_in_range"


class TimeRestriction(BaseModel):
    """Time-based restriction definition."""

    start_time: time = Field(..., description="Start time of restriction period")
    end_time: time = Field(..., description="End time of restriction period")
    days_of_week: List[int] = Field(
        default=[0, 1, 2, 3, 4, 5, 6],
        description="Days of week (0=Monday, 6=Sunday)"
    )
    timezone: str = Field(default="UTC", description="Timezone for time restriction")

    @validator("days_of_week")
    def validate_days(cls, v: List[int]) -> List[int]:
        """Validate days are in valid range."""
        for day in v:
            if day < 0 or day > 6:
                raise ValueError(f"Invalid day of week: {day}. Must be 0-6.")
        return v


class Condition(BaseModel):
    """Condition for conditional policies."""

    tag_id: str = Field(..., description="Tag ID to evaluate condition on")
    operator: ConditionOperator = Field(..., description="Comparison operator")
    value: Union[float, int, str, bool] = Field(..., description="Value to compare against")
    secondary_value: Optional[Union[float, int]] = Field(
        None,
        description="Secondary value for range operators"
    )

    @root_validator
    def validate_range_values(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate range operators have secondary value."""
        operator = values.get("operator")
        secondary = values.get("secondary_value")

        if operator in (ConditionOperator.IN_RANGE, ConditionOperator.NOT_IN_RANGE):
            if secondary is None:
                raise ValueError(f"Operator {operator} requires secondary_value")

        return values


class RateLimitSpec(BaseModel):
    """Rate limit specification."""

    max_change_per_second: Optional[float] = Field(
        None,
        ge=0,
        description="Maximum change per second"
    )
    max_change_per_minute: Optional[float] = Field(
        None,
        ge=0,
        description="Maximum change per minute"
    )
    max_writes_per_minute: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum write operations per minute"
    )
    cooldown_seconds: Optional[float] = Field(
        None,
        ge=0,
        description="Minimum seconds between writes"
    )


class BoundaryPolicy(BaseModel):
    """
    Safety boundary policy definition.

    Defines constraints that must be enforced on tag values before
    optimization and actuation. Policies are immutable once created.

    Attributes:
        policy_id: Unique identifier for this policy
        policy_type: Type of policy (limit, rate, whitelist, etc.)
        tag_pattern: Glob pattern for matching tags
        min_value: Minimum allowed value (for limit policies)
        max_value: Maximum allowed value (for limit policies)
        rate_limit: Rate limiting specification
        conditions: Conditions that must be met
        time_restrictions: Time-based restrictions
        severity: Severity level if violated
        safety_level: Safety integrity level
        description: Human-readable description
        enabled: Whether policy is currently active
        version: Policy version for audit trail

    Example:
        >>> policy = BoundaryPolicy(
        ...     policy_id="TEMP_MAX_001",
        ...     policy_type=PolicyType.ABSOLUTE_LIMIT,
        ...     tag_pattern="TI-*",
        ...     max_value=150.0,
        ...     severity=ViolationSeverity.CRITICAL
        ... )
    """

    policy_id: str = Field(..., min_length=1, description="Unique policy identifier")
    policy_type: PolicyType = Field(..., description="Type of boundary policy")
    tag_pattern: str = Field(..., description="Glob pattern for matching tag IDs")

    # Absolute limits
    min_value: Optional[float] = Field(None, description="Minimum allowed value")
    max_value: Optional[float] = Field(None, description="Maximum allowed value")
    engineering_units: Optional[str] = Field(None, description="Engineering units for values")

    # Rate limits
    rate_limit: Optional[RateLimitSpec] = Field(None, description="Rate limit specification")

    # Whitelist
    allowed_tags: Optional[Set[str]] = Field(None, description="Whitelist of allowed tag IDs")

    # Conditional
    conditions: Optional[List[Condition]] = Field(None, description="Conditions for policy")
    condition_logic: str = Field(default="AND", description="Logic for multiple conditions (AND/OR)")

    # Time-based
    time_restrictions: Optional[List[TimeRestriction]] = Field(
        None,
        description="Time-based restrictions"
    )

    # Metadata
    severity: ViolationSeverity = Field(
        default=ViolationSeverity.CRITICAL,
        description="Severity if violated"
    )
    safety_level: Optional[SafetyLevel] = Field(None, description="Safety integrity level")
    description: str = Field(default="", description="Human-readable description")
    enabled: bool = Field(default=True, description="Whether policy is active")
    version: str = Field(default="1.0.0", description="Policy version")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")

    @root_validator
    def validate_policy_type_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate required fields based on policy type."""
        policy_type = values.get("policy_type")

        if policy_type == PolicyType.ABSOLUTE_LIMIT:
            if values.get("min_value") is None and values.get("max_value") is None:
                raise ValueError("ABSOLUTE_LIMIT policy requires min_value and/or max_value")

        elif policy_type == PolicyType.RATE_LIMIT:
            if values.get("rate_limit") is None:
                raise ValueError("RATE_LIMIT policy requires rate_limit specification")

        elif policy_type == PolicyType.WHITELIST:
            if not values.get("allowed_tags"):
                raise ValueError("WHITELIST policy requires allowed_tags")

        elif policy_type == PolicyType.CONDITIONAL:
            if not values.get("conditions"):
                raise ValueError("CONDITIONAL policy requires conditions")

        elif policy_type == PolicyType.TIME_BASED:
            if not values.get("time_restrictions"):
                raise ValueError("TIME_BASED policy requires time_restrictions")

        return values

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of policy for immutability verification."""
        policy_str = (
            f"{self.policy_id}|{self.policy_type}|{self.tag_pattern}|"
            f"{self.min_value}|{self.max_value}|{self.severity}|{self.version}"
        )
        return hashlib.sha256(policy_str.encode()).hexdigest()

    class Config:
        """Pydantic config."""
        frozen = False  # Allow modification during creation
        use_enum_values = True


class BoundaryViolation(BaseModel):
    """
    Record of a boundary violation.

    Created when an action is blocked due to safety boundary violation.
    These records are immutable and include cryptographic hash for audit.

    Attributes:
        violation_id: Unique identifier (UUID)
        timestamp: When violation occurred
        policy_id: ID of violated policy
        policy_type: Type of policy violated
        tag_id: Tag that caused violation
        requested_value: Value that was requested
        current_value: Current value of tag
        boundary_value: Boundary that was violated
        violation_type: Type of violation
        severity: Severity level
        blocked: Whether action was blocked
        message: Human-readable violation message
        context: Additional context information
        provenance_hash: SHA-256 hash for audit trail

    Example:
        >>> violation = BoundaryViolation(
        ...     policy_id="TEMP_MAX_001",
        ...     tag_id="TI-101",
        ...     requested_value=175.0,
        ...     current_value=145.0,
        ...     boundary_value=150.0,
        ...     violation_type=ViolationType.OVER_MAX
        ... )
    """

    violation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique violation identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When violation occurred"
    )

    # Policy information
    policy_id: str = Field(..., description="ID of violated policy")
    policy_type: PolicyType = Field(
        default=PolicyType.ABSOLUTE_LIMIT,
        description="Type of policy violated"
    )

    # Tag information
    tag_id: str = Field(..., description="Tag that caused violation")
    requested_value: Optional[float] = Field(None, description="Value that was requested")
    current_value: Optional[float] = Field(None, description="Current tag value")
    boundary_value: Optional[float] = Field(None, description="Boundary that was violated")

    # Violation details
    violation_type: ViolationType = Field(..., description="Type of violation")
    severity: ViolationSeverity = Field(
        default=ViolationSeverity.CRITICAL,
        description="Severity level"
    )
    blocked: bool = Field(default=True, description="Whether action was blocked")

    # Context
    message: str = Field(default="", description="Human-readable message")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

    # Audit
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit trail")

    @validator("provenance_hash", always=True)
    def compute_provenance_hash(cls, v: str, values: Dict[str, Any]) -> str:
        """Compute provenance hash if not provided."""
        if v:
            return v

        # Create deterministic hash from violation data
        hash_input = (
            f"{values.get('timestamp', '')}|"
            f"{values.get('policy_id', '')}|"
            f"{values.get('tag_id', '')}|"
            f"{values.get('requested_value', '')}|"
            f"{values.get('violation_type', '')}|"
            f"{values.get('severity', '')}"
        )
        return hashlib.sha256(hash_input.encode()).hexdigest()

    class Config:
        """Pydantic config - violations are immutable."""
        frozen = True
        use_enum_values = True


class TagWriteRequest(BaseModel):
    """Request to write a value to a tag."""

    tag_id: str = Field(..., description="Tag identifier")
    value: float = Field(..., description="Value to write")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")
    source: str = Field(default="GL-001", description="Source of write request")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracing")

    class Config:
        """Pydantic config."""
        frozen = True


class ActionGateResult(BaseModel):
    """
    Result from the Safety Action Gate.

    Contains the decision and details about whether an action is allowed.

    Attributes:
        decision: Allow, block, or clamp
        original_request: Original write request
        final_value: Value to write (may be clamped)
        violations: List of violations encountered
        checks_passed: List of checks that passed
        checks_failed: List of checks that failed
        gate_timestamp: When gate evaluation occurred
        evaluation_time_ms: Time to evaluate gate
        provenance_hash: SHA-256 hash for audit

    Example:
        >>> result = gate.evaluate(request)
        >>> if result.decision == GateDecision.ALLOW:
        ...     write_to_plc(result.final_value)
    """

    decision: GateDecision = Field(..., description="Gate decision")
    original_request: TagWriteRequest = Field(..., description="Original write request")
    final_value: Optional[float] = Field(None, description="Final value to write (if allowed)")

    # Validation results
    violations: List[BoundaryViolation] = Field(
        default_factory=list,
        description="List of violations"
    )
    checks_passed: List[str] = Field(
        default_factory=list,
        description="List of checks that passed"
    )
    checks_failed: List[str] = Field(
        default_factory=list,
        description="List of checks that failed"
    )

    # Gate metadata
    gate_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When gate was evaluated"
    )
    evaluation_time_ms: float = Field(default=0.0, description="Evaluation duration in ms")

    # Audit
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit")

    @property
    def is_allowed(self) -> bool:
        """Check if action is allowed."""
        return self.decision in (GateDecision.ALLOW, GateDecision.CLAMP)

    @property
    def is_blocked(self) -> bool:
        """Check if action is blocked."""
        return self.decision == GateDecision.BLOCK

    def compute_provenance(self) -> str:
        """Compute provenance hash."""
        hash_input = (
            f"{self.gate_timestamp}|{self.decision}|"
            f"{self.original_request.tag_id}|{self.original_request.value}|"
            f"{self.final_value}|{len(self.violations)}"
        )
        return hashlib.sha256(hash_input.encode()).hexdigest()

    class Config:
        """Pydantic config."""
        use_enum_values = True


class SISState(BaseModel):
    """Safety Instrumented System state."""

    sis_id: str = Field(..., description="SIS identifier")
    is_active: bool = Field(default=True, description="Whether SIS is active")
    is_healthy: bool = Field(default=True, description="Whether SIS is healthy")
    last_heartbeat: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last heartbeat from SIS"
    )
    trip_status: bool = Field(default=False, description="Whether SIS is in trip state")
    bypass_active: bool = Field(default=False, description="Whether bypass is active")

    @property
    def is_operational(self) -> bool:
        """Check if SIS is operational."""
        return self.is_active and self.is_healthy and not self.bypass_active


class InterlockState(BaseModel):
    """Interlock state information."""

    interlock_id: str = Field(..., description="Interlock identifier")
    is_permissive: bool = Field(default=True, description="Whether interlock is permissive")
    cause: Optional[str] = Field(None, description="Cause if not permissive")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="State timestamp")


class AlarmState(BaseModel):
    """Alarm state information."""

    alarm_id: str = Field(..., description="Alarm identifier")
    tag_id: str = Field(..., description="Associated tag ID")
    is_active: bool = Field(default=False, description="Whether alarm is active")
    priority: str = Field(default="LOW", description="Alarm priority")
    acknowledged: bool = Field(default=False, description="Whether alarm is acknowledged")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Alarm timestamp")


class SafetyState(BaseModel):
    """
    Overall system safety state.

    Aggregates all safety-related state information for the system.

    Attributes:
        timestamp: Current state timestamp
        overall_safe: Whether system is in safe state
        sis_states: States of all SIS systems
        interlock_states: States of all interlocks
        alarm_states: States of active alarms
        active_violations: Currently active violations
        policies_enabled: Number of enabled policies
        policies_total: Total number of policies
        last_gate_result: Result of last action gate evaluation
        safety_level: Current safety integrity level

    Example:
        >>> state = safety_engine.get_state()
        >>> if state.overall_safe:
        ...     proceed_with_optimization()
    """

    timestamp: datetime = Field(default_factory=datetime.utcnow, description="State timestamp")
    overall_safe: bool = Field(default=True, description="Whether system is overall safe")

    # Component states
    sis_states: List[SISState] = Field(default_factory=list, description="SIS states")
    interlock_states: List[InterlockState] = Field(
        default_factory=list,
        description="Interlock states"
    )
    alarm_states: List[AlarmState] = Field(default_factory=list, description="Active alarms")

    # Violation tracking
    active_violations: List[BoundaryViolation] = Field(
        default_factory=list,
        description="Currently active violations"
    )
    violations_last_hour: int = Field(default=0, description="Violations in last hour")
    violations_last_24h: int = Field(default=0, description="Violations in last 24 hours")

    # Policy status
    policies_enabled: int = Field(default=0, description="Number of enabled policies")
    policies_total: int = Field(default=0, description="Total number of policies")

    # Last gate result
    last_gate_result: Optional[ActionGateResult] = Field(
        None,
        description="Last action gate result"
    )

    # Safety level
    safety_level: SafetyLevel = Field(
        default=SafetyLevel.SIL_2,
        description="Current safety integrity level"
    )

    @property
    def all_sis_operational(self) -> bool:
        """Check if all SIS systems are operational."""
        return all(sis.is_operational for sis in self.sis_states)

    @property
    def all_interlocks_permissive(self) -> bool:
        """Check if all interlocks are permissive."""
        return all(interlock.is_permissive for interlock in self.interlock_states)

    @property
    def has_critical_alarms(self) -> bool:
        """Check if there are any critical alarms."""
        return any(
            alarm.is_active and alarm.priority in ("CRITICAL", "EMERGENCY")
            for alarm in self.alarm_states
        )

    def compute_overall_safe(self) -> bool:
        """Compute overall safety status."""
        return (
            self.all_sis_operational and
            self.all_interlocks_permissive and
            not self.has_critical_alarms and
            len(self.active_violations) == 0
        )

    class Config:
        """Pydantic config."""
        use_enum_values = True


class SafetyAuditRecord(BaseModel):
    """
    Immutable audit record for safety events.

    Used for regulatory compliance and incident investigation.

    Attributes:
        record_id: Unique record identifier
        timestamp: When event occurred
        event_type: Type of safety event
        violation: Associated violation if any
        gate_result: Associated gate result if any
        action_taken: Action taken in response
        operator_notified: Whether operator was notified
        provenance_hash: SHA-256 hash for immutability
        previous_hash: Hash of previous record in chain
    """

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique record identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Event timestamp"
    )
    event_type: str = Field(..., description="Type of safety event")

    # Associated data
    violation: Optional[BoundaryViolation] = Field(None, description="Associated violation")
    gate_result: Optional[ActionGateResult] = Field(None, description="Associated gate result")

    # Response
    action_taken: str = Field(..., description="Action taken in response")
    operator_notified: bool = Field(default=False, description="Whether operator was notified")
    escalation_level: int = Field(default=0, description="Escalation level (0=none)")

    # Audit chain
    provenance_hash: str = Field(default="", description="SHA-256 hash of this record")
    previous_hash: str = Field(default="", description="Hash of previous record in chain")

    @validator("provenance_hash", always=True)
    def compute_provenance_hash(cls, v: str, values: Dict[str, Any]) -> str:
        """Compute provenance hash if not provided."""
        if v:
            return v

        hash_input = (
            f"{values.get('timestamp', '')}|"
            f"{values.get('event_type', '')}|"
            f"{values.get('action_taken', '')}|"
            f"{values.get('previous_hash', '')}"
        )
        return hashlib.sha256(hash_input.encode()).hexdigest()

    class Config:
        """Pydantic config - audit records are immutable."""
        frozen = True


# Type aliases for convenience
PolicyList = List[BoundaryPolicy]
ViolationList = List[BoundaryViolation]
TagValue = Union[float, int]
