"""
Control Schemas - Data models for control system operations.

This module defines Pydantic models for operating modes, setpoint writes,
mode transitions, and control cycle results. These schemas support the
graduated autonomy control system (OBSERVE -> ADVISORY -> CLOSED_LOOP).

Example:
    >>> from control_schemas import OperatingMode, SetpointWrite
    >>> write = SetpointWrite(
    ...     tag="FC-101.SP",
    ...     value=2.5,
    ...     reason="Optimization recommendation"
    ... )
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator, computed_field
import hashlib


class OperatingMode(str, Enum):
    """
    Control system operating modes with graduated autonomy.

    OBSERVE: Read-only monitoring, no control actions
    ADVISORY: Generates recommendations but requires operator approval
    CLOSED_LOOP: Automatic setpoint adjustments within safety limits
    FALLBACK: Safe degraded operation mode for error conditions
    """
    OBSERVE = "observe"
    ADVISORY = "advisory"
    CLOSED_LOOP = "closed_loop"
    FALLBACK = "fallback"
    MANUAL = "manual"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"


class WriteStatus(str, Enum):
    """Status of a setpoint write operation."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class ActionType(str, Enum):
    """Type of control action."""
    SETPOINT_CHANGE = "setpoint_change"
    MODE_CHANGE = "mode_change"
    ALARM_ACKNOWLEDGEMENT = "alarm_acknowledgement"
    EQUIPMENT_START = "equipment_start"
    EQUIPMENT_STOP = "equipment_stop"
    EMERGENCY_STOP = "emergency_stop"
    INTERLOCK_OVERRIDE = "interlock_override"
    PARAMETER_TUNE = "parameter_tune"


class ControlAuthority(str, Enum):
    """Level of control authority."""
    SYSTEM = "system"      # Automated system action
    OPERATOR = "operator"  # Operator-initiated action
    ENGINEER = "engineer"  # Engineering override
    ADMIN = "admin"        # Administrative override
    SAFETY = "safety"      # Safety system action (highest priority)


class AuditContext(BaseModel):
    """Audit context for tracking who/what/why for control actions."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    user_id: Optional[str] = Field(default=None, max_length=100, description="User ID who initiated action")
    user_name: Optional[str] = Field(default=None, max_length=200, description="User name for display")
    authority: ControlAuthority = Field(default=ControlAuthority.SYSTEM, description="Authority level")
    session_id: Optional[str] = Field(default=None, max_length=100, description="Session identifier")
    source_system: str = Field(default="BURNMASTER", max_length=100, description="System that generated this action")
    reason: str = Field(default="", max_length=1000, description="Reason for the action")
    ticket_number: Optional[str] = Field(default=None, max_length=50, description="Associated change ticket")
    optimization_id: Optional[str] = Field(default=None, max_length=50, description="Associated optimization result ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Audit timestamp")


class SetpointWrite(BaseModel):
    """
    Setpoint write request to control system.

    Represents a request to change a setpoint value with full audit context
    and safety validation status.

    Attributes:
        tag: Tag/variable name to write
        value: New setpoint value
        timestamp: When the write was requested
        audit_context: Full audit trail information
    """
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    write_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f"), description="Unique write request ID")
    tag: str = Field(..., min_length=1, max_length=100, description="Tag/variable name to write")
    value: float = Field(..., description="New setpoint value to write")
    unit: str = Field(default="", max_length=50, description="Engineering unit")

    # Previous value for audit
    previous_value: Optional[float] = Field(default=None, description="Previous value before write")

    # Bounds checking
    min_limit: Optional[float] = Field(default=None, description="Minimum allowed value")
    max_limit: Optional[float] = Field(default=None, description="Maximum allowed value")

    # Status and timing
    status: WriteStatus = Field(default=WriteStatus.PENDING, description="Current status of write request")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When write was requested")
    execution_timestamp: Optional[datetime] = Field(default=None, description="When write was executed")
    expiration_timestamp: Optional[datetime] = Field(default=None, description="When write request expires")

    # Audit and approval
    audit_context: AuditContext = Field(default_factory=AuditContext, description="Full audit context")
    requires_approval: bool = Field(default=True, description="Whether operator approval is required")
    approved_by: Optional[str] = Field(default=None, max_length=100, description="User who approved the write")
    approval_timestamp: Optional[datetime] = Field(default=None, description="When write was approved")

    # Safety validation
    safety_validated: bool = Field(default=False, description="Whether safety checks passed")
    safety_check_results: List[str] = Field(default_factory=list, description="Results of safety checks")

    # Error handling
    error_message: Optional[str] = Field(default=None, max_length=500, description="Error message if write failed")
    retry_count: int = Field(default=0, ge=0, description="Number of retry attempts")

    @computed_field
    @property
    def change_amount(self) -> Optional[float]:
        """Calculate change amount if previous value known."""
        if self.previous_value is not None:
            return self.value - self.previous_value
        return None

    @computed_field
    @property
    def change_percent(self) -> Optional[float]:
        """Calculate percentage change if previous value known."""
        if self.previous_value is not None and self.previous_value != 0:
            return ((self.value - self.previous_value) / self.previous_value) * 100.0
        return None

    @computed_field
    @property
    def is_within_limits(self) -> bool:
        """Check if value is within limits."""
        if self.min_limit is not None and self.value < self.min_limit:
            return False
        if self.max_limit is not None and self.value > self.max_limit:
            return False
        return True

    @computed_field
    @property
    def is_expired(self) -> bool:
        """Check if write request has expired."""
        if self.expiration_timestamp is None:
            return False
        return datetime.utcnow() > self.expiration_timestamp

    def approve(self, approver: str) -> None:
        """Approve the write request."""
        self.status = WriteStatus.APPROVED
        self.approved_by = approver
        self.approval_timestamp = datetime.utcnow()

    def reject(self, reason: str) -> None:
        """Reject the write request."""
        self.status = WriteStatus.REJECTED
        self.error_message = reason

    def execute(self) -> None:
        """Mark write as executed."""
        self.status = WriteStatus.EXECUTED
        self.execution_timestamp = datetime.utcnow()

    def fail(self, error: str) -> None:
        """Mark write as failed."""
        self.status = WriteStatus.FAILED
        self.error_message = error

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash for audit."""
        data = f"{self.write_id}:{self.tag}:{self.value}:{self.timestamp.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()


class ModeTransitionTrigger(str, Enum):
    """Trigger for mode transition."""
    OPERATOR_REQUEST = "operator_request"
    AUTOMATIC = "automatic"
    SAFETY_INTERLOCK = "safety_interlock"
    FAULT_CONDITION = "fault_condition"
    SCHEDULED = "scheduled"
    STARTUP_SEQUENCE = "startup_sequence"
    SHUTDOWN_SEQUENCE = "shutdown_sequence"
    PERFORMANCE_THRESHOLD = "performance_threshold"


class ModeTransition(BaseModel):
    """
    Record of a mode transition.

    Captures all details of a transition between operating modes
    for audit and debugging purposes.

    Attributes:
        from_mode: Mode before transition
        to_mode: Mode after transition
        trigger: What triggered the transition
        timestamp: When transition occurred
    """
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    transition_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f"), description="Unique transition ID")
    from_mode: OperatingMode = Field(..., description="Operating mode before transition")
    to_mode: OperatingMode = Field(..., description="Operating mode after transition")
    trigger: ModeTransitionTrigger = Field(..., description="What triggered the transition")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When transition occurred")

    # Transition details
    success: bool = Field(default=True, description="Whether transition succeeded")
    duration_ms: Optional[float] = Field(default=None, ge=0.0, description="Transition duration in milliseconds")
    error_message: Optional[str] = Field(default=None, max_length=500, description="Error if transition failed")

    # Conditions at transition
    conditions: Dict[str, Any] = Field(default_factory=dict, description="System conditions at transition")
    prerequisites_met: List[str] = Field(default_factory=list, description="Prerequisites that were verified")
    prerequisites_failed: List[str] = Field(default_factory=list, description="Prerequisites that failed")

    # Audit
    audit_context: AuditContext = Field(default_factory=AuditContext, description="Full audit context")
    rollback_available: bool = Field(default=True, description="Whether rollback to previous mode is available")

    @model_validator(mode='after')
    def validate_mode_change(self) -> 'ModeTransition':
        """Validate that mode actually changed."""
        if self.from_mode == self.to_mode:
            raise ValueError("from_mode and to_mode must be different")
        return self

    @computed_field
    @property
    def is_escalation(self) -> bool:
        """Check if this is an escalation to higher autonomy."""
        mode_order = {
            OperatingMode.MANUAL: 0,
            OperatingMode.OBSERVE: 1,
            OperatingMode.ADVISORY: 2,
            OperatingMode.CLOSED_LOOP: 3
        }
        from_order = mode_order.get(self.from_mode, 0)
        to_order = mode_order.get(self.to_mode, 0)
        return to_order > from_order

    @computed_field
    @property
    def is_degradation(self) -> bool:
        """Check if this is a degradation to lower autonomy."""
        mode_order = {
            OperatingMode.MANUAL: 0,
            OperatingMode.OBSERVE: 1,
            OperatingMode.ADVISORY: 2,
            OperatingMode.CLOSED_LOOP: 3
        }
        from_order = mode_order.get(self.from_mode, 0)
        to_order = mode_order.get(self.to_mode, 0)
        return to_order < from_order


class ControlAction(BaseModel):
    """Individual control action within a control cycle."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    action_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[:17], description="Unique action ID")
    action_type: ActionType = Field(..., description="Type of control action")
    target: str = Field(..., min_length=1, max_length=100, description="Target tag/equipment/parameter")
    value: Optional[float] = Field(default=None, description="Value for setpoint changes")
    previous_value: Optional[float] = Field(default=None, description="Previous value")
    success: bool = Field(default=True, description="Whether action succeeded")
    error_message: Optional[str] = Field(default=None, max_length=500, description="Error message if failed")
    execution_time_ms: Optional[float] = Field(default=None, ge=0.0, description="Execution time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When action was executed")


class StateSnapshot(BaseModel):
    """Snapshot of system state at a point in time."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    snapshot_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f"), description="Unique snapshot ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When snapshot was taken")

    # Operating mode
    operating_mode: OperatingMode = Field(..., description="Current operating mode")

    # Key process values
    fuel_flow_kg_per_s: Optional[float] = Field(default=None, description="Fuel flow rate")
    air_flow_kg_per_s: Optional[float] = Field(default=None, description="Air flow rate")
    o2_percent: Optional[float] = Field(default=None, description="O2 concentration")
    co_ppm: Optional[float] = Field(default=None, description="CO concentration")
    nox_ppm: Optional[float] = Field(default=None, description="NOx concentration")

    # Performance metrics
    efficiency_percent: Optional[float] = Field(default=None, description="Current efficiency")
    stability_score: Optional[float] = Field(default=None, description="Stability score")
    firing_rate_mw: Optional[float] = Field(default=None, description="Current firing rate")

    # Setpoints
    setpoints: Dict[str, float] = Field(default_factory=dict, description="Current setpoint values")

    # Additional values
    additional_values: Dict[str, Any] = Field(default_factory=dict, description="Additional state values")

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash for state verification."""
        import json
        data = self.model_dump(exclude={'snapshot_id', 'timestamp'})
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()


class ControlCycleResult(BaseModel):
    """
    Result of a single control cycle execution.

    Captures all actions taken, state before and after, and
    performance metrics for audit and analysis.

    Attributes:
        actions_taken: List of control actions executed
        state_before: System state before control cycle
        state_after: System state after control cycle
    """
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    cycle_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f"), description="Unique cycle identifier")
    cycle_number: int = Field(default=0, ge=0, description="Sequential cycle number")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When cycle was executed")

    # Mode and timing
    operating_mode: OperatingMode = Field(..., description="Operating mode during this cycle")
    cycle_time_ms: float = Field(..., ge=0.0, description="Total cycle execution time in milliseconds")

    # State snapshots
    state_before: StateSnapshot = Field(..., description="System state before control cycle")
    state_after: StateSnapshot = Field(..., description="System state after control cycle")

    # Actions
    actions_taken: List[ControlAction] = Field(default_factory=list, description="List of actions executed in this cycle")
    actions_pending: List[SetpointWrite] = Field(default_factory=list, description="Actions pending approval")
    recommendations_generated: int = Field(default=0, ge=0, description="Number of recommendations generated")

    # Outcomes
    success: bool = Field(default=True, description="Whether cycle completed successfully")
    errors: List[str] = Field(default_factory=list, description="List of errors encountered")
    warnings: List[str] = Field(default_factory=list, description="List of warnings generated")

    # Performance metrics
    optimization_run: bool = Field(default=False, description="Whether optimization was run this cycle")
    optimization_time_ms: Optional[float] = Field(default=None, ge=0.0, description="Optimization execution time")
    model_inference_time_ms: Optional[float] = Field(default=None, ge=0.0, description="ML model inference time")

    # Cost metrics
    instantaneous_fuel_cost_per_hr: Optional[float] = Field(default=None, description="Fuel cost rate at cycle end")
    instantaneous_emissions_kg_per_hr: Optional[float] = Field(default=None, description="Emissions rate at cycle end")

    # Provenance
    provenance_hash: Optional[str] = Field(default=None, description="SHA-256 hash for audit")

    @computed_field
    @property
    def action_count(self) -> int:
        """Count of actions taken."""
        return len(self.actions_taken)

    @computed_field
    @property
    def successful_actions(self) -> int:
        """Count of successful actions."""
        return sum(1 for a in self.actions_taken if a.success)

    @computed_field
    @property
    def failed_actions(self) -> int:
        """Count of failed actions."""
        return sum(1 for a in self.actions_taken if not a.success)

    @computed_field
    @property
    def efficiency_change(self) -> Optional[float]:
        """Calculate efficiency change during cycle."""
        if (self.state_before.efficiency_percent is not None and
            self.state_after.efficiency_percent is not None):
            return self.state_after.efficiency_percent - self.state_before.efficiency_percent
        return None

    @computed_field
    @property
    def stability_change(self) -> Optional[float]:
        """Calculate stability change during cycle."""
        if (self.state_before.stability_score is not None and
            self.state_after.stability_score is not None):
            return self.state_after.stability_score - self.state_before.stability_score
        return None

    def calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash for complete audit trail."""
        import json
        data = {
            "cycle_id": self.cycle_id,
            "cycle_number": self.cycle_number,
            "timestamp": self.timestamp.isoformat(),
            "operating_mode": self.operating_mode.value,
            "action_count": self.action_count,
            "state_before_hash": self.state_before.calculate_hash(),
            "state_after_hash": self.state_after.calculate_hash()
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()


class ControllerConfig(BaseModel):
    """Configuration for the control system."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    config_id: str = Field(default="default", min_length=1, max_length=50, description="Configuration identifier")
    name: str = Field(default="Default Controller Config", max_length=200, description="Configuration name")

    # Mode settings
    default_mode: OperatingMode = Field(default=OperatingMode.OBSERVE, description="Default operating mode on startup")
    allow_closed_loop: bool = Field(default=False, description="Whether closed-loop mode is enabled")
    require_approval_for_closed_loop: bool = Field(default=True, description="Require approval to enter closed-loop")

    # Cycle timing
    control_cycle_interval_s: float = Field(default=60.0, ge=1.0, le=3600.0, description="Control cycle interval in seconds")
    optimization_interval_s: float = Field(default=300.0, ge=10.0, le=3600.0, description="Optimization interval in seconds")

    # Setpoint limits
    max_setpoint_change_per_cycle_percent: float = Field(default=5.0, ge=0.1, le=50.0, description="Maximum setpoint change per cycle")
    setpoint_rate_limit_per_minute: float = Field(default=10.0, ge=0.0, le=100.0, description="Maximum setpoint changes per minute")

    # Safety
    safety_check_enabled: bool = Field(default=True, description="Enable safety checks before writes")
    auto_fallback_on_error: bool = Field(default=True, description="Automatically fallback on errors")

    # Audit
    audit_all_cycles: bool = Field(default=True, description="Audit every control cycle")
    retain_history_hours: int = Field(default=720, ge=1, le=8760, description="Hours to retain cycle history")

    # Version
    version: str = Field(default="1.0", description="Configuration version")
    effective_date: Optional[datetime] = Field(default=None, description="When this config became effective")


__all__ = [
    "OperatingMode",
    "WriteStatus",
    "ActionType",
    "ControlAuthority",
    "AuditContext",
    "SetpointWrite",
    "ModeTransitionTrigger",
    "ModeTransition",
    "ControlAction",
    "StateSnapshot",
    "ControlCycleResult",
    "ControllerConfig",
]
