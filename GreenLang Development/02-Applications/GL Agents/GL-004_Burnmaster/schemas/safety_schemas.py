"""
Safety Schemas - Data models for safety-related operations.

This module defines Pydantic models for safety limits, envelopes, checks,
interlocks, and hazard levels. These schemas ensure combustion systems
operate safely within defined boundaries.

Example:
    >>> from safety_schemas import SafetyLimit, HazardLevel
    >>> limit = SafetyLimit(
    ...     parameter="furnace_temperature_celsius",
    ...     min_value=200.0,
    ...     max_value=1200.0,
    ...     unit="C"
    ... )
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator, computed_field
import hashlib


class HazardLevel(str, Enum):
    """
    Hazard severity level classification.

    NONE: No hazard detected
    LOW: Minor concern, continue monitoring
    MEDIUM: Elevated risk, consider corrective action
    HIGH: Significant risk, immediate attention required
    CRITICAL: Imminent danger, emergency response required
    """
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyCategory(str, Enum):
    """Category of safety concern."""
    PROCESS = "process"              # Process safety (temperatures, pressures)
    EMISSIONS = "emissions"          # Emissions compliance
    EQUIPMENT = "equipment"          # Equipment protection
    PERSONNEL = "personnel"          # Personnel safety
    ENVIRONMENTAL = "environmental"  # Environmental protection
    REGULATORY = "regulatory"        # Regulatory compliance


class InterlockType(str, Enum):
    """Type of safety interlock."""
    PERMISSIVE = "permissive"    # Must be satisfied to proceed
    TRIP = "trip"                # Triggers shutdown when violated
    ALARM = "alarm"             # Generates alarm but doesn't trip
    MONITORING = "monitoring"    # Monitoring only, no automatic action


class InterlockState(str, Enum):
    """Current state of an interlock."""
    NORMAL = "normal"           # Interlock condition satisfied
    ALARMING = "alarming"       # Alarm condition, not yet tripped
    TRIPPED = "tripped"         # Interlock has tripped
    BYPASSED = "bypassed"       # Interlock manually bypassed
    FAILED = "failed"           # Interlock sensor/system failure
    UNKNOWN = "unknown"         # State cannot be determined


class SafetyCheckType(str, Enum):
    """Type of safety check."""
    LIMIT_CHECK = "limit_check"           # Value within limits
    RATE_OF_CHANGE = "rate_of_change"     # Rate of change acceptable
    INTERLOCK = "interlock"               # Interlock status
    PERMISSIVE = "permissive"             # Permissive conditions met
    EQUIPMENT_STATUS = "equipment_status"  # Equipment operational
    SENSOR_VALIDITY = "sensor_validity"    # Sensor data valid
    CALCULATION = "calculation"            # Calculated safety parameter


class SafetyLimit(BaseModel):
    """
    Safety limit definition for a single parameter.

    Defines the safe operating range for a process parameter with
    multiple alarm and trip thresholds.

    Attributes:
        parameter: Name of the monitored parameter
        min_value: Minimum safe value
        max_value: Maximum safe value
        unit: Engineering unit
    """
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    limit_id: str = Field(..., min_length=1, max_length=100, description="Unique identifier for this limit")
    parameter: str = Field(..., min_length=1, max_length=100, description="Parameter name being limited")
    description: str = Field(default="", max_length=500, description="Human-readable description")
    category: SafetyCategory = Field(default=SafetyCategory.PROCESS, description="Safety category")

    # Primary limits
    min_value: Optional[float] = Field(default=None, description="Minimum safe operating value")
    max_value: Optional[float] = Field(default=None, description="Maximum safe operating value")
    unit: str = Field(..., min_length=1, max_length=50, description="Engineering unit")

    # Low alarm/trip thresholds (below min_value)
    low_alarm: Optional[float] = Field(default=None, description="Low alarm threshold")
    low_low_alarm: Optional[float] = Field(default=None, description="Low-low alarm threshold")
    low_trip: Optional[float] = Field(default=None, description="Low trip threshold (emergency)")

    # High alarm/trip thresholds (above max_value)
    high_alarm: Optional[float] = Field(default=None, description="High alarm threshold")
    high_high_alarm: Optional[float] = Field(default=None, description="High-high alarm threshold")
    high_trip: Optional[float] = Field(default=None, description="High trip threshold (emergency)")

    # Response times
    alarm_delay_s: float = Field(default=0.0, ge=0.0, le=300.0, description="Delay before alarm activates")
    trip_delay_s: float = Field(default=0.0, ge=0.0, le=60.0, description="Delay before trip activates")

    # Deadband for alarm reset
    deadband_percent: float = Field(default=2.0, ge=0.0, le=20.0, description="Deadband for alarm reset as percentage")

    # Metadata
    hazard_level: HazardLevel = Field(default=HazardLevel.MEDIUM, description="Hazard level if violated")
    source: str = Field(default="", max_length=200, description="Source of this limit (standard, OEM, etc.)")
    active: bool = Field(default=True, description="Whether this limit is active")
    version: str = Field(default="1.0", description="Limit definition version")

    @model_validator(mode='after')
    def validate_limit_ordering(self) -> 'SafetyLimit':
        """Validate that limits are properly ordered."""
        # Check that we have at least one limit
        if self.min_value is None and self.max_value is None:
            raise ValueError("At least one of min_value or max_value must be specified")

        # Check min < max
        if self.min_value is not None and self.max_value is not None:
            if self.min_value >= self.max_value:
                raise ValueError("min_value must be less than max_value")

        # Check low alarm ordering: low_trip < low_low_alarm < low_alarm < min_value
        if self.min_value is not None:
            if self.low_alarm is not None and self.low_alarm >= self.min_value:
                raise ValueError("low_alarm must be less than min_value")
            if self.low_low_alarm is not None and self.low_alarm is not None:
                if self.low_low_alarm >= self.low_alarm:
                    raise ValueError("low_low_alarm must be less than low_alarm")
            if self.low_trip is not None and self.low_low_alarm is not None:
                if self.low_trip >= self.low_low_alarm:
                    raise ValueError("low_trip must be less than low_low_alarm")

        # Check high alarm ordering: max_value < high_alarm < high_high_alarm < high_trip
        if self.max_value is not None:
            if self.high_alarm is not None and self.high_alarm <= self.max_value:
                raise ValueError("high_alarm must be greater than max_value")
            if self.high_high_alarm is not None and self.high_alarm is not None:
                if self.high_high_alarm <= self.high_alarm:
                    raise ValueError("high_high_alarm must be greater than high_alarm")
            if self.high_trip is not None and self.high_high_alarm is not None:
                if self.high_trip <= self.high_high_alarm:
                    raise ValueError("high_trip must be greater than high_high_alarm")

        return self

    def check_value(self, value: float) -> Dict[str, Any]:
        """
        Check a value against all thresholds.

        Args:
            value: Value to check

        Returns:
            Dictionary with status and any violations
        """
        result = {
            "value": value,
            "in_range": True,
            "alarm_status": "normal",
            "violations": []
        }

        # Check trip thresholds first (most severe)
        if self.low_trip is not None and value <= self.low_trip:
            result["alarm_status"] = "trip_low"
            result["in_range"] = False
            result["violations"].append(f"Below low trip threshold ({self.low_trip})")
        elif self.high_trip is not None and value >= self.high_trip:
            result["alarm_status"] = "trip_high"
            result["in_range"] = False
            result["violations"].append(f"Above high trip threshold ({self.high_trip})")
        # Check high-high/low-low alarms
        elif self.low_low_alarm is not None and value <= self.low_low_alarm:
            result["alarm_status"] = "alarm_low_low"
            result["in_range"] = False
            result["violations"].append(f"Below low-low alarm ({self.low_low_alarm})")
        elif self.high_high_alarm is not None and value >= self.high_high_alarm:
            result["alarm_status"] = "alarm_high_high"
            result["in_range"] = False
            result["violations"].append(f"Above high-high alarm ({self.high_high_alarm})")
        # Check regular alarms
        elif self.low_alarm is not None and value <= self.low_alarm:
            result["alarm_status"] = "alarm_low"
            result["in_range"] = False
            result["violations"].append(f"Below low alarm ({self.low_alarm})")
        elif self.high_alarm is not None and value >= self.high_alarm:
            result["alarm_status"] = "alarm_high"
            result["in_range"] = False
            result["violations"].append(f"Above high alarm ({self.high_alarm})")
        # Check normal operating range
        elif self.min_value is not None and value < self.min_value:
            result["alarm_status"] = "below_range"
            result["in_range"] = False
            result["violations"].append(f"Below minimum ({self.min_value})")
        elif self.max_value is not None and value > self.max_value:
            result["alarm_status"] = "above_range"
            result["in_range"] = False
            result["violations"].append(f"Above maximum ({self.max_value})")

        return result

    def get_margin_to_limit(self, value: float) -> Dict[str, float]:
        """Calculate margin to nearest limit."""
        margins = {}
        if self.min_value is not None:
            margins["to_min"] = value - self.min_value
        if self.max_value is not None:
            margins["to_max"] = self.max_value - value
        if self.low_trip is not None:
            margins["to_low_trip"] = value - self.low_trip
        if self.high_trip is not None:
            margins["to_high_trip"] = self.high_trip - value
        return margins


class SafetyEnvelope(BaseModel):
    """
    Collection of safety limits defining the safe operating envelope.

    The safety envelope encompasses all safety limits that must be
    respected during operation.
    """
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    envelope_id: str = Field(..., min_length=1, max_length=50, description="Unique envelope identifier")
    name: str = Field(..., min_length=1, max_length=200, description="Human-readable envelope name")
    description: str = Field(default="", max_length=1000, description="Detailed description")

    # Safety limits
    limits: List[SafetyLimit] = Field(default_factory=list, description="List of safety limits in this envelope")

    # Metadata
    category: SafetyCategory = Field(default=SafetyCategory.PROCESS, description="Primary safety category")
    hazard_level: HazardLevel = Field(default=HazardLevel.MEDIUM, description="Overall hazard level")
    version: str = Field(default="1.0", description="Envelope version")
    effective_date: Optional[datetime] = Field(default=None, description="When envelope became effective")
    approved_by: Optional[str] = Field(default=None, max_length=100, description="Approval authority")
    review_date: Optional[datetime] = Field(default=None, description="Next review date")

    # Status
    active: bool = Field(default=True, description="Whether envelope is currently active")

    @field_validator('limits')
    @classmethod
    def validate_unique_limit_ids(cls, v: List[SafetyLimit]) -> List[SafetyLimit]:
        """Ensure no duplicate limit IDs."""
        ids = [limit.limit_id for limit in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate limit IDs in envelope")
        return v

    def get_limit(self, limit_id: str) -> Optional[SafetyLimit]:
        """Get a specific limit by ID."""
        for limit in self.limits:
            if limit.limit_id == limit_id:
                return limit
        return None

    def get_limit_for_parameter(self, parameter: str) -> Optional[SafetyLimit]:
        """Get the limit for a specific parameter."""
        for limit in self.limits:
            if limit.parameter == parameter and limit.active:
                return limit
        return None

    def check_all_values(self, values: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Check all values against their limits."""
        results = {}
        for limit in self.limits:
            if limit.active and limit.parameter in values:
                results[limit.parameter] = limit.check_value(values[limit.parameter])
        return results

    def is_within_envelope(self, values: Dict[str, float]) -> bool:
        """Check if all values are within the safety envelope."""
        results = self.check_all_values(values)
        return all(r["in_range"] for r in results.values())

    def get_active_limits(self) -> List[SafetyLimit]:
        """Get all active limits."""
        return [limit for limit in self.limits if limit.active]


class SafetyCheck(BaseModel):
    """
    Result of a single safety check.

    Represents the outcome of checking a safety condition with
    pass/fail status and detailed margin information.

    Attributes:
        check_type: Type of safety check performed
        passed: Whether the check passed
        margin: Safety margin (distance from limit)
        message: Human-readable result message
    """
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    check_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[:17], description="Unique check ID")
    check_type: SafetyCheckType = Field(..., description="Type of safety check")
    category: SafetyCategory = Field(default=SafetyCategory.PROCESS, description="Safety category")

    # Check target
    parameter: str = Field(..., min_length=1, max_length=100, description="Parameter being checked")
    value: Optional[float] = Field(default=None, description="Value that was checked")
    limit: Optional[float] = Field(default=None, description="Limit that was checked against")

    # Results
    passed: bool = Field(..., description="Whether the safety check passed")
    margin: Optional[float] = Field(default=None, description="Margin from limit (positive = safe)")
    margin_percent: Optional[float] = Field(default=None, description="Margin as percentage of limit")

    # Messages
    message: str = Field(..., min_length=1, max_length=500, description="Human-readable result message")
    details: Optional[str] = Field(default=None, max_length=1000, description="Additional details")

    # Severity
    hazard_level: HazardLevel = Field(default=HazardLevel.NONE, description="Hazard level if check failed")

    # Timing
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When check was performed")
    check_duration_ms: Optional[float] = Field(default=None, ge=0.0, description="Check duration in milliseconds")

    @computed_field
    @property
    def is_critical(self) -> bool:
        """Check if this is a critical failure."""
        return not self.passed and self.hazard_level in [HazardLevel.HIGH, HazardLevel.CRITICAL]


class InterlockStatus(BaseModel):
    """
    Status of a safety interlock.

    Captures the current state and history of a safety interlock
    for monitoring and audit purposes.

    Attributes:
        interlock_id: Unique identifier for the interlock
        active: Whether the interlock is currently active (tripped)
        reason: Reason for current state
    """
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    interlock_id: str = Field(..., min_length=1, max_length=100, description="Unique interlock identifier")
    name: str = Field(..., min_length=1, max_length=200, description="Human-readable interlock name")
    description: str = Field(default="", max_length=500, description="Detailed description")

    # Interlock type and state
    interlock_type: InterlockType = Field(..., description="Type of interlock")
    state: InterlockState = Field(default=InterlockState.NORMAL, description="Current interlock state")
    active: bool = Field(default=False, description="Whether interlock is currently active/tripped")

    # Trigger information
    trigger_parameter: Optional[str] = Field(default=None, max_length=100, description="Parameter that triggers interlock")
    trigger_value: Optional[float] = Field(default=None, description="Value that triggered interlock")
    trigger_threshold: Optional[float] = Field(default=None, description="Threshold for triggering")

    # State reason
    reason: str = Field(default="", max_length=500, description="Reason for current state")

    # Bypass information
    bypassed: bool = Field(default=False, description="Whether interlock is currently bypassed")
    bypass_reason: Optional[str] = Field(default=None, max_length=500, description="Reason for bypass")
    bypassed_by: Optional[str] = Field(default=None, max_length=100, description="User who bypassed")
    bypass_expires: Optional[datetime] = Field(default=None, description="When bypass expires")

    # Timing
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of this status")
    last_trip_time: Optional[datetime] = Field(default=None, description="When interlock last tripped")
    last_reset_time: Optional[datetime] = Field(default=None, description="When interlock was last reset")

    # Counters
    trip_count: int = Field(default=0, ge=0, description="Number of times this interlock has tripped")
    trip_count_24h: int = Field(default=0, ge=0, description="Trip count in last 24 hours")

    # Hazard level
    hazard_level: HazardLevel = Field(default=HazardLevel.HIGH, description="Hazard level if interlock trips")

    @computed_field
    @property
    def is_healthy(self) -> bool:
        """Check if interlock is in healthy state."""
        return self.state == InterlockState.NORMAL and not self.bypassed

    @computed_field
    @property
    def requires_attention(self) -> bool:
        """Check if interlock requires operator attention."""
        return (self.state in [InterlockState.TRIPPED, InterlockState.FAILED, InterlockState.UNKNOWN] or
                self.bypassed or self.trip_count_24h > 3)


class SafetyAssessment(BaseModel):
    """Comprehensive safety assessment result."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    assessment_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f"), description="Unique assessment ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Assessment timestamp")

    # Overall status
    overall_status: HazardLevel = Field(default=HazardLevel.NONE, description="Overall hazard level")
    safe_to_operate: bool = Field(default=True, description="Whether safe to continue operation")
    requires_action: bool = Field(default=False, description="Whether action is required")

    # Individual checks
    safety_checks: List[SafetyCheck] = Field(default_factory=list, description="Individual safety check results")
    interlock_statuses: List[InterlockStatus] = Field(default_factory=list, description="Interlock status results")

    # Summary counts
    checks_passed: int = Field(default=0, ge=0, description="Number of checks that passed")
    checks_failed: int = Field(default=0, ge=0, description="Number of checks that failed")
    interlocks_normal: int = Field(default=0, ge=0, description="Number of interlocks in normal state")
    interlocks_tripped: int = Field(default=0, ge=0, description="Number of tripped interlocks")
    interlocks_bypassed: int = Field(default=0, ge=0, description="Number of bypassed interlocks")

    # Recommendations
    recommendations: List[str] = Field(default_factory=list, description="Safety recommendations")
    required_actions: List[str] = Field(default_factory=list, description="Required actions")

    # Provenance
    provenance_hash: Optional[str] = Field(default=None, description="SHA-256 hash for audit")
    assessed_by: str = Field(default="BURNMASTER", max_length=100, description="System/user that performed assessment")

    @computed_field
    @property
    def critical_failures(self) -> List[SafetyCheck]:
        """Get list of critical safety check failures."""
        return [check for check in self.safety_checks if check.is_critical]

    def calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash for audit trail."""
        import json
        data = {
            "assessment_id": self.assessment_id,
            "timestamp": self.timestamp.isoformat(),
            "overall_status": self.overall_status.value,
            "safe_to_operate": self.safe_to_operate,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()


class EmergencyAction(BaseModel):
    """Definition of an emergency action to take."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    action_id: str = Field(..., min_length=1, max_length=50, description="Unique action identifier")
    name: str = Field(..., min_length=1, max_length=200, description="Action name")
    description: str = Field(..., max_length=1000, description="Detailed description")

    # Action details
    action_type: str = Field(..., max_length=50, description="Type of emergency action")
    target_equipment: List[str] = Field(default_factory=list, description="Equipment affected by action")
    setpoints_to_change: Dict[str, float] = Field(default_factory=dict, description="Setpoint changes to make")

    # Priority and timing
    priority: int = Field(default=1, ge=1, le=10, description="Priority (1=highest)")
    execution_time_limit_s: float = Field(default=60.0, ge=0.0, description="Maximum execution time")

    # Triggers
    trigger_conditions: List[str] = Field(default_factory=list, description="Conditions that trigger this action")
    hazard_levels: List[HazardLevel] = Field(default_factory=list, description="Hazard levels that trigger this action")

    # Metadata
    requires_confirmation: bool = Field(default=False, description="Whether operator confirmation required")
    active: bool = Field(default=True, description="Whether this action is active")


__all__ = [
    "HazardLevel",
    "SafetyCategory",
    "InterlockType",
    "InterlockState",
    "SafetyCheckType",
    "SafetyLimit",
    "SafetyEnvelope",
    "SafetyCheck",
    "InterlockStatus",
    "SafetyAssessment",
    "EmergencyAction",
]
