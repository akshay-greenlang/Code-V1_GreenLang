"""
Safety Boundaries Module - Safety Boundary Enforcement

This module provides safety boundary enforcement for process heat
systems. It defines, validates, and enforces safety limits on
process variables with full provenance tracking.

Key Capabilities:
    - Safety limit definition and validation
    - Real-time boundary checking
    - Limit enforcement with clamping
    - Violation detection and logging
    - Multi-variable boundary sets
    - Dynamic limit adjustment

Standards:
    - IEC 61511: Safety Instrumented Systems
    - NFPA 86: Standard for Ovens and Furnaces
    - Process industry best practices

Example:
    >>> boundary = SafetyBoundary(limits={
    ...     "temperature": (0.0, 1200.0),
    ...     "pressure": (0.0, 15.0),
    ...     "flow_rate": (10.0, 500.0),
    ... })
    >>> status = boundary.check_value("temperature", 1150.0)
    >>> if status.is_alarm:
    ...     print(f"Temperature alarm: {status.message}")

CRITICAL: All boundary calculations are DETERMINISTIC. NO LLM calls permitted.
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class LimitType(str, Enum):
    """Types of safety limits."""
    ABSOLUTE = "absolute"           # Hard limit, never exceed
    OPERATING = "operating"         # Normal operating range
    ALARM = "alarm"                 # Alarm threshold
    WARNING = "warning"             # Warning threshold
    INTERLOCK = "interlock"         # Interlock trip point
    PERMISSIVE = "permissive"       # Permissive condition
    RATE_OF_CHANGE = "rate_of_change"  # Rate limit


class EnforcementAction(str, Enum):
    """Actions when limit is violated."""
    NONE = "none"                   # Log only
    ALARM = "alarm"                 # Generate alarm
    CLAMP = "clamp"                 # Clamp to limit
    TRIP = "trip"                   # Initiate safety trip
    REJECT = "reject"              # Reject the value


class ViolationSeverity(str, Enum):
    """Severity levels for violations."""
    INFO = "info"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class SafetyLimit(BaseModel):
    """
    Definition of a safety limit for a process variable.

    Attributes:
        limit_id: Unique identifier
        name: Variable name
        low_limit: Low limit value
        high_limit: High limit value
        low_low_limit: Low-low alarm limit (optional)
        high_high_limit: High-high alarm limit (optional)
        unit: Engineering unit
        limit_type: Type of limit
        enforcement_action: Action on violation
        deadband: Deadband to prevent chatter
        time_delay_ms: Delay before enforcement
    """
    limit_id: str = Field(default="", description="Unique identifier")
    name: str = Field(..., description="Variable name")
    low_limit: Optional[float] = Field(None, description="Low limit")
    high_limit: Optional[float] = Field(None, description="High limit")
    low_low_limit: Optional[float] = Field(None, description="Low-low limit")
    high_high_limit: Optional[float] = Field(None, description="High-high limit")
    unit: str = Field("", description="Engineering unit")
    limit_type: LimitType = Field(LimitType.OPERATING, description="Limit type")
    enforcement_action: EnforcementAction = Field(
        EnforcementAction.ALARM, description="Enforcement action"
    )
    deadband: float = Field(0.0, ge=0, description="Deadband")
    time_delay_ms: float = Field(0.0, ge=0, description="Delay before action")
    description: Optional[str] = Field(None, description="Limit description")
    source_reference: Optional[str] = Field(None, description="Standard reference")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.limit_id:
            self.limit_id = f"LIM-{hash(self.name) % 10000:04d}"

    @validator('low_low_limit')
    def validate_low_low(cls, v, values):
        """Validate low-low is below low limit."""
        low = values.get('low_limit')
        if v is not None and low is not None and v >= low:
            raise ValueError("low_low_limit must be less than low_limit")
        return v

    @validator('high_high_limit')
    def validate_high_high(cls, v, values):
        """Validate high-high is above high limit."""
        high = values.get('high_limit')
        if v is not None and high is not None and v <= high:
            raise ValueError("high_high_limit must be greater than high_limit")
        return v


class SafetyStatus(BaseModel):
    """
    Status result from boundary check.

    Contains the check result and all relevant status information.
    """
    variable_name: str = Field(..., description="Variable name")
    current_value: float = Field(..., description="Current value")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Status flags
    is_normal: bool = Field(..., description="Within normal range")
    is_warning: bool = Field(False, description="In warning zone")
    is_alarm: bool = Field(False, description="In alarm zone")
    is_critical: bool = Field(False, description="Critical violation")

    # Violation details
    violation_type: Optional[str] = Field(None, description="high, low, high_high, low_low")
    violation_severity: Optional[ViolationSeverity] = Field(None, description="Severity")
    violation_amount: Optional[float] = Field(None, description="Amount over/under limit")
    violation_percent: Optional[float] = Field(None, description="Percent over/under limit")

    # Limits
    low_limit: Optional[float] = Field(None, description="Low limit")
    high_limit: Optional[float] = Field(None, description="High limit")
    unit: str = Field("", description="Unit")

    # Enforcement
    enforcement_action: EnforcementAction = Field(
        EnforcementAction.NONE, description="Action taken"
    )
    enforced_value: Optional[float] = Field(None, description="Value after enforcement")

    # Message
    message: str = Field("", description="Status message")
    status_hash: str = Field("", description="SHA-256 hash")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.status_hash:
            self.status_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of this status."""
        hash_data = {
            "variable_name": self.variable_name,
            "current_value": str(self.current_value),
            "timestamp": self.timestamp.isoformat(),
            "is_normal": self.is_normal,
            "violation_type": self.violation_type,
        }
        return hashlib.sha256(
            json.dumps(hash_data, sort_keys=True).encode()
        ).hexdigest()


class BoundaryViolation(BaseModel):
    """
    Record of a boundary violation event.

    Used for logging and audit trail of safety violations.
    """
    violation_id: str = Field(..., description="Unique violation ID")
    variable_name: str = Field(..., description="Variable name")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    value: float = Field(..., description="Value at violation")
    limit_value: float = Field(..., description="Limit that was violated")
    violation_type: str = Field(..., description="high, low, high_high, low_low")
    severity: ViolationSeverity = Field(..., description="Violation severity")
    enforcement_action: EnforcementAction = Field(..., description="Action taken")
    enforced_value: Optional[float] = Field(None, description="Value after enforcement")
    duration_ms: Optional[float] = Field(None, description="Duration of violation")
    acknowledged: bool = Field(False, description="Acknowledged by operator")
    acknowledged_by: Optional[str] = Field(None, description="Who acknowledged")
    acknowledged_at: Optional[datetime] = Field(None, description="When acknowledged")
    root_cause: Optional[str] = Field(None, description="Root cause analysis")
    provenance_hash: str = Field("", description="SHA-256 hash")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.provenance_hash:
            self.provenance_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of this violation."""
        hash_data = {
            "violation_id": self.violation_id,
            "variable_name": self.variable_name,
            "timestamp": self.timestamp.isoformat(),
            "value": str(self.value),
            "limit_value": str(self.limit_value),
            "violation_type": self.violation_type,
        }
        return hashlib.sha256(
            json.dumps(hash_data, sort_keys=True).encode()
        ).hexdigest()


class BoundaryCheckStep(BaseModel):
    """Individual boundary check step with provenance."""
    step_number: int = Field(..., description="Step number")
    description: str = Field(..., description="Check description")
    variable: str = Field(..., description="Variable checked")
    value: float = Field(..., description="Value checked")
    limit_type: str = Field(..., description="Limit type checked")
    limit_value: Optional[float] = Field(None, description="Limit value")
    is_violated: bool = Field(..., description="Violation detected")
    step_hash: str = Field("", description="SHA-256 hash")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.step_hash:
            self.step_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of this step."""
        hash_data = {
            "step_number": self.step_number,
            "variable": self.variable,
            "value": str(self.value),
            "limit_type": self.limit_type,
            "is_violated": self.is_violated,
        }
        return hashlib.sha256(
            json.dumps(hash_data, sort_keys=True).encode()
        ).hexdigest()


class SafetyBoundary:
    """
    Safety Boundary Enforcement System.

    This class provides safety boundary checking and enforcement
    for process variables. All checks are deterministic with
    full provenance tracking for regulatory compliance.

    Key Methods:
        check_value: Check single value against limits
        check_all: Check multiple values against all limits
        enforce_limits: Enforce limits by clamping values
        add_limit: Add a new safety limit
        get_violations: Get violation history

    Example:
        >>> boundary = SafetyBoundary(limits={
        ...     "temperature": (0.0, 1200.0),
        ...     "pressure": (0.0, 15.0),
        ... })
        >>> status = boundary.check_value("temperature", 1250.0)
        >>> if status.is_alarm:
        ...     print(f"ALARM: {status.message}")
        >>>
        >>> # Enforce limits
        >>> values = {"temperature": 1250.0, "pressure": 12.0}
        >>> enforced = boundary.enforce_limits(values)
        >>> print(f"Enforced temp: {enforced['temperature']}")  # 1200.0

    CRITICAL: All checks are DETERMINISTIC. NO LLM calls permitted.
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        limits: Optional[Dict[str, Tuple[float, float]]] = None,
        safety_limits: Optional[List[SafetyLimit]] = None,
    ):
        """
        Initialize Safety Boundary system.

        Args:
            limits: Simple dict of {name: (low, high)} limits
            safety_limits: List of SafetyLimit objects for advanced config
        """
        self._limits: Dict[str, SafetyLimit] = {}
        self._violations: List[BoundaryViolation] = []
        self._steps: List[BoundaryCheckStep] = []
        self._step_counter = 0
        self._start_time: Optional[float] = None
        self._violation_counter = 0

        # Add simple limits
        if limits:
            for name, (low, high) in limits.items():
                self.add_limit(SafetyLimit(
                    name=name,
                    low_limit=low,
                    high_limit=high,
                    limit_type=LimitType.OPERATING,
                    enforcement_action=EnforcementAction.ALARM,
                ))

        # Add advanced limits
        if safety_limits:
            for limit in safety_limits:
                self.add_limit(limit)

        logger.info(f"Safety Boundary initialized with {len(self._limits)} limits")

    def add_limit(self, limit: SafetyLimit) -> None:
        """
        Add a safety limit.

        Args:
            limit: SafetyLimit to add
        """
        self._limits[limit.name] = limit
        logger.debug(f"Added limit for '{limit.name}': {limit.low_limit} to {limit.high_limit}")

    def remove_limit(self, name: str) -> bool:
        """
        Remove a safety limit.

        Args:
            name: Variable name

        Returns:
            True if removed, False if not found
        """
        if name in self._limits:
            del self._limits[name]
            logger.debug(f"Removed limit for '{name}'")
            return True
        return False

    def get_limit(self, name: str) -> Optional[SafetyLimit]:
        """
        Get a safety limit by name.

        Args:
            name: Variable name

        Returns:
            SafetyLimit or None
        """
        return self._limits.get(name)

    def _start_check(self) -> None:
        """Reset check state."""
        self._steps = []
        self._step_counter = 0
        self._start_time = time.perf_counter()

    def _record_step(
        self,
        description: str,
        variable: str,
        value: float,
        limit_type: str,
        limit_value: Optional[float],
        is_violated: bool,
    ) -> BoundaryCheckStep:
        """Record a boundary check step."""
        self._step_counter += 1
        step = BoundaryCheckStep(
            step_number=self._step_counter,
            description=description,
            variable=variable,
            value=value,
            limit_type=limit_type,
            limit_value=limit_value,
            is_violated=is_violated,
        )
        self._steps.append(step)
        return step

    def check_value(
        self,
        name: str,
        value: float,
        record_violation: bool = True,
    ) -> SafetyStatus:
        """
        Check a value against its safety limits.

        DETERMINISTIC: Pure arithmetic comparison.

        Args:
            name: Variable name
            value: Current value
            record_violation: Whether to record violations

        Returns:
            SafetyStatus with check results
        """
        limit = self._limits.get(name)

        if limit is None:
            logger.warning(f"No limit defined for '{name}'")
            return SafetyStatus(
                variable_name=name,
                current_value=value,
                is_normal=True,
                message=f"No limits defined for {name}",
            )

        self._start_check()

        # Initialize status
        is_normal = True
        is_warning = False
        is_alarm = False
        is_critical = False
        violation_type: Optional[str] = None
        violation_severity: Optional[ViolationSeverity] = None
        violation_amount: Optional[float] = None
        violation_percent: Optional[float] = None
        message = ""

        # Check high-high limit (critical)
        if limit.high_high_limit is not None:
            exceeded = value > limit.high_high_limit
            self._record_step(
                description=f"Check high-high limit for {name}",
                variable=name,
                value=value,
                limit_type="high_high",
                limit_value=limit.high_high_limit,
                is_violated=exceeded,
            )
            if exceeded:
                is_normal = False
                is_critical = True
                violation_type = "high_high"
                violation_severity = ViolationSeverity.EMERGENCY
                violation_amount = value - limit.high_high_limit
                violation_percent = (violation_amount / limit.high_high_limit) * 100
                message = f"{name} CRITICAL HIGH: {value} > {limit.high_high_limit} {limit.unit}"

        # Check high limit (alarm)
        if limit.high_limit is not None and not is_critical:
            exceeded = value > limit.high_limit
            self._record_step(
                description=f"Check high limit for {name}",
                variable=name,
                value=value,
                limit_type="high",
                limit_value=limit.high_limit,
                is_violated=exceeded,
            )
            if exceeded:
                is_normal = False
                is_alarm = True
                violation_type = "high"
                violation_severity = ViolationSeverity.ALARM
                violation_amount = value - limit.high_limit
                violation_percent = (violation_amount / limit.high_limit) * 100
                message = f"{name} HIGH: {value} > {limit.high_limit} {limit.unit}"

        # Check low-low limit (critical)
        if limit.low_low_limit is not None:
            exceeded = value < limit.low_low_limit
            self._record_step(
                description=f"Check low-low limit for {name}",
                variable=name,
                value=value,
                limit_type="low_low",
                limit_value=limit.low_low_limit,
                is_violated=exceeded,
            )
            if exceeded:
                is_normal = False
                is_critical = True
                violation_type = "low_low"
                violation_severity = ViolationSeverity.EMERGENCY
                violation_amount = limit.low_low_limit - value
                violation_percent = (violation_amount / abs(limit.low_low_limit)) * 100 if limit.low_low_limit != 0 else float('inf')
                message = f"{name} CRITICAL LOW: {value} < {limit.low_low_limit} {limit.unit}"

        # Check low limit (alarm)
        if limit.low_limit is not None and not is_critical:
            exceeded = value < limit.low_limit
            self._record_step(
                description=f"Check low limit for {name}",
                variable=name,
                value=value,
                limit_type="low",
                limit_value=limit.low_limit,
                is_violated=exceeded,
            )
            if exceeded:
                is_normal = False
                is_alarm = True
                violation_type = "low"
                violation_severity = ViolationSeverity.ALARM
                violation_amount = limit.low_limit - value
                violation_percent = (violation_amount / abs(limit.low_limit)) * 100 if limit.low_limit != 0 else float('inf')
                message = f"{name} LOW: {value} < {limit.low_limit} {limit.unit}"

        # Check warning zone (within 10% of limit)
        if is_normal:
            if limit.high_limit is not None:
                margin = limit.high_limit - value
                if margin > 0 and margin < (limit.high_limit * 0.1):
                    is_warning = True
                    violation_severity = ViolationSeverity.WARNING
                    message = f"{name} approaching high limit: {value} (limit: {limit.high_limit})"

            if limit.low_limit is not None:
                margin = value - limit.low_limit
                if margin > 0 and margin < (abs(limit.low_limit) * 0.1):
                    is_warning = True
                    violation_severity = ViolationSeverity.WARNING
                    message = f"{name} approaching low limit: {value} (limit: {limit.low_limit})"

        if is_normal and not is_warning:
            message = f"{name} normal: {value} {limit.unit}"

        # Determine enforcement action
        enforcement_action = EnforcementAction.NONE
        enforced_value: Optional[float] = None

        if not is_normal and limit.enforcement_action != EnforcementAction.NONE:
            enforcement_action = limit.enforcement_action
            if enforcement_action == EnforcementAction.CLAMP:
                enforced_value = self._clamp_value(value, limit)

        # Record violation if needed
        if record_violation and not is_normal:
            self._record_violation(
                name=name,
                value=value,
                limit=limit,
                violation_type=violation_type,
                severity=violation_severity or ViolationSeverity.ALARM,
                enforcement_action=enforcement_action,
                enforced_value=enforced_value,
            )

        return SafetyStatus(
            variable_name=name,
            current_value=value,
            is_normal=is_normal,
            is_warning=is_warning,
            is_alarm=is_alarm,
            is_critical=is_critical,
            violation_type=violation_type,
            violation_severity=violation_severity,
            violation_amount=violation_amount,
            violation_percent=violation_percent,
            low_limit=limit.low_limit,
            high_limit=limit.high_limit,
            unit=limit.unit,
            enforcement_action=enforcement_action,
            enforced_value=enforced_value,
            message=message,
        )

    def check_all(
        self,
        values: Dict[str, float],
    ) -> Dict[str, SafetyStatus]:
        """
        Check multiple values against all defined limits.

        Args:
            values: Dictionary of {name: value}

        Returns:
            Dictionary of {name: SafetyStatus}
        """
        results = {}
        for name, value in values.items():
            results[name] = self.check_value(name, value)
        return results

    def enforce_limits(
        self,
        values: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Enforce limits by clamping values to their boundaries.

        DETERMINISTIC: Pure arithmetic clamping.

        Args:
            values: Dictionary of {name: value}

        Returns:
            Dictionary of {name: clamped_value}
        """
        enforced = {}
        for name, value in values.items():
            limit = self._limits.get(name)
            if limit is None:
                enforced[name] = value
            else:
                enforced[name] = self._clamp_value(value, limit)
        return enforced

    def _clamp_value(
        self,
        value: float,
        limit: SafetyLimit,
    ) -> float:
        """
        Clamp a value to its limits.

        DETERMINISTIC: min/max operations.

        Args:
            value: Value to clamp
            limit: SafetyLimit

        Returns:
            Clamped value
        """
        result = value

        if limit.low_limit is not None:
            result = max(result, limit.low_limit)

        if limit.high_limit is not None:
            result = min(result, limit.high_limit)

        return result

    def _record_violation(
        self,
        name: str,
        value: float,
        limit: SafetyLimit,
        violation_type: Optional[str],
        severity: ViolationSeverity,
        enforcement_action: EnforcementAction,
        enforced_value: Optional[float],
    ) -> BoundaryViolation:
        """Record a boundary violation."""
        self._violation_counter += 1
        violation_id = f"VIO-{int(time.time())}-{self._violation_counter:04d}"

        # Determine limit value based on violation type
        if violation_type == "high":
            limit_value = limit.high_limit or 0
        elif violation_type == "high_high":
            limit_value = limit.high_high_limit or 0
        elif violation_type == "low":
            limit_value = limit.low_limit or 0
        elif violation_type == "low_low":
            limit_value = limit.low_low_limit or 0
        else:
            limit_value = 0

        violation = BoundaryViolation(
            violation_id=violation_id,
            variable_name=name,
            value=value,
            limit_value=limit_value,
            violation_type=violation_type or "unknown",
            severity=severity,
            enforcement_action=enforcement_action,
            enforced_value=enforced_value,
        )

        self._violations.append(violation)

        logger.warning(
            f"BOUNDARY VIOLATION: {name}={value} {violation_type} limit={limit_value}, "
            f"severity={severity.value}, action={enforcement_action.value}"
        )

        return violation

    def get_violations(
        self,
        since: Optional[datetime] = None,
        variable_name: Optional[str] = None,
        severity: Optional[ViolationSeverity] = None,
    ) -> List[BoundaryViolation]:
        """
        Get violation history with optional filters.

        Args:
            since: Filter by timestamp
            variable_name: Filter by variable
            severity: Filter by severity

        Returns:
            List of BoundaryViolation records
        """
        violations = self._violations

        if since is not None:
            violations = [v for v in violations if v.timestamp >= since]

        if variable_name is not None:
            violations = [v for v in violations if v.variable_name == variable_name]

        if severity is not None:
            violations = [v for v in violations if v.severity == severity]

        return sorted(violations, key=lambda v: v.timestamp, reverse=True)

    def acknowledge_violation(
        self,
        violation_id: str,
        acknowledged_by: str,
    ) -> Optional[BoundaryViolation]:
        """
        Acknowledge a violation.

        Args:
            violation_id: Violation identifier
            acknowledged_by: Who acknowledged

        Returns:
            Updated BoundaryViolation or None
        """
        for violation in self._violations:
            if violation.violation_id == violation_id:
                violation.acknowledged = True
                violation.acknowledged_by = acknowledged_by
                violation.acknowledged_at = datetime.utcnow()
                logger.info(f"Violation {violation_id} acknowledged by {acknowledged_by}")
                return violation
        return None

    def get_unacknowledged_violations(self) -> List[BoundaryViolation]:
        """Get all unacknowledged violations."""
        return [v for v in self._violations if not v.acknowledged]

    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get summary of all limit statuses.

        Returns:
            Dictionary with status counts and details
        """
        return {
            "total_limits": len(self._limits),
            "total_violations": len(self._violations),
            "unacknowledged_violations": len(self.get_unacknowledged_violations()),
            "limits": {
                name: {
                    "low": limit.low_limit,
                    "high": limit.high_limit,
                    "unit": limit.unit,
                    "type": limit.limit_type.value,
                }
                for name, limit in self._limits.items()
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    def validate_all_limits(self) -> Tuple[bool, List[str]]:
        """
        Validate all configured limits for consistency.

        Returns:
            Tuple of (all_valid, messages)
        """
        messages = []
        all_valid = True

        for name, limit in self._limits.items():
            # Check low < high
            if limit.low_limit is not None and limit.high_limit is not None:
                if limit.low_limit >= limit.high_limit:
                    messages.append(f"{name}: low_limit >= high_limit")
                    all_valid = False

            # Check low-low < low
            if limit.low_low_limit is not None and limit.low_limit is not None:
                if limit.low_low_limit >= limit.low_limit:
                    messages.append(f"{name}: low_low_limit >= low_limit")
                    all_valid = False

            # Check high < high-high
            if limit.high_limit is not None and limit.high_high_limit is not None:
                if limit.high_limit >= limit.high_high_limit:
                    messages.append(f"{name}: high_limit >= high_high_limit")
                    all_valid = False

            # Check at least one limit is defined
            if all(x is None for x in [
                limit.low_limit, limit.high_limit,
                limit.low_low_limit, limit.high_high_limit
            ]):
                messages.append(f"{name}: no limits defined")
                all_valid = False

        if all_valid:
            messages.append("All limits validated successfully")

        return all_valid, messages

    def to_dict(self) -> Dict[str, Any]:
        """Export boundary configuration as dictionary."""
        return {
            "version": self.VERSION,
            "limits": {
                name: limit.dict()
                for name, limit in self._limits.items()
            },
            "violation_count": len(self._violations),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SafetyBoundary":
        """Create SafetyBoundary from dictionary."""
        limits = []
        for name, limit_data in data.get("limits", {}).items():
            limit_data["name"] = name
            limits.append(SafetyLimit(**limit_data))
        return cls(safety_limits=limits)
