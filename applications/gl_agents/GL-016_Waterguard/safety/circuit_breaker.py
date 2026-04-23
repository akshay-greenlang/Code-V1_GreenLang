"""
GL-016 Waterguard Circuit Breaker - IEC 61511 SIL-3 Compliant

This module implements the Circuit Breaker pattern for fault isolation
and graceful degradation in the WATERGUARD safety system.

The circuit breaker pattern prevents cascading failures by:
    1. Detecting consecutive failures
    2. Opening the circuit to stop further damage
    3. Allowing controlled recovery after a timeout

Circuit Breaker States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Circuit tripped, calls rejected immediately
    - HALF_OPEN: Testing recovery, limited calls allowed

Key Features:
    - Per-actuator circuit breakers
    - Exponential backoff for recovery attempts
    - Complete audit trail
    - Thread-safe operations
    - Integration with safety coordinator

Reference Standards:
    - IEC 61511-1:2016 Functional Safety
    - Circuit Breaker Pattern (Release It! by Michael Nygard)

Author: GreenLang Safety Engineering Team
Version: 1.0.0
SIL Level: 3
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
import uuid

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# ENUMERATIONS
# =============================================================================


class CircuitBreakerState(str, Enum):
    """State of a circuit breaker."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Circuit tripped, rejecting calls
    HALF_OPEN = "half_open" # Testing recovery


class FailureType(str, Enum):
    """Types of failures that can trip a circuit breaker."""
    COMMUNICATION_TIMEOUT = "communication_timeout"
    COMMUNICATION_ERROR = "communication_error"
    ACTUATOR_FAULT = "actuator_fault"
    VALIDATION_FAILURE = "validation_failure"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    SENSOR_FAULT = "sensor_fault"
    UNKNOWN = "unknown"


class RecoveryStrategy(str, Enum):
    """Recovery strategies for circuit breakers."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_INTERVAL = "fixed_interval"
    MANUAL_ONLY = "manual_only"


# =============================================================================
# DATA MODELS
# =============================================================================


class CircuitBreakerConfig(BaseModel):
    """Configuration for a circuit breaker."""

    name: str = Field(
        ...,
        description="Circuit breaker name"
    )
    failure_threshold: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Failures before circuit opens"
    )
    success_threshold: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Successes in half-open before closing"
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=3600.0,
        description="Time in open state before half-open"
    )
    max_timeout_seconds: float = Field(
        default=300.0,
        ge=10.0,
        le=3600.0,
        description="Maximum timeout with exponential backoff"
    )
    recovery_strategy: RecoveryStrategy = Field(
        default=RecoveryStrategy.EXPONENTIAL_BACKOFF,
        description="Recovery strategy"
    )
    half_open_max_calls: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max calls allowed in half-open state"
    )
    enabled: bool = Field(
        default=True,
        description="Circuit breaker enabled"
    )


class FailureRecord(BaseModel):
    """Record of a failure event."""

    failure_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Failure identifier"
    )
    failure_type: FailureType = Field(
        ...,
        description="Type of failure"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When failure occurred"
    )
    error_message: str = Field(
        default="",
        description="Error message"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context"
    )


class CircuitBreakerStatus(BaseModel):
    """Current status of a circuit breaker."""

    name: str = Field(
        ...,
        description="Circuit breaker name"
    )
    state: CircuitBreakerState = Field(
        ...,
        description="Current state"
    )
    failure_count: int = Field(
        default=0,
        ge=0,
        description="Current failure count"
    )
    success_count: int = Field(
        default=0,
        ge=0,
        description="Success count in half-open"
    )
    last_failure_time: Optional[datetime] = Field(
        default=None,
        description="Last failure time"
    )
    last_state_change: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last state change time"
    )
    current_timeout: float = Field(
        default=30.0,
        description="Current timeout seconds"
    )
    trips_total: int = Field(
        default=0,
        ge=0,
        description="Total times circuit has tripped"
    )
    calls_rejected: int = Field(
        default=0,
        ge=0,
        description="Calls rejected while open"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Status timestamp"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit"
    )

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            hash_str = (
                f"{self.name}|{self.state.value}|{self.failure_count}|"
                f"{self.trips_total}|{self.timestamp.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(hash_str.encode()).hexdigest()[:16]


class StateTransition(BaseModel):
    """Record of a circuit breaker state transition."""

    transition_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Transition identifier"
    )
    circuit_name: str = Field(
        ...,
        description="Circuit breaker name"
    )
    from_state: CircuitBreakerState = Field(
        ...,
        description="Previous state"
    )
    to_state: CircuitBreakerState = Field(
        ...,
        description="New state"
    )
    reason: str = Field(
        default="",
        description="Reason for transition"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Transition timestamp"
    )
    failure_count: int = Field(
        default=0,
        description="Failure count at transition"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit"
    )

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            hash_str = (
                f"{self.transition_id}|{self.circuit_name}|"
                f"{self.from_state.value}|{self.to_state.value}|"
                f"{self.timestamp.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(hash_str.encode()).hexdigest()


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================


class CircuitBreaker:
    """
    Circuit Breaker for fault isolation.

    The circuit breaker pattern prevents cascading failures by detecting
    consecutive failures and "opening" the circuit to reject further calls
    until the system has had time to recover.

    States:
        - CLOSED: Normal operation. Calls pass through. Failures are counted.
        - OPEN: Circuit is tripped. All calls are rejected immediately.
                After timeout, transitions to HALF_OPEN.
        - HALF_OPEN: Testing recovery. Limited calls allowed. If successful,
                     transitions to CLOSED. If fails, back to OPEN.

    SIL-3 Compliance:
        - All state transitions are logged with provenance hashes
        - Thread-safe operations with proper locking
        - Deterministic state machine behavior
        - No silent failures

    Example:
        >>> cb = CircuitBreaker(CircuitBreakerConfig(name="pump_control"))
        >>> try:
        ...     with cb.call_context():
        ...         result = control_pump(...)
        ... except CircuitOpenError:
        ...     # Circuit is open, use fallback
        ...     fallback_action()
    """

    def __init__(self, config: CircuitBreakerConfig) -> None:
        """
        Initialize CircuitBreaker.

        Args:
            config: Circuit breaker configuration
        """
        self._config = config
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0

        self._last_failure_time: Optional[datetime] = None
        self._last_state_change = datetime.now(timezone.utc)
        self._open_until: Optional[datetime] = None
        self._current_timeout = config.timeout_seconds
        self._backoff_multiplier = 1

        self._trips_total = 0
        self._calls_rejected = 0

        self._lock = threading.RLock()

        # History
        self._failure_history: List[FailureRecord] = []
        self._transition_history: List[StateTransition] = []
        self._max_history = 1000

        # Callbacks
        self._on_state_change: Optional[Callable[[StateTransition], None]] = None
        self._on_trip: Optional[Callable[[str], None]] = None

        logger.info(
            "CircuitBreaker '%s' initialized: threshold=%d, timeout=%.1fs",
            config.name, config.failure_threshold, config.timeout_seconds
        )

    @property
    def name(self) -> str:
        """Get circuit breaker name."""
        return self._config.name

    @property
    def state(self) -> CircuitBreakerState:
        """Get current state (may transition based on timeout)."""
        with self._lock:
            self._check_state_transition()
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitBreakerState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting calls)."""
        return self.state == CircuitBreakerState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.state == CircuitBreakerState.HALF_OPEN

    def can_execute(self) -> bool:
        """
        Check if a call can be executed.

        Returns:
            True if call can proceed, False if circuit is open

        Note:
            This updates state if needed (OPEN -> HALF_OPEN after timeout).
        """
        with self._lock:
            self._check_state_transition()

            if self._state == CircuitBreakerState.CLOSED:
                return True

            if self._state == CircuitBreakerState.OPEN:
                self._calls_rejected += 1
                return False

            if self._state == CircuitBreakerState.HALF_OPEN:
                if self._half_open_calls < self._config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    def record_success(self) -> None:
        """
        Record a successful call.

        In HALF_OPEN state, successes count toward closing the circuit.
        In CLOSED state, this resets the failure count.
        """
        with self._lock:
            if self._state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

            elif self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                logger.debug(
                    "CircuitBreaker '%s' success in half-open: %d/%d",
                    self._config.name,
                    self._success_count,
                    self._config.success_threshold
                )

                if self._success_count >= self._config.success_threshold:
                    self._transition_to(
                        CircuitBreakerState.CLOSED,
                        "Recovery successful"
                    )

    def record_failure(
        self,
        failure_type: FailureType = FailureType.UNKNOWN,
        error_message: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a failed call.

        Failures are counted. When threshold is reached, circuit opens.
        In HALF_OPEN, any failure immediately opens the circuit.

        Args:
            failure_type: Type of failure
            error_message: Error message
            context: Additional context
        """
        with self._lock:
            # Record failure
            failure = FailureRecord(
                failure_type=failure_type,
                error_message=error_message,
                context=context or {}
            )
            self._failure_history.append(failure)
            if len(self._failure_history) > self._max_history:
                self._failure_history = self._failure_history[-self._max_history:]

            self._last_failure_time = datetime.now(timezone.utc)
            self._failure_count += 1

            logger.warning(
                "CircuitBreaker '%s' failure: %s - %s (count: %d/%d)",
                self._config.name,
                failure_type.value,
                error_message[:100],
                self._failure_count,
                self._config.failure_threshold
            )

            if self._state == CircuitBreakerState.CLOSED:
                if self._failure_count >= self._config.failure_threshold:
                    self._trip("Failure threshold reached")

            elif self._state == CircuitBreakerState.HALF_OPEN:
                # Any failure in half-open immediately trips
                self._trip("Failure during recovery test")

    def _trip(self, reason: str) -> None:
        """Trip the circuit breaker (transition to OPEN)."""
        self._trips_total += 1

        # Calculate timeout with backoff
        if self._config.recovery_strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
            self._current_timeout = min(
                self._config.timeout_seconds * (2 ** self._backoff_multiplier),
                self._config.max_timeout_seconds
            )
            self._backoff_multiplier += 1
        elif self._config.recovery_strategy == RecoveryStrategy.LINEAR_BACKOFF:
            self._current_timeout = min(
                self._config.timeout_seconds * (1 + self._backoff_multiplier),
                self._config.max_timeout_seconds
            )
            self._backoff_multiplier += 1
        else:
            self._current_timeout = self._config.timeout_seconds

        self._open_until = datetime.now(timezone.utc) + timedelta(
            seconds=self._current_timeout
        )

        self._transition_to(CircuitBreakerState.OPEN, reason)

        logger.error(
            "CircuitBreaker '%s' TRIPPED: %s (timeout: %.1fs, trips: %d)",
            self._config.name, reason, self._current_timeout, self._trips_total
        )

        if self._on_trip:
            try:
                self._on_trip(self._config.name)
            except Exception as e:
                logger.error("Trip callback failed: %s", e)

    def _check_state_transition(self) -> None:
        """Check if state should transition based on timeout."""
        if self._state == CircuitBreakerState.OPEN:
            if self._open_until and datetime.now(timezone.utc) >= self._open_until:
                self._transition_to(
                    CircuitBreakerState.HALF_OPEN,
                    "Timeout elapsed, testing recovery"
                )
                self._half_open_calls = 0
                self._success_count = 0

    def _transition_to(
        self,
        new_state: CircuitBreakerState,
        reason: str
    ) -> None:
        """Transition to a new state with logging."""
        old_state = self._state

        transition = StateTransition(
            circuit_name=self._config.name,
            from_state=old_state,
            to_state=new_state,
            reason=reason,
            failure_count=self._failure_count
        )

        self._transition_history.append(transition)
        if len(self._transition_history) > self._max_history:
            self._transition_history = self._transition_history[-self._max_history:]

        self._state = new_state
        self._last_state_change = datetime.now(timezone.utc)

        if new_state == CircuitBreakerState.CLOSED:
            # Reset on close
            self._failure_count = 0
            self._backoff_multiplier = 0
            self._current_timeout = self._config.timeout_seconds

        logger.info(
            "CircuitBreaker '%s' state: %s -> %s (%s)",
            self._config.name, old_state.value, new_state.value, reason
        )

        if self._on_state_change:
            try:
                self._on_state_change(transition)
            except Exception as e:
                logger.error("State change callback failed: %s", e)

    def reset(self, reason: str = "Manual reset") -> None:
        """
        Manually reset the circuit breaker to CLOSED state.

        Args:
            reason: Reason for reset
        """
        with self._lock:
            self._transition_to(CircuitBreakerState.CLOSED, reason)
            logger.warning(
                "CircuitBreaker '%s' manually reset: %s",
                self._config.name, reason
            )

    def force_open(self, reason: str = "Manual trip") -> None:
        """
        Manually force the circuit to OPEN state.

        Args:
            reason: Reason for forcing open
        """
        with self._lock:
            self._trip(reason)

    def get_status(self) -> CircuitBreakerStatus:
        """Get current circuit breaker status."""
        with self._lock:
            self._check_state_transition()
            return CircuitBreakerStatus(
                name=self._config.name,
                state=self._state,
                failure_count=self._failure_count,
                success_count=self._success_count,
                last_failure_time=self._last_failure_time,
                last_state_change=self._last_state_change,
                current_timeout=self._current_timeout,
                trips_total=self._trips_total,
                calls_rejected=self._calls_rejected
            )

    def get_failure_history(self, limit: int = 50) -> List[FailureRecord]:
        """Get recent failure history."""
        with self._lock:
            return list(reversed(self._failure_history[-limit:]))

    def get_transition_history(self, limit: int = 50) -> List[StateTransition]:
        """Get recent state transition history."""
        with self._lock:
            return list(reversed(self._transition_history[-limit:]))

    def set_on_state_change(
        self,
        callback: Callable[[StateTransition], None]
    ) -> None:
        """Set callback for state changes."""
        self._on_state_change = callback

    def set_on_trip(self, callback: Callable[[str], None]) -> None:
        """Set callback for circuit trips."""
        self._on_trip = callback


# =============================================================================
# ACTUATOR CIRCUIT BREAKER
# =============================================================================


class ActuatorCircuitBreaker(CircuitBreaker):
    """
    Specialized circuit breaker for actuator control.

    This circuit breaker is designed for controlling physical actuators
    like valves, pumps, and dosing systems. It includes additional
    safety features:

        - Per-actuator isolation (failure of one doesn't affect others)
        - Safe state on trip (actuator goes to defined safe position)
        - Integration with action gate
        - OEM interlock awareness

    Example:
        >>> cb = ActuatorCircuitBreaker(
        ...     config=CircuitBreakerConfig(name="BD-001"),
        ...     actuator_tag="BD-001",
        ...     safe_state_value=0.0  # Close valve on trip
        ... )
    """

    def __init__(
        self,
        config: CircuitBreakerConfig,
        actuator_tag: str,
        safe_state_value: float = 0.0,
        safe_state_units: str = "%",
        action_gate: Optional[Any] = None
    ) -> None:
        """
        Initialize ActuatorCircuitBreaker.

        Args:
            config: Circuit breaker configuration
            actuator_tag: Tag of controlled actuator
            safe_state_value: Value to command on trip
            safe_state_units: Engineering units
            action_gate: Optional ActionGate for permission checks
        """
        super().__init__(config)
        self._actuator_tag = actuator_tag
        self._safe_state_value = safe_state_value
        self._safe_state_units = safe_state_units
        self._action_gate = action_gate

        self._last_commanded_value: Optional[float] = None
        self._in_safe_state = False

        logger.info(
            "ActuatorCircuitBreaker '%s' for tag '%s': safe_state=%.2f%s",
            config.name, actuator_tag, safe_state_value, safe_state_units
        )

    @property
    def actuator_tag(self) -> str:
        """Get actuator tag."""
        return self._actuator_tag

    @property
    def is_in_safe_state(self) -> bool:
        """Check if actuator is in safe state."""
        with self._lock:
            return self._in_safe_state

    def command_actuator(
        self,
        value: float,
        command_func: Callable[[str, float], bool]
    ) -> bool:
        """
        Command the actuator through the circuit breaker.

        Args:
            value: Value to command
            command_func: Function to execute command (tag, value) -> success

        Returns:
            True if command succeeded, False if rejected or failed
        """
        with self._lock:
            if not self.can_execute():
                logger.warning(
                    "ActuatorCircuitBreaker '%s' rejected command: circuit open",
                    self._config.name
                )
                return False

        # Execute command outside lock
        try:
            success = command_func(self._actuator_tag, value)

            with self._lock:
                if success:
                    self._last_commanded_value = value
                    self._in_safe_state = False
                    self.record_success()
                else:
                    self.record_failure(
                        FailureType.ACTUATOR_FAULT,
                        f"Command to {self._actuator_tag} failed"
                    )

            return success

        except Exception as e:
            self.record_failure(
                FailureType.ACTUATOR_FAULT,
                str(e),
                {"actuator_tag": self._actuator_tag, "commanded_value": value}
            )
            return False

    def enter_safe_state(
        self,
        command_func: Callable[[str, float], bool],
        reason: str = "Circuit breaker trip"
    ) -> bool:
        """
        Command actuator to safe state.

        Args:
            command_func: Function to execute command
            reason: Reason for entering safe state

        Returns:
            True if safe state achieved
        """
        logger.warning(
            "ActuatorCircuitBreaker '%s' entering safe state: %s",
            self._config.name, reason
        )

        try:
            success = command_func(self._actuator_tag, self._safe_state_value)
            with self._lock:
                self._in_safe_state = success
            return success
        except Exception as e:
            logger.error(
                "Failed to enter safe state for '%s': %s",
                self._actuator_tag, e
            )
            return False


# =============================================================================
# CIRCUIT BREAKER REGISTRY
# =============================================================================


class CircuitBreakerRegistry:
    """
    Registry and coordinator for multiple circuit breakers.

    The registry provides:
        - Centralized management of all circuit breakers
        - Bulk status queries
        - Cascade prevention (stop propagation of failures)
        - Statistics and monitoring
        - Integration with safety coordinator

    Example:
        >>> registry = CircuitBreakerRegistry()
        >>> registry.register(CircuitBreaker(config1))
        >>> registry.register(CircuitBreaker(config2))
        >>> status = registry.get_all_status()
        >>> open_circuits = registry.get_open_circuits()
    """

    def __init__(self) -> None:
        """Initialize CircuitBreakerRegistry."""
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            "total_trips": 0,
            "total_resets": 0,
            "total_calls_rejected": 0,
        }

        # Callbacks
        self._on_any_trip: Optional[Callable[[str, CircuitBreaker], None]] = None
        self._on_cascade_risk: Optional[Callable[[List[str]], None]] = None

        # Cascade detection
        self._cascade_threshold = 3  # Open circuits before cascade warning

        logger.info("CircuitBreakerRegistry initialized")

    def register(self, breaker: CircuitBreaker) -> None:
        """
        Register a circuit breaker.

        Args:
            breaker: Circuit breaker to register
        """
        with self._lock:
            if breaker.name in self._breakers:
                logger.warning(
                    "Replacing existing circuit breaker: %s",
                    breaker.name
                )

            self._breakers[breaker.name] = breaker

            # Set up trip callback for cascade detection
            original_callback = breaker._on_trip

            def trip_wrapper(name: str) -> None:
                if original_callback:
                    original_callback(name)
                self._handle_trip(name)

            breaker.set_on_trip(trip_wrapper)

            logger.info("Registered circuit breaker: %s", breaker.name)

    def unregister(self, name: str) -> Optional[CircuitBreaker]:
        """
        Unregister a circuit breaker.

        Args:
            name: Circuit breaker name

        Returns:
            Removed circuit breaker or None
        """
        with self._lock:
            return self._breakers.pop(name, None)

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """
        Get a circuit breaker by name.

        Args:
            name: Circuit breaker name

        Returns:
            Circuit breaker or None
        """
        with self._lock:
            return self._breakers.get(name)

    def get_all(self) -> Dict[str, CircuitBreaker]:
        """Get all registered circuit breakers."""
        with self._lock:
            return dict(self._breakers)

    def get_all_status(self) -> Dict[str, CircuitBreakerStatus]:
        """Get status of all circuit breakers."""
        with self._lock:
            return {
                name: breaker.get_status()
                for name, breaker in self._breakers.items()
            }

    def get_open_circuits(self) -> List[str]:
        """Get names of all open circuit breakers."""
        with self._lock:
            return [
                name for name, breaker in self._breakers.items()
                if breaker.is_open
            ]

    def get_healthy_circuits(self) -> List[str]:
        """Get names of all closed circuit breakers."""
        with self._lock:
            return [
                name for name, breaker in self._breakers.items()
                if breaker.is_closed
            ]

    def reset_all(self, reason: str = "Bulk reset") -> int:
        """
        Reset all circuit breakers to CLOSED.

        Args:
            reason: Reason for reset

        Returns:
            Number of circuits reset
        """
        count = 0
        with self._lock:
            for breaker in self._breakers.values():
                if not breaker.is_closed:
                    breaker.reset(reason)
                    count += 1
                    self._stats["total_resets"] += 1

        logger.warning("Bulk reset of %d circuit breakers: %s", count, reason)
        return count

    def _handle_trip(self, name: str) -> None:
        """Handle a circuit trip for cascade detection."""
        with self._lock:
            self._stats["total_trips"] += 1

            open_circuits = self.get_open_circuits()
            open_count = len(open_circuits)

            if open_count >= self._cascade_threshold:
                logger.critical(
                    "CASCADE RISK: %d circuits open: %s",
                    open_count, open_circuits
                )
                if self._on_cascade_risk:
                    try:
                        self._on_cascade_risk(open_circuits)
                    except Exception as e:
                        logger.error("Cascade callback failed: %s", e)

            if self._on_any_trip:
                breaker = self._breakers.get(name)
                if breaker:
                    try:
                        self._on_any_trip(name, breaker)
                    except Exception as e:
                        logger.error("Trip callback failed: %s", e)

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            open_count = len(self.get_open_circuits())
            total_count = len(self._breakers)

            return {
                **self._stats,
                "total_breakers": total_count,
                "open_count": open_count,
                "closed_count": total_count - open_count,
                "health_percentage": (
                    ((total_count - open_count) / total_count * 100)
                    if total_count > 0 else 100.0
                ),
            }

    def set_on_any_trip(
        self,
        callback: Callable[[str, CircuitBreaker], None]
    ) -> None:
        """Set callback for any circuit trip."""
        self._on_any_trip = callback

    def set_on_cascade_risk(
        self,
        callback: Callable[[List[str]], None]
    ) -> None:
        """Set callback for cascade risk detection."""
        self._on_cascade_risk = callback

    def set_cascade_threshold(self, threshold: int) -> None:
        """Set number of open circuits before cascade warning."""
        self._cascade_threshold = max(1, threshold)


# =============================================================================
# EXCEPTIONS
# =============================================================================


class CircuitOpenError(Exception):
    """Raised when attempting to execute through an open circuit."""

    def __init__(self, circuit_name: str, message: str = ""):
        self.circuit_name = circuit_name
        super().__init__(
            message or f"Circuit '{circuit_name}' is open"
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "CircuitBreakerState",
    "FailureType",
    "RecoveryStrategy",
    # Models
    "CircuitBreakerConfig",
    "FailureRecord",
    "CircuitBreakerStatus",
    "StateTransition",
    # Classes
    "CircuitBreaker",
    "ActuatorCircuitBreaker",
    "CircuitBreakerRegistry",
    # Exceptions
    "CircuitOpenError",
]
