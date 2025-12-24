# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Safety Circuit Breaker

Implements the circuit breaker pattern for safety-critical emission monitoring.
Provides automatic failure detection, isolation, and recovery with configurable
thresholds and escalation policies.

Design Principles:
    - Fail-safe: Default to safe state on any error
    - Deterministic: No ML inference for safety decisions
    - Auditable: Complete provenance tracking
    - Resilient: Automatic recovery with backoff

Reference: IEC 61508 Safety Integrity Level patterns

Author: GreenLang GL-010 EmissionsGuardian
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
import hashlib
import logging
import threading
import time
import uuid
import json
from collections import deque
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failures exceeded threshold, blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class FailureType(str, Enum):
    """Types of failures tracked by circuit breaker."""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    VALIDATION = "validation"
    THRESHOLD = "threshold"
    SAFETY = "safety"
    RATE_LIMIT = "rate_limit"


class EscalationLevel(str, Enum):
    """Escalation levels for safety events."""
    LEVEL_1 = "operator_alert"
    LEVEL_2 = "supervisor_alert"
    LEVEL_3 = "emergency_shutdown"
    LEVEL_4 = "regulatory_notification"


@dataclass
class FailureEvent:
    """Record of a circuit breaker failure."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    failure_type: FailureType = FailureType.EXCEPTION
    service_name: str = ""
    operation: str = ""
    error_message: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    severity: int = 1  # 1=low, 5=critical
    provenance_hash: str = ""

    def __post_init__(self):
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        content = f"{self.event_id}|{self.timestamp.isoformat()}|{self.failure_type.value}|{self.service_name}"
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    # Failure thresholds
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes in half-open before closing
    timeout_seconds: float = 30.0  # Default operation timeout

    # Recovery settings
    recovery_timeout_seconds: float = 60.0  # Time before half-open test
    reset_timeout_seconds: float = 300.0  # Time before full reset

    # Rate limiting
    max_calls_per_minute: int = 100
    rate_limit_window_seconds: float = 60.0

    # Safety settings
    safety_threshold: int = 3  # Safety violations before escalation
    auto_recovery_enabled: bool = True
    escalation_enabled: bool = True

    # Sliding window
    window_size_seconds: float = 120.0  # 2-minute window for failure counting


@dataclass
class CircuitBreakerState:
    """Current state of a circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    opened_at: Optional[datetime] = None
    half_opened_at: Optional[datetime] = None
    total_failures: int = 0
    total_successes: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreaker:
    """
    Safety-critical circuit breaker for emission monitoring.

    Implements automatic failure detection with configurable thresholds,
    supports escalation policies, and maintains complete audit trails.
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState()
        self._lock = threading.RLock()
        self._failure_window: deque = deque()
        self._call_timestamps: deque = deque()
        self._failure_history: List[FailureEvent] = []
        self._escalation_callbacks: Dict[EscalationLevel, List[Callable]] = {
            level: [] for level in EscalationLevel
        }
        self._state_change_callbacks: List[Callable] = []
        self._safety_violation_count = 0

        logger.info(f"CircuitBreaker '{name}' initialized: threshold={self.config.failure_threshold}")

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state.state

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking calls)."""
        return self.state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    def register_escalation_callback(
        self,
        level: EscalationLevel,
        callback: Callable[[FailureEvent], None],
    ) -> None:
        """Register callback for escalation events."""
        self._escalation_callbacks[level].append(callback)
        logger.info(f"Registered escalation callback for {level.value}")

    def register_state_change_callback(
        self,
        callback: Callable[[CircuitState, CircuitState], None],
    ) -> None:
        """Register callback for state changes."""
        self._state_change_callbacks.append(callback)

    def _clean_old_failures(self) -> None:
        """Remove failures outside the sliding window."""
        cutoff = datetime.utcnow() - timedelta(seconds=self.config.window_size_seconds)
        while self._failure_window and self._failure_window[0] < cutoff:
            self._failure_window.popleft()

    def _clean_old_calls(self) -> None:
        """Remove call timestamps outside rate limit window."""
        cutoff = datetime.utcnow() - timedelta(seconds=self.config.rate_limit_window_seconds)
        while self._call_timestamps and self._call_timestamps[0] < cutoff:
            self._call_timestamps.popleft()

    def _check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded."""
        self._clean_old_calls()
        return len(self._call_timestamps) < self.config.max_calls_per_minute

    def _record_call(self) -> None:
        """Record a call for rate limiting."""
        self._call_timestamps.append(datetime.utcnow())

    def _transition_state(self, new_state: CircuitState) -> None:
        """Transition to a new state with callbacks."""
        old_state = self._state.state
        if old_state == new_state:
            return

        self._state.state = new_state
        now = datetime.utcnow()

        if new_state == CircuitState.OPEN:
            self._state.opened_at = now
        elif new_state == CircuitState.HALF_OPEN:
            self._state.half_opened_at = now
            self._state.success_count = 0
        elif new_state == CircuitState.CLOSED:
            self._state.failure_count = 0
            self._state.consecutive_failures = 0

        logger.warning(f"CircuitBreaker '{self.name}': {old_state.value} -> {new_state.value}")

        # Notify callbacks
        for callback in self._state_change_callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")

    def _record_failure(
        self,
        failure_type: FailureType,
        error_message: str,
        operation: str = "",
        context: Optional[Dict] = None,
        severity: int = 1,
    ) -> FailureEvent:
        """Record a failure event."""
        event = FailureEvent(
            failure_type=failure_type,
            service_name=self.name,
            operation=operation,
            error_message=error_message,
            context=context or {},
            severity=severity,
        )

        self._failure_window.append(datetime.utcnow())
        self._failure_history.append(event)
        self._state.total_failures += 1
        self._state.consecutive_failures += 1
        self._state.consecutive_successes = 0
        self._state.last_failure_time = datetime.utcnow()

        # Check for safety violations
        if failure_type == FailureType.SAFETY:
            self._safety_violation_count += 1
            self._check_safety_escalation(event)

        # Keep history bounded
        if len(self._failure_history) > 1000:
            self._failure_history = self._failure_history[-500:]

        return event

    def _record_success(self) -> None:
        """Record a successful call."""
        self._state.total_successes += 1
        self._state.consecutive_successes += 1
        self._state.consecutive_failures = 0
        self._state.last_success_time = datetime.utcnow()

        if self._state.state == CircuitState.HALF_OPEN:
            self._state.success_count += 1
            if self._state.success_count >= self.config.success_threshold:
                self._transition_state(CircuitState.CLOSED)

    def _check_safety_escalation(self, event: FailureEvent) -> None:
        """Check if safety violations require escalation."""
        if not self.config.escalation_enabled:
            return

        if self._safety_violation_count >= self.config.safety_threshold:
            # Determine escalation level based on severity
            if event.severity >= 5:
                level = EscalationLevel.LEVEL_3
            elif event.severity >= 3:
                level = EscalationLevel.LEVEL_2
            else:
                level = EscalationLevel.LEVEL_1

            self._escalate(level, event)

    def _escalate(self, level: EscalationLevel, event: FailureEvent) -> None:
        """Trigger escalation callbacks."""
        logger.critical(f"ESCALATION {level.value}: {event.error_message}")

        for callback in self._escalation_callbacks[level]:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Escalation callback error: {e}")

    def _should_allow_call(self) -> Tuple[bool, str]:
        """Determine if a call should be allowed."""
        with self._lock:
            # Check rate limit
            if not self._check_rate_limit():
                return False, "Rate limit exceeded"

            self._clean_old_failures()

            if self._state.state == CircuitState.CLOSED:
                return True, "OK"

            elif self._state.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._state.opened_at:
                    elapsed = (datetime.utcnow() - self._state.opened_at).total_seconds()
                    if elapsed >= self.config.recovery_timeout_seconds:
                        self._transition_state(CircuitState.HALF_OPEN)
                        return True, "Testing recovery"
                return False, f"Circuit open since {self._state.opened_at}"

            elif self._state.state == CircuitState.HALF_OPEN:
                # Allow limited calls for testing
                return True, "Half-open testing"

            return False, "Unknown state"

    def call(
        self,
        func: Callable[..., T],
        *args,
        timeout: Optional[float] = None,
        fallback: Optional[Callable[..., T]] = None,
        **kwargs,
    ) -> T:
        """
        Execute a function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            timeout: Override default timeout
            fallback: Fallback function if circuit is open
            **kwargs: Function keyword arguments

        Returns:
            Function result or fallback result

        Raises:
            CircuitBreakerOpenError: If circuit is open and no fallback
        """
        allowed, reason = self._should_allow_call()

        if not allowed:
            if fallback:
                logger.warning(f"Circuit '{self.name}' blocked, using fallback: {reason}")
                return fallback(*args, **kwargs)
            raise CircuitBreakerOpenError(f"Circuit '{self.name}' is open: {reason}")

        with self._lock:
            self._record_call()

        try:
            # Execute with timeout
            timeout = timeout or self.config.timeout_seconds
            start_time = time.time()

            result = func(*args, **kwargs)

            elapsed = time.time() - start_time
            if elapsed > timeout:
                self._handle_timeout(elapsed, timeout, func.__name__)
            else:
                with self._lock:
                    self._record_success()

            return result

        except Exception as e:
            with self._lock:
                self._handle_exception(e, func.__name__)
            raise

    def _handle_timeout(self, elapsed: float, timeout: float, operation: str) -> None:
        """Handle timeout failure."""
        event = self._record_failure(
            FailureType.TIMEOUT,
            f"Operation timed out: {elapsed:.2f}s > {timeout:.2f}s",
            operation=operation,
            context={"elapsed": elapsed, "timeout": timeout},
        )

        if len(self._failure_window) >= self.config.failure_threshold:
            self._transition_state(CircuitState.OPEN)

    def _handle_exception(self, error: Exception, operation: str) -> None:
        """Handle exception failure."""
        event = self._record_failure(
            FailureType.EXCEPTION,
            str(error),
            operation=operation,
            context={"exception_type": type(error).__name__},
            severity=3 if "safety" in str(error).lower() else 1,
        )

        if len(self._failure_window) >= self.config.failure_threshold:
            self._transition_state(CircuitState.OPEN)

    def record_safety_violation(
        self,
        message: str,
        operation: str = "",
        context: Optional[Dict] = None,
        severity: int = 4,
    ) -> FailureEvent:
        """
        Record a safety-related violation.

        These are tracked separately and can trigger immediate escalation.
        """
        with self._lock:
            event = self._record_failure(
                FailureType.SAFETY,
                message,
                operation=operation,
                context=context,
                severity=severity,
            )

            # Safety violations may open circuit immediately
            if severity >= 4:
                self._transition_state(CircuitState.OPEN)

            return event

    def force_open(self, reason: str = "Manual override") -> None:
        """Force the circuit to open state."""
        with self._lock:
            logger.critical(f"CircuitBreaker '{self.name}' forced OPEN: {reason}")
            self._transition_state(CircuitState.OPEN)

    def force_close(self, reason: str = "Manual reset") -> None:
        """Force the circuit to closed state."""
        with self._lock:
            logger.warning(f"CircuitBreaker '{self.name}' forced CLOSED: {reason}")
            self._transition_state(CircuitState.CLOSED)
            self._safety_violation_count = 0

    def reset(self) -> None:
        """Reset the circuit breaker to initial state."""
        with self._lock:
            self._state = CircuitBreakerState()
            self._failure_window.clear()
            self._call_timestamps.clear()
            self._safety_violation_count = 0
            logger.info(f"CircuitBreaker '{self.name}' reset")

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.state.value,
                "failure_count": len(self._failure_window),
                "total_failures": self._state.total_failures,
                "total_successes": self._state.total_successes,
                "consecutive_failures": self._state.consecutive_failures,
                "consecutive_successes": self._state.consecutive_successes,
                "safety_violations": self._safety_violation_count,
                "last_failure": self._state.last_failure_time.isoformat() if self._state.last_failure_time else None,
                "last_success": self._state.last_success_time.isoformat() if self._state.last_success_time else None,
                "opened_at": self._state.opened_at.isoformat() if self._state.opened_at else None,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout_seconds,
                },
            }

    def get_failure_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent failure history."""
        with self._lock:
            return [
                {
                    "event_id": e.event_id,
                    "timestamp": e.timestamp.isoformat(),
                    "type": e.failure_type.value,
                    "operation": e.operation,
                    "message": e.error_message,
                    "severity": e.severity,
                    "provenance_hash": e.provenance_hash,
                }
                for e in self._failure_history[-limit:]
            ]


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


def circuit_breaker(
    breaker: CircuitBreaker,
    fallback: Optional[Callable] = None,
    timeout: Optional[float] = None,
):
    """
    Decorator to protect a function with a circuit breaker.

    Usage:
        @circuit_breaker(my_breaker)
        def risky_operation():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return breaker.call(func, *args, timeout=timeout, fallback=fallback, **kwargs)
        return wrapper
    return decorator


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Provides centralized monitoring and control of all circuit breakers.
    """

    _instance: Optional['CircuitBreakerRegistry'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'CircuitBreakerRegistry':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._registry_lock = threading.Lock()

    def register(self, breaker: CircuitBreaker) -> None:
        """Register a circuit breaker."""
        with self._registry_lock:
            self._breakers[breaker.name] = breaker
            logger.info(f"Registered circuit breaker: {breaker.name}")

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self._breakers.get(name)

    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        with self._registry_lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {name: cb.get_status() for name, cb in self._breakers.items()}

    def force_open_all(self, reason: str = "Emergency") -> None:
        """Force all circuit breakers to open state."""
        for breaker in self._breakers.values():
            breaker.force_open(reason)

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    return CircuitBreakerRegistry()


# Pre-defined circuit breakers for common emission monitoring operations
_registry = get_circuit_breaker_registry()

# CEMS data acquisition circuit breaker
CEMS_BREAKER = _registry.get_or_create(
    "cems_acquisition",
    CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout_seconds=30.0,
        safety_threshold=2,
    )
)

# Compliance calculation circuit breaker
COMPLIANCE_BREAKER = _registry.get_or_create(
    "compliance_engine",
    CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout_seconds=60.0,
        timeout_seconds=10.0,
    )
)

# Fugitive detection circuit breaker
FUGITIVE_BREAKER = _registry.get_or_create(
    "fugitive_detection",
    CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout_seconds=60.0,
        safety_threshold=3,
    )
)


# Export all public classes
__all__ = [
    "CircuitState",
    "FailureType",
    "EscalationLevel",
    "FailureEvent",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "CircuitBreakerRegistry",
    "circuit_breaker",
    "get_circuit_breaker_registry",
    "CEMS_BREAKER",
    "COMPLIANCE_BREAKER",
    "FUGITIVE_BREAKER",
]
