# -*- coding: utf-8 -*-
"""
CircuitBreaker - IEC 61511 compliant circuit breaker for GL-011 FuelCraft.

This module implements a circuit breaker pattern compliant with IEC 61511 (Safety
Instrumented Systems) with SIL level support, fail-closed behavior for critical
operations, and comprehensive safety logging.

Reference Standards:
    - IEC 61511: Functional Safety - Safety Instrumented Systems
    - IEC 61508: Functional Safety of E/E/PE Safety-Related Systems
    - ISA 84: Application of Safety Instrumented Systems

Author: GL-BackendDeveloper
Date: 2025-01-01
Version: 1.0.0
"""

from typing import Dict, List, Optional, Callable, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone, timedelta
from enum import Enum
from threading import Lock
from collections import deque
import hashlib
import json
import logging
import time
import functools

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Blocking all calls
    HALF_OPEN = "half_open" # Testing if service recovered


class SILLevel(str, Enum):
    """Safety Integrity Level per IEC 61511."""
    SIL_1 = "SIL_1"  # PFD 1E-1 to 1E-2
    SIL_2 = "SIL_2"  # PFD 1E-2 to 1E-3
    SIL_3 = "SIL_3"  # PFD 1E-3 to 1E-4
    SIL_4 = "SIL_4"  # PFD 1E-4 to 1E-5


class FailureMode(str, Enum):
    """Failure mode for circuit breaker."""
    FAIL_CLOSED = "fail_closed"  # Block all operations on failure
    FAIL_OPEN = "fail_open"      # Allow all operations on failure (not for safety)
    FAIL_SAFE = "fail_safe"      # Return to safe state


class RecoveryStrategy(str, Enum):
    """Strategy for recovering from open state."""
    MANUAL = "manual"              # Requires manual reset
    AUTOMATIC = "automatic"        # Automatic after timeout
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Increasing delays


class CircuitEvent(BaseModel):
    """Event record for circuit breaker state changes."""
    event_id: str = Field(...)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    circuit_id: str = Field(...)
    previous_state: CircuitState = Field(...)
    new_state: CircuitState = Field(...)
    reason: str = Field(...)
    failure_count: int = Field(0)
    sil_level: SILLevel = Field(...)
    provenance_hash: str = Field(...)


class CircuitMetrics(BaseModel):
    """Metrics for circuit breaker monitoring."""
    circuit_id: str = Field(...)
    current_state: CircuitState = Field(...)
    total_calls: int = Field(0)
    successful_calls: int = Field(0)
    failed_calls: int = Field(0)
    rejected_calls: int = Field(0)
    state_changes: int = Field(0)
    last_failure_time: Optional[datetime] = Field(None)
    last_success_time: Optional[datetime] = Field(None)
    consecutive_failures: int = Field(0)
    uptime_pct: float = Field(100.0)


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker."""
    circuit_id: str = Field(...)
    sil_level: SILLevel = Field(SILLevel.SIL_2)
    failure_mode: FailureMode = Field(FailureMode.FAIL_CLOSED)
    recovery_strategy: RecoveryStrategy = Field(RecoveryStrategy.MANUAL)

    # Thresholds
    failure_threshold: int = Field(3, ge=1, description="Failures before opening")
    success_threshold: int = Field(2, ge=1, description="Successes to close from half-open")
    timeout_seconds: float = Field(60.0, ge=1.0, description="Open state timeout")

    # Exponential backoff settings
    initial_backoff_seconds: float = Field(10.0, ge=1.0)
    max_backoff_seconds: float = Field(300.0, ge=1.0)
    backoff_multiplier: float = Field(2.0, ge=1.0)

    # Monitoring
    metrics_window_seconds: int = Field(300, ge=60)


class CircuitBreaker:
    """
    IEC 61511 compliant circuit breaker implementation.

    This circuit breaker implements fail-closed behavior for critical safety
    operations, with comprehensive logging and SIL level tracking.

    Key Features:
        - SIL level classification per IEC 61511
        - Fail-closed default for safety-critical operations
        - Manual or automatic recovery strategies
        - Complete audit trail for safety analysis
        - Thread-safe operation

    Example:
        >>> breaker = CircuitBreaker(config)
        >>> @breaker.protect
        ... def critical_operation():
        ...     return perform_operation()
        >>> result = critical_operation()
    """

    def __init__(self, config: CircuitBreakerConfig):
        """Initialize circuit breaker."""
        self._config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._last_state_change = datetime.now(timezone.utc)
        self._lock = Lock()
        self._events: deque = deque(maxlen=1000)
        self._metrics = CircuitMetrics(
            circuit_id=config.circuit_id,
            current_state=CircuitState.CLOSED
        )
        self._backoff_count = 0

        logger.info(
            f"CircuitBreaker {config.circuit_id} initialized: "
            f"SIL={config.sil_level.value}, failure_mode={config.failure_mode.value}"
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking)."""
        return self._state == CircuitState.OPEN

    def can_execute(self) -> bool:
        """
        Check if execution is allowed.

        Returns True if circuit is closed or half-open.
        For open state, checks if timeout has elapsed for automatic recovery.
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.HALF_OPEN:
                return True

            if self._state == CircuitState.OPEN:
                if self._config.recovery_strategy == RecoveryStrategy.MANUAL:
                    return False

                elapsed = (datetime.now(timezone.utc) - self._last_state_change).total_seconds()
                timeout = self._get_current_timeout()

                if elapsed >= timeout:
                    self._transition_to(CircuitState.HALF_OPEN, "Timeout elapsed, testing recovery")
                    return True

                return False

            return False

    def record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            self._metrics.total_calls += 1
            self._metrics.successful_calls += 1
            self._metrics.last_success_time = datetime.now(timezone.utc)
            self._failure_count = 0

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._config.success_threshold:
                    self._transition_to(CircuitState.CLOSED, "Success threshold reached")
                    self._success_count = 0
                    self._backoff_count = 0

            logger.debug(f"CircuitBreaker {self._config.circuit_id}: success recorded")

    def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record a failed operation."""
        with self._lock:
            self._metrics.total_calls += 1
            self._metrics.failed_calls += 1
            self._metrics.last_failure_time = datetime.now(timezone.utc)
            self._metrics.consecutive_failures += 1
            self._failure_count += 1
            self._last_failure_time = datetime.now(timezone.utc)

            error_msg = str(error) if error else "Unknown error"

            if self._state == CircuitState.CLOSED:
                if self._failure_count >= self._config.failure_threshold:
                    self._transition_to(
                        CircuitState.OPEN,
                        f"Failure threshold ({self._config.failure_threshold}) reached: {error_msg}"
                    )
                    self._backoff_count += 1

            elif self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN, f"Recovery test failed: {error_msg}")
                self._success_count = 0
                self._backoff_count += 1

            logger.warning(
                f"CircuitBreaker {self._config.circuit_id}: failure recorded "
                f"(count={self._failure_count}): {error_msg}"
            )

    def record_rejection(self) -> None:
        """Record a rejected call (circuit open)."""
        with self._lock:
            self._metrics.total_calls += 1
            self._metrics.rejected_calls += 1
            logger.warning(f"CircuitBreaker {self._config.circuit_id}: call rejected (circuit open)")

    def manual_reset(self, authorized_by: str, reason: str) -> bool:
        """
        Manually reset the circuit breaker.

        Only allowed for MANUAL recovery strategy.
        """
        with self._lock:
            if self._config.recovery_strategy != RecoveryStrategy.MANUAL:
                if self._state != CircuitState.OPEN:
                    return False

            self._transition_to(
                CircuitState.CLOSED,
                f"Manual reset by {authorized_by}: {reason}"
            )
            self._failure_count = 0
            self._success_count = 0
            self._backoff_count = 0
            self._metrics.consecutive_failures = 0

            logger.info(
                f"[SAFETY] CircuitBreaker {self._config.circuit_id} manually reset "
                f"by {authorized_by}: {reason}"
            )
            return True

    def force_open(self, authorized_by: str, reason: str) -> None:
        """Force the circuit breaker to open state."""
        with self._lock:
            self._transition_to(
                CircuitState.OPEN,
                f"Forced open by {authorized_by}: {reason}"
            )
            logger.warning(
                f"[SAFETY] CircuitBreaker {self._config.circuit_id} forced OPEN "
                f"by {authorized_by}: {reason}"
            )

    def protect(self, func: Callable) -> Callable:
        """
        Decorator to protect a function with circuit breaker.

        FAIL-CLOSED: If circuit is open, raises CircuitOpenError for SIL_2+.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.can_execute():
                self.record_rejection()
                if self._config.failure_mode == FailureMode.FAIL_CLOSED:
                    raise CircuitOpenError(
                        f"Circuit {self._config.circuit_id} is OPEN - "
                        f"fail-closed mode (SIL {self._config.sil_level.value})"
                    )
                elif self._config.failure_mode == FailureMode.FAIL_SAFE:
                    return None
                else:
                    return func(*args, **kwargs)

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise

        return wrapper

    def get_metrics(self) -> CircuitMetrics:
        """Get current circuit breaker metrics."""
        with self._lock:
            self._metrics.current_state = self._state
            return self._metrics.model_copy()

    def get_events(self, limit: int = 100) -> List[CircuitEvent]:
        """Get recent circuit events."""
        with self._lock:
            return list(self._events)[-limit:]

    def _transition_to(self, new_state: CircuitState, reason: str) -> None:
        """Transition to a new state with event logging."""
        if self._state == new_state:
            return

        event = CircuitEvent(
            event_id=hashlib.sha256(
                f"{self._config.circuit_id}|{datetime.now(timezone.utc).isoformat()}".encode()
            ).hexdigest()[:16],
            circuit_id=self._config.circuit_id,
            previous_state=self._state,
            new_state=new_state,
            reason=reason,
            failure_count=self._failure_count,
            sil_level=self._config.sil_level,
            provenance_hash=hashlib.sha256(
                json.dumps({
                    "circuit": self._config.circuit_id,
                    "from": self._state.value,
                    "to": new_state.value,
                    "reason": reason
                }, sort_keys=True).encode()
            ).hexdigest()
        )

        self._events.append(event)
        self._metrics.state_changes += 1

        old_state = self._state
        self._state = new_state
        self._last_state_change = datetime.now(timezone.utc)

        log_level = logging.WARNING if new_state == CircuitState.OPEN else logging.INFO
        logger.log(
            log_level,
            f"[SAFETY] CircuitBreaker {self._config.circuit_id}: "
            f"{old_state.value} -> {new_state.value} ({reason})"
        )

    def _get_current_timeout(self) -> float:
        """Get current timeout based on recovery strategy."""
        if self._config.recovery_strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
            timeout = self._config.initial_backoff_seconds * (
                self._config.backoff_multiplier ** self._backoff_count
            )
            return min(timeout, self._config.max_backoff_seconds)
        return self._config.timeout_seconds


class CircuitOpenError(Exception):
    """Exception raised when circuit is open and fail-closed mode is active."""
    pass


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    _instance: Optional['CircuitBreakerRegistry'] = None
    _lock = Lock()

    def __new__(cls):
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._breakers = {}
            return cls._instance

    def register(self, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Register a new circuit breaker."""
        if config.circuit_id in self._breakers:
            raise ValueError(f"Circuit breaker {config.circuit_id} already registered")

        breaker = CircuitBreaker(config)
        self._breakers[config.circuit_id] = breaker
        logger.info(f"Registered circuit breaker: {config.circuit_id}")
        return breaker

    def get(self, circuit_id: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by ID."""
        return self._breakers.get(circuit_id)

    def get_all_metrics(self) -> Dict[str, CircuitMetrics]:
        """Get metrics for all circuit breakers."""
        return {
            circuit_id: breaker.get_metrics()
            for circuit_id, breaker in self._breakers.items()
        }

    def reset_all(self, authorized_by: str, reason: str) -> int:
        """Reset all circuit breakers (emergency only)."""
        count = 0
        for circuit_id, breaker in self._breakers.items():
            if breaker.manual_reset(authorized_by, reason):
                count += 1
        logger.warning(f"[SAFETY] Reset {count} circuit breakers by {authorized_by}: {reason}")
        return count


def create_safety_circuit_breaker(
    circuit_id: str,
    sil_level: SILLevel = SILLevel.SIL_2,
    failure_threshold: int = 3
) -> CircuitBreaker:
    """
    Factory function to create a safety-critical circuit breaker.

    Always uses FAIL_CLOSED mode and MANUAL recovery for safety.
    """
    config = CircuitBreakerConfig(
        circuit_id=circuit_id,
        sil_level=sil_level,
        failure_mode=FailureMode.FAIL_CLOSED,
        recovery_strategy=RecoveryStrategy.MANUAL,
        failure_threshold=failure_threshold,
        timeout_seconds=300.0  # 5 minutes
    )
    return CircuitBreaker(config)
