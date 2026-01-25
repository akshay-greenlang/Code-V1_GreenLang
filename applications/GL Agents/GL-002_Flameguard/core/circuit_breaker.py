"""
GL-002 FLAMEGUARD - Circuit Breaker Module

This module implements the Circuit Breaker pattern for fault tolerance in
industrial control system integrations. Provides protection for OPC-UA,
Modbus, and other external service calls.

The circuit breaker prevents cascading failures by:
1. Tracking failure counts and timing
2. Opening the circuit when failures exceed thresholds
3. Allowing periodic recovery attempts in half-open state
4. Collecting metrics for monitoring and alerting

Standards Compliance:
    - IEC 61511 (Functional Safety - fail-safe behavior)
    - ISA-88 (Batch Control - state machine patterns)

Example:
    >>> breaker = CircuitBreaker(
    ...     name="modbus_plc",
    ...     failure_threshold=5,
    ...     recovery_timeout_s=30.0
    ... )
    >>> async with breaker:
    ...     result = await modbus_client.read_registers()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, TypeVar, Union
import asyncio
import hashlib
import logging
import time
import functools

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states following standard state machine pattern."""
    CLOSED = "closed"      # Normal operation, requests flow through
    OPEN = "open"          # Circuit tripped, requests fail fast
    HALF_OPEN = "half_open"  # Recovery testing, limited requests allowed


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""
    pass


class CircuitOpenError(CircuitBreakerError):
    """Raised when circuit is open and request cannot be made."""

    def __init__(self, breaker_name: str, time_until_retry: float) -> None:
        self.breaker_name = breaker_name
        self.time_until_retry = time_until_retry
        super().__init__(
            f"Circuit breaker '{breaker_name}' is OPEN. "
            f"Retry in {time_until_retry:.1f}s"
        )


class CircuitHalfOpenError(CircuitBreakerError):
    """Raised when too many requests attempted in half-open state."""

    def __init__(self, breaker_name: str) -> None:
        self.breaker_name = breaker_name
        super().__init__(
            f"Circuit breaker '{breaker_name}' is HALF_OPEN with max calls reached"
        )


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Number of failures before circuit opens
        recovery_timeout_s: Seconds to wait before attempting recovery
        half_open_max_calls: Max calls allowed in half-open state
        success_threshold: Successes needed in half-open to close circuit
        failure_rate_threshold: Failure rate (0-1) that triggers opening
        slow_call_duration_threshold_s: Duration to consider a call "slow"
        slow_call_rate_threshold: Rate of slow calls that triggers opening
        sliding_window_size: Size of sliding window for rate calculations
        sliding_window_type: Type of window ("count" or "time")
        wait_duration_in_half_open_s: Time to wait in half-open before closing
        record_exceptions: Exception types to record as failures
        ignore_exceptions: Exception types to NOT record as failures
    """
    failure_threshold: int = 5
    recovery_timeout_s: float = 30.0
    half_open_max_calls: int = 3
    success_threshold: int = 2
    failure_rate_threshold: float = 0.5
    slow_call_duration_threshold_s: float = 5.0
    slow_call_rate_threshold: float = 0.8
    sliding_window_size: int = 10
    sliding_window_type: str = "count"  # "count" or "time"
    wait_duration_in_half_open_s: float = 10.0
    record_exceptions: tuple = (Exception,)
    ignore_exceptions: tuple = ()

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.recovery_timeout_s <= 0:
            raise ValueError("recovery_timeout_s must be > 0")
        if self.half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be >= 1")
        if not 0 < self.failure_rate_threshold <= 1:
            raise ValueError("failure_rate_threshold must be in (0, 1]")


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring.

    Provides Prometheus-compatible metrics for observability.
    """
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0  # Calls rejected due to open circuit
    slow_calls: int = 0
    state_transitions: int = 0
    time_in_open_s: float = 0.0
    time_in_half_open_s: float = 0.0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_state_change_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    # Sliding window tracking
    _call_history: List[tuple] = field(default_factory=list)
    _failure_history: List[datetime] = field(default_factory=list)

    def record_success(self, duration_s: float) -> None:
        """Record a successful call."""
        now = datetime.now(timezone.utc)
        self.total_calls += 1
        self.successful_calls += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = now
        self._call_history.append((now, True, duration_s))

    def record_failure(self, duration_s: float = 0.0) -> None:
        """Record a failed call."""
        now = datetime.now(timezone.utc)
        self.total_calls += 1
        self.failed_calls += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = now
        self._failure_history.append(now)
        self._call_history.append((now, False, duration_s))

    def record_rejection(self) -> None:
        """Record a rejected call (circuit open)."""
        self.rejected_calls += 1

    def record_slow_call(self) -> None:
        """Record a slow call."""
        self.slow_calls += 1

    def record_state_transition(self) -> None:
        """Record a state transition."""
        self.state_transitions += 1
        self.last_state_change_time = datetime.now(timezone.utc)

    def get_failure_rate(self, window_size: int = 10) -> float:
        """Calculate failure rate over sliding window."""
        if not self._call_history:
            return 0.0
        recent = self._call_history[-window_size:]
        if not recent:
            return 0.0
        failures = sum(1 for _, success, _ in recent if not success)
        return failures / len(recent)

    def get_slow_call_rate(
        self,
        threshold_s: float,
        window_size: int = 10,
    ) -> float:
        """Calculate slow call rate over sliding window."""
        if not self._call_history:
            return 0.0
        recent = self._call_history[-window_size:]
        if not recent:
            return 0.0
        slow = sum(1 for _, _, duration in recent if duration >= threshold_s)
        return slow / len(recent)

    def cleanup_old_records(self, max_age_s: float = 3600.0) -> None:
        """Clean up old records to prevent memory growth."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_s)
        self._call_history = [
            (ts, success, dur)
            for ts, success, dur in self._call_history
            if ts > cutoff
        ]
        self._failure_history = [
            ts for ts in self._failure_history if ts > cutoff
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for API/monitoring."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "slow_calls": self.slow_calls,
            "state_transitions": self.state_transitions,
            "time_in_open_s": self.time_in_open_s,
            "time_in_half_open_s": self.time_in_half_open_s,
            "last_failure_time": (
                self.last_failure_time.isoformat()
                if self.last_failure_time else None
            ),
            "last_success_time": (
                self.last_success_time.isoformat()
                if self.last_success_time else None
            ),
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "failure_rate": self.get_failure_rate(),
        }


@dataclass
class CircuitBreakerState:
    """Internal state of the circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    opened_at: Optional[float] = None
    half_open_calls: int = 0
    half_open_successes: int = 0


class CircuitBreaker:
    """
    Circuit Breaker implementation for fault tolerance.

    This class implements the circuit breaker pattern to prevent cascading
    failures in distributed systems. It wraps external service calls and
    tracks their success/failure rates.

    States:
        CLOSED: Normal operation. Calls flow through.
        OPEN: Circuit is tripped. Calls fail fast with CircuitOpenError.
        HALF_OPEN: Testing recovery. Limited calls allowed.

    Attributes:
        name: Unique identifier for this breaker
        config: Configuration settings
        metrics: Collected metrics

    Example:
        >>> breaker = CircuitBreaker("scada_modbus")
        >>>
        >>> # Decorator usage
        >>> @breaker
        >>> async def read_plc():
        ...     return await modbus.read()
        >>>
        >>> # Context manager usage
        >>> async with breaker:
        ...     data = await opc_client.read_nodes()
        >>>
        >>> # Direct call usage
        >>> result = await breaker.call(modbus.read_registers)
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        failure_threshold: Optional[int] = None,
        recovery_timeout_s: Optional[float] = None,
        half_open_max_calls: Optional[int] = None,
        on_state_change: Optional[Callable[[str, CircuitState, CircuitState], None]] = None,
        on_failure: Optional[Callable[[str, Exception], None]] = None,
        on_success: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Initialize CircuitBreaker.

        Args:
            name: Unique name for this circuit breaker
            config: Full configuration object (overrides individual params)
            failure_threshold: Override config failure_threshold
            recovery_timeout_s: Override config recovery_timeout_s
            half_open_max_calls: Override config half_open_max_calls
            on_state_change: Callback for state transitions
            on_failure: Callback for failures
            on_success: Callback for successes
        """
        self.name = name

        # Build configuration
        if config:
            self.config = config
        else:
            self.config = CircuitBreakerConfig(
                failure_threshold=failure_threshold or 5,
                recovery_timeout_s=recovery_timeout_s or 30.0,
                half_open_max_calls=half_open_max_calls or 3,
            )

        # State management
        self._state = CircuitBreakerState()
        self._lock = asyncio.Lock()

        # Metrics
        self.metrics = CircuitBreakerMetrics()

        # Callbacks
        self._on_state_change = on_state_change
        self._on_failure = on_failure
        self._on_success = on_success

        # Timing
        self._opened_at: Optional[float] = None
        self._half_opened_at: Optional[float] = None

        logger.info(
            f"CircuitBreaker '{name}' initialized: "
            f"threshold={self.config.failure_threshold}, "
            f"timeout={self.config.recovery_timeout_s}s"
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state.state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self._state.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state.state == CircuitState.HALF_OPEN

    def _should_allow_request(self) -> bool:
        """Determine if a request should be allowed through."""
        if self._state.state == CircuitState.CLOSED:
            return True

        if self._state.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self._opened_at is not None:
                elapsed = time.monotonic() - self._opened_at
                if elapsed >= self.config.recovery_timeout_s:
                    # Transition to half-open
                    self._transition_to_half_open()
                    return True
            return False

        if self._state.state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open
            return self._state.half_open_calls < self.config.half_open_max_calls

        return False

    def _time_until_retry(self) -> float:
        """Calculate time remaining until retry is allowed."""
        if self._state.state != CircuitState.OPEN:
            return 0.0
        if self._opened_at is None:
            return 0.0
        elapsed = time.monotonic() - self._opened_at
        remaining = self.config.recovery_timeout_s - elapsed
        return max(0.0, remaining)

    def _transition_to_open(self) -> None:
        """Transition circuit to OPEN state."""
        old_state = self._state.state
        if old_state == CircuitState.OPEN:
            return

        self._state.state = CircuitState.OPEN
        self._opened_at = time.monotonic()
        self._state.half_open_calls = 0
        self._state.half_open_successes = 0

        self.metrics.record_state_transition()

        logger.warning(
            f"CircuitBreaker '{self.name}' OPENED: "
            f"failures={self._state.failure_count}, "
            f"threshold={self.config.failure_threshold}"
        )

        if self._on_state_change:
            try:
                self._on_state_change(self.name, old_state, CircuitState.OPEN)
            except Exception as e:
                logger.error(f"State change callback failed: {e}")

    def _transition_to_half_open(self) -> None:
        """Transition circuit to HALF_OPEN state."""
        old_state = self._state.state
        if old_state == CircuitState.HALF_OPEN:
            return

        self._state.state = CircuitState.HALF_OPEN
        self._half_opened_at = time.monotonic()
        self._state.half_open_calls = 0
        self._state.half_open_successes = 0

        self.metrics.record_state_transition()

        logger.info(
            f"CircuitBreaker '{self.name}' HALF_OPEN: "
            f"testing recovery"
        )

        if self._on_state_change:
            try:
                self._on_state_change(self.name, old_state, CircuitState.HALF_OPEN)
            except Exception as e:
                logger.error(f"State change callback failed: {e}")

    def _transition_to_closed(self) -> None:
        """Transition circuit to CLOSED state."""
        old_state = self._state.state
        if old_state == CircuitState.CLOSED:
            return

        # Track time spent in other states
        if old_state == CircuitState.OPEN and self._opened_at:
            self.metrics.time_in_open_s += time.monotonic() - self._opened_at
        elif old_state == CircuitState.HALF_OPEN and self._half_opened_at:
            self.metrics.time_in_half_open_s += time.monotonic() - self._half_opened_at

        self._state.state = CircuitState.CLOSED
        self._state.failure_count = 0
        self._state.success_count = 0
        self._opened_at = None
        self._half_opened_at = None

        self.metrics.record_state_transition()

        logger.info(f"CircuitBreaker '{self.name}' CLOSED: normal operation resumed")

        if self._on_state_change:
            try:
                self._on_state_change(self.name, old_state, CircuitState.CLOSED)
            except Exception as e:
                logger.error(f"State change callback failed: {e}")

    def _record_success(self, duration_s: float) -> None:
        """Record a successful call."""
        self._state.success_count += 1
        self.metrics.record_success(duration_s)

        if duration_s >= self.config.slow_call_duration_threshold_s:
            self.metrics.record_slow_call()

        if self._state.state == CircuitState.HALF_OPEN:
            self._state.half_open_successes += 1
            if self._state.half_open_successes >= self.config.success_threshold:
                self._transition_to_closed()

        if self._on_success:
            try:
                self._on_success(self.name)
            except Exception as e:
                logger.error(f"Success callback failed: {e}")

    def _record_failure(self, exception: Exception, duration_s: float = 0.0) -> None:
        """Record a failed call."""
        # Check if this exception should be ignored
        if isinstance(exception, self.config.ignore_exceptions):
            return

        # Check if this exception should be recorded
        if not isinstance(exception, self.config.record_exceptions):
            return

        self._state.failure_count += 1
        self._state.last_failure_time = time.monotonic()
        self.metrics.record_failure(duration_s)

        if self._on_failure:
            try:
                self._on_failure(self.name, exception)
            except Exception as e:
                logger.error(f"Failure callback failed: {e}")

        # Check if we should open the circuit
        if self._state.state == CircuitState.CLOSED:
            if self._state.failure_count >= self.config.failure_threshold:
                self._transition_to_open()
            elif self.metrics.get_failure_rate(
                self.config.sliding_window_size
            ) >= self.config.failure_rate_threshold:
                self._transition_to_open()
        elif self._state.state == CircuitState.HALF_OPEN:
            # Any failure in half-open opens the circuit again
            self._transition_to_open()

    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func execution

        Raises:
            CircuitOpenError: If circuit is open
            CircuitHalfOpenError: If half-open max calls exceeded
            Exception: Any exception raised by func
        """
        async with self._lock:
            if not self._should_allow_request():
                self.metrics.record_rejection()
                if self._state.state == CircuitState.OPEN:
                    raise CircuitOpenError(
                        self.name,
                        self._time_until_retry()
                    )
                else:
                    raise CircuitHalfOpenError(self.name)

            if self._state.state == CircuitState.HALF_OPEN:
                self._state.half_open_calls += 1

        start_time = time.monotonic()

        try:
            result = await func(*args, **kwargs)
            duration = time.monotonic() - start_time

            async with self._lock:
                self._record_success(duration)

            return result

        except Exception as e:
            duration = time.monotonic() - start_time

            async with self._lock:
                self._record_failure(e, duration)

            raise

    def __call__(
        self,
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        """
        Decorator to wrap async functions with circuit breaker.

        Example:
            >>> breaker = CircuitBreaker("api")
            >>> @breaker
            >>> async def call_api():
            ...     return await api.request()
        """
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await self.call(func, *args, **kwargs)
        return wrapper

    async def __aenter__(self) -> "CircuitBreaker":
        """Async context manager entry."""
        async with self._lock:
            if not self._should_allow_request():
                self.metrics.record_rejection()
                if self._state.state == CircuitState.OPEN:
                    raise CircuitOpenError(
                        self.name,
                        self._time_until_retry()
                    )
                else:
                    raise CircuitHalfOpenError(self.name)

            if self._state.state == CircuitState.HALF_OPEN:
                self._state.half_open_calls += 1

        self._context_start_time = time.monotonic()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Any,
    ) -> bool:
        """Async context manager exit."""
        duration = time.monotonic() - self._context_start_time

        async with self._lock:
            if exc_val is None:
                self._record_success(duration)
            else:
                self._record_failure(exc_val, duration)

        return False  # Don't suppress exceptions

    def reset(self) -> None:
        """Reset circuit breaker to initial closed state."""
        self._state = CircuitBreakerState()
        self._opened_at = None
        self._half_opened_at = None
        self.metrics = CircuitBreakerMetrics()
        logger.info(f"CircuitBreaker '{self.name}' reset to CLOSED")

    def force_open(self) -> None:
        """Force circuit to open state (for testing/maintenance)."""
        self._transition_to_open()
        logger.warning(f"CircuitBreaker '{self.name}' forced OPEN")

    def force_close(self) -> None:
        """Force circuit to closed state (for testing/maintenance)."""
        self._transition_to_closed()
        logger.info(f"CircuitBreaker '{self.name}' forced CLOSED")

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status for monitoring."""
        return {
            "name": self.name,
            "state": self._state.state.value,
            "failure_count": self._state.failure_count,
            "success_count": self._state.success_count,
            "half_open_calls": self._state.half_open_calls,
            "time_until_retry_s": self._time_until_retry(),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout_s": self.config.recovery_timeout_s,
                "half_open_max_calls": self.config.half_open_max_calls,
            },
            "metrics": self.metrics.to_dict(),
        }

    def get_provenance_hash(self) -> str:
        """Calculate SHA-256 hash of current state for audit trail."""
        state_str = (
            f"{self.name}|{self._state.state.value}|"
            f"{self._state.failure_count}|{self._state.success_count}|"
            f"{self.metrics.total_calls}"
        )
        return hashlib.sha256(state_str.encode()).hexdigest()


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Provides centralized access, monitoring, and control of all
    circuit breakers in the application.

    Example:
        >>> registry = CircuitBreakerRegistry()
        >>> scada_breaker = registry.get_or_create("scada_modbus")
        >>> status = registry.get_all_status()
    """

    _instance: Optional["CircuitBreakerRegistry"] = None

    def __new__(cls) -> "CircuitBreakerRegistry":
        """Singleton pattern for global registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._breakers = {}
            cls._instance._lock = asyncio.Lock()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    def __init__(self) -> None:
        """Initialize registry (no-op due to singleton)."""
        if not hasattr(self, "_breakers"):
            self._breakers: Dict[str, CircuitBreaker] = {}
            self._lock = asyncio.Lock()

    def register(self, breaker: CircuitBreaker) -> None:
        """Register a circuit breaker."""
        self._breakers[breaker.name] = breaker
        logger.debug(f"Registered circuit breaker: {breaker.name}")

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self._breakers.get(name)

    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        **kwargs: Any,
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        if name not in self._breakers:
            breaker = CircuitBreaker(name, config, **kwargs)
            self._breakers[name] = breaker
        return self._breakers[name]

    def remove(self, name: str) -> None:
        """Remove a circuit breaker from registry."""
        if name in self._breakers:
            del self._breakers[name]
            logger.debug(f"Removed circuit breaker: {name}")

    def get_all_status(self) -> Dict[str, Dict]:
        """Get status of all registered circuit breakers."""
        return {
            name: breaker.get_status()
            for name, breaker in self._breakers.items()
        }

    def get_open_breakers(self) -> List[str]:
        """Get names of all open circuit breakers."""
        return [
            name for name, breaker in self._breakers.items()
            if breaker.is_open
        ]

    def get_unhealthy_breakers(self) -> List[str]:
        """Get names of unhealthy (open or half-open) breakers."""
        return [
            name for name, breaker in self._breakers.items()
            if not breaker.is_closed
        ]

    def reset_all(self) -> None:
        """Reset all circuit breakers to closed state."""
        for breaker in self._breakers.values():
            breaker.reset()
        logger.info("All circuit breakers reset")

    def is_healthy(self) -> bool:
        """Check if all breakers are healthy (closed)."""
        return all(breaker.is_closed for breaker in self._breakers.values())

    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for all breakers."""
        total = len(self._breakers)
        open_count = sum(1 for b in self._breakers.values() if b.is_open)
        half_open_count = sum(1 for b in self._breakers.values() if b.is_half_open)
        closed_count = total - open_count - half_open_count

        return {
            "total_breakers": total,
            "closed": closed_count,
            "half_open": half_open_count,
            "open": open_count,
            "healthy": open_count == 0,
            "breakers": {
                name: breaker.state.value
                for name, breaker in self._breakers.items()
            },
        }


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout_s: float = 30.0,
    half_open_max_calls: int = 3,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator factory for creating circuit-breaker-protected functions.

    Args:
        name: Name for the circuit breaker
        failure_threshold: Failures before opening circuit
        recovery_timeout_s: Seconds to wait before recovery attempt
        half_open_max_calls: Max calls in half-open state

    Returns:
        Decorator function

    Example:
        >>> @circuit_breaker("modbus", failure_threshold=3)
        >>> async def read_modbus():
        ...     return await client.read()
    """
    registry = CircuitBreakerRegistry()
    breaker = registry.get_or_create(
        name,
        failure_threshold=failure_threshold,
        recovery_timeout_s=recovery_timeout_s,
        half_open_max_calls=half_open_max_calls,
    )

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await breaker.call(func, *args, **kwargs)
        return wrapper

    return decorator
