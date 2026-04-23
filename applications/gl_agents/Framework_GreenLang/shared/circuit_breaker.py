"""
GreenLang Framework - Circuit Breaker Module

Implements the Circuit Breaker pattern for fault tolerance in agent pipelines.
Provides automatic failure detection, recovery, and metrics collection.

Based on:
- Martin Fowler's Circuit Breaker pattern
- Netflix Hystrix patterns
- Resilience4j implementation patterns

The circuit breaker has three states:
- CLOSED: Normal operation, requests pass through
- OPEN: Failures exceeded threshold, requests are rejected
- HALF_OPEN: Testing if service has recovered

Example:
    >>> from greenlang.shared.circuit_breaker import CircuitBreaker
    >>>
    >>> breaker = CircuitBreaker(
    ...     name="erp_connection",
    ...     failure_threshold=5,
    ...     recovery_timeout=30.0,
    ... )
    >>>
    >>> @breaker
    ... def call_erp_service(data: dict) -> dict:
    ...     return external_service.call(data)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from functools import wraps
from threading import Lock, RLock
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    Awaitable,
)
import asyncio
import hashlib
import json
import logging
import time
import uuid


logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class CircuitState(Enum):
    """
    Circuit breaker states.

    CLOSED: Normal operation - requests pass through
    OPEN: Circuit tripped - requests are rejected immediately
    HALF_OPEN: Testing recovery - limited requests allowed
    """
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""
    pass


class CircuitOpenError(CircuitBreakerError):
    """Raised when circuit is open and request is rejected."""

    def __init__(
        self,
        message: str,
        breaker_name: str,
        time_until_recovery: float,
    ):
        super().__init__(message)
        self.breaker_name = breaker_name
        self.time_until_recovery = time_until_recovery


class CircuitHalfOpenError(CircuitBreakerError):
    """Raised when circuit is half-open and request limit exceeded."""

    def __init__(self, message: str, breaker_name: str):
        super().__init__(message)
        self.breaker_name = breaker_name


@dataclass
class CircuitMetrics:
    """
    Metrics for circuit breaker monitoring.

    Provides detailed statistics for observability and alerting.
    """
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_transitions: int = 0
    time_in_open_state_ms: float = 0.0
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    average_response_time_ms: float = 0.0
    _response_times: List[float] = field(default_factory=list)
    _max_response_times: int = 1000

    def record_success(self, response_time_ms: float) -> None:
        """Record a successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = datetime.now(timezone.utc)
        self._update_response_time(response_time_ms)

    def record_failure(self, response_time_ms: float) -> None:
        """Record a failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = datetime.now(timezone.utc)
        self._update_response_time(response_time_ms)

    def record_rejection(self) -> None:
        """Record a rejected request (circuit open)."""
        self.rejected_requests += 1

    def record_state_transition(self) -> None:
        """Record a state transition."""
        self.state_transitions += 1

    def _update_response_time(self, response_time_ms: float) -> None:
        """Update average response time."""
        self._response_times.append(response_time_ms)
        if len(self._response_times) > self._max_response_times:
            self._response_times = self._response_times[-self._max_response_times:]

        if self._response_times:
            self.average_response_time_ms = (
                sum(self._response_times) / len(self._response_times)
            )

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100.0

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "rejected_requests": self.rejected_requests,
            "success_rate": round(self.success_rate, 2),
            "failure_rate": round(self.failure_rate, 2),
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "last_success_time": (
                self.last_success_time.isoformat() if self.last_success_time else None
            ),
            "state_transitions": self.state_transitions,
            "time_in_open_state_ms": self.time_in_open_state_ms,
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "average_response_time_ms": round(self.average_response_time_ms, 2),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.rejected_requests = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.state_transitions = 0
        self.time_in_open_state_ms = 0.0
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.average_response_time_ms = 0.0
        self._response_times = []


@dataclass
class StateTransitionEvent:
    """Record of a circuit breaker state transition."""
    event_id: str
    breaker_name: str
    from_state: CircuitState
    to_state: CircuitState
    timestamp: datetime
    reason: str
    metrics_snapshot: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "breaker_name": self.breaker_name,
            "from_state": self.from_state.name,
            "to_state": self.to_state.name,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "metrics_snapshot": self.metrics_snapshot,
        }


class CircuitBreaker:
    """
    Circuit Breaker implementation for fault tolerance.

    Implements a state machine with three states:
    - CLOSED: Normal operation
    - OPEN: Failure threshold exceeded, requests rejected
    - HALF_OPEN: Testing if service has recovered

    Thread-safe implementation suitable for concurrent access.

    Example:
        >>> breaker = CircuitBreaker(
        ...     name="external_api",
        ...     failure_threshold=5,
        ...     recovery_timeout=30.0,
        ...     success_threshold=3,
        ... )
        >>>
        >>> @breaker
        ... def call_api(data):
        ...     return requests.post(url, json=data)
        ...
        >>> try:
        ...     result = call_api({"key": "value"})
        ... except CircuitOpenError:
        ...     # Handle circuit open
        ...     pass
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 3,
        half_open_max_calls: int = 3,
        excluded_exceptions: Optional[Set[type]] = None,
        on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None,
        on_failure: Optional[Callable[[Exception], None]] = None,
        on_success: Optional[Callable[[Any], None]] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Unique identifier for this circuit breaker
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before testing recovery
            success_threshold: Successes needed in HALF_OPEN to close circuit
            half_open_max_calls: Max concurrent calls allowed in HALF_OPEN state
            excluded_exceptions: Exceptions that should not count as failures
            on_state_change: Callback for state transitions
            on_failure: Callback for failures
            on_success: Callback for successes
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.half_open_max_calls = half_open_max_calls
        self.excluded_exceptions = excluded_exceptions or set()

        # Callbacks
        self._on_state_change = on_state_change
        self._on_failure = on_failure
        self._on_success = on_success

        # State
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._opened_at: Optional[datetime] = None
        self._half_open_calls = 0

        # Thread safety
        self._lock = RLock()

        # Metrics and events
        self._metrics = CircuitMetrics()
        self._transition_history: List[StateTransitionEvent] = []
        self._max_history = 100

        logger.info(
            f"Circuit breaker '{name}' initialized: "
            f"failure_threshold={failure_threshold}, "
            f"recovery_timeout={recovery_timeout}s"
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.state == CircuitState.HALF_OPEN

    @property
    def metrics(self) -> CircuitMetrics:
        """Get circuit breaker metrics."""
        return self._metrics

    @property
    def transition_history(self) -> List[StateTransitionEvent]:
        """Get state transition history."""
        return self._transition_history.copy()

    def _transition_to(self, new_state: CircuitState, reason: str) -> None:
        """
        Transition to a new state.

        Args:
            new_state: Target state
            reason: Reason for transition
        """
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state
        self._metrics.record_state_transition()

        # Track time in OPEN state
        now = datetime.now(timezone.utc)
        if old_state == CircuitState.OPEN and self._opened_at:
            open_duration = (now - self._opened_at).total_seconds() * 1000
            self._metrics.time_in_open_state_ms += open_duration

        # Record transition event
        event = StateTransitionEvent(
            event_id=str(uuid.uuid4()),
            breaker_name=self.name,
            from_state=old_state,
            to_state=new_state,
            timestamp=now,
            reason=reason,
            metrics_snapshot=self._metrics.to_dict(),
        )
        self._transition_history.append(event)
        if len(self._transition_history) > self._max_history:
            self._transition_history = self._transition_history[-self._max_history:]

        # State-specific actions
        if new_state == CircuitState.OPEN:
            self._opened_at = now
            self._half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0
            self._half_open_calls = 0
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
            self._opened_at = None

        logger.info(
            f"Circuit breaker '{self.name}' transitioned: "
            f"{old_state.name} -> {new_state.name} ({reason})"
        )

        # Invoke callback
        if self._on_state_change:
            try:
                self._on_state_change(old_state, new_state)
            except Exception as e:
                logger.warning(f"State change callback failed: {e}")

    def _should_allow_request(self) -> Tuple[bool, Optional[str]]:
        """
        Check if request should be allowed.

        Returns:
            Tuple of (allowed, rejection_reason)
        """
        now = datetime.now(timezone.utc)

        if self._state == CircuitState.CLOSED:
            return True, None

        elif self._state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self._opened_at:
                elapsed = (now - self._opened_at).total_seconds()
                if elapsed >= self.recovery_timeout:
                    # Transition to HALF_OPEN
                    self._transition_to(
                        CircuitState.HALF_OPEN,
                        f"Recovery timeout ({self.recovery_timeout}s) elapsed"
                    )
                    return True, None
                else:
                    time_remaining = self.recovery_timeout - elapsed
                    return False, f"Circuit open, {time_remaining:.1f}s until recovery"
            return False, "Circuit open"

        elif self._state == CircuitState.HALF_OPEN:
            # Allow limited requests in HALF_OPEN state
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True, None
            else:
                return False, "Half-open request limit exceeded"

        return False, "Unknown state"

    def _record_success(self, response_time_ms: float) -> None:
        """
        Record a successful request.

        Args:
            response_time_ms: Request response time in milliseconds
        """
        self._metrics.record_success(response_time_ms)

        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._transition_to(
                    CircuitState.CLOSED,
                    f"Success threshold ({self.success_threshold}) reached"
                )

        if self._on_success:
            try:
                self._on_success(None)
            except Exception as e:
                logger.warning(f"Success callback failed: {e}")

    def _record_failure(self, error: Exception, response_time_ms: float) -> None:
        """
        Record a failed request.

        Args:
            error: The exception that occurred
            response_time_ms: Request response time in milliseconds
        """
        # Check if exception is excluded
        if type(error) in self.excluded_exceptions:
            logger.debug(
                f"Circuit breaker '{self.name}': "
                f"Exception {type(error).__name__} is excluded"
            )
            return

        self._metrics.record_failure(response_time_ms)
        self._failure_count += 1
        self._last_failure_time = datetime.now(timezone.utc)

        if self._state == CircuitState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                self._transition_to(
                    CircuitState.OPEN,
                    f"Failure threshold ({self.failure_threshold}) exceeded"
                )

        elif self._state == CircuitState.HALF_OPEN:
            # Single failure in HALF_OPEN opens circuit
            self._transition_to(
                CircuitState.OPEN,
                "Failure during recovery test"
            )

        if self._on_failure:
            try:
                self._on_failure(error)
            except Exception as e:
                logger.warning(f"Failure callback failed: {e}")

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute a function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitOpenError: If circuit is open
            CircuitHalfOpenError: If half-open limit exceeded
        """
        with self._lock:
            allowed, reason = self._should_allow_request()

            if not allowed:
                self._metrics.record_rejection()
                if self._state == CircuitState.OPEN:
                    time_remaining = 0.0
                    if self._opened_at:
                        elapsed = (datetime.now(timezone.utc) - self._opened_at).total_seconds()
                        time_remaining = max(0, self.recovery_timeout - elapsed)
                    raise CircuitOpenError(
                        f"Circuit breaker '{self.name}' is open: {reason}",
                        breaker_name=self.name,
                        time_until_recovery=time_remaining,
                    )
                else:
                    raise CircuitHalfOpenError(
                        f"Circuit breaker '{self.name}' half-open limit: {reason}",
                        breaker_name=self.name,
                    )

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            response_time_ms = (time.time() - start_time) * 1000

            with self._lock:
                self._record_success(response_time_ms)

            return result

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000

            with self._lock:
                self._record_failure(e, response_time_ms)

            raise

    async def execute_async(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute an async function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitOpenError: If circuit is open
            CircuitHalfOpenError: If half-open limit exceeded
        """
        with self._lock:
            allowed, reason = self._should_allow_request()

            if not allowed:
                self._metrics.record_rejection()
                if self._state == CircuitState.OPEN:
                    time_remaining = 0.0
                    if self._opened_at:
                        elapsed = (datetime.now(timezone.utc) - self._opened_at).total_seconds()
                        time_remaining = max(0, self.recovery_timeout - elapsed)
                    raise CircuitOpenError(
                        f"Circuit breaker '{self.name}' is open: {reason}",
                        breaker_name=self.name,
                        time_until_recovery=time_remaining,
                    )
                else:
                    raise CircuitHalfOpenError(
                        f"Circuit breaker '{self.name}' half-open limit: {reason}",
                        breaker_name=self.name,
                    )

        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            response_time_ms = (time.time() - start_time) * 1000

            with self._lock:
                self._record_success(response_time_ms)

            return result

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000

            with self._lock:
                self._record_failure(e, response_time_ms)

            raise

    def __call__(self, func: F) -> F:
        """
        Use circuit breaker as a decorator.

        Args:
            func: Function to wrap

        Returns:
            Wrapped function

        Example:
            >>> @circuit_breaker
            ... def call_service(data):
            ...     return service.call(data)
        """
        is_async = asyncio.iscoroutinefunction(func)

        if is_async:
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await self.execute_async(func, *args, **kwargs)
            return async_wrapper  # type: ignore
        else:
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return self.execute(func, *args, **kwargs)
            return sync_wrapper  # type: ignore

    def reset(self) -> None:
        """Reset circuit breaker to CLOSED state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED, "Manual reset")
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._opened_at = None
            self._half_open_calls = 0
            logger.info(f"Circuit breaker '{self.name}' reset to CLOSED")

    def force_open(self, reason: str = "Manual override") -> None:
        """Force circuit breaker to OPEN state."""
        with self._lock:
            self._transition_to(CircuitState.OPEN, reason)
            logger.warning(f"Circuit breaker '{self.name}' forced OPEN: {reason}")

    def force_close(self, reason: str = "Manual override") -> None:
        """Force circuit breaker to CLOSED state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED, reason)
            logger.info(f"Circuit breaker '{self.name}' forced CLOSED: {reason}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current circuit breaker status.

        Returns:
            Dictionary with current status and metrics
        """
        with self._lock:
            time_until_recovery = None
            if self._state == CircuitState.OPEN and self._opened_at:
                elapsed = (datetime.now(timezone.utc) - self._opened_at).total_seconds()
                time_until_recovery = max(0, self.recovery_timeout - elapsed)

            return {
                "name": self.name,
                "state": self._state.name,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "success_threshold": self.success_threshold,
                "time_until_recovery": time_until_recovery,
                "last_failure_time": (
                    self._last_failure_time.isoformat()
                    if self._last_failure_time else None
                ),
                "metrics": self._metrics.to_dict(),
            }


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Provides centralized access to all circuit breakers in an application.

    Example:
        >>> registry = CircuitBreakerRegistry()
        >>> registry.register(CircuitBreaker("api_service"))
        >>> registry.register(CircuitBreaker("database"))
        >>>
        >>> api_breaker = registry.get("api_service")
        >>> health = registry.get_health_status()
    """

    def __init__(self):
        """Initialize registry."""
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = Lock()

    def register(self, breaker: CircuitBreaker) -> None:
        """
        Register a circuit breaker.

        Args:
            breaker: Circuit breaker to register
        """
        with self._lock:
            self._breakers[breaker.name] = breaker
            logger.info(f"Registered circuit breaker: {breaker.name}")

    def unregister(self, name: str) -> Optional[CircuitBreaker]:
        """
        Unregister a circuit breaker.

        Args:
            name: Name of circuit breaker to remove

        Returns:
            The removed circuit breaker or None
        """
        with self._lock:
            breaker = self._breakers.pop(name, None)
            if breaker:
                logger.info(f"Unregistered circuit breaker: {name}")
            return breaker

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """
        Get a circuit breaker by name.

        Args:
            name: Circuit breaker name

        Returns:
            CircuitBreaker or None if not found
        """
        with self._lock:
            return self._breakers.get(name)

    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        **kwargs: Any,
    ) -> CircuitBreaker:
        """
        Get existing or create new circuit breaker.

        Args:
            name: Circuit breaker name
            failure_threshold: Failures before opening
            recovery_timeout: Seconds before recovery test
            **kwargs: Additional CircuitBreaker arguments

        Returns:
            CircuitBreaker instance
        """
        with self._lock:
            if name not in self._breakers:
                breaker = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    **kwargs,
                )
                self._breakers[name] = breaker
            return self._breakers[name]

    def get_all(self) -> Dict[str, CircuitBreaker]:
        """Get all registered circuit breakers."""
        with self._lock:
            return self._breakers.copy()

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of all circuit breakers.

        Returns:
            Dictionary with overall health and individual statuses
        """
        with self._lock:
            statuses = {}
            open_count = 0
            half_open_count = 0

            for name, breaker in self._breakers.items():
                status = breaker.get_status()
                statuses[name] = status

                if breaker.is_open:
                    open_count += 1
                elif breaker.is_half_open:
                    half_open_count += 1

            total = len(self._breakers)
            closed_count = total - open_count - half_open_count

            # Determine overall health
            if open_count > 0:
                health = "DEGRADED" if closed_count > 0 else "CRITICAL"
            elif half_open_count > 0:
                health = "RECOVERING"
            else:
                health = "HEALTHY"

            return {
                "overall_health": health,
                "total_breakers": total,
                "closed": closed_count,
                "open": open_count,
                "half_open": half_open_count,
                "breakers": statuses,
            }

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            logger.info("All circuit breakers reset")


# Global registry instance
CIRCUIT_BREAKER_REGISTRY = CircuitBreakerRegistry()


def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    success_threshold: int = 3,
    excluded_exceptions: Optional[Set[type]] = None,
    use_registry: bool = True,
) -> Callable[[F], F]:
    """
    Decorator factory for circuit breaker protection.

    Args:
        name: Circuit breaker name (uses function name if not provided)
        failure_threshold: Failures before opening circuit
        recovery_timeout: Seconds before recovery test
        success_threshold: Successes needed to close circuit
        excluded_exceptions: Exceptions that don't count as failures
        use_registry: Whether to register in global registry

    Returns:
        Decorator function

    Example:
        >>> @circuit_breaker(name="api_call", failure_threshold=3)
        ... def call_api(data):
        ...     return requests.post(url, json=data)
    """
    def decorator(func: F) -> F:
        breaker_name = name or func.__name__

        if use_registry:
            breaker = CIRCUIT_BREAKER_REGISTRY.get_or_create(
                name=breaker_name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                success_threshold=success_threshold,
                excluded_exceptions=excluded_exceptions,
            )
        else:
            breaker = CircuitBreaker(
                name=breaker_name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                success_threshold=success_threshold,
                excluded_exceptions=excluded_exceptions,
            )

        return breaker(func)

    return decorator


__all__ = [
    # Core classes
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "CircuitMetrics",
    "StateTransitionEvent",
    # Enums
    "CircuitState",
    # Exceptions
    "CircuitBreakerError",
    "CircuitOpenError",
    "CircuitHalfOpenError",
    # Decorator
    "circuit_breaker",
    # Global registry
    "CIRCUIT_BREAKER_REGISTRY",
]
