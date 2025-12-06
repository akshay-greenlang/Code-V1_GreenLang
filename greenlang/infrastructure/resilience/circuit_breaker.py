"""
Circuit Breaker Implementation for GreenLang

This module provides a Resilience4j-style circuit breaker
for protecting services from cascading failures.

Features:
- State machine (CLOSED, OPEN, HALF_OPEN)
- Sliding window failure tracking
- Configurable thresholds
- Automatic recovery
- Metrics and observability

Example:
    >>> breaker = CircuitBreaker(config)
    >>> async with breaker:
    ...     result = await external_service()
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class FailureType(str, Enum):
    """Types of failures to track."""
    EXCEPTION = "exception"
    TIMEOUT = "timeout"
    SLOW_CALL = "slow_call"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    name: str = "default"
    # Failure thresholds
    failure_rate_threshold: float = 50.0  # Percentage
    slow_call_rate_threshold: float = 100.0  # Percentage
    slow_call_duration_threshold_ms: int = 60000  # 60 seconds
    # Sliding window
    sliding_window_size: int = 100
    sliding_window_type: str = "count"  # "count" or "time"
    minimum_number_of_calls: int = 10
    # State transition
    wait_duration_in_open_state_ms: int = 60000  # 60 seconds
    permitted_number_of_calls_in_half_open: int = 10
    # Recovery
    automatic_transition_from_open_to_half_open: bool = True
    # Exceptions
    record_exceptions: List[Type[Exception]] = field(default_factory=list)
    ignore_exceptions: List[Type[Exception]] = field(default_factory=list)


class CallResult(BaseModel):
    """Result of a call through the circuit breaker."""
    success: bool = Field(..., description="Call succeeded")
    duration_ms: float = Field(..., description="Call duration")
    exception: Optional[str] = Field(default=None, description="Exception type")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CircuitBreakerMetrics(BaseModel):
    """Metrics for circuit breaker."""
    state: CircuitState = Field(..., description="Current state")
    failure_rate: float = Field(default=0.0, description="Failure rate percentage")
    slow_call_rate: float = Field(default=0.0, description="Slow call rate percentage")
    total_calls: int = Field(default=0, description="Total calls")
    successful_calls: int = Field(default=0, description="Successful calls")
    failed_calls: int = Field(default=0, description="Failed calls")
    slow_calls: int = Field(default=0, description="Slow calls")
    not_permitted_calls: int = Field(default=0, description="Calls rejected by open circuit")
    state_transitions: int = Field(default=0, description="Number of state transitions")
    last_failure_time: Optional[datetime] = Field(default=None)
    last_success_time: Optional[datetime] = Field(default=None)


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, breaker_name: str, wait_time_ms: int):
        """Initialize error."""
        self.breaker_name = breaker_name
        self.wait_time_ms = wait_time_ms
        super().__init__(
            f"Circuit breaker '{breaker_name}' is open. "
            f"Wait {wait_time_ms}ms before retry."
        )


class SlidingWindow:
    """
    Sliding window for tracking call outcomes.

    Supports both count-based and time-based windows.
    """

    def __init__(
        self,
        size: int,
        window_type: str = "count"
    ):
        """
        Initialize sliding window.

        Args:
            size: Window size (count or seconds)
            window_type: "count" or "time"
        """
        self.size = size
        self.window_type = window_type
        self._results: List[CallResult] = []
        self._lock = asyncio.Lock()

    async def record(self, result: CallResult) -> None:
        """Record a call result."""
        async with self._lock:
            self._results.append(result)
            self._cleanup()

    def _cleanup(self) -> None:
        """Remove old entries."""
        if self.window_type == "count":
            if len(self._results) > self.size:
                self._results = self._results[-self.size:]
        else:
            cutoff = datetime.utcnow() - timedelta(seconds=self.size)
            self._results = [r for r in self._results if r.timestamp > cutoff]

    def get_failure_rate(self) -> float:
        """Calculate failure rate percentage."""
        if not self._results:
            return 0.0
        failed = sum(1 for r in self._results if not r.success)
        return (failed / len(self._results)) * 100

    def get_slow_call_rate(self, threshold_ms: int) -> float:
        """Calculate slow call rate percentage."""
        if not self._results:
            return 0.0
        slow = sum(1 for r in self._results if r.duration_ms > threshold_ms)
        return (slow / len(self._results)) * 100

    @property
    def call_count(self) -> int:
        """Get total call count."""
        return len(self._results)

    def clear(self) -> None:
        """Clear the window."""
        self._results.clear()


class CircuitBreaker:
    """
    Resilience4j-style circuit breaker.

    Protects services from cascading failures by opening
    the circuit when failure rate exceeds threshold.

    Attributes:
        config: Circuit breaker configuration
        state: Current circuit state
        metrics: Circuit breaker metrics

    Example:
        >>> config = CircuitBreakerConfig(
        ...     name="external-api",
        ...     failure_rate_threshold=50.0,
        ...     wait_duration_in_open_state_ms=30000
        ... )
        >>> breaker = CircuitBreaker(config)
        >>> @breaker
        ... async def call_api():
        ...     return await http_client.get("/data")
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration
        """
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._window = SlidingWindow(
            self.config.sliding_window_size,
            self.config.sliding_window_type
        )
        self._half_open_calls = 0
        self._opened_at: Optional[datetime] = None
        self._state_transitions = 0
        self._not_permitted_calls = 0
        self._lock = asyncio.Lock()
        self._state_change_callbacks: List[Callable] = []

        logger.info(f"CircuitBreaker '{self.config.name}' initialized")

    @property
    def state(self) -> CircuitState:
        """Get current state."""
        return self._state

    async def _set_state(self, new_state: CircuitState) -> None:
        """Set circuit state."""
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            self._state_transitions += 1

            if new_state == CircuitState.OPEN:
                self._opened_at = datetime.utcnow()
            elif new_state == CircuitState.HALF_OPEN:
                self._half_open_calls = 0

            logger.info(
                f"CircuitBreaker '{self.config.name}' "
                f"state changed: {old_state.value} -> {new_state.value}"
            )

            # Notify callbacks
            for callback in self._state_change_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(old_state, new_state)
                    else:
                        callback(old_state, new_state)
                except Exception as e:
                    logger.error(f"State change callback error: {e}")

    async def is_call_permitted(self) -> bool:
        """
        Check if a call is permitted.

        Returns:
            True if call is permitted
        """
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            elif self._state == CircuitState.OPEN:
                # Check if we should transition to HALF_OPEN
                if self.config.automatic_transition_from_open_to_half_open:
                    wait_time = self.config.wait_duration_in_open_state_ms / 1000
                    if self._opened_at:
                        elapsed = (datetime.utcnow() - self._opened_at).total_seconds()
                        if elapsed >= wait_time:
                            await self._set_state(CircuitState.HALF_OPEN)
                            return True
                return False

            else:  # HALF_OPEN
                if self._half_open_calls < self.config.permitted_number_of_calls_in_half_open:
                    self._half_open_calls += 1
                    return True
                return False

    async def record_success(self, duration_ms: float) -> None:
        """
        Record a successful call.

        Args:
            duration_ms: Call duration in milliseconds
        """
        result = CallResult(
            success=True,
            duration_ms=duration_ms
        )
        await self._window.record(result)

        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                # Check if we should close the circuit
                if self._half_open_calls >= self.config.permitted_number_of_calls_in_half_open:
                    failure_rate = self._window.get_failure_rate()
                    if failure_rate < self.config.failure_rate_threshold:
                        await self._set_state(CircuitState.CLOSED)
                        self._window.clear()

    async def record_failure(
        self,
        duration_ms: float,
        exception: Optional[Exception] = None
    ) -> None:
        """
        Record a failed call.

        Args:
            duration_ms: Call duration in milliseconds
            exception: Exception that occurred
        """
        # Check if exception should be recorded
        if exception:
            if self.config.ignore_exceptions:
                for exc_type in self.config.ignore_exceptions:
                    if isinstance(exception, exc_type):
                        return

            if self.config.record_exceptions:
                should_record = False
                for exc_type in self.config.record_exceptions:
                    if isinstance(exception, exc_type):
                        should_record = True
                        break
                if not should_record:
                    return

        result = CallResult(
            success=False,
            duration_ms=duration_ms,
            exception=type(exception).__name__ if exception else None
        )
        await self._window.record(result)

        async with self._lock:
            if self._state == CircuitState.CLOSED:
                # Check if we should open the circuit
                if self._window.call_count >= self.config.minimum_number_of_calls:
                    failure_rate = self._window.get_failure_rate()
                    slow_rate = self._window.get_slow_call_rate(
                        self.config.slow_call_duration_threshold_ms
                    )

                    if (failure_rate >= self.config.failure_rate_threshold or
                            slow_rate >= self.config.slow_call_rate_threshold):
                        await self._set_state(CircuitState.OPEN)

            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in HALF_OPEN transitions to OPEN
                await self._set_state(CircuitState.OPEN)

    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        if not await self.is_call_permitted():
            self._not_permitted_calls += 1
            wait_time = self.config.wait_duration_in_open_state_ms
            if self._opened_at:
                elapsed = (datetime.utcnow() - self._opened_at).total_seconds() * 1000
                wait_time = max(0, wait_time - int(elapsed))

            raise CircuitBreakerOpenError(self.config.name, wait_time)

        start_time = time.monotonic()

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            duration_ms = (time.monotonic() - start_time) * 1000
            await self.record_success(duration_ms)
            return result

        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            await self.record_failure(duration_ms, e)
            raise

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to wrap a function with circuit breaker.

        Args:
            func: Function to wrap

        Returns:
            Wrapped function
        """
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)

        return wrapper

    async def __aenter__(self) -> "CircuitBreaker":
        """Async context manager entry."""
        if not await self.is_call_permitted():
            wait_time = self.config.wait_duration_in_open_state_ms
            raise CircuitBreakerOpenError(self.config.name, wait_time)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        # Recording is done in execute method
        pass

    def on_state_change(
        self,
        callback: Callable[[CircuitState, CircuitState], None]
    ) -> None:
        """
        Register a state change callback.

        Args:
            callback: Function called on state changes
        """
        self._state_change_callbacks.append(callback)

    def get_metrics(self) -> CircuitBreakerMetrics:
        """
        Get circuit breaker metrics.

        Returns:
            Metrics object
        """
        return CircuitBreakerMetrics(
            state=self._state,
            failure_rate=self._window.get_failure_rate(),
            slow_call_rate=self._window.get_slow_call_rate(
                self.config.slow_call_duration_threshold_ms
            ),
            total_calls=self._window.call_count,
            successful_calls=sum(
                1 for r in self._window._results if r.success
            ),
            failed_calls=sum(
                1 for r in self._window._results if not r.success
            ),
            slow_calls=sum(
                1 for r in self._window._results
                if r.duration_ms > self.config.slow_call_duration_threshold_ms
            ),
            not_permitted_calls=self._not_permitted_calls,
            state_transitions=self._state_transitions,
        )

    async def reset(self) -> None:
        """Reset the circuit breaker."""
        async with self._lock:
            await self._set_state(CircuitState.CLOSED)
            self._window.clear()
            self._half_open_calls = 0
            self._opened_at = None
            self._not_permitted_calls = 0

        logger.info(f"CircuitBreaker '{self.config.name}' reset")

    async def force_open(self) -> None:
        """Force the circuit to open."""
        async with self._lock:
            await self._set_state(CircuitState.OPEN)

    async def force_close(self) -> None:
        """Force the circuit to close."""
        async with self._lock:
            await self._set_state(CircuitState.CLOSED)
            self._window.clear()
