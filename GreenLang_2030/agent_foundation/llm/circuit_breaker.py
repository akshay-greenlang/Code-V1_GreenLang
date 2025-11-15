"""
CircuitBreaker - Production-ready circuit breaker pattern implementation.

Implements the circuit breaker pattern to prevent cascading failures when
external services (LLM APIs) are down or slow.

States:
- CLOSED: Normal operation, requests flow through
- OPEN: Service is failing, requests fail fast
- HALF_OPEN: Testing if service recovered, limited requests allowed

Features:
- Configurable failure threshold (default: 5 failures)
- Configurable recovery timeout (default: 60 seconds)
- Exponential backoff for recovery attempts
- Thread-safe implementation

Example:
    >>> breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
    >>> async with breaker:
    ...     result = await call_external_api()
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    total_requests: int = 0
    rejected_requests: int = 0
    state_changes: int = 0


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after


class CircuitBreaker:
    """
    Production-ready circuit breaker implementation.

    The circuit breaker monitors failures and automatically opens the circuit
    when the failure threshold is exceeded, preventing additional requests
    until the service recovers.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        half_open_max_calls: Maximum concurrent calls in HALF_OPEN state
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
        name: str = "default",
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before opening (default: 5)
            recovery_timeout: Seconds to wait before testing recovery (default: 60)
            half_open_max_calls: Max concurrent calls in half-open state (default: 1)
            name: Name for logging and identification
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.name = name

        # State tracking
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._last_state_change: float = time.time()
        self._half_open_calls = 0

        # Statistics
        self._stats = CircuitBreakerStats()

        # Lock for thread safety
        self._lock = asyncio.Lock()

        self._logger = logging.getLogger(f"{__name__}.{name}")
        self._logger.info(
            f"Initialized CircuitBreaker: threshold={failure_threshold}, "
            f"recovery={recovery_timeout}s"
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        self._stats.state = self._state
        self._stats.failure_count = self._failure_count
        self._stats.success_count = self._success_count
        return self._stats

    async def __aenter__(self):
        """Context manager entry - check if request should proceed."""
        async with self._lock:
            self._stats.total_requests += 1

            # Check if we should attempt recovery
            if self._state == CircuitState.OPEN:
                time_since_failure = time.time() - (self._last_failure_time or 0)

                if time_since_failure >= self.recovery_timeout:
                    # Try half-open state
                    self._transition_to(CircuitState.HALF_OPEN)
                else:
                    # Still in failure state
                    self._stats.rejected_requests += 1
                    retry_after = self.recovery_timeout - time_since_failure
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is OPEN (threshold: {self.failure_threshold} failures)",
                        retry_after=retry_after,
                    )

            # In HALF_OPEN, limit concurrent requests
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    self._stats.rejected_requests += 1
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is HALF_OPEN (testing recovery)",
                        retry_after=self.recovery_timeout,
                    )
                self._half_open_calls += 1

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - record success or failure."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls -= 1

            if exc_type is None:
                # Success
                await self._record_success()
            else:
                # Failure
                await self._record_failure()

        return False  # Don't suppress exceptions

    async def call(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Call a function with circuit breaker protection.

        Args:
            func: Async function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Any exception from func
        """
        async with self:
            return await func(*args, **kwargs)

    async def _record_success(self) -> None:
        """Record successful call."""
        self._success_count += 1
        self._stats.success_count += 1
        self._stats.last_success_time = datetime.utcnow()

        if self._state == CircuitState.HALF_OPEN:
            # Recovery successful, close circuit
            self._logger.info(f"Circuit breaker '{self.name}' recovered - transitioning to CLOSED")
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0

    async def _record_failure(self) -> None:
        """Record failed call."""
        self._failure_count += 1
        self._stats.last_failure_time = datetime.utcnow()
        self._last_failure_time = time.time()

        self._logger.warning(
            f"Circuit breaker '{self.name}' recorded failure "
            f"({self._failure_count}/{self.failure_threshold})"
        )

        # Check if we should open circuit
        if self._state == CircuitState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                self._logger.error(
                    f"Circuit breaker '{self.name}' opening - "
                    f"failure threshold ({self.failure_threshold}) exceeded"
                )
                self._transition_to(CircuitState.OPEN)

        elif self._state == CircuitState.HALF_OPEN:
            # Failed during recovery, go back to open
            self._logger.error(
                f"Circuit breaker '{self.name}' failed during recovery - "
                f"returning to OPEN state"
            )
            self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.time()
        self._stats.state_changes += 1

        self._logger.info(
            f"Circuit breaker '{self.name}' transitioned: {old_state.value} -> {new_state.value}"
        )

        # Reset counters on state change
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0

    async def reset(self) -> None:
        """Manually reset circuit breaker to CLOSED state."""
        async with self._lock:
            self._logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED")
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None

    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    def is_open(self) -> bool:
        """Check if circuit is open (failing)."""
        return self._state == CircuitState.OPEN

    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN


# Example usage
if __name__ == "__main__":
    import asyncio

    async def flaky_api_call(should_fail: bool = False):
        """Simulate an API call that might fail."""
        await asyncio.sleep(0.1)
        if should_fail:
            raise Exception("API call failed")
        return "Success"

    async def main():
        """Test circuit breaker."""
        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=5.0,
            name="test-api"
        )

        # Successful calls
        for i in range(3):
            try:
                async with breaker:
                    result = await flaky_api_call(should_fail=False)
                    print(f"Call {i+1}: {result}")
            except Exception as e:
                print(f"Call {i+1} failed: {e}")

        # Failing calls to trigger circuit breaker
        for i in range(5):
            try:
                async with breaker:
                    result = await flaky_api_call(should_fail=True)
                    print(f"Call {i+4}: {result}")
            except CircuitBreakerOpenError as e:
                print(f"Call {i+4}: Circuit breaker OPEN - {e}")
            except Exception as e:
                print(f"Call {i+4} failed: {e}")

        print(f"\nCircuit state: {breaker.state}")
        print(f"Stats: {breaker.stats}")

        # Wait for recovery
        print("\nWaiting for recovery...")
        await asyncio.sleep(6)

        # Try again after recovery
        try:
            async with breaker:
                result = await flaky_api_call(should_fail=False)
                print(f"After recovery: {result}")
        except Exception as e:
            print(f"After recovery failed: {e}")

        print(f"\nFinal state: {breaker.state}")
        print(f"Final stats: {breaker.stats}")

    asyncio.run(main())
