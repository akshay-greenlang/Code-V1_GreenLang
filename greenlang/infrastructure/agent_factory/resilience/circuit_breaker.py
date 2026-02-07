"""
Circuit Breaker - Agent Factory Resilience (INFRA-010)

Implements the Circuit Breaker pattern for agent execution resilience.
Tracks failure rates over a sliding window and transitions between
CLOSED (normal), OPEN (failing), and HALF_OPEN (testing) states to
prevent cascading failures across the GreenLang agent pipeline.

Classes:
    - CircuitBreakerState: Enumeration of circuit breaker states.
    - CircuitBreakerConfig: Configuration for circuit breaker behaviour.
    - CircuitOpenError: Raised when the circuit is open and calls are rejected.
    - CircuitBreaker: Core circuit breaker implementation.

Example:
    >>> config = CircuitBreakerConfig(failure_rate_threshold=0.5)
    >>> cb = CircuitBreaker("intake-agent", config)
    >>> async with cb:
    ...     result = await agent.process(data)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Deque, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State Enumeration
# ---------------------------------------------------------------------------


class CircuitBreakerState(str, Enum):
    """Circuit breaker states following the standard state machine."""

    CLOSED = "closed"
    """Normal operation. Calls pass through. Failures are tracked."""

    OPEN = "open"
    """Circuit is tripped. All calls are rejected immediately."""

    HALF_OPEN = "half_open"
    """Testing state. A limited number of calls are allowed to probe recovery."""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit breaker is open.

    Attributes:
        agent_key: The agent whose circuit is open.
        state: Current circuit breaker state.
        open_since: Timestamp when the circuit opened.
        wait_remaining_s: Seconds remaining before half-open transition.
    """

    def __init__(
        self,
        agent_key: str,
        state: CircuitBreakerState,
        open_since: float,
        wait_remaining_s: float,
    ) -> None:
        self.agent_key = agent_key
        self.state = state
        self.open_since = open_since
        self.wait_remaining_s = wait_remaining_s
        super().__init__(
            f"Circuit breaker for '{agent_key}' is {state.value}. "
            f"Wait {wait_remaining_s:.1f}s before retry."
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Configuration for a circuit breaker instance.

    Attributes:
        failure_rate_threshold: Fraction of failures to trip the circuit (0.0-1.0).
        slow_call_threshold_s: Calls slower than this are counted as slow (seconds).
        slow_call_rate_threshold: Fraction of slow calls to trip the circuit.
        wait_in_open_s: Seconds to wait in OPEN before transitioning to HALF_OPEN.
        half_open_test_requests: Number of test requests allowed in HALF_OPEN.
        sliding_window_size_s: Duration of the sliding window for metrics (seconds).
        minimum_calls: Minimum calls in window before failure rate is evaluated.
    """

    failure_rate_threshold: float = 0.5
    slow_call_threshold_s: float = 5.0
    slow_call_rate_threshold: float = 0.8
    wait_in_open_s: float = 60.0
    half_open_test_requests: int = 3
    sliding_window_size_s: float = 60.0
    minimum_calls: int = 5


# ---------------------------------------------------------------------------
# Internal Data
# ---------------------------------------------------------------------------


@dataclass
class _CallRecord:
    """Record of a single call through the circuit breaker."""

    timestamp: float
    duration_s: float
    success: bool


# ---------------------------------------------------------------------------
# Metrics (lightweight counters for Prometheus export)
# ---------------------------------------------------------------------------


@dataclass
class CircuitBreakerMetrics:
    """Observable metrics for a single circuit breaker.

    These counters are designed for export to Prometheus via a collector.

    Attributes:
        total_calls: Total calls attempted (including rejected).
        success_count: Calls that completed successfully.
        failure_count: Calls that raised an exception.
        rejected_count: Calls rejected because the circuit was open.
        slow_call_count: Calls that exceeded the slow-call threshold.
        fallback_count: Times a fallback was invoked.
        state_transitions: Total number of state transitions.
    """

    total_calls: int = 0
    success_count: int = 0
    failure_count: int = 0
    rejected_count: int = 0
    slow_call_count: int = 0
    fallback_count: int = 0
    state_transitions: int = 0


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

# Type alias for state-change callbacks
StateChangeCallback = Callable[
    [str, CircuitBreakerState, CircuitBreakerState], Awaitable[None]
]


class CircuitBreaker:
    """Circuit breaker for agent execution resilience.

    Monitors call outcomes over a sliding window and transitions through
    CLOSED -> OPEN -> HALF_OPEN states to protect downstream services
    from cascading failures.

    Thread-safe via asyncio.Lock for all state mutations.

    Attributes:
        agent_key: Unique identifier of the protected agent.
        config: Circuit breaker configuration.
        state: Current state of the circuit breaker.
        metrics: Observable metrics counters.
    """

    # Class-level registry: agent_key -> CircuitBreaker
    _registry: Dict[str, CircuitBreaker] = {}

    def __init__(
        self,
        agent_key: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> None:
        """Initialize a circuit breaker for the given agent.

        Args:
            agent_key: Unique identifier of the agent to protect.
            config: Optional configuration. Uses defaults if None.
        """
        self.agent_key = agent_key
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState.CLOSED
        self._lock = asyncio.Lock()
        self._records: Deque[_CallRecord] = deque()
        self._opened_at: float = 0.0
        self._half_open_successes: int = 0
        self._half_open_calls: int = 0
        self.metrics = CircuitBreakerMetrics()
        self._on_state_change: Optional[StateChangeCallback] = None

        # Register in class-level registry
        CircuitBreaker._registry[agent_key] = self
        logger.info(
            "CircuitBreaker created for '%s' (threshold=%.0f%%, window=%.0fs)",
            agent_key,
            self.config.failure_rate_threshold * 100,
            self.config.sliding_window_size_s,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitBreakerState:
        """Current circuit breaker state."""
        return self._state

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, agent_key: str) -> Optional[CircuitBreaker]:
        """Retrieve a circuit breaker from the registry.

        Args:
            agent_key: The agent key to look up.

        Returns:
            The CircuitBreaker instance, or None if not registered.
        """
        return cls._registry.get(agent_key)

    @classmethod
    def get_or_create(
        cls,
        agent_key: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get an existing circuit breaker or create a new one.

        Args:
            agent_key: The agent key.
            config: Configuration for a new instance.

        Returns:
            The circuit breaker for the agent.
        """
        if agent_key in cls._registry:
            return cls._registry[agent_key]
        return cls(agent_key, config)

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered circuit breakers. Used for testing."""
        cls._registry.clear()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_state_change(self, callback: StateChangeCallback) -> None:
        """Register a callback for state transitions.

        Args:
            callback: Async function(agent_key, old_state, new_state).
        """
        self._on_state_change = callback

    # ------------------------------------------------------------------
    # Manual Overrides
    # ------------------------------------------------------------------

    async def force_open(self) -> None:
        """Manually force the circuit into OPEN state."""
        async with self._lock:
            old = self._state
            self._state = CircuitBreakerState.OPEN
            self._opened_at = time.monotonic()
            self.metrics.state_transitions += 1
        logger.warning("CircuitBreaker '%s' FORCED OPEN", self.agent_key)
        await self._emit_state_change(old, CircuitBreakerState.OPEN)

    async def force_close(self) -> None:
        """Manually force the circuit into CLOSED state and reset counters."""
        async with self._lock:
            old = self._state
            self._state = CircuitBreakerState.CLOSED
            self._records.clear()
            self._half_open_successes = 0
            self._half_open_calls = 0
            self.metrics.state_transitions += 1
        logger.info("CircuitBreaker '%s' FORCED CLOSED", self.agent_key)
        await self._emit_state_change(old, CircuitBreakerState.CLOSED)

    async def reset(self) -> None:
        """Reset the circuit breaker to initial CLOSED state with cleared metrics."""
        async with self._lock:
            old = self._state
            self._state = CircuitBreakerState.CLOSED
            self._records.clear()
            self._half_open_successes = 0
            self._half_open_calls = 0
            self._opened_at = 0.0
            self.metrics = CircuitBreakerMetrics()
        logger.info("CircuitBreaker '%s' RESET", self.agent_key)
        if old != CircuitBreakerState.CLOSED:
            await self._emit_state_change(old, CircuitBreakerState.CLOSED)

    # ------------------------------------------------------------------
    # Context Manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> CircuitBreaker:
        """Acquire the circuit breaker before a call."""
        await self._before_call()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        """Record the call outcome after execution.

        Returns False so exceptions propagate to the caller.
        """
        success = exc_type is None
        # Duration is not tracked via context manager; use record_call for full tracking
        await self._after_call(success=success, duration_s=0.0)
        return False

    # ------------------------------------------------------------------
    # Explicit Call Recording
    # ------------------------------------------------------------------

    async def record_call(self, success: bool, duration_s: float) -> None:
        """Record the outcome of a call explicitly.

        Use this when the context manager form does not capture timing.

        Args:
            success: Whether the call succeeded.
            duration_s: Wall-clock duration of the call in seconds.
        """
        await self._after_call(success=success, duration_s=duration_s)

    # ------------------------------------------------------------------
    # Internal State Machine
    # ------------------------------------------------------------------

    async def _before_call(self) -> None:
        """Gate a call based on current state. Raises CircuitOpenError if rejected."""
        async with self._lock:
            self.metrics.total_calls += 1

            if self._state == CircuitBreakerState.CLOSED:
                return

            if self._state == CircuitBreakerState.OPEN:
                elapsed = time.monotonic() - self._opened_at
                if elapsed >= self.config.wait_in_open_s:
                    # Transition to HALF_OPEN
                    old = self._state
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._half_open_successes = 0
                    self._half_open_calls = 0
                    self.metrics.state_transitions += 1
                    logger.info(
                        "CircuitBreaker '%s': OPEN -> HALF_OPEN after %.1fs",
                        self.agent_key, elapsed,
                    )
                    # Schedule callback outside lock
                    asyncio.get_event_loop().call_soon(
                        lambda: asyncio.ensure_future(
                            self._emit_state_change(old, CircuitBreakerState.HALF_OPEN)
                        )
                    )
                    return
                else:
                    remaining = self.config.wait_in_open_s - elapsed
                    self.metrics.rejected_count += 1
                    raise CircuitOpenError(
                        self.agent_key,
                        self._state,
                        self._opened_at,
                        remaining,
                    )

            if self._state == CircuitBreakerState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_test_requests:
                    self.metrics.rejected_count += 1
                    raise CircuitOpenError(
                        self.agent_key,
                        self._state,
                        self._opened_at,
                        0.0,
                    )
                self._half_open_calls += 1

    async def _after_call(self, success: bool, duration_s: float) -> None:
        """Process the outcome of a call and potentially transition state."""
        now = time.monotonic()
        is_slow = duration_s > self.config.slow_call_threshold_s

        async with self._lock:
            if success:
                self.metrics.success_count += 1
            else:
                self.metrics.failure_count += 1
            if is_slow:
                self.metrics.slow_call_count += 1

            record = _CallRecord(
                timestamp=now,
                duration_s=duration_s,
                success=success,
            )
            self._records.append(record)

            # Prune old records outside the sliding window
            cutoff = now - self.config.sliding_window_size_s
            while self._records and self._records[0].timestamp < cutoff:
                self._records.popleft()

            if self._state == CircuitBreakerState.HALF_OPEN:
                await self._evaluate_half_open(success)
            elif self._state == CircuitBreakerState.CLOSED:
                await self._evaluate_closed()

    async def _evaluate_closed(self) -> None:
        """Evaluate whether to trip the circuit from CLOSED to OPEN.

        Must be called while holding self._lock.
        """
        total = len(self._records)
        if total < self.config.minimum_calls:
            return

        failure_count = sum(1 for r in self._records if not r.success)
        failure_rate = failure_count / total

        slow_count = sum(
            1 for r in self._records
            if r.duration_s > self.config.slow_call_threshold_s
        )
        slow_rate = slow_count / total

        should_open = (
            failure_rate >= self.config.failure_rate_threshold
            or slow_rate >= self.config.slow_call_rate_threshold
        )

        if should_open:
            old = self._state
            self._state = CircuitBreakerState.OPEN
            self._opened_at = time.monotonic()
            self.metrics.state_transitions += 1
            logger.warning(
                "CircuitBreaker '%s': CLOSED -> OPEN "
                "(failure_rate=%.2f, slow_rate=%.2f, calls=%d)",
                self.agent_key, failure_rate, slow_rate, total,
            )
            asyncio.get_event_loop().call_soon(
                lambda: asyncio.ensure_future(
                    self._emit_state_change(old, CircuitBreakerState.OPEN)
                )
            )

    async def _evaluate_half_open(self, success: bool) -> None:
        """Evaluate whether to close or re-open from HALF_OPEN.

        Must be called while holding self._lock.
        """
        if not success:
            # Any failure in half-open re-opens the circuit
            old = self._state
            self._state = CircuitBreakerState.OPEN
            self._opened_at = time.monotonic()
            self._half_open_successes = 0
            self._half_open_calls = 0
            self.metrics.state_transitions += 1
            logger.warning(
                "CircuitBreaker '%s': HALF_OPEN -> OPEN (test request failed)",
                self.agent_key,
            )
            asyncio.get_event_loop().call_soon(
                lambda: asyncio.ensure_future(
                    self._emit_state_change(old, CircuitBreakerState.OPEN)
                )
            )
            return

        self._half_open_successes += 1
        if self._half_open_successes >= self.config.half_open_test_requests:
            old = self._state
            self._state = CircuitBreakerState.CLOSED
            self._records.clear()
            self._half_open_successes = 0
            self._half_open_calls = 0
            self.metrics.state_transitions += 1
            logger.info(
                "CircuitBreaker '%s': HALF_OPEN -> CLOSED (all %d test requests passed)",
                self.agent_key, self.config.half_open_test_requests,
            )
            asyncio.get_event_loop().call_soon(
                lambda: asyncio.ensure_future(
                    self._emit_state_change(old, CircuitBreakerState.CLOSED)
                )
            )

    async def _emit_state_change(
        self,
        old_state: CircuitBreakerState,
        new_state: CircuitBreakerState,
    ) -> None:
        """Fire the on_state_change callback if registered."""
        if self._on_state_change is not None:
            try:
                await self._on_state_change(self.agent_key, old_state, new_state)
            except Exception as exc:
                logger.error(
                    "State change callback failed for '%s': %s",
                    self.agent_key, exc,
                )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return a diagnostic snapshot of the circuit breaker state.

        Returns:
            Dictionary with current state, metrics, and configuration.
        """
        total = len(self._records)
        failure_count = sum(1 for r in self._records if not r.success)
        return {
            "agent_key": self.agent_key,
            "state": self._state.value,
            "window_calls": total,
            "window_failures": failure_count,
            "failure_rate": failure_count / total if total > 0 else 0.0,
            "metrics": {
                "total_calls": self.metrics.total_calls,
                "success_count": self.metrics.success_count,
                "failure_count": self.metrics.failure_count,
                "rejected_count": self.metrics.rejected_count,
                "slow_call_count": self.metrics.slow_call_count,
                "state_transitions": self.metrics.state_transitions,
            },
        }


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerMetrics",
    "CircuitBreakerState",
    "CircuitOpenError",
]
