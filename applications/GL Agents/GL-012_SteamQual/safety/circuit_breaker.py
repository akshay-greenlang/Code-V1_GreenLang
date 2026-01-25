"""
GL-012 STEAMQUAL - Circuit Breaker Pattern Implementation

This module implements the Circuit Breaker pattern for resilient external service
calls in the SteamQual Steam Quality Controller. It provides fault tolerance for
external integrations (OPC-UA, MQTT, SCADA, Historian) and prevents cascade failures.

The Circuit Breaker pattern is essential for:
1. Preventing cascade failures when external services are degraded
2. Providing fast-fail behavior to maintain system responsiveness
3. Enabling automatic recovery when services become healthy
4. Supporting safety-critical operations by isolating failures

State Diagram:

    CLOSED ----[failure_threshold exceeded]----> OPEN
       ^                                            |
       |                                            v
       +---[success]<--- HALF_OPEN <---[recovery_timeout]

    CLOSED: Normal operation, requests pass through
    OPEN: Failures exceeded threshold, requests fail immediately
    HALF_OPEN: Testing if service recovered, limited requests allowed

Reference Standards:
    - IEC 61508 (Functional Safety) - Fault Tolerance Requirements
    - IEC 61511 (SIS for Process Industries) - System Availability
    - NIST SP 800-160 (Systems Security Engineering) - Resilience Patterns
    - Martin Fowler's Circuit Breaker Pattern (Enterprise Architecture)

FAIL-SAFE Design:
When in doubt, the circuit breaker OPENS (fails safe). This prevents
potentially dangerous operations when external systems are unreliable.

Example:
    >>> from safety.circuit_breaker import SteamQualCircuitBreaker, circuit_protected
    >>>
    >>> # Manual circuit breaker usage
    >>> breaker = SteamQualCircuitBreaker(
    ...     name="scada_connector",
    ...     failure_threshold=5,
    ...     recovery_timeout_seconds=30.0
    ... )
    >>>
    >>> async with breaker.protect():
    ...     result = await scada_client.read_quality_data()
    >>>
    >>> # Decorator usage
    >>> @circuit_protected(name="historian_api", failure_threshold=3)
    ... async def fetch_from_historian(query: str) -> Dict:
    ...     return await historian_client.query(query)

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
)

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Type variables for generic circuit breaker
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# ENUMERATIONS
# =============================================================================


class CircuitBreakerState(str, Enum):
    """
    Circuit breaker state enumeration.

    States follow the standard circuit breaker pattern:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Circuit is tripped, requests fail immediately (FAIL-SAFE)
    - HALF_OPEN: Recovery testing, limited requests allowed

    Reference: Martin Fowler's Circuit Breaker Pattern
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerEvent(str, Enum):
    """Events emitted by circuit breaker state transitions."""

    FAILURE_RECORDED = "failure_recorded"
    SUCCESS_RECORDED = "success_recorded"
    CIRCUIT_OPENED = "circuit_opened"
    CIRCUIT_HALF_OPENED = "circuit_half_opened"
    CIRCUIT_CLOSED = "circuit_closed"
    CALL_BLOCKED = "call_blocked"
    CALL_ALLOWED = "call_allowed"
    DEGRADED_MODE_ENTERED = "degraded_mode_entered"
    DEGRADED_MODE_EXITED = "degraded_mode_exited"


# =============================================================================
# EXCEPTIONS
# =============================================================================


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""

    def __init__(
        self,
        message: str,
        breaker_name: str,
        state: CircuitBreakerState,
    ) -> None:
        super().__init__(message)
        self.breaker_name = breaker_name
        self.state = state


class CircuitOpenError(CircuitBreakerError):
    """Raised when circuit is open and calls are blocked (FAIL-SAFE behavior)."""

    def __init__(
        self,
        breaker_name: str,
        time_until_half_open: float,
    ) -> None:
        message = (
            f"Circuit breaker '{breaker_name}' is OPEN (fail-safe). "
            f"Retry in {time_until_half_open:.1f} seconds."
        )
        super().__init__(message, breaker_name, CircuitBreakerState.OPEN)
        self.time_until_half_open = time_until_half_open


class CircuitHalfOpenError(CircuitBreakerError):
    """Raised when circuit is half-open and max test calls exceeded."""

    def __init__(self, breaker_name: str) -> None:
        message = (
            f"Circuit breaker '{breaker_name}' is HALF_OPEN. "
            f"Maximum test calls in progress."
        )
        super().__init__(message, breaker_name, CircuitBreakerState.HALF_OPEN)


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================


class CircuitBreakerConfig(BaseModel):
    """
    Configuration for circuit breaker behavior.

    Attributes:
        name: Unique identifier for this circuit breaker
        failure_threshold: Number of failures before opening circuit
        recovery_timeout_seconds: Time to wait before attempting recovery
        half_open_max_calls: Maximum concurrent calls in half-open state
        success_threshold: Successes needed in half-open to close circuit
        failure_rate_threshold: Alternative: failure rate to trip circuit
        slow_call_duration_threshold_ms: Calls slower than this count as slow
        slow_call_rate_threshold: Rate of slow calls to trip circuit

    Reference:
        - Netflix Hystrix defaults
        - Resilience4j CircuitBreaker configuration
        - IEC 61508 fault tolerance recommendations
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique circuit breaker identifier"
    )
    failure_threshold: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Failures before opening circuit (IEC 61508 recommended: 3-5)"
    )
    recovery_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=600.0,
        description="Seconds before attempting recovery"
    )
    half_open_max_calls: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max concurrent calls in half-open state"
    )
    success_threshold: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Successes needed to close circuit"
    )
    failure_rate_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional: failure rate threshold (0.0-1.0)"
    )
    slow_call_duration_threshold_ms: float = Field(
        default=5000.0,
        ge=100.0,
        le=60000.0,
        description="Slow call threshold in milliseconds"
    )
    slow_call_rate_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Slow call rate to consider as failure"
    )
    sliding_window_size: int = Field(
        default=10,
        ge=5,
        le=100,
        description="Size of sliding window for metrics"
    )
    enabled: bool = Field(
        default=True,
        description="Enable/disable circuit breaker"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name contains only safe characters."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Name must contain only alphanumeric, underscore, or hyphen")
        return v


class CircuitBreakerMetrics(BaseModel):
    """
    Metrics for circuit breaker monitoring.

    These metrics support:
    - Prometheus/Grafana dashboards
    - SIL-level availability calculations
    - Trend analysis for predictive maintenance
    """

    name: str = Field(..., description="Circuit breaker name")
    state: CircuitBreakerState = Field(..., description="Current state")
    failure_count: int = Field(default=0, ge=0, description="Total failures")
    success_count: int = Field(default=0, ge=0, description="Total successes")
    consecutive_failures: int = Field(default=0, ge=0, description="Consecutive failures")
    consecutive_successes: int = Field(default=0, ge=0, description="Consecutive successes")
    last_failure_time: Optional[datetime] = Field(
        default=None,
        description="Last failure timestamp"
    )
    last_success_time: Optional[datetime] = Field(
        default=None,
        description="Last success timestamp"
    )
    state_changed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last state change timestamp"
    )
    calls_blocked: int = Field(default=0, ge=0, description="Calls blocked by open circuit")
    total_calls: int = Field(default=0, ge=0, description="Total call attempts")
    avg_call_duration_ms: float = Field(default=0.0, ge=0.0, description="Average call duration")

    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failure_count / self.total_calls

    @property
    def availability(self) -> float:
        """Calculate availability (success rate)."""
        return 1.0 - self.failure_rate

    def calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash for audit trail."""
        content = (
            f"{self.name}|{self.state.value}|{self.failure_count}|"
            f"{self.success_count}|{self.state_changed_at.isoformat()}"
        )
        return hashlib.sha256(content.encode()).hexdigest()


class CircuitBreakerAuditRecord(BaseModel):
    """
    Audit record for circuit breaker state changes.

    Supports IEC 61508 audit trail requirements for safety-critical systems.
    """

    record_id: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f"),
        description="Unique record identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Record timestamp"
    )
    breaker_name: str = Field(..., description="Circuit breaker name")
    event: CircuitBreakerEvent = Field(..., description="Event type")
    previous_state: Optional[CircuitBreakerState] = Field(
        default=None,
        description="Previous state"
    )
    new_state: CircuitBreakerState = Field(..., description="New state")
    failure_count: int = Field(default=0, description="Failure count at time of event")
    error_message: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Error message if applicable"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            content = (
                f"{self.record_id}|{self.timestamp.isoformat()}|"
                f"{self.breaker_name}|{self.event.value}|{self.new_state.value}"
            )
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


# =============================================================================
# SLIDING WINDOW FOR METRICS
# =============================================================================


@dataclass
class CallRecord:
    """Record of a single call for sliding window analysis."""

    timestamp: float
    duration_ms: float
    success: bool
    error: Optional[str] = None


class SlidingWindow:
    """
    Thread-safe sliding window for circuit breaker metrics.

    Maintains a fixed-size window of recent call records for
    calculating failure rates and slow call rates.
    """

    def __init__(self, size: int = 10) -> None:
        self._size = size
        self._records: List[CallRecord] = []
        self._lock = threading.Lock()

    def record(self, record: CallRecord) -> None:
        """Add a record to the window."""
        with self._lock:
            self._records.append(record)
            if len(self._records) > self._size:
                self._records.pop(0)

    def get_failure_rate(self) -> float:
        """Calculate failure rate from window."""
        with self._lock:
            if not self._records:
                return 0.0
            failures = sum(1 for r in self._records if not r.success)
            return failures / len(self._records)

    def get_slow_call_rate(self, threshold_ms: float) -> float:
        """Calculate slow call rate from window."""
        with self._lock:
            if not self._records:
                return 0.0
            slow = sum(1 for r in self._records if r.duration_ms > threshold_ms)
            return slow / len(self._records)

    def get_avg_duration(self) -> float:
        """Calculate average call duration."""
        with self._lock:
            if not self._records:
                return 0.0
            return sum(r.duration_ms for r in self._records) / len(self._records)

    def clear(self) -> None:
        """Clear all records."""
        with self._lock:
            self._records.clear()

    @property
    def count(self) -> int:
        """Get current record count."""
        with self._lock:
            return len(self._records)


# =============================================================================
# CIRCUIT BREAKER IMPLEMENTATION
# =============================================================================


class SteamQualCircuitBreaker:
    """
    Production-grade Circuit Breaker for GL-012 STEAMQUAL.

    Implements the circuit breaker pattern with FAIL-SAFE behavior:
    - When uncertain, the circuit OPENS (blocks calls)
    - Graceful degradation when external systems are unreliable
    - Automatic recovery testing after timeout

    Features:
    - Thread-safe state management
    - Sliding window metrics
    - Async context manager support
    - Comprehensive audit logging
    - Prometheus-compatible metrics

    Design follows:
    - IEC 61508 fault tolerance requirements
    - Netflix Hystrix / Resilience4j patterns
    - Martin Fowler's Circuit Breaker pattern

    Example:
        >>> breaker = SteamQualCircuitBreaker(
        ...     name="scada_quality_reader",
        ...     failure_threshold=5,
        ...     recovery_timeout_seconds=30.0
        ... )
        >>>
        >>> async with breaker.protect():
        ...     result = await scada_client.read_quality()
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0,
        half_open_max_calls: int = 3,
        success_threshold: int = 2,
        config: Optional[CircuitBreakerConfig] = None,
        event_callback: Optional[Callable[[CircuitBreakerAuditRecord], None]] = None,
    ) -> None:
        """
        Initialize circuit breaker.

        Args:
            name: Unique identifier for this circuit breaker
            failure_threshold: Failures before opening (default: 5 per IEC 61508)
            recovery_timeout_seconds: Recovery wait time (default: 30s)
            half_open_max_calls: Max test calls in half-open (default: 3)
            success_threshold: Successes to close circuit (default: 2)
            config: Optional full configuration object
            event_callback: Optional callback for state change events
        """
        if config:
            self._config = config
        else:
            self._config = CircuitBreakerConfig(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout_seconds=recovery_timeout_seconds,
                half_open_max_calls=half_open_max_calls,
                success_threshold=success_threshold,
            )

        self._state = CircuitBreakerState.CLOSED
        self._state_lock = threading.Lock()
        self._state_changed_at = time.monotonic()

        # Counters
        self._failure_count = 0
        self._success_count = 0
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._calls_blocked = 0
        self._total_calls = 0
        self._half_open_calls = 0

        # Timestamps
        self._last_failure_time: Optional[float] = None
        self._last_success_time: Optional[float] = None
        self._opened_at: Optional[float] = None

        # Sliding window
        self._window = SlidingWindow(self._config.sliding_window_size)

        # Event callback
        self._event_callback = event_callback

        # Audit trail
        self._audit_records: List[CircuitBreakerAuditRecord] = []

        logger.info(
            "SteamQualCircuitBreaker initialized: name=%s, failure_threshold=%d, "
            "recovery_timeout=%.1fs",
            self._config.name,
            self._config.failure_threshold,
            self._config.recovery_timeout_seconds,
        )

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def name(self) -> str:
        """Get circuit breaker name."""
        return self._config.name

    @property
    def state(self) -> CircuitBreakerState:
        """Get current state (thread-safe)."""
        with self._state_lock:
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitBreakerState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing safe)."""
        return self.state == CircuitBreakerState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.state == CircuitBreakerState.HALF_OPEN

    @property
    def config(self) -> CircuitBreakerConfig:
        """Get configuration."""
        return self._config

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def _transition_to(
        self,
        new_state: CircuitBreakerState,
        reason: str = "",
    ) -> None:
        """
        Transition to a new state (thread-safe).

        Args:
            new_state: Target state
            reason: Reason for transition
        """
        with self._state_lock:
            if self._state == new_state:
                return

            old_state = self._state
            self._state = new_state
            self._state_changed_at = time.monotonic()

            # Reset counters on state change
            if new_state == CircuitBreakerState.HALF_OPEN:
                self._half_open_calls = 0
                self._consecutive_successes = 0
            elif new_state == CircuitBreakerState.CLOSED:
                self._consecutive_failures = 0
                self._failure_count = 0
                self._window.clear()
            elif new_state == CircuitBreakerState.OPEN:
                self._opened_at = time.monotonic()

            # Determine event type
            event_type = {
                CircuitBreakerState.OPEN: CircuitBreakerEvent.CIRCUIT_OPENED,
                CircuitBreakerState.HALF_OPEN: CircuitBreakerEvent.CIRCUIT_HALF_OPENED,
                CircuitBreakerState.CLOSED: CircuitBreakerEvent.CIRCUIT_CLOSED,
            }[new_state]

            # Create audit record
            record = CircuitBreakerAuditRecord(
                breaker_name=self._config.name,
                event=event_type,
                previous_state=old_state,
                new_state=new_state,
                failure_count=self._failure_count,
                context={"reason": reason},
            )
            self._audit_records.append(record)

            # Invoke callback
            if self._event_callback:
                try:
                    self._event_callback(record)
                except Exception as e:
                    logger.error("Event callback failed: %s", e)

            logger.warning(
                "SteamQualCircuitBreaker state transition: %s -> %s (name=%s, reason=%s)",
                old_state.value,
                new_state.value,
                self._config.name,
                reason,
            )

    def _get_time_until_half_open(self) -> float:
        """Get seconds until circuit enters half-open state."""
        if self._state != CircuitBreakerState.OPEN:
            return 0.0
        if self._opened_at is None:
            return self._config.recovery_timeout_seconds
        elapsed = time.monotonic() - self._opened_at
        return max(0.0, self._config.recovery_timeout_seconds - elapsed)

    # =========================================================================
    # CALL RECORDING
    # =========================================================================

    def record_success(self, duration_ms: float = 0.0) -> None:
        """
        Record a successful call.

        Args:
            duration_ms: Call duration in milliseconds
        """
        with self._state_lock:
            self._success_count += 1
            self._consecutive_successes += 1
            self._consecutive_failures = 0
            self._last_success_time = time.monotonic()
            self._total_calls += 1

            # Add to sliding window
            self._window.record(CallRecord(
                timestamp=time.monotonic(),
                duration_ms=duration_ms,
                success=True,
            ))

            # State transitions
            if self._state == CircuitBreakerState.HALF_OPEN:
                if self._consecutive_successes >= self._config.success_threshold:
                    self._transition_to(
                        CircuitBreakerState.CLOSED,
                        reason=f"Success threshold reached ({self._consecutive_successes})"
                    )

            logger.debug(
                "SteamQualCircuitBreaker success: name=%s, consecutive=%d",
                self._config.name,
                self._consecutive_successes,
            )

    def record_failure(
        self,
        error: Optional[Exception] = None,
        duration_ms: float = 0.0,
    ) -> None:
        """
        Record a failed call.

        Args:
            error: Exception that caused the failure
            duration_ms: Call duration in milliseconds
        """
        with self._state_lock:
            self._failure_count += 1
            self._consecutive_failures += 1
            self._consecutive_successes = 0
            self._last_failure_time = time.monotonic()
            self._total_calls += 1

            error_msg = str(error) if error else None

            # Add to sliding window
            self._window.record(CallRecord(
                timestamp=time.monotonic(),
                duration_ms=duration_ms,
                success=False,
                error=error_msg,
            ))

            # State transitions
            if self._state == CircuitBreakerState.CLOSED:
                should_open = False
                reason = ""

                # Check failure count threshold
                if self._consecutive_failures >= self._config.failure_threshold:
                    should_open = True
                    reason = f"Failure threshold reached ({self._consecutive_failures})"

                # Check failure rate threshold
                elif self._config.failure_rate_threshold is not None:
                    rate = self._window.get_failure_rate()
                    if rate >= self._config.failure_rate_threshold:
                        should_open = True
                        reason = f"Failure rate threshold reached ({rate:.2%})"

                if should_open:
                    self._transition_to(CircuitBreakerState.OPEN, reason=reason)

            elif self._state == CircuitBreakerState.HALF_OPEN:
                # Any failure in half-open reopens the circuit (FAIL-SAFE)
                self._transition_to(
                    CircuitBreakerState.OPEN,
                    reason="Failure in half-open state"
                )

            logger.warning(
                "SteamQualCircuitBreaker failure: name=%s, consecutive=%d, error=%s",
                self._config.name,
                self._consecutive_failures,
                error_msg,
            )

    # =========================================================================
    # CALL EXECUTION
    # =========================================================================

    def allow_request(self) -> bool:
        """
        Check if a request should be allowed.

        Returns:
            True if request can proceed, False if blocked (FAIL-SAFE)
        """
        if not self._config.enabled:
            return True

        with self._state_lock:
            if self._state == CircuitBreakerState.CLOSED:
                return True

            elif self._state == CircuitBreakerState.OPEN:
                # Check if recovery timeout elapsed
                if self._opened_at is not None:
                    elapsed = time.monotonic() - self._opened_at
                    if elapsed >= self._config.recovery_timeout_seconds:
                        self._transition_to(
                            CircuitBreakerState.HALF_OPEN,
                            reason="Recovery timeout elapsed"
                        )
                        self._half_open_calls = 1
                        return True

                self._calls_blocked += 1
                return False

            elif self._state == CircuitBreakerState.HALF_OPEN:
                if self._half_open_calls < self._config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    @asynccontextmanager
    async def protect(self):
        """
        Async context manager for protected calls.

        FAIL-SAFE: If circuit is open, raises CircuitOpenError immediately
        to prevent potentially dangerous operations.

        Usage:
            async with breaker.protect():
                result = await risky_operation()

        Raises:
            CircuitOpenError: If circuit is open
            CircuitHalfOpenError: If half-open and max calls exceeded
        """
        if not self.allow_request():
            if self._state == CircuitBreakerState.OPEN:
                raise CircuitOpenError(
                    self._config.name,
                    self._get_time_until_half_open()
                )
            else:
                raise CircuitHalfOpenError(self._config.name)

        start_time = time.monotonic()
        try:
            yield
            duration_ms = (time.monotonic() - start_time) * 1000
            self.record_success(duration_ms)
        except Exception as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            self.record_failure(e, duration_ms)
            raise

    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        fallback: Optional[Callable[..., Awaitable[T]]] = None,
        **kwargs: Any,
    ) -> T:
        """
        Execute a function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            fallback: Optional fallback function if circuit is open
            **kwargs: Keyword arguments for func

        Returns:
            Result from func or fallback

        Raises:
            CircuitOpenError: If circuit is open and no fallback provided
        """
        try:
            async with self.protect():
                return await func(*args, **kwargs)
        except (CircuitOpenError, CircuitHalfOpenError) as e:
            if fallback is not None:
                logger.info(
                    "SteamQualCircuitBreaker fallback: name=%s, using fallback",
                    self._config.name,
                )
                return await fallback(*args, **kwargs)
            raise

    # =========================================================================
    # METRICS AND MONITORING
    # =========================================================================

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get current circuit breaker metrics."""
        with self._state_lock:
            return CircuitBreakerMetrics(
                name=self._config.name,
                state=self._state,
                failure_count=self._failure_count,
                success_count=self._success_count,
                consecutive_failures=self._consecutive_failures,
                consecutive_successes=self._consecutive_successes,
                last_failure_time=(
                    datetime.fromtimestamp(self._last_failure_time, tz=timezone.utc)
                    if self._last_failure_time
                    else None
                ),
                last_success_time=(
                    datetime.fromtimestamp(self._last_success_time, tz=timezone.utc)
                    if self._last_success_time
                    else None
                ),
                state_changed_at=datetime.fromtimestamp(
                    self._state_changed_at, tz=timezone.utc
                ),
                calls_blocked=self._calls_blocked,
                total_calls=self._total_calls,
                avg_call_duration_ms=self._window.get_avg_duration(),
            )

    def get_audit_records(
        self,
        limit: int = 100,
    ) -> List[CircuitBreakerAuditRecord]:
        """Get recent audit records."""
        return list(reversed(self._audit_records[-limit:]))

    def reset(self) -> None:
        """Reset circuit breaker to initial closed state."""
        with self._state_lock:
            old_state = self._state
            self._state = CircuitBreakerState.CLOSED
            self._state_changed_at = time.monotonic()
            self._failure_count = 0
            self._success_count = 0
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._calls_blocked = 0
            self._total_calls = 0
            self._half_open_calls = 0
            self._last_failure_time = None
            self._last_success_time = None
            self._opened_at = None
            self._window.clear()

            if old_state != CircuitBreakerState.CLOSED:
                record = CircuitBreakerAuditRecord(
                    breaker_name=self._config.name,
                    event=CircuitBreakerEvent.CIRCUIT_CLOSED,
                    previous_state=old_state,
                    new_state=CircuitBreakerState.CLOSED,
                    context={"reason": "Manual reset"},
                )
                self._audit_records.append(record)

        logger.info("SteamQualCircuitBreaker reset: name=%s", self._config.name)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SteamQualCircuitBreaker(name={self._config.name!r}, "
            f"state={self._state.value}, "
            f"failures={self._consecutive_failures})"
        )


# =============================================================================
# CIRCUIT BREAKER REGISTRY
# =============================================================================


class CircuitBreakerRegistry:
    """
    Thread-safe registry for managing multiple circuit breakers.

    Provides centralized management, monitoring, and configuration
    for all circuit breakers in the SteamQual system.
    """

    _instance: Optional["CircuitBreakerRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "CircuitBreakerRegistry":
        """Singleton pattern for global registry."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._breakers: Dict[str, SteamQualCircuitBreaker] = {}
                cls._instance._registry_lock = threading.Lock()
            return cls._instance

    def register(
        self,
        breaker: SteamQualCircuitBreaker,
    ) -> None:
        """Register a circuit breaker."""
        with self._registry_lock:
            self._breakers[breaker.name] = breaker
            logger.info("SteamQualCircuitBreaker registered: %s", breaker.name)

    def get(self, name: str) -> Optional[SteamQualCircuitBreaker]:
        """Get circuit breaker by name."""
        with self._registry_lock:
            return self._breakers.get(name)

    def get_or_create(
        self,
        name: str,
        **kwargs: Any,
    ) -> SteamQualCircuitBreaker:
        """Get existing or create new circuit breaker."""
        with self._registry_lock:
            if name in self._breakers:
                return self._breakers[name]
            breaker = SteamQualCircuitBreaker(name=name, **kwargs)
            self._breakers[name] = breaker
            return breaker

    def get_all_metrics(self) -> Dict[str, CircuitBreakerMetrics]:
        """Get metrics for all circuit breakers."""
        with self._registry_lock:
            return {
                name: breaker.get_metrics()
                for name, breaker in self._breakers.items()
            }

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._registry_lock:
            for breaker in self._breakers.values():
                breaker.reset()

    def clear(self) -> None:
        """Clear all registered circuit breakers."""
        with self._registry_lock:
            self._breakers.clear()


# Global registry instance
_registry = CircuitBreakerRegistry()


def get_circuit_breaker(name: str) -> Optional[SteamQualCircuitBreaker]:
    """Get circuit breaker from global registry."""
    return _registry.get(name)


def get_or_create_circuit_breaker(
    name: str,
    **kwargs: Any,
) -> SteamQualCircuitBreaker:
    """Get or create circuit breaker in global registry."""
    return _registry.get_or_create(name, **kwargs)


# =============================================================================
# DECORATOR
# =============================================================================


def circuit_protected(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout_seconds: float = 30.0,
    fallback: Optional[Callable[..., Awaitable[Any]]] = None,
    **config_kwargs: Any,
) -> Callable[[F], F]:
    """
    Decorator to protect async functions with a circuit breaker.

    Args:
        name: Circuit breaker name
        failure_threshold: Failures before opening circuit
        recovery_timeout_seconds: Recovery timeout
        fallback: Optional fallback function
        **config_kwargs: Additional configuration

    Returns:
        Decorated function

    Example:
        >>> @circuit_protected(
        ...     name="scada_api",
        ...     failure_threshold=5,
        ...     recovery_timeout_seconds=30.0
        ... )
        ... async def read_from_scada(tag_id: str) -> Dict:
        ...     return await scada_client.read(tag_id)
    """
    def decorator(func: F) -> F:
        # Get or create circuit breaker
        breaker = get_or_create_circuit_breaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout_seconds=recovery_timeout_seconds,
            **config_kwargs,
        )

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await breaker.execute(
                func,
                *args,
                fallback=fallback,
                **kwargs,
            )

        # Attach breaker reference for testing/introspection
        wrapper._circuit_breaker = breaker  # type: ignore

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# PRE-CONFIGURED CIRCUIT BREAKERS FOR STEAMQUAL
# =============================================================================


class SteamQualCircuitBreakers:
    """
    Pre-configured circuit breakers for GL-012 STEAMQUAL.

    Provides circuit breakers for all external integrations:
    - OPC-UA connections to SCADA
    - MQTT messaging for real-time quality data
    - Historian for historical data queries
    - ERP for production scheduling
    - SIS monitoring (read-only)
    """

    # Default configurations based on IEC 61508 recommendations
    SCADA_CONFIG = CircuitBreakerConfig(
        name="scada_connector",
        failure_threshold=3,  # Lower threshold for safety-critical
        recovery_timeout_seconds=15.0,  # Faster recovery for real-time
        half_open_max_calls=2,
        success_threshold=2,
        slow_call_duration_threshold_ms=2000.0,
    )

    MQTT_CONFIG = CircuitBreakerConfig(
        name="mqtt_broker",
        failure_threshold=5,
        recovery_timeout_seconds=30.0,
        half_open_max_calls=3,
        success_threshold=2,
    )

    HISTORIAN_CONFIG = CircuitBreakerConfig(
        name="historian_api",
        failure_threshold=5,
        recovery_timeout_seconds=60.0,  # Longer for historian
        half_open_max_calls=3,
        success_threshold=3,
        slow_call_duration_threshold_ms=10000.0,
    )

    SIS_CONFIG = CircuitBreakerConfig(
        name="sis_monitor",
        failure_threshold=3,  # Low threshold for safety system
        recovery_timeout_seconds=10.0,  # Fast recovery for safety
        half_open_max_calls=2,
        success_threshold=2,
        slow_call_duration_threshold_ms=1000.0,
    )

    def __init__(
        self,
        event_callback: Optional[Callable[[CircuitBreakerAuditRecord], None]] = None,
    ) -> None:
        """
        Initialize SteamQual circuit breakers.

        Args:
            event_callback: Optional callback for circuit breaker events
        """
        self._event_callback = event_callback

        # Create circuit breakers
        self.scada = SteamQualCircuitBreaker(
            name="scada_connector",
            config=self.SCADA_CONFIG,
            event_callback=event_callback,
        )
        self.mqtt = SteamQualCircuitBreaker(
            name="mqtt_broker",
            config=self.MQTT_CONFIG,
            event_callback=event_callback,
        )
        self.historian = SteamQualCircuitBreaker(
            name="historian_api",
            config=self.HISTORIAN_CONFIG,
            event_callback=event_callback,
        )
        self.sis = SteamQualCircuitBreaker(
            name="sis_monitor",
            config=self.SIS_CONFIG,
            event_callback=event_callback,
        )

        # Register all in global registry
        _registry.register(self.scada)
        _registry.register(self.mqtt)
        _registry.register(self.historian)
        _registry.register(self.sis)

        logger.info("SteamQualCircuitBreakers initialized")

    def get_all_metrics(self) -> Dict[str, CircuitBreakerMetrics]:
        """Get metrics for all SteamQual circuit breakers."""
        return {
            "scada": self.scada.get_metrics(),
            "mqtt": self.mqtt.get_metrics(),
            "historian": self.historian.get_metrics(),
            "sis": self.sis.get_metrics(),
        }

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health based on circuit breakers."""
        metrics = self.get_all_metrics()

        open_count = sum(
            1 for m in metrics.values()
            if m.state == CircuitBreakerState.OPEN
        )
        half_open_count = sum(
            1 for m in metrics.values()
            if m.state == CircuitBreakerState.HALF_OPEN
        )

        # Determine overall health
        # SIS circuit open is CRITICAL
        if metrics["sis"].state == CircuitBreakerState.OPEN:
            health = "critical"
        elif open_count >= 2:
            health = "critical"
        elif open_count >= 1 or half_open_count >= 2:
            health = "degraded"
        elif half_open_count >= 1:
            health = "warning"
        else:
            health = "healthy"

        return {
            "health": health,
            "open_circuits": open_count,
            "half_open_circuits": half_open_count,
            "total_circuits": len(metrics),
            "circuits": {
                name: m.state.value
                for name, m in metrics.items()
            },
        }

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        self.scada.reset()
        self.mqtt.reset()
        self.historian.reset()
        self.sis.reset()
        logger.info("All SteamQual circuit breakers reset")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "CircuitBreakerState",
    "CircuitBreakerEvent",
    # Exceptions
    "CircuitBreakerError",
    "CircuitOpenError",
    "CircuitHalfOpenError",
    # Models
    "CircuitBreakerConfig",
    "CircuitBreakerMetrics",
    "CircuitBreakerAuditRecord",
    # Core Classes
    "SteamQualCircuitBreaker",
    "CircuitBreakerRegistry",
    "SteamQualCircuitBreakers",
    # Functions
    "get_circuit_breaker",
    "get_or_create_circuit_breaker",
    "circuit_protected",
]
