# -*- coding: utf-8 -*-
"""
GL-015 Insulscan - Service Circuit Breaker

Implements the circuit breaker pattern for ML service failure handling
with graceful degradation to deterministic-only mode.

Features:
1. ML service failure detection and isolation
2. Graceful degradation to physics-based models
3. Automatic recovery with health checks
4. Exponential backoff for retry timing
5. Load shedding under stress

Safety Principles:
- Fail-safe: When ML service fails, use conservative deterministic models
- Graceful degradation: System continues to function with reduced capability
- Automatic recovery: Health checks enable automatic reconnection
- Audit trail: All state transitions are logged with provenance

Standards Reference:
- IEC 61508: Functional Safety - Fault Tolerance Requirements
- IEC 61511: SIS for Process Industries - System Availability

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import random
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
)

from pydantic import BaseModel, Field, field_validator

from .exceptions import ModelUnavailableError

logger = logging.getLogger(__name__)

# Type variables
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# ENUMERATIONS
# =============================================================================


class CircuitState(str, Enum):
    """
    Circuit breaker state enumeration.

    States follow the standard circuit breaker pattern:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Circuit is tripped, requests fail immediately (fail-safe)
    - HALF_OPEN: Recovery testing, limited requests allowed
    - DEGRADED: Operating in degraded mode (deterministic only)
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
    DEGRADED = "degraded"


class CircuitEvent(str, Enum):
    """Events emitted by circuit breaker state transitions."""

    FAILURE_RECORDED = "failure_recorded"
    SUCCESS_RECORDED = "success_recorded"
    CIRCUIT_OPENED = "circuit_opened"
    CIRCUIT_HALF_OPENED = "circuit_half_opened"
    CIRCUIT_CLOSED = "circuit_closed"
    DEGRADED_MODE_ENTERED = "degraded_mode_entered"
    DEGRADED_MODE_EXITED = "degraded_mode_exited"
    HEALTH_CHECK_PASSED = "health_check_passed"
    HEALTH_CHECK_FAILED = "health_check_failed"
    RECOVERY_STARTED = "recovery_started"
    RECOVERY_COMPLETED = "recovery_completed"


class ServiceType(str, Enum):
    """Types of services protected by circuit breaker."""

    CONDITION_PREDICTOR = "condition_predictor"
    HEAT_LOSS_CALCULATOR = "heat_loss_calculator"
    REPAIR_OPTIMIZER = "repair_optimizer"
    ANOMALY_DETECTOR = "anomaly_detector"
    THERMAL_SCANNER = "thermal_scanner"


class DegradationLevel(str, Enum):
    """Level of service degradation."""

    NONE = "none"  # Full ML service available
    PARTIAL = "partial"  # Some ML features unavailable
    FULL = "full"  # Deterministic-only mode


# =============================================================================
# CONFIGURATION
# =============================================================================


class InsulationCircuitBreakerConfig(BaseModel):
    """
    Configuration for insulation service circuit breaker.

    Attributes:
        name: Unique identifier for this circuit breaker
        failure_threshold: Consecutive failures to open circuit
        success_threshold: Consecutive successes to close circuit
        recovery_timeout_seconds: Time before attempting recovery
        max_recovery_timeout_seconds: Maximum recovery timeout (with backoff)
        half_open_max_calls: Max concurrent calls in half-open state
        health_check_interval_seconds: Interval between health checks
        enable_degraded_mode: Whether to enable degraded operation
        degraded_mode_timeout_seconds: Time to try recovery before degradation
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
        le=50,
        description="Consecutive failures to open circuit"
    )
    success_threshold: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Consecutive successes to close circuit"
    )
    recovery_timeout_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Initial recovery timeout"
    )
    max_recovery_timeout_seconds: float = Field(
        default=300.0,
        ge=60.0,
        le=600.0,
        description="Maximum recovery timeout with backoff"
    )
    half_open_max_calls: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max concurrent calls in half-open state"
    )
    health_check_interval_seconds: float = Field(
        default=60.0,
        ge=10.0,
        le=300.0,
        description="Interval between health checks"
    )
    enable_degraded_mode: bool = Field(
        default=True,
        description="Enable degraded (deterministic-only) mode"
    )
    degraded_mode_timeout_seconds: float = Field(
        default=300.0,
        ge=60.0,
        le=3600.0,
        description="Time before entering full degraded mode"
    )
    jitter_factor: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="Jitter factor for recovery timing"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name contains only safe characters."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Name must contain only alphanumeric, underscore, or hyphen")
        return v


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""

    name: str
    state: CircuitState
    failure_count: int = 0
    success_count: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    open_count: int = 0
    calls_blocked: int = 0
    total_calls: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    degradation_level: DegradationLevel = DegradationLevel.NONE
    recovery_attempts: int = 0
    avg_call_duration_ms: float = 0.0

    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failure_count / self.total_calls

    @property
    def availability(self) -> float:
        """Calculate service availability."""
        return 1.0 - self.failure_rate

    def calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash for audit trail."""
        content = (
            f"{self.name}|{self.state.value}|{self.failure_count}|"
            f"{self.success_count}|{self.state_changed_at.isoformat()}"
        )
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class CircuitBreakerEvent:
    """Audit record for circuit breaker events."""

    event_id: str
    timestamp: datetime
    breaker_name: str
    event_type: CircuitEvent
    previous_state: Optional[CircuitState]
    new_state: CircuitState
    degradation_level: DegradationLevel
    failure_count: int
    error_message: Optional[str] = None
    recovery_timeout: Optional[float] = None
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            content = (
                f"{self.event_id}|{self.timestamp.isoformat()}|"
                f"{self.breaker_name}|{self.event_type.value}|{self.new_state.value}"
            )
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


@dataclass
class FallbackResult(Generic[T]):
    """
    Result container indicating whether fallback was used.

    Attributes:
        value: The result value
        used_fallback: Whether fallback was used
        fallback_reason: Why fallback was used
        original_error: Original error if fallback was used
        degradation_level: Level of degradation
    """

    value: T
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
    original_error: Optional[Exception] = None
    degradation_level: DegradationLevel = DegradationLevel.NONE


# =============================================================================
# INSULATION CIRCUIT BREAKER
# =============================================================================


class InsulationCircuitBreaker:
    """
    Circuit breaker for insulation assessment ML service failure handling.

    Implements the circuit breaker pattern to:
    1. Detect ML service failures and prevent cascade failures
    2. Automatically recover when service becomes available
    3. Gracefully degrade to deterministic-only mode when needed
    4. Provide audit trail for all state transitions

    State Diagram:
        CLOSED ----[failure_threshold]----> OPEN
           ^                                   |
           |                                   v
           +---[success_threshold]<--- HALF_OPEN <---[recovery_timeout]
                                           |
           DEGRADED <---[degraded_timeout]-+

    Example:
        >>> breaker = InsulationCircuitBreaker(
        ...     name="condition_predictor",
        ...     failure_threshold=5,
        ...     recovery_timeout_seconds=30.0,
        ... )
        >>>
        >>> async with breaker.protect():
        ...     result = await ml_service.predict(features)

    Author: GL-BackendDeveloper
    Version: 1.0.0
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        recovery_timeout_seconds: float = 30.0,
        config: Optional[InsulationCircuitBreakerConfig] = None,
        fallback_handler: Optional[Callable[..., Any]] = None,
        health_check: Optional[Callable[[], Awaitable[bool]]] = None,
        event_callback: Optional[Callable[[CircuitBreakerEvent], None]] = None,
    ) -> None:
        """
        Initialize insulation circuit breaker.

        Args:
            name: Unique identifier
            failure_threshold: Consecutive failures to open circuit
            success_threshold: Consecutive successes to close circuit
            recovery_timeout_seconds: Time before attempting recovery
            config: Optional full configuration
            fallback_handler: Optional fallback for degraded mode
            health_check: Optional async health check function
            event_callback: Optional callback for events
        """
        if config:
            self._config = config
        else:
            self._config = InsulationCircuitBreakerConfig(
                name=name,
                failure_threshold=failure_threshold,
                success_threshold=success_threshold,
                recovery_timeout_seconds=recovery_timeout_seconds,
            )

        self._state = CircuitState.CLOSED
        self._state_lock = threading.RLock()
        self._state_changed_at = time.monotonic()

        # Counters
        self._failure_count = 0
        self._success_count = 0
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._calls_blocked = 0
        self._total_calls = 0
        self._half_open_calls = 0
        self._open_count = 0
        self._recovery_attempts = 0

        # Timestamps
        self._last_failure_time: Optional[float] = None
        self._last_success_time: Optional[float] = None
        self._opened_at: Optional[float] = None
        self._degraded_mode_entered_at: Optional[float] = None

        # Current recovery timeout (with backoff)
        self._current_recovery_timeout = self._config.recovery_timeout_seconds

        # Degradation tracking
        self._degradation_level = DegradationLevel.NONE

        # Handlers
        self._fallback_handler = fallback_handler
        self._health_check = health_check
        self._event_callback = event_callback

        # Event history
        self._events: List[CircuitBreakerEvent] = []

        # Call duration tracking
        self._call_durations: List[float] = []

        logger.info(
            f"InsulationCircuitBreaker initialized: name={self._config.name}, "
            f"failure_threshold={self._config.failure_threshold}, "
            f"recovery_timeout={self._config.recovery_timeout_seconds}s"
        )

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def name(self) -> str:
        """Get circuit breaker name."""
        return self._config.name

    @property
    def state(self) -> CircuitState:
        """Get current state."""
        with self._state_lock:
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking calls)."""
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.state == CircuitState.HALF_OPEN

    @property
    def is_degraded(self) -> bool:
        """Check if operating in degraded mode."""
        return self.state == CircuitState.DEGRADED or self._degradation_level != DegradationLevel.NONE

    @property
    def degradation_level(self) -> DegradationLevel:
        """Get current degradation level."""
        return self._degradation_level

    @property
    def config(self) -> InsulationCircuitBreakerConfig:
        """Get configuration."""
        return self._config

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def _transition_to(
        self,
        new_state: CircuitState,
        reason: str = "",
        error_message: Optional[str] = None,
    ) -> None:
        """
        Transition to a new state (thread-safe).

        Args:
            new_state: Target state
            reason: Reason for transition
            error_message: Error message if applicable
        """
        with self._state_lock:
            if self._state == new_state:
                return

            old_state = self._state
            self._state = new_state
            self._state_changed_at = time.monotonic()

            # State-specific actions
            if new_state == CircuitState.HALF_OPEN:
                self._half_open_calls = 0
                self._consecutive_successes = 0
                self._recovery_attempts += 1

            elif new_state == CircuitState.CLOSED:
                self._consecutive_failures = 0
                self._current_recovery_timeout = self._config.recovery_timeout_seconds
                self._degradation_level = DegradationLevel.NONE
                self._degraded_mode_entered_at = None

            elif new_state == CircuitState.OPEN:
                self._opened_at = time.monotonic()
                self._open_count += 1
                # Apply exponential backoff
                self._current_recovery_timeout = min(
                    self._current_recovery_timeout * 2,
                    self._config.max_recovery_timeout_seconds,
                )
                # Add jitter
                jitter = random.uniform(
                    -self._config.jitter_factor * self._current_recovery_timeout,
                    self._config.jitter_factor * self._current_recovery_timeout,
                )
                self._current_recovery_timeout = max(
                    self._config.recovery_timeout_seconds,
                    self._current_recovery_timeout + jitter,
                )

            elif new_state == CircuitState.DEGRADED:
                self._degradation_level = DegradationLevel.FULL
                if self._degraded_mode_entered_at is None:
                    self._degraded_mode_entered_at = time.monotonic()

            # Determine event type
            event_type_map = {
                CircuitState.OPEN: CircuitEvent.CIRCUIT_OPENED,
                CircuitState.HALF_OPEN: CircuitEvent.CIRCUIT_HALF_OPENED,
                CircuitState.CLOSED: CircuitEvent.CIRCUIT_CLOSED,
                CircuitState.DEGRADED: CircuitEvent.DEGRADED_MODE_ENTERED,
            }

            # Record event
            event = CircuitBreakerEvent(
                event_id=f"{self._config.name}_{int(time.time() * 1000)}",
                timestamp=datetime.now(timezone.utc),
                breaker_name=self._config.name,
                event_type=event_type_map.get(new_state, CircuitEvent.CIRCUIT_OPENED),
                previous_state=old_state,
                new_state=new_state,
                degradation_level=self._degradation_level,
                failure_count=self._consecutive_failures,
                error_message=error_message,
                recovery_timeout=self._current_recovery_timeout,
            )
            self._events.append(event)

            # Invoke callback
            if self._event_callback:
                try:
                    self._event_callback(event)
                except Exception as e:
                    logger.error(f"Event callback failed: {e}")

            logger.warning(
                f"Circuit breaker state transition: {old_state.value} -> {new_state.value} "
                f"(name={self._config.name}, reason={reason})"
            )

    def _get_time_until_half_open(self) -> float:
        """Get seconds until circuit can enter half-open state."""
        if self._state != CircuitState.OPEN:
            return 0.0
        if self._opened_at is None:
            return self._current_recovery_timeout
        elapsed = time.monotonic() - self._opened_at
        return max(0.0, self._current_recovery_timeout - elapsed)

    def _should_check_degradation(self) -> bool:
        """Check if we should enter full degraded mode."""
        if not self._config.enable_degraded_mode:
            return False
        if self._degraded_mode_entered_at is not None:
            return False
        if self._opened_at is None:
            return False

        elapsed_open = time.monotonic() - self._opened_at
        return elapsed_open >= self._config.degraded_mode_timeout_seconds

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

            # Track duration
            self._call_durations.append(duration_ms)
            if len(self._call_durations) > 100:
                self._call_durations.pop(0)

            # State transitions
            if self._state == CircuitState.HALF_OPEN:
                if self._consecutive_successes >= self._config.success_threshold:
                    self._transition_to(
                        CircuitState.CLOSED,
                        reason=f"Recovery successful ({self._consecutive_successes} successes)"
                    )

            # Exit degraded mode if in half-open and succeeding
            if self._degradation_level != DegradationLevel.NONE:
                self._degradation_level = DegradationLevel.PARTIAL

    def record_failure(
        self,
        error: Optional[Exception] = None,
        duration_ms: float = 0.0,
    ) -> None:
        """
        Record a failed call.

        Args:
            error: The exception that occurred
            duration_ms: Call duration in milliseconds
        """
        with self._state_lock:
            self._failure_count += 1
            self._consecutive_failures += 1
            self._consecutive_successes = 0
            self._last_failure_time = time.monotonic()
            self._total_calls += 1

            error_msg = str(error) if error else None

            # Track duration
            self._call_durations.append(duration_ms)
            if len(self._call_durations) > 100:
                self._call_durations.pop(0)

            # State transitions
            if self._state == CircuitState.CLOSED:
                if self._consecutive_failures >= self._config.failure_threshold:
                    self._transition_to(
                        CircuitState.OPEN,
                        reason=f"Failure threshold reached ({self._consecutive_failures})",
                        error_message=error_msg,
                    )

            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                self._transition_to(
                    CircuitState.OPEN,
                    reason="Failure during recovery testing",
                    error_message=error_msg,
                )

            # Check for degradation
            if self._should_check_degradation():
                self._transition_to(
                    CircuitState.DEGRADED,
                    reason="Extended outage, entering degraded mode",
                )

            logger.warning(
                f"Circuit breaker failure: name={self._config.name}, "
                f"consecutive={self._consecutive_failures}, error={error_msg}"
            )

    # =========================================================================
    # CALL EXECUTION
    # =========================================================================

    def allow_request(self) -> bool:
        """
        Check if a request should be allowed through.

        Returns:
            True if request can proceed, False if blocked
        """
        with self._state_lock:
            if self._state == CircuitState.CLOSED:
                return True

            elif self._state == CircuitState.DEGRADED:
                # In degraded mode, allow calls but expect fallback
                return True

            elif self._state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                if self._opened_at is not None:
                    elapsed = time.monotonic() - self._opened_at
                    if elapsed >= self._current_recovery_timeout:
                        self._transition_to(
                            CircuitState.HALF_OPEN,
                            reason="Recovery timeout elapsed"
                        )
                        self._half_open_calls = 1
                        return True

                self._calls_blocked += 1
                return False

            elif self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self._config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    @asynccontextmanager
    async def protect(self):
        """
        Async context manager for protected calls.

        Usage:
            async with breaker.protect():
                result = await risky_operation()

        Raises:
            ModelUnavailableError: If circuit is open and no fallback
        """
        if not self.allow_request():
            raise ModelUnavailableError(
                message=f"Circuit breaker '{self._config.name}' is {self._state.value}",
                service_name=self._config.name,
                retry_after_seconds=self._get_time_until_half_open(),
                fallback_available=self._fallback_handler is not None,
            )

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
    ) -> FallbackResult[T]:
        """
        Execute a function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments
            fallback: Optional fallback function
            **kwargs: Keyword arguments

        Returns:
            FallbackResult containing result and metadata
        """
        # Use provided fallback or configured one
        fallback_func = fallback or self._fallback_handler

        # Check if we should allow the request
        if not self.allow_request():
            # Try fallback
            if fallback_func is not None:
                logger.info(
                    f"Circuit breaker {self._config.name} using fallback "
                    f"(state={self._state.value})"
                )
                try:
                    result = await fallback_func(*args, **kwargs)
                    return FallbackResult(
                        value=result,
                        used_fallback=True,
                        fallback_reason=f"Circuit {self._state.value}",
                        degradation_level=self._degradation_level,
                    )
                except Exception as e:
                    logger.error(f"Fallback also failed: {e}")
                    raise

            # No fallback available
            raise ModelUnavailableError(
                message=f"Circuit breaker '{self._config.name}' is {self._state.value}",
                service_name=self._config.name,
                retry_after_seconds=self._get_time_until_half_open(),
                fallback_available=False,
            )

        # Try the main function
        start_time = time.monotonic()
        try:
            async with self.protect():
                result = await func(*args, **kwargs)
                return FallbackResult(
                    value=result,
                    used_fallback=False,
                    degradation_level=self._degradation_level,
                )
        except ModelUnavailableError:
            raise
        except Exception as e:
            # Try fallback on failure
            if fallback_func is not None:
                logger.warning(
                    f"Primary call failed, using fallback: {e}"
                )
                try:
                    result = await fallback_func(*args, **kwargs)
                    return FallbackResult(
                        value=result,
                        used_fallback=True,
                        fallback_reason=f"Primary failed: {str(e)[:100]}",
                        original_error=e,
                        degradation_level=self._degradation_level,
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    raise

            raise

    # =========================================================================
    # HEALTH CHECKING
    # =========================================================================

    async def check_health(self) -> bool:
        """
        Perform health check on the protected service.

        Returns:
            True if service is healthy
        """
        if self._health_check is None:
            # No health check configured, assume healthy if we can make calls
            return self.allow_request()

        try:
            is_healthy = await self._health_check()

            event_type = (
                CircuitEvent.HEALTH_CHECK_PASSED if is_healthy
                else CircuitEvent.HEALTH_CHECK_FAILED
            )

            event = CircuitBreakerEvent(
                event_id=f"{self._config.name}_health_{int(time.time() * 1000)}",
                timestamp=datetime.now(timezone.utc),
                breaker_name=self._config.name,
                event_type=event_type,
                previous_state=self._state,
                new_state=self._state,
                degradation_level=self._degradation_level,
                failure_count=self._consecutive_failures,
            )
            self._events.append(event)

            if is_healthy and self._state == CircuitState.OPEN:
                # Healthy service, try half-open
                self._transition_to(
                    CircuitState.HALF_OPEN,
                    reason="Health check passed"
                )

            return is_healthy

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def start_health_check_loop(self, stop_event: asyncio.Event) -> None:
        """
        Start background health check loop.

        Args:
            stop_event: Event to signal stopping the loop
        """
        while not stop_event.is_set():
            try:
                if self._state in (CircuitState.OPEN, CircuitState.DEGRADED):
                    await self.check_health()

                await asyncio.sleep(self._config.health_check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self._config.health_check_interval_seconds)

    # =========================================================================
    # MANUAL CONTROLS
    # =========================================================================

    def force_open(self, reason: str = "Manual force open") -> None:
        """Force circuit to open state."""
        self._transition_to(CircuitState.OPEN, reason=reason)

    def force_close(self, reason: str = "Manual force close") -> None:
        """Force circuit to closed state."""
        self._transition_to(CircuitState.CLOSED, reason=reason)

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self._state_lock:
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._state_changed_at = time.monotonic()
            self._current_recovery_timeout = self._config.recovery_timeout_seconds
            self._failure_count = 0
            self._success_count = 0
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._calls_blocked = 0
            self._total_calls = 0
            self._half_open_calls = 0
            self._recovery_attempts = 0
            self._last_failure_time = None
            self._last_success_time = None
            self._opened_at = None
            self._degraded_mode_entered_at = None
            self._degradation_level = DegradationLevel.NONE
            self._call_durations.clear()

            if old_state != CircuitState.CLOSED:
                event = CircuitBreakerEvent(
                    event_id=f"{self._config.name}_reset_{int(time.time() * 1000)}",
                    timestamp=datetime.now(timezone.utc),
                    breaker_name=self._config.name,
                    event_type=CircuitEvent.CIRCUIT_CLOSED,
                    previous_state=old_state,
                    new_state=CircuitState.CLOSED,
                    degradation_level=DegradationLevel.NONE,
                    failure_count=0,
                )
                self._events.append(event)

            logger.info(f"Circuit breaker reset: name={self._config.name}")

    # =========================================================================
    # METRICS AND MONITORING
    # =========================================================================

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get current circuit breaker metrics."""
        with self._state_lock:
            avg_duration = (
                sum(self._call_durations) / len(self._call_durations)
                if self._call_durations else 0.0
            )

            return CircuitBreakerMetrics(
                name=self._config.name,
                state=self._state,
                failure_count=self._failure_count,
                success_count=self._success_count,
                consecutive_failures=self._consecutive_failures,
                consecutive_successes=self._consecutive_successes,
                open_count=self._open_count,
                calls_blocked=self._calls_blocked,
                total_calls=self._total_calls,
                last_failure_time=(
                    datetime.fromtimestamp(self._last_failure_time, tz=timezone.utc)
                    if self._last_failure_time else None
                ),
                last_success_time=(
                    datetime.fromtimestamp(self._last_success_time, tz=timezone.utc)
                    if self._last_success_time else None
                ),
                state_changed_at=datetime.fromtimestamp(
                    self._state_changed_at, tz=timezone.utc
                ),
                degradation_level=self._degradation_level,
                recovery_attempts=self._recovery_attempts,
                avg_call_duration_ms=avg_duration,
            )

    def get_events(self, limit: int = 100) -> List[CircuitBreakerEvent]:
        """Get recent circuit breaker events."""
        with self._state_lock:
            return list(reversed(self._events[-limit:]))

    def __repr__(self) -> str:
        return (
            f"InsulationCircuitBreaker(name={self._config.name!r}, "
            f"state={self._state.value}, "
            f"failures={self._consecutive_failures}, "
            f"degradation={self._degradation_level.value})"
        )


# =============================================================================
# CIRCUIT BREAKER REGISTRY
# =============================================================================


class InsulationCircuitBreakerRegistry:
    """Thread-safe registry for managing multiple circuit breakers."""

    _instance: Optional["InsulationCircuitBreakerRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "InsulationCircuitBreakerRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._breakers: Dict[str, InsulationCircuitBreaker] = {}
                cls._instance._registry_lock = threading.RLock()
            return cls._instance

    def register(self, breaker: InsulationCircuitBreaker) -> None:
        """Register a circuit breaker."""
        with self._registry_lock:
            self._breakers[breaker.name] = breaker

    def get(self, name: str) -> Optional[InsulationCircuitBreaker]:
        """Get circuit breaker by name."""
        with self._registry_lock:
            return self._breakers.get(name)

    def get_or_create(
        self,
        name: str,
        **kwargs: Any,
    ) -> InsulationCircuitBreaker:
        """Get existing or create new circuit breaker."""
        with self._registry_lock:
            if name in self._breakers:
                return self._breakers[name]
            breaker = InsulationCircuitBreaker(name=name, **kwargs)
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
_registry = InsulationCircuitBreakerRegistry()


def get_insulation_circuit_breaker(name: str) -> Optional[InsulationCircuitBreaker]:
    """Get circuit breaker from global registry."""
    return _registry.get(name)


def get_or_create_insulation_circuit_breaker(name: str, **kwargs: Any) -> InsulationCircuitBreaker:
    """Get or create circuit breaker in global registry."""
    return _registry.get_or_create(name, **kwargs)


# =============================================================================
# DECORATOR
# =============================================================================


def insulation_circuit_protected(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout_seconds: float = 30.0,
    fallback: Optional[Callable[..., Awaitable[Any]]] = None,
    **config_kwargs: Any,
) -> Callable[[F], F]:
    """
    Decorator to protect async functions with a circuit breaker.

    Example:
        >>> @insulation_circuit_protected(
        ...     name="condition_predictor",
        ...     failure_threshold=5,
        ... )
        ... async def predict_condition(features: Dict) -> float:
        ...     return await ml_service.predict(features)
    """
    def decorator(func: F) -> F:
        breaker = get_or_create_insulation_circuit_breaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout_seconds=recovery_timeout_seconds,
            **config_kwargs,
        )

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = await breaker.execute(
                func,
                *args,
                fallback=fallback,
                **kwargs,
            )
            return result.value

        wrapper._circuit_breaker = breaker  # type: ignore
        return wrapper  # type: ignore

    return decorator


# =============================================================================
# PRE-CONFIGURED CIRCUIT BREAKERS
# =============================================================================


class InsulscanCircuitBreakers:
    """
    Pre-configured circuit breakers for GL-015 Insulscan.

    Provides circuit breakers for all ML services:
    - Condition predictor
    - Heat loss calculator
    - Repair optimizer
    - Anomaly detector
    """

    CONDITION_PREDICTOR_CONFIG = InsulationCircuitBreakerConfig(
        name="condition_predictor",
        failure_threshold=5,
        success_threshold=3,
        recovery_timeout_seconds=30.0,
        max_recovery_timeout_seconds=180.0,
        enable_degraded_mode=True,
    )

    HEAT_LOSS_CALCULATOR_CONFIG = InsulationCircuitBreakerConfig(
        name="heat_loss_calculator",
        failure_threshold=3,
        success_threshold=2,
        recovery_timeout_seconds=60.0,
        max_recovery_timeout_seconds=300.0,
        enable_degraded_mode=True,
    )

    REPAIR_OPTIMIZER_CONFIG = InsulationCircuitBreakerConfig(
        name="repair_optimizer",
        failure_threshold=5,
        success_threshold=3,
        recovery_timeout_seconds=30.0,
        enable_degraded_mode=True,
    )

    def __init__(
        self,
        event_callback: Optional[Callable[[CircuitBreakerEvent], None]] = None,
    ) -> None:
        """Initialize pre-configured circuit breakers."""
        self._event_callback = event_callback

        self.condition_predictor = InsulationCircuitBreaker(
            name="condition_predictor",
            config=self.CONDITION_PREDICTOR_CONFIG,
            event_callback=event_callback,
        )
        self.heat_loss_calculator = InsulationCircuitBreaker(
            name="heat_loss_calculator",
            config=self.HEAT_LOSS_CALCULATOR_CONFIG,
            event_callback=event_callback,
        )
        self.repair_optimizer = InsulationCircuitBreaker(
            name="repair_optimizer",
            config=self.REPAIR_OPTIMIZER_CONFIG,
            event_callback=event_callback,
        )

        # Register in global registry
        _registry.register(self.condition_predictor)
        _registry.register(self.heat_loss_calculator)
        _registry.register(self.repair_optimizer)

        logger.info("InsulscanCircuitBreakers initialized")

    def get_all_metrics(self) -> Dict[str, CircuitBreakerMetrics]:
        """Get metrics for all circuit breakers."""
        return {
            "condition_predictor": self.condition_predictor.get_metrics(),
            "heat_loss_calculator": self.heat_loss_calculator.get_metrics(),
            "repair_optimizer": self.repair_optimizer.get_metrics(),
        }

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health based on circuit breaker states."""
        metrics = self.get_all_metrics()

        open_count = sum(
            1 for m in metrics.values()
            if m.state in (CircuitState.OPEN, CircuitState.DEGRADED)
        )
        half_open_count = sum(
            1 for m in metrics.values()
            if m.state == CircuitState.HALF_OPEN
        )
        degraded_count = sum(
            1 for m in metrics.values()
            if m.degradation_level != DegradationLevel.NONE
        )

        if open_count >= 2:
            health = "critical"
        elif open_count >= 1 or degraded_count >= 2:
            health = "degraded"
        elif half_open_count >= 1 or degraded_count >= 1:
            health = "warning"
        else:
            health = "healthy"

        return {
            "health": health,
            "open_circuits": open_count,
            "half_open_circuits": half_open_count,
            "degraded_services": degraded_count,
            "circuits": {
                name: {
                    "state": m.state.value,
                    "degradation": m.degradation_level.value,
                    "availability": m.availability,
                }
                for name, m in metrics.items()
            },
        }

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        self.condition_predictor.reset()
        self.heat_loss_calculator.reset()
        self.repair_optimizer.reset()


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "CircuitState",
    "CircuitEvent",
    "ServiceType",
    "DegradationLevel",
    # Config
    "InsulationCircuitBreakerConfig",
    # Data models
    "CircuitBreakerMetrics",
    "CircuitBreakerEvent",
    "FallbackResult",
    # Main class
    "InsulationCircuitBreaker",
    # Registry
    "InsulationCircuitBreakerRegistry",
    "get_insulation_circuit_breaker",
    "get_or_create_insulation_circuit_breaker",
    # Decorator
    "insulation_circuit_protected",
    # Pre-configured
    "InsulscanCircuitBreakers",
]
