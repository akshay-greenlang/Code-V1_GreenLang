# -*- coding: utf-8 -*-
"""
CircuitBreaker - Resilient external integration protection for GL-005 COMBUSENSE.

This module implements the Circuit Breaker pattern per IEC 61511 safety requirements
for protecting combustion control systems against cascading failures when integrating
with external systems (DCS, PLC, SCADA, OPC-UA, analyzers, etc.).

The circuit breaker provides fail-safe operation for real-time control systems where
communication failures must be handled gracefully without compromising safety.

Design Principles:
    1. Fail-safe: Default to safe state on communication failure
    2. Deterministic: Predictable behavior under all conditions
    3. Observable: Full metrics and state visibility
    4. Auditable: SHA-256 provenance tracking

States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failure threshold exceeded, requests fail fast
    - HALF_OPEN: Testing if service recovered

Reference Standards:
    - IEC 61511: Functional safety for process industries
    - IEC 62443: Industrial cybersecurity
    - ISA-84: Safety Instrumented Systems
    - ISA-18.2: Alarm Management

Example:
    >>> config = CircuitBreakerConfig(
    ...     failure_threshold=3,
    ...     service_name="combustion_analyzer"
    ... )
    >>> breaker = CircuitBreaker(config)
    >>>
    >>> # Protect DCS communication
    >>> result = await breaker.call(dcs.read_process_variables)
    >>> if result.success:
    ...     process_data = result.value
    ... else:
    ...     # Use last known good values or trigger safe mode
    ...     process_data = cached_values

Author: GL-BackendDeveloper
Date: 2025-01-01
Version: 1.0.0
"""

from typing import (
    TypeVar, Generic, Callable, Optional, Any, Dict, List, Awaitable, Union
)
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime, timezone, timedelta
from enum import Enum
from dataclasses import dataclass, field
import asyncio
import hashlib
import json
import logging
import time
import functools
from collections import deque
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# Enums and Constants
# =============================================================================

class CircuitState(str, Enum):
    """
    Circuit breaker states per IEC 61511 safety state machine.

    State transitions:
        CLOSED -> OPEN: When failure_threshold exceeded
        OPEN -> HALF_OPEN: After recovery_timeout elapsed
        HALF_OPEN -> CLOSED: After success_threshold successes
        HALF_OPEN -> OPEN: On any failure
    """
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failure mode - fast fail
    HALF_OPEN = "half_open" # Recovery testing


class FailureType(str, Enum):
    """Classification of failure types for root cause analysis."""
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    RESPONSE_ERROR = "response_error"
    VALIDATION_ERROR = "validation_error"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    PROTOCOL_ERROR = "protocol_error"
    HARDWARE_ERROR = "hardware_error"
    UNKNOWN = "unknown"


class RecoveryStrategy(str, Enum):
    """
    Recovery strategies when circuit is open.

    For combustion control, FALLBACK_CACHE is typically preferred
    to maintain last known good values during communication failures.
    """
    FAIL_FAST = "fail_fast"           # Immediately return error
    FALLBACK_CACHE = "fallback_cache" # Use cached value (recommended)
    FALLBACK_DEFAULT = "fallback_default"  # Use safe default value
    SAFE_MODE = "safe_mode"           # Trigger safe mode operation


class SafetyAction(str, Enum):
    """Safety actions to take on circuit open."""
    NONE = "none"
    LOG_ALARM = "log_alarm"
    TRIGGER_ALARM = "trigger_alarm"
    SAFE_STATE = "safe_state"


# =============================================================================
# Pydantic Models
# =============================================================================

class CircuitBreakerConfig(BaseModel):
    """
    Configuration for CircuitBreaker with IEC 61511 safety parameters.

    Designed for combustion control real-time requirements:
    - Fast failure detection (low thresholds)
    - Quick recovery testing
    - Safe fallback values
    """

    # Failure thresholds (lower for safety-critical systems)
    failure_threshold: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Number of consecutive failures before opening circuit"
    )
    success_threshold: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Consecutive successes needed in half-open to close"
    )

    # Timing parameters (faster for real-time control)
    recovery_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=300.0,
        description="Time before attempting recovery (half-open)"
    )
    call_timeout_seconds: float = Field(
        default=5.0,
        ge=0.1,
        le=60.0,
        description="Default timeout for wrapped calls"
    )

    # Failure rate window
    failure_window_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Rolling window for failure rate calculation"
    )

    # Recovery strategy
    recovery_strategy: RecoveryStrategy = Field(
        default=RecoveryStrategy.FALLBACK_CACHE,
        description="Strategy when circuit is open"
    )

    # Half-open behavior
    half_open_max_calls: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Max concurrent calls in half-open state"
    )

    # IEC 61511 Safety parameters
    iec_61511_sil_level: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Safety Integrity Level (1-4)"
    )
    safety_action_on_open: SafetyAction = Field(
        default=SafetyAction.LOG_ALARM,
        description="Safety action when circuit opens"
    )
    enable_safety_logging: bool = Field(
        default=True,
        description="Enable detailed safety audit logging"
    )

    # Service identification
    service_name: str = Field(
        default="external_service",
        description="Name of protected service"
    )
    service_type: str = Field(
        default="generic",
        description="Type: dcs, plc, scada, analyzer, sensor"
    )

    # Monitoring
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )

    @field_validator('failure_threshold')
    @classmethod
    def validate_threshold_for_sil(cls, v: int, info) -> int:
        """Higher SIL levels require stricter thresholds."""
        sil_level = info.data.get('iec_61511_sil_level', 2)
        if sil_level >= 3 and v > 3:
            logger.warning(
                f"SIL-{sil_level} systems should use failure_threshold <= 3. "
                f"Current: {v}"
            )
        return v

    @model_validator(mode='after')
    def validate_timeout_consistency(self) -> 'CircuitBreakerConfig':
        """Ensure recovery timeout is greater than call timeout."""
        if self.recovery_timeout_seconds <= self.call_timeout_seconds:
            logger.warning(
                f"recovery_timeout ({self.recovery_timeout_seconds}s) should be > "
                f"call_timeout ({self.call_timeout_seconds}s)"
            )
        return self


class FailureRecord(BaseModel):
    """Immutable record of a single failure event."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    failure_type: FailureType = Field(...)
    error_message: str = Field(..., max_length=500)
    duration_ms: float = Field(..., ge=0)
    call_args_hash: str = Field(
        ...,
        description="SHA-256 hash of call arguments for traceability"
    )
    service_name: str = Field(default="unknown")

    class Config:
        frozen = True


class CircuitBreakerState(BaseModel):
    """Current state snapshot of the circuit breaker."""

    state: CircuitState = Field(...)
    failure_count: int = Field(default=0, ge=0)
    success_count: int = Field(default=0, ge=0)
    consecutive_failures: int = Field(default=0, ge=0)
    last_failure_time: Optional[datetime] = Field(default=None)
    last_success_time: Optional[datetime] = Field(default=None)
    last_state_change: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    total_calls: int = Field(default=0, ge=0)
    total_failures: int = Field(default=0, ge=0)
    total_successes: int = Field(default=0, ge=0)
    total_rejections: int = Field(default=0, ge=0)
    current_failure_rate: float = Field(default=0.0, ge=0, le=1)

    # IEC 61511 safety tracking
    safety_trips: int = Field(
        default=0,
        ge=0,
        description="Number of times circuit opened for safety"
    )

    # Audit trail
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail"
    )


class CircuitBreakerMetrics(BaseModel):
    """Metrics for monitoring and alerting (Prometheus compatible)."""

    service_name: str = Field(...)
    service_type: str = Field(default="generic")
    state: CircuitState = Field(...)

    # Counters
    total_calls: int = Field(default=0)
    total_successes: int = Field(default=0)
    total_failures: int = Field(default=0)
    total_rejections: int = Field(default=0)
    total_timeouts: int = Field(default=0)
    total_cache_hits: int = Field(default=0)

    # Rates
    success_rate: float = Field(default=0.0, ge=0, le=1)
    failure_rate: float = Field(default=0.0, ge=0, le=1)

    # Latencies (milliseconds)
    avg_latency_ms: float = Field(default=0.0, ge=0)
    p50_latency_ms: float = Field(default=0.0, ge=0)
    p95_latency_ms: float = Field(default=0.0, ge=0)
    p99_latency_ms: float = Field(default=0.0, ge=0)
    max_latency_ms: float = Field(default=0.0, ge=0)

    # State transitions
    state_transitions: int = Field(default=0)
    time_in_open_seconds: float = Field(default=0.0, ge=0)

    # Safety metrics
    safety_trips: int = Field(default=0)
    last_safety_trip: Optional[datetime] = Field(default=None)
    sil_level: int = Field(default=2)

    capture_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class CallResult(BaseModel, Generic[T]):
    """Result of a circuit breaker protected call."""

    success: bool = Field(...)
    value: Optional[Any] = Field(default=None)
    error: Optional[str] = Field(default=None)
    error_type: Optional[FailureType] = Field(default=None)
    duration_ms: float = Field(..., ge=0)
    from_cache: bool = Field(default=False)
    from_fallback: bool = Field(default=False)
    circuit_state: CircuitState = Field(...)
    service_name: str = Field(default="unknown")
    provenance_hash: str = Field(...)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    class Config:
        arbitrary_types_allowed = True


# =============================================================================
# Circuit Breaker Implementation
# =============================================================================

class CircuitBreaker:
    """
    Circuit Breaker for GL-005 Combustion Control System integration protection.

    Implements the Circuit Breaker pattern with IEC 61511 safety considerations
    for real-time combustion control applications.

    Key features for combustion control:
    - Fast failure detection (3 failures = open)
    - Quick recovery testing (10 second timeout)
    - Cache-based fallback for continuous operation
    - Safety alarm integration
    - Full provenance tracking

    Thread-safe implementation using asyncio locks.

    Example:
        >>> # Configure for combustion analyzer
        >>> config = CircuitBreakerConfig(
        ...     service_name="combustion_analyzer",
        ...     service_type="analyzer",
        ...     failure_threshold=3,
        ...     recovery_timeout_seconds=10.0,
        ...     recovery_strategy=RecoveryStrategy.FALLBACK_CACHE,
        ...     safety_action_on_open=SafetyAction.TRIGGER_ALARM
        ... )
        >>>
        >>> breaker = CircuitBreaker(config)
        >>>
        >>> # Protect analyzer communication
        >>> result = await breaker.call(analyzer.read_o2_percent)
        >>> if result.success:
        ...     o2_value = result.value
        ... elif result.from_cache:
        ...     o2_value = result.value  # Last known good value
        ...     log_alarm("Using cached O2 value")
        ... else:
        ...     trigger_safe_mode()
    """

    def __init__(
        self,
        config: Optional[CircuitBreakerConfig] = None,
        cache: Optional[Dict[str, Any]] = None,
        fallback_provider: Optional[Callable[..., Any]] = None,
        alarm_callback: Optional[Callable[[str, str], None]] = None
    ):
        """
        Initialize CircuitBreaker.

        Args:
            config: Circuit breaker configuration
            cache: Optional cache for fallback values
            fallback_provider: Optional function to provide fallback values
            alarm_callback: Optional callback for alarm generation
        """
        self.config = config or CircuitBreakerConfig()
        self._cache: Dict[str, Any] = cache or {}
        self._fallback_provider = fallback_provider
        self._alarm_callback = alarm_callback

        # State management
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._consecutive_failures = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._last_success_time: Optional[datetime] = None
        self._last_state_change = datetime.now(timezone.utc)

        # Failure tracking
        self._failure_records: deque = deque(maxlen=500)
        self._latencies: deque = deque(maxlen=1000)

        # Counters
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._total_rejections = 0
        self._total_timeouts = 0
        self._total_cache_hits = 0
        self._state_transitions = 0
        self._safety_trips = 0
        self._time_in_open_seconds = 0.0
        self._open_start_time: Optional[datetime] = None

        # Half-open state management
        self._half_open_calls = 0
        self._half_open_lock = asyncio.Lock()

        # Thread safety
        self._state_lock = asyncio.Lock()

        logger.info(
            f"CircuitBreaker initialized: service={self.config.service_name}, "
            f"type={self.config.service_type}, SIL-{self.config.iec_61511_sil_level}, "
            f"threshold={self.config.failure_threshold}, "
            f"recovery={self.config.recovery_timeout_seconds}s"
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
        """Check if circuit is open (failing fast)."""
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is in recovery testing mode."""
        return self._state == CircuitState.HALF_OPEN

    @property
    def service_name(self) -> str:
        """Get protected service name."""
        return self.config.service_name

    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        timeout: Optional[float] = None,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> CallResult:
        """
        Execute a protected call through the circuit breaker.

        Args:
            func: Async function to call
            *args: Positional arguments for func
            timeout: Optional timeout override
            cache_key: Optional cache key for fallback
            **kwargs: Keyword arguments for func

        Returns:
            CallResult with success/failure and value/error
        """
        start_time = time.perf_counter()
        call_timeout = timeout or self.config.call_timeout_seconds

        # Generate provenance hash
        provenance_hash = self._generate_provenance_hash(func, args, kwargs)

        async with self._state_lock:
            self._total_calls += 1

            # Check state transitions
            if self._state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    await self._transition_to_half_open()
                else:
                    self._total_rejections += 1
                    duration_ms = (time.perf_counter() - start_time) * 1000

                    return await self._handle_open_circuit(
                        cache_key, args, kwargs, duration_ms, provenance_hash
                    )

            # Check half-open call limit
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self._total_rejections += 1
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    return CallResult(
                        success=False,
                        error="Circuit half-open: testing in progress",
                        error_type=FailureType.RATE_LIMIT,
                        duration_ms=duration_ms,
                        circuit_state=self._state,
                        service_name=self.config.service_name,
                        provenance_hash=provenance_hash
                    )
                self._half_open_calls += 1

        # Execute the call
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=call_timeout
            )

            duration_ms = (time.perf_counter() - start_time) * 1000
            self._latencies.append(duration_ms)

            # Record success
            await self._record_success(duration_ms)

            # Cache successful result
            if cache_key:
                self._cache[cache_key] = {
                    "value": result,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

            return CallResult(
                success=True,
                value=result,
                duration_ms=duration_ms,
                circuit_state=self._state,
                service_name=self.config.service_name,
                provenance_hash=provenance_hash
            )

        except asyncio.TimeoutError:
            return await self._handle_timeout(
                start_time, call_timeout, args, kwargs, provenance_hash
            )

        except ConnectionError as e:
            return await self._handle_connection_error(
                start_time, e, args, kwargs, provenance_hash
            )

        except Exception as e:
            return await self._handle_generic_error(
                start_time, e, args, kwargs, provenance_hash
            )

        finally:
            if self._state == CircuitState.HALF_OPEN:
                async with self._state_lock:
                    self._half_open_calls = max(0, self._half_open_calls - 1)

    def protect(
        self,
        cache_key: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Callable:
        """
        Decorator to protect an async function with circuit breaker.

        Args:
            cache_key: Optional cache key for fallback
            timeout: Optional timeout override

        Returns:
            Decorated function

        Example:
            >>> @breaker.protect(cache_key="o2_reading")
            ... async def read_o2() -> float:
            ...     return await analyzer.read_o2_percent()
        """
        def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[CallResult]]:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs) -> CallResult:
                return await self.call(
                    func,
                    *args,
                    timeout=timeout,
                    cache_key=cache_key,
                    **kwargs
                )
            return wrapper
        return decorator

    async def _handle_timeout(
        self,
        start_time: float,
        timeout: float,
        args: tuple,
        kwargs: dict,
        provenance_hash: str
    ) -> CallResult:
        """Handle timeout failure."""
        duration_ms = (time.perf_counter() - start_time) * 1000
        self._total_timeouts += 1

        await self._record_failure(
            FailureType.TIMEOUT,
            f"Call timed out after {timeout}s",
            duration_ms,
            args,
            kwargs
        )

        return CallResult(
            success=False,
            error=f"Timeout after {timeout}s",
            error_type=FailureType.TIMEOUT,
            duration_ms=duration_ms,
            circuit_state=self._state,
            service_name=self.config.service_name,
            provenance_hash=provenance_hash
        )

    async def _handle_connection_error(
        self,
        start_time: float,
        error: Exception,
        args: tuple,
        kwargs: dict,
        provenance_hash: str
    ) -> CallResult:
        """Handle connection error."""
        duration_ms = (time.perf_counter() - start_time) * 1000

        await self._record_failure(
            FailureType.CONNECTION_ERROR,
            str(error),
            duration_ms,
            args,
            kwargs
        )

        return CallResult(
            success=False,
            error=str(error),
            error_type=FailureType.CONNECTION_ERROR,
            duration_ms=duration_ms,
            circuit_state=self._state,
            service_name=self.config.service_name,
            provenance_hash=provenance_hash
        )

    async def _handle_generic_error(
        self,
        start_time: float,
        error: Exception,
        args: tuple,
        kwargs: dict,
        provenance_hash: str
    ) -> CallResult:
        """Handle generic error."""
        duration_ms = (time.perf_counter() - start_time) * 1000
        failure_type = self._classify_exception(error)

        await self._record_failure(
            failure_type,
            str(error),
            duration_ms,
            args,
            kwargs
        )

        return CallResult(
            success=False,
            error=str(error),
            error_type=failure_type,
            duration_ms=duration_ms,
            circuit_state=self._state,
            service_name=self.config.service_name,
            provenance_hash=provenance_hash
        )

    async def _record_success(self, duration_ms: float) -> None:
        """Record a successful call."""
        async with self._state_lock:
            self._total_successes += 1
            self._consecutive_failures = 0
            self._last_success_time = datetime.now(timezone.utc)

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    await self._transition_to_closed()

    async def _record_failure(
        self,
        failure_type: FailureType,
        error_message: str,
        duration_ms: float,
        args: tuple,
        kwargs: dict
    ) -> None:
        """Record a failed call."""
        async with self._state_lock:
            self._total_failures += 1
            self._failure_count += 1
            self._consecutive_failures += 1
            self._last_failure_time = datetime.now(timezone.utc)

            record = FailureRecord(
                failure_type=failure_type,
                error_message=error_message[:500],
                duration_ms=duration_ms,
                call_args_hash=self._hash_args(args, kwargs),
                service_name=self.config.service_name
            )
            self._failure_records.append(record)

            if self.config.enable_safety_logging:
                logger.warning(
                    f"[SAFETY] CircuitBreaker failure: service={self.config.service_name}, "
                    f"type={failure_type.value}, "
                    f"consecutive={self._consecutive_failures}/{self.config.failure_threshold}"
                )

            if self._state == CircuitState.CLOSED:
                if self._consecutive_failures >= self.config.failure_threshold:
                    await self._transition_to_open()

            elif self._state == CircuitState.HALF_OPEN:
                await self._transition_to_open()

    async def _transition_to_open(self) -> None:
        """Transition to OPEN state with safety actions."""
        previous_state = self._state
        self._state = CircuitState.OPEN
        self._last_state_change = datetime.now(timezone.utc)
        self._state_transitions += 1
        self._safety_trips += 1
        self._open_start_time = datetime.now(timezone.utc)
        self._success_count = 0

        # Execute safety action
        if self.config.safety_action_on_open == SafetyAction.TRIGGER_ALARM:
            if self._alarm_callback:
                self._alarm_callback(
                    self.config.service_name,
                    f"Circuit breaker opened after {self._consecutive_failures} failures"
                )

        logger.warning(
            f"[SAFETY] CircuitBreaker OPENED: service={self.config.service_name}, "
            f"failures={self._consecutive_failures}, previous_state={previous_state.value}, "
            f"recovery_timeout={self.config.recovery_timeout_seconds}s, "
            f"safety_action={self.config.safety_action_on_open.value}"
        )

    async def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        if self._open_start_time:
            self._time_in_open_seconds += (
                datetime.now(timezone.utc) - self._open_start_time
            ).total_seconds()

        self._state = CircuitState.HALF_OPEN
        self._last_state_change = datetime.now(timezone.utc)
        self._state_transitions += 1
        self._half_open_calls = 0
        self._success_count = 0
        self._consecutive_failures = 0

        logger.info(
            f"CircuitBreaker HALF_OPEN: service={self.config.service_name}, "
            f"testing recovery with max_calls={self.config.half_open_max_calls}"
        )

    async def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._last_state_change = datetime.now(timezone.utc)
        self._state_transitions += 1
        self._failure_count = 0
        self._consecutive_failures = 0
        self._success_count = 0
        self._open_start_time = None

        logger.info(
            f"CircuitBreaker CLOSED: service={self.config.service_name}, "
            f"service recovered, normal operation resumed"
        )

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._last_failure_time is None:
            return True

        elapsed = (
            datetime.now(timezone.utc) - self._last_failure_time
        ).total_seconds()

        return elapsed >= self.config.recovery_timeout_seconds

    async def _handle_open_circuit(
        self,
        cache_key: Optional[str],
        args: tuple,
        kwargs: dict,
        duration_ms: float,
        provenance_hash: str
    ) -> CallResult:
        """Handle call when circuit is open based on recovery strategy."""

        # Try cache fallback
        if self.config.recovery_strategy in (
            RecoveryStrategy.FALLBACK_CACHE,
            RecoveryStrategy.SAFE_MODE
        ):
            if cache_key and cache_key in self._cache:
                cached = self._cache[cache_key]
                self._total_cache_hits += 1
                return CallResult(
                    success=True,
                    value=cached.get("value") if isinstance(cached, dict) else cached,
                    duration_ms=duration_ms,
                    from_cache=True,
                    circuit_state=self._state,
                    service_name=self.config.service_name,
                    provenance_hash=provenance_hash
                )

        # Try fallback provider
        if self.config.recovery_strategy == RecoveryStrategy.FALLBACK_DEFAULT:
            if self._fallback_provider:
                try:
                    fallback_value = self._fallback_provider(*args, **kwargs)
                    return CallResult(
                        success=True,
                        value=fallback_value,
                        duration_ms=duration_ms,
                        from_fallback=True,
                        circuit_state=self._state,
                        service_name=self.config.service_name,
                        provenance_hash=provenance_hash
                    )
                except Exception as e:
                    logger.warning(f"Fallback provider failed: {e}")

        # Default: fail fast
        return CallResult(
            success=False,
            error=f"Circuit open for {self.config.service_name}",
            error_type=FailureType.CONNECTION_ERROR,
            duration_ms=duration_ms,
            circuit_state=self._state,
            service_name=self.config.service_name,
            provenance_hash=provenance_hash
        )

    def _classify_exception(self, exc: Exception) -> FailureType:
        """Classify exception into failure type."""
        exc_name = type(exc).__name__.lower()
        exc_msg = str(exc).lower()

        if 'timeout' in exc_name or 'timeout' in exc_msg:
            return FailureType.TIMEOUT
        elif 'connection' in exc_name or 'network' in exc_name:
            return FailureType.CONNECTION_ERROR
        elif 'auth' in exc_name or 'permission' in exc_name:
            return FailureType.AUTHENTICATION
        elif 'ratelimit' in exc_name or 'throttle' in exc_name:
            return FailureType.RATE_LIMIT
        elif 'protocol' in exc_name or 'opc' in exc_name:
            return FailureType.PROTOCOL_ERROR
        elif 'hardware' in exc_name or 'device' in exc_name:
            return FailureType.HARDWARE_ERROR
        elif 'validation' in exc_name or 'invalid' in exc_name:
            return FailureType.VALIDATION_ERROR
        else:
            return FailureType.UNKNOWN

    def _generate_provenance_hash(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict
    ) -> str:
        """Generate SHA-256 provenance hash for call."""
        call_data = {
            "func": func.__name__ if hasattr(func, '__name__') else str(func),
            "args_hash": self._hash_args(args, kwargs),
            "service": self.config.service_name,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        return hashlib.sha256(
            json.dumps(call_data, sort_keys=True).encode()
        ).hexdigest()

    def _hash_args(self, args: tuple, kwargs: dict) -> str:
        """Create SHA-256 hash of call arguments."""
        try:
            args_data = {
                "args": [str(a)[:100] for a in args],
                "kwargs": {k: str(v)[:100] for k, v in kwargs.items()}
            }
            return hashlib.sha256(
                json.dumps(args_data, sort_keys=True).encode()
            ).hexdigest()[:16]
        except Exception:
            return "unhashable"

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state snapshot."""
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=self.config.failure_window_seconds)

        recent_failures = sum(
            1 for r in self._failure_records
            if r.timestamp >= window_start
        )

        window_calls = max(1, self._total_calls - self._total_rejections)
        failure_rate = min(1.0, recent_failures / window_calls)

        state_data = {
            "state": self._state.value,
            "service": self.config.service_name,
            "timestamp": now.isoformat()
        }
        provenance_hash = hashlib.sha256(
            json.dumps(state_data, sort_keys=True).encode()
        ).hexdigest()

        return CircuitBreakerState(
            state=self._state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            consecutive_failures=self._consecutive_failures,
            last_failure_time=self._last_failure_time,
            last_success_time=self._last_success_time,
            last_state_change=self._last_state_change,
            total_calls=self._total_calls,
            total_failures=self._total_failures,
            total_successes=self._total_successes,
            total_rejections=self._total_rejections,
            current_failure_rate=failure_rate,
            safety_trips=self._safety_trips,
            provenance_hash=provenance_hash
        )

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics for Prometheus."""
        total = max(1, self._total_calls - self._total_rejections)
        success_rate = self._total_successes / total
        failure_rate = self._total_failures / total

        latencies = list(self._latencies)
        if latencies:
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            avg_lat = sum(latencies) / n
            p50_lat = sorted_lat[int(n * 0.5)]
            p95_lat = sorted_lat[min(int(n * 0.95), n - 1)]
            p99_lat = sorted_lat[min(int(n * 0.99), n - 1)]
            max_lat = sorted_lat[-1]
        else:
            avg_lat = p50_lat = p95_lat = p99_lat = max_lat = 0.0

        return CircuitBreakerMetrics(
            service_name=self.config.service_name,
            service_type=self.config.service_type,
            state=self._state,
            total_calls=self._total_calls,
            total_successes=self._total_successes,
            total_failures=self._total_failures,
            total_rejections=self._total_rejections,
            total_timeouts=self._total_timeouts,
            total_cache_hits=self._total_cache_hits,
            success_rate=success_rate,
            failure_rate=failure_rate,
            avg_latency_ms=avg_lat,
            p50_latency_ms=p50_lat,
            p95_latency_ms=p95_lat,
            p99_latency_ms=p99_lat,
            max_latency_ms=max_lat,
            state_transitions=self._state_transitions,
            time_in_open_seconds=self._time_in_open_seconds,
            safety_trips=self._safety_trips,
            last_safety_trip=self._last_failure_time if self._safety_trips > 0 else None,
            sil_level=self.config.iec_61511_sil_level
        )

    def update_cache(self, key: str, value: Any) -> None:
        """Update cache with a value (for pre-populating fallbacks)."""
        self._cache[key] = {
            "value": value,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        async with self._state_lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._consecutive_failures = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._last_state_change = datetime.now(timezone.utc)

            logger.info(f"CircuitBreaker reset: service={self.config.service_name}")

    async def force_open(self, reason: str) -> None:
        """Manually force circuit to open state."""
        async with self._state_lock:
            self._state = CircuitState.OPEN
            self._last_state_change = datetime.now(timezone.utc)
            self._safety_trips += 1
            self._open_start_time = datetime.now(timezone.utc)

            logger.warning(
                f"[SAFETY] CircuitBreaker force-opened: "
                f"service={self.config.service_name}, reason={reason}"
            )


# =============================================================================
# Circuit Breaker Registry
# =============================================================================

class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Provides centralized management for all circuit breakers in the
    combustion control system.
    """

    def __init__(self):
        """Initialize registry."""
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Register a new circuit breaker."""
        async with self._lock:
            if name in self._breakers:
                return self._breakers[name]

            if config is None:
                config = CircuitBreakerConfig(service_name=name)

            breaker = CircuitBreaker(config)
            self._breakers[name] = breaker

            logger.info(f"Registered circuit breaker: {name}")
            return breaker

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self._breakers.get(name)

    def list_all(self) -> List[str]:
        """List all registered circuit breaker names."""
        return list(self._breakers.keys())

    def get_all_metrics(self) -> Dict[str, CircuitBreakerMetrics]:
        """Get metrics for all registered circuit breakers."""
        return {
            name: breaker.get_metrics()
            for name, breaker in self._breakers.items()
        }

    def get_all_states(self) -> Dict[str, CircuitBreakerState]:
        """Get states for all registered circuit breakers."""
        return {
            name: breaker.get_state()
            for name, breaker in self._breakers.items()
        }

    def get_open_circuits(self) -> List[str]:
        """Get list of currently open circuit breakers."""
        return [
            name for name, breaker in self._breakers.items()
            if breaker.is_open
        ]

    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            await breaker.reset()
        logger.info("All circuit breakers reset")


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()
