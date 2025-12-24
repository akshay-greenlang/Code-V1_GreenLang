# -*- coding: utf-8 -*-
"""
CircuitBreaker - Resilient external integration protection for GL-004 BURNMASTER.

This module implements the Circuit Breaker pattern per IEC 61511 safety requirements
for protecting against cascading failures when integrating with external systems
(OPC-UA, SCADA, DCS, ERP systems, etc.).

The circuit breaker prevents repeated calls to failing services, allowing systems
to recover while maintaining deterministic operation.

States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failure threshold exceeded, requests fail fast
    - HALF_OPEN: Testing if service recovered

Reference Standards:
    - IEC 61511: Functional safety for process industries
    - IEC 62443: Industrial cybersecurity
    - ISA-84: Safety Instrumented Systems

Example:
    >>> breaker = CircuitBreaker(config=CircuitBreakerConfig(failure_threshold=5))
    >>> result = await breaker.call(external_service.get_data, timeout=5.0)
    >>> if breaker.state == CircuitState.OPEN:
    ...     # Use cached/fallback values
    ...     pass

Author: GL-BackendDeveloper
Date: 2025-01-01
Version: 1.0.0
"""

from typing import (
    TypeVar, Generic, Callable, Optional, Any, Dict, List, Awaitable, Union
)
from pydantic import BaseModel, Field, field_validator
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
    """Circuit breaker states per IEC 61511 safety state machine."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failure mode - fast fail
    HALF_OPEN = "half_open" # Recovery testing


class FailureType(str, Enum):
    """Classification of failure types for analysis."""
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    RESPONSE_ERROR = "response_error"
    VALIDATION_ERROR = "validation_error"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    UNKNOWN = "unknown"


class RecoveryStrategy(str, Enum):
    """Recovery strategies when circuit is open."""
    FAIL_FAST = "fail_fast"           # Immediately return error
    FALLBACK_CACHE = "fallback_cache" # Use cached value
    FALLBACK_DEFAULT = "fallback_default"  # Use default value
    QUEUE_RETRY = "queue_retry"       # Queue for later retry


# =============================================================================
# Pydantic Models
# =============================================================================

class CircuitBreakerConfig(BaseModel):
    """Configuration for CircuitBreaker with IEC 61511 safety parameters."""

    # Failure thresholds
    failure_threshold: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of failures before opening circuit"
    )
    success_threshold: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Successes needed in half-open to close"
    )

    # Timing parameters (seconds)
    recovery_timeout_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=600.0,
        description="Time before attempting recovery (half-open)"
    )
    call_timeout_seconds: float = Field(
        default=10.0,
        ge=0.5,
        le=120.0,
        description="Default timeout for wrapped calls"
    )

    # Window for failure rate calculation
    failure_window_seconds: float = Field(
        default=60.0,
        ge=10.0,
        le=600.0,
        description="Rolling window for failure rate"
    )

    # Recovery strategy
    recovery_strategy: RecoveryStrategy = Field(
        default=RecoveryStrategy.FAIL_FAST,
        description="Strategy when circuit is open"
    )

    # Half-open behavior
    half_open_max_calls: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max concurrent calls in half-open state"
    )

    # Safety parameters per IEC 61511
    iec_61511_sil_level: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Safety Integrity Level (1-4)"
    )
    enable_safety_logging: bool = Field(
        default=True,
        description="Enable detailed safety audit logging"
    )

    # Monitoring
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )
    service_name: str = Field(
        default="external_service",
        description="Name of protected service"
    )

    @field_validator('failure_threshold')
    @classmethod
    def validate_threshold_for_sil(cls, v: int, info) -> int:
        """Validate threshold based on SIL level."""
        # Higher SIL levels require stricter thresholds
        if info.data.get('iec_61511_sil_level', 2) >= 3 and v > 3:
            logger.warning(
                f"SIL-3/4 systems should use lower failure threshold. "
                f"Current: {v}, Recommended: <=3"
            )
        return v


class FailureRecord(BaseModel):
    """Record of a single failure event."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    failure_type: FailureType = Field(...)
    error_message: str = Field(...)
    duration_ms: float = Field(..., ge=0)
    call_args_hash: str = Field(
        ...,
        description="SHA-256 hash of call arguments for traceability"
    )

    class Config:
        frozen = True


class CircuitBreakerState(BaseModel):
    """Current state of the circuit breaker."""

    state: CircuitState = Field(...)
    failure_count: int = Field(default=0, ge=0)
    success_count: int = Field(default=0, ge=0)
    last_failure_time: Optional[datetime] = Field(default=None)
    last_success_time: Optional[datetime] = Field(default=None)
    last_state_change: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    total_calls: int = Field(default=0, ge=0)
    total_failures: int = Field(default=0, ge=0)
    total_successes: int = Field(default=0, ge=0)
    current_failure_rate: float = Field(default=0.0, ge=0, le=1)

    # IEC 61511 safety tracking
    safety_trips: int = Field(
        default=0,
        ge=0,
        description="Number of times circuit opened for safety"
    )

    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail"
    )


class CircuitBreakerMetrics(BaseModel):
    """Metrics for monitoring and alerting."""

    service_name: str = Field(...)
    state: CircuitState = Field(...)

    # Counters
    total_calls: int = Field(default=0)
    total_successes: int = Field(default=0)
    total_failures: int = Field(default=0)
    total_rejections: int = Field(
        default=0,
        description="Calls rejected when open"
    )
    total_timeouts: int = Field(default=0)

    # Rates
    success_rate: float = Field(default=0.0, ge=0, le=1)
    failure_rate: float = Field(default=0.0, ge=0, le=1)

    # Latencies
    avg_latency_ms: float = Field(default=0.0, ge=0)
    p50_latency_ms: float = Field(default=0.0, ge=0)
    p95_latency_ms: float = Field(default=0.0, ge=0)
    p99_latency_ms: float = Field(default=0.0, ge=0)

    # State transitions
    state_transitions: int = Field(default=0)
    time_in_open_seconds: float = Field(default=0.0, ge=0)

    # Safety metrics
    safety_trips: int = Field(default=0)
    last_safety_trip: Optional[datetime] = Field(default=None)

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
    provenance_hash: str = Field(...)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# =============================================================================
# Circuit Breaker Implementation
# =============================================================================

class CircuitBreaker:
    """
    Circuit Breaker implementation for external system integration protection.

    Implements the Circuit Breaker pattern with IEC 61511 safety considerations:
    - State machine with CLOSED, OPEN, HALF_OPEN states
    - Failure rate tracking with rolling window
    - Configurable recovery strategies
    - SHA-256 provenance tracking for audit trails
    - Safety logging for regulatory compliance

    Thread-safe implementation using asyncio locks.

    Attributes:
        config: Circuit breaker configuration
        state: Current circuit state

    Example:
        >>> config = CircuitBreakerConfig(
        ...     failure_threshold=5,
        ...     recovery_timeout_seconds=30.0,
        ...     service_name="opc_ua_server"
        ... )
        >>> breaker = CircuitBreaker(config)
        >>>
        >>> # Protect an async call
        >>> result = await breaker.call(client.read_tags, tag_ids)
        >>>
        >>> # Or use as decorator
        >>> @breaker.protect
        ... async def fetch_sensor_data(sensor_id: str):
        ...     return await external_api.get(sensor_id)
    """

    def __init__(
        self,
        config: Optional[CircuitBreakerConfig] = None,
        cache: Optional[Dict[str, Any]] = None,
        fallback_provider: Optional[Callable[..., Any]] = None
    ):
        """
        Initialize CircuitBreaker.

        Args:
            config: Circuit breaker configuration
            cache: Optional cache for fallback values
            fallback_provider: Optional function to provide fallback values
        """
        self.config = config or CircuitBreakerConfig()
        self._cache: Dict[str, Any] = cache or {}
        self._fallback_provider = fallback_provider

        # State management
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._last_success_time: Optional[datetime] = None
        self._last_state_change = datetime.now(timezone.utc)

        # Failure tracking with rolling window
        self._failure_records: deque = deque(maxlen=1000)
        self._latencies: deque = deque(maxlen=1000)

        # Counters
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._total_rejections = 0
        self._total_timeouts = 0
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
            f"CircuitBreaker initialized for {self.config.service_name} "
            f"with SIL-{self.config.iec_61511_sil_level}, "
            f"threshold={self.config.failure_threshold}, "
            f"recovery_timeout={self.config.recovery_timeout_seconds}s"
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

        # Generate provenance hash for this call
        call_data = {
            "func": func.__name__ if hasattr(func, '__name__') else str(func),
            "args_hash": self._hash_args(args, kwargs),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        provenance_hash = hashlib.sha256(
            json.dumps(call_data, sort_keys=True).encode()
        ).hexdigest()

        async with self._state_lock:
            self._total_calls += 1

            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    await self._transition_to_half_open()
                else:
                    # Circuit is open - apply recovery strategy
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
                        error="Circuit half-open: max concurrent calls reached",
                        error_type=FailureType.RATE_LIMIT,
                        duration_ms=duration_ms,
                        circuit_state=self._state,
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
                self._cache[cache_key] = result

            return CallResult(
                success=True,
                value=result,
                duration_ms=duration_ms,
                circuit_state=self._state,
                provenance_hash=provenance_hash
            )

        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._total_timeouts += 1

            await self._record_failure(
                FailureType.TIMEOUT,
                f"Call timed out after {call_timeout}s",
                duration_ms,
                args,
                kwargs
            )

            return CallResult(
                success=False,
                error=f"Timeout after {call_timeout}s",
                error_type=FailureType.TIMEOUT,
                duration_ms=duration_ms,
                circuit_state=self._state,
                provenance_hash=provenance_hash
            )

        except ConnectionError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            await self._record_failure(
                FailureType.CONNECTION_ERROR,
                str(e),
                duration_ms,
                args,
                kwargs
            )

            return CallResult(
                success=False,
                error=str(e),
                error_type=FailureType.CONNECTION_ERROR,
                duration_ms=duration_ms,
                circuit_state=self._state,
                provenance_hash=provenance_hash
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            failure_type = self._classify_exception(e)

            await self._record_failure(
                failure_type,
                str(e),
                duration_ms,
                args,
                kwargs
            )

            return CallResult(
                success=False,
                error=str(e),
                error_type=failure_type,
                duration_ms=duration_ms,
                circuit_state=self._state,
                provenance_hash=provenance_hash
            )

        finally:
            # Decrement half-open counter
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
            >>> @breaker.protect(cache_key="sensor_data")
            ... async def get_sensor_data():
            ...     return await external_api.get_sensors()
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

    @asynccontextmanager
    async def context(self):
        """
        Context manager for circuit breaker protected blocks.

        Example:
            >>> async with breaker.context():
            ...     result = await external_service.call()
        """
        yield self

    async def _record_success(self, duration_ms: float) -> None:
        """Record a successful call."""
        async with self._state_lock:
            self._total_successes += 1
            self._last_success_time = datetime.now(timezone.utc)

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1

                if self._success_count >= self.config.success_threshold:
                    await self._transition_to_closed()

            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success in closed state
                self._failure_count = 0

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
            self._last_failure_time = datetime.now(timezone.utc)

            # Record failure details
            record = FailureRecord(
                failure_type=failure_type,
                error_message=error_message[:500],  # Truncate long messages
                duration_ms=duration_ms,
                call_args_hash=self._hash_args(args, kwargs)
            )
            self._failure_records.append(record)

            # Safety logging per IEC 61511
            if self.config.enable_safety_logging:
                logger.warning(
                    f"[SAFETY] CircuitBreaker failure for {self.config.service_name}: "
                    f"type={failure_type.value}, count={self._failure_count}/{self.config.failure_threshold}"
                )

            # Check if we should open the circuit
            if self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    await self._transition_to_open()

            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately opens circuit
                await self._transition_to_open()

    async def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        previous_state = self._state
        self._state = CircuitState.OPEN
        self._last_state_change = datetime.now(timezone.utc)
        self._state_transitions += 1
        self._safety_trips += 1
        self._open_start_time = datetime.now(timezone.utc)
        self._success_count = 0

        logger.warning(
            f"[SAFETY] CircuitBreaker OPENED for {self.config.service_name}: "
            f"failures={self._failure_count}, "
            f"previous_state={previous_state.value}, "
            f"recovery_timeout={self.config.recovery_timeout_seconds}s"
        )

    async def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        previous_state = self._state

        # Calculate time spent in open state
        if self._open_start_time:
            self._time_in_open_seconds += (
                datetime.now(timezone.utc) - self._open_start_time
            ).total_seconds()

        self._state = CircuitState.HALF_OPEN
        self._last_state_change = datetime.now(timezone.utc)
        self._state_transitions += 1
        self._half_open_calls = 0
        self._success_count = 0
        self._failure_count = 0

        logger.info(
            f"CircuitBreaker HALF_OPEN for {self.config.service_name}: "
            f"testing recovery, max_calls={self.config.half_open_max_calls}"
        )

    async def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._last_state_change = datetime.now(timezone.utc)
        self._state_transitions += 1
        self._failure_count = 0
        self._success_count = 0
        self._open_start_time = None

        logger.info(
            f"CircuitBreaker CLOSED for {self.config.service_name}: "
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

        if self.config.recovery_strategy == RecoveryStrategy.FALLBACK_CACHE:
            if cache_key and cache_key in self._cache:
                return CallResult(
                    success=True,
                    value=self._cache[cache_key],
                    duration_ms=duration_ms,
                    from_cache=True,
                    circuit_state=self._state,
                    provenance_hash=provenance_hash
                )

        elif self.config.recovery_strategy == RecoveryStrategy.FALLBACK_DEFAULT:
            if self._fallback_provider:
                try:
                    fallback_value = self._fallback_provider(*args, **kwargs)
                    return CallResult(
                        success=True,
                        value=fallback_value,
                        duration_ms=duration_ms,
                        from_fallback=True,
                        circuit_state=self._state,
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
            provenance_hash=provenance_hash
        )

    def _classify_exception(self, exc: Exception) -> FailureType:
        """Classify exception into failure type."""
        exc_name = type(exc).__name__.lower()

        if 'timeout' in exc_name:
            return FailureType.TIMEOUT
        elif 'connection' in exc_name or 'network' in exc_name:
            return FailureType.CONNECTION_ERROR
        elif 'auth' in exc_name or 'permission' in exc_name:
            return FailureType.AUTHENTICATION
        elif 'ratelimit' in exc_name or 'throttle' in exc_name:
            return FailureType.RATE_LIMIT
        elif 'validation' in exc_name or 'invalid' in exc_name:
            return FailureType.VALIDATION_ERROR
        else:
            return FailureType.UNKNOWN

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
        """Get current circuit breaker state."""
        # Calculate failure rate
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=self.config.failure_window_seconds)

        recent_failures = sum(
            1 for r in self._failure_records
            if r.timestamp >= window_start
        )

        window_calls = self._total_calls - self._total_rejections
        failure_rate = recent_failures / max(1, window_calls)

        # Create provenance hash
        state_data = {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "timestamp": now.isoformat()
        }
        provenance_hash = hashlib.sha256(
            json.dumps(state_data, sort_keys=True).encode()
        ).hexdigest()

        return CircuitBreakerState(
            state=self._state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            last_failure_time=self._last_failure_time,
            last_success_time=self._last_success_time,
            last_state_change=self._last_state_change,
            total_calls=self._total_calls,
            total_failures=self._total_failures,
            total_successes=self._total_successes,
            current_failure_rate=min(1.0, failure_rate),
            safety_trips=self._safety_trips,
            provenance_hash=provenance_hash
        )

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics for monitoring."""
        # Calculate rates
        total = max(1, self._total_calls - self._total_rejections)
        success_rate = self._total_successes / total
        failure_rate = self._total_failures / total

        # Calculate latencies
        latencies = list(self._latencies)
        if latencies:
            sorted_latencies = sorted(latencies)
            avg_latency = sum(latencies) / len(latencies)
            p50_idx = int(len(sorted_latencies) * 0.5)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)

            p50_latency = sorted_latencies[p50_idx] if p50_idx < len(sorted_latencies) else 0
            p95_latency = sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else 0
            p99_latency = sorted_latencies[p99_idx] if p99_idx < len(sorted_latencies) else 0
        else:
            avg_latency = p50_latency = p95_latency = p99_latency = 0.0

        return CircuitBreakerMetrics(
            service_name=self.config.service_name,
            state=self._state,
            total_calls=self._total_calls,
            total_successes=self._total_successes,
            total_failures=self._total_failures,
            total_rejections=self._total_rejections,
            total_timeouts=self._total_timeouts,
            success_rate=success_rate,
            failure_rate=failure_rate,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            state_transitions=self._state_transitions,
            time_in_open_seconds=self._time_in_open_seconds,
            safety_trips=self._safety_trips,
            last_safety_trip=self._last_failure_time if self._safety_trips > 0 else None
        )

    async def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        async with self._state_lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._last_state_change = datetime.now(timezone.utc)

            logger.info(
                f"CircuitBreaker manually reset for {self.config.service_name}"
            )

    async def force_open(self, reason: str) -> None:
        """Manually force circuit to open state."""
        async with self._state_lock:
            self._state = CircuitState.OPEN
            self._last_state_change = datetime.now(timezone.utc)
            self._safety_trips += 1

            logger.warning(
                f"[SAFETY] CircuitBreaker force-opened for {self.config.service_name}: {reason}"
            )


# =============================================================================
# Circuit Breaker Registry
# =============================================================================

class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Provides centralized management and monitoring of all circuit breakers
    in the application.
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

    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            await breaker.reset()

        logger.info("All circuit breakers reset")


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()
