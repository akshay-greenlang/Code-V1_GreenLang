# -*- coding: utf-8 -*-
"""GL-010 EmissionsGuardian - Enhanced Circuit Breaker with Critical Path Protection.

This module extends the base circuit breaker with:
- Critical path protection for emission reporting (always report emissions)
- Graceful degradation modes for non-critical paths
- Configurable recovery strategies
- Health metrics and monitoring
- EPA substitute data support for CEMS failures

Author: GreenLang GL-010 EmissionsGuardian
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Deque, Dict, Generic, List, Optional, Tuple, TypeVar, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PathCriticality(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DegradationMode(str, Enum):
    FAIL_FAST = "fail_fast"
    USE_CACHE = "use_cache"
    USE_DEFAULT = "use_default"
    USE_SUBSTITUTE = "use_substitute"
    BEST_EFFORT = "best_effort"


class RecoveryStrategy(str, Enum):
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIBONACCI = "fibonacci"
    ADAPTIVE = "adaptive"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class EnhancedCircuitConfig(BaseModel):
    failure_threshold: int = Field(default=5, ge=1, le=100)
    recovery_timeout_seconds: float = Field(default=30.0, ge=1.0, le=3600.0)
    half_open_max_calls: int = Field(default=3, ge=1, le=20)
    success_threshold: int = Field(default=2, ge=1, le=20)
    timeout_seconds: float = Field(default=30.0, ge=0.1, le=300.0)
    failure_window_seconds: float = Field(default=60.0, ge=10.0, le=600.0)
    criticality: PathCriticality = Field(default=PathCriticality.MEDIUM)
    degradation_mode: DegradationMode = Field(default=DegradationMode.FAIL_FAST)
    recovery_strategy: RecoveryStrategy = Field(default=RecoveryStrategy.EXPONENTIAL_BACKOFF)
    max_recovery_attempts: int = Field(default=5, ge=1, le=20)
    enable_substitute_data: bool = Field(default=False)
    cache_ttl_seconds: float = Field(default=300.0, ge=0.0, le=3600.0)

    class Config:
        frozen = True


class CriticalPathConfig(BaseModel):
    always_report: bool = Field(default=True)
    use_substitute_data: bool = Field(default=True)
    substitute_method: str = Field(default="epa_d_method")
    max_substitute_hours: int = Field(default=720)
    alert_threshold_hours: int = Field(default=24)
    require_audit_trail: bool = Field(default=True)

    class Config:
        frozen = True
\


@dataclass
class FailureEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error_type: str = "unknown"
    error_message: str = ""
    operation: str = ""
    response_time_ms: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        if not self.provenance_hash:
            content = f"{self.event_id}|{self.timestamp.isoformat()}|{self.error_type}|{self.error_message}"
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


@dataclass
class DegradedResult(Generic[T]):
    value: Optional[T] = None
    is_degraded: bool = True
    degradation_mode: DegradationMode = DegradationMode.FAIL_FAST
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.value is not None


@dataclass
class HealthMetrics:
    status: HealthStatus = HealthStatus.UNKNOWN
    success_rate: float = 0.0
    avg_response_time_ms: float = 0.0
    failure_count_last_hour: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    circuit_state: CircuitState = CircuitState.CLOSED
    uptime_percent: float = 100.0


class CircuitBreakerError(Exception):
    pass


class CircuitOpenError(CircuitBreakerError):
    def __init__(self, name: str, opened_at: Optional[datetime] = None,
                 recovery_at: Optional[datetime] = None, message: Optional[str] = None):
        self.name = name
        self.opened_at = opened_at
        self.recovery_at = recovery_at
        msg = message or f"Circuit breaker '{name}' is OPEN"
        if recovery_at:
            msg += f" (recovery at {recovery_at.isoformat()})"
        super().__init__(msg)


class CriticalPathFailureError(CircuitBreakerError):
    def __init__(self, name: str, message: str, substitute_available: bool = False):
        self.name = name
        self.substitute_available = substitute_available
        super().__init__(message)


class EnhancedCircuitBreaker(Generic[T]):
    GENESIS_HASH = "0" * 64

    def __init__(self, name: str, config: Optional[EnhancedCircuitConfig] = None,
                 critical_config: Optional[CriticalPathConfig] = None):
        self.name = name
        self.config = config or EnhancedCircuitConfig()
        self.critical_config = critical_config or CriticalPathConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._opened_at: Optional[datetime] = None
        self._last_failure: Optional[datetime] = None
        self._last_success: Optional[datetime] = None
        self._lock = threading.RLock()
        self._failure_window: Deque[datetime] = deque()
        self._response_times: Deque[float] = deque(maxlen=100)
        self._cache: Dict[str, Tuple[T, datetime]] = {}
        self._recovery_attempt = 0
        self._total_failures = 0
        self._total_successes = 0
        self._total_rejections = 0
        logger.info(f"EnhancedCircuitBreaker '{name}' initialized with {self.config.criticality.value} criticality")

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    @property
    def is_critical(self) -> bool:
        return self.config.criticality == PathCriticality.CRITICAL


    def _clean_failure_window(self) -> None:
        cutoff = datetime.utcnow() - timedelta(seconds=self.config.failure_window_seconds)
        while self._failure_window and self._failure_window[0] < cutoff:
            self._failure_window.popleft()

    def _get_recovery_delay(self) -> float:
        base = self.config.recovery_timeout_seconds
        if self.config.recovery_strategy == RecoveryStrategy.FIXED_DELAY:
            return base
        elif self.config.recovery_strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
            return min(base * (2 ** self._recovery_attempt), 600)
        elif self.config.recovery_strategy == RecoveryStrategy.FIBONACCI:
            a, b = 1, 1
            for _ in range(self._recovery_attempt):
                a, b = b, a + b
            return min(base * a, 600)
        return base

    def _transition_state(self, new_state: CircuitState, reason: str = "") -> None:
        old_state = self._state
        if old_state == new_state:
            return
        self._state = new_state
        if new_state == CircuitState.OPEN:
            self._opened_at = datetime.utcnow()
            self._half_open_calls = 0
            self._recovery_attempt += 1
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._recovery_attempt = 0
            self._opened_at = None
        logger.warning(f"CircuitBreaker '{self.name}' state: {old_state.value} -> {new_state.value} ({reason})")

    def _should_allow(self) -> Tuple[bool, str]:
        with self._lock:
            self._clean_failure_window()
            if self._state == CircuitState.CLOSED:
                return True, "normal"
            elif self._state == CircuitState.OPEN:
                if self._opened_at:
                    delay = self._get_recovery_delay()
                    elapsed = (datetime.utcnow() - self._opened_at).total_seconds()
                    if elapsed >= delay:
                        self._transition_state(CircuitState.HALF_OPEN, "recovery timeout")
                        return True, "half_open_test"
                self._total_rejections += 1
                return False, "circuit_open"
            elif self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True, "half_open_call"
                self._total_rejections += 1
                return False, "half_open_limit"
            return False, "unknown"


    def _record_failure(self, error_type: str, error_message: str, operation: str,
                        response_time_ms: Optional[float] = None) -> None:
        now = datetime.utcnow()
        self._failure_window.append(now)
        self._failure_count = len(self._failure_window)
        self._total_failures += 1
        self._last_failure = now
        if response_time_ms:
            self._response_times.append(response_time_ms)
        if self._failure_count >= self.config.failure_threshold:
            self._transition_state(CircuitState.OPEN, f"threshold {self.config.failure_threshold} exceeded")
        elif self._state == CircuitState.HALF_OPEN:
            self._transition_state(CircuitState.OPEN, "failure during half_open")

    def _record_success(self, response_time_ms: Optional[float] = None) -> None:
        now = datetime.utcnow()
        self._total_successes += 1
        self._last_success = now
        if response_time_ms:
            self._response_times.append(response_time_ms)
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._transition_state(CircuitState.CLOSED, "recovery successful")

    def _get_cached(self, cache_key: str) -> Optional[T]:
        if cache_key in self._cache:
            value, cached_at = self._cache[cache_key]
            if (datetime.utcnow() - cached_at).total_seconds() < self.config.cache_ttl_seconds:
                return value
            del self._cache[cache_key]
        return None

    def _set_cache(self, cache_key: str, value: T) -> None:
        self._cache[cache_key] = (value, datetime.utcnow())


    def _handle_degraded(self, cache_key: Optional[str], fallback: Optional[Callable[..., T]],
                         default: Optional[T], *args: Any, **kwargs: Any) -> Union[T, DegradedResult[T]]:
        mode = self.config.degradation_mode
        if mode == DegradationMode.FAIL_FAST:
            raise CircuitOpenError(self.name, self._opened_at)
        elif mode == DegradationMode.USE_CACHE and cache_key:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return DegradedResult(value=cached, degradation_mode=mode, source="cache")
        elif mode == DegradationMode.USE_DEFAULT and default is not None:
            return DegradedResult(value=default, degradation_mode=mode, source="default")
        elif mode == DegradationMode.USE_SUBSTITUTE:
            if fallback is not None:
                try:
                    result = fallback(*args, **kwargs)
                    return DegradedResult(value=result, degradation_mode=mode, source="substitute")
                except Exception:
                    pass
        elif mode == DegradationMode.BEST_EFFORT:
            if cache_key:
                cached = self._get_cached(cache_key)
                if cached is not None:
                    return DegradedResult(value=cached, degradation_mode=mode, source="cache")
            if default is not None:
                return DegradedResult(value=default, degradation_mode=mode, source="default")
            if fallback is not None:
                try:
                    result = fallback(*args, **kwargs)
                    return DegradedResult(value=result, degradation_mode=mode, source="substitute")
                except Exception:
                    pass
        raise CircuitOpenError(self.name, self._opened_at)

    def call(self, func: Callable[..., T], *args: Any, cache_key: Optional[str] = None,
             fallback: Optional[Callable[..., T]] = None, default: Optional[T] = None,
             **kwargs: Any) -> Union[T, DegradedResult[T]]:
        allowed, reason = self._should_allow()
        if not allowed:
            if self.is_critical and self.critical_config.always_report:
                return self._handle_critical_rejection(cache_key, fallback, default, *args, **kwargs)
            return self._handle_degraded(cache_key, fallback, default, *args, **kwargs)
        start = time.time()
        operation = getattr(func, "__name__", str(func))
        try:
            result = func(*args, **kwargs)
            elapsed = (time.time() - start) * 1000
            with self._lock:
                self._record_success(elapsed)
            if cache_key:
                self._set_cache(cache_key, result)
            return result
        except TimeoutError as e:
            with self._lock:
                self._record_failure("timeout", str(e), operation, (time.time() - start) * 1000)
            raise
        except ConnectionError as e:
            with self._lock:
                self._record_failure("connection", str(e), operation, (time.time() - start) * 1000)
            raise
        except Exception as e:
            with self._lock:
                self._record_failure("unknown", str(e), operation, (time.time() - start) * 1000)
            raise


    def _handle_critical_rejection(self, cache_key: Optional[str], fallback: Optional[Callable[..., T]],
                                   default: Optional[T], *args: Any, **kwargs: Any) -> Union[T, DegradedResult[T]]:
        logger.critical(f"CRITICAL PATH: Circuit '{self.name}' open but must report")
        if self.critical_config.use_substitute_data and fallback is not None:
            try:
                result = fallback(*args, **kwargs)
                logger.warning(f"CRITICAL PATH: Using substitute data for '{self.name}'")
                return DegradedResult(
                    value=result, degradation_mode=DegradationMode.USE_SUBSTITUTE,
                    source="epa_substitute", metadata={"substitute_method": self.critical_config.substitute_method}
                )
            except Exception as e:
                logger.error(f"CRITICAL PATH: Substitute data failed: {e}")
        if cache_key:
            cached = self._get_cached(cache_key)
            if cached is not None:
                logger.warning(f"CRITICAL PATH: Using cached data for '{self.name}'")
                return DegradedResult(value=cached, degradation_mode=DegradationMode.USE_CACHE, source="cache")
        if default is not None:
            logger.warning(f"CRITICAL PATH: Using default for '{self.name}'")
            return DegradedResult(value=default, degradation_mode=DegradationMode.USE_DEFAULT, source="default")
        raise CriticalPathFailureError(
            self.name, f"Critical path '{self.name}' failed with no substitute available",
            substitute_available=False
        )

    def get_health_metrics(self) -> HealthMetrics:
        with self._lock:
            total = self._total_successes + self._total_failures
            success_rate = (self._total_successes / total * 100) if total > 0 else 100.0
            avg_time = sum(self._response_times) / len(self._response_times) if self._response_times else 0.0
            hour_ago = datetime.utcnow() - timedelta(hours=1)
            failures_last_hour = sum(1 for t in self._failure_window if t > hour_ago)
            if self._state == CircuitState.CLOSED and success_rate >= 95:
                status = HealthStatus.HEALTHY
            elif self._state == CircuitState.OPEN:
                status = HealthStatus.UNHEALTHY
            elif success_rate >= 80:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.UNHEALTHY
            return HealthMetrics(
                status=status, success_rate=success_rate, avg_response_time_ms=avg_time,
                failure_count_last_hour=failures_last_hour, last_success=self._last_success,
                last_failure=self._last_failure, circuit_state=self._state
            )

    def protect(self, cache_key: Optional[str] = None, fallback: Optional[Callable[..., T]] = None,
                default: Optional[T] = None) -> Callable[[Callable[..., T]], Callable[..., Union[T, DegradedResult[T]]]]:
        def decorator(func: Callable[..., T]) -> Callable[..., Union[T, DegradedResult[T]]]:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Union[T, DegradedResult[T]]:
                return self.call(func, *args, cache_key=cache_key, fallback=fallback, default=default, **kwargs)
            return wrapper
        return decorator

    def force_open(self, reason: str = "Manual") -> None:
        with self._lock:
            self._transition_state(CircuitState.OPEN, reason)

    def force_close(self, reason: str = "Manual") -> None:
        with self._lock:
            self._transition_state(CircuitState.CLOSED, reason)

    def reset(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._opened_at = None
            self._recovery_attempt = 0
            self._failure_window.clear()
            logger.info(f"CircuitBreaker '{self.name}' reset")


def create_critical_emission_breaker(name: str, substitute_func: Optional[Callable] = None) -> EnhancedCircuitBreaker:
    config = EnhancedCircuitConfig(
        failure_threshold=3, recovery_timeout_seconds=15.0, half_open_max_calls=2,
        criticality=PathCriticality.CRITICAL, degradation_mode=DegradationMode.USE_SUBSTITUTE,
        enable_substitute_data=True
    )
    critical_config = CriticalPathConfig(
        always_report=True, use_substitute_data=True, substitute_method="epa_d_method",
        max_substitute_hours=720, require_audit_trail=True
    )
    return EnhancedCircuitBreaker(name, config, critical_config)


def create_compliance_breaker(name: str) -> EnhancedCircuitBreaker:
    config = EnhancedCircuitConfig(
        failure_threshold=5, recovery_timeout_seconds=60.0, half_open_max_calls=3,
        criticality=PathCriticality.HIGH, degradation_mode=DegradationMode.BEST_EFFORT,
        recovery_strategy=RecoveryStrategy.EXPONENTIAL_BACKOFF
    )
    return EnhancedCircuitBreaker(name, config)


def create_market_data_breaker(name: str) -> EnhancedCircuitBreaker:
    config = EnhancedCircuitConfig(
        failure_threshold=5, recovery_timeout_seconds=15.0, half_open_max_calls=3,
        criticality=PathCriticality.MEDIUM, degradation_mode=DegradationMode.USE_CACHE,
        cache_ttl_seconds=300.0
    )
    return EnhancedCircuitBreaker(name, config)


__all__ = [
    "PathCriticality", "DegradationMode", "RecoveryStrategy", "HealthStatus", "CircuitState",
    "EnhancedCircuitConfig", "CriticalPathConfig", "FailureEvent", "DegradedResult", "HealthMetrics",
    "CircuitBreakerError", "CircuitOpenError", "CriticalPathFailureError",
    "EnhancedCircuitBreaker", "create_critical_emission_breaker", "create_compliance_breaker",
    "create_market_data_breaker",
]
