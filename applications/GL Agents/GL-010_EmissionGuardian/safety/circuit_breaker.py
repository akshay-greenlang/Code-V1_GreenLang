# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Safety Circuit Breaker for External Integrations

Implements the circuit breaker pattern specifically designed for CEMS, EPA, and
market data external system integrations. Provides automatic failure detection,
isolation, and recovery with complete audit trails for regulatory compliance.

Design Principles:
    - Fail-safe: Default to safe state on any error
    - Deterministic: No ML inference for safety decisions
    - Auditable: Complete provenance tracking with SHA-256
    - Resilient: Automatic recovery with configurable backoff

Integration Points:
    - CEMS data acquisition systems
    - EPA electronic reporting APIs
    - Carbon market data providers (ICE, CME, CBL)
    - Offset registry verification APIs

Reference: IEC 61508 Safety Integrity Level patterns

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
from typing import (
    Any, Callable, Deque, Dict, Generic, List, Optional,
    Tuple, TypeVar,
)

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states: CLOSED (normal), OPEN (blocking), HALF_OPEN (testing)."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class FailureType(str, Enum):
    """Types of failures tracked by the circuit breaker."""
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    VALIDATION_ERROR = "validation_error"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    SERVER_ERROR = "server_error"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


class IntegrationType(str, Enum):
    """Types of external integrations protected by circuit breakers."""
    CEMS = "cems"
    EPA_REPORTING = "epa_reporting"
    MARKET_DATA = "market_data"
    OFFSET_REGISTRY = "offset_registry"
    ERP = "erp"
    SCADA = "scada"


@dataclass
class FailureRecord:
    """Record of a single failure event with provenance tracking."""
    failure_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    failure_type: FailureType = FailureType.UNKNOWN
    integration_type: IntegrationType = IntegrationType.CEMS
    operation: str = ""
    error_message: str = ""
    error_code: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    response_time_ms: Optional[float] = None
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_provenance_hash()

    def _calculate_provenance_hash(self) -> str:
        content = (
            f"{self.failure_id}|{self.timestamp.isoformat()}|"
            f"{self.failure_type.value}|{self.integration_type.value}|"
            f"{self.operation}|{self.error_message}"
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "timestamp": self.timestamp.isoformat(),
            "failure_type": self.failure_type.value,
            "integration_type": self.integration_type.value,
            "operation": self.operation,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "context": self.context,
            "response_time_ms": self.response_time_ms,
            "provenance_hash": self.provenance_hash,
        }


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = Field(default=5, ge=1, le=100, description="Failures before opening circuit")
    recovery_timeout_seconds: float = Field(default=30.0, ge=1.0, le=3600.0, description="Seconds before half-open test")
    half_open_max_calls: int = Field(default=3, ge=1, le=20, description="Maximum calls in half-open state")
    success_threshold: int = Field(default=2, ge=1, le=20, description="Successes to close from half-open")
    timeout_seconds: float = Field(default=30.0, ge=0.1, le=300.0, description="Default operation timeout")
    failure_window_seconds: float = Field(default=60.0, ge=10.0, le=600.0, description="Sliding window for failure counting")
    enable_audit_logging: bool = Field(default=True, description="Log all state changes for audit")
    timeout_threshold: Optional[int] = Field(default=None, description="Override threshold for timeout failures")
    connection_error_threshold: Optional[int] = Field(default=None, description="Override threshold for connection errors")
    validation_error_threshold: Optional[int] = Field(default=None, description="Override threshold for validation errors")

    @validator("success_threshold")
    def success_threshold_lte_half_open(cls, v: int, values: Dict) -> int:
        if "half_open_max_calls" in values and v > values["half_open_max_calls"]:
            raise ValueError("success_threshold must be <= half_open_max_calls")
        return v

    class Config:
        frozen = True


@dataclass
class CircuitBreakerState:
    """Current state of a circuit breaker with complete metrics."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    half_open_call_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_state_change_time: Optional[datetime] = None
    opened_at: Optional[datetime] = None
    total_failures: int = 0
    total_successes: int = 0
    total_rejections: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    def reset(self) -> None:
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_call_count = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.last_state_change_time = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "half_open_call_count": self.half_open_call_count,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "total_rejections": self.total_rejections,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
        }


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""
    pass


class CircuitOpenError(CircuitBreakerError):
    """Raised when circuit is open and call is rejected."""
    def __init__(self, circuit_name: str, opened_at: Optional[datetime] = None,
                 recovery_at: Optional[datetime] = None, message: Optional[str] = None):
        self.circuit_name = circuit_name
        self.opened_at = opened_at
        self.recovery_at = recovery_at
        if message is None:
            message = f"Circuit breaker '{circuit_name}' is OPEN"
            if recovery_at:
                message += f" (recovery at {recovery_at.isoformat()})"
        super().__init__(message)


class CircuitBreaker(Generic[T]):
    """
    Circuit breaker for external integration resilience.
    
    Implements the circuit breaker pattern with three states:
    - CLOSED: Normal operation, requests flow through
    - OPEN: Failures exceeded threshold, requests fail fast
    - HALF_OPEN: Testing if service has recovered
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None,
                 integration_type: IntegrationType = IntegrationType.CEMS):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.integration_type = integration_type
        self._state = CircuitBreakerState()
        self._lock = threading.RLock()
        self._failure_window: Deque[datetime] = deque()
        self._failure_history: List[FailureRecord] = []
        self._state_change_callbacks: List[Callable[[CircuitState, CircuitState], None]] = []
        logger.info(f"CircuitBreaker '{name}' initialized for {integration_type.value} with threshold={self.config.failure_threshold}")

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state.state

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    @property
    def is_half_open(self) -> bool:
        return self.state == CircuitState.HALF_OPEN

    def register_state_change_callback(self, callback: Callable[[CircuitState, CircuitState], None]) -> None:
        self._state_change_callbacks.append(callback)

    def _clean_failure_window(self) -> None:
        cutoff = datetime.utcnow() - timedelta(seconds=self.config.failure_window_seconds)
        while self._failure_window and self._failure_window[0] < cutoff:
            self._failure_window.popleft()

    def _get_failure_threshold(self, failure_type: FailureType) -> int:
        if failure_type == FailureType.TIMEOUT and self.config.timeout_threshold is not None:
            return self.config.timeout_threshold
        elif failure_type == FailureType.CONNECTION_ERROR and self.config.connection_error_threshold is not None:
            return self.config.connection_error_threshold
        elif failure_type == FailureType.VALIDATION_ERROR and self.config.validation_error_threshold is not None:
            return self.config.validation_error_threshold
        return self.config.failure_threshold

    def _transition_state(self, new_state: CircuitState, reason: str = "") -> None:
        old_state = self._state.state
        if old_state == new_state:
            return
        self._state.state = new_state
        self._state.last_state_change_time = datetime.utcnow()
        if new_state == CircuitState.OPEN:
            self._state.opened_at = datetime.utcnow()
            self._state.half_open_call_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._state.half_open_call_count = 0
            self._state.success_count = 0
        elif new_state == CircuitState.CLOSED:
            self._state.failure_count = 0
            self._state.consecutive_failures = 0
            self._state.opened_at = None
        if self.config.enable_audit_logging:
            logger.warning(f"CircuitBreaker '{self.name}' state change: {old_state.value} -> {new_state.value}{' (' + reason + ')' if reason else ''}")
        for callback in self._state_change_callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")


    def _record_failure(self, failure_type: FailureType, error_message: str, operation: str = "",
                       error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None,
                       response_time_ms: Optional[float] = None) -> FailureRecord:
        record = FailureRecord(failure_type=failure_type, integration_type=self.integration_type,
                              operation=operation, error_message=error_message, error_code=error_code,
                              context=context or {}, response_time_ms=response_time_ms)
        now = datetime.utcnow()
        self._failure_window.append(now)
        self._failure_history.append(record)
        self._state.failure_count = len(self._failure_window)
        self._state.total_failures += 1
        self._state.consecutive_failures += 1
        self._state.consecutive_successes = 0
        self._state.last_failure_time = now
        if len(self._failure_history) > 1000:
            self._failure_history = self._failure_history[-500:]
        if self.config.enable_audit_logging:
            logger.warning(f"CircuitBreaker '{self.name}' failure recorded: {failure_type.value} - {error_message[:100]}")
        return record

    def _record_success(self) -> None:
        now = datetime.utcnow()
        self._state.total_successes += 1
        self._state.consecutive_successes += 1
        self._state.consecutive_failures = 0
        self._state.last_success_time = now
        if self._state.state == CircuitState.HALF_OPEN:
            self._state.success_count += 1
            if self._state.success_count >= self.config.success_threshold:
                self._transition_state(CircuitState.CLOSED, f"Recovery successful after {self._state.success_count} successes")

    def _should_allow_request(self) -> Tuple[bool, str]:
        with self._lock:
            self._clean_failure_window()
            if self._state.state == CircuitState.CLOSED:
                return True, "Circuit closed - normal operation"
            elif self._state.state == CircuitState.OPEN:
                if self._state.opened_at:
                    elapsed = (datetime.utcnow() - self._state.opened_at).total_seconds()
                    if elapsed >= self.config.recovery_timeout_seconds:
                        self._transition_state(CircuitState.HALF_OPEN, "Recovery timeout elapsed")
                        return True, "Entering half-open state"
                self._state.total_rejections += 1
                recovery_at = self._state.opened_at + timedelta(seconds=self.config.recovery_timeout_seconds) if self._state.opened_at else None
                return False, f"Circuit open until {recovery_at}"
            elif self._state.state == CircuitState.HALF_OPEN:
                if self._state.half_open_call_count < self.config.half_open_max_calls:
                    self._state.half_open_call_count += 1
                    return True, "Half-open test call"
                else:
                    self._state.total_rejections += 1
                    return False, "Half-open call limit reached"
            return False, "Unknown state"

    def _handle_failure(self, failure_type: FailureType, error_message: str, operation: str,
                       error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None,
                       response_time_ms: Optional[float] = None) -> None:
        self._record_failure(failure_type=failure_type, error_message=error_message, operation=operation,
                            error_code=error_code, context=context, response_time_ms=response_time_ms)
        threshold = self._get_failure_threshold(failure_type)
        if len(self._failure_window) >= threshold:
            self._transition_state(CircuitState.OPEN, f"Threshold {threshold} exceeded for {failure_type.value}")
        elif self._state.state == CircuitState.HALF_OPEN:
            self._transition_state(CircuitState.OPEN, f"Failure during half-open test: {failure_type.value}")


    def call(self, func: Callable[..., T], *args: Any, timeout: Optional[float] = None,
             fallback: Optional[Callable[..., T]] = None, **kwargs: Any) -> T:
        """Execute a function with circuit breaker protection."""
        allowed, reason = self._should_allow_request()
        if not allowed:
            if fallback is not None:
                logger.info(f"CircuitBreaker '{self.name}' using fallback: {reason}")
                return fallback(*args, **kwargs)
            raise CircuitOpenError(self.name, opened_at=self._state.opened_at,
                                   recovery_at=(self._state.opened_at + timedelta(seconds=self.config.recovery_timeout_seconds)
                                               if self._state.opened_at else None))
        start_time = time.time()
        operation = getattr(func, "__name__", str(func))
        try:
            result = func(*args, **kwargs)
            elapsed_ms = (time.time() - start_time) * 1000
            effective_timeout = timeout or self.config.timeout_seconds
            if elapsed_ms > effective_timeout * 1000:
                logger.warning(f"CircuitBreaker '{self.name}' slow call: {elapsed_ms:.0f}ms > {effective_timeout * 1000:.0f}ms")
            with self._lock:
                self._record_success()
            return result
        except TimeoutError as e:
            with self._lock:
                self._handle_failure(FailureType.TIMEOUT, str(e), operation, response_time_ms=(time.time() - start_time) * 1000)
            raise
        except ConnectionError as e:
            with self._lock:
                self._handle_failure(FailureType.CONNECTION_ERROR, str(e), operation, response_time_ms=(time.time() - start_time) * 1000)
            raise
        except ValueError as e:
            with self._lock:
                self._handle_failure(FailureType.VALIDATION_ERROR, str(e), operation, response_time_ms=(time.time() - start_time) * 1000)
            raise
        except Exception as e:
            with self._lock:
                self._handle_failure(FailureType.UNKNOWN, str(e), operation, context={"exception_type": type(e).__name__},
                                    response_time_ms=(time.time() - start_time) * 1000)
            raise

    def protect(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to protect a function with this circuit breaker."""
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self.call(func, *args, **kwargs)
        return wrapper

    def __enter__(self) -> "CircuitBreaker[T]":
        allowed, reason = self._should_allow_request()
        if not allowed:
            raise CircuitOpenError(self.name, opened_at=self._state.opened_at)
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Any) -> bool:
        if exc_type is None:
            with self._lock:
                self._record_success()
        else:
            failure_type = FailureType.UNKNOWN
            if exc_type is TimeoutError:
                failure_type = FailureType.TIMEOUT
            elif exc_type is ConnectionError:
                failure_type = FailureType.CONNECTION_ERROR
            elif exc_type is ValueError:
                failure_type = FailureType.VALIDATION_ERROR
            with self._lock:
                self._handle_failure(failure_type=failure_type, error_message=str(exc_val) if exc_val else "Unknown error", operation="context_manager")
        return False

    def force_open(self, reason: str = "Manual override") -> None:
        with self._lock:
            logger.critical(f"CircuitBreaker '{self.name}' FORCED OPEN: {reason}")
            self._transition_state(CircuitState.OPEN, reason)

    def force_close(self, reason: str = "Manual reset") -> None:
        with self._lock:
            logger.warning(f"CircuitBreaker '{self.name}' FORCED CLOSED: {reason}")
            self._transition_state(CircuitState.CLOSED, reason)

    def reset(self) -> None:
        with self._lock:
            self._state.reset()
            self._failure_window.clear()
            logger.info(f"CircuitBreaker '{self.name}' reset")

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {"name": self.name, "integration_type": self.integration_type.value,
                    "config": {"failure_threshold": self.config.failure_threshold,
                               "recovery_timeout_seconds": self.config.recovery_timeout_seconds,
                               "half_open_max_calls": self.config.half_open_max_calls},
                    "state": self._state.to_dict()}

    def get_failure_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            return [record.to_dict() for record in self._failure_history[-limit:]]


class IntegrationCircuitBreakers:
    """Registry of circuit breakers for different integration types."""
    _instance: Optional["IntegrationCircuitBreakers"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "IntegrationCircuitBreakers":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._registry_lock = threading.Lock()
        logger.info("IntegrationCircuitBreakers registry initialized")

    def register(self, breaker: CircuitBreaker) -> None:
        with self._registry_lock:
            self._breakers[breaker.name] = breaker
            logger.info(f"Registered circuit breaker: {breaker.name}")

    def get(self, name: str) -> Optional[CircuitBreaker]:
        return self._breakers.get(name)

    def get_or_create(self, name: str, integration_type: IntegrationType,
                     config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        with self._registry_lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name=name, config=config, integration_type=integration_type)
            return self._breakers[name]

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        return {name: b.get_status() for name, b in self._breakers.items()}

    def get_by_type(self, integration_type: IntegrationType) -> List[CircuitBreaker]:
        return [b for b in self._breakers.values() if b.integration_type == integration_type]

    def force_open_all(self, reason: str = "Emergency") -> None:
        for breaker in self._breakers.values():
            breaker.force_open(reason)

    def reset_all(self) -> None:
        for breaker in self._breakers.values():
            breaker.reset()


_registry = IntegrationCircuitBreakers()


def get_cems_circuit_breaker(name: str = "cems_default", config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create a circuit breaker for CEMS integration."""
    if config is None:
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout_seconds=30.0, half_open_max_calls=2, timeout_seconds=10.0)
    return _registry.get_or_create(name, IntegrationType.CEMS, config)


def get_epa_circuit_breaker(name: str = "epa_default", config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create a circuit breaker for EPA reporting API."""
    if config is None:
        config = CircuitBreakerConfig(failure_threshold=5, recovery_timeout_seconds=60.0, half_open_max_calls=3, timeout_seconds=30.0)
    return _registry.get_or_create(name, IntegrationType.EPA_REPORTING, config)


def get_market_data_circuit_breaker(name: str = "market_data_default", config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create a circuit breaker for market data APIs."""
    if config is None:
        config = CircuitBreakerConfig(failure_threshold=5, recovery_timeout_seconds=15.0, half_open_max_calls=3, timeout_seconds=5.0)
    return _registry.get_or_create(name, IntegrationType.MARKET_DATA, config)


def circuit_protected(breaker: CircuitBreaker, fallback: Optional[Callable[..., T]] = None,
                     timeout: Optional[float] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator factory for circuit breaker protection."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return breaker.call(func, *args, timeout=timeout, fallback=fallback, **kwargs)
        return wrapper
    return decorator


__all__ = [
    "CircuitBreaker", "CircuitBreakerConfig", "CircuitBreakerState",
    "CircuitState", "FailureType", "IntegrationType", "FailureRecord",
    "CircuitBreakerError", "CircuitOpenError", "IntegrationCircuitBreakers",
    "get_cems_circuit_breaker", "get_epa_circuit_breaker", "get_market_data_circuit_breaker",
    "circuit_protected",
]
