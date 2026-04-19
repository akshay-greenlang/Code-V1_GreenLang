# -*- coding: utf-8 -*-
"""
GreenLang Circuit Breaker Implementation

Production-ready circuit breaker pattern for protecting external service calls.

Features:
- Three states: CLOSED, OPEN, HALF_OPEN
- Configurable failure thresholds and timeouts
- Automatic recovery attempts
- Prometheus metrics integration
- Thread-safe implementation
- Fallback support

Based on Netflix Hystrix and Martin Fowler's circuit breaker pattern.
Uses pybreaker library with GreenLang telemetry integration.

Author: GreenLang Platform Team
Version: 1.0.0
Date: 2025-11-09
"""

import time
import threading
from enum import Enum
from typing import Any, Callable, Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps

import pybreaker
from prometheus_client import Counter, Gauge, Histogram

from greenlang.telemetry import get_logger, MetricsCollector


# ============================================================================
# EXCEPTIONS
# ============================================================================

class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""
    pass


class CircuitOpenError(CircuitBreakerError):
    """Raised when circuit is open and calls are blocked."""
    pass


class CircuitBreakerTimeoutError(CircuitBreakerError):
    """Raised when a call times out."""
    pass


# ============================================================================
# STATE ENUM
# ============================================================================

class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, calls go through
    OPEN = "open"          # Circuit is open, calls are blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class CircuitBreakerConfig:
    """
    Circuit breaker configuration.

    Attributes:
        name: Unique name for this circuit breaker
        fail_max: Number of failures before opening circuit
        timeout_duration: Seconds to wait before attempting recovery (OPEN -> HALF_OPEN)
        expected_exception: Exception type(s) that count as failures
        reset_timeout: Time window for counting failures (seconds)
        exclude_exceptions: Exceptions that don't count as failures
        fallback_function: Optional fallback to call when circuit is open
        listeners: Event listeners for state changes
    """
    name: str
    fail_max: int = 5
    timeout_duration: int = 60
    reset_timeout: int = 60
    expected_exception: type = Exception
    exclude_exceptions: List[type] = field(default_factory=list)
    fallback_function: Optional[Callable] = None
    call_timeout: Optional[int] = None  # Timeout for individual calls


# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

# Circuit breaker state gauge
circuit_breaker_state = Gauge(
    'greenlang_circuit_breaker_state',
    'Current state of circuit breaker (0=closed, 1=open, 2=half_open)',
    ['circuit_name']
)

# Call counters
circuit_breaker_calls_total = Counter(
    'greenlang_circuit_breaker_calls_total',
    'Total number of calls through circuit breaker',
    ['circuit_name', 'status']  # status: success, failure, rejected
)

# State transition counter
circuit_breaker_state_transitions = Counter(
    'greenlang_circuit_breaker_state_transitions_total',
    'Total state transitions',
    ['circuit_name', 'from_state', 'to_state']
)

# Call duration histogram
circuit_breaker_call_duration = Histogram(
    'greenlang_circuit_breaker_call_duration_seconds',
    'Duration of calls through circuit breaker',
    ['circuit_name', 'status']
)

# Failure rate gauge
circuit_breaker_failure_rate = Gauge(
    'greenlang_circuit_breaker_failure_rate',
    'Current failure rate (failures/total calls)',
    ['circuit_name']
)


# ============================================================================
# CIRCUIT BREAKER LISTENER
# ============================================================================

class CircuitBreakerListener(pybreaker.CircuitBreakerListener):
    """
    Event listener for circuit breaker state changes.
    Integrates with GreenLang telemetry and Prometheus.
    """

    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"circuitbreaker.{name}")
        self._call_count = 0
        self._failure_count = 0
        self._lock = threading.Lock()

    def before_call(self, cb, func, *args, **kwargs):
        """Called before each call attempt."""
        self.logger.debug(
            f"Circuit breaker call attempt",
            extra={
                "circuit_name": self.name,
                "state": cb.current_state,
                "function": func.__name__,
            }
        )

    def state_change(self, cb, old_state, new_state):
        """Called when circuit breaker state changes."""
        old_state_name = self._state_to_string(old_state)
        new_state_name = self._state_to_string(new_state)

        self.logger.warning(
            f"Circuit breaker state transition: {old_state_name} -> {new_state_name}",
            extra={
                "circuit_name": self.name,
                "old_state": old_state_name,
                "new_state": new_state_name,
                "fail_counter": cb.fail_counter,
                "fail_max": cb.fail_max,
            }
        )

        # Update Prometheus metrics
        circuit_breaker_state_transitions.labels(
            circuit_name=self.name,
            from_state=old_state_name,
            to_state=new_state_name
        ).inc()

        self._update_state_gauge(new_state)

    def success(self, cb):
        """Called after successful call."""
        with self._lock:
            self._call_count += 1

        circuit_breaker_calls_total.labels(
            circuit_name=self.name,
            status="success"
        ).inc()

        self._update_failure_rate()

        self.logger.debug(
            f"Circuit breaker call succeeded",
            extra={
                "circuit_name": self.name,
                "state": cb.current_state,
            }
        )

    def failure(self, cb, exception):
        """Called after failed call."""
        with self._lock:
            self._call_count += 1
            self._failure_count += 1

        circuit_breaker_calls_total.labels(
            circuit_name=self.name,
            status="failure"
        ).inc()

        self._update_failure_rate()

        self.logger.error(
            f"Circuit breaker call failed",
            extra={
                "circuit_name": self.name,
                "state": cb.current_state,
                "exception": str(exception),
                "exception_type": type(exception).__name__,
                "fail_counter": cb.fail_counter,
                "fail_max": cb.fail_max,
            }
        )

    def call_rejected(self, cb):
        """Called when call is rejected (circuit is open)."""
        circuit_breaker_calls_total.labels(
            circuit_name=self.name,
            status="rejected"
        ).inc()

        self.logger.warning(
            f"Circuit breaker rejected call - circuit is OPEN",
            extra={
                "circuit_name": self.name,
                "state": "open",
            }
        )

    def _state_to_string(self, state) -> str:
        """Convert pybreaker state to string."""
        if isinstance(state, pybreaker.STATE_CLOSED):
            return "closed"
        elif isinstance(state, pybreaker.STATE_OPEN):
            return "open"
        elif isinstance(state, pybreaker.STATE_HALF_OPEN):
            return "half_open"
        return "unknown"

    def _update_state_gauge(self, state):
        """Update Prometheus state gauge."""
        state_value = 0
        if isinstance(state, pybreaker.STATE_CLOSED):
            state_value = 0
        elif isinstance(state, pybreaker.STATE_OPEN):
            state_value = 1
        elif isinstance(state, pybreaker.STATE_HALF_OPEN):
            state_value = 2

        circuit_breaker_state.labels(circuit_name=self.name).set(state_value)

    def _update_failure_rate(self):
        """Update failure rate metric."""
        with self._lock:
            if self._call_count > 0:
                rate = self._failure_count / self._call_count
                circuit_breaker_failure_rate.labels(circuit_name=self.name).set(rate)


# ============================================================================
# CIRCUIT BREAKER WRAPPER
# ============================================================================

class CircuitBreaker:
    """
    Production-ready circuit breaker for protecting external service calls.

    This wraps pybreaker.CircuitBreaker with GreenLang telemetry integration,
    Prometheus metrics, and additional safety features.

    States:
        CLOSED: Normal operation - calls go through
        OPEN: Circuit is open - calls are immediately rejected
        HALF_OPEN: Testing recovery - limited calls allowed

    Example:
        >>> config = CircuitBreakerConfig(
        ...     name="api_service",
        ...     fail_max=5,
        ...     timeout_duration=60
        ... )
        >>> cb = CircuitBreaker(config)
        >>>
        >>> @cb.protect
        ... def call_external_api():
        ...     return requests.get("https://api.example.com")
        >>>
        >>> result = call_external_api()
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.logger = get_logger(f"circuitbreaker.{config.name}")
        self.listener = CircuitBreakerListener(config.name)

        # Create pybreaker circuit breaker
        self._breaker = pybreaker.CircuitBreaker(
            fail_max=config.fail_max,
            timeout_duration=config.timeout_duration,
            reset_timeout=config.reset_timeout,
            expected_exception=config.expected_exception,
            exclude=config.exclude_exceptions,
            listeners=[self.listener],
            name=config.name,
        )

        self.logger.info(
            f"Circuit breaker initialized",
            extra={
                "circuit_name": config.name,
                "fail_max": config.fail_max,
                "timeout_duration": config.timeout_duration,
                "reset_timeout": config.reset_timeout,
            }
        )

    def protect(self, func: Callable) -> Callable:
        """
        Decorator to protect a function with circuit breaker.

        Args:
            func: Function to protect

        Returns:
            Wrapped function with circuit breaker protection

        Example:
            >>> @circuit_breaker.protect
            ... def risky_operation():
            ...     return external_api_call()
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call a function with circuit breaker protection.

        Args:
            func: Function to call
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Result from function call

        Raises:
            CircuitOpenError: If circuit is open
            CircuitBreakerError: On other circuit breaker errors
        """
        start_time = time.time()

        try:
            # Call through circuit breaker
            result = self._breaker.call(func, *args, **kwargs)

            # Record success duration
            duration = time.time() - start_time
            circuit_breaker_call_duration.labels(
                circuit_name=self.config.name,
                status="success"
            ).observe(duration)

            return result

        except pybreaker.CircuitBreakerError as e:
            # Circuit is open - try fallback if available
            duration = time.time() - start_time
            circuit_breaker_call_duration.labels(
                circuit_name=self.config.name,
                status="rejected"
            ).observe(duration)

            if self.config.fallback_function:
                self.logger.info(
                    f"Circuit open - using fallback",
                    extra={"circuit_name": self.config.name}
                )
                return self.config.fallback_function(*args, **kwargs)

            raise CircuitOpenError(
                f"Circuit breaker '{self.config.name}' is OPEN. "
                f"Service is temporarily unavailable."
            ) from e

        except Exception as e:
            # Other errors
            duration = time.time() - start_time
            circuit_breaker_call_duration.labels(
                circuit_name=self.config.name,
                status="failure"
            ).observe(duration)
            raise

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        current = self._breaker.current_state
        if isinstance(current, pybreaker.STATE_CLOSED):
            return CircuitBreakerState.CLOSED
        elif isinstance(current, pybreaker.STATE_OPEN):
            return CircuitBreakerState.OPEN
        elif isinstance(current, pybreaker.STATE_HALF_OPEN):
            return CircuitBreakerState.HALF_OPEN
        return CircuitBreakerState.CLOSED

    @property
    def fail_counter(self) -> int:
        """Get current failure count."""
        return self._breaker.fail_counter

    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        self._breaker.close()
        self.logger.info(
            f"Circuit breaker manually reset to CLOSED",
            extra={"circuit_name": self.config.name}
        )

    def open(self):
        """Manually open circuit breaker."""
        self._breaker.open()
        self.logger.warning(
            f"Circuit breaker manually opened",
            extra={"circuit_name": self.config.name}
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get circuit breaker statistics.

        Returns:
            Dictionary with current stats
        """
        return {
            "name": self.config.name,
            "state": self.state.value,
            "fail_counter": self.fail_counter,
            "fail_max": self.config.fail_max,
            "timeout_duration": self.config.timeout_duration,
            "total_calls": self.listener._call_count,
            "total_failures": self.listener._failure_count,
            "failure_rate": (
                self.listener._failure_count / self.listener._call_count
                if self.listener._call_count > 0 else 0
            ),
        }


# ============================================================================
# GLOBAL REGISTRY
# ============================================================================

_circuit_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def create_circuit_breaker(config: CircuitBreakerConfig) -> CircuitBreaker:
    """
    Create or get a circuit breaker from the global registry.

    Args:
        config: Circuit breaker configuration

    Returns:
        CircuitBreaker instance

    Example:
        >>> config = CircuitBreakerConfig(name="my_service", fail_max=3)
        >>> cb = create_circuit_breaker(config)
    """
    with _registry_lock:
        if config.name in _circuit_breakers:
            return _circuit_breakers[config.name]

        cb = CircuitBreaker(config)
        _circuit_breakers[config.name] = cb
        return cb


def get_circuit_breaker_stats(name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get statistics for circuit breakers.

    Args:
        name: Specific circuit breaker name, or None for all

    Returns:
        Dictionary of stats
    """
    with _registry_lock:
        if name:
            if name in _circuit_breakers:
                return _circuit_breakers[name].get_stats()
            return {}

        return {
            name: cb.get_stats()
            for name, cb in _circuit_breakers.items()
        }


# ============================================================================
# CONVENIENCE DECORATORS
# ============================================================================

def with_circuit_breaker(
    name: str,
    fail_max: int = 5,
    timeout_duration: int = 60,
    fallback: Optional[Callable] = None
):
    """
    Decorator to easily add circuit breaker protection to a function.

    Args:
        name: Circuit breaker name
        fail_max: Max failures before opening
        timeout_duration: Seconds before retry
        fallback: Optional fallback function

    Example:
        >>> @with_circuit_breaker("external_api", fail_max=3)
        ... def call_api():
        ...     return requests.get("https://api.example.com")
    """
    config = CircuitBreakerConfig(
        name=name,
        fail_max=fail_max,
        timeout_duration=timeout_duration,
        fallback_function=fallback,
    )
    cb = create_circuit_breaker(config)

    def decorator(func: Callable) -> Callable:
        return cb.protect(func)

    return decorator
