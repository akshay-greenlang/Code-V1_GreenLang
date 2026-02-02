"""
GL-009 ThermalIQ Safety Module

Provides resilience patterns for external system integration including
circuit breakers, rate limiters, and safety validators.

Components:
- CircuitBreaker: IEC 61511 compliant circuit breaker pattern
"""

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    FailureType,
    RecoveryStrategy,
    CallResult,
    CircuitBreakerState,
    CircuitBreakerMetrics,
)

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "FailureType",
    "RecoveryStrategy",
    "CallResult",
    "CircuitBreakerState",
    "CircuitBreakerMetrics",
]
