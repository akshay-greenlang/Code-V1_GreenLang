# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER Safety Module

Provides safety patterns for robust trap monitoring including circuit breakers,
fail-safe mechanisms, and graceful degradation strategies.

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
    FailureRecord,
    FailureType,
    RecoveryStrategy,
    SafetyIntegrityLevel,
    StateTransition,
    create_trap_sensor_breaker,
)

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerMetrics",
    "CircuitBreakerRegistry",
    "CircuitOpenError",
    "CircuitState",
    "FailureRecord",
    "FailureType",
    "RecoveryStrategy",
    "SafetyIntegrityLevel",
    "StateTransition",
    "create_trap_sensor_breaker",
]
