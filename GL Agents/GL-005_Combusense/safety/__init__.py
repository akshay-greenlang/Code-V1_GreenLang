# -*- coding: utf-8 -*-
"""
GL-005 COMBUSENSE Safety Module

This package provides safety-critical components for combustion control:
- Circuit breaker for external system integration
- Safety envelope validation
- IEC 61511 compliant safety functions

Author: GL-BackendDeveloper
"""

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CallResult,
    CircuitBreakerRegistry,
    circuit_breaker_registry
)

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "CallResult",
    "CircuitBreakerRegistry",
    "circuit_breaker_registry"
]
