# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Safety Module

Provides resilience patterns for external integrations including:
- Circuit breaker pattern for CEMS, EPA, and market data APIs
- Enhanced circuit breaker with critical path protection
- Graceful degradation for non-critical paths
- Recovery automation with configurable strategies
- EPA substitute data support for CEMS failures

Author: GreenLang GL-010 EmissionsGuardian
Version: 1.0.0
"""

from .circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState,
    CircuitState, FailureType, FailureRecord,
    CircuitBreakerError, CircuitOpenError,
    IntegrationCircuitBreakers,
    get_cems_circuit_breaker, get_epa_circuit_breaker,
    get_market_data_circuit_breaker, circuit_protected,
)

from .enhanced_circuit_breaker import (
    # Enums
    PathCriticality,
    DegradationMode,
    RecoveryStrategy,
    HealthStatus,
    CircuitState as EnhancedCircuitState,
    # Configs
    EnhancedCircuitConfig,
    CriticalPathConfig,
    # Data Classes
    FailureEvent,
    DegradedResult,
    HealthMetrics,
    # Exceptions
    CircuitBreakerError as EnhancedCircuitBreakerError,
    CircuitOpenError as EnhancedCircuitOpenError,
    CriticalPathFailureError,
    # Main Class
    EnhancedCircuitBreaker,
    # Factory Functions
    create_critical_emission_breaker,
    create_compliance_breaker,
    create_market_data_breaker,
)

__all__ = [
    # From circuit_breaker
    "CircuitBreaker", "CircuitBreakerConfig", "CircuitBreakerState",
    "CircuitState", "FailureType", "FailureRecord",
    "CircuitBreakerError", "CircuitOpenError", "IntegrationCircuitBreakers",
    "get_cems_circuit_breaker", "get_epa_circuit_breaker",
    "get_market_data_circuit_breaker", "circuit_protected",
    # From enhanced_circuit_breaker
    "PathCriticality", "DegradationMode", "RecoveryStrategy", "HealthStatus",
    "EnhancedCircuitState", "EnhancedCircuitConfig", "CriticalPathConfig",
    "FailureEvent", "DegradedResult", "HealthMetrics",
    "EnhancedCircuitBreakerError", "EnhancedCircuitOpenError", "CriticalPathFailureError",
    "EnhancedCircuitBreaker", "create_critical_emission_breaker",
    "create_compliance_breaker", "create_market_data_breaker",
]

__version__ = "1.0.0"
__author__ = "GreenLang GL-010 EmissionsGuardian"
