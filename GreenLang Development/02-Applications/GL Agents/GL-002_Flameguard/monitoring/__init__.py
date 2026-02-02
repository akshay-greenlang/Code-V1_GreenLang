"""
GL-002 FLAMEGUARD - Monitoring Module

Prometheus metrics, health checks, alerting, and circuit breaker health monitoring.
"""

from .metrics import (
    MetricsCollector,
    BoilerMetrics,
    CombustionMetrics,
    EfficiencyMetrics,
    SafetyMetrics,
)
from .health import (
    HealthChecker,
    HealthStatus,
    ComponentHealth,
)
from .alerting import (
    AlertManager,
    Alert,
    AlertSeverity,
    AlertRule,
)
from .circuit_breaker_health import (
    CircuitBreakerHealthMonitor,
    CircuitBreakerHealthStatus,
    CircuitBreakerHealthConfig,
    CircuitBreakerAlert,
    register_circuit_breaker_health_checks,
)

__all__ = [
    "MetricsCollector",
    "BoilerMetrics",
    "CombustionMetrics",
    "EfficiencyMetrics",
    "SafetyMetrics",
    "HealthChecker",
    "HealthStatus",
    "ComponentHealth",
    "AlertManager",
    "Alert",
    "AlertSeverity",
    "AlertRule",
    # Circuit Breaker Health Monitoring
    "CircuitBreakerHealthMonitor",
    "CircuitBreakerHealthStatus",
    "CircuitBreakerHealthConfig",
    "CircuitBreakerAlert",
    "register_circuit_breaker_health_checks",
]
