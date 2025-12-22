"""
GL-001 ThermalCommand - Chaos Engineering Test Suite

This module provides comprehensive chaos engineering tests for validating
the resilience and fault tolerance of the ThermalCommand Orchestrator.

Chaos testing categories:
- Network fault injection (partition, latency, packet loss)
- Resource exhaustion (CPU, memory, disk)
- External service failures
- State corruption scenarios
- Cascading failure tests

All tests are safe for CI/CD execution (no actual infrastructure damage).

Author: GreenLang Chaos Engineering Team
Version: 1.0.0
"""

from .chaos_runner import ChaosRunner, ChaosExperiment, ChaosResult
from .fault_injectors import (
    NetworkFaultInjector,
    ResourceFaultInjector,
    ServiceFaultInjector,
    StateFaultInjector,
)
from .steady_state import (
    SteadyStateHypothesis,
    SteadyStateMetric,
    SteadyStateValidator,
)
from .resilience_patterns import (
    CircuitBreakerTest,
    RetryMechanismTest,
    FallbackBehaviorTest,
    GracefulDegradationTest,
)

__all__ = [
    "ChaosRunner",
    "ChaosExperiment",
    "ChaosResult",
    "NetworkFaultInjector",
    "ResourceFaultInjector",
    "ServiceFaultInjector",
    "StateFaultInjector",
    "SteadyStateHypothesis",
    "SteadyStateMetric",
    "SteadyStateValidator",
    "CircuitBreakerTest",
    "RetryMechanismTest",
    "FallbackBehaviorTest",
    "GracefulDegradationTest",
]
