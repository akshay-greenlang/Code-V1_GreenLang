"""
GL-002 Flameguard - Chaos Engineering Test Suite

This module provides comprehensive chaos engineering tests for validating
the resilience and fault tolerance of the Flameguard Boiler Efficiency Agent.

Flameguard-specific chaos scenarios:
- Sensor data stream interruption
- Efficiency calculation failures
- CMMS integration failures
- OPC-UA communication faults
- Model inference failures

All tests are safe for CI/CD execution (no actual infrastructure damage).

Author: GreenLang Chaos Engineering Team
Version: 1.0.0
"""

# Re-export from shared chaos framework
import sys
import os

# Add parent path to allow importing from GL-001's chaos framework
gl001_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "GL-001_Thermalcommand", "tests", "chaos"))
if gl001_path not in sys.path:
    sys.path.insert(0, gl001_path)

from chaos_runner import ChaosRunner, ChaosExperiment, ChaosResult, ChaosSeverity
from fault_injectors import (
    NetworkFaultInjector,
    ResourceFaultInjector,
    ServiceFaultInjector,
    StateFaultInjector,
    get_fault_injector,
)
from steady_state import (
    SteadyStateHypothesis,
    SteadyStateMetric,
    SteadyStateValidator,
    ComparisonOperator,
)
from resilience_patterns import (
    CircuitBreakerTest,
    RetryMechanismTest,
    FallbackBehaviorTest,
    GracefulDegradationTest,
)
from kubernetes_chaos import (
    K8sPodDeletionTest,
    K8sNodeFailureTest,
    K8sResourceExhaustionTest,
)

# Agent-specific exports
from .flameguard_chaos import (
    FlameguardChaosConfig,
    SensorStreamFaultInjector,
    EfficiencyCalculationFaultInjector,
    create_flameguard_hypothesis,
)

__all__ = [
    "ChaosRunner",
    "ChaosExperiment",
    "ChaosResult",
    "ChaosSeverity",
    "NetworkFaultInjector",
    "ResourceFaultInjector",
    "ServiceFaultInjector",
    "StateFaultInjector",
    "SteadyStateHypothesis",
    "SteadyStateMetric",
    "SteadyStateValidator",
    "FlameguardChaosConfig",
    "SensorStreamFaultInjector",
    "EfficiencyCalculationFaultInjector",
    "create_flameguard_hypothesis",
]
