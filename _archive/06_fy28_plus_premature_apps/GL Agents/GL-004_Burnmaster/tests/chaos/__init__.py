"""
GL-004 Burnmaster - Chaos Engineering Test Suite

This module provides comprehensive chaos engineering tests for validating
the resilience and fault tolerance of the Burnmaster Combustion Optimizer Agent.

Burnmaster-specific chaos scenarios:
- Combustion air/fuel ratio imbalance
- Burner management system failures
- Flame detection failures
- Emission monitoring disruptions
- Oxygen trim control failures

All tests are safe for CI/CD execution (no actual infrastructure damage).

Author: GreenLang Chaos Engineering Team
Version: 1.0.0
"""

import sys
import os

gl001_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "GL-001_Thermalcommand", "tests", "chaos"))
if gl001_path not in sys.path:
    sys.path.insert(0, gl001_path)

from chaos_runner import ChaosRunner, ChaosExperiment, ChaosResult, ChaosSeverity
from fault_injectors import (
    NetworkFaultInjector,
    ResourceFaultInjector,
    ServiceFaultInjector,
    StateFaultInjector,
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

from .burnmaster_chaos import (
    BurnmasterChaosConfig,
    CombustionRatioFaultInjector,
    BurnerManagementFaultInjector,
    FlameDetectionFaultInjector,
    EmissionMonitoringFaultInjector,
    OxygenTrimFaultInjector,
    create_burnmaster_hypothesis,
)

__all__ = [
    "ChaosRunner",
    "ChaosExperiment",
    "ChaosResult",
    "ChaosSeverity",
    "BurnmasterChaosConfig",
    "CombustionRatioFaultInjector",
    "BurnerManagementFaultInjector",
    "FlameDetectionFaultInjector",
    "EmissionMonitoringFaultInjector",
    "OxygenTrimFaultInjector",
    "create_burnmaster_hypothesis",
]
