"""
GL-003 UnifiedSteam - Chaos Engineering Test Suite

This module provides comprehensive chaos engineering tests for validating
the resilience and fault tolerance of the UnifiedSteam Header Balance Agent.

UnifiedSteam-specific chaos scenarios:
- Steam header pressure imbalance
- Multi-boiler coordination failures
- Load balancing disruptions
- Demand forecasting errors
- Valve actuator failures

All tests are safe for CI/CD execution (no actual infrastructure damage).

Author: GreenLang Chaos Engineering Team
Version: 1.0.0
"""

import sys
import os

# Add GL-001 chaos framework to path
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

from .unifiedsteam_chaos import (
    UnifiedSteamChaosConfig,
    SteamHeaderFaultInjector,
    BoilerCoordinationFaultInjector,
    LoadBalancingFaultInjector,
    DemandForecastFaultInjector,
    ValveActuatorFaultInjector,
    create_unifiedsteam_hypothesis,
)

__all__ = [
    "ChaosRunner",
    "ChaosExperiment",
    "ChaosResult",
    "ChaosSeverity",
    "UnifiedSteamChaosConfig",
    "SteamHeaderFaultInjector",
    "BoilerCoordinationFaultInjector",
    "LoadBalancingFaultInjector",
    "DemandForecastFaultInjector",
    "ValveActuatorFaultInjector",
    "create_unifiedsteam_hypothesis",
]
