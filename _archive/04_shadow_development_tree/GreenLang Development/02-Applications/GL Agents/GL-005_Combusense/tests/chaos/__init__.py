"""
GL-005 Combusense - Chaos Engineering Test Suite

This module provides comprehensive chaos engineering tests for validating
the resilience and fault tolerance of the Combusense Emissions Analytics Agent.

Combusense-specific chaos scenarios:
- CEMS (Continuous Emission Monitoring System) failures
- Regulatory reporting disruptions
- Data quality degradation
- Correlation engine failures
- Predictive model failures

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

from .combusense_chaos import (
    CombusenseChaosConfig,
    CEMSFaultInjector,
    RegulatoryReportingFaultInjector,
    DataQualityFaultInjector,
    CorrelationEngineFaultInjector,
    PredictiveModelFaultInjector,
    create_combusense_hypothesis,
)

__all__ = [
    "ChaosRunner",
    "ChaosExperiment",
    "ChaosResult",
    "ChaosSeverity",
    "CombusenseChaosConfig",
    "CEMSFaultInjector",
    "RegulatoryReportingFaultInjector",
    "DataQualityFaultInjector",
    "CorrelationEngineFaultInjector",
    "PredictiveModelFaultInjector",
    "create_combusense_hypothesis",
]
