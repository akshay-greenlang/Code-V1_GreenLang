"""
Chaos Engineering Tests

This package contains comprehensive chaos testing suites for validating
the resilience and fault-tolerance of GreenLang agents under various
failure scenarios.

Modules:
- chaos_tests: Core chaos test runner and test scenarios
- conftest: Pytest configuration and fixtures

Test Categories:
- Agent Failover: Pod failure, backup takeover, cascading failures
- Database Resilience: Connection loss, retry logic, cache fallback
- High Latency: Timeout handling, request queuing
- Resource Pressure: Memory exhaustion, CPU saturation
"""

from .chaos_tests import (
    ChaosTestRunner,
    ChaosType,
    ChaosEvent,
    SteadyStateMetrics,
    ChaosTestResult,
)

__all__ = [
    "ChaosTestRunner",
    "ChaosType",
    "ChaosEvent",
    "SteadyStateMetrics",
    "ChaosTestResult",
]
