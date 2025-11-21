# -*- coding: utf-8 -*-
"""
GL-001 ProcessHeatOrchestrator Test Suite
Comprehensive testing for industrial process heat optimization agent.

Test Coverage Targets:
- Unit Tests: 95%
- Integration Tests: 85%
- Performance Tests: 100% of targets met
- Security Tests: 100% secure
- Compliance: 12/12 dimensions
"""

from .test_process_heat_orchestrator import TestProcessHeatOrchestrator
from .test_tools import TestProcessHeatTools
from .test_calculators import TestThermalCalculators
from .test_integrations import TestIntegrations
from .test_performance import TestPerformance
from .test_security import TestSecurity
from .test_determinism import TestDeterminism
from .test_compliance import TestCompliance

__all__ = [
    'TestProcessHeatOrchestrator',
    'TestProcessHeatTools',
    'TestThermalCalculators',
    'TestIntegrations',
    'TestPerformance',
    'TestSecurity',
    'TestDeterminism',
    'TestCompliance'
]

# Test suite metadata
TEST_SUITE_VERSION = "1.0.0"
COVERAGE_TARGET = 0.85
PERFORMANCE_TARGETS = {
    'agent_creation_ms': 100,
    'calculation_ms': 2000,
    'dashboard_ms': 5000,
    'concurrent_agents': 10000
}