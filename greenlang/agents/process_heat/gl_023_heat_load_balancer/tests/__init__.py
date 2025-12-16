"""
GL-023 HeatLoadBalancer Test Suite
==================================

Comprehensive test suite for the Heat Load Balancer Agent.
Target: 85%+ code coverage with unit, integration, and optimization tests.

Test Markers
------------
The following pytest markers are available for selective test execution:

@pytest.mark.unit
    Unit tests for individual functions and methods.
    Fast execution, no external dependencies.
    Run with: pytest -m unit

@pytest.mark.integration
    Integration tests for component interactions.
    Tests database, API, and external system integrations.
    Run with: pytest -m integration

@pytest.mark.optimization
    Optimization algorithm tests (MILP, heuristics).
    May have longer execution times.
    Run with: pytest -m optimization

@pytest.mark.safety
    Safety-critical tests for equipment protection.
    Tests N+1 redundancy, ramp rates, emergency reserves.
    Run with: pytest -m safety

@pytest.mark.performance
    Performance and benchmark tests.
    Tests throughput, latency, and resource usage.
    Run with: pytest -m performance

@pytest.mark.determinism
    Determinism verification tests.
    Ensures same input produces same output.
    Run with: pytest -m determinism

@pytest.mark.slow
    Slow tests (>1s execution time).
    Excluded by default in CI fast runs.
    Run with: pytest -m slow

@pytest.mark.critical
    Critical tests that must pass for deployment.
    Core functionality and safety validations.
    Run with: pytest -m critical

Test Structure
--------------
tests/
    __init__.py          - This file (markers documentation)
    conftest.py          - Shared fixtures and test configuration
    test_config.py       - Configuration validation tests
    test_calculators.py  - Calculator function tests
    test_optimizer.py    - Optimization algorithm tests
    test_balancer.py     - Main HeatLoadBalancer agent tests
    test_safety.py       - Safety validation tests

Running Tests
-------------
Full test suite:
    pytest greenlang/agents/process_heat/gl_023_heat_load_balancer/tests/

With coverage:
    pytest --cov=greenlang.agents.process_heat.gl_023_heat_load_balancer --cov-report=term-missing

Unit tests only:
    pytest -m unit

Skip slow tests:
    pytest -m "not slow"

Critical tests only:
    pytest -m critical

Coverage Requirements
---------------------
- Overall: 85%+ line coverage
- Calculators: 95%+ line coverage (zero-hallucination requirement)
- Safety: 100% branch coverage

Author: GL-TestEngineer
Version: 1.0.0
Last Updated: 2025-01-01
"""

__version__ = "1.0.0"
__author__ = "GL-TestEngineer"

# Test categories for reporting
TEST_CATEGORIES = {
    "unit": "Unit tests for individual functions",
    "integration": "Integration tests for component interactions",
    "optimization": "Optimization algorithm tests",
    "safety": "Safety-critical equipment protection tests",
    "performance": "Performance and benchmark tests",
    "determinism": "Determinism verification tests",
}

# Coverage targets by module
COVERAGE_TARGETS = {
    "formulas": 95.0,  # Zero-hallucination calculators
    "models": 90.0,    # Data models
    "agent": 85.0,     # Main agent logic
    "safety": 100.0,   # Safety-critical code
}
