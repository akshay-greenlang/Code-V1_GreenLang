# -*- coding: utf-8 -*-
"""
GL-010 EMISSIONWATCH Test Suite

Comprehensive test suite for EmissionsComplianceAgent providing:
- 200+ unit tests for all calculators and components
- 50+ integration tests for CEMS and reporting pipelines
- 15+ determinism tests for zero-hallucination verification
- 10+ performance tests for latency and throughput benchmarks

Coverage Target: 90%+

Test Structure:
    tests/
    ├── unit/           # Unit tests for individual components
    │   ├── test_nox_calculator.py
    │   ├── test_sox_calculator.py
    │   ├── test_co2_calculator.py
    │   ├── test_particulate_calculator.py
    │   ├── test_emission_factors.py
    │   ├── test_compliance_checker.py
    │   ├── test_violation_detector.py
    │   ├── test_orchestrator.py
    │   └── test_tools.py
    ├── integration/    # Integration tests
    │   ├── test_cems_integration.py
    │   ├── test_regulatory_reporting.py
    │   └── test_full_pipeline.py
    ├── e2e/            # End-to-end workflow tests
    │   └── test_complete_workflow.py
    ├── determinism/    # Reproducibility tests
    │   └── test_reproducibility.py
    ├── performance/    # Performance benchmarks
    │   └── test_benchmarks.py
    └── fixtures/       # Test data files
        ├── emissions_test_cases.json
        └── regulatory_limits.json

Usage:
    # Run all tests
    pytest

    # Run with coverage
    pytest --cov=.. --cov-report=html

    # Run specific test category
    pytest -m unit
    pytest -m integration
    pytest -m determinism
    pytest -m performance

    # Run specific pollutant tests
    pytest -m nox
    pytest -m sox

    # Run fast tests only
    pytest -m "not slow"

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "GreenLang Foundation Test Engineering"

# Test categories
UNIT_TESTS = [
    "test_nox_calculator",
    "test_sox_calculator",
    "test_co2_calculator",
    "test_particulate_calculator",
    "test_emission_factors",
    "test_compliance_checker",
    "test_violation_detector",
    "test_orchestrator",
    "test_tools",
]

INTEGRATION_TESTS = [
    "test_cems_integration",
    "test_regulatory_reporting",
    "test_full_pipeline",
]

E2E_TESTS = [
    "test_complete_workflow",
]

DETERMINISM_TESTS = [
    "test_reproducibility",
]

PERFORMANCE_TESTS = [
    "test_benchmarks",
]

# All test modules
ALL_TESTS = UNIT_TESTS + INTEGRATION_TESTS + E2E_TESTS + DETERMINISM_TESTS + PERFORMANCE_TESTS
