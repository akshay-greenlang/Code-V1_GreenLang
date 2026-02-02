# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN - Test Suite

Comprehensive test suite for the Insulation Scanning & Thermal Assessment Agent.

Test Categories:
- unit/: Unit tests for individual components (85%+ coverage target)
- integration/: End-to-end pipeline tests
- property/: Property-based tests using Hypothesis-style assertions
- chaos/: Chaos engineering resilience tests
- golden/: Golden master tests for determinism verification

Running Tests:
    # Run all tests
    pytest tests/

    # Run specific categories
    pytest tests/ -m unit
    pytest tests/ -m integration
    pytest tests/ -m property
    pytest tests/ -m chaos
    pytest tests/ -m golden

    # Run with coverage
    pytest tests/ --cov=insulscan --cov-report=html

    # Run performance tests
    pytest tests/ -m performance

CI/CD Integration:
    See pytest.ini for marker definitions and configuration.

Author: GL-TestEngineer
Version: 1.0.0
Target Coverage: 85%+
"""

__version__ = "1.0.0"
__author__ = "GL-TestEngineer"

__all__ = [
    "unit",
    "integration",
    "property",
    "chaos",
    "golden",
    "api",
]
