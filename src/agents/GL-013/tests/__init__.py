# -*- coding: utf-8 -*-
"""
GL-013 PREDICTMAINT - Test Suite Package
Comprehensive test coverage for predictive maintenance agent.

Test Categories:
- unit/: Unit tests for individual components (85%+ coverage target)
- integration/: Integration tests with external systems
- e2e/: End-to-end workflow tests
- determinism/: Determinism and reproducibility verification
- performance/: Performance benchmarks and load tests
- security/: Security and compliance tests

Run Tests:
    # Run all tests
    pytest

    # Run specific category
    pytest -m unit
    pytest -m performance
    pytest -m security

    # Run with coverage
    pytest --cov=calculators --cov-report=html

    # Run with verbosity
    pytest -v --tb=long

Author: GL-TestEngineer
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "GL-TestEngineer"
