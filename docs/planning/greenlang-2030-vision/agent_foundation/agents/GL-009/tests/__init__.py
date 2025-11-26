"""Test package for GL-009 THERMALIQ ThermalEfficiencyCalculator.

This test suite provides comprehensive coverage (90%+) for the
ThermalEfficiencyCalculator agent, including:

- Unit tests for all calculators and components
- Integration tests for connectors and APIs
- End-to-end workflow tests
- Determinism and reproducibility tests
- Performance and compliance tests

Test Structure:
    tests/
    ├── unit/               # Unit tests (isolated component testing)
    ├── integration/        # Integration tests (connector/API testing)
    ├── e2e/               # End-to-end workflow tests
    ├── determinism/       # Reproducibility and zero-hallucination tests
    └── fixtures/          # Test data fixtures

Running Tests:
    # All tests
    pytest tests/

    # Unit tests only
    pytest tests/unit/

    # With coverage report
    pytest tests/ --cov=. --cov-report=html

    # Specific test file
    pytest tests/unit/test_first_law_efficiency.py -v

    # With markers
    pytest tests/ -m "not slow"

Author: GL-TestEngineer
Version: 1.0.0
Target Coverage: 90%+
"""

__version__ = "1.0.0"
__author__ = "GL-TestEngineer"
