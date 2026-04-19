"""
GreenLang Golden Tests Framework

This module provides comprehensive golden test infrastructure for validating
zero-hallucination deterministic calculations across all GreenLang agents.

Golden tests ensure:
- Bit-perfect reproducibility of calculations
- Zero-hallucination determinism (same input = same output)
- Regulatory compliance with known reference values
- Provenance hash verification for audit trails

Usage:
    from tests.golden import GoldenTestRunner, GoldenTestLoader

    loader = GoldenTestLoader()
    tests = loader.load_tests_from_directory('tests/golden/carbon_emissions')

    runner = GoldenTestRunner()
    results = runner.run_tests(tests)
    assert results.all_passed
"""

from tests.golden.framework import (
    GoldenTestRunner,
    GoldenTestResult,
    GoldenTestSuite,
    ComparisonResult,
    ToleranceConfig,
)
from tests.golden.loader import (
    GoldenTestLoader,
    GoldenTestCase,
    ExpectedOutput,
)

__all__ = [
    "GoldenTestRunner",
    "GoldenTestResult",
    "GoldenTestSuite",
    "ComparisonResult",
    "ToleranceConfig",
    "GoldenTestLoader",
    "GoldenTestCase",
    "ExpectedOutput",
]

__version__ = "1.0.0"
