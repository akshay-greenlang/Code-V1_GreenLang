"""
GreenLang Testing Compatibility Module
=====================================

This module provides compatibility shims for testing utilities that might be moved
or reorganized in future versions. It ensures tests continue to work during transitions.
"""

import warnings


def _warn_deprecated_test_import(old_path: str, new_path: str) -> None:
    """Issue a deprecation warning for test imports."""
    warnings.warn(
        f"Importing from '{old_path}' is deprecated. "
        f"Use '{new_path}' instead. "
        f"This import path may be removed in v2.0.",
        DeprecationWarning,
        stacklevel=3,
    )


# Test utilities compatibility
try:
    # Try importing from new location first
    from tests.utils.numerics import assert_close, assert_within_tolerance

    _warn_deprecated_test_import("greenlang.compat.testing", "tests.utils.numerics")
except ImportError:
    try:
        # Fall back to old location
        from greenlang.testing.numerics import assert_close, assert_within_tolerance

        _warn_deprecated_test_import(
            "greenlang.compat.testing", "greenlang.testing.numerics"
        )
    except ImportError:
        # Provide basic implementations if nothing is available
        def assert_close(
            actual: float, expected: float, rel_tol: float = 1e-09, abs_tol: float = 0.0
        ) -> None:
            """Basic implementation of assert_close for compatibility."""
            import math

            if not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
                raise AssertionError(f"Values not close: {actual} != {expected}")

        def assert_within_tolerance(
            actual: float, expected: float, tolerance: float = 0.01
        ) -> None:
            """Basic implementation of assert_within_tolerance for compatibility."""
            if abs(actual - expected) > tolerance:
                raise AssertionError(
                    f"Value {actual} not within tolerance {tolerance} of {expected}"
                )


# Agent testing compatibility
class MockAgent:
    """
    Mock agent for testing purposes.
    Provides basic compatibility for tests that need agent-like objects.
    """

    def __init__(self, name: str = "mock_agent", **kwargs):
        self.name = name
        self.kwargs = kwargs
        self._results = {}

    def run(self, inputs: dict) -> dict:
        """Mock run method that returns configured results."""
        return self._results.get("default", {"status": "success", "data": inputs})

    def set_result(self, key: str, result: dict) -> None:
        """Set a mock result for testing."""
        self._results[key] = result


# Test data utilities
def create_test_data(data_type: str = "fuel", **kwargs) -> dict:
    """Create test data for various agent types."""
    _warn_deprecated_test_import(
        "greenlang.compat.testing.create_test_data", "tests.fixtures.data_factory"
    )

    if data_type == "fuel":
        return {
            "fuel_emissions": [
                {"fuel_type": "electricity", "amount": 1000, "unit": "kWh"},
                {"fuel_type": "natural_gas", "amount": 500, "unit": "therms"},
            ]
        }
    elif data_type == "carbon":
        return {
            "emissions": [
                {"source": "electricity", "co2_kg": 450.5},
                {"source": "natural_gas", "co2_kg": 123.2},
            ]
        }
    else:
        return {"data_type": data_type, **kwargs}


# Test fixtures compatibility
class TestFixtures:
    """
    Compatibility layer for test fixtures that might be moved.
    """

    @staticmethod
    def get_sample_pipeline() -> dict:
        """Get a sample pipeline for testing."""
        _warn_deprecated_test_import(
            "greenlang.compat.testing.TestFixtures", "tests.fixtures.pipelines"
        )
        return {
            "name": "test_pipeline",
            "version": "1.0",
            "steps": [
                {
                    "name": "step1",
                    "agent": "FuelAgent",
                    "inputs": {"fuel_data": "test_data"},
                }
            ],
        }

    @staticmethod
    def get_sample_manifest() -> dict:
        """Get a sample pack manifest for testing."""
        _warn_deprecated_test_import(
            "greenlang.compat.testing.TestFixtures", "tests.fixtures.manifests"
        )
        return {
            "name": "test-pack",
            "version": "1.0.0",
            "description": "Test pack for compatibility",
            "agents": [],
            "dependencies": [],
        }


__all__ = [
    "assert_close",
    "assert_within_tolerance",
    "MockAgent",
    "create_test_data",
    "TestFixtures",
]
