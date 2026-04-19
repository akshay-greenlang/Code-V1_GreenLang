"""
Golden Tests Configuration
==========================

Pytest fixtures and configuration for golden test suite.
"""

import json
import pytest
from pathlib import Path
from typing import Any, Dict, List, Tuple


# =============================================================================
# Constants
# =============================================================================

GOLDEN_TESTS_DIR = Path(__file__).parent
DEFAULT_TOLERANCE = 0.01  # 1% relative tolerance


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def golden_tests_dir():
    """Return the golden tests directory."""
    return GOLDEN_TESTS_DIR


@pytest.fixture
def tolerance():
    """Return default tolerance for numeric comparisons."""
    return DEFAULT_TOLERANCE


# =============================================================================
# Helper Functions
# =============================================================================

def load_golden_tests(json_file: Path) -> List[Dict[str, Any]]:
    """Load golden tests from a JSON file."""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("tests", [])


def compare_with_tolerance(actual: float, expected: float, tolerance: float = DEFAULT_TOLERANCE) -> Tuple[bool, float]:
    """
    Compare two values with relative tolerance.

    Returns:
        Tuple of (is_within_tolerance, relative_error)
    """
    if expected == 0:
        if actual == 0:
            return True, 0.0
        return abs(actual) < tolerance, abs(actual)

    relative_error = abs(actual - expected) / abs(expected)
    return relative_error <= tolerance, relative_error


# =============================================================================
# Pytest Hooks
# =============================================================================

def pytest_collection_modifyitems(config, items):
    """Add markers to golden tests for filtering."""
    for item in items:
        if "golden" in str(item.fspath):
            item.add_marker(pytest.mark.golden)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "golden: marks tests as golden tests (deselect with '-m \"not golden\"')"
    )
    config.addinivalue_line(
        "markers", "eudr: marks tests as EUDR compliance tests"
    )
    config.addinivalue_line(
        "markers", "cbam: marks tests as CBAM benchmark tests"
    )
    config.addinivalue_line(
        "markers", "fuel: marks tests as fuel emissions tests"
    )
    config.addinivalue_line(
        "markers", "building: marks tests as building energy tests"
    )


# =============================================================================
# Assertion Helpers
# =============================================================================

class GoldenAssert:
    """Assertion helpers for golden test validation."""

    @staticmethod
    def assert_within_tolerance(actual: float, expected: float, tolerance: float = DEFAULT_TOLERANCE, msg: str = ""):
        """Assert that actual value is within tolerance of expected."""
        is_match, relative_error = compare_with_tolerance(actual, expected, tolerance)
        if not is_match:
            raise AssertionError(
                f"{msg}Value {actual} is not within {tolerance*100:.1f}% of expected {expected}. "
                f"Relative error: {relative_error*100:.2f}%"
            )

    @staticmethod
    def assert_hash_valid(hash_value: str, expected_length: int = 64):
        """Assert that a hash value is valid SHA-256 format."""
        assert isinstance(hash_value, str), f"Hash must be string, got {type(hash_value)}"
        assert len(hash_value) == expected_length, f"Hash length must be {expected_length}, got {len(hash_value)}"
        try:
            int(hash_value, 16)
        except ValueError:
            raise AssertionError("Hash must be hexadecimal")

    @staticmethod
    def assert_output_matches(actual: Dict, expected: Dict, tolerance: float = DEFAULT_TOLERANCE):
        """Assert that actual output matches expected output with tolerance."""
        for key, expected_value in expected.items():
            assert key in actual, f"Missing key in output: {key}"
            actual_value = actual[key]

            if isinstance(expected_value, (int, float)):
                GoldenAssert.assert_within_tolerance(
                    actual_value, expected_value, tolerance, f"Key '{key}': "
                )
            elif isinstance(expected_value, dict):
                GoldenAssert.assert_output_matches(actual_value, expected_value, tolerance)
            elif isinstance(expected_value, list):
                assert len(actual_value) == len(expected_value), f"List length mismatch for {key}"
                for i, (a, e) in enumerate(zip(actual_value, expected_value)):
                    if isinstance(e, (int, float)):
                        GoldenAssert.assert_within_tolerance(a, e, tolerance, f"Key '{key}[{i}]': ")
                    else:
                        assert a == e, f"Value mismatch at {key}[{i}]: {a} != {e}"
            else:
                assert actual_value == expected_value, f"Value mismatch for {key}: {actual_value} != {expected_value}"


@pytest.fixture
def golden_assert():
    """Return GoldenAssert helper class."""
    return GoldenAssert
