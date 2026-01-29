"""
Pytest fixtures and configuration for Golden Test Suite.

This module provides shared fixtures, test data loaders, and configuration
for the GL-FOUND-X-003 golden file tests.

Features:
    - Automatic golden file discovery
    - YAML test case loading
    - Tolerance-based comparisons
    - Pint cross-validation support
    - Detailed diff reporting
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from decimal import Decimal
import math

import pytest
import yaml

# Add the package to the path for testing
PACKAGE_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PACKAGE_ROOT / "packages" / "gl-normalizer-core" / "src"))

try:
    from gl_normalizer_core.converter import UnitConverter, ConversionResult
    from gl_normalizer_core.parser import UnitParser, Quantity
    from gl_normalizer_core.resolver import ReferenceResolver, VocabEntry
    GL_NORMALIZER_AVAILABLE = True
except ImportError:
    GL_NORMALIZER_AVAILABLE = False

try:
    import pint
    PINT_AVAILABLE = True
    UREG = pint.UnitRegistry()
except ImportError:
    PINT_AVAILABLE = False
    UREG = None


# -----------------------------------------------------------------------------
# Path Configuration
# -----------------------------------------------------------------------------

GOLDEN_FILES_DIR = Path(__file__).parent / "golden_files"
UNIT_CONVERSIONS_DIR = GOLDEN_FILES_DIR / "unit_conversions"
ENTITY_RESOLUTION_DIR = GOLDEN_FILES_DIR / "entity_resolution"
FULL_PIPELINE_DIR = GOLDEN_FILES_DIR / "full_pipeline"


# -----------------------------------------------------------------------------
# YAML Loaders
# -----------------------------------------------------------------------------

def load_yaml_file(filepath: Path) -> Dict[str, Any]:
    """Load and parse a YAML file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_golden_files(directory: Path, pattern: str = "*.yaml") -> List[Path]:
    """Discover all golden files in a directory."""
    if not directory.exists():
        return []
    return sorted(directory.glob(pattern))


def load_test_cases(filepath: Path) -> List[Dict[str, Any]]:
    """Load test cases from a golden file."""
    data = load_yaml_file(filepath)
    return data.get("test_cases", [])


def load_all_unit_conversion_tests() -> List[Tuple[str, Dict[str, Any]]]:
    """Load all unit conversion test cases with their dimension category."""
    test_cases = []
    for golden_file in discover_golden_files(UNIT_CONVERSIONS_DIR):
        dimension = golden_file.stem  # e.g., "energy", "mass"
        for test_case in load_test_cases(golden_file):
            test_cases.append((dimension, test_case))
    return test_cases


def load_all_entity_resolution_tests() -> List[Tuple[str, Dict[str, Any]]]:
    """Load all entity resolution test cases with their entity type."""
    test_cases = []
    for golden_file in discover_golden_files(ENTITY_RESOLUTION_DIR):
        entity_type = golden_file.stem  # e.g., "fuels", "materials"
        for test_case in load_test_cases(golden_file):
            test_cases.append((entity_type, test_case))
    return test_cases


def load_all_pipeline_tests() -> List[Tuple[str, Dict[str, Any]]]:
    """Load all full pipeline test cases with their scenario type."""
    test_cases = []
    for golden_file in discover_golden_files(FULL_PIPELINE_DIR):
        scenario = golden_file.stem  # e.g., "ghg_protocol_scenarios"
        for test_case in load_test_cases(golden_file):
            test_cases.append((scenario, test_case))
    return test_cases


# -----------------------------------------------------------------------------
# Comparison Utilities
# -----------------------------------------------------------------------------

class GoldenTestResult:
    """Result of a golden test comparison."""

    def __init__(
        self,
        passed: bool,
        expected: Any,
        actual: Any,
        tolerance: Optional[float] = None,
        diff: Optional[str] = None,
    ):
        self.passed = passed
        self.expected = expected
        self.actual = actual
        self.tolerance = tolerance
        self.diff = diff

    def __bool__(self) -> bool:
        return self.passed

    def failure_message(self) -> str:
        """Generate detailed failure message."""
        msg = f"Expected: {self.expected}\n"
        msg += f"Actual:   {self.actual}\n"
        if self.tolerance is not None:
            msg += f"Tolerance: {self.tolerance}\n"
        if self.diff:
            msg += f"Diff: {self.diff}\n"
        return msg


def compare_values(
    expected: float,
    actual: float,
    tolerance: Optional[float] = None,
    relative: bool = True,
) -> GoldenTestResult:
    """
    Compare two numeric values with tolerance.

    Args:
        expected: Expected value
        actual: Actual value
        tolerance: Tolerance for comparison (default 1e-9)
        relative: Use relative tolerance (default True)

    Returns:
        GoldenTestResult with comparison details
    """
    # Ensure all values are floats
    expected = float(expected)
    actual = float(actual)
    if tolerance is None:
        tolerance = 1e-9
    else:
        tolerance = float(tolerance)

    if expected == 0:
        # Use absolute tolerance for zero
        passed = abs(actual) <= tolerance
        diff = f"Absolute difference: {abs(actual)}"
    elif relative:
        rel_diff = abs(actual - expected) / abs(expected)
        passed = rel_diff <= tolerance
        diff = f"Relative difference: {rel_diff:.2e} (max: {tolerance})"
    else:
        abs_diff = abs(actual - expected)
        passed = abs_diff <= tolerance
        diff = f"Absolute difference: {abs_diff:.2e} (max: {tolerance})"

    return GoldenTestResult(
        passed=passed,
        expected=expected,
        actual=actual,
        tolerance=tolerance,
        diff=diff if not passed else None,
    )


def compare_with_pint(
    value: float,
    source_unit: str,
    target_unit: str,
    expected_value: float,
    tolerance: float = 1e-9,
) -> GoldenTestResult:
    """
    Cross-validate conversion result with Pint library.

    Args:
        value: Input value
        source_unit: Source unit
        target_unit: Target unit
        expected_value: Expected converted value
        tolerance: Comparison tolerance

    Returns:
        GoldenTestResult from Pint validation
    """
    if not PINT_AVAILABLE:
        pytest.skip("Pint not available for cross-validation")

    try:
        pint_qty = UREG.Quantity(value, source_unit)
        pint_result = pint_qty.to(target_unit).magnitude
        return compare_values(expected_value, pint_result, tolerance)
    except Exception as e:
        return GoldenTestResult(
            passed=False,
            expected=expected_value,
            actual=None,
            diff=f"Pint error: {str(e)}",
        )


# -----------------------------------------------------------------------------
# Pytest Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def golden_files_dir() -> Path:
    """Return the golden files directory."""
    return GOLDEN_FILES_DIR


@pytest.fixture(scope="session")
def unit_converter() -> "UnitConverter":
    """Create a UnitConverter instance for testing."""
    if not GL_NORMALIZER_AVAILABLE:
        pytest.skip("gl_normalizer_core not available")
    return UnitConverter()


@pytest.fixture(scope="session")
def unit_parser() -> "UnitParser":
    """Create a UnitParser instance for testing."""
    if not GL_NORMALIZER_AVAILABLE:
        pytest.skip("gl_normalizer_core not available")
    return UnitParser()


@pytest.fixture(scope="session")
def reference_resolver() -> "ReferenceResolver":
    """Create a ReferenceResolver instance for testing."""
    if not GL_NORMALIZER_AVAILABLE:
        pytest.skip("gl_normalizer_core not available")
    return ReferenceResolver()


@pytest.fixture(scope="session")
def pint_ureg():
    """Return Pint UnitRegistry for cross-validation."""
    if not PINT_AVAILABLE:
        pytest.skip("Pint not available")
    return UREG


@pytest.fixture(scope="session")
def canonical_units_config() -> Dict[str, Any]:
    """Load the canonical units configuration."""
    config_path = PACKAGE_ROOT / "config" / "canonical_units.yaml"
    if config_path.exists():
        return load_yaml_file(config_path)
    return {}


# -----------------------------------------------------------------------------
# Test Parameter Generators
# -----------------------------------------------------------------------------

def pytest_generate_tests(metafunc):
    """Generate test parameters from golden files."""

    # Unit conversion tests
    if "unit_conversion_case" in metafunc.fixturenames:
        test_cases = load_all_unit_conversion_tests()
        if test_cases:
            ids = [f"{dim}::{tc.get('name', 'unnamed')}" for dim, tc in test_cases]
            metafunc.parametrize("unit_conversion_case", test_cases, ids=ids)

    # Entity resolution tests
    if "entity_resolution_case" in metafunc.fixturenames:
        test_cases = load_all_entity_resolution_tests()
        if test_cases:
            ids = [f"{etype}::{tc.get('name', 'unnamed')}" for etype, tc in test_cases]
            metafunc.parametrize("entity_resolution_case", test_cases, ids=ids)

    # Full pipeline tests
    if "pipeline_case" in metafunc.fixturenames:
        test_cases = load_all_pipeline_tests()
        if test_cases:
            ids = [f"{scenario}::{tc.get('name', 'unnamed')}" for scenario, tc in test_cases]
            metafunc.parametrize("pipeline_case", test_cases, ids=ids)


# -----------------------------------------------------------------------------
# Custom Markers
# -----------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "pint_cross_validation: tests that cross-validate with Pint"
    )
    config.addinivalue_line(
        "markers", "compliance: regulatory compliance tests"
    )
    config.addinivalue_line(
        "markers", "edge_case: edge case tests"
    )
    config.addinivalue_line(
        "markers", "dimension(name): tests for specific dimension"
    )


# -----------------------------------------------------------------------------
# Test Helpers
# -----------------------------------------------------------------------------

class ConversionTestHelper:
    """Helper class for unit conversion tests."""

    @staticmethod
    def get_conversion_factor(
        canonical_config: Dict[str, Any],
        dimension: str,
        unit: str,
    ) -> Optional[float]:
        """Get conversion factor from canonical config."""
        dimensions = canonical_config.get("dimensions", {})
        dim_config = dimensions.get(dimension, {})
        input_units = dim_config.get("input_units", {})
        unit_config = input_units.get(unit, {})
        return unit_config.get("factor")

    @staticmethod
    def is_exact_conversion(
        canonical_config: Dict[str, Any],
        dimension: str,
        unit: str,
    ) -> bool:
        """Check if conversion is marked as exact."""
        dimensions = canonical_config.get("dimensions", {})
        dim_config = dimensions.get(dimension, {})
        input_units = dim_config.get("input_units", {})
        unit_config = input_units.get(unit, {})
        return unit_config.get("exact", False)


class EntityResolutionTestHelper:
    """Helper class for entity resolution tests."""

    @staticmethod
    def normalize_query(query: str) -> str:
        """Normalize a query string for comparison."""
        return query.strip().lower()

    @staticmethod
    def check_match_confidence(
        expected_confidence: float,
        actual_confidence: float,
        tolerance: float = 0.01,
    ) -> bool:
        """Check if confidence scores match within tolerance."""
        return abs(expected_confidence - actual_confidence) <= tolerance


# Export helpers
conversion_helper = ConversionTestHelper()
entity_helper = EntityResolutionTestHelper()
