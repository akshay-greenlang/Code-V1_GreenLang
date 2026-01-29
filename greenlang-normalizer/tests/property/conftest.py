"""
Pytest Configuration for GL-FOUND-X-003 Property-Based Tests.

This module provides shared fixtures, configuration, and test setup
for the property-based test suite using Hypothesis.

Configuration:
    - Hypothesis settings for thorough testing
    - Shared fixtures for normalizer components
    - Test markers for categorization
    - Database and example management

Usage:
    Fixtures defined here are automatically available to all test modules
    in the property test suite.
"""

import sys
from pathlib import Path
from typing import Generator, Dict, Any, List

import pytest
from hypothesis import settings, Verbosity, Phase, HealthCheck

# Add the source directory to Python path for imports
src_path = Path(__file__).parent.parent.parent / "packages" / "gl-normalizer-core" / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# =============================================================================
# Hypothesis Settings Profiles
# =============================================================================

# Development profile: Fast feedback during development
settings.register_profile(
    "dev",
    max_examples=50,
    verbosity=Verbosity.normal,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# CI profile: Balanced thoroughness and speed for CI pipelines
settings.register_profile(
    "ci",
    max_examples=200,
    verbosity=Verbosity.normal,
    deadline=30000,  # 30 seconds per test
    suppress_health_check=[HealthCheck.too_slow],
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target],
)

# Thorough profile: Comprehensive testing (1000 examples as specified)
settings.register_profile(
    "thorough",
    max_examples=1000,
    verbosity=Verbosity.verbose,
    deadline=60000,  # 60 seconds per test
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target, Phase.shrink],
    stateful_step_count=50,
)

# Debug profile: Verbose output for debugging failures
settings.register_profile(
    "debug",
    max_examples=10,
    verbosity=Verbosity.verbose,
    deadline=None,
    suppress_health_check=list(HealthCheck),
    report_multiple_bugs=True,
)

# Load the appropriate profile based on environment
# Default to "thorough" for production, "dev" otherwise
import os
profile_name = os.environ.get("HYPOTHESIS_PROFILE", "thorough")
settings.load_profile(profile_name)


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "property: Property-based tests using Hypothesis"
    )
    config.addinivalue_line(
        "markers",
        "conversion: Tests for unit conversion properties"
    )
    config.addinivalue_line(
        "markers",
        "dimension: Tests for dimensional analysis properties"
    )
    config.addinivalue_line(
        "markers",
        "roundtrip: Tests for roundtrip/serialization properties"
    )
    config.addinivalue_line(
        "markers",
        "slow: Tests that take longer to run"
    )
    config.addinivalue_line(
        "markers",
        "audit: Tests for audit trail properties"
    )


def pytest_collection_modifyitems(config, items):
    """Add markers based on test location."""
    for item in items:
        # Add property marker to all tests in this package
        if "property" in str(item.fspath):
            item.add_marker(pytest.mark.property)

        # Add specific markers based on test module name
        if "conversion" in item.nodeid:
            item.add_marker(pytest.mark.conversion)
        elif "dimension" in item.nodeid:
            item.add_marker(pytest.mark.dimension)
        elif "roundtrip" in item.nodeid:
            item.add_marker(pytest.mark.roundtrip)


# =============================================================================
# Shared Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def unit_converter():
    """
    Create a UnitConverter instance for the test session.

    Returns:
        UnitConverter: Configured unit converter instance
    """
    try:
        from gl_normalizer_core import UnitConverter
        return UnitConverter(use_pint=True)
    except ImportError:
        pytest.skip("gl_normalizer_core not available")


@pytest.fixture(scope="session")
def unit_parser():
    """
    Create a UnitParser instance for the test session.

    Returns:
        UnitParser: Configured unit parser instance
    """
    try:
        from gl_normalizer_core import UnitParser
        return UnitParser(strict_mode=False)
    except ImportError:
        pytest.skip("gl_normalizer_core not available")


@pytest.fixture(scope="session")
def dimension_analyzer():
    """
    Create a DimensionAnalyzer instance for the test session.

    Returns:
        DimensionAnalyzer: Configured dimension analyzer instance
    """
    try:
        from gl_normalizer_core import DimensionAnalyzer
        return DimensionAnalyzer()
    except ImportError:
        pytest.skip("gl_normalizer_core not available")


@pytest.fixture(scope="session")
def reference_resolver():
    """
    Create a ReferenceResolver instance for the test session.

    Returns:
        ReferenceResolver: Configured reference resolver instance
    """
    try:
        from gl_normalizer_core import ReferenceResolver
        return ReferenceResolver(min_confidence=70.0, fuzzy_enabled=True)
    except ImportError:
        pytest.skip("gl_normalizer_core not available")


@pytest.fixture(scope="session")
def hash_chain_generator():
    """
    Create a HashChainGenerator instance for the test session.

    Returns:
        HashChainGenerator: Configured hash chain generator instance
    """
    try:
        from gl_normalizer_core.audit.chain import HashChainGenerator
        return HashChainGenerator()
    except ImportError:
        pytest.skip("gl_normalizer_core not available")


@pytest.fixture
def dimension_class():
    """
    Get the Dimension class for testing.

    Returns:
        type: The Dimension class
    """
    try:
        from gl_normalizer_core.dimension import Dimension
        return Dimension
    except ImportError:
        pytest.skip("gl_normalizer_core not available")


@pytest.fixture
def quantity_class():
    """
    Get the Quantity class for testing.

    Returns:
        type: The Quantity class
    """
    try:
        from gl_normalizer_core.parser import Quantity
        return Quantity
    except ImportError:
        pytest.skip("gl_normalizer_core not available")


# =============================================================================
# Conversion Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def conversion_factors() -> Dict[tuple, float]:
    """
    Provide known conversion factors for validation.

    Returns:
        Dictionary mapping (source, target) to conversion factor
    """
    return {
        ("kilogram", "gram"): 1000.0,
        ("gram", "kilogram"): 0.001,
        ("kilogram", "metric_ton"): 0.001,
        ("metric_ton", "kilogram"): 1000.0,
        ("kilogram", "pound"): 2.20462,
        ("pound", "kilogram"): 0.453592,
        ("kilowatt_hour", "megajoule"): 3.6,
        ("megajoule", "kilowatt_hour"): 0.277778,
        ("megawatt_hour", "kilowatt_hour"): 1000.0,
        ("kilowatt_hour", "megawatt_hour"): 0.001,
        ("gigajoule", "megajoule"): 1000.0,
        ("megajoule", "gigajoule"): 0.001,
        ("liter", "gallon"): 0.264172,
        ("gallon", "liter"): 3.78541,
        ("cubic_meter", "liter"): 1000.0,
        ("liter", "cubic_meter"): 0.001,
        ("kilometer", "mile"): 0.621371,
        ("mile", "kilometer"): 1.60934,
    }


@pytest.fixture(scope="session")
def dimension_mappings() -> Dict[str, Dict[str, int]]:
    """
    Provide unit to dimension mappings for validation.

    Returns:
        Dictionary mapping unit names to dimension exponents
    """
    return {
        "kilogram": {"mass": 1},
        "gram": {"mass": 1},
        "metric_ton": {"mass": 1},
        "pound": {"mass": 1},
        "meter": {"length": 1},
        "kilometer": {"length": 1},
        "mile": {"length": 1},
        "second": {"time": 1},
        "hour": {"time": 1},
        "joule": {"mass": 1, "length": 2, "time": -2},
        "kilowatt_hour": {"mass": 1, "length": 2, "time": -2},
        "megajoule": {"mass": 1, "length": 2, "time": -2},
        "watt": {"mass": 1, "length": 2, "time": -3},
        "kilowatt": {"mass": 1, "length": 2, "time": -3},
        "liter": {"length": 3},
        "cubic_meter": {"length": 3},
    }


# =============================================================================
# Tolerance Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def default_tolerance() -> float:
    """
    Provide default numerical tolerance for comparisons.

    Per NFR-037: Numerical tolerance bounds (default 1e-9 relative).

    Returns:
        Default relative tolerance
    """
    return 1e-9


@pytest.fixture(scope="session")
def relaxed_tolerance() -> float:
    """
    Provide relaxed tolerance for floating-point comparisons.

    Used when precision loss is acceptable (e.g., multi-step conversions).

    Returns:
        Relaxed relative tolerance
    """
    return 1e-6


@pytest.fixture(scope="session")
def strict_tolerance() -> float:
    """
    Provide strict tolerance for high-precision comparisons.

    Returns:
        Strict relative tolerance
    """
    return 1e-12


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_quantities() -> List[Dict[str, Any]]:
    """
    Provide sample quantities for testing.

    Returns:
        List of sample quantity dictionaries
    """
    return [
        {"magnitude": 100.0, "unit": "kilogram"},
        {"magnitude": 1.5, "unit": "metric_ton"},
        {"magnitude": 1000.0, "unit": "kilowatt_hour"},
        {"magnitude": 3.6, "unit": "megajoule"},
        {"magnitude": 500.0, "unit": "liter"},
    ]


@pytest.fixture
def sample_fuels() -> List[str]:
    """
    Provide sample fuel names for resolution testing.

    Returns:
        List of fuel name strings
    """
    return [
        "Natural Gas",
        "Diesel",
        "Petrol",
        "Gasoline",
        "Coal",
        "LPG",
        "Biomass",
    ]


@pytest.fixture
def sample_audit_events() -> List[Dict[str, Any]]:
    """
    Provide sample audit events for testing.

    Returns:
        List of sample audit event dictionaries
    """
    return [
        {
            "event_id": "norm-evt-001",
            "source_record_id": "rec-001",
            "status": "success",
            "measurements": [
                {"field": "energy", "raw_value": 100, "raw_unit": "kWh"}
            ],
            "entities": [],
        },
        {
            "event_id": "norm-evt-002",
            "source_record_id": "rec-002",
            "status": "success",
            "measurements": [
                {"field": "mass", "raw_value": 50, "raw_unit": "kg"}
            ],
            "entities": [
                {"field": "fuel", "raw_name": "Diesel"}
            ],
        },
    ]


# =============================================================================
# Helper Functions Available to Tests
# =============================================================================

def approx_equal(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 1e-15) -> bool:
    """
    Check if two floats are approximately equal.

    Args:
        a: First value
        b: Second value
        rel_tol: Relative tolerance
        abs_tol: Absolute tolerance

    Returns:
        True if values are approximately equal
    """
    if a == b:
        return True
    if a == 0 or b == 0:
        return abs(a - b) < abs_tol
    return abs(a - b) / max(abs(a), abs(b)) < rel_tol


def dimensions_equal(dim_a: Dict[str, int], dim_b: Dict[str, int]) -> bool:
    """
    Check if two dimension dictionaries are equal.

    Handles missing keys as zero exponents.

    Args:
        dim_a: First dimension dictionary
        dim_b: Second dimension dictionary

    Returns:
        True if dimensions are equal
    """
    all_keys = set(dim_a.keys()) | set(dim_b.keys())
    for key in all_keys:
        if dim_a.get(key, 0) != dim_b.get(key, 0):
            return False
    return True


# Make helper functions available to tests
@pytest.fixture
def approx_equal_func():
    """Provide approx_equal function to tests."""
    return approx_equal


@pytest.fixture
def dimensions_equal_func():
    """Provide dimensions_equal function to tests."""
    return dimensions_equal
