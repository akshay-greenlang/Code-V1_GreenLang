# -*- coding: utf-8 -*-
"""
GL-VCCI Scope 3 Platform - Centralized Pytest Configuration
Root-level conftest.py for shared fixtures across all test modules

This conftest.py provides:
1. Shared fixtures for all calculator category tests
2. Mock configurations for external dependencies
3. Common test data helpers
4. Test environment setup and teardown

Version: 1.0.0
Created: 2025-11-08
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any
from datetime import datetime
from greenlang.determinism import DeterministicClock


# ============================================================================
# SHARED FIXTURES - External Dependencies
# ============================================================================

@pytest.fixture
def mock_factor_broker():
    """
    Mock FactorBroker for all calculator tests.

    The FactorBroker resolves emission factors from various sources.
    This mock provides a consistent interface for testing.

    Returns:
        Mock: Configured FactorBroker mock with async resolve method
    """
    broker = Mock()
    broker.resolve = AsyncMock()

    # Default return value for factor resolution
    broker.resolve.return_value = Mock(
        factor_value=1.5,  # Default emission factor kgCO2e/unit
        factor_unit="kgCO2e/kg",
        source="test_database",
        region="US",
        year=2024,
        quality_score=0.9,
        metadata={"test": True}
    )

    return broker


@pytest.fixture
def mock_llm_client():
    """
    Mock LLM Client (Claude) for all tests using AI classification/estimation.

    Used by Tier 3 calculations and various classification tasks.

    Returns:
        AsyncMock: Configured LLM client mock
    """
    client = AsyncMock()

    # Default LLM response (can be overridden in specific tests)
    client.complete = AsyncMock(return_value='{"classification": "default", "confidence": 0.8}')

    return client


@pytest.fixture
def mock_uncertainty_engine():
    """
    Mock Uncertainty Engine for all calculator tests.

    The UncertaintyEngine propagates uncertainty through calculations.

    Returns:
        AsyncMock: Configured UncertaintyEngine mock
    """
    engine = AsyncMock()

    # Default uncertainty propagation (no-op for most tests)
    engine.propagate = AsyncMock(return_value=None)

    # Default uncertainty calculation
    engine.calculate_uncertainty = Mock(return_value=0.15)  # 15% default

    return engine


@pytest.fixture
def mock_provenance_builder():
    """
    Mock Provenance Chain Builder for all calculator tests.

    The ProvenanceChainBuilder creates audit trails for calculations.

    Returns:
        Mock: Configured ProvenanceChainBuilder mock
    """
    from services.agents.calculator.models import ProvenanceChain

    builder = Mock()

    # Hash factor info method
    builder.hash_factor_info = Mock(return_value="mock_hash_12345")

    # Build provenance chain method
    def mock_build(**kwargs):
        return ProvenanceChain(
            calculation_id=kwargs.get("calculation_id", "test-calc-001"),
            timestamp=DeterministicClock.now(),
            inputs_hash="mock_input_hash",
            factors_used=[],
            tier_applied=kwargs.get("tier", "tier_1"),
            data_quality_score=0.9,
            metadata={"test": True}
        )

    builder.build = AsyncMock(side_effect=mock_build)

    return builder


# ============================================================================
# SHARED FIXTURES - Test Data Helpers
# ============================================================================

@pytest.fixture
def sample_tier1_input():
    """
    Sample Tier 1 input data (highest quality - customer-provided data).

    Can be used as a base for category-specific tests.
    """
    return {
        "quantity": 1000.0,
        "quantity_unit": "kg",
        "region": "US",
        "year": 2024,
        "tier_preference": "tier_1"
    }


@pytest.fixture
def sample_tier2_input():
    """
    Sample Tier 2 input data (medium quality - calculated from secondary data).

    Can be used as a base for category-specific tests.
    """
    return {
        "quantity": 1000.0,
        "quantity_unit": "kg",
        "region": "US",
        "year": 2024,
        "tier_preference": "tier_2"
    }


@pytest.fixture
def sample_tier3_input():
    """
    Sample Tier 3 input data (lowest quality - LLM-estimated).

    Can be used as a base for category-specific tests.
    """
    return {
        "quantity": 1000.0,
        "quantity_unit": "kg",
        "region": "US",
        "year": 2024,
        "tier_preference": "tier_3"
    }


# ============================================================================
# SHARED FIXTURES - Test Database/State (if needed)
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """
    Create an event loop for async tests.

    This fixture ensures proper async/await handling across all tests.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def reset_mocks(mock_factor_broker, mock_llm_client, mock_uncertainty_engine, mock_provenance_builder):
    """
    Auto-reset all mocks after each test to prevent cross-test contamination.

    This fixture runs automatically after every test.
    """
    yield
    # Reset all mocks
    mock_factor_broker.reset_mock()
    mock_llm_client.reset_mock()
    mock_uncertainty_engine.reset_mock()
    mock_provenance_builder.reset_mock()


# ============================================================================
# PYTEST HOOKS - Test Execution Customization
# ============================================================================

def pytest_configure(config):
    """
    Pytest configuration hook - runs before test collection.

    Can be used to set up test environment, register custom markers, etc.
    """
    # Create test-reports directory if it doesn't exist
    import os
    os.makedirs("test-reports", exist_ok=True)

    # Set up any global test configuration
    config.addinivalue_line(
        "markers",
        "requires_api: mark test as requiring external API access"
    )


def pytest_collection_modifyitems(config, items):
    """
    Pytest hook to modify test collection.

    Can be used to add markers, skip tests, reorder tests, etc.
    """
    # Example: Skip slow tests if --fast flag is passed
    if config.getoption("--fast", default=False):
        skip_slow = pytest.mark.skip(reason="Skipping slow tests (--fast mode)")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


# ============================================================================
# HELPER FUNCTIONS - Available to All Tests
# ============================================================================

def assert_emissions_within_range(actual: float, expected: float, tolerance_pct: float = 5.0):
    """
    Helper function to assert emissions are within acceptable range.

    Args:
        actual: Actual emissions value
        expected: Expected emissions value
        tolerance_pct: Acceptable percentage tolerance (default 5%)

    Raises:
        AssertionError: If actual is outside tolerance range
    """
    lower_bound = expected * (1 - tolerance_pct / 100)
    upper_bound = expected * (1 + tolerance_pct / 100)

    assert lower_bound <= actual <= upper_bound, \
        f"Emissions {actual} kgCO2e outside acceptable range [{lower_bound}, {upper_bound}] " \
        f"(expected {expected} ± {tolerance_pct}%)"


def create_mock_factor_response(
    factor_value: float,
    factor_unit: str = "kgCO2e/unit",
    region: str = "US",
    source: str = "test_db",
    quality_score: float = 0.9
):
    """
    Helper to create consistent FactorResponse mocks.

    Args:
        factor_value: Emission factor value
        factor_unit: Unit of the emission factor
        region: Geographic region
        source: Data source identifier
        quality_score: Data quality score (0-1)

    Returns:
        Mock: Configured FactorResponse mock
    """
    response = Mock()
    response.factor_value = factor_value
    response.factor_unit = factor_unit
    response.region = region
    response.source = source
    response.year = 2024
    response.quality_score = quality_score
    response.metadata = {"created": "test"}

    return response


# ============================================================================
# PYTEST COMMAND LINE OPTIONS
# ============================================================================

def pytest_addoption(parser):
    """
    Add custom command line options to pytest.
    """
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Run fast tests only (skip slow tests)"
    )

    parser.addoption(
        "--category",
        action="store",
        default=None,
        help="Run tests for specific category (e.g., category_11)"
    )

    parser.addoption(
        "--tier",
        action="store",
        default=None,
        help="Run tests for specific tier (tier_1, tier_2, tier_3)"
    )


# ============================================================================
# NOTES
# ============================================================================
#
# This conftest.py provides shared fixtures for all test files in the
# GL-VCCI Scope 3 Platform. Individual test files can still define their
# own fixtures for category-specific needs.
#
# Fixture Scope:
# - Most fixtures are function-scoped (default) - created fresh for each test
# - event_loop is session-scoped - shared across all tests
# - reset_mocks is autouse - runs automatically after each test
#
# Usage in test files:
# Simply include the fixture name as a parameter in your test function:
#
#   @pytest.mark.asyncio
#   async def test_something(mock_factor_broker, mock_llm_client):
#       # Fixtures are automatically injected
#       result = await calculator.calculate(...)
#       assert result is not None
#
# ============================================================================
