"""
Pytest Configuration for Safety Tests

Provides fixtures and configuration for all safety tests.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from safety.boundary_policies import reset_policy_manager


@pytest.fixture(autouse=True)
def reset_policies_before_test():
    """Reset policy manager before each test."""
    reset_policy_manager()
    yield
    reset_policy_manager()


@pytest.fixture
def sample_allowed_tags():
    """Provide sample allowed tags for testing."""
    return ["TIC-101", "TIC-102", "PIC-101", "FIC-101", "XV-101"]


@pytest.fixture
def sample_sis_tags():
    """Provide sample SIS tags for testing."""
    return ["SIS-101", "ESD-001", "PSV-101", "TRIP-001", "XV-SIS-101"]


@pytest.fixture
def sample_temperatures():
    """Provide sample temperature values for testing."""
    return {
        "within_limits": 100.0,
        "at_min": -40.0,
        "at_max": 200.0,
        "under_min": -50.0,
        "over_max": 250.0,
    }


@pytest.fixture
def sample_pressures():
    """Provide sample pressure values for testing."""
    return {
        "within_limits": 500.0,
        "at_min": 0.0,
        "at_max": 1500.0,
        "under_min": -10.0,
        "over_max": 2000.0,
    }
