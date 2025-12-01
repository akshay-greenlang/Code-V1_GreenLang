# -*- coding: utf-8 -*-
"""
Determinism Test Configuration for GL-006 HEATRECLAIM.

This module provides shared fixtures and configuration for determinism testing.
"""

import pytest
import sys
from pathlib import Path
from decimal import Decimal

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def pytest_configure(config):
    """Configure pytest markers for determinism tests."""
    config.addinivalue_line(
        "markers", "determinism: mark test as a determinism/reproducibility test"
    )


@pytest.fixture(scope="session")
def precision_config():
    """Define precision configuration for deterministic calculations."""
    return {
        "decimal_places": 6,
        "temperature_precision": Decimal("0.01"),
        "heat_duty_precision": Decimal("0.001"),
        "efficiency_precision": Decimal("0.0001"),
        "financial_precision": Decimal("0.01"),
    }


@pytest.fixture(scope="session")
def reproducibility_iterations():
    """Number of iterations for reproducibility verification."""
    return 1000


@pytest.fixture(scope="session")
def golden_values():
    """Known golden values for verification."""
    return {
        "carnot_efficiency_300_500": Decimal("0.4000"),  # 1 - 300/500
        "lmtd_60_30": Decimal("43.2808"),  # (60-30)/ln(60/30)
        "simple_payback_500k_150k": Decimal("3.33"),  # 500000/150000
    }
