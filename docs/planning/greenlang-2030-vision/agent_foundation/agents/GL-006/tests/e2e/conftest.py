# -*- coding: utf-8 -*-
"""
E2E Test Configuration for GL-006 HEATRECLAIM.

This module provides shared fixtures and configuration for end-to-end testing.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def pytest_configure(config):
    """Configure pytest markers for E2E tests."""
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end workflow test"
    )


@dataclass
class TestPlantConfig:
    """Test plant configuration for E2E scenarios."""
    plant_id: str = "PLANT-TEST-001"
    operating_hours_year: int = 8000
    min_approach_temp_c: float = 10.0
    target_payback_years: float = 3.0
    steam_cost_usd_ton: float = 30.0
    electricity_cost_usd_kwh: float = 0.10


@dataclass
class TestStream:
    """Test stream for E2E scenarios."""
    stream_id: str
    stream_type: str
    supply_temp_c: float
    target_temp_c: float
    heat_capacity_flow_kw_k: float
    flow_rate_kg_s: float = 5.0


@pytest.fixture(scope="session")
def plant_config():
    """Create test plant configuration."""
    return TestPlantConfig()


@pytest.fixture(scope="function")
def standard_test_streams():
    """Create standard test streams for E2E tests."""
    return [
        TestStream("H1", "hot", 180.0, 60.0, 10.0),
        TestStream("H2", "hot", 150.0, 40.0, 8.0),
        TestStream("H3", "hot", 120.0, 35.0, 6.0),
        TestStream("C1", "cold", 20.0, 135.0, 7.5),
        TestStream("C2", "cold", 80.0, 140.0, 12.0),
    ]


@pytest.fixture(scope="session")
def e2e_timeout_config():
    """Define timeout configuration for E2E tests."""
    return {
        "connection_timeout_s": 5.0,
        "optimization_timeout_s": 30.0,
        "pipeline_timeout_s": 60.0,
    }


@pytest.fixture(scope="session")
def expected_results():
    """Define expected results for validation."""
    return {
        "min_pinch_temp_c": 80.0,
        "max_pinch_temp_c": 120.0,
        "min_heat_recovery_percent": 50.0,
        "max_payback_years": 5.0,
        "min_exergy_efficiency_percent": 50.0,
    }
