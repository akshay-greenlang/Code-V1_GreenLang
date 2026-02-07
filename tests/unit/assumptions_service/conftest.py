# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Assumptions Registry Service Unit Tests (AGENT-FOUND-004)
==============================================================================

Provides shared fixtures for testing the assumptions registry service config,
models, registry, scenarios, validator, provenance, dependencies, metrics,
setup facade, and API router.

All tests are self-contained with no external dependencies.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Environment cleanup fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_assumptions_env(monkeypatch):
    """Remove any GL_ASSUMPTIONS_ env vars between tests."""
    prefix = "GL_ASSUMPTIONS_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Sample Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_assumption_data() -> Dict[str, Any]:
    """Sample assumption data for creation."""
    return {
        "assumption_id": "diesel_ef_us",
        "name": "US Diesel Emission Factor",
        "description": "Emission factor for diesel combustion in the United States",
        "category": "emission_factor",
        "data_type": "float",
        "value": 2.68,
        "unit": "kgCO2e/L",
        "source": "EPA GHG Emission Factors Hub 2024",
        "tags": ["diesel", "US", "scope1"],
        "metadata": {
            "region": "US",
            "fuel_type": "diesel",
            "sector": "stationary_combustion",
        },
    }


@pytest.fixture
def sample_scenario_data() -> Dict[str, Any]:
    """Sample scenario data for creation."""
    return {
        "name": "Conservative 2030",
        "description": "Conservative emission factor projections for 2030",
        "scenario_type": "conservative",
        "overrides": {
            "diesel_ef_us": 3.10,
            "natural_gas_ef_us": 2.25,
        },
        "tags": ["2030", "conservative", "projection"],
    }


@pytest.fixture
def sample_validation_rules() -> List[Dict[str, Any]]:
    """Sample validation rules for assumptions."""
    return [
        {
            "rule_id": "ef_positive",
            "assumption_id": "diesel_ef_us",
            "rule_type": "min_value",
            "parameters": {"min_value": 0.0},
            "severity": "error",
            "message": "Emission factor must be positive",
        },
        {
            "rule_id": "ef_max_bound",
            "assumption_id": "diesel_ef_us",
            "rule_type": "max_value",
            "parameters": {"max_value": 100.0},
            "severity": "error",
            "message": "Emission factor exceeds maximum bound",
        },
        {
            "rule_id": "ef_reasonable_range",
            "assumption_id": "diesel_ef_us",
            "rule_type": "max_value",
            "parameters": {"max_value": 10.0},
            "severity": "warning",
            "message": "Emission factor outside typical range",
        },
    ]


# ---------------------------------------------------------------------------
# Mock Prometheus Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_prometheus():
    """Mock prometheus_client for metrics testing."""
    mock_counter = MagicMock()
    mock_counter.labels.return_value = mock_counter
    mock_histogram = MagicMock()
    mock_histogram.labels.return_value = mock_histogram
    mock_gauge = MagicMock()
    mock_gauge.labels.return_value = mock_gauge

    mock_prom = MagicMock()
    mock_prom.Counter.return_value = mock_counter
    mock_prom.Histogram.return_value = mock_histogram
    mock_prom.Gauge.return_value = mock_gauge
    mock_prom.generate_latest.return_value = (
        b"# HELP test_metric\n# TYPE test_metric counter\n"
    )
    return mock_prom
