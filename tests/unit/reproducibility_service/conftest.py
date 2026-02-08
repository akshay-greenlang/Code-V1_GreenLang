# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Reproducibility Service Unit Tests (AGENT-FOUND-008)
=========================================================================

Provides shared fixtures for testing the reproducibility service config,
models, artifact hasher, determinism verifier, drift detector, replay engine,
environment capture, seed manager, version pinner, provenance tracker,
metrics, setup facade, and API router.

All tests are self-contained with no external dependencies.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Environment cleanup fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_reproducibility_env(monkeypatch):
    """Remove any GL_REPRODUCIBILITY_ env vars between tests."""
    prefix = "GL_REPRODUCIBILITY_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_input_data() -> Dict[str, Any]:
    """Sample input data for reproducibility verification."""
    return {
        "emissions": 100.5,
        "fuel_type": "diesel",
        "quantity": 1000,
        "region": "US",
    }


@pytest.fixture
def sample_output_data() -> Dict[str, Any]:
    """Sample output data for reproducibility verification."""
    return {
        "total_emissions": 2680.0,
        "unit": "kg_co2e",
        "scope": "scope_1",
        "confidence": 0.95,
    }


@pytest.fixture
def sample_baseline_data() -> Dict[str, Any]:
    """Sample baseline data for drift detection."""
    return {
        "total_emissions": 2680.0,
        "unit": "kg_co2e",
        "scope": "scope_1",
        "confidence": 0.95,
    }


@pytest.fixture
def sample_drifted_data() -> Dict[str, Any]:
    """Sample data that has drifted from baseline."""
    return {
        "total_emissions": 2750.0,
        "unit": "kg_co2e",
        "scope": "scope_1",
        "confidence": 0.92,
    }


@pytest.fixture
def sample_nested_data() -> Dict[str, Any]:
    """Sample nested data structure."""
    return {
        "level1": {
            "level2": {
                "value": 42.0,
                "list": [1, 2, 3],
            },
            "name": "test",
        },
        "items": [
            {"id": 1, "value": 10.0},
            {"id": 2, "value": 20.0},
        ],
    }


# ---------------------------------------------------------------------------
# Mock Prometheus fixtures
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
