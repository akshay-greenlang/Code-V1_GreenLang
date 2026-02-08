# -*- coding: utf-8 -*-
"""
Pytest Fixtures for QA Test Harness Service Unit Tests (AGENT-FOUND-009)
=========================================================================

Provides shared fixtures for testing the QA Test Harness config, models,
test runner, assertion engine, golden file manager, regression detector,
performance benchmarker, coverage tracker, report generator, provenance
tracker, metrics, setup facade, and API router.

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
# Environment cleanup fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_qa_test_harness_env(monkeypatch):
    """Remove any GL_QA_TEST_HARNESS_ env vars between tests."""
    prefix = "GL_QA_TEST_HARNESS_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_input_data() -> Dict[str, Any]:
    """Sample input data for QA test harness verification."""
    return {
        "emissions": 100.5,
        "fuel_type": "diesel",
        "quantity": 1000,
        "region": "US",
    }


@pytest.fixture
def sample_output_data() -> Dict[str, Any]:
    """Sample output data for QA test harness verification."""
    return {
        "total_emissions": 2680.0,
        "unit": "kg_co2e",
        "scope": "scope_1",
        "confidence": 0.95,
    }


@pytest.fixture
def sample_agent_result() -> Dict[str, Any]:
    """Sample agent result dictionary."""
    return {
        "success": True,
        "data": {
            "total_emissions": 2680.0,
            "unit": "kg_co2e",
            "provenance_id": "prov-abc12345",
            "timestamp": "2026-01-01T00:00:00Z",
        },
        "error": None,
        "metrics": {"duration_ms": 5.0},
    }


@pytest.fixture
def sample_test_results() -> List[Dict[str, Any]]:
    """Sample test results for report generation."""
    return [
        {"test_id": "t1", "name": "test_one", "status": "passed", "duration_ms": 1.0},
        {"test_id": "t2", "name": "test_two", "status": "failed", "duration_ms": 2.0},
        {"test_id": "t3", "name": "test_three", "status": "passed", "duration_ms": 0.5},
    ]


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
