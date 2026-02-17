# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-DATA-014 Time Series Gap Filler Agent tests.

Provides reusable test fixtures for configuration, sample data, mock objects,
and pre-built model instances used across all test modules.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Autouse fixture: clean GL_TSGF_ environment variables before each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_tsgf_env(monkeypatch):
    """Remove all GL_TSGF_ environment variables before each test.

    This prevents leakage of environment state between tests that set
    GL_TSGF_ prefixed variables via monkeypatch or os.environ.
    """
    keys_to_remove = [k for k in os.environ if k.startswith("GL_TSGF_")]
    for key in keys_to_remove:
        monkeypatch.delenv(key, raising=False)

    # Reset the singleton config so each test starts fresh
    from greenlang.time_series_gap_filler.config import reset_config
    reset_config()

    yield

    # Post-test cleanup: reset singleton again
    reset_config()


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_config():
    """Create a fresh default TimeSeriesGapFillerConfig.

    The autouse fixture already calls reset_config() before each test,
    so this simply returns a new default instance.
    """
    from greenlang.time_series_gap_filler.config import (
        TimeSeriesGapFillerConfig,
        reset_config,
    )
    reset_config()
    cfg = TimeSeriesGapFillerConfig()
    yield cfg
    reset_config()


# ---------------------------------------------------------------------------
# Sample time-series data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_values() -> List[Optional[float]]:
    """Return 10 values with 3 gaps (None) for testing gap filling."""
    return [1.0, 2.0, None, 4.0, 5.0, None, None, 8.0, 9.0, 10.0]


@pytest.fixture
def sample_timestamps() -> List[str]:
    """Return 10 hourly-spaced ISO 8601 timestamp strings."""
    return [
        "2026-01-15T00:00:00+00:00",
        "2026-01-15T01:00:00+00:00",
        "2026-01-15T02:00:00+00:00",
        "2026-01-15T03:00:00+00:00",
        "2026-01-15T04:00:00+00:00",
        "2026-01-15T05:00:00+00:00",
        "2026-01-15T06:00:00+00:00",
        "2026-01-15T07:00:00+00:00",
        "2026-01-15T08:00:00+00:00",
        "2026-01-15T09:00:00+00:00",
    ]


@pytest.fixture
def sample_long_series() -> List[Optional[float]]:
    """Return 100 float values with periodic gaps every 10th position."""
    result: List[Optional[float]] = []
    for i in range(100):
        if i % 10 == 0 and i > 0:
            result.append(None)
        else:
            result.append(float(i) + 0.5)
    return result


# ---------------------------------------------------------------------------
# Mock service fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_service():
    """Create a mock TimeSeriesGapFillerService."""
    service = MagicMock()
    service.health_check.return_value = {"status": "healthy"}
    service.get_statistics.return_value = MagicMock(total_jobs=0)
    return service


# ---------------------------------------------------------------------------
# Mock prometheus
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_prometheus():
    """Mock prometheus_client Counter, Histogram, and Gauge classes."""
    mock_counter = MagicMock()
    mock_counter.labels.return_value.inc = MagicMock()

    mock_histogram = MagicMock()
    mock_histogram.labels.return_value.observe = MagicMock()

    mock_gauge = MagicMock()
    mock_gauge.set = MagicMock()

    return {
        "Counter": mock_counter,
        "Histogram": mock_histogram,
        "Gauge": mock_gauge,
    }
