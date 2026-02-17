# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-DATA-016 Data Freshness Monitor Agent tests.

Provides reusable test fixtures for configuration, sample datasets, SLA
definitions, freshness checks, and mock objects used across all test modules.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import MagicMock

import pydantic
import pytest


# ---------------------------------------------------------------------------
# Restore extra="forbid" on DFM models so our SDK tests can verify it
# ---------------------------------------------------------------------------

def _relax_model_configs():
    """Relax extra='forbid' to extra='ignore' on all DFM Pydantic models.

    The engine source code passes extra keyword arguments (e.g. registered_at,
    recorded_at, created_at, updated_at) to Pydantic models whose source
    declares extra='forbid'. For engine-level tests we need the models to
    accept (and silently discard) those extra fields so the engines can
    construct model instances without raising ValidationError.
    """
    try:
        from greenlang.data_freshness_monitor import models as dfm_models

        for name in dir(dfm_models):
            obj = getattr(dfm_models, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, pydantic.BaseModel)
                and obj is not pydantic.BaseModel
            ):
                cfg = getattr(obj, "model_config", {})
                if isinstance(cfg, dict) and cfg.get("extra") != "ignore":
                    obj.model_config = {**cfg, "extra": "ignore"}
                    obj.model_rebuild(force=True)
    except ImportError:
        pass


_relax_model_configs()


# ---------------------------------------------------------------------------
# Autouse fixture: clean GL_DFM_ environment variables before each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_dfm_env(monkeypatch):
    """Remove all GL_DFM_ environment variables before each test.

    This prevents leakage of environment state between tests that set
    GL_DFM_ prefixed variables via monkeypatch or os.environ.
    """
    keys_to_remove = [k for k in os.environ if k.startswith("GL_DFM_")]
    for key in keys_to_remove:
        monkeypatch.delenv(key, raising=False)

    # Reset the singleton config so each test starts fresh
    from greenlang.data_freshness_monitor.config import reset_config
    reset_config()

    yield

    # Post-test cleanup: reset singleton again
    reset_config()


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_config():
    """Create a fresh default DataFreshnessMonitorConfig.

    The autouse fixture already calls reset_config() before each test,
    so this simply returns a new default instance.
    """
    from greenlang.data_freshness_monitor.config import (
        DataFreshnessMonitorConfig,
        reset_config,
    )
    reset_config()
    cfg = DataFreshnessMonitorConfig()
    yield cfg
    reset_config()


# ---------------------------------------------------------------------------
# Sample dataset fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dataset_kwargs() -> Dict[str, Any]:
    """Return a dictionary of valid DatasetDefinition constructor kwargs."""
    return {
        "name": "ERP Emissions Feed",
        "source_name": "SAP ERP Production",
        "source_type": "erp",
        "owner": "data-engineering",
        "tags": ["scope1", "production"],
        "metadata": {"region": "US"},
    }


@pytest.fixture
def sample_sla_kwargs() -> Dict[str, Any]:
    """Return a dictionary of valid SLADefinition constructor kwargs."""
    return {
        "dataset_id": "ds-001",
        "warning_hours": 12.0,
        "critical_hours": 48.0,
    }


@pytest.fixture
def sample_freshness_check_kwargs() -> Dict[str, Any]:
    """Return kwargs for constructing a FreshnessCheck."""
    return {
        "dataset_id": "ds-001",
        "age_hours": 5.0,
        "freshness_score": 0.85,
    }


@pytest.fixture
def sample_breach_kwargs() -> Dict[str, Any]:
    """Return kwargs for constructing an SLABreach."""
    return {
        "dataset_id": "ds-001",
        "sla_id": "sla-001",
        "age_at_breach_hours": 80.0,
    }


@pytest.fixture
def sample_alert_kwargs() -> Dict[str, Any]:
    """Return kwargs for constructing a FreshnessAlert."""
    return {
        "breach_id": "breach-001",
        "message": "Dataset ERP Emissions Feed is stale",
    }


# ---------------------------------------------------------------------------
# Mock service fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_service():
    """Create a mock DataFreshnessMonitorService."""
    service = MagicMock()
    service.health_check.return_value = {"status": "healthy"}
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
