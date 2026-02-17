# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-DATA-015 Cross-Source Reconciliation Agent tests.

Provides reusable test fixtures for configuration, sample sources, sample records,
match keys, discrepancies, and mock objects used across all test modules.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation (GL-DATA-X-018)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Autouse fixture: clean GL_CSR_ environment variables before each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_csr_env(monkeypatch):
    """Remove all GL_CSR_ environment variables before each test.

    This prevents leakage of environment state between tests that set
    GL_CSR_ prefixed variables via monkeypatch or os.environ.
    """
    keys_to_remove = [k for k in os.environ if k.startswith("GL_CSR_")]
    for key in keys_to_remove:
        monkeypatch.delenv(key, raising=False)

    # Reset the singleton config so each test starts fresh
    from greenlang.cross_source_reconciliation.config import reset_config
    reset_config()

    yield

    # Post-test cleanup: reset singleton again
    reset_config()


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_config():
    """Create a fresh default CrossSourceReconciliationConfig.

    The autouse fixture already calls reset_config() before each test,
    so this simply returns a new default instance.
    """
    from greenlang.cross_source_reconciliation.config import (
        CrossSourceReconciliationConfig,
        reset_config,
    )
    reset_config()
    cfg = CrossSourceReconciliationConfig()
    yield cfg
    reset_config()


# ---------------------------------------------------------------------------
# Sample source fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_source_data() -> Dict[str, Any]:
    """Return a dictionary of valid SourceDefinition constructor kwargs."""
    return {
        "name": "SAP ERP Production",
        "source_type": "erp",
        "priority": 80,
        "credibility_score": 0.9,
        "schema_info": {"columns": ["entity_id", "value", "period"]},
        "refresh_cadence": "daily",
        "description": "Production SAP ERP data feed",
        "tags": ["erp", "production", "scope1"],
    }


@pytest.fixture
def sample_source_b_data() -> Dict[str, Any]:
    """Return a second valid source for matching tests."""
    return {
        "name": "Utility Provider Feed",
        "source_type": "utility",
        "priority": 60,
        "credibility_score": 0.75,
        "schema_info": {"columns": ["facility_id", "consumption", "month"]},
        "refresh_cadence": "monthly",
        "description": "Utility meter data feed",
        "tags": ["utility", "scope2"],
    }


# ---------------------------------------------------------------------------
# Sample match key fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_match_key_a_kwargs() -> Dict[str, Any]:
    """Return kwargs for constructing a MatchKey from source A."""
    return {
        "entity_id": "facility-001",
        "period": "2025-Q1",
        "metric_name": "electricity_kwh",
        "source_id": "source-a",
    }


@pytest.fixture
def sample_match_key_b_kwargs() -> Dict[str, Any]:
    """Return kwargs for constructing a MatchKey from source B."""
    return {
        "entity_id": "facility-001",
        "period": "2025-Q1",
        "metric_name": "electricity_kwh",
        "source_id": "source-b",
    }


# ---------------------------------------------------------------------------
# Sample record data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_records() -> List[Dict[str, Any]]:
    """Return a list of 5 sample reconciliation records."""
    return [
        {
            "entity_id": "facility-001",
            "period": "2025-Q1",
            "metric": "electricity_kwh",
            "value": 12500.0,
            "source": "erp",
        },
        {
            "entity_id": "facility-001",
            "period": "2025-Q1",
            "metric": "electricity_kwh",
            "value": 12650.0,
            "source": "utility",
        },
        {
            "entity_id": "facility-002",
            "period": "2025-Q1",
            "metric": "natural_gas_m3",
            "value": 3400.0,
            "source": "erp",
        },
        {
            "entity_id": "facility-002",
            "period": "2025-Q1",
            "metric": "natural_gas_m3",
            "value": 3380.0,
            "source": "meter",
        },
        {
            "entity_id": "facility-003",
            "period": "2025-Q2",
            "metric": "diesel_litres",
            "value": 800.0,
            "source": "spreadsheet",
        },
    ]


# ---------------------------------------------------------------------------
# Mock service fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_service():
    """Create a mock CrossSourceReconciliationService."""
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
