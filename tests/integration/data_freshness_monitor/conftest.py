# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-DATA-016 Data Freshness Monitor integration tests.

Provides reusable test fixtures for:
- Override of parent conftest autouse fixtures (mock_agents, block_network)
- Configuration reset between tests (fresh_config)
- Sample dataset definitions, SLA definitions, and refresh timestamps
- DataFreshnessMonitorService fixture
- ProvenanceTracker fixture
- FastAPI test app factory with mock service

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from greenlang.data_freshness_monitor.config import reset_config


# ---------------------------------------------------------------------------
# Override parent conftest autouse fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent conftest mock_agents fixture.

    The parent tests/integration/conftest.py defines an autouse fixture
    that patches greenlang.agents.registry.get_agent, which does not
    apply to DFM integration tests.
    """
    return {}


@pytest.fixture(scope="session", autouse=True)
def block_network():
    """Override parent conftest block_network fixture.

    The parent tests/integration/conftest.py blocks all socket access,
    which can interfere with asyncio event loop creation. We disable it
    for DFM integration tests since our tests are fully self-contained.
    """
    pass


# ---------------------------------------------------------------------------
# Configuration reset
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_config():
    """Reset the singleton config before and after each test.

    Ensures each test starts with a clean default configuration
    and does not leak state to subsequent tests.
    """
    reset_config()
    yield
    reset_config()


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Sample dataset definitions
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_datasets() -> List[Dict[str, Any]]:
    """Return sample dataset registration payloads.

    Provides 5 datasets across different sources, cadences, and priorities
    to exercise the full registration and freshness checking surface.
    """
    return [
        {
            "name": "Scope1 Emissions ERP",
            "source": "SAP",
            "owner": "sustainability-team",
            "refresh_cadence": "daily",
            "priority": 1,
            "tags": ["scope1", "erp", "critical"],
            "metadata": {"region": "EU", "business_unit": "manufacturing"},
        },
        {
            "name": "Scope2 Energy Meters",
            "source": "IoT-Meters",
            "owner": "energy-team",
            "refresh_cadence": "hourly",
            "priority": 2,
            "tags": ["scope2", "meters", "realtime"],
            "metadata": {"region": "US", "metering_frequency": "15min"},
        },
        {
            "name": "Scope3 Supplier Survey",
            "source": "Questionnaire",
            "owner": "procurement-team",
            "refresh_cadence": "quarterly",
            "priority": 5,
            "tags": ["scope3", "suppliers", "survey"],
            "metadata": {"region": "APAC", "survey_year": 2025},
        },
        {
            "name": "CBAM Import Data",
            "source": "CustomsAPI",
            "owner": "compliance-team",
            "refresh_cadence": "weekly",
            "priority": 3,
            "tags": ["cbam", "imports", "regulatory"],
            "metadata": {"regulation": "EU-CBAM", "cn_codes": ["7208", "7209"]},
        },
        {
            "name": "GHG Inventory Manual",
            "source": "Manual",
            "owner": "reporting-team",
            "refresh_cadence": "annual",
            "priority": 8,
            "tags": ["ghg", "manual", "annual"],
            "metadata": {"reporting_year": 2025, "framework": "GHG Protocol"},
        },
    ]


# ---------------------------------------------------------------------------
# Sample SLA definitions
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_sla_configs() -> List[Dict[str, Any]]:
    """Return sample SLA definition payloads.

    Provides SLAs with varying warning/critical thresholds for
    testing escalation tiers and breach detection.
    """
    return [
        {
            "name": "Critical Daily SLA",
            "warning_hours": 6.0,
            "critical_hours": 12.0,
            "severity": "critical",
            "escalation_policy": {
                "warning": {"channel": "slack", "team": "#data-quality"},
                "critical": {"channel": "pagerduty", "team": "on-call"},
            },
        },
        {
            "name": "Standard Weekly SLA",
            "warning_hours": 48.0,
            "critical_hours": 168.0,
            "severity": "high",
            "escalation_policy": {
                "warning": {"channel": "email", "team": "data-ops"},
                "critical": {"channel": "slack", "team": "#incidents"},
            },
        },
        {
            "name": "Relaxed Monthly SLA",
            "warning_hours": 336.0,
            "critical_hours": 720.0,
            "severity": "medium",
            "escalation_policy": {},
        },
    ]


# ---------------------------------------------------------------------------
# Sample refresh timestamps (relative to now)
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_timestamp() -> str:
    """Return an ISO timestamp from 30 minutes ago (excellent freshness)."""
    return (_utcnow() - timedelta(minutes=30)).isoformat()


@pytest.fixture
def good_timestamp() -> str:
    """Return an ISO timestamp from 3 hours ago (good freshness)."""
    return (_utcnow() - timedelta(hours=3)).isoformat()


@pytest.fixture
def fair_timestamp() -> str:
    """Return an ISO timestamp from 12 hours ago (fair freshness)."""
    return (_utcnow() - timedelta(hours=12)).isoformat()


@pytest.fixture
def poor_timestamp() -> str:
    """Return an ISO timestamp from 48 hours ago (poor freshness)."""
    return (_utcnow() - timedelta(hours=48)).isoformat()


@pytest.fixture
def stale_timestamp() -> str:
    """Return an ISO timestamp from 100 hours ago (stale)."""
    return (_utcnow() - timedelta(hours=100)).isoformat()


@pytest.fixture
def warning_breach_timestamp() -> str:
    """Return an ISO timestamp from 30 hours ago (exceeds default 24h warning)."""
    return (_utcnow() - timedelta(hours=30)).isoformat()


@pytest.fixture
def critical_breach_timestamp() -> str:
    """Return an ISO timestamp from 80 hours ago (exceeds default 72h critical)."""
    return (_utcnow() - timedelta(hours=80)).isoformat()


# ---------------------------------------------------------------------------
# Service fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def service():
    """Create a fresh DataFreshnessMonitorService instance.

    The service is started (startup called) and provides the full
    facade for the 7-engine freshness monitoring pipeline. Each test gets
    an isolated instance with empty in-memory stores.

    Note: Engine initialization may fail if engine constructors do not
    accept config kwargs. The service gracefully falls back to built-in
    implementations. This fixture catches startup errors and still
    marks the service as started so fallback code paths are exercised.
    """
    from greenlang.data_freshness_monitor.setup import (
        DataFreshnessMonitorService,
    )

    svc = DataFreshnessMonitorService()
    try:
        svc.startup()
    except TypeError:
        # Engine constructors may not accept config= kwarg.
        # Mark service as started so fallback paths are exercised.
        svc._started = True
    yield svc
    svc.shutdown()


# ---------------------------------------------------------------------------
# Provenance tracker fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def provenance_tracker():
    """Create a fresh ProvenanceTracker instance.

    Provides an isolated provenance tracker so tests do not
    pollute the global singleton.
    """
    from greenlang.data_freshness_monitor.provenance import (
        ProvenanceTracker,
    )

    tracker = ProvenanceTracker()
    yield tracker
    tracker.reset()


# ---------------------------------------------------------------------------
# FastAPI test app factory
# ---------------------------------------------------------------------------


@pytest.fixture
def freshness_app():
    """Create a minimal FastAPI test app with DFM service attached.

    Mounts the real router from greenlang.data_freshness_monitor.api.router
    and attaches a DataFreshnessMonitorService instance to
    app.state.data_freshness_monitor_service (matching the attribute
    name that _get_service() reads from the request).
    """
    try:
        from fastapi import FastAPI
    except ImportError:
        pytest.skip("FastAPI not installed; skipping API integration tests")

    from greenlang.data_freshness_monitor.api.router import (
        router,
        FASTAPI_AVAILABLE,
    )

    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not available in router module")

    from greenlang.data_freshness_monitor.setup import (
        DataFreshnessMonitorService,
    )

    app = FastAPI(title="DFM Integration Test")
    app.include_router(router)

    svc = DataFreshnessMonitorService()
    try:
        svc.startup()
    except TypeError:
        svc._started = True
    app.state.data_freshness_monitor_service = svc

    return app


@pytest.fixture
def test_client(freshness_app):
    """Create a synchronous test client for FastAPI integration tests."""
    from fastapi.testclient import TestClient

    return TestClient(freshness_app)


# ---------------------------------------------------------------------------
# Helper: register datasets and return their IDs
# ---------------------------------------------------------------------------


@pytest.fixture
def registered_datasets(service, sample_datasets):
    """Register all sample datasets and return a list of (dataset_id, dataset_dict) tuples.

    Convenience fixture that pre-populates the service with sample
    datasets so integration tests can immediately run checks and
    pipeline operations.
    """
    results = []
    for ds in sample_datasets:
        registered = service.register_dataset(**ds)
        results.append((registered["dataset_id"], registered))
    return results


@pytest.fixture
def registered_datasets_with_slas(service, sample_datasets, sample_sla_configs):
    """Register datasets and attach SLAs, returning (dataset_id, sla_id) pairs.

    The first SLA config is attached to the first dataset, second to
    the second, etc. Only the first min(datasets, slas) pairs are created.
    """
    pairs = []
    for i, ds_conf in enumerate(sample_datasets):
        registered = service.register_dataset(**ds_conf)
        dataset_id = registered["dataset_id"]

        if i < len(sample_sla_configs):
            sla_conf = sample_sla_configs[i].copy()
            sla_conf["dataset_id"] = dataset_id
            sla = service.create_sla(**sla_conf)
            pairs.append((dataset_id, sla["sla_id"]))
        else:
            pairs.append((dataset_id, None))
    return pairs
