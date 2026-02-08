# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Access & Policy Guard Service Unit Tests (AGENT-FOUND-006)
===============================================================================

Provides shared fixtures for testing the access guard service config, models,
policy engine, rate limiter, classifier, audit logger, OPA integration,
provenance tracker, metrics, setup facade, and API router.

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
def _clean_access_guard_env(monkeypatch):
    """Remove any GL_ACCESS_GUARD_ env vars between tests."""
    prefix = "GL_ACCESS_GUARD_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Sample Principal Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_principal_data() -> Dict[str, Any]:
    """Sample analyst principal data for tenant-1."""
    return {
        "principal_id": "user-analyst-001",
        "principal_type": "user",
        "tenant_id": "tenant-1",
        "roles": ["analyst"],
        "attributes": {"department": "sustainability", "region": "US"},
        "clearance_level": "internal",
        "groups": ["sustainability-team"],
        "authenticated": True,
        "session_id": "session-001",
    }


@pytest.fixture
def sample_admin_principal_data() -> Dict[str, Any]:
    """Sample admin principal with confidential clearance."""
    return {
        "principal_id": "user-admin-001",
        "principal_type": "user",
        "tenant_id": "tenant-1",
        "roles": ["admin"],
        "attributes": {"department": "it", "region": "US"},
        "clearance_level": "confidential",
        "groups": ["admin-team"],
        "authenticated": True,
        "session_id": "session-002",
    }


@pytest.fixture
def sample_service_principal_data() -> Dict[str, Any]:
    """Sample service account principal."""
    return {
        "principal_id": "svc-ingest-001",
        "principal_type": "service",
        "tenant_id": "tenant-1",
        "roles": ["service_account"],
        "attributes": {"service_name": "data-ingest"},
        "clearance_level": "confidential",
        "groups": [],
        "authenticated": True,
        "session_id": None,
    }


# ---------------------------------------------------------------------------
# Sample Resource Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_resource_data() -> Dict[str, Any]:
    """Sample data resource in tenant-1, INTERNAL classification."""
    return {
        "resource_id": "res-data-001",
        "resource_type": "data",
        "tenant_id": "tenant-1",
        "classification": "internal",
        "owner_id": "user-analyst-001",
        "attributes": {"category": "emissions", "scope": "scope_1"},
        "geographic_location": "US",
    }


@pytest.fixture
def sample_restricted_resource_data() -> Dict[str, Any]:
    """Sample restricted resource (PII data)."""
    return {
        "resource_id": "res-pii-001",
        "resource_type": "data",
        "tenant_id": "tenant-1",
        "classification": "restricted",
        "owner_id": "user-admin-001",
        "attributes": {"category": "employee_data", "contains_pii": True},
        "geographic_location": "EU",
    }


# ---------------------------------------------------------------------------
# Sample Policy / Rule Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_policy_rule_data() -> Dict[str, Any]:
    """Sample data access allow rule for analysts."""
    return {
        "rule_id": "rule-001",
        "name": "Analyst Read Access",
        "description": "Allow analysts to read data resources",
        "policy_type": "data_access",
        "priority": 100,
        "enabled": True,
        "effect": "allow",
        "actions": ["read"],
        "resources": ["type:data"],
        "principals": ["role:analyst"],
        "conditions": {},
    }


@pytest.fixture
def sample_policy_data(sample_policy_rule_data) -> Dict[str, Any]:
    """Sample policy with two rules."""
    return {
        "policy_id": "policy-001",
        "name": "Analyst Data Access Policy",
        "description": "Controls analyst access to data resources",
        "version": "1.0.0",
        "enabled": True,
        "rules": [
            sample_policy_rule_data,
            {
                "rule_id": "rule-002",
                "name": "Analyst Write Deny",
                "description": "Deny analysts write access",
                "policy_type": "data_access",
                "priority": 50,
                "enabled": True,
                "effect": "deny",
                "actions": ["write", "delete"],
                "resources": ["type:data"],
                "principals": ["role:analyst"],
                "conditions": {},
            },
        ],
        "parent_policy_id": None,
        "allow_override": True,
        "tenant_id": "tenant-1",
        "applies_to": ["data"],
    }


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
