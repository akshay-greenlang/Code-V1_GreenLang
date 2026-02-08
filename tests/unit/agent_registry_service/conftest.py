# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Agent Registry & Service Catalog Unit Tests (AGENT-FOUND-007)
=================================================================================

Provides shared fixtures for testing the agent registry service config, models,
registry, health checker, dependency resolver, capability matcher, provenance
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
# Environment cleanup fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_agent_registry_env(monkeypatch):
    """Remove any GL_AGENT_REGISTRY_ env vars between tests."""
    prefix = "GL_AGENT_REGISTRY_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Sample Agent Metadata Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_agent_metadata() -> Dict[str, Any]:
    """Sample agent metadata for GL-001 Carbon Calculator."""
    return {
        "agent_id": "gl-001-carbon-calculator",
        "name": "Carbon Calculator Agent",
        "description": "Calculates Scope 1/2/3 carbon emissions",
        "version": "2.1.0",
        "layer": "calculation",
        "sector_classifications": ["energy", "manufacturing"],
        "execution_mode": "glip_v1",
        "idempotency_support": "full",
        "health_status": "healthy",
        "tags": ["carbon", "scope1", "scope2", "ghg"],
        "capabilities": [
            {
                "name": "carbon_calculation",
                "category": "calculation",
                "input_types": ["emission_factor", "activity_data"],
                "output_types": ["carbon_footprint"],
                "description": "Calculate carbon emissions from activity data",
            }
        ],
        "dependencies": [],
        "resource_profile": {
            "cpu_request": "100m",
            "cpu_limit": "500m",
            "memory_request": "128Mi",
            "memory_limit": "512Mi",
        },
    }


@pytest.fixture
def sample_capabilities() -> List[Dict[str, Any]]:
    """Sample capabilities list."""
    return [
        {
            "name": "carbon_calculation",
            "category": "calculation",
            "input_types": ["emission_factor", "activity_data"],
            "output_types": ["carbon_footprint"],
        },
        {
            "name": "cbam_reporting",
            "category": "reporting",
            "input_types": ["import_data", "emission_data"],
            "output_types": ["cbam_report"],
        },
        {
            "name": "data_validation",
            "category": "validation",
            "input_types": ["raw_data"],
            "output_types": ["validated_data"],
        },
    ]


@pytest.fixture
def sample_dependencies() -> List[Dict[str, Any]]:
    """Sample dependency list."""
    return [
        {
            "agent_id": "gl-001-carbon-calculator",
            "version_constraint": ">=2.0.0",
            "optional": False,
        },
        {
            "agent_id": "gl-010-data-validator",
            "version_constraint": ">=1.0.0",
            "optional": True,
        },
    ]


@pytest.fixture
def sample_admin_agent_metadata() -> Dict[str, Any]:
    """Sample admin/orchestrator agent metadata."""
    return {
        "agent_id": "gl-000-orchestrator",
        "name": "GreenLang Orchestrator",
        "description": "DAG-based agent orchestration",
        "version": "1.5.0",
        "layer": "orchestration",
        "sector_classifications": [],
        "execution_mode": "glip_v1",
        "idempotency_support": "full",
        "health_status": "healthy",
        "tags": ["orchestrator", "dag", "pipeline"],
        "capabilities": [
            {
                "name": "dag_execution",
                "category": "orchestration",
                "input_types": ["dag_definition"],
                "output_types": ["execution_result"],
            }
        ],
        "dependencies": [],
    }


@pytest.fixture
def sample_legacy_agent_metadata() -> Dict[str, Any]:
    """Sample legacy HTTP agent metadata."""
    return {
        "agent_id": "gl-legacy-001",
        "name": "Legacy Emissions Agent",
        "description": "Legacy HTTP-based emissions calculator",
        "version": "1.0.0",
        "layer": "calculation",
        "sector_classifications": ["energy"],
        "execution_mode": "legacy_http",
        "idempotency_support": "none",
        "health_status": "healthy",
        "tags": ["legacy", "emissions"],
        "capabilities": [],
        "dependencies": [],
        "legacy_http_config": {
            "endpoint": "http://legacy-emissions:8080/calculate",
            "auth_type": "bearer",
            "timeout_seconds": 30,
        },
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
