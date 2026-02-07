# -*- coding: utf-8 -*-
"""
Pytest Fixtures for SLO Service Integration Tests (OBS-005)
============================================================

Provides fixtures for integration testing with mocked external
systems (Prometheus, Redis, database).

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
import socket
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

# Capture original socket class at import time, before the parent conftest's
# NetworkBlocker replaces it during session setup.
_ORIGINAL_SOCKET = socket.socket
_ORIGINAL_CREATE_CONNECTION = socket.create_connection

from greenlang.infrastructure.slo_service.config import (
    SLOServiceConfig,
    reset_config,
    set_config,
)
from greenlang.infrastructure.slo_service.models import (
    BudgetStatus,
    ErrorBudget,
    SLI,
    SLIType,
    SLO,
    SLOWindow,
)
from greenlang.infrastructure.slo_service.slo_manager import SLOManager


@pytest.fixture
def integration_config():
    """SLO config for integration tests."""
    config = SLOServiceConfig(
        service_name="integration-slo",
        environment="test",
        prometheus_url="http://localhost:9090",
        redis_url="redis://localhost:6379/15",
        database_url="postgresql://test:test@localhost:5432/test_slo",
        evaluation_interval_seconds=5,
        budget_exhausted_action="alert_only",
    )
    set_config(config)
    return config


@pytest.fixture
def populated_manager():
    """SLO manager pre-populated with test SLOs."""
    mgr = SLOManager()
    slos_data = [
        {
            "slo_id": "api-avail-99-9",
            "name": "API Availability 99.9",
            "service": "api-gateway",
            "target": 99.9,
            "team": "platform",
            "sli": SLI(
                name="api_availability",
                sli_type=SLIType.AVAILABILITY,
                good_query='http_requests_total{code!~"5.."}',
                total_query="http_requests_total",
            ),
        },
        {
            "slo_id": "api-latency-p99",
            "name": "API Latency P99 500ms",
            "service": "api-gateway",
            "target": 99.0,
            "team": "platform",
            "sli": SLI(
                name="api_latency",
                sli_type=SLIType.LATENCY,
                good_query='http_request_duration_seconds_bucket{le="0.5"}',
                total_query="http_request_duration_seconds_count",
                threshold_ms=500.0,
            ),
        },
        {
            "slo_id": "calc-correct-99-99",
            "name": "Calculation Correctness 99.99",
            "service": "emissions-engine",
            "target": 99.99,
            "team": "data-platform",
            "sli": SLI(
                name="calc_correctness",
                sli_type=SLIType.CORRECTNESS,
                good_query="emissions_calculations_correct_total",
                total_query="emissions_calculations_total",
            ),
        },
    ]

    for data in slos_data:
        slo = SLO(
            slo_id=data["slo_id"],
            name=data["name"],
            service=data["service"],
            sli=data["sli"],
            target=data["target"],
            team=data["team"],
        )
        mgr.create(slo)

    return mgr


@pytest.fixture
def mock_prometheus():
    """Mock Prometheus returning realistic SLI data."""
    async def _query(url, query, timeout_seconds=30):
        if "availability" in query or 'code!~"5.."' in query:
            return 0.9995
        elif "latency" in query or "duration" in query:
            return 0.9920
        elif "correctness" in query or "correct" in query:
            return 0.99995
        return 0.999

    return _query


@pytest.fixture(autouse=True)
def reset_integration_config():
    """Reset config between integration tests."""
    yield
    reset_config()


@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent conftest mock_agents (not needed for SLO service tests)."""
    yield


