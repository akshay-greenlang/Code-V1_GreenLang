# -*- coding: utf-8 -*-
"""
Pytest Fixtures for SLO Service Unit Tests (OBS-005)
=====================================================

Provides common fixtures for testing the SLO/SLI Definitions & Error
Budget Management Service.  All external dependencies (Prometheus,
Redis, database) are mocked so tests run without network access.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.infrastructure.slo_service.config import (
    SLOServiceConfig,
    reset_config,
)
from greenlang.infrastructure.slo_service.models import (
    BudgetStatus,
    BurnRateAlert,
    BurnRateWindow,
    ErrorBudget,
    SLI,
    SLIType,
    SLO,
    SLOReport,
    SLOReportEntry,
    SLOWindow,
)


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config():
    """Create an SLOServiceConfig with test defaults."""
    return SLOServiceConfig(
        service_name="test-slo",
        environment="test",
        enabled=True,
        prometheus_url="http://prometheus-test:9090",
        prometheus_timeout_seconds=5,
        redis_url="redis://redis-test:6379/3",
        redis_cache_ttl_seconds=30,
        database_url="postgresql://test:test@localhost:5432/test",
        budget_threshold_warning=20.0,
        budget_threshold_critical=50.0,
        budget_threshold_exhausted=100.0,
        budget_exhausted_action="alert_only",
        evaluation_interval_seconds=60,
        compliance_enabled=True,
        alerting_bridge_enabled=True,
    )


# ---------------------------------------------------------------------------
# SLI fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_sli_availability():
    """Create a sample availability SLI."""
    return SLI(
        name="api_availability",
        sli_type=SLIType.AVAILABILITY,
        good_query='http_requests_total{code!~"5.."}',
        total_query='http_requests_total',
        description="API request success rate",
    )


@pytest.fixture
def sample_sli_latency():
    """Create a sample latency SLI."""
    return SLI(
        name="api_latency_p99",
        sli_type=SLIType.LATENCY,
        good_query='http_request_duration_seconds_bucket{le="0.5"}',
        total_query='http_request_duration_seconds_count',
        threshold_ms=500.0,
        unit="ms",
        description="API p99 latency under 500ms",
    )


@pytest.fixture
def sample_sli_correctness():
    """Create a sample correctness SLI."""
    return SLI(
        name="calculation_correctness",
        sli_type=SLIType.CORRECTNESS,
        good_query='emissions_calculations_correct_total',
        total_query='emissions_calculations_total',
        description="Emissions calculation correctness rate",
    )


@pytest.fixture
def sample_sli_throughput():
    """Create a sample throughput SLI."""
    return SLI(
        name="pipeline_throughput",
        sli_type=SLIType.THROUGHPUT,
        good_query='pipeline_records_processed_total',
        total_query='pipeline_records_expected_total',
        description="Data pipeline throughput ratio",
    )


@pytest.fixture
def sample_sli_freshness():
    """Create a sample freshness SLI."""
    return SLI(
        name="data_freshness",
        sli_type=SLIType.FRESHNESS,
        good_query='data_freshness_checks_fresh_total',
        total_query='data_freshness_checks_total',
        description="Data freshness check pass rate",
    )


# ---------------------------------------------------------------------------
# SLO fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_slo(sample_sli_availability):
    """Create a sample SLO with availability SLI."""
    return SLO(
        slo_id="api-availability-99-9",
        name="API Availability",
        service="api-gateway",
        sli=sample_sli_availability,
        target=99.9,
        window=SLOWindow.THIRTY_DAYS,
        description="API gateway must maintain 99.9% availability",
        team="platform",
        labels={"tier": "critical", "env": "production"},
    )


@pytest.fixture
def sample_slo_latency(sample_sli_latency):
    """Create a sample SLO with latency SLI."""
    return SLO(
        slo_id="api-latency-p99-500ms",
        name="API Latency P99",
        service="api-gateway",
        sli=sample_sli_latency,
        target=99.0,
        window=SLOWindow.TWENTY_EIGHT_DAYS,
        description="99% of API requests must complete within 500ms",
        team="platform",
    )


@pytest.fixture
def sample_slo_list(sample_sli_availability, sample_sli_latency, sample_sli_correctness):
    """Create a list of sample SLOs for multi-SLO tests."""
    return [
        SLO(
            slo_id="api-availability-99-9",
            name="API Availability",
            service="api-gateway",
            sli=sample_sli_availability,
            target=99.9,
            team="platform",
        ),
        SLO(
            slo_id="api-latency-p99",
            name="API Latency P99",
            service="api-gateway",
            sli=sample_sli_latency,
            target=99.0,
            team="platform",
        ),
        SLO(
            slo_id="calc-correctness-99-99",
            name="Calculation Correctness",
            service="emissions-engine",
            sli=sample_sli_correctness,
            target=99.99,
            team="data-platform",
        ),
    ]


def _make_slo(
    slo_id: str = "test-slo-1",
    name: str = "Test SLO",
    service: str = "test-service",
    target: float = 99.9,
    sli_type: SLIType = SLIType.AVAILABILITY,
    window: SLOWindow = SLOWindow.THIRTY_DAYS,
    team: str = "platform",
    enabled: bool = True,
) -> SLO:
    """Factory helper for creating test SLOs."""
    sli = SLI(
        name=f"{name}_sli",
        sli_type=sli_type,
        good_query='http_requests_total{code!~"5.."}',
        total_query='http_requests_total',
    )
    return SLO(
        slo_id=slo_id,
        name=name,
        service=service,
        sli=sli,
        target=target,
        window=window,
        team=team,
        enabled=enabled,
    )


@pytest.fixture
def slo_factory():
    """Provide the SLO factory helper for tests."""
    return _make_slo


# ---------------------------------------------------------------------------
# ErrorBudget fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_error_budget():
    """Create a sample healthy error budget."""
    return ErrorBudget(
        slo_id="api-availability-99-9",
        total_minutes=43.2,       # 30d * 0.001 * 24 * 60
        consumed_minutes=4.32,    # 10% consumed
        remaining_minutes=38.88,
        remaining_percent=90.0,
        consumed_percent=10.0,
        status=BudgetStatus.HEALTHY,
        sli_value=99.99,
        window="30d",
    )


@pytest.fixture
def sample_error_budget_warning():
    """Create a warning-level error budget."""
    return ErrorBudget(
        slo_id="api-availability-99-9",
        total_minutes=43.2,
        consumed_minutes=15.12,
        remaining_minutes=28.08,
        remaining_percent=65.0,
        consumed_percent=35.0,
        status=BudgetStatus.WARNING,
        sli_value=99.965,
        window="30d",
    )


@pytest.fixture
def sample_error_budget_critical():
    """Create a critical-level error budget."""
    return ErrorBudget(
        slo_id="api-availability-99-9",
        total_minutes=43.2,
        consumed_minutes=32.4,
        remaining_minutes=10.8,
        remaining_percent=25.0,
        consumed_percent=75.0,
        status=BudgetStatus.CRITICAL,
        sli_value=99.925,
        window="30d",
    )


@pytest.fixture
def sample_error_budget_exhausted():
    """Create an exhausted error budget."""
    return ErrorBudget(
        slo_id="api-availability-99-9",
        total_minutes=43.2,
        consumed_minutes=43.2,
        remaining_minutes=0.0,
        remaining_percent=0.0,
        consumed_percent=100.0,
        status=BudgetStatus.EXHAUSTED,
        sli_value=99.8,
        window="30d",
    )


# ---------------------------------------------------------------------------
# BurnRateAlert fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_burn_rate_alert():
    """Create a sample burn rate alert."""
    return BurnRateAlert(
        slo_id="api-availability-99-9",
        slo_name="API Availability",
        burn_window="fast",
        burn_rate_long=16.5,
        burn_rate_short=18.2,
        threshold=14.4,
        severity="critical",
        service="api-gateway",
        message="SLO 'API Availability' burn rate alert (fast): long=16.50x, short=18.20x",
    )


# ---------------------------------------------------------------------------
# Mock clients
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis = MagicMock()
    redis.get.return_value = None
    redis.setex.return_value = True
    redis.delete.return_value = 1
    return redis


@pytest.fixture
def mock_redis_with_data(sample_error_budget):
    """Create a mock Redis client with pre-populated data."""
    redis = MagicMock()
    budget_json = json.dumps(sample_error_budget.to_dict(), default=str)
    redis.get.return_value = budget_json
    redis.setex.return_value = True
    redis.delete.return_value = 1
    return redis


@pytest.fixture
def mock_prometheus_response():
    """Create a mock successful Prometheus response."""
    return {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [
                {"metric": {}, "value": [1707307200, "0.9995"]}
            ],
        },
    }


@pytest.fixture
def mock_httpx_client(mock_prometheus_response):
    """Create an AsyncMock of httpx.AsyncClient."""
    client = AsyncMock()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = mock_prometheus_response
    response.raise_for_status = MagicMock()
    client.get.return_value = response
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


@pytest.fixture
def mock_db_pool():
    """Create a mock database connection pool."""
    pool = AsyncMock()
    conn = AsyncMock()
    conn.execute.return_value = None
    conn.fetchone.return_value = None
    conn.fetchall.return_value = []
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    return pool


# ---------------------------------------------------------------------------
# SLO YAML data fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_slo_yaml_data():
    """Return dict matching expected YAML SLO definitions format."""
    return {
        "slos": [
            {
                "slo_id": "api-availability-99-9",
                "name": "API Availability",
                "service": "api-gateway",
                "target": 99.9,
                "window": "30d",
                "team": "platform",
                "sli": {
                    "name": "api_availability",
                    "sli_type": "availability",
                    "good_query": 'http_requests_total{code!~"5.."}',
                    "total_query": "http_requests_total",
                },
            },
            {
                "slo_id": "api-latency-p99",
                "name": "API Latency P99",
                "service": "api-gateway",
                "target": 99.0,
                "window": "28d",
                "team": "platform",
                "sli": {
                    "name": "api_latency_p99",
                    "sli_type": "latency",
                    "good_query": 'http_request_duration_seconds_bucket{le="0.5"}',
                    "total_query": "http_request_duration_seconds_count",
                    "threshold_ms": 500.0,
                },
            },
        ]
    }


# ---------------------------------------------------------------------------
# Singleton reset (autouse)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_slo_config():
    """Reset the SLOServiceConfig singleton between tests."""
    yield
    reset_config()
