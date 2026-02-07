# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Monitoring Tests
====================================

Provides common fixtures for testing the monitoring subsystem.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Generator, Dict, Any
import time
import json


# -----------------------------------------------------------------------------
# PushGateway Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_pushgateway() -> Generator[Dict[str, MagicMock], None, None]:
    """
    Mock the prometheus_client push and delete functions.

    Yields:
        Dictionary with 'push' and 'delete' mocks
    """
    with patch("prometheus_client.push_to_gateway") as mock_push:
        with patch("prometheus_client.delete_from_gateway") as mock_delete:
            yield {"push": mock_push, "delete": mock_delete}


@pytest.fixture
def mock_pushgateway_with_errors() -> Generator[Dict[str, MagicMock], None, None]:
    """
    Mock PushGateway that simulates connection errors.

    Yields:
        Dictionary with mocks configured to raise errors
    """
    with patch("prometheus_client.push_to_gateway") as mock_push:
        with patch("prometheus_client.delete_from_gateway") as mock_delete:
            mock_push.side_effect = ConnectionError("Connection refused")
            mock_delete.side_effect = ConnectionError("Connection refused")
            yield {"push": mock_push, "delete": mock_delete}


@pytest.fixture
def mock_pushgateway_timeout() -> Generator[Dict[str, MagicMock], None, None]:
    """
    Mock PushGateway that simulates timeout errors.

    Yields:
        Dictionary with mocks configured to raise timeout
    """
    with patch("prometheus_client.push_to_gateway") as mock_push:
        with patch("prometheus_client.delete_from_gateway") as mock_delete:
            from urllib.error import URLError
            import socket
            mock_push.side_effect = URLError(socket.timeout("timed out"))
            yield {"push": mock_push, "delete": mock_delete}


# -----------------------------------------------------------------------------
# Registry Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_registry() -> "CollectorRegistry":
    """
    Create a fresh Prometheus CollectorRegistry for isolated testing.

    Returns:
        New CollectorRegistry instance
    """
    from prometheus_client import CollectorRegistry
    return CollectorRegistry()


@pytest.fixture
def mock_registry_with_metrics(mock_registry: "CollectorRegistry") -> "CollectorRegistry":
    """
    Create a registry pre-populated with common metrics.

    Args:
        mock_registry: Base registry fixture

    Returns:
        Registry with pre-registered metrics
    """
    from prometheus_client import Counter, Gauge, Histogram

    # Register common metrics
    Counter(
        "gl_test_counter_total",
        "Test counter",
        ["label1"],
        registry=mock_registry
    )

    Gauge(
        "gl_test_gauge",
        "Test gauge",
        registry=mock_registry
    )

    Histogram(
        "gl_test_duration_seconds",
        "Test histogram",
        ["operation"],
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
        registry=mock_registry
    )

    return mock_registry


# -----------------------------------------------------------------------------
# Time Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_time() -> Generator[MagicMock, None, None]:
    """
    Mock time.time() for deterministic testing.

    Yields:
        Mock time function
    """
    with patch("time.time") as mock_t:
        mock_t.return_value = 1704067200.0  # 2024-01-01 00:00:00 UTC
        yield mock_t


@pytest.fixture
def mock_time_advancing() -> Generator[MagicMock, None, None]:
    """
    Mock time.time() that advances with each call.

    Yields:
        Mock time function that returns incrementing values
    """
    with patch("time.time") as mock_t:
        start_time = 1704067200.0
        call_count = [0]

        def advancing_time():
            call_count[0] += 1
            return start_time + call_count[0] * 0.1

        mock_t.side_effect = advancing_time
        yield mock_t


# -----------------------------------------------------------------------------
# Configuration Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def pushgateway_config() -> Dict[str, Any]:
    """
    Standard PushGateway configuration for testing.

    Returns:
        Configuration dictionary
    """
    return {
        "url": "http://pushgateway.monitoring.svc:9091",
        "job_name": "test-job",
        "timeout": 10.0,
        "retry_count": 3,
        "retry_delay": 1.0,
    }


@pytest.fixture
def pushgateway_config_custom_grouping(pushgateway_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    PushGateway configuration with custom grouping key.

    Args:
        pushgateway_config: Base configuration

    Returns:
        Configuration with grouping_key
    """
    config = pushgateway_config.copy()
    config["grouping_key"] = {
        "instance": "worker-1",
        "environment": "test",
    }
    return config


# -----------------------------------------------------------------------------
# HTTP Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_http_response_success() -> MagicMock:
    """
    Mock successful HTTP response.

    Returns:
        Mock response with status 200
    """
    response = MagicMock()
    response.status_code = 200
    response.text = "OK"
    response.json.return_value = {"status": "success"}
    return response


@pytest.fixture
def mock_http_response_error() -> MagicMock:
    """
    Mock error HTTP response.

    Returns:
        Mock response with status 500
    """
    response = MagicMock()
    response.status_code = 500
    response.text = "Internal Server Error"
    response.raise_for_status.side_effect = Exception("HTTP 500")
    return response


# -----------------------------------------------------------------------------
# Async Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_async_push() -> Generator[AsyncMock, None, None]:
    """
    Mock async push function.

    Yields:
        Async mock for push operations
    """
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_response = AsyncMock()
        mock_response.status = 202
        mock_response.text = AsyncMock(return_value="Accepted")
        mock_post.return_value.__aenter__.return_value = mock_response
        yield mock_post


# -----------------------------------------------------------------------------
# Sample Data Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_job_metrics() -> Dict[str, Any]:
    """
    Sample batch job metrics data.

    Returns:
        Dictionary with sample metrics values
    """
    return {
        "duration_seconds": 125.5,
        "records_processed": 10000,
        "records_failed": 5,
        "memory_peak_bytes": 536870912,  # 512 MB
        "status": "success",
    }


@pytest.fixture
def sample_error_metrics() -> Dict[str, Any]:
    """
    Sample batch job error metrics data.

    Returns:
        Dictionary with error metrics values
    """
    return {
        "duration_seconds": 30.2,
        "records_processed": 500,
        "records_failed": 500,
        "memory_peak_bytes": 268435456,  # 256 MB
        "status": "error",
        "error_type": "ValidationError",
        "error_message": "Invalid data format",
    }


# -----------------------------------------------------------------------------
# Prometheus Rules Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_recording_rules() -> Dict[str, Any]:
    """
    Sample recording rules for validation.

    Returns:
        Recording rules YAML structure as dict
    """
    return {
        "groups": [
            {
                "name": "gl_recording_rules",
                "interval": "1m",
                "rules": [
                    {
                        "record": "gl:api_request_rate:5m",
                        "expr": "sum(rate(gl_api_requests_total[5m])) by (service)",
                    },
                    {
                        "record": "gl:api_latency_p99:5m",
                        "expr": "histogram_quantile(0.99, sum(rate(gl_api_request_duration_seconds_bucket[5m])) by (le, service))",
                    },
                ],
            }
        ]
    }


@pytest.fixture
def sample_alert_rules() -> Dict[str, Any]:
    """
    Sample alert rules for validation.

    Returns:
        Alert rules YAML structure as dict
    """
    return {
        "groups": [
            {
                "name": "gl_alerts",
                "rules": [
                    {
                        "alert": "HighErrorRate",
                        "expr": "sum(rate(gl_api_errors_total[5m])) / sum(rate(gl_api_requests_total[5m])) > 0.01",
                        "for": "5m",
                        "labels": {
                            "severity": "critical",
                        },
                        "annotations": {
                            "summary": "High error rate detected",
                            "description": "Error rate is {{ $value | humanizePercentage }}",
                        },
                    },
                ],
            }
        ]
    }


# -----------------------------------------------------------------------------
# Cleanup Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def cleanup_prometheus_registry():
    """
    Automatically cleanup the default Prometheus registry after each test.

    This prevents metric name conflicts between tests.
    """
    yield

    # Cleanup is handled by using isolated registries per test
    # This fixture is a placeholder for any global cleanup needed


# -----------------------------------------------------------------------------
# Environment Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def env_development(monkeypatch):
    """Set environment to development."""
    monkeypatch.setenv("GREENLANG_ENV", "development")
    monkeypatch.setenv("PUSHGATEWAY_URL", "http://localhost:9091")


@pytest.fixture
def env_production(monkeypatch):
    """Set environment to production."""
    monkeypatch.setenv("GREENLANG_ENV", "production")
    monkeypatch.setenv("PUSHGATEWAY_URL", "http://pushgateway.monitoring.svc:9091")


@pytest.fixture
def env_with_custom_pushgateway(monkeypatch):
    """Set custom PushGateway URL."""
    monkeypatch.setenv("PUSHGATEWAY_URL", "http://custom-pushgateway:9091")
    monkeypatch.setenv("PUSHGATEWAY_TIMEOUT", "30")
    monkeypatch.setenv("PUSHGATEWAY_RETRY_COUNT", "5")
