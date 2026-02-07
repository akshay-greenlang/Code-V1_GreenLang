# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Tracing Service Unit Tests (OBS-003)
========================================================

Provides common fixtures for testing the OpenTelemetry distributed
tracing SDK.  All OTel dependencies are mocked so tests run without
the OTel SDK installed.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import importlib
import threading
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# TracingConfig fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def tracing_config():
    """Create a default TracingConfig for testing.

    Returns a fresh TracingConfig with sensible test defaults.
    """
    from greenlang.infrastructure.tracing_service.config import TracingConfig

    return TracingConfig(
        service_name="test-service",
        service_version="1.0.0",
        environment="test",
        otlp_endpoint="http://localhost:4317",
        otlp_insecure=True,
        otlp_timeout=5,
        sampling_rate=1.0,
        enabled=True,
        console_exporter=False,
        instrument_fastapi=True,
        instrument_httpx=True,
        instrument_psycopg=True,
        instrument_redis=True,
        instrument_celery=True,
        instrument_requests=True,
        tenant_header="X-Tenant-ID",
        propagate_baggage=True,
        enrich_spans=True,
    )


@pytest.fixture
def disabled_config():
    """Create a TracingConfig with tracing disabled."""
    from greenlang.infrastructure.tracing_service.config import TracingConfig

    return TracingConfig(
        service_name="disabled-service",
        environment="test",
        enabled=False,
    )


# ---------------------------------------------------------------------------
# Mock OTel tracer / span
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_span():
    """Create a mock OpenTelemetry span.

    The mock supports context-manager usage (``with`` blocks), attribute
    setting, status setting, exception recording, and event addition.
    """
    span = MagicMock()
    span.set_attribute = MagicMock()
    span.set_status = MagicMock()
    span.record_exception = MagicMock()
    span.add_event = MagicMock()
    span.end = MagicMock()
    span.is_recording.return_value = True
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=False)

    # Span context
    span_context = MagicMock()
    span_context.trace_id = 0xABCDEF1234567890ABCDEF1234567890
    span_context.span_id = 0x1234567890ABCDEF
    span_context.is_valid = True
    span_context.is_remote = False
    span.get_span_context.return_value = span_context

    return span


@pytest.fixture
def mock_tracer(mock_span):
    """Create a mock OpenTelemetry tracer.

    Returns a ``(tracer, span)`` tuple where the tracer's
    ``start_as_current_span`` context manager yields the mock span.
    """
    tracer = MagicMock()
    tracer.start_as_current_span.return_value = mock_span
    tracer.start_span.return_value = mock_span
    return tracer, mock_span


# ---------------------------------------------------------------------------
# Provider state reset
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_provider_state():
    """Reset module-level provider state between tests.

    The ``provider.py`` module stores global state (``_provider``,
    ``_initialized``, ``_config``).  This fixture resets them so that
    each test starts from a clean slate.
    """
    yield

    # Reset after each test
    try:
        from greenlang.infrastructure.tracing_service import provider

        with provider._lock:
            provider._provider = None
            provider._config = None
            provider._initialized = False
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Mock ASGI app / scope / send / receive
# ---------------------------------------------------------------------------


@pytest.fixture
def asgi_scope():
    """Create a minimal ASGI HTTP scope for middleware testing."""
    return {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/agents",
        "query_string": b"",
        "headers": [
            (b"host", b"localhost:8000"),
            (b"user-agent", b"test-agent/1.0"),
            (b"x-tenant-id", b"t-acme"),
            (b"x-request-id", b"req-12345"),
        ],
        "server": ("localhost", 8000),
        "root_path": "",
    }


@pytest.fixture
def asgi_scope_health():
    """Create an ASGI scope for a health-check endpoint."""
    return {
        "type": "http",
        "method": "GET",
        "path": "/healthz",
        "query_string": b"",
        "headers": [],
        "server": ("localhost", 8000),
        "root_path": "",
    }


@pytest.fixture
def asgi_receive():
    """Create a mock ASGI receive callable."""

    async def receive():
        return {"type": "http.request", "body": b""}

    return receive


@pytest.fixture
def asgi_send():
    """Create a mock ASGI send callable that captures sent messages."""
    sent: List[Dict[str, Any]] = []

    async def send(message: Dict[str, Any]):
        sent.append(message)

    send.sent = sent  # type: ignore[attr-defined]
    return send


# ---------------------------------------------------------------------------
# Environment variable helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def env_otel_vars(monkeypatch):
    """Set standard OTEL_* environment variables for config tests."""
    monkeypatch.setenv("OTEL_SERVICE_NAME", "env-service")
    monkeypatch.setenv("GL_SERVICE_VERSION", "2.0.0")
    monkeypatch.setenv("GL_ENVIRONMENT", "staging")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4317")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_INSECURE", "false")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "30")
    monkeypatch.setenv("OTEL_TRACES_SAMPLER_ARG", "0.25")
    monkeypatch.setenv("OTEL_TRACES_CONSOLE", "true")
    monkeypatch.setenv("OTEL_TRACES_ENABLED", "true")


@pytest.fixture
def env_disabled(monkeypatch):
    """Set environment to disable tracing."""
    monkeypatch.setenv("OTEL_TRACES_ENABLED", "false")


# ---------------------------------------------------------------------------
# Captured spans helper
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_spans():
    """Provide a list to accumulate spans for assertion.

    Test code can append mock spans to this list and then assert on the
    collected entries.
    """
    return []


# ---------------------------------------------------------------------------
# Prometheus mock
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_prometheus():
    """Mock prometheus_client metrics objects."""
    counter = MagicMock()
    histogram = MagicMock()
    gauge = MagicMock()
    return {
        "counter": counter,
        "histogram": histogram,
        "gauge": gauge,
    }
