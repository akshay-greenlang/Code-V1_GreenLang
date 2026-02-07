# -*- coding: utf-8 -*-
"""
Integration Test Fixtures for Tracing Service (OBS-003)
=======================================================

Provides shared fixtures for integration testing the full trace pipeline:
SDK -> OTel Collector -> Tempo -> Query.

Tests use mock/in-process exporters by default. When TEMPO_ENDPOINT is set,
tests can run against a real Tempo instance.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from greenlang.infrastructure.tracing_service.config import TracingConfig


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

TEMPO_ENDPOINT = os.getenv("TEMPO_ENDPOINT", "")
OTEL_COLLECTOR_ENDPOINT = os.getenv(
    "OTEL_COLLECTOR_ENDPOINT", "http://localhost:4317"
)
SKIP_INTEGRATION = not TEMPO_ENDPOINT

skip_without_tempo = pytest.mark.skipif(
    SKIP_INTEGRATION,
    reason="TEMPO_ENDPOINT not set; skipping live integration tests",
)


# ---------------------------------------------------------------------------
# In-memory span exporter
# ---------------------------------------------------------------------------


class InMemorySpanExporter:
    """Collects exported spans in memory for test assertions.

    This is a simplified version of the OTel SDK's InMemorySpanExporter
    that works without requiring the full SDK.
    """

    def __init__(self) -> None:
        self.spans: List[Dict[str, Any]] = []
        self._shutdown = False

    def export(self, spans: Any) -> None:
        """Record spans."""
        if self._shutdown:
            return
        for span in spans:
            self.spans.append(
                {
                    "name": getattr(span, "name", str(span)),
                    "attributes": dict(getattr(span, "attributes", {}) or {}),
                    "status": getattr(span, "status", None),
                    "start_time": getattr(span, "start_time", 0),
                    "end_time": getattr(span, "end_time", 0),
                    "trace_id": getattr(
                        getattr(span, "context", None), "trace_id", 0
                    ),
                    "span_id": getattr(
                        getattr(span, "context", None), "span_id", 0
                    ),
                    "parent_id": getattr(span, "parent", None),
                }
            )

    def shutdown(self) -> None:
        """Mark exporter as shutdown."""
        self._shutdown = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """No-op flush."""
        return True

    def clear(self) -> None:
        """Clear collected spans."""
        self.spans.clear()

    def get_finished_spans(self) -> List[Dict[str, Any]]:
        """Return all collected spans."""
        return list(self.spans)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_exporter():
    """Provide a fresh InMemorySpanExporter."""
    return InMemorySpanExporter()


@pytest.fixture
def integration_config():
    """TracingConfig configured for integration testing."""
    return TracingConfig(
        service_name="integration-test-service",
        service_version="1.0.0-test",
        environment="test",
        otlp_endpoint=OTEL_COLLECTOR_ENDPOINT,
        otlp_insecure=True,
        otlp_timeout=5,
        sampling_rate=1.0,
        enabled=True,
        console_exporter=False,
        instrument_fastapi=False,
        instrument_httpx=False,
        instrument_psycopg=False,
        instrument_redis=False,
        instrument_celery=False,
        instrument_requests=False,
    )


@pytest.fixture
def mock_tracer_with_exporter(in_memory_exporter):
    """Create a mock tracer that records spans to the in-memory exporter."""
    tracer = MagicMock()
    spans_created: List[MagicMock] = []

    def make_span(name, **kwargs):
        span = MagicMock()
        span.name = name
        span._attributes = {}
        span.set_attribute = lambda k, v: span._attributes.__setitem__(k, v)
        span.is_recording.return_value = True
        span.__enter__ = MagicMock(return_value=span)
        span.__exit__ = MagicMock(return_value=False)
        spans_created.append(span)
        return span

    tracer.start_as_current_span = MagicMock(side_effect=make_span)
    tracer.start_span = MagicMock(side_effect=make_span)
    tracer._spans = spans_created
    return tracer, in_memory_exporter


@pytest.fixture(autouse=True)
def reset_tracing_state():
    """Reset tracing module state between integration tests."""
    yield
    try:
        from greenlang.infrastructure.tracing_service import setup
        from greenlang.infrastructure.tracing_service import provider

        setup._initialized = False
        setup._active_config = None
        with provider._lock:
            provider._provider = None
            provider._config = None
            provider._initialized = False
    except Exception:
        pass


@pytest.fixture
def trace_context_headers():
    """Sample W3C TraceContext headers for propagation tests."""
    return {
        "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        "tracestate": "congo=t61rcWkgMzE",
    }


@pytest.fixture
def sample_tenant_headers():
    """Sample GreenLang request headers with tenant context."""
    return {
        "x-tenant-id": "t-integration-test",
        "x-request-id": "req-integ-001",
        "x-correlation-id": "corr-integ-001",
        "x-user-id": "user-integ-001",
    }
