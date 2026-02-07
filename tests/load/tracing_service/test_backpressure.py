# -*- coding: utf-8 -*-
"""
Load tests - Tracing backpressure and graceful degradation (OBS-003)

Tests that the tracing SDK degrades gracefully when the OTel Collector
is unavailable or the export queue is full.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import time
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.infrastructure.tracing_service.config import TracingConfig
from greenlang.infrastructure.tracing_service.provider import (
    _NoOpTracer,
    _NoOpSpan,
    get_tracer,
)
from greenlang.infrastructure.tracing_service.middleware import TracingMiddleware
from greenlang.infrastructure.tracing_service.metrics_bridge import MetricsBridge


# ============================================================================
# Constants
# ============================================================================

BURST_SIZE = 5000
TARGET_P99_OVERHEAD_MS = 5.0


# ============================================================================
# NoOp fallback under load
# ============================================================================


class TestNoOpFallbackUnderLoad:
    """Test that the NoOp path handles burst traffic without issues."""

    def test_burst_span_creation(self):
        """Create a burst of 5000 spans on the NoOp path without errors."""
        tracer = _NoOpTracer()
        errors = 0

        for i in range(BURST_SIZE):
            try:
                with tracer.start_as_current_span(f"burst-op-{i}") as span:
                    span.set_attribute("burst.index", i)
                    span.set_attribute("gl.tenant_id", "t-burst")
            except Exception:
                errors += 1

        assert errors == 0, f"{errors} errors during burst span creation"

    def test_noop_span_context_manager_safety(self):
        """NoOp span context manager is safe to use in deeply nested calls."""
        tracer = _NoOpTracer()

        def nested_trace(depth: int) -> int:
            if depth <= 0:
                return 0
            with tracer.start_as_current_span(f"level-{depth}") as span:
                span.set_attribute("depth", depth)
                return 1 + nested_trace(depth - 1)

        total = nested_trace(100)
        assert total == 100


# ============================================================================
# Middleware under backpressure
# ============================================================================


class TestMiddlewareBackpressure:
    """Test middleware continues serving requests when tracing is degraded."""

    @pytest.mark.asyncio
    async def test_middleware_serves_when_otel_unavailable(self):
        """Middleware passes requests through when OTel is not available."""
        responses = []

        async def inner_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        mw = TracingMiddleware(inner_app, service_name="backpressure-svc")

        with patch(
            "greenlang.infrastructure.tracing_service.middleware.OTEL_AVAILABLE",
            False,
        ):
            for i in range(100):
                scope = {
                    "type": "http",
                    "method": "GET",
                    "path": f"/api/v1/test/{i}",
                    "query_string": b"",
                    "headers": [(b"host", b"localhost")],
                }

                sent: List[dict] = []

                async def send(msg, _sent=sent):
                    _sent.append(msg)

                await mw(scope, lambda: {"type": "http.request", "body": b""}, send)
                responses.append(sent)

        assert len(responses) == 100

    @pytest.mark.asyncio
    async def test_middleware_overhead_during_burst(self):
        """Middleware overhead stays < 5ms P99 during burst traffic."""
        durations: List[float] = []

        async def inner_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        mw = TracingMiddleware(inner_app, service_name="burst-svc")

        with patch(
            "greenlang.infrastructure.tracing_service.middleware.OTEL_AVAILABLE",
            False,
        ):
            for i in range(1000):
                scope = {
                    "type": "http",
                    "method": "GET",
                    "path": f"/api/v1/agents/{i}",
                    "query_string": b"",
                    "headers": [(b"host", b"localhost")],
                }

                async def send(msg):
                    pass

                start = time.perf_counter()
                await mw(
                    scope,
                    lambda: {"type": "http.request", "body": b""},
                    send,
                )
                durations.append((time.perf_counter() - start) * 1000)

        p99 = sorted(durations)[int(len(durations) * 0.99)]
        assert p99 < TARGET_P99_OVERHEAD_MS, (
            f"Middleware P99 overhead {p99:.3f}ms exceeds {TARGET_P99_OVERHEAD_MS}ms"
        )


# ============================================================================
# MetricsBridge under backpressure
# ============================================================================


class TestMetricsBridgeBackpressure:
    """Test MetricsBridge handles burst recording without data loss."""

    def test_burst_recording(self):
        """MetricsBridge handles a burst of 5000 record_span calls."""
        bridge = MetricsBridge(service_name="burst-svc")
        mock_hist = MagicMock()
        mock_counter = MagicMock()
        errors = 0

        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._SPAN_DURATION",
            mock_hist,
        ), patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._SPAN_COUNT",
            mock_counter,
        ):
            for i in range(BURST_SIZE):
                try:
                    bridge.record_span(f"op-{i}", 0.001 * i, status="ok")
                except Exception:
                    errors += 1

        assert errors == 0
        assert mock_hist.labels.call_count == BURST_SIZE
        assert mock_counter.labels.call_count == BURST_SIZE

    def test_no_prometheus_burst(self):
        """MetricsBridge handles burst recording when Prometheus is unavailable."""
        bridge = MetricsBridge(service_name="no-prom-svc")
        errors = 0

        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._SPAN_DURATION",
            None,
        ), patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._SPAN_COUNT",
            None,
        ):
            for i in range(BURST_SIZE):
                try:
                    bridge.record_span(f"op-{i}", 0.001 * i)
                except Exception:
                    errors += 1

        assert errors == 0
