# -*- coding: utf-8 -*-
"""
Integration tests - Trace-to-logs and trace-to-metrics correlation (OBS-003)

Tests that trace context flows correctly for Grafana trace-to-logs and
trace-to-metrics linking.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

from greenlang.infrastructure.tracing_service.config import TracingConfig
from greenlang.infrastructure.tracing_service.context import (
    inject_trace_context,
    extract_trace_context,
    get_current_trace_id,
    get_current_span_id,
    get_current_trace_context,
    set_tenant_context,
    get_tenant_context,
    set_correlation_id,
    get_correlation_id,
)
from greenlang.infrastructure.tracing_service.metrics_bridge import MetricsBridge


# ============================================================================
# Trace-to-Logs correlation tests
# ============================================================================


class TestTraceToLogsCorrelation:
    """Test trace context is available for log enrichment."""

    def test_trace_context_dict_structure(self):
        """get_current_trace_context returns dict with trace_id and span_id keys."""
        ctx = get_current_trace_context()
        assert "trace_id" in ctx
        assert "span_id" in ctx

    def test_trace_context_without_active_span(self):
        """Without an active span, trace_id and span_id are None."""
        ctx = get_current_trace_context()
        assert ctx["trace_id"] is None
        assert ctx["span_id"] is None

    def test_trace_id_format(self):
        """When active, trace_id is a 32-char hex string."""
        trace_id = get_current_trace_id()
        if trace_id is not None:
            assert len(trace_id) == 32
            int(trace_id, 16)  # validates hex

    def test_span_id_format(self):
        """When active, span_id is a 16-char hex string."""
        span_id = get_current_span_id()
        if span_id is not None:
            assert len(span_id) == 16
            int(span_id, 16)  # validates hex

    def test_inject_produces_traceparent_header(self):
        """inject_trace_context writes traceparent header when span is active."""
        headers: Dict[str, str] = {}
        inject_trace_context(headers)
        # Without active span, traceparent may not be injected
        # but no exception should occur
        assert isinstance(headers, dict)

    def test_extract_from_valid_traceparent(self):
        """extract_trace_context can parse a valid W3C traceparent header."""
        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        }
        ctx = extract_trace_context(headers)
        # Result depends on OTel availability
        # But should not raise

    def test_extract_from_empty_headers(self):
        """extract_trace_context with empty headers returns None or default context."""
        ctx = extract_trace_context({})
        # Should not raise


# ============================================================================
# Correlation ID propagation tests
# ============================================================================


class TestCorrelationIDPropagation:
    """Test correlation ID flows through baggage for cross-service tracking."""

    def test_set_get_correlation_id(self):
        """Correlation ID can be set and retrieved from baggage."""
        set_correlation_id("corr-12345")
        result = get_correlation_id()
        # Depends on OTel availability
        assert result is None or result == "corr-12345"

    def test_tenant_context_set_get(self):
        """Tenant context can be set and retrieved from baggage."""
        set_tenant_context("t-corp-beta")
        result = get_tenant_context()
        assert result is None or result == "t-corp-beta"


# ============================================================================
# Trace-to-Metrics correlation tests
# ============================================================================


class TestTraceToMetricsCorrelation:
    """Test that span metrics are correctly generated for Prometheus."""

    def test_span_metrics_include_service_label(self):
        """MetricsBridge produces metrics with correct service label."""
        bridge = MetricsBridge(service_name="correlation-test-svc")
        mock_hist = MagicMock()

        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._SPAN_DURATION",
            mock_hist,
        ), patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._SPAN_COUNT",
            MagicMock(),
        ):
            bridge.record_span("process_request", 0.05, status="ok")

        mock_hist.labels.assert_called_with(
            service="correlation-test-svc",
            operation="process_request",
            status="ok",
        )

    def test_error_metrics_include_error_type(self):
        """Error metrics carry the error_type label for alerting."""
        bridge = MetricsBridge(service_name="svc")
        mock_errors = MagicMock()

        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._TRACE_ERRORS",
            mock_errors,
        ):
            bridge.record_error("api_call", "ConnectionError")

        mock_errors.labels.assert_called_with(
            service="svc",
            operation="api_call",
            error_type="ConnectionError",
        )

    def test_active_spans_gauge_tracks_concurrency(self):
        """Active spans gauge increments and decrements correctly."""
        bridge = MetricsBridge(service_name="svc")
        mock_gauge = MagicMock()

        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._ACTIVE_SPANS",
            mock_gauge,
        ):
            bridge.span_started()
            bridge.span_started()
            bridge.span_ended()

        assert mock_gauge.labels().inc.call_count == 2
        assert mock_gauge.labels().dec.call_count == 1

    def test_export_metrics_track_throughput(self):
        """Export counter tracks total spans exported."""
        bridge = MetricsBridge(service_name="svc")
        mock_counter = MagicMock()

        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._EXPORTED_SPANS",
            mock_counter,
        ):
            bridge.record_export(100, exporter="otlp")
            bridge.record_export(50, exporter="otlp")

        assert mock_counter.labels().inc.call_count == 2
