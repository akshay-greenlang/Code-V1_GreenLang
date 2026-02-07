# -*- coding: utf-8 -*-
"""
Unit tests for MetricsBridge (OBS-003)

Tests trace-to-Prometheus metrics bridging: span recording, error tracking,
active span gauges, export counting, and singleton access.

Coverage target: 85%+ of metrics_bridge.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from greenlang.infrastructure.tracing_service.metrics_bridge import (
    MetricsBridge,
    get_metrics_bridge,
    PROMETHEUS_AVAILABLE,
)


# ============================================================================
# MetricsBridge construction tests
# ============================================================================


class TestMetricsBridgeInit:
    """Tests for MetricsBridge construction."""

    def test_default_service_name(self):
        """Default service name is 'greenlang'."""
        bridge = MetricsBridge()
        assert bridge.service_name == "greenlang"

    def test_custom_service_name(self):
        """Custom service name is respected."""
        bridge = MetricsBridge(service_name="api-service")
        assert bridge.service_name == "api-service"


# ============================================================================
# Span recording tests
# ============================================================================


class TestRecordSpan:
    """Tests for MetricsBridge.record_span."""

    def test_record_span_with_prometheus(self):
        """record_span observes histogram and increments counter when Prometheus is available."""
        bridge = MetricsBridge(service_name="test-svc")

        mock_histogram = MagicMock()
        mock_counter = MagicMock()

        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._SPAN_DURATION",
            mock_histogram,
        ), patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._SPAN_COUNT",
            mock_counter,
        ):
            bridge.record_span("calculate_emissions", 0.045, status="ok")

        mock_histogram.labels.assert_called_once_with(
            service="test-svc", operation="calculate_emissions", status="ok"
        )
        mock_histogram.labels().observe.assert_called_once_with(0.045)
        mock_counter.labels.assert_called_once_with(
            service="test-svc", operation="calculate_emissions", status="ok"
        )
        mock_counter.labels().inc.assert_called_once()

    def test_record_span_no_prometheus(self):
        """record_span is a no-op when Prometheus metrics are None."""
        bridge = MetricsBridge(service_name="test-svc")
        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._SPAN_DURATION",
            None,
        ), patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._SPAN_COUNT",
            None,
        ):
            # Should not raise
            bridge.record_span("op", 0.1)

    def test_record_span_default_status_ok(self):
        """Default status is 'ok'."""
        bridge = MetricsBridge(service_name="svc")
        mock_histogram = MagicMock()

        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._SPAN_DURATION",
            mock_histogram,
        ), patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._SPAN_COUNT",
            MagicMock(),
        ):
            bridge.record_span("op", 1.0)

        mock_histogram.labels.assert_called_with(
            service="svc", operation="op", status="ok"
        )

    def test_record_span_error_status(self):
        """Error status is passed through correctly."""
        bridge = MetricsBridge(service_name="svc")
        mock_counter = MagicMock()

        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._SPAN_DURATION",
            MagicMock(),
        ), patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._SPAN_COUNT",
            mock_counter,
        ):
            bridge.record_span("op", 0.5, status="error")

        mock_counter.labels.assert_called_with(
            service="svc", operation="op", status="error"
        )


# ============================================================================
# Error recording tests
# ============================================================================


class TestRecordError:
    """Tests for MetricsBridge.record_error."""

    def test_record_error(self):
        """record_error increments the error counter."""
        bridge = MetricsBridge(service_name="test-svc")
        mock_counter = MagicMock()

        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._TRACE_ERRORS",
            mock_counter,
        ):
            bridge.record_error("calculate_emissions", "ValueError")

        mock_counter.labels.assert_called_once_with(
            service="test-svc",
            operation="calculate_emissions",
            error_type="ValueError",
        )
        mock_counter.labels().inc.assert_called_once()

    def test_record_error_no_prometheus(self):
        """record_error is a no-op when counter is None."""
        bridge = MetricsBridge()
        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._TRACE_ERRORS",
            None,
        ):
            bridge.record_error("op", "RuntimeError")


# ============================================================================
# Active span tracking tests
# ============================================================================


class TestActiveSpanTracking:
    """Tests for span_started/span_ended gauge management."""

    def test_span_started_increments_gauge(self):
        """span_started increments the active spans gauge."""
        bridge = MetricsBridge(service_name="svc")
        mock_gauge = MagicMock()

        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._ACTIVE_SPANS",
            mock_gauge,
        ):
            bridge.span_started()

        mock_gauge.labels.assert_called_with(service="svc")
        mock_gauge.labels().inc.assert_called_once()

    def test_span_ended_decrements_gauge(self):
        """span_ended decrements the active spans gauge."""
        bridge = MetricsBridge(service_name="svc")
        mock_gauge = MagicMock()

        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._ACTIVE_SPANS",
            mock_gauge,
        ):
            bridge.span_ended()

        mock_gauge.labels.assert_called_with(service="svc")
        mock_gauge.labels().dec.assert_called_once()

    def test_span_started_no_prometheus(self):
        """span_started is a no-op when gauge is None."""
        bridge = MetricsBridge()
        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._ACTIVE_SPANS",
            None,
        ):
            bridge.span_started()

    def test_span_ended_no_prometheus(self):
        """span_ended is a no-op when gauge is None."""
        bridge = MetricsBridge()
        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._ACTIVE_SPANS",
            None,
        ):
            bridge.span_ended()


# ============================================================================
# Export tracking tests
# ============================================================================


class TestRecordExport:
    """Tests for MetricsBridge.record_export."""

    def test_record_export(self):
        """record_export increments the exported counter by count."""
        bridge = MetricsBridge(service_name="svc")
        mock_counter = MagicMock()

        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._EXPORTED_SPANS",
            mock_counter,
        ):
            bridge.record_export(42, exporter="otlp")

        mock_counter.labels.assert_called_with(service="svc", exporter="otlp")
        mock_counter.labels().inc.assert_called_once_with(42)

    def test_record_export_default_exporter(self):
        """Default exporter label is 'otlp'."""
        bridge = MetricsBridge(service_name="svc")
        mock_counter = MagicMock()

        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._EXPORTED_SPANS",
            mock_counter,
        ):
            bridge.record_export(10)

        mock_counter.labels.assert_called_with(service="svc", exporter="otlp")

    def test_record_export_no_prometheus(self):
        """record_export is a no-op when counter is None."""
        bridge = MetricsBridge()
        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._EXPORTED_SPANS",
            None,
        ):
            bridge.record_export(100)


# ============================================================================
# Singleton tests
# ============================================================================


class TestGetMetricsBridge:
    """Tests for the get_metrics_bridge singleton factory."""

    def test_returns_instance(self):
        """get_metrics_bridge returns a MetricsBridge instance."""
        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._bridge",
            None,
        ):
            bridge = get_metrics_bridge("test-svc")
            assert isinstance(bridge, MetricsBridge)
            assert bridge.service_name == "test-svc"

    def test_singleton_returns_same_instance(self):
        """Subsequent calls return the same instance."""
        with patch(
            "greenlang.infrastructure.tracing_service.metrics_bridge._bridge",
            None,
        ):
            b1 = get_metrics_bridge("svc")
            with patch(
                "greenlang.infrastructure.tracing_service.metrics_bridge._bridge",
                b1,
            ):
                b2 = get_metrics_bridge("different-svc")
                assert b1 is b2
