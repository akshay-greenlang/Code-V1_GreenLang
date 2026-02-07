# -*- coding: utf-8 -*-
"""
MetricsBridge - Trace-derived Prometheus metrics for GreenLang (OBS-003)

Bridges the gap between distributed traces and Prometheus metrics by recording
span-level measurements as Prometheus Histogram / Counter values.  This
enables SLI/SLO dashboards (p50/p95/p99 latencies, error rates) without
requiring a separate metrics pipeline.

The bridge is instantiated once per service and called from decorators,
middleware, or explicit span hooks.

All Prometheus client access is behind a try/except so the module loads
cleanly in environments without ``prometheus_client``.

Example:
    >>> from greenlang.infrastructure.tracing_service.metrics_bridge import MetricsBridge
    >>> bridge = MetricsBridge(service_name="api-service")
    >>> bridge.record_span("calculate_emissions", 0.045, status="ok")
    >>> bridge.record_error("calculate_emissions", "ValueError")

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Prometheus imports
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Histogram, Counter, Gauge

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Metric definitions (module-level singletons)
# ---------------------------------------------------------------------------

# Span duration histogram
_SPAN_DURATION: Optional[Any] = None
# Total spans counter
_SPAN_COUNT: Optional[Any] = None
# Error counter
_TRACE_ERRORS: Optional[Any] = None
# Active spans gauge
_ACTIVE_SPANS: Optional[Any] = None
# Exported spans counter
_EXPORTED_SPANS: Optional[Any] = None

if PROMETHEUS_AVAILABLE:
    _SPAN_DURATION = Histogram(
        "gl_trace_span_duration_seconds",
        "Duration of traced spans in seconds",
        ["service", "operation", "status"],
        buckets=[
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1,
            0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
        ],
    )

    _SPAN_COUNT = Counter(
        "gl_trace_spans_total",
        "Total number of spans created",
        ["service", "operation", "status"],
    )

    _TRACE_ERRORS = Counter(
        "gl_trace_errors_total",
        "Total number of trace errors",
        ["service", "operation", "error_type"],
    )

    _ACTIVE_SPANS = Gauge(
        "gl_trace_active_spans",
        "Number of currently active (in-flight) spans",
        ["service"],
    )

    _EXPORTED_SPANS = Counter(
        "gl_trace_exported_spans_total",
        "Total number of spans exported to the collector",
        ["service", "exporter"],
    )


# ---------------------------------------------------------------------------
# Any type for Prometheus unavailability
# ---------------------------------------------------------------------------
from typing import Any


# ---------------------------------------------------------------------------
# MetricsBridge
# ---------------------------------------------------------------------------

class MetricsBridge:
    """Bridge between OpenTelemetry traces and Prometheus metrics.

    Creates Prometheus observations from span lifecycle events so that
    latency histograms and error counters are available in Grafana without
    a separate metrics pipeline.

    Attributes:
        service_name: Logical service name used as a label.
    """

    def __init__(self, service_name: str = "greenlang") -> None:
        """Initialise the bridge.

        Args:
            service_name: Service name for Prometheus labels.
        """
        self.service_name = service_name
        if not PROMETHEUS_AVAILABLE:
            logger.debug(
                "prometheus_client not installed; MetricsBridge is no-op"
            )

    # ---- Span recording -----------------------------------------------------

    def record_span(
        self,
        operation: str,
        duration_seconds: float,
        status: str = "ok",
    ) -> None:
        """Record a completed span as Prometheus metrics.

        Args:
            operation: Span/operation name.
            duration_seconds: Span duration in seconds.
            status: Span status string ("ok" or "error").
        """
        if _SPAN_DURATION is not None:
            _SPAN_DURATION.labels(
                service=self.service_name,
                operation=operation,
                status=status,
            ).observe(duration_seconds)

        if _SPAN_COUNT is not None:
            _SPAN_COUNT.labels(
                service=self.service_name,
                operation=operation,
                status=status,
            ).inc()

    def record_error(
        self,
        operation: str,
        error_type: str,
    ) -> None:
        """Record a span error.

        Args:
            operation: Span/operation name.
            error_type: Exception class name or error category.
        """
        if _TRACE_ERRORS is not None:
            _TRACE_ERRORS.labels(
                service=self.service_name,
                operation=operation,
                error_type=error_type,
            ).inc()

    # ---- Active-span tracking ------------------------------------------------

    def span_started(self) -> None:
        """Increment the active-spans gauge."""
        if _ACTIVE_SPANS is not None:
            _ACTIVE_SPANS.labels(service=self.service_name).inc()

    def span_ended(self) -> None:
        """Decrement the active-spans gauge."""
        if _ACTIVE_SPANS is not None:
            _ACTIVE_SPANS.labels(service=self.service_name).dec()

    # ---- Export tracking -----------------------------------------------------

    def record_export(self, count: int, exporter: str = "otlp") -> None:
        """Record exported span count.

        Args:
            count: Number of spans exported.
            exporter: Exporter name ("otlp", "console").
        """
        if _EXPORTED_SPANS is not None:
            _EXPORTED_SPANS.labels(
                service=self.service_name,
                exporter=exporter,
            ).inc(count)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_bridge: Optional[MetricsBridge] = None


def get_metrics_bridge(service_name: str = "greenlang") -> MetricsBridge:
    """Return a shared MetricsBridge instance.

    Args:
        service_name: Service name for Prometheus labels.

    Returns:
        A MetricsBridge singleton.
    """
    global _bridge
    if _bridge is None:
        _bridge = MetricsBridge(service_name=service_name)
    return _bridge


__all__ = [
    "MetricsBridge",
    "get_metrics_bridge",
    "PROMETHEUS_AVAILABLE",
]
