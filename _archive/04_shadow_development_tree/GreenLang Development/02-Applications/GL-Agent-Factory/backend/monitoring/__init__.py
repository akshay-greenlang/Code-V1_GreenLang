"""
GreenLang Monitoring Module

Production-grade observability for the GreenLang Agent Factory.
Provides Prometheus metrics, distributed tracing, and structured logging.
"""

from backend.monitoring.metrics import (
    # Counters
    calculations_total,
    ef_lookups_total,
    errors_total,
    http_requests_total,
    # Histograms
    calculation_duration,
    ef_lookup_duration,
    http_request_duration,
    # Gauges
    active_agents,
    registry_agents_count,
    active_calculations,
    cache_size,
    # Helper functions
    track_calculation,
    track_ef_lookup,
    track_error,
    track_http_request,
)

from backend.monitoring.middleware import (
    PrometheusMiddleware,
    setup_metrics_endpoint,
)

__all__ = [
    # Counters
    "calculations_total",
    "ef_lookups_total",
    "errors_total",
    "http_requests_total",
    # Histograms
    "calculation_duration",
    "ef_lookup_duration",
    "http_request_duration",
    # Gauges
    "active_agents",
    "registry_agents_count",
    "active_calculations",
    "cache_size",
    # Helper functions
    "track_calculation",
    "track_ef_lookup",
    "track_error",
    "track_http_request",
    # Middleware
    "PrometheusMiddleware",
    "setup_metrics_endpoint",
]
