# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-FOUND-010: Observability Agent

12 Prometheus metrics for observability agent service self-monitoring with graceful
fallback when prometheus_client is not installed.

Metrics:
    1.  gl_obs_metrics_recorded_total (Counter, labels: metric_type, tenant_id)
    2.  gl_obs_operation_duration_seconds (Histogram)
    3.  gl_obs_spans_created_total (Counter, labels: status)
    4.  gl_obs_spans_active (Gauge)
    5.  gl_obs_logs_ingested_total (Counter, labels: level)
    6.  gl_obs_alerts_evaluated_total (Counter, labels: result)
    7.  gl_obs_alerts_firing (Gauge)
    8.  gl_obs_health_checks_total (Counter, labels: status, probe_type)
    9.  gl_obs_health_status (Gauge)
    10. gl_obs_slo_compliance_ratio (Gauge, labels: service_name, slo_type)
    11. gl_obs_error_budget_remaining (Gauge, labels: service_name)
    12. gl_obs_dashboard_queries_total (Counter)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-010 Observability Agent
Status: Production Ready
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful prometheus_client import
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info(
        "prometheus_client not installed; observability agent metrics disabled"
    )


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Metrics recorded by type and tenant
    obs_metrics_recorded_total = Counter(
        "gl_obs_metrics_recorded_total",
        "Total metrics recorded by the observability agent",
        labelnames=["metric_type", "tenant_id"],
    )

    # 2. Operation duration
    obs_operation_duration_seconds = Histogram(
        "gl_obs_operation_duration_seconds",
        "Observability agent operation duration in seconds",
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    # 3. Spans created by status
    obs_spans_created_total = Counter(
        "gl_obs_spans_created_total",
        "Total trace spans created",
        labelnames=["status"],
    )

    # 4. Active spans gauge
    obs_spans_active = Gauge(
        "gl_obs_spans_active",
        "Number of currently active trace spans",
    )

    # 5. Logs ingested by level
    obs_logs_ingested_total = Counter(
        "gl_obs_logs_ingested_total",
        "Total log entries ingested by level",
        labelnames=["level"],
    )

    # 6. Alerts evaluated by result
    obs_alerts_evaluated_total = Counter(
        "gl_obs_alerts_evaluated_total",
        "Total alert evaluations performed",
        labelnames=["result"],
    )

    # 7. Alerts currently firing
    obs_alerts_firing = Gauge(
        "gl_obs_alerts_firing",
        "Number of alerts currently in firing state",
    )

    # 8. Health checks by status and probe type
    obs_health_checks_total = Counter(
        "gl_obs_health_checks_total",
        "Total health checks executed",
        labelnames=["status", "probe_type"],
    )

    # 9. Overall health status gauge
    obs_health_status = Gauge(
        "gl_obs_health_status",
        "Current health status (1.0=healthy, 0.5=degraded, 0.0=unhealthy)",
    )

    # 10. SLO compliance ratio by service and SLO type
    obs_slo_compliance_ratio = Gauge(
        "gl_obs_slo_compliance_ratio",
        "Current SLO compliance ratio per service and SLO type",
        labelnames=["service_name", "slo_type"],
    )

    # 11. Error budget remaining by service
    obs_error_budget_remaining = Gauge(
        "gl_obs_error_budget_remaining",
        "Remaining error budget fraction per service",
        labelnames=["service_name"],
    )

    # 12. Dashboard queries executed
    obs_dashboard_queries_total = Counter(
        "gl_obs_dashboard_queries_total",
        "Total dashboard queries executed",
    )

else:
    # No-op placeholders
    obs_metrics_recorded_total = None  # type: ignore[assignment]
    obs_operation_duration_seconds = None  # type: ignore[assignment]
    obs_spans_created_total = None  # type: ignore[assignment]
    obs_spans_active = None  # type: ignore[assignment]
    obs_logs_ingested_total = None  # type: ignore[assignment]
    obs_alerts_evaluated_total = None  # type: ignore[assignment]
    obs_alerts_firing = None  # type: ignore[assignment]
    obs_health_checks_total = None  # type: ignore[assignment]
    obs_health_status = None  # type: ignore[assignment]
    obs_slo_compliance_ratio = None  # type: ignore[assignment]
    obs_error_budget_remaining = None  # type: ignore[assignment]
    obs_dashboard_queries_total = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_metric_recorded(metric_type: str, tenant_id: str) -> None:
    """Record that a metric was recorded by type and tenant.

    Args:
        metric_type: Type of metric (counter, gauge, histogram, summary).
        tenant_id: Tenant identifier.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    obs_metrics_recorded_total.labels(metric_type=metric_type, tenant_id=tenant_id).inc()


def record_operation_duration(duration_seconds: float) -> None:
    """Record an observability agent operation duration.

    Args:
        duration_seconds: Operation duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    obs_operation_duration_seconds.observe(duration_seconds)


def record_span_created(status: str) -> None:
    """Record a trace span creation with its status.

    Args:
        status: Span status (ok, error, timeout, cancelled).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    obs_spans_created_total.labels(status=status).inc()


def update_active_spans(count: int) -> None:
    """Update the currently active spans gauge.

    Args:
        count: Current number of active spans.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    obs_spans_active.set(count)


def record_log_ingested(level: str) -> None:
    """Record a log entry ingestion by level.

    Args:
        level: Log level (debug, info, warning, error, critical).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    obs_logs_ingested_total.labels(level=level).inc()


def record_alert_evaluated(result: str) -> None:
    """Record an alert evaluation result.

    Args:
        result: Evaluation result (firing, resolved, pending, suppressed).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    obs_alerts_evaluated_total.labels(result=result).inc()


def update_firing_alerts(count: int) -> None:
    """Update the number of currently firing alerts.

    Args:
        count: Current number of firing alerts.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    obs_alerts_firing.set(count)


def record_health_check(status: str, probe_type: str) -> None:
    """Record a health check execution.

    Args:
        status: Health check result (healthy, degraded, unhealthy).
        probe_type: Type of health probe (liveness, readiness, startup).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    obs_health_checks_total.labels(status=status, probe_type=probe_type).inc()


def update_health_status(value: float) -> None:
    """Update the overall health status gauge.

    Args:
        value: Health status value (1.0=healthy, 0.5=degraded, 0.0=unhealthy).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    obs_health_status.set(value)


def update_slo_compliance(service_name: str, slo_type: str, ratio: float) -> None:
    """Update the SLO compliance ratio for a service.

    Args:
        service_name: Name of the monitored service.
        slo_type: SLO type (availability, latency, throughput, error_rate).
        ratio: Compliance ratio (0.0 to 1.0).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    obs_slo_compliance_ratio.labels(service_name=service_name, slo_type=slo_type).set(ratio)


def update_error_budget(service_name: str, remaining: float) -> None:
    """Update the remaining error budget for a service.

    Args:
        service_name: Name of the monitored service.
        remaining: Remaining error budget fraction (0.0 to 1.0).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    obs_error_budget_remaining.labels(service_name=service_name).set(remaining)


def record_dashboard_query() -> None:
    """Record a dashboard query execution."""
    if not PROMETHEUS_AVAILABLE:
        return
    obs_dashboard_queries_total.inc()


__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "obs_metrics_recorded_total",
    "obs_operation_duration_seconds",
    "obs_spans_created_total",
    "obs_spans_active",
    "obs_logs_ingested_total",
    "obs_alerts_evaluated_total",
    "obs_alerts_firing",
    "obs_health_checks_total",
    "obs_health_status",
    "obs_slo_compliance_ratio",
    "obs_error_budget_remaining",
    "obs_dashboard_queries_total",
    # Helper functions
    "record_metric_recorded",
    "record_operation_duration",
    "record_span_created",
    "update_active_spans",
    "record_log_ingested",
    "record_alert_evaluated",
    "update_firing_alerts",
    "record_health_check",
    "update_health_status",
    "update_slo_compliance",
    "update_error_budget",
    "record_dashboard_query",
]
