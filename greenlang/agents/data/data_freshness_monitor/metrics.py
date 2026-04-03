# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-016: Data Freshness Monitor Agent

12 Prometheus metrics for data freshness monitoring service with
graceful fallback when prometheus_client is not installed.

Standard metrics (via MetricsFactory):
    1.  gl_dfm_operations_total (Counter, labels: type, tenant_id)
    2.  gl_dfm_processing_duration_seconds (Histogram, 12 buckets)
    3.  gl_dfm_validation_errors_total (Counter, labels: severity, type)
    4.  gl_dfm_batch_jobs_total (Counter, labels: status)
    5.  gl_dfm_active_jobs (Gauge)
    6.  gl_dfm_queue_size (Gauge)

Agent-specific metrics:
    7.  gl_dfm_checks_performed_total (Counter, labels: dataset, result)
    8.  gl_dfm_sla_breaches_total (Counter, labels: severity)
    9.  gl_dfm_alerts_sent_total (Counter, labels: channel, severity)
    10. gl_dfm_freshness_score (Histogram, labels: dataset)
    11. gl_dfm_data_age_hours (Histogram, labels: dataset)
    12. gl_dfm_active_breaches (Gauge)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
Status: Production Ready
"""

from __future__ import annotations

from greenlang.data_commons.metrics import (
    PROMETHEUS_AVAILABLE,
    MetricsFactory,
)

# ---------------------------------------------------------------------------
# Standard metrics (6 of 12) via factory
# ---------------------------------------------------------------------------

m = MetricsFactory(
    "gl_dfm",
    "Data Freshness Monitor",
    duration_buckets=(
        0.01, 0.05, 0.1, 0.5, 1.0,
        5.0, 10.0, 30.0, 60.0,
    ),
)

# ---------------------------------------------------------------------------
# Agent-specific metrics (6 of 12)
# ---------------------------------------------------------------------------

dfm_checks_performed_total = m.create_custom_counter(
    "checks_performed_total",
    "Total data freshness checks performed",
    labelnames=["dataset", "result"],
)

dfm_sla_breaches_total = m.create_custom_counter(
    "sla_breaches_total",
    "Total SLA breaches detected for data freshness",
    labelnames=["severity"],
)

dfm_alerts_sent_total = m.create_custom_counter(
    "alerts_sent_total",
    "Total freshness alerts sent to notification channels",
    labelnames=["channel", "severity"],
)

dfm_datasets_registered_total = m.create_custom_counter(
    "datasets_registered_total",
    "Total datasets registered for freshness monitoring",
    labelnames=["status"],
)

dfm_refresh_events_total = m.create_custom_counter(
    "refresh_events_total",
    "Total data refresh events recorded",
    labelnames=["source"],
)

dfm_predictions_made_total = m.create_custom_counter(
    "predictions_made_total",
    "Total refresh time predictions made",
    labelnames=["status"],
)

dfm_freshness_score = m.create_custom_histogram(
    "freshness_score",
    "Distribution of data freshness scores (0.0 - 1.0)",
    buckets=(
        0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0,
    ),
    labelnames=["dataset"],
)

dfm_data_age_hours = m.create_custom_histogram(
    "data_age_hours",
    "Distribution of data age in hours since last refresh",
    buckets=(
        0.5, 1.0, 2.0, 4.0, 8.0,
        12.0, 24.0, 48.0, 72.0, 168.0, 720.0,
    ),
    labelnames=["dataset"],
)

dfm_processing_duration_seconds = m.create_custom_histogram(
    "processing_duration_seconds_detail",
    "Data freshness monitor processing duration in seconds",
    buckets=(
        0.01, 0.05, 0.1, 0.5, 1.0,
        5.0, 10.0, 30.0, 60.0,
    ),
    labelnames=["operation"],
)

dfm_active_breaches = m.create_custom_gauge(
    "active_breaches",
    "Number of currently active data freshness SLA breaches",
)

dfm_monitored_datasets = m.create_custom_gauge(
    "monitored_datasets",
    "Number of datasets currently monitored for freshness",
)

dfm_processing_errors_total = m.create_custom_counter(
    "processing_errors_total",
    "Total processing errors encountered in freshness monitoring",
    labelnames=["error_type"],
)


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_check(dataset: str, result: str) -> None:
    """Record a freshness check performed event.

    Args:
        dataset: Name of the dataset checked.
        result: Check result (fresh, stale, warning, critical,
            unknown, skipped).
    """
    m.safe_inc(dfm_checks_performed_total, 1, dataset=dataset, result=result)


def record_breach(severity: str, count: int = 1) -> None:
    """Record an SLA breach detected event.

    Args:
        severity: Breach severity (critical, high, medium, low, info).
        count: Number of breaches detected.
    """
    m.safe_inc(dfm_sla_breaches_total, count, severity=severity)


def record_alert(channel: str, severity: str) -> None:
    """Record a freshness alert sent event.

    Args:
        channel: Notification channel (email, slack, pagerduty,
            opsgenie, teams, webhook).
        severity: Alert severity (critical, high, medium, low, info).
    """
    m.safe_inc(dfm_alerts_sent_total, 1, channel=channel, severity=severity)


def record_dataset_registered(status: str, count: int = 1) -> None:
    """Record a dataset registration event.

    Args:
        status: Registration status (active, inactive, paused,
            deregistered, pending).
        count: Number of datasets registered.
    """
    m.safe_inc(dfm_datasets_registered_total, count, status=status)


def record_refresh_event(source: str, count: int = 1) -> None:
    """Record a data refresh event.

    Args:
        source: Data source that refreshed (erp, api, file_upload,
            scheduled_etl, manual, streaming, webhook).
        count: Number of refresh events.
    """
    m.safe_inc(dfm_refresh_events_total, count, source=source)


def record_prediction(status: str) -> None:
    """Record a refresh prediction made event.

    Args:
        status: Prediction status (accurate, inaccurate, pending,
            expired, failed).
    """
    m.safe_inc(dfm_predictions_made_total, 1, status=status)


def observe_freshness_score(dataset: str, score: float) -> None:
    """Record a freshness score observation for a dataset.

    Args:
        dataset: Name of the dataset.
        score: Freshness score (0.0 - 1.0).
    """
    m.safe_observe(dfm_freshness_score, score, dataset=dataset)


def observe_data_age(dataset: str, age_hours: float) -> None:
    """Record data age in hours for a dataset.

    Args:
        dataset: Name of the dataset.
        age_hours: Age of the data in hours since last refresh.
    """
    m.safe_observe(dfm_data_age_hours, age_hours, dataset=dataset)


def observe_duration(operation: str, duration: float) -> None:
    """Record processing duration for a freshness monitoring operation.

    Args:
        operation: Operation name (check_freshness, evaluate_sla,
            detect_breach, send_alert, predict_refresh,
            register_dataset, record_refresh).
        duration: Duration in seconds.
    """
    m.safe_observe(dfm_processing_duration_seconds, duration, operation=operation)


def set_active_breaches(count: int) -> None:
    """Set the active SLA breaches gauge to an absolute value.

    Args:
        count: Number of currently active SLA breaches.
    """
    m.safe_set(dfm_active_breaches, count)


def set_monitored_datasets(count: int) -> None:
    """Set the monitored datasets gauge to an absolute value.

    Args:
        count: Number of datasets currently monitored for freshness.
    """
    m.safe_set(dfm_monitored_datasets, count)


def record_error(error_type: str) -> None:
    """Record a processing error event.

    Args:
        error_type: Error classification (validation, timeout, data,
            integration, sla_evaluation, prediction, alerting,
            registration, refresh_tracking, unknown).
    """
    m.safe_inc(dfm_processing_errors_total, 1, error_type=error_type)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "m",
    # Metric objects
    "dfm_checks_performed_total",
    "dfm_sla_breaches_total",
    "dfm_alerts_sent_total",
    "dfm_datasets_registered_total",
    "dfm_refresh_events_total",
    "dfm_predictions_made_total",
    "dfm_freshness_score",
    "dfm_data_age_hours",
    "dfm_processing_duration_seconds",
    "dfm_active_breaches",
    "dfm_monitored_datasets",
    "dfm_processing_errors_total",
    # Helper functions
    "record_check",
    "record_breach",
    "record_alert",
    "record_dataset_registered",
    "record_refresh_event",
    "record_prediction",
    "observe_freshness_score",
    "observe_data_age",
    "observe_duration",
    "set_active_breaches",
    "set_monitored_datasets",
    "record_error",
]
