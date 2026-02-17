# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-016: Data Freshness Monitor Agent

12 Prometheus metrics for data freshness monitoring service with
graceful fallback when prometheus_client is not installed.

Metrics:
    1.  gl_dfm_checks_performed_total (Counter, labels: dataset, result)
    2.  gl_dfm_sla_breaches_total (Counter, labels: severity)
    3.  gl_dfm_alerts_sent_total (Counter, labels: channel, severity)
    4.  gl_dfm_datasets_registered_total (Counter, labels: status)
    5.  gl_dfm_refresh_events_total (Counter, labels: source)
    6.  gl_dfm_predictions_made_total (Counter, labels: status)
    7.  gl_dfm_freshness_score (Histogram, labels: dataset)
    8.  gl_dfm_data_age_hours (Histogram, labels: dataset)
    9.  gl_dfm_processing_duration_seconds (Histogram, labels: operation)
    10. gl_dfm_active_breaches (Gauge)
    11. gl_dfm_monitored_datasets (Gauge)
    12. gl_dfm_processing_errors_total (Counter, labels: error_type)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
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
        "prometheus_client not installed; "
        "data freshness monitor metrics disabled"
    )


# ---------------------------------------------------------------------------
# Dummy fallback classes
# ---------------------------------------------------------------------------


class _DummyLabeled:
    """Dummy labeled metric that silently discards all observations."""

    def inc(self, amount: float = 1) -> None:
        """No-op increment."""

    def observe(self, amount: float) -> None:
        """No-op observe."""

    def set(self, value: float) -> None:
        """No-op set."""


class DummyCounter:
    """Fallback Counter that silently discards all increments."""

    def labels(self, **kwargs: str) -> _DummyLabeled:
        """Return a dummy labeled metric.

        Args:
            kwargs: Label key-value pairs (ignored).

        Returns:
            A no-op labeled metric instance.
        """
        return _DummyLabeled()

    def inc(self, amount: float = 1) -> None:
        """No-op increment.

        Args:
            amount: Increment amount (ignored).
        """


class DummyHistogram:
    """Fallback Histogram that silently discards all observations."""

    def labels(self, **kwargs: str) -> _DummyLabeled:
        """Return a dummy labeled metric.

        Args:
            kwargs: Label key-value pairs (ignored).

        Returns:
            A no-op labeled metric instance.
        """
        return _DummyLabeled()

    def observe(self, amount: float) -> None:
        """No-op observe.

        Args:
            amount: Observation value (ignored).
        """


class DummyGauge:
    """Fallback Gauge that silently discards all set/inc/dec operations."""

    def labels(self, **kwargs: str) -> _DummyLabeled:
        """Return a dummy labeled metric.

        Args:
            kwargs: Label key-value pairs (ignored).

        Returns:
            A no-op labeled metric instance.
        """
        return _DummyLabeled()

    def set(self, value: float) -> None:
        """No-op set.

        Args:
            value: Gauge value (ignored).
        """

    def inc(self, amount: float = 1) -> None:
        """No-op increment.

        Args:
            amount: Increment amount (ignored).
        """

    def dec(self, amount: float = 1) -> None:
        """No-op decrement.

        Args:
            amount: Decrement amount (ignored).
        """


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Freshness checks performed by dataset and result
    dfm_checks_performed_total = Counter(
        "gl_dfm_checks_performed_total",
        "Total data freshness checks performed",
        labelnames=["dataset", "result"],
    )

    # 2. SLA breaches detected by severity
    dfm_sla_breaches_total = Counter(
        "gl_dfm_sla_breaches_total",
        "Total SLA breaches detected for data freshness",
        labelnames=["severity"],
    )

    # 3. Alerts sent by channel and severity
    dfm_alerts_sent_total = Counter(
        "gl_dfm_alerts_sent_total",
        "Total freshness alerts sent to notification channels",
        labelnames=["channel", "severity"],
    )

    # 4. Datasets registered by status
    dfm_datasets_registered_total = Counter(
        "gl_dfm_datasets_registered_total",
        "Total datasets registered for freshness monitoring",
        labelnames=["status"],
    )

    # 5. Refresh events recorded by source
    dfm_refresh_events_total = Counter(
        "gl_dfm_refresh_events_total",
        "Total data refresh events recorded",
        labelnames=["source"],
    )

    # 6. Predictions made by status
    dfm_predictions_made_total = Counter(
        "gl_dfm_predictions_made_total",
        "Total refresh time predictions made",
        labelnames=["status"],
    )

    # 7. Freshness score distribution by dataset
    dfm_freshness_score = Histogram(
        "gl_dfm_freshness_score",
        "Distribution of data freshness scores (0.0 - 1.0)",
        labelnames=["dataset"],
        buckets=(
            0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0,
        ),
    )

    # 8. Data age distribution in hours by dataset
    dfm_data_age_hours = Histogram(
        "gl_dfm_data_age_hours",
        "Distribution of data age in hours since last refresh",
        labelnames=["dataset"],
        buckets=(
            0.5, 1.0, 2.0, 4.0, 8.0,
            12.0, 24.0, 48.0, 72.0, 168.0, 720.0,
        ),
    )

    # 9. Processing duration histogram by operation
    dfm_processing_duration_seconds = Histogram(
        "gl_dfm_processing_duration_seconds",
        "Data freshness monitor processing duration in seconds",
        labelnames=["operation"],
        buckets=(
            0.01, 0.05, 0.1, 0.5, 1.0,
            5.0, 10.0, 30.0, 60.0,
        ),
    )

    # 10. Currently active SLA breaches gauge
    dfm_active_breaches = Gauge(
        "gl_dfm_active_breaches",
        "Number of currently active data freshness SLA breaches",
    )

    # 11. Number of monitored datasets gauge
    dfm_monitored_datasets = Gauge(
        "gl_dfm_monitored_datasets",
        "Number of datasets currently monitored for freshness",
    )

    # 12. Processing errors by error type
    dfm_processing_errors_total = Counter(
        "gl_dfm_processing_errors_total",
        "Total processing errors encountered in freshness monitoring",
        labelnames=["error_type"],
    )

else:
    # Dummy fallback instances
    dfm_checks_performed_total: Counter = DummyCounter()  # type: ignore[assignment]
    dfm_sla_breaches_total: Counter = DummyCounter()  # type: ignore[assignment]
    dfm_alerts_sent_total: Counter = DummyCounter()  # type: ignore[assignment]
    dfm_datasets_registered_total: Counter = DummyCounter()  # type: ignore[assignment]
    dfm_refresh_events_total: Counter = DummyCounter()  # type: ignore[assignment]
    dfm_predictions_made_total: Counter = DummyCounter()  # type: ignore[assignment]
    dfm_freshness_score: Histogram = DummyHistogram()  # type: ignore[assignment]
    dfm_data_age_hours: Histogram = DummyHistogram()  # type: ignore[assignment]
    dfm_processing_duration_seconds: Histogram = DummyHistogram()  # type: ignore[assignment]
    dfm_active_breaches: Gauge = DummyGauge()  # type: ignore[assignment]
    dfm_monitored_datasets: Gauge = DummyGauge()  # type: ignore[assignment]
    dfm_processing_errors_total: Counter = DummyCounter()  # type: ignore[assignment]


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
    if not PROMETHEUS_AVAILABLE:
        return
    dfm_checks_performed_total.labels(dataset=dataset, result=result).inc()


def record_breach(severity: str, count: int = 1) -> None:
    """Record an SLA breach detected event.

    Args:
        severity: Breach severity (critical, high, medium, low, info).
        count: Number of breaches detected.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dfm_sla_breaches_total.labels(severity=severity).inc(count)


def record_alert(channel: str, severity: str) -> None:
    """Record a freshness alert sent event.

    Args:
        channel: Notification channel (email, slack, pagerduty,
            opsgenie, teams, webhook).
        severity: Alert severity (critical, high, medium, low, info).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dfm_alerts_sent_total.labels(channel=channel, severity=severity).inc()


def record_dataset_registered(status: str, count: int = 1) -> None:
    """Record a dataset registration event.

    Args:
        status: Registration status (active, inactive, paused,
            deregistered, pending).
        count: Number of datasets registered.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dfm_datasets_registered_total.labels(status=status).inc(count)


def record_refresh_event(source: str, count: int = 1) -> None:
    """Record a data refresh event.

    Args:
        source: Data source that refreshed (erp, api, file_upload,
            scheduled_etl, manual, streaming, webhook).
        count: Number of refresh events.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dfm_refresh_events_total.labels(source=source).inc(count)


def record_prediction(status: str) -> None:
    """Record a refresh prediction made event.

    Args:
        status: Prediction status (accurate, inaccurate, pending,
            expired, failed).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dfm_predictions_made_total.labels(status=status).inc()


def observe_freshness_score(dataset: str, score: float) -> None:
    """Record a freshness score observation for a dataset.

    Args:
        dataset: Name of the dataset.
        score: Freshness score (0.0 - 1.0).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dfm_freshness_score.labels(dataset=dataset).observe(score)


def observe_data_age(dataset: str, age_hours: float) -> None:
    """Record data age in hours for a dataset.

    Args:
        dataset: Name of the dataset.
        age_hours: Age of the data in hours since last refresh.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dfm_data_age_hours.labels(dataset=dataset).observe(age_hours)


def observe_duration(operation: str, duration: float) -> None:
    """Record processing duration for a freshness monitoring operation.

    Args:
        operation: Operation name (check_freshness, evaluate_sla,
            detect_breach, send_alert, predict_refresh,
            register_dataset, record_refresh).
        duration: Duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dfm_processing_duration_seconds.labels(operation=operation).observe(duration)


def set_active_breaches(count: int) -> None:
    """Set the active SLA breaches gauge to an absolute value.

    Args:
        count: Number of currently active SLA breaches.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dfm_active_breaches.set(count)


def set_monitored_datasets(count: int) -> None:
    """Set the monitored datasets gauge to an absolute value.

    Args:
        count: Number of datasets currently monitored for freshness.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dfm_monitored_datasets.set(count)


def record_error(error_type: str) -> None:
    """Record a processing error event.

    Args:
        error_type: Error classification (validation, timeout, data,
            integration, sla_evaluation, prediction, alerting,
            registration, refresh_tracking, unknown).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dfm_processing_errors_total.labels(error_type=error_type).inc()


__all__ = [
    "PROMETHEUS_AVAILABLE",
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
    # Dummy fallback classes
    "DummyCounter",
    "DummyHistogram",
    "DummyGauge",
]
