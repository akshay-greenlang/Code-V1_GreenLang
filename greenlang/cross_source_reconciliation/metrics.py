# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-015: Cross-Source Reconciliation Agent

12 Prometheus metrics for cross-source reconciliation service monitoring
with graceful fallback when prometheus_client is not installed.

Metrics:
    1.  gl_csr_jobs_processed_total (Counter, labels: status)
    2.  gl_csr_records_matched_total (Counter, labels: strategy)
    3.  gl_csr_comparisons_total (Counter, labels: result)
    4.  gl_csr_discrepancies_detected_total (Counter, labels: type, severity)
    5.  gl_csr_resolutions_applied_total (Counter, labels: strategy)
    6.  gl_csr_golden_records_created_total (Counter, labels: status)
    7.  gl_csr_processing_errors_total (Counter, labels: error_type)
    8.  gl_csr_match_confidence (Histogram)
    9.  gl_csr_processing_duration_seconds (Histogram)
    10. gl_csr_discrepancy_magnitude (Histogram)
    11. gl_csr_active_jobs (Gauge)
    12. gl_csr_pending_reviews (Gauge)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation
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
        "cross-source reconciliation metrics disabled"
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
    # 1. Reconciliation jobs processed by status
    csr_jobs_processed_total = Counter(
        "gl_csr_jobs_processed_total",
        "Total cross-source reconciliation jobs processed",
        labelnames=["status"],
    )

    # 2. Records matched by matching strategy
    csr_records_matched_total = Counter(
        "gl_csr_records_matched_total",
        "Total records matched across sources",
        labelnames=["strategy"],
    )

    # 3. Field comparisons by result
    csr_comparisons_total = Counter(
        "gl_csr_comparisons_total",
        "Total field-level comparisons performed",
        labelnames=["result"],
    )

    # 4. Discrepancies detected by type and severity
    csr_discrepancies_detected_total = Counter(
        "gl_csr_discrepancies_detected_total",
        "Total discrepancies detected between sources",
        labelnames=["type", "severity"],
    )

    # 5. Resolutions applied by strategy
    csr_resolutions_applied_total = Counter(
        "gl_csr_resolutions_applied_total",
        "Total conflict resolutions applied",
        labelnames=["strategy"],
    )

    # 6. Golden records created by status
    csr_golden_records_created_total = Counter(
        "gl_csr_golden_records_created_total",
        "Total golden records created from reconciliation",
        labelnames=["status"],
    )

    # 7. Processing errors by error type
    csr_processing_errors_total = Counter(
        "gl_csr_processing_errors_total",
        "Total processing errors encountered",
        labelnames=["error_type"],
    )

    # 8. Match confidence score distribution
    csr_match_confidence = Histogram(
        "gl_csr_match_confidence",
        "Distribution of record match confidence scores",
        buckets=(
            0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0,
        ),
    )

    # 9. Processing duration histogram
    csr_processing_duration_seconds = Histogram(
        "gl_csr_processing_duration_seconds",
        "Cross-source reconciliation processing duration in seconds",
        buckets=(
            0.01, 0.05, 0.1, 0.5, 1.0,
            5.0, 10.0, 30.0, 60.0,
        ),
    )

    # 10. Discrepancy magnitude distribution
    csr_discrepancy_magnitude = Histogram(
        "gl_csr_discrepancy_magnitude",
        "Distribution of discrepancy magnitudes (percentage deviation)",
        buckets=(
            1, 5, 10, 25, 50,
            75, 100, 200, 500,
        ),
    )

    # 11. Currently active reconciliation jobs gauge
    csr_active_jobs = Gauge(
        "gl_csr_active_jobs",
        "Number of currently active cross-source reconciliation jobs",
    )

    # 12. Pending human reviews gauge
    csr_pending_reviews = Gauge(
        "gl_csr_pending_reviews",
        "Number of reconciliation results pending human review",
    )

else:
    # Dummy fallback instances
    csr_jobs_processed_total: Counter = DummyCounter()  # type: ignore[assignment]
    csr_records_matched_total: Counter = DummyCounter()  # type: ignore[assignment]
    csr_comparisons_total: Counter = DummyCounter()  # type: ignore[assignment]
    csr_discrepancies_detected_total: Counter = DummyCounter()  # type: ignore[assignment]
    csr_resolutions_applied_total: Counter = DummyCounter()  # type: ignore[assignment]
    csr_golden_records_created_total: Counter = DummyCounter()  # type: ignore[assignment]
    csr_processing_errors_total: Counter = DummyCounter()  # type: ignore[assignment]
    csr_match_confidence: Histogram = DummyHistogram()  # type: ignore[assignment]
    csr_processing_duration_seconds: Histogram = DummyHistogram()  # type: ignore[assignment]
    csr_discrepancy_magnitude: Histogram = DummyHistogram()  # type: ignore[assignment]
    csr_active_jobs: Gauge = DummyGauge()  # type: ignore[assignment]
    csr_pending_reviews: Gauge = DummyGauge()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def inc_jobs_processed(status: str) -> None:
    """Record a reconciliation job processed event.

    Args:
        status: Job status (completed, failed, cancelled, timeout,
            partial, pending_review).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    csr_jobs_processed_total.labels(status=status).inc()


def inc_records_matched(strategy: str, count: int = 1) -> None:
    """Record records matched across sources.

    Args:
        strategy: Matching strategy used (exact, fuzzy, composite,
            rule_based, ml_assisted, manual).
        count: Number of records matched.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    csr_records_matched_total.labels(strategy=strategy).inc(count)


def inc_comparisons(result: str, count: int = 1) -> None:
    """Record field-level comparison results.

    Args:
        result: Comparison result (match, mismatch, partial_match,
            missing_left, missing_right, type_mismatch, skipped).
        count: Number of comparisons.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    csr_comparisons_total.labels(result=result).inc(count)


def inc_discrepancies(
    discrepancy_type: str,
    severity: str,
    count: int = 1,
) -> None:
    """Record discrepancies detected between sources.

    Args:
        discrepancy_type: Type of discrepancy (value_mismatch,
            missing_record, duplicate, format_difference,
            unit_mismatch, temporal_drift, semantic_conflict).
        severity: Severity level (critical, high, medium, low, info).
        count: Number of discrepancies detected.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    csr_discrepancies_detected_total.labels(
        type=discrepancy_type,
        severity=severity,
    ).inc(count)


def inc_resolutions(strategy: str, count: int = 1) -> None:
    """Record conflict resolutions applied.

    Args:
        strategy: Resolution strategy used (source_priority,
            most_recent, most_complete, average, median,
            manual_override, rule_based, ml_suggested).
        count: Number of resolutions applied.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    csr_resolutions_applied_total.labels(strategy=strategy).inc(count)


def inc_golden_records(status: str, count: int = 1) -> None:
    """Record golden records created.

    Args:
        status: Golden record status (created, updated, merged,
            rejected, pending_review).
        count: Number of golden records.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    csr_golden_records_created_total.labels(status=status).inc(count)


def observe_confidence(confidence: float) -> None:
    """Record a record match confidence score observation.

    Args:
        confidence: Confidence score (0.0 - 1.0).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    csr_match_confidence.observe(confidence)


def observe_duration(duration: float) -> None:
    """Record processing duration for a reconciliation operation.

    Args:
        duration: Duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    csr_processing_duration_seconds.observe(duration)


def observe_magnitude(magnitude: float) -> None:
    """Record the magnitude of a detected discrepancy.

    Args:
        magnitude: Discrepancy magnitude as percentage deviation.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    csr_discrepancy_magnitude.observe(magnitude)


def set_active_jobs(count: int) -> None:
    """Set the active reconciliation jobs gauge to an absolute value.

    Args:
        count: Number of currently active reconciliation jobs.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    csr_active_jobs.set(count)


def set_pending_reviews(count: int) -> None:
    """Set the pending human reviews gauge.

    Args:
        count: Number of reconciliation results pending human review.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    csr_pending_reviews.set(count)


def inc_errors(error_type: str) -> None:
    """Record a processing error event.

    Args:
        error_type: Error classification (validation, timeout, data,
            integration, matching, comparison, resolution, merge,
            golden_record, unknown).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    csr_processing_errors_total.labels(error_type=error_type).inc()


__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "csr_jobs_processed_total",
    "csr_records_matched_total",
    "csr_comparisons_total",
    "csr_discrepancies_detected_total",
    "csr_resolutions_applied_total",
    "csr_golden_records_created_total",
    "csr_processing_errors_total",
    "csr_match_confidence",
    "csr_processing_duration_seconds",
    "csr_discrepancy_magnitude",
    "csr_active_jobs",
    "csr_pending_reviews",
    # Helper functions
    "inc_jobs_processed",
    "inc_records_matched",
    "inc_comparisons",
    "inc_discrepancies",
    "inc_resolutions",
    "inc_golden_records",
    "observe_confidence",
    "observe_duration",
    "observe_magnitude",
    "set_active_jobs",
    "set_pending_reviews",
    "inc_errors",
    # Dummy fallback classes
    "DummyCounter",
    "DummyHistogram",
    "DummyGauge",
]
