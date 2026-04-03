# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-015: Cross-Source Reconciliation Agent

12 Prometheus metrics for cross-source reconciliation service monitoring
with graceful fallback when prometheus_client is not installed.

Standard metrics (via MetricsFactory):
    1.  gl_csr_operations_total (Counter, labels: type, tenant_id)
    2.  gl_csr_processing_duration_seconds (Histogram, 12 buckets)
    3.  gl_csr_validation_errors_total (Counter, labels: severity, type)
    4.  gl_csr_batch_jobs_total (Counter, labels: status)
    5.  gl_csr_active_jobs (Gauge)
    6.  gl_csr_queue_size (Gauge)

Agent-specific metrics:
    7.  gl_csr_jobs_processed_total (Counter, labels: status)
    8.  gl_csr_records_matched_total (Counter, labels: strategy)
    9.  gl_csr_discrepancies_detected_total (Counter, labels: type, severity)
    10. gl_csr_match_confidence (Histogram)
    11. gl_csr_discrepancy_magnitude (Histogram)
    12. gl_csr_pending_reviews (Gauge)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation
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
    "gl_csr",
    "Cross-Source Reconciliation",
    duration_buckets=(
        0.01, 0.05, 0.1, 0.5, 1.0,
        5.0, 10.0, 30.0, 60.0,
    ),
)

# Backward-compat alias
csr_active_jobs = m.active_jobs

# ---------------------------------------------------------------------------
# Agent-specific metrics (6 of 12)
# ---------------------------------------------------------------------------

csr_jobs_processed_total = m.create_custom_counter(
    "jobs_processed_total",
    "Total cross-source reconciliation jobs processed",
    labelnames=["status"],
)

csr_records_matched_total = m.create_custom_counter(
    "records_matched_total",
    "Total records matched across sources",
    labelnames=["strategy"],
)

csr_comparisons_total = m.create_custom_counter(
    "comparisons_total",
    "Total field-level comparisons performed",
    labelnames=["result"],
)

csr_discrepancies_detected_total = m.create_custom_counter(
    "discrepancies_detected_total",
    "Total discrepancies detected between sources",
    labelnames=["type", "severity"],
)

csr_resolutions_applied_total = m.create_custom_counter(
    "resolutions_applied_total",
    "Total conflict resolutions applied",
    labelnames=["strategy"],
)

csr_golden_records_created_total = m.create_custom_counter(
    "golden_records_created_total",
    "Total golden records created from reconciliation",
    labelnames=["status"],
)

csr_processing_errors_total = m.create_custom_counter(
    "processing_errors_total",
    "Total processing errors encountered",
    labelnames=["error_type"],
)

csr_match_confidence = m.create_custom_histogram(
    "match_confidence",
    "Distribution of record match confidence scores",
    buckets=(
        0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0,
    ),
)

csr_processing_duration_seconds = m.create_custom_histogram(
    "processing_duration_seconds_detail",
    "Cross-source reconciliation processing duration in seconds",
    buckets=(
        0.01, 0.05, 0.1, 0.5, 1.0,
        5.0, 10.0, 30.0, 60.0,
    ),
)

csr_discrepancy_magnitude = m.create_custom_histogram(
    "discrepancy_magnitude",
    "Distribution of discrepancy magnitudes (percentage deviation)",
    buckets=(1, 5, 10, 25, 50, 75, 100, 200, 500),
)

csr_pending_reviews = m.create_custom_gauge(
    "pending_reviews",
    "Number of reconciliation results pending human review",
)


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def inc_jobs_processed(status: str) -> None:
    """Record a reconciliation job processed event.

    Args:
        status: Job status (completed, failed, cancelled, timeout,
            partial, pending_review).
    """
    m.safe_inc(csr_jobs_processed_total, 1, status=status)


def inc_records_matched(strategy: str, count: int = 1) -> None:
    """Record records matched across sources.

    Args:
        strategy: Matching strategy used (exact, fuzzy, composite,
            rule_based, ml_assisted, manual).
        count: Number of records matched.
    """
    m.safe_inc(csr_records_matched_total, count, strategy=strategy)


def inc_comparisons(result: str, count: int = 1) -> None:
    """Record field-level comparison results.

    Args:
        result: Comparison result (match, mismatch, partial_match,
            missing_left, missing_right, type_mismatch, skipped).
        count: Number of comparisons.
    """
    m.safe_inc(csr_comparisons_total, count, result=result)


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
    m.safe_inc(
        csr_discrepancies_detected_total, count,
        type=discrepancy_type, severity=severity,
    )


def inc_resolutions(strategy: str, count: int = 1) -> None:
    """Record conflict resolutions applied.

    Args:
        strategy: Resolution strategy used (source_priority,
            most_recent, most_complete, average, median,
            manual_override, rule_based, ml_suggested).
        count: Number of resolutions applied.
    """
    m.safe_inc(csr_resolutions_applied_total, count, strategy=strategy)


def inc_golden_records(status: str, count: int = 1) -> None:
    """Record golden records created.

    Args:
        status: Golden record status (created, updated, merged,
            rejected, pending_review).
        count: Number of golden records.
    """
    m.safe_inc(csr_golden_records_created_total, count, status=status)


def observe_confidence(confidence: float) -> None:
    """Record a record match confidence score observation.

    Args:
        confidence: Confidence score (0.0 - 1.0).
    """
    m.safe_observe(csr_match_confidence, confidence)


def observe_duration(duration: float) -> None:
    """Record processing duration for a reconciliation operation.

    Args:
        duration: Duration in seconds.
    """
    m.safe_observe(csr_processing_duration_seconds, duration)


def observe_magnitude(magnitude: float) -> None:
    """Record the magnitude of a detected discrepancy.

    Args:
        magnitude: Discrepancy magnitude as percentage deviation.
    """
    m.safe_observe(csr_discrepancy_magnitude, magnitude)


def set_active_jobs(count: int) -> None:
    """Set the active reconciliation jobs gauge to an absolute value.

    Args:
        count: Number of currently active reconciliation jobs.
    """
    m.safe_set(csr_active_jobs, count)


def set_pending_reviews(count: int) -> None:
    """Set the pending human reviews gauge.

    Args:
        count: Number of reconciliation results pending human review.
    """
    m.safe_set(csr_pending_reviews, count)


def inc_errors(error_type: str) -> None:
    """Record a processing error event.

    Args:
        error_type: Error classification (validation, timeout, data,
            integration, matching, comparison, resolution, merge,
            golden_record, unknown).
    """
    m.safe_inc(csr_processing_errors_total, 1, error_type=error_type)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "m",
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
]
