# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-013: Outlier Detection Agent

12 Prometheus metrics for outlier detection service monitoring with
graceful fallback when prometheus_client is not installed.

Standard metrics (via MetricsFactory):
    1.  gl_od_operations_total (Counter, labels: type, tenant_id)
    2.  gl_od_processing_duration_seconds (Histogram, 12 buckets)
    3.  gl_od_validation_errors_total (Counter, labels: severity, type)
    4.  gl_od_batch_jobs_total (Counter, labels: status)
    5.  gl_od_active_jobs (Gauge)
    6.  gl_od_queue_size (Gauge)

Agent-specific metrics:
    7.  gl_od_jobs_processed_total (Counter, labels: status)
    8.  gl_od_outliers_detected_total (Counter, labels: method)
    9.  gl_od_outliers_classified_total (Counter, labels: outlier_class)
    10. gl_od_ensemble_score (Histogram, labels: method)
    11. gl_od_detection_confidence (Histogram, labels: method)
    12. gl_od_total_outliers_flagged (Gauge)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
Status: Production Ready
"""

from __future__ import annotations

from greenlang.data_commons.metrics import (
    CONFIDENCE_BUCKETS,
    PROMETHEUS_AVAILABLE,
    MetricsFactory,
)

# ---------------------------------------------------------------------------
# Standard metrics (6 of 12) via factory
# ---------------------------------------------------------------------------

m = MetricsFactory(
    "gl_od",
    "Outlier Detector",
    duration_buckets=(
        0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
        5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
    ),
)

# Backward-compat alias
od_active_jobs = m.active_jobs

# ---------------------------------------------------------------------------
# Agent-specific metrics (6 of 12)
# ---------------------------------------------------------------------------

od_jobs_processed_total = m.create_custom_counter(
    "jobs_processed_total",
    "Total outlier detection jobs processed",
    labelnames=["status"],
)

od_outliers_detected_total = m.create_custom_counter(
    "outliers_detected_total",
    "Total outliers detected",
    labelnames=["method"],
)

od_outliers_classified_total = m.create_custom_counter(
    "outliers_classified_total",
    "Total outliers classified",
    labelnames=["outlier_class"],
)

od_treatments_applied_total = m.create_custom_counter(
    "treatments_applied_total",
    "Total outlier treatments applied",
    labelnames=["strategy"],
)

od_thresholds_evaluated_total = m.create_custom_counter(
    "thresholds_evaluated_total",
    "Total threshold evaluations performed",
    labelnames=["source"],
)

od_feedback_received_total = m.create_custom_counter(
    "feedback_received_total",
    "Total outlier feedback entries received",
    labelnames=["feedback_type"],
)

od_processing_errors_total = m.create_custom_counter(
    "processing_errors_total",
    "Total processing errors encountered",
    labelnames=["error_type"],
)

od_ensemble_score = m.create_custom_histogram(
    "ensemble_score",
    "Ensemble outlier score distribution",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    labelnames=["method"],
)

od_processing_duration_seconds = m.create_custom_histogram(
    "processing_duration_seconds_detail",
    "Outlier detection processing duration in seconds",
    buckets=(
        0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
        5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
    ),
    labelnames=["operation"],
)

od_detection_confidence = m.create_custom_histogram(
    "detection_confidence",
    "Detection confidence score distribution",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    labelnames=["method"],
)

od_total_outliers_flagged = m.create_custom_gauge(
    "total_outliers_flagged",
    "Total outliers currently flagged across active datasets",
)


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def inc_jobs(status: str) -> None:
    """Record a detection job processed event.

    Args:
        status: Job status (completed, failed, cancelled, timeout).
    """
    m.safe_inc(od_jobs_processed_total, 1, status=status)


def inc_outliers_detected(method: str, count: int = 1) -> None:
    """Record outliers detected.

    Args:
        method: Detection method used (iqr, zscore, modified_zscore,
            mad, grubbs, tukey, percentile, lof, isolation_forest,
            mahalanobis, dbscan, contextual, temporal, ensemble).
        count: Number of outliers detected.
    """
    m.safe_inc(od_outliers_detected_total, count, method=method)


def inc_outliers_classified(outlier_class: str, count: int = 1) -> None:
    """Record outliers classified.

    Args:
        outlier_class: Classification (error, genuine_extreme,
            data_entry, regime_change, sensor_fault).
        count: Number of outliers classified.
    """
    m.safe_inc(od_outliers_classified_total, count, outlier_class=outlier_class)


def inc_treatments(strategy: str, count: int = 1) -> None:
    """Record treatments applied.

    Args:
        strategy: Treatment strategy (cap, winsorize, flag, remove,
            replace, investigate).
        count: Number of treatments applied.
    """
    m.safe_inc(od_treatments_applied_total, count, strategy=strategy)


def inc_thresholds(source: str, count: int = 1) -> None:
    """Record threshold evaluations.

    Args:
        source: Threshold source (domain, statistical, regulatory,
            custom, learned).
        count: Number of evaluations.
    """
    m.safe_inc(od_thresholds_evaluated_total, count, source=source)


def inc_feedback(feedback_type: str) -> None:
    """Record a feedback entry received.

    Args:
        feedback_type: Feedback type (confirmed_outlier, false_positive,
            reclassified, unknown).
    """
    m.safe_inc(od_feedback_received_total, 1, feedback_type=feedback_type)


def inc_errors(error_type: str) -> None:
    """Record a processing error event.

    Args:
        error_type: Error classification (validation, timeout, data,
            integration, detection, classification, treatment, unknown).
    """
    m.safe_inc(od_processing_errors_total, 1, error_type=error_type)


def observe_ensemble_score(method: str, score: float) -> None:
    """Record an ensemble outlier score observation.

    Args:
        method: Ensemble method used.
        score: Ensemble score (0.0 - 1.0).
    """
    m.safe_observe(od_ensemble_score, score, method=method)


def observe_duration(operation: str, duration: float) -> None:
    """Record processing duration for an operation.

    Args:
        operation: Operation type (detect, classify, treat, validate,
            document, pipeline, ensemble, contextual, temporal).
        duration: Duration in seconds.
    """
    m.safe_observe(od_processing_duration_seconds, duration, operation=operation)


def observe_confidence(method: str, confidence: float) -> None:
    """Record a detection confidence score observation.

    Args:
        method: Detection method used.
        confidence: Confidence score (0.0 - 1.0).
    """
    m.safe_observe(od_detection_confidence, confidence, method=method)


def set_active_jobs(count: int) -> None:
    """Set the active jobs gauge to an absolute value.

    Args:
        count: Number of currently active detection jobs.
    """
    m.safe_set(od_active_jobs, count)


def set_total_outliers_flagged(count: int) -> None:
    """Set the total outliers flagged gauge.

    Args:
        count: Total number of outliers currently flagged.
    """
    m.safe_set(od_total_outliers_flagged, count)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "m",
    # Metric objects
    "od_jobs_processed_total",
    "od_outliers_detected_total",
    "od_outliers_classified_total",
    "od_treatments_applied_total",
    "od_thresholds_evaluated_total",
    "od_feedback_received_total",
    "od_processing_errors_total",
    "od_ensemble_score",
    "od_processing_duration_seconds",
    "od_detection_confidence",
    "od_active_jobs",
    "od_total_outliers_flagged",
    # Helper functions
    "inc_jobs",
    "inc_outliers_detected",
    "inc_outliers_classified",
    "inc_treatments",
    "inc_thresholds",
    "inc_feedback",
    "inc_errors",
    "observe_ensemble_score",
    "observe_duration",
    "observe_confidence",
    "set_active_jobs",
    "set_total_outliers_flagged",
]
