# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-013: Outlier Detection Agent

12 Prometheus metrics for outlier detection service monitoring with
graceful fallback when prometheus_client is not installed.

Metrics:
    1.  gl_od_jobs_processed_total (Counter, labels: status)
    2.  gl_od_outliers_detected_total (Counter, labels: method)
    3.  gl_od_outliers_classified_total (Counter, labels: outlier_class)
    4.  gl_od_treatments_applied_total (Counter, labels: strategy)
    5.  gl_od_thresholds_evaluated_total (Counter, labels: source)
    6.  gl_od_feedback_received_total (Counter, labels: feedback_type)
    7.  gl_od_processing_errors_total (Counter, labels: error_type)
    8.  gl_od_ensemble_score (Histogram, labels: method)
    9.  gl_od_processing_duration_seconds (Histogram, labels: operation)
    10. gl_od_detection_confidence (Histogram, labels: method)
    11. gl_od_active_jobs (Gauge)
    12. gl_od_total_outliers_flagged (Gauge)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
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
        "prometheus_client not installed; outlier detector metrics disabled"
    )


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Detection jobs processed by status
    od_jobs_processed_total = Counter(
        "gl_od_jobs_processed_total",
        "Total outlier detection jobs processed",
        labelnames=["status"],
    )

    # 2. Outliers detected by detection method
    od_outliers_detected_total = Counter(
        "gl_od_outliers_detected_total",
        "Total outliers detected",
        labelnames=["method"],
    )

    # 3. Outliers classified by class
    od_outliers_classified_total = Counter(
        "gl_od_outliers_classified_total",
        "Total outliers classified",
        labelnames=["outlier_class"],
    )

    # 4. Treatments applied by strategy
    od_treatments_applied_total = Counter(
        "gl_od_treatments_applied_total",
        "Total outlier treatments applied",
        labelnames=["strategy"],
    )

    # 5. Thresholds evaluated by source
    od_thresholds_evaluated_total = Counter(
        "gl_od_thresholds_evaluated_total",
        "Total threshold evaluations performed",
        labelnames=["source"],
    )

    # 6. Feedback received by type
    od_feedback_received_total = Counter(
        "gl_od_feedback_received_total",
        "Total outlier feedback entries received",
        labelnames=["feedback_type"],
    )

    # 7. Processing errors by error type
    od_processing_errors_total = Counter(
        "gl_od_processing_errors_total",
        "Total processing errors encountered",
        labelnames=["error_type"],
    )

    # 8. Ensemble score distribution by method
    od_ensemble_score = Histogram(
        "gl_od_ensemble_score",
        "Ensemble outlier score distribution",
        labelnames=["method"],
        buckets=(
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 1.0,
        ),
    )

    # 9. Processing duration histogram by operation type
    od_processing_duration_seconds = Histogram(
        "gl_od_processing_duration_seconds",
        "Outlier detection processing duration in seconds",
        labelnames=["operation"],
        buckets=(
            0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
            5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
        ),
    )

    # 10. Detection confidence histogram by method
    od_detection_confidence = Histogram(
        "gl_od_detection_confidence",
        "Detection confidence score distribution",
        labelnames=["method"],
        buckets=(
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 1.0,
        ),
    )

    # 11. Currently active detection jobs gauge
    od_active_jobs = Gauge(
        "gl_od_active_jobs",
        "Number of currently active outlier detection jobs",
    )

    # 12. Total outliers flagged gauge
    od_total_outliers_flagged = Gauge(
        "gl_od_total_outliers_flagged",
        "Total outliers currently flagged across active datasets",
    )

else:
    # No-op placeholders
    od_jobs_processed_total = None  # type: ignore[assignment]
    od_outliers_detected_total = None  # type: ignore[assignment]
    od_outliers_classified_total = None  # type: ignore[assignment]
    od_treatments_applied_total = None  # type: ignore[assignment]
    od_thresholds_evaluated_total = None  # type: ignore[assignment]
    od_feedback_received_total = None  # type: ignore[assignment]
    od_processing_errors_total = None  # type: ignore[assignment]
    od_ensemble_score = None  # type: ignore[assignment]
    od_processing_duration_seconds = None  # type: ignore[assignment]
    od_detection_confidence = None  # type: ignore[assignment]
    od_active_jobs = None  # type: ignore[assignment]
    od_total_outliers_flagged = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def inc_jobs(status: str) -> None:
    """Record a detection job processed event.

    Args:
        status: Job status (completed, failed, cancelled, timeout).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    od_jobs_processed_total.labels(
        status=status,
    ).inc()


def inc_outliers_detected(method: str, count: int = 1) -> None:
    """Record outliers detected.

    Args:
        method: Detection method used (iqr, zscore, modified_zscore,
            mad, grubbs, tukey, percentile, lof, isolation_forest,
            mahalanobis, dbscan, contextual, temporal, ensemble).
        count: Number of outliers detected.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    od_outliers_detected_total.labels(
        method=method,
    ).inc(count)


def inc_outliers_classified(outlier_class: str, count: int = 1) -> None:
    """Record outliers classified.

    Args:
        outlier_class: Classification (error, genuine_extreme,
            data_entry, regime_change, sensor_fault).
        count: Number of outliers classified.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    od_outliers_classified_total.labels(
        outlier_class=outlier_class,
    ).inc(count)


def inc_treatments(strategy: str, count: int = 1) -> None:
    """Record treatments applied.

    Args:
        strategy: Treatment strategy (cap, winsorize, flag, remove,
            replace, investigate).
        count: Number of treatments applied.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    od_treatments_applied_total.labels(
        strategy=strategy,
    ).inc(count)


def inc_thresholds(source: str, count: int = 1) -> None:
    """Record threshold evaluations.

    Args:
        source: Threshold source (domain, statistical, regulatory,
            custom, learned).
        count: Number of evaluations.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    od_thresholds_evaluated_total.labels(
        source=source,
    ).inc(count)


def inc_feedback(feedback_type: str) -> None:
    """Record a feedback entry received.

    Args:
        feedback_type: Feedback type (confirmed_outlier, false_positive,
            reclassified, unknown).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    od_feedback_received_total.labels(
        feedback_type=feedback_type,
    ).inc()


def inc_errors(error_type: str) -> None:
    """Record a processing error event.

    Args:
        error_type: Error classification (validation, timeout, data,
            integration, detection, classification, treatment, unknown).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    od_processing_errors_total.labels(
        error_type=error_type,
    ).inc()


def observe_ensemble_score(method: str, score: float) -> None:
    """Record an ensemble outlier score observation.

    Args:
        method: Ensemble method used.
        score: Ensemble score (0.0 - 1.0).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    od_ensemble_score.labels(
        method=method,
    ).observe(score)


def observe_duration(operation: str, duration: float) -> None:
    """Record processing duration for an operation.

    Args:
        operation: Operation type (detect, classify, treat, validate,
            document, pipeline, ensemble, contextual, temporal).
        duration: Duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    od_processing_duration_seconds.labels(
        operation=operation,
    ).observe(duration)


def observe_confidence(method: str, confidence: float) -> None:
    """Record a detection confidence score observation.

    Args:
        method: Detection method used.
        confidence: Confidence score (0.0 - 1.0).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    od_detection_confidence.labels(
        method=method,
    ).observe(confidence)


def set_active_jobs(count: int) -> None:
    """Set the active jobs gauge to an absolute value.

    Args:
        count: Number of currently active detection jobs.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    od_active_jobs.set(count)


def set_total_outliers_flagged(count: int) -> None:
    """Set the total outliers flagged gauge.

    Args:
        count: Total number of outliers currently flagged.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    od_total_outliers_flagged.set(count)


__all__ = [
    "PROMETHEUS_AVAILABLE",
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
