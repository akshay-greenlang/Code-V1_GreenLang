# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-010: Data Quality Profiler

12 Prometheus metrics for data quality profiler service monitoring with
graceful fallback when prometheus_client is not installed.

Metrics:
    1.  gl_dq_datasets_profiled_total (Counter, labels: source)
    2.  gl_dq_columns_profiled_total (Counter, labels: data_type)
    3.  gl_dq_assessments_completed_total (Counter, labels: quality_level)
    4.  gl_dq_rules_evaluated_total (Counter, labels: result)
    5.  gl_dq_anomalies_detected_total (Counter, labels: method)
    6.  gl_dq_gates_evaluated_total (Counter, labels: outcome)
    7.  gl_dq_overall_quality_score (Histogram, buckets: 0.1-1.0)
    8.  gl_dq_processing_duration_seconds (Histogram, labels: operation)
    9.  gl_dq_active_profiles (Gauge)
    10. gl_dq_total_issues_found (Gauge)
    11. gl_dq_processing_errors_total (Counter, labels: error_type)
    12. gl_dq_freshness_checks_total (Counter, labels: status)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-010 Data Quality Profiler (GL-DATA-X-013)
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
        "prometheus_client not installed; data quality profiler metrics disabled"
    )


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Datasets profiled by source
    dq_datasets_profiled_total = Counter(
        "gl_dq_datasets_profiled_total",
        "Total datasets profiled",
        labelnames=["source"],
    )

    # 2. Columns profiled by detected data type
    dq_columns_profiled_total = Counter(
        "gl_dq_columns_profiled_total",
        "Total columns profiled",
        labelnames=["data_type"],
    )

    # 3. Quality assessments completed by quality level
    dq_assessments_completed_total = Counter(
        "gl_dq_assessments_completed_total",
        "Total quality assessments completed",
        labelnames=["quality_level"],
    )

    # 4. Quality rules evaluated by result
    dq_rules_evaluated_total = Counter(
        "gl_dq_rules_evaluated_total",
        "Total quality rules evaluated",
        labelnames=["result"],
    )

    # 5. Anomalies detected by detection method
    dq_anomalies_detected_total = Counter(
        "gl_dq_anomalies_detected_total",
        "Total anomalies detected",
        labelnames=["method"],
    )

    # 6. Quality gates evaluated by outcome
    dq_gates_evaluated_total = Counter(
        "gl_dq_gates_evaluated_total",
        "Total quality gates evaluated",
        labelnames=["outcome"],
    )

    # 7. Overall quality score distribution
    dq_overall_quality_score = Histogram(
        "gl_dq_overall_quality_score",
        "Overall quality score distribution",
        buckets=(
            0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 1.0,
        ),
    )

    # 8. Processing duration histogram by operation type
    dq_processing_duration_seconds = Histogram(
        "gl_dq_processing_duration_seconds",
        "Data quality profiler processing duration in seconds",
        labelnames=["operation"],
        buckets=(
            0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
            5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
        ),
    )

    # 9. Currently active profiles gauge
    dq_active_profiles = Gauge(
        "gl_dq_active_profiles",
        "Number of currently active profiling operations",
    )

    # 10. Total issues found gauge
    dq_total_issues_found = Gauge(
        "gl_dq_total_issues_found",
        "Total cumulative data quality issues found",
    )

    # 11. Processing errors by error type
    dq_processing_errors_total = Counter(
        "gl_dq_processing_errors_total",
        "Total processing errors encountered",
        labelnames=["error_type"],
    )

    # 12. Freshness checks by status
    dq_freshness_checks_total = Counter(
        "gl_dq_freshness_checks_total",
        "Total freshness checks performed",
        labelnames=["status"],
    )

else:
    # No-op placeholders
    dq_datasets_profiled_total = None  # type: ignore[assignment]
    dq_columns_profiled_total = None  # type: ignore[assignment]
    dq_assessments_completed_total = None  # type: ignore[assignment]
    dq_rules_evaluated_total = None  # type: ignore[assignment]
    dq_anomalies_detected_total = None  # type: ignore[assignment]
    dq_gates_evaluated_total = None  # type: ignore[assignment]
    dq_overall_quality_score = None  # type: ignore[assignment]
    dq_processing_duration_seconds = None  # type: ignore[assignment]
    dq_active_profiles = None  # type: ignore[assignment]
    dq_total_issues_found = None  # type: ignore[assignment]
    dq_processing_errors_total = None  # type: ignore[assignment]
    dq_freshness_checks_total = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_profile(source: str) -> None:
    """Record a dataset profiling event.

    Args:
        source: Profiling source (csv, excel, api, erp, manual).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dq_datasets_profiled_total.labels(
        source=source,
    ).inc()


def record_column_profile(data_type: str) -> None:
    """Record a column profiling event.

    Args:
        data_type: Detected data type (string, integer, float, date, boolean, etc.).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dq_columns_profiled_total.labels(
        data_type=data_type,
    ).inc()


def record_assessment(quality_level: str) -> None:
    """Record a quality assessment completion event.

    Args:
        quality_level: Quality level result (EXCELLENT, GOOD, FAIR, POOR, CRITICAL).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dq_assessments_completed_total.labels(
        quality_level=quality_level,
    ).inc()


def record_rule_evaluation(result: str) -> None:
    """Record a quality rule evaluation event.

    Args:
        result: Evaluation result (pass, fail, skip, error).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dq_rules_evaluated_total.labels(
        result=result,
    ).inc()


def record_anomaly(method: str) -> None:
    """Record an anomaly detection event.

    Args:
        method: Detection method (zscore, iqr, isolation_forest, dbscan, percentile).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dq_anomalies_detected_total.labels(
        method=method,
    ).inc()


def record_gate_evaluation(outcome: str) -> None:
    """Record a quality gate evaluation event.

    Args:
        outcome: Gate outcome (pass, fail, warn).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dq_gates_evaluated_total.labels(
        outcome=outcome,
    ).inc()


def record_quality_score(score: float) -> None:
    """Record an overall quality score observation.

    Args:
        score: Quality score (0.0 - 1.0).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dq_overall_quality_score.observe(score)


def record_processing_duration(operation: str, duration: float) -> None:
    """Record processing duration for an operation.

    Args:
        operation: Operation type (profile, assess, validate, detect, etc.).
        duration: Duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dq_processing_duration_seconds.labels(
        operation=operation,
    ).observe(duration)


def update_active_profiles(delta: int) -> None:
    """Update the active profiles gauge.

    Args:
        delta: Positive to increment, negative to decrement.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    if delta > 0:
        dq_active_profiles.inc(delta)
    elif delta < 0:
        dq_active_profiles.dec(abs(delta))


def update_total_issues(delta: int) -> None:
    """Update the total issues found gauge.

    Args:
        delta: Number of issues to add (may be negative for corrections).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    if delta > 0:
        dq_total_issues_found.inc(delta)
    elif delta < 0:
        dq_total_issues_found.dec(abs(delta))


def record_processing_error(error_type: str) -> None:
    """Record a processing error event.

    Args:
        error_type: Error classification (validation, timeout, data, integration, unknown).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dq_processing_errors_total.labels(
        error_type=error_type,
    ).inc()


def record_freshness_check(status: str) -> None:
    """Record a freshness check event.

    Args:
        status: Freshness check status (fresh, stale, expired, unknown).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dq_freshness_checks_total.labels(
        status=status,
    ).inc()


__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "dq_datasets_profiled_total",
    "dq_columns_profiled_total",
    "dq_assessments_completed_total",
    "dq_rules_evaluated_total",
    "dq_anomalies_detected_total",
    "dq_gates_evaluated_total",
    "dq_overall_quality_score",
    "dq_processing_duration_seconds",
    "dq_active_profiles",
    "dq_total_issues_found",
    "dq_processing_errors_total",
    "dq_freshness_checks_total",
    # Helper functions
    "record_profile",
    "record_column_profile",
    "record_assessment",
    "record_rule_evaluation",
    "record_anomaly",
    "record_gate_evaluation",
    "record_quality_score",
    "record_processing_duration",
    "update_active_profiles",
    "update_total_issues",
    "record_processing_error",
    "record_freshness_check",
]
