# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-010: Data Quality Profiler

12 Prometheus metrics for data quality profiler service monitoring with
graceful fallback when prometheus_client is not installed.

Standard metrics (via MetricsFactory):
    1.  gl_dq_operations_total (Counter, labels: type, tenant_id)
    2.  gl_dq_processing_duration_seconds (Histogram, 12 buckets)
    3.  gl_dq_validation_errors_total (Counter, labels: severity, type)
    4.  gl_dq_batch_jobs_total (Counter, labels: status)
    5.  gl_dq_active_jobs (Gauge)
    6.  gl_dq_queue_size (Gauge)

Agent-specific metrics:
    7.  gl_dq_datasets_profiled_total (Counter, labels: source)
    8.  gl_dq_columns_profiled_total (Counter, labels: data_type)
    9.  gl_dq_assessments_completed_total (Counter, labels: quality_level)
    10. gl_dq_anomalies_detected_total (Counter, labels: method)
    11. gl_dq_overall_quality_score (Histogram, buckets: 0.1-1.0)
    12. gl_dq_freshness_checks_total (Counter, labels: status)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-010 Data Quality Profiler (GL-DATA-X-013)
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
    "gl_dq",
    "Data Quality Profiler",
    duration_buckets=(
        0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
        5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
    ),
)

# ---------------------------------------------------------------------------
# Agent-specific metrics (6 of 12)
# ---------------------------------------------------------------------------

dq_datasets_profiled_total = m.create_custom_counter(
    "datasets_profiled_total",
    "Total datasets profiled",
    labelnames=["source"],
)

dq_columns_profiled_total = m.create_custom_counter(
    "columns_profiled_total",
    "Total columns profiled",
    labelnames=["data_type"],
)

dq_assessments_completed_total = m.create_custom_counter(
    "assessments_completed_total",
    "Total quality assessments completed",
    labelnames=["quality_level"],
)

dq_rules_evaluated_total = m.create_custom_counter(
    "rules_evaluated_total",
    "Total quality rules evaluated",
    labelnames=["result"],
)

dq_anomalies_detected_total = m.create_custom_counter(
    "anomalies_detected_total",
    "Total anomalies detected",
    labelnames=["method"],
)

dq_gates_evaluated_total = m.create_custom_counter(
    "gates_evaluated_total",
    "Total quality gates evaluated",
    labelnames=["outcome"],
)

dq_overall_quality_score = m.create_custom_histogram(
    "overall_quality_score",
    "Overall quality score distribution",
    buckets=CONFIDENCE_BUCKETS,
)

dq_active_profiles = m.create_custom_gauge(
    "active_profiles",
    "Number of currently active profiling operations",
)

dq_total_issues_found = m.create_custom_gauge(
    "total_issues_found",
    "Total cumulative data quality issues found",
)

dq_processing_errors_total = m.create_custom_counter(
    "processing_errors_total",
    "Total processing errors encountered",
    labelnames=["error_type"],
)

dq_freshness_checks_total = m.create_custom_counter(
    "freshness_checks_total",
    "Total freshness checks performed",
    labelnames=["status"],
)

# Backward-compat alias for standard metrics expected by __init__.py
dq_processing_duration_seconds = m.processing_duration


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_profile(source: str) -> None:
    """Record a dataset profiling event.

    Args:
        source: Profiling source (csv, excel, api, erp, manual).
    """
    m.safe_inc(dq_datasets_profiled_total, 1, source=source)


def record_column_profile(data_type: str) -> None:
    """Record a column profiling event.

    Args:
        data_type: Detected data type (string, integer, float, date, boolean, etc.).
    """
    m.safe_inc(dq_columns_profiled_total, 1, data_type=data_type)


def record_assessment(quality_level: str) -> None:
    """Record a quality assessment completion event.

    Args:
        quality_level: Quality level result (EXCELLENT, GOOD, FAIR, POOR, CRITICAL).
    """
    m.safe_inc(dq_assessments_completed_total, 1, quality_level=quality_level)


def record_rule_evaluation(result: str) -> None:
    """Record a quality rule evaluation event.

    Args:
        result: Evaluation result (pass, fail, skip, error).
    """
    m.safe_inc(dq_rules_evaluated_total, 1, result=result)


def record_anomaly(method: str) -> None:
    """Record an anomaly detection event.

    Args:
        method: Detection method (zscore, iqr, isolation_forest, dbscan, percentile).
    """
    m.safe_inc(dq_anomalies_detected_total, 1, method=method)


def record_gate_evaluation(outcome: str) -> None:
    """Record a quality gate evaluation event.

    Args:
        outcome: Gate outcome (pass, fail, warn).
    """
    m.safe_inc(dq_gates_evaluated_total, 1, outcome=outcome)


def record_quality_score(score: float) -> None:
    """Record an overall quality score observation.

    Args:
        score: Quality score (0.0 - 1.0).
    """
    m.safe_observe(dq_overall_quality_score, score)


def record_processing_duration(operation: str, duration: float) -> None:
    """Record processing duration for an operation.

    Args:
        operation: Operation type (profile, assess, validate, detect, etc.).
        duration: Duration in seconds.
    """
    m.record_operation(duration, type=operation, tenant_id="default")


def update_active_profiles(delta: int) -> None:
    """Update the active profiles gauge.

    Args:
        delta: Positive to increment, negative to decrement.
    """
    if not m.available:
        return
    if dq_active_profiles is not None:
        if delta > 0:
            dq_active_profiles.inc(delta)
        elif delta < 0:
            dq_active_profiles.dec(abs(delta))


def update_total_issues(delta: int) -> None:
    """Update the total issues found gauge.

    Args:
        delta: Number of issues to add (may be negative for corrections).
    """
    if not m.available:
        return
    if dq_total_issues_found is not None:
        if delta > 0:
            dq_total_issues_found.inc(delta)
        elif delta < 0:
            dq_total_issues_found.dec(abs(delta))


def record_processing_error(error_type: str) -> None:
    """Record a processing error event.

    Args:
        error_type: Error classification (validation, timeout, data, integration, unknown).
    """
    m.safe_inc(dq_processing_errors_total, 1, error_type=error_type)


def record_freshness_check(status: str) -> None:
    """Record a freshness check event.

    Args:
        status: Freshness check status (fresh, stale, expired, unknown).
    """
    m.safe_inc(dq_freshness_checks_total, 1, status=status)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "m",
    # Metric objects
    "dq_datasets_profiled_total",
    "dq_columns_profiled_total",
    "dq_assessments_completed_total",
    "dq_rules_evaluated_total",
    "dq_anomalies_detected_total",
    "dq_gates_evaluated_total",
    "dq_overall_quality_score",
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
