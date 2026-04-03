# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-012: Missing Value Imputer Agent

12 Prometheus metrics for missing value imputer service monitoring with
graceful fallback when prometheus_client is not installed.

Standard metrics (via MetricsFactory):
    1.  gl_mvi_operations_total (Counter, labels: type, tenant_id)
    2.  gl_mvi_processing_duration_seconds (Histogram, 12 buckets)
    3.  gl_mvi_validation_errors_total (Counter, labels: severity, type)
    4.  gl_mvi_batch_jobs_total (Counter, labels: status)
    5.  gl_mvi_active_jobs (Gauge)
    6.  gl_mvi_queue_size (Gauge)

Agent-specific metrics:
    7.  gl_mvi_jobs_processed_total (Counter, labels: status)
    8.  gl_mvi_values_imputed_total (Counter, labels: strategy)
    9.  gl_mvi_analyses_completed_total (Counter, labels: missingness_type)
    10. gl_mvi_confidence_score (Histogram, labels: strategy)
    11. gl_mvi_completeness_improvement (Histogram, labels: strategy)
    12. gl_mvi_total_missing_detected (Gauge)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
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
    "gl_mvi",
    "Missing Value Imputer",
    duration_buckets=(
        0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
        5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
    ),
)

# Backward-compat alias
mvi_active_jobs = m.active_jobs

# ---------------------------------------------------------------------------
# Agent-specific metrics (6 of 12)
# ---------------------------------------------------------------------------

mvi_jobs_processed_total = m.create_custom_counter(
    "jobs_processed_total",
    "Total imputation jobs processed",
    labelnames=["status"],
)

mvi_values_imputed_total = m.create_custom_counter(
    "values_imputed_total",
    "Total missing values imputed",
    labelnames=["strategy"],
)

mvi_analyses_completed_total = m.create_custom_counter(
    "analyses_completed_total",
    "Total missingness analyses completed",
    labelnames=["missingness_type"],
)

mvi_validations_passed_total = m.create_custom_counter(
    "validations_passed_total",
    "Total imputation validations passed",
    labelnames=["method"],
)

mvi_rules_evaluated_total = m.create_custom_counter(
    "rules_evaluated_total",
    "Total imputation rules evaluated",
    labelnames=["priority"],
)

mvi_strategies_selected_total = m.create_custom_counter(
    "strategies_selected_total",
    "Total imputation strategies selected",
    labelnames=["strategy"],
)

mvi_processing_errors_total = m.create_custom_counter(
    "processing_errors_total",
    "Total processing errors encountered",
    labelnames=["error_type"],
)

mvi_confidence_score = m.create_custom_histogram(
    "confidence_score",
    "Imputation confidence score distribution",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    labelnames=["strategy"],
)

mvi_processing_duration_seconds = m.create_custom_histogram(
    "processing_duration_seconds_detail",
    "Missing value imputer processing duration in seconds",
    buckets=(
        0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
        5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
    ),
    labelnames=["operation"],
)

mvi_completeness_improvement = m.create_custom_histogram(
    "completeness_improvement",
    "Completeness improvement after imputation (fraction 0.0-1.0)",
    buckets=(
        0.0, 0.05, 0.10, 0.15, 0.20, 0.30,
        0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0,
    ),
    labelnames=["strategy"],
)

mvi_total_missing_detected = m.create_custom_gauge(
    "total_missing_detected",
    "Total missing values detected across active datasets",
)


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def inc_jobs(status: str) -> None:
    """Record an imputation job processed event.

    Args:
        status: Job status (completed, failed, cancelled, timeout).
    """
    m.safe_inc(mvi_jobs_processed_total, 1, status=status)


def inc_values_imputed(strategy: str, count: int = 1) -> None:
    """Record missing values imputed.

    Args:
        strategy: Imputation strategy used (mean, median, mode, knn,
            regression, mice, random_forest, gradient_boosting,
            linear_interpolation, spline_interpolation,
            seasonal_decomposition, rule_based, lookup_table,
            regulatory_default, hot_deck, locf, nocb).
        count: Number of values imputed.
    """
    m.safe_inc(mvi_values_imputed_total, count, strategy=strategy)


def inc_analyses(missingness_type: str) -> None:
    """Record a missingness analysis completed.

    Args:
        missingness_type: Type of missingness detected (mcar, mar, mnar,
            unknown).
    """
    m.safe_inc(mvi_analyses_completed_total, 1, missingness_type=missingness_type)


def inc_validations(method: str) -> None:
    """Record a validation passed event.

    Args:
        method: Validation method (ks_test, chi_square,
            plausibility_range, distribution_preservation,
            cross_validation).
    """
    m.safe_inc(mvi_validations_passed_total, 1, method=method)


def inc_rules_evaluated(priority: str, count: int = 1) -> None:
    """Record imputation rules evaluated.

    Args:
        priority: Rule priority level (critical, high, medium, low,
            default).
        count: Number of rules evaluated.
    """
    m.safe_inc(mvi_rules_evaluated_total, count, priority=priority)


def inc_strategies_selected(strategy: str) -> None:
    """Record an imputation strategy selection.

    Args:
        strategy: Strategy selected for imputation.
    """
    m.safe_inc(mvi_strategies_selected_total, 1, strategy=strategy)


def inc_errors(error_type: str) -> None:
    """Record a processing error event.

    Args:
        error_type: Error classification (validation, timeout, data,
            integration, imputation, strategy, convergence, unknown).
    """
    m.safe_inc(mvi_processing_errors_total, 1, error_type=error_type)


def observe_confidence(strategy: str, score: float) -> None:
    """Record an imputation confidence score observation.

    Args:
        strategy: Imputation strategy used.
        score: Confidence score (0.0 - 1.0).
    """
    m.safe_observe(mvi_confidence_score, score, strategy=strategy)


def observe_duration(operation: str, duration: float) -> None:
    """Record processing duration for an operation.

    Args:
        operation: Operation type (analyze, strategize, impute, validate,
            document, pipeline, job, rule_evaluate).
        duration: Duration in seconds.
    """
    m.safe_observe(mvi_processing_duration_seconds, duration, operation=operation)


def observe_completeness_improvement(strategy: str, improvement: float) -> None:
    """Record completeness improvement after imputation.

    Args:
        strategy: Imputation strategy used.
        improvement: Completeness improvement fraction (0.0 - 1.0).
    """
    m.safe_observe(mvi_completeness_improvement, improvement, strategy=strategy)


def set_active_jobs(count: int) -> None:
    """Set the active jobs gauge to an absolute value.

    Args:
        count: Number of currently active imputation jobs.
    """
    m.safe_set(mvi_active_jobs, count)


def set_total_missing_detected(count: int) -> None:
    """Set the total missing values detected gauge.

    Args:
        count: Total number of missing values currently detected.
    """
    m.safe_set(mvi_total_missing_detected, count)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "m",
    # Metric objects
    "mvi_jobs_processed_total",
    "mvi_values_imputed_total",
    "mvi_analyses_completed_total",
    "mvi_validations_passed_total",
    "mvi_rules_evaluated_total",
    "mvi_strategies_selected_total",
    "mvi_processing_errors_total",
    "mvi_confidence_score",
    "mvi_processing_duration_seconds",
    "mvi_completeness_improvement",
    "mvi_active_jobs",
    "mvi_total_missing_detected",
    # Helper functions
    "inc_jobs",
    "inc_values_imputed",
    "inc_analyses",
    "inc_validations",
    "inc_rules_evaluated",
    "inc_strategies_selected",
    "inc_errors",
    "observe_confidence",
    "observe_duration",
    "observe_completeness_improvement",
    "set_active_jobs",
    "set_total_missing_detected",
]
