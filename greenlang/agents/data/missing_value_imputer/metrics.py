# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-012: Missing Value Imputer Agent

12 Prometheus metrics for missing value imputer service monitoring with
graceful fallback when prometheus_client is not installed.

Metrics:
    1.  gl_mvi_jobs_processed_total (Counter, labels: status)
    2.  gl_mvi_values_imputed_total (Counter, labels: strategy)
    3.  gl_mvi_analyses_completed_total (Counter, labels: missingness_type)
    4.  gl_mvi_validations_passed_total (Counter, labels: method)
    5.  gl_mvi_rules_evaluated_total (Counter, labels: priority)
    6.  gl_mvi_strategies_selected_total (Counter, labels: strategy)
    7.  gl_mvi_processing_errors_total (Counter, labels: error_type)
    8.  gl_mvi_confidence_score (Histogram, labels: strategy)
    9.  gl_mvi_processing_duration_seconds (Histogram, labels: operation)
    10. gl_mvi_completeness_improvement (Histogram, labels: strategy)
    11. gl_mvi_active_jobs (Gauge)
    12. gl_mvi_total_missing_detected (Gauge)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
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
        "prometheus_client not installed; missing value imputer metrics disabled"
    )


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Imputation jobs processed by status
    mvi_jobs_processed_total = Counter(
        "gl_mvi_jobs_processed_total",
        "Total imputation jobs processed",
        labelnames=["status"],
    )

    # 2. Values imputed by strategy
    mvi_values_imputed_total = Counter(
        "gl_mvi_values_imputed_total",
        "Total missing values imputed",
        labelnames=["strategy"],
    )

    # 3. Missingness analyses completed by type
    mvi_analyses_completed_total = Counter(
        "gl_mvi_analyses_completed_total",
        "Total missingness analyses completed",
        labelnames=["missingness_type"],
    )

    # 4. Validations passed by method
    mvi_validations_passed_total = Counter(
        "gl_mvi_validations_passed_total",
        "Total imputation validations passed",
        labelnames=["method"],
    )

    # 5. Rules evaluated by priority
    mvi_rules_evaluated_total = Counter(
        "gl_mvi_rules_evaluated_total",
        "Total imputation rules evaluated",
        labelnames=["priority"],
    )

    # 6. Strategies selected by type
    mvi_strategies_selected_total = Counter(
        "gl_mvi_strategies_selected_total",
        "Total imputation strategies selected",
        labelnames=["strategy"],
    )

    # 7. Processing errors by error type
    mvi_processing_errors_total = Counter(
        "gl_mvi_processing_errors_total",
        "Total processing errors encountered",
        labelnames=["error_type"],
    )

    # 8. Confidence score distribution by strategy
    mvi_confidence_score = Histogram(
        "gl_mvi_confidence_score",
        "Imputation confidence score distribution",
        labelnames=["strategy"],
        buckets=(
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 1.0,
        ),
    )

    # 9. Processing duration histogram by operation type
    mvi_processing_duration_seconds = Histogram(
        "gl_mvi_processing_duration_seconds",
        "Missing value imputer processing duration in seconds",
        labelnames=["operation"],
        buckets=(
            0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
            5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
        ),
    )

    # 10. Completeness improvement histogram by strategy
    mvi_completeness_improvement = Histogram(
        "gl_mvi_completeness_improvement",
        "Completeness improvement after imputation (fraction 0.0-1.0)",
        labelnames=["strategy"],
        buckets=(
            0.0, 0.05, 0.10, 0.15, 0.20, 0.30,
            0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0,
        ),
    )

    # 11. Currently active imputation jobs gauge
    mvi_active_jobs = Gauge(
        "gl_mvi_active_jobs",
        "Number of currently active imputation jobs",
    )

    # 12. Total missing values detected gauge
    mvi_total_missing_detected = Gauge(
        "gl_mvi_total_missing_detected",
        "Total missing values detected across active datasets",
    )

else:
    # No-op placeholders
    mvi_jobs_processed_total = None  # type: ignore[assignment]
    mvi_values_imputed_total = None  # type: ignore[assignment]
    mvi_analyses_completed_total = None  # type: ignore[assignment]
    mvi_validations_passed_total = None  # type: ignore[assignment]
    mvi_rules_evaluated_total = None  # type: ignore[assignment]
    mvi_strategies_selected_total = None  # type: ignore[assignment]
    mvi_processing_errors_total = None  # type: ignore[assignment]
    mvi_confidence_score = None  # type: ignore[assignment]
    mvi_processing_duration_seconds = None  # type: ignore[assignment]
    mvi_completeness_improvement = None  # type: ignore[assignment]
    mvi_active_jobs = None  # type: ignore[assignment]
    mvi_total_missing_detected = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def inc_jobs(status: str) -> None:
    """Record an imputation job processed event.

    Args:
        status: Job status (completed, failed, cancelled, timeout).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mvi_jobs_processed_total.labels(
        status=status,
    ).inc()


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
    if not PROMETHEUS_AVAILABLE:
        return
    mvi_values_imputed_total.labels(
        strategy=strategy,
    ).inc(count)


def inc_analyses(missingness_type: str) -> None:
    """Record a missingness analysis completed.

    Args:
        missingness_type: Type of missingness detected (mcar, mar, mnar,
            unknown).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mvi_analyses_completed_total.labels(
        missingness_type=missingness_type,
    ).inc()


def inc_validations(method: str) -> None:
    """Record a validation passed event.

    Args:
        method: Validation method (ks_test, chi_square,
            plausibility_range, distribution_preservation,
            cross_validation).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mvi_validations_passed_total.labels(
        method=method,
    ).inc()


def inc_rules_evaluated(priority: str, count: int = 1) -> None:
    """Record imputation rules evaluated.

    Args:
        priority: Rule priority level (critical, high, medium, low,
            default).
        count: Number of rules evaluated.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mvi_rules_evaluated_total.labels(
        priority=priority,
    ).inc(count)


def inc_strategies_selected(strategy: str) -> None:
    """Record an imputation strategy selection.

    Args:
        strategy: Strategy selected for imputation.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mvi_strategies_selected_total.labels(
        strategy=strategy,
    ).inc()


def inc_errors(error_type: str) -> None:
    """Record a processing error event.

    Args:
        error_type: Error classification (validation, timeout, data,
            integration, imputation, strategy, convergence, unknown).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mvi_processing_errors_total.labels(
        error_type=error_type,
    ).inc()


def observe_confidence(strategy: str, score: float) -> None:
    """Record an imputation confidence score observation.

    Args:
        strategy: Imputation strategy used.
        score: Confidence score (0.0 - 1.0).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mvi_confidence_score.labels(
        strategy=strategy,
    ).observe(score)


def observe_duration(operation: str, duration: float) -> None:
    """Record processing duration for an operation.

    Args:
        operation: Operation type (analyze, strategize, impute, validate,
            document, pipeline, job, rule_evaluate).
        duration: Duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mvi_processing_duration_seconds.labels(
        operation=operation,
    ).observe(duration)


def observe_completeness_improvement(strategy: str, improvement: float) -> None:
    """Record completeness improvement after imputation.

    Args:
        strategy: Imputation strategy used.
        improvement: Completeness improvement fraction (0.0 - 1.0).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mvi_completeness_improvement.labels(
        strategy=strategy,
    ).observe(improvement)


def set_active_jobs(count: int) -> None:
    """Set the active jobs gauge to an absolute value.

    Args:
        count: Number of currently active imputation jobs.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mvi_active_jobs.set(count)


def set_total_missing_detected(count: int) -> None:
    """Set the total missing values detected gauge.

    Args:
        count: Total number of missing values currently detected.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mvi_total_missing_detected.set(count)


__all__ = [
    "PROMETHEUS_AVAILABLE",
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
