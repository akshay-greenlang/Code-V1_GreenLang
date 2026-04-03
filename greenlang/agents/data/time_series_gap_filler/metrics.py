# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-014: Time Series Gap Filler Agent

12 Prometheus metrics for time series gap filler service monitoring with
graceful fallback when prometheus_client is not installed.

Standard metrics (via MetricsFactory):
    1.  gl_tsgf_operations_total (Counter, labels: type, tenant_id)
    2.  gl_tsgf_processing_duration_seconds (Histogram, 12 buckets)
    3.  gl_tsgf_validation_errors_total (Counter, labels: severity, type)
    4.  gl_tsgf_batch_jobs_total (Counter, labels: status)
    5.  gl_tsgf_active_jobs (Gauge)
    6.  gl_tsgf_queue_size (Gauge)

Agent-specific metrics:
    7.  gl_tsgf_jobs_processed_total (Counter, labels: status)
    8.  gl_tsgf_gaps_detected_total (Counter, labels: gap_type)
    9.  gl_tsgf_gaps_filled_total (Counter, labels: method)
    10. gl_tsgf_fill_confidence (Histogram)
    11. gl_tsgf_gap_duration_seconds (Histogram)
    12. gl_tsgf_total_gaps_open (Gauge)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
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
    "gl_tsgf",
    "Time Series Gap Filler",
    duration_buckets=(
        0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
        5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
    ),
)

# Backward-compat alias
tsgf_active_jobs = m.active_jobs

# ---------------------------------------------------------------------------
# Agent-specific metrics (6 of 12)
# ---------------------------------------------------------------------------

tsgf_jobs_processed_total = m.create_custom_counter(
    "jobs_processed_total",
    "Total time series gap fill jobs processed",
    labelnames=["status"],
)

tsgf_gaps_detected_total = m.create_custom_counter(
    "gaps_detected_total",
    "Total gaps detected in time series data",
    labelnames=["gap_type"],
)

tsgf_gaps_filled_total = m.create_custom_counter(
    "gaps_filled_total",
    "Total gaps filled in time series data",
    labelnames=["method"],
)

tsgf_validations_passed_total = m.create_custom_counter(
    "validations_passed_total",
    "Total validation checks executed",
    labelnames=["result"],
)

tsgf_frequencies_detected_total = m.create_custom_counter(
    "frequencies_detected_total",
    "Total frequency detection operations performed",
    labelnames=["level"],
)

tsgf_strategies_selected_total = m.create_custom_counter(
    "strategies_selected_total",
    "Total fill strategy selections performed",
    labelnames=["strategy"],
)

tsgf_fill_confidence = m.create_custom_histogram(
    "fill_confidence",
    "Distribution of gap fill confidence scores",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

tsgf_processing_duration_seconds = m.create_custom_histogram(
    "processing_duration_seconds_detail",
    "Time series gap filler processing duration in seconds",
    buckets=(
        0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
        5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
    ),
    labelnames=["operation"],
)

tsgf_gap_duration_seconds = m.create_custom_histogram(
    "gap_duration_seconds",
    "Distribution of detected gap durations in seconds",
    buckets=(
        60.0, 300.0, 900.0, 1800.0, 3600.0,
        7200.0, 14400.0, 28800.0, 43200.0,
        86400.0, 172800.0, 604800.0, 2592000.0,
    ),
)

tsgf_total_gaps_open = m.create_custom_gauge(
    "total_gaps_open",
    "Total gaps currently open across active datasets",
)

tsgf_processing_errors_total = m.create_custom_counter(
    "processing_errors_total",
    "Total processing errors encountered",
    labelnames=["error_type"],
)


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def inc_jobs_processed(status: str) -> None:
    """Record a gap fill job processed event.

    Args:
        status: Job status (completed, failed, cancelled, timeout,
            partial).
    """
    m.safe_inc(tsgf_jobs_processed_total, 1, status=status)


def inc_gaps_detected(gap_type: str, count: int = 1) -> None:
    """Record gaps detected in time series data.

    Args:
        gap_type: Type of gap detected (missing, null, irregular,
            duplicated, truncated, sparse, block).
        count: Number of gaps detected.
    """
    m.safe_inc(tsgf_gaps_detected_total, count, gap_type=gap_type)


def inc_gaps_filled(method: str, count: int = 1) -> None:
    """Record gaps filled by a specific method.

    Args:
        method: Fill method used (linear, spline, forward_fill,
            backward_fill, mean, median, seasonal, kalman,
            regression, ensemble, custom).
        count: Number of gaps filled.
    """
    m.safe_inc(tsgf_gaps_filled_total, count, method=method)


def inc_validations(result: str, count: int = 1) -> None:
    """Record validation check results.

    Args:
        result: Validation result (passed, failed, warning, skipped).
        count: Number of validation checks.
    """
    m.safe_inc(tsgf_validations_passed_total, count, result=result)


def inc_frequencies(level: str, count: int = 1) -> None:
    """Record frequency detection operations.

    Args:
        level: Frequency level detected (sub_minute, minutely,
            hourly, daily, weekly, monthly, quarterly, yearly,
            irregular, unknown).
        count: Number of frequency detections.
    """
    m.safe_inc(tsgf_frequencies_detected_total, count, level=level)


def inc_strategies(strategy: str, count: int = 1) -> None:
    """Record strategy selection events.

    Args:
        strategy: Strategy selected (interpolation, extrapolation,
            imputation, seasonal_decomposition, model_based,
            rule_based, hybrid, manual).
        count: Number of strategy selections.
    """
    m.safe_inc(tsgf_strategies_selected_total, count, strategy=strategy)


def observe_confidence(confidence: float) -> None:
    """Record a gap fill confidence score observation.

    Args:
        confidence: Confidence score (0.0 - 1.0).
    """
    m.safe_observe(tsgf_fill_confidence, confidence)


def observe_duration(operation: str, duration: float) -> None:
    """Record processing duration for an operation.

    Args:
        operation: Operation type (detect_gaps, detect_frequency,
            select_strategy, fill_gaps, validate, report, pipeline,
            batch, export).
        duration: Duration in seconds.
    """
    m.safe_observe(tsgf_processing_duration_seconds, duration, operation=operation)


def observe_gap_duration(duration: float) -> None:
    """Record the duration of a detected gap.

    Args:
        duration: Gap duration in seconds.
    """
    m.safe_observe(tsgf_gap_duration_seconds, duration)


def set_active_jobs(count: int) -> None:
    """Set the active jobs gauge to an absolute value.

    Args:
        count: Number of currently active gap fill jobs.
    """
    m.safe_set(tsgf_active_jobs, count)


def set_gaps_open(count: int) -> None:
    """Set the total open gaps gauge.

    Args:
        count: Total number of gaps currently open across datasets.
    """
    m.safe_set(tsgf_total_gaps_open, count)


def inc_errors(error_type: str) -> None:
    """Record a processing error event.

    Args:
        error_type: Error classification (validation, timeout, data,
            integration, detection, frequency, strategy, fill,
            interpolation, extrapolation, unknown).
    """
    m.safe_inc(tsgf_processing_errors_total, 1, error_type=error_type)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "m",
    # Metric objects
    "tsgf_jobs_processed_total",
    "tsgf_gaps_detected_total",
    "tsgf_gaps_filled_total",
    "tsgf_validations_passed_total",
    "tsgf_frequencies_detected_total",
    "tsgf_strategies_selected_total",
    "tsgf_fill_confidence",
    "tsgf_processing_duration_seconds",
    "tsgf_gap_duration_seconds",
    "tsgf_active_jobs",
    "tsgf_total_gaps_open",
    "tsgf_processing_errors_total",
    # Helper functions
    "inc_jobs_processed",
    "inc_gaps_detected",
    "inc_gaps_filled",
    "inc_validations",
    "inc_frequencies",
    "inc_strategies",
    "observe_confidence",
    "observe_duration",
    "observe_gap_duration",
    "set_active_jobs",
    "set_gaps_open",
    "inc_errors",
]
