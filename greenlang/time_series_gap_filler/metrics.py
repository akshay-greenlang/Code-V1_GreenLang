# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-014: Time Series Gap Filler Agent

12 Prometheus metrics for time series gap filler service monitoring with
graceful fallback when prometheus_client is not installed.

Metrics:
    1.  gl_tsgf_jobs_processed_total (Counter, labels: status)
    2.  gl_tsgf_gaps_detected_total (Counter, labels: gap_type)
    3.  gl_tsgf_gaps_filled_total (Counter, labels: method)
    4.  gl_tsgf_validations_passed_total (Counter, labels: result)
    5.  gl_tsgf_frequencies_detected_total (Counter, labels: level)
    6.  gl_tsgf_strategies_selected_total (Counter, labels: strategy)
    7.  gl_tsgf_fill_confidence (Histogram)
    8.  gl_tsgf_processing_duration_seconds (Histogram, labels: operation)
    9.  gl_tsgf_gap_duration_seconds (Histogram)
    10. gl_tsgf_active_jobs (Gauge)
    11. gl_tsgf_total_gaps_open (Gauge)
    12. gl_tsgf_processing_errors_total (Counter, labels: error_type)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
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
        "prometheus_client not installed; time series gap filler metrics disabled"
    )


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Gap fill jobs processed by status
    tsgf_jobs_processed_total = Counter(
        "gl_tsgf_jobs_processed_total",
        "Total time series gap fill jobs processed",
        labelnames=["status"],
    )

    # 2. Gaps detected by gap type
    tsgf_gaps_detected_total = Counter(
        "gl_tsgf_gaps_detected_total",
        "Total gaps detected in time series data",
        labelnames=["gap_type"],
    )

    # 3. Gaps filled by fill method
    tsgf_gaps_filled_total = Counter(
        "gl_tsgf_gaps_filled_total",
        "Total gaps filled in time series data",
        labelnames=["method"],
    )

    # 4. Validations passed by result
    tsgf_validations_passed_total = Counter(
        "gl_tsgf_validations_passed_total",
        "Total validation checks executed",
        labelnames=["result"],
    )

    # 5. Frequencies detected by level
    tsgf_frequencies_detected_total = Counter(
        "gl_tsgf_frequencies_detected_total",
        "Total frequency detection operations performed",
        labelnames=["level"],
    )

    # 6. Strategies selected by strategy name
    tsgf_strategies_selected_total = Counter(
        "gl_tsgf_strategies_selected_total",
        "Total fill strategy selections performed",
        labelnames=["strategy"],
    )

    # 7. Fill confidence score distribution
    tsgf_fill_confidence = Histogram(
        "gl_tsgf_fill_confidence",
        "Distribution of gap fill confidence scores",
        buckets=(
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 1.0,
        ),
    )

    # 8. Processing duration histogram by operation type
    tsgf_processing_duration_seconds = Histogram(
        "gl_tsgf_processing_duration_seconds",
        "Time series gap filler processing duration in seconds",
        labelnames=["operation"],
        buckets=(
            0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
            5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
        ),
    )

    # 9. Gap duration histogram (how long each detected gap spans)
    tsgf_gap_duration_seconds = Histogram(
        "gl_tsgf_gap_duration_seconds",
        "Distribution of detected gap durations in seconds",
        buckets=(
            60.0, 300.0, 900.0, 1800.0, 3600.0,
            7200.0, 14400.0, 28800.0, 43200.0,
            86400.0, 172800.0, 604800.0, 2592000.0,
        ),
    )

    # 10. Currently active gap fill jobs gauge
    tsgf_active_jobs = Gauge(
        "gl_tsgf_active_jobs",
        "Number of currently active time series gap fill jobs",
    )

    # 11. Total open gaps gauge
    tsgf_total_gaps_open = Gauge(
        "gl_tsgf_total_gaps_open",
        "Total gaps currently open across active datasets",
    )

    # 12. Processing errors by error type
    tsgf_processing_errors_total = Counter(
        "gl_tsgf_processing_errors_total",
        "Total processing errors encountered",
        labelnames=["error_type"],
    )

else:
    # No-op placeholders
    tsgf_jobs_processed_total = None  # type: ignore[assignment]
    tsgf_gaps_detected_total = None  # type: ignore[assignment]
    tsgf_gaps_filled_total = None  # type: ignore[assignment]
    tsgf_validations_passed_total = None  # type: ignore[assignment]
    tsgf_frequencies_detected_total = None  # type: ignore[assignment]
    tsgf_strategies_selected_total = None  # type: ignore[assignment]
    tsgf_fill_confidence = None  # type: ignore[assignment]
    tsgf_processing_duration_seconds = None  # type: ignore[assignment]
    tsgf_gap_duration_seconds = None  # type: ignore[assignment]
    tsgf_active_jobs = None  # type: ignore[assignment]
    tsgf_total_gaps_open = None  # type: ignore[assignment]
    tsgf_processing_errors_total = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def inc_jobs_processed(status: str) -> None:
    """Record a gap fill job processed event.

    Args:
        status: Job status (completed, failed, cancelled, timeout,
            partial).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    tsgf_jobs_processed_total.labels(
        status=status,
    ).inc()


def inc_gaps_detected(gap_type: str, count: int = 1) -> None:
    """Record gaps detected in time series data.

    Args:
        gap_type: Type of gap detected (missing, null, irregular,
            duplicated, truncated, sparse, block).
        count: Number of gaps detected.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    tsgf_gaps_detected_total.labels(
        gap_type=gap_type,
    ).inc(count)


def inc_gaps_filled(method: str, count: int = 1) -> None:
    """Record gaps filled by a specific method.

    Args:
        method: Fill method used (linear, spline, forward_fill,
            backward_fill, mean, median, seasonal, kalman,
            regression, ensemble, custom).
        count: Number of gaps filled.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    tsgf_gaps_filled_total.labels(
        method=method,
    ).inc(count)


def inc_validations(result: str, count: int = 1) -> None:
    """Record validation check results.

    Args:
        result: Validation result (passed, failed, warning, skipped).
        count: Number of validation checks.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    tsgf_validations_passed_total.labels(
        result=result,
    ).inc(count)


def inc_frequencies(level: str, count: int = 1) -> None:
    """Record frequency detection operations.

    Args:
        level: Frequency level detected (sub_minute, minutely,
            hourly, daily, weekly, monthly, quarterly, yearly,
            irregular, unknown).
        count: Number of frequency detections.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    tsgf_frequencies_detected_total.labels(
        level=level,
    ).inc(count)


def inc_strategies(strategy: str, count: int = 1) -> None:
    """Record strategy selection events.

    Args:
        strategy: Strategy selected (interpolation, extrapolation,
            imputation, seasonal_decomposition, model_based,
            rule_based, hybrid, manual).
        count: Number of strategy selections.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    tsgf_strategies_selected_total.labels(
        strategy=strategy,
    ).inc(count)


def observe_confidence(confidence: float) -> None:
    """Record a gap fill confidence score observation.

    Args:
        confidence: Confidence score (0.0 - 1.0).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    tsgf_fill_confidence.observe(confidence)


def observe_duration(operation: str, duration: float) -> None:
    """Record processing duration for an operation.

    Args:
        operation: Operation type (detect_gaps, detect_frequency,
            select_strategy, fill_gaps, validate, report, pipeline,
            batch, export).
        duration: Duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    tsgf_processing_duration_seconds.labels(
        operation=operation,
    ).observe(duration)


def observe_gap_duration(duration: float) -> None:
    """Record the duration of a detected gap.

    Args:
        duration: Gap duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    tsgf_gap_duration_seconds.observe(duration)


def set_active_jobs(count: int) -> None:
    """Set the active jobs gauge to an absolute value.

    Args:
        count: Number of currently active gap fill jobs.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    tsgf_active_jobs.set(count)


def set_gaps_open(count: int) -> None:
    """Set the total open gaps gauge.

    Args:
        count: Total number of gaps currently open across datasets.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    tsgf_total_gaps_open.set(count)


def inc_errors(error_type: str) -> None:
    """Record a processing error event.

    Args:
        error_type: Error classification (validation, timeout, data,
            integration, detection, frequency, strategy, fill,
            interpolation, extrapolation, unknown).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    tsgf_processing_errors_total.labels(
        error_type=error_type,
    ).inc()


__all__ = [
    "PROMETHEUS_AVAILABLE",
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
