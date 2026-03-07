# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-007: GPS Coordinate Validator

18 Prometheus metrics for GPS coordinate validation service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_gcv_`` prefix (GreenLang EUDR GPS
Coordinate Validator) for consistent identification in Prometheus
queries, Grafana dashboards, and alerting rules across the GreenLang
platform.

Metrics:
    1.  gl_eudr_gcv_coordinates_parsed_total         (Counter)
    2.  gl_eudr_gcv_coordinates_validated_total       (Counter, labels: result)
    3.  gl_eudr_gcv_datum_transforms_total            (Counter, labels: source_datum, target_datum)
    4.  gl_eudr_gcv_precision_analyses_total          (Counter, labels: level)
    5.  gl_eudr_gcv_plausibility_checks_total         (Counter, labels: check_type, result)
    6.  gl_eudr_gcv_reverse_geocodes_total            (Counter)
    7.  gl_eudr_gcv_accuracy_assessments_total        (Counter, labels: tier)
    8.  gl_eudr_gcv_certificates_issued_total         (Counter, labels: status)
    9.  gl_eudr_gcv_auto_corrections_total            (Counter, labels: correction_type)
    10. gl_eudr_gcv_batch_jobs_total                  (Counter, labels: status)
    11. gl_eudr_gcv_batch_coordinates_processed_total (Counter)
    12. gl_eudr_gcv_validation_errors_total           (Counter, labels: error_type, severity)
    13. gl_eudr_gcv_parse_duration_seconds             (Histogram)
    14. gl_eudr_gcv_validation_duration_seconds         (Histogram)
    15. gl_eudr_gcv_batch_duration_seconds              (Histogram)
    16. gl_eudr_gcv_errors_total                        (Counter, labels: operation)
    17. gl_eudr_gcv_active_batch_jobs                   (Gauge)
    18. gl_eudr_gcv_avg_accuracy_score                  (Gauge)

Label Values Reference:
    result:
        valid, invalid, warning, corrected.
    level:
        survey_grade, high, moderate, low, inadequate.
    check_type:
        ocean, country, commodity, elevation, urban, protected_area.
    tier:
        gold, silver, bronze, unverified.
    status (certificate):
        compliant, non_compliant, needs_review, insufficient_data.
    status (batch):
        completed, failed, cancelled.
    correction_type:
        swap_lat_lon, negate_lat, negate_lon, add_hemisphere,
        remove_hemisphere, datum_transform.
    error_type:
        out_of_range, swapped_lat_lon, sign_error, hemisphere_error,
        null_island, nan_value, inf_value, null_value, duplicate,
        near_duplicate, truncated, artificially_rounded, format_error.
    severity:
        error, warning, info.
    operation:
        parse, validate, transform, analyze, check, geocode, assess,
        certify, correct, batch.

Example:
    >>> from greenlang.agents.eudr.gps_coordinate_validator.metrics import (
    ...     record_coordinate_parsed,
    ...     record_coordinate_validated,
    ...     observe_validation_duration,
    ...     set_avg_accuracy_score,
    ... )
    >>> record_coordinate_parsed()
    >>> record_coordinate_validated("valid")
    >>> observe_validation_duration(1.5)
    >>> set_avg_accuracy_score(82.5)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GCV-007)
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
        "GPS coordinate validator metrics disabled"
    )

# ---------------------------------------------------------------------------
# Safe metric registration helpers (avoid collisions with other modules)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    from prometheus_client import REGISTRY as _REGISTRY

    def _safe_counter(
        name: str, doc: str, labelnames: list = None,  # type: ignore[assignment]
    ):
        """Create a Counter or retrieve existing one to avoid registry collisions."""
        try:
            return Counter(name, doc, labelnames=labelnames or [])
        except ValueError:
            for collector in _REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            from prometheus_client import CollectorRegistry
            return Counter(
                name, doc, labelnames=labelnames or [],
                registry=CollectorRegistry(),
            )

    def _safe_histogram(
        name: str, doc: str, labelnames: list = None,  # type: ignore[assignment]
        buckets: tuple = (),
    ):
        """Create a Histogram or retrieve existing one."""
        try:
            kw = {}
            if buckets:
                kw["buckets"] = buckets
            return Histogram(name, doc, labelnames=labelnames or [], **kw)
        except ValueError:
            for collector in _REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            from prometheus_client import CollectorRegistry
            kw = {}
            if buckets:
                kw["buckets"] = buckets
            return Histogram(
                name, doc, labelnames=labelnames or [],
                registry=CollectorRegistry(), **kw,
            )

    def _safe_gauge(
        name: str, doc: str, labelnames: list = None,  # type: ignore[assignment]
    ):
        """Create a Gauge or retrieve existing one."""
        try:
            return Gauge(name, doc, labelnames=labelnames or [])
        except ValueError:
            for collector in _REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            from prometheus_client import CollectorRegistry
            return Gauge(
                name, doc, labelnames=labelnames or [],
                registry=CollectorRegistry(),
            )

# ---------------------------------------------------------------------------
# Metric definitions (18 metrics per PRD Section 7.6)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Coordinate parsing operations
    gcv_coordinates_parsed_total = _safe_counter(
        "gl_eudr_gcv_coordinates_parsed_total",
        "Total coordinate parsing operations performed",
    )

    # 2. Coordinate validations by result
    gcv_coordinates_validated_total = _safe_counter(
        "gl_eudr_gcv_coordinates_validated_total",
        "Total coordinate validations by result",
        labelnames=["result"],
    )

    # 3. Datum transformations by source and target datum
    gcv_datum_transforms_total = _safe_counter(
        "gl_eudr_gcv_datum_transforms_total",
        "Total datum transformations by source and target datum",
        labelnames=["source_datum", "target_datum"],
    )

    # 4. Precision analyses by level
    gcv_precision_analyses_total = _safe_counter(
        "gl_eudr_gcv_precision_analyses_total",
        "Total precision analyses by classified level",
        labelnames=["level"],
    )

    # 5. Plausibility checks by check type and result
    gcv_plausibility_checks_total = _safe_counter(
        "gl_eudr_gcv_plausibility_checks_total",
        "Total plausibility checks by type and result",
        labelnames=["check_type", "result"],
    )

    # 6. Reverse geocoding operations
    gcv_reverse_geocodes_total = _safe_counter(
        "gl_eudr_gcv_reverse_geocodes_total",
        "Total reverse geocoding operations performed",
    )

    # 7. Accuracy assessments by tier
    gcv_accuracy_assessments_total = _safe_counter(
        "gl_eudr_gcv_accuracy_assessments_total",
        "Total accuracy assessments by assigned tier",
        labelnames=["tier"],
    )

    # 8. Compliance certificates issued by status
    gcv_certificates_issued_total = _safe_counter(
        "gl_eudr_gcv_certificates_issued_total",
        "Total compliance certificates issued by status",
        labelnames=["status"],
    )

    # 9. Auto-corrections applied by correction type
    gcv_auto_corrections_total = _safe_counter(
        "gl_eudr_gcv_auto_corrections_total",
        "Total auto-corrections applied by type",
        labelnames=["correction_type"],
    )

    # 10. Batch validation jobs by status
    gcv_batch_jobs_total = _safe_counter(
        "gl_eudr_gcv_batch_jobs_total",
        "Total batch validation jobs by status",
        labelnames=["status"],
    )

    # 11. Coordinates processed in batch jobs
    gcv_batch_coordinates_processed_total = _safe_counter(
        "gl_eudr_gcv_batch_coordinates_processed_total",
        "Total coordinates processed in batch validation jobs",
    )

    # 12. Validation errors by error type and severity
    gcv_validation_errors_total = _safe_counter(
        "gl_eudr_gcv_validation_errors_total",
        "Total validation errors detected by type and severity",
        labelnames=["error_type", "severity"],
    )

    # 13. Coordinate parsing latency
    gcv_parse_duration_seconds = _safe_histogram(
        "gl_eudr_gcv_parse_duration_seconds",
        "Duration of coordinate parsing operations in seconds",
        buckets=(
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1,
            0.25, 0.5, 1.0, 2.5, 5.0,
        ),
    )

    # 14. Coordinate validation latency
    gcv_validation_duration_seconds = _safe_histogram(
        "gl_eudr_gcv_validation_duration_seconds",
        "Duration of coordinate validation operations in seconds",
        buckets=(
            0.01, 0.05, 0.1, 0.25, 0.5, 1.0,
            2.5, 5.0, 10.0, 30.0,
        ),
    )

    # 15. Batch job total duration
    gcv_batch_duration_seconds = _safe_histogram(
        "gl_eudr_gcv_batch_duration_seconds",
        "Duration of complete batch validation jobs in seconds",
        buckets=(
            1.0, 5.0, 10.0, 30.0, 60.0, 120.0,
            300.0, 600.0, 1800.0, 3600.0,
        ),
    )

    # 16. Errors by operation type
    gcv_errors_total = _safe_counter(
        "gl_eudr_gcv_errors_total",
        "Total errors encountered by operation type",
        labelnames=["operation"],
    )

    # 17. Currently running batch jobs
    gcv_active_batch_jobs = _safe_gauge(
        "gl_eudr_gcv_active_batch_jobs",
        "Number of currently running batch validation jobs",
    )

    # 18. Average accuracy score across all validated coordinates
    gcv_avg_accuracy_score = _safe_gauge(
        "gl_eudr_gcv_avg_accuracy_score",
        "Average accuracy score across all validated coordinates",
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    gcv_coordinates_parsed_total = None               # type: ignore[assignment]
    gcv_coordinates_validated_total = None             # type: ignore[assignment]
    gcv_datum_transforms_total = None                  # type: ignore[assignment]
    gcv_precision_analyses_total = None                # type: ignore[assignment]
    gcv_plausibility_checks_total = None               # type: ignore[assignment]
    gcv_reverse_geocodes_total = None                  # type: ignore[assignment]
    gcv_accuracy_assessments_total = None              # type: ignore[assignment]
    gcv_certificates_issued_total = None               # type: ignore[assignment]
    gcv_auto_corrections_total = None                  # type: ignore[assignment]
    gcv_batch_jobs_total = None                        # type: ignore[assignment]
    gcv_batch_coordinates_processed_total = None       # type: ignore[assignment]
    gcv_validation_errors_total = None                 # type: ignore[assignment]
    gcv_parse_duration_seconds = None                  # type: ignore[assignment]
    gcv_validation_duration_seconds = None             # type: ignore[assignment]
    gcv_batch_duration_seconds = None                  # type: ignore[assignment]
    gcv_errors_total = None                            # type: ignore[assignment]
    gcv_active_batch_jobs = None                       # type: ignore[assignment]
    gcv_avg_accuracy_score = None                      # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_coordinate_parsed() -> None:
    """Record a coordinate parsing event."""
    if not PROMETHEUS_AVAILABLE:
        return
    gcv_coordinates_parsed_total.inc()


def record_coordinate_validated(result: str) -> None:
    """Record a coordinate validation event.

    Args:
        result: Validation result ('valid', 'invalid', 'warning', 'corrected').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gcv_coordinates_validated_total.labels(result=result).inc()


def record_datum_transform(source_datum: str, target_datum: str) -> None:
    """Record a datum transformation event.

    Args:
        source_datum: Source datum identifier.
        target_datum: Target datum identifier.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gcv_datum_transforms_total.labels(
        source_datum=source_datum,
        target_datum=target_datum,
    ).inc()


def record_precision_analysis(level: str) -> None:
    """Record a precision analysis event.

    Args:
        level: Classified precision level ('survey_grade', 'high',
            'moderate', 'low', 'inadequate').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gcv_precision_analyses_total.labels(level=level).inc()


def record_plausibility_check(check_type: str, result: str) -> None:
    """Record a plausibility check event.

    Args:
        check_type: Type of check ('ocean', 'country', 'commodity',
            'elevation', 'urban', 'protected_area').
        result: Check result ('pass', 'fail').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gcv_plausibility_checks_total.labels(
        check_type=check_type,
        result=result,
    ).inc()


def record_reverse_geocode() -> None:
    """Record a reverse geocoding event."""
    if not PROMETHEUS_AVAILABLE:
        return
    gcv_reverse_geocodes_total.inc()


def record_accuracy_assessment(tier: str) -> None:
    """Record an accuracy assessment event.

    Args:
        tier: Assigned accuracy tier ('gold', 'silver', 'bronze',
            'unverified').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gcv_accuracy_assessments_total.labels(tier=tier).inc()


def record_certificate_issued(status: str) -> None:
    """Record a compliance certificate issuance event.

    Args:
        status: Certificate compliance status ('compliant',
            'non_compliant', 'needs_review', 'insufficient_data').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gcv_certificates_issued_total.labels(status=status).inc()


def record_auto_correction(correction_type: str) -> None:
    """Record an auto-correction application event.

    Args:
        correction_type: Type of correction applied ('swap_lat_lon',
            'negate_lat', 'negate_lon', 'add_hemisphere',
            'remove_hemisphere', 'datum_transform').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gcv_auto_corrections_total.labels(correction_type=correction_type).inc()


def record_batch_job(status: str) -> None:
    """Record a batch validation job event.

    Args:
        status: Job status ('completed', 'failed', 'cancelled').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gcv_batch_jobs_total.labels(status=status).inc()


def record_batch_coordinate_processed() -> None:
    """Record a single coordinate processed within a batch job."""
    if not PROMETHEUS_AVAILABLE:
        return
    gcv_batch_coordinates_processed_total.inc()


def record_validation_error(error_type: str, severity: str) -> None:
    """Record a validation error detection event.

    Args:
        error_type: Classification of the error (e.g., 'out_of_range',
            'swapped_lat_lon', 'null_island').
        severity: Severity level ('error', 'warning', 'info').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gcv_validation_errors_total.labels(
        error_type=error_type,
        severity=severity,
    ).inc()


def observe_parse_duration(seconds: float) -> None:
    """Record the duration of a coordinate parsing operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gcv_parse_duration_seconds.observe(seconds)


def observe_validation_duration(seconds: float) -> None:
    """Record the duration of a coordinate validation operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gcv_validation_duration_seconds.observe(seconds)


def observe_batch_duration(seconds: float) -> None:
    """Record the duration of a complete batch validation job.

    Args:
        seconds: Total batch job wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gcv_batch_duration_seconds.observe(seconds)


def record_error(operation: str) -> None:
    """Record an error event by operation type.

    Args:
        operation: Type of operation that failed (e.g., 'parse',
            'validate', 'transform', 'analyze', 'check', 'geocode',
            'assess', 'certify', 'correct', 'batch').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gcv_errors_total.labels(operation=operation).inc()


def set_active_batch_jobs(count: int) -> None:
    """Set the gauge for currently running batch validation jobs.

    Args:
        count: Number of active batch jobs. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gcv_active_batch_jobs.set(count)


def set_avg_accuracy_score(score: float) -> None:
    """Set the gauge for average accuracy score across all validated coordinates.

    Args:
        score: Average accuracy score (0-100).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    gcv_avg_accuracy_score.set(score)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "gcv_coordinates_parsed_total",
    "gcv_coordinates_validated_total",
    "gcv_datum_transforms_total",
    "gcv_precision_analyses_total",
    "gcv_plausibility_checks_total",
    "gcv_reverse_geocodes_total",
    "gcv_accuracy_assessments_total",
    "gcv_certificates_issued_total",
    "gcv_auto_corrections_total",
    "gcv_batch_jobs_total",
    "gcv_batch_coordinates_processed_total",
    "gcv_validation_errors_total",
    "gcv_parse_duration_seconds",
    "gcv_validation_duration_seconds",
    "gcv_batch_duration_seconds",
    "gcv_errors_total",
    "gcv_active_batch_jobs",
    "gcv_avg_accuracy_score",
    # Helper functions
    "record_coordinate_parsed",
    "record_coordinate_validated",
    "record_datum_transform",
    "record_precision_analysis",
    "record_plausibility_check",
    "record_reverse_geocode",
    "record_accuracy_assessment",
    "record_certificate_issued",
    "record_auto_correction",
    "record_batch_job",
    "record_batch_coordinate_processed",
    "record_validation_error",
    "observe_parse_duration",
    "observe_validation_duration",
    "observe_batch_duration",
    "record_error",
    "set_active_batch_jobs",
    "set_avg_accuracy_score",
]
