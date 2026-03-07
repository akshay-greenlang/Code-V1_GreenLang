# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-002: Geolocation Verification Agent

15 Prometheus metrics for geolocation verification agent service
monitoring with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_geo_`` prefix (GreenLang EUDR
Geolocation Verification) for consistent identification in Prometheus
queries, Grafana dashboards, and alerting rules across the GreenLang
platform.

Metrics:
    1.  gl_eudr_geo_coordinates_validated_total       (Counter)
    2.  gl_eudr_geo_polygons_verified_total           (Counter)
    3.  gl_eudr_geo_protected_area_checks_total       (Counter)
    4.  gl_eudr_geo_deforestation_checks_total        (Counter)
    5.  gl_eudr_geo_plots_verified_total              (Counter, labels: level, result)
    6.  gl_eudr_geo_batch_jobs_total                  (Counter)
    7.  gl_eudr_geo_batch_plots_processed_total       (Counter)
    8.  gl_eudr_geo_scores_calculated_total           (Counter)
    9.  gl_eudr_geo_compliance_reports_total           (Counter)
    10. gl_eudr_geo_issues_detected_total             (Counter, labels: issue_type, severity)
    11. gl_eudr_geo_verification_duration_seconds      (Histogram, labels: level)
    12. gl_eudr_geo_batch_duration_seconds             (Histogram)
    13. gl_eudr_geo_errors_total                       (Counter, labels: operation)
    14. gl_eudr_geo_active_batch_jobs                  (Gauge)
    15. gl_eudr_geo_avg_accuracy_score                 (Gauge)

Label Values Reference:
    level:
        quick, standard, deep.
    result:
        passed, failed, warning.
    issue_type:
        out_of_bounds, low_precision, likely_transposed, country_mismatch,
        on_water, duplicate_coordinate, cluster_anomaly,
        elevation_implausible, zero_coordinate, hemisphere_mismatch,
        rounded_coordinate, unclosed_ring, wrong_winding_order,
        self_intersection, insufficient_vertices, excessive_vertices,
        area_mismatch, sliver_polygon, spike_vertex, duplicate_vertex,
        exceeds_max_area, vertex_density_low, degenerate_polygon.
    severity:
        error, warning.
    operation:
        coordinate_validate, polygon_verify, protected_area_check,
        deforestation_verify, score_calculate, temporal_analyze,
        batch_process, report_generate, plot_verify.

Example:
    >>> from greenlang.agents.eudr.geolocation_verification.metrics import (
    ...     record_coordinate_validated,
    ...     record_polygon_verified,
    ...     observe_verification_duration,
    ...     set_avg_accuracy_score,
    ... )
    >>> record_coordinate_validated()
    >>> record_polygon_verified()
    >>> observe_verification_duration("standard", 2.5)
    >>> set_avg_accuracy_score(78.3)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-002 Geolocation Verification Agent (GL-EUDR-GEO-002)
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
        "geolocation verification metrics disabled"
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
# Metric definitions (15 metrics per PRD Section 7.6)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Coordinate validations performed
    geo_coordinates_validated_total = _safe_counter(
        "gl_eudr_geo_coordinates_validated_total",
        "Total coordinate validations performed",
    )

    # 2. Polygon verifications performed
    geo_polygons_verified_total = _safe_counter(
        "gl_eudr_geo_polygons_verified_total",
        "Total polygon topology verifications performed",
    )

    # 3. Protected area screenings
    geo_protected_area_checks_total = _safe_counter(
        "gl_eudr_geo_protected_area_checks_total",
        "Total protected area intersection screenings performed",
    )

    # 4. Deforestation cutoff verifications
    geo_deforestation_checks_total = _safe_counter(
        "gl_eudr_geo_deforestation_checks_total",
        "Total deforestation cutoff verifications performed",
    )

    # 5. Full plot verifications by level and result
    geo_plots_verified_total = _safe_counter(
        "gl_eudr_geo_plots_verified_total",
        "Total full plot verifications by verification level and result",
        labelnames=["level", "result"],
    )

    # 6. Batch verification jobs submitted
    geo_batch_jobs_total = _safe_counter(
        "gl_eudr_geo_batch_jobs_total",
        "Total batch verification jobs submitted",
    )

    # 7. Plots processed in batch jobs
    geo_batch_plots_processed_total = _safe_counter(
        "gl_eudr_geo_batch_plots_processed_total",
        "Total plots processed in batch verification jobs",
    )

    # 8. Accuracy scores calculated
    geo_scores_calculated_total = _safe_counter(
        "gl_eudr_geo_scores_calculated_total",
        "Total geolocation accuracy scores calculated",
    )

    # 9. Compliance reports generated
    geo_compliance_reports_total = _safe_counter(
        "gl_eudr_geo_compliance_reports_total",
        "Total Article 9 compliance reports generated",
    )

    # 10. Issues detected by type and severity
    geo_issues_detected_total = _safe_counter(
        "gl_eudr_geo_issues_detected_total",
        "Total geolocation issues detected by type and severity",
        labelnames=["issue_type", "severity"],
    )

    # 11. Verification latency by verification level
    geo_verification_duration_seconds = _safe_histogram(
        "gl_eudr_geo_verification_duration_seconds",
        "Duration of plot verification operations in seconds by level",
        labelnames=["level"],
        buckets=(
            0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
            10.0, 30.0, 60.0, 120.0,
        ),
    )

    # 12. Batch job total duration
    geo_batch_duration_seconds = _safe_histogram(
        "gl_eudr_geo_batch_duration_seconds",
        "Duration of complete batch verification jobs in seconds",
        buckets=(
            1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
            600.0, 1800.0, 3600.0,
        ),
    )

    # 13. Errors by operation type
    geo_errors_total = _safe_counter(
        "gl_eudr_geo_errors_total",
        "Total errors encountered by operation type",
        labelnames=["operation"],
    )

    # 14. Currently running batch jobs
    geo_active_batch_jobs = _safe_gauge(
        "gl_eudr_geo_active_batch_jobs",
        "Number of currently running batch verification jobs",
    )

    # 15. Average accuracy score across all verified plots
    geo_avg_accuracy_score = _safe_gauge(
        "gl_eudr_geo_avg_accuracy_score",
        "Average geolocation accuracy score across all verified plots",
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    geo_coordinates_validated_total = None       # type: ignore[assignment]
    geo_polygons_verified_total = None           # type: ignore[assignment]
    geo_protected_area_checks_total = None       # type: ignore[assignment]
    geo_deforestation_checks_total = None        # type: ignore[assignment]
    geo_plots_verified_total = None              # type: ignore[assignment]
    geo_batch_jobs_total = None                  # type: ignore[assignment]
    geo_batch_plots_processed_total = None       # type: ignore[assignment]
    geo_scores_calculated_total = None           # type: ignore[assignment]
    geo_compliance_reports_total = None          # type: ignore[assignment]
    geo_issues_detected_total = None             # type: ignore[assignment]
    geo_verification_duration_seconds = None     # type: ignore[assignment]
    geo_batch_duration_seconds = None            # type: ignore[assignment]
    geo_errors_total = None                      # type: ignore[assignment]
    geo_active_batch_jobs = None                 # type: ignore[assignment]
    geo_avg_accuracy_score = None                # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_coordinate_validated() -> None:
    """Record a coordinate validation event."""
    if not PROMETHEUS_AVAILABLE:
        return
    geo_coordinates_validated_total.inc()


def record_polygon_verified() -> None:
    """Record a polygon topology verification event."""
    if not PROMETHEUS_AVAILABLE:
        return
    geo_polygons_verified_total.inc()


def record_protected_area_check() -> None:
    """Record a protected area screening event."""
    if not PROMETHEUS_AVAILABLE:
        return
    geo_protected_area_checks_total.inc()


def record_deforestation_check() -> None:
    """Record a deforestation cutoff verification event."""
    if not PROMETHEUS_AVAILABLE:
        return
    geo_deforestation_checks_total.inc()


def record_plot_verified(level: str, result: str) -> None:
    """Record a full plot verification event.

    Args:
        level: Verification level ('quick', 'standard', 'deep').
        result: Verification result ('passed', 'failed', 'warning').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    geo_plots_verified_total.labels(level=level, result=result).inc()


def record_batch_job() -> None:
    """Record a batch verification job submission event."""
    if not PROMETHEUS_AVAILABLE:
        return
    geo_batch_jobs_total.inc()


def record_batch_plot_processed() -> None:
    """Record a single plot processed within a batch job."""
    if not PROMETHEUS_AVAILABLE:
        return
    geo_batch_plots_processed_total.inc()


def record_score_calculated() -> None:
    """Record an accuracy score calculation event."""
    if not PROMETHEUS_AVAILABLE:
        return
    geo_scores_calculated_total.inc()


def record_compliance_report() -> None:
    """Record a compliance report generation event."""
    if not PROMETHEUS_AVAILABLE:
        return
    geo_compliance_reports_total.inc()


def record_issue_detected(issue_type: str, severity: str) -> None:
    """Record a geolocation issue detection event.

    Args:
        issue_type: Classification of the issue (e.g., 'out_of_bounds',
            'self_intersection', 'low_precision').
        severity: Severity level ('error' or 'warning').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    geo_issues_detected_total.labels(
        issue_type=issue_type,
        severity=severity,
    ).inc()


def observe_verification_duration(level: str, seconds: float) -> None:
    """Record the duration of a plot verification operation.

    Args:
        level: Verification level ('quick', 'standard', 'deep').
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    geo_verification_duration_seconds.labels(level=level).observe(seconds)


def observe_batch_duration(seconds: float) -> None:
    """Record the duration of a complete batch verification job.

    Args:
        seconds: Total batch job wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    geo_batch_duration_seconds.observe(seconds)


def record_error(operation: str) -> None:
    """Record an error event by operation type.

    Args:
        operation: Type of operation that failed (e.g.,
            'coordinate_validate', 'polygon_verify',
            'protected_area_check', 'deforestation_verify',
            'score_calculate', 'batch_process', 'report_generate',
            'plot_verify', 'temporal_analyze').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    geo_errors_total.labels(operation=operation).inc()


def set_active_batch_jobs(count: int) -> None:
    """Set the gauge for currently running batch verification jobs.

    Args:
        count: Number of active batch jobs. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    geo_active_batch_jobs.set(count)


def set_avg_accuracy_score(score: float) -> None:
    """Set the gauge for average accuracy score across all verified plots.

    Args:
        score: Average geolocation accuracy score (0-100).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    geo_avg_accuracy_score.set(score)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "geo_coordinates_validated_total",
    "geo_polygons_verified_total",
    "geo_protected_area_checks_total",
    "geo_deforestation_checks_total",
    "geo_plots_verified_total",
    "geo_batch_jobs_total",
    "geo_batch_plots_processed_total",
    "geo_scores_calculated_total",
    "geo_compliance_reports_total",
    "geo_issues_detected_total",
    "geo_verification_duration_seconds",
    "geo_batch_duration_seconds",
    "geo_errors_total",
    "geo_active_batch_jobs",
    "geo_avg_accuracy_score",
    # Helper functions
    "record_coordinate_validated",
    "record_polygon_verified",
    "record_protected_area_check",
    "record_deforestation_check",
    "record_plot_verified",
    "record_batch_job",
    "record_batch_plot_processed",
    "record_score_calculated",
    "record_compliance_report",
    "record_issue_detected",
    "observe_verification_duration",
    "observe_batch_duration",
    "record_error",
    "set_active_batch_jobs",
    "set_avg_accuracy_score",
]
