# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-006: Plot Boundary Manager Agent

18 Prometheus metrics for plot boundary manager agent service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_pbm_`` prefix (GreenLang EUDR
Plot Boundary Manager) for consistent identification in Prometheus
queries, Grafana dashboards, and alerting rules across the GreenLang
platform.

Metrics:
    1.  gl_eudr_pbm_boundaries_created_total            (Counter, labels: commodity, country)
    2.  gl_eudr_pbm_boundaries_updated_total            (Counter, labels: change_reason)
    3.  gl_eudr_pbm_validations_total                    (Counter, labels: result, status)
    4.  gl_eudr_pbm_validation_errors_total              (Counter, labels: error_type)
    5.  gl_eudr_pbm_repairs_total                        (Counter, labels: strategy, status)
    6.  gl_eudr_pbm_area_calculations_total              (Counter, labels: method, classification)
    7.  gl_eudr_pbm_overlaps_detected_total              (Counter, labels: severity)
    8.  gl_eudr_pbm_overlap_scans_total                  (Counter, labels: status)
    9.  gl_eudr_pbm_versions_created_total               (Counter, labels: change_reason)
    10. gl_eudr_pbm_simplifications_total                (Counter, labels: method, status)
    11. gl_eudr_pbm_splits_total                         (Counter, labels: status)
    12. gl_eudr_pbm_merges_total                         (Counter, labels: status)
    13. gl_eudr_pbm_exports_total                        (Counter, labels: format, status)
    14. gl_eudr_pbm_batch_jobs_total                     (Counter, labels: operation, status)
    15. gl_eudr_pbm_api_errors_total                     (Counter, labels: operation)
    16. gl_eudr_pbm_operation_duration_seconds            (Histogram, labels: operation)
    17. gl_eudr_pbm_vertex_count                          (Histogram)
    18. gl_eudr_pbm_area_hectares                         (Histogram)

Label Values Reference:
    commodity:
        cattle, cocoa, coffee, palm_oil, rubber, soya, wood.
    country:
        ISO 3166-1 alpha-2 country codes.
    change_reason:
        survey_update, split, merge, correction, seasonal, initial,
        repair, import.
    result (validation):
        valid, invalid.
    status:
        success, error, cached.
    error_type:
        self_intersection, unclosed_ring, duplicate_vertices, spike,
        sliver, wrong_orientation, invalid_coordinates,
        too_few_vertices, hole_outside_shell, overlapping_holes,
        nested_shells, zero_area.
    strategy:
        node_insertion, ring_closure, vertex_removal, spike_removal,
        orientation_reversal, hole_removal, convex_hull_fallback,
        interpolation.
    method (area calculation):
        karney, spherical, utm.
    classification:
        polygon_required, point_sufficient.
    severity:
        minor, moderate, major, critical.
    method (simplification):
        douglas_peucker, visvalingam_whyatt, topology_preserving.
    format:
        geojson, kml, wkt, wkb, shapefile, eudr_xml, gpx, gml.
    operation:
        create, update, validate, repair, area_calc, overlap_detect,
        simplify, split, merge, export, batch.

Example:
    >>> from greenlang.agents.eudr.plot_boundary.metrics import (
    ...     record_boundary_created,
    ...     record_validation,
    ...     record_area_calculation,
    ...     record_api_error,
    ... )
    >>> record_boundary_created("soya", "BR")
    >>> record_validation("valid", "success")
    >>> record_area_calculation("karney", "polygon_required")
    >>> record_api_error("validate")

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-006 Plot Boundary Manager Agent (GL-EUDR-PBM-006)
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
        "plot boundary manager metrics disabled"
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
    # 1. Boundaries created by commodity and country
    pbm_boundaries_created_total = _safe_counter(
        "gl_eudr_pbm_boundaries_created_total",
        "Total plot boundaries created",
        labelnames=["commodity", "country"],
    )

    # 2. Boundaries updated by change reason
    pbm_boundaries_updated_total = _safe_counter(
        "gl_eudr_pbm_boundaries_updated_total",
        "Total plot boundaries updated",
        labelnames=["change_reason"],
    )

    # 3. Validations by result and status
    pbm_validations_total = _safe_counter(
        "gl_eudr_pbm_validations_total",
        "Total boundary validations performed",
        labelnames=["result", "status"],
    )

    # 4. Validation errors by error type
    pbm_validation_errors_total = _safe_counter(
        "gl_eudr_pbm_validation_errors_total",
        "Total validation errors detected by type",
        labelnames=["error_type"],
    )

    # 5. Repairs by strategy and status
    pbm_repairs_total = _safe_counter(
        "gl_eudr_pbm_repairs_total",
        "Total geometry repairs performed",
        labelnames=["strategy", "status"],
    )

    # 6. Area calculations by method and threshold classification
    pbm_area_calculations_total = _safe_counter(
        "gl_eudr_pbm_area_calculations_total",
        "Total geodetic area calculations performed",
        labelnames=["method", "classification"],
    )

    # 7. Overlaps detected by severity
    pbm_overlaps_detected_total = _safe_counter(
        "gl_eudr_pbm_overlaps_detected_total",
        "Total boundary overlaps detected",
        labelnames=["severity"],
    )

    # 8. Overlap scans by status
    pbm_overlap_scans_total = _safe_counter(
        "gl_eudr_pbm_overlap_scans_total",
        "Total overlap detection scans performed",
        labelnames=["status"],
    )

    # 9. Versions created by change reason
    pbm_versions_created_total = _safe_counter(
        "gl_eudr_pbm_versions_created_total",
        "Total boundary versions created",
        labelnames=["change_reason"],
    )

    # 10. Simplifications by method and status
    pbm_simplifications_total = _safe_counter(
        "gl_eudr_pbm_simplifications_total",
        "Total polygon simplifications performed",
        labelnames=["method", "status"],
    )

    # 11. Splits by status
    pbm_splits_total = _safe_counter(
        "gl_eudr_pbm_splits_total",
        "Total boundary split operations",
        labelnames=["status"],
    )

    # 12. Merges by status
    pbm_merges_total = _safe_counter(
        "gl_eudr_pbm_merges_total",
        "Total boundary merge operations",
        labelnames=["status"],
    )

    # 13. Exports by format and status
    pbm_exports_total = _safe_counter(
        "gl_eudr_pbm_exports_total",
        "Total boundary exports performed",
        labelnames=["format", "status"],
    )

    # 14. Batch jobs by operation and status
    pbm_batch_jobs_total = _safe_counter(
        "gl_eudr_pbm_batch_jobs_total",
        "Total batch boundary jobs processed",
        labelnames=["operation", "status"],
    )

    # 15. API errors by operation type
    pbm_api_errors_total = _safe_counter(
        "gl_eudr_pbm_api_errors_total",
        "Total API errors encountered by operation type",
        labelnames=["operation"],
    )

    # 16. Operation duration by operation type
    pbm_operation_duration_seconds = _safe_histogram(
        "gl_eudr_pbm_operation_duration_seconds",
        "Duration of plot boundary operations in seconds",
        labelnames=["operation"],
        buckets=(
            0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
            2.5, 5.0, 10.0, 30.0, 60.0,
        ),
    )

    # 17. Vertex count distribution
    pbm_vertex_count = _safe_histogram(
        "gl_eudr_pbm_vertex_count",
        "Distribution of polygon vertex counts",
        buckets=(
            4, 10, 25, 50, 100, 250, 500,
            1000, 2500, 5000, 10000, 50000, 100000,
        ),
    )

    # 18. Area distribution in hectares
    pbm_area_hectares = _safe_histogram(
        "gl_eudr_pbm_area_hectares",
        "Distribution of plot boundary areas in hectares",
        buckets=(
            0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 10.0,
            50.0, 100.0, 500.0, 1000.0, 5000.0, 50000.0,
        ),
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    pbm_boundaries_created_total = None          # type: ignore[assignment]
    pbm_boundaries_updated_total = None          # type: ignore[assignment]
    pbm_validations_total = None                 # type: ignore[assignment]
    pbm_validation_errors_total = None           # type: ignore[assignment]
    pbm_repairs_total = None                     # type: ignore[assignment]
    pbm_area_calculations_total = None           # type: ignore[assignment]
    pbm_overlaps_detected_total = None           # type: ignore[assignment]
    pbm_overlap_scans_total = None               # type: ignore[assignment]
    pbm_versions_created_total = None            # type: ignore[assignment]
    pbm_simplifications_total = None             # type: ignore[assignment]
    pbm_splits_total = None                      # type: ignore[assignment]
    pbm_merges_total = None                      # type: ignore[assignment]
    pbm_exports_total = None                     # type: ignore[assignment]
    pbm_batch_jobs_total = None                  # type: ignore[assignment]
    pbm_api_errors_total = None                  # type: ignore[assignment]
    pbm_operation_duration_seconds = None         # type: ignore[assignment]
    pbm_vertex_count = None                      # type: ignore[assignment]
    pbm_area_hectares = None                     # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_boundary_created(commodity: str, country: str) -> None:
    """Record a plot boundary creation event.

    Args:
        commodity: EUDR commodity (e.g., 'soya', 'palm_oil').
        country: ISO 3166-1 alpha-2 country code (e.g., 'BR').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pbm_boundaries_created_total.labels(
        commodity=commodity, country=country,
    ).inc()


def record_boundary_updated(change_reason: str) -> None:
    """Record a plot boundary update event.

    Args:
        change_reason: Reason for the update (e.g., 'survey_update',
            'correction', 'repair').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pbm_boundaries_updated_total.labels(change_reason=change_reason).inc()


def record_validation(result: str, status: str) -> None:
    """Record a boundary validation event.

    Args:
        result: Validation result ('valid' or 'invalid').
        status: Operation status ('success', 'error', 'cached').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pbm_validations_total.labels(result=result, status=status).inc()


def record_validation_error(error_type: str) -> None:
    """Record a specific validation error by type.

    Args:
        error_type: Type of validation error (e.g.,
            'self_intersection', 'unclosed_ring', 'spike').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pbm_validation_errors_total.labels(error_type=error_type).inc()


def record_repair(strategy: str, status: str) -> None:
    """Record a geometry repair event.

    Args:
        strategy: Repair strategy applied (e.g., 'node_insertion',
            'ring_closure', 'vertex_removal').
        status: Repair status ('success', 'error').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pbm_repairs_total.labels(strategy=strategy, status=status).inc()


def record_area_calculation(method: str, classification: str) -> None:
    """Record a geodetic area calculation event.

    Args:
        method: Calculation method ('karney', 'spherical', 'utm').
        classification: EUDR threshold result ('polygon_required',
            'point_sufficient').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pbm_area_calculations_total.labels(
        method=method, classification=classification,
    ).inc()


def record_overlap_detected(severity: str) -> None:
    """Record a detected boundary overlap event.

    Args:
        severity: Overlap severity ('minor', 'moderate', 'major',
            'critical').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pbm_overlaps_detected_total.labels(severity=severity).inc()


def record_overlap_scan(status: str) -> None:
    """Record an overlap detection scan event.

    Args:
        status: Scan status ('success', 'error').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pbm_overlap_scans_total.labels(status=status).inc()


def record_version_created(change_reason: str) -> None:
    """Record a boundary version creation event.

    Args:
        change_reason: Reason for the version (e.g., 'survey_update',
            'split', 'merge', 'initial').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pbm_versions_created_total.labels(change_reason=change_reason).inc()


def record_simplification(method: str, status: str) -> None:
    """Record a polygon simplification event.

    Args:
        method: Simplification method ('douglas_peucker',
            'visvalingam_whyatt', 'topology_preserving').
        status: Operation status ('success', 'error').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pbm_simplifications_total.labels(method=method, status=status).inc()


def record_split(status: str) -> None:
    """Record a boundary split operation event.

    Args:
        status: Operation status ('success', 'error').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pbm_splits_total.labels(status=status).inc()


def record_merge(status: str) -> None:
    """Record a boundary merge operation event.

    Args:
        status: Operation status ('success', 'error').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pbm_merges_total.labels(status=status).inc()


def record_export(export_format: str, status: str) -> None:
    """Record a boundary export event.

    Args:
        export_format: Export format ('geojson', 'kml', 'wkt', 'wkb',
            'shapefile', 'eudr_xml', 'gpx', 'gml').
        status: Operation status ('success', 'error').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pbm_exports_total.labels(format=export_format, status=status).inc()


def record_batch_job(operation: str, status: str) -> None:
    """Record a batch boundary job event.

    Args:
        operation: Batch operation type (e.g., 'validate', 'simplify').
        status: Job status ('success', 'error', 'cancelled').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pbm_batch_jobs_total.labels(operation=operation, status=status).inc()


def record_operation_duration(operation: str, seconds: float) -> None:
    """Record the duration of a plot boundary operation.

    Args:
        operation: Type of operation (e.g., 'create', 'validate',
            'area_calc', 'overlap_detect', 'simplify', 'split',
            'merge', 'export').
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pbm_operation_duration_seconds.labels(operation=operation).observe(seconds)


def record_vertex_count(count: int) -> None:
    """Record the vertex count of a processed polygon.

    Args:
        count: Number of vertices in the polygon.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pbm_vertex_count.observe(count)


def record_area_hectares(area_ha: float) -> None:
    """Record the area in hectares of a processed plot boundary.

    Args:
        area_ha: Area of the plot boundary in hectares.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pbm_area_hectares.observe(area_ha)


def record_api_error(operation: str) -> None:
    """Record an API error event by operation type.

    Args:
        operation: Type of operation that failed (e.g., 'create',
            'update', 'validate', 'repair', 'area_calc',
            'overlap_detect', 'simplify', 'split', 'merge',
            'export', 'batch').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pbm_api_errors_total.labels(operation=operation).inc()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "pbm_boundaries_created_total",
    "pbm_boundaries_updated_total",
    "pbm_validations_total",
    "pbm_validation_errors_total",
    "pbm_repairs_total",
    "pbm_area_calculations_total",
    "pbm_overlaps_detected_total",
    "pbm_overlap_scans_total",
    "pbm_versions_created_total",
    "pbm_simplifications_total",
    "pbm_splits_total",
    "pbm_merges_total",
    "pbm_exports_total",
    "pbm_batch_jobs_total",
    "pbm_api_errors_total",
    "pbm_operation_duration_seconds",
    "pbm_vertex_count",
    "pbm_area_hectares",
    # Helper functions
    "record_boundary_created",
    "record_boundary_updated",
    "record_validation",
    "record_validation_error",
    "record_repair",
    "record_area_calculation",
    "record_overlap_detected",
    "record_overlap_scan",
    "record_version_created",
    "record_simplification",
    "record_split",
    "record_merge",
    "record_export",
    "record_batch_job",
    "record_operation_duration",
    "record_vertex_count",
    "record_area_hectares",
    "record_api_error",
]
