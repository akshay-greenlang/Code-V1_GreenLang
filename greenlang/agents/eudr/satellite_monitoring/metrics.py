# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-003: Satellite Monitoring Agent

18 Prometheus metrics for satellite monitoring agent service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_sat_`` prefix (GreenLang EUDR
Satellite Monitoring) for consistent identification in Prometheus
queries, Grafana dashboards, and alerting rules across the GreenLang
platform.

Metrics:
    1.  gl_eudr_sat_scenes_queried_total              (Counter, labels: source, status)
    2.  gl_eudr_sat_scenes_downloaded_total            (Counter, labels: source)
    3.  gl_eudr_sat_imagery_download_bytes_total       (Counter, labels: source)
    4.  gl_eudr_sat_baselines_established_total        (Counter, labels: commodity, country)
    5.  gl_eudr_sat_ndvi_calculations_total            (Counter, labels: index_type)
    6.  gl_eudr_sat_change_detections_total            (Counter, labels: method, result)
    7.  gl_eudr_sat_deforestation_detected_total       (Counter, labels: commodity, country, severity)
    8.  gl_eudr_sat_alerts_generated_total             (Counter, labels: severity)
    9.  gl_eudr_sat_evidence_packages_total            (Counter, labels: format)
    10. gl_eudr_sat_monitoring_executions_total        (Counter, labels: status)
    11. gl_eudr_sat_cloud_gap_fills_total              (Counter, labels: method)
    12. gl_eudr_sat_fusion_analyses_total              (Counter, labels: sources_count)
    13. gl_eudr_sat_analysis_duration_seconds           (Histogram, labels: operation)
    14. gl_eudr_sat_batch_duration_seconds              (Histogram)
    15. gl_eudr_sat_active_monitoring_plots             (Gauge)
    16. gl_eudr_sat_avg_detection_confidence            (Gauge)
    17. gl_eudr_sat_api_errors_total                    (Counter, labels: operation)
    18. gl_eudr_sat_data_quality_score                  (Gauge, labels: source)

Label Values Reference:
    source:
        sentinel_2, landsat_8, landsat_9, sentinel_1_sar, gfw_alerts.
    status:
        success, error, no_results, cached.
    method (detection):
        ndvi_differencing, spectral_angle, time_series_break,
        multi_source_fusion, sar_backscatter.
    result:
        no_change, deforestation, degradation, reforestation, regrowth.
    severity:
        critical, warning, info.
    commodity:
        cattle, cocoa, coffee, oil_palm, rubber, soya, wood.
    country:
        ISO 3166-1 alpha-2 country codes.
    format:
        json, pdf, csv, eudr_xml.
    method (cloud fill):
        temporal_composite, sar_fusion, interpolation, nearest_clear.
    sources_count:
        1, 2, 3, 4, 5 (number of sources fused).
    operation:
        scene_search, scene_download, baseline_establish,
        change_detect, fuse_sources, cloud_fill, monitoring_execute,
        alert_generate, evidence_generate, batch_analyze.

Example:
    >>> from greenlang.agents.eudr.satellite_monitoring.metrics import (
    ...     record_scene_queried,
    ...     record_baseline_established,
    ...     observe_analysis_duration,
    ...     set_avg_detection_confidence,
    ... )
    >>> record_scene_queried("sentinel_2", "success")
    >>> record_baseline_established("soya", "BR")
    >>> observe_analysis_duration("change_detect", 2.5)
    >>> set_avg_detection_confidence(85.2)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-003 Satellite Monitoring Agent (GL-EUDR-SAT-003)
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
        "satellite monitoring metrics disabled"
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
    # 1. Scene queries performed by source and status
    sat_scenes_queried_total = _safe_counter(
        "gl_eudr_sat_scenes_queried_total",
        "Total satellite scene queries performed",
        labelnames=["source", "status"],
    )

    # 2. Scenes downloaded by source
    sat_scenes_downloaded_total = _safe_counter(
        "gl_eudr_sat_scenes_downloaded_total",
        "Total satellite scenes downloaded",
        labelnames=["source"],
    )

    # 3. Total imagery download volume in bytes
    sat_imagery_download_bytes_total = _safe_counter(
        "gl_eudr_sat_imagery_download_bytes_total",
        "Total satellite imagery download volume in bytes",
        labelnames=["source"],
    )

    # 4. Baselines established by commodity and country
    sat_baselines_established_total = _safe_counter(
        "gl_eudr_sat_baselines_established_total",
        "Total spectral baselines established",
        labelnames=["commodity", "country"],
    )

    # 5. Spectral index calculations by index type
    sat_ndvi_calculations_total = _safe_counter(
        "gl_eudr_sat_ndvi_calculations_total",
        "Total spectral index calculations performed",
        labelnames=["index_type"],
    )

    # 6. Change detections by method and result
    sat_change_detections_total = _safe_counter(
        "gl_eudr_sat_change_detections_total",
        "Total change detection analyses performed",
        labelnames=["method", "result"],
    )

    # 7. Deforestation events detected by commodity, country, severity
    sat_deforestation_detected_total = _safe_counter(
        "gl_eudr_sat_deforestation_detected_total",
        "Total deforestation events detected",
        labelnames=["commodity", "country", "severity"],
    )

    # 8. Alerts generated by severity
    sat_alerts_generated_total = _safe_counter(
        "gl_eudr_sat_alerts_generated_total",
        "Total satellite monitoring alerts generated",
        labelnames=["severity"],
    )

    # 9. Evidence packages generated by format
    sat_evidence_packages_total = _safe_counter(
        "gl_eudr_sat_evidence_packages_total",
        "Total EUDR evidence packages generated",
        labelnames=["format"],
    )

    # 10. Monitoring executions by status
    sat_monitoring_executions_total = _safe_counter(
        "gl_eudr_sat_monitoring_executions_total",
        "Total monitoring schedule executions",
        labelnames=["status"],
    )

    # 11. Cloud gap fills by method
    sat_cloud_gap_fills_total = _safe_counter(
        "gl_eudr_sat_cloud_gap_fills_total",
        "Total cloud gap fill operations performed",
        labelnames=["method"],
    )

    # 12. Fusion analyses by number of sources fused
    sat_fusion_analyses_total = _safe_counter(
        "gl_eudr_sat_fusion_analyses_total",
        "Total multi-source fusion analyses performed",
        labelnames=["sources_count"],
    )

    # 13. Analysis operation duration by operation type
    sat_analysis_duration_seconds = _safe_histogram(
        "gl_eudr_sat_analysis_duration_seconds",
        "Duration of satellite analysis operations in seconds",
        labelnames=["operation"],
        buckets=(
            0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
            10.0, 30.0, 60.0, 120.0, 300.0,
        ),
    )

    # 14. Batch analysis total duration
    sat_batch_duration_seconds = _safe_histogram(
        "gl_eudr_sat_batch_duration_seconds",
        "Duration of complete batch analysis jobs in seconds",
        buckets=(
            1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
            600.0, 1800.0, 3600.0,
        ),
    )

    # 15. Currently active monitoring plots
    sat_active_monitoring_plots = _safe_gauge(
        "gl_eudr_sat_active_monitoring_plots",
        "Number of plots under active satellite monitoring",
    )

    # 16. Average detection confidence across all monitored plots
    sat_avg_detection_confidence = _safe_gauge(
        "gl_eudr_sat_avg_detection_confidence",
        "Average change detection confidence across monitored plots",
    )

    # 17. API errors by operation type
    sat_api_errors_total = _safe_counter(
        "gl_eudr_sat_api_errors_total",
        "Total API errors encountered by operation type",
        labelnames=["operation"],
    )

    # 18. Data quality score by source
    sat_data_quality_score = _safe_gauge(
        "gl_eudr_sat_data_quality_score",
        "Data quality score by satellite source (0-100)",
        labelnames=["source"],
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    sat_scenes_queried_total = None            # type: ignore[assignment]
    sat_scenes_downloaded_total = None         # type: ignore[assignment]
    sat_imagery_download_bytes_total = None    # type: ignore[assignment]
    sat_baselines_established_total = None     # type: ignore[assignment]
    sat_ndvi_calculations_total = None         # type: ignore[assignment]
    sat_change_detections_total = None         # type: ignore[assignment]
    sat_deforestation_detected_total = None    # type: ignore[assignment]
    sat_alerts_generated_total = None          # type: ignore[assignment]
    sat_evidence_packages_total = None         # type: ignore[assignment]
    sat_monitoring_executions_total = None     # type: ignore[assignment]
    sat_cloud_gap_fills_total = None           # type: ignore[assignment]
    sat_fusion_analyses_total = None           # type: ignore[assignment]
    sat_analysis_duration_seconds = None       # type: ignore[assignment]
    sat_batch_duration_seconds = None          # type: ignore[assignment]
    sat_active_monitoring_plots = None         # type: ignore[assignment]
    sat_avg_detection_confidence = None        # type: ignore[assignment]
    sat_api_errors_total = None                # type: ignore[assignment]
    sat_data_quality_score = None              # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_scene_queried(source: str, status: str) -> None:
    """Record a satellite scene query event.

    Args:
        source: Satellite source queried (e.g., 'sentinel_2',
            'landsat_8', 'gfw_alerts').
        status: Query status ('success', 'error', 'no_results',
            'cached').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sat_scenes_queried_total.labels(source=source, status=status).inc()


def record_scene_downloaded(source: str) -> None:
    """Record a satellite scene download event.

    Args:
        source: Satellite source downloaded from.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sat_scenes_downloaded_total.labels(source=source).inc()


def record_imagery_download_bytes(source: str, bytes_count: int) -> None:
    """Record imagery download volume in bytes.

    Args:
        source: Satellite source downloaded from.
        bytes_count: Number of bytes downloaded.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sat_imagery_download_bytes_total.labels(source=source).inc(bytes_count)


def record_baseline_established(commodity: str, country: str) -> None:
    """Record a spectral baseline establishment event.

    Args:
        commodity: EUDR commodity (e.g., 'soya', 'oil_palm').
        country: ISO 3166-1 alpha-2 country code (e.g., 'BR', 'ID').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sat_baselines_established_total.labels(
        commodity=commodity, country=country,
    ).inc()


def record_ndvi_calculation(index_type: str) -> None:
    """Record a spectral index calculation event.

    Args:
        index_type: Type of spectral index calculated (e.g., 'ndvi',
            'evi', 'nbr', 'ndmi', 'savi').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sat_ndvi_calculations_total.labels(index_type=index_type).inc()


def record_change_detection(method: str, result: str) -> None:
    """Record a change detection analysis event.

    Args:
        method: Detection method used (e.g., 'ndvi_differencing',
            'multi_source_fusion').
        result: Detection result ('no_change', 'deforestation',
            'degradation', 'reforestation', 'regrowth').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sat_change_detections_total.labels(method=method, result=result).inc()


def record_deforestation_detected(
    commodity: str, country: str, severity: str,
) -> None:
    """Record a deforestation detection event.

    Args:
        commodity: EUDR commodity affected.
        country: ISO 3166-1 alpha-2 country code.
        severity: Alert severity ('critical', 'warning', 'info').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sat_deforestation_detected_total.labels(
        commodity=commodity, country=country, severity=severity,
    ).inc()


def record_alert_generated(severity: str) -> None:
    """Record a satellite monitoring alert generation event.

    Args:
        severity: Alert severity ('critical', 'warning', 'info').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sat_alerts_generated_total.labels(severity=severity).inc()


def record_evidence_package(format: str) -> None:
    """Record an evidence package generation event.

    Args:
        format: Evidence package format ('json', 'pdf', 'csv',
            'eudr_xml').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sat_evidence_packages_total.labels(format=format).inc()


def record_monitoring_execution(status: str) -> None:
    """Record a monitoring schedule execution event.

    Args:
        status: Execution status ('success', 'error', 'timeout',
            'skipped').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sat_monitoring_executions_total.labels(status=status).inc()


def record_cloud_gap_fill(method: str) -> None:
    """Record a cloud gap fill operation event.

    Args:
        method: Gap fill method used ('temporal_composite',
            'sar_fusion', 'interpolation', 'nearest_clear').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sat_cloud_gap_fills_total.labels(method=method).inc()


def record_fusion_analysis(sources_count: int) -> None:
    """Record a multi-source fusion analysis event.

    Args:
        sources_count: Number of satellite sources fused (1-5).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sat_fusion_analyses_total.labels(
        sources_count=str(sources_count),
    ).inc()


def observe_analysis_duration(operation: str, seconds: float) -> None:
    """Record the duration of a satellite analysis operation.

    Args:
        operation: Type of analysis operation (e.g., 'scene_search',
            'baseline_establish', 'change_detect', 'fuse_sources',
            'cloud_fill', 'monitoring_execute', 'alert_generate',
            'evidence_generate', 'batch_analyze').
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sat_analysis_duration_seconds.labels(operation=operation).observe(seconds)


def observe_batch_duration(seconds: float) -> None:
    """Record the duration of a complete batch analysis job.

    Args:
        seconds: Total batch job wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sat_batch_duration_seconds.observe(seconds)


def set_active_monitoring_plots(count: int) -> None:
    """Set the gauge for number of plots under active monitoring.

    Args:
        count: Number of actively monitored plots. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sat_active_monitoring_plots.set(count)


def set_avg_detection_confidence(score: float) -> None:
    """Set the gauge for average detection confidence across monitored plots.

    Args:
        score: Average detection confidence score (0-100).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sat_avg_detection_confidence.set(score)


def record_api_error(operation: str) -> None:
    """Record an API error event by operation type.

    Args:
        operation: Type of operation that failed (e.g., 'scene_search',
            'scene_download', 'baseline_establish', 'change_detect',
            'fuse_sources', 'cloud_fill', 'monitoring_execute',
            'alert_generate', 'evidence_generate', 'batch_analyze').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sat_api_errors_total.labels(operation=operation).inc()


def set_data_quality_score(source: str, score: float) -> None:
    """Set the data quality score gauge for a satellite source.

    Args:
        source: Satellite source identifier (e.g., 'sentinel_2',
            'landsat_8', 'gfw_alerts').
        score: Data quality score (0-100).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sat_data_quality_score.labels(source=source).set(score)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "sat_scenes_queried_total",
    "sat_scenes_downloaded_total",
    "sat_imagery_download_bytes_total",
    "sat_baselines_established_total",
    "sat_ndvi_calculations_total",
    "sat_change_detections_total",
    "sat_deforestation_detected_total",
    "sat_alerts_generated_total",
    "sat_evidence_packages_total",
    "sat_monitoring_executions_total",
    "sat_cloud_gap_fills_total",
    "sat_fusion_analyses_total",
    "sat_analysis_duration_seconds",
    "sat_batch_duration_seconds",
    "sat_active_monitoring_plots",
    "sat_avg_detection_confidence",
    "sat_api_errors_total",
    "sat_data_quality_score",
    # Helper functions
    "record_scene_queried",
    "record_scene_downloaded",
    "record_imagery_download_bytes",
    "record_baseline_established",
    "record_ndvi_calculation",
    "record_change_detection",
    "record_deforestation_detected",
    "record_alert_generated",
    "record_evidence_package",
    "record_monitoring_execution",
    "record_cloud_gap_fill",
    "record_fusion_analysis",
    "observe_analysis_duration",
    "observe_batch_duration",
    "set_active_monitoring_plots",
    "set_avg_detection_confidence",
    "record_api_error",
    "set_data_quality_score",
]
