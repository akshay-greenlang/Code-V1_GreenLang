# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-007: Deforestation Satellite Connector

12 Prometheus metrics for deforestation satellite connector service
monitoring with graceful fallback when prometheus_client is not installed.

Metrics:
    1.  gl_deforestation_sat_scenes_acquired_total (Counter) [satellite, status]
    2.  gl_deforestation_sat_acquisition_duration_seconds (Histogram) [satellite]
    3.  gl_deforestation_sat_change_detections_total (Counter) [change_type, status]
    4.  gl_deforestation_sat_alerts_processed_total (Counter) [source, severity]
    5.  gl_deforestation_sat_baseline_checks_total (Counter) [country, compliance_status]
    6.  gl_deforestation_sat_classifications_total (Counter) [land_cover_type]
    7.  gl_deforestation_sat_compliance_reports_total (Counter) [status]
    8.  gl_deforestation_sat_pipeline_runs_total (Counter) [stage, status]
    9.  gl_deforestation_sat_active_monitoring_jobs (Gauge) []
    10. gl_deforestation_sat_processing_errors_total (Counter) [engine, error_type]
    11. gl_deforestation_sat_forest_area_monitored_ha (Gauge) []
    12. gl_deforestation_sat_pipeline_duration_seconds (Histogram) [stage]

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-007 Deforestation Satellite Connector Agent (GL-DATA-GEO-003)
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
        "prometheus_client not installed; deforestation satellite metrics disabled"
    )


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Satellite scenes acquired by satellite type and status
    scenes_acquired_total = Counter(
        "gl_deforestation_sat_scenes_acquired_total",
        "Total satellite scenes acquired",
        labelnames=["satellite", "status"],
    )

    # 2. Acquisition duration histogram (buckets from sub-second to minutes)
    acquisition_duration_seconds = Histogram(
        "gl_deforestation_sat_acquisition_duration_seconds",
        "Satellite acquisition duration in seconds",
        labelnames=["satellite"],
        buckets=(
            0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
            10.0, 30.0, 60.0, 120.0, 300.0, 600.0,
        ),
    )

    # 3. Change detections by change type and status
    change_detections_total = Counter(
        "gl_deforestation_sat_change_detections_total",
        "Total forest change detections performed",
        labelnames=["change_type", "status"],
    )

    # 4. Alerts processed by source and severity
    alerts_processed_total = Counter(
        "gl_deforestation_sat_alerts_processed_total",
        "Total deforestation alerts processed",
        labelnames=["source", "severity"],
    )

    # 5. Baseline checks by country and compliance status
    baseline_checks_total = Counter(
        "gl_deforestation_sat_baseline_checks_total",
        "Total baseline compliance checks performed",
        labelnames=["country", "compliance_status"],
    )

    # 6. Classifications by land cover type
    classifications_total = Counter(
        "gl_deforestation_sat_classifications_total",
        "Total land cover classifications performed",
        labelnames=["land_cover_type"],
    )

    # 7. Compliance reports by status
    compliance_reports_total = Counter(
        "gl_deforestation_sat_compliance_reports_total",
        "Total EUDR compliance reports generated",
        labelnames=["status"],
    )

    # 8. Pipeline runs by stage and status
    pipeline_runs_total = Counter(
        "gl_deforestation_sat_pipeline_runs_total",
        "Total pipeline stage executions",
        labelnames=["stage", "status"],
    )

    # 9. Active monitoring jobs gauge
    active_monitoring_jobs = Gauge(
        "gl_deforestation_sat_active_monitoring_jobs",
        "Number of currently active monitoring jobs",
    )

    # 10. Processing errors by engine and error type
    processing_errors_total = Counter(
        "gl_deforestation_sat_processing_errors_total",
        "Total processing errors encountered",
        labelnames=["engine", "error_type"],
    )

    # 11. Forest area monitored gauge (hectares)
    forest_area_monitored_ha = Gauge(
        "gl_deforestation_sat_forest_area_monitored_ha",
        "Total forest area currently under monitoring in hectares",
    )

    # 12. Pipeline duration histogram by stage
    pipeline_duration_seconds = Histogram(
        "gl_deforestation_sat_pipeline_duration_seconds",
        "Pipeline stage execution duration in seconds",
        labelnames=["stage"],
        buckets=(
            0.01, 0.05, 0.1, 0.25, 0.5, 1.0,
            2.5, 5.0, 10.0, 30.0, 60.0, 300.0,
        ),
    )

else:
    # No-op placeholders
    scenes_acquired_total = None  # type: ignore[assignment]
    acquisition_duration_seconds = None  # type: ignore[assignment]
    change_detections_total = None  # type: ignore[assignment]
    alerts_processed_total = None  # type: ignore[assignment]
    baseline_checks_total = None  # type: ignore[assignment]
    classifications_total = None  # type: ignore[assignment]
    compliance_reports_total = None  # type: ignore[assignment]
    pipeline_runs_total = None  # type: ignore[assignment]
    active_monitoring_jobs = None  # type: ignore[assignment]
    processing_errors_total = None  # type: ignore[assignment]
    forest_area_monitored_ha = None  # type: ignore[assignment]
    pipeline_duration_seconds = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_scene_acquired(satellite: str, status: str = "success") -> None:
    """Record a satellite scene acquisition event.

    Args:
        satellite: Satellite source (sentinel2, landsat8, landsat9, modis).
        status: Acquisition status (success, failed, partial).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    scenes_acquired_total.labels(satellite=satellite, status=status).inc()


def record_change_detection(change_type: str, status: str = "success") -> None:
    """Record a forest change detection event.

    Args:
        change_type: Detected change type (clear_cut, degradation, etc.).
        status: Detection status (success, failed).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    change_detections_total.labels(change_type=change_type, status=status).inc()


def record_alert_processed(source: str, severity: str) -> None:
    """Record a deforestation alert processing event.

    Args:
        source: Alert source (glad, radd, firms, gfw, internal).
        severity: Alert severity (low, medium, high, critical).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    alerts_processed_total.labels(source=source, severity=severity).inc()


def record_baseline_check(country: str, compliance_status: str) -> None:
    """Record a baseline compliance check event.

    Args:
        country: ISO 3166-1 alpha-3 country code.
        compliance_status: Compliance determination (compliant, review_required,
            non_compliant).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    baseline_checks_total.labels(
        country=country, compliance_status=compliance_status,
    ).inc()


def record_classification(land_cover_type: str) -> None:
    """Record a land cover classification event.

    Args:
        land_cover_type: Classified land cover type (dense_forest, open_forest,
            shrubland, grassland, etc.).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    classifications_total.labels(land_cover_type=land_cover_type).inc()


def record_compliance_report(status: str) -> None:
    """Record a compliance report generation event.

    Args:
        status: Report compliance status (compliant, review_required,
            non_compliant).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    compliance_reports_total.labels(status=status).inc()


def record_pipeline_run(stage: str, status: str = "completed") -> None:
    """Record a pipeline stage execution event.

    Args:
        stage: Pipeline stage name (initialization, image_acquisition, etc.).
        status: Execution status (completed, failed, skipped).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pipeline_runs_total.labels(stage=stage, status=status).inc()


def record_processing_error(engine: str, error_type: str) -> None:
    """Record a processing error event.

    Args:
        engine: Engine that produced the error (satellite_data, forest_change,
            alert_aggregation, baseline_assessment, classifier, compliance_report,
            monitoring_pipeline).
        error_type: Error classification (validation, connection, timeout,
            computation, data, unknown).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    processing_errors_total.labels(engine=engine, error_type=error_type).inc()


def update_active_jobs(count: int) -> None:
    """Set the active monitoring jobs gauge.

    Args:
        count: Current number of active monitoring jobs.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    active_monitoring_jobs.set(count)


def update_forest_area(area_ha: float) -> None:
    """Set the forest area monitored gauge.

    Args:
        area_ha: Total forest area under monitoring in hectares.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    forest_area_monitored_ha.set(area_ha)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "scenes_acquired_total",
    "acquisition_duration_seconds",
    "change_detections_total",
    "alerts_processed_total",
    "baseline_checks_total",
    "classifications_total",
    "compliance_reports_total",
    "pipeline_runs_total",
    "active_monitoring_jobs",
    "processing_errors_total",
    "forest_area_monitored_ha",
    "pipeline_duration_seconds",
    # Helper functions
    "record_scene_acquired",
    "record_change_detection",
    "record_alert_processed",
    "record_baseline_check",
    "record_classification",
    "record_compliance_report",
    "record_pipeline_run",
    "record_processing_error",
    "update_active_jobs",
    "update_forest_area",
]
