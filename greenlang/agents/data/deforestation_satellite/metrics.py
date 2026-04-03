# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-007: Deforestation Satellite Connector

12 Prometheus metrics for deforestation satellite connector service
monitoring with graceful fallback when prometheus_client is not installed.

Standard metrics (via MetricsFactory):
    1.  gl_deforestation_sat_operations_total (Counter, labels: type, tenant_id)
    2.  gl_deforestation_sat_processing_duration_seconds (Histogram, 12 buckets)
    3.  gl_deforestation_sat_validation_errors_total (Counter, labels: severity, type)
    4.  gl_deforestation_sat_batch_jobs_total (Counter, labels: status)
    5.  gl_deforestation_sat_active_jobs (Gauge)
    6.  gl_deforestation_sat_queue_size (Gauge)

Agent-specific metrics:
    7.  gl_deforestation_sat_scenes_acquired_total (Counter, labels: satellite, status)
    8.  gl_deforestation_sat_change_detections_total (Counter, labels: change_type, status)
    9.  gl_deforestation_sat_alerts_processed_total (Counter, labels: source, severity)
    10. gl_deforestation_sat_baseline_checks_total (Counter, labels: country, compliance_status)
    11. gl_deforestation_sat_classifications_total (Counter, labels: land_cover_type)
    12. gl_deforestation_sat_compliance_reports_total (Counter, labels: status)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-007 Deforestation Satellite Connector Agent (GL-DATA-GEO-003)
Status: Production Ready
"""

from __future__ import annotations

from greenlang.data_commons.metrics import (
    LONG_DURATION_BUCKETS,
    PROMETHEUS_AVAILABLE,
    MetricsFactory,
)

# ---------------------------------------------------------------------------
# Standard metrics (6 of 12) via factory
# ---------------------------------------------------------------------------

m = MetricsFactory(
    "gl_deforestation_sat",
    "Deforestation Satellite",
    duration_buckets=LONG_DURATION_BUCKETS,
)

# ---------------------------------------------------------------------------
# Agent-specific metrics (6 of 12)
# ---------------------------------------------------------------------------

scenes_acquired_total = m.create_custom_counter(
    "scenes_acquired_total",
    "Total satellite scenes acquired",
    labelnames=["satellite", "status"],
)

acquisition_duration_seconds = m.create_custom_histogram(
    "acquisition_duration_seconds",
    "Satellite acquisition duration in seconds",
    buckets=LONG_DURATION_BUCKETS,
    labelnames=["satellite"],
)

change_detections_total = m.create_custom_counter(
    "change_detections_total",
    "Total forest change detections performed",
    labelnames=["change_type", "status"],
)

alerts_processed_total = m.create_custom_counter(
    "alerts_processed_total",
    "Total deforestation alerts processed",
    labelnames=["source", "severity"],
)

baseline_checks_total = m.create_custom_counter(
    "baseline_checks_total",
    "Total baseline compliance checks performed",
    labelnames=["country", "compliance_status"],
)

classifications_total = m.create_custom_counter(
    "classifications_total",
    "Total land cover classifications performed",
    labelnames=["land_cover_type"],
)

compliance_reports_total = m.create_custom_counter(
    "compliance_reports_total",
    "Total EUDR compliance reports generated",
    labelnames=["status"],
)

pipeline_runs_total = m.create_custom_counter(
    "pipeline_runs_total",
    "Total pipeline stage executions",
    labelnames=["stage", "status"],
)

active_monitoring_jobs = m.create_custom_gauge(
    "active_monitoring_jobs",
    "Number of currently active monitoring jobs",
)

processing_errors_total = m.create_custom_counter(
    "processing_errors_total",
    "Total processing errors encountered",
    labelnames=["engine", "error_type"],
)

forest_area_monitored_ha = m.create_custom_gauge(
    "forest_area_monitored_ha",
    "Total forest area currently under monitoring in hectares",
)

pipeline_duration_seconds = m.create_custom_histogram(
    "pipeline_duration_seconds",
    "Pipeline stage execution duration in seconds",
    buckets=(
        0.01, 0.05, 0.1, 0.25, 0.5, 1.0,
        2.5, 5.0, 10.0, 30.0, 60.0, 300.0,
    ),
    labelnames=["stage"],
)


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_scene_acquired(satellite: str, status: str = "success") -> None:
    """Record a satellite scene acquisition event.

    Args:
        satellite: Satellite source (sentinel2, landsat8, landsat9, modis).
        status: Acquisition status (success, failed, partial).
    """
    m.safe_inc(scenes_acquired_total, 1, satellite=satellite, status=status)


def record_change_detection(change_type: str, status: str = "success") -> None:
    """Record a forest change detection event.

    Args:
        change_type: Detected change type (clear_cut, degradation, etc.).
        status: Detection status (success, failed).
    """
    m.safe_inc(change_detections_total, 1, change_type=change_type, status=status)


def record_alert_processed(source: str, severity: str) -> None:
    """Record a deforestation alert processing event.

    Args:
        source: Alert source (glad, radd, firms, gfw, internal).
        severity: Alert severity (low, medium, high, critical).
    """
    m.safe_inc(alerts_processed_total, 1, source=source, severity=severity)


def record_baseline_check(country: str, compliance_status: str) -> None:
    """Record a baseline compliance check event.

    Args:
        country: ISO 3166-1 alpha-3 country code.
        compliance_status: Compliance determination (compliant, review_required,
            non_compliant).
    """
    m.safe_inc(
        baseline_checks_total, 1,
        country=country, compliance_status=compliance_status,
    )


def record_classification(land_cover_type: str) -> None:
    """Record a land cover classification event.

    Args:
        land_cover_type: Classified land cover type (dense_forest, open_forest,
            shrubland, grassland, etc.).
    """
    m.safe_inc(classifications_total, 1, land_cover_type=land_cover_type)


def record_compliance_report(status: str) -> None:
    """Record a compliance report generation event.

    Args:
        status: Report compliance status (compliant, review_required,
            non_compliant).
    """
    m.safe_inc(compliance_reports_total, 1, status=status)


def record_pipeline_run(stage: str, status: str = "completed") -> None:
    """Record a pipeline stage execution event.

    Args:
        stage: Pipeline stage name (initialization, image_acquisition, etc.).
        status: Execution status (completed, failed, skipped).
    """
    m.safe_inc(pipeline_runs_total, 1, stage=stage, status=status)


def record_processing_error(engine: str, error_type: str) -> None:
    """Record a processing error event.

    Args:
        engine: Engine that produced the error (satellite_data, forest_change,
            alert_aggregation, baseline_assessment, classifier, compliance_report,
            monitoring_pipeline).
        error_type: Error classification (validation, connection, timeout,
            computation, data, unknown).
    """
    m.safe_inc(processing_errors_total, 1, engine=engine, error_type=error_type)


def update_active_jobs(count: int) -> None:
    """Set the active monitoring jobs gauge.

    Args:
        count: Current number of active monitoring jobs.
    """
    m.safe_set(active_monitoring_jobs, count)


def update_forest_area(area_ha: float) -> None:
    """Set the forest area monitored gauge.

    Args:
        area_ha: Total forest area under monitoring in hectares.
    """
    m.safe_set(forest_area_monitored_ha, area_ha)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "m",
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
