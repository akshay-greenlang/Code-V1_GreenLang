# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-022: Protected Area Validator

20 Prometheus metrics for protected area validator agent service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_pav_`` prefix (GreenLang EUDR Protected
Area Validator) for consistent identification in Prometheus queries, Grafana
dashboards, and alerting rules.

Metrics (20):
    Counters (10):
        1.  gl_eudr_pav_overlaps_detected_total         - Spatial overlaps detected
        2.  gl_eudr_pav_buffer_analyses_total            - Buffer zone analyses completed
        3.  gl_eudr_pav_designation_validations_total    - Designation validations
        4.  gl_eudr_pav_proximity_alerts_total           - Proximity alerts generated
        5.  gl_eudr_pav_compliance_records_total         - Compliance records created
        6.  gl_eudr_pav_conservation_assessments_total   - Conservation assessments
        7.  gl_eudr_pav_reports_generated_total          - Reports generated
        8.  gl_eudr_pav_integration_events_total         - Integration events published
        9.  gl_eudr_pav_batch_jobs_total                 - Batch screening jobs
        10. gl_eudr_pav_api_errors_total                 - API errors by operation

    Histograms (4):
        11. gl_eudr_pav_overlap_latency_seconds          - Overlap detection latency
        12. gl_eudr_pav_buffer_analysis_seconds           - Buffer analysis duration
        13. gl_eudr_pav_risk_scoring_seconds              - Risk scoring duration
        14. gl_eudr_pav_report_generation_seconds         - Report generation duration

    Gauges (6):
        15. gl_eudr_pav_protected_areas_loaded            - Protected areas in database
        16. gl_eudr_pav_active_overlaps                   - Active overlap detections
        17. gl_eudr_pav_active_violations                 - Active compliance violations
        18. gl_eudr_pav_pending_alerts                    - Pending proximity alerts
        19. gl_eudr_pav_wdpa_data_age_days               - WDPA data age in days
        20. gl_eudr_pav_batch_backlog                     - Batch processing backlog

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-022 Protected Area Validator (GL-EUDR-PAV-022)
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
        "protected area validator metrics disabled"
    )

# ---------------------------------------------------------------------------
# Safe metric registration helpers
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    from prometheus_client import REGISTRY as _REGISTRY

    def _safe_counter(name: str, doc: str, labels: list) -> Counter:
        """Safely create or retrieve a Counter metric."""
        try:
            return Counter(name, doc, labels, registry=_REGISTRY)
        except ValueError:
            return _REGISTRY._names_to_collectors[name]

    def _safe_histogram(name: str, doc: str, labels: list, buckets: tuple) -> Histogram:
        """Safely create or retrieve a Histogram metric."""
        try:
            return Histogram(name, doc, labels, buckets=buckets, registry=_REGISTRY)
        except ValueError:
            return _REGISTRY._names_to_collectors[name]

    def _safe_gauge(name: str, doc: str, labels: list) -> Gauge:
        """Safely create or retrieve a Gauge metric."""
        try:
            return Gauge(name, doc, labels, registry=_REGISTRY)
        except ValueError:
            return _REGISTRY._names_to_collectors[name]

# ---------------------------------------------------------------------------
# Metric definitions (20 metrics)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # Counters (10)
    _overlaps_detected_total = _safe_counter(
        "gl_eudr_pav_overlaps_detected_total",
        "Total spatial overlaps detected between plots and protected areas",
        ["iucn_category", "overlap_type", "country_code"],
    )
    _buffer_analyses_total = _safe_counter(
        "gl_eudr_pav_buffer_analyses_total",
        "Total buffer zone analyses completed",
        ["proximity_tier", "country_code"],
    )
    _designation_validations_total = _safe_counter(
        "gl_eudr_pav_designation_validations_total",
        "Total designation validations performed",
        ["designation_level"],
    )
    _proximity_alerts_total = _safe_counter(
        "gl_eudr_pav_proximity_alerts_total",
        "Total high-risk proximity alerts generated",
        ["severity", "country_code"],
    )
    _compliance_records_total = _safe_counter(
        "gl_eudr_pav_compliance_records_total",
        "Total compliance records created",
        ["status"],
    )
    _conservation_assessments_total = _safe_counter(
        "gl_eudr_pav_conservation_assessments_total",
        "Total conservation status assessments completed",
        ["country_code"],
    )
    _reports_generated_total = _safe_counter(
        "gl_eudr_pav_reports_generated_total",
        "Total compliance reports generated",
        ["report_type", "format"],
    )
    _integration_events_total = _safe_counter(
        "gl_eudr_pav_integration_events_total",
        "Total integration events published to event bus",
        ["event_type"],
    )
    _batch_jobs_total = _safe_counter(
        "gl_eudr_pav_batch_jobs_total",
        "Total batch screening jobs completed",
        ["status"],
    )
    _api_errors_total = _safe_counter(
        "gl_eudr_pav_api_errors_total",
        "Total API errors by operation",
        ["operation"],
    )

    # Histograms (4)
    _overlap_latency_seconds = _safe_histogram(
        "gl_eudr_pav_overlap_latency_seconds",
        "Spatial overlap detection latency in seconds",
        ["iucn_category"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _buffer_analysis_seconds = _safe_histogram(
        "gl_eudr_pav_buffer_analysis_seconds",
        "Buffer zone analysis duration in seconds",
        [],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    _risk_scoring_seconds = _safe_histogram(
        "gl_eudr_pav_risk_scoring_seconds",
        "Risk scoring computation duration in seconds",
        [],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    )
    _report_generation_seconds = _safe_histogram(
        "gl_eudr_pav_report_generation_seconds",
        "Compliance report generation duration in seconds",
        ["report_type"],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
    )

    # Gauges (6)
    _protected_areas_loaded = _safe_gauge(
        "gl_eudr_pav_protected_areas_loaded",
        "Number of protected areas loaded in spatial database",
        [],
    )
    _active_overlaps = _safe_gauge(
        "gl_eudr_pav_active_overlaps",
        "Number of active overlap detections",
        ["risk_level"],
    )
    _active_violations = _safe_gauge(
        "gl_eudr_pav_active_violations",
        "Number of active compliance violations",
        ["status"],
    )
    _pending_alerts = _safe_gauge(
        "gl_eudr_pav_pending_alerts",
        "Number of pending proximity alerts",
        ["severity"],
    )
    _wdpa_data_age_days = _safe_gauge(
        "gl_eudr_pav_wdpa_data_age_days",
        "WDPA data age in days since last update",
        [],
    )
    _batch_backlog = _safe_gauge(
        "gl_eudr_pav_batch_backlog",
        "Number of plots in batch processing backlog",
        [],
    )


# ---------------------------------------------------------------------------
# Helper functions (20 matching 20 metrics)
# ---------------------------------------------------------------------------

def record_overlap_detected(
    iucn_category: str = "NR",
    overlap_type: str = "clear",
    country_code: str = "unknown",
) -> None:
    """Record a spatial overlap detection."""
    if PROMETHEUS_AVAILABLE:
        _overlaps_detected_total.labels(
            iucn_category=iucn_category,
            overlap_type=overlap_type,
            country_code=country_code,
        ).inc()


def record_buffer_analysis(
    proximity_tier: str = "peripheral",
    country_code: str = "unknown",
) -> None:
    """Record a buffer zone analysis completion."""
    if PROMETHEUS_AVAILABLE:
        _buffer_analyses_total.labels(
            proximity_tier=proximity_tier,
            country_code=country_code,
        ).inc()


def record_designation_validation(designation_level: str = "national") -> None:
    """Record a designation validation completion."""
    if PROMETHEUS_AVAILABLE:
        _designation_validations_total.labels(designation_level=designation_level).inc()


def record_proximity_alert(severity: str = "standard", country_code: str = "unknown") -> None:
    """Record a proximity alert generation."""
    if PROMETHEUS_AVAILABLE:
        _proximity_alerts_total.labels(severity=severity, country_code=country_code).inc()


def record_compliance_record(status: str = "detected") -> None:
    """Record a compliance record creation."""
    if PROMETHEUS_AVAILABLE:
        _compliance_records_total.labels(status=status).inc()


def record_conservation_assessment(country_code: str = "unknown") -> None:
    """Record a conservation assessment completion."""
    if PROMETHEUS_AVAILABLE:
        _conservation_assessments_total.labels(country_code=country_code).inc()


def record_report_generated(report_type: str = "full_compliance", fmt: str = "pdf") -> None:
    """Record a compliance report generation."""
    if PROMETHEUS_AVAILABLE:
        _reports_generated_total.labels(report_type=report_type, format=fmt).inc()


def record_integration_event(event_type: str = "overlap_detected") -> None:
    """Record an integration event publication."""
    if PROMETHEUS_AVAILABLE:
        _integration_events_total.labels(event_type=event_type).inc()


def record_batch_job(status: str = "completed") -> None:
    """Record a batch screening job completion."""
    if PROMETHEUS_AVAILABLE:
        _batch_jobs_total.labels(status=status).inc()


def record_api_error(operation: str) -> None:
    """Record an API error."""
    if PROMETHEUS_AVAILABLE:
        _api_errors_total.labels(operation=operation).inc()


def observe_overlap_latency(duration_seconds: float, iucn_category: str = "NR") -> None:
    """Observe overlap detection latency."""
    if PROMETHEUS_AVAILABLE:
        _overlap_latency_seconds.labels(iucn_category=iucn_category).observe(duration_seconds)


def observe_buffer_analysis_duration(duration_seconds: float) -> None:
    """Observe buffer analysis duration."""
    if PROMETHEUS_AVAILABLE:
        _buffer_analysis_seconds.observe(duration_seconds)


def observe_risk_scoring_duration(duration_seconds: float) -> None:
    """Observe risk scoring duration."""
    if PROMETHEUS_AVAILABLE:
        _risk_scoring_seconds.observe(duration_seconds)


def observe_report_generation_duration(duration_seconds: float, report_type: str = "full_compliance") -> None:
    """Observe report generation duration."""
    if PROMETHEUS_AVAILABLE:
        _report_generation_seconds.labels(report_type=report_type).observe(duration_seconds)


def set_protected_areas_loaded(count: int) -> None:
    """Set the number of protected areas loaded."""
    if PROMETHEUS_AVAILABLE:
        _protected_areas_loaded.set(count)


def set_active_overlaps(count: int, risk_level: str = "all") -> None:
    """Set the number of active overlap detections."""
    if PROMETHEUS_AVAILABLE:
        _active_overlaps.labels(risk_level=risk_level).set(count)


def set_active_violations(count: int, status: str = "all") -> None:
    """Set the number of active compliance violations."""
    if PROMETHEUS_AVAILABLE:
        _active_violations.labels(status=status).set(count)


def set_pending_alerts(count: int, severity: str = "all") -> None:
    """Set the number of pending proximity alerts."""
    if PROMETHEUS_AVAILABLE:
        _pending_alerts.labels(severity=severity).set(count)


def set_wdpa_data_age_days(days: int) -> None:
    """Set the WDPA data age in days."""
    if PROMETHEUS_AVAILABLE:
        _wdpa_data_age_days.set(days)


def set_batch_backlog(count: int) -> None:
    """Set the batch processing backlog size."""
    if PROMETHEUS_AVAILABLE:
        _batch_backlog.set(count)
