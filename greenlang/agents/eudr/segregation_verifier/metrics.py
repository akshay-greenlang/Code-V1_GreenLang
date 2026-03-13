# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-010: Segregation Verifier

18 Prometheus metrics for segregation verifier agent service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_sgv_`` prefix (GreenLang EUDR
Segregation Verifier) for consistent identification in Prometheus
queries, Grafana dashboards, and alerting rules across the GreenLang
platform.

Metrics (18 per PRD Section 7.3):
    Counters (13):
        1.  gl_eudr_sgv_scp_validations_total                  - SCP validations
        2.  gl_eudr_sgv_scp_failures_total                     - SCP validation failures
        3.  gl_eudr_sgv_storage_audits_total                   - Storage zone audits
        4.  gl_eudr_sgv_transport_checks_total                 - Transport checks
        5.  gl_eudr_sgv_processing_checks_total                - Processing line checks
        6.  gl_eudr_sgv_contamination_events_total             - Contamination events
        7.  gl_eudr_sgv_contamination_critical_total           - Critical contamination
        8.  gl_eudr_sgv_labels_verified_total                  - Labels verified
        9.  gl_eudr_sgv_label_failures_total                   - Label failures
        10. gl_eudr_sgv_assessments_total                      - Facility assessments
        11. gl_eudr_sgv_reports_generated_total                - Reports generated
        12. gl_eudr_sgv_batch_jobs_total                       - Batch processing jobs
        13. gl_eudr_sgv_api_errors_total                       - API errors

    Histograms (3):
        14. gl_eudr_sgv_scp_validation_duration_seconds        - SCP validation latency
        15. gl_eudr_sgv_contamination_detection_duration_seconds - Detection latency
        16. gl_eudr_sgv_assessment_duration_seconds             - Assessment latency

    Gauges (2):
        17. gl_eudr_sgv_active_segregation_points              - Active SCPs by status
        18. gl_eudr_sgv_avg_facility_score                     - Avg facility score

Label Values Reference:
    scp_type:
        storage, transport, processing, handling, loading_unloading.
    commodity:
        cattle, cocoa, coffee, oil_palm, rubber, soya, wood.
    reason:
        barrier_inadequate, distance_violation, labeling_missing,
        documentation_gap, cleaning_not_verified, changeover_insufficient,
        contamination_detected, verification_expired.
    facility_id:
        Unique facility identifier string.
    storage_type:
        silo, warehouse_bay, tank, container_yard, cold_room,
        dry_store, bonded_area, open_yard, covered_shed,
        sealed_unit, locked_cage, segregated_floor.
    transport_type:
        bulk_truck, container_truck, tanker, dry_bulk_vessel,
        container_vessel, tanker_vessel, rail_hopper,
        rail_container, barge, air_freight.
    dedicated:
        true, false.
    line_type:
        extraction, pressing, milling, refining, roasting,
        fermenting, drying, cutting, tanning, spinning,
        smelting, fractionation, blending_line, packaging, grading.
    pathway:
        shared_storage, shared_transport, shared_processing,
        shared_equipment, temporal_overlap, adjacent_storage,
        residual_material, handling_error, labeling_error,
        documentation_error.
    severity:
        critical, major, minor, observation.
    label_type:
        compliance_tag, zone_sign, vehicle_placard,
        container_seal_label, batch_sticker, pallet_marker,
        silo_sign, processing_line_marker.
    report_type:
        facility, scp, contamination, labeling, assessment, summary.
    format:
        json, pdf, csv, eudr_xml.
    status:
        success, failure, partial.
    endpoint:
        register_scp, validate_scp, register_zone, record_event,
        register_vehicle, verify_transport, register_line,
        record_changeover, detect_contamination, record_contamination,
        verify_labels, run_assessment, generate_report, search_scp,
        batch_import.
    status_code:
        400, 404, 409, 422, 500, 503.

Example:
    >>> from greenlang.agents.eudr.segregation_verifier.metrics import (
    ...     record_scp_validation,
    ...     record_contamination_event,
    ...     observe_scp_validation_duration,
    ...     set_active_segregation_points,
    ... )
    >>> record_scp_validation("storage", "cocoa")
    >>> record_contamination_event("shared_storage", "major")
    >>> observe_scp_validation_duration(0.045)
    >>> set_active_segregation_points("verified", 156)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-010 Segregation Verifier (GL-EUDR-SGV-010)
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
        "segregation verifier metrics disabled"
    )

# ---------------------------------------------------------------------------
# Safe metric registration helpers (avoid collisions with other modules)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    from prometheus_client import REGISTRY as _REGISTRY

    def _safe_counter(
        name: str, doc: str, labelnames: list = None,  # type: ignore[assignment]
    ):  # type: ignore[return]
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
    ):  # type: ignore[return]
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
    ):  # type: ignore[return]
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
# Metric definitions (18 metrics per PRD Section 7.3)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # -- Counters (13) -------------------------------------------------------

    # 1. SCP validations by type and commodity
    sgv_scp_validations_total = _safe_counter(
        "gl_eudr_sgv_scp_validations_total",
        "Total segregation control point validations by type and commodity",
        labelnames=["scp_type", "commodity"],
    )

    # 2. SCP validation failures by type and reason
    sgv_scp_failures_total = _safe_counter(
        "gl_eudr_sgv_scp_failures_total",
        "Total SCP validation failures by type and reason",
        labelnames=["scp_type", "reason"],
    )

    # 3. Storage zone audits by facility and storage type
    sgv_storage_audits_total = _safe_counter(
        "gl_eudr_sgv_storage_audits_total",
        "Total storage zone audits by facility and storage type",
        labelnames=["facility_id", "storage_type"],
    )

    # 4. Transport segregation checks by transport type and dedication
    sgv_transport_checks_total = _safe_counter(
        "gl_eudr_sgv_transport_checks_total",
        "Total transport segregation checks by type and dedication status",
        labelnames=["transport_type", "dedicated"],
    )

    # 5. Processing line checks by line type and dedication
    sgv_processing_checks_total = _safe_counter(
        "gl_eudr_sgv_processing_checks_total",
        "Total processing line checks by type and dedication status",
        labelnames=["line_type", "dedicated"],
    )

    # 6. Contamination events by pathway and severity
    sgv_contamination_events_total = _safe_counter(
        "gl_eudr_sgv_contamination_events_total",
        "Total contamination events detected by pathway and severity",
        labelnames=["pathway", "severity"],
    )

    # 7. Critical contamination events by facility
    sgv_contamination_critical_total = _safe_counter(
        "gl_eudr_sgv_contamination_critical_total",
        "Total critical contamination events by facility",
        labelnames=["facility_id"],
    )

    # 8. Labels verified by label type
    sgv_labels_verified_total = _safe_counter(
        "gl_eudr_sgv_labels_verified_total",
        "Total labels verified by label type",
        labelnames=["label_type"],
    )

    # 9. Label failures by label type and reason
    sgv_label_failures_total = _safe_counter(
        "gl_eudr_sgv_label_failures_total",
        "Total label verification failures by type and reason",
        labelnames=["label_type", "reason"],
    )

    # 10. Facility assessments by facility
    sgv_assessments_total = _safe_counter(
        "gl_eudr_sgv_assessments_total",
        "Total facility assessments performed by facility",
        labelnames=["facility_id"],
    )

    # 11. Reports generated by report type and format
    sgv_reports_generated_total = _safe_counter(
        "gl_eudr_sgv_reports_generated_total",
        "Total segregation reports generated by type and format",
        labelnames=["report_type", "format"],
    )

    # 12. Batch processing jobs by status
    sgv_batch_jobs_total = _safe_counter(
        "gl_eudr_sgv_batch_jobs_total",
        "Total batch processing jobs by status",
        labelnames=["status"],
    )

    # 13. API errors by endpoint and status code
    sgv_api_errors_total = _safe_counter(
        "gl_eudr_sgv_api_errors_total",
        "Total API errors by endpoint and status code",
        labelnames=["endpoint", "status_code"],
    )

    # -- Histograms (3) ------------------------------------------------------

    # 14. SCP validation latency
    sgv_scp_validation_duration_seconds = _safe_histogram(
        "gl_eudr_sgv_scp_validation_duration_seconds",
        "Duration of SCP validation operations in seconds",
        buckets=(
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
            1.0, 2.5, 5.0, 10.0, 30.0,
        ),
    )

    # 15. Contamination detection latency
    sgv_contamination_detection_duration_seconds = _safe_histogram(
        "gl_eudr_sgv_contamination_detection_duration_seconds",
        "Duration of contamination detection scans in seconds",
        buckets=(
            0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
            1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
        ),
    )

    # 16. Facility assessment latency
    sgv_assessment_duration_seconds = _safe_histogram(
        "gl_eudr_sgv_assessment_duration_seconds",
        "Duration of facility assessment operations in seconds",
        buckets=(
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
            1.0, 2.5, 5.0, 10.0,
        ),
    )

    # -- Gauges (2) ----------------------------------------------------------

    # 17. Active segregation control points by status
    sgv_active_segregation_points = _safe_gauge(
        "gl_eudr_sgv_active_segregation_points",
        "Number of active segregation control points by status",
        labelnames=["status"],
    )

    # 18. Average facility assessment score
    sgv_avg_facility_score = _safe_gauge(
        "gl_eudr_sgv_avg_facility_score",
        "Average facility segregation assessment score (0-100)",
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    sgv_scp_validations_total = None                        # type: ignore[assignment]
    sgv_scp_failures_total = None                           # type: ignore[assignment]
    sgv_storage_audits_total = None                         # type: ignore[assignment]
    sgv_transport_checks_total = None                       # type: ignore[assignment]
    sgv_processing_checks_total = None                      # type: ignore[assignment]
    sgv_contamination_events_total = None                   # type: ignore[assignment]
    sgv_contamination_critical_total = None                 # type: ignore[assignment]
    sgv_labels_verified_total = None                        # type: ignore[assignment]
    sgv_label_failures_total = None                         # type: ignore[assignment]
    sgv_assessments_total = None                            # type: ignore[assignment]
    sgv_reports_generated_total = None                      # type: ignore[assignment]
    sgv_batch_jobs_total = None                             # type: ignore[assignment]
    sgv_api_errors_total = None                             # type: ignore[assignment]
    sgv_scp_validation_duration_seconds = None              # type: ignore[assignment]
    sgv_contamination_detection_duration_seconds = None     # type: ignore[assignment]
    sgv_assessment_duration_seconds = None                  # type: ignore[assignment]
    sgv_active_segregation_points = None                    # type: ignore[assignment]
    sgv_avg_facility_score = None                           # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_scp_validation(scp_type: str, commodity: str) -> None:
    """Record a segregation control point validation event.

    Args:
        scp_type: Type of SCP (storage, transport, processing,
            handling, loading_unloading).
        commodity: EUDR commodity (cattle, cocoa, coffee, oil_palm,
            rubber, soya, wood).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sgv_scp_validations_total.labels(
        scp_type=scp_type, commodity=commodity,
    ).inc()


def record_scp_failure(scp_type: str, reason: str) -> None:
    """Record a segregation control point validation failure.

    Args:
        scp_type: Type of SCP that failed validation.
        reason: Reason for failure (barrier_inadequate,
            distance_violation, labeling_missing, etc.).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sgv_scp_failures_total.labels(
        scp_type=scp_type, reason=reason,
    ).inc()


def record_storage_audit(facility_id: str, storage_type: str) -> None:
    """Record a storage zone audit event.

    Args:
        facility_id: Identifier of the facility audited.
        storage_type: Type of storage zone audited.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sgv_storage_audits_total.labels(
        facility_id=facility_id, storage_type=storage_type,
    ).inc()


def record_transport_check(
    transport_type: str, dedicated: str,
) -> None:
    """Record a transport segregation check event.

    Args:
        transport_type: Type of transport vehicle checked.
        dedicated: Whether vehicle is dedicated ("true" or "false").
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sgv_transport_checks_total.labels(
        transport_type=transport_type, dedicated=dedicated,
    ).inc()


def record_processing_check(
    line_type: str, dedicated: str,
) -> None:
    """Record a processing line segregation check event.

    Args:
        line_type: Type of processing line checked.
        dedicated: Whether line is dedicated ("true" or "false").
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sgv_processing_checks_total.labels(
        line_type=line_type, dedicated=dedicated,
    ).inc()


def record_contamination_event(
    pathway: str, severity: str,
) -> None:
    """Record a contamination event detection.

    Args:
        pathway: Contamination pathway (shared_storage,
            shared_transport, shared_processing, etc.).
        severity: Severity level (critical, major, minor, observation).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sgv_contamination_events_total.labels(
        pathway=pathway, severity=severity,
    ).inc()


def record_contamination_critical(facility_id: str) -> None:
    """Record a critical contamination event at a facility.

    Args:
        facility_id: Identifier of the affected facility.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sgv_contamination_critical_total.labels(
        facility_id=facility_id,
    ).inc()


def record_label_verified(label_type: str) -> None:
    """Record a label verification event.

    Args:
        label_type: Type of label verified (compliance_tag,
            zone_sign, vehicle_placard, etc.).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sgv_labels_verified_total.labels(label_type=label_type).inc()


def record_label_failure(label_type: str, reason: str) -> None:
    """Record a label verification failure.

    Args:
        label_type: Type of label that failed verification.
        reason: Reason for failure (missing, damaged, expired, etc.).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sgv_label_failures_total.labels(
        label_type=label_type, reason=reason,
    ).inc()


def record_assessment(facility_id: str) -> None:
    """Record a facility assessment event.

    Args:
        facility_id: Identifier of the assessed facility.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sgv_assessments_total.labels(facility_id=facility_id).inc()


def record_report_generated(
    report_type: str, report_format: str,
) -> None:
    """Record a report generation event.

    Args:
        report_type: Type of report (facility, scp, contamination,
            labeling, assessment, summary).
        report_format: Format of the report (json, pdf, csv, eudr_xml).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sgv_reports_generated_total.labels(
        report_type=report_type, format=report_format,
    ).inc()


def record_batch_job(status: str) -> None:
    """Record a batch processing job event.

    Args:
        status: Job status (success, failure, partial).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sgv_batch_jobs_total.labels(status=status).inc()


def record_api_error(endpoint: str, status_code: str) -> None:
    """Record an API error event.

    Args:
        endpoint: API endpoint that generated the error (register_scp,
            validate_scp, register_zone, etc.).
        status_code: HTTP status code (400, 404, 409, 422, 500, 503).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sgv_api_errors_total.labels(
        endpoint=endpoint, status_code=status_code,
    ).inc()


def observe_scp_validation_duration(seconds: float) -> None:
    """Record the duration of an SCP validation operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sgv_scp_validation_duration_seconds.observe(seconds)


def observe_contamination_detection_duration(seconds: float) -> None:
    """Record the duration of a contamination detection scan.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sgv_contamination_detection_duration_seconds.observe(seconds)


def observe_assessment_duration(seconds: float) -> None:
    """Record the duration of a facility assessment operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sgv_assessment_duration_seconds.observe(seconds)


def set_active_segregation_points(status: str, count: int) -> None:
    """Set the gauge for active segregation control points by status.

    Args:
        status: SCP status (verified, unverified, failed, expired,
            pending_inspection).
        count: Number of active SCPs with this status. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sgv_active_segregation_points.labels(status=status).set(count)


def set_avg_facility_score(score: float) -> None:
    """Set the gauge for average facility assessment score.

    Args:
        score: Average assessment score (0.0-100.0).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    sgv_avg_facility_score.set(score)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "sgv_scp_validations_total",
    "sgv_scp_failures_total",
    "sgv_storage_audits_total",
    "sgv_transport_checks_total",
    "sgv_processing_checks_total",
    "sgv_contamination_events_total",
    "sgv_contamination_critical_total",
    "sgv_labels_verified_total",
    "sgv_label_failures_total",
    "sgv_assessments_total",
    "sgv_reports_generated_total",
    "sgv_batch_jobs_total",
    "sgv_api_errors_total",
    "sgv_scp_validation_duration_seconds",
    "sgv_contamination_detection_duration_seconds",
    "sgv_assessment_duration_seconds",
    "sgv_active_segregation_points",
    "sgv_avg_facility_score",
    # Helper functions
    "record_scp_validation",
    "record_scp_failure",
    "record_storage_audit",
    "record_transport_check",
    "record_processing_check",
    "record_contamination_event",
    "record_contamination_critical",
    "record_label_verified",
    "record_label_failure",
    "record_assessment",
    "record_report_generated",
    "record_batch_job",
    "record_api_error",
    "observe_scp_validation_duration",
    "observe_contamination_detection_duration",
    "observe_assessment_duration",
    "set_active_segregation_points",
    "set_avg_facility_score",
]
