# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-031: Stakeholder Engagement Tool

18+ Prometheus metrics for stakeholder engagement service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_set_`` prefix for consistent
identification in Prometheus queries, Grafana dashboards, and alerting
rules across the GreenLang platform.

Metrics (18 per PRD Section 7.6):
    Counters (10):
        1.  gl_eudr_set_stakeholders_mapped_total              - Stakeholders mapped [category, commodity]
        2.  gl_eudr_set_fpic_initiated_total                    - FPIC workflows initiated [commodity]
        3.  gl_eudr_set_fpic_consents_total                     - FPIC consent decisions [commodity, status]
        4.  gl_eudr_set_grievances_submitted_total              - Grievances submitted [severity]
        5.  gl_eudr_set_grievances_resolved_total               - Grievances resolved [severity]
        6.  gl_eudr_set_consultations_conducted_total           - Consultation records created [type]
        7.  gl_eudr_set_communications_sent_total               - Communications sent [channel]
        8.  gl_eudr_set_assessments_completed_total             - Engagement assessments completed [stakeholder_id]
        9.  gl_eudr_set_reports_generated_total                 - Compliance reports generated [report_type, format]
        10. gl_eudr_set_api_errors_total                        - API errors [operation]

    Histograms (5):
        11. gl_eudr_set_stakeholder_mapping_duration_seconds    - Stakeholder mapping latency [commodity]
        12. gl_eudr_set_fpic_stage_duration_seconds             - FPIC stage duration [stage]
        13. gl_eudr_set_grievance_resolution_duration_seconds   - Grievance resolution latency [severity]
        14. gl_eudr_set_consultation_duration_seconds            - Consultation recording latency [type]
        15. gl_eudr_set_communication_delivery_duration_seconds  - Communication delivery latency [channel]

    Gauges (4):
        16. gl_eudr_set_active_stakeholders                     - Active stakeholder count
        17. gl_eudr_set_active_fpic_workflows                   - Active FPIC workflows count
        18. gl_eudr_set_open_grievances                         - Open grievances count
        19. gl_eudr_set_pending_communications                  - Pending communications count

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-031 (GL-EUDR-SET-031)
Status: Production Ready
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.debug("prometheus_client not available; metrics disabled")


# ---------------------------------------------------------------------------
# Metric Definitions
# ---------------------------------------------------------------------------

if _PROMETHEUS_AVAILABLE:
    # Counters (10)
    _STAKEHOLDERS_MAPPED = Counter(
        "gl_eudr_set_stakeholders_mapped_total",
        "Stakeholders mapped in the engagement registry",
        ["category", "commodity"],
    )
    _FPIC_INITIATED = Counter(
        "gl_eudr_set_fpic_initiated_total",
        "FPIC workflows initiated",
        ["commodity"],
    )
    _FPIC_CONSENTS = Counter(
        "gl_eudr_set_fpic_consents_total",
        "FPIC consent decisions recorded",
        ["commodity", "status"],
    )
    _GRIEVANCES_SUBMITTED = Counter(
        "gl_eudr_set_grievances_submitted_total",
        "Grievances submitted to the mechanism",
        ["severity"],
    )
    _GRIEVANCES_RESOLVED = Counter(
        "gl_eudr_set_grievances_resolved_total",
        "Grievances resolved",
        ["severity"],
    )
    _CONSULTATIONS_CONDUCTED = Counter(
        "gl_eudr_set_consultations_conducted_total",
        "Consultation records created",
        ["type"],
    )
    _COMMUNICATIONS_SENT = Counter(
        "gl_eudr_set_communications_sent_total",
        "Communications sent to stakeholders",
        ["channel"],
    )
    _ASSESSMENTS_COMPLETED = Counter(
        "gl_eudr_set_assessments_completed_total",
        "Engagement quality assessments completed",
        ["stakeholder_id"],
    )
    _REPORTS_GENERATED = Counter(
        "gl_eudr_set_reports_generated_total",
        "Compliance reports generated",
        ["report_type", "format"],
    )
    _API_ERRORS = Counter(
        "gl_eudr_set_api_errors_total",
        "API errors by operation",
        ["operation"],
    )

    # Histograms (5)
    _STAKEHOLDER_MAPPING_DURATION = Histogram(
        "gl_eudr_set_stakeholder_mapping_duration_seconds",
        "Stakeholder mapping operation latency",
        ["commodity"],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _FPIC_STAGE_DURATION = Histogram(
        "gl_eudr_set_fpic_stage_duration_seconds",
        "FPIC workflow stage duration from start to completion",
        ["stage"],
        buckets=(
            3600.0, 86400.0, 604800.0, 2592000.0, 7776000.0,
            15552000.0, 31536000.0,
        ),
    )
    _GRIEVANCE_RESOLUTION_DURATION = Histogram(
        "gl_eudr_set_grievance_resolution_duration_seconds",
        "Grievance resolution latency from submission to resolution",
        ["severity"],
        buckets=(
            3600.0, 86400.0, 604800.0, 1209600.0, 2592000.0,
            5184000.0,
        ),
    )
    _CONSULTATION_DURATION = Histogram(
        "gl_eudr_set_consultation_duration_seconds",
        "Consultation recording operation latency",
        ["type"],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 60.0, 120.0, 300.0),
    )
    _COMMUNICATION_DELIVERY_DURATION = Histogram(
        "gl_eudr_set_communication_delivery_duration_seconds",
        "Communication delivery latency",
        ["channel"],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    # Gauges (4)
    _ACTIVE_STAKEHOLDERS = Gauge(
        "gl_eudr_set_active_stakeholders",
        "Number of active stakeholders in the registry",
    )
    _ACTIVE_FPIC_WORKFLOWS = Gauge(
        "gl_eudr_set_active_fpic_workflows",
        "Number of active FPIC workflows in progress",
    )
    _OPEN_GRIEVANCES = Gauge(
        "gl_eudr_set_open_grievances",
        "Number of open (unresolved) grievances",
    )
    _PENDING_COMMUNICATIONS = Gauge(
        "gl_eudr_set_pending_communications",
        "Number of communications pending delivery",
    )


# ---------------------------------------------------------------------------
# Helper Functions - Counters
# ---------------------------------------------------------------------------


def record_stakeholder_mapped(category: str, commodity: str) -> None:
    """Record a stakeholder mapping metric."""
    if _PROMETHEUS_AVAILABLE:
        _STAKEHOLDERS_MAPPED.labels(category=category, commodity=commodity).inc()


def record_fpic_initiated(commodity: str) -> None:
    """Record an FPIC workflow initiation metric."""
    if _PROMETHEUS_AVAILABLE:
        _FPIC_INITIATED.labels(commodity=commodity).inc()


def record_fpic_consent(commodity: str, status: str) -> None:
    """Record an FPIC consent decision metric."""
    if _PROMETHEUS_AVAILABLE:
        _FPIC_CONSENTS.labels(commodity=commodity, status=status).inc()


def record_grievance_submitted(severity: str) -> None:
    """Record a grievance submission metric."""
    if _PROMETHEUS_AVAILABLE:
        _GRIEVANCES_SUBMITTED.labels(severity=severity).inc()


def record_grievance_resolved(severity: str) -> None:
    """Record a grievance resolution metric."""
    if _PROMETHEUS_AVAILABLE:
        _GRIEVANCES_RESOLVED.labels(severity=severity).inc()


def record_consultation_conducted(consultation_type: str) -> None:
    """Record a consultation creation metric."""
    if _PROMETHEUS_AVAILABLE:
        _CONSULTATIONS_CONDUCTED.labels(type=consultation_type).inc()


def record_communication_sent(channel: str) -> None:
    """Record a communication sent metric."""
    if _PROMETHEUS_AVAILABLE:
        _COMMUNICATIONS_SENT.labels(channel=channel).inc()


def record_assessment_completed(stakeholder_id: str) -> None:
    """Record an engagement assessment completion metric."""
    if _PROMETHEUS_AVAILABLE:
        _ASSESSMENTS_COMPLETED.labels(stakeholder_id=stakeholder_id).inc()


def record_report_generated(report_type: str, report_format: str) -> None:
    """Record a compliance report generation metric."""
    if _PROMETHEUS_AVAILABLE:
        _REPORTS_GENERATED.labels(
            report_type=report_type, format=report_format
        ).inc()


def record_api_error(operation: str) -> None:
    """Record an API error metric."""
    if _PROMETHEUS_AVAILABLE:
        _API_ERRORS.labels(operation=operation).inc()


# ---------------------------------------------------------------------------
# Helper Functions - Histograms
# ---------------------------------------------------------------------------


def observe_stakeholder_mapping_duration(
    commodity: str, duration: float,
) -> None:
    """Observe stakeholder mapping operation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _STAKEHOLDER_MAPPING_DURATION.labels(commodity=commodity).observe(
            duration
        )


def observe_fpic_stage_duration(stage: str, duration: float) -> None:
    """Observe FPIC workflow stage duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _FPIC_STAGE_DURATION.labels(stage=stage).observe(duration)


def observe_grievance_resolution_duration(
    severity: str, duration: float,
) -> None:
    """Observe grievance resolution duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _GRIEVANCE_RESOLUTION_DURATION.labels(severity=severity).observe(
            duration
        )


def observe_consultation_duration(
    consultation_type: str, duration: float,
) -> None:
    """Observe consultation recording operation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _CONSULTATION_DURATION.labels(type=consultation_type).observe(
            duration
        )


def observe_communication_delivery_duration(
    channel: str, duration: float,
) -> None:
    """Observe communication delivery duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _COMMUNICATION_DELIVERY_DURATION.labels(channel=channel).observe(
            duration
        )


# ---------------------------------------------------------------------------
# Helper Functions - Gauges
# ---------------------------------------------------------------------------


def set_active_stakeholders(count: int) -> None:
    """Set gauge of active stakeholders."""
    if _PROMETHEUS_AVAILABLE:
        _ACTIVE_STAKEHOLDERS.set(count)


def set_active_fpic_workflows(count: int) -> None:
    """Set gauge of active FPIC workflows."""
    if _PROMETHEUS_AVAILABLE:
        _ACTIVE_FPIC_WORKFLOWS.set(count)


def set_open_grievances(count: int) -> None:
    """Set gauge of open grievances."""
    if _PROMETHEUS_AVAILABLE:
        _OPEN_GRIEVANCES.set(count)


def set_pending_communications(count: int) -> None:
    """Set gauge of pending communications."""
    if _PROMETHEUS_AVAILABLE:
        _PENDING_COMMUNICATIONS.set(count)
