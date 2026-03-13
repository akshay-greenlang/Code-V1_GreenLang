# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-024: Third-Party Audit Manager

20 Prometheus metrics for third-party audit management agent service
monitoring with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_tam_`` prefix (GreenLang EUDR
Third-party Audit Manager) for consistent identification in Prometheus
queries, Grafana dashboards, and alerting rules.

Metrics (20 per PRD Section 7.3):
    Counters (10):
        1.  gl_eudr_tam_audits_scheduled_total          - Audits scheduled
        2.  gl_eudr_tam_audits_completed_total          - Audits completed
        3.  gl_eudr_tam_ncs_detected_total              - NCs detected
        4.  gl_eudr_tam_cars_issued_total               - CARs issued
        5.  gl_eudr_tam_cars_closed_total               - CARs closed
        6.  gl_eudr_tam_reports_generated_total         - Reports generated
        7.  gl_eudr_tam_authority_interactions_total     - Authority interactions
        8.  gl_eudr_tam_auditor_matches_total           - Auditor matches
        9.  gl_eudr_tam_cert_syncs_total                - Certificate syncs
        10. gl_eudr_tam_api_errors_total                - API errors

    Histograms (4):
        11. gl_eudr_tam_scheduling_duration_seconds     - Scheduling duration
        12. gl_eudr_tam_nc_classification_seconds       - NC classification time
        13. gl_eudr_tam_report_generation_seconds       - Report generation time
        14. gl_eudr_tam_analytics_calculation_seconds   - Analytics calc time

    Gauges (6):
        15. gl_eudr_tam_active_audits                   - Active audits
        16. gl_eudr_tam_open_cars                       - Open CARs
        17. gl_eudr_tam_overdue_cars                    - Overdue CARs
        18. gl_eudr_tam_pending_authority_responses     - Pending responses
        19. gl_eudr_tam_registered_auditors             - Registered auditors
        20. gl_eudr_tam_active_certificates             - Active certificates

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024 Third-Party Audit Manager (GL-EUDR-TAM-024)
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
        "third-party audit manager metrics disabled"
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
    _audits_scheduled_total = _safe_counter(
        "gl_eudr_tam_audits_scheduled_total",
        "Total number of audits scheduled via risk-based planning",
        ["scope", "modality", "country_code"],
    )
    _audits_completed_total = _safe_counter(
        "gl_eudr_tam_audits_completed_total",
        "Total number of audits completed",
        ["scope", "country_code", "commodity"],
    )
    _ncs_detected_total = _safe_counter(
        "gl_eudr_tam_ncs_detected_total",
        "Total number of non-conformances detected",
        ["severity", "country_code"],
    )
    _cars_issued_total = _safe_counter(
        "gl_eudr_tam_cars_issued_total",
        "Total number of corrective action requests issued",
        ["severity", "authority_issued"],
    )
    _cars_closed_total = _safe_counter(
        "gl_eudr_tam_cars_closed_total",
        "Total number of CARs closed (verified effective)",
        ["severity", "within_sla"],
    )
    _reports_generated_total = _safe_counter(
        "gl_eudr_tam_reports_generated_total",
        "Total number of ISO 19011 audit reports generated",
        ["format", "language"],
    )
    _authority_interactions_total = _safe_counter(
        "gl_eudr_tam_authority_interactions_total",
        "Total number of competent authority interactions logged",
        ["interaction_type", "member_state"],
    )
    _auditor_matches_total = _safe_counter(
        "gl_eudr_tam_auditor_matches_total",
        "Total number of auditor matching operations",
        ["commodity", "scheme"],
    )
    _cert_syncs_total = _safe_counter(
        "gl_eudr_tam_cert_syncs_total",
        "Total number of certification scheme sync operations",
        ["scheme", "status"],
    )
    _api_errors_total = _safe_counter(
        "gl_eudr_tam_api_errors_total",
        "Total number of API errors by operation",
        ["operation"],
    )

    # Histograms (4)
    _scheduling_duration_seconds = _safe_histogram(
        "gl_eudr_tam_scheduling_duration_seconds",
        "Audit scheduling computation duration in seconds",
        ["scope"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    )
    _nc_classification_seconds = _safe_histogram(
        "gl_eudr_tam_nc_classification_seconds",
        "Non-conformance classification duration in seconds",
        ["severity"],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    )
    _report_generation_seconds = _safe_histogram(
        "gl_eudr_tam_report_generation_seconds",
        "Audit report generation duration in seconds",
        ["format"],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0),
    )
    _analytics_calculation_seconds = _safe_histogram(
        "gl_eudr_tam_analytics_calculation_seconds",
        "Analytics calculation duration in seconds",
        ["metric_type"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0),
    )

    # Gauges (6)
    _active_audits = _safe_gauge(
        "gl_eudr_tam_active_audits",
        "Number of currently active (in-progress) audits",
        ["status"],
    )
    _open_cars = _safe_gauge(
        "gl_eudr_tam_open_cars",
        "Number of open (unresolved) corrective action requests",
        ["severity"],
    )
    _overdue_cars = _safe_gauge(
        "gl_eudr_tam_overdue_cars",
        "Number of overdue CARs exceeding SLA deadline",
        ["severity"],
    )
    _pending_authority_responses = _safe_gauge(
        "gl_eudr_tam_pending_authority_responses",
        "Number of pending competent authority responses",
        ["member_state"],
    )
    _registered_auditors = _safe_gauge(
        "gl_eudr_tam_registered_auditors",
        "Number of registered auditors in the registry",
        ["status"],
    )
    _active_certificates = _safe_gauge(
        "gl_eudr_tam_active_certificates",
        "Number of active certification scheme certificates",
        ["scheme"],
    )


# ---------------------------------------------------------------------------
# Helper functions (20 functions matching 20 metrics)
# ---------------------------------------------------------------------------


def record_audit_scheduled(
    scope: str = "full",
    modality: str = "on_site",
    country_code: str = "unknown",
) -> None:
    """Record an audit scheduling event.

    Args:
        scope: Audit scope (full, targeted, surveillance, unscheduled).
        modality: Audit modality (on_site, remote, hybrid, unannounced).
        country_code: ISO 3166-1 alpha-2 country code.
    """
    if PROMETHEUS_AVAILABLE:
        _audits_scheduled_total.labels(
            scope=scope, modality=modality, country_code=country_code,
        ).inc()


def record_audit_completed(
    scope: str = "full",
    country_code: str = "unknown",
    commodity: str = "unknown",
) -> None:
    """Record an audit completion event.

    Args:
        scope: Audit scope.
        country_code: ISO 3166-1 alpha-2 country code.
        commodity: EUDR commodity audited.
    """
    if PROMETHEUS_AVAILABLE:
        _audits_completed_total.labels(
            scope=scope, country_code=country_code, commodity=commodity,
        ).inc()


def record_nc_detected(
    severity: str = "minor",
    country_code: str = "unknown",
) -> None:
    """Record a non-conformance detection.

    Args:
        severity: NC severity (critical, major, minor, observation).
        country_code: ISO 3166-1 alpha-2 country code.
    """
    if PROMETHEUS_AVAILABLE:
        _ncs_detected_total.labels(
            severity=severity, country_code=country_code,
        ).inc()


def record_car_issued(
    severity: str = "minor",
    authority_issued: str = "false",
) -> None:
    """Record a CAR issuance event.

    Args:
        severity: CAR severity.
        authority_issued: Whether CAR was authority-issued ("true"/"false").
    """
    if PROMETHEUS_AVAILABLE:
        _cars_issued_total.labels(
            severity=severity, authority_issued=authority_issued,
        ).inc()


def record_car_closed(
    severity: str = "minor",
    within_sla: str = "true",
) -> None:
    """Record a CAR closure event.

    Args:
        severity: CAR severity.
        within_sla: Whether closed within SLA ("true"/"false").
    """
    if PROMETHEUS_AVAILABLE:
        _cars_closed_total.labels(
            severity=severity, within_sla=within_sla,
        ).inc()


def record_report_generated(
    format: str = "pdf",
    language: str = "en",
) -> None:
    """Record a report generation event.

    Args:
        format: Report output format.
        language: Report language.
    """
    if PROMETHEUS_AVAILABLE:
        _reports_generated_total.labels(
            format=format, language=language,
        ).inc()


def record_authority_interaction(
    interaction_type: str = "document_request",
    member_state: str = "unknown",
) -> None:
    """Record a competent authority interaction.

    Args:
        interaction_type: Type of authority interaction.
        member_state: EU Member State code.
    """
    if PROMETHEUS_AVAILABLE:
        _authority_interactions_total.labels(
            interaction_type=interaction_type, member_state=member_state,
        ).inc()


def record_auditor_match(
    commodity: str = "unknown",
    scheme: str = "none",
) -> None:
    """Record an auditor matching operation.

    Args:
        commodity: EUDR commodity.
        scheme: Certification scheme.
    """
    if PROMETHEUS_AVAILABLE:
        _auditor_matches_total.labels(
            commodity=commodity, scheme=scheme,
        ).inc()


def record_cert_sync(
    scheme: str = "unknown",
    status: str = "success",
) -> None:
    """Record a certification scheme sync operation.

    Args:
        scheme: Certification scheme synced.
        status: Sync status (success, failure).
    """
    if PROMETHEUS_AVAILABLE:
        _cert_syncs_total.labels(
            scheme=scheme, status=status,
        ).inc()


def record_api_error(operation: str) -> None:
    """Record an API error.

    Args:
        operation: Operation name.
    """
    if PROMETHEUS_AVAILABLE:
        _api_errors_total.labels(operation=operation).inc()


def observe_scheduling_duration(
    duration_seconds: float,
    scope: str = "full",
) -> None:
    """Observe audit scheduling computation duration.

    Args:
        duration_seconds: Duration in seconds.
        scope: Audit scope.
    """
    if PROMETHEUS_AVAILABLE:
        _scheduling_duration_seconds.labels(scope=scope).observe(duration_seconds)


def observe_nc_classification_duration(
    duration_seconds: float,
    severity: str = "minor",
) -> None:
    """Observe NC classification duration.

    Args:
        duration_seconds: Duration in seconds.
        severity: Classified severity.
    """
    if PROMETHEUS_AVAILABLE:
        _nc_classification_seconds.labels(severity=severity).observe(duration_seconds)


def observe_report_generation_duration(
    duration_seconds: float,
    format: str = "pdf",
) -> None:
    """Observe report generation duration.

    Args:
        duration_seconds: Duration in seconds.
        format: Report format.
    """
    if PROMETHEUS_AVAILABLE:
        _report_generation_seconds.labels(format=format).observe(duration_seconds)


def observe_analytics_calculation_duration(
    duration_seconds: float,
    metric_type: str = "finding_trends",
) -> None:
    """Observe analytics calculation duration.

    Args:
        duration_seconds: Duration in seconds.
        metric_type: Type of analytics metric calculated.
    """
    if PROMETHEUS_AVAILABLE:
        _analytics_calculation_seconds.labels(
            metric_type=metric_type,
        ).observe(duration_seconds)


def set_active_audits(count: int, status: str = "all") -> None:
    """Set the number of active audits.

    Args:
        count: Number of active audits.
        status: Status filter (or "all").
    """
    if PROMETHEUS_AVAILABLE:
        _active_audits.labels(status=status).set(count)


def set_open_cars(count: int, severity: str = "all") -> None:
    """Set the number of open CARs.

    Args:
        count: Number of open CARs.
        severity: Severity filter (or "all").
    """
    if PROMETHEUS_AVAILABLE:
        _open_cars.labels(severity=severity).set(count)


def set_overdue_cars(count: int, severity: str = "all") -> None:
    """Set the number of overdue CARs.

    Args:
        count: Number of overdue CARs.
        severity: Severity filter (or "all").
    """
    if PROMETHEUS_AVAILABLE:
        _overdue_cars.labels(severity=severity).set(count)


def set_pending_authority_responses(
    count: int,
    member_state: str = "all",
) -> None:
    """Set the number of pending authority responses.

    Args:
        count: Number of pending responses.
        member_state: Member State filter (or "all").
    """
    if PROMETHEUS_AVAILABLE:
        _pending_authority_responses.labels(
            member_state=member_state,
        ).set(count)


def set_registered_auditors(count: int, status: str = "all") -> None:
    """Set the number of registered auditors.

    Args:
        count: Number of registered auditors.
        status: Status filter (or "all").
    """
    if PROMETHEUS_AVAILABLE:
        _registered_auditors.labels(status=status).set(count)


def set_active_certificates(count: int, scheme: str = "all") -> None:
    """Set the number of active certificates.

    Args:
        count: Number of active certificates.
        scheme: Scheme filter (or "all").
    """
    if PROMETHEUS_AVAILABLE:
        _active_certificates.labels(scheme=scheme).set(count)
