# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-036: EU Information System Interface

18 Prometheus metrics for EU Information System Interface service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_euis_`` prefix for consistent
identification in Prometheus queries, Grafana dashboards, and alerting
rules across the GreenLang platform.

Metrics (18 per PRD Section 7.6):
    Counters (8):
        1.  gl_eudr_euis_dds_submitted_total          - DDS submissions [commodity, dds_type]
        2.  gl_eudr_euis_dds_accepted_total            - DDS accepted by EU IS [commodity]
        3.  gl_eudr_euis_dds_rejected_total            - DDS rejected by EU IS [commodity, reason]
        4.  gl_eudr_euis_operators_registered_total    - Operator registrations [member_state]
        5.  gl_eudr_euis_packages_assembled_total      - Document packages assembled [commodity]
        6.  gl_eudr_euis_status_checks_total           - Status checks performed [result]
        7.  gl_eudr_euis_api_calls_total               - EU IS API calls [method, endpoint]
        8.  gl_eudr_euis_api_errors_total              - API errors [operation, error_type]

    Histograms (5):
        9.  gl_eudr_euis_submission_duration_seconds    - DDS submission latency [commodity]
        10. gl_eudr_euis_geolocation_format_duration_seconds - Geolocation formatting latency
        11. gl_eudr_euis_package_assembly_duration_seconds - Package assembly latency [commodity]
        12. gl_eudr_euis_api_call_duration_seconds       - EU IS API call latency [endpoint]
        13. gl_eudr_euis_status_check_duration_seconds   - Status check latency

    Gauges (5):
        14. gl_eudr_euis_active_submissions             - Active submission count
        15. gl_eudr_euis_pending_dds                    - Pending DDS count
        16. gl_eudr_euis_registered_operators           - Registered operator count
        17. gl_eudr_euis_eu_api_health                  - EU IS API health (1=up, 0=down)
        18. gl_eudr_euis_audit_records_count            - Audit record count

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-036 (GL-EUDR-EUIS-036)
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
    # Counters (8)
    _DDS_SUBMITTED = Counter(
        "gl_eudr_euis_dds_submitted_total",
        "DDS submissions to EU Information System",
        ["commodity", "dds_type"],
    )
    _DDS_ACCEPTED = Counter(
        "gl_eudr_euis_dds_accepted_total",
        "DDS accepted by EU Information System",
        ["commodity"],
    )
    _DDS_REJECTED = Counter(
        "gl_eudr_euis_dds_rejected_total",
        "DDS rejected by EU Information System",
        ["commodity", "reason"],
    )
    _OPERATORS_REGISTERED = Counter(
        "gl_eudr_euis_operators_registered_total",
        "Operator registrations in EU Information System",
        ["member_state"],
    )
    _PACKAGES_ASSEMBLED = Counter(
        "gl_eudr_euis_packages_assembled_total",
        "Document packages assembled for submission",
        ["commodity"],
    )
    _STATUS_CHECKS = Counter(
        "gl_eudr_euis_status_checks_total",
        "DDS status checks performed",
        ["result"],
    )
    _API_CALLS = Counter(
        "gl_eudr_euis_api_calls_total",
        "EU IS API calls made",
        ["method", "endpoint"],
    )
    _API_ERRORS = Counter(
        "gl_eudr_euis_api_errors_total",
        "API errors by operation type",
        ["operation", "error_type"],
    )

    # Histograms (5)
    _SUBMISSION_DURATION = Histogram(
        "gl_eudr_euis_submission_duration_seconds",
        "DDS submission latency",
        ["commodity"],
        buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )
    _GEOLOCATION_FORMAT_DURATION = Histogram(
        "gl_eudr_euis_geolocation_format_duration_seconds",
        "Geolocation formatting latency",
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    )
    _PACKAGE_ASSEMBLY_DURATION = Histogram(
        "gl_eudr_euis_package_assembly_duration_seconds",
        "Document package assembly latency",
        ["commodity"],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    _API_CALL_DURATION = Histogram(
        "gl_eudr_euis_api_call_duration_seconds",
        "EU IS API call latency",
        ["endpoint"],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    _STATUS_CHECK_DURATION = Histogram(
        "gl_eudr_euis_status_check_duration_seconds",
        "Status check latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    # Gauges (5)
    _ACTIVE_SUBMISSIONS = Gauge(
        "gl_eudr_euis_active_submissions",
        "Currently active DDS submissions",
    )
    _PENDING_DDS = Gauge(
        "gl_eudr_euis_pending_dds",
        "DDS in pending/draft state",
    )
    _REGISTERED_OPERATORS = Gauge(
        "gl_eudr_euis_registered_operators",
        "Total registered operators",
    )
    _EU_API_HEALTH = Gauge(
        "gl_eudr_euis_eu_api_health",
        "EU Information System API health (1=up, 0=down)",
    )
    _AUDIT_RECORDS_COUNT = Gauge(
        "gl_eudr_euis_audit_records_count",
        "Total audit records stored",
    )


# ---------------------------------------------------------------------------
# Helper Functions - Counters
# ---------------------------------------------------------------------------


def record_dds_submitted(commodity: str, dds_type: str) -> None:
    """Record a DDS submission metric."""
    if _PROMETHEUS_AVAILABLE:
        _DDS_SUBMITTED.labels(
            commodity=commodity, dds_type=dds_type
        ).inc()


def record_dds_accepted(commodity: str) -> None:
    """Record a DDS acceptance metric."""
    if _PROMETHEUS_AVAILABLE:
        _DDS_ACCEPTED.labels(commodity=commodity).inc()


def record_dds_rejected(commodity: str, reason: str) -> None:
    """Record a DDS rejection metric."""
    if _PROMETHEUS_AVAILABLE:
        _DDS_REJECTED.labels(
            commodity=commodity, reason=reason
        ).inc()


def record_operator_registered(member_state: str) -> None:
    """Record an operator registration metric."""
    if _PROMETHEUS_AVAILABLE:
        _OPERATORS_REGISTERED.labels(member_state=member_state).inc()


def record_package_assembled(commodity: str) -> None:
    """Record a package assembly metric."""
    if _PROMETHEUS_AVAILABLE:
        _PACKAGES_ASSEMBLED.labels(commodity=commodity).inc()


def record_status_check(result: str) -> None:
    """Record a status check metric."""
    if _PROMETHEUS_AVAILABLE:
        _STATUS_CHECKS.labels(result=result).inc()


def record_api_call(method: str, endpoint: str) -> None:
    """Record an EU IS API call metric."""
    if _PROMETHEUS_AVAILABLE:
        _API_CALLS.labels(method=method, endpoint=endpoint).inc()


def record_api_error(operation: str, error_type: str = "unknown") -> None:
    """Record an API error metric."""
    if _PROMETHEUS_AVAILABLE:
        _API_ERRORS.labels(
            operation=operation, error_type=error_type
        ).inc()


# ---------------------------------------------------------------------------
# Helper Functions - Histograms
# ---------------------------------------------------------------------------


def observe_submission_duration(commodity: str, duration: float) -> None:
    """Observe DDS submission duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _SUBMISSION_DURATION.labels(commodity=commodity).observe(duration)


def observe_geolocation_format_duration(duration: float) -> None:
    """Observe geolocation formatting duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _GEOLOCATION_FORMAT_DURATION.observe(duration)


def observe_package_assembly_duration(commodity: str, duration: float) -> None:
    """Observe package assembly duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _PACKAGE_ASSEMBLY_DURATION.labels(commodity=commodity).observe(duration)


def observe_api_call_duration(endpoint: str, duration: float) -> None:
    """Observe EU IS API call duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _API_CALL_DURATION.labels(endpoint=endpoint).observe(duration)


def observe_status_check_duration(duration: float) -> None:
    """Observe status check duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _STATUS_CHECK_DURATION.observe(duration)


# ---------------------------------------------------------------------------
# Helper Functions - Gauges
# ---------------------------------------------------------------------------


def set_active_submissions(count: int) -> None:
    """Set gauge of currently active DDS submissions."""
    if _PROMETHEUS_AVAILABLE:
        _ACTIVE_SUBMISSIONS.set(count)


def set_pending_dds(count: int) -> None:
    """Set gauge of pending DDS."""
    if _PROMETHEUS_AVAILABLE:
        _PENDING_DDS.set(count)


def set_registered_operators(count: int) -> None:
    """Set gauge of registered operators."""
    if _PROMETHEUS_AVAILABLE:
        _REGISTERED_OPERATORS.set(count)


def set_eu_api_health(healthy: bool) -> None:
    """Set gauge of EU IS API health."""
    if _PROMETHEUS_AVAILABLE:
        _EU_API_HEALTH.set(1 if healthy else 0)


def set_audit_records_count(count: int) -> None:
    """Set gauge of total audit records."""
    if _PROMETHEUS_AVAILABLE:
        _AUDIT_RECORDS_COUNT.set(count)
