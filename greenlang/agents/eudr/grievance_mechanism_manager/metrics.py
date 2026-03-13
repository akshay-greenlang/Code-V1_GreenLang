# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-032: Grievance Mechanism Manager

18 Prometheus metrics for grievance mechanism management service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_gmm_`` prefix for consistent
identification in Prometheus queries, Grafana dashboards, and alerting
rules across the GreenLang platform.

Metrics (18 per PRD Section 7.6):
    Counters (9):
        1.  gl_eudr_gmm_analytics_created_total          - Analytics records created [pattern_type]
        2.  gl_eudr_gmm_root_causes_analyzed_total        - Root cause analyses completed [method]
        3.  gl_eudr_gmm_mediations_initiated_total        - Mediations initiated [mediator_type]
        4.  gl_eudr_gmm_mediations_completed_total        - Mediations completed [settlement_status]
        5.  gl_eudr_gmm_remediations_created_total        - Remediations created [type]
        6.  gl_eudr_gmm_remediations_verified_total       - Remediations verified [status]
        7.  gl_eudr_gmm_risk_scores_computed_total        - Risk scores computed [scope]
        8.  gl_eudr_gmm_collective_created_total          - Collective grievances created
        9.  gl_eudr_gmm_regulatory_reports_generated_total - Regulatory reports generated [report_type]

    Histograms (6):
        10. gl_eudr_gmm_analytics_duration_seconds        - Analytics computation latency
        11. gl_eudr_gmm_root_cause_duration_seconds       - Root cause analysis latency
        12. gl_eudr_gmm_mediation_session_duration_seconds - Mediation session duration
        13. gl_eudr_gmm_remediation_verification_duration_seconds - Remediation verification latency
        14. gl_eudr_gmm_risk_scoring_duration_seconds     - Risk scoring computation latency
        15. gl_eudr_gmm_report_generation_duration_seconds - Report generation latency

    Gauges (3):
        16. gl_eudr_gmm_active_mediations                 - Active mediation count
        17. gl_eudr_gmm_open_remediations                 - Open remediation count
        18. gl_eudr_gmm_high_risk_entities                - High/critical risk entity count

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-032 (GL-EUDR-GMM-032)
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
    # Counters (9)
    _ANALYTICS_CREATED = Counter(
        "gl_eudr_gmm_analytics_created_total",
        "Grievance analytics records created",
        ["pattern_type"],
    )
    _ROOT_CAUSES_ANALYZED = Counter(
        "gl_eudr_gmm_root_causes_analyzed_total",
        "Root cause analyses completed",
        ["method"],
    )
    _MEDIATIONS_INITIATED = Counter(
        "gl_eudr_gmm_mediations_initiated_total",
        "Mediations initiated",
        ["mediator_type"],
    )
    _MEDIATIONS_COMPLETED = Counter(
        "gl_eudr_gmm_mediations_completed_total",
        "Mediations completed",
        ["settlement_status"],
    )
    _REMEDIATIONS_CREATED = Counter(
        "gl_eudr_gmm_remediations_created_total",
        "Remediations created",
        ["type"],
    )
    _REMEDIATIONS_VERIFIED = Counter(
        "gl_eudr_gmm_remediations_verified_total",
        "Remediations verified",
        ["status"],
    )
    _RISK_SCORES_COMPUTED = Counter(
        "gl_eudr_gmm_risk_scores_computed_total",
        "Risk scores computed",
        ["scope"],
    )
    _COLLECTIVE_CREATED = Counter(
        "gl_eudr_gmm_collective_created_total",
        "Collective grievances created",
    )
    _REGULATORY_REPORTS_GENERATED = Counter(
        "gl_eudr_gmm_regulatory_reports_generated_total",
        "Regulatory compliance reports generated",
        ["report_type"],
    )

    # Histograms (6)
    _ANALYTICS_DURATION = Histogram(
        "gl_eudr_gmm_analytics_duration_seconds",
        "Grievance analytics computation latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    _ROOT_CAUSE_DURATION = Histogram(
        "gl_eudr_gmm_root_cause_duration_seconds",
        "Root cause analysis computation latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _MEDIATION_SESSION_DURATION = Histogram(
        "gl_eudr_gmm_mediation_session_duration_seconds",
        "Mediation session duration (minutes converted to seconds)",
        buckets=(
            1800.0, 3600.0, 5400.0, 7200.0, 10800.0, 14400.0,
        ),
    )
    _REMEDIATION_VERIFICATION_DURATION = Histogram(
        "gl_eudr_gmm_remediation_verification_duration_seconds",
        "Remediation verification latency",
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    _RISK_SCORING_DURATION = Histogram(
        "gl_eudr_gmm_risk_scoring_duration_seconds",
        "Risk scoring computation latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _REPORT_GENERATION_DURATION = Histogram(
        "gl_eudr_gmm_report_generation_duration_seconds",
        "Regulatory report generation latency",
        ["report_type"],
        buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 60.0),
    )

    # Gauges (3)
    _ACTIVE_MEDIATIONS = Gauge(
        "gl_eudr_gmm_active_mediations",
        "Number of active (non-closed) mediations",
    )
    _OPEN_REMEDIATIONS = Gauge(
        "gl_eudr_gmm_open_remediations",
        "Number of open (in-progress) remediations",
    )
    _HIGH_RISK_ENTITIES = Gauge(
        "gl_eudr_gmm_high_risk_entities",
        "Number of entities scored as high or critical risk",
    )


# ---------------------------------------------------------------------------
# Helper Functions - Counters
# ---------------------------------------------------------------------------


def record_analytics_created(pattern_type: str) -> None:
    """Record an analytics record creation metric."""
    if _PROMETHEUS_AVAILABLE:
        _ANALYTICS_CREATED.labels(pattern_type=pattern_type).inc()


def record_root_cause_analyzed(method: str) -> None:
    """Record a root cause analysis completion metric."""
    if _PROMETHEUS_AVAILABLE:
        _ROOT_CAUSES_ANALYZED.labels(method=method).inc()


def record_mediation_initiated(mediator_type: str) -> None:
    """Record a mediation initiation metric."""
    if _PROMETHEUS_AVAILABLE:
        _MEDIATIONS_INITIATED.labels(mediator_type=mediator_type).inc()


def record_mediation_completed(settlement_status: str) -> None:
    """Record a mediation completion metric."""
    if _PROMETHEUS_AVAILABLE:
        _MEDIATIONS_COMPLETED.labels(settlement_status=settlement_status).inc()


def record_remediation_created(remediation_type: str) -> None:
    """Record a remediation creation metric."""
    if _PROMETHEUS_AVAILABLE:
        _REMEDIATIONS_CREATED.labels(type=remediation_type).inc()


def record_remediation_verified(status: str) -> None:
    """Record a remediation verification metric."""
    if _PROMETHEUS_AVAILABLE:
        _REMEDIATIONS_VERIFIED.labels(status=status).inc()


def record_risk_score_computed(scope: str) -> None:
    """Record a risk score computation metric."""
    if _PROMETHEUS_AVAILABLE:
        _RISK_SCORES_COMPUTED.labels(scope=scope).inc()


def record_collective_created() -> None:
    """Record a collective grievance creation metric."""
    if _PROMETHEUS_AVAILABLE:
        _COLLECTIVE_CREATED.inc()


def record_regulatory_report_generated(report_type: str) -> None:
    """Record a regulatory report generation metric."""
    if _PROMETHEUS_AVAILABLE:
        _REGULATORY_REPORTS_GENERATED.labels(report_type=report_type).inc()


# ---------------------------------------------------------------------------
# Helper Functions - Histograms
# ---------------------------------------------------------------------------


def observe_analytics_duration(duration: float) -> None:
    """Observe analytics computation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _ANALYTICS_DURATION.observe(duration)


def observe_root_cause_duration(duration: float) -> None:
    """Observe root cause analysis duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _ROOT_CAUSE_DURATION.observe(duration)


def observe_mediation_session_duration(duration: float) -> None:
    """Observe mediation session duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _MEDIATION_SESSION_DURATION.observe(duration)


def observe_remediation_verification_duration(duration: float) -> None:
    """Observe remediation verification duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _REMEDIATION_VERIFICATION_DURATION.observe(duration)


def observe_risk_scoring_duration(duration: float) -> None:
    """Observe risk scoring computation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _RISK_SCORING_DURATION.observe(duration)


def observe_report_generation_duration(report_type: str, duration: float) -> None:
    """Observe regulatory report generation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _REPORT_GENERATION_DURATION.labels(report_type=report_type).observe(duration)


# ---------------------------------------------------------------------------
# Helper Functions - Gauges
# ---------------------------------------------------------------------------


def set_active_mediations(count: int) -> None:
    """Set gauge of active mediations."""
    if _PROMETHEUS_AVAILABLE:
        _ACTIVE_MEDIATIONS.set(count)


def set_open_remediations(count: int) -> None:
    """Set gauge of open remediations."""
    if _PROMETHEUS_AVAILABLE:
        _OPEN_REMEDIATIONS.set(count)


def set_high_risk_entities(count: int) -> None:
    """Set gauge of high/critical risk entities."""
    if _PROMETHEUS_AVAILABLE:
        _HIGH_RISK_ENTITIES.set(count)
