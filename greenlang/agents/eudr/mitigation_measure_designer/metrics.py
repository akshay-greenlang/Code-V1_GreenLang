# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-029: Mitigation Measure Designer

18 Prometheus metrics for mitigation measure designer service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_mmd_`` prefix for consistent
identification in Prometheus queries, Grafana dashboards, and alerting
rules across the GreenLang platform.

Metrics (18 per PRD Section 7.6):
    Counters (8):
        1.  gl_eudr_mmd_strategies_designed_total       - Strategies designed [commodity, risk_level]
        2.  gl_eudr_mmd_measures_proposed_total          - Measures proposed [category, dimension]
        3.  gl_eudr_mmd_measures_approved_total          - Measures approved [category]
        4.  gl_eudr_mmd_measures_completed_total         - Measures completed [category, result]
        5.  gl_eudr_mmd_verifications_total              - Verifications performed [result]
        6.  gl_eudr_mmd_reports_generated_total          - Reports generated [commodity]
        7.  gl_eudr_mmd_workflows_closed_total           - Workflows closed [commodity, outcome]
        8.  gl_eudr_mmd_api_errors_total                 - API errors [operation]

    Histograms (5):
        9.  gl_eudr_mmd_strategy_design_duration_seconds   - Strategy design latency [commodity]
        10. gl_eudr_mmd_effectiveness_estimation_duration_seconds - Effectiveness estimation latency
        11. gl_eudr_mmd_verification_duration_seconds      - Verification latency [commodity]
        12. gl_eudr_mmd_report_generation_duration_seconds - Report generation latency [commodity]
        13. gl_eudr_mmd_workflow_duration_seconds           - Full workflow duration [commodity]

    Gauges (5):
        14. gl_eudr_mmd_active_workflows                 - Active workflow count
        15. gl_eudr_mmd_overdue_measures                 - Overdue measure count
        16. gl_eudr_mmd_average_risk_reduction            - Average risk reduction gauge
        17. gl_eudr_mmd_pending_approvals                - Pending approval count
        18. gl_eudr_mmd_template_library_size             - Template library size

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-029 (GL-EUDR-MMD-029)
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
    _STRATEGIES_DESIGNED = Counter(
        "gl_eudr_mmd_strategies_designed_total",
        "Mitigation strategies designed",
        ["commodity", "risk_level"],
    )
    _MEASURES_PROPOSED = Counter(
        "gl_eudr_mmd_measures_proposed_total",
        "Mitigation measures proposed",
        ["category", "dimension"],
    )
    _MEASURES_APPROVED = Counter(
        "gl_eudr_mmd_measures_approved_total",
        "Mitigation measures approved",
        ["category"],
    )
    _MEASURES_COMPLETED = Counter(
        "gl_eudr_mmd_measures_completed_total",
        "Mitigation measures completed",
        ["category", "result"],
    )
    _VERIFICATIONS = Counter(
        "gl_eudr_mmd_verifications_total",
        "Post-implementation verifications performed",
        ["result"],
    )
    _REPORTS_GENERATED = Counter(
        "gl_eudr_mmd_reports_generated_total",
        "Mitigation reports generated",
        ["commodity"],
    )
    _WORKFLOWS_CLOSED = Counter(
        "gl_eudr_mmd_workflows_closed_total",
        "Mitigation workflows closed",
        ["commodity", "outcome"],
    )
    _API_ERRORS = Counter(
        "gl_eudr_mmd_api_errors_total",
        "API errors by operation type",
        ["operation"],
    )

    # Histograms (5)
    _STRATEGY_DESIGN_DURATION = Histogram(
        "gl_eudr_mmd_strategy_design_duration_seconds",
        "Strategy design latency",
        ["commodity"],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    _EFFECTIVENESS_ESTIMATION_DURATION = Histogram(
        "gl_eudr_mmd_effectiveness_estimation_duration_seconds",
        "Effectiveness estimation latency",
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    _VERIFICATION_DURATION = Histogram(
        "gl_eudr_mmd_verification_duration_seconds",
        "Post-implementation verification latency",
        ["commodity"],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    _REPORT_GENERATION_DURATION = Histogram(
        "gl_eudr_mmd_report_generation_duration_seconds",
        "Report generation latency",
        ["commodity"],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    _WORKFLOW_DURATION = Histogram(
        "gl_eudr_mmd_workflow_duration_seconds",
        "Full mitigation workflow duration",
        ["commodity"],
        buckets=(60.0, 300.0, 900.0, 3600.0, 86400.0, 604800.0, 2592000.0),
    )

    # Gauges (5)
    _ACTIVE_WORKFLOWS = Gauge(
        "gl_eudr_mmd_active_workflows",
        "Currently active mitigation workflows",
    )
    _OVERDUE_MEASURES = Gauge(
        "gl_eudr_mmd_overdue_measures",
        "Number of overdue mitigation measures",
    )
    _AVERAGE_RISK_REDUCTION = Gauge(
        "gl_eudr_mmd_average_risk_reduction",
        "Rolling average risk reduction across verified strategies",
    )
    _PENDING_APPROVALS = Gauge(
        "gl_eudr_mmd_pending_approvals",
        "Number of measures pending approval",
    )
    _TEMPLATE_LIBRARY_SIZE = Gauge(
        "gl_eudr_mmd_template_library_size",
        "Number of measure templates in the library",
    )


# ---------------------------------------------------------------------------
# Helper Functions - Counters
# ---------------------------------------------------------------------------


def record_strategy_designed(commodity: str, risk_level: str) -> None:
    """Record a mitigation strategy design metric."""
    if _PROMETHEUS_AVAILABLE:
        _STRATEGIES_DESIGNED.labels(
            commodity=commodity, risk_level=risk_level
        ).inc()


def record_measure_proposed(category: str, dimension: str) -> None:
    """Record a mitigation measure proposal metric."""
    if _PROMETHEUS_AVAILABLE:
        _MEASURES_PROPOSED.labels(
            category=category, dimension=dimension
        ).inc()


def record_measure_approved(category: str) -> None:
    """Record a mitigation measure approval metric."""
    if _PROMETHEUS_AVAILABLE:
        _MEASURES_APPROVED.labels(category=category).inc()


def record_measure_completed(category: str, result: str) -> None:
    """Record a mitigation measure completion metric."""
    if _PROMETHEUS_AVAILABLE:
        _MEASURES_COMPLETED.labels(category=category, result=result).inc()


def record_verification(result: str) -> None:
    """Record a post-implementation verification metric."""
    if _PROMETHEUS_AVAILABLE:
        _VERIFICATIONS.labels(result=result).inc()


def record_report_generated(commodity: str) -> None:
    """Record a mitigation report generation metric."""
    if _PROMETHEUS_AVAILABLE:
        _REPORTS_GENERATED.labels(commodity=commodity).inc()


def record_workflow_closed(commodity: str, outcome: str) -> None:
    """Record a mitigation workflow closure metric."""
    if _PROMETHEUS_AVAILABLE:
        _WORKFLOWS_CLOSED.labels(commodity=commodity, outcome=outcome).inc()


def record_api_error(operation: str) -> None:
    """Record an API error metric."""
    if _PROMETHEUS_AVAILABLE:
        _API_ERRORS.labels(operation=operation).inc()


# ---------------------------------------------------------------------------
# Helper Functions - Histograms
# ---------------------------------------------------------------------------


def observe_strategy_design_duration(commodity: str, duration: float) -> None:
    """Observe strategy design duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _STRATEGY_DESIGN_DURATION.labels(commodity=commodity).observe(duration)


def observe_effectiveness_estimation_duration(duration: float) -> None:
    """Observe effectiveness estimation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _EFFECTIVENESS_ESTIMATION_DURATION.observe(duration)


def observe_verification_duration(commodity: str, duration: float) -> None:
    """Observe post-implementation verification duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _VERIFICATION_DURATION.labels(commodity=commodity).observe(duration)


def observe_report_generation_duration(commodity: str, duration: float) -> None:
    """Observe report generation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _REPORT_GENERATION_DURATION.labels(commodity=commodity).observe(duration)


def observe_workflow_duration(commodity: str, duration: float) -> None:
    """Observe full mitigation workflow duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _WORKFLOW_DURATION.labels(commodity=commodity).observe(duration)


# ---------------------------------------------------------------------------
# Helper Functions - Gauges
# ---------------------------------------------------------------------------


def set_active_workflows(count: int) -> None:
    """Set gauge of currently active mitigation workflows."""
    if _PROMETHEUS_AVAILABLE:
        _ACTIVE_WORKFLOWS.set(count)


def set_overdue_measures(count: int) -> None:
    """Set gauge of overdue mitigation measures."""
    if _PROMETHEUS_AVAILABLE:
        _OVERDUE_MEASURES.set(count)


def set_average_risk_reduction(value: float) -> None:
    """Set gauge of average risk reduction across verified strategies."""
    if _PROMETHEUS_AVAILABLE:
        _AVERAGE_RISK_REDUCTION.set(value)


def set_pending_approvals(count: int) -> None:
    """Set gauge of measures pending approval."""
    if _PROMETHEUS_AVAILABLE:
        _PENDING_APPROVALS.set(count)


def set_template_library_size(count: int) -> None:
    """Set gauge of measure templates in the library."""
    if _PROMETHEUS_AVAILABLE:
        _TEMPLATE_LIBRARY_SIZE.set(count)
