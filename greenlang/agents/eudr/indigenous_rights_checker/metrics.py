# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-021: Indigenous Rights Checker

15 Prometheus metrics for indigenous rights checker agent service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_irc_`` prefix (GreenLang EUDR Indigenous
Rights Checker) for consistent identification in Prometheus queries,
Grafana dashboards, and alerting rules.

Metrics (15 per PRD Section 7.6):
    Counters (10):
        1.  gl_eudr_irc_territory_queries_total          - Territory database queries
        2.  gl_eudr_irc_fpic_assessments_total           - FPIC assessments performed
        3.  gl_eudr_irc_overlaps_detected_total          - Territory overlaps detected
        4.  gl_eudr_irc_consultations_recorded_total     - Consultation records created
        5.  gl_eudr_irc_violations_ingested_total        - Violation reports ingested
        6.  gl_eudr_irc_violations_correlated_total      - Violations correlated
        7.  gl_eudr_irc_workflows_created_total          - FPIC workflows initiated
        8.  gl_eudr_irc_workflow_transitions_total       - Workflow stage transitions
        9.  gl_eudr_irc_reports_generated_total          - Compliance reports generated
        10. gl_eudr_irc_api_errors_total                 - API errors by endpoint

    Histograms (2):
        11. gl_eudr_irc_overlap_query_duration_seconds   - Overlap detection latency
        12. gl_eudr_irc_fpic_assessment_duration_seconds  - FPIC assessment latency

    Gauges (3):
        13. gl_eudr_irc_active_territories               - Total territories in database
        14. gl_eudr_irc_active_overlaps                   - Currently active overlaps
        15. gl_eudr_irc_active_workflows                  - Currently active workflows

Example:
    >>> from greenlang.agents.eudr.indigenous_rights_checker.metrics import (
    ...     record_territory_query,
    ...     record_fpic_assessment,
    ...     record_overlap_detected,
    ...     observe_overlap_query_duration,
    ...     set_active_territories,
    ... )
    >>> record_territory_query("BR", "funai")
    >>> record_fpic_assessment("consent_obtained")
    >>> record_overlap_detected("direct")
    >>> observe_overlap_query_duration(0.045)
    >>> set_active_territories(50234)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 Indigenous Rights Checker (GL-EUDR-IRC-021)
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
        "indigenous rights checker metrics disabled"
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

    def _safe_histogram(
        name: str, doc: str, labels: list, buckets: tuple
    ) -> Histogram:
        """Safely create or retrieve a Histogram metric."""
        try:
            return Histogram(
                name, doc, labels, buckets=buckets, registry=_REGISTRY
            )
        except ValueError:
            return _REGISTRY._names_to_collectors[name]

    def _safe_gauge(name: str, doc: str, labels: list) -> Gauge:
        """Safely create or retrieve a Gauge metric."""
        try:
            return Gauge(name, doc, labels, registry=_REGISTRY)
        except ValueError:
            return _REGISTRY._names_to_collectors[name]

# ---------------------------------------------------------------------------
# Metric definitions (15 metrics)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # Counters (10)
    _territory_queries_total = _safe_counter(
        "gl_eudr_irc_territory_queries_total",
        "Total number of territory database queries performed",
        ["country_code", "data_source"],
    )

    _fpic_assessments_total = _safe_counter(
        "gl_eudr_irc_fpic_assessments_total",
        "Total number of FPIC assessments performed",
        ["fpic_status"],
    )

    _overlaps_detected_total = _safe_counter(
        "gl_eudr_irc_overlaps_detected_total",
        "Total number of territory overlaps detected",
        ["overlap_type"],
    )

    _consultations_recorded_total = _safe_counter(
        "gl_eudr_irc_consultations_recorded_total",
        "Total number of consultation records created",
        ["consultation_stage"],
    )

    _violations_ingested_total = _safe_counter(
        "gl_eudr_irc_violations_ingested_total",
        "Total number of violation reports ingested",
        ["source"],
    )

    _violations_correlated_total = _safe_counter(
        "gl_eudr_irc_violations_correlated_total",
        "Total number of violations correlated with supply chain",
        ["severity_level"],
    )

    _workflows_created_total = _safe_counter(
        "gl_eudr_irc_workflows_created_total",
        "Total number of FPIC workflows initiated",
        [],
    )

    _workflow_transitions_total = _safe_counter(
        "gl_eudr_irc_workflow_transitions_total",
        "Total number of workflow stage transitions",
        ["from_stage", "to_stage"],
    )

    _reports_generated_total = _safe_counter(
        "gl_eudr_irc_reports_generated_total",
        "Total number of compliance reports generated",
        ["report_type"],
    )

    _api_errors_total = _safe_counter(
        "gl_eudr_irc_api_errors_total",
        "Total number of API errors by endpoint",
        ["endpoint", "status_code"],
    )

    # Histograms (2)
    _overlap_query_duration_seconds = _safe_histogram(
        "gl_eudr_irc_overlap_query_duration_seconds",
        "Overlap detection query duration in seconds",
        [],
        buckets=(
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
        ),
    )

    _fpic_assessment_duration_seconds = _safe_histogram(
        "gl_eudr_irc_fpic_assessment_duration_seconds",
        "FPIC assessment processing duration in seconds",
        [],
        buckets=(
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
        ),
    )

    # Gauges (3)
    _active_territories = _safe_gauge(
        "gl_eudr_irc_active_territories",
        "Total number of territories in database",
        [],
    )

    _active_overlaps = _safe_gauge(
        "gl_eudr_irc_active_overlaps",
        "Number of currently active territory overlaps",
        [],
    )

    _active_workflows = _safe_gauge(
        "gl_eudr_irc_active_workflows",
        "Number of currently active FPIC workflows",
        [],
    )


# ---------------------------------------------------------------------------
# Helper functions (15 functions matching 15 metrics)
# ---------------------------------------------------------------------------


def record_territory_query(
    country_code: str = "unknown",
    data_source: str = "unknown",
) -> None:
    """Record a territory database query.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        data_source: Territory data source identifier.
    """
    if PROMETHEUS_AVAILABLE:
        _territory_queries_total.labels(
            country_code=country_code,
            data_source=data_source,
        ).inc()


def record_fpic_assessment(fpic_status: str = "unknown") -> None:
    """Record an FPIC assessment completion.

    Args:
        fpic_status: Resulting FPIC status classification.
    """
    if PROMETHEUS_AVAILABLE:
        _fpic_assessments_total.labels(fpic_status=fpic_status).inc()


def record_overlap_detected(overlap_type: str = "unknown") -> None:
    """Record a territory overlap detection.

    Args:
        overlap_type: Overlap classification (direct, partial, etc.).
    """
    if PROMETHEUS_AVAILABLE:
        _overlaps_detected_total.labels(overlap_type=overlap_type).inc()


def record_consultation_recorded(
    consultation_stage: str = "unknown",
) -> None:
    """Record a consultation activity creation.

    Args:
        consultation_stage: Consultation lifecycle stage.
    """
    if PROMETHEUS_AVAILABLE:
        _consultations_recorded_total.labels(
            consultation_stage=consultation_stage,
        ).inc()


def record_violation_ingested(source: str = "unknown") -> None:
    """Record a violation report ingestion.

    Args:
        source: Violation report source identifier.
    """
    if PROMETHEUS_AVAILABLE:
        _violations_ingested_total.labels(source=source).inc()


def record_violation_correlated(severity_level: str = "unknown") -> None:
    """Record a violation-supply chain correlation.

    Args:
        severity_level: Alert severity level.
    """
    if PROMETHEUS_AVAILABLE:
        _violations_correlated_total.labels(
            severity_level=severity_level,
        ).inc()


def record_workflow_created() -> None:
    """Record a new FPIC workflow creation."""
    if PROMETHEUS_AVAILABLE:
        _workflows_created_total.inc()


def record_workflow_transition(
    from_stage: str = "unknown",
    to_stage: str = "unknown",
) -> None:
    """Record a FPIC workflow stage transition.

    Args:
        from_stage: Source workflow stage.
        to_stage: Target workflow stage.
    """
    if PROMETHEUS_AVAILABLE:
        _workflow_transitions_total.labels(
            from_stage=from_stage,
            to_stage=to_stage,
        ).inc()


def record_report_generated(report_type: str = "unknown") -> None:
    """Record a compliance report generation.

    Args:
        report_type: Type of report generated.
    """
    if PROMETHEUS_AVAILABLE:
        _reports_generated_total.labels(report_type=report_type).inc()


def record_api_error(
    endpoint: str = "unknown",
    status_code: str = "500",
) -> None:
    """Record an API error.

    Args:
        endpoint: API endpoint path.
        status_code: HTTP status code string.
    """
    if PROMETHEUS_AVAILABLE:
        _api_errors_total.labels(
            endpoint=endpoint,
            status_code=status_code,
        ).inc()


def observe_overlap_query_duration(duration_seconds: float) -> None:
    """Observe overlap detection query duration.

    Args:
        duration_seconds: Duration in seconds.
    """
    if PROMETHEUS_AVAILABLE:
        _overlap_query_duration_seconds.observe(duration_seconds)


def observe_fpic_assessment_duration(duration_seconds: float) -> None:
    """Observe FPIC assessment processing duration.

    Args:
        duration_seconds: Duration in seconds.
    """
    if PROMETHEUS_AVAILABLE:
        _fpic_assessment_duration_seconds.observe(duration_seconds)


def set_active_territories(count: int) -> None:
    """Set the total number of territories in database.

    Args:
        count: Number of territories.
    """
    if PROMETHEUS_AVAILABLE:
        _active_territories.set(count)


def set_active_overlaps(count: int) -> None:
    """Set the number of currently active territory overlaps.

    Args:
        count: Number of active overlaps.
    """
    if PROMETHEUS_AVAILABLE:
        _active_overlaps.set(count)


def set_active_workflows(count: int) -> None:
    """Set the number of currently active FPIC workflows.

    Args:
        count: Number of active workflows.
    """
    if PROMETHEUS_AVAILABLE:
        _active_workflows.set(count)
