# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-020: Deforestation Alert System

20 Prometheus metrics for deforestation alert system agent service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_das_`` prefix (GreenLang EUDR Deforestation
Alert System) for consistent identification in Prometheus queries, Grafana
dashboards, and alerting rules across the GreenLang platform.

Metrics (20 per PRD Section 7.3):
    Counters (10):
        1.  gl_eudr_das_satellite_detections_total       - Satellite change detections processed
        2.  gl_eudr_das_alerts_generated_total            - Deforestation alerts generated
        3.  gl_eudr_das_severity_classifications_total    - Severity classifications completed
        4.  gl_eudr_das_buffer_checks_total               - Buffer zone checks performed
        5.  gl_eudr_das_cutoff_verifications_total        - Cutoff date verifications completed
        6.  gl_eudr_das_baseline_comparisons_total        - Baseline comparisons completed
        7.  gl_eudr_das_workflow_transitions_total        - Workflow state transitions
        8.  gl_eudr_das_compliance_assessments_total      - Compliance impact assessments
        9.  gl_eudr_das_false_positives_total             - False positive detections
        10. gl_eudr_das_api_errors_total                  - API errors by operation

    Histograms (4):
        11. gl_eudr_das_detection_latency_seconds         - Satellite detection processing latency
        12. gl_eudr_das_alert_generation_seconds           - Alert generation duration
        13. gl_eudr_das_severity_scoring_seconds           - Severity scoring duration
        14. gl_eudr_das_compliance_assessment_seconds      - Compliance assessment duration

    Gauges (6):
        15. gl_eudr_das_active_alerts                      - Number of active alerts
        16. gl_eudr_das_monitored_plots                    - Number of monitored supply chain plots
        17. gl_eudr_das_active_buffers                     - Number of active buffer zones
        18. gl_eudr_das_pending_investigations             - Alerts pending investigation
        19. gl_eudr_das_sla_breaches                       - Current SLA breaches
        20. gl_eudr_das_detection_backlog                  - Detection processing backlog size

Label Values Reference:
    source:
        sentinel2, landsat8, landsat9, glad, hansen_gfc, radd, planet, custom.
    change_type:
        deforestation, degradation, fire, logging, clearing, regrowth, no_change.
    severity:
        critical, high, medium, low, informational.
    status:
        pending, triaged, investigating, resolved, escalated, false_positive, expired.
    country_code:
        ISO 3166-1 alpha-2 codes (e.g. BR, ID, CD, CI, GH, CO, PE).
    commodity:
        cattle, cocoa, coffee, palm_oil, rubber, soya, wood.
    action:
        triage, assign, investigate, resolve, escalate, close, reopen.
    operation:
        detect_changes, generate_alerts, classify_severity, check_buffer,
        verify_cutoff, compare_baseline, transition_workflow, assess_compliance.

Example:
    >>> from greenlang.agents.eudr.deforestation_alert_system.metrics import (
    ...     record_satellite_detection,
    ...     record_alert_generated,
    ...     record_severity_classification,
    ...     observe_detection_latency,
    ...     set_active_alerts,
    ...     set_monitored_plots,
    ... )
    >>> record_satellite_detection("sentinel2", "deforestation", "BR")
    >>> record_alert_generated("high", "BR", "palm_oil")
    >>> record_severity_classification("high")
    >>> observe_detection_latency(0.045, "sentinel2")
    >>> set_active_alerts(42, "high")
    >>> set_monitored_plots(1250)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-020 Deforestation Alert System (GL-EUDR-DAS-020)
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
        "deforestation alert system metrics disabled"
    )

# ---------------------------------------------------------------------------
# Safe metric registration helpers (avoid collisions with other modules)
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
    _satellite_detections_total = _safe_counter(
        "gl_eudr_das_satellite_detections_total",
        "Total number of satellite change detections processed",
        ["source", "change_type", "country_code"],
    )

    _alerts_generated_total = _safe_counter(
        "gl_eudr_das_alerts_generated_total",
        "Total number of deforestation alerts generated",
        ["severity", "country_code", "commodity"],
    )

    _severity_classifications_total = _safe_counter(
        "gl_eudr_das_severity_classifications_total",
        "Total number of severity classifications completed",
        ["severity"],
    )

    _buffer_checks_total = _safe_counter(
        "gl_eudr_das_buffer_checks_total",
        "Total number of buffer zone checks performed",
        ["country_code", "commodity"],
    )

    _cutoff_verifications_total = _safe_counter(
        "gl_eudr_das_cutoff_verifications_total",
        "Total number of EUDR cutoff date verifications completed",
        ["result", "country_code"],
    )

    _baseline_comparisons_total = _safe_counter(
        "gl_eudr_das_baseline_comparisons_total",
        "Total number of historical baseline comparisons completed",
        ["change_type", "country_code"],
    )

    _workflow_transitions_total = _safe_counter(
        "gl_eudr_das_workflow_transitions_total",
        "Total number of workflow state transitions",
        ["action", "status"],
    )

    _compliance_assessments_total = _safe_counter(
        "gl_eudr_das_compliance_assessments_total",
        "Total number of compliance impact assessments completed",
        ["outcome", "country_code"],
    )

    _false_positives_total = _safe_counter(
        "gl_eudr_das_false_positives_total",
        "Total number of false positive detections identified",
        ["source", "country_code"],
    )

    _api_errors_total = _safe_counter(
        "gl_eudr_das_api_errors_total",
        "Total number of API errors by operation",
        ["operation"],
    )

    # Histograms (4)
    _detection_latency_seconds = _safe_histogram(
        "gl_eudr_das_detection_latency_seconds",
        "Satellite detection processing latency in seconds",
        ["source"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    _alert_generation_seconds = _safe_histogram(
        "gl_eudr_das_alert_generation_seconds",
        "Alert generation duration in seconds",
        ["severity"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )

    _severity_scoring_seconds = _safe_histogram(
        "gl_eudr_das_severity_scoring_seconds",
        "Severity scoring computation duration in seconds",
        [],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    )

    _compliance_assessment_seconds = _safe_histogram(
        "gl_eudr_das_compliance_assessment_seconds",
        "Compliance impact assessment duration in seconds",
        ["outcome"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )

    # Gauges (6)
    _active_alerts = _safe_gauge(
        "gl_eudr_das_active_alerts",
        "Number of active (unresolved) deforestation alerts",
        ["severity"],
    )

    _monitored_plots = _safe_gauge(
        "gl_eudr_das_monitored_plots",
        "Number of supply chain plots being monitored",
        [],
    )

    _active_buffers = _safe_gauge(
        "gl_eudr_das_active_buffers",
        "Number of active spatial buffer zones",
        ["commodity"],
    )

    _pending_investigations = _safe_gauge(
        "gl_eudr_das_pending_investigations",
        "Number of alerts pending investigation",
        ["severity"],
    )

    _sla_breaches = _safe_gauge(
        "gl_eudr_das_sla_breaches",
        "Number of current SLA breaches",
        ["severity"],
    )

    _detection_backlog = _safe_gauge(
        "gl_eudr_das_detection_backlog",
        "Number of detections in processing backlog",
        ["source"],
    )


# ---------------------------------------------------------------------------
# Helper functions (20 functions matching 20 metrics)
# ---------------------------------------------------------------------------


def record_satellite_detection(
    source: str = "unknown",
    change_type: str = "unknown",
    country_code: str = "unknown",
) -> None:
    """Record a satellite change detection event.

    Args:
        source: Satellite source (sentinel2, landsat8, glad, etc.).
        change_type: Type of change detected (deforestation, degradation, etc.).
        country_code: ISO 3166-1 alpha-2 country code.
    """
    if PROMETHEUS_AVAILABLE:
        _satellite_detections_total.labels(
            source=source,
            change_type=change_type,
            country_code=country_code,
        ).inc()


def record_alert_generated(
    severity: str = "medium",
    country_code: str = "unknown",
    commodity: str = "unknown",
) -> None:
    """Record a deforestation alert generation.

    Args:
        severity: Alert severity (critical, high, medium, low, informational).
        country_code: ISO 3166-1 alpha-2 country code.
        commodity: EUDR commodity affected.
    """
    if PROMETHEUS_AVAILABLE:
        _alerts_generated_total.labels(
            severity=severity,
            country_code=country_code,
            commodity=commodity,
        ).inc()


def record_severity_classification(
    severity: str = "medium",
) -> None:
    """Record a severity classification completion.

    Args:
        severity: Classified severity level.
    """
    if PROMETHEUS_AVAILABLE:
        _severity_classifications_total.labels(
            severity=severity,
        ).inc()


def record_buffer_check(
    country_code: str = "unknown",
    commodity: str = "unknown",
) -> None:
    """Record a buffer zone check.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        commodity: EUDR commodity associated with the buffer.
    """
    if PROMETHEUS_AVAILABLE:
        _buffer_checks_total.labels(
            country_code=country_code,
            commodity=commodity,
        ).inc()


def record_cutoff_verification(
    result: str = "uncertain",
    country_code: str = "unknown",
) -> None:
    """Record a cutoff date verification completion.

    Args:
        result: Verification result (pre_cutoff, post_cutoff, uncertain, ongoing).
        country_code: ISO 3166-1 alpha-2 country code.
    """
    if PROMETHEUS_AVAILABLE:
        _cutoff_verifications_total.labels(
            result=result,
            country_code=country_code,
        ).inc()


def record_baseline_comparison(
    change_type: str = "no_change",
    country_code: str = "unknown",
) -> None:
    """Record a historical baseline comparison completion.

    Args:
        change_type: Type of change detected in comparison.
        country_code: ISO 3166-1 alpha-2 country code.
    """
    if PROMETHEUS_AVAILABLE:
        _baseline_comparisons_total.labels(
            change_type=change_type,
            country_code=country_code,
        ).inc()


def record_workflow_transition(
    action: str = "triage",
    status: str = "pending",
) -> None:
    """Record a workflow state transition.

    Args:
        action: Workflow action performed.
        status: New workflow status after transition.
    """
    if PROMETHEUS_AVAILABLE:
        _workflow_transitions_total.labels(
            action=action,
            status=status,
        ).inc()


def record_compliance_assessment(
    outcome: str = "under_review",
    country_code: str = "unknown",
) -> None:
    """Record a compliance impact assessment completion.

    Args:
        outcome: Compliance outcome (compliant, non_compliant, etc.).
        country_code: ISO 3166-1 alpha-2 country code.
    """
    if PROMETHEUS_AVAILABLE:
        _compliance_assessments_total.labels(
            outcome=outcome,
            country_code=country_code,
        ).inc()


def record_false_positive(
    source: str = "unknown",
    country_code: str = "unknown",
) -> None:
    """Record a false positive detection.

    Args:
        source: Satellite source that produced the false positive.
        country_code: ISO 3166-1 alpha-2 country code.
    """
    if PROMETHEUS_AVAILABLE:
        _false_positives_total.labels(
            source=source,
            country_code=country_code,
        ).inc()


def record_api_error(operation: str) -> None:
    """Record an API error.

    Args:
        operation: Operation name (detect_changes, generate_alerts, etc.).
    """
    if PROMETHEUS_AVAILABLE:
        _api_errors_total.labels(operation=operation).inc()


def observe_detection_latency(
    duration_seconds: float,
    source: str = "unknown",
) -> None:
    """Observe satellite detection processing latency.

    Args:
        duration_seconds: Duration in seconds.
        source: Satellite source (sentinel2, landsat8, etc.).
    """
    if PROMETHEUS_AVAILABLE:
        _detection_latency_seconds.labels(
            source=source,
        ).observe(duration_seconds)


def observe_alert_generation_duration(
    duration_seconds: float,
    severity: str = "medium",
) -> None:
    """Observe alert generation duration.

    Args:
        duration_seconds: Duration in seconds.
        severity: Alert severity level.
    """
    if PROMETHEUS_AVAILABLE:
        _alert_generation_seconds.labels(
            severity=severity,
        ).observe(duration_seconds)


def observe_severity_scoring_duration(
    duration_seconds: float,
) -> None:
    """Observe severity scoring computation duration.

    Args:
        duration_seconds: Duration in seconds.
    """
    if PROMETHEUS_AVAILABLE:
        _severity_scoring_seconds.observe(duration_seconds)


def observe_compliance_assessment_duration(
    duration_seconds: float,
    outcome: str = "under_review",
) -> None:
    """Observe compliance impact assessment duration.

    Args:
        duration_seconds: Duration in seconds.
        outcome: Compliance outcome.
    """
    if PROMETHEUS_AVAILABLE:
        _compliance_assessment_seconds.labels(
            outcome=outcome,
        ).observe(duration_seconds)


def set_active_alerts(
    count: int,
    severity: str = "all",
) -> None:
    """Set the number of active (unresolved) deforestation alerts.

    Args:
        count: Number of active alerts.
        severity: Severity filter (or "all").
    """
    if PROMETHEUS_AVAILABLE:
        _active_alerts.labels(severity=severity).set(count)


def set_monitored_plots(count: int) -> None:
    """Set the number of supply chain plots being monitored.

    Args:
        count: Number of monitored plots.
    """
    if PROMETHEUS_AVAILABLE:
        _monitored_plots.set(count)


def set_active_buffers(
    count: int,
    commodity: str = "all",
) -> None:
    """Set the number of active spatial buffer zones.

    Args:
        count: Number of active buffers.
        commodity: EUDR commodity filter (or "all").
    """
    if PROMETHEUS_AVAILABLE:
        _active_buffers.labels(commodity=commodity).set(count)


def set_pending_investigations(
    count: int,
    severity: str = "all",
) -> None:
    """Set the number of alerts pending investigation.

    Args:
        count: Number of pending investigations.
        severity: Severity filter (or "all").
    """
    if PROMETHEUS_AVAILABLE:
        _pending_investigations.labels(severity=severity).set(count)


def set_sla_breaches(
    count: int,
    severity: str = "all",
) -> None:
    """Set the number of current SLA breaches.

    Args:
        count: Number of SLA breaches.
        severity: Severity filter (or "all").
    """
    if PROMETHEUS_AVAILABLE:
        _sla_breaches.labels(severity=severity).set(count)


def set_detection_backlog(
    count: int,
    source: str = "all",
) -> None:
    """Set the detection processing backlog size.

    Args:
        count: Number of detections in backlog.
        source: Satellite source filter (or "all").
    """
    if PROMETHEUS_AVAILABLE:
        _detection_backlog.labels(source=source).set(count)
