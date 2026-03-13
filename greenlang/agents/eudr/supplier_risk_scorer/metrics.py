# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-017: Supplier Risk Scorer

18 Prometheus metrics for supplier risk scorer agent service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_srs_`` prefix (GreenLang EUDR Supplier
Risk Scorer) for consistent identification in Prometheus queries,
Grafana dashboards, and alerting rules across the GreenLang platform.

Metrics (18 per PRD Section 7.3):
    Counters (8):
        1.  gl_eudr_srs_assessments_total              - Supplier risk assessments completed
        2.  gl_eudr_srs_dd_records_total               - Due diligence records created
        3.  gl_eudr_srs_documents_analyzed_total       - Documents analyzed
        4.  gl_eudr_srs_certifications_validated_total - Certifications validated
        5.  gl_eudr_srs_networks_analyzed_total        - Supplier networks analyzed
        6.  gl_eudr_srs_alerts_generated_total         - Alerts generated
        7.  gl_eudr_srs_reports_generated_total        - Risk reports generated
        8.  gl_eudr_srs_api_errors_total               - API errors by operation

    Histograms (5):
        9.  gl_eudr_srs_assessment_duration_seconds     - Supplier assessment latency
        10. gl_eudr_srs_dd_tracking_duration_seconds    - Due diligence tracking latency
        11. gl_eudr_srs_document_analysis_duration_seconds - Document analysis latency
        12. gl_eudr_srs_certification_validation_duration_seconds - Certification validation latency
        13. gl_eudr_srs_report_generation_duration_seconds - Report generation latency

    Gauges (5):
        14. gl_eudr_srs_active_suppliers                - Active suppliers in portfolio
        15. gl_eudr_srs_high_risk_suppliers             - Suppliers classified as high/critical risk
        16. gl_eudr_srs_pending_dd                      - Pending due diligence activities
        17. gl_eudr_srs_expiring_certifications         - Certifications expiring within buffer
        18. gl_eudr_srs_active_alerts                   - Active unresolved alerts

Label Values Reference:
    risk_level:
        low, medium, high, critical.
    supplier_type:
        producer, trader, processor, exporter, importer, broker, cooperative.
    commodity:
        cattle, cocoa, coffee, oil_palm, rubber, soya, wood.
    dd_level:
        simplified, standard, enhanced.
    report_type:
        individual, portfolio, comparative, trend, audit_package, executive.
    report_format:
        json, html, pdf, excel, csv.
    alert_severity:
        info, warning, high, critical.
    alert_type:
        risk_threshold, certification_expiry, document_missing,
        dd_overdue, sanction_hit, behavior_change.
    operation:
        assess, track, analyze, validate, generate, configure,
        compare, search, export.

Example:
    >>> from greenlang.agents.eudr.supplier_risk_scorer.metrics import (
    ...     record_assessment_completed,
    ...     record_dd_record_created,
    ...     record_document_analyzed,
    ...     observe_assessment_duration,
    ...     set_active_suppliers,
    ... )
    >>> record_assessment_completed("high")
    >>> record_dd_record_created("enhanced")
    >>> record_document_analyzed("geolocation")
    >>> observe_assessment_duration(0.125)
    >>> set_active_suppliers(350)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-017 Supplier Risk Scorer (GL-EUDR-SRS-017)
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
        "supplier risk scorer metrics disabled"
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
            # Metric already exists, retrieve it
            return _REGISTRY._names_to_collectors[name]

    def _safe_histogram(name: str, doc: str, labels: list, buckets: tuple) -> Histogram:
        """Safely create or retrieve a Histogram metric."""
        try:
            return Histogram(name, doc, labels, buckets=buckets, registry=_REGISTRY)
        except ValueError:
            # Metric already exists, retrieve it
            return _REGISTRY._names_to_collectors[name]

    def _safe_gauge(name: str, doc: str, labels: list) -> Gauge:
        """Safely create or retrieve a Gauge metric."""
        try:
            return Gauge(name, doc, labels, registry=_REGISTRY)
        except ValueError:
            # Metric already exists, retrieve it
            return _REGISTRY._names_to_collectors[name]

# ---------------------------------------------------------------------------
# Metric definitions (18 metrics)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # Counters (8)
    _assessments_total = _safe_counter(
        "gl_eudr_srs_assessments_total",
        "Total number of supplier risk assessments completed",
        ["risk_level", "supplier_type", "commodity"],
    )

    _dd_records_total = _safe_counter(
        "gl_eudr_srs_dd_records_total",
        "Total number of due diligence records created",
        ["dd_level", "status"],
    )

    _documents_analyzed_total = _safe_counter(
        "gl_eudr_srs_documents_analyzed_total",
        "Total number of documents analyzed",
        ["document_type", "status"],
    )

    _certifications_validated_total = _safe_counter(
        "gl_eudr_srs_certifications_validated_total",
        "Total number of certifications validated",
        ["scheme", "status"],
    )

    _networks_analyzed_total = _safe_counter(
        "gl_eudr_srs_networks_analyzed_total",
        "Total number of supplier networks analyzed",
        ["depth"],
    )

    _alerts_generated_total = _safe_counter(
        "gl_eudr_srs_alerts_generated_total",
        "Total number of alerts generated",
        ["alert_type", "alert_severity"],
    )

    _reports_generated_total = _safe_counter(
        "gl_eudr_srs_reports_generated_total",
        "Total number of risk reports generated",
        ["report_type", "report_format"],
    )

    _api_errors_total = _safe_counter(
        "gl_eudr_srs_api_errors_total",
        "Total number of API errors by operation",
        ["operation"],
    )

    # Histograms (5)
    _assessment_duration_seconds = _safe_histogram(
        "gl_eudr_srs_assessment_duration_seconds",
        "Supplier risk assessment duration in seconds",
        ["risk_level"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    _dd_tracking_duration_seconds = _safe_histogram(
        "gl_eudr_srs_dd_tracking_duration_seconds",
        "Due diligence tracking duration in seconds",
        ["dd_level"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    _document_analysis_duration_seconds = _safe_histogram(
        "gl_eudr_srs_document_analysis_duration_seconds",
        "Document analysis duration in seconds",
        ["document_type"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    _certification_validation_duration_seconds = _safe_histogram(
        "gl_eudr_srs_certification_validation_duration_seconds",
        "Certification validation duration in seconds",
        ["scheme"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    _report_generation_duration_seconds = _safe_histogram(
        "gl_eudr_srs_report_generation_duration_seconds",
        "Risk report generation duration in seconds",
        ["report_type", "report_format"],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
    )

    # Gauges (5)
    _active_suppliers = _safe_gauge(
        "gl_eudr_srs_active_suppliers",
        "Number of active suppliers in portfolio",
        [],
    )

    _high_risk_suppliers = _safe_gauge(
        "gl_eudr_srs_high_risk_suppliers",
        "Number of suppliers classified as high or critical risk",
        ["risk_level"],
    )

    _pending_dd = _safe_gauge(
        "gl_eudr_srs_pending_dd",
        "Number of pending due diligence activities",
        ["dd_level"],
    )

    _expiring_certifications = _safe_gauge(
        "gl_eudr_srs_expiring_certifications",
        "Number of certifications expiring within buffer period",
        ["scheme"],
    )

    _active_alerts = _safe_gauge(
        "gl_eudr_srs_active_alerts",
        "Number of active unresolved alerts",
        ["alert_type", "alert_severity"],
    )

# ---------------------------------------------------------------------------
# Helper functions (18 functions)
# ---------------------------------------------------------------------------


def record_assessment_completed(risk_level: str, supplier_type: str = "unknown", commodity: str = "unknown") -> None:
    """Record a supplier risk assessment completion.

    Args:
        risk_level: Risk level (low, medium, high, critical).
        supplier_type: Supplier type (producer, trader, etc.).
        commodity: Commodity type (cattle, cocoa, etc.).
    """
    if PROMETHEUS_AVAILABLE:
        _assessments_total.labels(
            risk_level=risk_level,
            supplier_type=supplier_type,
            commodity=commodity,
        ).inc()


def record_dd_record_created(dd_level: str, status: str = "in_progress") -> None:
    """Record a due diligence record creation.

    Args:
        dd_level: Due diligence level (simplified, standard, enhanced).
        status: Due diligence status (not_started, in_progress, completed, overdue).
    """
    if PROMETHEUS_AVAILABLE:
        _dd_records_total.labels(dd_level=dd_level, status=status).inc()


def record_document_analyzed(document_type: str, status: str = "verified") -> None:
    """Record a document analysis.

    Args:
        document_type: Document type (geolocation, dds_reference, etc.).
        status: Document status (submitted, verified, rejected, expired, missing).
    """
    if PROMETHEUS_AVAILABLE:
        _documents_analyzed_total.labels(
            document_type=document_type,
            status=status,
        ).inc()


def record_certification_validated(scheme: str, status: str = "valid") -> None:
    """Record a certification validation.

    Args:
        scheme: Certification scheme (FSC, PEFC, RSPO, etc.).
        status: Certification status (valid, expired, suspended, revoked, pending).
    """
    if PROMETHEUS_AVAILABLE:
        _certifications_validated_total.labels(scheme=scheme, status=status).inc()


def record_network_analyzed(depth: int) -> None:
    """Record a supplier network analysis.

    Args:
        depth: Network depth (1-10).
    """
    if PROMETHEUS_AVAILABLE:
        _networks_analyzed_total.labels(depth=str(depth)).inc()


def record_alert_generated(alert_type: str, alert_severity: str) -> None:
    """Record an alert generation.

    Args:
        alert_type: Alert type (risk_threshold, certification_expiry, etc.).
        alert_severity: Alert severity (info, warning, high, critical).
    """
    if PROMETHEUS_AVAILABLE:
        _alerts_generated_total.labels(
            alert_type=alert_type,
            alert_severity=alert_severity,
        ).inc()


def record_report_generated(report_type: str, report_format: str) -> None:
    """Record a risk report generation.

    Args:
        report_type: Report type (individual, portfolio, comparative, etc.).
        report_format: Report format (json, html, pdf, excel, csv).
    """
    if PROMETHEUS_AVAILABLE:
        _reports_generated_total.labels(
            report_type=report_type,
            report_format=report_format,
        ).inc()


def record_api_error(operation: str) -> None:
    """Record an API error.

    Args:
        operation: Operation name (assess, track, analyze, etc.).
    """
    if PROMETHEUS_AVAILABLE:
        _api_errors_total.labels(operation=operation).inc()


def observe_assessment_duration(duration_seconds: float, risk_level: str = "unknown") -> None:
    """Observe supplier risk assessment duration.

    Args:
        duration_seconds: Duration in seconds.
        risk_level: Risk level (low, medium, high, critical).
    """
    if PROMETHEUS_AVAILABLE:
        _assessment_duration_seconds.labels(risk_level=risk_level).observe(duration_seconds)


def observe_dd_tracking_duration(duration_seconds: float, dd_level: str = "standard") -> None:
    """Observe due diligence tracking duration.

    Args:
        duration_seconds: Duration in seconds.
        dd_level: Due diligence level (simplified, standard, enhanced).
    """
    if PROMETHEUS_AVAILABLE:
        _dd_tracking_duration_seconds.labels(dd_level=dd_level).observe(duration_seconds)


def observe_document_analysis_duration(duration_seconds: float, document_type: str = "unknown") -> None:
    """Observe document analysis duration.

    Args:
        duration_seconds: Duration in seconds.
        document_type: Document type (geolocation, dds_reference, etc.).
    """
    if PROMETHEUS_AVAILABLE:
        _document_analysis_duration_seconds.labels(document_type=document_type).observe(duration_seconds)


def observe_certification_validation_duration(duration_seconds: float, scheme: str = "unknown") -> None:
    """Observe certification validation duration.

    Args:
        duration_seconds: Duration in seconds.
        scheme: Certification scheme (FSC, PEFC, RSPO, etc.).
    """
    if PROMETHEUS_AVAILABLE:
        _certification_validation_duration_seconds.labels(scheme=scheme).observe(duration_seconds)


def observe_report_generation_duration(duration_seconds: float, report_type: str = "individual", report_format: str = "pdf") -> None:
    """Observe risk report generation duration.

    Args:
        duration_seconds: Duration in seconds.
        report_type: Report type (individual, portfolio, comparative, etc.).
        report_format: Report format (json, html, pdf, excel, csv).
    """
    if PROMETHEUS_AVAILABLE:
        _report_generation_duration_seconds.labels(
            report_type=report_type,
            report_format=report_format,
        ).observe(duration_seconds)


def set_active_suppliers(count: int) -> None:
    """Set the number of active suppliers in portfolio.

    Args:
        count: Number of active suppliers.
    """
    if PROMETHEUS_AVAILABLE:
        _active_suppliers.set(count)


def set_high_risk_suppliers(count: int, risk_level: str = "high") -> None:
    """Set the number of high/critical risk suppliers.

    Args:
        count: Number of high/critical risk suppliers.
        risk_level: Risk level (high or critical).
    """
    if PROMETHEUS_AVAILABLE:
        _high_risk_suppliers.labels(risk_level=risk_level).set(count)


def set_pending_dd(count: int, dd_level: str = "standard") -> None:
    """Set the number of pending due diligence activities.

    Args:
        count: Number of pending DD activities.
        dd_level: Due diligence level (simplified, standard, enhanced).
    """
    if PROMETHEUS_AVAILABLE:
        _pending_dd.labels(dd_level=dd_level).set(count)


def set_expiring_certifications(count: int, scheme: str = "all") -> None:
    """Set the number of certifications expiring within buffer period.

    Args:
        count: Number of expiring certifications.
        scheme: Certification scheme (FSC, PEFC, RSPO, etc., or "all").
    """
    if PROMETHEUS_AVAILABLE:
        _expiring_certifications.labels(scheme=scheme).set(count)


def set_active_alerts(count: int, alert_type: str = "all", alert_severity: str = "all") -> None:
    """Set the number of active unresolved alerts.

    Args:
        count: Number of active alerts.
        alert_type: Alert type (risk_threshold, certification_expiry, etc., or "all").
        alert_severity: Alert severity (info, warning, high, critical, or "all").
    """
    if PROMETHEUS_AVAILABLE:
        _active_alerts.labels(
            alert_type=alert_type,
            alert_severity=alert_severity,
        ).set(count)
