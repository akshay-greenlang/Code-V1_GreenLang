# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-008: Multi-Tier Supplier Tracker

18 Prometheus metrics for multi-tier supplier tracking service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_mst_`` prefix (GreenLang EUDR Multi-Tier
Supplier Tracker) for consistent identification in Prometheus queries,
Grafana dashboards, and alerting rules across the GreenLang platform.

Metrics:
    1.  gl_eudr_mst_suppliers_discovered_total         (Counter)
    2.  gl_eudr_mst_suppliers_onboarded_total           (Counter)
    3.  gl_eudr_mst_relationships_created_total         (Counter)
    4.  gl_eudr_mst_tier_depth_assessments_total        (Counter)
    5.  gl_eudr_mst_risk_assessments_total              (Counter)
    6.  gl_eudr_mst_risk_alerts_total                   (Counter, labels: severity)
    7.  gl_eudr_mst_compliance_checks_total             (Counter, labels: status)
    8.  gl_eudr_mst_compliance_alerts_total             (Counter, labels: alert_type)
    9.  gl_eudr_mst_gaps_detected_total                 (Counter, labels: severity)
    10. gl_eudr_mst_gaps_remediated_total               (Counter)
    11. gl_eudr_mst_reports_generated_total              (Counter, labels: report_type, format)
    12. gl_eudr_mst_batch_jobs_total                     (Counter, labels: status)
    13. gl_eudr_mst_discovery_duration_seconds            (Histogram)
    14. gl_eudr_mst_risk_propagation_duration_seconds     (Histogram)
    15. gl_eudr_mst_compliance_check_duration_seconds     (Histogram)
    16. gl_eudr_mst_active_suppliers                      (Gauge)
    17. gl_eudr_mst_avg_tier_depth                        (Gauge)
    18. gl_eudr_mst_api_errors_total                      (Counter, labels: operation)

Label Values Reference:
    severity (risk_alerts):
        warning, critical.
    status (compliance_checks):
        compliant, conditionally_compliant, non_compliant, unverified, expired.
    alert_type (compliance_alerts):
        dds_expiry, cert_expiry, status_change, geolocation_gap,
        deforestation_alert.
    severity (gaps_detected):
        critical, major, minor.
    report_type:
        audit, tier_summary, gaps, risk_propagation, dds_readiness.
    format:
        json, pdf, csv, eudr_xml.
    status (batch_jobs):
        completed, failed, cancelled.
    operation (api_errors):
        discover, create, update, delete, assess, propagate, check,
        certify, analyze, remediate, export, batch.

Example:
    >>> from greenlang.agents.eudr.multi_tier_supplier.metrics import (
    ...     record_supplier_discovered,
    ...     record_compliance_check,
    ...     observe_discovery_duration,
    ...     set_active_suppliers,
    ... )
    >>> record_supplier_discovered()
    >>> record_compliance_check("compliant")
    >>> observe_discovery_duration(2.5)
    >>> set_active_suppliers(1500)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
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
        "multi-tier supplier tracker metrics disabled"
    )

# ---------------------------------------------------------------------------
# Safe metric registration helpers (avoid collisions with other modules)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    from prometheus_client import REGISTRY as _REGISTRY

    def _safe_counter(
        name: str, doc: str, labelnames: list = None,  # type: ignore[assignment]
    ):
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
    ):
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
    ):
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
    # 1. Total suppliers discovered
    mst_suppliers_discovered_total = _safe_counter(
        "gl_eudr_mst_suppliers_discovered_total",
        "Total suppliers discovered through hierarchy mapping",
    )

    # 2. Total suppliers onboarded
    mst_suppliers_onboarded_total = _safe_counter(
        "gl_eudr_mst_suppliers_onboarded_total",
        "Total suppliers successfully onboarded with profiles",
    )

    # 3. Total relationships created
    mst_relationships_created_total = _safe_counter(
        "gl_eudr_mst_relationships_created_total",
        "Total supplier-to-supplier relationships created",
    )

    # 4. Total tier depth assessments
    mst_tier_depth_assessments_total = _safe_counter(
        "gl_eudr_mst_tier_depth_assessments_total",
        "Total tier depth assessments performed",
    )

    # 5. Total risk assessments
    mst_risk_assessments_total = _safe_counter(
        "gl_eudr_mst_risk_assessments_total",
        "Total risk assessments performed",
    )

    # 6. Risk threshold alerts by severity
    mst_risk_alerts_total = _safe_counter(
        "gl_eudr_mst_risk_alerts_total",
        "Total risk threshold alert notifications",
        labelnames=["severity"],
    )

    # 7. Compliance checks by resulting status
    mst_compliance_checks_total = _safe_counter(
        "gl_eudr_mst_compliance_checks_total",
        "Total compliance checks by resulting status",
        labelnames=["status"],
    )

    # 8. Compliance status change alerts by type
    mst_compliance_alerts_total = _safe_counter(
        "gl_eudr_mst_compliance_alerts_total",
        "Total compliance status change alert notifications",
        labelnames=["alert_type"],
    )

    # 9. Data gaps detected by severity
    mst_gaps_detected_total = _safe_counter(
        "gl_eudr_mst_gaps_detected_total",
        "Total data gaps detected by severity",
        labelnames=["severity"],
    )

    # 10. Data gaps remediated
    mst_gaps_remediated_total = _safe_counter(
        "gl_eudr_mst_gaps_remediated_total",
        "Total data gaps successfully remediated",
    )

    # 11. Reports generated by type and format
    mst_reports_generated_total = _safe_counter(
        "gl_eudr_mst_reports_generated_total",
        "Total reports generated by type and format",
        labelnames=["report_type", "format"],
    )

    # 12. Batch jobs by status
    mst_batch_jobs_total = _safe_counter(
        "gl_eudr_mst_batch_jobs_total",
        "Total batch processing jobs by status",
        labelnames=["status"],
    )

    # 13. Discovery operation latency
    mst_discovery_duration_seconds = _safe_histogram(
        "gl_eudr_mst_discovery_duration_seconds",
        "Duration of supplier discovery operations in seconds",
        buckets=(
            0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
            10.0, 30.0, 60.0, 120.0, 300.0,
        ),
    )

    # 14. Risk propagation latency
    mst_risk_propagation_duration_seconds = _safe_histogram(
        "gl_eudr_mst_risk_propagation_duration_seconds",
        "Duration of risk propagation operations in seconds",
        buckets=(
            0.01, 0.05, 0.1, 0.25, 0.5, 1.0,
            2.5, 5.0, 10.0, 30.0,
        ),
    )

    # 15. Compliance check latency
    mst_compliance_check_duration_seconds = _safe_histogram(
        "gl_eudr_mst_compliance_check_duration_seconds",
        "Duration of compliance check operations in seconds",
        buckets=(
            0.01, 0.05, 0.1, 0.25, 0.5, 1.0,
            2.5, 5.0, 10.0, 30.0,
        ),
    )

    # 16. Currently active suppliers
    mst_active_suppliers = _safe_gauge(
        "gl_eudr_mst_active_suppliers",
        "Number of currently active supplier profiles",
    )

    # 17. Average tier depth
    mst_avg_tier_depth = _safe_gauge(
        "gl_eudr_mst_avg_tier_depth",
        "Average tier depth across all active supply chains",
    )

    # 18. API errors by operation type
    mst_api_errors_total = _safe_counter(
        "gl_eudr_mst_api_errors_total",
        "Total API errors encountered by operation type",
        labelnames=["operation"],
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    mst_suppliers_discovered_total = None            # type: ignore[assignment]
    mst_suppliers_onboarded_total = None             # type: ignore[assignment]
    mst_relationships_created_total = None           # type: ignore[assignment]
    mst_tier_depth_assessments_total = None          # type: ignore[assignment]
    mst_risk_assessments_total = None                # type: ignore[assignment]
    mst_risk_alerts_total = None                     # type: ignore[assignment]
    mst_compliance_checks_total = None               # type: ignore[assignment]
    mst_compliance_alerts_total = None               # type: ignore[assignment]
    mst_gaps_detected_total = None                   # type: ignore[assignment]
    mst_gaps_remediated_total = None                 # type: ignore[assignment]
    mst_reports_generated_total = None               # type: ignore[assignment]
    mst_batch_jobs_total = None                      # type: ignore[assignment]
    mst_discovery_duration_seconds = None            # type: ignore[assignment]
    mst_risk_propagation_duration_seconds = None     # type: ignore[assignment]
    mst_compliance_check_duration_seconds = None     # type: ignore[assignment]
    mst_active_suppliers = None                      # type: ignore[assignment]
    mst_avg_tier_depth = None                        # type: ignore[assignment]
    mst_api_errors_total = None                      # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_supplier_discovered() -> None:
    """Record a supplier discovery event."""
    if not PROMETHEUS_AVAILABLE:
        return
    mst_suppliers_discovered_total.inc()


def record_supplier_onboarded() -> None:
    """Record a supplier onboarding event."""
    if not PROMETHEUS_AVAILABLE:
        return
    mst_suppliers_onboarded_total.inc()


def record_relationship_created() -> None:
    """Record a relationship creation event."""
    if not PROMETHEUS_AVAILABLE:
        return
    mst_relationships_created_total.inc()


def record_tier_depth_assessment() -> None:
    """Record a tier depth assessment event."""
    if not PROMETHEUS_AVAILABLE:
        return
    mst_tier_depth_assessments_total.inc()


def record_risk_assessment() -> None:
    """Record a risk assessment event."""
    if not PROMETHEUS_AVAILABLE:
        return
    mst_risk_assessments_total.inc()


def record_risk_alert(severity: str) -> None:
    """Record a risk threshold alert event.

    Args:
        severity: Alert severity ('warning', 'critical').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mst_risk_alerts_total.labels(severity=severity).inc()


def record_compliance_check(status: str) -> None:
    """Record a compliance check event.

    Args:
        status: Resulting compliance status ('compliant',
            'conditionally_compliant', 'non_compliant',
            'unverified', 'expired').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mst_compliance_checks_total.labels(status=status).inc()


def record_compliance_alert(alert_type: str) -> None:
    """Record a compliance alert event.

    Args:
        alert_type: Type of alert ('dds_expiry', 'cert_expiry',
            'status_change', 'geolocation_gap', 'deforestation_alert').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mst_compliance_alerts_total.labels(alert_type=alert_type).inc()


def record_gap_detected(severity: str) -> None:
    """Record a data gap detection event.

    Args:
        severity: Gap severity ('critical', 'major', 'minor').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mst_gaps_detected_total.labels(severity=severity).inc()


def record_gap_remediated() -> None:
    """Record a data gap remediation event."""
    if not PROMETHEUS_AVAILABLE:
        return
    mst_gaps_remediated_total.inc()


def record_report_generated(report_type: str, format: str) -> None:
    """Record a report generation event.

    Args:
        report_type: Type of report ('audit', 'tier_summary', 'gaps',
            'risk_propagation', 'dds_readiness').
        format: Output format ('json', 'pdf', 'csv', 'eudr_xml').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mst_reports_generated_total.labels(
        report_type=report_type,
        format=format,
    ).inc()


def record_batch_job(status: str) -> None:
    """Record a batch job event.

    Args:
        status: Job status ('completed', 'failed', 'cancelled').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mst_batch_jobs_total.labels(status=status).inc()


def observe_discovery_duration(seconds: float) -> None:
    """Record the duration of a supplier discovery operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mst_discovery_duration_seconds.observe(seconds)


def observe_risk_propagation_duration(seconds: float) -> None:
    """Record the duration of a risk propagation operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mst_risk_propagation_duration_seconds.observe(seconds)


def observe_compliance_check_duration(seconds: float) -> None:
    """Record the duration of a compliance check operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mst_compliance_check_duration_seconds.observe(seconds)


def set_active_suppliers(count: int) -> None:
    """Set the gauge for currently active supplier profiles.

    Args:
        count: Number of active suppliers. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mst_active_suppliers.set(count)


def set_avg_tier_depth(depth: float) -> None:
    """Set the gauge for average tier depth across active supply chains.

    Args:
        depth: Average tier depth value.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mst_avg_tier_depth.set(depth)


def record_api_error(operation: str) -> None:
    """Record an API error event by operation type.

    Args:
        operation: Type of operation that failed ('discover', 'create',
            'update', 'delete', 'assess', 'propagate', 'check',
            'certify', 'analyze', 'remediate', 'export', 'batch').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    mst_api_errors_total.labels(operation=operation).inc()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "mst_suppliers_discovered_total",
    "mst_suppliers_onboarded_total",
    "mst_relationships_created_total",
    "mst_tier_depth_assessments_total",
    "mst_risk_assessments_total",
    "mst_risk_alerts_total",
    "mst_compliance_checks_total",
    "mst_compliance_alerts_total",
    "mst_gaps_detected_total",
    "mst_gaps_remediated_total",
    "mst_reports_generated_total",
    "mst_batch_jobs_total",
    "mst_discovery_duration_seconds",
    "mst_risk_propagation_duration_seconds",
    "mst_compliance_check_duration_seconds",
    "mst_active_suppliers",
    "mst_avg_tier_depth",
    "mst_api_errors_total",
    # Helper functions
    "record_supplier_discovered",
    "record_supplier_onboarded",
    "record_relationship_created",
    "record_tier_depth_assessment",
    "record_risk_assessment",
    "record_risk_alert",
    "record_compliance_check",
    "record_compliance_alert",
    "record_gap_detected",
    "record_gap_remediated",
    "record_report_generated",
    "record_batch_job",
    "observe_discovery_duration",
    "observe_risk_propagation_duration",
    "observe_compliance_check_duration",
    "set_active_suppliers",
    "set_avg_tier_depth",
    "record_api_error",
]
