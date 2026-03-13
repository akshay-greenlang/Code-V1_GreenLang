# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-033: Continuous Monitoring Agent

40 Prometheus metrics for continuous monitoring service observability
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_cm_`` prefix for consistent
identification in Prometheus queries, Grafana dashboards, and alerting
rules across the GreenLang platform.

Metrics (40 per PRD Section 7.6):
    Counters (16):
        1.  gl_eudr_cm_supply_chain_scans_total          - Supply chain scans [status]
        2.  gl_eudr_cm_supplier_changes_detected_total    - Supplier changes detected [change_type]
        3.  gl_eudr_cm_certification_expiries_total       - Certification expiry alerts [status]
        4.  gl_eudr_cm_geolocation_drifts_total           - Geolocation drifts detected
        5.  gl_eudr_cm_deforestation_checks_total         - Deforestation checks [status]
        6.  gl_eudr_cm_deforestation_correlations_total   - Deforestation correlations found
        7.  gl_eudr_cm_investigations_triggered_total     - Investigations triggered [type]
        8.  gl_eudr_cm_compliance_audits_total            - Compliance audits [status]
        9.  gl_eudr_cm_compliance_checks_passed_total     - Individual checks passed
        10. gl_eudr_cm_compliance_checks_failed_total     - Individual checks failed
        11. gl_eudr_cm_changes_detected_total             - Changes detected [change_type, impact]
        12. gl_eudr_cm_risk_monitors_total                - Risk score monitors [trend]
        13. gl_eudr_cm_risk_degradations_total            - Risk degradations detected
        14. gl_eudr_cm_freshness_checks_total             - Data freshness checks [status]
        15. gl_eudr_cm_regulatory_updates_total           - Regulatory updates found [impact]
        16. gl_eudr_cm_alerts_generated_total             - Monitoring alerts generated [severity]

    Histograms (14):
        17. gl_eudr_cm_supply_chain_scan_duration_seconds  - Supply chain scan latency
        18. gl_eudr_cm_deforestation_check_duration_seconds - Deforestation check latency
        19. gl_eudr_cm_compliance_audit_duration_seconds    - Compliance audit latency
        20. gl_eudr_cm_change_detection_duration_seconds    - Change detection latency
        21. gl_eudr_cm_risk_monitor_duration_seconds        - Risk monitoring latency
        22. gl_eudr_cm_freshness_check_duration_seconds     - Freshness check latency
        23. gl_eudr_cm_regulatory_check_duration_seconds    - Regulatory check latency
        24. gl_eudr_cm_impact_assessment_duration_seconds   - Impact assessment latency
        25. gl_eudr_cm_correlation_duration_seconds         - Correlation computation latency
        26. gl_eudr_cm_investigation_trigger_duration_seconds - Investigation trigger latency
        27. gl_eudr_cm_trend_analysis_duration_seconds      - Trend analysis latency
        28. gl_eudr_cm_freshness_report_duration_seconds    - Freshness report generation latency
        29. gl_eudr_cm_notification_duration_seconds        - Notification dispatch latency
        30. gl_eudr_cm_entity_mapping_duration_seconds      - Entity mapping latency

    Gauges (10):
        31. gl_eudr_cm_active_scans                        - Active scan count
        32. gl_eudr_cm_pending_investigations               - Pending investigation count
        33. gl_eudr_cm_expiring_certifications              - Expiring certification count
        34. gl_eudr_cm_stale_entities                       - Stale entity count
        35. gl_eudr_cm_open_alerts                          - Open alert count
        36. gl_eudr_cm_critical_alerts                      - Critical alert count
        37. gl_eudr_cm_high_risk_entities                   - High/critical risk entities
        38. gl_eudr_cm_compliance_score                     - Latest compliance score
        39. gl_eudr_cm_freshness_percentage                 - Latest freshness percentage
        40. gl_eudr_cm_monitored_suppliers                  - Total monitored suppliers

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-033 (GL-EUDR-CM-033)
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
    # Counters (16)
    _SUPPLY_CHAIN_SCANS = Counter(
        "gl_eudr_cm_supply_chain_scans_total",
        "Supply chain scans completed",
        ["status"],
    )
    _SUPPLIER_CHANGES_DETECTED = Counter(
        "gl_eudr_cm_supplier_changes_detected_total",
        "Supplier changes detected",
        ["change_type"],
    )
    _CERTIFICATION_EXPIRIES = Counter(
        "gl_eudr_cm_certification_expiries_total",
        "Certification expiry alerts",
        ["status"],
    )
    _GEOLOCATION_DRIFTS = Counter(
        "gl_eudr_cm_geolocation_drifts_total",
        "Geolocation drifts detected",
    )
    _DEFORESTATION_CHECKS = Counter(
        "gl_eudr_cm_deforestation_checks_total",
        "Deforestation checks completed",
        ["status"],
    )
    _DEFORESTATION_CORRELATIONS = Counter(
        "gl_eudr_cm_deforestation_correlations_total",
        "Deforestation correlations found",
    )
    _INVESTIGATIONS_TRIGGERED = Counter(
        "gl_eudr_cm_investigations_triggered_total",
        "Investigations triggered",
        ["type"],
    )
    _COMPLIANCE_AUDITS = Counter(
        "gl_eudr_cm_compliance_audits_total",
        "Compliance audits completed",
        ["status"],
    )
    _COMPLIANCE_CHECKS_PASSED = Counter(
        "gl_eudr_cm_compliance_checks_passed_total",
        "Individual compliance checks passed",
    )
    _COMPLIANCE_CHECKS_FAILED = Counter(
        "gl_eudr_cm_compliance_checks_failed_total",
        "Individual compliance checks failed",
    )
    _CHANGES_DETECTED = Counter(
        "gl_eudr_cm_changes_detected_total",
        "Changes detected across entities",
        ["change_type", "impact"],
    )
    _RISK_MONITORS = Counter(
        "gl_eudr_cm_risk_monitors_total",
        "Risk score monitors completed",
        ["trend"],
    )
    _RISK_DEGRADATIONS = Counter(
        "gl_eudr_cm_risk_degradations_total",
        "Risk degradations detected",
    )
    _FRESHNESS_CHECKS = Counter(
        "gl_eudr_cm_freshness_checks_total",
        "Data freshness checks completed",
        ["status"],
    )
    _REGULATORY_UPDATES = Counter(
        "gl_eudr_cm_regulatory_updates_total",
        "Regulatory updates found",
        ["impact"],
    )
    _ALERTS_GENERATED = Counter(
        "gl_eudr_cm_alerts_generated_total",
        "Monitoring alerts generated",
        ["severity"],
    )

    # Histograms (14)
    _SC_SCAN_DURATION = Histogram(
        "gl_eudr_cm_supply_chain_scan_duration_seconds",
        "Supply chain scan latency",
        buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )
    _DEFORESTATION_CHECK_DURATION = Histogram(
        "gl_eudr_cm_deforestation_check_duration_seconds",
        "Deforestation check latency",
        buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    )
    _COMPLIANCE_AUDIT_DURATION = Histogram(
        "gl_eudr_cm_compliance_audit_duration_seconds",
        "Compliance audit latency",
        buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    )
    _CHANGE_DETECTION_DURATION = Histogram(
        "gl_eudr_cm_change_detection_duration_seconds",
        "Change detection latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _RISK_MONITOR_DURATION = Histogram(
        "gl_eudr_cm_risk_monitor_duration_seconds",
        "Risk monitoring latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _FRESHNESS_CHECK_DURATION = Histogram(
        "gl_eudr_cm_freshness_check_duration_seconds",
        "Freshness check latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _REGULATORY_CHECK_DURATION = Histogram(
        "gl_eudr_cm_regulatory_check_duration_seconds",
        "Regulatory check latency",
        buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    _IMPACT_ASSESSMENT_DURATION = Histogram(
        "gl_eudr_cm_impact_assessment_duration_seconds",
        "Impact assessment latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _CORRELATION_DURATION = Histogram(
        "gl_eudr_cm_correlation_duration_seconds",
        "Correlation computation latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    _INVESTIGATION_TRIGGER_DURATION = Histogram(
        "gl_eudr_cm_investigation_trigger_duration_seconds",
        "Investigation trigger latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    _TREND_ANALYSIS_DURATION = Histogram(
        "gl_eudr_cm_trend_analysis_duration_seconds",
        "Trend analysis latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _FRESHNESS_REPORT_DURATION = Histogram(
        "gl_eudr_cm_freshness_report_duration_seconds",
        "Freshness report generation latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _NOTIFICATION_DURATION = Histogram(
        "gl_eudr_cm_notification_duration_seconds",
        "Notification dispatch latency",
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    _ENTITY_MAPPING_DURATION = Histogram(
        "gl_eudr_cm_entity_mapping_duration_seconds",
        "Entity mapping latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )

    # Gauges (10)
    _ACTIVE_SCANS = Gauge(
        "gl_eudr_cm_active_scans",
        "Number of active monitoring scans",
    )
    _PENDING_INVESTIGATIONS = Gauge(
        "gl_eudr_cm_pending_investigations",
        "Number of pending investigations",
    )
    _EXPIRING_CERTIFICATIONS = Gauge(
        "gl_eudr_cm_expiring_certifications",
        "Number of expiring certifications",
    )
    _STALE_ENTITIES = Gauge(
        "gl_eudr_cm_stale_entities",
        "Number of stale entities",
    )
    _OPEN_ALERTS = Gauge(
        "gl_eudr_cm_open_alerts",
        "Number of open alerts",
    )
    _CRITICAL_ALERTS = Gauge(
        "gl_eudr_cm_critical_alerts",
        "Number of critical alerts",
    )
    _HIGH_RISK_ENTITIES = Gauge(
        "gl_eudr_cm_high_risk_entities",
        "Number of entities scored as high or critical risk",
    )
    _COMPLIANCE_SCORE = Gauge(
        "gl_eudr_cm_compliance_score",
        "Latest overall compliance score",
    )
    _FRESHNESS_PERCENTAGE = Gauge(
        "gl_eudr_cm_freshness_percentage",
        "Latest data freshness percentage",
    )
    _MONITORED_SUPPLIERS = Gauge(
        "gl_eudr_cm_monitored_suppliers",
        "Total number of monitored suppliers",
    )


# ---------------------------------------------------------------------------
# Helper Functions - Counters
# ---------------------------------------------------------------------------


def record_supply_chain_scan(status: str) -> None:
    """Record a supply chain scan completion metric."""
    if _PROMETHEUS_AVAILABLE:
        _SUPPLY_CHAIN_SCANS.labels(status=status).inc()


def record_supplier_change_detected(change_type: str) -> None:
    """Record a supplier change detection metric."""
    if _PROMETHEUS_AVAILABLE:
        _SUPPLIER_CHANGES_DETECTED.labels(change_type=change_type).inc()


def record_certification_expiry(status: str) -> None:
    """Record a certification expiry alert metric."""
    if _PROMETHEUS_AVAILABLE:
        _CERTIFICATION_EXPIRIES.labels(status=status).inc()


def record_geolocation_drift() -> None:
    """Record a geolocation drift detection metric."""
    if _PROMETHEUS_AVAILABLE:
        _GEOLOCATION_DRIFTS.inc()


def record_deforestation_check(status: str) -> None:
    """Record a deforestation check metric."""
    if _PROMETHEUS_AVAILABLE:
        _DEFORESTATION_CHECKS.labels(status=status).inc()


def record_deforestation_correlation() -> None:
    """Record a deforestation correlation metric."""
    if _PROMETHEUS_AVAILABLE:
        _DEFORESTATION_CORRELATIONS.inc()


def record_investigation_triggered(investigation_type: str) -> None:
    """Record an investigation trigger metric."""
    if _PROMETHEUS_AVAILABLE:
        _INVESTIGATIONS_TRIGGERED.labels(type=investigation_type).inc()


def record_compliance_audit(status: str) -> None:
    """Record a compliance audit metric."""
    if _PROMETHEUS_AVAILABLE:
        _COMPLIANCE_AUDITS.labels(status=status).inc()


def record_compliance_check_passed() -> None:
    """Record a compliance check pass metric."""
    if _PROMETHEUS_AVAILABLE:
        _COMPLIANCE_CHECKS_PASSED.inc()


def record_compliance_check_failed() -> None:
    """Record a compliance check failure metric."""
    if _PROMETHEUS_AVAILABLE:
        _COMPLIANCE_CHECKS_FAILED.inc()


def record_change_detected(change_type: str, impact: str) -> None:
    """Record a change detection metric."""
    if _PROMETHEUS_AVAILABLE:
        _CHANGES_DETECTED.labels(change_type=change_type, impact=impact).inc()


def record_risk_monitor(trend: str) -> None:
    """Record a risk monitoring metric."""
    if _PROMETHEUS_AVAILABLE:
        _RISK_MONITORS.labels(trend=trend).inc()


def record_risk_degradation() -> None:
    """Record a risk degradation detection metric."""
    if _PROMETHEUS_AVAILABLE:
        _RISK_DEGRADATIONS.inc()


def record_freshness_check(status: str) -> None:
    """Record a data freshness check metric."""
    if _PROMETHEUS_AVAILABLE:
        _FRESHNESS_CHECKS.labels(status=status).inc()


def record_regulatory_update(impact: str) -> None:
    """Record a regulatory update metric."""
    if _PROMETHEUS_AVAILABLE:
        _REGULATORY_UPDATES.labels(impact=impact).inc()


def record_alert_generated(severity: str) -> None:
    """Record an alert generation metric."""
    if _PROMETHEUS_AVAILABLE:
        _ALERTS_GENERATED.labels(severity=severity).inc()


# ---------------------------------------------------------------------------
# Helper Functions - Histograms
# ---------------------------------------------------------------------------


def observe_supply_chain_scan_duration(duration: float) -> None:
    """Observe supply chain scan duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _SC_SCAN_DURATION.observe(duration)


def observe_deforestation_check_duration(duration: float) -> None:
    """Observe deforestation check duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _DEFORESTATION_CHECK_DURATION.observe(duration)


def observe_compliance_audit_duration(duration: float) -> None:
    """Observe compliance audit duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _COMPLIANCE_AUDIT_DURATION.observe(duration)


def observe_change_detection_duration(duration: float) -> None:
    """Observe change detection duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _CHANGE_DETECTION_DURATION.observe(duration)


def observe_risk_monitor_duration(duration: float) -> None:
    """Observe risk monitoring duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _RISK_MONITOR_DURATION.observe(duration)


def observe_freshness_check_duration(duration: float) -> None:
    """Observe freshness check duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _FRESHNESS_CHECK_DURATION.observe(duration)


def observe_regulatory_check_duration(duration: float) -> None:
    """Observe regulatory check duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _REGULATORY_CHECK_DURATION.observe(duration)


def observe_impact_assessment_duration(duration: float) -> None:
    """Observe impact assessment duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _IMPACT_ASSESSMENT_DURATION.observe(duration)


def observe_correlation_duration(duration: float) -> None:
    """Observe correlation computation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _CORRELATION_DURATION.observe(duration)


def observe_investigation_trigger_duration(duration: float) -> None:
    """Observe investigation trigger duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _INVESTIGATION_TRIGGER_DURATION.observe(duration)


def observe_trend_analysis_duration(duration: float) -> None:
    """Observe trend analysis duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _TREND_ANALYSIS_DURATION.observe(duration)


def observe_freshness_report_duration(duration: float) -> None:
    """Observe freshness report generation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _FRESHNESS_REPORT_DURATION.observe(duration)


def observe_notification_duration(duration: float) -> None:
    """Observe notification dispatch duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _NOTIFICATION_DURATION.observe(duration)


def observe_entity_mapping_duration(duration: float) -> None:
    """Observe entity mapping duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _ENTITY_MAPPING_DURATION.observe(duration)


# ---------------------------------------------------------------------------
# Helper Functions - Gauges
# ---------------------------------------------------------------------------


def set_active_scans(count: int) -> None:
    """Set gauge of active monitoring scans."""
    if _PROMETHEUS_AVAILABLE:
        _ACTIVE_SCANS.set(count)


def set_pending_investigations(count: int) -> None:
    """Set gauge of pending investigations."""
    if _PROMETHEUS_AVAILABLE:
        _PENDING_INVESTIGATIONS.set(count)


def set_expiring_certifications(count: int) -> None:
    """Set gauge of expiring certifications."""
    if _PROMETHEUS_AVAILABLE:
        _EXPIRING_CERTIFICATIONS.set(count)


def set_stale_entities(count: int) -> None:
    """Set gauge of stale entities."""
    if _PROMETHEUS_AVAILABLE:
        _STALE_ENTITIES.set(count)


def set_open_alerts(count: int) -> None:
    """Set gauge of open alerts."""
    if _PROMETHEUS_AVAILABLE:
        _OPEN_ALERTS.set(count)


def set_critical_alerts(count: int) -> None:
    """Set gauge of critical alerts."""
    if _PROMETHEUS_AVAILABLE:
        _CRITICAL_ALERTS.set(count)


def set_high_risk_entities(count: int) -> None:
    """Set gauge of high/critical risk entities."""
    if _PROMETHEUS_AVAILABLE:
        _HIGH_RISK_ENTITIES.set(count)


def set_compliance_score(score: float) -> None:
    """Set gauge of latest compliance score."""
    if _PROMETHEUS_AVAILABLE:
        _COMPLIANCE_SCORE.set(score)


def set_freshness_percentage(percentage: float) -> None:
    """Set gauge of latest freshness percentage."""
    if _PROMETHEUS_AVAILABLE:
        _FRESHNESS_PERCENTAGE.set(percentage)


def set_monitored_suppliers(count: int) -> None:
    """Set gauge of total monitored suppliers."""
    if _PROMETHEUS_AVAILABLE:
        _MONITORED_SUPPLIERS.set(count)
