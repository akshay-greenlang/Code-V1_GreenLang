# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-023: Legal Compliance Verifier

20 Prometheus metrics for the Legal Compliance Verifier agent service
monitoring with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_lcv_`` prefix for consistent
identification in Prometheus queries and Grafana dashboards.

Metrics (20 per Architecture Specification Section 7):
    Counters (10):
        1.  gl_eudr_lcv_framework_queries_total
        2.  gl_eudr_lcv_document_verifications_total
        3.  gl_eudr_lcv_certification_validations_total
        4.  gl_eudr_lcv_red_flag_scans_total
        5.  gl_eudr_lcv_red_flags_triggered_total
        6.  gl_eudr_lcv_compliance_assessments_total
        7.  gl_eudr_lcv_audit_reports_processed_total
        8.  gl_eudr_lcv_reports_generated_total
        9.  gl_eudr_lcv_batch_jobs_total
        10. gl_eudr_lcv_api_errors_total

    Histograms (5):
        11. gl_eudr_lcv_compliance_check_duration_seconds
        12. gl_eudr_lcv_full_assessment_duration_seconds
        13. gl_eudr_lcv_document_verification_duration_seconds
        14. gl_eudr_lcv_red_flag_scan_duration_seconds
        15. gl_eudr_lcv_report_generation_duration_seconds

    Gauges (5):
        16. gl_eudr_lcv_countries_covered
        17. gl_eudr_lcv_active_red_flags
        18. gl_eudr_lcv_expiring_documents_30d
        19. gl_eudr_lcv_non_compliant_suppliers
        20. gl_eudr_lcv_cache_hit_ratio

Example:
    >>> from greenlang.agents.eudr.legal_compliance_verifier.metrics import (
    ...     record_framework_query,
    ...     record_document_verification,
    ...     observe_compliance_check_duration,
    ...     set_countries_covered,
    ... )
    >>> record_framework_query("BR", "forest_related_rules")
    >>> record_document_verification("BR", "passed")
    >>> observe_compliance_check_duration(0.35)
    >>> set_countries_covered(27)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023 Legal Compliance Verifier (GL-EUDR-LCV-023)
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
        "legal compliance verifier metrics disabled"
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
    _framework_queries_total = _safe_counter(
        "gl_eudr_lcv_framework_queries_total",
        "Total number of legal framework queries performed",
        ["country_code", "category"],
    )
    _document_verifications_total = _safe_counter(
        "gl_eudr_lcv_document_verifications_total",
        "Total number of document verifications performed",
        ["country_code", "result"],
    )
    _certification_validations_total = _safe_counter(
        "gl_eudr_lcv_certification_validations_total",
        "Total number of certification validations performed",
        ["scheme", "result"],
    )
    _red_flag_scans_total = _safe_counter(
        "gl_eudr_lcv_red_flag_scans_total",
        "Total number of red flag scans executed",
        ["country_code", "commodity"],
    )
    _red_flags_triggered_total = _safe_counter(
        "gl_eudr_lcv_red_flags_triggered_total",
        "Total number of individual red flags triggered",
        ["flag_category", "severity"],
    )
    _compliance_assessments_total = _safe_counter(
        "gl_eudr_lcv_compliance_assessments_total",
        "Total number of compliance assessments completed",
        ["country_code", "commodity", "status"],
    )
    _audit_reports_processed_total = _safe_counter(
        "gl_eudr_lcv_audit_reports_processed_total",
        "Total number of audit reports processed",
        ["audit_type"],
    )
    _reports_generated_total = _safe_counter(
        "gl_eudr_lcv_reports_generated_total",
        "Total number of compliance reports generated",
        ["report_type", "format"],
    )
    _batch_jobs_total = _safe_counter(
        "gl_eudr_lcv_batch_jobs_total",
        "Total number of batch processing jobs",
        ["job_type", "status"],
    )
    _api_errors_total = _safe_counter(
        "gl_eudr_lcv_api_errors_total",
        "Total number of API errors by operation",
        ["operation", "status_code"],
    )

    # Histograms (5)
    _compliance_check_duration = _safe_histogram(
        "gl_eudr_lcv_compliance_check_duration_seconds",
        "Single compliance check latency in seconds",
        [],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
    )
    _full_assessment_duration = _safe_histogram(
        "gl_eudr_lcv_full_assessment_duration_seconds",
        "Full 8-category assessment latency in seconds",
        [],
        buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    )
    _document_verification_duration = _safe_histogram(
        "gl_eudr_lcv_document_verification_duration_seconds",
        "Document verification latency in seconds",
        [],
        buckets=(0.1, 0.5, 1.0, 2.0, 5.0),
    )
    _red_flag_scan_duration = _safe_histogram(
        "gl_eudr_lcv_red_flag_scan_duration_seconds",
        "Red flag scan latency in seconds",
        [],
        buckets=(0.1, 0.5, 1.0, 3.0, 5.0),
    )
    _report_generation_duration = _safe_histogram(
        "gl_eudr_lcv_report_generation_duration_seconds",
        "Report generation latency in seconds",
        [],
        buckets=(1.0, 2.0, 5.0, 10.0, 30.0),
    )

    # Gauges (5)
    _countries_covered = _safe_gauge(
        "gl_eudr_lcv_countries_covered",
        "Number of countries with legal framework data",
        [],
    )
    _active_red_flags = _safe_gauge(
        "gl_eudr_lcv_active_red_flags",
        "Number of unacknowledged red flags",
        [],
    )
    _expiring_documents_30d = _safe_gauge(
        "gl_eudr_lcv_expiring_documents_30d",
        "Documents expiring within 30 days",
        [],
    )
    _non_compliant_suppliers = _safe_gauge(
        "gl_eudr_lcv_non_compliant_suppliers",
        "Suppliers with non-compliant status",
        [],
    )
    _cache_hit_ratio = _safe_gauge(
        "gl_eudr_lcv_cache_hit_ratio",
        "Redis cache hit ratio (0.0-1.0)",
        [],
    )


# ---------------------------------------------------------------------------
# Helper functions (20 functions matching 20 metrics)
# ---------------------------------------------------------------------------


def record_framework_query(
    country_code: str = "unknown",
    category: str = "all",
) -> None:
    """Record a legal framework query.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        category: Legislation category queried.
    """
    if PROMETHEUS_AVAILABLE:
        _framework_queries_total.labels(
            country_code=country_code, category=category,
        ).inc()


def record_document_verification(
    country_code: str = "unknown",
    result: str = "unknown",
) -> None:
    """Record a document verification.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        result: Verification result (passed, failed, unverifiable).
    """
    if PROMETHEUS_AVAILABLE:
        _document_verifications_total.labels(
            country_code=country_code, result=result,
        ).inc()


def record_certification_validation(
    scheme: str = "unknown",
    result: str = "unknown",
) -> None:
    """Record a certification validation.

    Args:
        scheme: Certification scheme (fsc_fm, rspo_pc, etc.).
        result: Validation result (valid, expired, suspended, invalid).
    """
    if PROMETHEUS_AVAILABLE:
        _certification_validations_total.labels(
            scheme=scheme, result=result,
        ).inc()


def record_red_flag_scan(
    country_code: str = "unknown",
    commodity: str = "unknown",
) -> None:
    """Record a red flag scan execution.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        commodity: EUDR commodity type.
    """
    if PROMETHEUS_AVAILABLE:
        _red_flag_scans_total.labels(
            country_code=country_code, commodity=commodity,
        ).inc()


def record_red_flag_triggered(
    flag_category: str = "unknown",
    severity: str = "unknown",
) -> None:
    """Record a red flag being triggered.

    Args:
        flag_category: Red flag category.
        severity: Red flag severity (low, moderate, high, critical).
    """
    if PROMETHEUS_AVAILABLE:
        _red_flags_triggered_total.labels(
            flag_category=flag_category, severity=severity,
        ).inc()


def record_compliance_assessment(
    country_code: str = "unknown",
    commodity: str = "unknown",
    status: str = "unknown",
) -> None:
    """Record a compliance assessment completion.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        commodity: EUDR commodity type.
        status: Compliance status result.
    """
    if PROMETHEUS_AVAILABLE:
        _compliance_assessments_total.labels(
            country_code=country_code, commodity=commodity, status=status,
        ).inc()


def record_audit_report_processed(audit_type: str = "unknown") -> None:
    """Record an audit report processing completion.

    Args:
        audit_type: Audit report type (FSC, PEFC, RSPO, etc.).
    """
    if PROMETHEUS_AVAILABLE:
        _audit_reports_processed_total.labels(audit_type=audit_type).inc()


def record_report_generated(
    report_type: str = "unknown",
    report_format: str = "unknown",
) -> None:
    """Record a compliance report generation.

    Args:
        report_type: Report type (full_assessment, etc.).
        report_format: Report format (pdf, json, etc.).
    """
    if PROMETHEUS_AVAILABLE:
        _reports_generated_total.labels(
            report_type=report_type, format=report_format,
        ).inc()


def record_batch_job(
    job_type: str = "unknown",
    status: str = "unknown",
) -> None:
    """Record a batch processing job.

    Args:
        job_type: Batch job type.
        status: Job status (pending, completed, failed).
    """
    if PROMETHEUS_AVAILABLE:
        _batch_jobs_total.labels(job_type=job_type, status=status).inc()


def record_api_error(
    operation: str = "unknown",
    status_code: str = "500",
) -> None:
    """Record an API error.

    Args:
        operation: Operation name.
        status_code: HTTP status code.
    """
    if PROMETHEUS_AVAILABLE:
        _api_errors_total.labels(
            operation=operation, status_code=status_code,
        ).inc()


def observe_compliance_check_duration(duration_seconds: float) -> None:
    """Observe single compliance check duration.

    Args:
        duration_seconds: Duration in seconds.
    """
    if PROMETHEUS_AVAILABLE:
        _compliance_check_duration.observe(duration_seconds)


def observe_full_assessment_duration(duration_seconds: float) -> None:
    """Observe full 8-category assessment duration.

    Args:
        duration_seconds: Duration in seconds.
    """
    if PROMETHEUS_AVAILABLE:
        _full_assessment_duration.observe(duration_seconds)


def observe_document_verification_duration(duration_seconds: float) -> None:
    """Observe document verification duration.

    Args:
        duration_seconds: Duration in seconds.
    """
    if PROMETHEUS_AVAILABLE:
        _document_verification_duration.observe(duration_seconds)


def observe_red_flag_scan_duration(duration_seconds: float) -> None:
    """Observe red flag scan duration.

    Args:
        duration_seconds: Duration in seconds.
    """
    if PROMETHEUS_AVAILABLE:
        _red_flag_scan_duration.observe(duration_seconds)


def observe_report_generation_duration(duration_seconds: float) -> None:
    """Observe report generation duration.

    Args:
        duration_seconds: Duration in seconds.
    """
    if PROMETHEUS_AVAILABLE:
        _report_generation_duration.observe(duration_seconds)


def set_countries_covered(count: int) -> None:
    """Set the number of countries with legal framework data.

    Args:
        count: Number of countries covered.
    """
    if PROMETHEUS_AVAILABLE:
        _countries_covered.set(count)


def set_active_red_flags(count: int) -> None:
    """Set the number of unacknowledged red flags.

    Args:
        count: Number of active red flags.
    """
    if PROMETHEUS_AVAILABLE:
        _active_red_flags.set(count)


def set_expiring_documents_30d(count: int) -> None:
    """Set the number of documents expiring within 30 days.

    Args:
        count: Number of expiring documents.
    """
    if PROMETHEUS_AVAILABLE:
        _expiring_documents_30d.set(count)


def set_non_compliant_suppliers(count: int) -> None:
    """Set the number of non-compliant suppliers.

    Args:
        count: Number of non-compliant suppliers.
    """
    if PROMETHEUS_AVAILABLE:
        _non_compliant_suppliers.set(count)


def set_cache_hit_ratio(ratio: float) -> None:
    """Set the Redis cache hit ratio.

    Args:
        ratio: Cache hit ratio (0.0 to 1.0).
    """
    if PROMETHEUS_AVAILABLE:
        _cache_hit_ratio.set(ratio)
