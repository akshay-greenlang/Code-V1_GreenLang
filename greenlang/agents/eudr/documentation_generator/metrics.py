# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-030: Documentation Generator

18 Prometheus metrics for documentation generator service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_dgn_`` prefix for consistent
identification in Prometheus queries, Grafana dashboards, and alerting
rules across the GreenLang platform.

Metrics (18 per PRD Section 7.6):
    Counters (8):
        1.  gl_eudr_dgn_dds_generated_total             - DDS documents generated [commodity, status]
        2.  gl_eudr_dgn_article9_assemblies_total        - Article 9 packages assembled [commodity]
        3.  gl_eudr_dgn_risk_docs_total                  - Risk assessment docs generated [commodity, risk_level]
        4.  gl_eudr_dgn_mitigation_docs_total            - Mitigation docs generated [commodity]
        5.  gl_eudr_dgn_compliance_packages_total        - Compliance packages built [commodity]
        6.  gl_eudr_dgn_submissions_total                - Submissions to EU IS [commodity, status]
        7.  gl_eudr_dgn_validations_total                - Document validations [document_type, result]
        8.  gl_eudr_dgn_api_errors_total                 - API errors [operation]

    Histograms (5):
        9.  gl_eudr_dgn_dds_generation_duration_seconds        - DDS generation latency [commodity]
        10. gl_eudr_dgn_article9_assembly_duration_seconds     - Article 9 assembly latency [commodity]
        11. gl_eudr_dgn_package_build_duration_seconds         - Compliance package build latency [commodity]
        12. gl_eudr_dgn_submission_duration_seconds            - EU IS submission latency
        13. gl_eudr_dgn_validation_duration_seconds            - Document validation latency [document_type]

    Gauges (5):
        14. gl_eudr_dgn_active_drafts                    - Active DDS drafts count
        15. gl_eudr_dgn_pending_submissions              - Pending submissions count
        16. gl_eudr_dgn_rejected_submissions             - Rejected submissions count
        17. gl_eudr_dgn_document_versions                - Total document versions tracked
        18. gl_eudr_dgn_retention_documents              - Documents under retention management

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-030 (GL-EUDR-DGN-030)
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
    _DDS_GENERATED = Counter(
        "gl_eudr_dgn_dds_generated_total",
        "Due Diligence Statements generated",
        ["commodity", "status"],
    )
    _ARTICLE9_ASSEMBLIES = Counter(
        "gl_eudr_dgn_article9_assemblies_total",
        "Article 9 packages assembled",
        ["commodity"],
    )
    _RISK_DOCS = Counter(
        "gl_eudr_dgn_risk_docs_total",
        "Risk assessment documents generated",
        ["commodity", "risk_level"],
    )
    _MITIGATION_DOCS = Counter(
        "gl_eudr_dgn_mitigation_docs_total",
        "Mitigation documents generated",
        ["commodity"],
    )
    _COMPLIANCE_PACKAGES = Counter(
        "gl_eudr_dgn_compliance_packages_total",
        "Compliance packages built",
        ["commodity"],
    )
    _SUBMISSIONS = Counter(
        "gl_eudr_dgn_submissions_total",
        "Submissions to EU Information System",
        ["commodity", "status"],
    )
    _VALIDATIONS = Counter(
        "gl_eudr_dgn_validations_total",
        "Document validations performed",
        ["document_type", "result"],
    )
    _API_ERRORS = Counter(
        "gl_eudr_dgn_api_errors_total",
        "API errors by operation type",
        ["operation"],
    )

    # Histograms (5)
    _DDS_GENERATION_DURATION = Histogram(
        "gl_eudr_dgn_dds_generation_duration_seconds",
        "DDS generation latency",
        ["commodity"],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    _ARTICLE9_ASSEMBLY_DURATION = Histogram(
        "gl_eudr_dgn_article9_assembly_duration_seconds",
        "Article 9 package assembly latency",
        ["commodity"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    _PACKAGE_BUILD_DURATION = Histogram(
        "gl_eudr_dgn_package_build_duration_seconds",
        "Compliance package build latency",
        ["commodity"],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    _SUBMISSION_DURATION = Histogram(
        "gl_eudr_dgn_submission_duration_seconds",
        "EU Information System submission latency",
        buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )
    _VALIDATION_DURATION = Histogram(
        "gl_eudr_dgn_validation_duration_seconds",
        "Document validation latency",
        ["document_type"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )

    # Gauges (5)
    _ACTIVE_DRAFTS = Gauge(
        "gl_eudr_dgn_active_drafts",
        "Number of active DDS drafts",
    )
    _PENDING_SUBMISSIONS = Gauge(
        "gl_eudr_dgn_pending_submissions",
        "Number of submissions pending acknowledgement",
    )
    _REJECTED_SUBMISSIONS = Gauge(
        "gl_eudr_dgn_rejected_submissions",
        "Number of rejected submissions awaiting resubmission",
    )
    _DOCUMENT_VERSIONS = Gauge(
        "gl_eudr_dgn_document_versions",
        "Total document versions under management",
    )
    _RETENTION_DOCUMENTS = Gauge(
        "gl_eudr_dgn_retention_documents",
        "Documents under retention management",
    )


# ---------------------------------------------------------------------------
# Helper Functions - Counters
# ---------------------------------------------------------------------------


def record_dds_generated(commodity: str, status: str) -> None:
    """Record a DDS generation metric."""
    if _PROMETHEUS_AVAILABLE:
        _DDS_GENERATED.labels(commodity=commodity, status=status).inc()


def record_article9_assembly(commodity: str) -> None:
    """Record an Article 9 package assembly metric."""
    if _PROMETHEUS_AVAILABLE:
        _ARTICLE9_ASSEMBLIES.labels(commodity=commodity).inc()


def record_risk_doc(commodity: str, risk_level: str) -> None:
    """Record a risk assessment document generation metric."""
    if _PROMETHEUS_AVAILABLE:
        _RISK_DOCS.labels(commodity=commodity, risk_level=risk_level).inc()


def record_mitigation_doc(commodity: str) -> None:
    """Record a mitigation document generation metric."""
    if _PROMETHEUS_AVAILABLE:
        _MITIGATION_DOCS.labels(commodity=commodity).inc()


def record_compliance_package(commodity: str) -> None:
    """Record a compliance package build metric."""
    if _PROMETHEUS_AVAILABLE:
        _COMPLIANCE_PACKAGES.labels(commodity=commodity).inc()


def record_submission(commodity: str, status: str) -> None:
    """Record a submission to the EU Information System metric."""
    if _PROMETHEUS_AVAILABLE:
        _SUBMISSIONS.labels(commodity=commodity, status=status).inc()


def record_validation(document_type: str, result: str) -> None:
    """Record a document validation metric."""
    if _PROMETHEUS_AVAILABLE:
        _VALIDATIONS.labels(document_type=document_type, result=result).inc()


def record_api_error(operation: str) -> None:
    """Record an API error metric."""
    if _PROMETHEUS_AVAILABLE:
        _API_ERRORS.labels(operation=operation).inc()


# ---------------------------------------------------------------------------
# Helper Functions - Histograms
# ---------------------------------------------------------------------------


def observe_dds_generation_duration(commodity: str, duration: float) -> None:
    """Observe DDS generation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _DDS_GENERATION_DURATION.labels(commodity=commodity).observe(duration)


def observe_article9_assembly_duration(commodity: str, duration: float) -> None:
    """Observe Article 9 package assembly duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _ARTICLE9_ASSEMBLY_DURATION.labels(commodity=commodity).observe(duration)


def observe_package_build_duration(commodity: str, duration: float) -> None:
    """Observe compliance package build duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _PACKAGE_BUILD_DURATION.labels(commodity=commodity).observe(duration)


def observe_submission_duration(duration: float) -> None:
    """Observe EU Information System submission duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _SUBMISSION_DURATION.observe(duration)


def observe_validation_duration(document_type: str, duration: float) -> None:
    """Observe document validation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _VALIDATION_DURATION.labels(document_type=document_type).observe(duration)


# ---------------------------------------------------------------------------
# Helper Functions - Gauges
# ---------------------------------------------------------------------------


def set_active_drafts(count: int) -> None:
    """Set gauge of active DDS drafts."""
    if _PROMETHEUS_AVAILABLE:
        _ACTIVE_DRAFTS.set(count)


def set_pending_submissions(count: int) -> None:
    """Set gauge of pending submissions."""
    if _PROMETHEUS_AVAILABLE:
        _PENDING_SUBMISSIONS.set(count)


def set_rejected_submissions(count: int) -> None:
    """Set gauge of rejected submissions."""
    if _PROMETHEUS_AVAILABLE:
        _REJECTED_SUBMISSIONS.set(count)


def set_document_versions(count: int) -> None:
    """Set gauge of total document versions."""
    if _PROMETHEUS_AVAILABLE:
        _DOCUMENT_VERSIONS.set(count)


def set_retention_documents(count: int) -> None:
    """Set gauge of documents under retention management."""
    if _PROMETHEUS_AVAILABLE:
        _RETENTION_DOCUMENTS.set(count)
