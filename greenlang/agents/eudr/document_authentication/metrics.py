# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-012: Document Authentication

18 Prometheus metrics for document authentication agent service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_dav_`` prefix (GreenLang EUDR Document
Authentication Verification) for consistent identification in Prometheus
queries, Grafana dashboards, and alerting rules across the GreenLang
platform.

Metrics (18 per PRD Section 7.3):
    Counters (14):
        1.  gl_eudr_dav_documents_processed_total        - Documents processed
        2.  gl_eudr_dav_classifications_total             - Classifications performed
        3.  gl_eudr_dav_signatures_verified_total         - Signatures verified
        4.  gl_eudr_dav_signatures_invalid_total          - Invalid signatures detected
        5.  gl_eudr_dav_hashes_computed_total             - Hashes computed
        6.  gl_eudr_dav_duplicates_detected_total         - Duplicate documents detected
        7.  gl_eudr_dav_tampering_detected_total          - Document tampering detected
        8.  gl_eudr_dav_cert_chains_validated_total       - Certificate chains validated
        9.  gl_eudr_dav_cert_revocations_total            - Certificate revocations found
        10. gl_eudr_dav_fraud_alerts_total                - Fraud alerts generated
        11. gl_eudr_dav_fraud_critical_total              - Critical fraud alerts
        12. gl_eudr_dav_crossref_queries_total            - Cross-reference queries
        13. gl_eudr_dav_reports_generated_total           - Reports generated
        14. gl_eudr_dav_api_errors_total                  - API errors by operation

    Histograms (3):
        15. gl_eudr_dav_classification_duration_seconds   - Classification latency
        16. gl_eudr_dav_verification_duration_seconds     - Verification latency
        17. gl_eudr_dav_crossref_duration_seconds         - Cross-reference latency

    Gauges (1):
        18. gl_eudr_dav_active_verifications              - Active verification jobs

Label Values Reference:
    document_type:
        coo, pc, bol, cde, cdi, rspo_cert, fsc_cert, iscc_cert,
        ft_cert, utz_cert, ltr, ltd, fmp, fc, wqc, dds_draft,
        ssd, ic, tc, wr.
    confidence:
        high, medium, low, unknown.
    signature_status:
        valid, invalid, expired, revoked, no_signature,
        unknown_signer, stripped.
    severity:
        low, medium, high, critical.
    registry_type:
        fsc, rspo, iscc, fairtrade, utz_ra, ippc, national_customs.
    report_format:
        json, pdf, csv, eudr_xml.
    operation:
        classify, verify_signature, compute_hash, verify_hash,
        validate_chain, extract_metadata, detect_fraud,
        cross_reference, generate_report, batch_verify.

Example:
    >>> from greenlang.agents.eudr.document_authentication.metrics import (
    ...     record_document_processed,
    ...     record_classification,
    ...     record_signature_verified,
    ...     observe_classification_duration,
    ...     set_active_verifications,
    ... )
    >>> record_document_processed("coo")
    >>> record_classification("coo", "high")
    >>> record_signature_verified("valid")
    >>> observe_classification_duration(0.045)
    >>> set_active_verifications(3)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-012 Document Authentication (GL-EUDR-DAV-012)
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
        "document authentication metrics disabled"
    )

# ---------------------------------------------------------------------------
# Safe metric registration helpers (avoid collisions with other modules)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    from prometheus_client import REGISTRY as _REGISTRY

    def _safe_counter(
        name: str, doc: str, labelnames: list = None,  # type: ignore[assignment]
    ):  # type: ignore[return]
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
    ):  # type: ignore[return]
        """Create a Histogram or retrieve existing one."""
        try:
            kw = {}
            if buckets:
                kw["buckets"] = buckets
            return Histogram(
                name, doc, labelnames=labelnames or [], **kw,
            )
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
    ):  # type: ignore[return]
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
    # -- Counters (14) -------------------------------------------------------

    # 1. Documents processed by document type
    dav_documents_processed_total = _safe_counter(
        "gl_eudr_dav_documents_processed_total",
        "Total documents processed by document type",
        labelnames=["document_type"],
    )

    # 2. Classifications performed by document type and confidence level
    dav_classifications_total = _safe_counter(
        "gl_eudr_dav_classifications_total",
        "Total classifications performed by document type and confidence",
        labelnames=["document_type", "confidence"],
    )

    # 3. Signatures verified by status
    dav_signatures_verified_total = _safe_counter(
        "gl_eudr_dav_signatures_verified_total",
        "Total signatures verified by signature status",
        labelnames=["signature_status"],
    )

    # 4. Invalid signatures detected
    dav_signatures_invalid_total = _safe_counter(
        "gl_eudr_dav_signatures_invalid_total",
        "Total invalid signatures detected",
    )

    # 5. Hashes computed by algorithm
    dav_hashes_computed_total = _safe_counter(
        "gl_eudr_dav_hashes_computed_total",
        "Total hashes computed by algorithm",
        labelnames=["algorithm"],
    )

    # 6. Duplicate documents detected
    dav_duplicates_detected_total = _safe_counter(
        "gl_eudr_dav_duplicates_detected_total",
        "Total duplicate documents detected via hash registry",
    )

    # 7. Document tampering detected
    dav_tampering_detected_total = _safe_counter(
        "gl_eudr_dav_tampering_detected_total",
        "Total documents where tampering was detected",
    )

    # 8. Certificate chains validated by status
    dav_cert_chains_validated_total = _safe_counter(
        "gl_eudr_dav_cert_chains_validated_total",
        "Total certificate chains validated by leaf status",
        labelnames=["leaf_status"],
    )

    # 9. Certificate revocations found
    dav_cert_revocations_total = _safe_counter(
        "gl_eudr_dav_cert_revocations_total",
        "Total certificate revocations found during validation",
    )

    # 10. Fraud alerts generated by severity
    dav_fraud_alerts_total = _safe_counter(
        "gl_eudr_dav_fraud_alerts_total",
        "Total fraud alerts generated by severity",
        labelnames=["severity"],
    )

    # 11. Critical fraud alerts
    dav_fraud_critical_total = _safe_counter(
        "gl_eudr_dav_fraud_critical_total",
        "Total critical fraud alerts generated",
    )

    # 12. Cross-reference queries by registry type
    dav_crossref_queries_total = _safe_counter(
        "gl_eudr_dav_crossref_queries_total",
        "Total cross-reference queries by registry type",
        labelnames=["registry_type"],
    )

    # 13. Reports generated by format
    dav_reports_generated_total = _safe_counter(
        "gl_eudr_dav_reports_generated_total",
        "Total reports generated by output format",
        labelnames=["report_format"],
    )

    # 14. API errors by operation
    dav_api_errors_total = _safe_counter(
        "gl_eudr_dav_api_errors_total",
        "Total API errors encountered by operation",
        labelnames=["operation"],
    )

    # -- Histograms (3) ------------------------------------------------------

    # 15. Classification latency
    dav_classification_duration_seconds = _safe_histogram(
        "gl_eudr_dav_classification_duration_seconds",
        "Duration of document classification operations in seconds",
        buckets=(
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
            1.0, 2.5, 5.0, 10.0, 30.0,
        ),
    )

    # 16. Verification latency (signature + hash + cert chain)
    dav_verification_duration_seconds = _safe_histogram(
        "gl_eudr_dav_verification_duration_seconds",
        "Duration of document verification operations in seconds",
        buckets=(
            0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
            1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
        ),
    )

    # 17. Cross-reference query latency
    dav_crossref_duration_seconds = _safe_histogram(
        "gl_eudr_dav_crossref_duration_seconds",
        "Duration of cross-reference verification queries in seconds",
        buckets=(
            0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
            1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
        ),
    )

    # -- Gauges (1) ----------------------------------------------------------

    # 18. Active verification jobs
    dav_active_verifications = _safe_gauge(
        "gl_eudr_dav_active_verifications",
        "Number of currently active document verification jobs",
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    dav_documents_processed_total = None        # type: ignore[assignment]
    dav_classifications_total = None             # type: ignore[assignment]
    dav_signatures_verified_total = None         # type: ignore[assignment]
    dav_signatures_invalid_total = None          # type: ignore[assignment]
    dav_hashes_computed_total = None             # type: ignore[assignment]
    dav_duplicates_detected_total = None         # type: ignore[assignment]
    dav_tampering_detected_total = None          # type: ignore[assignment]
    dav_cert_chains_validated_total = None       # type: ignore[assignment]
    dav_cert_revocations_total = None            # type: ignore[assignment]
    dav_fraud_alerts_total = None                # type: ignore[assignment]
    dav_fraud_critical_total = None              # type: ignore[assignment]
    dav_crossref_queries_total = None            # type: ignore[assignment]
    dav_reports_generated_total = None           # type: ignore[assignment]
    dav_api_errors_total = None                  # type: ignore[assignment]
    dav_classification_duration_seconds = None   # type: ignore[assignment]
    dav_verification_duration_seconds = None     # type: ignore[assignment]
    dav_crossref_duration_seconds = None         # type: ignore[assignment]
    dav_active_verifications = None              # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_document_processed(document_type: str) -> None:
    """Record a document processed event by document type.

    Args:
        document_type: Type of document processed (coo, pc, bol,
            cde, cdi, rspo_cert, fsc_cert, iscc_cert, ft_cert,
            utz_cert, ltr, ltd, fmp, fc, wqc, dds_draft, ssd,
            ic, tc, wr).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dav_documents_processed_total.labels(
        document_type=document_type,
    ).inc()


def record_classification(document_type: str, confidence: str) -> None:
    """Record a classification event by document type and confidence.

    Args:
        document_type: Classified document type.
        confidence: Confidence level (high, medium, low, unknown).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dav_classifications_total.labels(
        document_type=document_type, confidence=confidence,
    ).inc()


def record_signature_verified(signature_status: str) -> None:
    """Record a signature verification event by status.

    Args:
        signature_status: Verification status (valid, invalid,
            expired, revoked, no_signature, unknown_signer, stripped).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dav_signatures_verified_total.labels(
        signature_status=signature_status,
    ).inc()


def record_signature_invalid() -> None:
    """Record an invalid signature detection event."""
    if not PROMETHEUS_AVAILABLE:
        return
    dav_signatures_invalid_total.inc()


def record_hash_computed(algorithm: str) -> None:
    """Record a hash computation event by algorithm.

    Args:
        algorithm: Hash algorithm used (sha256, sha512, hmac_sha256).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dav_hashes_computed_total.labels(algorithm=algorithm).inc()


def record_duplicate_detected() -> None:
    """Record a duplicate document detection event."""
    if not PROMETHEUS_AVAILABLE:
        return
    dav_duplicates_detected_total.inc()


def record_tampering_detected() -> None:
    """Record a document tampering detection event."""
    if not PROMETHEUS_AVAILABLE:
        return
    dav_tampering_detected_total.inc()


def record_cert_chain_validated(leaf_status: str) -> None:
    """Record a certificate chain validation event by leaf status.

    Args:
        leaf_status: Status of the leaf certificate (valid, expired,
            revoked, self_signed, weak_key, unknown).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dav_cert_chains_validated_total.labels(
        leaf_status=leaf_status,
    ).inc()


def record_cert_revocation() -> None:
    """Record a certificate revocation detection event."""
    if not PROMETHEUS_AVAILABLE:
        return
    dav_cert_revocations_total.inc()


def record_fraud_alert(severity: str) -> None:
    """Record a fraud alert generation event by severity.

    Args:
        severity: Alert severity (low, medium, high, critical).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dav_fraud_alerts_total.labels(severity=severity).inc()


def record_fraud_critical() -> None:
    """Record a critical fraud alert event."""
    if not PROMETHEUS_AVAILABLE:
        return
    dav_fraud_critical_total.inc()


def record_crossref_query(registry_type: str) -> None:
    """Record a cross-reference query event by registry type.

    Args:
        registry_type: Registry queried (fsc, rspo, iscc, fairtrade,
            utz_ra, ippc, national_customs).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dav_crossref_queries_total.labels(
        registry_type=registry_type,
    ).inc()


def record_report_generated(report_format: str) -> None:
    """Record a report generation event by format.

    Args:
        report_format: Output format (json, pdf, csv, eudr_xml).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dav_reports_generated_total.labels(
        report_format=report_format,
    ).inc()


def record_api_error(operation: str) -> None:
    """Record an API error event by operation type.

    Args:
        operation: Type of operation that failed (classify,
            verify_signature, compute_hash, verify_hash,
            validate_chain, extract_metadata, detect_fraud,
            cross_reference, generate_report, batch_verify).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dav_api_errors_total.labels(operation=operation).inc()


def observe_classification_duration(seconds: float) -> None:
    """Record the duration of a document classification operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dav_classification_duration_seconds.observe(seconds)


def observe_verification_duration(seconds: float) -> None:
    """Record the duration of a document verification operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dav_verification_duration_seconds.observe(seconds)


def observe_crossref_duration(seconds: float) -> None:
    """Record the duration of a cross-reference verification query.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dav_crossref_duration_seconds.observe(seconds)


def set_active_verifications(count: int) -> None:
    """Set the gauge for currently active verification jobs.

    Args:
        count: Number of active verification jobs. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    dav_active_verifications.set(count)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "dav_documents_processed_total",
    "dav_classifications_total",
    "dav_signatures_verified_total",
    "dav_signatures_invalid_total",
    "dav_hashes_computed_total",
    "dav_duplicates_detected_total",
    "dav_tampering_detected_total",
    "dav_cert_chains_validated_total",
    "dav_cert_revocations_total",
    "dav_fraud_alerts_total",
    "dav_fraud_critical_total",
    "dav_crossref_queries_total",
    "dav_reports_generated_total",
    "dav_api_errors_total",
    "dav_classification_duration_seconds",
    "dav_verification_duration_seconds",
    "dav_crossref_duration_seconds",
    "dav_active_verifications",
    # Helper functions
    "record_document_processed",
    "record_classification",
    "record_signature_verified",
    "record_signature_invalid",
    "record_hash_computed",
    "record_duplicate_detected",
    "record_tampering_detected",
    "record_cert_chain_validated",
    "record_cert_revocation",
    "record_fraud_alert",
    "record_fraud_critical",
    "record_crossref_query",
    "record_report_generated",
    "record_api_error",
    "observe_classification_duration",
    "observe_verification_duration",
    "observe_crossref_duration",
    "set_active_verifications",
]
