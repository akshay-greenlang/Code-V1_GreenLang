# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-014: QR Code Generator

18 Prometheus metrics for QR code generator agent service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_qrg_`` prefix (GreenLang EUDR QR Code
Generator) for consistent identification in Prometheus queries, Grafana
dashboards, and alerting rules across the GreenLang platform.

Metrics (18 per PRD Section 7.3):
    Counters (12):
        1.  gl_eudr_qrg_codes_generated_total           - QR codes generated
        2.  gl_eudr_qrg_labels_generated_total           - Labels rendered
        3.  gl_eudr_qrg_payloads_composed_total          - Payloads composed
        4.  gl_eudr_qrg_batch_codes_total                - Batch codes generated
        5.  gl_eudr_qrg_verification_urls_total          - Verification URLs built
        6.  gl_eudr_qrg_scans_total                      - Scan events recorded
        7.  gl_eudr_qrg_counterfeit_detections_total     - Counterfeit detections
        8.  gl_eudr_qrg_bulk_jobs_total                  - Bulk jobs submitted
        9.  gl_eudr_qrg_bulk_codes_total                 - Codes generated via bulk
        10. gl_eudr_qrg_revocations_total                - Codes revoked
        11. gl_eudr_qrg_signature_verifications_total    - Signature verifications
        12. gl_eudr_qrg_api_errors_total                 - API errors by operation

    Histograms (4):
        13. gl_eudr_qrg_generation_duration_seconds      - Code generation latency
        14. gl_eudr_qrg_label_duration_seconds           - Label rendering latency
        15. gl_eudr_qrg_bulk_duration_seconds            - Bulk job duration
        16. gl_eudr_qrg_verification_duration_seconds    - Scan verification latency

    Gauges (2):
        17. gl_eudr_qrg_active_bulk_jobs                 - Active bulk generation jobs
        18. gl_eudr_qrg_active_codes                     - Active (scannable) codes

Label Values Reference:
    output_format:
        png, svg, pdf, zpl, eps.
    content_type:
        full_traceability, compact_verification, consumer_summary,
        batch_identifier, blockchain_anchor.
    error_correction:
        L, M, Q, H.
    commodity:
        cattle, cocoa, coffee, oil_palm, rubber, soya, wood.
    compliance_status:
        compliant, pending, non_compliant, under_review.
    scan_outcome:
        verified, counterfeit_suspected, expired_code,
        revoked_code, error.
    counterfeit_risk:
        low, medium, high, critical.
    template:
        product_label, shipping_label, pallet_label,
        container_label, consumer_label.
    bulk_status:
        queued, processing, completed, failed, cancelled.
    operation:
        generate, compose, render, encode, build_url, sign,
        verify, scan, activate, deactivate, revoke, cancel,
        bulk_submit, reprint, search.

Example:
    >>> from greenlang.agents.eudr.qr_code_generator.metrics import (
    ...     record_code_generated,
    ...     record_label_generated,
    ...     record_scan,
    ...     observe_generation_duration,
    ...     set_active_codes,
    ... )
    >>> record_code_generated("png", "compact_verification")
    >>> record_label_generated("product_label")
    >>> record_scan("verified")
    >>> observe_generation_duration(0.35)
    >>> set_active_codes(1500)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014 QR Code Generator (GL-EUDR-QRG-014)
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
        "QR code generator metrics disabled"
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
    # -- Counters (12) -------------------------------------------------------

    # 1. QR codes generated by output format and content type
    qrg_codes_generated_total = _safe_counter(
        "gl_eudr_qrg_codes_generated_total",
        "Total QR codes generated by output format and content type",
        labelnames=["output_format", "content_type"],
    )

    # 2. Labels rendered by template
    qrg_labels_generated_total = _safe_counter(
        "gl_eudr_qrg_labels_generated_total",
        "Total labels rendered by template type",
        labelnames=["template"],
    )

    # 3. Payloads composed by content type
    qrg_payloads_composed_total = _safe_counter(
        "gl_eudr_qrg_payloads_composed_total",
        "Total payloads composed by content type",
        labelnames=["content_type"],
    )

    # 4. Batch codes generated by commodity
    qrg_batch_codes_total = _safe_counter(
        "gl_eudr_qrg_batch_codes_total",
        "Total batch codes generated by commodity",
        labelnames=["commodity"],
    )

    # 5. Verification URLs built
    qrg_verification_urls_total = _safe_counter(
        "gl_eudr_qrg_verification_urls_total",
        "Total verification URLs constructed",
    )

    # 6. Scan events recorded by outcome
    qrg_scans_total = _safe_counter(
        "gl_eudr_qrg_scans_total",
        "Total scan events recorded by scan outcome",
        labelnames=["scan_outcome"],
    )

    # 7. Counterfeit detections by risk level
    qrg_counterfeit_detections_total = _safe_counter(
        "gl_eudr_qrg_counterfeit_detections_total",
        "Total counterfeit detections by risk level",
        labelnames=["counterfeit_risk"],
    )

    # 8. Bulk jobs submitted by status
    qrg_bulk_jobs_total = _safe_counter(
        "gl_eudr_qrg_bulk_jobs_total",
        "Total bulk generation jobs submitted by final status",
        labelnames=["bulk_status"],
    )

    # 9. Codes generated via bulk jobs
    qrg_bulk_codes_total = _safe_counter(
        "gl_eudr_qrg_bulk_codes_total",
        "Total QR codes generated through bulk job processing",
    )

    # 10. Codes revoked by commodity
    qrg_revocations_total = _safe_counter(
        "gl_eudr_qrg_revocations_total",
        "Total QR codes revoked by commodity",
        labelnames=["commodity"],
    )

    # 11. Signature verifications performed
    qrg_signature_verifications_total = _safe_counter(
        "gl_eudr_qrg_signature_verifications_total",
        "Total HMAC signature verifications performed",
    )

    # 12. API errors by operation
    qrg_api_errors_total = _safe_counter(
        "gl_eudr_qrg_api_errors_total",
        "Total API errors encountered by operation",
        labelnames=["operation"],
    )

    # -- Histograms (4) ------------------------------------------------------

    # 13. QR code generation latency
    qrg_generation_duration_seconds = _safe_histogram(
        "gl_eudr_qrg_generation_duration_seconds",
        "Duration of QR code generation in seconds",
        buckets=(
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
            0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
        ),
    )

    # 14. Label rendering latency
    qrg_label_duration_seconds = _safe_histogram(
        "gl_eudr_qrg_label_duration_seconds",
        "Duration of label rendering in seconds",
        buckets=(
            0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
            1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
        ),
    )

    # 15. Bulk job total duration
    qrg_bulk_duration_seconds = _safe_histogram(
        "gl_eudr_qrg_bulk_duration_seconds",
        "Duration of bulk generation jobs in seconds",
        buckets=(
            1.0, 5.0, 10.0, 30.0, 60.0, 120.0,
            300.0, 600.0, 1800.0, 3600.0,
        ),
    )

    # 16. Scan verification latency
    qrg_verification_duration_seconds = _safe_histogram(
        "gl_eudr_qrg_verification_duration_seconds",
        "Duration of scan verification operations in seconds",
        buckets=(
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1,
            0.25, 0.5, 1.0, 2.5, 5.0,
        ),
    )

    # -- Gauges (2) ----------------------------------------------------------

    # 17. Active bulk generation jobs
    qrg_active_bulk_jobs = _safe_gauge(
        "gl_eudr_qrg_active_bulk_jobs",
        "Number of currently active bulk generation jobs",
    )

    # 18. Active (scannable) QR codes
    qrg_active_codes = _safe_gauge(
        "gl_eudr_qrg_active_codes",
        "Number of currently active (scannable) QR codes",
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    qrg_codes_generated_total = None              # type: ignore[assignment]
    qrg_labels_generated_total = None              # type: ignore[assignment]
    qrg_payloads_composed_total = None             # type: ignore[assignment]
    qrg_batch_codes_total = None                   # type: ignore[assignment]
    qrg_verification_urls_total = None             # type: ignore[assignment]
    qrg_scans_total = None                         # type: ignore[assignment]
    qrg_counterfeit_detections_total = None         # type: ignore[assignment]
    qrg_bulk_jobs_total = None                     # type: ignore[assignment]
    qrg_bulk_codes_total = None                    # type: ignore[assignment]
    qrg_revocations_total = None                   # type: ignore[assignment]
    qrg_signature_verifications_total = None        # type: ignore[assignment]
    qrg_api_errors_total = None                    # type: ignore[assignment]
    qrg_generation_duration_seconds = None         # type: ignore[assignment]
    qrg_label_duration_seconds = None              # type: ignore[assignment]
    qrg_bulk_duration_seconds = None               # type: ignore[assignment]
    qrg_verification_duration_seconds = None       # type: ignore[assignment]
    qrg_active_bulk_jobs = None                    # type: ignore[assignment]
    qrg_active_codes = None                        # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_code_generated(output_format: str, content_type: str) -> None:
    """Record a QR code generation event by output format and content type.

    Args:
        output_format: Output image format (png, svg, pdf, zpl, eps).
        content_type: Payload content type (full_traceability,
            compact_verification, consumer_summary, batch_identifier,
            blockchain_anchor).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    qrg_codes_generated_total.labels(
        output_format=output_format, content_type=content_type,
    ).inc()


def record_label_generated(template: str) -> None:
    """Record a label rendering event by template type.

    Args:
        template: Label template used (product_label, shipping_label,
            pallet_label, container_label, consumer_label).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    qrg_labels_generated_total.labels(template=template).inc()


def record_payload_composed(content_type: str) -> None:
    """Record a payload composition event by content type.

    Args:
        content_type: Payload content type.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    qrg_payloads_composed_total.labels(
        content_type=content_type,
    ).inc()


def record_batch_code_generated(commodity: str) -> None:
    """Record a batch code generation event by commodity.

    Args:
        commodity: EUDR-regulated commodity type (cattle, cocoa,
            coffee, oil_palm, rubber, soya, wood).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    qrg_batch_codes_total.labels(commodity=commodity).inc()


def record_verification_url_built() -> None:
    """Record a verification URL construction event."""
    if not PROMETHEUS_AVAILABLE:
        return
    qrg_verification_urls_total.inc()


def record_scan(scan_outcome: str) -> None:
    """Record a scan event by outcome.

    Args:
        scan_outcome: Scan verification outcome (verified,
            counterfeit_suspected, expired_code, revoked_code, error).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    qrg_scans_total.labels(scan_outcome=scan_outcome).inc()


def record_counterfeit_detection(counterfeit_risk: str) -> None:
    """Record a counterfeit detection event by risk level.

    Args:
        counterfeit_risk: Assessed risk level (low, medium, high,
            critical).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    qrg_counterfeit_detections_total.labels(
        counterfeit_risk=counterfeit_risk,
    ).inc()


def record_bulk_job(bulk_status: str) -> None:
    """Record a bulk job event by final status.

    Args:
        bulk_status: Final job status (queued, processing, completed,
            failed, cancelled).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    qrg_bulk_jobs_total.labels(bulk_status=bulk_status).inc()


def record_bulk_codes(count: int) -> None:
    """Record codes generated through a bulk job.

    Args:
        count: Number of codes generated in this batch.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    qrg_bulk_codes_total.inc(count)


def record_revocation(commodity: str) -> None:
    """Record a QR code revocation event by commodity.

    Args:
        commodity: EUDR-regulated commodity type.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    qrg_revocations_total.labels(commodity=commodity).inc()


def record_signature_verification() -> None:
    """Record an HMAC signature verification event."""
    if not PROMETHEUS_AVAILABLE:
        return
    qrg_signature_verifications_total.inc()


def record_api_error(operation: str) -> None:
    """Record an API error event by operation type.

    Args:
        operation: Type of operation that failed (generate, compose,
            render, encode, build_url, sign, verify, scan, activate,
            deactivate, revoke, cancel, bulk_submit, reprint, search).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    qrg_api_errors_total.labels(operation=operation).inc()


def observe_generation_duration(seconds: float) -> None:
    """Record the duration of a QR code generation operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    qrg_generation_duration_seconds.observe(seconds)


def observe_label_duration(seconds: float) -> None:
    """Record the duration of a label rendering operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    qrg_label_duration_seconds.observe(seconds)


def observe_bulk_duration(seconds: float) -> None:
    """Record the duration of a bulk generation job.

    Args:
        seconds: Job wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    qrg_bulk_duration_seconds.observe(seconds)


def observe_verification_duration(seconds: float) -> None:
    """Record the duration of a scan verification operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    qrg_verification_duration_seconds.observe(seconds)


def set_active_bulk_jobs(count: int) -> None:
    """Set the gauge for currently active bulk generation jobs.

    Args:
        count: Number of active bulk jobs. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    qrg_active_bulk_jobs.set(count)


def set_active_codes(count: int) -> None:
    """Set the gauge for currently active (scannable) QR codes.

    Args:
        count: Number of active QR codes. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    qrg_active_codes.set(count)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "qrg_codes_generated_total",
    "qrg_labels_generated_total",
    "qrg_payloads_composed_total",
    "qrg_batch_codes_total",
    "qrg_verification_urls_total",
    "qrg_scans_total",
    "qrg_counterfeit_detections_total",
    "qrg_bulk_jobs_total",
    "qrg_bulk_codes_total",
    "qrg_revocations_total",
    "qrg_signature_verifications_total",
    "qrg_api_errors_total",
    "qrg_generation_duration_seconds",
    "qrg_label_duration_seconds",
    "qrg_bulk_duration_seconds",
    "qrg_verification_duration_seconds",
    "qrg_active_bulk_jobs",
    "qrg_active_codes",
    # Helper functions
    "record_code_generated",
    "record_label_generated",
    "record_payload_composed",
    "record_batch_code_generated",
    "record_verification_url_built",
    "record_scan",
    "record_counterfeit_detection",
    "record_bulk_job",
    "record_bulk_codes",
    "record_revocation",
    "record_signature_verification",
    "record_api_error",
    "observe_generation_duration",
    "observe_label_duration",
    "observe_bulk_duration",
    "observe_verification_duration",
    "set_active_bulk_jobs",
    "set_active_codes",
]
