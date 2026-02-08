# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-001: PDF & Invoice Extractor

12 Prometheus metrics for PDF extractor service monitoring with graceful
fallback when prometheus_client is not installed.

Metrics:
    1.  gl_pdf_documents_processed_total (Counter, labels: document_type, tenant_id)
    2.  gl_pdf_processing_duration_seconds (Histogram, 12 buckets)
    3.  gl_pdf_pages_extracted_total (Counter)
    4.  gl_pdf_fields_extracted_total (Counter, labels: status, document_type)
    5.  gl_pdf_extraction_confidence (Histogram, buckets: 0.1-1.0 by 0.1)
    6.  gl_pdf_ocr_operations_total (Counter, labels: engine, status)
    7.  gl_pdf_validation_errors_total (Counter, labels: severity, document_type)
    8.  gl_pdf_classification_total (Counter, labels: document_type)
    9.  gl_pdf_line_items_extracted_total (Counter)
    10. gl_pdf_batch_jobs_total (Counter, labels: status)
    11. gl_pdf_active_jobs (Gauge)
    12. gl_pdf_queue_size (Gauge)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-001 PDF & Invoice Extractor
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
        "prometheus_client not installed; PDF extractor metrics disabled"
    )


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Documents processed by type and tenant
    pdf_documents_processed_total = Counter(
        "gl_pdf_documents_processed_total",
        "Total PDF documents processed",
        labelnames=["document_type", "tenant_id"],
    )

    # 2. Processing duration histogram (12 buckets covering sub-second to
    #    multi-minute OCR-heavy extractions)
    pdf_processing_duration_seconds = Histogram(
        "gl_pdf_processing_duration_seconds",
        "PDF document processing duration in seconds",
        buckets=(
            0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
            10.0, 15.0, 30.0, 60.0, 120.0, 300.0,
        ),
    )

    # 3. Total pages extracted across all documents
    pdf_pages_extracted_total = Counter(
        "gl_pdf_pages_extracted_total",
        "Total pages extracted from PDF documents",
    )

    # 4. Fields extracted by status and document type
    pdf_fields_extracted_total = Counter(
        "gl_pdf_fields_extracted_total",
        "Total fields extracted from PDF documents",
        labelnames=["status", "document_type"],
    )

    # 5. Extraction confidence scores (0.1 to 1.0 in 0.1 increments)
    pdf_extraction_confidence = Histogram(
        "gl_pdf_extraction_confidence",
        "Extraction confidence score distribution",
        buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    )

    # 6. OCR operations by engine and status
    pdf_ocr_operations_total = Counter(
        "gl_pdf_ocr_operations_total",
        "Total OCR operations performed",
        labelnames=["engine", "status"],
    )

    # 7. Validation errors by severity and document type
    pdf_validation_errors_total = Counter(
        "gl_pdf_validation_errors_total",
        "Total validation errors detected during extraction",
        labelnames=["severity", "document_type"],
    )

    # 8. Classification counts by document type
    pdf_classification_total = Counter(
        "gl_pdf_classification_total",
        "Total document classification operations",
        labelnames=["document_type"],
    )

    # 9. Line items extracted across all invoices/manifests
    pdf_line_items_extracted_total = Counter(
        "gl_pdf_line_items_extracted_total",
        "Total line items extracted from documents",
    )

    # 10. Batch jobs by status
    pdf_batch_jobs_total = Counter(
        "gl_pdf_batch_jobs_total",
        "Total batch extraction jobs",
        labelnames=["status"],
    )

    # 11. Currently active extraction jobs
    pdf_active_jobs = Gauge(
        "gl_pdf_active_jobs",
        "Number of currently active extraction jobs",
    )

    # 12. Current extraction queue depth
    pdf_queue_size = Gauge(
        "gl_pdf_queue_size",
        "Current number of documents waiting in extraction queue",
    )

else:
    # No-op placeholders
    pdf_documents_processed_total = None  # type: ignore[assignment]
    pdf_processing_duration_seconds = None  # type: ignore[assignment]
    pdf_pages_extracted_total = None  # type: ignore[assignment]
    pdf_fields_extracted_total = None  # type: ignore[assignment]
    pdf_extraction_confidence = None  # type: ignore[assignment]
    pdf_ocr_operations_total = None  # type: ignore[assignment]
    pdf_validation_errors_total = None  # type: ignore[assignment]
    pdf_classification_total = None  # type: ignore[assignment]
    pdf_line_items_extracted_total = None  # type: ignore[assignment]
    pdf_batch_jobs_total = None  # type: ignore[assignment]
    pdf_active_jobs = None  # type: ignore[assignment]
    pdf_queue_size = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_document_processed(
    document_type: str,
    tenant_id: str,
    duration_seconds: float,
) -> None:
    """Record a document processing completion with duration.

    Args:
        document_type: Type of document processed (invoice, manifest, etc.).
        tenant_id: Tenant identifier.
        duration_seconds: Total processing duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pdf_documents_processed_total.labels(
        document_type=document_type, tenant_id=tenant_id,
    ).inc()
    pdf_processing_duration_seconds.observe(duration_seconds)


def record_pages_extracted(page_count: int) -> None:
    """Record the number of pages extracted from a document.

    Args:
        page_count: Number of pages extracted.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pdf_pages_extracted_total.inc(page_count)


def record_fields_extracted(
    count: int,
    status: str,
    document_type: str,
) -> None:
    """Record extracted fields by status.

    Args:
        count: Number of fields extracted.
        status: Extraction status (success, low_confidence, failed).
        document_type: Type of source document.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pdf_fields_extracted_total.labels(
        status=status, document_type=document_type,
    ).inc(count)


def record_extraction_confidence(confidence: float) -> None:
    """Record an extraction confidence score.

    Args:
        confidence: Confidence score between 0.0 and 1.0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pdf_extraction_confidence.observe(confidence)


def record_ocr_operation(engine: str, status: str) -> None:
    """Record an OCR engine operation.

    Args:
        engine: OCR engine name (tesseract, textract, azure, google).
        status: Operation status (success, error, timeout).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pdf_ocr_operations_total.labels(engine=engine, status=status).inc()


def record_validation_error(severity: str, document_type: str) -> None:
    """Record a validation error during extraction.

    Args:
        severity: Error severity (critical, high, medium, low, info).
        document_type: Type of document that failed validation.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pdf_validation_errors_total.labels(
        severity=severity, document_type=document_type,
    ).inc()


def record_classification(document_type: str) -> None:
    """Record a document classification event.

    Args:
        document_type: Classified document type.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pdf_classification_total.labels(document_type=document_type).inc()


def record_line_items_extracted(count: int) -> None:
    """Record the number of line items extracted.

    Args:
        count: Number of line items extracted.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pdf_line_items_extracted_total.inc(count)


def record_batch_job(status: str) -> None:
    """Record a batch job event.

    Args:
        status: Batch job status (submitted, completed, failed, partial).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pdf_batch_jobs_total.labels(status=status).inc()


def update_active_jobs(delta: int) -> None:
    """Update the active jobs gauge.

    Args:
        delta: Positive to increment, negative to decrement.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    if delta > 0:
        pdf_active_jobs.inc(delta)
    elif delta < 0:
        pdf_active_jobs.dec(abs(delta))


def update_queue_size(size: int) -> None:
    """Set the current extraction queue size.

    Args:
        size: Current queue depth.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    pdf_queue_size.set(size)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "pdf_documents_processed_total",
    "pdf_processing_duration_seconds",
    "pdf_pages_extracted_total",
    "pdf_fields_extracted_total",
    "pdf_extraction_confidence",
    "pdf_ocr_operations_total",
    "pdf_validation_errors_total",
    "pdf_classification_total",
    "pdf_line_items_extracted_total",
    "pdf_batch_jobs_total",
    "pdf_active_jobs",
    "pdf_queue_size",
    # Helper functions
    "record_document_processed",
    "record_pages_extracted",
    "record_fields_extracted",
    "record_extraction_confidence",
    "record_ocr_operation",
    "record_validation_error",
    "record_classification",
    "record_line_items_extracted",
    "record_batch_job",
    "update_active_jobs",
    "update_queue_size",
]
