# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-001: PDF & Invoice Extractor

12 Prometheus metrics for PDF extractor service monitoring with graceful
fallback when prometheus_client is not installed.

Standard metrics (via MetricsFactory):
    1.  gl_pdf_operations_total (Counter, labels: type, tenant_id)
    2.  gl_pdf_processing_duration_seconds (Histogram, 12 buckets)
    3.  gl_pdf_validation_errors_total (Counter, labels: severity, type)
    4.  gl_pdf_batch_jobs_total (Counter, labels: status)
    5.  gl_pdf_active_jobs (Gauge)
    6.  gl_pdf_queue_size (Gauge)

Agent-specific metrics:
    7.  gl_pdf_pages_extracted_total (Counter)
    8.  gl_pdf_fields_extracted_total (Counter, labels: status, document_type)
    9.  gl_pdf_extraction_confidence (Histogram, buckets: 0.1-1.0 by 0.1)
    10. gl_pdf_ocr_operations_total (Counter, labels: engine, status)
    11. gl_pdf_classification_total (Counter, labels: document_type)
    12. gl_pdf_line_items_extracted_total (Counter)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-001 PDF & Invoice Extractor
Status: Production Ready
"""

from __future__ import annotations

from greenlang.data_commons.metrics import (
    CONFIDENCE_BUCKETS,
    PROMETHEUS_AVAILABLE,
    MetricsFactory,
)

# ---------------------------------------------------------------------------
# Standard metrics (6 of 12) via factory
# ---------------------------------------------------------------------------

m = MetricsFactory("gl_pdf", "PDF Extractor")

# ---------------------------------------------------------------------------
# Agent-specific metrics (6 of 12)
# ---------------------------------------------------------------------------

pdf_pages_extracted_total = m.create_custom_counter(
    "pages_extracted_total",
    "Total pages extracted from PDF documents",
)

pdf_fields_extracted_total = m.create_custom_counter(
    "fields_extracted_total",
    "Total fields extracted from PDF documents",
    labelnames=["status", "document_type"],
)

pdf_extraction_confidence = m.create_custom_histogram(
    "extraction_confidence",
    "Extraction confidence score distribution",
    buckets=CONFIDENCE_BUCKETS,
)

pdf_ocr_operations_total = m.create_custom_counter(
    "ocr_operations_total",
    "Total OCR operations performed",
    labelnames=["engine", "status"],
)

pdf_classification_total = m.create_custom_counter(
    "classification_total",
    "Total document classification operations",
    labelnames=["document_type"],
)

pdf_line_items_extracted_total = m.create_custom_counter(
    "line_items_extracted_total",
    "Total line items extracted from documents",
)


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
    m.record_operation(duration_seconds, type=document_type, tenant_id=tenant_id)


def record_pages_extracted(page_count: int) -> None:
    """Record the number of pages extracted from a document.

    Args:
        page_count: Number of pages extracted.
    """
    m.safe_inc(pdf_pages_extracted_total, page_count)


def record_fields_extracted(count: int, status: str, document_type: str) -> None:
    """Record extracted fields by status.

    Args:
        count: Number of fields extracted.
        status: Extraction status (success, low_confidence, failed).
        document_type: Type of source document.
    """
    m.safe_inc(pdf_fields_extracted_total, count, status=status, document_type=document_type)


def record_extraction_confidence(confidence: float) -> None:
    """Record an extraction confidence score.

    Args:
        confidence: Confidence score between 0.0 and 1.0.
    """
    m.safe_observe(pdf_extraction_confidence, confidence)


def record_ocr_operation(engine: str, status: str) -> None:
    """Record an OCR engine operation.

    Args:
        engine: OCR engine name (tesseract, textract, azure, google).
        status: Operation status (success, error, timeout).
    """
    m.safe_inc(pdf_ocr_operations_total, 1, engine=engine, status=status)


def record_validation_error(severity: str, document_type: str) -> None:
    """Record a validation error during extraction.

    Args:
        severity: Error severity (critical, high, medium, low, info).
        document_type: Type of document that failed validation.
    """
    m.record_validation_error(severity=severity, type=document_type)


def record_classification(document_type: str) -> None:
    """Record a document classification event.

    Args:
        document_type: Classified document type.
    """
    m.safe_inc(pdf_classification_total, 1, document_type=document_type)


def record_line_items_extracted(count: int) -> None:
    """Record the number of line items extracted.

    Args:
        count: Number of line items extracted.
    """
    m.safe_inc(pdf_line_items_extracted_total, count)


def record_batch_job(status: str) -> None:
    """Record a batch job event.

    Args:
        status: Batch job status (submitted, completed, failed, partial).
    """
    m.record_batch_job(status)


def update_active_jobs(delta: int) -> None:
    """Update the active jobs gauge.

    Args:
        delta: Positive to increment, negative to decrement.
    """
    m.update_active_jobs(delta)


def update_queue_size(size: int) -> None:
    """Set the current extraction queue size.

    Args:
        size: Current queue depth.
    """
    m.update_queue_size(size)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "m",
    # Agent-specific metric objects
    "pdf_pages_extracted_total",
    "pdf_fields_extracted_total",
    "pdf_extraction_confidence",
    "pdf_ocr_operations_total",
    "pdf_classification_total",
    "pdf_line_items_extracted_total",
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
