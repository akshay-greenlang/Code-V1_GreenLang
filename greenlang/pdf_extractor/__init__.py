# -*- coding: utf-8 -*-
"""
GL-DATA-X-001: GreenLang PDF & Invoice Extractor Service SDK
=============================================================

This package provides document parsing, OCR text extraction, field extraction,
invoice processing, manifest processing, document classification, cross-field
validation, and provenance tracking SDK for the GreenLang framework. It
supports:

- PDF, PNG, JPG, TIFF, and BMP document ingestion
- Multi-engine OCR (Tesseract, AWS Textract, Azure Vision, Google Vision)
- Pattern-based field extraction with confidence scoring
- Invoice data extraction (vendor, amount, date, line items)
- Shipping manifest / BOL extraction (weight, pieces, carrier)
- Utility bill extraction (consumption, meter readings, rates)
- Document type classification and routing
- Cross-field validation with configurable rules
- Batch document processing with parallel workers
- Extraction template management for custom layouts
- SHA-256 provenance chain tracking for complete audit trails
- 12 Prometheus metrics for observability
- FastAPI REST API with 20 endpoints
- Thread-safe configuration with GL_PDF_EXTRACTOR_ env prefix

Key Components:
    - config: PDFExtractorConfig with GL_PDF_EXTRACTOR_ env prefix
    - models: Pydantic v2 models for all data structures
    - document_parser: PDF/image document parsing engine
    - ocr_engine: Multi-engine OCR adapter with fallback
    - field_extractor: Pattern-based field extraction engine
    - invoice_processor: Invoice-specific extraction and validation
    - manifest_processor: Shipping manifest / BOL processing
    - document_classifier: Document type detection engine
    - validation_engine: Cross-field validation rule engine
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics
    - api: FastAPI HTTP service
    - setup: PDFExtractorService facade

Example:
    >>> from greenlang.pdf_extractor import PDFExtractorService
    >>> service = PDFExtractorService()
    >>> result = service.ingest_document("invoice.pdf")
    >>> print(result.extraction_status)
    success

Agent ID: GL-DATA-X-001
Agent Name: Document Ingestion & OCR Agent
"""

__version__ = "1.0.0"
__agent_id__ = "GL-DATA-X-001"
__agent_name__ = "Document Ingestion & OCR Agent"

# SDK availability flag
PDF_EXTRACTOR_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.pdf_extractor.config import (
    PDFExtractorConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Models (enums, Layer 1, SDK)
# ---------------------------------------------------------------------------
from greenlang.pdf_extractor.models import (
    # Layer 1 enumerations
    DocumentType,
    ExtractionStatus,
    OCREngine,
    # Layer 1 models
    BoundingBox,
    ExtractedField,
    LineItem,
    InvoiceData,
    ManifestData,
    UtilityBillData,
    DocumentIngestionInput,
    DocumentIngestionOutput,
    # New enumerations
    DocumentFormat,
    TemplateType,
    JobStatus,
    ValidationSeverity,
    # SDK models
    DocumentRecord,
    PageContent,
    ExtractionJob,
    InvoiceExtraction,
    ManifestExtraction,
    UtilityBillExtraction,
    ExtractionTemplate,
    ValidationResult,
    BatchJob,
    PDFStatistics,
    # Request models
    IngestDocumentRequest,
    BatchIngestRequest,
    ClassifyDocumentRequest,
    ExtractInvoiceRequest,
    CreateTemplateRequest,
)

# ---------------------------------------------------------------------------
# Core engines
# ---------------------------------------------------------------------------
from greenlang.pdf_extractor.document_parser import DocumentParser
from greenlang.pdf_extractor.ocr_engine import OCREngineAdapter
from greenlang.pdf_extractor.field_extractor import FieldExtractor
from greenlang.pdf_extractor.invoice_processor import InvoiceProcessor
from greenlang.pdf_extractor.manifest_processor import ManifestProcessor
from greenlang.pdf_extractor.document_classifier import DocumentClassifier
from greenlang.pdf_extractor.validation_engine import ValidationEngine
from greenlang.pdf_extractor.provenance import ProvenanceTracker

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from greenlang.pdf_extractor.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    pdf_documents_processed_total,
    pdf_processing_duration_seconds,
    pdf_pages_extracted_total,
    pdf_fields_extracted_total,
    pdf_extraction_confidence,
    pdf_ocr_operations_total,
    pdf_validation_errors_total,
    pdf_classification_total,
    pdf_line_items_extracted_total,
    pdf_batch_jobs_total,
    pdf_active_jobs,
    pdf_queue_size,
    # Helper functions
    record_document_processed,
    record_pages_extracted,
    record_fields_extracted,
    record_extraction_confidence,
    record_ocr_operation,
    record_validation_error,
    record_classification,
    record_line_items_extracted,
    record_batch_job,
    update_active_jobs,
    update_queue_size,
)

# ---------------------------------------------------------------------------
# Service setup facade
# ---------------------------------------------------------------------------
from greenlang.pdf_extractor.setup import (
    PDFExtractorService,
    configure_pdf_extractor,
    get_pdf_extractor,
    get_router,
)

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "PDF_EXTRACTOR_SDK_AVAILABLE",
    # Configuration
    "PDFExtractorConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Layer 1 enumerations
    "DocumentType",
    "ExtractionStatus",
    "OCREngine",
    # Layer 1 models
    "BoundingBox",
    "ExtractedField",
    "LineItem",
    "InvoiceData",
    "ManifestData",
    "UtilityBillData",
    "DocumentIngestionInput",
    "DocumentIngestionOutput",
    # New enumerations
    "DocumentFormat",
    "TemplateType",
    "JobStatus",
    "ValidationSeverity",
    # SDK models
    "DocumentRecord",
    "PageContent",
    "ExtractionJob",
    "InvoiceExtraction",
    "ManifestExtraction",
    "UtilityBillExtraction",
    "ExtractionTemplate",
    "ValidationResult",
    "BatchJob",
    "PDFStatistics",
    # Request models
    "IngestDocumentRequest",
    "BatchIngestRequest",
    "ClassifyDocumentRequest",
    "ExtractInvoiceRequest",
    "CreateTemplateRequest",
    # Core engines
    "DocumentParser",
    "OCREngineAdapter",
    "FieldExtractor",
    "InvoiceProcessor",
    "ManifestProcessor",
    "DocumentClassifier",
    "ValidationEngine",
    "ProvenanceTracker",
    # Metric objects
    "PROMETHEUS_AVAILABLE",
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
    # Metric helper functions
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
    # Service setup facade
    "PDFExtractorService",
    "configure_pdf_extractor",
    "get_pdf_extractor",
    "get_router",
]
