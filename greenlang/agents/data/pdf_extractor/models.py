# -*- coding: utf-8 -*-
"""
PDF & Invoice Extractor Service Data Models - AGENT-DATA-001: PDF Extractor

Pydantic v2 data models for the PDF & Invoice Extractor SDK. Re-exports the
Layer 1 enumerations and models from the foundation agent, and defines
additional SDK models for document records, page content, extraction jobs,
invoice/manifest/utility bill extractions, templates, validation results,
batch jobs, statistics, and request wrappers.

Models:
    - Re-exported enums: DocumentType, ExtractionStatus, OCREngine
    - Re-exported Layer 1: BoundingBox, ExtractedField, LineItem, InvoiceData,
        ManifestData, UtilityBillData, DocumentIngestionInput,
        DocumentIngestionOutput
    - New enums: DocumentFormat, TemplateType, JobStatus, ValidationSeverity
    - SDK models: DocumentRecord, PageContent, ExtractionJob,
        InvoiceExtraction, ManifestExtraction, UtilityBillExtraction,
        ExtractionTemplate, ValidationResult, BatchJob, PDFStatistics
    - Request models: IngestDocumentRequest, BatchIngestRequest,
        ClassifyDocumentRequest, ExtractInvoiceRequest, CreateTemplateRequest

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-001 PDF & Invoice Extractor
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Re-export Layer 1 enumerations
# ---------------------------------------------------------------------------

from greenlang.agents.data.document_ingestion_agent import (
    DocumentType,
    ExtractionStatus,
    OCREngine,
)

# ---------------------------------------------------------------------------
# Re-export Layer 1 models
# ---------------------------------------------------------------------------

from greenlang.agents.data.document_ingestion_agent import (
    BoundingBox,
    ExtractedField,
    LineItem,
    InvoiceData,
    ManifestData,
    UtilityBillData,
    DocumentIngestionInput,
    DocumentIngestionOutput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# =============================================================================
# New Enumerations
# =============================================================================


class DocumentFormat(str, Enum):
    """Supported document file formats for ingestion."""

    PDF = "pdf"
    PNG = "png"
    JPG = "jpg"
    TIFF = "tiff"
    BMP = "bmp"


class TemplateType(str, Enum):
    """Types of extraction templates for field pattern matching."""

    INVOICE = "invoice"
    MANIFEST = "manifest"
    UTILITY_BILL = "utility_bill"
    RECEIPT = "receipt"
    PURCHASE_ORDER = "purchase_order"
    WEIGHT_TICKET = "weight_ticket"
    CUSTOM = "custom"


class JobStatus(str, Enum):
    """Lifecycle status of an extraction or batch job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ValidationSeverity(str, Enum):
    """Severity level for validation results."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# =============================================================================
# SDK Data Models
# =============================================================================


class DocumentRecord(BaseModel):
    """Persistent record of an ingested document.

    Captures metadata about a document that has been uploaded and registered
    in the extraction system for processing and audit purposes.

    Attributes:
        document_id: Unique identifier for this document record.
        file_name: Original file name of the uploaded document.
        file_path: Storage path (local or S3) for the document.
        file_size_bytes: Size of the document file in bytes.
        file_hash: SHA-256 hash of the raw file content for deduplication.
        document_type: Classified or specified document type.
        document_format: File format of the document.
        page_count: Number of pages in the document.
        upload_timestamp: Timestamp when the document was uploaded.
        tenant_id: Tenant identifier for multi-tenant isolation.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    document_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this document record",
    )
    file_name: str = Field(
        ..., description="Original file name of the uploaded document",
    )
    file_path: str = Field(
        default="", description="Storage path for the document",
    )
    file_size_bytes: int = Field(
        default=0, ge=0, description="Size of the document file in bytes",
    )
    file_hash: str = Field(
        default="", description="SHA-256 hash of the raw file content",
    )
    document_type: DocumentType = Field(
        default=DocumentType.UNKNOWN,
        description="Classified or specified document type",
    )
    document_format: DocumentFormat = Field(
        default=DocumentFormat.PDF,
        description="File format of the document",
    )
    page_count: int = Field(
        default=1, ge=1, description="Number of pages in the document",
    )
    upload_timestamp: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the document was uploaded",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, v: str) -> str:
        """Validate file_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("file_name must be non-empty")
        return v


class PageContent(BaseModel):
    """OCR-extracted content for a single page of a document.

    Captures the raw text, word count, confidence, and bounding box
    information for a single page after OCR processing.

    Attributes:
        page_id: Unique identifier for this page content record.
        document_id: Parent document identifier.
        page_number: Page number within the document (1-indexed).
        raw_text: Raw OCR-extracted text content.
        word_count: Number of words detected on the page.
        confidence: Overall OCR confidence score for the page.
        ocr_engine_used: OCR engine that processed this page.
        bounding_boxes: List of bounding box dictionaries for text regions.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    page_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this page content record",
    )
    document_id: str = Field(
        ..., description="Parent document identifier",
    )
    page_number: int = Field(
        ..., ge=1, description="Page number within the document",
    )
    raw_text: str = Field(
        default="", description="Raw OCR-extracted text content",
    )
    word_count: int = Field(
        default=0, ge=0, description="Number of words detected on the page",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Overall OCR confidence score for the page",
    )
    ocr_engine_used: OCREngine = Field(
        default=OCREngine.SIMULATED,
        description="OCR engine that processed this page",
    )
    bounding_boxes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of bounding box dictionaries for text regions",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: str) -> str:
        """Validate document_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("document_id must be non-empty")
        return v


class ExtractionJob(BaseModel):
    """Record of a document extraction job execution.

    Tracks the lifecycle, configuration, and results of a single
    extraction job for monitoring and audit purposes.

    Attributes:
        job_id: Unique identifier for this extraction job.
        document_id: Document being processed.
        job_status: Current lifecycle status of the job.
        document_type: Document type used for extraction patterns.
        ocr_engine: OCR engine used for text extraction.
        confidence_threshold: Minimum confidence threshold applied.
        started_at: Timestamp when the job started processing.
        completed_at: Timestamp when the job finished (if completed).
        duration_ms: Total processing duration in milliseconds.
        error_message: Error description if the job failed.
        fields_extracted: Number of fields successfully extracted.
        pages_processed: Number of pages processed by OCR.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    job_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this extraction job",
    )
    document_id: str = Field(
        ..., description="Document being processed",
    )
    job_status: JobStatus = Field(
        default=JobStatus.PENDING,
        description="Current lifecycle status of the job",
    )
    document_type: DocumentType = Field(
        default=DocumentType.UNKNOWN,
        description="Document type used for extraction patterns",
    )
    ocr_engine: OCREngine = Field(
        default=OCREngine.SIMULATED,
        description="OCR engine used for text extraction",
    )
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Minimum confidence threshold applied",
    )
    started_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the job started processing",
    )
    completed_at: Optional[datetime] = Field(
        None, description="Timestamp when the job finished",
    )
    duration_ms: float = Field(
        default=0.0, ge=0.0,
        description="Total processing duration in milliseconds",
    )
    error_message: Optional[str] = Field(
        None, description="Error description if the job failed",
    )
    fields_extracted: int = Field(
        default=0, ge=0,
        description="Number of fields successfully extracted",
    )
    pages_processed: int = Field(
        default=0, ge=0,
        description="Number of pages processed by OCR",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: str) -> str:
        """Validate document_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("document_id must be non-empty")
        return v


class InvoiceExtraction(BaseModel):
    """Structured extraction result for an invoice document.

    Contains the full invoice data, per-field confidence scores,
    validation status, and provenance information.

    Attributes:
        extraction_id: Unique identifier for this extraction result.
        document_id: Source document identifier.
        invoice_data: Structured invoice data extracted from the document.
        overall_confidence: Aggregate confidence score across all fields.
        field_confidences: Per-field confidence scores.
        validation_passed: Whether all cross-field validations passed.
        validation_errors: List of validation error messages.
        tenant_id: Tenant identifier for multi-tenant isolation.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    extraction_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this extraction result",
    )
    document_id: str = Field(
        ..., description="Source document identifier",
    )
    invoice_data: InvoiceData = Field(
        default_factory=InvoiceData,
        description="Structured invoice data extracted from the document",
    )
    overall_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Aggregate confidence score across all fields",
    )
    field_confidences: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-field confidence scores",
    )
    validation_passed: bool = Field(
        default=False,
        description="Whether all cross-field validations passed",
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="List of validation error messages",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: str) -> str:
        """Validate document_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("document_id must be non-empty")
        return v


class ManifestExtraction(BaseModel):
    """Structured extraction result for a shipping manifest document.

    Contains the full manifest data, per-field confidence scores,
    validation status, and provenance information.

    Attributes:
        extraction_id: Unique identifier for this extraction result.
        document_id: Source document identifier.
        manifest_data: Structured manifest data extracted from the document.
        overall_confidence: Aggregate confidence score across all fields.
        field_confidences: Per-field confidence scores.
        validation_passed: Whether all cross-field validations passed.
        tenant_id: Tenant identifier for multi-tenant isolation.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    extraction_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this extraction result",
    )
    document_id: str = Field(
        ..., description="Source document identifier",
    )
    manifest_data: ManifestData = Field(
        default_factory=ManifestData,
        description="Structured manifest data extracted from the document",
    )
    overall_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Aggregate confidence score across all fields",
    )
    field_confidences: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-field confidence scores",
    )
    validation_passed: bool = Field(
        default=False,
        description="Whether all cross-field validations passed",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: str) -> str:
        """Validate document_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("document_id must be non-empty")
        return v


class UtilityBillExtraction(BaseModel):
    """Structured extraction result for a utility bill document.

    Contains the full utility bill data, per-field confidence scores,
    and provenance information.

    Attributes:
        extraction_id: Unique identifier for this extraction result.
        document_id: Source document identifier.
        utility_data: Structured utility bill data from the document.
        overall_confidence: Aggregate confidence score across all fields.
        field_confidences: Per-field confidence scores.
        tenant_id: Tenant identifier for multi-tenant isolation.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    extraction_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this extraction result",
    )
    document_id: str = Field(
        ..., description="Source document identifier",
    )
    utility_data: UtilityBillData = Field(
        default_factory=UtilityBillData,
        description="Structured utility bill data from the document",
    )
    overall_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Aggregate confidence score across all fields",
    )
    field_confidences: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-field confidence scores",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: str) -> str:
        """Validate document_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("document_id must be non-empty")
        return v


class ExtractionTemplate(BaseModel):
    """Reusable template defining field extraction patterns and rules.

    Templates allow users to define custom extraction patterns for
    specific document layouts, enabling accurate field extraction
    without retraining OCR models.

    Attributes:
        template_id: Unique identifier for this template.
        name: Human-readable template name.
        description: Detailed description of the template purpose.
        template_type: Type of document this template applies to.
        field_patterns: Mapping of field names to regex pattern lists.
        validation_rules: List of cross-field validation rule definitions.
        created_at: Timestamp when the template was created.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    template_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this template",
    )
    name: str = Field(
        ..., description="Human-readable template name",
    )
    description: str = Field(
        default="",
        description="Detailed description of the template purpose",
    )
    template_type: TemplateType = Field(
        default=TemplateType.CUSTOM,
        description="Type of document this template applies to",
    )
    field_patterns: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Mapping of field names to regex pattern lists",
    )
    validation_rules: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of cross-field validation rule definitions",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the template was created",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v


class ValidationResult(BaseModel):
    """Result of a single validation check on an extracted field.

    Captures the validation rule, severity, expected vs actual values,
    and descriptive message for audit and debugging.

    Attributes:
        result_id: Unique identifier for this validation result.
        document_id: Document that was validated.
        field_name: Name of the field that was validated.
        severity: Severity level of the validation result.
        message: Human-readable validation message.
        expected_value: Expected value (if applicable).
        actual_value: Actual extracted value.
        rule_name: Name of the validation rule that produced this result.
    """

    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this validation result",
    )
    document_id: str = Field(
        ..., description="Document that was validated",
    )
    field_name: str = Field(
        ..., description="Name of the field that was validated",
    )
    severity: ValidationSeverity = Field(
        default=ValidationSeverity.INFO,
        description="Severity level of the validation result",
    )
    message: str = Field(
        default="", description="Human-readable validation message",
    )
    expected_value: Optional[str] = Field(
        None, description="Expected value (if applicable)",
    )
    actual_value: Optional[str] = Field(
        None, description="Actual extracted value",
    )
    rule_name: str = Field(
        default="", description="Name of the validation rule",
    )

    model_config = {"extra": "forbid"}

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: str) -> str:
        """Validate document_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("document_id must be non-empty")
        return v

    @field_validator("field_name")
    @classmethod
    def validate_field_name(cls, v: str) -> str:
        """Validate field_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("field_name must be non-empty")
        return v


class BatchJob(BaseModel):
    """Record of a batch document processing job.

    Tracks the overall status and progress of a batch of documents
    being processed together.

    Attributes:
        batch_id: Unique identifier for this batch job.
        document_ids: List of document IDs included in the batch.
        status: Current lifecycle status of the batch job.
        total_documents: Total number of documents in the batch.
        completed: Number of documents that completed successfully.
        failed: Number of documents that failed processing.
        started_at: Timestamp when the batch job started.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this batch job",
    )
    document_ids: List[str] = Field(
        default_factory=list,
        description="List of document IDs included in the batch",
    )
    status: JobStatus = Field(
        default=JobStatus.PENDING,
        description="Current lifecycle status of the batch job",
    )
    total_documents: int = Field(
        default=0, ge=0,
        description="Total number of documents in the batch",
    )
    completed: int = Field(
        default=0, ge=0,
        description="Number of documents that completed successfully",
    )
    failed: int = Field(
        default=0, ge=0,
        description="Number of documents that failed processing",
    )
    started_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the batch job started",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}


class PDFStatistics(BaseModel):
    """Aggregated statistics for the PDF extractor service.

    Attributes:
        total_documents: Total number of documents processed.
        total_pages: Total number of pages processed across all documents.
        total_fields: Total number of fields extracted.
        total_invoices: Total number of invoice extractions completed.
        total_manifests: Total number of manifest extractions completed.
        total_utility_bills: Total number of utility bill extractions.
        avg_confidence: Average extraction confidence across all fields.
        validation_pass_rate: Percentage of documents passing validation.
    """

    total_documents: int = Field(
        default=0, ge=0,
        description="Total number of documents processed",
    )
    total_pages: int = Field(
        default=0, ge=0,
        description="Total number of pages processed",
    )
    total_fields: int = Field(
        default=0, ge=0,
        description="Total number of fields extracted",
    )
    total_invoices: int = Field(
        default=0, ge=0,
        description="Total number of invoice extractions completed",
    )
    total_manifests: int = Field(
        default=0, ge=0,
        description="Total number of manifest extractions completed",
    )
    total_utility_bills: int = Field(
        default=0, ge=0,
        description="Total number of utility bill extractions",
    )
    avg_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Average extraction confidence across all fields",
    )
    validation_pass_rate: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage of documents passing validation",
    )

    model_config = {"extra": "forbid"}


# =============================================================================
# Request Models
# =============================================================================


class IngestDocumentRequest(BaseModel):
    """Request body for ingesting a single document.

    Attributes:
        file_name: Name of the document file.
        file_path: Path or URL to the document.
        document_type: Expected document type (or unknown for auto-detect).
        ocr_engine: OCR engine to use for extraction.
        confidence_threshold: Minimum confidence for accepted fields.
        extract_line_items: Whether to extract line items.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    file_name: str = Field(
        ..., description="Name of the document file",
    )
    file_path: str = Field(
        default="", description="Path or URL to the document",
    )
    document_type: DocumentType = Field(
        default=DocumentType.UNKNOWN,
        description="Expected document type",
    )
    ocr_engine: OCREngine = Field(
        default=OCREngine.SIMULATED,
        description="OCR engine to use for extraction",
    )
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Minimum confidence for accepted fields",
    )
    extract_line_items: bool = Field(
        default=True,
        description="Whether to extract line items",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, v: str) -> str:
        """Validate file_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("file_name must be non-empty")
        return v


class BatchIngestRequest(BaseModel):
    """Request body for ingesting a batch of documents.

    Attributes:
        documents: List of individual ingest requests.
        ocr_engine: Default OCR engine for the batch.
        confidence_threshold: Default confidence threshold for the batch.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    documents: List[IngestDocumentRequest] = Field(
        ..., description="List of individual ingest requests",
    )
    ocr_engine: OCREngine = Field(
        default=OCREngine.SIMULATED,
        description="Default OCR engine for the batch",
    )
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Default confidence threshold for the batch",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("documents")
    @classmethod
    def validate_documents(cls, v: List[IngestDocumentRequest]) -> List[IngestDocumentRequest]:
        """Validate documents list is non-empty."""
        if not v:
            raise ValueError("documents list must be non-empty")
        return v


class ClassifyDocumentRequest(BaseModel):
    """Request body for classifying a document type.

    Attributes:
        document_id: Identifier of the document to classify.
        text: Raw text content to classify (if available).
        file_path: Path to the document file (if text not provided).
    """

    document_id: str = Field(
        ..., description="Identifier of the document to classify",
    )
    text: Optional[str] = Field(
        None, description="Raw text content to classify",
    )
    file_path: Optional[str] = Field(
        None, description="Path to the document file",
    )

    model_config = {"extra": "forbid"}

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: str) -> str:
        """Validate document_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("document_id must be non-empty")
        return v


class ExtractInvoiceRequest(BaseModel):
    """Request body for extracting invoice data from a document.

    Attributes:
        document_id: Identifier of the document to extract from.
        file_path: Path to the invoice document.
        ocr_engine: OCR engine to use for extraction.
        confidence_threshold: Minimum confidence for accepted fields.
        validate_totals: Whether to validate invoice total consistency.
        template_id: Optional template ID for custom extraction patterns.
    """

    document_id: str = Field(
        ..., description="Identifier of the document to extract from",
    )
    file_path: str = Field(
        default="", description="Path to the invoice document",
    )
    ocr_engine: OCREngine = Field(
        default=OCREngine.SIMULATED,
        description="OCR engine to use for extraction",
    )
    confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Minimum confidence for accepted fields",
    )
    validate_totals: bool = Field(
        default=True,
        description="Whether to validate invoice total consistency",
    )
    template_id: Optional[str] = Field(
        None, description="Optional template ID for custom extraction",
    )

    model_config = {"extra": "forbid"}

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: str) -> str:
        """Validate document_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("document_id must be non-empty")
        return v


class CreateTemplateRequest(BaseModel):
    """Request body for creating a new extraction template.

    Attributes:
        name: Human-readable template name.
        description: Detailed description of the template purpose.
        template_type: Type of document this template applies to.
        field_patterns: Mapping of field names to regex pattern lists.
        validation_rules: List of cross-field validation rule definitions.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    name: str = Field(
        ..., description="Human-readable template name",
    )
    description: str = Field(
        default="",
        description="Detailed description of the template purpose",
    )
    template_type: TemplateType = Field(
        default=TemplateType.CUSTOM,
        description="Type of document this template applies to",
    )
    field_patterns: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Mapping of field names to regex pattern lists",
    )
    validation_rules: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of cross-field validation rule definitions",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v


__all__ = [
    # Re-exported enums
    "DocumentType",
    "ExtractionStatus",
    "OCREngine",
    # Re-exported Layer 1 models
    "BoundingBox",
    "ExtractedField",
    "LineItem",
    "InvoiceData",
    "ManifestData",
    "UtilityBillData",
    "DocumentIngestionInput",
    "DocumentIngestionOutput",
    # New enums
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
]
