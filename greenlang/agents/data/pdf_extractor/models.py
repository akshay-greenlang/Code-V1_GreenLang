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

from pydantic import Field, field_validator

from greenlang.schemas import (
    GreenLangBase,
    GreenLangRecord,
    GreenLangRequest,
    ProvenanceMixin,
    TenantMixin,
    utcnow,
    new_uuid,
)

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

from greenlang.schemas.enums import JobStatus, ValidationSeverity
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

# _utcnow retained as local alias for backward compatibility
_utcnow = utcnow

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

# =============================================================================
# SDK Data Models
# =============================================================================

class DocumentRecord(GreenLangRecord):
    """Persistent record of an ingested document.

    Inherits from GreenLangRecord: created_at, updated_at, tenant_id,
    provenance_hash, model_config (extra=forbid, validate_default=True).

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
    """

    document_id: str = Field(
        default_factory=new_uuid,
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
        default_factory=utcnow,
        description="Timestamp when the document was uploaded",
    )

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, v: str) -> str:
        """Validate file_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("file_name must be non-empty")
        return v

class PageContent(GreenLangBase, ProvenanceMixin):
    """OCR-extracted content for a single page of a document.

    Inherits from GreenLangBase + ProvenanceMixin: provenance_hash,
    model_config (extra=forbid, validate_default=True).

    Attributes:
        page_id: Unique identifier for this page content record.
        document_id: Parent document identifier.
        page_number: Page number within the document (1-indexed).
        raw_text: Raw OCR-extracted text content.
        word_count: Number of words detected on the page.
        confidence: Overall OCR confidence score for the page.
        ocr_engine_used: OCR engine that processed this page.
        bounding_boxes: List of bounding box dictionaries for text regions.
    """

    page_id: str = Field(
        default_factory=new_uuid,
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

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: str) -> str:
        """Validate document_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("document_id must be non-empty")
        return v

class ExtractionJob(GreenLangBase, TenantMixin):
    """Record of a document extraction job execution.

    Inherits from GreenLangBase + TenantMixin: tenant_id,
    model_config (extra=forbid, validate_default=True).

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
    """

    job_id: str = Field(
        default_factory=new_uuid,
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
        default_factory=utcnow,
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

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: str) -> str:
        """Validate document_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("document_id must be non-empty")
        return v

class InvoiceExtraction(GreenLangRecord):
    """Structured extraction result for an invoice document.

    Inherits from GreenLangRecord: created_at, updated_at, tenant_id,
    provenance_hash, model_config (extra=forbid, validate_default=True).

    Attributes:
        extraction_id: Unique identifier for this extraction result.
        document_id: Source document identifier.
        invoice_data: Structured invoice data extracted from the document.
        overall_confidence: Aggregate confidence score across all fields.
        field_confidences: Per-field confidence scores.
        validation_passed: Whether all cross-field validations passed.
        validation_errors: List of validation error messages.
    """

    extraction_id: str = Field(
        default_factory=new_uuid,
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

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: str) -> str:
        """Validate document_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("document_id must be non-empty")
        return v

class ManifestExtraction(GreenLangRecord):
    """Structured extraction result for a shipping manifest document.

    Inherits from GreenLangRecord: created_at, updated_at, tenant_id,
    provenance_hash, model_config (extra=forbid, validate_default=True).

    Attributes:
        extraction_id: Unique identifier for this extraction result.
        document_id: Source document identifier.
        manifest_data: Structured manifest data extracted from the document.
        overall_confidence: Aggregate confidence score across all fields.
        field_confidences: Per-field confidence scores.
        validation_passed: Whether all cross-field validations passed.
    """

    extraction_id: str = Field(
        default_factory=new_uuid,
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

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: str) -> str:
        """Validate document_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("document_id must be non-empty")
        return v

class UtilityBillExtraction(GreenLangRecord):
    """Structured extraction result for a utility bill document.

    Inherits from GreenLangRecord: created_at, updated_at, tenant_id,
    provenance_hash, model_config (extra=forbid, validate_default=True).

    Attributes:
        extraction_id: Unique identifier for this extraction result.
        document_id: Source document identifier.
        utility_data: Structured utility bill data from the document.
        overall_confidence: Aggregate confidence score across all fields.
        field_confidences: Per-field confidence scores.
    """

    extraction_id: str = Field(
        default_factory=new_uuid,
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

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: str) -> str:
        """Validate document_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("document_id must be non-empty")
        return v

class ExtractionTemplate(GreenLangBase, TenantMixin):
    """Reusable template defining field extraction patterns and rules.

    Inherits from GreenLangBase + TenantMixin: tenant_id,
    model_config (extra=forbid, validate_default=True).

    Attributes:
        template_id: Unique identifier for this template.
        name: Human-readable template name.
        description: Detailed description of the template purpose.
        template_type: Type of document this template applies to.
        field_patterns: Mapping of field names to regex pattern lists.
        validation_rules: List of cross-field validation rule definitions.
        created_at: Timestamp when the template was created.
    """

    template_id: str = Field(
        default_factory=new_uuid,
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
        default_factory=utcnow,
        description="Timestamp when the template was created",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v

class ValidationResult(GreenLangBase):
    """Result of a single validation check on an extracted field.

    Inherits from GreenLangBase: model_config (extra=forbid, validate_default=True).

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
        default_factory=new_uuid,
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

class BatchJob(GreenLangBase, TenantMixin):
    """Record of a batch document processing job.

    Inherits from GreenLangBase + TenantMixin: tenant_id,
    model_config (extra=forbid, validate_default=True).

    Attributes:
        batch_id: Unique identifier for this batch job.
        document_ids: List of document IDs included in the batch.
        status: Current lifecycle status of the batch job.
        total_documents: Total number of documents in the batch.
        completed: Number of documents that completed successfully.
        failed: Number of documents that failed processing.
        started_at: Timestamp when the batch job started.
    """

    batch_id: str = Field(
        default_factory=new_uuid,
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
        default_factory=utcnow,
        description="Timestamp when the batch job started",
    )

class PDFStatistics(GreenLangBase):
    """Aggregated statistics for the PDF extractor service.

    Inherits from GreenLangBase: model_config (extra=forbid, validate_default=True).

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

# =============================================================================
# Request Models
# =============================================================================

class IngestDocumentRequest(GreenLangRequest, TenantMixin):
    """Request body for ingesting a single document.

    Inherits from GreenLangRequest + TenantMixin: tenant_id,
    model_config (extra=forbid, validate_default=True).

    Attributes:
        file_name: Name of the document file.
        file_path: Path or URL to the document.
        document_type: Expected document type (or unknown for auto-detect).
        ocr_engine: OCR engine to use for extraction.
        confidence_threshold: Minimum confidence for accepted fields.
        extract_line_items: Whether to extract line items.
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

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, v: str) -> str:
        """Validate file_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("file_name must be non-empty")
        return v

class BatchIngestRequest(GreenLangRequest, TenantMixin):
    """Request body for ingesting a batch of documents.

    Inherits from GreenLangRequest + TenantMixin: tenant_id,
    model_config (extra=forbid, validate_default=True).

    Attributes:
        documents: List of individual ingest requests.
        ocr_engine: Default OCR engine for the batch.
        confidence_threshold: Default confidence threshold for the batch.
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

    @field_validator("documents")
    @classmethod
    def validate_documents(cls, v: List[IngestDocumentRequest]) -> List[IngestDocumentRequest]:
        """Validate documents list is non-empty."""
        if not v:
            raise ValueError("documents list must be non-empty")
        return v

class ClassifyDocumentRequest(GreenLangRequest):
    """Request body for classifying a document type.

    Inherits from GreenLangRequest: model_config (extra=forbid, validate_default=True).

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

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: str) -> str:
        """Validate document_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("document_id must be non-empty")
        return v

class ExtractInvoiceRequest(GreenLangRequest):
    """Request body for extracting invoice data from a document.

    Inherits from GreenLangRequest: model_config (extra=forbid, validate_default=True).

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

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: str) -> str:
        """Validate document_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("document_id must be non-empty")
        return v

class CreateTemplateRequest(GreenLangRequest, TenantMixin):
    """Request body for creating a new extraction template.

    Inherits from GreenLangRequest + TenantMixin: tenant_id,
    model_config (extra=forbid, validate_default=True).

    Attributes:
        name: Human-readable template name.
        description: Detailed description of the template purpose.
        template_type: Type of document this template applies to.
        field_patterns: Mapping of field names to regex pattern lists.
        validation_rules: List of cross-field validation rule definitions.
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
