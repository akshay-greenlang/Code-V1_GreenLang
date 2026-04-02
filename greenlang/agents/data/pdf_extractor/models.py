# -*- coding: utf-8 -*-
"""
PDF & Invoice Extractor Service Data Models - AGENT-DATA-001: PDF Extractor

Pydantic v2 data models for the PDF & Invoice Extractor SDK. This module is
the **canonical** location for document ingestion enumerations and models.
The parent ``document_ingestion_agent.py`` re-exports from here.

Models:
    - Core enums: DocumentType, ExtractionStatus, OCREngine
    - Core models: BoundingBox, ExtractedField, LineItem, InvoiceData,
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
from datetime import datetime, date, timezone
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
# Core Enumerations (canonical definitions)
# ---------------------------------------------------------------------------

class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    INVOICE = "invoice"
    MANIFEST = "manifest"
    BILL_OF_LADING = "bill_of_lading"
    WEIGHT_TICKET = "weight_ticket"
    UTILITY_BILL = "utility_bill"
    RECEIPT = "receipt"
    PURCHASE_ORDER = "purchase_order"
    UNKNOWN = "unknown"


class ExtractionStatus(str, Enum):
    """Status of field extraction."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    LOW_CONFIDENCE = "low_confidence"


class OCREngine(str, Enum):
    """Supported OCR engines."""
    TESSERACT = "tesseract"
    AZURE_VISION = "azure_vision"
    AWS_TEXTRACT = "aws_textract"
    GOOGLE_VISION = "google_vision"
    SIMULATED = "simulated"


# ---------------------------------------------------------------------------
# Core Models (canonical definitions)
# ---------------------------------------------------------------------------

class BoundingBox(GreenLangBase):
    """Bounding box for extracted text region."""
    x: float = Field(..., description="X coordinate (left)")
    y: float = Field(..., description="Y coordinate (top)")
    width: float = Field(..., description="Box width")
    height: float = Field(..., description="Box height")
    page: int = Field(default=1, ge=1, description="Page number")


class ExtractedField(GreenLangBase):
    """A single extracted field with confidence score."""
    field_name: str = Field(..., description="Name of the field")
    value: Any = Field(..., description="Extracted value")
    raw_text: str = Field(..., description="Raw OCR text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")
    bounding_box: Optional[BoundingBox] = Field(None, description="Location in document")
    extraction_method: str = Field(default="pattern", description="Method used for extraction")
    validated: bool = Field(default=False, description="Whether value passed validation")


class LineItem(GreenLangBase):
    """A line item from an invoice or manifest."""
    line_number: int = Field(..., ge=1, description="Line number")
    description: str = Field(..., description="Item description")
    quantity: Optional[float] = Field(None, description="Quantity")
    unit: Optional[str] = Field(None, description="Unit of measure")
    unit_price: Optional[float] = Field(None, description="Unit price")
    total_price: Optional[float] = Field(None, description="Line total")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class InvoiceData(GreenLangBase):
    """Structured data extracted from an invoice."""
    invoice_number: Optional[str] = Field(None, description="Invoice number")
    invoice_date: Optional[date] = Field(None, description="Invoice date")
    due_date: Optional[date] = Field(None, description="Payment due date")
    vendor_name: Optional[str] = Field(None, description="Vendor/supplier name")
    vendor_address: Optional[str] = Field(None, description="Vendor address")
    vendor_tax_id: Optional[str] = Field(None, description="Vendor tax ID")
    buyer_name: Optional[str] = Field(None, description="Buyer/customer name")
    buyer_address: Optional[str] = Field(None, description="Buyer address")
    subtotal: Optional[float] = Field(None, description="Subtotal before tax")
    tax_amount: Optional[float] = Field(None, description="Tax amount")
    total_amount: Optional[float] = Field(None, description="Total invoice amount")
    currency: str = Field(default="USD", description="Currency code")
    line_items: List[LineItem] = Field(default_factory=list, description="Line items")
    payment_terms: Optional[str] = Field(None, description="Payment terms")


class ManifestData(GreenLangBase):
    """Structured data extracted from a shipping manifest."""
    manifest_number: Optional[str] = Field(None, description="Manifest/BOL number")
    shipment_date: Optional[date] = Field(None, description="Shipment date")
    carrier_name: Optional[str] = Field(None, description="Carrier name")
    origin: Optional[str] = Field(None, description="Origin location")
    destination: Optional[str] = Field(None, description="Destination location")
    shipper_name: Optional[str] = Field(None, description="Shipper name")
    consignee_name: Optional[str] = Field(None, description="Consignee name")
    total_weight: Optional[float] = Field(None, description="Total weight")
    weight_unit: str = Field(default="kg", description="Weight unit")
    total_pieces: Optional[int] = Field(None, description="Total pieces/packages")
    line_items: List[LineItem] = Field(default_factory=list, description="Line items")
    vehicle_id: Optional[str] = Field(None, description="Vehicle/trailer ID")
    seal_numbers: List[str] = Field(default_factory=list, description="Seal numbers")


class UtilityBillData(GreenLangBase):
    """Structured data extracted from a utility bill."""
    account_number: Optional[str] = Field(None, description="Account number")
    billing_period_start: Optional[date] = Field(None, description="Billing period start")
    billing_period_end: Optional[date] = Field(None, description="Billing period end")
    utility_type: Optional[str] = Field(None, description="Type: electricity, gas, water")
    meter_number: Optional[str] = Field(None, description="Meter number")
    previous_reading: Optional[float] = Field(None, description="Previous meter reading")
    current_reading: Optional[float] = Field(None, description="Current meter reading")
    consumption: Optional[float] = Field(None, description="Total consumption")
    consumption_unit: Optional[str] = Field(None, description="Consumption unit")
    rate: Optional[float] = Field(None, description="Rate per unit")
    total_amount: Optional[float] = Field(None, description="Total amount due")
    currency: str = Field(default="USD", description="Currency code")


class DocumentIngestionInput(GreenLangBase):
    """Input for document ingestion."""
    document_id: str = Field(..., description="Unique document identifier")
    file_path: Optional[str] = Field(None, description="Path to document file")
    file_content: Optional[bytes] = Field(None, description="Document content as bytes")
    file_base64: Optional[str] = Field(None, description="Document content as base64")
    document_type: DocumentType = Field(default=DocumentType.UNKNOWN)
    ocr_engine: OCREngine = Field(default=OCREngine.SIMULATED)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    extract_line_items: bool = Field(default=True)
    validate_totals: bool = Field(default=True)
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class DocumentIngestionOutput(GreenLangBase):
    """Output from document ingestion."""
    document_id: str = Field(..., description="Document identifier")
    document_type: DocumentType = Field(..., description="Detected/specified document type")
    page_count: int = Field(..., ge=1, description="Number of pages")
    extraction_status: ExtractionStatus = Field(..., description="Overall extraction status")
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    extracted_fields: List[ExtractedField] = Field(default_factory=list)
    structured_data: Dict[str, Any] = Field(default_factory=dict)
    raw_text: str = Field(default="", description="Full raw text")
    ocr_engine_used: OCREngine = Field(..., description="OCR engine used")
    processing_time_ms: float = Field(..., description="Processing duration")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")
    validation_errors: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Shared schema enums
# ---------------------------------------------------------------------------

from greenlang.schemas.enums import JobStatus, ValidationSeverity

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
    # Core enums (canonical definitions)
    "DocumentType",
    "ExtractionStatus",
    "OCREngine",
    # Core models (canonical definitions)
    "BoundingBox",
    "ExtractedField",
    "LineItem",
    "InvoiceData",
    "ManifestData",
    "UtilityBillData",
    "DocumentIngestionInput",
    "DocumentIngestionOutput",
    # Additional enums
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
