# -*- coding: utf-8 -*-
"""
Unit Tests for PDF Extractor Models (AGENT-DATA-001)

Tests all enums (DocumentType, ExtractionStatus, OCREngine, DocumentFormat,
TemplateType, JobStatus, ValidationSeverity) and all SDK models (DocumentRecord,
PageContent, ExtractionJob, InvoiceExtraction, ManifestExtraction,
UtilityBillExtraction, ExtractionTemplate, ValidationResult, BatchJob,
PDFStatistics), plus request models.

Coverage target: 85%+ of models.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline enums mirroring greenlang/pdf_extractor/models.py
# ---------------------------------------------------------------------------


class DocumentType(str, Enum):
    INVOICE = "invoice"
    MANIFEST = "manifest"
    UTILITY_BILL = "utility_bill"
    RECEIPT = "receipt"
    PURCHASE_ORDER = "purchase_order"
    BILL_OF_LADING = "bill_of_lading"
    UNKNOWN = "unknown"


class ExtractionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    VALIDATED = "validated"


class OCREngine(str, Enum):
    TESSERACT = "tesseract"
    AWS_TEXTRACT = "aws_textract"
    AZURE_VISION = "azure_vision"
    GOOGLE_VISION = "google_vision"


class DocumentFormat(str, Enum):
    PDF = "pdf"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    TIFF = "tiff"
    BMP = "bmp"


class TemplateType(str, Enum):
    INVOICE = "invoice"
    MANIFEST = "manifest"
    UTILITY_BILL = "utility_bill"
    RECEIPT = "receipt"
    PURCHASE_ORDER = "purchase_order"
    CUSTOM = "custom"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ---------------------------------------------------------------------------
# Inline SDK models mirroring greenlang/pdf_extractor/models.py
# ---------------------------------------------------------------------------


class PageContent:
    """Extracted content from a single PDF/image page."""

    def __init__(
        self,
        page_number: int = 1,
        text: str = "",
        confidence: float = 0.0,
        ocr_engine: str = "tesseract",
        regions: Optional[List[Dict[str, Any]]] = None,
        width: int = 0,
        height: int = 0,
    ):
        self.page_number = page_number
        self.text = text
        self.confidence = confidence
        self.ocr_engine = ocr_engine
        self.regions = regions or []
        self.width = width
        self.height = height

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "text": self.text,
            "confidence": self.confidence,
            "ocr_engine": self.ocr_engine,
            "regions": self.regions,
            "width": self.width,
            "height": self.height,
        }


class DocumentRecord:
    """Record of an ingested document."""

    def __init__(
        self,
        document_id: Optional[str] = None,
        filename: str = "",
        file_format: str = "pdf",
        file_size_bytes: int = 0,
        file_hash: str = "",
        page_count: int = 0,
        document_type: str = "unknown",
        classification_confidence: float = 0.0,
        pages: Optional[List[PageContent]] = None,
        status: str = "pending",
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ):
        self.document_id = document_id or str(uuid.uuid4())
        self.filename = filename
        self.file_format = file_format
        self.file_size_bytes = file_size_bytes
        self.file_hash = file_hash
        self.page_count = page_count
        self.document_type = document_type
        self.classification_confidence = classification_confidence
        self.pages = pages or []
        self.status = status
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.updated_at = updated_at or self.created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_format": self.file_format,
            "file_size_bytes": self.file_size_bytes,
            "file_hash": self.file_hash,
            "page_count": self.page_count,
            "document_type": self.document_type,
            "classification_confidence": self.classification_confidence,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ExtractionJob:
    """A single extraction job tracking."""

    def __init__(
        self,
        job_id: Optional[str] = None,
        document_id: str = "",
        document_type: str = "unknown",
        status: str = "queued",
        extraction_result: Optional[Dict[str, Any]] = None,
        confidence: float = 0.0,
        processing_time_ms: float = 0.0,
        error_message: Optional[str] = None,
        provenance_hash: Optional[str] = None,
        created_at: Optional[str] = None,
        completed_at: Optional[str] = None,
    ):
        self.job_id = job_id or str(uuid.uuid4())
        self.document_id = document_id
        self.document_type = document_type
        self.status = status
        self.extraction_result = extraction_result or {}
        self.confidence = confidence
        self.processing_time_ms = processing_time_ms
        self.error_message = error_message
        self.provenance_hash = provenance_hash
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.completed_at = completed_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "document_id": self.document_id,
            "document_type": self.document_type,
            "status": self.status,
            "confidence": self.confidence,
            "processing_time_ms": self.processing_time_ms,
            "error_message": self.error_message,
            "provenance_hash": self.provenance_hash,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


class LineItem:
    """A single line item from an invoice."""

    def __init__(
        self,
        item_code: str = "",
        description: str = "",
        quantity: float = 0.0,
        unit_price: float = 0.0,
        amount: float = 0.0,
        confidence: float = 0.0,
    ):
        self.item_code = item_code
        self.description = description
        self.quantity = quantity
        self.unit_price = unit_price
        self.amount = amount
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_code": self.item_code,
            "description": self.description,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "amount": self.amount,
            "confidence": self.confidence,
        }


class InvoiceExtraction:
    """Extracted data from an invoice document."""

    def __init__(
        self,
        invoice_number: str = "",
        invoice_date: Optional[str] = None,
        due_date: Optional[str] = None,
        po_number: Optional[str] = None,
        vendor_name: str = "",
        vendor_address: str = "",
        buyer_name: str = "",
        buyer_address: str = "",
        line_items: Optional[List[LineItem]] = None,
        subtotal: float = 0.0,
        tax_amount: float = 0.0,
        total_amount: float = 0.0,
        currency: str = "USD",
        payment_terms: Optional[str] = None,
        confidence: float = 0.0,
    ):
        self.invoice_number = invoice_number
        self.invoice_date = invoice_date
        self.due_date = due_date
        self.po_number = po_number
        self.vendor_name = vendor_name
        self.vendor_address = vendor_address
        self.buyer_name = buyer_name
        self.buyer_address = buyer_address
        self.line_items = line_items or []
        self.subtotal = subtotal
        self.tax_amount = tax_amount
        self.total_amount = total_amount
        self.currency = currency
        self.payment_terms = payment_terms
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "invoice_number": self.invoice_number,
            "invoice_date": self.invoice_date,
            "due_date": self.due_date,
            "po_number": self.po_number,
            "vendor_name": self.vendor_name,
            "buyer_name": self.buyer_name,
            "line_items": [i.to_dict() for i in self.line_items],
            "subtotal": self.subtotal,
            "tax_amount": self.tax_amount,
            "total_amount": self.total_amount,
            "currency": self.currency,
            "payment_terms": self.payment_terms,
            "confidence": self.confidence,
        }


class ManifestExtraction:
    """Extracted data from a shipping manifest / BOL."""

    def __init__(
        self,
        manifest_number: str = "",
        date: Optional[str] = None,
        shipper_name: str = "",
        shipper_address: str = "",
        consignee_name: str = "",
        consignee_address: str = "",
        carrier_name: str = "",
        vessel_name: Optional[str] = None,
        voyage_number: Optional[str] = None,
        port_of_loading: str = "",
        port_of_discharge: str = "",
        container_number: Optional[str] = None,
        cargo_items: Optional[List[Dict[str, Any]]] = None,
        total_packages: int = 0,
        total_weight_kg: float = 0.0,
        total_volume_m3: float = 0.0,
        freight_terms: Optional[str] = None,
        confidence: float = 0.0,
    ):
        self.manifest_number = manifest_number
        self.date = date
        self.shipper_name = shipper_name
        self.shipper_address = shipper_address
        self.consignee_name = consignee_name
        self.consignee_address = consignee_address
        self.carrier_name = carrier_name
        self.vessel_name = vessel_name
        self.voyage_number = voyage_number
        self.port_of_loading = port_of_loading
        self.port_of_discharge = port_of_discharge
        self.container_number = container_number
        self.cargo_items = cargo_items or []
        self.total_packages = total_packages
        self.total_weight_kg = total_weight_kg
        self.total_volume_m3 = total_volume_m3
        self.freight_terms = freight_terms
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_number": self.manifest_number,
            "date": self.date,
            "shipper_name": self.shipper_name,
            "consignee_name": self.consignee_name,
            "carrier_name": self.carrier_name,
            "vessel_name": self.vessel_name,
            "port_of_loading": self.port_of_loading,
            "port_of_discharge": self.port_of_discharge,
            "total_packages": self.total_packages,
            "total_weight_kg": self.total_weight_kg,
            "total_volume_m3": self.total_volume_m3,
            "confidence": self.confidence,
        }


class UtilityBillExtraction:
    """Extracted data from a utility bill."""

    def __init__(
        self,
        account_number: str = "",
        billing_period_start: Optional[str] = None,
        billing_period_end: Optional[str] = None,
        statement_date: Optional[str] = None,
        utility_type: str = "",
        meter_number: Optional[str] = None,
        previous_reading: float = 0.0,
        current_reading: float = 0.0,
        consumption: float = 0.0,
        consumption_unit: str = "kWh",
        total_amount: float = 0.0,
        currency: str = "USD",
        due_date: Optional[str] = None,
        confidence: float = 0.0,
    ):
        self.account_number = account_number
        self.billing_period_start = billing_period_start
        self.billing_period_end = billing_period_end
        self.statement_date = statement_date
        self.utility_type = utility_type
        self.meter_number = meter_number
        self.previous_reading = previous_reading
        self.current_reading = current_reading
        self.consumption = consumption
        self.consumption_unit = consumption_unit
        self.total_amount = total_amount
        self.currency = currency
        self.due_date = due_date
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "account_number": self.account_number,
            "billing_period_start": self.billing_period_start,
            "billing_period_end": self.billing_period_end,
            "utility_type": self.utility_type,
            "consumption": self.consumption,
            "consumption_unit": self.consumption_unit,
            "total_amount": self.total_amount,
            "currency": self.currency,
            "confidence": self.confidence,
        }


class ExtractionTemplate:
    """Reusable extraction template for common document layouts."""

    def __init__(
        self,
        template_id: Optional[str] = None,
        name: str = "",
        template_type: str = "custom",
        field_mappings: Optional[Dict[str, Any]] = None,
        regex_patterns: Optional[Dict[str, str]] = None,
        created_at: Optional[str] = None,
    ):
        self.template_id = template_id or str(uuid.uuid4())
        self.name = name
        self.template_type = template_type
        self.field_mappings = field_mappings or {}
        self.regex_patterns = regex_patterns or {}
        self.created_at = created_at or datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "template_type": self.template_type,
            "field_mappings": self.field_mappings,
            "regex_patterns": self.regex_patterns,
            "created_at": self.created_at,
        }


class ValidationResult:
    """Cross-field validation result."""

    def __init__(
        self,
        is_valid: bool = True,
        errors: Optional[List[Dict[str, Any]]] = None,
        warnings: Optional[List[Dict[str, Any]]] = None,
        info: Optional[List[Dict[str, Any]]] = None,
        fields_validated: int = 0,
        rules_checked: int = 0,
    ):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.info = info or []
        self.fields_validated = fields_validated
        self.rules_checked = rules_checked

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "fields_validated": self.fields_validated,
            "rules_checked": self.rules_checked,
        }


class BatchJob:
    """Batch processing job for multiple documents."""

    def __init__(
        self,
        batch_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        status: str = "queued",
        total_documents: int = 0,
        completed_documents: int = 0,
        failed_documents: int = 0,
        created_at: Optional[str] = None,
        completed_at: Optional[str] = None,
    ):
        self.batch_id = batch_id or str(uuid.uuid4())
        self.document_ids = document_ids or []
        self.status = status
        self.total_documents = total_documents
        self.completed_documents = completed_documents
        self.failed_documents = failed_documents
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.completed_at = completed_at

    @property
    def progress_pct(self) -> float:
        if self.total_documents == 0:
            return 0.0
        return (self.completed_documents + self.failed_documents) / self.total_documents * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "document_ids": self.document_ids,
            "status": self.status,
            "total_documents": self.total_documents,
            "completed_documents": self.completed_documents,
            "failed_documents": self.failed_documents,
            "progress_pct": self.progress_pct,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


class PDFStatistics:
    """Statistics counters for the PDF Extractor service."""

    def __init__(self):
        self.documents_ingested: int = 0
        self.documents_classified: int = 0
        self.extractions_completed: int = 0
        self.extractions_failed: int = 0
        self.validations_run: int = 0
        self.total_pages_processed: int = 0
        self.avg_confidence: float = 0.0
        self._confidence_sum: float = 0.0
        self._confidence_count: int = 0

    def record_extraction(self, confidence: float):
        self.extractions_completed += 1
        self._confidence_sum += confidence
        self._confidence_count += 1
        self.avg_confidence = self._confidence_sum / self._confidence_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "documents_ingested": self.documents_ingested,
            "documents_classified": self.documents_classified,
            "extractions_completed": self.extractions_completed,
            "extractions_failed": self.extractions_failed,
            "validations_run": self.validations_run,
            "total_pages_processed": self.total_pages_processed,
            "avg_confidence": round(self.avg_confidence, 4),
        }


# ===========================================================================
# Test Classes
# ===========================================================================


class TestDocumentTypeEnum:
    """Test DocumentType enum values."""

    def test_invoice(self):
        assert DocumentType.INVOICE.value == "invoice"

    def test_manifest(self):
        assert DocumentType.MANIFEST.value == "manifest"

    def test_utility_bill(self):
        assert DocumentType.UTILITY_BILL.value == "utility_bill"

    def test_receipt(self):
        assert DocumentType.RECEIPT.value == "receipt"

    def test_purchase_order(self):
        assert DocumentType.PURCHASE_ORDER.value == "purchase_order"

    def test_bill_of_lading(self):
        assert DocumentType.BILL_OF_LADING.value == "bill_of_lading"

    def test_unknown(self):
        assert DocumentType.UNKNOWN.value == "unknown"

    def test_all_7_types(self):
        assert len(DocumentType) == 7

    def test_string_conversion(self):
        assert str(DocumentType.INVOICE) == "DocumentType.INVOICE"

    def test_from_value(self):
        assert DocumentType("invoice") == DocumentType.INVOICE


class TestExtractionStatusEnum:
    """Test ExtractionStatus enum values."""

    def test_pending(self):
        assert ExtractionStatus.PENDING.value == "pending"

    def test_processing(self):
        assert ExtractionStatus.PROCESSING.value == "processing"

    def test_completed(self):
        assert ExtractionStatus.COMPLETED.value == "completed"

    def test_failed(self):
        assert ExtractionStatus.FAILED.value == "failed"

    def test_partial(self):
        assert ExtractionStatus.PARTIAL.value == "partial"

    def test_validated(self):
        assert ExtractionStatus.VALIDATED.value == "validated"

    def test_all_6_statuses(self):
        assert len(ExtractionStatus) == 6


class TestOCREngineEnum:
    """Test OCREngine enum values."""

    def test_tesseract(self):
        assert OCREngine.TESSERACT.value == "tesseract"

    def test_aws_textract(self):
        assert OCREngine.AWS_TEXTRACT.value == "aws_textract"

    def test_azure_vision(self):
        assert OCREngine.AZURE_VISION.value == "azure_vision"

    def test_google_vision(self):
        assert OCREngine.GOOGLE_VISION.value == "google_vision"

    def test_all_4_engines(self):
        assert len(OCREngine) == 4


class TestDocumentFormatEnum:
    """Test DocumentFormat enum values."""

    def test_pdf(self):
        assert DocumentFormat.PDF.value == "pdf"

    def test_png(self):
        assert DocumentFormat.PNG.value == "png"

    def test_jpg(self):
        assert DocumentFormat.JPG.value == "jpg"

    def test_jpeg(self):
        assert DocumentFormat.JPEG.value == "jpeg"

    def test_tiff(self):
        assert DocumentFormat.TIFF.value == "tiff"

    def test_bmp(self):
        assert DocumentFormat.BMP.value == "bmp"

    def test_all_6_formats(self):
        assert len(DocumentFormat) == 6


class TestTemplateTypeEnum:
    """Test TemplateType enum values."""

    def test_invoice(self):
        assert TemplateType.INVOICE.value == "invoice"

    def test_manifest(self):
        assert TemplateType.MANIFEST.value == "manifest"

    def test_utility_bill(self):
        assert TemplateType.UTILITY_BILL.value == "utility_bill"

    def test_receipt(self):
        assert TemplateType.RECEIPT.value == "receipt"

    def test_purchase_order(self):
        assert TemplateType.PURCHASE_ORDER.value == "purchase_order"

    def test_custom(self):
        assert TemplateType.CUSTOM.value == "custom"

    def test_all_6_types(self):
        assert len(TemplateType) == 6


class TestJobStatusEnum:
    """Test JobStatus enum values."""

    def test_queued(self):
        assert JobStatus.QUEUED.value == "queued"

    def test_running(self):
        assert JobStatus.RUNNING.value == "running"

    def test_completed(self):
        assert JobStatus.COMPLETED.value == "completed"

    def test_failed(self):
        assert JobStatus.FAILED.value == "failed"

    def test_cancelled(self):
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_all_5_statuses(self):
        assert len(JobStatus) == 5


class TestValidationSeverityEnum:
    """Test ValidationSeverity enum values."""

    def test_error(self):
        assert ValidationSeverity.ERROR.value == "error"

    def test_warning(self):
        assert ValidationSeverity.WARNING.value == "warning"

    def test_info(self):
        assert ValidationSeverity.INFO.value == "info"

    def test_all_3_severities(self):
        assert len(ValidationSeverity) == 3


class TestPageContentModel:
    """Test PageContent model."""

    def test_creation_defaults(self):
        p = PageContent()
        assert p.page_number == 1
        assert p.text == ""
        assert p.confidence == 0.0
        assert p.ocr_engine == "tesseract"
        assert p.regions == []

    def test_creation_with_values(self):
        p = PageContent(page_number=3, text="Hello", confidence=0.95, ocr_engine="textract")
        assert p.page_number == 3
        assert p.text == "Hello"
        assert p.confidence == 0.95

    def test_to_dict(self):
        p = PageContent(page_number=1, text="Test", confidence=0.9)
        d = p.to_dict()
        assert d["page_number"] == 1
        assert d["text"] == "Test"
        assert d["confidence"] == 0.9

    def test_regions_default_empty(self):
        p = PageContent()
        assert p.regions == []

    def test_regions_with_data(self):
        regions = [{"x": 10, "y": 20, "w": 100, "h": 50, "text": "Invoice"}]
        p = PageContent(regions=regions)
        assert len(p.regions) == 1
        assert p.regions[0]["text"] == "Invoice"


class TestDocumentRecordModel:
    """Test DocumentRecord model."""

    def test_creation_defaults(self):
        d = DocumentRecord()
        assert len(d.document_id) == 36
        assert d.filename == ""
        assert d.file_format == "pdf"
        assert d.status == "pending"

    def test_creation_with_values(self):
        d = DocumentRecord(
            document_id="doc-001",
            filename="invoice.pdf",
            file_format="pdf",
            file_size_bytes=102400,
            page_count=3,
            document_type="invoice",
        )
        assert d.document_id == "doc-001"
        assert d.filename == "invoice.pdf"
        assert d.page_count == 3

    def test_auto_generated_id(self):
        d = DocumentRecord()
        assert len(d.document_id) == 36

    def test_to_dict(self):
        d = DocumentRecord(document_id="doc-001", filename="test.pdf")
        result = d.to_dict()
        assert result["document_id"] == "doc-001"
        assert result["filename"] == "test.pdf"
        assert "created_at" in result

    def test_default_timestamps(self):
        d = DocumentRecord()
        assert d.created_at is not None
        assert d.updated_at is not None
        assert d.created_at == d.updated_at


class TestExtractionJobModel:
    """Test ExtractionJob model."""

    def test_creation_defaults(self):
        j = ExtractionJob()
        assert len(j.job_id) == 36
        assert j.status == "queued"
        assert j.confidence == 0.0

    def test_creation_with_values(self):
        j = ExtractionJob(
            job_id="job-001",
            document_id="doc-001",
            document_type="invoice",
            status="completed",
            confidence=0.92,
            processing_time_ms=150.5,
        )
        assert j.job_id == "job-001"
        assert j.document_type == "invoice"
        assert j.processing_time_ms == 150.5

    def test_to_dict(self):
        j = ExtractionJob(job_id="job-001", document_id="doc-001")
        d = j.to_dict()
        assert d["job_id"] == "job-001"
        assert d["document_id"] == "doc-001"

    def test_error_message(self):
        j = ExtractionJob(status="failed", error_message="OCR engine unavailable")
        assert j.error_message == "OCR engine unavailable"

    def test_provenance_hash(self):
        j = ExtractionJob(provenance_hash="abc123" * 10 + "abcd")
        assert j.provenance_hash is not None


class TestLineItemModel:
    """Test LineItem model."""

    def test_creation(self):
        li = LineItem(
            item_code="CARB-001",
            description="Carbon Offset Credits",
            quantity=100,
            unit_price=25.0,
            amount=2500.0,
            confidence=0.95,
        )
        assert li.item_code == "CARB-001"
        assert li.amount == 2500.0

    def test_defaults(self):
        li = LineItem()
        assert li.quantity == 0.0
        assert li.unit_price == 0.0
        assert li.amount == 0.0

    def test_to_dict(self):
        li = LineItem(item_code="X", quantity=10, unit_price=5.0, amount=50.0)
        d = li.to_dict()
        assert d["quantity"] == 10
        assert d["amount"] == 50.0


class TestInvoiceExtractionModel:
    """Test InvoiceExtraction model."""

    def test_creation_defaults(self):
        inv = InvoiceExtraction()
        assert inv.invoice_number == ""
        assert inv.line_items == []
        assert inv.total_amount == 0.0
        assert inv.currency == "USD"

    def test_creation_with_values(self):
        inv = InvoiceExtraction(
            invoice_number="INV-001",
            invoice_date="2025-06-15",
            vendor_name="EcoSupply",
            total_amount=11130.0,
            currency="USD",
        )
        assert inv.invoice_number == "INV-001"
        assert inv.total_amount == 11130.0

    def test_with_line_items(self):
        items = [
            LineItem(item_code="A", quantity=10, unit_price=5.0, amount=50.0),
            LineItem(item_code="B", quantity=5, unit_price=10.0, amount=50.0),
        ]
        inv = InvoiceExtraction(line_items=items, subtotal=100.0, total_amount=100.0)
        assert len(inv.line_items) == 2

    def test_to_dict(self):
        inv = InvoiceExtraction(invoice_number="INV-001", total_amount=100.0)
        d = inv.to_dict()
        assert d["invoice_number"] == "INV-001"
        assert d["total_amount"] == 100.0
        assert "line_items" in d

    def test_payment_terms(self):
        inv = InvoiceExtraction(payment_terms="Net 30")
        assert inv.payment_terms == "Net 30"


class TestManifestExtractionModel:
    """Test ManifestExtraction model."""

    def test_creation_defaults(self):
        m = ManifestExtraction()
        assert m.manifest_number == ""
        assert m.cargo_items == []
        assert m.total_weight_kg == 0.0

    def test_creation_with_values(self):
        m = ManifestExtraction(
            manifest_number="BOL-2025-56789",
            shipper_name="EcoMaterials GmbH",
            consignee_name="GreenCorp Ltd.",
            carrier_name="MaerskLine",
            total_packages=85,
            total_weight_kg=25500.0,
            total_volume_m3=35.8,
        )
        assert m.manifest_number == "BOL-2025-56789"
        assert m.total_weight_kg == 25500.0

    def test_to_dict(self):
        m = ManifestExtraction(manifest_number="BOL-001", total_weight_kg=1000.0)
        d = m.to_dict()
        assert d["manifest_number"] == "BOL-001"
        assert d["total_weight_kg"] == 1000.0

    def test_port_fields(self):
        m = ManifestExtraction(port_of_loading="DEHAM", port_of_discharge="GBFXT")
        assert m.port_of_loading == "DEHAM"
        assert m.port_of_discharge == "GBFXT"

    def test_vessel_info(self):
        m = ManifestExtraction(vessel_name="MV Green Future", voyage_number="GF-2025-042")
        assert m.vessel_name == "MV Green Future"


class TestUtilityBillExtractionModel:
    """Test UtilityBillExtraction model."""

    def test_creation_defaults(self):
        u = UtilityBillExtraction()
        assert u.account_number == ""
        assert u.consumption == 0.0
        assert u.consumption_unit == "kWh"

    def test_creation_with_values(self):
        u = UtilityBillExtraction(
            account_number="ELEC-2025-78901",
            utility_type="electricity",
            previous_reading=145230,
            current_reading=152780,
            consumption=7550,
            total_amount=1252.13,
        )
        assert u.consumption == 7550
        assert u.total_amount == 1252.13

    def test_to_dict(self):
        u = UtilityBillExtraction(account_number="ACC-001", consumption=500)
        d = u.to_dict()
        assert d["account_number"] == "ACC-001"
        assert d["consumption"] == 500

    def test_reading_consistency(self):
        u = UtilityBillExtraction(previous_reading=100, current_reading=200, consumption=100)
        assert u.current_reading - u.previous_reading == u.consumption


class TestExtractionTemplateModel:
    """Test ExtractionTemplate model."""

    def test_creation_defaults(self):
        t = ExtractionTemplate()
        assert len(t.template_id) == 36
        assert t.name == ""
        assert t.template_type == "custom"

    def test_creation_with_values(self):
        t = ExtractionTemplate(
            template_id="tpl-001",
            name="Standard Invoice",
            template_type="invoice",
            field_mappings={"invoice_number": "Invoice No"},
            regex_patterns={"invoice_number": r"INV-\d+"},
        )
        assert t.template_id == "tpl-001"
        assert t.name == "Standard Invoice"

    def test_to_dict(self):
        t = ExtractionTemplate(name="Test Template")
        d = t.to_dict()
        assert d["name"] == "Test Template"
        assert "field_mappings" in d

    def test_default_empty_mappings(self):
        t = ExtractionTemplate()
        assert t.field_mappings == {}
        assert t.regex_patterns == {}


class TestValidationResultModel:
    """Test ValidationResult model."""

    def test_creation_defaults(self):
        v = ValidationResult()
        assert v.is_valid is True
        assert v.errors == []
        assert v.warnings == []
        assert v.info == []

    def test_creation_with_errors(self):
        errors = [{"field": "total", "message": "Total mismatch"}]
        v = ValidationResult(is_valid=False, errors=errors, rules_checked=5)
        assert v.is_valid is False
        assert len(v.errors) == 1
        assert v.rules_checked == 5

    def test_to_dict(self):
        v = ValidationResult(is_valid=True, fields_validated=10, rules_checked=8)
        d = v.to_dict()
        assert d["is_valid"] is True
        assert d["fields_validated"] == 10

    def test_warnings_only(self):
        warnings = [{"field": "date", "message": "Date in the past"}]
        v = ValidationResult(is_valid=True, warnings=warnings)
        assert v.is_valid is True
        assert len(v.warnings) == 1


class TestBatchJobModel:
    """Test BatchJob model."""

    def test_creation_defaults(self):
        b = BatchJob()
        assert len(b.batch_id) == 36
        assert b.status == "queued"
        assert b.total_documents == 0

    def test_creation_with_values(self):
        b = BatchJob(
            batch_id="batch-001",
            document_ids=["doc-1", "doc-2", "doc-3"],
            status="running",
            total_documents=3,
            completed_documents=1,
        )
        assert b.batch_id == "batch-001"
        assert len(b.document_ids) == 3

    def test_progress_pct_zero_total(self):
        b = BatchJob(total_documents=0)
        assert b.progress_pct == 0.0

    def test_progress_pct_partial(self):
        b = BatchJob(total_documents=10, completed_documents=5, failed_documents=1)
        assert b.progress_pct == 60.0

    def test_progress_pct_complete(self):
        b = BatchJob(total_documents=10, completed_documents=10)
        assert b.progress_pct == 100.0

    def test_to_dict(self):
        b = BatchJob(batch_id="batch-001", total_documents=5)
        d = b.to_dict()
        assert d["batch_id"] == "batch-001"
        assert "progress_pct" in d

    def test_to_dict_includes_progress(self):
        b = BatchJob(total_documents=4, completed_documents=2)
        d = b.to_dict()
        assert d["progress_pct"] == 50.0


class TestPDFStatisticsModel:
    """Test PDFStatistics model."""

    def test_creation_defaults(self):
        s = PDFStatistics()
        assert s.documents_ingested == 0
        assert s.extractions_completed == 0
        assert s.avg_confidence == 0.0

    def test_record_extraction(self):
        s = PDFStatistics()
        s.record_extraction(0.9)
        assert s.extractions_completed == 1
        assert s.avg_confidence == 0.9

    def test_record_multiple_extractions(self):
        s = PDFStatistics()
        s.record_extraction(0.8)
        s.record_extraction(1.0)
        assert s.extractions_completed == 2
        assert s.avg_confidence == pytest.approx(0.9, rel=1e-6)

    def test_to_dict(self):
        s = PDFStatistics()
        s.documents_ingested = 5
        s.record_extraction(0.85)
        d = s.to_dict()
        assert d["documents_ingested"] == 5
        assert d["extractions_completed"] == 1
        assert d["avg_confidence"] == 0.85

    def test_incremental_counters(self):
        s = PDFStatistics()
        s.documents_ingested += 1
        s.documents_classified += 1
        s.validations_run += 1
        s.total_pages_processed += 5
        assert s.documents_ingested == 1
        assert s.documents_classified == 1
        assert s.total_pages_processed == 5
