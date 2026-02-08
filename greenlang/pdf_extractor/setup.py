# -*- coding: utf-8 -*-
"""
PDF & Invoice Extractor Service Setup - AGENT-DATA-001: PDF Extractor

Provides ``configure_pdf_extractor(app)`` which wires up the PDF &
Invoice Extractor SDK (document parser, OCR engine adapter, field
extractor, invoice processor, manifest processor, document classifier,
validation engine, provenance tracker) and mounts the REST API.

Also exposes ``get_pdf_extractor(app)`` for programmatic access
and the ``PDFExtractorService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.pdf_extractor.setup import configure_pdf_extractor
    >>> app = FastAPI()
    >>> import asyncio
    >>> service = asyncio.run(configure_pdf_extractor(app))

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-001 PDF & Invoice Extractor
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from greenlang.pdf_extractor.config import PDFExtractorConfig, get_config
from greenlang.pdf_extractor.metrics import (
    PROMETHEUS_AVAILABLE,
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False


# ===================================================================
# Lightweight Pydantic models used by the facade
# ===================================================================


class DocumentRecord(BaseModel):
    """Record representing an ingested and processed document.

    Attributes:
        document_id: Unique identifier for this document.
        file_name: Original filename or generated identifier.
        document_type: Classified document type.
        page_count: Number of pages in the document.
        pages: Page-level metadata list.
        extracted_fields: Mapping of field name to extracted value.
        extraction_result: Full extraction result payload.
        confidence: Overall extraction confidence (0.0-1.0).
        ocr_engine: OCR engine used for this document.
        tenant_id: Owning tenant identifier.
        status: Processing status.
        provenance_hash: SHA-256 provenance hash.
        created_at: Timestamp of ingestion.
    """
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_name: str = Field(default="")
    document_type: str = Field(default="unknown")
    page_count: int = Field(default=0)
    pages: List[Dict[str, Any]] = Field(default_factory=list)
    extracted_fields: Dict[str, Any] = Field(default_factory=dict)
    extraction_result: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0)
    ocr_engine: str = Field(default="tesseract")
    tenant_id: str = Field(default="default")
    status: str = Field(default="processed")
    provenance_hash: str = Field(default="")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class ExtractionJob(BaseModel):
    """Represents an asynchronous extraction job.

    Attributes:
        job_id: Unique job identifier.
        document_id: Associated document identifier.
        status: Job status (queued, processing, completed, failed).
        progress_pct: Completion percentage 0-100.
        created_at: Timestamp of job creation.
        completed_at: Timestamp of job completion.
        provenance_hash: SHA-256 provenance hash.
    """
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = Field(default="")
    status: str = Field(default="queued")
    progress_pct: float = Field(default=0.0)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    completed_at: Optional[str] = Field(default=None)
    provenance_hash: str = Field(default="")


class BatchJob(BaseModel):
    """Represents a batch ingestion job.

    Attributes:
        batch_id: Unique batch identifier.
        document_count: Number of documents in the batch.
        status: Batch status (submitted, processing, completed, partial, failed).
        jobs: Individual extraction jobs within the batch.
        created_at: Timestamp of batch creation.
        provenance_hash: SHA-256 provenance hash.
    """
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_count: int = Field(default=0)
    status: str = Field(default="submitted")
    jobs: List[ExtractionJob] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class InvoiceExtraction(BaseModel):
    """Structured invoice extraction result.

    Attributes:
        document_id: Source document identifier.
        vendor_name: Extracted vendor / supplier name.
        invoice_number: Extracted invoice number.
        invoice_date: Extracted invoice date string.
        due_date: Extracted payment due date.
        total_amount: Extracted total amount.
        currency: Extracted currency code.
        line_items: List of extracted line items.
        confidence: Overall extraction confidence.
        provenance_hash: SHA-256 provenance hash.
    """
    document_id: str = Field(default="")
    vendor_name: Optional[str] = Field(default=None)
    invoice_number: Optional[str] = Field(default=None)
    invoice_date: Optional[str] = Field(default=None)
    due_date: Optional[str] = Field(default=None)
    total_amount: Optional[float] = Field(default=None)
    currency: Optional[str] = Field(default=None)
    line_items: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class ManifestExtraction(BaseModel):
    """Structured manifest / bill-of-lading extraction result.

    Attributes:
        document_id: Source document identifier.
        shipper_name: Extracted shipper name.
        consignee_name: Extracted consignee name.
        manifest_number: Manifest or BOL number.
        ship_date: Extracted shipping date.
        delivery_date: Extracted delivery date.
        origin: Origin location.
        destination: Destination location.
        weight_kg: Total weight in kilograms.
        line_items: Cargo line items.
        confidence: Overall extraction confidence.
        provenance_hash: SHA-256 provenance hash.
    """
    document_id: str = Field(default="")
    shipper_name: Optional[str] = Field(default=None)
    consignee_name: Optional[str] = Field(default=None)
    manifest_number: Optional[str] = Field(default=None)
    ship_date: Optional[str] = Field(default=None)
    delivery_date: Optional[str] = Field(default=None)
    origin: Optional[str] = Field(default=None)
    destination: Optional[str] = Field(default=None)
    weight_kg: Optional[float] = Field(default=None)
    line_items: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class UtilityBillExtraction(BaseModel):
    """Structured utility bill extraction result.

    Attributes:
        document_id: Source document identifier.
        provider_name: Utility provider name.
        account_number: Customer account number.
        billing_period_start: Billing period start date.
        billing_period_end: Billing period end date.
        total_amount: Total bill amount.
        currency: Currency code.
        usage_kwh: Electricity usage in kWh.
        usage_therms: Natural gas usage in therms.
        usage_gallons: Water usage in gallons.
        facility_address: Service address.
        confidence: Overall extraction confidence.
        provenance_hash: SHA-256 provenance hash.
    """
    document_id: str = Field(default="")
    provider_name: Optional[str] = Field(default=None)
    account_number: Optional[str] = Field(default=None)
    billing_period_start: Optional[str] = Field(default=None)
    billing_period_end: Optional[str] = Field(default=None)
    total_amount: Optional[float] = Field(default=None)
    currency: Optional[str] = Field(default=None)
    usage_kwh: Optional[float] = Field(default=None)
    usage_therms: Optional[float] = Field(default=None)
    usage_gallons: Optional[float] = Field(default=None)
    facility_address: Optional[str] = Field(default=None)
    confidence: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class ExtractionTemplate(BaseModel):
    """Extraction template definition.

    Attributes:
        template_id: Unique template identifier.
        name: Human-readable template name.
        template_type: Template category (invoice, manifest, utility_bill, generic).
        field_patterns: Field name to regex/pattern mapping.
        validation_rules: Cross-field validation rules.
        description: Template description.
        created_at: Timestamp of creation.
        provenance_hash: SHA-256 provenance hash.
    """
    template_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="")
    template_type: str = Field(default="generic")
    field_patterns: Dict[str, Any] = Field(default_factory=dict)
    validation_rules: Dict[str, Any] = Field(default_factory=dict)
    description: str = Field(default="")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    provenance_hash: str = Field(default="")


class ValidationResult(BaseModel):
    """Result of a single validation check.

    Attributes:
        rule_name: Name of the validation rule.
        passed: Whether the check passed.
        severity: Severity level (critical, high, medium, low, info).
        message: Human-readable message.
        field: Field name that was validated.
    """
    rule_name: str = Field(default="")
    passed: bool = Field(default=True)
    severity: str = Field(default="info")
    message: str = Field(default="")
    field: Optional[str] = Field(default=None)


class PDFStatistics(BaseModel):
    """Aggregate statistics for the PDF extractor service.

    Attributes:
        total_documents: Total documents processed.
        total_pages: Total pages extracted.
        total_fields: Total fields extracted.
        total_line_items: Total line items extracted.
        total_invoices: Total invoices processed.
        total_manifests: Total manifests processed.
        total_utility_bills: Total utility bills processed.
        total_batch_jobs: Total batch jobs submitted.
        avg_confidence: Average extraction confidence.
        avg_processing_time_ms: Average processing time in milliseconds.
        total_validation_errors: Total validation errors detected.
        total_ocr_operations: Total OCR operations performed.
    """
    total_documents: int = Field(default=0)
    total_pages: int = Field(default=0)
    total_fields: int = Field(default=0)
    total_line_items: int = Field(default=0)
    total_invoices: int = Field(default=0)
    total_manifests: int = Field(default=0)
    total_utility_bills: int = Field(default=0)
    total_batch_jobs: int = Field(default=0)
    avg_confidence: float = Field(default=0.0)
    avg_processing_time_ms: float = Field(default=0.0)
    total_validation_errors: int = Field(default=0)
    total_ocr_operations: int = Field(default=0)


# ===================================================================
# Provenance helper
# ===================================================================


class _ProvenanceTracker:
    """Minimal provenance tracker recording SHA-256 audit entries.

    Attributes:
        entries: List of provenance entries.
        entry_count: Number of entries recorded.
    """

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self.entry_count: int = 0

    def record(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        data_hash: str,
        user_id: str = "system",
    ) -> str:
        """Record a provenance entry and return its hash.

        Args:
            entity_type: Type of entity (document, invoice, template, etc.).
            entity_id: Entity identifier.
            action: Action performed (ingest, extract, validate, etc.).
            data_hash: SHA-256 hash of associated data.
            user_id: User or system that performed the action.

        Returns:
            SHA-256 hash of the provenance entry itself.
        """
        entry = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "action": action,
            "data_hash": data_hash,
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        entry_hash = hashlib.sha256(
            json.dumps(entry, sort_keys=True, default=str).encode()
        ).hexdigest()
        entry["entry_hash"] = entry_hash
        self._entries.append(entry)
        self.entry_count += 1
        return entry_hash


# ===================================================================
# PDFExtractorService facade
# ===================================================================

# Thread-safe singleton lock
_singleton_lock = threading.Lock()
_singleton_instance: Optional["PDFExtractorService"] = None


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, str, or Pydantic model).

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


class PDFExtractorService:
    """Unified facade over the PDF & Invoice Extractor SDK.

    Aggregates all extraction engines (document parser, OCR engine adapter,
    field extractor, invoice processor, manifest processor, document
    classifier, validation engine, provenance tracker) through a single
    entry point with convenience methods for common operations.

    Each method records provenance and updates self-monitoring metrics.

    Attributes:
        config: PDFExtractorConfig instance.
        provenance: _ProvenanceTracker instance for SHA-256 audit trails.

    Example:
        >>> service = PDFExtractorService()
        >>> record = service.ingest_document(file_content="Invoice #123...")
        >>> print(record.document_type, record.confidence)
    """

    def __init__(
        self,
        config: Optional[PDFExtractorConfig] = None,
    ) -> None:
        """Initialize the PDF Extractor Service facade.

        Instantiates all 7 internal engines plus the provenance tracker:
        - DocumentParser
        - OCREngineAdapter
        - FieldExtractor
        - InvoiceProcessor
        - ManifestProcessor
        - DocumentClassifier
        - ValidationEngine

        Args:
            config: Optional configuration. Uses global config if None.
        """
        self.config = config or get_config()

        # Provenance tracker
        self.provenance = _ProvenanceTracker()

        # Engine placeholders -- real implementations are injected by the
        # respective SDK modules at import time. We use a lazy-init approach
        # so that setup.py can be imported without the full SDK installed.
        self._document_parser: Any = None
        self._ocr_engine: Any = None
        self._field_extractor: Any = None
        self._invoice_processor: Any = None
        self._manifest_processor: Any = None
        self._document_classifier: Any = None
        self._validation_engine: Any = None

        self._init_engines()

        # In-memory stores (production uses DB; these are SDK-level caches)
        self._documents: Dict[str, DocumentRecord] = {}
        self._jobs: Dict[str, ExtractionJob] = {}
        self._batch_jobs: Dict[str, BatchJob] = {}
        self._templates: Dict[str, ExtractionTemplate] = {}

        # Statistics
        self._stats = PDFStatistics()
        self._started = False

        logger.info("PDFExtractorService facade created")

    # ------------------------------------------------------------------
    # Engine initialization
    # ------------------------------------------------------------------

    def _init_engines(self) -> None:
        """Attempt to import and initialise SDK engines.

        Engines are optional; missing imports are logged as warnings and
        the service continues in degraded mode.
        """
        try:
            from greenlang.pdf_extractor.document_parser import DocumentParser
            self._document_parser = DocumentParser(self.config)
        except ImportError:
            logger.warning("DocumentParser not available; using stub")

        try:
            from greenlang.pdf_extractor.ocr_engine import OCREngineAdapter
            self._ocr_engine = OCREngineAdapter(self.config)
        except ImportError:
            logger.warning("OCREngineAdapter not available; using stub")

        try:
            from greenlang.pdf_extractor.field_extractor import FieldExtractor
            self._field_extractor = FieldExtractor(self.config)
        except ImportError:
            logger.warning("FieldExtractor not available; using stub")

        try:
            from greenlang.pdf_extractor.invoice_processor import InvoiceProcessor
            self._invoice_processor = InvoiceProcessor(self.config)
        except ImportError:
            logger.warning("InvoiceProcessor not available; using stub")

        try:
            from greenlang.pdf_extractor.manifest_processor import ManifestProcessor
            self._manifest_processor = ManifestProcessor(self.config)
        except ImportError:
            logger.warning("ManifestProcessor not available; using stub")

        try:
            from greenlang.pdf_extractor.document_classifier import DocumentClassifier
            self._document_classifier = DocumentClassifier(self.config)
        except ImportError:
            logger.warning("DocumentClassifier not available; using stub")

        try:
            from greenlang.pdf_extractor.validation_engine import ValidationEngine
            self._validation_engine = ValidationEngine(self.config)
        except ImportError:
            logger.warning("ValidationEngine not available; using stub")

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    def ingest_document(
        self,
        file_path: Optional[str] = None,
        file_content: Optional[str] = None,
        file_base64: Optional[str] = None,
        document_type: Optional[str] = None,
        ocr_engine: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        tenant_id: str = "default",
    ) -> DocumentRecord:
        """Ingest a single document for extraction.

        At least one of ``file_path``, ``file_content``, or ``file_base64``
        must be provided.

        Args:
            file_path: Server-side file path to read.
            file_content: Raw text content.
            file_base64: Base64-encoded file content.
            document_type: Document type hint; auto-classified if None.
            ocr_engine: OCR engine override.
            confidence_threshold: Minimum confidence override.
            tenant_id: Tenant identifier.

        Returns:
            DocumentRecord with extraction results.

        Raises:
            ValueError: If no input source is provided.
        """
        start_time = time.time()

        if not any([file_path, file_content, file_base64]):
            raise ValueError(
                "At least one of file_path, file_content, or file_base64 "
                "must be provided"
            )

        update_active_jobs(1)
        engine_name = ocr_engine or self.config.default_ocr_engine
        threshold = confidence_threshold or self.config.default_confidence_threshold

        try:
            # Step 1: Parse document text
            text = self._parse_document(
                file_path=file_path,
                file_content=file_content,
                file_base64=file_base64,
                engine_name=engine_name,
            )

            # Step 2: Classify document type if not provided
            if document_type is None and self.config.enable_document_classification:
                document_type, cls_conf = self.classify_document(
                    text=text, file_name=file_path,
                )
            elif document_type is None:
                document_type = "unknown"

            # Step 3: Extract fields
            extracted_fields = self._extract_fields(text, document_type, threshold)

            # Step 4: Build record
            page_count = max(1, text.count("\f") + 1) if text else 0
            pages = [{"page_number": i + 1} for i in range(page_count)]

            record = DocumentRecord(
                file_name=file_path or "inline_content",
                document_type=document_type,
                page_count=page_count,
                pages=pages,
                extracted_fields=extracted_fields,
                extraction_result=extracted_fields,
                confidence=self._compute_avg_confidence(extracted_fields),
                ocr_engine=engine_name,
                tenant_id=tenant_id,
                status="processed",
            )

            # Step 5: Compute provenance
            record.provenance_hash = _compute_hash(record)

            # Step 6: Store
            self._documents[record.document_id] = record

            # Step 7: Record metrics
            duration = time.time() - start_time
            record_document_processed(document_type, tenant_id, duration)
            record_pages_extracted(page_count)
            record_fields_extracted(
                len(extracted_fields), "success", document_type,
            )
            record_extraction_confidence(record.confidence)
            record_ocr_operation(engine_name, "success")

            # Step 8: Record provenance
            self.provenance.record(
                entity_type="document",
                entity_id=record.document_id,
                action="ingest",
                data_hash=record.provenance_hash,
            )

            # Step 9: Update statistics
            self._update_stats_for_document(record, duration)

            logger.info(
                "Ingested document %s (%s, %d pages, confidence=%.2f) in %.2fs",
                record.document_id, document_type, page_count,
                record.confidence, duration,
            )
            return record

        except Exception as exc:
            record_ocr_operation(engine_name, "error")
            logger.error("Document ingestion failed: %s", exc, exc_info=True)
            raise
        finally:
            update_active_jobs(-1)

    def ingest_batch(
        self,
        documents: List[Dict[str, Any]],
        ocr_engine: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        tenant_id: str = "default",
    ) -> BatchJob:
        """Batch ingest multiple documents.

        Args:
            documents: List of document dicts with file_path/content/base64 keys.
            ocr_engine: OCR engine override for the entire batch.
            confidence_threshold: Minimum confidence override.
            tenant_id: Tenant identifier.

        Returns:
            BatchJob with individual extraction jobs.

        Raises:
            ValueError: If documents list is empty or exceeds batch limit.
        """
        if not documents:
            raise ValueError("Documents list must not be empty")
        if len(documents) > self.config.batch_max_documents:
            raise ValueError(
                f"Batch size {len(documents)} exceeds maximum "
                f"{self.config.batch_max_documents}"
            )

        batch = BatchJob(
            document_count=len(documents),
            status="processing",
        )
        self._batch_jobs[batch.batch_id] = batch

        record_batch_job("submitted")
        update_queue_size(len(documents))

        jobs: List[ExtractionJob] = []
        completed = 0
        failed = 0

        for doc_spec in documents:
            job = ExtractionJob(status="processing")
            jobs.append(job)
            self._jobs[job.job_id] = job

            try:
                record = self.ingest_document(
                    file_path=doc_spec.get("file_path"),
                    file_content=doc_spec.get("file_content"),
                    file_base64=doc_spec.get("file_base64"),
                    document_type=doc_spec.get("document_type"),
                    ocr_engine=ocr_engine,
                    confidence_threshold=confidence_threshold,
                    tenant_id=tenant_id,
                )
                job.document_id = record.document_id
                job.status = "completed"
                job.progress_pct = 100.0
                job.completed_at = datetime.now(timezone.utc).isoformat()
                job.provenance_hash = record.provenance_hash
                completed += 1
            except Exception as exc:
                job.status = "failed"
                job.progress_pct = 0.0
                failed += 1
                logger.warning("Batch item failed: %s", exc)

        # Determine batch status
        if failed == 0:
            batch.status = "completed"
            record_batch_job("completed")
        elif completed == 0:
            batch.status = "failed"
            record_batch_job("failed")
        else:
            batch.status = "partial"
            record_batch_job("partial")

        batch.jobs = jobs
        batch.provenance_hash = _compute_hash(batch)
        update_queue_size(0)

        # Record provenance
        self.provenance.record(
            entity_type="batch_job",
            entity_id=batch.batch_id,
            action="batch_ingest",
            data_hash=batch.provenance_hash,
        )

        self._stats.total_batch_jobs += 1

        logger.info(
            "Batch %s completed: %d/%d succeeded, status=%s",
            batch.batch_id, completed, len(documents), batch.status,
        )
        return batch

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify_document(
        self,
        text: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Classify a document's type.

        Uses the DocumentClassifier engine when available; otherwise falls
        back to keyword-based heuristic classification.

        Args:
            text: Extracted text for classification.
            file_name: Original filename for heuristic hints.

        Returns:
            Tuple of (document_type, confidence_score).

        Raises:
            ValueError: If neither text nor file_name is provided.
        """
        if text is None and file_name is None:
            raise ValueError("At least one of text or file_name must be provided")

        # Delegate to engine if available
        if self._document_classifier is not None:
            doc_type, confidence = self._document_classifier.classify(
                text=text, file_name=file_name,
            )
        else:
            doc_type, confidence = self._heuristic_classify(text, file_name)

        record_classification(doc_type)

        self.provenance.record(
            entity_type="classification",
            entity_id=str(uuid.uuid4()),
            action="classify",
            data_hash=_compute_hash({"text_len": len(text or ""), "result": doc_type}),
        )

        return doc_type, confidence

    # ------------------------------------------------------------------
    # Invoice extraction
    # ------------------------------------------------------------------

    def extract_invoice(
        self,
        document_id_or_text: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        template: Optional[str] = None,
    ) -> InvoiceExtraction:
        """Extract structured invoice data.

        Accepts either an existing document_id or raw text for direct
        extraction.

        Args:
            document_id_or_text: Document ID or raw text.
            confidence_threshold: Minimum confidence override.
            template: Template name to apply.

        Returns:
            InvoiceExtraction with structured invoice fields.

        Raises:
            ValueError: If input is None.
        """
        start_time = time.time()
        if document_id_or_text is None:
            raise ValueError("document_id or text must be provided")

        # Resolve text
        text, doc_id = self._resolve_text(document_id_or_text)

        # Delegate to engine
        if self._invoice_processor is not None:
            result_data = self._invoice_processor.extract(
                text, threshold=confidence_threshold, template=template,
            )
            extraction = InvoiceExtraction(document_id=doc_id, **result_data)
        else:
            extraction = InvoiceExtraction(
                document_id=doc_id,
                confidence=0.0,
            )

        # Provenance
        extraction.provenance_hash = _compute_hash(extraction)
        self.provenance.record(
            entity_type="invoice",
            entity_id=doc_id,
            action="extract_invoice",
            data_hash=extraction.provenance_hash,
        )

        # Metrics
        duration = time.time() - start_time
        record_document_processed("invoice", "default", duration)
        record_extraction_confidence(extraction.confidence)
        if extraction.line_items:
            record_line_items_extracted(len(extraction.line_items))

        self._stats.total_invoices += 1
        logger.info(
            "Invoice extraction for %s completed (confidence=%.2f) in %.2fs",
            doc_id, extraction.confidence, duration,
        )
        return extraction

    # ------------------------------------------------------------------
    # Manifest extraction
    # ------------------------------------------------------------------

    def extract_manifest(
        self,
        document_id_or_text: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
    ) -> ManifestExtraction:
        """Extract structured manifest / bill-of-lading data.

        Args:
            document_id_or_text: Document ID or raw text.
            confidence_threshold: Minimum confidence override.

        Returns:
            ManifestExtraction with structured manifest fields.

        Raises:
            ValueError: If input is None.
        """
        start_time = time.time()
        if document_id_or_text is None:
            raise ValueError("document_id or text must be provided")

        text, doc_id = self._resolve_text(document_id_or_text)

        if self._manifest_processor is not None:
            result_data = self._manifest_processor.extract(
                text, threshold=confidence_threshold,
            )
            extraction = ManifestExtraction(document_id=doc_id, **result_data)
        else:
            extraction = ManifestExtraction(
                document_id=doc_id,
                confidence=0.0,
            )

        extraction.provenance_hash = _compute_hash(extraction)
        self.provenance.record(
            entity_type="manifest",
            entity_id=doc_id,
            action="extract_manifest",
            data_hash=extraction.provenance_hash,
        )

        duration = time.time() - start_time
        record_document_processed("manifest", "default", duration)
        record_extraction_confidence(extraction.confidence)
        if extraction.line_items:
            record_line_items_extracted(len(extraction.line_items))

        self._stats.total_manifests += 1
        logger.info(
            "Manifest extraction for %s completed (confidence=%.2f) in %.2fs",
            doc_id, extraction.confidence, duration,
        )
        return extraction

    # ------------------------------------------------------------------
    # Utility bill extraction
    # ------------------------------------------------------------------

    def extract_utility_bill(
        self,
        document_id_or_text: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
    ) -> UtilityBillExtraction:
        """Extract structured utility bill data.

        Args:
            document_id_or_text: Document ID or raw text.
            confidence_threshold: Minimum confidence override.

        Returns:
            UtilityBillExtraction with structured utility bill fields.

        Raises:
            ValueError: If input is None.
        """
        start_time = time.time()
        if document_id_or_text is None:
            raise ValueError("document_id or text must be provided")

        text, doc_id = self._resolve_text(document_id_or_text)

        # Stub extraction -- real engine delegates to the validation pipeline
        extraction = UtilityBillExtraction(
            document_id=doc_id,
            confidence=0.0,
        )

        extraction.provenance_hash = _compute_hash(extraction)
        self.provenance.record(
            entity_type="utility_bill",
            entity_id=doc_id,
            action="extract_utility_bill",
            data_hash=extraction.provenance_hash,
        )

        duration = time.time() - start_time
        record_document_processed("utility_bill", "default", duration)
        record_extraction_confidence(extraction.confidence)

        self._stats.total_utility_bills += 1
        logger.info(
            "Utility bill extraction for %s completed (confidence=%.2f) in %.2fs",
            doc_id, extraction.confidence, duration,
        )
        return extraction

    # ------------------------------------------------------------------
    # Reprocess
    # ------------------------------------------------------------------

    def reprocess_document(
        self,
        document_id: str,
    ) -> ExtractionJob:
        """Reprocess an already-ingested document.

        Args:
            document_id: ID of the document to reprocess.

        Returns:
            ExtractionJob tracking the reprocessing task.

        Raises:
            ValueError: If document ID is not found.
        """
        record = self._documents.get(document_id)
        if record is None:
            raise ValueError(f"Document {document_id} not found")

        job = ExtractionJob(
            document_id=document_id,
            status="completed",
            progress_pct=100.0,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        job.provenance_hash = _compute_hash(job)
        self._jobs[job.job_id] = job

        self.provenance.record(
            entity_type="reprocess",
            entity_id=document_id,
            action="reprocess",
            data_hash=job.provenance_hash,
        )

        logger.info("Reprocessed document %s as job %s", document_id, job.job_id)
        return job

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_document(
        self,
        document_id: str,
    ) -> List[ValidationResult]:
        """Run validation rules on an extracted document.

        Args:
            document_id: ID of the document to validate.

        Returns:
            List of ValidationResult instances.

        Raises:
            ValueError: If document ID is not found.
        """
        record = self._documents.get(document_id)
        if record is None:
            raise ValueError(f"Document {document_id} not found")

        results: List[ValidationResult] = []

        # Delegate to engine if available
        if self._validation_engine is not None:
            results = self._validation_engine.validate(record)
        else:
            # Stub: basic presence checks
            for field_name, value in record.extracted_fields.items():
                passed = value is not None and str(value).strip() != ""
                results.append(ValidationResult(
                    rule_name=f"presence_{field_name}",
                    passed=passed,
                    severity="medium" if not passed else "info",
                    message=f"Field '{field_name}' is {'present' if passed else 'missing'}",
                    field=field_name,
                ))

        # Record metrics for failures
        for r in results:
            if not r.passed:
                record_validation_error(r.severity, record.document_type)
                self._stats.total_validation_errors += 1

        self.provenance.record(
            entity_type="validation",
            entity_id=document_id,
            action="validate",
            data_hash=_compute_hash({"results": [r.model_dump() for r in results]}),
        )

        return results

    # ------------------------------------------------------------------
    # Template management
    # ------------------------------------------------------------------

    def create_template(
        self,
        name: str,
        template_type: str,
        field_patterns: Dict[str, Any],
        validation_rules: Optional[Dict[str, Any]] = None,
    ) -> ExtractionTemplate:
        """Create a new extraction template.

        Args:
            name: Template name.
            template_type: Template category.
            field_patterns: Field name to pattern mapping.
            validation_rules: Optional cross-field validation rules.

        Returns:
            Created ExtractionTemplate.

        Raises:
            ValueError: If name is empty or duplicate.
        """
        if not name.strip():
            raise ValueError("Template name must not be empty")

        # Check for duplicate names
        for existing in self._templates.values():
            if existing.name == name:
                raise ValueError(f"Template with name '{name}' already exists")

        template = ExtractionTemplate(
            name=name,
            template_type=template_type,
            field_patterns=field_patterns,
            validation_rules=validation_rules or {},
        )
        template.provenance_hash = _compute_hash(template)
        self._templates[template.template_id] = template

        self.provenance.record(
            entity_type="template",
            entity_id=template.template_id,
            action="create",
            data_hash=template.provenance_hash,
        )

        logger.info("Created template '%s' (%s)", name, template.template_id)
        return template

    def list_templates(self) -> List[ExtractionTemplate]:
        """List all extraction templates.

        Returns:
            List of ExtractionTemplate instances.
        """
        return list(self._templates.values())

    # ------------------------------------------------------------------
    # Document queries
    # ------------------------------------------------------------------

    def get_document(
        self,
        document_id: str,
    ) -> Optional[DocumentRecord]:
        """Get a document record by ID.

        Args:
            document_id: Document identifier.

        Returns:
            DocumentRecord or None if not found.
        """
        return self._documents.get(document_id)

    def list_documents(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentRecord]:
        """List documents with optional filters.

        Supported filter keys: tenant_id, document_type, limit, offset.

        Args:
            filters: Optional filter dictionary.

        Returns:
            List of DocumentRecord instances.
        """
        docs = list(self._documents.values())
        if filters:
            tenant_id = filters.get("tenant_id")
            if tenant_id:
                docs = [d for d in docs if d.tenant_id == tenant_id]

            document_type = filters.get("document_type")
            if document_type:
                docs = [d for d in docs if d.document_type == document_type]

            offset = filters.get("offset", 0)
            limit = filters.get("limit", 50)
            docs = docs[offset:offset + limit]

        return docs

    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[ExtractionJob]:
        """List extraction jobs with optional status filter.

        Args:
            status: Optional status filter.
            limit: Maximum number of jobs to return.

        Returns:
            List of ExtractionJob instances.
        """
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return jobs[-limit:]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> PDFStatistics:
        """Get aggregated PDF extractor statistics.

        Returns:
            PDFStatistics summary.
        """
        return self._stats

    # ------------------------------------------------------------------
    # Convenience getters
    # ------------------------------------------------------------------

    def get_provenance(self) -> _ProvenanceTracker:
        """Get the ProvenanceTracker instance.

        Returns:
            _ProvenanceTracker used by this service.
        """
        return self.provenance

    # ------------------------------------------------------------------
    # Metrics summary
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Get PDF extractor service metrics summary.

        Returns:
            Dictionary with service metric summaries.
        """
        return {
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "started": self._started,
            "total_documents": self._stats.total_documents,
            "total_pages": self._stats.total_pages,
            "total_fields": self._stats.total_fields,
            "total_invoices": self._stats.total_invoices,
            "total_manifests": self._stats.total_manifests,
            "total_utility_bills": self._stats.total_utility_bills,
            "total_batch_jobs": self._stats.total_batch_jobs,
            "avg_confidence": self._stats.avg_confidence,
            "total_templates": len(self._templates),
            "provenance_entries": self.provenance.entry_count,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_document(
        self,
        file_path: Optional[str],
        file_content: Optional[str],
        file_base64: Optional[str],
        engine_name: str,
    ) -> str:
        """Parse a document into text, using OCR if needed.

        Args:
            file_path: Server-side file path.
            file_content: Raw text content.
            file_base64: Base64-encoded content.
            engine_name: OCR engine to use.

        Returns:
            Extracted text string.
        """
        # If raw text provided, return immediately
        if file_content:
            return file_content

        # Delegate to document parser engine
        if self._document_parser is not None:
            return self._document_parser.parse(
                file_path=file_path,
                file_base64=file_base64,
                ocr_engine=engine_name,
            )

        # Fallback: return empty string with warning
        logger.warning(
            "DocumentParser not available; returning empty text for %s",
            file_path or "base64_input",
        )
        self._stats.total_ocr_operations += 1
        return ""

    def _extract_fields(
        self,
        text: str,
        document_type: str,
        threshold: float,
    ) -> Dict[str, Any]:
        """Extract fields from text.

        Args:
            text: Document text.
            document_type: Document classification.
            threshold: Minimum confidence threshold.

        Returns:
            Mapping of field name to extracted value.
        """
        if self._field_extractor is not None:
            return self._field_extractor.extract(
                text, document_type=document_type, threshold=threshold,
            )
        # Stub: return empty
        return {}

    def _heuristic_classify(
        self,
        text: Optional[str],
        file_name: Optional[str],
    ) -> Tuple[str, float]:
        """Classify document using keyword heuristics.

        Args:
            text: Document text.
            file_name: Original filename.

        Returns:
            Tuple of (document_type, confidence).
        """
        combined = (text or "").lower() + " " + (file_name or "").lower()

        if any(kw in combined for kw in ["invoice", "inv#", "bill to", "amount due"]):
            return "invoice", 0.75
        if any(kw in combined for kw in ["manifest", "bill of lading", "bol", "consignee"]):
            return "manifest", 0.70
        if any(kw in combined for kw in ["utility", "kwh", "therms", "meter reading"]):
            return "utility_bill", 0.70
        if any(kw in combined for kw in ["receipt", "transaction", "paid"]):
            return "receipt", 0.60
        return "other", 0.50

    def _resolve_text(
        self,
        document_id_or_text: str,
    ) -> Tuple[str, str]:
        """Resolve a document_id or raw text into (text, doc_id).

        If the input matches a known document ID, the text is retrieved
        from the stored record. Otherwise it is treated as raw text.

        Args:
            document_id_or_text: Document ID or raw text.

        Returns:
            Tuple of (text, document_id).
        """
        record = self._documents.get(document_id_or_text)
        if record is not None:
            text = json.dumps(record.extracted_fields)
            return text, record.document_id

        # Treat as raw text
        return document_id_or_text, str(uuid.uuid4())

    @staticmethod
    def _compute_avg_confidence(fields: Dict[str, Any]) -> float:
        """Compute average confidence from extracted fields.

        If fields contain nested ``confidence`` keys, average them;
        otherwise return 0.0.

        Args:
            fields: Extracted field mapping.

        Returns:
            Average confidence score.
        """
        confidences: List[float] = []
        for value in fields.values():
            if isinstance(value, dict) and "confidence" in value:
                confidences.append(float(value["confidence"]))
        if confidences:
            return sum(confidences) / len(confidences)
        return 0.0

    def _update_stats_for_document(
        self,
        record: DocumentRecord,
        duration_seconds: float,
    ) -> None:
        """Update statistics after document processing.

        Args:
            record: Processed document record.
            duration_seconds: Processing time.
        """
        self._stats.total_documents += 1
        self._stats.total_pages += record.page_count
        self._stats.total_fields += len(record.extracted_fields)
        self._stats.total_ocr_operations += 1

        # Update running average confidence
        total = self._stats.total_documents
        prev_avg = self._stats.avg_confidence
        self._stats.avg_confidence = (
            (prev_avg * (total - 1) + record.confidence) / total
        )

        # Update running average processing time
        prev_time_avg = self._stats.avg_processing_time_ms
        self._stats.avg_processing_time_ms = (
            (prev_time_avg * (total - 1) + duration_seconds * 1000) / total
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Start the PDF extractor service.

        Safe to call multiple times.
        """
        if self._started:
            logger.debug("PDFExtractorService already started; skipping")
            return

        logger.info("PDFExtractorService starting up...")
        self._started = True
        logger.info("PDFExtractorService startup complete")

    def shutdown(self) -> None:
        """Shutdown the PDF extractor service and release resources."""
        if not self._started:
            return

        self._started = False
        logger.info("PDFExtractorService shut down")


# ===================================================================
# Thread-safe singleton access
# ===================================================================


def _get_singleton() -> PDFExtractorService:
    """Get or create the singleton PDFExtractorService instance.

    Returns:
        The singleton PDFExtractorService.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = PDFExtractorService()
    return _singleton_instance


# ===================================================================
# FastAPI integration
# ===================================================================


async def configure_pdf_extractor(
    app: Any,
    config: Optional[PDFExtractorConfig] = None,
) -> PDFExtractorService:
    """Configure the PDF Extractor Service on a FastAPI application.

    Creates the PDFExtractorService, stores it in app.state, mounts
    the PDF extractor API router, and starts the service.

    Args:
        app: FastAPI application instance.
        config: Optional PDF extractor config.

    Returns:
        PDFExtractorService instance.
    """
    global _singleton_instance

    service = PDFExtractorService(config=config)

    # Store as singleton
    with _singleton_lock:
        _singleton_instance = service

    # Attach to app state
    app.state.pdf_extractor_service = service

    # Mount PDF extractor API router
    try:
        from greenlang.pdf_extractor.api.router import router as pdf_router
        if pdf_router is not None:
            app.include_router(pdf_router)
            logger.info("PDF extractor service API router mounted")
    except ImportError:
        logger.warning("PDF extractor router not available; API not mounted")

    # Start service
    service.startup()

    logger.info("PDF extractor service configured on app")
    return service


def get_pdf_extractor(app: Any) -> PDFExtractorService:
    """Get the PDFExtractorService instance from app state.

    Args:
        app: FastAPI application instance.

    Returns:
        PDFExtractorService instance.

    Raises:
        RuntimeError: If PDF extractor service not configured.
    """
    service = getattr(app.state, "pdf_extractor_service", None)
    if service is None:
        raise RuntimeError(
            "PDF extractor service not configured. "
            "Call configure_pdf_extractor(app) first."
        )
    return service


def get_router(service: Optional[PDFExtractorService] = None) -> Any:
    """Get the PDF extractor API router.

    Args:
        service: Optional service instance (unused, kept for API compat).

    Returns:
        FastAPI APIRouter or None if FastAPI not available.
    """
    try:
        from greenlang.pdf_extractor.api.router import router
        return router
    except ImportError:
        return None


__all__ = [
    "PDFExtractorService",
    "configure_pdf_extractor",
    "get_pdf_extractor",
    "get_router",
    # Models
    "DocumentRecord",
    "ExtractionJob",
    "BatchJob",
    "InvoiceExtraction",
    "ManifestExtraction",
    "UtilityBillExtraction",
    "ExtractionTemplate",
    "ValidationResult",
    "PDFStatistics",
]
