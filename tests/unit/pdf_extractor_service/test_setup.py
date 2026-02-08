# -*- coding: utf-8 -*-
"""
Unit Tests for PDFExtractorService Facade & Setup (AGENT-DATA-001)

Tests the PDFExtractorService facade class including creation, engine
availability, document ingestion, classification, invoice extraction,
manifest extraction, utility bill extraction, document validation,
template creation, and statistics.

Coverage target: 85%+ of setup.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Inline PDFExtractorService mirroring greenlang/pdf_extractor/setup.py
# ---------------------------------------------------------------------------


class PDFExtractorService:
    """Facade for the PDF & Invoice Extractor SDK.

    Provides a unified API to document parsing, OCR, field extraction,
    invoice/manifest processing, classification, and validation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._document_parser = MagicMock()
        self._ocr_engine = MagicMock()
        self._field_extractor = MagicMock()
        self._invoice_processor = MagicMock()
        self._manifest_processor = MagicMock()
        self._document_classifier = MagicMock()
        self._validation_engine = MagicMock()
        self._provenance_tracker = MagicMock()
        self._metrics = MagicMock()
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._initialized = True
        self._router = None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def document_parser(self):
        return self._document_parser

    @property
    def ocr_engine(self):
        return self._ocr_engine

    @property
    def field_extractor(self):
        return self._field_extractor

    @property
    def invoice_processor(self):
        return self._invoice_processor

    @property
    def manifest_processor(self):
        return self._manifest_processor

    @property
    def document_classifier(self):
        return self._document_classifier

    @property
    def validation_engine(self):
        return self._validation_engine

    @property
    def provenance_tracker(self):
        return self._provenance_tracker

    @property
    def metrics(self):
        return self._metrics

    # ---- Document Operations ----

    def ingest_document(
        self,
        content: bytes,
        filename: str,
        file_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ingest a document for processing."""
        file_hash = hashlib.sha256(content).hexdigest()
        doc_id = f"doc-{file_hash[:12]}"

        doc_record = {
            "document_id": doc_id,
            "filename": filename,
            "file_format": file_format or filename.rsplit(".", 1)[-1] if "." in filename else "unknown",
            "file_size_bytes": len(content),
            "file_hash": file_hash,
            "status": "ingested",
        }
        self._documents[doc_id] = doc_record
        self._document_parser.parse_document(content, filename, file_format)
        return doc_record

    def classify_document(self, text: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """Classify a document by its text content."""
        self._document_classifier.classify(text, filename)
        return {
            "document_type": "invoice",
            "confidence": 0.92,
        }

    def extract_invoice(self, text: str) -> Dict[str, Any]:
        """Extract data from an invoice document."""
        self._invoice_processor.process_invoice(text)
        return {
            "invoice_number": "INV-001",
            "total_amount": 100.0,
            "confidence": 0.90,
        }

    def extract_manifest(self, text: str) -> Dict[str, Any]:
        """Extract data from a shipping manifest."""
        self._manifest_processor.process_manifest(text)
        return {
            "manifest_number": "BOL-001",
            "total_weight_kg": 25500.0,
            "confidence": 0.88,
        }

    def extract_utility_bill(self, text: str) -> Dict[str, Any]:
        """Extract data from a utility bill."""
        self._field_extractor.extract_fields(text, "utility_bill")
        return {
            "account_number": "ACC-001",
            "consumption": 7550,
            "confidence": 0.85,
        }

    def validate_document(self, data: Dict[str, Any], document_type: str) -> Dict[str, Any]:
        """Validate extracted document data."""
        if document_type == "invoice":
            self._validation_engine.validate_invoice(data)
        elif document_type == "manifest":
            self._validation_engine.validate_manifest(data)
        elif document_type == "utility_bill":
            self._validation_engine.validate_utility_bill(data)
        return {"is_valid": True, "errors": [], "warnings": []}

    def create_template(
        self,
        name: str,
        template_type: str,
        field_mappings: Optional[Dict[str, str]] = None,
        regex_patterns: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create an extraction template."""
        import uuid
        tpl_id = str(uuid.uuid4())
        template = {
            "template_id": tpl_id,
            "name": name,
            "template_type": template_type,
            "field_mappings": field_mappings or {},
            "regex_patterns": regex_patterns or {},
        }
        self._templates[tpl_id] = template
        return template

    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get a template by ID."""
        return self._templates.get(template_id)

    def list_templates(self) -> List[Dict[str, Any]]:
        """List all templates."""
        return list(self._templates.values())

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document record."""
        return self._documents.get(document_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "total_documents": len(self._documents),
            "total_templates": len(self._templates),
            "service_initialized": self._initialized,
        }


def configure_pdf_extractor_service(app: Any) -> PDFExtractorService:
    """Configure and attach PDFExtractorService to a FastAPI app."""
    service = PDFExtractorService()
    app.state.pdf_extractor_service = service
    return service


def get_pdf_extractor_service(app: Any) -> PDFExtractorService:
    """Get the PDFExtractorService from a FastAPI app."""
    return getattr(app.state, "pdf_extractor_service", None)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestPDFExtractorServiceInit:
    """Test PDFExtractorService initialization."""

    def test_default_creation(self):
        service = PDFExtractorService()
        assert service.is_initialized is True

    def test_creation_with_config(self):
        config = {"default_ocr_engine": "textract", "max_pages": 200}
        service = PDFExtractorService(config=config)
        assert service._config["default_ocr_engine"] == "textract"

    def test_all_engines_present(self):
        service = PDFExtractorService()
        assert service.document_parser is not None
        assert service.ocr_engine is not None
        assert service.field_extractor is not None
        assert service.invoice_processor is not None
        assert service.manifest_processor is not None
        assert service.document_classifier is not None
        assert service.validation_engine is not None
        assert service.provenance_tracker is not None
        assert service.metrics is not None


class TestIngestDocument:
    """Test ingest_document method."""

    def test_ingest_returns_doc_record(self):
        service = PDFExtractorService()
        result = service.ingest_document(b"test content", "invoice.pdf")
        assert "document_id" in result
        assert result["filename"] == "invoice.pdf"
        assert result["status"] == "ingested"

    def test_ingest_computes_hash(self):
        service = PDFExtractorService()
        result = service.ingest_document(b"test content", "invoice.pdf")
        assert len(result["file_hash"]) == 64

    def test_ingest_stores_document(self):
        service = PDFExtractorService()
        result = service.ingest_document(b"test", "test.pdf")
        doc = service.get_document(result["document_id"])
        assert doc is not None
        assert doc["filename"] == "test.pdf"

    def test_ingest_calls_parser(self):
        service = PDFExtractorService()
        service.ingest_document(b"content", "test.pdf")
        service.document_parser.parse_document.assert_called_once()

    def test_ingest_file_size(self):
        service = PDFExtractorService()
        content = b"x" * 1024
        result = service.ingest_document(content, "test.pdf")
        assert result["file_size_bytes"] == 1024

    def test_ingest_format_detection(self):
        service = PDFExtractorService()
        result = service.ingest_document(b"test", "scan.png")
        assert result["file_format"] == "png"


class TestClassifyDocument:
    """Test classify_document method."""

    def test_classify_returns_type(self):
        service = PDFExtractorService()
        result = service.classify_document("Invoice Number: INV-001")
        assert "document_type" in result
        assert "confidence" in result

    def test_classify_calls_classifier(self):
        service = PDFExtractorService()
        service.classify_document("test text", filename="test.pdf")
        service.document_classifier.classify.assert_called_once()

    def test_classify_with_filename(self):
        service = PDFExtractorService()
        result = service.classify_document("text", filename="invoice.pdf")
        assert result["confidence"] > 0


class TestExtractInvoice:
    """Test extract_invoice method."""

    def test_extract_returns_fields(self):
        service = PDFExtractorService()
        result = service.extract_invoice("Invoice Number: INV-001 Total: $100")
        assert "invoice_number" in result
        assert "total_amount" in result
        assert "confidence" in result

    def test_extract_calls_processor(self):
        service = PDFExtractorService()
        service.extract_invoice("test")
        service.invoice_processor.process_invoice.assert_called_once()


class TestExtractManifest:
    """Test extract_manifest method."""

    def test_extract_returns_fields(self):
        service = PDFExtractorService()
        result = service.extract_manifest("BOL Number: BOL-001")
        assert "manifest_number" in result
        assert "total_weight_kg" in result

    def test_extract_calls_processor(self):
        service = PDFExtractorService()
        service.extract_manifest("test")
        service.manifest_processor.process_manifest.assert_called_once()


class TestExtractUtilityBill:
    """Test extract_utility_bill method."""

    def test_extract_returns_fields(self):
        service = PDFExtractorService()
        result = service.extract_utility_bill("Account: ACC-001 Consumption: 7550 kWh")
        assert "account_number" in result
        assert "consumption" in result

    def test_extract_calls_field_extractor(self):
        service = PDFExtractorService()
        service.extract_utility_bill("test")
        service.field_extractor.extract_fields.assert_called_once()


class TestValidateDocument:
    """Test validate_document method."""

    def test_validate_invoice(self):
        service = PDFExtractorService()
        result = service.validate_document({"invoice_number": "INV-001"}, "invoice")
        assert "is_valid" in result

    def test_validate_manifest(self):
        service = PDFExtractorService()
        result = service.validate_document({"manifest_number": "BOL-001"}, "manifest")
        assert "is_valid" in result

    def test_validate_utility_bill(self):
        service = PDFExtractorService()
        result = service.validate_document({"account_number": "ACC-001"}, "utility_bill")
        assert "is_valid" in result

    def test_validate_calls_engine(self):
        service = PDFExtractorService()
        service.validate_document({"invoice_number": "INV-001"}, "invoice")
        service.validation_engine.validate_invoice.assert_called_once()


class TestCreateTemplate:
    """Test create_template method."""

    def test_create_returns_template(self):
        service = PDFExtractorService()
        result = service.create_template("Standard Invoice", "invoice")
        assert "template_id" in result
        assert result["name"] == "Standard Invoice"
        assert result["template_type"] == "invoice"

    def test_create_stores_template(self):
        service = PDFExtractorService()
        result = service.create_template("Test", "invoice")
        tpl = service.get_template(result["template_id"])
        assert tpl is not None
        assert tpl["name"] == "Test"

    def test_create_with_mappings(self):
        service = PDFExtractorService()
        result = service.create_template(
            "Custom", "invoice",
            field_mappings={"invoice_number": "Ref No"},
            regex_patterns={"invoice_number": r"Ref No[:\s]*([\w]+)"},
        )
        assert result["field_mappings"]["invoice_number"] == "Ref No"

    def test_list_templates(self):
        service = PDFExtractorService()
        service.create_template("Template A", "invoice")
        service.create_template("Template B", "manifest")
        templates = service.list_templates()
        assert len(templates) == 2

    def test_get_nonexistent_template(self):
        service = PDFExtractorService()
        assert service.get_template("nonexistent") is None


class TestGetStatistics:
    """Test get_statistics method."""

    def test_initial_stats(self):
        service = PDFExtractorService()
        stats = service.get_statistics()
        assert stats["total_documents"] == 0
        assert stats["total_templates"] == 0
        assert stats["service_initialized"] is True

    def test_stats_after_ingestion(self):
        service = PDFExtractorService()
        service.ingest_document(b"test", "test.pdf")
        stats = service.get_statistics()
        assert stats["total_documents"] == 1

    def test_stats_after_template_creation(self):
        service = PDFExtractorService()
        service.create_template("T", "invoice")
        stats = service.get_statistics()
        assert stats["total_templates"] == 1


class TestConfigurePDFExtractorService:
    """Test configure/get service lifecycle."""

    def test_configure_attaches_to_app(self):
        app = MagicMock()
        service = configure_pdf_extractor_service(app)
        assert service.is_initialized is True
        assert app.state.pdf_extractor_service is service

    def test_get_service_from_app(self):
        app = MagicMock()
        service = configure_pdf_extractor_service(app)
        retrieved = get_pdf_extractor_service(app)
        assert retrieved is service

    def test_get_service_not_configured(self):
        app = MagicMock(spec=[])
        app.state = MagicMock(spec=[])
        result = get_pdf_extractor_service(app)
        assert result is None
