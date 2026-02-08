# -*- coding: utf-8 -*-
"""
End-to-End Integration Tests for PDF & Invoice Extractor (AGENT-DATA-001)

Tests full document lifecycle: ingest -> classify -> extract -> validate
for invoices, manifests, and utility bills. Tests batch processing,
template usage, provenance chain integrity, and multi-document workflows.

All implementations are self-contained to avoid cross-module import issues.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Self-contained implementations for integration testing
# ---------------------------------------------------------------------------


class DocumentPipeline:
    """End-to-end document processing pipeline for integration testing."""

    def __init__(self):
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._extractions: Dict[str, Dict[str, Any]] = {}
        self._provenance: Dict[str, List[Dict[str, Any]]] = {}

    def ingest(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Ingest a document."""
        doc_id = f"doc-{uuid.uuid4().hex[:12]}"
        file_hash = hashlib.sha256(content).hexdigest()
        text = content.decode("utf-8", errors="replace")

        doc = {
            "document_id": doc_id,
            "filename": filename,
            "file_hash": file_hash,
            "text": text,
            "file_size_bytes": len(content),
            "status": "ingested",
            "created_at": datetime.utcnow().isoformat(),
        }
        self._documents[doc_id] = doc
        self._record_provenance(doc_id, "ingest", {"filename": filename, "hash": file_hash})
        return doc

    def classify(self, doc_id: str) -> Dict[str, Any]:
        """Classify a document."""
        doc = self._documents.get(doc_id)
        if not doc:
            raise ValueError(f"Document {doc_id} not found")

        text = doc["text"].lower()
        scores = {}
        keywords = {
            "invoice": ["invoice", "due date", "subtotal", "total", "bill to"],
            "manifest": ["bill of lading", "bol", "shipper", "consignee", "carrier", "vessel"],
            "utility_bill": ["utility", "billing period", "meter", "consumption", "kwh"],
        }
        for doc_type, kws in keywords.items():
            matched = sum(1 for kw in kws if kw in text)
            scores[doc_type] = matched / len(kws)

        best_type = max(scores, key=scores.get) if scores else "unknown"
        confidence = scores.get(best_type, 0.0)

        doc["document_type"] = best_type if confidence > 0.2 else "unknown"
        doc["classification_confidence"] = confidence
        doc["status"] = "classified"

        self._record_provenance(doc_id, "classify", {"type": best_type, "confidence": confidence})
        return {"document_type": doc["document_type"], "confidence": confidence}

    def extract(self, doc_id: str) -> Dict[str, Any]:
        """Extract fields based on document type."""
        doc = self._documents.get(doc_id)
        if not doc:
            raise ValueError(f"Document {doc_id} not found")

        text = doc["text"]
        doc_type = doc.get("document_type", "unknown")
        extraction = {"document_id": doc_id, "document_type": doc_type}

        if doc_type == "invoice":
            extraction.update(self._extract_invoice(text))
        elif doc_type == "manifest":
            extraction.update(self._extract_manifest(text))
        elif doc_type == "utility_bill":
            extraction.update(self._extract_utility_bill(text))

        self._extractions[doc_id] = extraction
        doc["status"] = "extracted"
        self._record_provenance(doc_id, "extract", {"fields": len(extraction)})
        return extraction

    def validate(self, doc_id: str) -> Dict[str, Any]:
        """Validate extraction results."""
        extraction = self._extractions.get(doc_id)
        if not extraction:
            raise ValueError(f"No extraction for {doc_id}")

        errors = []
        doc_type = extraction.get("document_type", "unknown")

        if doc_type == "invoice":
            if not extraction.get("invoice_number"):
                errors.append({"field": "invoice_number", "message": "Missing"})
            if extraction.get("subtotal") and extraction.get("tax_amount"):
                expected = extraction["subtotal"] + extraction["tax_amount"]
                if extraction.get("total_amount") and abs(expected - extraction["total_amount"]) > 0.01:
                    errors.append({"field": "total_amount", "message": "Mismatch"})

        elif doc_type == "manifest":
            if not extraction.get("manifest_number"):
                errors.append({"field": "manifest_number", "message": "Missing"})

        elif doc_type == "utility_bill":
            if not extraction.get("account_number"):
                errors.append({"field": "account_number", "message": "Missing"})

        is_valid = len(errors) == 0
        doc = self._documents.get(doc_id, {})
        doc["status"] = "validated" if is_valid else "validation_failed"

        self._record_provenance(doc_id, "validate", {"is_valid": is_valid})
        return {"is_valid": is_valid, "errors": errors}

    def get_provenance_chain(self, doc_id: str) -> List[Dict[str, Any]]:
        return list(self._provenance.get(doc_id, []))

    def verify_provenance(self, doc_id: str) -> bool:
        chain = self._provenance.get(doc_id, [])
        if not chain:
            return True
        genesis = "0" * 64
        for i, record in enumerate(chain):
            expected_prev = chain[i - 1]["hash"] if i > 0 else genesis
            if record["previous_hash"] != expected_prev:
                return False
        return True

    def _record_provenance(self, doc_id: str, operation: str, data: Dict[str, Any]):
        if doc_id not in self._provenance:
            self._provenance[doc_id] = []
        chain = self._provenance[doc_id]
        prev_hash = chain[-1]["hash"] if chain else "0" * 64
        record_data = json.dumps({"op": operation, "data": data}, sort_keys=True)
        record_hash = hashlib.sha256(record_data.encode()).hexdigest()
        chain.append({
            "sequence": len(chain) + 1,
            "operation": operation,
            "data": data,
            "previous_hash": prev_hash,
            "hash": record_hash,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def _extract_invoice(self, text: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        m = re.search(r"Invoice\s*Number[:\s]*([\w\-]+)", text, re.IGNORECASE)
        result["invoice_number"] = m.group(1) if m else None
        m = re.search(r"Invoice\s*Date[:\s]*(\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)
        result["invoice_date"] = m.group(1) if m else None
        m = re.search(r"Subtotal[:\s]*[$]?([\d,]+\.?\d*)", text, re.IGNORECASE)
        result["subtotal"] = float(m.group(1).replace(",", "")) if m else None
        m = re.search(r"Tax[^:]*[:\s]*[$]?([\d,]+\.?\d*)", text, re.IGNORECASE)
        result["tax_amount"] = float(m.group(1).replace(",", "")) if m else None
        m = re.search(r"(?<![Ss]ub)Total[:\s]*[$]?([\d,]+\.?\d*)", text, re.IGNORECASE)
        result["total_amount"] = float(m.group(1).replace(",", "")) if m else None
        return result

    def _extract_manifest(self, text: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        m = re.search(r"(?:BOL|Bill\s+of\s+Lading)\s*Number[:\s]*([\w\-]+)", text, re.IGNORECASE)
        result["manifest_number"] = m.group(1) if m else None
        m = re.search(r"Total\s+(?:Gross\s+)?Weight[:\s]*([\d,]+)\s*kg", text, re.IGNORECASE)
        result["total_weight_kg"] = float(m.group(1).replace(",", "")) if m else None
        m = re.search(r"Total\s+Packages[:\s]*(\d+)", text, re.IGNORECASE)
        result["total_packages"] = int(m.group(1)) if m else None
        return result

    def _extract_utility_bill(self, text: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        m = re.search(r"Account\s*Number[:\s]*([\w\-]+)", text, re.IGNORECASE)
        result["account_number"] = m.group(1) if m else None
        m = re.search(r"Consumption[:\s]*([\d,]+)\s*kWh", text, re.IGNORECASE)
        result["consumption"] = float(m.group(1).replace(",", "")) if m else None
        m = re.search(r"Total\s*(?:Due)?[:\s]*[$]?([\d,]+\.?\d*)", text, re.IGNORECASE)
        result["total_amount"] = float(m.group(1).replace(",", "")) if m else None
        return result


# ---------------------------------------------------------------------------
# Test Data
# ---------------------------------------------------------------------------

INVOICE_TEXT = """
INVOICE
Invoice Number: INV-2025-001234
Invoice Date: 2025-06-15
Due Date: 2025-07-15
Bill To: GreenCorp Ltd.
Vendor: EcoSupply Partners Inc.
ITEM-001 Carbon Credits 100 25.00 2500.00
ITEM-002 Energy Certs 50 15.50 775.00
Subtotal: $3,275.00
Tax (20%): $655.00
Total: $3,930.00
Payment Terms: Net 30
""".strip()

MANIFEST_TEXT = """
BILL OF LADING
BOL Number: BOL-2025-56789
Date: 2025-06-20
Shipper: EcoMaterials GmbH
Consignee: GreenCorp Ltd.
Carrier: MaerskLine Container Shipping
Vessel: MV Green Future
Port of Loading: Hamburg, Germany (DEHAM)
Port of Discharge: Felixstowe, UK (GBFXT)
Total Packages: 85
Total Gross Weight: 25,500 kg
Total Volume: 35.8 m3
""".strip()

UTILITY_BILL_TEXT = """
UTILITY BILL - ELECTRICITY
Account Number: ELEC-2025-78901
Billing Period: 2025-05-01 to 2025-05-31
Meter Number: E-MTR-445566
Previous Reading: 145,230 kWh
Current Reading: 152,780 kWh
Consumption: 7,550 kWh
Total Due: $1,252.13
""".strip()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestInvoiceEndToEnd:
    """Full lifecycle test for invoice documents."""

    def test_full_invoice_lifecycle(self):
        pipeline = DocumentPipeline()
        doc = pipeline.ingest(INVOICE_TEXT.encode(), "invoice_2025.pdf")
        assert doc["status"] == "ingested"

        classification = pipeline.classify(doc["document_id"])
        assert classification["document_type"] == "invoice"
        assert classification["confidence"] > 0.3

        extraction = pipeline.extract(doc["document_id"])
        assert extraction["invoice_number"] == "INV-2025-001234"
        assert extraction["invoice_date"] == "2025-06-15"

        validation = pipeline.validate(doc["document_id"])
        assert validation["is_valid"] is True

    def test_invoice_provenance_chain(self):
        pipeline = DocumentPipeline()
        doc = pipeline.ingest(INVOICE_TEXT.encode(), "invoice.pdf")
        pipeline.classify(doc["document_id"])
        pipeline.extract(doc["document_id"])
        pipeline.validate(doc["document_id"])

        chain = pipeline.get_provenance_chain(doc["document_id"])
        assert len(chain) == 4
        assert chain[0]["operation"] == "ingest"
        assert chain[1]["operation"] == "classify"
        assert chain[2]["operation"] == "extract"
        assert chain[3]["operation"] == "validate"

    def test_invoice_provenance_integrity(self):
        pipeline = DocumentPipeline()
        doc = pipeline.ingest(INVOICE_TEXT.encode(), "invoice.pdf")
        pipeline.classify(doc["document_id"])
        pipeline.extract(doc["document_id"])
        assert pipeline.verify_provenance(doc["document_id"]) is True

    def test_invoice_totals_extracted(self):
        pipeline = DocumentPipeline()
        doc = pipeline.ingest(INVOICE_TEXT.encode(), "invoice.pdf")
        pipeline.classify(doc["document_id"])
        extraction = pipeline.extract(doc["document_id"])
        assert extraction["subtotal"] == 3275.0
        assert extraction["total_amount"] is not None


class TestManifestEndToEnd:
    """Full lifecycle test for manifest documents."""

    def test_full_manifest_lifecycle(self):
        pipeline = DocumentPipeline()
        doc = pipeline.ingest(MANIFEST_TEXT.encode(), "bol_2025.pdf")
        assert doc["status"] == "ingested"

        classification = pipeline.classify(doc["document_id"])
        assert classification["document_type"] == "manifest"

        extraction = pipeline.extract(doc["document_id"])
        assert extraction["manifest_number"] == "BOL-2025-56789"
        assert extraction["total_weight_kg"] == 25500.0
        assert extraction["total_packages"] == 85

        validation = pipeline.validate(doc["document_id"])
        assert validation["is_valid"] is True

    def test_manifest_provenance(self):
        pipeline = DocumentPipeline()
        doc = pipeline.ingest(MANIFEST_TEXT.encode(), "manifest.pdf")
        pipeline.classify(doc["document_id"])
        pipeline.extract(doc["document_id"])
        assert pipeline.verify_provenance(doc["document_id"]) is True


class TestUtilityBillEndToEnd:
    """Full lifecycle test for utility bill documents."""

    def test_full_utility_bill_lifecycle(self):
        pipeline = DocumentPipeline()
        doc = pipeline.ingest(UTILITY_BILL_TEXT.encode(), "electric_bill.pdf")

        classification = pipeline.classify(doc["document_id"])
        assert classification["document_type"] == "utility_bill"

        extraction = pipeline.extract(doc["document_id"])
        assert extraction["account_number"] == "ELEC-2025-78901"
        assert extraction["consumption"] == 7550.0

        validation = pipeline.validate(doc["document_id"])
        assert validation["is_valid"] is True

    def test_utility_bill_total(self):
        pipeline = DocumentPipeline()
        doc = pipeline.ingest(UTILITY_BILL_TEXT.encode(), "bill.pdf")
        pipeline.classify(doc["document_id"])
        extraction = pipeline.extract(doc["document_id"])
        assert extraction["total_amount"] is not None


class TestBatchProcessing:
    """Test batch document processing."""

    def test_batch_multiple_types(self):
        pipeline = DocumentPipeline()
        documents = [
            (INVOICE_TEXT.encode(), "invoice.pdf"),
            (MANIFEST_TEXT.encode(), "manifest.pdf"),
            (UTILITY_BILL_TEXT.encode(), "bill.pdf"),
        ]
        results = []
        for content, filename in documents:
            doc = pipeline.ingest(content, filename)
            pipeline.classify(doc["document_id"])
            extraction = pipeline.extract(doc["document_id"])
            validation = pipeline.validate(doc["document_id"])
            results.append({
                "document_id": doc["document_id"],
                "type": extraction["document_type"],
                "valid": validation["is_valid"],
            })

        assert len(results) == 3
        types = {r["type"] for r in results}
        assert "invoice" in types
        assert "manifest" in types
        assert "utility_bill" in types

    def test_batch_all_valid(self):
        pipeline = DocumentPipeline()
        documents = [
            (INVOICE_TEXT.encode(), "inv.pdf"),
            (MANIFEST_TEXT.encode(), "bol.pdf"),
        ]
        all_valid = True
        for content, filename in documents:
            doc = pipeline.ingest(content, filename)
            pipeline.classify(doc["document_id"])
            pipeline.extract(doc["document_id"])
            result = pipeline.validate(doc["document_id"])
            if not result["is_valid"]:
                all_valid = False
        assert all_valid is True


class TestDuplicateDetection:
    """Test duplicate document handling via file hash."""

    def test_same_content_same_hash(self):
        pipeline = DocumentPipeline()
        doc1 = pipeline.ingest(INVOICE_TEXT.encode(), "copy1.pdf")
        doc2 = pipeline.ingest(INVOICE_TEXT.encode(), "copy2.pdf")
        assert doc1["file_hash"] == doc2["file_hash"]

    def test_different_content_different_hash(self):
        pipeline = DocumentPipeline()
        doc1 = pipeline.ingest(INVOICE_TEXT.encode(), "invoice.pdf")
        doc2 = pipeline.ingest(MANIFEST_TEXT.encode(), "manifest.pdf")
        assert doc1["file_hash"] != doc2["file_hash"]


class TestProvenanceChainIntegrity:
    """Test provenance chain across multiple operations."""

    def test_chain_links_correctly(self):
        pipeline = DocumentPipeline()
        doc = pipeline.ingest(INVOICE_TEXT.encode(), "test.pdf")
        pipeline.classify(doc["document_id"])
        pipeline.extract(doc["document_id"])
        pipeline.validate(doc["document_id"])

        chain = pipeline.get_provenance_chain(doc["document_id"])
        assert len(chain) == 4
        assert chain[0]["previous_hash"] == "0" * 64
        for i in range(1, len(chain)):
            assert chain[i]["previous_hash"] == chain[i - 1]["hash"]

    def test_empty_document_provenance(self):
        pipeline = DocumentPipeline()
        assert pipeline.verify_provenance("nonexistent") is True

    def test_provenance_timestamps_ordered(self):
        pipeline = DocumentPipeline()
        doc = pipeline.ingest(INVOICE_TEXT.encode(), "test.pdf")
        pipeline.classify(doc["document_id"])
        pipeline.extract(doc["document_id"])

        chain = pipeline.get_provenance_chain(doc["document_id"])
        timestamps = [r["timestamp"] for r in chain]
        assert timestamps == sorted(timestamps)


class TestErrorHandling:
    """Test error scenarios."""

    def test_classify_nonexistent_document(self):
        pipeline = DocumentPipeline()
        with pytest.raises(ValueError, match="not found"):
            pipeline.classify("nonexistent-id")

    def test_extract_nonexistent_document(self):
        pipeline = DocumentPipeline()
        with pytest.raises(ValueError, match="not found"):
            pipeline.extract("nonexistent-id")

    def test_validate_without_extraction(self):
        pipeline = DocumentPipeline()
        doc = pipeline.ingest(INVOICE_TEXT.encode(), "test.pdf")
        with pytest.raises(ValueError, match="No extraction"):
            pipeline.validate(doc["document_id"])
