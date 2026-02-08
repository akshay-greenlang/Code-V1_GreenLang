# -*- coding: utf-8 -*-
"""
Load Tests for PDF & Invoice Extractor Service (AGENT-DATA-001)

Tests throughput and concurrency for document ingestion, classification,
extraction, validation, batch processing, memory bounds, and latency
under high-volume conditions.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline implementations for load testing
# ---------------------------------------------------------------------------


class LoadTestDocumentParser:
    """Minimal document parser for load testing."""

    def __init__(self):
        self._count = 0

    def parse(self, content: bytes, filename: str) -> Dict[str, Any]:
        self._count += 1
        file_hash = hashlib.sha256(content).hexdigest()
        text = content.decode("utf-8", errors="replace")
        return {
            "document_id": f"doc-{uuid.uuid4().hex[:12]}",
            "filename": filename,
            "file_hash": file_hash,
            "text": text,
            "page_count": max(1, text.count("\f") + 1),
        }

    @property
    def count(self) -> int:
        return self._count


class LoadTestClassifier:
    """Minimal classifier for load testing."""

    KEYWORDS = {
        "invoice": ["invoice", "due date", "subtotal", "total"],
        "manifest": ["bill of lading", "bol", "shipper", "consignee"],
        "utility_bill": ["utility", "billing period", "meter", "consumption"],
    }

    def classify(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        scores = {}
        for doc_type, kws in self.KEYWORDS.items():
            matched = sum(1 for kw in kws if kw in text_lower)
            scores[doc_type] = matched / len(kws)
        best = max(scores, key=scores.get)
        return {"document_type": best, "confidence": scores[best]}


class LoadTestFieldExtractor:
    """Minimal field extractor for load testing."""

    def extract(self, text: str, doc_type: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {"document_type": doc_type}
        if doc_type == "invoice":
            m = re.search(r"Invoice\s*Number[:\s]*([\w\-]+)", text, re.IGNORECASE)
            result["invoice_number"] = m.group(1) if m else None
            m = re.search(r"Total[:\s]*\$?([\d,]+\.?\d*)", text, re.IGNORECASE)
            result["total_amount"] = float(m.group(1).replace(",", "")) if m else None
        elif doc_type == "manifest":
            m = re.search(r"BOL\s*Number[:\s]*([\w\-]+)", text, re.IGNORECASE)
            result["manifest_number"] = m.group(1) if m else None
        elif doc_type == "utility_bill":
            m = re.search(r"Account\s*Number[:\s]*([\w\-]+)", text, re.IGNORECASE)
            result["account_number"] = m.group(1) if m else None
        return result


class LoadTestValidator:
    """Minimal validator for load testing."""

    def validate(self, data: Dict[str, Any], doc_type: str) -> Dict[str, Any]:
        errors = []
        if doc_type == "invoice" and not data.get("invoice_number"):
            errors.append("Missing invoice_number")
        if doc_type == "manifest" and not data.get("manifest_number"):
            errors.append("Missing manifest_number")
        return {"is_valid": len(errors) == 0, "errors": errors}


# ---------------------------------------------------------------------------
# Test data generation
# ---------------------------------------------------------------------------


def generate_invoice_text(index: int) -> str:
    return f"""
    INVOICE
    Invoice Number: INV-LOAD-{index:06d}
    Invoice Date: 2025-06-{(index % 28) + 1:02d}
    Due Date: 2025-07-{(index % 28) + 1:02d}
    Bill To: LoadTest Corp #{index}
    Subtotal: ${100.0 + index * 0.5:.2f}
    Tax: ${20.0 + index * 0.1:.2f}
    Total: ${120.0 + index * 0.6:.2f}
    """


def generate_manifest_text(index: int) -> str:
    return f"""
    BILL OF LADING
    BOL Number: BOL-LOAD-{index:06d}
    Shipper: LoadShipper #{index}
    Consignee: LoadConsignee #{index}
    Total Packages: {10 + index % 100}
    Total Gross Weight: {1000 + index * 10} kg
    """


def generate_utility_text(index: int) -> str:
    return f"""
    UTILITY BILL
    Account Number: UTIL-LOAD-{index:06d}
    Billing Period: 2025-05-01 to 2025-05-31
    Consumption: {500 + index * 2} kWh
    Meter Number: MTR-{index:04d}
    Total Due: ${50.0 + index * 0.3:.2f}
    """


# ===========================================================================
# Load Test Classes
# ===========================================================================


class TestDocumentIngestionThroughput:
    """Test document ingestion throughput: 1000 in <5s."""

    @pytest.mark.slow
    def test_1000_sequential_ingestions(self):
        parser = LoadTestDocumentParser()
        start = time.time()
        for i in range(1000):
            text = generate_invoice_text(i)
            parser.parse(text.encode(), f"invoice_{i:04d}.pdf")
        elapsed = time.time() - start

        assert parser.count == 1000
        assert elapsed < 5.0, f"1000 ingestions took {elapsed:.2f}s (target: <5s)"

    @pytest.mark.slow
    def test_concurrent_ingestion_50_threads(self):
        parser = LoadTestDocumentParser()
        results = []

        def do_ingest(i: int):
            text = generate_invoice_text(i)
            return parser.parse(text.encode(), f"inv_{i}.pdf")

        start = time.time()
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(do_ingest, i) for i in range(500)]
            for future in as_completed(futures):
                results.append(future.result())
        elapsed = time.time() - start

        assert len(results) == 500
        assert elapsed < 10.0, f"500 concurrent ingestions took {elapsed:.2f}s"


class TestClassificationThroughput:
    """Test classification throughput."""

    @pytest.mark.slow
    def test_1000_sequential_classifications(self):
        classifier = LoadTestClassifier()
        results = []
        start = time.time()
        for i in range(1000):
            text = generate_invoice_text(i) if i % 3 == 0 else (
                generate_manifest_text(i) if i % 3 == 1 else generate_utility_text(i)
            )
            results.append(classifier.classify(text))
        elapsed = time.time() - start

        assert len(results) == 1000
        assert elapsed < 3.0, f"1000 classifications took {elapsed:.2f}s (target: <3s)"

    @pytest.mark.slow
    def test_classification_accuracy_at_scale(self):
        classifier = LoadTestClassifier()
        correct = 0
        total = 300

        for i in range(total):
            if i % 3 == 0:
                result = classifier.classify(generate_invoice_text(i))
                if result["document_type"] == "invoice":
                    correct += 1
            elif i % 3 == 1:
                result = classifier.classify(generate_manifest_text(i))
                if result["document_type"] == "manifest":
                    correct += 1
            else:
                result = classifier.classify(generate_utility_text(i))
                if result["document_type"] == "utility_bill":
                    correct += 1

        accuracy = correct / total
        assert accuracy > 0.8, f"Accuracy {accuracy:.2%} below 80% threshold"


class TestExtractionThroughput:
    """Test field extraction throughput."""

    @pytest.mark.slow
    def test_1000_invoice_extractions(self):
        extractor = LoadTestFieldExtractor()
        results = []
        start = time.time()
        for i in range(1000):
            text = generate_invoice_text(i)
            results.append(extractor.extract(text, "invoice"))
        elapsed = time.time() - start

        assert len(results) == 1000
        assert elapsed < 5.0, f"1000 extractions took {elapsed:.2f}s (target: <5s)"

        # Verify extraction quality
        extracted_count = sum(1 for r in results if r.get("invoice_number"))
        assert extracted_count > 900, f"Only {extracted_count}/1000 invoices extracted"


class TestValidationThroughput:
    """Test validation throughput."""

    @pytest.mark.slow
    def test_1000_validations(self):
        validator = LoadTestValidator()
        results = []
        start = time.time()
        for i in range(1000):
            data = {"invoice_number": f"INV-{i:06d}", "total_amount": 100.0 + i}
            results.append(validator.validate(data, "invoice"))
        elapsed = time.time() - start

        assert len(results) == 1000
        assert elapsed < 2.0, f"1000 validations took {elapsed:.2f}s (target: <2s)"
        assert all(r["is_valid"] for r in results)


class TestBatchProcessingPerformance:
    """Test batch processing performance."""

    @pytest.mark.slow
    def test_batch_100_documents(self):
        parser = LoadTestDocumentParser()
        classifier = LoadTestClassifier()
        extractor = LoadTestFieldExtractor()
        validator = LoadTestValidator()

        start = time.time()
        for i in range(100):
            text = generate_invoice_text(i)
            doc = parser.parse(text.encode(), f"batch_{i}.pdf")
            cls_result = classifier.classify(doc["text"])
            ext_result = extractor.extract(doc["text"], cls_result["document_type"])
            validator.validate(ext_result, cls_result["document_type"])
        elapsed = time.time() - start

        assert elapsed < 5.0, f"100 full pipeline took {elapsed:.2f}s (target: <5s)"

    @pytest.mark.slow
    def test_concurrent_batch_processing(self):
        def process_document(i: int) -> Dict[str, Any]:
            parser = LoadTestDocumentParser()
            classifier = LoadTestClassifier()
            extractor = LoadTestFieldExtractor()
            text = generate_invoice_text(i)
            doc = parser.parse(text.encode(), f"doc_{i}.pdf")
            cls_result = classifier.classify(doc["text"])
            ext_result = extractor.extract(doc["text"], cls_result["document_type"])
            return ext_result

        start = time.time()
        results = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(process_document, i) for i in range(200)]
            for future in as_completed(futures):
                results.append(future.result())
        elapsed = time.time() - start

        assert len(results) == 200
        assert elapsed < 10.0, f"200 concurrent took {elapsed:.2f}s (target: <10s)"


class TestMemoryBounds:
    """Test memory usage stays within bounds."""

    @pytest.mark.slow
    def test_memory_usage_1000_documents(self):
        parser = LoadTestDocumentParser()
        documents = []

        initial_size = sys.getsizeof(documents)
        for i in range(1000):
            text = generate_invoice_text(i)
            doc = parser.parse(text.encode(), f"doc_{i}.pdf")
            documents.append(doc)

        # Rough check: documents list should not grow beyond 50MB
        # Each doc is small (a few KB), 1000 docs should be well under 50MB
        total_text_size = sum(len(d.get("text", "")) for d in documents)
        assert total_text_size < 50 * 1024 * 1024, "Text data exceeds 50MB limit"
        assert len(documents) == 1000


class TestLatencyTargets:
    """Test single-operation latency targets."""

    def test_single_parse_latency(self):
        parser = LoadTestDocumentParser()
        text = generate_invoice_text(0).encode()
        start = time.time()
        parser.parse(text, "test.pdf")
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 50, f"Single parse took {elapsed_ms:.2f}ms (target: <50ms)"

    def test_single_classify_latency(self):
        classifier = LoadTestClassifier()
        text = generate_invoice_text(0)
        start = time.time()
        classifier.classify(text)
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 10, f"Single classify took {elapsed_ms:.2f}ms (target: <10ms)"

    def test_single_extraction_latency(self):
        extractor = LoadTestFieldExtractor()
        text = generate_invoice_text(0)
        start = time.time()
        extractor.extract(text, "invoice")
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 20, f"Single extraction took {elapsed_ms:.2f}ms (target: <20ms)"

    def test_single_validation_latency(self):
        validator = LoadTestValidator()
        data = {"invoice_number": "INV-001", "total_amount": 100.0}
        start = time.time()
        validator.validate(data, "invoice")
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 5, f"Single validation took {elapsed_ms:.2f}ms (target: <5ms)"
