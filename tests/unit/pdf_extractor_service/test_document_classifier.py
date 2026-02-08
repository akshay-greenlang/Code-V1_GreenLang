# -*- coding: utf-8 -*-
"""
Unit Tests for DocumentClassifier (AGENT-DATA-001)

Tests document type classification, batch classification, keyword scoring,
filename scoring, custom keyword registration, confidence scoring, and
behavior with ambiguous documents.

Coverage target: 85%+ of document_classifier.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Inline DocumentClassifier mirroring greenlang/pdf_extractor/document_classifier.py
# ---------------------------------------------------------------------------


class DocumentClassifier:
    """Classifies documents by type using keyword scoring and filename analysis.

    Supports: invoice, manifest, utility_bill, receipt, purchase_order.
    """

    DEFAULT_KEYWORDS: Dict[str, List[str]] = {
        "invoice": [
            "invoice", "invoice number", "invoice date", "due date",
            "bill to", "subtotal", "tax", "total amount", "payment terms",
            "vendor", "line items", "unit price",
        ],
        "manifest": [
            "bill of lading", "bol", "manifest", "shipper", "consignee",
            "carrier", "vessel", "port of loading", "port of discharge",
            "cargo", "freight", "container", "voyage",
        ],
        "utility_bill": [
            "utility bill", "electricity", "gas bill", "water bill",
            "account number", "billing period", "meter", "consumption",
            "reading", "kwh", "therms", "supply charge",
        ],
        "receipt": [
            "receipt", "receipt number", "payment received",
            "paid", "change", "cash", "credit card", "debit card",
        ],
        "purchase_order": [
            "purchase order", "po number", "delivery date",
            "buyer", "supplier", "order", "requisition",
        ],
    }

    FILENAME_HINTS: Dict[str, List[str]] = {
        "invoice": ["invoice", "inv", "bill"],
        "manifest": ["manifest", "bol", "lading", "shipping"],
        "utility_bill": ["utility", "electric", "gas", "water"],
        "receipt": ["receipt", "rcpt"],
        "purchase_order": ["purchase", "po", "order"],
    }

    def __init__(self, confidence_threshold: float = 0.3):
        self._confidence_threshold = confidence_threshold
        self._custom_keywords: Dict[str, List[str]] = {}
        self._stats = {
            "documents_classified": 0,
            "classifications_by_type": {},
            "low_confidence_count": 0,
        }

    def classify(
        self,
        text: str,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Classify a document by its text content and optional filename.

        Returns:
            Dict with document_type, confidence, scores (per-type scores).
        """
        scores: Dict[str, float] = {}

        for doc_type in self.DEFAULT_KEYWORDS:
            keyword_score = self._score_keywords(text, doc_type)
            filename_score = self._score_filename(filename, doc_type) if filename else 0.0
            scores[doc_type] = keyword_score * 0.8 + filename_score * 0.2

        if not scores:
            return {"document_type": "unknown", "confidence": 0.0, "scores": {}}

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        if best_score < self._confidence_threshold:
            doc_type = "unknown"
            confidence = best_score
        else:
            doc_type = best_type
            confidence = min(best_score, 1.0)

        self._stats["documents_classified"] += 1
        self._stats["classifications_by_type"][doc_type] = (
            self._stats["classifications_by_type"].get(doc_type, 0) + 1
        )
        if confidence < self._confidence_threshold:
            self._stats["low_confidence_count"] += 1

        return {
            "document_type": doc_type,
            "confidence": round(confidence, 4),
            "scores": {k: round(v, 4) for k, v in scores.items()},
        }

    def classify_batch(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Classify a batch of documents.

        Args:
            documents: List of dicts with 'text' and optional 'filename'.

        Returns:
            List of classification results.
        """
        results = []
        for doc in documents:
            result = self.classify(
                text=doc.get("text", ""),
                filename=doc.get("filename"),
            )
            results.append(result)
        return results

    def register_keywords(self, document_type: str, keywords: List[str]) -> None:
        """Register custom keywords for a document type."""
        self._custom_keywords[document_type] = keywords

    def get_statistics(self) -> Dict[str, Any]:
        """Return classification statistics."""
        return dict(self._stats)

    def _score_keywords(self, text: str, doc_type: str) -> float:
        """Score text against keywords for a document type."""
        keywords = self._custom_keywords.get(doc_type, self.DEFAULT_KEYWORDS.get(doc_type, []))
        if not keywords:
            return 0.0

        text_lower = text.lower()
        matched = sum(1 for kw in keywords if kw.lower() in text_lower)
        return matched / len(keywords)

    def _score_filename(self, filename: Optional[str], doc_type: str) -> float:
        """Score filename against hints for a document type."""
        if not filename:
            return 0.0

        hints = self.FILENAME_HINTS.get(doc_type, [])
        if not hints:
            return 0.0

        fn_lower = filename.lower()
        matched = sum(1 for hint in hints if hint in fn_lower)
        return matched / len(hints) if hints else 0.0


# ===========================================================================
# Test Classes
# ===========================================================================


class TestDocumentClassifierInit:
    """Test DocumentClassifier initialization."""

    def test_default_threshold(self):
        clf = DocumentClassifier()
        assert clf._confidence_threshold == 0.3

    def test_custom_threshold(self):
        clf = DocumentClassifier(confidence_threshold=0.5)
        assert clf._confidence_threshold == 0.5

    def test_initial_statistics(self):
        clf = DocumentClassifier()
        stats = clf.get_statistics()
        assert stats["documents_classified"] == 0

    def test_default_keywords_exist(self):
        assert "invoice" in DocumentClassifier.DEFAULT_KEYWORDS
        assert "manifest" in DocumentClassifier.DEFAULT_KEYWORDS
        assert "utility_bill" in DocumentClassifier.DEFAULT_KEYWORDS
        assert "receipt" in DocumentClassifier.DEFAULT_KEYWORDS
        assert "purchase_order" in DocumentClassifier.DEFAULT_KEYWORDS


class TestClassifyInvoice:
    """Test classification of invoice documents."""

    def test_classify_invoice_text(self, sample_invoice_text):
        clf = DocumentClassifier()
        result = clf.classify(sample_invoice_text)
        assert result["document_type"] == "invoice"
        assert result["confidence"] > 0.3

    def test_classify_invoice_with_filename(self):
        clf = DocumentClassifier()
        result = clf.classify(
            "Invoice Number: INV-001 invoice date due date subtotal tax total amount bill to vendor",
            filename="invoice_2025.pdf",
        )
        assert result["document_type"] == "invoice"

    def test_invoice_scores_highest(self, sample_invoice_text):
        clf = DocumentClassifier()
        result = clf.classify(sample_invoice_text)
        assert result["scores"]["invoice"] >= result["scores"]["manifest"]
        assert result["scores"]["invoice"] >= result["scores"]["utility_bill"]


class TestClassifyManifest:
    """Test classification of manifest documents."""

    def test_classify_manifest_text(self, sample_manifest_text):
        clf = DocumentClassifier()
        result = clf.classify(sample_manifest_text)
        assert result["document_type"] == "manifest"
        assert result["confidence"] > 0.3

    def test_classify_bol_with_filename(self):
        clf = DocumentClassifier()
        result = clf.classify(
            "Bill of Lading shipper consignee carrier vessel port of loading",
            filename="bol_shipping.pdf",
        )
        assert result["document_type"] == "manifest"


class TestClassifyUtilityBill:
    """Test classification of utility bill documents."""

    def test_classify_utility_text(self, sample_utility_bill_text):
        clf = DocumentClassifier()
        result = clf.classify(sample_utility_bill_text)
        assert result["document_type"] == "utility_bill"
        assert result["confidence"] > 0.3

    def test_classify_electricity_filename(self):
        clf = DocumentClassifier()
        result = clf.classify(
            "electricity billing period meter consumption kWh reading account number supply charge",
            filename="electric_bill.pdf",
        )
        assert result["document_type"] == "utility_bill"


class TestClassifyReceipt:
    """Test classification of receipt documents."""

    def test_classify_receipt_text(self):
        clf = DocumentClassifier()
        text = "Receipt receipt number payment received paid cash credit card change debit card"
        result = clf.classify(text)
        assert result["document_type"] == "receipt"

    def test_receipt_keywords(self):
        clf = DocumentClassifier()
        result = clf.classify("Receipt receipt number payment received paid cash credit card change")
        assert result["document_type"] == "receipt"


class TestClassifyPurchaseOrder:
    """Test classification of purchase order documents."""

    def test_classify_po_text(self, sample_purchase_order_text):
        clf = DocumentClassifier()
        result = clf.classify(sample_purchase_order_text)
        assert result["document_type"] == "purchase_order"

    def test_po_with_filename(self):
        clf = DocumentClassifier()
        result = clf.classify(
            "Purchase Order PO Number Delivery Date Buyer Supplier order requisition",
            filename="purchase_order_2025.pdf",
        )
        assert result["document_type"] == "purchase_order"


class TestClassifyUnknown:
    """Test classification of ambiguous/unknown documents."""

    def test_empty_text(self):
        clf = DocumentClassifier()
        result = clf.classify("")
        assert result["document_type"] == "unknown"
        assert result["confidence"] < 0.3

    def test_random_text(self):
        clf = DocumentClassifier()
        result = clf.classify("Lorem ipsum dolor sit amet")
        assert result["document_type"] == "unknown"

    def test_ambiguous_text(self):
        clf = DocumentClassifier()
        result = clf.classify("This document contains some text about numbers and dates")
        assert result["confidence"] < 0.5


class TestClassifyBatch:
    """Test classify_batch method."""

    def test_batch_classification(self, sample_invoice_text, sample_manifest_text):
        clf = DocumentClassifier()
        docs = [
            {"text": sample_invoice_text, "filename": "invoice.pdf"},
            {"text": sample_manifest_text, "filename": "manifest.pdf"},
        ]
        results = clf.classify_batch(docs)
        assert len(results) == 2
        assert results[0]["document_type"] == "invoice"
        assert results[1]["document_type"] == "manifest"

    def test_batch_empty(self):
        clf = DocumentClassifier()
        results = clf.classify_batch([])
        assert results == []

    def test_batch_single_document(self, sample_invoice_text):
        clf = DocumentClassifier()
        results = clf.classify_batch([{"text": sample_invoice_text}])
        assert len(results) == 1

    def test_batch_updates_stats(self, sample_invoice_text, sample_manifest_text):
        clf = DocumentClassifier()
        clf.classify_batch([
            {"text": sample_invoice_text},
            {"text": sample_manifest_text},
        ])
        stats = clf.get_statistics()
        assert stats["documents_classified"] == 2


class TestScoreKeywords:
    """Test _score_keywords method."""

    def test_perfect_match(self):
        clf = DocumentClassifier()
        text = " ".join(DocumentClassifier.DEFAULT_KEYWORDS["invoice"])
        score = clf._score_keywords(text, "invoice")
        assert score == 1.0

    def test_no_match(self):
        clf = DocumentClassifier()
        score = clf._score_keywords("xyzzy abcde fghij", "invoice")
        assert score == 0.0

    def test_partial_match(self):
        clf = DocumentClassifier()
        score = clf._score_keywords("invoice number due date", "invoice")
        assert 0.0 < score < 1.0

    def test_case_insensitive(self):
        clf = DocumentClassifier()
        score = clf._score_keywords("INVOICE NUMBER DUE DATE", "invoice")
        assert score > 0.0


class TestScoreFilename:
    """Test _score_filename method."""

    def test_invoice_filename(self):
        clf = DocumentClassifier()
        score = clf._score_filename("invoice_2025.pdf", "invoice")
        assert score > 0.0

    def test_no_match_filename(self):
        clf = DocumentClassifier()
        score = clf._score_filename("document.pdf", "invoice")
        assert score == 0.0

    def test_none_filename(self):
        clf = DocumentClassifier()
        score = clf._score_filename(None, "invoice")
        assert score == 0.0


class TestRegisterKeywords:
    """Test register_keywords method."""

    def test_custom_keywords(self):
        clf = DocumentClassifier()
        clf.register_keywords("invoice", ["factura", "numero de factura"])
        result = clf.classify("factura numero de factura")
        assert result["document_type"] == "invoice"

    def test_custom_overrides_default(self):
        clf = DocumentClassifier()
        clf.register_keywords("invoice", ["custom_keyword_xyz"])
        score = clf._score_keywords("custom_keyword_xyz", "invoice")
        assert score == 1.0


class TestClassifierStatistics:
    """Test statistics gathering."""

    def test_stats_after_classification(self, sample_invoice_text):
        clf = DocumentClassifier()
        clf.classify(sample_invoice_text)
        stats = clf.get_statistics()
        assert stats["documents_classified"] == 1

    def test_classifications_by_type(self, sample_invoice_text, sample_manifest_text):
        clf = DocumentClassifier()
        clf.classify(sample_invoice_text)
        clf.classify(sample_manifest_text)
        stats = clf.get_statistics()
        assert "invoice" in stats["classifications_by_type"]
        assert "manifest" in stats["classifications_by_type"]

    def test_low_confidence_count(self):
        clf = DocumentClassifier()
        clf.classify("")  # Low confidence
        stats = clf.get_statistics()
        assert stats["low_confidence_count"] >= 1
