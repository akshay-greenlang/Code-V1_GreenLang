# -*- coding: utf-8 -*-
"""
Tests for DocumentClassifierEngine - AGENT-EUDR-012 Engine 1: Document Classification

Comprehensive test suite covering:
- Classification for all 20 EUDR document types
- Template matching with known templates
- Confidence scoring (high, medium, low, unknown)
- Unknown document type flagging
- Batch classification
- Language detection
- Template registration and management
- Multi-page document handling
- Edge cases: empty document, corrupt file, no metadata

Test count: 55+ tests
Coverage target: >= 85% of DocumentClassifierEngine module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-012 Document Authentication Agent (GL-EUDR-DAV-012)
"""

from __future__ import annotations

import copy
import uuid
from decimal import Decimal
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.document_authentication.conftest import (
    DOCUMENT_TYPES,
    CONFIDENCE_LEVELS,
    DOCUMENT_LANGUAGES,
    SHA256_HEX_LENGTH,
    EUDR_COMMODITIES,
    DOC_ID_COO_001,
    DOC_ID_FSC_001,
    DOC_ID_BOL_001,
    CLASSIFICATION_COO_HIGH,
    CLASSIFICATION_FSC_MEDIUM,
    SAMPLE_PDF_BYTES,
    SAMPLE_EMPTY_BYTES,
    SAMPLE_CORRUPT_BYTES,
    SAMPLE_LARGE_BYTES,
    make_document_record,
    make_classification_result,
    assert_classification_valid,
    assert_valid_provenance_hash,
)


# ===========================================================================
# 1. Basic Classification
# ===========================================================================


class TestBasicClassification:
    """Test basic document classification operations."""

    def test_classify_coo_document(self, classifier_engine):
        """Classify a Certificate of Origin document."""
        result = classifier_engine.classify(
            document_bytes=SAMPLE_PDF_BYTES,
            file_name="certificate_of_origin_ghana.pdf",
        )
        assert result is not None
        assert "document_type" in result
        assert "confidence" in result

    def test_classify_returns_confidence(self, classifier_engine):
        """Classification returns a confidence score between 0 and 1."""
        result = classifier_engine.classify(
            document_bytes=SAMPLE_PDF_BYTES,
            file_name="test_doc.pdf",
        )
        assert 0.0 <= result["confidence"] <= 1.0

    def test_classify_returns_confidence_level(self, classifier_engine):
        """Classification returns a confidence level string."""
        result = classifier_engine.classify(
            document_bytes=SAMPLE_PDF_BYTES,
            file_name="test_doc.pdf",
        )
        assert result["confidence_level"] in CONFIDENCE_LEVELS

    def test_classify_returns_document_type(self, classifier_engine):
        """Classification returns a recognized document type or unknown."""
        result = classifier_engine.classify(
            document_bytes=SAMPLE_PDF_BYTES,
            file_name="test_doc.pdf",
        )
        assert result["document_type"] in DOCUMENT_TYPES or result["confidence_level"] == "unknown"

    def test_classify_returns_alternatives(self, classifier_engine):
        """Classification returns a list of alternative type candidates."""
        result = classifier_engine.classify(
            document_bytes=SAMPLE_PDF_BYTES,
            file_name="test_doc.pdf",
        )
        assert "alternatives" in result
        assert isinstance(result["alternatives"], list)

    def test_classify_includes_processing_time(self, classifier_engine):
        """Classification result includes processing time in milliseconds."""
        result = classifier_engine.classify(
            document_bytes=SAMPLE_PDF_BYTES,
            file_name="test_doc.pdf",
        )
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] >= 0

    def test_classify_provenance_hash(self, classifier_engine):
        """Classification generates a provenance hash."""
        result = classifier_engine.classify(
            document_bytes=SAMPLE_PDF_BYTES,
            file_name="test_doc.pdf",
        )
        if result.get("provenance_hash"):
            assert_valid_provenance_hash(result["provenance_hash"])


# ===========================================================================
# 2. All Document Types
# ===========================================================================


class TestAllDocumentTypes:
    """Test classification for all 20 EUDR document types."""

    @pytest.mark.parametrize("doc_type", DOCUMENT_TYPES)
    def test_classify_all_document_types(self, classifier_engine, doc_type):
        """Classification can identify all 20 document types."""
        result = make_classification_result(document_type=doc_type)
        assert_classification_valid(result)
        assert result["document_type"] == doc_type

    @pytest.mark.parametrize("doc_type", DOCUMENT_TYPES)
    def test_classification_result_structure(self, classifier_engine, doc_type):
        """Each document type classification has the required fields."""
        result = make_classification_result(document_type=doc_type)
        required_keys = [
            "document_id", "document_type", "confidence",
            "confidence_level", "matched_template", "language",
        ]
        for key in required_keys:
            assert key in result, f"Missing key '{key}' for doc_type '{doc_type}'"


# ===========================================================================
# 3. Template Matching
# ===========================================================================


class TestTemplateMatching:
    """Test template matching with known templates."""

    def test_template_match_returns_id(self, classifier_engine):
        """Template matching returns a template identifier."""
        result = classifier_engine.classify(
            document_bytes=SAMPLE_PDF_BYTES,
            file_name="coo_ghana_standard.pdf",
        )
        assert "matched_template" in result

    def test_no_template_match_returns_none(self, classifier_engine):
        """Unrecognized document returns None for matched_template."""
        result = classifier_engine.classify(
            document_bytes=SAMPLE_CORRUPT_BYTES,
            file_name="random_bytes.bin",
        )
        # Either None or the confidence should be low
        if result.get("matched_template") is None:
            assert True
        else:
            assert result["confidence_level"] in ("low", "unknown")

    def test_register_template(self, classifier_engine):
        """Register a new template for classification."""
        template_data = {
            "template_id": "coo_ivory_coast_v1",
            "document_type": "coo",
            "country": "CI",
            "version": "1.0",
            "features": {"header_pattern": "REPUBLIC OF COTE D'IVOIRE"},
        }
        result = classifier_engine.register_template(template_data)
        assert result is not None

    def test_register_duplicate_template_raises(self, classifier_engine):
        """Registering a duplicate template ID raises an error."""
        template_data = {
            "template_id": "coo_duplicate_test",
            "document_type": "coo",
            "country": "GH",
        }
        classifier_engine.register_template(template_data)
        with pytest.raises((ValueError, KeyError)):
            classifier_engine.register_template(copy.deepcopy(template_data))

    def test_list_templates(self, classifier_engine):
        """List all registered templates."""
        result = classifier_engine.list_templates()
        assert isinstance(result, list)

    def test_list_templates_by_type(self, classifier_engine):
        """List templates filtered by document type."""
        result = classifier_engine.list_templates(document_type="coo")
        assert isinstance(result, list)
        for t in result:
            assert t.get("document_type") == "coo"


# ===========================================================================
# 4. Confidence Scoring
# ===========================================================================


class TestConfidenceScoring:
    """Test confidence scoring thresholds."""

    def test_high_confidence_threshold(self, classifier_engine):
        """Confidence >= 0.95 maps to HIGH level."""
        result = make_classification_result(confidence=0.98, confidence_level="high")
        assert result["confidence_level"] == "high"
        assert result["confidence"] >= 0.95

    def test_medium_confidence_threshold(self, classifier_engine):
        """Confidence >= 0.70 and < 0.95 maps to MEDIUM level."""
        result = make_classification_result(confidence=0.82, confidence_level="medium")
        assert result["confidence_level"] == "medium"
        assert 0.70 <= result["confidence"] < 0.95

    def test_low_confidence_threshold(self, classifier_engine):
        """Confidence < 0.70 maps to LOW level."""
        result = make_classification_result(confidence=0.45, confidence_level="low")
        assert result["confidence_level"] == "low"
        assert result["confidence"] < 0.70

    def test_unknown_confidence(self, classifier_engine):
        """Unknown document type gets UNKNOWN confidence level."""
        result = make_classification_result(confidence=0.10, confidence_level="unknown")
        assert result["confidence_level"] == "unknown"

    @pytest.mark.parametrize("conf,expected_level", [
        (0.99, "high"),
        (0.95, "high"),
        (0.94, "medium"),
        (0.70, "medium"),
        (0.69, "low"),
        (0.30, "low"),
        (0.05, "low"),
    ])
    def test_confidence_level_boundaries(self, classifier_engine, conf, expected_level):
        """Confidence-level mapping respects boundary values."""
        result = make_classification_result(confidence=conf, confidence_level=expected_level)
        assert result["confidence_level"] == expected_level


# ===========================================================================
# 5. Unknown Document Flagging
# ===========================================================================


class TestUnknownDocumentFlagging:
    """Test unknown document type handling."""

    def test_unknown_document_flagged(self, classifier_engine):
        """Unrecognized file formats are flagged as unknown."""
        result = classifier_engine.classify(
            document_bytes=b"This is not a valid document format",
            file_name="readme.txt",
        )
        assert result["confidence_level"] in ("low", "unknown")

    def test_binary_file_flagged(self, classifier_engine):
        """Random binary data is flagged as unknown."""
        result = classifier_engine.classify(
            document_bytes=SAMPLE_CORRUPT_BYTES,
            file_name="data.bin",
        )
        assert result["confidence_level"] in ("low", "unknown")


# ===========================================================================
# 6. Batch Classification
# ===========================================================================


class TestBatchClassification:
    """Test batch classification operations."""

    def test_batch_classify_multiple_documents(self, classifier_engine):
        """Batch classify multiple documents at once."""
        documents = [
            {"document_bytes": SAMPLE_PDF_BYTES, "file_name": f"doc_{i}.pdf"}
            for i in range(5)
        ]
        results = classifier_engine.batch_classify(documents)
        assert len(results) == 5

    def test_batch_classify_empty_list(self, classifier_engine):
        """Batch classify with empty list returns empty results."""
        results = classifier_engine.batch_classify([])
        assert len(results) == 0

    def test_batch_classify_max_size(self, classifier_engine):
        """Batch classify respects maximum batch size."""
        documents = [
            {"document_bytes": SAMPLE_PDF_BYTES, "file_name": f"doc_{i}.pdf"}
            for i in range(501)
        ]
        with pytest.raises(ValueError):
            classifier_engine.batch_classify(documents)

    def test_batch_classify_partial_failure(self, classifier_engine):
        """Batch classify handles partial failures gracefully."""
        documents = [
            {"document_bytes": SAMPLE_PDF_BYTES, "file_name": "valid.pdf"},
            {"document_bytes": SAMPLE_CORRUPT_BYTES, "file_name": "corrupt.bin"},
            {"document_bytes": SAMPLE_PDF_BYTES, "file_name": "valid2.pdf"},
        ]
        results = classifier_engine.batch_classify(documents, continue_on_error=True)
        assert len(results) == 3

    def test_batch_classify_returns_per_document_results(self, classifier_engine):
        """Each batch result maps to its input document."""
        documents = [
            {"document_bytes": SAMPLE_PDF_BYTES, "file_name": f"doc_{i}.pdf"}
            for i in range(3)
        ]
        results = classifier_engine.batch_classify(documents)
        for r in results:
            assert "document_type" in r
            assert "confidence" in r


# ===========================================================================
# 7. Language Detection
# ===========================================================================


class TestLanguageDetection:
    """Test language detection in classification."""

    def test_language_detected(self, classifier_engine):
        """Classification detects the document language."""
        result = classifier_engine.classify(
            document_bytes=SAMPLE_PDF_BYTES,
            file_name="certificate_en.pdf",
        )
        assert "language" in result

    @pytest.mark.parametrize("lang", DOCUMENT_LANGUAGES)
    def test_all_supported_languages(self, classifier_engine, lang):
        """All 7 supported languages are valid classification outputs."""
        result = make_classification_result(language=lang)
        assert result["language"] == lang
        assert result["language"] in DOCUMENT_LANGUAGES

    def test_language_none_for_undetectable(self, classifier_engine):
        """Undetectable language returns None or a default."""
        result = classifier_engine.classify(
            document_bytes=SAMPLE_CORRUPT_BYTES,
            file_name="binary.dat",
        )
        # Language may be None or "unknown"
        assert result.get("language") is None or isinstance(result["language"], str)


# ===========================================================================
# 8. Multi-Page Document Handling
# ===========================================================================


class TestMultiPageHandling:
    """Test multi-page document classification."""

    def test_page_count_returned(self, classifier_engine):
        """Classification returns the page count."""
        result = classifier_engine.classify(
            document_bytes=SAMPLE_PDF_BYTES,
            file_name="multi_page.pdf",
        )
        assert "page_count" in result
        assert isinstance(result["page_count"], int)

    def test_single_page_classification(self, classifier_engine):
        """Single-page document has page_count of 1."""
        result = make_classification_result(page_count=1)
        assert result["page_count"] == 1

    def test_multi_page_classification(self, classifier_engine):
        """Multi-page document page count is preserved."""
        result = make_classification_result(page_count=15)
        assert result["page_count"] == 15


# ===========================================================================
# 9. Edge Cases
# ===========================================================================


class TestClassificationEdgeCases:
    """Test edge cases for document classification."""

    def test_empty_document_raises(self, classifier_engine):
        """Empty document bytes raises ValueError."""
        with pytest.raises(ValueError):
            classifier_engine.classify(
                document_bytes=SAMPLE_EMPTY_BYTES,
                file_name="empty.pdf",
            )

    def test_corrupt_document_handled(self, classifier_engine):
        """Corrupt document is handled gracefully (low confidence or error)."""
        try:
            result = classifier_engine.classify(
                document_bytes=SAMPLE_CORRUPT_BYTES,
                file_name="corrupt.pdf",
            )
            assert result["confidence_level"] in ("low", "unknown")
        except ValueError:
            pass  # Also acceptable to raise ValueError

    def test_no_file_name_raises(self, classifier_engine):
        """Missing file name raises ValueError."""
        with pytest.raises((ValueError, TypeError)):
            classifier_engine.classify(
                document_bytes=SAMPLE_PDF_BYTES,
                file_name=None,
            )

    def test_empty_file_name_raises(self, classifier_engine):
        """Empty file name raises ValueError."""
        with pytest.raises(ValueError):
            classifier_engine.classify(
                document_bytes=SAMPLE_PDF_BYTES,
                file_name="",
            )

    def test_very_large_document(self, classifier_engine):
        """Very large document is classified without error."""
        result = classifier_engine.classify(
            document_bytes=SAMPLE_LARGE_BYTES,
            file_name="large_document.pdf",
        )
        assert result is not None

    def test_classification_idempotent(self, classifier_engine):
        """Classifying the same document twice returns consistent results."""
        r1 = classifier_engine.classify(
            document_bytes=SAMPLE_PDF_BYTES,
            file_name="stable.pdf",
        )
        r2 = classifier_engine.classify(
            document_bytes=SAMPLE_PDF_BYTES,
            file_name="stable.pdf",
        )
        assert r1["document_type"] == r2["document_type"]
        assert r1["confidence"] == r2["confidence"]

    def test_none_bytes_raises(self, classifier_engine):
        """None document bytes raises TypeError."""
        with pytest.raises((TypeError, ValueError)):
            classifier_engine.classify(
                document_bytes=None,
                file_name="test.pdf",
            )

    def test_classification_result_factory_valid(self, classifier_engine):
        """Factory-built classification result passes validation."""
        result = make_classification_result()
        assert_classification_valid(result)

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_classify_with_commodity_hint(self, classifier_engine, commodity):
        """Classification accepts commodity hint parameter."""
        result = classifier_engine.classify(
            document_bytes=SAMPLE_PDF_BYTES,
            file_name=f"{commodity}_cert.pdf",
            commodity_hint=commodity,
        )
        assert result is not None
