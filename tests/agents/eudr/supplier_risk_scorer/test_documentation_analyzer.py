# -*- coding: utf-8 -*-
"""
Unit tests for DocumentationAnalyzer - AGENT-EUDR-017 Engine 3

Tests comprehensive supplier documentation analysis per EUDR Articles 4, 9, 31
covering document completeness scoring, accuracy assessment, consistency validation,
timeliness tracking, EUDR-required document verification, expiry tracking, and
gap analysis.

Target: 50+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-017 Supplier Risk Scorer (GL-EUDR-SRS-017)
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.supplier_risk_scorer.documentation_analyzer import (
    DocumentationAnalyzer,
)
from greenlang.agents.eudr.supplier_risk_scorer.models import (
    DocumentType,
    DocumentStatus,
    CommodityType,
)


class TestDocumentationAnalyzerInit:
    """Tests for DocumentationAnalyzer initialization."""

    @pytest.mark.unit
    def test_initialization_creates_empty_stores(self, mock_config):
        analyzer = DocumentationAnalyzer()
        assert analyzer._documentation_profiles == {}


class TestAnalyzeDocuments:
    """Tests for analyze_documents method."""

    @pytest.mark.unit
    def test_analyze_documents_returns_profile(
        self, documentation_analyzer, sample_documentation
    ):
        result = documentation_analyzer.analyze_documents(
            supplier_id=sample_documentation["supplier_id"],
            documents=sample_documentation["documents"],
            commodity=CommodityType.SOYA,
        )
        assert result is not None
        assert "profile_id" in result
        assert result["supplier_id"] == sample_documentation["supplier_id"]

    @pytest.mark.unit
    def test_analyze_documents_all_types(self, documentation_analyzer):
        doc_types = [
            DocumentType.GEOLOCATION, DocumentType.DDS_REFERENCE,
            DocumentType.PRODUCT_DESCRIPTION, DocumentType.QUANTITY_DECLARATION,
            DocumentType.HARVEST_DATE, DocumentType.COMPLIANCE_DECLARATION,
            DocumentType.CERTIFICATE, DocumentType.TRADE_LICENSE,
            DocumentType.PHYTOSANITARY
        ]
        documents = [
            {"type": doc_type, "status": DocumentStatus.VERIFIED}
            for doc_type in doc_types
        ]
        result = documentation_analyzer.analyze_documents(
            supplier_id="SUPP-001",
            documents=documents,
            commodity=CommodityType.SOYA,
        )
        assert len(result["documents"]) == len(doc_types)


class TestScoreCompleteness:
    """Tests for completeness scoring."""

    @pytest.mark.unit
    def test_score_completeness_full_documents(self, documentation_analyzer):
        required_docs = [
            DocumentType.GEOLOCATION, DocumentType.DDS_REFERENCE,
            DocumentType.PRODUCT_DESCRIPTION, DocumentType.QUANTITY_DECLARATION,
            DocumentType.HARVEST_DATE, DocumentType.COMPLIANCE_DECLARATION
        ]
        documents = [
            {"type": doc_type, "status": DocumentStatus.VERIFIED}
            for doc_type in required_docs
        ]
        score = documentation_analyzer.score_completeness(
            documents=documents,
            commodity=CommodityType.SOYA,
        )
        assert score == Decimal("1.0")  # 100% complete

    @pytest.mark.unit
    def test_score_completeness_partial_documents(self, documentation_analyzer):
        # Only 3 out of 6 required
        documents = [
            {"type": DocumentType.GEOLOCATION, "status": DocumentStatus.VERIFIED},
            {"type": DocumentType.DDS_REFERENCE, "status": DocumentStatus.VERIFIED},
            {"type": DocumentType.PRODUCT_DESCRIPTION, "status": DocumentStatus.VERIFIED},
        ]
        score = documentation_analyzer.score_completeness(
            documents=documents,
            commodity=CommodityType.SOYA,
        )
        assert Decimal("0.0") < score < Decimal("1.0")


class TestIdentifyGaps:
    """Tests for gap identification."""

    @pytest.mark.unit
    def test_identify_gaps_returns_missing_docs(self, documentation_analyzer):
        documents = [
            {"type": DocumentType.GEOLOCATION, "status": DocumentStatus.VERIFIED},
        ]
        gaps = documentation_analyzer.identify_gaps(
            documents=documents,
            commodity=CommodityType.SOYA,
        )
        assert len(gaps) > 0
        assert DocumentType.DDS_REFERENCE in gaps or "dds_reference" in [g.lower() for g in gaps]


class TestCheckExpiry:
    """Tests for expiry checking."""

    @pytest.mark.unit
    def test_check_expiry_warns_for_expiring_docs(
        self, documentation_analyzer, mock_config
    ):
        expiry_date = datetime.now(timezone.utc) + timedelta(days=60)  # Within 90 day buffer
        documents = [
            {
                "type": DocumentType.CERTIFICATE,
                "status": DocumentStatus.VERIFIED,
                "expiry_date": expiry_date.isoformat(),
            }
        ]
        expiring = documentation_analyzer.check_expiry(
            documents=documents,
            warning_days=mock_config.expiry_warning_days,
        )
        assert len(expiring) > 0

    @pytest.mark.unit
    def test_check_expiry_no_warnings_for_valid_docs(
        self, documentation_analyzer, mock_config
    ):
        expiry_date = datetime.now(timezone.utc) + timedelta(days=365)  # Far future
        documents = [
            {
                "type": DocumentType.CERTIFICATE,
                "status": DocumentStatus.VERIFIED,
                "expiry_date": expiry_date.isoformat(),
            }
        ]
        expiring = documentation_analyzer.check_expiry(
            documents=documents,
            warning_days=mock_config.expiry_warning_days,
        )
        assert len(expiring) == 0


class TestValidateAuthenticity:
    """Tests for authenticity validation."""

    @pytest.mark.unit
    def test_validate_authenticity_indicators(self, documentation_analyzer):
        document = {
            "type": DocumentType.CERTIFICATE,
            "status": DocumentStatus.VERIFIED,
            "supplier_id": "SUPP-001",
            "certificate_number": "FSC-C123456",
        }
        result = documentation_analyzer.validate_authenticity(document)
        assert "authenticity_score" in result
        assert Decimal("0.0") <= result["authenticity_score"] <= Decimal("100.0")


class TestDetectLanguage:
    """Tests for language detection."""

    @pytest.mark.unit
    def test_detect_language_english(self, documentation_analyzer):
        text = "This is a compliance declaration for EUDR purposes."
        lang = documentation_analyzer.detect_language(text)
        assert lang == "en"

    @pytest.mark.unit
    def test_detect_language_portuguese(self, documentation_analyzer):
        text = "Esta é uma declaração de conformidade para fins EUDR."
        lang = documentation_analyzer.detect_language(text)
        assert lang == "pt"


class TestGenerateRequest:
    """Tests for document request generation."""

    @pytest.mark.unit
    def test_generate_document_request(self, documentation_analyzer):
        missing_docs = [DocumentType.HARVEST_DATE, DocumentType.COMPLIANCE_DECLARATION]
        request = documentation_analyzer.generate_document_request(
            supplier_id="SUPP-001",
            missing_documents=missing_docs,
            due_date=datetime.now(timezone.utc) + timedelta(days=30),
        )
        assert "request_id" in request
        assert len(request["requested_documents"]) == len(missing_docs)


class TestQualityTrend:
    """Tests for quality trend analysis."""

    @pytest.mark.unit
    def test_analyze_quality_trend(self, documentation_analyzer):
        supplier_id = "SUPP-TREND"
        # Simulate improving quality over time
        history = [
            {"quality_score": Decimal("60.0"), "date": datetime.now(timezone.utc) - timedelta(days=360)},
            {"quality_score": Decimal("70.0"), "date": datetime.now(timezone.utc) - timedelta(days=180)},
            {"quality_score": Decimal("80.0"), "date": datetime.now(timezone.utc) - timedelta(days=90)},
        ]
        trend = documentation_analyzer.analyze_quality_trend(supplier_id, history)
        assert trend == "improving"


class TestProvenance:
    """Tests for provenance tracking."""

    @pytest.mark.unit
    def test_analysis_includes_provenance_hash(
        self, documentation_analyzer, sample_documentation
    ):
        result = documentation_analyzer.analyze_documents(
            supplier_id=sample_documentation["supplier_id"],
            documents=sample_documentation["documents"],
            commodity=CommodityType.SOYA,
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.unit
    def test_invalid_supplier_id_raises_error(self, documentation_analyzer):
        with pytest.raises(ValueError):
            documentation_analyzer.analyze_documents(
                supplier_id="",
                documents=[],
                commodity=CommodityType.SOYA,
            )


@pytest.mark.unit
@pytest.mark.parametrize("commodity", [
    CommodityType.CATTLE, CommodityType.COCOA, CommodityType.COFFEE,
    CommodityType.OIL_PALM, CommodityType.RUBBER, CommodityType.SOYA,
    CommodityType.WOOD
])
def test_required_documents_by_commodity(documentation_analyzer, commodity):
    """Test that required documents are correctly identified for each commodity."""
    required = documentation_analyzer.get_required_documents(commodity)
    assert len(required) >= 6  # All commodities need at least 6 base docs
