# -*- coding: utf-8 -*-
"""
Tests for MitigationDocumenter Engine - AGENT-EUDR-030

Tests the Mitigation Documenter including document(), _calculate_risk_reduction(),
_verify_article11_compliance(), _format_measures(), and Article 11(2) category mapping.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest
from decimal import Decimal

from greenlang.agents.eudr.documentation_generator.mitigation_documenter import (
    MitigationDocumenter,
    _ARTICLE11_REFERENCES,
    _EFFECTIVENESS_THRESHOLDS,
)
from greenlang.agents.eudr.documentation_generator.models import (
    EUDRCommodity,
    MeasureSummary,
    MitigationDoc,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def documenter() -> MitigationDocumenter:
    """Create MitigationDocumenter instance."""
    return MitigationDocumenter()


@pytest.fixture
def sample_measures() -> list[MeasureSummary]:
    """Create sample mitigation measures."""
    return [
        MeasureSummary(
            measure_id="msr-001",
            title="Enhanced Supplier Audit",
            category="independent_audit",
            status="completed",
            reduction=Decimal("25.0"),
        ),
        MeasureSummary(
            measure_id="msr-002",
            title="Additional Geolocation Verification",
            category="additional_info",
            status="completed",
            reduction=Decimal("15.0"),
        ),
    ]


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------

def test_documenter_initialization(documenter):
    """Test MitigationDocumenter initializes correctly."""
    assert documenter._config is not None
    assert documenter._provenance is not None


# ---------------------------------------------------------------------------
# Test: generate_mitigation_doc - Success Paths
# ---------------------------------------------------------------------------

def test_generate_mitigation_doc_basic(documenter, sample_measures):
    """Test basic mitigation doc generation."""
    doc = documenter.generate_mitigation_doc(
        strategy_id="stg-001",
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        pre_score=Decimal("72.0"),
        post_score=Decimal("28.0"),
        measures=sample_measures,
    )

    assert doc.doc_id.startswith("mid-")
    assert doc.strategy_id == "stg-001"
    assert doc.operator_id == "OP-001"
    assert doc.commodity == EUDRCommodity.COFFEE
    assert doc.pre_score == Decimal("72.0")
    assert doc.post_score == Decimal("28.0")
    assert len(doc.measures_summary) == 2


def test_generate_mitigation_doc_with_verification(documenter, sample_measures):
    """Test mitigation doc with verification result."""
    doc = documenter.generate_mitigation_doc(
        strategy_id="stg-002",
        operator_id="OP-002",
        commodity=EUDRCommodity.COCOA,
        pre_score=Decimal("80.0"),
        post_score=Decimal("25.0"),
        measures=sample_measures,
        verification_result="sufficient",
    )

    assert doc.verification_result == "sufficient"


def test_generate_mitigation_doc_partial_verification(documenter, sample_measures):
    """Test mitigation doc with partial verification."""
    doc = documenter.generate_mitigation_doc(
        strategy_id="stg-003",
        operator_id="OP-003",
        commodity=EUDRCommodity.WOOD,
        pre_score=Decimal("70.0"),
        post_score=Decimal("40.0"),
        measures=sample_measures,
        verification_result="partial",
    )

    assert doc.verification_result == "partial"


def test_generate_mitigation_doc_no_measures(documenter):
    """Test mitigation doc with empty measures list."""
    doc = documenter.generate_mitigation_doc(
        strategy_id="stg-empty",
        operator_id="OP-004",
        commodity=EUDRCommodity.PALM_OIL,
        pre_score=Decimal("50.0"),
        post_score=Decimal("50.0"),
        measures=[],
    )

    assert len(doc.measures_summary) == 0


# ---------------------------------------------------------------------------
# Test: Measures Section Building
# ---------------------------------------------------------------------------

def test_build_measures_section(documenter, sample_measures):
    """Test building measures section with Article 11(2) mapping."""
    section = documenter._build_measures_section(sample_measures)

    assert len(section) == 2
    assert section[0]["index"] == 1
    assert section[0]["measure_id"] == "msr-001"
    assert section[0]["title"] == "Enhanced Supplier Audit"
    assert section[0]["category"] == "independent_audit"
    assert section[0]["article11_reference"] == "EUDR Article 11(1)(b)"


def test_build_measures_section_article11_mapping(documenter):
    """Test Article 11 category mapping for different categories."""
    measures = [
        MeasureSummary(
            measure_id="m1",
            title="Additional Info",
            category="additional_info",
            status="completed",
            reduction=Decimal("10"),
        ),
        MeasureSummary(
            measure_id="m2",
            title="Independent Audit",
            category="independent_audit",
            status="completed",
            reduction=Decimal("20"),
        ),
        MeasureSummary(
            measure_id="m3",
            title="Other Measure",
            category="other_measures",
            status="completed",
            reduction=Decimal("15"),
        ),
    ]

    section = documenter._build_measures_section(measures)

    assert section[0]["article11_reference"] == "EUDR Article 11(1)(a)"
    assert section[1]["article11_reference"] == "EUDR Article 11(1)(b)"
    assert section[2]["article11_reference"] == "EUDR Article 11(1)(c)"


def test_build_measures_section_unknown_category(documenter):
    """Test measures section with unknown category defaults to other_measures."""
    measures = [
        MeasureSummary(
            measure_id="m1",
            title="Unknown Category",
            category="unknown_category",
            status="pending",
            reduction=Decimal("0"),
        )
    ]

    section = documenter._build_measures_section(measures)

    # Should default to other_measures reference
    assert "EUDR Article 11" in section[0]["article11_reference"]


# ---------------------------------------------------------------------------
# Test: Effectiveness Section Building
# ---------------------------------------------------------------------------

def test_build_effectiveness_section(documenter):
    """Test effectiveness section building."""
    section = documenter._build_effectiveness_section(
        pre=Decimal("72.0"),
        post=Decimal("28.0"),
    )

    assert section["pre_mitigation_score"] == "72.0"
    assert section["post_mitigation_score"] == "28.0"
    assert section["absolute_reduction"] == "44.0"
    # (44 / 72) * 100 = 61.11%
    assert Decimal(section["percentage_reduction"]) > Decimal("61")
    assert section["effectiveness_classification"] == "highly_effective"
    assert section["target_achieved"] is True


def test_build_effectiveness_section_moderate(documenter):
    """Test effectiveness with moderate reduction."""
    section = documenter._build_effectiveness_section(
        pre=Decimal("60.0"),
        post=Decimal("40.0"),
    )

    # (20 / 60) * 100 = 33.33%
    assert section["effectiveness_classification"] == "moderately_effective"


def test_build_effectiveness_section_marginal(documenter):
    """Test effectiveness with marginal reduction."""
    section = documenter._build_effectiveness_section(
        pre=Decimal("50.0"),
        post=Decimal("44.0"),
    )

    # (6 / 50) * 100 = 12%
    assert section["effectiveness_classification"] == "marginally_effective"


def test_build_effectiveness_section_ineffective(documenter):
    """Test effectiveness with ineffective reduction."""
    section = documenter._build_effectiveness_section(
        pre=Decimal("50.0"),
        post=Decimal("49.0"),
    )

    # (1 / 50) * 100 = 2%
    assert section["effectiveness_classification"] == "ineffective"


def test_build_effectiveness_section_zero_pre_score(documenter):
    """Test effectiveness section with zero pre-score."""
    section = documenter._build_effectiveness_section(
        pre=Decimal("0"),
        post=Decimal("0"),
    )

    assert section["percentage_reduction"] == "0"


# ---------------------------------------------------------------------------
# Test: Effectiveness Classification
# ---------------------------------------------------------------------------

def test_classify_effectiveness_highly_effective(documenter):
    """Test highly effective classification (>=50%)."""
    classification = documenter._classify_effectiveness(Decimal("75"))
    assert classification == "highly_effective"


def test_classify_effectiveness_moderately_effective(documenter):
    """Test moderately effective classification (25-49%)."""
    classification = documenter._classify_effectiveness(Decimal("35"))
    assert classification == "moderately_effective"


def test_classify_effectiveness_marginally_effective(documenter):
    """Test marginally effective classification (10-24%)."""
    classification = documenter._classify_effectiveness(Decimal("15"))
    assert classification == "marginally_effective"


def test_classify_effectiveness_ineffective(documenter):
    """Test ineffective classification (<10%)."""
    classification = documenter._classify_effectiveness(Decimal("5"))
    assert classification == "ineffective"


# ---------------------------------------------------------------------------
# Test: Score to Risk Level Conversion
# ---------------------------------------------------------------------------

def test_score_to_risk_level_negligible(documenter):
    """Test score to risk level for negligible range."""
    level = documenter._score_to_risk_level(Decimal("10"))
    assert level == "negligible"


def test_score_to_risk_level_low(documenter):
    """Test score to risk level for low range."""
    level = documenter._score_to_risk_level(Decimal("25"))
    assert level == "low"


def test_score_to_risk_level_standard(documenter):
    """Test score to risk level for standard range."""
    level = documenter._score_to_risk_level(Decimal("45"))
    assert level == "standard"


def test_score_to_risk_level_high(documenter):
    """Test score to risk level for high range."""
    level = documenter._score_to_risk_level(Decimal("75"))
    assert level == "high"


def test_score_to_risk_level_critical(documenter):
    """Test score to risk level for critical range."""
    level = documenter._score_to_risk_level(Decimal("90"))
    assert level == "critical"


# ---------------------------------------------------------------------------
# Test: DDS Formatting
# ---------------------------------------------------------------------------

def test_format_for_dds_inclusion(documenter):
    """Test formatting mitigation doc for DDS inclusion."""
    doc = MitigationDoc(
        doc_id="mid-001",
        strategy_id="stg-001",
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        pre_score=Decimal("72.0"),
        post_score=Decimal("28.0"),
        measures_summary=[
            MeasureSummary(
                measure_id="m1",
                title="Audit",
                category="independent_audit",
                status="completed",
                reduction=Decimal("44"),
            )
        ],
        verification_result="sufficient",
    )

    summary = documenter.format_for_dds_inclusion(doc)

    assert summary["doc_id"] == "mid-001"
    assert summary["strategy_id"] == "stg-001"
    assert summary["pre_score"] == "72.0"
    assert summary["post_score"] == "28.0"
    assert summary["measure_count"] == 1
    assert summary["verification_result"] == "sufficient"
    assert summary["article_reference"] == "EUDR Article 11"


def test_format_for_dds_inclusion_with_category_breakdown(documenter):
    """Test DDS formatting includes category breakdown."""
    doc = MitigationDoc(
        doc_id="mid-002",
        strategy_id="stg-002",
        operator_id="OP-002",
        commodity=EUDRCommodity.COCOA,
        pre_score=Decimal("60.0"),
        post_score=Decimal("30.0"),
        measures_summary=[
            MeasureSummary(
                measure_id="m1",
                title="Audit 1",
                category="independent_audit",
                status="completed",
                reduction=Decimal("20"),
            ),
            MeasureSummary(
                measure_id="m2",
                title="Audit 2",
                category="independent_audit",
                status="completed",
                reduction=Decimal("10"),
            ),
            MeasureSummary(
                measure_id="m3",
                title="Info",
                category="additional_info",
                status="completed",
                reduction=Decimal("5"),
            ),
        ],
    )

    summary = documenter.format_for_dds_inclusion(doc)

    assert "category_breakdown" in summary
    assert summary["category_breakdown"]["independent_audit"] == 2
    assert summary["category_breakdown"]["additional_info"] == 1


# ---------------------------------------------------------------------------
# Test: Article 11 Reference Retrieval
# ---------------------------------------------------------------------------

def test_get_article11_reference_additional_info(documenter):
    """Test getting Article 11 reference for additional_info."""
    ref = documenter.get_article11_reference("additional_info")

    assert ref["primary"] == "EUDR Article 11(1)(a)"
    assert "information" in ref["description"].lower()


def test_get_article11_reference_independent_audit(documenter):
    """Test getting Article 11 reference for independent_audit."""
    ref = documenter.get_article11_reference("independent_audit")

    assert ref["primary"] == "EUDR Article 11(1)(b)"
    assert "audit" in ref["description"].lower()


def test_get_article11_reference_other_measures(documenter):
    """Test getting Article 11 reference for other_measures."""
    ref = documenter.get_article11_reference("other_measures")

    assert ref["primary"] == "EUDR Article 11(1)(c)"


def test_get_article11_reference_unknown_category(documenter):
    """Test getting Article 11 reference for unknown category."""
    ref = documenter.get_article11_reference("unknown_category")

    # Should return default
    assert "EUDR Article 11" in ref["primary"]


# ---------------------------------------------------------------------------
# Test: Effectiveness Thresholds
# ---------------------------------------------------------------------------

def test_get_effectiveness_thresholds(documenter):
    """Test getting effectiveness thresholds."""
    thresholds = documenter.get_effectiveness_thresholds()

    assert "highly_effective" in thresholds
    assert "moderately_effective" in thresholds
    assert "marginally_effective" in thresholds
    assert "ineffective" in thresholds
    assert Decimal(thresholds["highly_effective"]) == Decimal("50")


# ---------------------------------------------------------------------------
# Test: Health Check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_check(documenter):
    """Test health check returns correct status."""
    status = await documenter.health_check()

    assert status["engine"] == "MitigationDocumenter"
    assert status["status"] == "available"
    assert "config" in status
    assert status["article11_categories"] == 3


# ---------------------------------------------------------------------------
# Test: Different Commodities
# ---------------------------------------------------------------------------

def test_generate_mitigation_doc_rubber(documenter, sample_measures):
    """Test mitigation doc for rubber commodity."""
    doc = documenter.generate_mitigation_doc(
        strategy_id="stg-rubber",
        operator_id="OP-RUBBER",
        commodity=EUDRCommodity.RUBBER,
        pre_score=Decimal("65.0"),
        post_score=Decimal("30.0"),
        measures=sample_measures,
    )

    assert doc.commodity == EUDRCommodity.RUBBER


def test_generate_mitigation_doc_cattle(documenter, sample_measures):
    """Test mitigation doc for cattle commodity."""
    doc = documenter.generate_mitigation_doc(
        strategy_id="stg-cattle",
        operator_id="OP-CATTLE",
        commodity=EUDRCommodity.CATTLE,
        pre_score=Decimal("70.0"),
        post_score=Decimal("25.0"),
        measures=sample_measures,
    )

    assert doc.commodity == EUDRCommodity.CATTLE


# ---------------------------------------------------------------------------
# Test: Edge Cases
# ---------------------------------------------------------------------------

def test_generate_mitigation_doc_no_reduction(documenter):
    """Test mitigation doc with no actual reduction."""
    measures = [
        MeasureSummary(
            measure_id="m1",
            title="Ineffective Measure",
            category="other_measures",
            status="completed",
            reduction=Decimal("0"),
        )
    ]

    doc = documenter.generate_mitigation_doc(
        strategy_id="stg-no-reduction",
        operator_id="OP-005",
        commodity=EUDRCommodity.SOYA,
        pre_score=Decimal("50.0"),
        post_score=Decimal("50.0"),
        measures=measures,
    )

    assert doc.pre_score == doc.post_score


def test_generate_mitigation_doc_target_achieved(documenter, sample_measures):
    """Test mitigation doc where post-score achieves target (<=30)."""
    doc = documenter.generate_mitigation_doc(
        strategy_id="stg-target",
        operator_id="OP-006",
        commodity=EUDRCommodity.COFFEE,
        pre_score=Decimal("75.0"),
        post_score=Decimal("25.0"),  # Below 30 threshold
        measures=sample_measures,
    )

    assert doc.post_score <= Decimal("30")
