# -*- coding: utf-8 -*-
"""
Tests for RiskAssessmentDocumenter Engine - AGENT-EUDR-030

Tests the Risk Assessment Documenter including document(),
_format_risk_level(), _build_contributing_factors(),
_add_regulatory_references(), and Article 10 criterion mapping.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest
from decimal import Decimal

from greenlang.agents.eudr.documentation_generator.risk_assessment_documenter import (
    RiskAssessmentDocumenter,
    _RISK_LEVEL_LABELS,
    _ARTICLE10_CRITERIA,
    _BENCHMARK_DESCRIPTIONS,
)
from greenlang.agents.eudr.documentation_generator.models import (
    EUDRCommodity,
    RiskAssessmentDoc,
    RiskLevel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def documenter() -> RiskAssessmentDocumenter:
    """Create RiskAssessmentDocumenter instance."""
    return RiskAssessmentDocumenter()


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------

def test_documenter_initialization(documenter):
    """Test RiskAssessmentDocumenter initializes correctly."""
    assert documenter._config is not None
    assert documenter._provenance is not None


# ---------------------------------------------------------------------------
# Test: generate_risk_doc - Success Paths
# ---------------------------------------------------------------------------

def test_generate_risk_doc_minimal(documenter):
    """Test generating risk doc with minimal data."""
    doc = documenter.generate_risk_doc(
        assessment_id="asr-001",
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        composite_score=Decimal("45.5"),
        risk_level=RiskLevel.STANDARD,
    )

    assert doc.doc_id.startswith("rad-")
    assert doc.assessment_id == "asr-001"
    assert doc.operator_id == "OP-001"
    assert doc.commodity == EUDRCommodity.COFFEE
    assert doc.composite_score == Decimal("45.5")
    assert doc.risk_level == RiskLevel.STANDARD


def test_generate_risk_doc_with_criteria(documenter):
    """Test risk doc generation with criterion evaluations."""
    criteria = [
        {
            "criterion_id": "10.2.a",
            "score": "65",
            "assessment": "moderate_risk",
            "evidence": ["satellite-data"],
        },
        {
            "criterion_id": "10.2.b",
            "score": "30",
            "assessment": "low_risk",
            "evidence": [],
        },
    ]

    doc = documenter.generate_risk_doc(
        assessment_id="asr-002",
        operator_id="OP-002",
        commodity=EUDRCommodity.COCOA,
        composite_score=Decimal("52.0"),
        risk_level=RiskLevel.STANDARD,
        criterion_evaluations=criteria,
    )

    assert len(doc.criterion_evaluations) == 7  # All Article 10(2) criteria


def test_generate_risk_doc_with_country_benchmark(documenter):
    """Test risk doc with country benchmarking."""
    doc = documenter.generate_risk_doc(
        assessment_id="asr-003",
        operator_id="OP-003",
        commodity=EUDRCommodity.WOOD,
        composite_score=Decimal("12.0"),
        risk_level=RiskLevel.LOW,
        country_benchmark="low",
        simplified_dd_eligible=True,
    )

    assert doc.country_benchmark == "low"
    assert doc.simplified_dd_eligible is True


def test_generate_risk_doc_with_decomposition(documenter):
    """Test risk doc with risk decomposition."""
    dimensions = {
        "country": {
            "score": Decimal("45"),
            "weight": Decimal("0.25"),
            "contribution": Decimal("11.25"),
        },
        "supplier": {
            "score": Decimal("60"),
            "weight": Decimal("0.30"),
            "contribution": Decimal("18.0"),
        },
    }

    doc = documenter.generate_risk_doc(
        assessment_id="asr-004",
        operator_id="OP-004",
        commodity=EUDRCommodity.COFFEE,
        composite_score=Decimal("48.0"),
        risk_level=RiskLevel.STANDARD,
        risk_dimensions=dimensions,
    )

    assert doc.doc_id.startswith("rad-")


# ---------------------------------------------------------------------------
# Test: Criterion Section Building
# ---------------------------------------------------------------------------

def test_build_criterion_section_empty(documenter):
    """Test criterion section with no upstream evaluations."""
    result = documenter._build_criterion_section([])

    assert len(result) == 7  # All Article 10(2) criteria
    assert result[0]["criterion_id"] == "10.2.a"
    assert result[0]["assessment"] == "not_evaluated"


def test_build_criterion_section_with_evaluations(documenter):
    """Test criterion section merges upstream evaluations."""
    evals = [
        {
            "criterion_id": "10.2.a",
            "score": "75",
            "assessment": "high_risk",
            "evidence": ["deforestation-alert"],
            "notes": "High deforestation in region",
        }
    ]

    result = documenter._build_criterion_section(evals)

    criterion_a = next(c for c in result if c["criterion_id"] == "10.2.a")
    assert criterion_a["score"] == "75"
    assert criterion_a["assessment"] == "high_risk"
    assert "deforestation-alert" in criterion_a["evidence"]


def test_build_criterion_section_all_criteria_present(documenter):
    """Test all Article 10(2) criteria are included."""
    result = documenter._build_criterion_section([])

    criterion_ids = [c["criterion_id"] for c in result]
    assert "10.2.a" in criterion_ids
    assert "10.2.b" in criterion_ids
    assert "10.2.c" in criterion_ids
    assert "10.2.d" in criterion_ids
    assert "10.2.e" in criterion_ids
    assert "10.2.f" in criterion_ids
    assert "10.2.g" in criterion_ids


# ---------------------------------------------------------------------------
# Test: Decomposition Section Building
# ---------------------------------------------------------------------------

def test_build_decomposition_section(documenter):
    """Test decomposition section building."""
    dimensions = {
        "country": {
            "score": Decimal("50"),
            "weight": Decimal("0.25"),
            "contribution": Decimal("12.5"),
        },
        "supplier": {
            "score": Decimal("60"),
            "weight": Decimal("0.30"),
            "contribution": Decimal("18.0"),
        },
    }

    section = documenter._build_decomposition_section(dimensions)

    assert section["total_dimensions"] == 2
    assert "country" in section["dimensions"]
    assert "supplier" in section["dimensions"]
    assert section["dimensions"]["country"]["score"] == "50"
    assert section["dimensions"]["country"]["weighted_contribution"] == "12.5"


def test_build_decomposition_section_with_simple_values(documenter):
    """Test decomposition with simple score values."""
    dimensions = {
        "country": Decimal("50"),
        "supplier": Decimal("60"),
    }

    section = documenter._build_decomposition_section(dimensions)

    assert section["total_dimensions"] == 2
    assert section["dimensions"]["country"]["score"] == "50"


# ---------------------------------------------------------------------------
# Test: Country Benchmark Section
# ---------------------------------------------------------------------------

def test_build_country_benchmark_section_low(documenter):
    """Test country benchmark section for low-risk country."""
    section = documenter._build_country_benchmark_section("low")

    assert section["classification"] == "low"
    assert section["simplified_dd_applicable"] is True
    assert section["enhanced_scrutiny_required"] is False
    assert section["article_reference"] == "EUDR Article 29"


def test_build_country_benchmark_section_high(documenter):
    """Test country benchmark section for high-risk country."""
    section = documenter._build_country_benchmark_section("high")

    assert section["classification"] == "high"
    assert section["simplified_dd_applicable"] is False
    assert section["enhanced_scrutiny_required"] is True


def test_build_country_benchmark_section_standard(documenter):
    """Test country benchmark section for standard-risk country."""
    section = documenter._build_country_benchmark_section("standard")

    assert section["classification"] == "standard"
    assert section["simplified_dd_applicable"] is False
    assert section["enhanced_scrutiny_required"] is False


def test_build_country_benchmark_section_unknown(documenter):
    """Test country benchmark section for unknown classification."""
    section = documenter._build_country_benchmark_section("unknown")

    assert section["classification"] == "unknown"


# ---------------------------------------------------------------------------
# Test: Dimension Score Classification
# ---------------------------------------------------------------------------

def test_classify_dimension_score_negligible(documenter):
    """Test dimension score classification for negligible range."""
    classification = documenter._classify_dimension_score(Decimal("10"))
    assert classification == "negligible"


def test_classify_dimension_score_low(documenter):
    """Test dimension score classification for low range."""
    classification = documenter._classify_dimension_score(Decimal("25"))
    assert classification == "low"


def test_classify_dimension_score_standard(documenter):
    """Test dimension score classification for standard range."""
    classification = documenter._classify_dimension_score(Decimal("45"))
    assert classification == "standard"


def test_classify_dimension_score_high(documenter):
    """Test dimension score classification for high range."""
    classification = documenter._classify_dimension_score(Decimal("75"))
    assert classification == "high"


def test_classify_dimension_score_critical(documenter):
    """Test dimension score classification for critical range."""
    classification = documenter._classify_dimension_score(Decimal("90"))
    assert classification == "critical"


def test_classify_dimension_score_invalid(documenter):
    """Test dimension score classification with invalid input."""
    classification = documenter._classify_dimension_score("invalid")
    assert classification == "unknown"


# ---------------------------------------------------------------------------
# Test: DDS Formatting
# ---------------------------------------------------------------------------

def test_format_for_dds_inclusion(documenter):
    """Test formatting risk doc for DDS inclusion."""
    doc = RiskAssessmentDoc(
        doc_id="rad-001",
        assessment_id="asr-001",
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        composite_score=Decimal("45.0"),
        risk_level=RiskLevel.STANDARD,
        criterion_evaluations=[{"id": "10.2.a"}],
        country_benchmark="standard",
        simplified_dd_eligible=False,
    )

    summary = documenter.format_for_dds_inclusion(doc)

    assert summary["doc_id"] == "rad-001"
    assert summary["assessment_id"] == "asr-001"
    assert summary["composite_score"] == "45.0"
    assert summary["risk_level"] == "standard"
    assert summary["criterion_count"] == 1
    assert summary["article_reference"] == "EUDR Article 10"
    assert "country_benchmark" in summary


def test_format_for_dds_inclusion_no_benchmark(documenter):
    """Test DDS formatting without country benchmark."""
    doc = RiskAssessmentDoc(
        doc_id="rad-002",
        assessment_id="asr-002",
        operator_id="OP-002",
        commodity=EUDRCommodity.COCOA,
        composite_score=Decimal("30.0"),
        risk_level=RiskLevel.LOW,
        criterion_evaluations=[],
        country_benchmark="",
        simplified_dd_eligible=False,
    )

    summary = documenter.format_for_dds_inclusion(doc)

    assert "country_benchmark" not in summary or summary.get("country_benchmark") == ""


# ---------------------------------------------------------------------------
# Test: Risk Level Labels
# ---------------------------------------------------------------------------

def test_get_risk_level_label(documenter):
    """Test getting risk level labels."""
    label = documenter.get_risk_level_label(RiskLevel.HIGH)
    assert "high" in label.lower()
    assert "mitigation" in label.lower()


def test_get_risk_level_label_all_levels(documenter):
    """Test labels exist for all risk levels."""
    for level in RiskLevel:
        label = documenter.get_risk_level_label(level)
        assert len(label) > 0


# ---------------------------------------------------------------------------
# Test: Criteria List
# ---------------------------------------------------------------------------

def test_get_criteria_list(documenter):
    """Test getting Article 10(2) criteria list."""
    criteria = documenter.get_criteria_list()

    assert len(criteria) == 7
    assert all("id" in c for c in criteria)
    assert all("description" in c for c in criteria)
    assert all("article" in c for c in criteria)


# ---------------------------------------------------------------------------
# Test: Health Check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_check(documenter):
    """Test health check returns correct status."""
    status = await documenter.health_check()

    assert status["engine"] == "RiskAssessmentDocumenter"
    assert status["status"] == "available"
    assert "config" in status
    assert status["criteria_count"] == 7


# ---------------------------------------------------------------------------
# Test: Different Risk Levels
# ---------------------------------------------------------------------------

def test_generate_risk_doc_negligible(documenter):
    """Test risk doc for negligible risk level."""
    doc = documenter.generate_risk_doc(
        assessment_id="asr-neg",
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        composite_score=Decimal("5.0"),
        risk_level=RiskLevel.NEGLIGIBLE,
    )

    assert doc.risk_level == RiskLevel.NEGLIGIBLE


def test_generate_risk_doc_critical(documenter):
    """Test risk doc for critical risk level."""
    doc = documenter.generate_risk_doc(
        assessment_id="asr-crit",
        operator_id="OP-001",
        commodity=EUDRCommodity.WOOD,
        composite_score=Decimal("95.0"),
        risk_level=RiskLevel.CRITICAL,
    )

    assert doc.risk_level == RiskLevel.CRITICAL


# ---------------------------------------------------------------------------
# Test: Different Commodities
# ---------------------------------------------------------------------------

def test_generate_risk_doc_palm_oil(documenter):
    """Test risk doc for palm oil commodity."""
    doc = documenter.generate_risk_doc(
        assessment_id="asr-palm",
        operator_id="OP-PALM",
        commodity=EUDRCommodity.PALM_OIL,
        composite_score=Decimal("60.0"),
        risk_level=RiskLevel.STANDARD,
    )

    assert doc.commodity == EUDRCommodity.PALM_OIL


def test_generate_risk_doc_soya(documenter):
    """Test risk doc for soya commodity."""
    doc = documenter.generate_risk_doc(
        assessment_id="asr-soya",
        operator_id="OP-SOYA",
        commodity=EUDRCommodity.SOYA,
        composite_score=Decimal("40.0"),
        risk_level=RiskLevel.STANDARD,
    )

    assert doc.commodity == EUDRCommodity.SOYA
