# -*- coding: utf-8 -*-
"""
Unit tests for OffsetPortfolioEngine (PACK-021 Engine 6).

Tests credit quality scoring, portfolio diversification, standard assessment,
vintage management, SBTi compliance, VCMI claims alignment, Oxford Principles
alignment, cost analysis, retirement tracking, and provenance hashing.

Author:  GL-TestEngineer
Pack:    PACK-021 Net Zero Starter
"""

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.offset_portfolio_engine import (
    CATEGORY_COST_RANGES,
    CreditCategory,
    CreditEntry,
    CreditQualityScore,
    CreditStandard,
    CreditType,
    OffsetPortfolioEngine,
    OxfordAlignmentResult,
    OXFORD_PRINCIPLES_TARGETS,
    PortfolioResult,
    PortfolioSummary,
    QUALITY_WEIGHTS,
    QualityDimension,
    RetirementStatus,
    SBTiComplianceResult,
    SBTiCreditUse,
    STANDARD_BENCHMARKS,
    VCMIAlignmentResult,
    VCMIClaim,
    VCMI_THRESHOLDS,
)


# ========================================================================
# Fixtures
# ========================================================================


@pytest.fixture
def engine() -> OffsetPortfolioEngine:
    """Fresh OffsetPortfolioEngine."""
    return OffsetPortfolioEngine()


@pytest.fixture
def removal_credit() -> CreditEntry:
    """High-quality biochar removal credit."""
    return CreditEntry(
        standard=CreditStandard.VERRA_VCS,
        credit_type=CreditType.REMOVAL,
        category=CreditCategory.BIOCHAR,
        project_name="Biochar Project Alpha",
        vintage_year=2025,
        quantity_tco2e=Decimal("500"),
        unit_price_usd=Decimal("120"),
        additionality_score=4,
        permanence_score=4,
        co_benefits_score=4,
        leakage_risk_score=4,
        mrv_quality_score=4,
        sbti_use=SBTiCreditUse.NEUTRALIZATION,
    )


@pytest.fixture
def avoidance_credit() -> CreditEntry:
    """Medium-quality renewable energy avoidance credit."""
    return CreditEntry(
        standard=CreditStandard.GOLD_STANDARD,
        credit_type=CreditType.AVOIDANCE,
        category=CreditCategory.RENEWABLE_ENERGY,
        project_name="Wind Farm Beta",
        vintage_year=2024,
        quantity_tco2e=Decimal("1000"),
        unit_price_usd=Decimal("5"),
        additionality_score=3,
        permanence_score=2,
        co_benefits_score=3,
        leakage_risk_score=4,
        mrv_quality_score=3,
        sbti_use=SBTiCreditUse.BVCM_COMPENSATION,
    )


@pytest.fixture
def reduction_credit() -> CreditEntry:
    """Reduction credit for methane capture."""
    return CreditEntry(
        standard=CreditStandard.ACR,
        credit_type=CreditType.REDUCTION,
        category=CreditCategory.METHANE_CAPTURE,
        project_name="Landfill Gas Capture",
        vintage_year=2023,
        quantity_tco2e=Decimal("300"),
        unit_price_usd=Decimal("12"),
        additionality_score=4,
        permanence_score=3,
        co_benefits_score=3,
        leakage_risk_score=4,
        mrv_quality_score=4,
        sbti_use=SBTiCreditUse.BVCM_COMPENSATION,
    )


@pytest.fixture
def populated_engine(engine, removal_credit, avoidance_credit, reduction_credit):
    """Engine with three credits added."""
    engine.add_credit(removal_credit)
    engine.add_credit(avoidance_credit)
    engine.add_credit(reduction_credit)
    return engine


# ========================================================================
# Instantiation
# ========================================================================


class TestOffsetPortfolioEngineInstantiation:
    """Tests for engine creation."""

    def test_engine_instantiates(self):
        """Engine creates without error."""
        engine = OffsetPortfolioEngine()
        assert engine is not None
        assert engine.engine_version == "1.0.0"

    def test_engine_starts_empty(self, engine):
        """Internal credit list starts empty."""
        assert len(engine._credits) == 0


# ========================================================================
# Credit Quality Scoring
# ========================================================================


class TestCreditQualityScoring:
    """Tests for per-credit quality scoring."""

    def test_score_credit_returns_quality_score(self, engine, removal_credit):
        """score_credit returns a CreditQualityScore."""
        score = engine.score_credit(removal_credit)
        assert isinstance(score, CreditQualityScore)

    def test_score_dimensions_normalized_to_100(self, engine, removal_credit):
        """Each dimension score is on 0-100 scale (1-5 * 20)."""
        score = engine.score_credit(removal_credit)
        # removal_credit has all scores = 4 => 4*20 = 80
        assert score.additionality == Decimal("80.0")
        assert score.permanence == Decimal("80.0")
        assert score.co_benefits == Decimal("80.0")
        assert score.leakage_risk == Decimal("80.0")
        assert score.mrv_quality == Decimal("80.0")

    def test_overall_score_is_weighted_average(self, engine, removal_credit):
        """Overall score = weighted sum of dimension scores."""
        score = engine.score_credit(removal_credit)
        # All dimensions at 80, weights sum to 1.0, so overall = 80
        assert score.overall_score == Decimal("80.0")

    def test_quality_tier_high(self, engine, removal_credit):
        """Score >= 70 yields 'High' tier."""
        score = engine.score_credit(removal_credit)
        assert score.quality_tier == "High"

    def test_quality_tier_medium(self, engine):
        """Score between 45 and 70 yields 'Medium' tier."""
        credit = CreditEntry(
            standard=CreditStandard.CDM,
            credit_type=CreditType.AVOIDANCE,
            vintage_year=2022,
            quantity_tco2e=Decimal("100"),
            additionality_score=3,
            permanence_score=3,
            co_benefits_score=2,
            leakage_risk_score=3,
            mrv_quality_score=3,
        )
        score = engine.score_credit(credit)
        # Scores: 60,60,40,60,60 -> weighted = 60*0.25+60*0.25+40*0.15+60*0.15+60*0.20
        # = 15+15+6+9+12 = 57
        assert score.quality_tier == "Medium"

    def test_quality_tier_low(self, engine):
        """Score < 45 yields 'Low' tier."""
        credit = CreditEntry(
            standard=CreditStandard.CUSTOM,
            credit_type=CreditType.AVOIDANCE,
            vintage_year=2022,
            quantity_tco2e=Decimal("100"),
            additionality_score=1,
            permanence_score=1,
            co_benefits_score=1,
            leakage_risk_score=1,
            mrv_quality_score=1,
        )
        score = engine.score_credit(credit)
        # All 1 => 20 per dimension => overall = 20
        assert score.quality_tier == "Low"
        assert score.overall_score == Decimal("20.0")

    @pytest.mark.parametrize(
        "add,perm,co,leak,mrv,expected_overall",
        [
            (5, 5, 5, 5, 5, Decimal("100.0")),
            (1, 1, 1, 1, 1, Decimal("20.0")),
            (3, 3, 3, 3, 3, Decimal("60.0")),
            (5, 1, 3, 3, 5, Decimal("68.0")),
        ],
    )
    def test_parameterized_quality_scores(
        self, engine, add, perm, co, leak, mrv, expected_overall
    ):
        """Validate overall score from various score combinations."""
        credit = CreditEntry(
            standard=CreditStandard.VERRA_VCS,
            credit_type=CreditType.REMOVAL,
            vintage_year=2025,
            quantity_tco2e=Decimal("100"),
            additionality_score=add,
            permanence_score=perm,
            co_benefits_score=co,
            leakage_risk_score=leak,
            mrv_quality_score=mrv,
        )
        score = engine.score_credit(credit)
        # Overall = 20*add*0.25 + 20*perm*0.25 + 20*co*0.15 + 20*leak*0.15 + 20*mrv*0.20
        assert score.overall_score == expected_overall

    def test_provenance_hash_on_quality_score(self, engine, removal_credit):
        """Quality score has a provenance hash."""
        score = engine.score_credit(removal_credit)
        assert score.provenance_hash
        assert len(score.provenance_hash) == 64


# ========================================================================
# Portfolio Diversification
# ========================================================================


class TestPortfolioDiversification:
    """Tests for portfolio diversification scoring."""

    def test_single_credit_diversification(self, engine, removal_credit):
        """Single credit has low diversification."""
        engine.add_credit(removal_credit)
        result = engine.analyze_portfolio()
        assert result.portfolio_summary.credit_count == 1

    def test_diverse_portfolio_higher_score(self, populated_engine):
        """Multiple credit types yield diversification > 0."""
        result = populated_engine.analyze_portfolio()
        assert result.portfolio_summary.diversification_score > Decimal("0")

    def test_portfolio_credits_by_type(self, populated_engine):
        """Summary breaks down credits by type."""
        result = populated_engine.analyze_portfolio()
        assert "removal" in result.portfolio_summary.credits_by_type
        assert "avoidance" in result.portfolio_summary.credits_by_type


# ========================================================================
# Standard Assessment
# ========================================================================


class TestStandardAssessment:
    """Tests for credit standard quality benchmarks."""

    @pytest.mark.parametrize(
        "standard,expected_name",
        [
            (CreditStandard.VERRA_VCS, "Verified Carbon Standard (Verra)"),
            (CreditStandard.GOLD_STANDARD, "Gold Standard for the Global Goals"),
            (CreditStandard.ACR, "American Carbon Registry"),
            (CreditStandard.CAR, "Climate Action Reserve"),
            (CreditStandard.CORSIA, "CORSIA Eligible Emissions Units"),
            (CreditStandard.ART_TREES, "ART TREES (Jurisdictional REDD+)"),
        ],
    )
    def test_standard_benchmark_names(self, standard, expected_name):
        """Standard benchmarks contain correct names."""
        assert STANDARD_BENCHMARKS[standard.value]["name"] == expected_name

    def test_gold_standard_highest_base_quality(self):
        """Gold Standard has the highest base quality among common standards."""
        gs_score = STANDARD_BENCHMARKS[CreditStandard.GOLD_STANDARD.value]["base_quality_score"]
        vcs_score = STANDARD_BENCHMARKS[CreditStandard.VERRA_VCS.value]["base_quality_score"]
        assert gs_score > vcs_score


# ========================================================================
# Vintage Management
# ========================================================================


class TestVintageManagement:
    """Tests for credit vintage tracking."""

    def test_credits_by_vintage_in_summary(self, populated_engine):
        """Portfolio summary includes vintage breakdown."""
        result = populated_engine.analyze_portfolio()
        summary = result.portfolio_summary
        assert len(summary.credits_by_vintage) > 0

    def test_average_vintage_year_computed(self, populated_engine):
        """Average vintage year is computed."""
        result = populated_engine.analyze_portfolio()
        assert result.portfolio_summary.average_vintage_year > Decimal("0")


# ========================================================================
# SBTi Compliance
# ========================================================================


class TestSBTiComplianceCheck:
    """Tests for SBTi compliance assessment."""

    def test_sbti_compliance_present(self, populated_engine):
        """SBTi compliance result is generated."""
        result = populated_engine.analyze_portfolio()
        assert result.sbti_compliance is not None
        assert isinstance(result.sbti_compliance, SBTiComplianceResult)

    def test_sbti_distinguishes_bvcm_from_neutralization(self, populated_engine):
        """SBTi assessment separates BVCM from neutralization credits."""
        result = populated_engine.analyze_portfolio()
        sbti = result.sbti_compliance
        assert sbti.bvcm_credits_tco2e >= Decimal("0")
        assert sbti.neutralization_credits_tco2e >= Decimal("0")

    def test_neutralization_removal_only_check(self, engine, removal_credit):
        """Pure removal portfolio passes neutralization removal check."""
        engine.add_credit(removal_credit)
        result = engine.analyze_portfolio()
        assert result.sbti_compliance.neutralization_is_removal_only is True


# ========================================================================
# VCMI Claims Alignment
# ========================================================================


class TestVCMIClaimsAlignment:
    """Tests for VCMI Claims Code tier alignment."""

    def test_vcmi_alignment_present(self, populated_engine):
        """VCMI alignment result is generated."""
        result = populated_engine.analyze_portfolio(
            reduction_progress_pct=Decimal("55"),
        )
        assert result.vcmi_alignment is not None
        assert isinstance(result.vcmi_alignment, VCMIAlignmentResult)

    @pytest.mark.parametrize(
        "progress_pct,expected_claim_minimum",
        [
            (Decimal("90"), VCMIClaim.PLATINUM),
            (Decimal("65"), VCMIClaim.GOLD),
            (Decimal("45"), VCMIClaim.SILVER),
            (Decimal("10"), VCMIClaim.NOT_ELIGIBLE),
        ],
    )
    def test_vcmi_tier_by_progress(self, engine, removal_credit, progress_pct, expected_claim_minimum):
        """VCMI tier depends on reduction progress percentage."""
        # High-quality removal credit
        engine.add_credit(removal_credit)
        result = engine.analyze_portfolio(
            reduction_progress_pct=progress_pct,
        )
        vcmi = result.vcmi_alignment
        # Verify the eligible claim is at least as high as expected
        # (may exceed if quality is exceptionally high)
        tier_order = [VCMIClaim.NOT_ELIGIBLE, VCMIClaim.SILVER, VCMIClaim.GOLD, VCMIClaim.PLATINUM]
        actual_idx = tier_order.index(vcmi.eligible_claim)
        expected_idx = tier_order.index(expected_claim_minimum)
        assert actual_idx >= expected_idx

    def test_vcmi_thresholds_reference_data(self):
        """VCMI thresholds match expected values."""
        assert VCMI_THRESHOLDS[VCMIClaim.PLATINUM.value]["min_reduction_progress_pct"] == Decimal("80")
        assert VCMI_THRESHOLDS[VCMIClaim.GOLD.value]["min_reduction_progress_pct"] == Decimal("60")
        assert VCMI_THRESHOLDS[VCMIClaim.SILVER.value]["min_reduction_progress_pct"] == Decimal("40")


# ========================================================================
# Oxford Principles Alignment
# ========================================================================


class TestOxfordPrinciplesAlignment:
    """Tests for Oxford Principles alignment."""

    def test_oxford_alignment_present(self, populated_engine):
        """Oxford Principles alignment result is generated."""
        result = populated_engine.analyze_portfolio(assessment_year=2026)
        assert result.oxford_alignment is not None
        assert isinstance(result.oxford_alignment, OxfordAlignmentResult)

    def test_oxford_removal_share_calculated(self, populated_engine):
        """Oxford assessment computes current removal share."""
        result = populated_engine.analyze_portfolio(assessment_year=2026)
        oxford = result.oxford_alignment
        assert oxford.current_removal_share_pct >= Decimal("0")

    def test_oxford_targets_by_decade(self):
        """Oxford Principles targets increase over decades."""
        t2025 = OXFORD_PRINCIPLES_TARGETS["2025"]["min_removal_share_pct"]
        t2030 = OXFORD_PRINCIPLES_TARGETS["2030"]["min_removal_share_pct"]
        t2050 = OXFORD_PRINCIPLES_TARGETS["2050"]["min_removal_share_pct"]
        assert t2025 < t2030 < t2050


# ========================================================================
# Cost Per tCO2e
# ========================================================================


class TestCostPerTCO2e:
    """Tests for credit cost calculations."""

    def test_auto_total_cost_calculation(self, engine):
        """Total cost is auto-calculated from unit price * quantity."""
        credit = CreditEntry(
            standard=CreditStandard.VERRA_VCS,
            credit_type=CreditType.REMOVAL,
            vintage_year=2025,
            quantity_tco2e=Decimal("100"),
            unit_price_usd=Decimal("50"),
        )
        registered = engine.add_credit(credit)
        assert registered.total_cost_usd == Decimal("5000.000")

    def test_weighted_average_price(self, populated_engine):
        """Portfolio summary computes weighted average price."""
        result = populated_engine.analyze_portfolio()
        assert result.portfolio_summary.weighted_avg_price_usd > Decimal("0")

    def test_total_portfolio_cost(self, populated_engine):
        """Total portfolio cost is positive."""
        result = populated_engine.analyze_portfolio()
        assert result.portfolio_summary.total_cost_usd > Decimal("0")


# ========================================================================
# Retirement Tracking
# ========================================================================


class TestRetirementTracking:
    """Tests for credit retirement operations."""

    def test_retire_credit_changes_status(self, engine, removal_credit):
        """Retiring a credit sets status to RETIRED."""
        added = engine.add_credit(removal_credit)
        retired = engine.retire_credit(added.credit_id)
        assert retired is not None
        assert retired.status == RetirementStatus.RETIRED
        assert retired.retirement_date is not None

    def test_retire_nonexistent_credit(self, engine):
        """Retiring a nonexistent credit returns None."""
        result = engine.retire_credit("nonexistent-id")
        assert result is None

    def test_retired_credits_in_summary(self, engine, removal_credit):
        """Retired credits appear in the summary total."""
        added = engine.add_credit(removal_credit)
        engine.retire_credit(added.credit_id)
        result = engine.analyze_portfolio()
        assert result.portfolio_summary.total_retired_tco2e > Decimal("0")

    def test_clear_portfolio(self, populated_engine):
        """clear_portfolio empties all credits."""
        populated_engine.clear_portfolio()
        assert len(populated_engine._credits) == 0


# ========================================================================
# Provenance Hash
# ========================================================================


class TestOffsetProvenanceHash:
    """Tests for SHA-256 provenance on portfolio results."""

    def test_portfolio_result_has_provenance(self, populated_engine):
        """Portfolio result has a 64-char hex hash."""
        result = populated_engine.analyze_portfolio()
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_provenance_hash_is_valid_sha256(self, engine):
        """Portfolio provenance hash is a valid SHA-256 hex string."""
        c = CreditEntry(
            standard=CreditStandard.VERRA_VCS,
            credit_type=CreditType.REMOVAL,
            vintage_year=2025,
            quantity_tco2e=Decimal("100"),
            unit_price_usd=Decimal("50"),
            additionality_score=4,
            permanence_score=4,
            co_benefits_score=4,
            leakage_risk_score=4,
            mrv_quality_score=4,
        )
        engine.add_credit(c)
        r1 = engine.analyze_portfolio(
            reduction_progress_pct=Decimal("50"), assessment_year=2026
        )
        assert len(r1.provenance_hash) == 64
        assert all(ch in "0123456789abcdef" for ch in r1.provenance_hash)

    def test_credit_entry_has_provenance_after_add(self, engine, removal_credit):
        """Credit entry gets a provenance hash upon add."""
        added = engine.add_credit(removal_credit)
        assert added.provenance_hash
        assert len(added.provenance_hash) == 64


# ========================================================================
# Result Structure
# ========================================================================


class TestPortfolioResultStructure:
    """Validate complete result structure."""

    def test_result_fields(self, populated_engine):
        """PortfolioResult has all expected fields."""
        result = populated_engine.analyze_portfolio(
            reduction_progress_pct=Decimal("55"), assessment_year=2026
        )
        assert isinstance(result, PortfolioResult)
        assert result.result_id
        assert result.engine_version == "1.0.0"
        assert len(result.credits) == 3
        assert isinstance(result.portfolio_summary, PortfolioSummary)
        assert len(result.quality_scores) == 3
        assert result.average_quality_score > Decimal("0")
        assert result.processing_time_ms >= 0.0
        assert isinstance(result.recommendations, list)

    def test_empty_portfolio_analysis(self, engine):
        """Analyzing empty portfolio produces valid result."""
        result = engine.analyze_portfolio()
        assert isinstance(result, PortfolioResult)
        assert result.portfolio_summary.credit_count == 0
        assert result.portfolio_summary.total_credits_tco2e == Decimal("0")
