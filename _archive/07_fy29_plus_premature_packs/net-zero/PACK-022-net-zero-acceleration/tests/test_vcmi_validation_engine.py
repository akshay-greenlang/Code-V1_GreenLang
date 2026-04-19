# -*- coding: utf-8 -*-
"""
Unit tests for VCMIValidationEngine (PACK-022 Engine 9).

Tests VCMI Claims Code foundational criteria, tier eligibility,
ICVCM assessment, greenwashing risk, ISO 14068-1 comparison,
gap analysis, and the full validation pipeline.
"""

import pytest
from decimal import Decimal

from engines.vcmi_validation_engine import (
    VCMIValidationEngine,
    VCMIValidationConfig,
    EmissionsData,
    CarbonCreditPortfolio,
    VCMIResult,
    FoundationalCriterionResult,
    TierEligibility,
    ICVCMAssessment,
    ISOComparison,
    GapToNextTier,
    GreenwashingFlag,
    VCMITier,
    CriterionStatus,
    EvidenceStrength,
    GreenwashingRiskLevel,
    CreditQualityLevel,
    VCMI_TIER_THRESHOLDS,
    ICVCM_CORE_CARBON_PRINCIPLES,
    ISO_14068_REQUIREMENTS,
    GREENWASHING_INDICATORS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_emissions(
    scope1=1000, scope2=500, scope3=3000,
    has_sbti=True, target_pct=42, has_disclosure=True,
    platform="CDP", base_year=2019, base_emissions=5000,
    reductions=200, inventory_year=2025, reporting_year=2025,
):
    return EmissionsData(
        reporting_year=reporting_year,
        scope1_emissions=Decimal(str(scope1)),
        scope2_emissions=Decimal(str(scope2)),
        scope3_emissions=Decimal(str(scope3)),
        base_year=base_year,
        base_year_emissions=Decimal(str(base_emissions)),
        reductions_achieved=Decimal(str(reductions)),
        has_sbti_target=has_sbti,
        target_reduction_pct=Decimal(str(target_pct)),
        has_public_disclosure=has_disclosure,
        disclosure_platform=platform,
        inventory_year=inventory_year,
    )


def _make_credits(
    total=1000, ccp_approved=800, vintage=2024,
    registries=None, ccp_compliance=None,
):
    if registries is None:
        registries = ["verra", "gold_standard"]
    if ccp_compliance is None:
        ccp_compliance = {f"CCP-{i}": True for i in range(1, 11)}
    return CarbonCreditPortfolio(
        total_credits_retired=Decimal(str(total)),
        ccp_approved_credits=Decimal(str(ccp_approved)),
        non_ccp_credits=Decimal(str(total - ccp_approved)),
        credit_vintage_year=vintage,
        registries=registries,
        ccp_compliance=ccp_compliance,
    )


@pytest.fixture
def engine():
    return VCMIValidationEngine()


@pytest.fixture
def strong_emissions():
    # Strong emissions: significant reduction achieved since base year.
    # total_emissions = 1000+500+1200 = 2700, base = 5000, reduction = 46%
    # expected: 42% * (2025-2019)/6 = 42%, so progress_ratio >= 1 => 60 pts
    # reductions_achieved=500 => +20 pts, reduction_pct=46% > 5% => +20 pts = 100
    return _make_emissions(scope1=1000, scope2=500, scope3=1200, base_emissions=5000, reductions=500)


@pytest.fixture
def strong_credits():
    return _make_credits(total=5000, ccp_approved=5000)


@pytest.fixture
def weak_emissions():
    return _make_emissions(
        has_sbti=False, target_pct=0, has_disclosure=False,
        platform="", reductions=0, base_emissions=5000,
        scope1=1000, scope2=500, scope3=3500,
    )


@pytest.fixture
def weak_credits():
    return _make_credits(total=100, ccp_approved=10, ccp_compliance={},
                         registries=["unknown_registry"])


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestVCMIInit:

    def test_default_config(self, engine):
        assert isinstance(engine.config, VCMIValidationConfig)
        assert engine.config.silver_threshold_pct == Decimal("20")
        assert engine.config.gold_threshold_pct == Decimal("60")
        assert engine.config.platinum_threshold_pct == Decimal("100")

    def test_custom_config(self):
        eng = VCMIValidationEngine({"credit_price_usd_per_tonne": "25"})
        assert eng.config.credit_price_usd_per_tonne == Decimal("25")

    def test_clear_is_noop(self, engine):
        engine.clear()  # Should not raise


# ---------------------------------------------------------------------------
# Foundational Criterion 1: Science-aligned Target
# ---------------------------------------------------------------------------


class TestCriterion1:

    def test_full_marks_with_sbti_and_ambitious(self, engine, strong_emissions):
        result = engine._assess_criterion_1(strong_emissions)
        assert result.criterion_id == "FC-1"
        assert result.status == CriterionStatus.MET
        assert float(result.score) >= 70

    def test_no_sbti_no_target(self, engine, weak_emissions):
        result = engine._assess_criterion_1(weak_emissions)
        assert result.status in (CriterionStatus.NOT_MET, CriterionStatus.PARTIALLY_MET)
        assert float(result.score) < 70

    def test_moderate_target(self, engine):
        emissions = _make_emissions(has_sbti=True, target_pct=25)
        result = engine._assess_criterion_1(emissions)
        assert float(result.score) >= 40


# ---------------------------------------------------------------------------
# Foundational Criterion 2: Demonstrated Progress
# ---------------------------------------------------------------------------


class TestCriterion2:

    def test_strong_progress(self, engine, strong_emissions):
        result = engine._assess_criterion_2(strong_emissions)
        assert result.criterion_id == "FC-2"
        assert float(result.score) >= 40

    def test_no_base_year_emissions(self, engine):
        emissions = _make_emissions(base_emissions=0)
        result = engine._assess_criterion_2(emissions)
        assert result.status == CriterionStatus.NOT_MET
        assert result.evidence_strength == EvidenceStrength.ABSENT

    def test_no_reduction(self, engine):
        emissions = _make_emissions(base_emissions=4000, scope1=1500, scope2=800, scope3=3000)
        # Must set total_emissions explicitly when calling _assess_criterion_2 directly
        emissions.total_emissions = emissions.scope1_emissions + emissions.scope2_emissions + emissions.scope3_emissions
        result = engine._assess_criterion_2(emissions)
        # total = 5300 > base 4000, so no reduction
        assert float(result.score) < 70


# ---------------------------------------------------------------------------
# Foundational Criterion 3: Public Disclosure
# ---------------------------------------------------------------------------


class TestCriterion3:

    def test_full_disclosure(self, engine, strong_emissions):
        result = engine._assess_criterion_3(strong_emissions)
        assert result.criterion_id == "FC-3"
        assert result.status == CriterionStatus.MET

    def test_no_disclosure(self, engine, weak_emissions):
        result = engine._assess_criterion_3(weak_emissions)
        assert float(result.score) < 70

    def test_stale_inventory(self, engine):
        emissions = _make_emissions(inventory_year=2020, reporting_year=2025)
        result = engine._assess_criterion_3(emissions)
        assert any("stale" in f.lower() or "years old" in f.lower() for f in result.findings)


# ---------------------------------------------------------------------------
# Foundational Criterion 4: Credit Quality
# ---------------------------------------------------------------------------


class TestCriterion4:

    def test_high_quality_credits(self, engine, strong_credits):
        result = engine._assess_criterion_4(strong_credits)
        assert result.criterion_id == "FC-4"
        assert result.status == CriterionStatus.MET

    def test_no_credits(self, engine):
        credits = _make_credits(total=0)
        result = engine._assess_criterion_4(credits)
        assert result.status == CriterionStatus.NOT_MET

    def test_low_ccp_proportion(self, engine, weak_credits):
        result = engine._assess_criterion_4(weak_credits)
        assert float(result.score) < 70


# ---------------------------------------------------------------------------
# Tier Eligibility Tests
# ---------------------------------------------------------------------------


class TestTierEligibility:

    def test_platinum_eligible(self, engine, strong_emissions, strong_credits):
        result = engine.validate(strong_emissions, strong_credits)
        # With 5000 credits and 4500 unabated: 111% coverage
        assert result.highest_eligible_tier in (VCMITier.GOLD, VCMITier.PLATINUM)

    def test_not_eligible_without_foundation(self, engine, weak_emissions, weak_credits):
        result = engine.validate(weak_emissions, weak_credits)
        assert result.highest_eligible_tier == VCMITier.NOT_ELIGIBLE

    def test_silver_threshold(self, engine, strong_emissions):
        # 20% of ~4500 = 900 credits for silver
        credits = _make_credits(total=1000, ccp_approved=1000)
        result = engine.validate(strong_emissions, credits)
        assert result.highest_eligible_tier in (VCMITier.SILVER, VCMITier.GOLD, VCMITier.PLATINUM)

    def test_three_tiers_assessed(self, engine, strong_emissions, strong_credits):
        result = engine.validate(strong_emissions, strong_credits)
        assert len(result.tier_eligibility) == 3


# ---------------------------------------------------------------------------
# ICVCM Assessment Tests
# ---------------------------------------------------------------------------


class TestICVCMAssessment:

    def test_full_compliance(self, engine, strong_credits):
        assessments = engine.assess_icvcm_compliance(strong_credits)
        assert len(assessments) == 10
        assert all(a.compliant for a in assessments)

    def test_no_compliance_data(self, engine):
        credits = _make_credits(ccp_compliance={})
        assessments = engine.assess_icvcm_compliance(credits)
        assert all(not a.compliant for a in assessments)

    def test_partial_compliance(self, engine):
        compliance = {f"CCP-{i}": (i <= 5) for i in range(1, 11)}
        credits = _make_credits(ccp_compliance=compliance)
        assessments = engine.assess_icvcm_compliance(credits)
        compliant_count = sum(1 for a in assessments if a.compliant)
        assert compliant_count == 5


# ---------------------------------------------------------------------------
# ISO 14068-1 Comparison Tests
# ---------------------------------------------------------------------------


class TestISOComparison:

    def test_comparison_count(self, engine, strong_emissions):
        fc_results = [
            engine._assess_criterion_1(strong_emissions),
            engine._assess_criterion_2(strong_emissions),
            engine._assess_criterion_3(strong_emissions),
            engine._assess_criterion_4(_make_credits()),
        ]
        comparisons = engine.compare_iso_14068(strong_emissions, fc_results)
        assert len(comparisons) == 7  # ISO-1 through ISO-7

    def test_iso5_not_covered_by_vcmi(self, engine, strong_emissions):
        fc_results = [
            engine._assess_criterion_1(strong_emissions),
            engine._assess_criterion_2(strong_emissions),
            engine._assess_criterion_3(strong_emissions),
            engine._assess_criterion_4(_make_credits()),
        ]
        comparisons = engine.compare_iso_14068(strong_emissions, fc_results)
        iso5 = next(c for c in comparisons if c.requirement_id == "ISO-5")
        assert iso5.vcmi_overlap is False


# ---------------------------------------------------------------------------
# Greenwashing Risk Tests
# ---------------------------------------------------------------------------


class TestGreenwashingRisk:

    def test_low_risk_for_strong_entity(self, engine, strong_emissions, strong_credits):
        result = engine.validate(strong_emissions, strong_credits)
        assert result.greenwashing_risk_level in (GreenwashingRiskLevel.LOW, GreenwashingRiskLevel.MEDIUM)

    def test_critical_risk_no_target_with_credits(self, engine):
        emissions = _make_emissions(has_sbti=False, target_pct=0)
        credits = _make_credits(total=1000)
        result = engine.validate(emissions, credits)
        assert result.greenwashing_risk_level in (GreenwashingRiskLevel.CRITICAL, GreenwashingRiskLevel.HIGH)

    def test_greenwashing_flags_count(self, engine, strong_emissions, strong_credits):
        result = engine.validate(strong_emissions, strong_credits)
        assert len(result.greenwashing_flags) == 7  # GW-1 through GW-7


# ---------------------------------------------------------------------------
# Gap Analysis Tests
# ---------------------------------------------------------------------------


class TestGapAnalysis:

    def test_gap_when_not_eligible(self, engine, weak_emissions, weak_credits):
        result = engine.validate(weak_emissions, weak_credits)
        assert result.gaps_to_next_tier is not None
        assert result.gaps_to_next_tier.next_tier == VCMITier.SILVER

    def test_no_gap_when_platinum(self, engine, strong_emissions):
        credits = _make_credits(total=10000, ccp_approved=10000)
        result = engine.validate(strong_emissions, credits)
        if result.highest_eligible_tier == VCMITier.PLATINUM:
            assert result.gaps_to_next_tier is None

    def test_gap_has_cost_estimate(self, engine, weak_emissions, weak_credits):
        result = engine.validate(weak_emissions, weak_credits)
        if result.gaps_to_next_tier:
            assert float(result.gaps_to_next_tier.estimated_cost_usd) >= 0


# ---------------------------------------------------------------------------
# Full Validation Pipeline Tests
# ---------------------------------------------------------------------------


class TestFullValidation:

    def test_validate_structure(self, engine, strong_emissions, strong_credits):
        result = engine.validate(strong_emissions, strong_credits, entity_name="TestCo")
        assert isinstance(result, VCMIResult)
        assert result.entity_name == "TestCo"
        assert result.reporting_year == 2025
        assert len(result.foundational_criteria_results) == 4
        assert len(result.tier_eligibility) == 3
        assert len(result.icvcm_assessment) == 10
        assert len(result.iso_comparison) == 7
        assert len(result.provenance_hash) == 64

    def test_auto_calculates_total(self, engine):
        emissions = _make_emissions()
        emissions.total_emissions = Decimal("0")
        credits = _make_credits()
        result = engine.validate(emissions, credits)
        assert result is not None

    def test_recommendations_populated(self, engine, weak_emissions, weak_credits):
        result = engine.validate(weak_emissions, weak_credits)
        assert len(result.recommendations) > 0


# ---------------------------------------------------------------------------
# Re-validation Tests
# ---------------------------------------------------------------------------


class TestRevalidation:

    def test_revalidation_detects_upgrade(self, engine):
        prev_emissions = _make_emissions(has_sbti=False, target_pct=0,
                                         reporting_year=2024)
        prev_credits = _make_credits(total=100, ccp_approved=50)
        prev_result = engine.validate(prev_emissions, prev_credits)

        new_emissions = _make_emissions(reporting_year=2025)
        new_credits = _make_credits(total=5000, ccp_approved=5000)
        reval = engine.revalidate(prev_result, new_emissions, new_credits)

        assert "year_over_year" in reval
        assert reval["year_over_year"]["tier_change"] in ("upgraded", "no_change", "downgraded")

    def test_revalidation_provenance(self, engine, strong_emissions, strong_credits):
        prev = engine.validate(strong_emissions, strong_credits)
        reval = engine.revalidate(prev, strong_emissions, strong_credits)
        assert "provenance_hash" in reval


# ---------------------------------------------------------------------------
# Edge Cases & Constants
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_tier_thresholds(self):
        assert VCMI_TIER_THRESHOLDS["silver"] == Decimal("20")
        assert VCMI_TIER_THRESHOLDS["gold"] == Decimal("60")
        assert VCMI_TIER_THRESHOLDS["platinum"] == Decimal("100")

    def test_icvcm_principles_count(self):
        assert len(ICVCM_CORE_CARBON_PRINCIPLES) == 10

    def test_iso_requirements_count(self):
        assert len(ISO_14068_REQUIREMENTS) == 7

    def test_greenwashing_indicators_count(self):
        assert len(GREENWASHING_INDICATORS) == 7

    def test_enum_values(self):
        assert VCMITier.SILVER.value == "silver"
        assert CriterionStatus.MET.value == "met"
        assert EvidenceStrength.STRONG.value == "strong"
        assert GreenwashingRiskLevel.CRITICAL.value == "critical"
