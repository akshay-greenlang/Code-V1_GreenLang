# -*- coding: utf-8 -*-
"""
Deep tests for PledgeCommitmentEngine (Engine 1 of 10).

Covers: 8 eligibility criteria validation, 7 actor types with scope
coverage requirements, quality scoring (STRONG/ADEQUATE/WEAK/INELIGIBLE),
partner alignment scoring, SHA-256 provenance, Decimal arithmetic,
edge cases, and performance benchmarks.

Target: ~60 tests.

Author: GreenLang Platform Team
Pack: PACK-025 Race to Zero Pack
"""

import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from engines.pledge_commitment_engine import (
    ActorType,
    CriterionStatus,
    EligibilityStatus,
    PartnerInitiative,
    PledgeCommitmentEngine,
    PledgeCommitmentInput,
    PledgeCriterionInput,
    PledgeQuality,
    PartnerAlignmentInput,
    CRITERION_IDS,
    CRITERION_WEIGHTS,
    CORE_CRITERIA,
    SCOPE_COVERAGE_REQUIREMENTS,
    ACTOR_PARTNER_MAP,
    MAX_NET_ZERO_YEAR,
    STATUS_SCORES,
    QUALITY_THRESHOLDS,
)

from conftest import assert_decimal_close, assert_provenance_hash, timed_block


# ========================================================================
# Criterion ID & Weight Validation
# ========================================================================


class TestCriterionConstants:
    """Validate criterion IDs, weights, and core criteria constants."""

    def test_exactly_8_criterion_ids(self):
        assert len(CRITERION_IDS) == 8

    def test_all_criterion_ids_present(self):
        expected = {
            "net_zero_commitment", "partner_initiative", "interim_target",
            "action_plan", "annual_reporting", "scope_coverage",
            "governance", "public_disclosure",
        }
        assert set(CRITERION_IDS) == expected

    def test_weights_sum_to_one(self):
        total = sum(CRITERION_WEIGHTS.values())
        assert_decimal_close(total, Decimal("1.00"), Decimal("0.001"))

    def test_each_weight_positive(self):
        for cid, w in CRITERION_WEIGHTS.items():
            assert w > Decimal("0"), f"Weight for {cid} must be positive"

    def test_all_weights_have_matching_id(self):
        for cid in CRITERION_IDS:
            assert cid in CRITERION_WEIGHTS

    def test_core_criteria_count(self):
        assert len(CORE_CRITERIA) == 5

    def test_core_criteria_are_subset_of_ids(self):
        for c in CORE_CRITERIA:
            assert c in CRITERION_IDS

    def test_net_zero_commitment_is_highest_weight(self):
        max_cid = max(CRITERION_WEIGHTS, key=CRITERION_WEIGHTS.get)
        assert max_cid == "net_zero_commitment"

    def test_net_zero_weight_is_020(self):
        assert CRITERION_WEIGHTS["net_zero_commitment"] == Decimal("0.20")

    def test_public_disclosure_is_lowest_weight(self):
        min_cid = min(CRITERION_WEIGHTS, key=CRITERION_WEIGHTS.get)
        assert min_cid == "public_disclosure"


# ========================================================================
# Enum Validation
# ========================================================================


class TestPledgeEnums:
    """Test enum values and counts."""

    def test_actor_type_has_7_values(self):
        assert len(ActorType) == 7

    @pytest.mark.parametrize("actor", [
        "corporate", "financial_institution", "city",
        "region", "sme", "university", "healthcare",
    ])
    def test_actor_type_value(self, actor):
        assert ActorType(actor) is not None

    def test_partner_initiative_count(self):
        assert len(PartnerInitiative) >= 15

    def test_eligibility_status_values(self):
        assert EligibilityStatus.ELIGIBLE.value == "eligible"
        assert EligibilityStatus.CONDITIONAL.value == "conditional"
        assert EligibilityStatus.INELIGIBLE.value == "ineligible"

    def test_pledge_quality_values(self):
        assert PledgeQuality.STRONG.value == "strong"
        assert PledgeQuality.ADEQUATE.value == "adequate"
        assert PledgeQuality.WEAK.value == "weak"
        assert PledgeQuality.INELIGIBLE.value == "ineligible"

    def test_criterion_status_values(self):
        assert CriterionStatus.PASS.value == "pass"
        assert CriterionStatus.PARTIAL.value == "partial"
        assert CriterionStatus.FAIL.value == "fail"
        assert CriterionStatus.NOT_APPLICABLE.value == "not_applicable"

    def test_status_scores_mapping(self):
        assert STATUS_SCORES["pass"] == Decimal("1.0")
        assert STATUS_SCORES["partial"] == Decimal("0.5")
        assert STATUS_SCORES["fail"] == Decimal("0.0")


# ========================================================================
# Scope Coverage Requirements
# ========================================================================


class TestScopeCoverageRequirements:
    """Validate scope coverage requirements per actor type."""

    def test_all_actor_types_have_requirements(self):
        for at in ActorType:
            assert at.value in SCOPE_COVERAGE_REQUIREMENTS

    @pytest.mark.parametrize("actor_type,expected_s3", [
        ("corporate", Decimal("67")),
        ("financial_institution", Decimal("67")),
        ("city", Decimal("0")),
        ("region", Decimal("0")),
        ("sme", Decimal("40")),
        ("university", Decimal("50")),
        ("healthcare", Decimal("50")),
    ])
    def test_scope3_requirements(self, actor_type, expected_s3):
        req = SCOPE_COVERAGE_REQUIREMENTS[actor_type]
        assert req["scope3_pct"] == expected_s3

    def test_corporate_scope1_requirement(self):
        req = SCOPE_COVERAGE_REQUIREMENTS["corporate"]
        assert req["scope1_pct"] == Decimal("95")

    def test_city_scope1_lower_requirement(self):
        req = SCOPE_COVERAGE_REQUIREMENTS["city"]
        assert req["scope1_pct"] == Decimal("90")


# ========================================================================
# Actor-Partner Mapping
# ========================================================================


class TestActorPartnerMapping:
    """Validate actor-type to partner initiative mapping."""

    def test_all_actor_types_mapped(self):
        for at in ActorType:
            assert at.value in ACTOR_PARTNER_MAP

    def test_corporate_partners_include_sbti(self):
        assert "sbti" in ACTOR_PARTNER_MAP["corporate"]

    def test_financial_partners_include_gfanz(self):
        assert "gfanz" in ACTOR_PARTNER_MAP["financial_institution"]

    def test_city_partners_include_c40(self):
        assert "c40" in ACTOR_PARTNER_MAP["city"]

    def test_sme_partners_include_sme_hub(self):
        assert "sme_climate_hub" in ACTOR_PARTNER_MAP["sme"]

    def test_university_partners_include_second_nature(self):
        assert "second_nature" in ACTOR_PARTNER_MAP["university"]

    def test_healthcare_partners_include_hcwh(self):
        assert "hcwh" in ACTOR_PARTNER_MAP["healthcare"]

    def test_region_partners_include_under2(self):
        assert "under2" in ACTOR_PARTNER_MAP["region"]


# ========================================================================
# Quality Thresholds
# ========================================================================


class TestQualityThresholds:
    """Validate quality tier thresholds."""

    def test_strong_threshold_85(self):
        assert QUALITY_THRESHOLDS[0] == (Decimal("85"), "strong")

    def test_adequate_threshold_65(self):
        assert QUALITY_THRESHOLDS[1] == (Decimal("65"), "adequate")

    def test_weak_threshold_40(self):
        assert QUALITY_THRESHOLDS[2] == (Decimal("40"), "weak")

    def test_ineligible_threshold_0(self):
        assert QUALITY_THRESHOLDS[3] == (Decimal("0"), "ineligible")

    def test_max_net_zero_year(self):
        assert MAX_NET_ZERO_YEAR == 2050


# ========================================================================
# Pydantic Model Validation
# ========================================================================


class TestPledgeInputModel:
    """Validate PledgeCommitmentInput Pydantic model."""

    def test_strong_input_constructs(self, strong_pledge_input):
        assert strong_pledge_input.entity_name == "GreenCorp International"
        assert strong_pledge_input.net_zero_target_year == 2050

    def test_weak_input_constructs(self, weak_pledge_input):
        assert weak_pledge_input.entity_name == "SlowStart Ltd"
        assert weak_pledge_input.net_zero_target_year == 2055

    def test_input_scope_coverage_decimal(self, strong_pledge_input):
        assert isinstance(strong_pledge_input.scope1_coverage_pct, Decimal)
        assert strong_pledge_input.scope1_coverage_pct == Decimal("100")

    def test_input_baseline_emissions_decimal(self, strong_pledge_input):
        assert isinstance(strong_pledge_input.baseline_emissions_tco2e, Decimal)

    def test_partner_alignment_input_validation(self):
        p = PartnerAlignmentInput(partner_id="sbti", membership_status="active")
        assert p.partner_id == "sbti"

    def test_invalid_partner_id_raises(self):
        with pytest.raises(Exception):
            PartnerAlignmentInput(partner_id="fake_partner")

    def test_invalid_membership_status_raises(self):
        with pytest.raises(Exception):
            PartnerAlignmentInput(
                partner_id="sbti", membership_status="bogus"
            )

    def test_criterion_input_validation(self):
        c = PledgeCriterionInput(
            criterion_id="net_zero_commitment",
            status="pass",
        )
        assert c.criterion_id == "net_zero_commitment"

    def test_invalid_criterion_id_raises(self):
        with pytest.raises(Exception):
            PledgeCriterionInput(criterion_id="fake_id")

    def test_invalid_criterion_status_raises(self):
        with pytest.raises(Exception):
            PledgeCriterionInput(
                criterion_id="net_zero_commitment",
                status="bogus_status",
            )


# ========================================================================
# Engine Instantiation & Configuration
# ========================================================================


class TestPledgeEngineInstantiation:
    """Tests for engine creation and configuration."""

    def test_default_instantiation(self, pledge_engine):
        assert pledge_engine is not None

    def test_custom_config(self, pledge_engine_custom):
        assert pledge_engine_custom is not None

    def test_engine_has_calculate(self, pledge_engine):
        assert callable(getattr(pledge_engine, "assess", None))

    def test_engine_class_name(self):
        assert PledgeCommitmentEngine.__name__ == "PledgeCommitmentEngine"


# ========================================================================
# Engine Calculation -- Strong Pledge
# ========================================================================


class TestStrongPledgeAssessment:
    """Tests for a strong/eligible pledge assessment."""

    def test_strong_pledge_calculates(self, pledge_engine, strong_pledge_input):
        result = pledge_engine.assess(strong_pledge_input)
        assert result is not None

    def test_strong_pledge_eligible(self, pledge_engine, strong_pledge_input):
        result = pledge_engine.assess(strong_pledge_input)
        assert result.eligibility_status in (
            EligibilityStatus.ELIGIBLE.value, "eligible"
        )

    def test_strong_pledge_quality_strong_or_adequate(
        self, pledge_engine, strong_pledge_input,
    ):
        result = pledge_engine.assess(strong_pledge_input)
        assert result.pledge_quality in ("strong", "adequate")

    def test_strong_pledge_quality_score_above_65(
        self, pledge_engine, strong_pledge_input,
    ):
        result = pledge_engine.assess(strong_pledge_input)
        assert result.quality_score >= Decimal("65")

    def test_strong_pledge_has_provenance(
        self, pledge_engine, strong_pledge_input,
    ):
        result = pledge_engine.assess(strong_pledge_input)
        assert_provenance_hash(result)

    def test_strong_pledge_has_processing_time(
        self, pledge_engine, strong_pledge_input,
    ):
        result = pledge_engine.assess(strong_pledge_input)
        assert result.processing_time_ms >= 0

    def test_strong_pledge_has_result_id(
        self, pledge_engine, strong_pledge_input,
    ):
        result = pledge_engine.assess(strong_pledge_input)
        assert result.result_id is not None
        assert len(result.result_id) > 0

    def test_strong_pledge_has_criterion_results(
        self, pledge_engine, strong_pledge_input,
    ):
        result = pledge_engine.assess(strong_pledge_input)
        assert hasattr(result, "criteria_results") or hasattr(result, "criterion_results")

    def test_strong_pledge_entity_name(
        self, pledge_engine, strong_pledge_input,
    ):
        result = pledge_engine.assess(strong_pledge_input)
        assert result.entity_name == "GreenCorp International"

    def test_strong_pledge_performance(
        self, pledge_engine, strong_pledge_input,
    ):
        """Assessment should complete within 5 seconds."""
        with timed_block("strong_pledge_assessment", max_seconds=5.0):
            pledge_engine.assess(strong_pledge_input)


# ========================================================================
# Engine Calculation -- Weak Pledge
# ========================================================================


class TestWeakPledgeAssessment:
    """Tests for a weak/ineligible pledge assessment."""

    def test_weak_pledge_calculates(self, pledge_engine, weak_pledge_input):
        result = pledge_engine.assess(weak_pledge_input)
        assert result is not None

    def test_weak_pledge_ineligible_or_conditional(
        self, pledge_engine, weak_pledge_input,
    ):
        result = pledge_engine.assess(weak_pledge_input)
        assert result.eligibility_status in (
            "ineligible", "conditional",
        )

    def test_weak_pledge_quality_below_strong(
        self, pledge_engine, weak_pledge_input,
    ):
        result = pledge_engine.assess(weak_pledge_input)
        assert result.quality_score < Decimal("85")

    def test_weak_pledge_has_gaps(self, pledge_engine, weak_pledge_input):
        result = pledge_engine.assess(weak_pledge_input)
        has_gaps = (
            hasattr(result, "gaps") and len(result.gaps) > 0
        ) or (
            hasattr(result, "recommendations") and len(result.recommendations) > 0
        )
        assert has_gaps

    def test_weak_pledge_provenance(self, pledge_engine, weak_pledge_input):
        result = pledge_engine.assess(weak_pledge_input)
        assert_provenance_hash(result)


# ========================================================================
# Determinism & Reproducibility
# ========================================================================


class TestPledgeDeterminism:
    """Tests for deterministic calculation output."""

    def test_same_input_same_quality_score(
        self, pledge_engine, strong_pledge_input,
    ):
        r1 = pledge_engine.assess(strong_pledge_input)
        r2 = pledge_engine.assess(strong_pledge_input)
        assert r1.quality_score == r2.quality_score

    def test_same_input_same_eligibility(
        self, pledge_engine, strong_pledge_input,
    ):
        r1 = pledge_engine.assess(strong_pledge_input)
        r2 = pledge_engine.assess(strong_pledge_input)
        assert r1.eligibility_status == r2.eligibility_status

    def test_same_input_same_quality_tier(
        self, pledge_engine, strong_pledge_input,
    ):
        r1 = pledge_engine.assess(strong_pledge_input)
        r2 = pledge_engine.assess(strong_pledge_input)
        assert r1.pledge_quality == r2.pledge_quality
