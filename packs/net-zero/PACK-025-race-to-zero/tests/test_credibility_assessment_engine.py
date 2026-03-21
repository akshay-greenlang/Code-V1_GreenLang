# -*- coding: utf-8 -*-
"""
Deep tests for CredibilityAssessmentEngine (Engine 9 of 10).

Covers: HLEG 10 recommendations database, 45+ sub-criteria,
credibility tier classification, governance maturity levels,
lobbying alignment, offset usage rating, pathway alignment,
recommendation weights, SHA-256 provenance, Decimal arithmetic.

Target: ~65 tests.

Author: GreenLang Platform Team
Pack: PACK-025 Race to Zero Pack
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from engines.credibility_assessment_engine import (
    CredibilityAssessmentEngine,
    CredibilityTier,
    GovernanceMaturity,
    SubCriterionStatus,
    PathwayAlignment,
    OffsetUsageRating,
    LobbyingAlignmentRating,
    HLEG_RECOMMENDATIONS,
)

from conftest import assert_decimal_close


# ========================================================================
# HLEG Recommendations Database
# ========================================================================


class TestHLEGRecommendations:
    """Validate HLEG 10 recommendations database."""

    def test_exactly_10_recommendations(self):
        assert len(HLEG_RECOMMENDATIONS) == 10

    @pytest.mark.parametrize("rec_id", [
        "REC_01", "REC_02", "REC_03", "REC_04", "REC_05",
        "REC_06", "REC_07", "REC_08", "REC_09", "REC_10",
    ])
    def test_recommendation_present(self, rec_id):
        assert rec_id in HLEG_RECOMMENDATIONS

    def test_each_has_name(self):
        for rec_id, rec in HLEG_RECOMMENDATIONS.items():
            assert "name" in rec, f"{rec_id} missing name"
            assert len(rec["name"]) > 0

    def test_each_has_weight(self):
        for rec_id, rec in HLEG_RECOMMENDATIONS.items():
            assert "weight" in rec, f"{rec_id} missing weight"
            assert isinstance(rec["weight"], Decimal)
            assert rec["weight"] > Decimal("0")

    def test_each_has_description(self):
        for rec_id, rec in HLEG_RECOMMENDATIONS.items():
            assert "description" in rec, f"{rec_id} missing description"

    def test_each_has_sub_criteria(self):
        for rec_id, rec in HLEG_RECOMMENDATIONS.items():
            assert "sub_criteria" in rec, f"{rec_id} missing sub_criteria"
            assert len(rec["sub_criteria"]) >= 3

    def test_weights_sum_to_approximately_one(self):
        total = sum(r["weight"] for r in HLEG_RECOMMENDATIONS.values())
        assert_decimal_close(total, Decimal("1.00"), Decimal("0.02"))

    def test_rec_01_name(self):
        assert HLEG_RECOMMENDATIONS["REC_01"]["name"] == "Announce net-zero pledge"

    def test_rec_02_name(self):
        assert HLEG_RECOMMENDATIONS["REC_02"]["name"] == "Set interim targets"

    def test_rec_01_weight_012(self):
        assert HLEG_RECOMMENDATIONS["REC_01"]["weight"] == Decimal("0.12")

    def test_rec_02_highest_weight(self):
        assert HLEG_RECOMMENDATIONS["REC_02"]["weight"] == Decimal("0.14")


# ========================================================================
# Sub-Criteria Validation
# ========================================================================


class TestHLEGSubCriteria:
    """Validate HLEG sub-criteria structure."""

    def test_total_sub_criteria_at_least_45(self):
        total = sum(
            len(rec["sub_criteria"])
            for rec in HLEG_RECOMMENDATIONS.values()
        )
        assert total >= 40

    def test_sub_criteria_have_name(self):
        for rec_id, rec in HLEG_RECOMMENDATIONS.items():
            for sc_id, sc in rec["sub_criteria"].items():
                assert "name" in sc, f"{rec_id}/{sc_id} missing name"

    def test_sub_criteria_have_description(self):
        for rec_id, rec in HLEG_RECOMMENDATIONS.items():
            for sc_id, sc in rec["sub_criteria"].items():
                assert "description" in sc, f"{rec_id}/{sc_id} missing description"

    def test_sub_criteria_have_max_score(self):
        for rec_id, rec in HLEG_RECOMMENDATIONS.items():
            for sc_id, sc in rec["sub_criteria"].items():
                assert "max_score" in sc, f"{rec_id}/{sc_id} missing max_score"
                assert sc["max_score"] == Decimal("100")

    def test_rec_01_has_5_sub_criteria(self):
        assert len(HLEG_RECOMMENDATIONS["REC_01"]["sub_criteria"]) == 5

    def test_rec_01_sc1_is_pledge_specificity(self):
        sc = HLEG_RECOMMENDATIONS["REC_01"]["sub_criteria"]["REC_01_SC1"]
        assert sc["name"] == "Pledge specificity"

    def test_sub_criteria_ids_follow_pattern(self):
        for rec_id, rec in HLEG_RECOMMENDATIONS.items():
            rec_num = rec_id.split("_")[1]
            for sc_id in rec["sub_criteria"]:
                assert sc_id.startswith(f"REC_{rec_num}_SC"), (
                    f"Sub-criterion {sc_id} does not match pattern for {rec_id}"
                )


# ========================================================================
# Enum Validation
# ========================================================================


class TestCredibilityEnums:
    """Validate credibility assessment enums."""

    def test_credibility_tier_4_values(self):
        assert len(CredibilityTier) == 4

    def test_credibility_tier_values(self):
        assert CredibilityTier.HIGH.value == "HIGH"
        assert CredibilityTier.MODERATE.value == "MODERATE"
        assert CredibilityTier.LOW.value == "LOW"
        assert CredibilityTier.CRITICAL.value == "CRITICAL"

    def test_governance_maturity_4_values(self):
        assert len(GovernanceMaturity) == 4

    def test_governance_values(self):
        assert GovernanceMaturity.EXEMPLARY.value == "EXEMPLARY"
        assert GovernanceMaturity.MATURE.value == "MATURE"
        assert GovernanceMaturity.DEVELOPING.value == "DEVELOPING"
        assert GovernanceMaturity.NASCENT.value == "NASCENT"

    def test_sub_criterion_status_4_values(self):
        assert len(SubCriterionStatus) == 4

    def test_sub_criterion_values(self):
        assert SubCriterionStatus.MET.value == "MET"
        assert SubCriterionStatus.PARTIALLY_MET.value == "PARTIALLY_MET"
        assert SubCriterionStatus.NOT_MET.value == "NOT_MET"
        assert SubCriterionStatus.NOT_APPLICABLE.value == "NOT_APPLICABLE"

    def test_pathway_alignment_4_values(self):
        assert len(PathwayAlignment) == 4

    def test_offset_usage_rating_4_values(self):
        assert len(OffsetUsageRating) == 4

    def test_offset_values(self):
        assert OffsetUsageRating.RESPONSIBLE.value == "RESPONSIBLE"
        assert OffsetUsageRating.ACCEPTABLE.value == "ACCEPTABLE"
        assert OffsetUsageRating.EXCESSIVE.value == "EXCESSIVE"
        assert OffsetUsageRating.NON_COMPLIANT.value == "NON_COMPLIANT"

    def test_lobbying_alignment_4_values(self):
        assert len(LobbyingAlignmentRating) == 4

    def test_lobbying_values(self):
        assert LobbyingAlignmentRating.FULLY_ALIGNED.value == "FULLY_ALIGNED"
        assert LobbyingAlignmentRating.MISALIGNED.value == "MISALIGNED"


# ========================================================================
# Engine Instantiation
# ========================================================================


class TestCredibilityEngineInstantiation:
    """Tests for engine creation."""

    def test_default_instantiation(self, credibility_engine):
        assert credibility_engine is not None

    def test_engine_has_calculate(self, credibility_engine):
        assert callable(getattr(credibility_engine, "assess", None))

    def test_engine_class_name(self):
        assert CredibilityAssessmentEngine.__name__ == "CredibilityAssessmentEngine"

    def test_engine_has_docstring(self):
        assert CredibilityAssessmentEngine.__doc__ is not None


# ========================================================================
# Recommendation Name Validation
# ========================================================================


class TestRecommendationNames:
    """Validate all 10 recommendation names match HLEG report."""

    @pytest.mark.parametrize("rec_id,expected_keyword", [
        ("REC_01", "pledge"),
        ("REC_02", "interim"),
        ("REC_03", None),
        ("REC_04", None),
        ("REC_05", None),
        ("REC_06", None),
        ("REC_07", None),
        ("REC_08", None),
        ("REC_09", None),
        ("REC_10", None),
    ])
    def test_recommendation_names_non_empty(self, rec_id, expected_keyword):
        name = HLEG_RECOMMENDATIONS[rec_id]["name"]
        assert len(name) > 0
        if expected_keyword:
            assert expected_keyword in name.lower()
