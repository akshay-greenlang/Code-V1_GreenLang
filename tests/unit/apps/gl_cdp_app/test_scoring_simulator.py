# -*- coding: utf-8 -*-
"""
Unit tests for ScoringSimulator -- CDP scoring algorithm.

Tests scoring for each of 17 categories, category weighting, overall score
calculation, score band determination, what-if analysis, A-level eligibility,
score trajectory, confidence intervals, and score comparison with 42+ tests.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import ScoringCategory, ScoringLevel, ScoreBand
from services.models import (
    CDPCategoryScore,
    CDPScoringResult,
    CDPResponse,
    _new_id,
)
from services.scoring_simulator import ScoringSimulator


# ---------------------------------------------------------------------------
# Category scoring
# ---------------------------------------------------------------------------

class TestCategoryScoring:
    """Test scoring for individual categories."""

    CATEGORY_WEIGHTS_MGMT = {
        "CAT01": Decimal("0.07"), "CAT02": Decimal("0.06"),
        "CAT03": Decimal("0.05"), "CAT04": Decimal("0.05"),
        "CAT05": Decimal("0.06"), "CAT06": Decimal("0.05"),
        "CAT07": Decimal("0.08"), "CAT08": Decimal("0.07"),
        "CAT09": Decimal("0.10"), "CAT10": Decimal("0.08"),
        "CAT11": Decimal("0.06"), "CAT12": Decimal("0.04"),
        "CAT13": Decimal("0.06"), "CAT14": Decimal("0.03"),
        "CAT15": Decimal("0.06"), "CAT16": Decimal("0.05"),
        "CAT17": Decimal("0.03"),
    }

    def test_governance_scoring(self, scoring_simulator):
        score = scoring_simulator.score_category(
            category=ScoringCategory.GOVERNANCE,
            responses=[],
            disclosure_pct=Decimal("100"),
            awareness_pct=Decimal("80"),
            management_pct=Decimal("60"),
            leadership_pct=Decimal("40"),
        )
        assert isinstance(score, CDPCategoryScore)
        assert score.raw_score >= Decimal("0")
        assert score.raw_score <= Decimal("100")

    def test_all_17_categories_scoreable(self, scoring_simulator):
        for category in ScoringCategory:
            score = scoring_simulator.score_category(
                category=category,
                responses=[],
                disclosure_pct=Decimal("50"),
                awareness_pct=Decimal("50"),
                management_pct=Decimal("50"),
                leadership_pct=Decimal("50"),
            )
            assert score is not None
            assert score.raw_score >= Decimal("0")

    def test_perfect_score_category(self, scoring_simulator):
        score = scoring_simulator.score_category(
            category=ScoringCategory.SCOPE_1_2_EMISSIONS,
            responses=[],
            disclosure_pct=Decimal("100"),
            awareness_pct=Decimal("100"),
            management_pct=Decimal("100"),
            leadership_pct=Decimal("100"),
        )
        assert score.raw_score == Decimal("100")

    def test_zero_score_category(self, scoring_simulator):
        score = scoring_simulator.score_category(
            category=ScoringCategory.CARBON_PRICING,
            responses=[],
            disclosure_pct=Decimal("0"),
            awareness_pct=Decimal("0"),
            management_pct=Decimal("0"),
            leadership_pct=Decimal("0"),
        )
        assert score.raw_score == Decimal("0")


# ---------------------------------------------------------------------------
# Category weighting
# ---------------------------------------------------------------------------

class TestCategoryWeighting:
    """Test category weighting application."""

    def test_management_weights_sum_to_one(self, scoring_simulator):
        weights = scoring_simulator.get_category_weights(level="management")
        total = sum(weights.values())
        assert Decimal("0.98") <= total <= Decimal("1.02")

    def test_leadership_weights_sum_to_one(self, scoring_simulator):
        weights = scoring_simulator.get_category_weights(level="leadership")
        total = sum(weights.values())
        assert Decimal("0.98") <= total <= Decimal("1.02")

    def test_leadership_transition_plan_weight_higher(self, scoring_simulator):
        mgmt = scoring_simulator.get_category_weights(level="management")
        lead = scoring_simulator.get_category_weights(level="leadership")
        assert lead["CAT15"] >= mgmt["CAT15"]

    def test_scope_1_2_weight_is_ten_percent(self, scoring_simulator):
        weights = scoring_simulator.get_category_weights(level="management")
        assert weights["CAT09"] == Decimal("0.10")

    def test_weighted_score_calculation(self, scoring_simulator):
        raw = Decimal("80.0")
        weight = Decimal("0.10")
        weighted = scoring_simulator.calculate_weighted_score(raw, weight)
        assert weighted == Decimal("8.0")


# ---------------------------------------------------------------------------
# Overall score calculation
# ---------------------------------------------------------------------------

class TestOverallScore:
    """Test overall score computation and band assignment."""

    def test_overall_score_from_categories(self, scoring_simulator, sample_category_scores):
        result = scoring_simulator.calculate_overall_score(sample_category_scores)
        assert isinstance(result, CDPScoringResult)
        assert Decimal("0") <= result.overall_score <= Decimal("100")

    def test_overall_score_deterministic(self, scoring_simulator, sample_category_scores):
        r1 = scoring_simulator.calculate_overall_score(sample_category_scores)
        r2 = scoring_simulator.calculate_overall_score(sample_category_scores)
        assert r1.overall_score == r2.overall_score

    def test_empty_categories_zero_score(self, scoring_simulator):
        result = scoring_simulator.calculate_overall_score([])
        assert result.overall_score == Decimal("0")


# ---------------------------------------------------------------------------
# Score band determination
# ---------------------------------------------------------------------------

class TestScoreBand:
    """Test score-to-band mapping."""

    @pytest.mark.parametrize("score,expected_band", [
        (Decimal("92"), "A"),
        (Decimal("80"), "A"),
        (Decimal("75"), "A-"),
        (Decimal("70"), "A-"),
        (Decimal("65"), "B"),
        (Decimal("60"), "B"),
        (Decimal("55"), "B-"),
        (Decimal("50"), "B-"),
        (Decimal("45"), "C"),
        (Decimal("40"), "C"),
        (Decimal("35"), "C-"),
        (Decimal("30"), "C-"),
        (Decimal("25"), "D"),
        (Decimal("20"), "D"),
        (Decimal("15"), "D-"),
        (Decimal("5"), "D-"),
        (Decimal("0"), "D-"),
    ])
    def test_score_to_band(self, scoring_simulator, score, expected_band):
        band = scoring_simulator.determine_score_band(score)
        assert band == expected_band

    def test_boundary_80_is_a(self, scoring_simulator):
        assert scoring_simulator.determine_score_band(Decimal("80")) == "A"

    def test_boundary_70_is_a_minus(self, scoring_simulator):
        assert scoring_simulator.determine_score_band(Decimal("70")) == "A-"

    def test_boundary_60_is_b(self, scoring_simulator):
        assert scoring_simulator.determine_score_band(Decimal("60")) == "B"


# ---------------------------------------------------------------------------
# What-if analysis
# ---------------------------------------------------------------------------

class TestWhatIfAnalysis:
    """Test score change prediction from improving responses."""

    def test_what_if_single_improvement(self, scoring_simulator, sample_category_scores):
        baseline = scoring_simulator.calculate_overall_score(sample_category_scores)
        scenario = scoring_simulator.what_if_analysis(
            current_scores=sample_category_scores,
            improvements={"CAT06": Decimal("85.0")},  # Scenario analysis: 55 -> 85
        )
        assert scenario.overall_score >= baseline.overall_score

    def test_what_if_multiple_improvements(self, scoring_simulator, sample_category_scores):
        baseline = scoring_simulator.calculate_overall_score(sample_category_scores)
        scenario = scoring_simulator.what_if_analysis(
            current_scores=sample_category_scores,
            improvements={
                "CAT06": Decimal("80.0"),
                "CAT12": Decimal("75.0"),
                "CAT14": Decimal("70.0"),
            },
        )
        assert scenario.overall_score > baseline.overall_score

    def test_what_if_no_improvement(self, scoring_simulator, sample_category_scores):
        baseline = scoring_simulator.calculate_overall_score(sample_category_scores)
        scenario = scoring_simulator.what_if_analysis(
            current_scores=sample_category_scores,
            improvements={},
        )
        assert scenario.overall_score == baseline.overall_score

    def test_what_if_delta_calculation(self, scoring_simulator, sample_category_scores):
        baseline = scoring_simulator.calculate_overall_score(sample_category_scores)
        scenario = scoring_simulator.what_if_analysis(
            current_scores=sample_category_scores,
            improvements={"CAT09": Decimal("95.0")},
        )
        delta = scenario.overall_score - baseline.overall_score
        assert delta >= Decimal("0")


# ---------------------------------------------------------------------------
# A-level eligibility
# ---------------------------------------------------------------------------

class TestALevelEligibility:
    """Test A-level eligibility requirements checker."""

    def test_all_requirements_met(self, scoring_simulator):
        eligible = scoring_simulator.check_a_level_eligibility(
            has_transition_plan=True,
            has_complete_inventory=True,
            scope1_2_verified_pct=Decimal("100"),
            scope3_verified_pct=Decimal("75"),
            annual_reduction_pct=Decimal("5.0"),
        )
        assert eligible.is_eligible is True
        assert len(eligible.unmet_requirements) == 0

    def test_missing_transition_plan(self, scoring_simulator):
        eligible = scoring_simulator.check_a_level_eligibility(
            has_transition_plan=False,
            has_complete_inventory=True,
            scope1_2_verified_pct=Decimal("100"),
            scope3_verified_pct=Decimal("75"),
            annual_reduction_pct=Decimal("5.0"),
        )
        assert eligible.is_eligible is False
        assert "transition_plan" in str(eligible.unmet_requirements).lower()

    def test_insufficient_scope3_verification(self, scoring_simulator):
        eligible = scoring_simulator.check_a_level_eligibility(
            has_transition_plan=True,
            has_complete_inventory=True,
            scope1_2_verified_pct=Decimal("100"),
            scope3_verified_pct=Decimal("50"),  # Below 70% threshold
            annual_reduction_pct=Decimal("5.0"),
        )
        assert eligible.is_eligible is False

    def test_insufficient_reduction_rate(self, scoring_simulator):
        eligible = scoring_simulator.check_a_level_eligibility(
            has_transition_plan=True,
            has_complete_inventory=True,
            scope1_2_verified_pct=Decimal("100"),
            scope3_verified_pct=Decimal("75"),
            annual_reduction_pct=Decimal("2.0"),  # Below 4.2% threshold
        )
        assert eligible.is_eligible is False

    def test_five_requirements_checked(self, scoring_simulator):
        eligible = scoring_simulator.check_a_level_eligibility(
            has_transition_plan=False,
            has_complete_inventory=False,
            scope1_2_verified_pct=Decimal("0"),
            scope3_verified_pct=Decimal("0"),
            annual_reduction_pct=Decimal("0"),
        )
        assert len(eligible.unmet_requirements) == 5


# ---------------------------------------------------------------------------
# Score trajectory prediction
# ---------------------------------------------------------------------------

class TestScoreTrajectory:
    """Test score trajectory based on current completion."""

    def test_predict_with_partial_completion(self, scoring_simulator, sample_category_scores):
        predicted = scoring_simulator.predict_trajectory(
            current_scores=sample_category_scores,
            completion_pct=Decimal("60"),
        )
        assert predicted.predicted_band is not None

    def test_predict_full_completion(self, scoring_simulator, sample_category_scores):
        predicted = scoring_simulator.predict_trajectory(
            current_scores=sample_category_scores,
            completion_pct=Decimal("100"),
        )
        assert predicted.confidence_lower <= predicted.overall_score
        assert predicted.overall_score <= predicted.confidence_upper


# ---------------------------------------------------------------------------
# Confidence interval
# ---------------------------------------------------------------------------

class TestConfidenceInterval:
    """Test confidence interval calculation on predicted scores."""

    def test_confidence_interval_range(self, scoring_simulator, sample_category_scores):
        result = scoring_simulator.calculate_confidence_interval(
            category_scores=sample_category_scores,
            confidence_level=95,
        )
        assert result["lower"] <= result["predicted"]
        assert result["predicted"] <= result["upper"]

    def test_higher_confidence_wider_interval(self, scoring_simulator, sample_category_scores):
        ci_90 = scoring_simulator.calculate_confidence_interval(
            category_scores=sample_category_scores, confidence_level=90,
        )
        ci_99 = scoring_simulator.calculate_confidence_interval(
            category_scores=sample_category_scores, confidence_level=99,
        )
        width_90 = ci_90["upper"] - ci_90["lower"]
        width_99 = ci_99["upper"] - ci_99["lower"]
        assert width_99 >= width_90


# ---------------------------------------------------------------------------
# Score comparison
# ---------------------------------------------------------------------------

class TestScoreComparison:
    """Test current vs. previous score comparison."""

    def test_compare_scores_improvement(self, scoring_simulator):
        comparison = scoring_simulator.compare_scores(
            current_score=Decimal("72.5"),
            current_band="A-",
            previous_score=Decimal("58.0"),
            previous_band="B-",
        )
        assert comparison["score_delta"] == Decimal("14.5")
        assert comparison["band_change"] == "B- -> A-"
        assert comparison["direction"] == "improvement"

    def test_compare_scores_decline(self, scoring_simulator):
        comparison = scoring_simulator.compare_scores(
            current_score=Decimal("45.0"),
            current_band="C",
            previous_score=Decimal("62.0"),
            previous_band="B",
        )
        assert comparison["score_delta"] == Decimal("-17.0")
        assert comparison["direction"] == "decline"

    def test_compare_scores_no_change(self, scoring_simulator):
        comparison = scoring_simulator.compare_scores(
            current_score=Decimal("55.0"),
            current_band="B-",
            previous_score=Decimal("55.0"),
            previous_band="B-",
        )
        assert comparison["score_delta"] == Decimal("0")
        assert comparison["direction"] == "stable"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestScoringEdgeCases:
    """Test edge cases in scoring."""

    def test_no_responses_score_zero(self, scoring_simulator):
        result = scoring_simulator.calculate_overall_score([])
        assert result.overall_score == Decimal("0")
        assert result.score_band == "D-"

    def test_all_categories_perfect(self, scoring_simulator):
        perfect_scores = []
        for i, cat in enumerate(ScoringCategory, 1):
            perfect_scores.append(CDPCategoryScore(
                scoring_result_id=_new_id(),
                category_code=f"CAT{i:02d}",
                category_name=cat.value,
                raw_score=Decimal("100"),
                weighted_score=Decimal("100") * Decimal("0.0588"),  # approx
                weight=Decimal("0.0588"),
                disclosure_score=Decimal("100"),
                awareness_score=Decimal("100"),
                management_score=Decimal("100"),
                leadership_score=Decimal("100"),
            ))
        result = scoring_simulator.calculate_overall_score(perfect_scores)
        assert result.overall_score >= Decimal("95")
        assert result.score_band == "A"

    def test_partial_responses_some_categories(self, scoring_simulator, sample_category_scores):
        partial = sample_category_scores[:5]
        result = scoring_simulator.calculate_overall_score(partial)
        assert result.overall_score >= Decimal("0")
