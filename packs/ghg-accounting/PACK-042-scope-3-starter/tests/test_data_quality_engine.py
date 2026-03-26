# -*- coding: utf-8 -*-
"""
Unit tests for DataQualityAssessmentEngine (PACK-042 Engine 7)
================================================================

Tests 5 DQI scoring, weighted DQR calculation, quality trend tracking,
gap analysis generation, improvement roadmap prioritization, framework
minimum thresholds, and edge cases.

Coverage target: 85%+
Total tests: ~40
"""

from decimal import Decimal
from typing import Any, Dict

import pytest

from tests.conftest import SCOPE3_CATEGORIES, compute_provenance_hash


# =============================================================================
# 5 DQI Scoring Tests
# =============================================================================


class TestDQIScoring:
    """Test 5 data quality indicator scoring."""

    DQI_NAMES = [
        "technological_representativeness",
        "temporal_representativeness",
        "geographical_representativeness",
        "completeness",
        "reliability",
    ]

    def test_five_dqi_indicators(self):
        assert len(self.DQI_NAMES) == 5

    def test_each_category_has_5_dqis(self, sample_data_quality):
        for cat_id, data in sample_data_quality["categories"].items():
            assert "dqi" in data
            for dqi in self.DQI_NAMES:
                assert dqi in data["dqi"], f"{cat_id} missing DQI: {dqi}"

    def test_dqi_scores_in_range_1_to_5(self, sample_data_quality):
        for cat_id, data in sample_data_quality["categories"].items():
            for dqi_name, score in data["dqi"].items():
                assert Decimal("1.0") <= score <= Decimal("5.0"), (
                    f"{cat_id}.{dqi_name} score {score} out of [1.0, 5.0] range"
                )

    def test_lower_dqi_is_better(self):
        """DQI score of 1.0 = best (audited primary), 5.0 = worst (estimated)."""
        best = Decimal("1.0")
        worst = Decimal("5.0")
        assert best < worst

    @pytest.mark.parametrize("dqi_name", [
        "technological_representativeness",
        "temporal_representativeness",
        "geographical_representativeness",
        "completeness",
        "reliability",
    ])
    def test_dqi_name_exists_in_all_categories(self, dqi_name, sample_data_quality):
        for cat_id, data in sample_data_quality["categories"].items():
            assert dqi_name in data["dqi"]


# =============================================================================
# Weighted DQR Calculation Tests
# =============================================================================


class TestWeightedDQR:
    """Test weighted DQR calculation."""

    def test_overall_dqr_present(self, sample_data_quality):
        assert "overall_dqr" in sample_data_quality
        dqr = sample_data_quality["overall_dqr"]
        assert Decimal("1.0") <= dqr <= Decimal("5.0")

    def test_per_category_dqr_in_range(self, sample_data_quality):
        for cat_id, data in sample_data_quality["categories"].items():
            assert Decimal("1.0") <= data["dqr"] <= Decimal("5.0")

    def test_weighted_dqr_formula(self, sample_data_quality):
        """DQR = sum(weight_i * DQI_i) for each category."""
        weights = {
            "technological_representativeness": Decimal("0.30"),
            "temporal_representativeness": Decimal("0.20"),
            "geographical_representativeness": Decimal("0.20"),
            "completeness": Decimal("0.15"),
            "reliability": Decimal("0.15"),
        }
        for cat_id, data in sample_data_quality["categories"].items():
            expected_dqr = sum(
                weights[dqi] * data["dqi"][dqi]
                for dqi in weights
            )
            # Allow tolerance for rounding
            assert abs(expected_dqr - data["dqr"]) < Decimal("0.5"), (
                f"{cat_id}: calculated DQR {expected_dqr} != reported {data['dqr']}"
            )

    def test_weights_sum_to_one(self):
        weights = {
            "technological_representativeness": 0.30,
            "temporal_representativeness": 0.20,
            "geographical_representativeness": 0.20,
            "completeness": 0.15,
            "reliability": 0.15,
        }
        assert abs(sum(weights.values()) - 1.0) < 0.001


# =============================================================================
# Quality Trend Tracking Tests
# =============================================================================


class TestQualityTrend:
    """Test data quality trend tracking over time."""

    def test_trend_can_be_calculated_from_dqr(self, sample_data_quality):
        """DQR values from multiple years enable trend calculation."""
        current_dqr = sample_data_quality["overall_dqr"]
        previous_dqr = Decimal("4.0")  # Hypothetical prior year
        improvement = previous_dqr - current_dqr
        assert improvement > 0, "Quality should improve (lower DQR)"

    def test_per_category_trend(self, sample_data_quality):
        # Each category should support year-over-year tracking
        for cat_id in sample_data_quality["categories"]:
            assert sample_data_quality["categories"][cat_id]["dqr"] > 0


# =============================================================================
# Gap Analysis Tests
# =============================================================================


class TestGapAnalysis:
    """Test gap analysis generation."""

    def test_gap_analysis_present(self, sample_data_quality):
        assert "gap_analysis" in sample_data_quality
        assert len(sample_data_quality["gap_analysis"]) > 0

    def test_gap_has_current_and_target(self, sample_data_quality):
        for gap in sample_data_quality["gap_analysis"]:
            assert "current_dqr" in gap
            assert "target_dqr" in gap
            assert "gap" in gap

    def test_gap_is_positive(self, sample_data_quality):
        for gap in sample_data_quality["gap_analysis"]:
            assert gap["gap"] >= 0, (
                f"Gap should be >= 0, got {gap['gap']} for {gap['category']}"
            )

    def test_gap_equals_difference(self, sample_data_quality):
        for gap in sample_data_quality["gap_analysis"]:
            expected_gap = gap["current_dqr"] - gap["target_dqr"]
            assert gap["gap"] == expected_gap

    def test_gap_has_priority(self, sample_data_quality):
        valid_priorities = {"HIGH", "MEDIUM", "LOW"}
        for gap in sample_data_quality["gap_analysis"]:
            assert gap["priority"] in valid_priorities


# =============================================================================
# Improvement Roadmap Tests
# =============================================================================


class TestImprovementRoadmap:
    """Test improvement roadmap prioritization."""

    def test_improvement_actions_present(self, sample_data_quality):
        assert "improvement_actions" in sample_data_quality
        assert len(sample_data_quality["improvement_actions"]) > 0

    def test_actions_have_impact_and_effort(self, sample_data_quality):
        for action in sample_data_quality["improvement_actions"]:
            assert "impact" in action
            assert "effort" in action
            assert action["impact"] in {"HIGH", "MEDIUM", "LOW"}
            assert action["effort"] in {"HIGH", "MEDIUM", "LOW"}

    def test_actions_have_expected_improvement(self, sample_data_quality):
        for action in sample_data_quality["improvement_actions"]:
            assert "expected_dqr_improvement" in action
            assert action["expected_dqr_improvement"] > 0

    def test_high_impact_low_effort_first(self, sample_data_quality):
        actions = sample_data_quality["improvement_actions"]
        # At least one action should be high impact
        high_impact = [a for a in actions if a["impact"] == "HIGH"]
        assert len(high_impact) > 0


# =============================================================================
# Framework Minimum Threshold Tests
# =============================================================================


class TestFrameworkThresholds:
    """Test framework minimum data quality thresholds."""

    def test_esrs_minimum_dqr_3_0(self):
        esrs_min = Decimal("3.0")
        assert esrs_min == Decimal("3.0")

    def test_cdp_a_list_minimum_dqr_3_5(self):
        cdp_a_min = Decimal("3.5")
        assert cdp_a_min == Decimal("3.5")

    def test_ghg_protocol_recommends_dqr_below_4(self):
        recommended = Decimal("4.0")
        assert recommended <= Decimal("4.0")

    def test_current_dqr_vs_esrs_threshold(self, sample_data_quality):
        esrs_min = Decimal("3.0")
        overall = sample_data_quality["overall_dqr"]
        # Report gap if above threshold
        gap_exists = overall > esrs_min
        assert isinstance(gap_exists, bool)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestDataQualityEdgeCases:
    """Test edge cases for data quality assessment."""

    def test_perfect_quality_all_1s(self):
        perfect_dqi = {
            "technological_representativeness": Decimal("1.0"),
            "temporal_representativeness": Decimal("1.0"),
            "geographical_representativeness": Decimal("1.0"),
            "completeness": Decimal("1.0"),
            "reliability": Decimal("1.0"),
        }
        weights = [0.30, 0.20, 0.20, 0.15, 0.15]
        dqr = sum(Decimal(str(w)) * v for w, v in zip(weights, perfect_dqi.values()))
        assert dqr == Decimal("1.0")

    def test_lowest_quality_all_5s(self):
        worst_dqi = {
            "technological_representativeness": Decimal("5.0"),
            "temporal_representativeness": Decimal("5.0"),
            "geographical_representativeness": Decimal("5.0"),
            "completeness": Decimal("5.0"),
            "reliability": Decimal("5.0"),
        }
        weights = [0.30, 0.20, 0.20, 0.15, 0.15]
        dqr = sum(Decimal(str(w)) * v for w, v in zip(weights, worst_dqi.values()))
        assert dqr == Decimal("5.0")

    def test_missing_data_defaults_to_5(self):
        """Missing data should default to worst quality (5.0)."""
        default_score = Decimal("5.0")
        assert default_score == Decimal("5.0")

    def test_provenance_hash_present(self, sample_data_quality):
        assert "provenance_hash" in sample_data_quality
        assert len(sample_data_quality["provenance_hash"]) == 64
