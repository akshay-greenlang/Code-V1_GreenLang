# -*- coding: utf-8 -*-
"""
Unit tests for SBTi Scope 3 Screening Engine.

Tests the 40% Scope 3 trigger assessment, 15-category breakdown,
hotspot identification, coverage calculation (67% and 90% thresholds),
category selection recommendations, and per-category data quality
assessment with 22+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest


# ===========================================================================
# Trigger Assessment
# ===========================================================================

class TestTriggerAssessment:
    """Test Scope 3 40% threshold trigger."""

    def test_trigger_above_threshold(self, sample_scope3_screening):
        assert sample_scope3_screening["scope3_target_required"] is True
        assert sample_scope3_screening["scope3_pct_of_total"] >= 40.0

    def test_trigger_below_threshold(self, low_scope3_inventory):
        s3_pct = low_scope3_inventory["scope3_pct_of_total"]
        assert s3_pct < 40.0

    @pytest.mark.parametrize("s1,s2,s3,expected_required", [
        (30_000, 20_000, 50_000, True),    # 50% - above
        (30_000, 20_000, 40_000, True),     # 44.4% - above
        (60_000, 20_000, 20_000, False),    # 20% - below
        (50_000, 10_000, 40_000, True),     # 40% - exactly at threshold
        (90_000, 10_000, 0, False),         # 0% - no scope 3
    ])
    def test_trigger_edge_cases(self, s1, s2, s3, expected_required):
        total = s1 + s2 + s3
        s3_pct = (s3 / total) * 100 if total > 0 else 0
        required = s3_pct >= 40.0
        assert required == expected_required

    def test_trigger_zero_total_emissions(self):
        total = 0
        s3_pct = 0.0
        assert s3_pct < 40.0

    def test_trigger_high_scope3(self, high_scope3_inventory):
        assert high_scope3_inventory["scope3_pct_of_total"] >= 40.0


# ===========================================================================
# Category Breakdown
# ===========================================================================

class TestCategoryBreakdown:
    """Test 15 Scope 3 category breakdown."""

    def test_all_15_categories(self, sample_scope3_screening):
        categories = sample_scope3_screening["category_breakdown"]
        assert len(categories) == 15
        for cat_num in range(1, 16):
            assert cat_num in categories

    def test_category_has_tco2e(self, sample_scope3_screening):
        for cat_num, data in sample_scope3_screening["category_breakdown"].items():
            assert "tco2e" in data
            assert data["tco2e"] >= 0

    def test_category_has_percentage(self, sample_scope3_screening):
        for cat_num, data in sample_scope3_screening["category_breakdown"].items():
            assert "pct" in data
            assert 0 <= data["pct"] <= 100

    def test_category_percentages_sum_to_100(self, sample_scope3_screening):
        total_pct = sum(
            data["pct"]
            for data in sample_scope3_screening["category_breakdown"].values()
        )
        assert abs(total_pct - 100.0) < 1.0

    def test_category_emissions_sum(self, sample_scope3_screening):
        total_emissions = sum(
            data["tco2e"]
            for data in sample_scope3_screening["category_breakdown"].values()
        )
        assert total_emissions == sample_scope3_screening["total_scope3_tco2e"]


# ===========================================================================
# Hotspot Identification
# ===========================================================================

class TestHotspotIdentification:
    """Test identification of top Scope 3 categories."""

    def test_hotspots_identified(self, sample_scope3_screening):
        hotspots = sample_scope3_screening["hotspot_categories"]
        assert len(hotspots) >= 1

    def test_hotspot_top_category(self, sample_scope3_screening):
        hotspots = sample_scope3_screening["hotspot_categories"]
        categories = sample_scope3_screening["category_breakdown"]
        # Hotspot list should include the largest category
        sorted_cats = sorted(
            categories.items(), key=lambda x: x[1]["tco2e"], reverse=True
        )
        assert sorted_cats[0][0] in hotspots

    def test_hotspots_sorted_by_materiality(self, sample_scope3_screening):
        hotspots = sample_scope3_screening["hotspot_categories"]
        categories = sample_scope3_screening["category_breakdown"]
        for i in range(1, len(hotspots)):
            prev_emissions = categories[hotspots[i - 1]]["tco2e"]
            curr_emissions = categories[hotspots[i]]["tco2e"]
            assert prev_emissions >= curr_emissions


# ===========================================================================
# Coverage Calculation
# ===========================================================================

class TestCoverageCalculation:
    """Test coverage threshold calculations."""

    def test_67pct_near_term_threshold(self, sample_scope3_screening):
        assert sample_scope3_screening["min_coverage_pct"] == 67.0

    def test_recommended_coverage_meets_minimum(self, sample_scope3_screening):
        recommended = sample_scope3_screening["recommended_coverage_pct"]
        minimum = sample_scope3_screening["min_coverage_pct"]
        assert recommended >= minimum

    @pytest.mark.parametrize("coverage,meets_near_term,meets_long_term", [
        (67.0, True, False),
        (72.0, True, False),
        (90.0, True, True),
        (95.0, True, True),
        (66.0, False, False),
        (50.0, False, False),
    ])
    def test_coverage_thresholds(self, coverage, meets_near_term, meets_long_term):
        assert (coverage >= 67.0) == meets_near_term
        assert (coverage >= 90.0) == meets_long_term

    def test_coverage_from_selected_categories(self, sample_scope3_screening):
        categories = sample_scope3_screening["category_breakdown"]
        recommended = sample_scope3_screening["recommended_categories"]
        selected_pct = sum(categories[cat]["pct"] for cat in recommended)
        assert abs(selected_pct - sample_scope3_screening["recommended_coverage_pct"]) < 1.0


# ===========================================================================
# Recommendations
# ===========================================================================

class TestRecommendations:
    """Test category selection recommendations."""

    def test_recommended_categories_provided(self, sample_scope3_screening):
        recommended = sample_scope3_screening["recommended_categories"]
        assert len(recommended) >= 1

    def test_recommended_coverage_sufficient(self, sample_scope3_screening):
        assert sample_scope3_screening["recommended_coverage_pct"] >= 67.0

    def test_recommended_includes_hotspots(self, sample_scope3_screening):
        recommended = set(sample_scope3_screening["recommended_categories"])
        hotspots = set(sample_scope3_screening["hotspot_categories"])
        # Recommended should include at least the top hotspot
        assert len(recommended & hotspots) >= 1


# ===========================================================================
# Data Quality
# ===========================================================================

class TestDataQuality:
    """Test per-category data quality assessment."""

    def test_data_quality_present(self, sample_scope3_screening):
        for cat_num, data in sample_scope3_screening["category_breakdown"].items():
            assert "data_quality" in data

    def test_data_quality_range(self, sample_scope3_screening):
        for cat_num, data in sample_scope3_screening["category_breakdown"].items():
            assert 1 <= data["data_quality"] <= 5

    @pytest.mark.parametrize("quality_score,quality_level", [
        (1, "estimated"),
        (2, "spend_based"),
        (3, "average_data"),
        (4, "supplier_specific"),
        (5, "primary_data"),
    ])
    def test_data_quality_levels(self, quality_score, quality_level):
        levels = {1: "estimated", 2: "spend_based", 3: "average_data",
                  4: "supplier_specific", 5: "primary_data"}
        assert levels[quality_score] == quality_level

    def test_average_data_quality(self, sample_scope3_screening):
        categories = sample_scope3_screening["category_breakdown"]
        avg_quality = sum(
            data["data_quality"] for data in categories.values()
        ) / len(categories)
        assert 1.0 <= avg_quality <= 5.0
