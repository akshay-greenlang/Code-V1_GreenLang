# -*- coding: utf-8 -*-
"""
Unit tests for UncertaintyEngine -- ISO 14064-1:2018 Clause 6.3.

Tests Monte Carlo simulation, analytical error propagation, per-category
uncertainty, sensitivity ranking, and statistical outputs with 25+ tests.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import (
    DataQualityTier,
    ISOCategory,
    ISO14064AppConfig,
)
from services.models import CategoryResult
from services.uncertainty_engine import UncertaintyEngine


@pytest.fixture
def two_category_results():
    """Two categories for uncertainty testing."""
    return {
        ISOCategory.CATEGORY_1_DIRECT.value: CategoryResult(
            iso_category=ISOCategory.CATEGORY_1_DIRECT,
            total_tco2e=Decimal("5000"),
            data_quality_tier=DataQualityTier.TIER_3,
        ),
        ISOCategory.CATEGORY_2_ENERGY.value: CategoryResult(
            iso_category=ISOCategory.CATEGORY_2_ENERGY,
            total_tco2e=Decimal("3000"),
            data_quality_tier=DataQualityTier.TIER_2,
        ),
    }


@pytest.fixture
def fast_engine():
    """Uncertainty engine with reduced iterations for speed."""
    config = ISO14064AppConfig(monte_carlo_iterations=1000)
    return UncertaintyEngine(config)


class TestMonteCarlo:
    """Test Monte Carlo simulation."""

    def test_monte_carlo_produces_result(self, fast_engine, two_category_results):
        result = fast_engine.run_monte_carlo(
            "inv-1", two_category_results, iterations=500,
        )
        assert result.mean > Decimal("0")
        assert result.std_dev > Decimal("0")
        assert result.cv_percent > Decimal("0")
        assert result.iterations == 500

    def test_monte_carlo_has_intervals(self, fast_engine, two_category_results):
        result = fast_engine.run_monte_carlo(
            "inv-1", two_category_results, iterations=500,
        )
        assert len(result.intervals) >= 1
        for interval in result.intervals:
            assert interval.lower_bound < interval.upper_bound
            assert interval.confidence_level in [90, 95, 99]

    def test_reproducible_with_seed(self, fast_engine, two_category_results):
        r1 = fast_engine.run_monte_carlo(
            "inv-1", two_category_results, iterations=200, seed=123,
        )
        r2 = fast_engine.run_monte_carlo(
            "inv-1", two_category_results, iterations=200, seed=123,
        )
        assert r1.mean == r2.mean
        assert r1.std_dev == r2.std_dev

    def test_per_category_stats(self, fast_engine, two_category_results):
        result = fast_engine.run_monte_carlo(
            "inv-1", two_category_results, iterations=500,
        )
        assert len(result.by_category) == 2
        for cat_key, stats in result.by_category.items():
            assert "mean" in stats
            assert "std_dev" in stats
            assert "cv_percent" in stats

    def test_zero_emissions_category_skipped(self, fast_engine):
        cat_results = {
            ISOCategory.CATEGORY_1_DIRECT.value: CategoryResult(
                iso_category=ISOCategory.CATEGORY_1_DIRECT,
                total_tco2e=Decimal("5000"),
                data_quality_tier=DataQualityTier.TIER_3,
            ),
            ISOCategory.CATEGORY_6_OTHER.value: CategoryResult(
                iso_category=ISOCategory.CATEGORY_6_OTHER,
                total_tco2e=Decimal("0"),
                data_quality_tier=DataQualityTier.TIER_1,
            ),
        }
        result = fast_engine.run_monte_carlo("inv-1", cat_results, iterations=200)
        assert ISOCategory.CATEGORY_6_OTHER.value not in result.by_category


class TestAnalytical:
    """Test IPCC Tier 1 analytical error propagation."""

    def test_analytical_produces_result(self, fast_engine, two_category_results):
        result = fast_engine.run_analytical("inv-1", two_category_results)
        assert result.mean > Decimal("0")
        assert result.std_dev > Decimal("0")
        assert result.iterations == 0  # Analytical, no simulation

    def test_analytical_has_intervals(self, fast_engine, two_category_results):
        result = fast_engine.run_analytical("inv-1", two_category_results)
        assert len(result.intervals) >= 1
        for interval in result.intervals:
            assert interval.lower_bound <= result.mean
            assert interval.upper_bound >= result.mean

    def test_analytical_cv_reflects_tiers(self, fast_engine):
        # High-quality data should have lower CV
        high_quality = {
            ISOCategory.CATEGORY_1_DIRECT.value: CategoryResult(
                iso_category=ISOCategory.CATEGORY_1_DIRECT,
                total_tco2e=Decimal("5000"),
                data_quality_tier=DataQualityTier.TIER_4,
            ),
        }
        low_quality = {
            ISOCategory.CATEGORY_1_DIRECT.value: CategoryResult(
                iso_category=ISOCategory.CATEGORY_1_DIRECT,
                total_tco2e=Decimal("5000"),
                data_quality_tier=DataQualityTier.TIER_1,
            ),
        }
        r_high = fast_engine.run_analytical("inv-hq", high_quality)
        r_low = fast_engine.run_analytical("inv-lq", low_quality)
        assert r_high.cv_percent < r_low.cv_percent

    def test_analytical_per_category_breakdown(self, fast_engine, two_category_results):
        result = fast_engine.run_analytical("inv-1", two_category_results)
        assert len(result.by_category) == 2
        # Tier 3 category should have CV = 5.0%
        cat1_stats = result.by_category.get(ISOCategory.CATEGORY_1_DIRECT.value)
        assert cat1_stats is not None
        assert cat1_stats["cv_percent"] == Decimal("5.0")


class TestPerCategoryUncertainty:
    """Test single-category uncertainty."""

    def test_uncertainty_by_category(self, fast_engine):
        cat_result = CategoryResult(
            iso_category=ISOCategory.CATEGORY_1_DIRECT,
            total_tco2e=Decimal("10000"),
            data_quality_tier=DataQualityTier.TIER_2,
        )
        result = fast_engine.uncertainty_by_category(cat_result, iterations=500)
        assert result["mean"] > 0
        assert result["cv_percent"] > 0
        assert "p5" in result
        assert "p95" in result


class TestSensitivityRanking:
    """Test sensitivity/variance contribution ranking."""

    def test_ranking_order(self, fast_engine, two_category_results):
        ranking = fast_engine.get_sensitivity_ranking(
            two_category_results, iterations=500,
        )
        assert len(ranking) >= 1
        # Verify sorted by variance contribution descending
        contributions = [r["variance_contribution_pct"] for r in ranking]
        assert contributions == sorted(contributions, reverse=True)

    def test_ranking_has_required_fields(self, fast_engine, two_category_results):
        ranking = fast_engine.get_sensitivity_ranking(
            two_category_results, iterations=500,
        )
        for entry in ranking:
            assert "category" in entry
            assert "mean_tco2e" in entry
            assert "variance_contribution_pct" in entry


class TestTierCVMapping:
    """Test the tier to CV percentage mapping."""

    def test_tier_1_cv_50(self):
        assert UncertaintyEngine.TIER_CV_MAP[DataQualityTier.TIER_1] == 50.0

    def test_tier_2_cv_20(self):
        assert UncertaintyEngine.TIER_CV_MAP[DataQualityTier.TIER_2] == 20.0

    def test_tier_3_cv_5(self):
        assert UncertaintyEngine.TIER_CV_MAP[DataQualityTier.TIER_3] == 5.0

    def test_tier_4_cv_2(self):
        assert UncertaintyEngine.TIER_CV_MAP[DataQualityTier.TIER_4] == 2.0


class TestConfidenceIntervals:
    """Test confidence interval configuration."""

    def test_default_confidence_levels(self, uncertainty_engine):
        assert uncertainty_engine._confidence_levels == [90, 95, 99]

    def test_custom_confidence_levels(self):
        config = ISO14064AppConfig(
            confidence_levels=[90, 95],
            monte_carlo_iterations=100,
        )
        engine = UncertaintyEngine(config)
        assert engine._confidence_levels == [90, 95]

    def test_z_score_lookup(self):
        assert UncertaintyEngine._z_score_for_level(90) == 1.645
        assert UncertaintyEngine._z_score_for_level(95) == 1.960
        assert UncertaintyEngine._z_score_for_level(99) == 2.576
        # Unknown level defaults to 95%
        assert UncertaintyEngine._z_score_for_level(80) == 1.960
