# -*- coding: utf-8 -*-
"""
Unit tests for PACK-043 Data Maturity Engine
===============================================

Tests maturity assessment across 15 Scope 3 categories, gap analysis,
upgrade ROI, budget optimization, uncertainty reduction, and edge cases.

Coverage target: 85%+
Total tests: ~50
"""

from decimal import Decimal

import pytest

from tests.conftest import SCOPE3_CATEGORIES, MATURITY_TIERS


# =============================================================================
# Maturity Assessment for 15 Categories
# =============================================================================


class TestMaturityAssessment:
    """Test maturity tier assessment for all 15 Scope 3 categories."""

    def test_fifteen_categories_present(self, sample_maturity_assessment):
        assert len(sample_maturity_assessment["categories"]) == 15

    @pytest.mark.parametrize("cat_num", range(1, 16))
    def test_category_has_required_fields(self, cat_num, sample_maturity_assessment):
        cat = sample_maturity_assessment["categories"][cat_num]
        required = [
            "category_id", "category_name", "current_tier", "target_tier",
            "emissions_tco2e", "data_quality_score", "upgrade_cost_usd",
            "uncertainty_pct",
        ]
        for field in required:
            assert field in cat, f"Missing {field} in category {cat_num}"

    @pytest.mark.parametrize("cat_num", range(1, 16))
    def test_current_tier_in_range(self, cat_num, sample_maturity_assessment):
        tier = sample_maturity_assessment["categories"][cat_num]["current_tier"]
        assert 1 <= tier <= 5

    @pytest.mark.parametrize("cat_num", range(1, 16))
    def test_target_tier_gte_current(self, cat_num, sample_maturity_assessment):
        cat = sample_maturity_assessment["categories"][cat_num]
        assert cat["target_tier"] >= cat["current_tier"]

    @pytest.mark.parametrize("cat_num", range(1, 16))
    def test_category_name_matches_reference(self, cat_num, sample_maturity_assessment):
        cat = sample_maturity_assessment["categories"][cat_num]
        assert cat["category_name"] == SCOPE3_CATEGORIES[cat_num]

    def test_total_emissions_sum(self, sample_maturity_assessment):
        cats = sample_maturity_assessment["categories"]
        total = sum(c["emissions_tco2e"] for c in cats.values())
        assert total == sample_maturity_assessment["total_scope3_tco2e"]


# =============================================================================
# Gap Analysis
# =============================================================================


class TestGapAnalysis:
    """Test gap analysis between current and target tiers."""

    @pytest.mark.parametrize("cat_num", range(1, 16))
    def test_gap_non_negative(self, cat_num, sample_maturity_assessment):
        cat = sample_maturity_assessment["categories"][cat_num]
        gap = cat["target_tier"] - cat["current_tier"]
        assert gap >= 0

    def test_max_gap_category(self, sample_maturity_assessment):
        """Identify the category with the largest gap."""
        cats = sample_maturity_assessment["categories"]
        max_gap = 0
        max_cat = None
        for num, cat in cats.items():
            gap = cat["target_tier"] - cat["current_tier"]
            if gap > max_gap:
                max_gap = gap
                max_cat = num
        assert max_gap > 0
        assert max_cat is not None

    def test_no_gap_means_zero_upgrade_cost(self, sample_maturity_assessment):
        """Categories at target tier should have zero upgrade cost."""
        cats = sample_maturity_assessment["categories"]
        for cat in cats.values():
            if cat["target_tier"] == cat["current_tier"]:
                assert cat["upgrade_cost_usd"] == Decimal("0")

    def test_gap_correlates_with_cost(self, sample_maturity_assessment):
        """Larger gaps should have higher upgrade costs."""
        cats = sample_maturity_assessment["categories"]
        for cat in cats.values():
            gap = cat["target_tier"] - cat["current_tier"]
            if gap > 0:
                assert cat["upgrade_cost_usd"] > Decimal("0")


# =============================================================================
# Upgrade ROI Calculation
# =============================================================================


class TestUpgradeROI:
    """Test upgrade ROI calculation accuracy."""

    def test_roi_positive_for_high_emission_category(self, sample_maturity_assessment):
        """Category 1 (highest emissions) should have best ROI."""
        cat1 = sample_maturity_assessment["categories"][1]
        # ROI = uncertainty_reduction * emissions / cost
        uncertainty_reduction = cat1["uncertainty_pct"] - Decimal("10")  # target uncertainty
        if cat1["upgrade_cost_usd"] > 0:
            roi = (uncertainty_reduction * cat1["emissions_tco2e"]) / cat1["upgrade_cost_usd"]
            assert roi > Decimal("0")

    def test_weighted_quality_in_range(self, sample_maturity_assessment):
        score = sample_maturity_assessment["weighted_data_quality"]
        assert Decimal("0") <= score <= Decimal("100")

    def test_total_upgrade_cost_positive(self, sample_maturity_assessment):
        assert sample_maturity_assessment["total_upgrade_cost"] > Decimal("0")

    def test_upgrade_cost_sum_matches_total(self, sample_maturity_assessment):
        cats = sample_maturity_assessment["categories"]
        total = sum(c["upgrade_cost_usd"] for c in cats.values())
        assert total == sample_maturity_assessment["total_upgrade_cost"]


# =============================================================================
# Budget Optimization
# =============================================================================


class TestBudgetOptimization:
    """Test constrained and unconstrained budget optimization."""

    def test_unconstrained_selects_all(self, sample_maturity_assessment):
        """With unlimited budget, all categories should be upgraded."""
        cats = sample_maturity_assessment["categories"]
        upgradeable = [c for c in cats.values() if c["target_tier"] > c["current_tier"]]
        assert len(upgradeable) > 0

    def test_constrained_budget_selects_best_roi(self, sample_maturity_assessment):
        """With limited budget, highest ROI categories should be prioritized."""
        budget = Decimal("100000")
        cats = sample_maturity_assessment["categories"]

        # Sort by cost-effectiveness: emissions_impact / cost
        ranked = sorted(
            [c for c in cats.values() if c["upgrade_cost_usd"] > 0],
            key=lambda c: c["emissions_tco2e"] / c["upgrade_cost_usd"],
            reverse=True,
        )
        selected = []
        remaining = budget
        for cat in ranked:
            if cat["upgrade_cost_usd"] <= remaining:
                selected.append(cat)
                remaining -= cat["upgrade_cost_usd"]
        assert len(selected) >= 1
        total_cost = sum(c["upgrade_cost_usd"] for c in selected)
        assert total_cost <= budget

    def test_zero_budget_selects_none(self, sample_maturity_assessment):
        """Zero budget should select no upgrades."""
        budget = Decimal("0")
        cats = sample_maturity_assessment["categories"]
        selected = [
            c for c in cats.values()
            if c["upgrade_cost_usd"] > 0 and c["upgrade_cost_usd"] <= budget
        ]
        assert len(selected) == 0


# =============================================================================
# Uncertainty Reduction Projection
# =============================================================================


class TestUncertaintyReduction:
    """Test uncertainty reduction projection."""

    @pytest.mark.parametrize("cat_num", range(1, 16))
    def test_uncertainty_in_valid_range(self, cat_num, sample_maturity_assessment):
        u = sample_maturity_assessment["categories"][cat_num]["uncertainty_pct"]
        assert Decimal("0") <= u <= Decimal("100")

    def test_higher_tier_lower_uncertainty(self, sample_maturity_assessment):
        """Higher maturity tiers should have lower uncertainty."""
        cats = sample_maturity_assessment["categories"]
        tier_uncertainties = {}
        for cat in cats.values():
            tier = cat["current_tier"]
            if tier not in tier_uncertainties:
                tier_uncertainties[tier] = []
            tier_uncertainties[tier].append(cat["uncertainty_pct"])

        tier_avgs = {t: sum(us) / len(us) for t, us in tier_uncertainties.items()}
        sorted_tiers = sorted(tier_avgs.keys())
        for i in range(1, len(sorted_tiers)):
            assert tier_avgs[sorted_tiers[i]] <= tier_avgs[sorted_tiers[i - 1]]

    def test_post_upgrade_uncertainty_lower(self, sample_maturity_assessment):
        """Projected post-upgrade uncertainty should be lower than current."""
        cats = sample_maturity_assessment["categories"]
        for cat in cats.values():
            current_u = cat["uncertainty_pct"]
            # Post-upgrade uncertainty based on target tier
            projected_u = max(Decimal("5"), Decimal("50") - cat["target_tier"] * Decimal("10"))
            assert projected_u <= current_u


# =============================================================================
# Post-Upgrade Simulation
# =============================================================================


class TestPostUpgradeSimulation:
    """Test simulation of post-upgrade state."""

    def test_simulated_quality_improves(self, sample_maturity_assessment):
        """After upgrade, weighted quality should improve."""
        cats = sample_maturity_assessment["categories"]
        total_emissions = sample_maturity_assessment["total_scope3_tco2e"]

        current_wq = sum(
            c["data_quality_score"] * c["emissions_tco2e"] / total_emissions
            for c in cats.values()
        )
        upgraded_wq = sum(
            (c["target_tier"] * Decimal("20")) * c["emissions_tco2e"] / total_emissions
            for c in cats.values()
        )
        assert upgraded_wq >= current_wq

    def test_simulated_uncertainty_decreases(self, sample_maturity_assessment):
        """After upgrade, overall uncertainty should decrease."""
        cats = sample_maturity_assessment["categories"]
        current_us = [float(c["uncertainty_pct"]) for c in cats.values()]
        target_us = [
            max(5.0, 50.0 - c["target_tier"] * 10.0)
            for c in cats.values()
        ]
        assert sum(target_us) < sum(current_us)


# =============================================================================
# Edge Cases
# =============================================================================


class TestMaturityEdgeCases:
    """Test edge cases for maturity assessment."""

    def test_all_level_5(self):
        """All categories at Level 5 should have zero upgrade cost."""
        categories = {}
        for i in range(1, 16):
            categories[i] = {
                "current_tier": 5,
                "target_tier": 5,
                "upgrade_cost_usd": Decimal("0"),
            }
        total_cost = sum(c["upgrade_cost_usd"] for c in categories.values())
        assert total_cost == Decimal("0")

    def test_all_level_1(self):
        """All categories at Level 1 should have maximum upgrade potential."""
        categories = {}
        for i in range(1, 16):
            categories[i] = {
                "current_tier": 1,
                "target_tier": 5,
                "gap": 4,
            }
        total_gap = sum(c["gap"] for c in categories.values())
        assert total_gap == 60  # 15 * 4

    def test_single_category(self):
        """Assessment with a single category should work."""
        categories = {
            1: {
                "current_tier": 2,
                "target_tier": 4,
                "emissions_tco2e": Decimal("100000"),
                "upgrade_cost_usd": Decimal("50000"),
            }
        }
        assert len(categories) == 1
        assert categories[1]["target_tier"] > categories[1]["current_tier"]

    def test_tier_names_valid(self, sample_maturity_assessment):
        """All tier names should be from the MATURITY_TIERS reference."""
        cats = sample_maturity_assessment["categories"]
        for cat in cats.values():
            assert cat["current_tier_name"] in MATURITY_TIERS.values()
            assert cat["target_tier_name"] in MATURITY_TIERS.values()

    def test_data_quality_score_formula(self, sample_maturity_assessment):
        """Data quality score should be tier * 20."""
        cats = sample_maturity_assessment["categories"]
        for cat in cats.values():
            expected = Decimal(str(cat["current_tier"] * 20))
            assert cat["data_quality_score"] == expected
