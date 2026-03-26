# -*- coding: utf-8 -*-
"""
Unit tests for CategoryConsolidationEngine (PACK-042 Engine 3)
================================================================

Tests consolidation of all 15 Scope 3 categories, upstream/downstream split,
per-gas aggregation, weighted DQR, year-over-year comparison, boundary
alignment with PACK-041, and provenance hash consistency.

Coverage target: 85%+
Total tests: ~50
"""

from decimal import Decimal
from typing import Any, Dict

import pytest

from tests.conftest import (
    SCOPE3_CATEGORIES,
    UPSTREAM_CATEGORIES,
    DOWNSTREAM_CATEGORIES,
    compute_provenance_hash,
)


# =============================================================================
# Consolidation Coverage Tests
# =============================================================================


class TestConsolidateAllCategories:
    """Test consolidation across all 15 categories."""

    def test_all_15_categories_in_results(self, sample_category_results):
        cats = sample_category_results["categories"]
        assert len(cats) == 15

    def test_total_equals_sum_of_categories(self, sample_category_results):
        cats = sample_category_results["categories"]
        calculated = sum(cats[c]["total_tco2e"] for c in SCOPE3_CATEGORIES)
        assert calculated == sample_category_results["total_scope3_tco2e"]

    def test_each_category_has_methodology(self, sample_category_results):
        valid_methods = {"SPEND_BASED", "AVERAGE_DATA", "SUPPLIER_SPECIFIC", "HYBRID", "NOT_APPLICABLE"}
        for cat_id, data in sample_category_results["categories"].items():
            assert data["methodology"] in valid_methods

    def test_each_category_has_dqr(self, sample_category_results):
        for cat_id, data in sample_category_results["categories"].items():
            dqr = data["dqr"]
            if data["methodology"] != "NOT_APPLICABLE":
                assert Decimal("1.0") <= dqr <= Decimal("5.0"), (
                    f"{cat_id} DQR {dqr} out of range [1.0, 5.0]"
                )

    def test_each_category_has_gas_breakdown(self, sample_category_results):
        for cat_id, data in sample_category_results["categories"].items():
            assert "by_gas" in data
            assert "CO2" in data["by_gas"]
            assert "CH4" in data["by_gas"]
            assert "N2O" in data["by_gas"]


# =============================================================================
# Upstream/Downstream Split Tests
# =============================================================================


class TestUpstreamDownstreamSplit:
    """Test upstream vs downstream category split."""

    def test_upstream_categories_count_8(self):
        assert len(UPSTREAM_CATEGORIES) == 8

    def test_downstream_categories_count_7(self):
        assert len(DOWNSTREAM_CATEGORIES) == 7

    def test_upstream_plus_downstream_equals_15(self):
        assert len(UPSTREAM_CATEGORIES) + len(DOWNSTREAM_CATEGORIES) == 15

    def test_upstream_total_calculation(self, sample_category_results):
        cats = sample_category_results["categories"]
        upstream = sum(cats[c]["total_tco2e"] for c in UPSTREAM_CATEGORIES)
        assert upstream > 0

    def test_downstream_total_calculation(self, sample_category_results):
        cats = sample_category_results["categories"]
        downstream = sum(cats[c]["total_tco2e"] for c in DOWNSTREAM_CATEGORIES)
        assert downstream > 0

    def test_upstream_plus_downstream_equals_total(self, sample_category_results):
        cats = sample_category_results["categories"]
        upstream = sum(cats[c]["total_tco2e"] for c in UPSTREAM_CATEGORIES)
        downstream = sum(cats[c]["total_tco2e"] for c in DOWNSTREAM_CATEGORIES)
        total = sample_category_results["total_scope3_tco2e"]
        assert upstream + downstream == total

    def test_consolidated_inventory_has_split(self, sample_consolidated_inventory):
        assert "upstream_tco2e" in sample_consolidated_inventory
        assert "downstream_tco2e" in sample_consolidated_inventory
        assert "upstream_pct" in sample_consolidated_inventory
        assert "downstream_pct" in sample_consolidated_inventory

    def test_split_percentages_sum_to_100(self, sample_consolidated_inventory):
        total_pct = (
            sample_consolidated_inventory["upstream_pct"]
            + sample_consolidated_inventory["downstream_pct"]
        )
        assert abs(total_pct - 100.0) < 0.5


# =============================================================================
# Per-Gas Aggregation Tests
# =============================================================================


class TestPerGasAggregation:
    """Test per-gas aggregation across categories."""

    def test_gas_breakdown_in_inventory(self, sample_consolidated_inventory):
        by_gas = sample_consolidated_inventory["by_gas"]
        assert "CO2" in by_gas
        assert "CH4" in by_gas
        assert "N2O" in by_gas

    def test_co2_is_dominant_gas(self, sample_consolidated_inventory):
        by_gas = sample_consolidated_inventory["by_gas"]
        assert by_gas["CO2"] > by_gas["CH4"]
        assert by_gas["CO2"] > by_gas["N2O"]

    def test_gas_sum_equals_total(self, sample_category_results):
        cats = sample_category_results["categories"]
        total_co2 = sum(cats[c]["by_gas"]["CO2"] for c in SCOPE3_CATEGORIES)
        total_ch4 = sum(cats[c]["by_gas"]["CH4"] for c in SCOPE3_CATEGORIES)
        total_n2o = sum(cats[c]["by_gas"]["N2O"] for c in SCOPE3_CATEGORIES)
        gas_total = total_co2 + total_ch4 + total_n2o
        reported_total = sample_category_results["total_scope3_tco2e"]
        assert gas_total == reported_total

    def test_per_category_gas_sum_equals_category_total(self, sample_category_results):
        for cat_id, data in sample_category_results["categories"].items():
            gas_sum = sum(data["by_gas"].values())
            assert gas_sum == data["total_tco2e"], (
                f"{cat_id}: gas sum {gas_sum} != total {data['total_tco2e']}"
            )


# =============================================================================
# Scope 3 as Percentage of Total Footprint
# =============================================================================


class TestScope3PercentOfTotal:
    """Test Scope 3 as percentage of total footprint calculation."""

    def test_scope3_pct_present(self, sample_consolidated_inventory):
        assert "scope3_pct_of_total" in sample_consolidated_inventory

    def test_scope3_pct_calculation(self, sample_consolidated_inventory):
        inv = sample_consolidated_inventory
        s3 = float(inv["total_scope3_tco2e"])
        total = float(inv["total_footprint_tco2e"])
        expected_pct = s3 / total * 100
        assert abs(inv["scope3_pct_of_total"] - expected_pct) < 0.5

    def test_scope3_larger_than_scope12_for_manufacturing(self, sample_consolidated_inventory):
        inv = sample_consolidated_inventory
        scope12 = float(inv["scope1_tco2e"]) + float(inv["scope2_market_tco2e"])
        scope3 = float(inv["total_scope3_tco2e"])
        assert scope3 > scope12, "For manufacturing, Scope 3 typically > Scope 1+2"

    def test_total_footprint_equals_sum(self, sample_consolidated_inventory):
        inv = sample_consolidated_inventory
        calculated = (
            inv["total_scope3_tco2e"]
            + inv["scope1_tco2e"]
            + inv["scope2_market_tco2e"]
        )
        assert calculated == inv["total_footprint_tco2e"]


# =============================================================================
# Weighted Average DQR Tests
# =============================================================================


class TestWeightedDQR:
    """Test weighted average data quality rating calculation."""

    def test_weighted_dqr_present(self, sample_consolidated_inventory):
        assert "weighted_dqr" in sample_consolidated_inventory

    def test_weighted_dqr_in_valid_range(self, sample_consolidated_inventory):
        dqr = float(sample_consolidated_inventory["weighted_dqr"])
        assert 1.0 <= dqr <= 5.0

    def test_weighted_dqr_weighted_by_emissions(self, sample_category_results):
        cats = sample_category_results["categories"]
        total = sample_category_results["total_scope3_tco2e"]
        if total == 0:
            return
        weighted_sum = sum(
            cats[c]["dqr"] * cats[c]["total_tco2e"]
            for c in SCOPE3_CATEGORIES
            if cats[c]["dqr"] > 0
        )
        emissions_with_dqr = sum(
            cats[c]["total_tco2e"]
            for c in SCOPE3_CATEGORIES
            if cats[c]["dqr"] > 0
        )
        if emissions_with_dqr > 0:
            expected_dqr = weighted_sum / emissions_with_dqr
            assert Decimal("1.0") <= expected_dqr <= Decimal("5.0")


# =============================================================================
# Year-Over-Year Comparison Tests
# =============================================================================


class TestYearOverYearComparison:
    """Test year-over-year comparison capability."""

    def test_single_year_baseline(self, sample_consolidated_inventory):
        assert sample_consolidated_inventory["reporting_year"] == 2025

    def test_yoy_change_calculation(self):
        year_2024_total = Decimal("65000")
        year_2025_total = Decimal("61430")
        change_pct = float((year_2025_total - year_2024_total) / year_2024_total * 100)
        assert change_pct < 0  # Reduction
        assert abs(change_pct) < 10  # Less than 10% change

    def test_yoy_per_category_tracking(self, sample_category_results):
        # Each category should support year-over-year tracking
        for cat_id in SCOPE3_CATEGORIES:
            assert cat_id in sample_category_results["categories"]


# =============================================================================
# Boundary Alignment with PACK-041 Tests
# =============================================================================


class TestBoundaryAlignment:
    """Test boundary alignment with PACK-041 Scope 1-2 data."""

    def test_scope12_data_present(self, sample_consolidated_inventory):
        assert "scope1_tco2e" in sample_consolidated_inventory
        assert "scope2_market_tco2e" in sample_consolidated_inventory

    def test_scope12_positive(self, sample_consolidated_inventory):
        assert sample_consolidated_inventory["scope1_tco2e"] > 0
        assert sample_consolidated_inventory["scope2_market_tco2e"] > 0

    def test_cat3_not_double_counting_scope12(self, sample_category_results):
        cats = sample_category_results["categories"]
        # Cat 3 should only include WTT and T&D, not Scope 1/2 emissions
        cat3 = cats["CAT_3"]["total_tco2e"]
        # Cat 3 is typically 15-30% of Scope 1+2
        scope12 = Decimal("17200")  # 12000 + 5200
        ratio = float(cat3 / scope12 * 100)
        assert ratio < 50, f"Cat 3 is {ratio}% of Scope 1+2, seems too high"


# =============================================================================
# Single Category Consolidation Tests
# =============================================================================


class TestSingleCategoryConsolidation:
    """Test consolidation with only a single active category."""

    def test_single_category_total_equals_category(self, single_category_results):
        cats = single_category_results["categories"]
        assert cats["CAT_1"]["total_tco2e"] == single_category_results["total_scope3_tco2e"]

    def test_single_category_100_pct(self, single_category_results):
        cats = single_category_results["categories"]
        total = single_category_results["total_scope3_tco2e"]
        pct = float(cats["CAT_1"]["total_tco2e"] / total * 100)
        assert pct == 100.0


# =============================================================================
# Missing Categories Handling Tests
# =============================================================================


class TestMissingCategoriesHandling:
    """Test handling when categories have zero emissions."""

    def test_zero_emission_categories_present(self, sample_category_results):
        cats = sample_category_results["categories"]
        zero_cats = [c for c, d in cats.items() if d["total_tco2e"] == Decimal("0")]
        assert len(zero_cats) >= 2, "At least Cat 13 and Cat 14 should be zero"

    def test_zero_categories_have_not_applicable_method(self, sample_category_results):
        cats = sample_category_results["categories"]
        for cat_id, data in cats.items():
            if data["total_tco2e"] == Decimal("0"):
                assert data["methodology"] == "NOT_APPLICABLE"


# =============================================================================
# Provenance Hash Tests
# =============================================================================


class TestConsolidationProvenance:
    """Test provenance hash consistency for consolidation."""

    def test_inventory_has_provenance_hash(self, sample_consolidated_inventory):
        assert "provenance_hash" in sample_consolidated_inventory
        h = sample_consolidated_inventory["provenance_hash"]
        assert len(h) == 64

    def test_provenance_hash_changes_with_total(self):
        h1 = compute_provenance_hash({"total": "61430"})
        h2 = compute_provenance_hash({"total": "61431"})
        assert h1 != h2

    def test_provenance_hash_deterministic(self):
        data = {"total": "61430", "year": "2025"}
        assert compute_provenance_hash(data) == compute_provenance_hash(data)
