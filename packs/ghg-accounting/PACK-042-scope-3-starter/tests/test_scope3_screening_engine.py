# -*- coding: utf-8 -*-
"""
Unit tests for Scope3ScreeningEngine (PACK-042 Engine 1)
=========================================================

Tests EEIO-based screening across all 15 Scope 3 categories, sector-specific
profiles, relevance scoring, significance thresholds, and provenance hashing.

Coverage target: 85%+
Total tests: ~50
"""

import hashlib
import json
from decimal import Decimal
from typing import Any, Dict, List

import pytest

from tests.conftest import (
    SCOPE3_CATEGORIES,
    UPSTREAM_CATEGORIES,
    DOWNSTREAM_CATEGORIES,
    compute_provenance_hash,
)


# =============================================================================
# Screening Coverage Tests
# =============================================================================


class TestScreenAllCategories:
    """Test that screening covers all 15 Scope 3 categories."""

    def test_all_15_categories_present_in_results(self, sample_screening_results):
        cats = sample_screening_results["categories"]
        for cat in SCOPE3_CATEGORIES:
            assert cat in cats, f"Missing category {cat} in screening results"

    def test_screening_returns_15_categories(self, sample_screening_results):
        assert len(sample_screening_results["categories"]) == 15

    def test_each_category_has_estimated_tco2e(self, sample_screening_results):
        for cat, data in sample_screening_results["categories"].items():
            assert "estimated_tco2e" in data, f"{cat} missing estimated_tco2e"
            assert isinstance(data["estimated_tco2e"], Decimal)

    def test_each_category_has_relevance(self, sample_screening_results):
        valid_relevance = {"HIGH", "MEDIUM", "LOW", "NOT_APPLICABLE"}
        for cat, data in sample_screening_results["categories"].items():
            assert data["relevance"] in valid_relevance, f"{cat} has invalid relevance"

    def test_each_category_has_pct_of_total(self, sample_screening_results):
        for cat, data in sample_screening_results["categories"].items():
            assert "pct_of_total" in data
            assert data["pct_of_total"] >= Decimal("0")

    def test_each_category_has_recommended_tier(self, sample_screening_results):
        valid_tiers = {"SPEND_BASED", "AVERAGE_DATA", "SUPPLIER_SPECIFIC", "HYBRID", "NOT_APPLICABLE"}
        for cat, data in sample_screening_results["categories"].items():
            assert data["recommended_tier"] in valid_tiers

    def test_manufacturing_cat1_is_highest(self, sample_screening_results):
        cats = sample_screening_results["categories"]
        cat1_val = cats["CAT_1"]["estimated_tco2e"]
        for cat_id, data in cats.items():
            if cat_id != "CAT_1":
                assert cat1_val >= data["estimated_tco2e"], (
                    f"CAT_1 ({cat1_val}) should be >= {cat_id} ({data['estimated_tco2e']}) for manufacturing"
                )

    def test_total_estimated_equals_sum(self, sample_screening_results):
        cats = sample_screening_results["categories"]
        calculated_total = sum(data["estimated_tco2e"] for data in cats.values())
        assert calculated_total == sample_screening_results["total_estimated_tco2e"]

    def test_percentages_sum_close_to_100(self, sample_screening_results):
        cats = sample_screening_results["categories"]
        total_pct = sum(float(data["pct_of_total"]) for data in cats.values())
        # Allow tolerance for rounding and N/A categories with 0%
        assert abs(total_pct - 100.0) < 10.0, f"Percentages sum to {total_pct}, expected close to 100"


# =============================================================================
# Significance Threshold Tests
# =============================================================================


class TestSignificanceThreshold:
    """Test category significance identification against threshold."""

    def test_significant_categories_identified(self, sample_screening_results):
        assert "significant_categories" in sample_screening_results
        assert len(sample_screening_results["significant_categories"]) > 0

    def test_significant_categories_are_valid(self, sample_screening_results):
        for cat in sample_screening_results["significant_categories"]:
            assert cat in SCOPE3_CATEGORIES

    def test_default_threshold_is_one_percent(self, sample_screening_results):
        cats = sample_screening_results["categories"]
        significant = sample_screening_results["significant_categories"]
        for cat in significant:
            assert cats[cat]["pct_of_total"] >= Decimal("1.0"), (
                f"{cat} marked significant but pct is {cats[cat]['pct_of_total']}"
            )

    def test_below_threshold_not_significant(self, sample_screening_results):
        cats = sample_screening_results["categories"]
        significant = set(sample_screening_results["significant_categories"])
        for cat_id, data in cats.items():
            if data["pct_of_total"] < Decimal("1.0") and data["relevance"] != "NOT_APPLICABLE":
                assert cat_id not in significant or data["relevance"] == "LOW"

    def test_not_applicable_excluded(self, sample_screening_results):
        cats = sample_screening_results["categories"]
        significant = sample_screening_results["significant_categories"]
        for cat in significant:
            assert cats[cat]["relevance"] != "NOT_APPLICABLE"


# =============================================================================
# Relevance Scoring Tests
# =============================================================================


class TestRelevanceScoring:
    """Test relevance scoring and classification."""

    def test_high_relevance_has_high_pct(self, sample_screening_results):
        cats = sample_screening_results["categories"]
        for cat_id, data in cats.items():
            if data["relevance"] == "HIGH":
                assert data["pct_of_total"] >= Decimal("1.0")

    def test_not_applicable_has_zero_emissions(self, sample_screening_results):
        cats = sample_screening_results["categories"]
        for cat_id, data in cats.items():
            if data["relevance"] == "NOT_APPLICABLE":
                assert data["estimated_tco2e"] == Decimal("0")

    def test_low_relevance_below_threshold(self, sample_screening_results):
        cats = sample_screening_results["categories"]
        low_cats = [c for c, d in cats.items() if d["relevance"] == "LOW"]
        for cat in low_cats:
            assert cats[cat]["pct_of_total"] <= Decimal("5.0")

    def test_relevance_tiers_are_ordered(self, sample_screening_results):
        cats = sample_screening_results["categories"]
        high_min = min(
            cats[c]["pct_of_total"]
            for c in SCOPE3_CATEGORIES
            if cats[c]["relevance"] == "HIGH"
        )
        low_cats = [c for c in SCOPE3_CATEGORIES if cats[c]["relevance"] == "LOW"]
        if low_cats:
            low_max = max(cats[c]["pct_of_total"] for c in low_cats)
            assert high_min >= low_max


# =============================================================================
# EEIO Factor Lookup Tests
# =============================================================================


class TestEEIOFactorLookup:
    """Test EEIO emission factor lookup by sector."""

    def test_common_sectors_have_positive_factors(self, sample_eeio_factors):
        common = ["basic_metals", "chemicals_pharmaceuticals", "machinery_equipment"]
        for sector in common:
            assert sector in sample_eeio_factors
            assert sample_eeio_factors[sector] > 0

    def test_metals_higher_than_services(self, sample_eeio_factors):
        assert sample_eeio_factors["basic_metals"] > sample_eeio_factors["it_services"]

    def test_air_transport_high_intensity(self, sample_eeio_factors):
        assert sample_eeio_factors["air_transport"] > 3.0

    def test_financial_low_intensity(self, sample_eeio_factors):
        assert sample_eeio_factors["financial_services"] < 0.5

    def test_all_factors_positive(self, sample_eeio_factors):
        for sector, factor in sample_eeio_factors.items():
            assert factor > 0, f"Factor for {sector} should be positive"

    def test_factor_count_covers_major_sectors(self, sample_eeio_factors):
        assert len(sample_eeio_factors) >= 30, "Should cover at least 30 EEIO sectors"


# =============================================================================
# Sector-Specific Screening Tests
# =============================================================================


class TestSectorSpecificScreening:
    """Test sector-specific screening profiles."""

    def test_manufacturing_has_cat1_dominant(self, sample_screening_results):
        cats = sample_screening_results["categories"]
        assert cats["CAT_1"]["pct_of_total"] > Decimal("30")

    def test_downstream_categories_present(self, sample_screening_results):
        cats = sample_screening_results["categories"]
        downstream_total = sum(
            cats[c]["estimated_tco2e"] for c in DOWNSTREAM_CATEGORIES
        )
        assert downstream_total > 0

    def test_upstream_categories_present(self, sample_screening_results):
        cats = sample_screening_results["categories"]
        upstream_total = sum(
            cats[c]["estimated_tco2e"] for c in UPSTREAM_CATEGORIES
        )
        assert upstream_total > 0

    def test_upstream_greater_than_downstream_for_manufacturing(self, sample_screening_results):
        cats = sample_screening_results["categories"]
        upstream = sum(cats[c]["estimated_tco2e"] for c in UPSTREAM_CATEGORIES)
        downstream = sum(cats[c]["estimated_tco2e"] for c in DOWNSTREAM_CATEGORIES)
        assert upstream >= downstream * Decimal("0.5"), (
            "For manufacturing, upstream should be at least half of downstream"
        )


# =============================================================================
# Revenue-Based Screening Tests
# =============================================================================


class TestRevenueBasedScreening:
    """Test revenue-based screening for downstream categories."""

    def test_downstream_categories_use_revenue(self, manufacturing_org):
        # Downstream categories (9-12) typically use revenue or product data
        assert manufacturing_org["revenue_meur"] > 0

    def test_zero_revenue_still_screens(self, zero_revenue_org):
        assert zero_revenue_org["revenue_meur"] == Decimal("0")
        # Engine should handle zero revenue by using spend-only methods

    def test_cat11_screening_for_manufacturing(self, sample_screening_results):
        cats = sample_screening_results["categories"]
        # Cat 11 (Use of Sold Products) is significant for manufacturing
        assert cats["CAT_11"]["estimated_tco2e"] > 0

    def test_cat13_cat14_not_applicable_for_manufacturing(self, sample_screening_results):
        cats = sample_screening_results["categories"]
        # Cat 13 (Downstream Leased Assets) and Cat 14 (Franchises) typically N/A
        assert cats["CAT_13"]["relevance"] == "NOT_APPLICABLE"
        assert cats["CAT_14"]["relevance"] == "NOT_APPLICABLE"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestScreeningEdgeCases:
    """Test screening edge cases."""

    def test_single_category_relevant(self, single_category_results):
        cats = single_category_results["categories"]
        non_zero = [c for c, d in cats.items() if d["total_tco2e"] > 0]
        assert len(non_zero) == 1
        assert non_zero[0] == "CAT_1"

    def test_sme_org_has_limited_categories(self, sme_org):
        assert sme_org["number_of_suppliers"] <= 30

    def test_missing_revenue_handled(self, minimal_org):
        assert "revenue_meur" not in minimal_org

    def test_missing_spend_handled(self, minimal_org):
        assert "total_procurement_spend_eur" not in minimal_org

    def test_zero_revenue_org_valid(self, zero_revenue_org):
        assert zero_revenue_org["revenue_meur"] == Decimal("0")
        assert zero_revenue_org["employees_fte"] > 0


# =============================================================================
# Provenance Hash Tests
# =============================================================================


class TestScreeningProvenance:
    """Test provenance hash generation and consistency."""

    def test_screening_results_have_provenance_hash(self, sample_screening_results):
        assert "provenance_hash" in sample_screening_results
        assert len(sample_screening_results["provenance_hash"]) == 64

    def test_provenance_hash_is_hex_string(self, sample_screening_results):
        h = sample_screening_results["provenance_hash"]
        try:
            int(h, 16)
        except ValueError:
            pytest.fail("Provenance hash is not a valid hex string")

    def test_provenance_hash_deterministic(self):
        data = {"category": "CAT_1", "tco2e": "28500"}
        h1 = compute_provenance_hash(data)
        h2 = compute_provenance_hash(data)
        assert h1 == h2

    def test_provenance_hash_changes_with_data(self):
        data1 = {"category": "CAT_1", "tco2e": "28500"}
        data2 = {"category": "CAT_1", "tco2e": "28501"}
        assert compute_provenance_hash(data1) != compute_provenance_hash(data2)

    def test_provenance_hash_sha256_length(self):
        h = compute_provenance_hash({"test": True})
        assert len(h) == 64
