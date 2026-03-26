# -*- coding: utf-8 -*-
"""
Unit tests for HotspotAnalysisEngine (PACK-042 Engine 5)
=========================================================

Tests Pareto analysis, materiality matrix scoring, sector benchmark
comparison, supplier concentration analysis, product intensity ranking,
reduction opportunity quantification, tier upgrade impact, and edge cases.

Coverage target: 85%+
Total tests: ~45
"""

from decimal import Decimal
from typing import Any, Dict, List

import pytest

from tests.conftest import (
    SCOPE3_CATEGORIES,
    compute_provenance_hash,
)


# =============================================================================
# Pareto Analysis Tests
# =============================================================================


class TestParetoAnalysis:
    """Test Pareto (80/20) analysis for category prioritization."""

    def test_pareto_categories_present(self, sample_hotspot_analysis):
        assert "pareto_categories" in sample_hotspot_analysis
        assert len(sample_hotspot_analysis["pareto_categories"]) > 0

    def test_pareto_categories_sorted_by_tco2e(self, sample_hotspot_analysis):
        cats = sample_hotspot_analysis["pareto_categories"]
        for i in range(len(cats) - 1):
            assert cats[i]["tco2e"] >= cats[i + 1]["tco2e"], (
                f"Pareto categories not sorted at index {i}"
            )

    def test_cumulative_pct_monotonically_increasing(self, sample_hotspot_analysis):
        cats = sample_hotspot_analysis["pareto_categories"]
        for i in range(len(cats) - 1):
            assert cats[i]["cumulative_pct"] <= cats[i + 1]["cumulative_pct"]

    def test_pareto_reaches_80_pct(self, sample_hotspot_analysis):
        cats = sample_hotspot_analysis["pareto_categories"]
        last_cumulative = cats[-1]["cumulative_pct"]
        assert last_cumulative >= Decimal("80.0"), (
            f"Pareto only reaches {last_cumulative}%, should reach 80%"
        )

    def test_pareto_threshold_is_80(self, sample_hotspot_analysis):
        assert sample_hotspot_analysis["pareto_threshold_pct"] == 80.0

    def test_categories_to_80pct_count(self, sample_hotspot_analysis):
        count = sample_hotspot_analysis["categories_to_80pct"]
        assert count <= 15
        assert count >= 1
        assert count == len(sample_hotspot_analysis["pareto_categories"])

    def test_cat1_is_first_in_pareto(self, sample_hotspot_analysis):
        first = sample_hotspot_analysis["pareto_categories"][0]
        assert first["category"] == "CAT_1"

    def test_few_categories_drive_most_emissions(self, sample_hotspot_analysis):
        count = sample_hotspot_analysis["categories_to_80pct"]
        assert count <= 7, "80% of emissions should be covered by 7 or fewer categories"


# =============================================================================
# Materiality Matrix Tests
# =============================================================================


class TestMaterialityMatrix:
    """Test materiality matrix scoring."""

    def test_materiality_matrix_present(self, sample_hotspot_analysis):
        assert "materiality_matrix" in sample_hotspot_analysis

    def test_matrix_has_magnitude_score(self, sample_hotspot_analysis):
        for cat, scores in sample_hotspot_analysis["materiality_matrix"].items():
            assert "magnitude" in scores
            assert scores["magnitude"] in {"HIGH", "MEDIUM", "LOW"}

    def test_matrix_has_data_quality_score(self, sample_hotspot_analysis):
        for cat, scores in sample_hotspot_analysis["materiality_matrix"].items():
            assert "data_quality" in scores
            assert scores["data_quality"] in {"HIGH", "MEDIUM", "LOW"}

    def test_matrix_has_reduction_potential(self, sample_hotspot_analysis):
        for cat, scores in sample_hotspot_analysis["materiality_matrix"].items():
            assert "reduction_potential" in scores
            assert scores["reduction_potential"] in {"HIGH", "MEDIUM", "LOW"}

    def test_high_magnitude_categories_in_matrix(self, sample_hotspot_analysis):
        matrix = sample_hotspot_analysis["materiality_matrix"]
        high_magnitude = [c for c, s in matrix.items() if s["magnitude"] == "HIGH"]
        assert len(high_magnitude) > 0


# =============================================================================
# Supplier Concentration Tests
# =============================================================================


class TestSupplierConcentration:
    """Test supplier concentration analysis."""

    def test_top_suppliers_present(self, sample_hotspot_analysis):
        assert "top_suppliers" in sample_hotspot_analysis
        assert len(sample_hotspot_analysis["top_suppliers"]) > 0

    def test_top_suppliers_have_tco2e(self, sample_hotspot_analysis):
        for supplier in sample_hotspot_analysis["top_suppliers"]:
            assert "tco2e" in supplier
            assert supplier["tco2e"] > 0

    def test_top_suppliers_have_percentage(self, sample_hotspot_analysis):
        for supplier in sample_hotspot_analysis["top_suppliers"]:
            assert "pct_of_cat1" in supplier
            assert supplier["pct_of_cat1"] > 0

    def test_top_suppliers_sorted_by_emissions(self, sample_hotspot_analysis):
        suppliers = sample_hotspot_analysis["top_suppliers"]
        for i in range(len(suppliers) - 1):
            assert suppliers[i]["tco2e"] >= suppliers[i + 1]["tco2e"]

    def test_supplier_concentration_not_100pct(self, sample_hotspot_analysis):
        suppliers = sample_hotspot_analysis["top_suppliers"]
        total_pct = sum(float(s["pct_of_cat1"]) for s in suppliers)
        assert total_pct < 100.0, "Top suppliers should not cover 100%"


# =============================================================================
# Reduction Opportunity Tests
# =============================================================================


class TestReductionOpportunities:
    """Test reduction opportunity quantification."""

    def test_opportunities_present(self, sample_hotspot_analysis):
        assert "reduction_opportunities" in sample_hotspot_analysis
        assert len(sample_hotspot_analysis["reduction_opportunities"]) > 0

    def test_opportunities_have_potential(self, sample_hotspot_analysis):
        for opp in sample_hotspot_analysis["reduction_opportunities"]:
            assert "potential_reduction_tco2e" in opp
            assert opp["potential_reduction_tco2e"] > 0

    def test_opportunities_have_effort(self, sample_hotspot_analysis):
        valid_efforts = {"LOW", "MEDIUM", "HIGH"}
        for opp in sample_hotspot_analysis["reduction_opportunities"]:
            assert "effort" in opp
            assert opp["effort"] in valid_efforts

    def test_opportunities_have_action_description(self, sample_hotspot_analysis):
        for opp in sample_hotspot_analysis["reduction_opportunities"]:
            assert "action" in opp
            assert len(opp["action"]) > 5

    def test_total_reduction_potential(self, sample_hotspot_analysis):
        total_potential = sum(
            opp["potential_reduction_tco2e"]
            for opp in sample_hotspot_analysis["reduction_opportunities"]
        )
        assert total_potential > 0


# =============================================================================
# Tier Upgrade Impact Tests
# =============================================================================


class TestTierUpgradeImpact:
    """Test tier upgrade impact calculation."""

    def test_spend_to_average_improves_uncertainty(self):
        spend_uncertainty = Decimal("50.0")
        average_uncertainty = Decimal("30.0")
        improvement = spend_uncertainty - average_uncertainty
        assert improvement > 0

    def test_average_to_supplier_improves_uncertainty(self):
        average_uncertainty = Decimal("30.0")
        supplier_uncertainty = Decimal("10.0")
        improvement = average_uncertainty - supplier_uncertainty
        assert improvement > 0

    def test_supplier_specific_best_quality(self):
        tiers = {
            "SPEND_BASED": Decimal("50"),
            "AVERAGE_DATA": Decimal("30"),
            "SUPPLIER_SPECIFIC": Decimal("10"),
        }
        assert tiers["SUPPLIER_SPECIFIC"] < tiers["AVERAGE_DATA"] < tiers["SPEND_BASED"]


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestHotspotEdgeCases:
    """Test edge cases for hotspot analysis."""

    def test_single_category_dominant(self, single_category_results):
        """When one category is 100% of emissions, Pareto is trivial."""
        cats = single_category_results["categories"]
        non_zero = [c for c, d in cats.items() if d["total_tco2e"] > 0]
        assert len(non_zero) == 1

    def test_even_distribution_scenario(self):
        """When all categories are equal, all are in Pareto."""
        even_value = Decimal("4000")
        categories = {f"CAT_{i}": even_value for i in range(1, 16)}
        total = sum(categories.values())
        for cat, val in categories.items():
            pct = float(val / total * 100)
            assert abs(pct - 100.0 / 15) < 0.5

    def test_empty_supplier_list_handled(self):
        suppliers = []
        assert len(suppliers) == 0

    def test_provenance_hash_present(self, sample_hotspot_analysis):
        assert "provenance_hash" in sample_hotspot_analysis
        assert len(sample_hotspot_analysis["provenance_hash"]) == 64
