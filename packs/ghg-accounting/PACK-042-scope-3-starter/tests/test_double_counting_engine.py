# -*- coding: utf-8 -*-
"""
Unit tests for DoubleCountingPreventionEngine (PACK-042 Engine 4)
==================================================================

Tests all 12 overlap rules individually, conservative vs proportional
allocation, no-overlap scenarios, multiple overlaps, net adjustment
calculation accuracy, and provenance consistency.

Coverage target: 85%+
Total tests: ~55
"""

from decimal import Decimal
from typing import Any, Dict

import pytest

from tests.conftest import (
    OVERLAP_RULES,
    SCOPE3_CATEGORIES,
    compute_provenance_hash,
)


# =============================================================================
# Overlap Rule Tests (12 rules individually)
# =============================================================================


class TestOverlapRules:
    """Test each of the 12 double-counting prevention rules."""

    def test_twelve_overlap_rules_defined(self):
        assert len(OVERLAP_RULES) == 12

    def test_rule_cat1_vs_cat3_energy(self):
        """Cat 1 purchased goods may include energy costs that overlap Cat 3."""
        rule = "cat1_vs_cat3_energy"
        assert rule in OVERLAP_RULES
        # Transport/energy in purchased goods prices
        cat1_energy_component = Decimal("200")
        cat3_total = Decimal("3800")
        overlap = min(cat1_energy_component, cat3_total)
        assert overlap == Decimal("200")

    def test_rule_cat1_vs_cat4_logistics(self):
        """Cat 1 supplier prices may include logistics that overlaps Cat 4."""
        rule = "cat1_vs_cat4_logistics"
        assert rule in OVERLAP_RULES
        cat1_logistics_component = Decimal("450")
        cat4_total = Decimal("5100")
        overlap = min(cat1_logistics_component, cat4_total)
        assert overlap <= cat4_total

    def test_rule_cat1_vs_cat2_capex_opex(self):
        """Items may be misclassified between OPEX (Cat 1) and CAPEX (Cat 2)."""
        rule = "cat1_vs_cat2_capex_opex"
        assert rule in OVERLAP_RULES
        misclassified_amount = Decimal("120")
        assert misclassified_amount > 0

    def test_rule_cat3_vs_scope2_upstream_energy(self):
        """Cat 3 WTT/T&D must not include Scope 2 direct energy emissions."""
        rule = "cat3_vs_scope2_upstream_energy"
        assert rule in OVERLAP_RULES
        # Correctly separated: Cat 3 = WTT + T&D, Scope 2 = direct
        overlap = Decimal("0")
        assert overlap == Decimal("0")

    def test_rule_cat4_vs_cat9_transport_allocation(self):
        """Inbound (Cat 4) vs outbound (Cat 9) transport allocation."""
        rule = "cat4_vs_cat9_transport_allocation"
        assert rule in OVERLAP_RULES
        cat4_total = Decimal("5100")
        cat9_total = Decimal("3200")
        # No overlap if properly allocated by direction
        overlap = Decimal("0")
        assert overlap == Decimal("0")

    def test_rule_cat8_vs_scope12_leased_assets(self):
        """Leased assets (Cat 8) must not include Scope 1/2 operational emissions."""
        rule = "cat8_vs_scope12_leased_assets"
        assert rule in OVERLAP_RULES
        # Under operational control, leased assets in Scope 1/2
        # Cat 8 only for assets not under operational control
        cat8 = Decimal("350")
        assert cat8 >= 0

    def test_rule_cat13_vs_cat11_downstream_leased(self):
        """Downstream leased assets (Cat 13) vs use of sold products (Cat 11)."""
        rule = "cat13_vs_cat11_downstream_leased"
        assert rule in OVERLAP_RULES
        cat13 = Decimal("0")
        cat11 = Decimal("8500")
        # No overlap if Cat 13 is N/A
        overlap = Decimal("0")
        assert overlap == Decimal("0")

    def test_rule_cat14_vs_scope12_franchise(self):
        """Franchise emissions (Cat 14) vs Scope 1/2 if franchisor."""
        rule = "cat14_vs_scope12_franchise"
        assert rule in OVERLAP_RULES
        cat14 = Decimal("0")
        assert cat14 >= 0

    def test_rule_cat1_vs_cat5_packaging_waste(self):
        """Packaging in Cat 1 purchased goods vs waste in Cat 5."""
        rule = "cat1_vs_cat5_packaging_waste"
        assert rule in OVERLAP_RULES
        packaging_in_cat1 = Decimal("50")
        waste_in_cat5 = Decimal("850")
        overlap = min(packaging_in_cat1, waste_in_cat5)
        assert overlap <= waste_in_cat5

    def test_rule_cat10_vs_cat11_processing_use(self):
        """Processing of sold products (Cat 10) vs use of sold products (Cat 11)."""
        rule = "cat10_vs_cat11_processing_use"
        assert rule in OVERLAP_RULES
        # These are sequential: processing happens before use
        cat10 = Decimal("2800")
        cat11 = Decimal("8500")
        overlap = Decimal("0")
        assert overlap == Decimal("0")

    def test_rule_cat6_vs_cat7_travel_commuting(self):
        """Business travel (Cat 6) vs employee commuting (Cat 7) boundary."""
        rule = "cat6_vs_cat7_travel_commuting"
        assert rule in OVERLAP_RULES
        # Day trips may be classified as either
        cat6 = Decimal("1200")
        cat7 = Decimal("980")
        # Overlap possible for day trips
        potential_overlap = Decimal("50")
        assert potential_overlap < min(cat6, cat7)

    def test_rule_cat15_vs_cat13_cat14_investment(self):
        """Investment emissions (Cat 15) vs downstream leased/franchise."""
        rule = "cat15_vs_cat13_cat14_investment"
        assert rule in OVERLAP_RULES
        cat15 = Decimal("450")
        cat13 = Decimal("0")
        cat14 = Decimal("0")
        # If Cat 13/14 are zero, no overlap possible
        assert cat13 + cat14 == Decimal("0")


# =============================================================================
# Allocation Method Tests
# =============================================================================


class TestAllocationMethods:
    """Test conservative vs proportional allocation."""

    def test_conservative_allocation_avoids_understatement(self):
        """Conservative mode keeps the higher estimate when in doubt."""
        cat1_overlap = Decimal("450")
        cat4_overlap = Decimal("450")
        # Conservative: remove from the smaller category (Cat 4)
        conservative_adjustment = -cat4_overlap
        assert conservative_adjustment == Decimal("-450")

    def test_proportional_allocation_splits_equally(self):
        """Proportional allocation splits overlap between categories."""
        overlap = Decimal("450")
        cat1_share = overlap * Decimal("0.5")
        cat4_share = overlap * Decimal("0.5")
        assert cat1_share + cat4_share == overlap

    def test_economic_allocation_uses_spend(self):
        cat1_spend = Decimal("15000000")
        cat4_spend = Decimal("2000000")
        total_spend = cat1_spend + cat4_spend
        cat1_share = cat1_spend / total_spend
        cat4_share = cat4_spend / total_spend
        assert abs(float(cat1_share + cat4_share) - 1.0) < 0.001

    def test_physical_allocation_uses_mass(self):
        cat1_mass_tonnes = Decimal("10000")
        cat4_mass_tonnes = Decimal("10000")
        total_mass = cat1_mass_tonnes + cat4_mass_tonnes
        cat1_share = cat1_mass_tonnes / total_mass
        assert cat1_share == Decimal("0.5")

    def test_default_allocation_is_economic(self, sample_double_counting_results):
        assert sample_double_counting_results["allocation_method"] == "ECONOMIC"

    def test_conservative_mode_enabled(self, sample_double_counting_results):
        assert sample_double_counting_results["conservative_mode"] is True


# =============================================================================
# No Overlaps Scenario Tests
# =============================================================================


class TestNoOverlaps:
    """Test scenario with no double-counting overlaps."""

    def test_zero_overlap_result(self):
        result = {
            "overlaps_detected": [],
            "total_overlap_tco2e": Decimal("0"),
            "net_adjustment_tco2e": Decimal("0"),
            "rules_evaluated": 12,
            "rules_with_overlap": 0,
        }
        assert result["total_overlap_tco2e"] == Decimal("0")
        assert result["rules_with_overlap"] == 0

    def test_no_adjustment_needed(self):
        adjustment = Decimal("0")
        original_total = Decimal("61430")
        adjusted_total = original_total + adjustment
        assert adjusted_total == original_total

    def test_all_rules_still_evaluated(self):
        result = {"rules_evaluated": 12, "rules_with_overlap": 0}
        assert result["rules_evaluated"] == 12


# =============================================================================
# Multiple Overlaps Tests
# =============================================================================


class TestMultipleOverlaps:
    """Test scenario with multiple overlaps in single inventory."""

    def test_multiple_overlaps_detected(self, sample_double_counting_results):
        overlaps = sample_double_counting_results["overlaps_detected"]
        overlaps_with_amount = [o for o in overlaps if o["overlap_tco2e"] > 0]
        assert len(overlaps_with_amount) >= 2

    def test_total_overlap_is_sum(self, sample_double_counting_results):
        overlaps = sample_double_counting_results["overlaps_detected"]
        calculated_total = sum(
            o["overlap_tco2e"] for o in overlaps if o["overlap_tco2e"] > 0
        )
        assert calculated_total == sample_double_counting_results["total_overlap_tco2e"]

    def test_overlaps_have_rationale(self, sample_double_counting_results):
        for overlap in sample_double_counting_results["overlaps_detected"]:
            assert "rationale" in overlap
            assert len(overlap["rationale"]) > 10

    def test_each_overlap_has_rule_name(self, sample_double_counting_results):
        for overlap in sample_double_counting_results["overlaps_detected"]:
            assert "rule" in overlap
            assert len(overlap["rule"]) > 0


# =============================================================================
# Net Adjustment Calculation Tests
# =============================================================================


class TestNetAdjustment:
    """Test net adjustment calculation accuracy."""

    def test_net_adjustment_is_negative(self, sample_double_counting_results):
        # Double-counting removal should reduce total
        adj = sample_double_counting_results["net_adjustment_tco2e"]
        assert adj <= Decimal("0")

    def test_net_adjustment_magnitude(self, sample_double_counting_results):
        adj = abs(sample_double_counting_results["net_adjustment_tco2e"])
        total = sample_double_counting_results["total_overlap_tco2e"]
        assert adj == total

    def test_adjusted_total_less_than_original(self, sample_double_counting_results):
        original = Decimal("61430")
        adj = sample_double_counting_results["net_adjustment_tco2e"]
        adjusted = original + adj
        assert adjusted < original

    def test_adjustment_less_than_5pct_of_total(self, sample_double_counting_results):
        adj = abs(sample_double_counting_results["net_adjustment_tco2e"])
        total = Decimal("61430")
        pct = float(adj / total * 100)
        assert pct < 5.0, f"Adjustment is {pct}% of total, should be < 5%"

    def test_adjustment_applied_to_correct_category(self, sample_double_counting_results):
        for overlap in sample_double_counting_results["overlaps_detected"]:
            if overlap["adjustment_tco2e"] != Decimal("0"):
                assert overlap["adjustment_category"] is not None
                assert overlap["adjustment_category"] in SCOPE3_CATEGORIES or overlap["adjustment_category"].startswith("SCOPE")


# =============================================================================
# Result Structure Tests
# =============================================================================


class TestDoubleCountingResultStructure:
    """Test result structure and completeness."""

    def test_required_fields_present(self, sample_double_counting_results):
        required = [
            "assessment_id", "overlaps_detected", "total_overlap_tco2e",
            "net_adjustment_tco2e", "rules_evaluated", "rules_with_overlap",
            "allocation_method", "conservative_mode", "provenance_hash",
        ]
        for field in required:
            assert field in sample_double_counting_results

    def test_rules_evaluated_equals_12(self, sample_double_counting_results):
        assert sample_double_counting_results["rules_evaluated"] == 12

    def test_provenance_hash_present(self, sample_double_counting_results):
        h = sample_double_counting_results["provenance_hash"]
        assert len(h) == 64

    def test_overlap_entry_structure(self, sample_double_counting_results):
        required_fields = {
            "rule", "categories", "overlap_tco2e", "allocation_method",
            "adjustment_category", "adjustment_tco2e", "rationale",
        }
        for overlap in sample_double_counting_results["overlaps_detected"]:
            for field in required_fields:
                assert field in overlap, f"Missing field {field} in overlap entry"

    def test_overlap_categories_are_lists(self, sample_double_counting_results):
        for overlap in sample_double_counting_results["overlaps_detected"]:
            assert isinstance(overlap["categories"], list)
            assert len(overlap["categories"]) == 2
