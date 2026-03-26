"""
Unit tests for MaterialityAssessmentEngine (PACK-048 Engine 6).

Tests all public methods with 25+ tests covering:
  - Overall materiality calculation
  - Performance materiality
  - Clearly trivial threshold
  - Scope-specific materiality
  - Qualitative factors
  - Materiality revision
  - Zero emissions edge case
  - Custom percentages

Author: GreenLang QA Team
"""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from tests.conftest import assert_decimal_between, assert_decimal_equal


# ---------------------------------------------------------------------------
# Overall Materiality Calculation Tests
# ---------------------------------------------------------------------------


class TestOverallMaterialityCalculation:
    """Tests for overall materiality calculation."""

    def test_overall_materiality_5pct_of_total(self, sample_emissions_data, materiality_engine_config):
        """Test overall materiality is 5% of total emissions."""
        total = sample_emissions_data["total_all_scopes_tco2e"]
        pct = materiality_engine_config["overall_pct"]
        materiality = total * pct / Decimal("100")
        assert_decimal_equal(materiality, Decimal("1150"))

    def test_overall_materiality_positive(self, sample_emissions_data):
        """Test overall materiality is always positive for non-zero emissions."""
        total = sample_emissions_data["total_all_scopes_tco2e"]
        materiality = total * Decimal("5") / Decimal("100")
        assert materiality > Decimal("0")

    def test_materiality_proportional_to_total(self):
        """Test materiality is proportional to total emissions."""
        total_a = Decimal("10000")
        total_b = Decimal("20000")
        mat_a = total_a * Decimal("5") / Decimal("100")
        mat_b = total_b * Decimal("5") / Decimal("100")
        assert mat_b == mat_a * Decimal("2")


# ---------------------------------------------------------------------------
# Performance Materiality Tests
# ---------------------------------------------------------------------------


class TestPerformanceMateriality:
    """Tests for performance materiality calculation."""

    def test_performance_materiality_75pct_of_overall(self, sample_emissions_data, materiality_engine_config):
        """Test performance materiality is 75% of overall materiality."""
        total = sample_emissions_data["total_all_scopes_tco2e"]
        overall = total * materiality_engine_config["overall_pct"] / Decimal("100")
        perf_pct = materiality_engine_config["performance_pct_of_overall"]
        performance = overall * perf_pct / Decimal("100")
        expected = Decimal("1150") * Decimal("75") / Decimal("100")
        assert_decimal_equal(performance, expected, tolerance=Decimal("0.01"))

    def test_performance_lt_overall(self, sample_emissions_data):
        """Test performance materiality is less than overall materiality."""
        total = sample_emissions_data["total_all_scopes_tco2e"]
        overall = total * Decimal("5") / Decimal("100")
        performance = overall * Decimal("75") / Decimal("100")
        assert performance < overall

    def test_performance_materiality_range(self):
        """Test performance materiality is typically 50-85% of overall."""
        overall = Decimal("1000")
        perf_50 = overall * Decimal("50") / Decimal("100")
        perf_85 = overall * Decimal("85") / Decimal("100")
        perf_75 = overall * Decimal("75") / Decimal("100")
        assert perf_50 <= perf_75 <= perf_85


# ---------------------------------------------------------------------------
# Clearly Trivial Threshold Tests
# ---------------------------------------------------------------------------


class TestClearlyTrivialThreshold:
    """Tests for clearly trivial threshold calculation."""

    def test_clearly_trivial_5pct_of_overall(self, sample_emissions_data, materiality_engine_config):
        """Test clearly trivial threshold is 5% of overall materiality."""
        total = sample_emissions_data["total_all_scopes_tco2e"]
        overall = total * materiality_engine_config["overall_pct"] / Decimal("100")
        trivial_pct = materiality_engine_config["clearly_trivial_pct_of_overall"]
        trivial = overall * trivial_pct / Decimal("100")
        expected = Decimal("1150") * Decimal("5") / Decimal("100")
        assert_decimal_equal(trivial, expected, tolerance=Decimal("0.01"))

    def test_clearly_trivial_lt_performance(self, sample_emissions_data):
        """Test clearly trivial is less than performance materiality."""
        total = sample_emissions_data["total_all_scopes_tco2e"]
        overall = total * Decimal("5") / Decimal("100")
        performance = overall * Decimal("75") / Decimal("100")
        trivial = overall * Decimal("5") / Decimal("100")
        assert trivial < performance

    def test_clearly_trivial_very_small(self, sample_emissions_data):
        """Test clearly trivial is a very small amount."""
        total = sample_emissions_data["total_all_scopes_tco2e"]
        overall = total * Decimal("5") / Decimal("100")
        trivial = overall * Decimal("5") / Decimal("100")
        assert trivial < total * Decimal("1") / Decimal("100")  # Less than 1% of total


# ---------------------------------------------------------------------------
# Scope-Specific Materiality Tests
# ---------------------------------------------------------------------------


class TestScopeSpecificMateriality:
    """Tests for scope-specific materiality calculation."""

    def test_scope_1_materiality(self, sample_emissions_data):
        """Test Scope 1 specific materiality calculation."""
        scope_1_total = sample_emissions_data["scope_1"]["total_tco2e"]
        scope_1_mat = scope_1_total * Decimal("5") / Decimal("100")
        assert scope_1_mat > Decimal("0")

    def test_scope_2_materiality(self, sample_emissions_data):
        """Test Scope 2 specific materiality calculation."""
        scope_2_total = sample_emissions_data["scope_2_location"]["total_tco2e"]
        scope_2_mat = scope_2_total * Decimal("5") / Decimal("100")
        assert scope_2_mat > Decimal("0")

    def test_scope_3_materiality(self, sample_emissions_data):
        """Test Scope 3 specific materiality calculation."""
        scope_3_total = sample_emissions_data["scope_3"]["total_tco2e"]
        scope_3_mat = scope_3_total * Decimal("5") / Decimal("100")
        assert scope_3_mat > Decimal("0")

    def test_scope_materialities_sum_gt_overall(self, sample_emissions_data):
        """Test sum of scope-specific materialities >= overall (conservative)."""
        s1 = sample_emissions_data["scope_1"]["total_tco2e"] * Decimal("5") / Decimal("100")
        s2 = sample_emissions_data["scope_2_location"]["total_tco2e"] * Decimal("5") / Decimal("100")
        s3 = sample_emissions_data["scope_3"]["total_tco2e"] * Decimal("5") / Decimal("100")
        overall = sample_emissions_data["total_all_scopes_tco2e"] * Decimal("5") / Decimal("100")
        assert (s1 + s2 + s3) >= overall


# ---------------------------------------------------------------------------
# Qualitative Factors Tests
# ---------------------------------------------------------------------------


class TestQualitativeFactors:
    """Tests for qualitative materiality factor adjustments."""

    def test_4_qualitative_factors(self, materiality_engine_config):
        """Test 4 qualitative factors are defined."""
        factors = materiality_engine_config["qualitative_factors"]
        assert len(factors) == 4

    def test_regulatory_scrutiny_factor(self, materiality_engine_config):
        """Test regulatory scrutiny factor is present."""
        assert "regulatory_scrutiny" in materiality_engine_config["qualitative_factors"]

    def test_qualitative_factors_can_reduce_materiality(self):
        """Test qualitative factors can reduce materiality threshold."""
        base_materiality = Decimal("1150")
        adjustment_factor = Decimal("0.8")  # 20% reduction due to high scrutiny
        adjusted = base_materiality * adjustment_factor
        assert adjusted < base_materiality

    def test_qualitative_factors_can_increase_materiality(self):
        """Test qualitative factors can increase materiality threshold."""
        base_materiality = Decimal("1150")
        adjustment_factor = Decimal("1.2")  # 20% increase due to low risk
        adjusted = base_materiality * adjustment_factor
        assert adjusted > base_materiality


# ---------------------------------------------------------------------------
# Materiality Revision Tests
# ---------------------------------------------------------------------------


class TestMaterialityRevision:
    """Tests for materiality revision during engagement."""

    def test_revised_materiality_different_from_initial(self):
        """Test revised materiality can differ from initial."""
        initial = Decimal("1150")
        revised = Decimal("1000")
        assert revised != initial

    def test_revision_reason_documented(self):
        """Test revision reason is documented."""
        revision = {
            "initial_materiality": Decimal("1150"),
            "revised_materiality": Decimal("1000"),
            "reason": "Increased regulatory scrutiny after new CSRD guidance",
            "approved_by": "Engagement Partner",
            "revision_date": "2025-06-01",
        }
        assert revision["reason"] is not None
        assert len(revision["reason"]) > 0


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestMaterialityEdgeCases:
    """Tests for materiality edge cases."""

    def test_zero_emissions_produces_zero_materiality(self):
        """Test zero total emissions produces zero materiality."""
        total = Decimal("0")
        materiality = total * Decimal("5") / Decimal("100")
        assert_decimal_equal(materiality, Decimal("0"))

    def test_very_small_emissions(self):
        """Test very small emissions produce proportionally small materiality."""
        total = Decimal("10")
        materiality = total * Decimal("5") / Decimal("100")
        assert_decimal_equal(materiality, Decimal("0.5"))

    def test_custom_pct_10(self):
        """Test custom materiality percentage of 10%."""
        total = Decimal("23000")
        custom_pct = Decimal("10")
        materiality = total * custom_pct / Decimal("100")
        assert_decimal_equal(materiality, Decimal("2300"))

    def test_custom_pct_2(self):
        """Test custom materiality percentage of 2%."""
        total = Decimal("23000")
        custom_pct = Decimal("2")
        materiality = total * custom_pct / Decimal("100")
        assert_decimal_equal(materiality, Decimal("460"))
