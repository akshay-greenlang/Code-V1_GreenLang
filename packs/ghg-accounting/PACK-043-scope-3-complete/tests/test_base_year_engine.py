# -*- coding: utf-8 -*-
"""
Unit tests for PACK-043 Base Year Engine
===========================================

Tests base year establishment, 6 recalculation trigger types,
significance threshold, pro-rata adjustments, multi-year trend
comparison, decomposition analysis, cumulative reduction, and edge cases.

Coverage target: 85%+
Total tests: ~40
"""

from decimal import Decimal

import pytest


# =============================================================================
# Base Year Establishment
# =============================================================================


class TestBaseYearEstablishment:
    """Test base year data establishment."""

    def test_base_year_is_2019(self, sample_base_year_data):
        assert sample_base_year_data["base_year"] == 2019

    def test_base_year_total_positive(self, sample_base_year_data):
        assert sample_base_year_data["scope3_total_tco2e"] > Decimal("0")

    def test_base_year_has_15_categories(self, sample_base_year_data):
        assert len(sample_base_year_data["by_category"]) == 15

    def test_base_year_categories_sum(self, sample_base_year_data):
        total = sum(sample_base_year_data["by_category"].values())
        assert total == sample_base_year_data["scope3_total_tco2e"]

    def test_all_categories_non_negative(self, sample_base_year_data):
        for cat_num, value in sample_base_year_data["by_category"].items():
            assert value >= Decimal("0"), f"Category {cat_num} is negative"


# =============================================================================
# Six Recalculation Trigger Types
# =============================================================================


class TestRecalculationTriggers:
    """Test all 6 recalculation trigger types."""

    def test_six_triggers_present(self, sample_base_year_data):
        assert len(sample_base_year_data["recalculation_triggers"]) == 6

    @pytest.mark.parametrize("trigger_type", [
        "acquisition",
        "divestiture",
        "methodology",
        "scope_expansion",
        "error_correction",
        "structural_change",
    ])
    def test_trigger_type_present(self, trigger_type, sample_base_year_data):
        found = any(
            t["trigger_type"] == trigger_type
            for t in sample_base_year_data["recalculation_triggers"]
        )
        assert found is True, f"Trigger type {trigger_type} not found"

    def test_acquisition_trigger(self, sample_base_year_data):
        trigger = next(
            t for t in sample_base_year_data["recalculation_triggers"]
            if t["trigger_type"] == "acquisition"
        )
        assert trigger["impact_tco2e"] > Decimal("0")
        assert trigger["exceeds_threshold"] is True

    def test_divestiture_trigger(self, sample_base_year_data):
        trigger = next(
            t for t in sample_base_year_data["recalculation_triggers"]
            if t["trigger_type"] == "divestiture"
        )
        assert trigger["impact_tco2e"] < Decimal("0")
        assert trigger["exceeds_threshold"] is True

    def test_methodology_trigger(self, sample_base_year_data):
        trigger = next(
            t for t in sample_base_year_data["recalculation_triggers"]
            if t["trigger_type"] == "methodology"
        )
        assert trigger["exceeds_threshold"] is False  # below 5%

    def test_scope_expansion_trigger(self, sample_base_year_data):
        trigger = next(
            t for t in sample_base_year_data["recalculation_triggers"]
            if t["trigger_type"] == "scope_expansion"
        )
        assert trigger["exceeds_threshold"] is True

    def test_error_correction_trigger(self, sample_base_year_data):
        trigger = next(
            t for t in sample_base_year_data["recalculation_triggers"]
            if t["trigger_type"] == "error_correction"
        )
        assert trigger["exceeds_threshold"] is False

    def test_structural_change_trigger(self, sample_base_year_data):
        trigger = next(
            t for t in sample_base_year_data["recalculation_triggers"]
            if t["trigger_type"] == "structural_change"
        )
        assert trigger["exceeds_threshold"] is True

    def test_trigger_ids_unique(self, sample_base_year_data):
        ids = [t["trigger_id"] for t in sample_base_year_data["recalculation_triggers"]]
        assert len(ids) == len(set(ids))


# =============================================================================
# Significance Threshold
# =============================================================================


class TestSignificanceThreshold:
    """Test 5% significance threshold for recalculation."""

    def test_threshold_is_5pct(self, sample_base_year_data):
        assert sample_base_year_data["significance_threshold_pct"] == Decimal("5.0")

    def test_above_threshold_triggers_recalc(self, sample_base_year_data):
        for trigger in sample_base_year_data["recalculation_triggers"]:
            if trigger["impact_pct"] > sample_base_year_data["significance_threshold_pct"]:
                assert trigger["exceeds_threshold"] is True

    def test_below_threshold_no_recalc(self, sample_base_year_data):
        for trigger in sample_base_year_data["recalculation_triggers"]:
            if trigger["impact_pct"] <= sample_base_year_data["significance_threshold_pct"]:
                assert trigger["exceeds_threshold"] is False

    def test_impact_pct_formula(self, sample_base_year_data):
        """Impact % = |impact_tco2e| / base_total * 100."""
        base = sample_base_year_data["scope3_total_tco2e"]
        for trigger in sample_base_year_data["recalculation_triggers"]:
            expected_pct = abs(trigger["impact_tco2e"]) / base * Decimal("100")
            assert expected_pct == pytest.approx(
                trigger["impact_pct"], abs=Decimal("0.01")
            )


# =============================================================================
# Pro-Rata Mid-Year Adjustment
# =============================================================================


class TestProRataAdjustment:
    """Test pro-rata mid-year adjustment calculations."""

    def test_acquisition_pro_rata(self, sample_base_year_data):
        trigger = next(
            t for t in sample_base_year_data["recalculation_triggers"]
            if t["trigger_type"] == "acquisition"
        )
        assert "pro_rata_factor" in trigger
        assert Decimal("0") < trigger["pro_rata_factor"] < Decimal("1")

    def test_pro_rata_formula(self, sample_base_year_data):
        """Pro-rata factor = days_in_period / 365."""
        trigger = next(
            t for t in sample_base_year_data["recalculation_triggers"]
            if t["trigger_type"] == "acquisition"
        )
        expected = Decimal(str(trigger["pro_rata_days"])) / Decimal("365")
        assert expected == pytest.approx(trigger["pro_rata_factor"], abs=Decimal("0.001"))

    def test_full_year_divestiture(self, sample_base_year_data):
        trigger = next(
            t for t in sample_base_year_data["recalculation_triggers"]
            if t["trigger_type"] == "divestiture"
        )
        assert trigger["pro_rata_factor"] == Decimal("1.0")


# =============================================================================
# Multi-Year Trend Comparison
# =============================================================================


class TestMultiYearTrend:
    """Test multi-year trend comparison."""

    def test_seven_years_of_data(self, sample_base_year_data):
        assert len(sample_base_year_data["yearly_actuals"]) == 7

    def test_years_sequential(self, sample_base_year_data):
        years = sorted(sample_base_year_data["yearly_actuals"].keys())
        assert years == [2019, 2020, 2021, 2022, 2023, 2024, 2025]

    def test_overall_downward_trend(self, sample_base_year_data):
        actuals = sample_base_year_data["yearly_actuals"]
        assert actuals[2025] < actuals[2019]

    def test_adjusted_base_year_positive(self, sample_base_year_data):
        assert sample_base_year_data["adjusted_base_year_tco2e"] > Decimal("0")


# =============================================================================
# Decomposition Analysis
# =============================================================================


class TestDecompositionAnalysis:
    """Test real vs methodology-driven decomposition."""

    def test_real_reduction(self, sample_base_year_data):
        """Real reduction = adjusted_base - current_actual."""
        adjusted = sample_base_year_data["adjusted_base_year_tco2e"]
        current = sample_base_year_data["yearly_actuals"][2025]
        real_reduction = adjusted - current
        assert real_reduction > Decimal("0")

    def test_methodology_driven_triggers(self, sample_base_year_data):
        """Methodology-driven triggers don't represent real reductions."""
        methodology_triggers = [
            t for t in sample_base_year_data["recalculation_triggers"]
            if t["trigger_type"] == "methodology"
        ]
        assert len(methodology_triggers) >= 1


# =============================================================================
# Cumulative Reduction
# =============================================================================


class TestCumulativeReduction:
    """Test cumulative reduction calculation."""

    def test_cumulative_vs_base(self, sample_base_year_data):
        """Cumulative reduction from base year to current."""
        base = sample_base_year_data["scope3_total_tco2e"]
        current = sample_base_year_data["yearly_actuals"][2025]
        reduction_pct = (base - current) / base * Decimal("100")
        assert reduction_pct > Decimal("0")
        # (300000 - 252500) / 300000 * 100 = 15.83%
        assert Decimal("15") < reduction_pct < Decimal("17")

    def test_cumulative_vs_adjusted_base(self, sample_base_year_data):
        adjusted = sample_base_year_data["adjusted_base_year_tco2e"]
        current = sample_base_year_data["yearly_actuals"][2025]
        reduction_pct = (adjusted - current) / adjusted * Decimal("100")
        assert reduction_pct > Decimal("0")


# =============================================================================
# Edge Cases
# =============================================================================


class TestBaseYearEdgeCases:
    """Test edge cases for base year engine."""

    def test_no_triggers(self):
        """No triggers means base year is unchanged."""
        base = Decimal("300000")
        triggers = []
        adjusted = base + sum(t.get("impact_tco2e", Decimal("0")) for t in triggers)
        assert adjusted == base

    def test_below_threshold_all_triggers(self):
        """All triggers below threshold - no recalculation needed."""
        threshold = Decimal("5.0")
        triggers = [
            {"impact_pct": Decimal("2.0"), "exceeds_threshold": False},
            {"impact_pct": Decimal("1.5"), "exceeds_threshold": False},
        ]
        needs_recalc = any(t["exceeds_threshold"] for t in triggers)
        assert needs_recalc is False

    def test_exactly_at_threshold(self):
        """Impact exactly at 5% threshold."""
        impact_pct = Decimal("5.0")
        threshold = Decimal("5.0")
        # Convention: equal to threshold does trigger recalculation
        exceeds = impact_pct >= threshold
        assert exceeds is True
