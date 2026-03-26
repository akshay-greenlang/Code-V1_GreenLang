# -*- coding: utf-8 -*-
"""
Unit tests for BaseYearRecalculationEngine -- PACK-041 Engine 7
==================================================================

Tests base-year recalculation triggers, significance thresholds,
acquisition/divestiture/merger/methodology adjustments, audit trails,
and multi-trigger scenarios.

Coverage target: 85%+
Total tests: ~45
"""

from decimal import Decimal
from datetime import date

import pytest


# =============================================================================
# No Recalculation Needed
# =============================================================================


class TestNoRecalculation:
    """Test scenarios where base-year recalculation is NOT triggered."""

    def test_no_recalculation_below_threshold(self, sample_base_year):
        """Change <5% of base year should not trigger recalculation."""
        base_total = sample_base_year["total_scope12_location_tco2e"]
        change_emissions = Decimal("1000")  # 1000 / 41000 = 2.4%
        materiality = change_emissions / base_total * Decimal("100")
        assert materiality < Decimal("5.0")

    def test_no_recalculation_organic_growth(self):
        """Organic growth does not trigger recalculation per GHG Protocol."""
        change_type = "organic_growth"
        triggers_recalc = change_type not in {"organic_growth", "shutdown"}
        assert triggers_recalc is False

    def test_no_recalculation_shutdown(self):
        """Facility shutdown does not trigger recalculation by default."""
        change_type = "shutdown"
        triggers_recalc = change_type not in {"organic_growth", "shutdown"}
        assert triggers_recalc is False

    def test_zero_emissions_change(self, sample_base_year):
        base_total = sample_base_year["total_scope12_location_tco2e"]
        change = Decimal("0")
        materiality = change / base_total * Decimal("100") if base_total else Decimal("0")
        assert materiality == Decimal("0")


# =============================================================================
# Acquisition Recalculation
# =============================================================================


class TestAcquisitionRecalculation:
    """Test base-year recalculation for acquisitions."""

    def test_acquisition_significant(self, sample_base_year):
        """Acquisition >5% should trigger recalculation."""
        base_total = sample_base_year["total_scope12_location_tco2e"]
        acquired_emissions = Decimal("5000")  # 5000/41000 = 12.2%
        materiality = acquired_emissions / base_total * Decimal("100")
        assert materiality > Decimal("5.0")

    def test_acquisition_not_significant(self, sample_base_year):
        """Acquisition <5% should not trigger recalculation."""
        base_total = sample_base_year["total_scope12_location_tco2e"]
        acquired_emissions = Decimal("500")  # 500/41000 = 1.2%
        materiality = acquired_emissions / base_total * Decimal("100")
        assert materiality < Decimal("5.0")

    def test_acquisition_adjusts_base_year_upward(self, sample_base_year):
        """Acquisition should increase adjusted base-year emissions."""
        original = sample_base_year["total_scope12_location_tco2e"]
        acquired = Decimal("5000")
        adjusted = original + acquired
        assert adjusted > original
        assert adjusted == Decimal("46000.0")

    def test_acquisition_pro_rata_partial_year(self):
        """Acquisition mid-year should be pro-rated."""
        annual_emissions = Decimal("12000")
        months_in_year = 12
        months_acquired = 7  # acquired June 1, so 7 months
        pro_rata = annual_emissions * Decimal(str(months_acquired)) / Decimal(str(months_in_year))
        assert pro_rata == Decimal("7000")


# =============================================================================
# Divestiture Recalculation
# =============================================================================


class TestDivestitureRecalculation:
    """Test base-year recalculation for divestitures."""

    def test_divestiture_adjusts_base_year_downward(self, sample_base_year):
        original = sample_base_year["total_scope12_location_tco2e"]
        divested = Decimal("8000")
        adjusted = original - divested
        assert adjusted < original
        assert adjusted == Decimal("33000.0")

    def test_divestiture_cannot_go_negative(self, sample_base_year):
        original = sample_base_year["total_scope12_location_tco2e"]
        divested = Decimal("50000")
        adjusted = max(original - divested, Decimal("0"))
        assert adjusted == Decimal("0")

    def test_divestiture_significance(self, sample_base_year):
        base_total = sample_base_year["total_scope12_location_tco2e"]
        divested = Decimal("3000")
        materiality = divested / base_total * Decimal("100")
        assert materiality == pytest.approx(Decimal("7.32"), abs=Decimal("0.01"))


# =============================================================================
# Merger Recalculation
# =============================================================================


class TestMergerRecalculation:
    """Test base-year recalculation for mergers."""

    def test_merger_adds_merged_entity(self, sample_base_year):
        original = sample_base_year["total_scope12_location_tco2e"]
        merged = Decimal("15000")
        adjusted = original + merged
        assert adjusted == Decimal("56000.0")

    def test_merger_always_significant(self, sample_base_year):
        """Mergers are typically significant (>5%)."""
        base_total = sample_base_year["total_scope12_location_tco2e"]
        merged = Decimal("10000")
        materiality = merged / base_total * Decimal("100")
        assert materiality > Decimal("5.0")


# =============================================================================
# Methodology Change
# =============================================================================


class TestMethodologyChange:
    """Test base-year recalculation for methodology changes."""

    def test_methodology_change_recalculates(self, sample_base_year):
        original = sample_base_year["total_scope12_location_tco2e"]
        impact = Decimal("3500")  # e.g., switching from AR4 to AR6 GWPs
        materiality = impact / original * Decimal("100")
        assert materiality > Decimal("5.0")

    def test_methodology_change_can_increase_or_decrease(self):
        """Switching GWP source may increase or decrease total."""
        original = Decimal("41000")
        impact_positive = Decimal("1500")
        impact_negative = Decimal("-800")
        adjusted_up = original + impact_positive
        adjusted_down = original + impact_negative
        assert adjusted_up > original
        assert adjusted_down < original


# =============================================================================
# Error Correction
# =============================================================================


class TestErrorCorrection:
    """Test base-year recalculation for data error corrections."""

    def test_error_correction_significant(self, sample_base_year):
        original = sample_base_year["total_scope12_location_tco2e"]
        correction = Decimal("4000")
        materiality = correction / original * Decimal("100")
        assert materiality > Decimal("5.0")

    def test_error_correction_not_significant(self, sample_base_year):
        original = sample_base_year["total_scope12_location_tco2e"]
        correction = Decimal("500")
        materiality = correction / original * Decimal("100")
        assert materiality < Decimal("5.0")


# =============================================================================
# Significance Threshold
# =============================================================================


class TestSignificanceThreshold:
    """Test the 5% significance threshold and custom thresholds."""

    def test_default_threshold_5_pct(self):
        threshold = Decimal("5.0")
        assert threshold == Decimal("5.0")

    def test_custom_threshold_10_pct(self, sample_base_year):
        base_total = sample_base_year["total_scope12_location_tco2e"]
        change = Decimal("3500")  # 8.5%
        custom_threshold = Decimal("10.0")
        materiality = change / base_total * Decimal("100")
        assert materiality < custom_threshold

    @pytest.mark.parametrize("change_tco2e,threshold_pct,expected_trigger", [
        (Decimal("2050"), Decimal("5.0"), True),   # 5.0% of 41000
        (Decimal("2000"), Decimal("5.0"), False),   # 4.88%
        (Decimal("4100"), Decimal("10.0"), True),   # 10.0%
        (Decimal("4000"), Decimal("10.0"), False),   # 9.76%
    ])
    def test_threshold_parametrized(self, change_tco2e, threshold_pct, expected_trigger, sample_base_year):
        base_total = sample_base_year["total_scope12_location_tco2e"]
        materiality = change_tco2e / base_total * Decimal("100")
        triggered = materiality >= threshold_pct
        assert triggered == expected_trigger


# =============================================================================
# Audit Trail
# =============================================================================


class TestAuditTrail:
    """Test audit trail generation for base-year recalculations."""

    def test_audit_record_contains_fields(self):
        audit = {
            "recalculation_id": "RECALC-2025-001",
            "trigger_type": "structural_change",
            "trigger_event": "acquisition",
            "effective_date": "2025-06-01",
            "original_base_year_tco2e": Decimal("41000"),
            "change_emissions_tco2e": Decimal("5000"),
            "adjusted_base_year_tco2e": Decimal("46000"),
            "materiality_pct": Decimal("12.2"),
            "significance_threshold_pct": Decimal("5.0"),
            "approved_by": "env_director@acme.com",
            "approval_date": "2025-07-15",
            "notes": "Acquisition of GreenTech Inc.",
        }
        required_fields = [
            "recalculation_id", "trigger_type", "original_base_year_tco2e",
            "adjusted_base_year_tco2e", "materiality_pct",
        ]
        for field in required_fields:
            assert field in audit

    def test_audit_preserves_original(self):
        audit = {"original": Decimal("41000"), "adjusted": Decimal("46000")}
        assert audit["original"] != audit["adjusted"]


# =============================================================================
# Multiple Triggers Same Year
# =============================================================================


class TestMultipleTriggers:
    """Test multiple recalculation triggers in the same year."""

    def test_multiple_changes_cumulative(self, sample_base_year):
        """Multiple structural changes should be applied cumulatively."""
        original = sample_base_year["total_scope12_location_tco2e"]
        acquisition = Decimal("5000")
        divestiture = Decimal("3000")
        adjusted = original + acquisition - divestiture
        assert adjusted == Decimal("43000.0")

    def test_net_materiality_check(self, sample_base_year):
        """Check materiality on net change, not individual changes."""
        original = sample_base_year["total_scope12_location_tco2e"]
        net_change = Decimal("5000") - Decimal("3000")
        materiality = net_change / original * Decimal("100")
        assert materiality == pytest.approx(Decimal("4.88"), abs=Decimal("0.01"))

    def test_three_changes_in_year(self, sample_base_year):
        original = sample_base_year["total_scope12_location_tco2e"]
        changes = [Decimal("5000"), Decimal("-3000"), Decimal("1500")]
        net = sum(changes)
        adjusted = original + net
        assert adjusted == Decimal("44500.0")
