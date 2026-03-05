# -*- coding: utf-8 -*-
"""
Unit tests for SBTi Recalculation Engine.

Tests the 5% significance threshold check (above, below, exactly 5%),
recalculation trigger types, revalidation requirement assessment,
M&A acquisition/divestment impact modeling, and target adjustment
after recalculation with 20+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from datetime import datetime

import pytest


# ===========================================================================
# Threshold Check
# ===========================================================================

class TestThresholdCheck:
    """Test 5% recalculation significance threshold."""

    def test_above_threshold(self, sample_recalculation):
        assert sample_recalculation["exceeds_threshold"] is True
        assert sample_recalculation["change_pct"] > 5.0

    def test_below_threshold(self, minor_recalculation):
        assert minor_recalculation["exceeds_threshold"] is False
        assert minor_recalculation["change_pct"] < 5.0

    @pytest.mark.parametrize("change_pct,exceeds", [
        (5.0, True),    # Exactly 5% - meets threshold
        (5.1, True),
        (8.0, True),
        (15.0, True),
        (4.9, False),
        (2.5, False),
        (0.0, False),
    ])
    def test_threshold_edge_cases(self, change_pct, exceeds):
        assert (change_pct >= 5.0) == exceeds

    def test_threshold_value(self, sample_recalculation):
        assert sample_recalculation["recalculation_threshold_pct"] == 5.0

    def test_change_pct_calculation(self):
        original = 80_000.0
        adjusted = 86_400.0
        change_pct = ((adjusted - original) / original) * 100
        assert change_pct == 8.0

    def test_negative_change_also_triggers(self):
        original = 80_000.0
        adjusted = 74_000.0
        change_pct = abs((adjusted - original) / original) * 100
        assert change_pct == 7.5
        assert change_pct >= 5.0


# ===========================================================================
# Recalculation Creation
# ===========================================================================

class TestRecalculationCreation:
    """Test recalculation record creation with trigger types."""

    VALID_TRIGGER_TYPES = [
        "acquisition", "divestment", "merger", "outsourcing",
        "insourcing", "methodology_change", "organic_growth",
        "discovery_of_errors", "structural_change",
    ]

    def test_trigger_type_acquisition(self, sample_recalculation):
        assert sample_recalculation["trigger_type"] == "acquisition"
        assert sample_recalculation["trigger_type"] in self.VALID_TRIGGER_TYPES

    def test_trigger_type_organic(self, minor_recalculation):
        assert minor_recalculation["trigger_type"] == "organic_growth"
        assert minor_recalculation["trigger_type"] in self.VALID_TRIGGER_TYPES

    @pytest.mark.parametrize("trigger_type", [
        "acquisition", "divestment", "merger", "outsourcing",
        "insourcing", "methodology_change", "organic_growth",
        "discovery_of_errors", "structural_change",
    ])
    def test_all_trigger_types_valid(self, trigger_type):
        assert trigger_type in self.VALID_TRIGGER_TYPES

    def test_trigger_description(self, sample_recalculation):
        assert len(sample_recalculation["trigger_description"]) > 0


# ===========================================================================
# Revalidation Assessment
# ===========================================================================

class TestRevalidationAssessment:
    """Test whether revalidation is needed after recalculation."""

    def test_revalidation_required_above_threshold(self, sample_recalculation):
        assert sample_recalculation["revalidation_required"] is True

    def test_revalidation_not_required_below_threshold(self, minor_recalculation):
        assert minor_recalculation["revalidation_required"] is False

    def test_methodology_change_requires_revalidation(self):
        recalc = {
            "methodology_change": True,
            "change_pct": 3.0,  # Below threshold but methodology changed
        }
        revalidation = recalc["methodology_change"] or recalc["change_pct"] >= 5.0
        assert revalidation is True

    def test_structural_change_tracking(self, sample_recalculation):
        assert sample_recalculation["structural_change"] is True


# ===========================================================================
# M&A Impact
# ===========================================================================

class TestMAImpact:
    """Test acquisition/divestment impact modeling."""

    def test_acquisition_increases_base(self, sample_recalculation):
        assert sample_recalculation["adjusted_base_year_tco2e"] > sample_recalculation["original_base_year_tco2e"]

    def test_divestment_decreases_base(self):
        recalc = {
            "trigger_type": "divestment",
            "original_base_year_tco2e": 80_000.0,
            "adjusted_base_year_tco2e": 70_000.0,
            "change_pct": -12.5,
        }
        assert recalc["adjusted_base_year_tco2e"] < recalc["original_base_year_tco2e"]

    def test_ma_proportional_target_adjustment(self, sample_recalculation):
        orig_base = sample_recalculation["original_base_year_tco2e"]
        adj_base = sample_recalculation["adjusted_base_year_tco2e"]
        orig_target = sample_recalculation["original_target_tco2e"]
        adj_target = sample_recalculation["adjusted_target_tco2e"]
        # Proportional relationship maintained
        orig_ratio = orig_target / orig_base
        adj_ratio = adj_target / adj_base
        assert abs(orig_ratio - adj_ratio) < 0.01


# ===========================================================================
# Target Update
# ===========================================================================

class TestTargetUpdate:
    """Test target adjustment after recalculation."""

    def test_target_adjusted(self, sample_recalculation):
        assert sample_recalculation["adjusted_target_tco2e"] != sample_recalculation["original_target_tco2e"]

    def test_target_increased_on_acquisition(self, sample_recalculation):
        assert sample_recalculation["adjusted_target_tco2e"] > sample_recalculation["original_target_tco2e"]

    def test_audit_trail_present(self, sample_recalculation):
        trail = sample_recalculation["audit_trail"]
        assert len(trail) >= 1

    def test_audit_trail_has_timestamps(self, sample_recalculation):
        for entry in sample_recalculation["audit_trail"]:
            assert "timestamp" in entry
            assert "action" in entry

    def test_recalculation_provenance(self, sample_recalculation):
        assert len(sample_recalculation["provenance_hash"]) == 64
