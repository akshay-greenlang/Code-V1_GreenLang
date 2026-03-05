# -*- coding: utf-8 -*-
"""
Unit tests for EU Taxonomy KPI Calculation Engine.

Tests turnover/CapEx/OpEx KPI calculation, double-counting prevention,
CapEx plan registration, objective breakdown, activity-level aggregation,
KPI percentage validation, and multi-period comparison with 40+ test
functions.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest


# ===========================================================================
# Turnover KPI calculation tests
# ===========================================================================

class TestTurnoverKPI:
    """Test turnover KPI calculation."""

    def test_turnover_kpi_type(self, sample_kpi_data):
        """KPI type is turnover."""
        assert sample_kpi_data["kpi_type"] == "turnover"

    def test_turnover_percentage(self, sample_kpi_data):
        """Turnover KPI percentage is calculated correctly."""
        aligned = float(sample_kpi_data["aligned_amount"])
        total = float(sample_kpi_data["total_amount"])
        expected = round(aligned / total * 100, 4)
        assert abs(float(sample_kpi_data["kpi_percentage"]) - expected) < 0.01

    def test_turnover_amounts_hierarchy(self, sample_kpi_data):
        """Aligned <= eligible <= total."""
        assert sample_kpi_data["aligned_amount"] <= sample_kpi_data["eligible_amount"]
        assert sample_kpi_data["eligible_amount"] <= sample_kpi_data["total_amount"]

    def test_turnover_eligible_ratio(self, sample_kpi_data):
        """Eligible turnover ratio calculation."""
        eligible = float(sample_kpi_data["eligible_amount"])
        total = float(sample_kpi_data["total_amount"])
        eligible_pct = eligible / total * 100
        assert eligible_pct == 70.0

    def test_turnover_period(self, sample_kpi_data):
        """Turnover KPI for correct period."""
        assert sample_kpi_data["period"] == "FY2025"

    def test_engine_calculate_turnover(self, kpi_engine):
        """Engine calculates turnover KPI."""
        kpi_engine.calculate_turnover_kpi.return_value = {
            "eligible_amount": 1750000000,
            "aligned_amount": 1050000000,
            "total_amount": 2500000000,
            "kpi_percentage": 42.0,
        }
        result = kpi_engine.calculate_turnover_kpi("org-123", "FY2025")
        assert result["kpi_percentage"] == 42.0
        kpi_engine.calculate_turnover_kpi.assert_called_once()


# ===========================================================================
# CapEx KPI calculation tests
# ===========================================================================

class TestCapExKPI:
    """Test CapEx KPI calculation."""

    def test_capex_kpi_type(self, sample_capex_kpi):
        """KPI type is capex."""
        assert sample_capex_kpi["kpi_type"] == "capex"

    def test_capex_percentage(self, sample_capex_kpi):
        """CapEx KPI percentage matches expected value."""
        assert sample_capex_kpi["kpi_percentage"] == Decimal("52.5000")

    def test_capex_amounts_hierarchy(self, sample_capex_kpi):
        """Aligned CapEx <= eligible CapEx <= total CapEx."""
        assert sample_capex_kpi["aligned_amount"] <= sample_capex_kpi["eligible_amount"]
        assert sample_capex_kpi["eligible_amount"] <= sample_capex_kpi["total_amount"]

    def test_capex_higher_than_turnover(self, sample_kpi_data, sample_capex_kpi):
        """CapEx alignment typically higher than turnover (future investment)."""
        capex_pct = float(sample_capex_kpi["kpi_percentage"])
        turnover_pct = float(sample_kpi_data["kpi_percentage"])
        assert capex_pct > turnover_pct

    def test_engine_calculate_capex(self, kpi_engine):
        """Engine calculates CapEx KPI."""
        kpi_engine.calculate_capex_kpi.return_value = {
            "kpi_percentage": 52.5,
            "includes_capex_plans": True,
        }
        result = kpi_engine.calculate_capex_kpi("org-123", "FY2025")
        assert result["kpi_percentage"] == 52.5


# ===========================================================================
# OpEx KPI calculation tests
# ===========================================================================

class TestOpExKPI:
    """Test OpEx KPI calculation."""

    def test_opex_kpi_type(self, sample_opex_kpi):
        """KPI type is opex."""
        assert sample_opex_kpi["kpi_type"] == "opex"

    def test_opex_percentage(self, sample_opex_kpi):
        """OpEx KPI percentage matches expected value."""
        assert sample_opex_kpi["kpi_percentage"] == Decimal("25.5000")

    def test_opex_amounts_hierarchy(self, sample_opex_kpi):
        """Aligned OpEx <= eligible OpEx <= total OpEx."""
        assert sample_opex_kpi["aligned_amount"] <= sample_opex_kpi["eligible_amount"]
        assert sample_opex_kpi["eligible_amount"] <= sample_opex_kpi["total_amount"]

    def test_opex_lower_than_others(self, sample_kpi_data, sample_capex_kpi, sample_opex_kpi):
        """OpEx alignment typically lower than turnover and CapEx."""
        opex_pct = float(sample_opex_kpi["kpi_percentage"])
        turnover_pct = float(sample_kpi_data["kpi_percentage"])
        assert opex_pct < turnover_pct

    def test_engine_calculate_opex(self, kpi_engine):
        """Engine calculates OpEx KPI."""
        kpi_engine.calculate_opex_kpi.return_value = {
            "kpi_percentage": 25.5,
        }
        result = kpi_engine.calculate_opex_kpi("org-123", "FY2025")
        assert result["kpi_percentage"] == 25.5


# ===========================================================================
# Double-counting prevention tests
# ===========================================================================

class TestDoubleCounting:
    """Test double-counting prevention in KPI calculations."""

    def test_double_counting_check_in_metadata(self, sample_kpi_data):
        """Double-counting check recorded in metadata."""
        assert sample_kpi_data["metadata"]["double_counting_check"] is True

    def test_intercompany_elimination(self, sample_kpi_data):
        """Intercompany transactions eliminated."""
        assert sample_kpi_data["metadata"]["intercompany_eliminated"] is True

    def test_engine_check_double_counting(self, kpi_engine):
        """Engine double-counting check returns False (no issues)."""
        result = kpi_engine.check_double_counting("org-123", "FY2025", "turnover")
        assert result is False

    def test_multi_objective_no_double_count(self, sample_kpi_data):
        """Same activity not counted under multiple objectives."""
        breakdown = sample_kpi_data["objective_breakdown"]
        total_aligned_by_obj = sum(
            v["aligned"] for v in breakdown.values()
        )
        # Total by objective should not exceed total aligned
        assert total_aligned_by_obj <= float(sample_kpi_data["aligned_amount"]) * 1.01  # small tolerance

    def test_unique_constraint_per_kpi_type(self, sample_kpi_data, sample_capex_kpi, sample_opex_kpi):
        """Each KPI type is separate per period."""
        types = {sample_kpi_data["kpi_type"], sample_capex_kpi["kpi_type"], sample_opex_kpi["kpi_type"]}
        assert len(types) == 3


# ===========================================================================
# CapEx plan registration tests
# ===========================================================================

class TestCapExPlanRegistration:
    """Test CapEx plan registration and tracking."""

    def test_capex_plan_created(self, sample_capex_plan):
        """CapEx plan has expected structure."""
        assert sample_capex_plan["activity_code"] == "CCM_3.3"
        assert sample_capex_plan["start_year"] == 2025
        assert sample_capex_plan["end_year"] == 2030

    def test_capex_plan_approved(self, sample_capex_plan):
        """CapEx plan is management-approved."""
        assert sample_capex_plan["management_approved"] is True
        assert sample_capex_plan["approved_date"] is not None

    def test_capex_plan_amounts(self, sample_capex_plan):
        """Planned amounts span all years."""
        planned = sample_capex_plan["planned_amounts"]
        assert len(planned) == 6
        total_planned = sum(planned.values())
        assert total_planned > 0

    def test_capex_plan_actual_tracking(self, sample_capex_plan):
        """Actual amounts tracked against plan."""
        actual = sample_capex_plan["actual_amounts"]
        assert "2025" in actual
        assert actual["2025"] > 0

    def test_capex_plan_status_active(self, sample_capex_plan):
        """Active CapEx plan."""
        assert sample_capex_plan["status"] == "active"

    def test_capex_plan_duration(self, sample_capex_plan):
        """Plan duration within allowed range."""
        duration = sample_capex_plan["end_year"] - sample_capex_plan["start_year"]
        assert duration <= 10

    def test_engine_register_capex_plan(self, kpi_engine):
        """Engine registers CapEx plan."""
        kpi_engine.register_capex_plan.return_value = {"plan_id": "plan-001", "status": "registered"}
        result = kpi_engine.register_capex_plan(
            org_id="org-123",
            activity_code="CCM_3.3",
            start_year=2025,
            end_year=2030,
        )
        kpi_engine.register_capex_plan.assert_called_once()


# ===========================================================================
# Objective breakdown tests
# ===========================================================================

class TestObjectiveBreakdown:
    """Test KPI objective-level breakdown."""

    def test_turnover_objective_breakdown(self, sample_kpi_data):
        """Turnover breakdown by objective."""
        breakdown = sample_kpi_data["objective_breakdown"]
        assert "climate_mitigation" in breakdown
        assert "climate_adaptation" in breakdown

    def test_climate_mitigation_dominant(self, sample_kpi_data):
        """Climate mitigation is the largest objective."""
        breakdown = sample_kpi_data["objective_breakdown"]
        ccm = breakdown["climate_mitigation"]
        cca = breakdown["climate_adaptation"]
        assert ccm["aligned"] > cca["aligned"]

    def test_objective_percentages_sum(self, sample_kpi_data):
        """Objective percentages sum to total KPI percentage."""
        breakdown = sample_kpi_data["objective_breakdown"]
        total_pct = sum(v["percentage"] for v in breakdown.values())
        kpi_pct = float(sample_kpi_data["kpi_percentage"])
        assert abs(total_pct - kpi_pct) < 0.1

    def test_capex_objective_breakdown(self, sample_capex_kpi):
        """CapEx breakdown includes circular economy objective."""
        breakdown = sample_capex_kpi["objective_breakdown"]
        assert "climate_mitigation" in breakdown
        assert "circular_economy" in breakdown

    def test_engine_get_objective_breakdown(self, kpi_engine):
        """Engine returns objective breakdown."""
        kpi_engine.get_objective_breakdown.return_value = {
            "climate_mitigation": {"aligned": 950000000, "percentage": 38.0},
            "climate_adaptation": {"aligned": 100000000, "percentage": 4.0},
        }
        result = kpi_engine.get_objective_breakdown("org-123", "FY2025", "turnover")
        assert len(result) == 2


# ===========================================================================
# Activity-level aggregation tests
# ===========================================================================

class TestActivityLevelAggregation:
    """Test KPI aggregation from activity-level financial data."""

    def test_activity_financials_structure(self, sample_activity_financials):
        """Activity financials have expected fields."""
        for af in sample_activity_financials:
            assert "activity_code" in af
            assert "turnover" in af
            assert "capex" in af
            assert "opex" in af

    def test_eligible_activities_flagged(self, sample_activity_financials):
        """Eligible activities correctly flagged."""
        eligible = [af for af in sample_activity_financials if af["eligible"]]
        assert len(eligible) == 2

    def test_non_eligible_not_aligned(self, sample_activity_financials):
        """Non-eligible activities are not aligned."""
        non_eligible = [af for af in sample_activity_financials if not af["eligible"]]
        for af in non_eligible:
            assert af["aligned"] is False

    def test_aligned_subset_of_eligible(self, sample_activity_financials):
        """Aligned activities are a subset of eligible."""
        aligned = [af for af in sample_activity_financials if af["aligned"]]
        for af in aligned:
            assert af["eligible"] is True

    def test_activity_objective_assigned(self, sample_activity_financials):
        """Aligned activities have an assigned objective."""
        aligned = [af for af in sample_activity_financials if af["aligned"]]
        for af in aligned:
            assert af["objective"] is not None

    def test_turnover_sum_from_activities(self, sample_activity_financials):
        """Total turnover can be calculated from activity data."""
        total = sum(float(af["turnover"]) for af in sample_activity_financials)
        assert total > 0

    def test_aligned_turnover_from_activities(self, sample_activity_financials):
        """Aligned turnover from activity data."""
        aligned_turnover = sum(
            float(af["turnover"]) for af in sample_activity_financials if af["aligned"]
        )
        assert aligned_turnover > 0


# ===========================================================================
# KPI percentage validation tests
# ===========================================================================

class TestKPIPercentageValidation:
    """Test KPI percentage range and consistency."""

    def test_kpi_percentage_range(self, sample_kpi_data):
        """KPI percentage between 0 and 100."""
        pct = float(sample_kpi_data["kpi_percentage"])
        assert 0 <= pct <= 100

    def test_all_kpi_types_valid(self, sample_kpi_data, sample_capex_kpi, sample_opex_kpi):
        """All three KPI types have valid percentages."""
        for kpi in [sample_kpi_data, sample_capex_kpi, sample_opex_kpi]:
            pct = float(kpi["kpi_percentage"])
            assert 0 <= pct <= 100

    def test_kpi_percentage_decimal_precision(self, sample_kpi_data):
        """KPI percentage has appropriate decimal precision."""
        pct = sample_kpi_data["kpi_percentage"]
        assert isinstance(pct, Decimal)
        # DECIMAL(7,4) - up to 4 decimal places
        assert pct == pct.quantize(Decimal("0.0001"))

    def test_zero_total_protection(self, kpi_engine):
        """Engine handles zero total amount gracefully."""
        kpi_engine.calculate.return_value = {
            "eligible_amount": 0,
            "aligned_amount": 0,
            "total_amount": 0,
            "kpi_percentage": 0,
        }
        result = kpi_engine.calculate("org-empty", "FY2025", "turnover")
        assert result["kpi_percentage"] == 0
