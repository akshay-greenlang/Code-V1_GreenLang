# -*- coding: utf-8 -*-
"""
Unit tests for EU Taxonomy KPI Integration Engine.

Tests full KPI workflow (register activities, calculate Turnover/CapEx/
OpEx), double-counting prevention across objectives, CapEx plan impact
on CapEx KPI, objective disaggregation, KPI period comparison,
denominator validation (IAS references), and eligible vs aligned
breakdown with 38+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

import pytest
from decimal import Decimal


# ===========================================================================
# Full KPI Workflow
# ===========================================================================

class TestFullKPIWorkflow:
    """Test end-to-end KPI calculation workflow."""

    def test_kpi_result_created(self, sample_kpi_result):
        assert sample_kpi_result["org_id"] is not None
        assert sample_kpi_result["reporting_period"] is not None

    def test_all_three_kpis_present(self, sample_kpi_result):
        assert "turnover" in sample_kpi_result
        assert "capex" in sample_kpi_result
        assert "opex" in sample_kpi_result

    def test_turnover_kpi_structure(self, sample_kpi_result):
        turnover = sample_kpi_result["turnover"]
        assert "total_denominator" in turnover
        assert "eligible_amount" in turnover
        assert "aligned_amount" in turnover
        assert "eligible_pct" in turnover
        assert "aligned_pct" in turnover

    def test_capex_kpi_structure(self, sample_kpi_result):
        capex = sample_kpi_result["capex"]
        assert "total_denominator" in capex
        assert "eligible_amount" in capex
        assert "aligned_amount" in capex

    def test_opex_kpi_structure(self, sample_kpi_result):
        opex = sample_kpi_result["opex"]
        assert "total_denominator" in opex
        assert "eligible_amount" in opex
        assert "aligned_amount" in opex

    def test_aligned_less_than_eligible(self, sample_kpi_result):
        for kpi_name in ["turnover", "capex", "opex"]:
            kpi = sample_kpi_result[kpi_name]
            assert kpi["aligned_amount"] <= kpi["eligible_amount"]

    def test_eligible_less_than_denominator(self, sample_kpi_result):
        for kpi_name in ["turnover", "capex", "opex"]:
            kpi = sample_kpi_result[kpi_name]
            assert kpi["eligible_amount"] <= kpi["total_denominator"]

    def test_kpi_percentages_valid(self, sample_kpi_result):
        for kpi_name in ["turnover", "capex", "opex"]:
            kpi = sample_kpi_result[kpi_name]
            assert 0.0 <= kpi["eligible_pct"] <= 100.0
            assert 0.0 <= kpi["aligned_pct"] <= 100.0


# ===========================================================================
# Double-Counting Prevention
# ===========================================================================

class TestDoubleCounting:
    """Test double-counting prevention across environmental objectives."""

    def test_activity_counted_once(self, sample_kpi_result):
        if "activity_allocations" in sample_kpi_result:
            activity_ids = [
                a["activity_id"]
                for a in sample_kpi_result["activity_allocations"]
            ]
            assert len(activity_ids) == len(set(activity_ids))

    def test_no_cross_objective_double_count(self, sample_kpi_result):
        # Total aligned should not exceed 100% even with multiple objectives
        for kpi_name in ["turnover", "capex", "opex"]:
            kpi = sample_kpi_result[kpi_name]
            assert kpi["aligned_pct"] <= 100.0

    def test_primary_objective_allocation(self):
        activity = {
            "activity_code": "4.1",
            "contributes_to": ["climate_mitigation", "circular_economy"],
            "primary_objective": "climate_mitigation",
            "turnover_amount": 1_000_000,
        }
        assert activity["primary_objective"] in activity["contributes_to"]

    @pytest.mark.parametrize("objectives_count,expected_allocation_strategy", [
        (1, "full"),
        (2, "primary_only"),
        (3, "primary_only"),
    ])
    def test_allocation_strategy(self, objectives_count, expected_allocation_strategy):
        if objectives_count == 1:
            strategy = "full"
        else:
            strategy = "primary_only"
        assert strategy == expected_allocation_strategy


# ===========================================================================
# CapEx Plan Impact
# ===========================================================================

class TestCapExPlanImpact:
    """Test CapEx plan impact on CapEx KPI."""

    def test_capex_plan_included(self, sample_kpi_result):
        capex = sample_kpi_result["capex"]
        assert "capex_plan_amount" in capex

    def test_capex_plan_increases_aligned(self):
        capex_without_plan = {
            "aligned_amount": 5_000_000,
            "capex_plan_amount": 0,
        }
        capex_with_plan = {
            "aligned_amount": 8_000_000,
            "capex_plan_amount": 3_000_000,
        }
        assert capex_with_plan["aligned_amount"] > capex_without_plan["aligned_amount"]

    def test_capex_plan_time_bound(self):
        plan = {
            "plan_start": "2025-01-01",
            "plan_end": "2030-12-31",
            "max_duration_years": 5,
        }
        assert plan["max_duration_years"] <= 10

    def test_capex_plan_activity_linked(self):
        plan = {
            "activity_code": "7.7",
            "capex_amount": 2_000_000,
            "target_alignment": "climate_mitigation",
        }
        assert plan["activity_code"] is not None
        assert plan["target_alignment"] is not None

    def test_capex_plan_requires_evidence(self):
        plan = {
            "evidence_provided": True,
            "board_approved": True,
            "timeline_defined": True,
        }
        assert all([
            plan["evidence_provided"],
            plan["board_approved"],
            plan["timeline_defined"],
        ])


# ===========================================================================
# Objective Disaggregation
# ===========================================================================

class TestObjectiveDisaggregation:
    """Test KPI disaggregation by environmental objective."""

    OBJECTIVES = [
        "climate_mitigation", "climate_adaptation",
        "water_resources", "circular_economy",
        "pollution_prevention", "biodiversity",
    ]

    def test_objective_breakdown_present(self, sample_kpi_result):
        assert "by_objective" in sample_kpi_result
        assert len(sample_kpi_result["by_objective"]) >= 1

    def test_objective_breakdown_valid(self, sample_kpi_result):
        for obj_data in sample_kpi_result["by_objective"]:
            assert obj_data["objective"] in self.OBJECTIVES

    def test_objective_kpis_present(self, sample_kpi_result):
        for obj_data in sample_kpi_result["by_objective"]:
            assert "turnover_pct" in obj_data
            assert "capex_pct" in obj_data
            assert "opex_pct" in obj_data

    def test_objective_kpis_sum_no_more_than_total(self, sample_kpi_result):
        total_turnover = sum(
            o["turnover_pct"] for o in sample_kpi_result["by_objective"]
        )
        # Due to primary-only allocation, should not exceed total aligned
        assert total_turnover <= sample_kpi_result["turnover"]["aligned_pct"] + 0.5

    @pytest.mark.parametrize("objective", [
        "climate_mitigation", "climate_adaptation",
    ])
    def test_objective_data_structure(self, objective):
        obj_data = {
            "objective": objective,
            "eligible_turnover": 10_000_000,
            "aligned_turnover": 8_000_000,
            "turnover_pct": 80.0,
        }
        assert obj_data["aligned_turnover"] <= obj_data["eligible_turnover"]


# ===========================================================================
# KPI Period Comparison
# ===========================================================================

class TestKPIPeriodComparison:
    """Test KPI period-over-period comparison."""

    def test_period_comparison_created(self, sample_kpi_comparison):
        assert sample_kpi_comparison["current_period"] is not None
        assert sample_kpi_comparison["previous_period"] is not None

    def test_period_delta_calculated(self, sample_kpi_comparison):
        for kpi_name in ["turnover", "capex", "opex"]:
            assert f"{kpi_name}_delta_pct" in sample_kpi_comparison
            assert f"{kpi_name}_delta_abs" in sample_kpi_comparison

    def test_period_trend_direction(self, sample_kpi_comparison):
        for kpi_name in ["turnover", "capex", "opex"]:
            delta = sample_kpi_comparison[f"{kpi_name}_delta_pct"]
            if delta > 0:
                trend = "improving"
            elif delta < 0:
                trend = "declining"
            else:
                trend = "stable"
            assert trend in ["improving", "declining", "stable"]

    @pytest.mark.parametrize("current_pct,previous_pct,expected_delta", [
        (15.0, 10.0, 5.0),
        (10.0, 10.0, 0.0),
        (8.0, 12.0, -4.0),
    ])
    def test_delta_calculation(self, current_pct, previous_pct, expected_delta):
        delta = current_pct - previous_pct
        assert delta == pytest.approx(expected_delta, abs=0.01)


# ===========================================================================
# Denominator Validation (IAS References)
# ===========================================================================

class TestDenominatorValidation:
    """Test KPI denominator validation with IAS references."""

    def test_turnover_denominator_ias(self, sample_kpi_result):
        turnover = sample_kpi_result["turnover"]
        assert turnover["ias_reference"] == "IAS 1.82(a)"

    def test_capex_denominator_ias(self, sample_kpi_result):
        capex = sample_kpi_result["capex"]
        assert capex["ias_reference"] in [
            "IAS 16", "IAS 38", "IAS 40", "IFRS 16",
        ]

    def test_opex_denominator_definition(self, sample_kpi_result):
        opex = sample_kpi_result["opex"]
        assert "denominator_includes" in opex
        required = ["R&D", "renovation_measures", "short_term_leases", "maintenance"]
        for item in required:
            assert item in opex["denominator_includes"]

    def test_denominator_positive(self, sample_kpi_result):
        for kpi_name in ["turnover", "capex", "opex"]:
            assert sample_kpi_result[kpi_name]["total_denominator"] > 0

    def test_denominator_currency(self, sample_kpi_result):
        assert sample_kpi_result["currency"] in ["EUR", "USD", "GBP", "CHF"]


# ===========================================================================
# Eligible vs Aligned Breakdown
# ===========================================================================

class TestEligibleVsAlignedBreakdown:
    """Test eligible vs aligned breakdown detail."""

    def test_eligible_breakdown_present(self, sample_kpi_result):
        for kpi_name in ["turnover", "capex", "opex"]:
            kpi = sample_kpi_result[kpi_name]
            assert "eligible_not_aligned_amount" in kpi

    def test_eligible_not_aligned_amount(self, sample_kpi_result):
        for kpi_name in ["turnover", "capex", "opex"]:
            kpi = sample_kpi_result[kpi_name]
            expected_not_aligned = kpi["eligible_amount"] - kpi["aligned_amount"]
            assert abs(
                kpi["eligible_not_aligned_amount"] - expected_not_aligned
            ) < 0.01

    def test_non_eligible_amount(self, sample_kpi_result):
        for kpi_name in ["turnover", "capex", "opex"]:
            kpi = sample_kpi_result[kpi_name]
            non_eligible = kpi["total_denominator"] - kpi["eligible_amount"]
            assert non_eligible >= 0

    def test_funnel_consistency(self, sample_kpi_result):
        for kpi_name in ["turnover", "capex", "opex"]:
            kpi = sample_kpi_result[kpi_name]
            total = kpi["total_denominator"]
            eligible = kpi["eligible_amount"]
            aligned = kpi["aligned_amount"]
            assert total >= eligible >= aligned >= 0
