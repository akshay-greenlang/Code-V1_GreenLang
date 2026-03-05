# -*- coding: utf-8 -*-
"""
Unit tests for ManagementPlanEngine -- ISO 14064-1:2018 Clause 9.

Tests plan CRUD, action lifecycle, progress tracking, cost-benefit
analysis, MAC curve, action prioritization, annual review, and
plan summary with 25+ tests.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import (
    ActionCategory,
    ActionStatus,
    ISOCategory,
)
from services.management_plan import ManagementPlanEngine


class TestPlanCRUD:
    """Test management plan creation, retrieval, and deletion."""

    def test_create_plan(self, management_engine):
        plan = management_engine.create_plan("org-1", 2025)
        assert plan.org_id == "org-1"
        assert plan.reporting_year == 2025
        assert len(plan.id) == 36

    def test_get_plan(self, management_engine):
        plan = management_engine.create_plan("org-1", 2025)
        retrieved = management_engine.get_plan(plan.id)
        assert retrieved is not None
        assert retrieved.id == plan.id

    def test_get_nonexistent_plan(self, management_engine):
        assert management_engine.get_plan("bad-id") is None

    def test_get_plans_for_org(self, management_engine):
        management_engine.create_plan("org-1", 2024)
        management_engine.create_plan("org-1", 2025)
        management_engine.create_plan("org-2", 2025)
        plans = management_engine.get_plans_for_org("org-1")
        assert len(plans) == 2

    def test_get_plan_for_year(self, management_engine):
        management_engine.create_plan("org-1", 2024)
        management_engine.create_plan("org-1", 2025)
        result = management_engine.get_plan_for_year("org-1", 2025)
        assert result is not None
        assert result.reporting_year == 2025

    def test_get_plan_for_year_not_found(self, management_engine):
        management_engine.create_plan("org-1", 2024)
        result = management_engine.get_plan_for_year("org-1", 2030)
        assert result is None

    def test_delete_plan(self, management_engine):
        plan = management_engine.create_plan("org-1", 2025)
        result = management_engine.delete_plan(plan.id)
        assert result is True
        assert management_engine.get_plan(plan.id) is None

    def test_delete_nonexistent_plan(self, management_engine):
        result = management_engine.delete_plan("bad-id")
        assert result is False


class TestActionCRUD:
    """Test improvement action management."""

    def test_add_action(self, management_engine):
        plan = management_engine.create_plan("org-1", 2025)
        action = management_engine.add_action(
            plan.id,
            name="Install solar panels",
            category=ActionCategory.EMISSION_REDUCTION,
            target_reduction_tco2e=Decimal("500"),
            estimated_cost_usd=Decimal("100000"),
        )
        assert action.name == "Install solar panels"
        assert action.category == ActionCategory.EMISSION_REDUCTION
        assert action.status == ActionStatus.PLANNED

    def test_add_action_with_iso_category(self, management_engine):
        plan = management_engine.create_plan("org-1", 2025)
        action = management_engine.add_action(
            plan.id,
            name="LED retrofit",
            category=ActionCategory.EMISSION_REDUCTION,
            iso_category=ISOCategory.CATEGORY_2_ENERGY,
        )
        assert action.iso_category == ISOCategory.CATEGORY_2_ENERGY

    def test_add_action_invalid_plan_raises(self, management_engine):
        with pytest.raises(ValueError, match="not found"):
            management_engine.add_action(
                "bad-id",
                name="X",
                category=ActionCategory.EMISSION_REDUCTION,
            )

    def test_remove_action(self, management_engine):
        plan = management_engine.create_plan("org-1", 2025)
        action = management_engine.add_action(
            plan.id, "Solar", ActionCategory.EMISSION_REDUCTION,
        )
        result = management_engine.remove_action(plan.id, action.id)
        assert result is True
        assert len(management_engine.get_plan(plan.id).actions) == 0

    def test_remove_nonexistent_action(self, management_engine):
        plan = management_engine.create_plan("org-1", 2025)
        result = management_engine.remove_action(plan.id, "bad-action-id")
        assert result is False


class TestActionStatus:
    """Test action status updates."""

    def test_update_status(self, management_engine):
        plan = management_engine.create_plan("org-1", 2025)
        action = management_engine.add_action(
            plan.id, "Solar", ActionCategory.EMISSION_REDUCTION,
        )
        updated = management_engine.update_action_status(
            plan.id, action.id, ActionStatus.IN_PROGRESS,
        )
        assert updated.status == ActionStatus.IN_PROGRESS

    def test_update_status_invalid_action_raises(self, management_engine):
        plan = management_engine.create_plan("org-1", 2025)
        with pytest.raises(ValueError, match="not found"):
            management_engine.update_action_status(
                plan.id, "bad-id", ActionStatus.COMPLETED,
            )


class TestProgressTracking:
    """Test progress update logic."""

    def test_update_progress(self, management_engine):
        plan = management_engine.create_plan("org-1", 2025)
        action = management_engine.add_action(
            plan.id, "Solar", ActionCategory.EMISSION_REDUCTION,
        )
        updated = management_engine.update_progress(
            plan.id, action.id, Decimal("50"), notes="Halfway done",
        )
        assert updated.progress_pct == Decimal("50")
        assert updated.notes == "Halfway done"

    def test_progress_auto_sets_in_progress(self, management_engine):
        plan = management_engine.create_plan("org-1", 2025)
        action = management_engine.add_action(
            plan.id, "Solar", ActionCategory.EMISSION_REDUCTION,
        )
        updated = management_engine.update_progress(
            plan.id, action.id, Decimal("10"),
        )
        assert updated.status == ActionStatus.IN_PROGRESS

    def test_progress_100_auto_completes(self, management_engine):
        plan = management_engine.create_plan("org-1", 2025)
        action = management_engine.add_action(
            plan.id, "Solar", ActionCategory.EMISSION_REDUCTION,
        )
        updated = management_engine.update_progress(
            plan.id, action.id, Decimal("100"),
        )
        assert updated.status == ActionStatus.COMPLETED


class TestCostBenefitAnalysis:
    """Test MAC curve and cost-benefit analysis."""

    def test_cost_benefit_structure(self, management_engine):
        plan = management_engine.create_plan("org-1", 2025)
        management_engine.add_action(
            plan.id, "Solar", ActionCategory.EMISSION_REDUCTION,
            target_reduction_tco2e=Decimal("500"),
            estimated_cost_usd=Decimal("100000"),
        )
        result = management_engine.get_cost_benefit_analysis(plan.id)
        assert result["total_estimated_cost_usd"] == "100000"
        assert result["total_target_reduction_tco2e"] == "500"
        assert result["action_count"] == 1

    def test_mac_curve_sorted_cheapest_first(self, management_engine):
        plan = management_engine.create_plan("org-1", 2025)
        # Expensive per tCO2e
        management_engine.add_action(
            plan.id, "CCS", ActionCategory.REMOVAL_ENHANCEMENT,
            target_reduction_tco2e=Decimal("100"),
            estimated_cost_usd=Decimal("50000"),
        )
        # Cheap per tCO2e
        management_engine.add_action(
            plan.id, "LED", ActionCategory.EMISSION_REDUCTION,
            target_reduction_tco2e=Decimal("200"),
            estimated_cost_usd=Decimal("10000"),
        )
        result = management_engine.get_cost_benefit_analysis(plan.id)
        mac_curve = result["mac_curve"]
        assert len(mac_curve) == 2
        # LED first (50 $/tCO2e) < CCS (500 $/tCO2e)
        assert mac_curve[0]["action"] == "LED"
        assert mac_curve[1]["action"] == "CCS"

    def test_mac_curve_empty_plan(self, management_engine):
        plan = management_engine.create_plan("org-1", 2025)
        result = management_engine.get_cost_benefit_analysis(plan.id)
        assert result["mac_curve"] == []

    def test_invalid_plan_raises(self, management_engine):
        with pytest.raises(ValueError, match="not found"):
            management_engine.get_cost_benefit_analysis("bad-id")


class TestProgressSummary:
    """Test progress summary generation."""

    def test_summary_structure(self, management_engine):
        plan = management_engine.create_plan("org-1", 2025)
        management_engine.add_action(
            plan.id, "Solar", ActionCategory.EMISSION_REDUCTION,
            target_reduction_tco2e=Decimal("500"),
        )
        result = management_engine.get_progress_summary(plan.id)
        assert result["plan_id"] == plan.id
        assert result["action_count"] == 1
        assert "by_status" in result
        assert "by_category" in result

    def test_completed_reduction_tracked(self, management_engine):
        plan = management_engine.create_plan("org-1", 2025)
        action = management_engine.add_action(
            plan.id, "Solar", ActionCategory.EMISSION_REDUCTION,
            target_reduction_tco2e=Decimal("500"),
        )
        management_engine.update_progress(plan.id, action.id, Decimal("100"))
        result = management_engine.get_progress_summary(plan.id)
        assert result["completed_reduction_tco2e"] == "500"
        assert result["reduction_progress_pct"] == "100.0"


class TestPrioritizedActions:
    """Test action prioritization by MAC."""

    def test_prioritized_returns_list(self, management_engine):
        plan = management_engine.create_plan("org-1", 2025)
        management_engine.add_action(
            plan.id, "Solar", ActionCategory.EMISSION_REDUCTION,
            target_reduction_tco2e=Decimal("500"),
            estimated_cost_usd=Decimal("50000"),
        )
        result = management_engine.get_prioritized_actions(plan.id)
        assert len(result) == 1
        assert result[0]["name"] == "Solar"

    def test_completed_actions_sorted_last(self, management_engine):
        plan = management_engine.create_plan("org-1", 2025)
        a1 = management_engine.add_action(
            plan.id, "Done", ActionCategory.EMISSION_REDUCTION,
            target_reduction_tco2e=Decimal("100"),
            estimated_cost_usd=Decimal("5000"),
        )
        management_engine.update_progress(plan.id, a1.id, Decimal("100"))
        management_engine.add_action(
            plan.id, "Pending", ActionCategory.EMISSION_REDUCTION,
            target_reduction_tco2e=Decimal("200"),
            estimated_cost_usd=Decimal("10000"),
        )
        result = management_engine.get_prioritized_actions(plan.id)
        assert result[0]["name"] == "Pending"
        assert result[1]["name"] == "Done"


class TestAnnualReview:
    """Test annual review process."""

    def test_annual_review_structure(self, management_engine):
        plan = management_engine.create_plan("org-1", 2025)
        management_engine.add_action(
            plan.id, "Solar", ActionCategory.EMISSION_REDUCTION,
            target_reduction_tco2e=Decimal("500"),
            estimated_cost_usd=Decimal("100000"),
        )
        result = management_engine.conduct_annual_review(plan.id, "Annual check")
        assert result["plan_id"] == plan.id
        assert result["notes"] == "Annual check"
        assert "progress_summary" in result
        assert "cost_benefit_summary" in result
        assert "recommendations" in result

    def test_review_recommends_when_low_progress(self, management_engine):
        plan = management_engine.create_plan("org-1", 2025)
        management_engine.add_action(
            plan.id, "Solar", ActionCategory.EMISSION_REDUCTION,
            target_reduction_tco2e=Decimal("500"),
        )
        result = management_engine.conduct_annual_review(plan.id)
        recs = result["recommendations"]
        assert len(recs) >= 1

    def test_review_recommends_unassigned(self, management_engine):
        plan = management_engine.create_plan("org-1", 2025)
        management_engine.add_action(
            plan.id, "Solar", ActionCategory.EMISSION_REDUCTION,
        )
        result = management_engine.conduct_annual_review(plan.id)
        # Should flag unassigned actions
        recs_text = " ".join(result["recommendations"])
        assert "assigned" in recs_text.lower() or "no assigned" in recs_text.lower()


class TestPlanSummary:
    """Test comprehensive plan summary."""

    def test_summary_structure(self, management_engine):
        plan = management_engine.create_plan("org-1", 2025)
        management_engine.add_action(
            plan.id, "Solar", ActionCategory.EMISSION_REDUCTION,
            target_reduction_tco2e=Decimal("500"),
            estimated_cost_usd=Decimal("100000"),
        )
        result = management_engine.generate_plan_summary(plan.id)
        assert result["plan_id"] == plan.id
        assert result["org_id"] == "org-1"
        assert result["total_actions"] == 1
        assert result["total_planned_reduction_tco2e"] == "500"
        assert len(result["actions"]) == 1

    def test_summary_nonexistent_plan_raises(self, management_engine):
        with pytest.raises(ValueError, match="not found"):
            management_engine.generate_plan_summary("bad-id")
